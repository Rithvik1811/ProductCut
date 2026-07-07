"""
Shared Qwen JSON-call helper for the Critic Chain (docs/TECHNICAL_DOCUMENTATION.md §5.4).

Every LLM-backed checker in the Critic Chain (Hook-Checker §5.4.1, Body-Checker
§5.4.3, CTA-Checker §5.4.4, Tone-Checker §5.4.5, Meta-Critic §5.4.6, ...) makes
the *same* shape of call: send a system rubric + a user payload of the four
script variants, and get back a small structured JSON object of per-variant
scores. The Pacing-Checker (§5.4.2) is the deliberate exception — it is pure
arithmetic and makes no LLM call at all.

Rather than have each checker re-implement "construct the DashScope OpenAI-compatible
client from env, force JSON output, retry on a transient hiccup, parse the string
back to a dict", this module centralises that one pattern. It intentionally does
*not* know anything about scoring rubrics or variant shapes — those live in each
checker module, so this helper stays reusable by the not-yet-built checkers too.

Why the OpenAI SDK and not the native `dashscope` SDK: the text-reasoning models
are served over DashScope's OpenAI-compatible endpoint (see backend/.env:
DASHSCOPE_BASE_URL .../compatible-mode/v1), and requirements.txt already pins
`openai` for exactly this ("used against DashScope's OpenAI-compatible endpoint").
The native `dashscope` dep is reserved for Wan video synthesis / TTS task polling.

This module reads config from the environment (DASHSCOPE_API_KEY,
DASHSCOPE_BASE_URL, MODEL_TEXT) — it never hard-codes model ids or keys, so the
region-locked key and the `qwen3.7-plus` text model are both swappable via .env
without touching code.

Two transport-hardening fixes live here (both learned from live end-to-end smoke
tests where a checker call misbehaved in a way the streaming switch above did NOT
cover):

1. Explicit request timeout (_client). The OpenAI SDK's own default is a 600s
   *read* timeout (openai._constants.DEFAULT_TIMEOUT == httpx.Timeout(connect=5,
   read=600, write=600, pool=600)). A call that genuinely stalls mid-stream (no
   chunk arrives, but nothing errors either) therefore hangs for a full 10 minutes
   per attempt — and with the 3-attempt retry below, up to ~30 minutes — before
   any error reaches the caller. A live run hit exactly this: a checker hung for
   10+ minutes with zero output before being killed. We set an explicit, calibrated
   timeout so a truly stuck call fails reasonably promptly. The read budget is sized
   well ABOVE the slowest *observed-successful* checker call (Body-Checker ~165s;
   CTA/Tone ~66-77s; a trivial prompt ~7s) so we never convert a legitimately-slow-
   but-working call into a guaranteed timeout — a flat 60s (as Hook-Checker uses for
   its own, faster single-axis call) would do exactly that here.

2. Empty/blank streamed response is retry-eligible (_create_with_retry). The
   streaming switch handles a HARD connection drop (RemoteProtocolError /
   APIConnectionError). It does NOT handle the *soft* empty response observed live:
   the HTTP call succeeds end-to-end, the streaming loop completes with no exception,
   yet zero content came through — the assembled string is literally ''. That is not
   "the model tried and produced garbage" (a content problem a same-prompt retry
   can't fix, which is why QwenJSONError is deliberately excluded from the retry set
   below); it is "nothing came through at all", which points at a transport hiccup
   and IS worth a same-prompt retry. So an empty/whitespace-only assembly raises the
   transient marker _EmptyStreamContent, exhausting the same 3-attempt/backoff policy;
   only if content is STILL empty after all attempts does call_qwen_json surface the
   existing QwenJSONError (a genuine failure worth reporting clearly, not swallowing).
"""
from __future__ import annotations

import json
import logging
import os
from typing import Optional

import httpx
from openai import (
    APIConnectionError,
    InternalServerError,
    OpenAI,
    RateLimitError,
)
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger("productcut.critics.llm")


# Read budget (240s) with real margin over the slowest observed-successful checker
# call (~165s), so a genuinely-stuck call fails in ~4 min/attempt (vs the SDK's 600s
# default) without misclassifying a slow-but-working call as a failure. Connect stays
# short (10s): a connection that won't even establish is stuck, not slow.
_REQUEST_TIMEOUT = httpx.Timeout(connect=10.0, read=240.0, write=30.0, pool=10.0)


class QwenJSONError(RuntimeError):
    """Raised when the model's response cannot be parsed as a JSON object.

    Distinct from the transport/API errors tenacity retries on: this is a
    *content* failure (the model returned non-JSON despite json_object mode),
    which a caller may want to handle by re-prompting rather than crashing —
    mirroring the "re-prompt once, then fall back" policy the Concept Agent
    and Shot-List Agent use elsewhere in the pipeline.
    """


class _EmptyStreamContent(RuntimeError):
    """Internal transient marker: the stream finished with no content at all.

    Raised inside _create_with_retry (and listed in _TRANSIENT_ERRORS) so an
    empty/whitespace-only assembled response re-runs under the same retry policy
    as a network blip. Deliberately NOT a subclass of the OpenAI SDK's
    APIConnectionError: that exception's constructor requires a live
    `request: httpx.Request` we have no clean way to fabricate here, so a tiny
    purpose-built exception is cleaner than faking one. Never escapes this module
    — call_qwen_json converts a still-empty result after all retries into the
    public QwenJSONError.
    """


def _client() -> OpenAI:
    """Build an OpenAI client pointed at the DashScope compatible endpoint.

    Reads DASHSCOPE_API_KEY and DASHSCOPE_BASE_URL from the environment. Raises
    a clear error if the key is missing/placeholder, so a misconfigured .env
    fails loudly at call time instead of surfacing as an opaque 401 later.
    """
    api_key = os.getenv("DASHSCOPE_API_KEY", "")
    base_url = os.getenv("DASHSCOPE_BASE_URL", "")
    if not api_key or api_key.startswith("<"):
        raise RuntimeError(
            "DASHSCOPE_API_KEY is not set (or is still the .env.example placeholder). "
            "Set it in backend/.env before calling any Critic Chain checker."
        )
    if not base_url or base_url.startswith("<"):
        raise RuntimeError(
            "DASHSCOPE_BASE_URL is not set. It must be the region-matched "
            "OpenAI-compatible endpoint, e.g. https://dashscope-us.aliyuncs.com/compatible-mode/v1"
        )
    # Explicit timeout: without it the SDK falls back to its 600s-read default, so a
    # stalled call hangs ~10 min/attempt (~30 min across retries) before erroring.
    return OpenAI(api_key=api_key, base_url=base_url, timeout=_REQUEST_TIMEOUT)


# Retry only on genuinely transient failures (network blips / 5xx / rate limits /
# an empty stream that never delivered content). Non-transient API errors (401 auth,
# 400 bad request, ...) are NOT retried — they would fail identically three times and
# only delay the real error by the full backoff. We also deliberately do NOT retry
# QwenJSONError here: a re-run with the same prompt and temperature=0 would just
# reproduce the same malformed (but non-empty) content, so that is the caller's
# re-prompt concern, not a mechanical retry. An EMPTY stream is the exception — see
# _EmptyStreamContent — because "nothing arrived" is a transport symptom, not a
# content-quality one.
_TRANSIENT_ERRORS = (
    APIConnectionError,   # network blip / timeout (APITimeoutError subclasses this)
    RateLimitError,       # 429
    InternalServerError,  # 5xx
    _EmptyStreamContent,  # stream finished with zero content (soft empty response)
)


def _log_retry(retry_state) -> None:
    """before_sleep hook: surface WHY a checker call is being retried.

    Without this, transient retries (now including the soft-empty-stream case) are
    silent, so a call that succeeds only on attempt 2/3 looks instantaneous in the
    logs and the underlying flakiness stays invisible. Kept at WARNING because a
    retry always means something went wrong on the wire the first time.
    """
    exc = retry_state.outcome.exception() if retry_state.outcome else None
    logger.warning(
        "critic_llm: transient failure on attempt %d (%s: %s) — retrying",
        retry_state.attempt_number,
        type(exc).__name__ if exc else "?",
        exc,
    )


@retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    retry=retry_if_exception_type(_TRANSIENT_ERRORS),
    before_sleep=_log_retry,
)
def _create_with_retry(client: OpenAI, **kwargs) -> str:
    """Call the chat endpoint with streaming and return the assembled content string.

    Uses stream=True rather than a single blocking call: empirically, non-streaming
    calls against this project's DashScope endpoint intermittently die mid-wait
    (httpx.RemoteProtocolError "Server disconnected without sending a response" /
    openai.APIConnectionError) once the model takes any real time to respond --
    something in the network path kills the idle-waiting connection. Streaming keeps
    the connection actively receiving data instead of idle-waiting for one large
    response, avoiding that failure mode. This mirrors the fix already applied in
    agents/_retry.py's create_completion(). Returns the final string directly (not a
    response object) -- callers only need the assembled content to json.loads().

    If the stream completes with NO content (assembled string empty/whitespace-only)
    despite raising no exception -- the soft-empty response observed live -- this
    raises _EmptyStreamContent so the same retry policy re-attempts it. See the module
    docstring for why an empty response is treated as transport, not content, failure.
    """
    stream = client.chat.completions.create(**kwargs, stream=True)
    parts: list[str] = []
    for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta.content
        if delta:
            parts.append(delta)
    content = "".join(parts)
    if not content.strip():
        raise _EmptyStreamContent(
            "DashScope stream returned no content (empty/whitespace-only assembly)"
        )
    return content


def call_qwen_json(
    system_prompt: str,
    user_prompt: str,
    *,
    model: Optional[str] = None,
    temperature: float = 0.0,
) -> dict:
    """Call the Qwen text model and return its response parsed as a JSON object.

    Args:
        system_prompt: the rubric / role instructions (the checker's calibration).
        user_prompt:   the payload — typically the script variants to score.
        model:         override the model id; defaults to MODEL_TEXT from .env
                       (qwen3.7-plus), which §5.4 designates for the checkers.
        temperature:   defaults to 0.0 — scoring is a judgment task we want
                       *stable and reproducible*, not creative sampling. This is
                       the same reason the Pacing-Checker is deterministic code:
                       we don't want a variant's score to wobble run-to-run.

    Returns:
        The parsed top-level JSON object (a dict).

    Raises:
        QwenJSONError: if the model output isn't a JSON object, OR if the stream
                       still returns no content at all after all retries (an empty
                       response that survived 3 attempts is a genuine failure worth
                       surfacing clearly, not silently swallowing).
        Exception:     the underlying OpenAI/transport error, after 3 attempts.
    """
    model_id = model or os.getenv("MODEL_TEXT", "qwen3.7-plus")
    client = _client()

    try:
        content = _create_with_retry(
            client,
            model=model_id,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            # Force structured output so we never have to strip prose/markdown fences.
            response_format={"type": "json_object"},
        )
    except _EmptyStreamContent as exc:
        # Retries are exhausted and content is STILL empty -- a genuine failure now.
        logger.error("Qwen returned empty content after all retries")
        raise QwenJSONError(
            f"Model {model_id} returned no content after 3 attempts: {exc}"
        ) from exc
    try:
        parsed = json.loads(content)
    except (json.JSONDecodeError, TypeError) as exc:
        logger.error("Qwen returned non-JSON content: %r", content[:500])
        raise QwenJSONError(
            f"Model {model_id} did not return valid JSON: {exc}"
        ) from exc

    if not isinstance(parsed, dict):
        raise QwenJSONError(
            f"Model {model_id} returned a JSON {type(parsed).__name__}, expected an object."
        )
    return parsed


__all__ = ["call_qwen_json", "QwenJSONError"]
