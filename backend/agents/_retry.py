"""
Shared transient-failure retry policy for DashScope API calls.

requirements.txt already lists tenacity specifically for this ("retry/backoff
for Qwen and video-gen API calls"), but nothing used it yet -- this is that
policy, applied uniformly instead of each agent module hand-rolling its own.

Scope, deliberately narrow: only retries transport-level failures (dropped
connections, timeouts). Never retries on a malformed/invalid model response --
that's a content problem, already owned by each agent's own re-prompt-once-
then-degrade logic, and retrying transport failures inside that logic would
conflate two different failure classes.
"""
from __future__ import annotations

import logging

from openai import APIConnectionError, APITimeoutError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

logger = logging.getLogger("productcut.agents.retry")


def _log_retry(retry_state) -> None:
    logger.warning(
        "DashScope call failed (%s), retrying (attempt %d/3)...",
        retry_state.outcome.exception(), retry_state.attempt_number,
    )


dashscope_retry = retry(
    retry=retry_if_exception_type((APIConnectionError, APITimeoutError)),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    stop=stop_after_attempt(3),
    reraise=True,
    before_sleep=_log_retry,
)


@dashscope_retry
async def create_completion(client, *, model: str, messages: list[dict]) -> str:
    """The one place every agent should call the DashScope chat endpoint.

    Uses streaming rather than a single blocking call. Empirically, non-streaming
    calls against this project's DashScope endpoint intermittently die mid-wait
    (ReadError / ReadTimeout) once the model takes any real time to respond --
    something in the network path appears to kill idle connections. Streaming
    keeps the connection actively receiving data instead of idle-waiting for one
    large response, which avoids that failure mode. Returns the assembled text
    content directly (not a response object) -- callers don't need chunk-level
    access, just the final string to hand to _parse_json_response.
    """
    stream = await client.chat.completions.create(model=model, messages=messages, stream=True)
    parts: list[str] = []
    async for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta.content
        if delta:
            parts.append(delta)
    return "".join(parts)
