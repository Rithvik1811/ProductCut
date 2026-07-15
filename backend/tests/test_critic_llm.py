"""
Hermetic unit tests for agents/critic_llm.py -- the shared Qwen JSON-call helper
every LLM-backed Critic-Chain checker routes through (Body §5.4.3, CTA §5.4.4,
Tone §5.4.5, Meta-Critic §5.4.6, Merge Coherence Validator §5.4.7).

Focus is the two transport-hardening fixes this module owns, exercised entirely
against the fake sync OpenAI client (no real network):

  1. A soft *empty* streamed response (assembled content == '') is retry-eligible:
     empty-then-valid succeeds transparently; empty-on-every-attempt surfaces the
     public QwenJSONError after the 3-attempt policy is exhausted.
  2. A genuinely malformed-but-NON-empty response is NOT retried (unchanged
     behaviour -- a same-prompt retry can't fix a content problem).
  3. _client() actually hands OpenAI(...) an explicit `timeout=`.

Network is faked at `agents.critic_llm.OpenAI`, the single construction point --
same monkeypatch target test_meta_critic.py / test_merge_validator.py use.
conftest.py's autouse `_fake_dashscope_env` supplies dummy DASHSCOPE_* env so
`_client()` doesn't raise before the fake client is returned.
"""
from __future__ import annotations

import json

import httpx
import pytest

from agents import critic_llm
from agents.critic_llm import (
    QwenJSONError,
    _EmptyStreamContent,
    call_qwen_json,
    call_qwen_json_validated,
)
from tests._fakes import FakeSyncOpenAIClient, make_fake_sync_openai


@pytest.fixture(autouse=True)
def _no_retry_sleep(monkeypatch):
    """Zero out tenacity's backoff so the retrying tests run instantly.

    Targets the Retrying instance the @retry decorator attaches to
    _create_with_retry, so only this module's retries are affected.
    """
    monkeypatch.setattr(critic_llm._create_with_retry.retry, "sleep", lambda _s: None)


def _shared_client_factory(client: FakeSyncOpenAIClient):
    """Monkeypatch factory returning ONE shared instance, so a test can inspect
    its call_count after the call (make_fake_sync_openai builds a fresh instance
    per construction, which hides how many .create() calls actually happened --
    fine there, but we need the count here)."""

    def _factory(*_a, **_k):
        return client

    return _factory


# ===========================================================================
# Happy path.
# ===========================================================================


def test_valid_json_returns_dict_single_call(monkeypatch):
    client = FakeSyncOpenAIClient([json.dumps({"ok": True, "score": 4})])
    monkeypatch.setattr("agents.critic_llm.OpenAI", _shared_client_factory(client))

    result = call_qwen_json("system rubric", "user payload")

    assert result == {"ok": True, "score": 4}
    assert client.call_count == 1  # no retry on a clean first response


# ===========================================================================
# Empty-stream retry behaviour (fix #2).
# ===========================================================================


def test_empty_content_then_success_retries_transparently(monkeypatch):
    # First .create() yields an empty stream (soft empty response); the retry then
    # gets real JSON. call_qwen_json should return the real content, hiding the blip.
    good = json.dumps({"recovered": True})
    client = FakeSyncOpenAIClient(["", good])
    monkeypatch.setattr("agents.critic_llm.OpenAI", _shared_client_factory(client))

    result = call_qwen_json("system rubric", "user payload")

    assert result == {"recovered": True}
    assert client.call_count == 2  # attempt 1 empty -> retried -> attempt 2 succeeded


def test_whitespace_only_content_is_treated_as_empty(monkeypatch):
    # A stream that delivers only whitespace is as useless as a truly empty one --
    # it must also be retried, not fed to json.loads (which would raise QwenJSONError).
    good = json.dumps({"recovered": True})
    client = FakeSyncOpenAIClient(["   \n\t  ", good])
    monkeypatch.setattr("agents.critic_llm.OpenAI", _shared_client_factory(client))

    result = call_qwen_json("system rubric", "user payload")

    assert result == {"recovered": True}
    assert client.call_count == 2


def test_empty_content_on_all_attempts_raises_qwenjsonerror(monkeypatch):
    # Every attempt returns empty -> retries exhausted -> the public QwenJSONError
    # is surfaced (NOT silently returning {} or leaking the internal marker).
    client = FakeSyncOpenAIClient([""])
    monkeypatch.setattr("agents.critic_llm.OpenAI", _shared_client_factory(client))

    with pytest.raises(QwenJSONError) as excinfo:
        call_qwen_json("system rubric", "user payload")

    assert client.call_count == 3  # stop_after_attempt(3) -- all three tried
    # The underlying transient marker is chained as the cause, not swallowed.
    assert isinstance(excinfo.value.__cause__, _EmptyStreamContent)
    assert "no content" in str(excinfo.value)


def test_empty_marker_is_registered_as_transient():
    # Guard the wiring itself: if _EmptyStreamContent ever drops out of the retry
    # set, the empty-stream case would fail on the first attempt instead of retrying.
    assert _EmptyStreamContent in critic_llm._TRANSIENT_ERRORS


# ===========================================================================
# Malformed-but-NON-empty is NOT retried (existing behaviour must not regress).
# ===========================================================================


def test_malformed_nonempty_json_not_retried(monkeypatch):
    client = FakeSyncOpenAIClient(["not json at all"])
    monkeypatch.setattr("agents.critic_llm.OpenAI", _shared_client_factory(client))

    with pytest.raises(QwenJSONError):
        call_qwen_json("system rubric", "user payload")

    assert client.call_count == 1  # content problem -> exactly one attempt, no retry


def test_valid_json_non_object_raises_and_not_retried(monkeypatch):
    # A JSON array is valid JSON but not the expected top-level object -- still a
    # content problem, so still exactly one attempt.
    client = FakeSyncOpenAIClient(["[1, 2, 3]"])
    monkeypatch.setattr("agents.critic_llm.OpenAI", _shared_client_factory(client))

    with pytest.raises(QwenJSONError):
        call_qwen_json("system rubric", "user payload")

    assert client.call_count == 1


# ===========================================================================
# _client() passes an explicit timeout (fix #1).
# ===========================================================================


def test_client_passes_explicit_timeout(monkeypatch):
    captured: dict = {}

    def _recording_ctor(*_a, **kwargs):
        captured.update(kwargs)
        return FakeSyncOpenAIClient([json.dumps({"ok": True})])

    monkeypatch.setattr("agents.critic_llm.OpenAI", _recording_ctor)

    call_qwen_json("system rubric", "user payload")

    assert "timeout" in captured, "OpenAI() was constructed without an explicit timeout"
    timeout = captured["timeout"]
    assert isinstance(timeout, httpx.Timeout)
    # The calibrated read budget must sit well above the slowest observed-successful
    # checker call (~165s) so a slow-but-working call is never a guaranteed timeout.
    assert timeout.read == 240.0
    assert timeout.connect == 10.0
    assert captured["timeout"] is critic_llm._REQUEST_TIMEOUT


def test_missing_api_key_raises_before_any_client(monkeypatch):
    # Loud-fail-on-misconfig behaviour must survive the timeout change.
    monkeypatch.setattr("agents.critic_llm.OpenAI", make_fake_sync_openai(["{}"]))
    monkeypatch.setenv("DASHSCOPE_API_KEY", "<your-key-here>")

    with pytest.raises(RuntimeError, match="DASHSCOPE_API_KEY"):
        call_qwen_json("system rubric", "user payload")


# ===========================================================================
# call_qwen_json_validated -- bounded retry-on-VALIDATION-failure (video-gen-
# fidelity branch fix). Distinct from the transport/empty-stream retries above:
# this is the "well-formed JSON that fails the caller's OWN schema check" case,
# which `call_qwen_json` alone has zero protection against (see its own
# docstring's NOTE, added alongside this fix).
# ===========================================================================


def _reject_once_then_accept():
    """A validate_fn that raises ValueError on its first call and returns the
    parsed dict unchanged on any call after -- simulates a one-off content
    failure that a re-prompt actually fixes."""
    calls = {"n": 0}

    def _validate(parsed: dict) -> dict:
        calls["n"] += 1
        if calls["n"] == 1:
            raise ValueError("simulated validation failure: bad field name")
        return parsed

    return _validate


def test_validated_happy_path_no_retry_when_validate_fn_passes_first_try(monkeypatch):
    client = FakeSyncOpenAIClient([json.dumps({"ok": True})])
    monkeypatch.setattr("agents.critic_llm.OpenAI", _shared_client_factory(client))

    result = call_qwen_json_validated(
        "system rubric", "user payload", lambda parsed: parsed
    )

    assert result == {"ok": True}
    assert client.call_count == 1, "validate_fn passed first try -- no re-prompt needed"


def test_validated_retries_once_on_validation_error_and_recovers(monkeypatch):
    client = FakeSyncOpenAIClient([json.dumps({"ok": True})])
    monkeypatch.setattr("agents.critic_llm.OpenAI", _shared_client_factory(client))

    result = call_qwen_json_validated(
        "system rubric", "user payload", _reject_once_then_accept()
    )

    assert result == {"ok": True}
    assert client.call_count == 2, "one failed validation + one successful re-prompt call"


class _RecordingSyncClient:
    """Like FakeSyncOpenAIClient, but records every `messages` list it was
    called with -- needed to assert on the actual re-prompt turn's content,
    which the plain fake doesn't expose."""

    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self.call_count = 0
        self.seen_messages: list[list[dict]] = []
        self.chat = self
        self.completions = self

    def create(self, model: str, messages: list[dict], **_kwargs):
        from tests._fakes import _FakeSyncStream

        self.seen_messages.append(messages)
        content = self._responses[min(self.call_count, len(self._responses) - 1)]
        self.call_count += 1
        return _FakeSyncStream(content)


def test_validated_reprompt_message_names_the_exact_error(monkeypatch):
    """The re-prompt turn sent back to the model must name the exact error, not
    a generic 'try again' -- mirroring product_truth_extractor.py's
    _reprompt_message / hook_checker.py's retry pattern -- and must carry the
    original assistant reply forward as real conversation history, not just a
    fresh one-shot prompt."""
    client = _RecordingSyncClient([json.dumps({"ok": True})])
    monkeypatch.setattr("agents.critic_llm.OpenAI", lambda *a, **k: client)

    call_qwen_json_validated("system rubric", "user payload", _reject_once_then_accept())

    assert client.call_count == 2
    second_call_messages = client.seen_messages[1]
    assert len(second_call_messages) == 4, "system, user, assistant(raw reply), user(error) turns"
    assert second_call_messages[0]["role"] == "system"
    assert second_call_messages[1]["role"] == "user"
    assert second_call_messages[2]["role"] == "assistant"
    assert second_call_messages[3]["role"] == "user"
    assert "bad field name" in second_call_messages[3]["content"], (
        "the re-prompt must name the EXACT validation error, not a generic retry message"
    )


def test_validated_raises_clearly_after_max_attempts_exhausted(monkeypatch):
    """Every attempt fails validation -> bounded at max_attempts (default 2),
    never hangs, never silently degrades -- raises the real, last ValueError."""
    client = FakeSyncOpenAIClient([json.dumps({"ok": True})])
    monkeypatch.setattr("agents.critic_llm.OpenAI", _shared_client_factory(client))

    def _always_fail(parsed: dict) -> dict:
        raise ValueError("simulated persistent validation failure")

    with pytest.raises(ValueError, match="simulated persistent validation failure"):
        call_qwen_json_validated("system rubric", "user payload", _always_fail)

    assert client.call_count == 2, "bounded at the default max_attempts=2, not retried forever"


def test_validated_respects_custom_max_attempts(monkeypatch):
    client = FakeSyncOpenAIClient([json.dumps({"ok": True})])
    monkeypatch.setattr("agents.critic_llm.OpenAI", _shared_client_factory(client))

    def _always_fail(parsed: dict) -> dict:
        raise ValueError("still bad")

    with pytest.raises(ValueError, match="still bad"):
        call_qwen_json_validated(
            "system rubric", "user payload", _always_fail, max_attempts=3
        )

    assert client.call_count == 3


def test_validated_non_valueerror_from_validate_fn_is_not_retried(monkeypatch):
    """A validate_fn exception that is NOT a ValueError propagates immediately,
    unretried -- deliberately narrow, matching every other re-prompt loop in
    this codebase (they only catch the specific content-failure type their own
    validator raises)."""
    client = FakeSyncOpenAIClient([json.dumps({"ok": True})])
    monkeypatch.setattr("agents.critic_llm.OpenAI", _shared_client_factory(client))

    def _raise_type_error(parsed: dict) -> dict:
        raise TypeError("not a ValueError")

    with pytest.raises(TypeError):
        call_qwen_json_validated("system rubric", "user payload", _raise_type_error)

    assert client.call_count == 1, "non-ValueError is not retried"
