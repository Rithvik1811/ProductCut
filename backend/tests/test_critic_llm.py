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
from agents.critic_llm import QwenJSONError, _EmptyStreamContent, call_qwen_json
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
