"""
Shared fake OpenAI-compatible client for tests. Mimics the STREAMING interface
agents._retry.create_completion actually uses: `client.chat.completions.create(
..., stream=True)` returning an async-iterable of chunks, each exposing
`.choices[0].delta.content` -- not the old single-shot `.choices[0].message.content`
shape. If create_completion's interface changes again, this is the one place
to update, not every test file individually.
"""
from __future__ import annotations

from typing import Callable


class _FakeDelta:
    def __init__(self, content: str):
        self.content = content


class _FakeChunkChoice:
    def __init__(self, content: str):
        self.delta = _FakeDelta(content)


class _FakeChunk:
    def __init__(self, content: str):
        self.choices = [_FakeChunkChoice(content)]


class _FakeStream:
    """Async iterable yielding a single chunk carrying the whole content.

    Real streaming yields many small token-sized chunks; one chunk is
    functionally equivalent here since create_completion just concatenates
    whatever chunks it receives.
    """

    def __init__(self, content: str):
        self._content = content

    def __aiter__(self):
        return self._gen()

    async def _gen(self):
        yield _FakeChunk(self._content)


class FakeOpenAIClient:
    """Stand-in for openai.AsyncOpenAI's chat.completions.create, streaming shape.

    Returns pre-programmed JSON responses in order; call N+ (beyond the list)
    reuses the last one. Usable two ways:
      - directly, passed as the `client=` arg agents already accept for testing
      - wrapped by `make_fake_async_openai()` to replace AsyncOpenAI itself via
        monkeypatch, for the own_client=True path (which also calls .close())
    """

    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self.call_count = 0
        self.chat = self  # let `client.chat.completions.create` resolve onto us
        self.completions = self

    async def create(self, model: str, messages: list[dict], stream: bool = False) -> _FakeStream:  # noqa: ARG002
        content = self._responses[min(self.call_count, len(self._responses) - 1)]
        self.call_count += 1
        return _FakeStream(content)

    async def close(self) -> None:
        pass


def make_fake_async_openai(responses: list[str]) -> Callable[..., FakeOpenAIClient]:
    """Factory for monkeypatching `agents.<module>.AsyncOpenAI` itself.

    Ignores whatever constructor args the real code passes (api_key, base_url,
    timeout, ...) and always returns a FakeOpenAIClient seeded with `responses`.
    """

    def _factory(*_args, **_kwargs) -> FakeOpenAIClient:
        return FakeOpenAIClient(responses)

    return _factory


class _FakeSyncStream:
    """Sync iterable yielding a single chunk carrying the whole content.

    Mirrors `_FakeStream` above but for `agents.critic_llm`'s SYNC `OpenAI`
    client (`for chunk in stream`, not `async for`) -- `call_qwen_json`'s
    `_create_with_retry` uses `client.chat.completions.create(**kwargs,
    stream=True)` against the sync client, so its fake needs a plain
    `__iter__`, not `__aiter__`.
    """

    def __init__(self, content: str):
        self._content = content

    def __iter__(self):
        yield _FakeChunk(self._content)


class FakeSyncOpenAIClient:
    """Stand-in for openai.OpenAI's chat.completions.create, streaming shape.

    Sync counterpart of `FakeOpenAIClient`, for `agents.critic_llm.call_qwen_json`
    (Body-Checker, CTA-Checker, Tone-Checker, Meta-Critic all route through it).
    That module has no `client=` injection param -- it builds its own `OpenAI()`
    internally via `_client()` -- so tests must monkeypatch
    `agents.critic_llm.OpenAI` itself with `make_fake_sync_openai(...)`, not pass
    an instance directly.
    """

    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self.call_count = 0
        self.chat = self
        self.completions = self

    def create(self, model: str, messages: list[dict], **_kwargs) -> _FakeSyncStream:
        content = self._responses[min(self.call_count, len(self._responses) - 1)]
        self.call_count += 1
        return _FakeSyncStream(content)


def make_fake_sync_openai(responses: list[str]) -> Callable[..., FakeSyncOpenAIClient]:
    """Factory for monkeypatching `agents.critic_llm.OpenAI` itself.

    Ignores whatever constructor args the real code passes (api_key, base_url)
    and always returns a FakeSyncOpenAIClient seeded with `responses`.
    """

    def _factory(*_args, **_kwargs) -> FakeSyncOpenAIClient:
        return FakeSyncOpenAIClient(responses)

    return _factory


class ContentRoutedSyncOpenAIClient:
    """Sync fake for `agents.critic_llm.OpenAI` that routes by SYSTEM prompt text.

    `critic_llm.call_qwen_json` builds a fresh `OpenAI()` on EVERY call, so every
    checker that routes through it (Body §5.4.3, CTA §5.4.4, Tone §5.4.5,
    Meta-Critic §5.4.6) gets its own brand-new instance with call_count==0 — a
    flat `FakeSyncOpenAIClient` response list would therefore serve every one of
    them `responses[0]`, which cannot satisfy four different expected JSON shapes.

    This variant instead inspects each call's system prompt (unique per checker)
    and returns the first `routes` entry whose `needle` substring is present. That
    also makes it robust to the non-deterministic order LangGraph runs the
    parallel checker branches in. Purely additive: it does not touch
    `FakeSyncOpenAIClient` / `make_fake_sync_openai`, whose behaviour is unchanged.
    """

    def __init__(self, routes: list[tuple[str, str]]):
        self._routes = list(routes)
        self.chat = self
        self.completions = self

    def create(self, model: str, messages: list[dict], **_kwargs) -> _FakeSyncStream:  # noqa: ARG002
        system_prompt = messages[0]["content"] if messages else ""
        for needle, content in self._routes:
            if needle in system_prompt:
                return _FakeSyncStream(content)
        raise AssertionError(
            f"ContentRoutedSyncOpenAIClient: no route matched system prompt starting "
            f"{system_prompt[:80]!r}"
        )


def make_content_routed_sync_openai(
    routes: list[tuple[str, str]],
) -> Callable[..., ContentRoutedSyncOpenAIClient]:
    """Factory for monkeypatching `agents.critic_llm.OpenAI` with prompt routing.

    `routes` is a list of (needle, response_content) pairs; each call returns the
    response whose needle appears in the call's system prompt. See
    ContentRoutedSyncOpenAIClient for why routing (not a flat list) is required.
    """

    def _factory(*_args, **_kwargs) -> ContentRoutedSyncOpenAIClient:
        return ContentRoutedSyncOpenAIClient(routes)

    return _factory
