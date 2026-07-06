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
