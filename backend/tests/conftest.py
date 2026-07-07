"""Shared test fixtures. Keeps tests hermetic -- no dependency on a real .env."""
import pytest


@pytest.fixture(autouse=True)
def _fake_dashscope_env(monkeypatch):
    """Every test gets dummy env vars, regardless of what's in the real .env.

    Tests inject a fake client directly, so these values are never actually
    used to make a network call -- they just need to exist so os.environ[...]
    lookups in the agent modules don't KeyError.
    """
    monkeypatch.setenv("MODEL_VISION", "test-vision-model")
    monkeypatch.setenv("MODEL_TEXT", "test-text-model")
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")
    monkeypatch.setenv("DASHSCOPE_BASE_URL", "https://example.invalid/compatible-mode/v1")
    monkeypatch.setenv("OSS_ACCESS_KEY_ID", "test-oss-key")
    monkeypatch.setenv("OSS_ACCESS_KEY_SECRET", "test-oss-secret")
    monkeypatch.setenv("OSS_ENDPOINT", "https://oss.example.invalid")
    monkeypatch.setenv("OSS_BUCKET", "test-bucket")
    # Force MemorySaver in graph.build regardless of what the real shell has
    # exported -- these tests must never touch a real Postgres instance.
    monkeypatch.delenv("DATABASE_URL", raising=False)
