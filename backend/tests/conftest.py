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
    # Voiceover + Caption Agent's TTS calls use a separate, dedicated key/base
    # URL (this account's TTS access is scoped to a different DashScope
    # region/workspace than text/vision/video -- see
    # agents/voiceover_caption_agent.py's module docstring). Tests always
    # inject a fake synth_fn/mock the SDK call directly, so this value is
    # never actually used to make a network call -- it just needs to exist.
    monkeypatch.setenv("DASHSCOPE_TTS_API_KEY", "test-tts-key")
    monkeypatch.setenv("DASHSCOPE_TTS_BASE_URL", "https://example.invalid/compatible-mode/v1")
    monkeypatch.setenv("OSS_ACCESS_KEY_ID", "test-oss-key")
    monkeypatch.setenv("OSS_ACCESS_KEY_SECRET", "test-oss-secret")
    monkeypatch.setenv("OSS_ENDPOINT", "https://oss.example.invalid")
    monkeypatch.setenv("OSS_BUCKET", "test-bucket")
    # Force MemorySaver in graph.build regardless of what the real shell has
    # exported -- these tests must never touch a real Postgres instance.
    monkeypatch.delenv("DATABASE_URL", raising=False)
    # app/main.py runs load_dotenv() at import, which leaks the real Singapore
    # video key into os.environ for the whole session. The Video-Gen Node's
    # T2I scene generator activates whenever this key is present, and would then
    # make a real wan2.7-image-pro network call even in tests that inject a fake
    # generate_fn. Clear it (same hermetic posture as DATABASE_URL) so the T2I
    # path stays off unless a test explicitly opts in.
    monkeypatch.delenv("DASHSCOPE_VIDEO_INTL_API_KEY", raising=False)
