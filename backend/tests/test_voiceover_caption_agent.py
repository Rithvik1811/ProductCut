"""
Unit tests for the Voiceover + Caption Agent (§5.11).

Covers: happy-path generation + OSS upload (real ffmpeg concat/probe, gated on
ffmpeg/ffprobe being on PATH -- same `_skip_no_ffmpeg` convention
test_ken_burns_fallback_node.py established), caption-timing correctness (real
measured durations) and the word-count estimation heuristic for a permanently
failed beat, the per-beat retry-then-degrade failure path (failure_reason /
silent_beat_indices), the real Qwen3-TTS-Flash response-shape handling
(url vs. base64 `data`, non-200 status), and the node wrapper's `vo_ready`
event + reasoning trace.

Every test injects a fake `synth_fn` (or mocks `SpeechSynthesizer.call`
directly for the response-shape tests) -- no real DashScope call is ever made.
"""
from __future__ import annotations

import base64
import json
import os
import shutil
import subprocess

import pytest
from langchain_core.runnables import RunnableLambda

from agents.voiceover_caption_agent import (
    FAILURE_TYPE_API_ERROR,
    FAILURE_TYPE_TIMEOUT,
    MIN_ESTIMATED_BEAT_SEC,
    VoiceoverAPIError,
    VoiceoverTimeoutError,
    _call_qwen_tts_sync,
    _estimate_duration_sec,
    _synthesize_beat,
    generate_voiceover,
    voiceover_caption_agent_node,
)
from agents.pacing_checker import WORDS_PER_SECOND

_HAS_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None
_skip_no_ffmpeg = pytest.mark.skipif(not _HAS_FFMPEG, reason="ffmpeg/ffprobe not on PATH")


def _make_real_audio(tmp_path, name: str, duration_sec: float) -> str:
    """A real short audio clip via ffmpeg's own lavfi sine source -- exercises
    the actual ffprobe/concat pipeline instead of faking bytes, matching this
    repo's established preference for real ffmpeg execution coverage."""
    path = str(tmp_path / name)
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", f"sine=frequency=440:duration={duration_sec}",
         "-ac", "2", "-ar", "44100", path],
        check=True, capture_output=True,
    )
    return path


def _winning_script(lines: list[str], beat_len_sec: float = 2.0) -> dict:
    beats = []
    t = 0.0
    for line in lines:
        beats.append({"t_start": t, "t_end": t + beat_len_sec, "line": line})
        t += beat_len_sec
    return {"text": " ".join(lines), "beats": beats, "source_variant_ids": ["v1"]}


class _RecordingUploader:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def __call__(self, local_path: str) -> str:
        assert os.path.exists(local_path)  # must still exist at call time
        self.calls.append(local_path)
        return f"https://oss.example.invalid/{os.path.basename(local_path)}"


# ---------------------------------------------------------------------------
# _estimate_duration_sec (requirement 2's word-count heuristic)
# ---------------------------------------------------------------------------
def test_estimate_duration_uses_pacing_checkers_words_per_second():
    text = "one two three four five six seven"  # 7 words
    assert _estimate_duration_sec(text) == pytest.approx(7 / WORDS_PER_SECOND)


def test_estimate_duration_floors_at_minimum_for_near_empty_line():
    assert _estimate_duration_sec("hi") == MIN_ESTIMATED_BEAT_SEC
    assert _estimate_duration_sec("") == MIN_ESTIMATED_BEAT_SEC


# ---------------------------------------------------------------------------
# _synthesize_beat: exactly one retry, then degrade (requirement 5)
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_synthesize_beat_succeeds_first_try():
    calls = []

    async def fn(text):
        calls.append(text)
        return "/tmp/fake.mp3"

    path, failure = await _synthesize_beat("hello", fn)
    assert path == "/tmp/fake.mp3"
    assert failure is None
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_synthesize_beat_recovers_on_retry():
    calls = []

    async def fn(text):
        calls.append(text)
        if len(calls) == 1:
            raise VoiceoverAPIError("transient")
        return "/tmp/fake.mp3"

    path, failure = await _synthesize_beat("hello", fn)
    assert path == "/tmp/fake.mp3"
    assert failure is None
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_synthesize_beat_degrades_after_two_failures():
    calls = []

    async def fn(text):
        calls.append(text)
        raise VoiceoverTimeoutError("never responds")

    path, failure = await _synthesize_beat("hello", fn)
    assert path is None
    assert failure == {"type": FAILURE_TYPE_TIMEOUT, "detail": "never responds"}
    assert len(calls) == 2  # exactly one retry, never more


# ---------------------------------------------------------------------------
# _call_qwen_tts_sync: real confirmed Qwen3-TTS-Flash response shape
# (dashscope.audio.qwen_tts.speech_synthesizer.TextToSpeechResponse), url vs.
# base64 data, and a non-200 status.
#
# `_fake_audio()` returns a plain dict, NOT an attribute-accessible object --
# this matches the REAL confirmed shape (derisk/test_tts_smoke.py found
# `type(resp.output.audio) is dict` live), and deliberately replaces an
# earlier version of this fixture that used a `self.url = url` attribute
# object. That earlier fixture was WRONG and let a real bug ship:
# `_call_qwen_tts_sync` used `getattr(audio, "url", None)`, which always
# silently returned `None` against the real dict shape, so every real
# production call failed with "neither url nor data" despite the API
# genuinely succeeding -- these tests passed the whole time because the fake
# didn't reproduce the real shape. Fixed in both `_extract_audio_field` (dict
# access first) and here (a realistic fake).
# ---------------------------------------------------------------------------
def _fake_audio(url=None, data=None) -> dict:
    return {"url": url, "data": data, "expires_at": 0, "id": "test-audio-id"}


class _FakeOutput:
    def __init__(self, audio):
        self.audio = audio


class _FakeResponse:
    def __init__(self, status_code, output=None, code="", message=""):
        self.status_code = status_code
        self.output = output
        self.code = code
        self.message = message


def test_call_qwen_tts_sync_downloads_from_url(monkeypatch, tmp_path):
    monkeypatch.setenv("MODEL_TTS", "qwen3-tts-flash")
    downloaded = tmp_path / "clip.mp3"
    downloaded.write_bytes(b"real-bytes")

    def fake_call(**kwargs):
        assert kwargs["model"] == "qwen3-tts-flash"
        assert kwargs["text"] == "hello there"
        return _FakeResponse(200, output=_FakeOutput(_fake_audio(url="http://tts.example/clip.mp3")))

    monkeypatch.setattr("agents.voiceover_caption_agent.SpeechSynthesizer.call", fake_call)
    monkeypatch.setattr(
        "agents.voiceover_caption_agent._download_to_temp",
        lambda url: str(downloaded),
    )

    path = _call_qwen_tts_sync("hello there")
    assert path == str(downloaded)


def test_call_qwen_tts_sync_decodes_base64_data(monkeypatch, tmp_path):
    monkeypatch.setenv("MODEL_TTS", "qwen3-tts-flash")
    raw = b"decoded-audio-bytes"
    encoded = base64.b64encode(raw).decode("ascii")

    def fake_call(**kwargs):
        return _FakeResponse(200, output=_FakeOutput(_fake_audio(data=encoded)))

    monkeypatch.setattr("agents.voiceover_caption_agent.SpeechSynthesizer.call", fake_call)

    path = _call_qwen_tts_sync("hello")
    try:
        with open(path, "rb") as fh:
            assert fh.read() == raw
    finally:
        os.remove(path)


def test_call_qwen_tts_sync_raises_on_non_200(monkeypatch):
    monkeypatch.setenv("MODEL_TTS", "qwen3-tts-flash")

    def fake_call(**kwargs):
        return _FakeResponse(400, code="InvalidParameter", message="bad voice")

    monkeypatch.setattr("agents.voiceover_caption_agent.SpeechSynthesizer.call", fake_call)

    with pytest.raises(VoiceoverAPIError, match="InvalidParameter"):
        _call_qwen_tts_sync("hello")


def test_call_qwen_tts_sync_raises_when_audio_has_neither_url_nor_data(monkeypatch):
    monkeypatch.setenv("MODEL_TTS", "qwen3-tts-flash")

    def fake_call(**kwargs):
        return _FakeResponse(200, output=_FakeOutput(_fake_audio()))

    monkeypatch.setattr("agents.voiceover_caption_agent.SpeechSynthesizer.call", fake_call)

    with pytest.raises(VoiceoverAPIError, match="neither url nor data"):
        _call_qwen_tts_sync("hello")


def test_call_qwen_tts_sync_treats_empty_string_data_as_absent(monkeypatch, tmp_path):
    """Regression guard for the real confirmed response shape: a live call's
    `data` key comes back as `""` (empty string), not absent/None, when only
    `url` is populated -- must not be treated as truthy "real" data."""
    monkeypatch.setenv("MODEL_TTS", "qwen3-tts-flash")
    downloaded = tmp_path / "clip.wav"
    downloaded.write_bytes(b"real-bytes")

    def fake_call(**kwargs):
        return _FakeResponse(
            200, output=_FakeOutput(_fake_audio(url="http://tts.example/clip.wav", data=""))
        )

    monkeypatch.setattr("agents.voiceover_caption_agent.SpeechSynthesizer.call", fake_call)
    monkeypatch.setattr("agents.voiceover_caption_agent._download_to_temp", lambda url: str(downloaded))

    path = _call_qwen_tts_sync("hello")
    assert path == str(downloaded)


# ---------------------------------------------------------------------------
# generate_voiceover: happy path -- real ffmpeg concat/probe + OSS upload
# ---------------------------------------------------------------------------
@_skip_no_ffmpeg
@pytest.mark.asyncio
async def test_generate_voiceover_happy_path_real_audio(tmp_path):
    lines = ["Your coffee is cold in twelve minutes.", "Double wall keeps it hot for hours.", "Buy now."]
    durations = [1.5, 2.0, 0.8]

    async def synth_fn(text):
        idx = lines.index(text)
        return _make_real_audio(tmp_path, f"beat{idx}.wav", durations[idx])

    audio_uploader = _RecordingUploader()
    captions_uploader = _RecordingUploader()

    voiceover, captions = await generate_voiceover(
        _winning_script(lines),
        "job-vo-1",
        synth_fn=synth_fn,
        upload_audio_fn=audio_uploader,
        upload_captions_fn=captions_uploader,
    )

    assert voiceover["audio_uri"] == "https://oss.example.invalid/" + os.path.basename(audio_uploader.calls[0])
    assert voiceover["caption_track_uri"] == "https://oss.example.invalid/" + os.path.basename(captions_uploader.calls[0])
    assert "failure_reason" not in voiceover
    assert "silent_beat_indices" not in voiceover

    assert len(captions) == 3
    assert [c["text"] for c in captions] == lines
    # Real measured durations (not an even split, not the script's own 2s-per-beat t_start/t_end).
    assert captions[0]["start_ts"] == 0.0
    assert captions[0]["end_ts"] == pytest.approx(durations[0], abs=0.15)
    assert captions[1]["start_ts"] == pytest.approx(durations[0], abs=0.15)
    assert captions[1]["end_ts"] == pytest.approx(durations[0] + durations[1], abs=0.2)
    assert captions[2]["start_ts"] == pytest.approx(durations[0] + durations[1], abs=0.2)

    # temp files cleaned up
    for path in audio_uploader.calls + captions_uploader.calls:
        assert not os.path.exists(path)


@_skip_no_ffmpeg
@pytest.mark.asyncio
async def test_generate_voiceover_uploads_valid_captions_json(tmp_path):
    lines = ["Hook line here.", "Call to action now."]

    async def synth_fn(text):
        return _make_real_audio(tmp_path, f"{abs(hash(text))}.wav", 1.0)

    captured_json = {}

    def capture_captions(local_path):
        with open(local_path, "r", encoding="utf-8") as fh:
            captured_json["value"] = json.load(fh)
        return "https://oss.example.invalid/captions.json"

    voiceover, captions = await generate_voiceover(
        _winning_script(lines),
        "job-vo-2",
        synth_fn=synth_fn,
        upload_audio_fn=lambda p: "https://oss.example.invalid/voiceover.mp3",
        upload_captions_fn=capture_captions,
    )

    assert captured_json["value"] == captions
    assert all(set(entry.keys()) == {"text", "start_ts", "end_ts"} for entry in captured_json["value"])


# ---------------------------------------------------------------------------
# generate_voiceover: failure path -- one beat permanently fails, degrades to
# a silent gap with an estimated-duration caption; siblings keep real audio.
# ---------------------------------------------------------------------------
@_skip_no_ffmpeg
@pytest.mark.asyncio
async def test_generate_voiceover_degrades_one_failed_beat(tmp_path):
    lines = ["This beat synthesizes fine.", "This beat always fails to synthesize completely."]
    good_duration = 1.4

    async def synth_fn(text):
        if text == lines[1]:
            raise VoiceoverAPIError("simulated persistent TTS failure")
        return _make_real_audio(tmp_path, "good.wav", good_duration)

    voiceover, captions = await generate_voiceover(
        _winning_script(lines),
        "job-vo-3",
        synth_fn=synth_fn,
        upload_audio_fn=lambda p: "https://oss.example.invalid/voiceover.mp3",
        upload_captions_fn=lambda p: "https://oss.example.invalid/captions.json",
    )

    assert voiceover["failure_reason"] == {"type": FAILURE_TYPE_API_ERROR, "detail": "simulated persistent TTS failure"}
    assert voiceover["silent_beat_indices"] == [1]
    # audio_uri still produced (the doc's "assemble with captions only" --
    # implemented as a real, fully-formed track with a silent gap, not a
    # missing/optional key).
    assert voiceover["audio_uri"] == "https://oss.example.invalid/voiceover.mp3"

    assert captions[0]["end_ts"] == pytest.approx(good_duration, abs=0.15)
    expected_estimate = len(lines[1].split()) / WORDS_PER_SECOND
    assert captions[1]["end_ts"] - captions[1]["start_ts"] == pytest.approx(expected_estimate, abs=0.01)
    assert captions[1]["start_ts"] == captions[0]["end_ts"]


@_skip_no_ffmpeg
@pytest.mark.asyncio
async def test_generate_voiceover_all_beats_fail_still_produces_full_silent_track(tmp_path):
    lines = ["Always fails one.", "Always fails two."]

    async def synth_fn(text):
        raise VoiceoverTimeoutError("simulated total outage")

    voiceover, captions = await generate_voiceover(
        _winning_script(lines),
        "job-vo-4",
        synth_fn=synth_fn,
        upload_audio_fn=lambda p: "https://oss.example.invalid/voiceover.mp3",
        upload_captions_fn=lambda p: "https://oss.example.invalid/captions.json",
    )

    assert voiceover["silent_beat_indices"] == [0, 1]
    assert voiceover["failure_reason"]["type"] == FAILURE_TYPE_TIMEOUT
    assert voiceover["audio_uri"] == "https://oss.example.invalid/voiceover.mp3"
    assert len(captions) == 2


@pytest.mark.asyncio
async def test_generate_voiceover_empty_beats_degrades_gracefully():
    """A malformed winning_script with no beats must never crash the node --
    produces a minimal silent placeholder instead."""
    empty_script = {"text": "", "beats": [], "source_variant_ids": []}

    async def synth_fn(text):
        raise AssertionError("should never be called for an empty beat list")

    voiceover, captions = await generate_voiceover(
        empty_script,
        "job-vo-5",
        synth_fn=synth_fn,
        upload_audio_fn=lambda p: "https://oss.example.invalid/voiceover.mp3",
        upload_captions_fn=lambda p: "https://oss.example.invalid/captions.json",
    )
    assert voiceover["audio_uri"] == "https://oss.example.invalid/voiceover.mp3"
    assert len(captions) == 1


# ---------------------------------------------------------------------------
# voiceover_caption_agent_node: state I/O, vo_ready event, reasoning trace.
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_node_requires_winning_script(monkeypatch):
    state = {"job_id": "job-1"}
    with pytest.raises(KeyError):
        await voiceover_caption_agent_node(state)


@pytest.mark.asyncio
async def test_node_happy_path_writes_voiceover_and_trace(monkeypatch):
    async def fake_generate(winning_script, job_id, **kwargs):
        assert job_id == "job-node-1"
        return {"audio_uri": "https://oss.example.invalid/a.mp3", "caption_track_uri": "https://oss.example.invalid/c.json"}, [
            {"text": "hi", "start_ts": 0.0, "end_ts": 1.0}
        ]

    monkeypatch.setattr("agents.voiceover_caption_agent.generate_voiceover", fake_generate)

    state = {
        "job_id": "job-node-1",
        "winning_script": _winning_script(["hi"]),
        "reasoning_trace": "",
    }
    result = await RunnableLambda(voiceover_caption_agent_node).ainvoke(state)

    assert result["voiceover"]["audio_uri"] == "https://oss.example.invalid/a.mp3"
    assert "synthesized 1 caption beat(s)" in result["reasoning_trace"]
    assert "silent" not in result["reasoning_trace"]


@pytest.mark.asyncio
async def test_node_trace_mentions_degradation_when_present(monkeypatch):
    async def fake_generate(winning_script, job_id, **kwargs):
        return {
            "audio_uri": "https://oss.example.invalid/a.mp3",
            "caption_track_uri": "https://oss.example.invalid/c.json",
            "failure_reason": {"type": FAILURE_TYPE_API_ERROR, "detail": "boom"},
            "silent_beat_indices": [0],
        }, [{"text": "hi", "start_ts": 0.0, "end_ts": 1.0}]

    monkeypatch.setattr("agents.voiceover_caption_agent.generate_voiceover", fake_generate)

    state = {"job_id": "job-node-2", "winning_script": _winning_script(["hi"]), "reasoning_trace": ""}
    result = await RunnableLambda(voiceover_caption_agent_node).ainvoke(state)

    assert "1 beat(s) silent" in result["reasoning_trace"]
    assert "api_error" in result["reasoning_trace"]


@pytest.mark.asyncio
async def test_node_emits_vo_ready_event(monkeypatch):
    async def fake_generate(winning_script, job_id, **kwargs):
        return {"audio_uri": "https://oss.example.invalid/a.mp3", "caption_track_uri": "https://oss.example.invalid/c.json"}, [
            {"text": "a", "start_ts": 0.0, "end_ts": 1.0},
            {"text": "b", "start_ts": 1.0, "end_ts": 2.0},
        ]

    monkeypatch.setattr("agents.voiceover_caption_agent.generate_voiceover", fake_generate)

    state = {"job_id": "job-node-3", "winning_script": _winning_script(["a", "b"]), "reasoning_trace": ""}
    events = [
        e
        async for e in RunnableLambda(voiceover_caption_agent_node).astream_events(state, version="v2")
        if e.get("event") == "on_custom_event" and e.get("name") == "vo_ready"
    ]

    assert len(events) == 1
    payload = events[0]["data"]
    assert payload["caption_count"] == 2
    assert payload["degraded"] is False
    assert payload["voiceover"]["audio_uri"] == "https://oss.example.invalid/a.mp3"


@pytest.mark.asyncio
async def test_node_vo_ready_event_flags_degraded_true(monkeypatch):
    async def fake_generate(winning_script, job_id, **kwargs):
        return {
            "audio_uri": "https://oss.example.invalid/a.mp3",
            "caption_track_uri": "https://oss.example.invalid/c.json",
            "failure_reason": {"type": FAILURE_TYPE_TIMEOUT, "detail": "boom"},
            "silent_beat_indices": [0],
        }, [{"text": "a", "start_ts": 0.0, "end_ts": 1.0}]

    monkeypatch.setattr("agents.voiceover_caption_agent.generate_voiceover", fake_generate)

    state = {"job_id": "job-node-4", "winning_script": _winning_script(["a"]), "reasoning_trace": ""}
    events = [
        e
        async for e in RunnableLambda(voiceover_caption_agent_node).astream_events(state, version="v2")
        if e.get("event") == "on_custom_event" and e.get("name") == "vo_ready"
    ]

    assert events[0]["data"]["degraded"] is True
