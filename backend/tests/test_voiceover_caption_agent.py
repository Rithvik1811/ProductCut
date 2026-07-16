"""
Unit tests for the Voiceover + Caption Agent (§5.11).

Covers: happy-path generation + OSS upload (real ffmpeg concat/probe, gated on
ffmpeg/ffprobe being on PATH -- same `_skip_no_ffmpeg` convention
test_ken_burns_fallback_node.py established), caption-timing correctness (real
measured durations) and the word-count estimation heuristic for a permanently
failed beat, the per-beat retry-then-degrade failure path (failure_reason /
silent_beat_indices), the CosyVoice sync call (mocked dashscope SDK), directed-beat
spoken_text/emotion/pacing selection, and the node wrapper's `vo_ready` event +
reasoning trace.

Every test injects a fake `synth_fn` (or mocks the dashscope SDK client
directly for the sync-call test) -- no real CosyVoice API call is ever made.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess

import pytest
from langchain_core.runnables import RunnableLambda

from agents.voiceover_caption_agent import (
    COSYVOICE_VOICE_ID,
    FAILURE_TYPE_API_ERROR,
    FAILURE_TYPE_TIMEOUT,
    MIN_ESTIMATED_BEAT_SEC,
    VoiceoverAPIError,
    VoiceoverTimeoutError,
    _call_cosyvoice_sync,
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
# _call_cosyvoice_sync: calls SpeechSynthesizer.call() and writes returned
# bytes to a temp MP3 file. The dashscope SDK is mocked so no real API call
# is made.
# ---------------------------------------------------------------------------
def test_call_cosyvoice_sync_creates_temp_file(monkeypatch):
    """_call_cosyvoice_sync calls SpeechSynthesizer.call() and writes MP3 bytes."""
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")

    fake_audio = b"fake-cosyvoice-mp3-data"
    captured = {}

    class _FakeSynthesizer:
        def __init__(self, **kwargs):
            captured["model"] = kwargs.get("model")
            captured["voice"] = kwargs.get("voice")
            captured["speech_rate"] = kwargs.get("speech_rate")

        def call(self, text):
            captured["text"] = text
            return fake_audio

    import dashscope.audio.tts_v2 as tts_module
    monkeypatch.setattr(tts_module, "SpeechSynthesizer", _FakeSynthesizer)

    path = _call_cosyvoice_sync("hello world", "fast")
    try:
        with open(path, "rb") as fh:
            assert fh.read() == fake_audio
        assert captured["voice"] == COSYVOICE_VOICE_ID
        assert captured["text"] == "hello world"
        assert captured["speech_rate"] == pytest.approx(1.10)
    finally:
        os.remove(path)


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
# generate_voiceover with directed_beats: spoken_text + emotion + pacing are
# used to build each beat's CosyVoice fn (verified by patching _make_cosyvoice_fn).
# ---------------------------------------------------------------------------
@_skip_no_ffmpeg
@pytest.mark.asyncio
async def test_generate_voiceover_uses_directed_beats_spoken_text(tmp_path):
    """When directed_beats is provided (and no synth_fn injection), each beat
    uses its spoken_text + emotion + pacing, not the raw winning_script line."""
    import agents.voiceover_caption_agent as vca_module

    captured = []
    original_make = vca_module._make_cosyvoice_fn

    def capturing_make(pacing="normal"):
        async def _fn(text):
            captured.append({"text": text, "pacing": pacing})
            return _make_real_audio(tmp_path, f"cap_{len(captured)}.wav", 0.3)
        return _fn

    vca_module._make_cosyvoice_fn = capturing_make
    try:
        directed = [
            {"beat_index": 0, "spoken_text": "spoken one", "emotion": "excited", "pacing": "fast"},
            {"beat_index": 1, "spoken_text": "spoken two", "emotion": "warm", "pacing": "normal"},
        ]
        vo, caps = await generate_voiceover(
            _winning_script(["raw line one", "raw line two"]),
            "test-job",
            directed_beats=directed,
            upload_audio_fn=lambda p: "https://oss.example.invalid/voiceover.mp3",
            upload_captions_fn=lambda p: "https://oss.example.invalid/captions.json",
        )
        assert captured[0]["text"] == "spoken one"
        assert captured[0]["pacing"] == "fast"
        assert captured[1]["text"] == "spoken two"
        assert captured[1]["pacing"] == "normal"
        # captions carry the spoken_text, not the raw line
        assert [c["text"] for c in caps] == ["spoken one", "spoken two"]
    finally:
        vca_module._make_cosyvoice_fn = original_make


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
        "voiceover_reasoning_trace": "",
    }
    result = await RunnableLambda(voiceover_caption_agent_node).ainvoke(state)

    assert result["voiceover"]["audio_uri"] == "https://oss.example.invalid/a.mp3"
    assert "synthesized 1 caption beat(s)" in result["voiceover_reasoning_trace"]
    assert "silent" not in result["voiceover_reasoning_trace"]


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

    state = {"job_id": "job-node-2", "winning_script": _winning_script(["hi"]), "voiceover_reasoning_trace": ""}
    result = await RunnableLambda(voiceover_caption_agent_node).ainvoke(state)

    assert "1 beat(s) silent" in result["voiceover_reasoning_trace"]
    assert "api_error" in result["voiceover_reasoning_trace"]


@pytest.mark.asyncio
async def test_node_emits_vo_ready_event(monkeypatch):
    async def fake_generate(winning_script, job_id, **kwargs):
        return {"audio_uri": "https://oss.example.invalid/a.mp3", "caption_track_uri": "https://oss.example.invalid/c.json"}, [
            {"text": "a", "start_ts": 0.0, "end_ts": 1.0},
            {"text": "b", "start_ts": 1.0, "end_ts": 2.0},
        ]

    monkeypatch.setattr("agents.voiceover_caption_agent.generate_voiceover", fake_generate)

    state = {"job_id": "job-node-3", "winning_script": _winning_script(["a", "b"]), "voiceover_reasoning_trace": ""}
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

    state = {"job_id": "job-node-4", "winning_script": _winning_script(["a"]), "voiceover_reasoning_trace": ""}
    events = [
        e
        async for e in RunnableLambda(voiceover_caption_agent_node).astream_events(state, version="v2")
        if e.get("event") == "on_custom_event" and e.get("name") == "vo_ready"
    ]

    assert events[0]["data"]["degraded"] is True
