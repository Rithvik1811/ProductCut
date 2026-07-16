"""
Voiceover + Caption Agent -- CosyVoice v2 (cosyvoice-v2, DashScope international)
with per-beat pacing control (Phase 5).
Spec of record: docs/TECHNICAL_DOCUMENTATION.md §5.11.

TTS ENGINE: CosyVoice v2 (DashScope native), replacing ElevenLabs.
Keeps the entire hackathon stack within the Qwen/Alibaba Cloud ecosystem.
NOTE: cosyvoice-v3 (which supports instruct_text emotion control) is only
available in the Beijing/mainland region. cosyvoice-v2 is used here because
this account runs on the international (US/Singapore) region. The Voice
Direction Agent (agents/voice_direction_agent.py) still rewrites each
ScriptBeat.line into natural spoken English and assigns emotion + pacing --
the spoken_text rewrite reaches TTS directly (more natural phrasing, better
TTS prosody), and pacing maps to cosyvoice-v2's speech_rate parameter.
Emotion tagging is preserved in state for forward-compatibility with v3 once
the international region supports it.
When no directed beats are present (e.g. the Voice Direction Agent was
skipped), falls back to the raw beat lines at normal speech_rate.

Confirmed against the ACTUAL merged code:
  * `graph.state.WinningScript` (C1, frozen): {text, beats: list[ScriptBeat],
    source_variant_ids}. `ScriptBeat` = {t_start, t_end, line}. There is no
    `beat_function` on a ScriptBeat; beat segmentation here is keyed on beat
    INDEX only.
  * `graph.state.DirectedBeat` (C1 v12): {beat_index, spoken_text, emotion,
    pacing}. `spoken_text` is the natural spoken rewrite fed to CosyVoice;
    `emotion` selects the Chinese instruct_text; `pacing` sets speech_rate.
  * `graph.state.Voiceover` (C1, frozen): `{audio_uri, caption_track_uri}` --
    both plain strings (OSS refs). The `{text, start_ts, end_ts}` caption array
    is serialized to a file, uploaded to OSS, and only the resulting signed URL
    lands in state, as `caption_track_uri`.

PARALLEL BRANCH, NOT A DOWNSTREAM STEP OF VIDEO-GEN. This module's entry point,
`voiceover_caption_agent_node`, reads ONLY `state["winning_script"]`,
`state["directed_script_beats"]`, and `state["job_id"]` -- never `treatment`,
`shot_list`, `budget_ledger`, or `generated_shots` -- and writes only
`state["voiceover"]` + `voiceover_reasoning_trace` (its OWN dedicated trace key,
not the shared `reasoning_trace`; see graph/state.py's v7 changelog).

WIRING (graph/build.py): merge_validator's "finalize"/"fallback" routes fan out
to BOTH visual_direction_agent AND voice_direction_agent (parallel sub-branches).
voice_direction_agent -> voiceover_caption_agent -> assembly_agent (defer=True).

REQUIREMENT 1 -- PER-BEAT vs. WHOLE-SCRIPT TTS. Chosen: PER-BEAT (one
CosyVoice call per beat). Why: docs §5.11's failure-handling sentence ("If
synthesis fails for a line, the node retries that line") only makes sense at
per-line granularity; per-beat also gives natural pause control at beat
boundaries, per-beat emotion/pacing tuning, and an exact (ffprobe-measured)
duration per beat before concatenating.

REQUIREMENT 2 -- CAPTION GRANULARITY AND TIMING. One caption entry per beat.
For a beat whose synthesis SUCCEEDED, start_ts/end_ts come from that beat's REAL
measured clip duration (ffmpeg.probe, cumulative from 0). For a beat whose
synthesis PERMANENTLY failed, the span is ESTIMATED: len(text.split()) /
WORDS_PER_SECOND (reusing agents.pacing_checker.WORDS_PER_SECOND), floored at
MIN_ESTIMATED_BEAT_SEC.

REQUIREMENT 3 -- OSS PERSISTENCE. Reuses agents/_oss.py (upload_audio_to_oss /
upload_json_to_oss) exclusively.

REQUIREMENT 4 -- EVENT. Emits the proposed C2 `vo_ready` event
({voiceover, caption_count, degraded}) via adispatch_custom_event.

REQUIREMENT 5 -- FAILURE HANDLING / DEGRADE PATH. Per-beat retry once, then
degrade that single beat to a silent gap + estimated-timing caption, NEVER halt
the whole node/job over one bad line (docs §5.11). Each beat gets at most 2
attempts (1 initial + 1 retry) via `_synthesize_beat`; a beat that fails both
contributes a silent gap (ffmpeg anullsrc) of its estimated duration. The
returned `voiceover` dict always has audio_uri + caption_track_uri; when >=1
beat degraded it additionally carries `failure_reason` ({type, detail}, reusing
graph.state.FailureReason's frozen "timeout"/"api_error" vocabulary) and
`silent_beat_indices: list[int]`.

OUTPUT AUDIO FORMAT -- MP3 (libmp3lame); ffmpeg re-encodes every per-beat clip
through `aformat` before concatenation, so the output format is fixed
regardless of the per-beat source format.

CREDENTIALS: DashScope reads DASHSCOPE_API_KEY from the environment automatically
(same key used by every other DashScope call in this codebase). Voice and model
are env-overridable (COSYVOICE_MODEL_ID / COSYVOICE_VOICE_ID). See .env.example.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
from typing import Awaitable, Callable, Optional

import ffmpeg
from langchain_core.callbacks.manager import adispatch_custom_event
from langchain_core.runnables import RunnableConfig

from agents._oss import upload_audio_to_oss, upload_json_to_oss
from agents.pacing_checker import WORDS_PER_SECOND
from graph.state import DirectedBeat, ProductCutState, WinningScript

logger = logging.getLogger("productcut.agents.voiceover_caption_agent")

# CosyVoice v2 model/voice, env-overridable. cosyvoice-v2 is the international-
# region model (v3 is Beijing-only). longxiaochun_v2 is a deep commercial
# narrator; other options: longyingxiao (female), longjiqi (male).
COSYVOICE_MODEL_ID = os.getenv("COSYVOICE_MODEL_ID", "cosyvoice-v2")
COSYVOICE_VOICE_ID = os.getenv("COSYVOICE_VOICE_ID", "longxiaochun_v2")

# CosyVoice synthesizes one short line per call -- generous margin above the
# handful-of-seconds a single beat's line should realistically take.
DEFAULT_COSYVOICE_TIMEOUT_SEC = float(os.getenv("VOICEOVER_TTS_TIMEOUT_SEC", "45"))

# Floor so a near-empty/blank line never estimates to ~0s (requirement 2).
MIN_ESTIMATED_BEAT_SEC = 0.5

# Reused verbatim from graph.state.FailureReason's frozen Literal -- see
# module docstring's REQUIREMENT 5 note on why no new failure-type is invented.
FAILURE_TYPE_TIMEOUT = "timeout"
FAILURE_TYPE_API_ERROR = "api_error"

# Per-pacing speech_rate multiplier for CosyVoice v2 (1.0 = normal speed).
# cosyvoice-v2 does not support instruct_text emotion control (v3 only, and
# v3 is Beijing-region only). Pacing is the only TTS-level delivery control
# available on v2; emotion from the Voice Direction Agent still improves
# quality indirectly via the spoken_text rewrite that reaches TTS.
PACING_SPEECH_RATE: dict[str, float] = {
    "slow": 0.85,
    "normal": 1.0,
    "fast": 1.10,
}

_DEFAULT_EMOTION = "conversational"

_SILENCE_SAMPLE_RATE = 44100
_SILENCE_CHANNEL_LAYOUT = "stereo"


class VoiceoverTimeoutError(Exception):
    """A single beat's CosyVoice call never returned within the wait timeout."""


class VoiceoverAPIError(Exception):
    """The CosyVoice call returned a hard failure (SDK error or no usable audio)."""


# ---------------------------------------------------------------------------
# Real CosyVoice v3 call (dashscope SDK). One short line per call, tuned by
# the beat's emotion (EMOTION_INSTRUCTIONS instruct_text) and pacing
# (PACING_SPEECH_RATE speech_rate). Uses DASHSCOPE_API_KEY automatically.
# ---------------------------------------------------------------------------
def _call_cosyvoice_sync(
    text: str,
    pacing: str = "normal",
) -> str:
    """Blocking CosyVoice v2 synthesis of one line -> local temp MP3 path.

    Returns the local path to the synthesized audio (never a remote URL) so the
    caller can `ffmpeg.probe` its real duration and concatenate it with sibling
    beats. `pacing` controls the speech_rate multiplier; cosyvoice-v2 does not
    support instruct_text emotion control (that is v3-only, Beijing region).
    """
    import dashscope
    from dashscope.audio.tts_v2 import SpeechSynthesizer, AudioFormat

    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if api_key:
        dashscope.api_key = api_key

    speech_rate = PACING_SPEECH_RATE.get(pacing, 1.0)

    synthesizer = SpeechSynthesizer(
        model=COSYVOICE_MODEL_ID,
        voice=COSYVOICE_VOICE_ID,
        format=AudioFormat.MP3_22050HZ_MONO_256KBPS,
        speech_rate=speech_rate,
    )
    audio_bytes: bytes = synthesizer.call(text)
    if not audio_bytes:
        raise VoiceoverAPIError("CosyVoice returned empty audio bytes")

    fd, path = tempfile.mkstemp(suffix=".mp3", prefix="cosyvoice_tts_")
    with os.fdopen(fd, "wb") as fh:
        fh.write(audio_bytes)
    return path


async def _call_cosyvoice(
    text: str,
    pacing: str = "normal",
) -> str:
    """Async wrapper: the SDK call is blocking, so it runs in a thread with an
    explicit wait timeout (matching video_gen_node.py's posture on its own
    blocking-call timeout).
    """
    try:
        return await asyncio.wait_for(
            asyncio.to_thread(_call_cosyvoice_sync, text, pacing),
            timeout=DEFAULT_COSYVOICE_TIMEOUT_SEC,
        )
    except asyncio.TimeoutError as exc:
        raise VoiceoverTimeoutError(
            f"CosyVoice exceeded the {DEFAULT_COSYVOICE_TIMEOUT_SEC:.0f}s wait timeout"
        ) from exc
    except VoiceoverAPIError:
        raise
    except Exception as exc:  # noqa: BLE001 -- any transport/SDK error is a classified api_error, never crashes the batch
        raise VoiceoverAPIError(str(exc)) from exc


SynthesizeFn = Callable[[str], Awaitable[str]]


def _make_cosyvoice_fn(
    pacing: str = "normal",
) -> SynthesizeFn:
    """Closure factory: bind pacing into a single-arg `SynthesizeFn` so the
    per-beat retry logic (`_synthesize_beat`) stays unchanged (it only ever
    passes `text`). emotion is not passed to cosyvoice-v2 (no instruct_text
    support); it still shaped the spoken_text rewrite in voice_direction_agent."""
    async def _fn(text: str) -> str:
        return await _call_cosyvoice(text, pacing)

    return _fn


# ---------------------------------------------------------------------------
# Per-beat retry (requirement 5 -- "the node retries that line").
# ---------------------------------------------------------------------------
async def _synthesize_beat(text: str, synth_fn: SynthesizeFn) -> tuple[Optional[str], Optional[dict]]:
    """Returns (local_audio_path, failure_reason) -- exactly one of the two is
    set. Attempts the beat's line at most twice (1 initial + 1 retry); on a
    second failure returns (None, failure_reason) so the caller can degrade
    that single beat rather than aborting the whole job.
    """
    failure: Optional[dict] = None
    for attempt in (1, 2):
        try:
            return await synth_fn(text), None
        except VoiceoverTimeoutError as exc:
            failure = {"type": FAILURE_TYPE_TIMEOUT, "detail": str(exc)}
        except VoiceoverAPIError as exc:
            failure = {"type": FAILURE_TYPE_API_ERROR, "detail": str(exc)}
        except Exception as exc:  # noqa: BLE001 -- any other unexpected failure still degrades cleanly
            failure = {"type": FAILURE_TYPE_API_ERROR, "detail": str(exc)}
        logger.warning(
            "Voiceover: beat synthesis attempt %d/2 failed (%s) for line %r",
            attempt, failure["detail"], text[:80],
        )
    return None, failure


def _estimate_duration_sec(text: str) -> float:
    """Word-count-proportional estimate for a beat whose synthesis permanently
    failed -- see module docstring REQUIREMENT 2. Reuses the Pacing-Checker's
    own canonical spoken rate rather than a second, possibly-drifting constant.
    """
    word_count = len(text.split())
    return max(MIN_ESTIMATED_BEAT_SEC, word_count / WORDS_PER_SECOND)


def _probe_duration_sec(local_path: str) -> float:
    """Real, measured duration of a synthesized clip (requirement 2 -- exact,
    not estimated, alignment for every beat that actually has audio)."""
    info = ffmpeg.probe(local_path)
    return float(info["format"]["duration"])


# ---------------------------------------------------------------------------
# ffmpeg concat (CPU/IO-bound; callers run it via asyncio.to_thread, matching
# ken_burns_fallback_node.py's posture on its own blocking ffmpeg calls).
# ---------------------------------------------------------------------------
def _render_silence_clip(duration: float) -> str:
    """Render one `anullsrc` gap to its own real temp WAV file.

    Necessary, not just tidy: ffmpeg-python's graph builder treats two
    `ffmpeg.input()` calls with IDENTICAL args (same lavfi string + same `t=`)
    as the SAME node -- two beats that happen to share an estimated duration
    (e.g. two failed beats with the same word count) would otherwise collapse
    into one shared silence node, and the later `aformat` filter would then
    have two outgoing edges to `concat`'s two input slots, which ffmpeg-python
    rejects with "a `split` filter is probably required". Rendering each gap
    to its own always-unique temp file path sidesteps that dedup entirely.
    """
    fd, path = tempfile.mkstemp(suffix=".wav", prefix="voiceover_silence_")
    os.close(fd)
    (
        ffmpeg
        .input(f"anullsrc=r={_SILENCE_SAMPLE_RATE}:cl={_SILENCE_CHANNEL_LAYOUT}", f="lavfi", t=duration)
        .output(path, acodec="pcm_s16le")
        .overwrite_output()
        .run(capture_stdout=True, capture_stderr=True)
    )
    return path


def _concat_audio_segments(segments: list[tuple[Optional[str], float]]) -> str:
    """Concatenate per-beat audio (real clip, or a silent gap for a
    permanently-failed beat) into one MP3, in beat order.

    Every input is a real file on disk (see `_render_silence_clip` for why a
    gap is rendered to one rather than referenced as an in-graph lavfi node),
    passed through `aformat` first so mismatched sample rates/channel layouts
    across sources never break ffmpeg's concat filter, which requires uniform
    parameters across all inputs.
    """
    rendered_silence: list[str] = []
    streams = []
    try:
        for local_path, duration in segments:
            if local_path is None:
                local_path = _render_silence_clip(duration)
                rendered_silence.append(local_path)
            src = ffmpeg.input(local_path).audio
            streams.append(
                src.filter("aformat", sample_rates=_SILENCE_SAMPLE_RATE, channel_layouts=_SILENCE_CHANNEL_LAYOUT)
            )

        joined = ffmpeg.concat(*streams, v=0, a=1)
        fd, out_path = tempfile.mkstemp(suffix=".mp3", prefix="voiceover_")
        os.close(fd)
        (
            ffmpeg
            .output(joined, out_path, acodec="libmp3lame", audio_bitrate="128k")
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
        return out_path
    finally:
        for path in rendered_silence:
            if os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    pass


def _write_captions_json(captions: list[dict]) -> str:
    fd, path = tempfile.mkstemp(suffix=".json", prefix="captions_")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(captions, fh, ensure_ascii=False, indent=2)
    return path


# ---------------------------------------------------------------------------
# OSS upload bindings (job-level, not per-shot -- see agents/_oss.py's
# additive "oss_job_asset_key" note).
# ---------------------------------------------------------------------------
def _make_audio_upload_fn(job_id: str) -> Callable[[str], str]:
    def _upload(local_path: str) -> str:
        return upload_audio_to_oss(local_path, job_id)

    return _upload


def _make_captions_upload_fn(job_id: str) -> Callable[[str], str]:
    def _upload(local_path: str) -> str:
        return upload_json_to_oss(local_path, job_id, filename="captions.json")

    return _upload


# ---------------------------------------------------------------------------
# Main entry point.
# ---------------------------------------------------------------------------
async def generate_voiceover(
    winning_script: WinningScript,
    job_id: str,
    *,
    directed_beats: Optional[list[DirectedBeat]] = None,
    synth_fn: Optional[SynthesizeFn] = None,
    upload_audio_fn: Optional[Callable[[str], str]] = None,
    upload_captions_fn: Optional[Callable[[str], str]] = None,
) -> tuple[dict, list[dict]]:
    """Synthesize VO audio + caption timing for the finalized winning_script.

    Returns (voiceover, captions):
      * `voiceover` is the real, C1-shaped `{audio_uri, caption_track_uri}` dict
        (plus `failure_reason`/`silent_beat_indices` when >=1 beat degraded --
        see module docstring REQUIREMENT 5), ready to assign to
        `state["voiceover"]` verbatim.
      * `captions` is the full `[{text, start_ts, end_ts}, ...]` list (one
        entry per beat).

    Per-beat text + delivery selection:
      * If `synth_fn` is injected (tests), it's used as-is for every beat with
        the RAW beat line -- emotion/directed_beats are ignored (the injected
        fn is the whole boundary under test).
      * Else if `directed_beats` is available, each beat uses its `spoken_text`
        (falling back to the raw line) and a CosyVoice fn tuned with the beat's
        emotion instruction and pacing speech_rate.
      * Else, each beat uses its raw line with a neutral (conversational) voice.
    """
    upload_audio = upload_audio_fn or _make_audio_upload_fn(job_id)
    upload_captions = upload_captions_fn or _make_captions_upload_fn(job_id)

    raw_beats = winning_script.get("beats") or []
    if not raw_beats:
        # Degenerate case (should not happen -- merge_validator_node always
        # produces beats -- but never crash on a malformed winning_script).
        logger.warning("Voiceover: winning_script has no beats -- producing an empty, fully-silent track.")
        raw_beats = [{"t_start": 0.0, "t_end": MIN_ESTIMATED_BEAT_SEC, "line": ""}]
        directed_beats = None

    segments: list[tuple[Optional[str], float]] = []
    captions: list[dict] = []
    representative_failure: Optional[dict] = None
    silent_beat_indices: list[int] = []
    running_ts = 0.0

    for i, beat in enumerate(raw_beats):
        if synth_fn is not None:
            # Test injection -- use the injected fn as-is with the raw line.
            text = beat.get("line", "")
            beat_fn = synth_fn
        elif directed_beats and i < len(directed_beats):
            d = directed_beats[i]
            text = d.get("spoken_text") or beat.get("line", "")
            pacing = d.get("pacing", "normal")
            beat_fn = _make_cosyvoice_fn(pacing)
        else:
            text = beat.get("line", "")
            beat_fn = _make_cosyvoice_fn()

        local_path, beat_failure = await _synthesize_beat(text, beat_fn)
        if local_path is not None:
            duration = await asyncio.to_thread(_probe_duration_sec, local_path)
        else:
            duration = _estimate_duration_sec(text)
            silent_beat_indices.append(i)
            representative_failure = representative_failure or beat_failure

        segments.append((local_path, duration))
        captions.append({"text": text, "start_ts": round(running_ts, 3), "end_ts": round(running_ts + duration, 3)})
        running_ts += duration

    local_audio_path = await asyncio.to_thread(_concat_audio_segments, segments)
    captions_path = _write_captions_json(captions)
    try:
        audio_uri = await asyncio.to_thread(upload_audio, local_audio_path)
        caption_track_uri = await asyncio.to_thread(upload_captions, captions_path)
    finally:
        for local_path, _ in segments:
            if local_path and os.path.exists(local_path):
                try:
                    os.remove(local_path)
                except OSError:
                    pass
        if os.path.exists(local_audio_path):
            try:
                os.remove(local_audio_path)
            except OSError:
                pass
        if os.path.exists(captions_path):
            try:
                os.remove(captions_path)
            except OSError:
                pass

    voiceover: dict = {"audio_uri": audio_uri, "caption_track_uri": caption_track_uri}
    if representative_failure is not None:
        # Extra, non-C1 keys -- cost-free (Voiceover has no Pydantic/
        # extra="forbid" validator), same precedent as GeneratedShot's
        # resolution_used/etc. in video_gen_node.py. See REQUIREMENT 5.
        voiceover["failure_reason"] = representative_failure
        voiceover["silent_beat_indices"] = silent_beat_indices

    return voiceover, captions


# ---------------------------------------------------------------------------
# LangGraph node wrapper.
# ---------------------------------------------------------------------------
async def voiceover_caption_agent_node(
    state: ProductCutState,
    config: Optional[RunnableConfig] = None,
) -> dict:
    """LangGraph node wrapper: reads `winning_script`/`directed_script_beats`/
    `job_id` from state, synthesizes VO audio + caption timing, persists both to
    OSS, and emits the proposed C2 `vo_ready` event (see module docstring
    REQUIREMENT 4).

    `state["winning_script"]` is accessed directly (KeyError, not `.get(...)`)
    -- this node's one hard precondition is that `merge_validator_node` already
    finalized it. `directed_script_beats` is read with `.get(...)` -- it is a
    NotRequired enhancement (the Voice Direction Agent runs as a serial pre-step
    but this node still degrades gracefully to the raw beat lines if it's absent).
    """
    job_id = state.get("job_id", "unknown_job")
    winning_script = state["winning_script"]
    directed_beats = state.get("directed_script_beats")

    voiceover, captions = await generate_voiceover(
        winning_script, job_id, directed_beats=directed_beats
    )

    await adispatch_custom_event(
        "vo_ready",
        {
            "voiceover": voiceover,
            "caption_count": len(captions),
            "degraded": "failure_reason" in voiceover,
        },
        config=config,
    )

    trace_note = f"\n[voiceover_caption_agent] synthesized {len(captions)} caption beat(s)."
    if "failure_reason" in voiceover:
        trace_note += (
            f" {len(voiceover['silent_beat_indices'])} beat(s) silent after persistent "
            f"TTS failure ({voiceover['failure_reason']['type']}) -- assembling with "
            "captions only for those segments."
        )

    return {
        "voiceover": voiceover,
        # Dedicated key, NOT the shared `reasoning_trace` -- see graph/state.py's
        # v7 changelog note for the full reasoning.
        "voiceover_reasoning_trace": state.get("voiceover_reasoning_trace", "") + trace_note,
    }


__all__ = [
    "COSYVOICE_MODEL_ID",
    "COSYVOICE_VOICE_ID",
    "PACING_SPEECH_RATE",
    "DEFAULT_COSYVOICE_TIMEOUT_SEC",
    "MIN_ESTIMATED_BEAT_SEC",
    "FAILURE_TYPE_TIMEOUT",
    "FAILURE_TYPE_API_ERROR",
    "VoiceoverTimeoutError",
    "VoiceoverAPIError",
    "SynthesizeFn",
    "_call_cosyvoice_sync",
    "_call_cosyvoice",
    "_make_cosyvoice_fn",
    "_synthesize_beat",
    "_estimate_duration_sec",
    "generate_voiceover",
    "voiceover_caption_agent_node",
]
