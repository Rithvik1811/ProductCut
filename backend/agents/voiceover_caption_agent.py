"""
Voiceover + Caption Agent -- Qwen3-TTS-Flash via the native DashScope SDK (Phase 5).
Spec of record: docs/TECHNICAL_DOCUMENTATION.md §5.11.

Confirmed against the ACTUAL merged code before writing anything (per this
task's own instruction not to assume field names):
  * `graph.state.WinningScript` (C1, frozen): {text, beats: list[ScriptBeat],
    source_variant_ids}. `ScriptBeat` = {t_start, t_end, line} -- NOTE there is
    no `beat_function` (hook/problem/demo/proof/cta) on a ScriptBeat; that enum
    lives on `Treatment.beat_treatments[]`, written later by the Treatment
    Agent. Since this node is a parallel branch off `winning_script` alone
    (see PARALLEL BRANCH below), Treatment may not have run yet -- so beat
    segmentation here is keyed on beat INDEX/`t_start`/`t_end`/`line` only,
    never on `beat_function`.
  * `graph.state.Voiceover` (C1, frozen): `{audio_uri, caption_track_uri}` --
    BOTH plain strings (OSS refs), confirmed against docs/TECHNICAL_DOCUMENTATION.md
    §5.11's own output contract ("voiceover = {audio_uri, caption_track_uri}").
    This task's own instructions suggested writing `vo_audio_ref` and
    `caption_timing` fields -- those names do not exist anywhere in the real,
    merged graph/state.py. Using the real ones instead: `state["voiceover"]`,
    `.audio_uri`, `.caption_track_uri`. The `{text, start_ts, end_ts}` caption
    array requirement 2 asks for is real work this module does, it is just not
    stored inline in state -- like every other sizeable artifact in this
    codebase (video_uri, master_cut_uri, exports), it is serialized to a file,
    uploaded to OSS, and only the resulting signed URL lands in state, as
    `caption_track_uri`.
  * `.env.example`: `MODEL_TTS=qwen3-tts-flash` (confirmed exact value -- no
    other snapshot pinned).
  * `requirements.txt`'s own comment on the pinned `dashscope` dependency:
    "native SDK: Wan/HappyHorse video synthesis + CosyVoice TTS task polling"
    -- i.e. TTS is native-SDK territory in this codebase, NOT the OpenAI-
    compatible endpoint every LLM/vision agent uses (docs/TECHNICAL_DOCUMENTATION.md
    §9.7's "every speech call goes through the OpenAI-compatible endpoint" is
    the aspirational/planning-doc claim; the actual pinned dependency comment
    is the real, current answer, and it says native SDK -- the same
    OpenAI-compatible-vs-native split agents/video_gen_node.py already
    documented and followed for Wan).
  * The installed `dashscope` package (`backend/.venv/.../dashscope/audio/qwen_tts/
    speech_synthesizer.py`) was read directly rather than guessed: `SpeechSynthesizer.call(
    model, text, api_key=None, workspace=None, voice=..., stream=...)` ->
    `TextToSpeechResponse` whose `.output.audio` is a `TextToSpeechAudio` with
    exactly `{expires_at, id, data, url}` -- CONFIRMED: no word/phoneme-level
    timestamp field of any kind. This directly settles requirement 2's open
    question: per-beat caption timing MUST be derived, not read off the API
    response, for every beat whose synthesis succeeds it is derived from a
    REAL measured clip duration (ffprobe), never a heuristic; the word-count
    heuristic below is used ONLY as the estimate for a beat whose synthesis
    permanently failed (see FAILURE HANDLING).

PARALLEL BRANCH, NOT A DOWNSTREAM STEP OF VIDEO-GEN (per the phase goal and
docs/TECHNICAL_DOCUMENTATION.md §5.11: "runs as a parallel branch that starts
as soon as the winning script is finalized... because voiceover depends only
on the script, not on the rendered video"). This module's entry point,
`voiceover_caption_agent_node`, reads ONLY `state["winning_script"]` +
`state["job_id"]` -- it never reads `treatment`, `shot_list`, `budget_ledger`,
or `generated_shots`, and writes only `state["voiceover"]` +
`voiceover_reasoning_trace` (its OWN dedicated trace key, not the shared
`reasoning_trace` every other node writes -- see graph/state.py's v7 changelog:
this node now genuinely runs in the same superstep as treatment_agent, and two
nodes read-modify-writing one plain-string channel raises LangGraph's
InvalidUpdateError; a single-writer key sidesteps that without touching the
other dozen agent modules' already-established read-modify-write contract).
Nothing here depends on, blocks, or is blocked by Video-Gen/Ken-Burns/Continuity.

HOW THIS COMPOSES WITH THE REST OF THE GRAPH (checked graph/build.py and
agents/merge_validator.py's `route_after_merge_validation` before deciding --
not a guess): `winning_script` is written by `merge_validator_node` on EITHER
a "finalize" or a "fallback" verdict (graph/build.py's own docstring). This is
NOT a `Send()` fan-out (that pattern, used internally by video_gen_node.py, is
for fanning one shot list out to N parallel per-shot branches within a single
node -- a different problem). The mechanism used for a SECOND node to run in
parallel off the SAME conditional edge is LangGraph's documented support for a
conditional-edge path function returning a *list* of routing keys, not just one
(confirmed by reading the installed `langgraph.graph.state.StateGraph.add_conditional_edges`'s
own type signature: `path: Callable[..., Hashable | Sequence[Hashable]]`).

WIRED INTO graph/build.py (follow-up integration commit, matching this
codebase's established rhythm -- see video_gen_node.py's own precedent). What
actually landed, exactly as planned below except item 1 (kept
`route_after_merge_validation` itself untouched -- it's independently
unit-tested in test_merge_validator.py and called internally by
`merge_validator_node`; graph/build.py instead wraps it in a small
`_route_after_merge_validation_with_vo` path function used only by the
conditional edge):
  1. The conditional-edge path function's "finalize"/"fallback" branches return
     `["treatment_agent", "voiceover_caption_agent"]` instead of a bare string
     (its "copy_editor"/"meta_critic" retry-loop branches are unaffected --
     winning_script isn't final yet on those paths, so VOX must not fire).
  2. `builder.add_node("voiceover_caption_agent", voiceover_caption_agent_node)`
     and `"voiceover_caption_agent": "voiceover_caption_agent"` added to the
     `path_map` passed to the existing `add_conditional_edges("merge_validator", ...)`
     call.
  3. `builder.add_edge("voiceover_caption_agent", END)` -- nothing consumes
     `state["voiceover"]` yet (Assembly, §5.12, is not built), so this branch
     is a leaf for now, exactly like Continuity's "end" branch before Phase 4
     added a consumer downstream of it.

WAS NOT WIRED INTO graph/build.py FOR ONE PASS -- deliberately, matching this
codebase's own established rhythm (agents/video_gen_node.py, Phase 3, was
merged standalone before its own later integration commit; RR then wired it +
Ken-Burns in as a separate follow-up). This module followed the same
precedent: built + tested standalone first, then wired in as a separate
follow-up commit (the "WIRED INTO graph/build.py" section above) that also
updated tests/test_graph_build.py, tests/test_graph_end_to_end.py,
tests/test_graph_critic_chain.py, tests/test_graph_merge_validator.py,
tests/test_continuity_loop_e2e.py, tests/test_phase4_integration_edge_cases.py,
and tests/test_pipeline_integration_edge_cases.py (every test driving the
compiled graph through merge_validator now fakes this node's TTS/OSS boundary
too, via tests/_phase3_graph.py's new `patch_voiceover_boundaries`).

REQUIREMENT 1 -- PER-BEAT vs. WHOLE-SCRIPT TTS, tradeoff and decision. Chosen:
PER-BEAT (one `SpeechSynthesizer.call` per `ScriptBeat.line`), not one call for
the whole script. Why: docs §5.11's own failure-handling sentence -- "If
synthesis fails for a line, the node retries that line" -- only makes sense
under a per-line/per-beat call granularity; a single whole-script call has no
"a line" to selectively retry. Per-beat also gives natural, REAL pause control
at beat boundaries (each beat's audio is a genuinely separate clip, concatenated
in order) and -- combined with the confirmed absence of word-level timestamps
above -- is the only way to get an exact (not estimated) duration per beat via
ffprobe on each clip before concatenating. Cost: N TTS calls instead of 1 (N =
beat count, typically 4-6 per §5.6's 3-7 shots / one beat per shot-ish scale),
each a small, fast synthesis of a single short line rather than one longer
call -- more calls, but each cheaper and independently retryable/degradable.

REQUIREMENT 2 -- CAPTION GRANULARITY AND TIMING. One caption entry per beat
(the finest structured text unit `WinningScript.beats` actually has -- there is
no sub-beat/per-word structure anywhere in C1 to split further without
inventing one). For a beat whose synthesis SUCCEEDED, `start_ts`/`end_ts` come
from that beat's REAL measured clip duration (`ffmpeg.probe`, cumulative from
0 at the first beat) -- exact alignment to the actual audio, not the script's
own `t_start`/`t_end` (which are the *target* pacing timestamps validated by
Pacing-Checker, not necessarily identical to how long Qwen3-TTS-Flash actually
took to say the line). For a beat whose synthesis PERMANENTLY failed (see
FAILURE HANDLING), there is no real audio to measure, so its `start_ts`/`end_ts`
span is ESTIMATED: `len(line.split()) / WORDS_PER_SECOND` (reusing
`agents.pacing_checker.WORDS_PER_SECOND` = 2.3, the same canonical spoken-rate
constant the Pacing-Checker already validated this exact script against --
not a second, possibly-drifting copy of that number), floored at
`MIN_ESTIMATED_BEAT_SEC` so a near-empty line never collapses to ~0s. This is
explicitly an ESTIMATE, not exact alignment -- flagged via `failure_reason` +
`silent_beat_indices` on the returned voiceover dict (see FAILURE HANDLING).

REQUIREMENT 3 -- OSS PERSISTENCE. Reuses agents/_oss.py exclusively -- no new
put_object_from_file/sign_url code here. The concatenated VO audio and the
caption-timing JSON are two job-level (not per-shot) assets, so this task
required a small ADDITIVE extension to _oss.py (see that module's own
docstring "ADDITIVE (Phase 5, ...)" note): `oss_job_asset_key` (a job-only
`jobs/{job_id}/{filename}` key, alongside the existing shot-scoped
`oss_object_key`) and two thin wrappers, `upload_audio_to_oss` /
`upload_json_to_oss`, both built on a shared `_put_and_sign` helper factored
out of the pre-existing `upload_video_to_oss` (behavior-preserving refactor --
test_oss.py's existing assertions are unchanged and still pass).

REQUIREMENT 4 -- EVENT. No `vo_ready` (or equivalent) event existed in C2
before this task (checked graph/events.py's full `EventType` Literal: the
original 10 + v2's "merge_validated" cover Ingest through job completion, none
of them fire when VO synthesis finishes). Per this task's own instruction --
"flagged clearly as a proposed additive change, not silently invented" -- this
is added as `graph/events.py` v4: `"vo_ready"` + `VoReadyPayload {voiceover,
caption_count, degraded}`, documented in that file's own version-history
docstring exactly like every prior additive C2 change (v2's "merge_validated",
v3's `ShotGeneratedPayload.status` addition), and explicitly marked there as a
proposed change pending a sync with whoever builds the dashboard's VO panel --
mirroring how `status="fallback_requested"` was flagged as self-invented and
unconfirmed in video_gen_node.py before RR's Phase 3 sync formalized it.
Dispatched via `adispatch_custom_event`, the same mechanism/precedent as
`shot_generated` (video_gen_node.py) and `truth_extracted`
(product_truth_extractor.py).

REQUIREMENT 5 -- FAILURE HANDLING / DEGRADE PATH. Chosen: per-beat retry once,
then degrade that single beat to "silent gap + estimated-timing caption",
NEVER halt the whole node/job over one bad line. This is not an invented
policy -- it is docs §5.11's own explicit sentence, followed literally: "If
synthesis fails for a line, the node retries that line; on persistent failure
the ad can assemble with captions only (silent-with-captions is a valid
short-form format), recorded in the trace." Concretely:
  * Each beat gets at most 2 attempts (1 initial + 1 retry) via `_synthesize_beat`.
  * A beat that fails both attempts contributes a SILENT gap (ffmpeg `anullsrc`)
    of its estimated duration to the concatenated audio timeline, not a missing
    chunk -- the final `audio_uri` always spans the full ad, so downstream
    ffmpeg concat/mixing (Assembly, not yet built) never has to special-case a
    variable-length track.
  * The RETURNED `voiceover` dict always has `audio_uri` + `caption_track_uri`
    (the frozen, non-`NotRequired` C1 shape is honestly satisfied in every
    case -- even a job where EVERY beat fails still produces a real, fully
    silent audio_uri spanning the estimated total length, i.e. the doc's
    literal "the ad can assemble with captions only", implemented as silence
    rather than as a missing/optional key that would violate the frozen shape).
  * Two EXTRA, non-C1 keys are added when >=1 beat degraded -- the same
    "extra keys cost nothing, `Voiceover` has no Pydantic/`extra=forbid`
    validator anywhere in this codebase" precedent video_gen_node.py's
    `GeneratedShot.resolution_used`/etc. established: `failure_reason` (reuses
    the EXISTING `graph.state.FailureReason` shape `{type, detail}` verbatim --
    a TTS failure is always classified "timeout" or "api_error", both already
    real values in that frozen Literal, so no new failure-type vocabulary is
    invented) and `silent_beat_indices: list[int]`, naming exactly which beats
    are silent so a future Assembly Agent can mute/gap those spans precisely
    instead of guessing from a single boolean. Their ABSENCE (no
    `failure_reason` key at all) is itself the "fully real, nothing degraded"
    signal -- Assembly's obvious check is `"failure_reason" in state["voiceover"]`.

OUTPUT AUDIO FORMAT -- MP3 (`libmp3lame`), chosen (not confirmed against a real
Qwen3-TTS-Flash account response -- see next paragraph) as a broadly-compatible
default for the concatenated master VO track; ffmpeg re-encodes every input
through `aformat` before concatenation, so the OUTPUT format is fixed
regardless of whatever container/codec each per-beat clip actually arrives in.

RESOLVED -- CROSS-REGION TTS CREDENTIALS (live-confirmed via
`derisk/test_tts_smoke.py`). The three items below were originally flagged as
a single "KNOWN GAP" (response field + base-URL routing both unconfirmed
against a live account). Live testing during that de-risk pass found this
account's TTS access is genuinely **on a different DashScope region/workspace
than every other model** -- `DASHSCOPE_API_KEY` (Virginia/`dashscope-us`,
already `.env`-documented) returns a real, explicit `model_not_found` for
every TTS model id tried (`qwen3-tts-flash`, `qwen-tts`, `cosyvoice-v1/v2`)
via both the native SDK and the OpenAI-compatible endpoint, while the SAME
account's TTS access genuinely works (confirmed via the DashScope console
Playground) once addressed with dedicated Singapore/`dashscope-intl`
credentials. This is an account/workspace-scoping fact, not a bug to fix in
code -- so this module (and ONLY this module) reads a separate pair of env
vars instead of the shared `DASHSCOPE_API_KEY`/`DASHSCOPE_BASE_URL` every
other agent uses:
  * `DASHSCOPE_TTS_API_KEY` (required, no fallback to `DASHSCOPE_API_KEY` --
    the shared key is confirmed NOT to carry TTS scope on this account, so
    silently falling back would just reproduce the `model_not_found` failure).
  * `DASHSCOPE_TTS_BASE_URL` (required for a real call to succeed) -- set in
    the SAME `.../compatible-mode/v1` flavor as `DASHSCOPE_BASE_URL` for
    consistency (`.env.example`), even though this module calls the native
    SDK, not the OpenAI-compatible client. `_resolve_tts_native_base_url()`
    derives the native `.../api/v1` path from it (same host, different path,
    exactly the relationship `docs/DERISK_VIDEO_GEN_RESULT.md` §5 documents
    for Wan) -- confirmed live: the compatible-mode path itself does NOT work
    against `dashscope.audio.qwen_tts.SpeechSynthesizer` (native SDK call),
    only the derived native path does.
  1. Response field: `TextToSpeechAudio.url` is the one actually populated by
     a real call on this account (a signed `dashscope-result-sgp` OSS URL,
     ~24h `expires_at`); `.data` came back as an empty string. `_call_qwen_tts`
     below still handles a `.data`-only response defensively (never assume a
     one-account observation is universal), but `.url` is the confirmed,
     exercised path.
  2. Base-URL routing: see "RESOLVED" above -- `DASHSCOPE_TTS_BASE_URL` is
     effectively required for this account (not "most accounts need neither
     override" as originally guessed), since TTS lives on a different
     workspace/region than every other model this codebase calls. The
     REAL RISK this creates -- unique to this being a genuinely parallel
     branch (every other agent in this codebase runs strictly sequentially)
     -- is now MOOT for the base-URL race originally flagged here:
     `dashscope.base_http_api_url` is global, mutable SDK state, but
     Video-Gen's optional `DASHSCOPE_VIDEO_BASE_URL` and this module's
     `DASHSCOPE_TTS_BASE_URL` are two independently-resolved values only
     ever written from within each module's own synchronous call (`asyncio.to_thread`),
     so a genuine data race would require two threads mutating the same
     process-global attribute between two DIFFERENT nodes' concurrent calls --
     still a real, narrow risk if a job's Video-Gen and Voiceover branches
     execute genuinely concurrently AND both need their override applied at
     the same instant; flagged, not solved with a lock, matching
     video_gen_node.py's own proportionate posture on its analogous gap.
  3. `DEFAULT_TTS_VOICE` ("Cherry") is now CONFIRMED against a real account
     (the de-risk call above succeeded with `voice="Cherry"`) -- no longer a
     guess. Still env-overridable via `TTS_VOICE`.
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import tempfile
from typing import Awaitable, Callable, Optional

import ffmpeg
from dashscope.audio.qwen_tts import SpeechSynthesizer
from langchain_core.callbacks.manager import adispatch_custom_event
from langchain_core.runnables import RunnableConfig

from agents._oss import _download_to_temp, upload_audio_to_oss, upload_json_to_oss
from agents.pacing_checker import WORDS_PER_SECOND
from graph.state import ProductCutState, WinningScript

logger = logging.getLogger("productcut.agents.voiceover_caption_agent")

# See KNOWN GAP #3.
DEFAULT_TTS_VOICE = os.getenv("TTS_VOICE", "Cherry")

# Qwen3-TTS-Flash synthesizes one short line per call -- generous margin above
# the handful-of-seconds a single beat's line should realistically take,
# without the multi-minute budget video_gen_node.py needs for a whole clip.
DEFAULT_TTS_TIMEOUT_SEC = float(os.getenv("VOICEOVER_TTS_TIMEOUT_SEC", "60"))

# Floor so a near-empty/blank line never estimates to ~0s (requirement 2).
MIN_ESTIMATED_BEAT_SEC = 0.5

# Reused verbatim from graph.state.FailureReason's frozen Literal -- see
# module docstring's REQUIREMENT 5 note on why no new failure-type is invented.
FAILURE_TYPE_TIMEOUT = "timeout"
FAILURE_TYPE_API_ERROR = "api_error"

_SILENCE_SAMPLE_RATE = 44100
_SILENCE_CHANNEL_LAYOUT = "stereo"


class VoiceoverTimeoutError(Exception):
    """A single beat's Qwen3-TTS-Flash call never returned within the wait timeout."""


class VoiceoverAPIError(Exception):
    """The Qwen3-TTS-Flash call returned a hard failure (non-200 or no usable audio)."""


# ---------------------------------------------------------------------------
# Real Qwen3-TTS-Flash call (native dashscope SDK -- see module docstring's
# "RESOLVED" section on cross-region credentials and the response shape).
# ---------------------------------------------------------------------------
def _extract_audio_field(audio, key: str) -> Optional[str]:
    """Read `key` off a real `SpeechSynthesizer.call()` response's `output.audio`.

    BUG FIX (found live, via derisk/test_tts_smoke.py): the installed SDK's
    `TextToSpeechAudio` dataclass documents `url`/`data` as attributes, but a
    REAL response's `output.audio` comes back as a plain `dict` (confirmed:
    `type(resp.output.audio) is dict`), not a `TextToSpeechAudio` instance --
    so the original `getattr(audio, "url", None)` silently returned `None`
    every time, even on a fully successful call with a real signed URL
    present. Every real production TTS call would have failed with "neither
    url nor data" despite the API genuinely succeeding. Unit tests never
    caught this because their `_FakeAudio` fixture used real attributes
    (`self.url = url`), not a dict, which doesn't reproduce the real SDK's
    shape. Handles both here defensively (dict-style first, since that's the
    confirmed real shape; attribute-style as a fallback in case a future SDK
    version wraps it in a real object again).
    """
    if isinstance(audio, dict):
        return audio.get(key)
    return getattr(audio, key, None)


def _write_base64_audio_to_temp(data: str) -> str:
    """Decode a base64 `TextToSpeechAudio.data` payload to a local temp file.

    Suffix is deliberately generic (`.audio`, not a guessed `.mp3`/`.wav`) --
    ffmpeg's demuxer probes actual content, not the extension, and the real
    encoding this path returns is unconfirmed (see module docstring KNOWN GAP #1).
    """
    raw = base64.b64decode(data)
    fd, path = tempfile.mkstemp(suffix=".audio", prefix="qwen_tts_b64_")
    with os.fdopen(fd, "wb") as fh:
        fh.write(raw)
    return path


def _resolve_tts_native_base_url() -> Optional[str]:
    """Derive the native dashscope SDK base URL from `DASHSCOPE_TTS_BASE_URL`.

    `DASHSCOPE_TTS_BASE_URL` is set in the same `.../compatible-mode/v1`
    flavor as `DASHSCOPE_BASE_URL` (`.env.example`) for consistency, but
    `dashscope.audio.qwen_tts.SpeechSynthesizer` is a NATIVE SDK call, which
    needs the native `.../api/v1` path on the SAME host -- confirmed live
    (see module docstring "RESOLVED" section): the compatible-mode path
    itself returns nothing usable against this native call, only the derived
    native path does.
    """
    compatible_url = os.getenv("DASHSCOPE_TTS_BASE_URL")
    if not compatible_url:
        return None
    return compatible_url.replace("/compatible-mode/v1", "/api/v1")


def _call_qwen_tts_sync(text: str) -> str:
    """Blocking Qwen3-TTS-Flash call + download/decode to a local temp file.

    Returns the local path to the synthesized audio (never a remote URL --
    unlike video_gen_node.py's Wan call, this module needs the raw bytes
    immediately, to `ffmpeg.probe` its real duration and later concatenate it
    with its sibling beats, so downloading here rather than deferring to a
    separate persistence step avoids a second round trip).

    Uses `DASHSCOPE_TTS_API_KEY`/`DASHSCOPE_TTS_BASE_URL`, NOT the shared
    `DASHSCOPE_API_KEY`/`DASHSCOPE_BASE_URL` every other agent in this
    codebase uses -- see module docstring "RESOLVED -- CROSS-REGION TTS
    CREDENTIALS" for why this account's TTS access is scoped to a different
    DashScope region/workspace than text/vision/video.
    """
    model = os.environ["MODEL_TTS"]  # KeyError is intentional -- see product_truth_extractor.py's precedent on MODEL_VISION
    native_base_url = _resolve_tts_native_base_url()
    if native_base_url:
        import dashscope

        dashscope.base_http_api_url = native_base_url

    response = SpeechSynthesizer.call(
        model=model,
        text=text,
        voice=DEFAULT_TTS_VOICE,
        api_key=os.environ["DASHSCOPE_TTS_API_KEY"],  # dedicated TTS-scoped key -- see module docstring
    )
    if response.status_code != 200:
        raise VoiceoverAPIError(
            f"Qwen3-TTS-Flash call failed (code={getattr(response, 'code', '')!r}, "
            f"message={getattr(response, 'message', '')!r})"
        )

    audio = response.output.audio if response.output else None
    if audio is None:
        raise VoiceoverAPIError("Qwen3-TTS-Flash response had no output.audio at all")

    url = _extract_audio_field(audio, "url")
    if url:
        return _download_to_temp(url)
    data = _extract_audio_field(audio, "data")
    if data:
        return _write_base64_audio_to_temp(data)
    raise VoiceoverAPIError("Qwen3-TTS-Flash response's output.audio had neither url nor data")


async def _call_qwen_tts(text: str) -> str:
    """Async wrapper: the native SDK call is blocking, so it runs in a thread
    (matching ken_burns_fallback_node.py's posture on its own blocking ffmpeg
    calls), with an explicit wait timeout (matching video_gen_node.py's
    `_call_wan_video_gen` posture on its own blocking-call timeout).
    """
    try:
        return await asyncio.wait_for(
            asyncio.to_thread(_call_qwen_tts_sync, text),
            timeout=DEFAULT_TTS_TIMEOUT_SEC,
        )
    except asyncio.TimeoutError as exc:
        raise VoiceoverTimeoutError(
            f"Qwen3-TTS-Flash exceeded the {DEFAULT_TTS_TIMEOUT_SEC:.0f}s wait timeout"
        ) from exc
    except VoiceoverAPIError:
        raise
    except Exception as exc:  # noqa: BLE001 -- any transport/SDK error is a classified api_error, never crashes the batch
        raise VoiceoverAPIError(str(exc)) from exc


SynthesizeFn = Callable[[str], Awaitable[str]]


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
    to its own always-unique temp file path sidesteps that dedup entirely --
    two files can share identical audio content and duration without ever
    comparing equal as graph nodes.
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
        entry per beat) -- not itself part of state (see module docstring),
        but returned so the node wrapper can report `caption_count` on the
        `vo_ready` event and in the reasoning trace without re-deriving it.

    `synth_fn` / `upload_audio_fn` / `upload_captions_fn` are injectable (same
    `generate_fn`/`client=None` pattern every other agent module here uses) --
    default to the real Qwen3-TTS-Flash call and the real OSS uploads.
    """
    fn: SynthesizeFn = synth_fn or _call_qwen_tts
    upload_audio = upload_audio_fn or _make_audio_upload_fn(job_id)
    upload_captions = upload_captions_fn or _make_captions_upload_fn(job_id)

    beats = winning_script.get("beats") or []
    if not beats:
        # Degenerate case (should not happen -- merge_validator_node always
        # produces beats -- but never crash on a malformed winning_script).
        logger.warning("Voiceover: winning_script has no beats -- producing an empty, fully-silent track.")
        beats = [{"t_start": 0.0, "t_end": MIN_ESTIMATED_BEAT_SEC, "line": ""}]

    segments: list[tuple[Optional[str], float]] = []
    captions: list[dict] = []
    representative_failure: Optional[dict] = None
    silent_beat_indices: list[int] = []
    running_ts = 0.0

    for i, beat in enumerate(beats):
        text = beat.get("line", "")
        local_path, beat_failure = await _synthesize_beat(text, fn)
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
# LangGraph node wrapper. NOT wired into graph/build.py yet -- see module
# docstring's "HOW THIS COMPOSES..." / "NOT WIRED..." sections.
# ---------------------------------------------------------------------------
async def voiceover_caption_agent_node(
    state: ProductCutState,
    config: Optional[RunnableConfig] = None,
) -> dict:
    """LangGraph node wrapper: reads `winning_script`/`job_id` from state,
    synthesizes VO audio + caption timing, persists both to OSS, and emits the
    proposed C2 `vo_ready` event (see module docstring REQUIREMENT 4).

    `state["winning_script"]` is accessed directly (KeyError, not `.get(...)`)
    -- this node's one hard precondition is that `merge_validator_node` already
    finalized it; a KeyError here means a wiring bug (this node invoked before
    winning_script exists), not a normal runtime-data gap, matching this
    codebase's posture on other hard preconditions (e.g.
    product_truth_extractor_node's direct `state["product_photos"]` access).
    """
    job_id = state.get("job_id", "unknown_job")
    winning_script = state["winning_script"]

    voiceover, captions = await generate_voiceover(winning_script, job_id)

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
        # Dedicated key, NOT the shared `reasoning_trace` -- this node runs as a
        # genuine parallel branch alongside treatment_agent (graph/build.py), and
        # two nodes read-modify-writing the same plain-string channel in the same
        # superstep raises LangGraph's InvalidUpdateError. See graph/state.py's
        # v7 changelog note for the full reasoning.
        "voiceover_reasoning_trace": state.get("voiceover_reasoning_trace", "") + trace_note,
    }


__all__ = [
    "DEFAULT_TTS_VOICE",
    "DEFAULT_TTS_TIMEOUT_SEC",
    "MIN_ESTIMATED_BEAT_SEC",
    "FAILURE_TYPE_TIMEOUT",
    "FAILURE_TYPE_API_ERROR",
    "VoiceoverTimeoutError",
    "VoiceoverAPIError",
    "generate_voiceover",
    "voiceover_caption_agent_node",
]
