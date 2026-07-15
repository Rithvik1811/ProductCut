"""
Assembly Agent -- stitches shot clips + voiceover + captions into one master
cut (Phase 5). Spec of record: docs/TECHNICAL_DOCUMENTATION.md §5.12.

Confirmed against the ACTUAL merged code before writing anything:
  * `graph.state.Shot.justification.treatment_ref` matches
    `Treatment.beat_treatments[].beat_index`, which matches the winning
    script beat's own position -- the ONLY reliable way to map a `Shot` back
    to "which script beat is this" (shot_list is NOT guaranteed positional
    after `agents/budget_gate.py` cuts a shot).
  * `agents/budget_gate.py` can and does drop a shot entirely (cut-only,
    never merge -- see that module's own docstring), so
    `len(shot_list) <= len(winning_script.beats)` is a real, common case, not
    a defensive-only edge.
  * `agents/voiceover_caption_agent.py` produces ONE caption entry per
    `winning_script.beats` element, unconditionally, with `start_ts`/`end_ts`
    cumulative from REAL measured TTS clip durations -- it has no idea what
    Budget Gate did to the shot list. The caption track is therefore the
    complete, authoritative timeline; the shot list is a partial visual
    covering of it.
  * `agents/ken_burns_fallback_node.py` fallback clips are hardcoded
    1920x1080 @ 30fps; real Wan clips can independently land at 720p or
    1080p AND can be portrait (a real generation this session measured
    784x1174) -- a single job's shots are routinely a resolution/orientation
    mix, so every segment is normalized (scale+letterbox, never a bare
    scale-to-fill) onto one shared canvas before concatenation.
  * `GeneratedShot.duration_sec_used` is the PLANNED duration on both the
    real and fallback paths (`ken_burns_fallback_node.py` sets it to
    `shot["duration_sec"]` verbatim; a live pipeline run this session showed
    Wan clips landing at 3.01s measured against a 3.0/4.0s planned value) --
    never trusted here; every downloaded clip is `ffmpeg.probe`d for its
    real duration/resolution.
  * `agents._oss._download_to_temp` / `_put_and_sign` / `oss_job_asset_key`
    are reused verbatim (no second hand-rolled download/upload path) --
    `upload_master_cut_to_oss` is the one small additive wrapper this task
    needed there, mirroring `upload_audio_to_oss`/`upload_json_to_oss`.

TWO-STAGE FFMPEG PIPELINE (ffmpeg 8.1.2-full_build-www.gyan.dev, confirmed on
this machine; a real end-to-end prototype was run against it -- concat
demuxer of video-only segments + chained `drawtext` + external audio input +
`apad`/`-shortest` -- before any of this was written, so the filter graph
below is empirically verified, not merely researched).

STAGE 1 (one ffmpeg run PER SHOT): download the clip, probe its REAL
duration/resolution, normalize it onto the job's shared canvas
(`scale ... force_original_aspect_ratio=decrease,pad,setsar=1`), force true
CFR at `FPS` (`fps=30` -- this, not a bare copy, is what prevents a
stutter/desync at a Wan<->Ken-Burns cut, since Ken-Burns is native CFR30 and
Wan's realized fps is never assumed), strip all audio (`-an` -- every clip's
baked-in audio is Wan's own incidental ambient noise or Ken-Burns's literal
silence, never meaningful, never mixed into the final ad), and conform the
VIDEO's duration to its beat's REAL measured VO window (never the other way
around -- see DURATION CONFORMING below). Written to a small, fast
(`-preset veryfast`) intermediate file; it gets re-encoded once more in
Stage 2 anyway.

STAGE 2 (one ffmpeg run): concat-DEMUX (not concat-filter -- safe specifically
BECAUSE Stage 1 already normalized every segment to identical codec/size/fps/
pix_fmt/no-audio) the Stage-1 segments in beat order, burn every beat's
caption via a `drawtext` chain gated with `enable=between(t,start_ts,end_ts)`
straight off the REAL captions JSON, map the downloaded VO audio file as the
SOLE audio track (`apad` + `-shortest` absorbs sub-frame rounding drift
rather than truncating a video frame or leaving a dangling silent tail), and
encode the final mezzanine (`-crf 16 -preset slow` -- this is NOT the final
per-aspect-ratio deliverable; a later, not-yet-built Format Export node
re-encodes it again, so this pass avoids stacking a second lossy compression
on top of Wan's already-compressed output).

DURATION CONFORMING -- THE CORE DESIGN DECISION. The voiceover is the master
clock and is NEVER time-stretched: its captions' timestamps are cumulative
offsets baked into ONE already-concatenated audio file (§5.11), so stretching
any beat's audio would desync every subsequent beat's caption. Instead each
shot's VIDEO is conformed to its own beat's real VO window
(`target = caption.end_ts - caption.start_ts`):
  * `actual >= target`            -> trim (`trim=duration=target`).
  * `(target-actual)/actual<=15%` -> imperceptible speed-up (`setpts`).
  * otherwise                     -> freeze-frame pad (`tpad stop_mode=clone`).
Once every segment is conformed to its own beat's real VO duration, the
concatenated video timeline becomes IDENTICAL to the VO/caption timeline, so
Stage 2's `enable=between(...)` gating just works directly off the real
captions JSON -- no separate caption-alignment pass is needed.

BEAT/SHOT MISMATCH POLICY (a real gap nothing upstream has addressed --
Budget Gate can and does cut a shot, but the captions cover every beat
regardless). CHOSEN: HOLD a neighboring shot's frame across the orphaned
beat's VO window (extend the PRECEDING shot's last frame if one exists,
else the FOLLOWING shot's first frame for a leading gap) -- never drop the
VO audio or its caption for that beat. Rejected alternative: cut the
orphaned beat's audio out of the concatenated VO track. Why HOLD wins:
  1. It never touches the VO audio a second time. The whole point of
     "duration conforming" above is that the VO's baked-in cumulative
     timestamps are sacrosanct once synthesized; cutting a slice out would
     require re-deriving every LATER beat's caption timing to shift back by
     the removed span -- reintroducing exactly the kind of derived-timing
     drift the real-measured-duration design elsewhere in this pipeline
     exists to avoid.
  2. Per `agents/budget_gate.py`'s own docstring, a cut shot is
     DETERMINISTICALLY the single LOWEST-WEIGHT shot in the job -- by
     construction never the hook/cta/top-proof shot. The beat losing its
     dedicated clip is therefore the least narratively-critical one; a held
     frame across that one beat's window is a materially smaller quality hit
     than silently deleting a line of ad copy the seller's script (and the
     already-synthesized VO) actually contains.
  3. Precedent: `agents/ken_burns_fallback_node.py` already treats "a
     visibly frozen/broken-looking frame" as a real defect worth engineering
     around (its faintest zoom is applied even to a `static` camera move for
     exactly this reason). The HOLD mechanism here reuses the same
     `tpad stop_mode=clone`/`start_mode=clone` primitive Stage 1 already uses
     for a normal freeze-pad conform, so an orphaned beat is rendered with
     the identical "extend a real frame" technique as a beat whose own real
     clip just happened to be shorter than its VO window -- not a special
     second code path.
A shot's clip that fails to DOWNLOAD (network/OSS error) is treated by the
exact same mechanism -- see `assemble_master_cut`'s per-shot try/except: one
shot's fetch failure degrades that beat to a held-frame gap (logged, traced,
never crashes the whole assembly), mirroring `video_gen_node.py`'s per-shot
failure isolation posture. `test_assembly_agent.py::test_shot_download_failure_is_isolated_as_a_held_frame_gap`
locks this in.

CANVAS SELECTION. Computed once per job from the REAL (ffprobe'd, never
`resolution_used`) dimensions of every `status=="passed"` (real Wan, not
Ken-Burns) clip -- the largest-area one wins, rounded down to even via
`force_divisible_by=2`. If every usable shot in the job is a Ken-Burns
fallback, 1920x1080 is used (§5.9's own fixed fallback spec).

CAPTION TEXT ZONE. `Shot.text_overlay_zone == "none"` (or no shot at all, for
an orphaned/held beat) falls back to `lower_third` rather than skipping the
caption -- most short-form ad video autoplays muted, so an unburned caption
is a functionally DROPPED line of ad copy, not a cosmetic omission.

PUBLIC SIGNATURE NOTE. `assemble_master_cut` returns only the master-cut URI
(per this task's own requested signature). The node wrapper needs a little
more (shot_count / total_duration_sec for the `master_cut_ready` C2 event) --
rather than widen the public contract, `assemble_master_cut` is a thin
wrapper around `_assemble_master_cut_impl`, which returns the richer
`AssemblyResult`; the node wrapper calls the richer function directly. Tests
of the "pure" logic call the public function per this task's spec; the
richer function is exercised by the node-wrapper tests.

video-gen-fidelity PHASE 3 -- CTA/ending fix (a real, independently confirmed
bug, distinct from the prompt-fidelity fix in video_gen_node.py/
shot_list_agent.py on the same branch). Confirmed root cause: the ad's final
shot's own natural motion (lift -> swing -> SETTLE) was being hard-trimmed
right at its beat's measured VO-window boundary -- e.g. a real 4.0s clip
conformed down to a 3.0s VO window, cutting off exactly the moment the clip
would be settling/resolving, with zero fade anywhere in the master cut. Two
changes, both scoped to ONLY the LAST rendered segment (the one that ends the
master cut) and ONLY the trim-direction default for mid-ad human-interaction
shots elsewhere:
  1. The LAST segment's own duration-conform NEVER hard-trims its tail
     anymore (`_ConformPlan("keep", ...)`, new mode). If it overruns its VO
     window, its full natural length is kept; Stage 2's existing `apad` +
     `-shortest` combo (already in the pipeline for absorbing sub-frame
     rounding drift) transparently extends the VO audio with trailing silence
     to match a LONGER video too -- `apad` with no explicit pad length pads
     indefinitely, and `-shortest` then just stops the output at whichever
     stream is actually longer, which is now the video. No new audio-side
     code needed for that half. If it's short, the freeze-frame hold is used
     unconditionally (never the imperceptible stretch) -- a barely-perceptible
     speed-up is the wrong trade specifically at the one moment the ad should
     feel most settled, not merely "not stretched too far."
  2. A short fade-out (`AUDIO_FADE_SEC`/`VIDEO_FADE_SEC`) is burned into the
     very end of the master cut in Stage 2, so the ad now visibly/audibly
     resolves instead of stopping dead the instant the last caption's window
     closes. Computed from the REAL summed duration of the rendered Stage-1
     segments (each segment's own render function now returns the exact
     duration it produced), not `target`, so the fade lands correctly even
     when the last segment's "keep" mode made it longer than planned.
  3. Separately (not the ending fix, but the same "which end of a clip is
     weakest" question): for `product_in_hand`/`worn_in_use` MID-ad shots
     specifically, a `trim`-mode conform now prefers cutting from the START
     of the clip, not the end -- i2v models' documented bias toward
     under-motion early in a clip (the rendered motion "leaks" from the
     static reference image and only departs from it over time -- NeurIPS
     2024 "Conditional Image Leakage in I2V Diffusion," arXiv:2406.15735)
     means the clip's own weakest, least-in-motion frames are its opening
     ones; entering the segment mid-motion (standard "enter late" editing
     practice) discards exactly that weak opening instead of the clip's own
     natural end. The hook shot and static product-alone shot types keep the
     original end-trim default -- a hook needs its opening frame intact for
     attention-grabbing, and a static product-alone shot's identity fidelity
     is strongest closest to the reference photo, i.e. in its own opening
     frames.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
import textwrap
from typing import Awaitable, Callable, NamedTuple, Optional

import ffmpeg
from langchain_core.callbacks.manager import adispatch_custom_event
from langchain_core.runnables import RunnableConfig

from agents._oss import _download_to_temp, upload_master_cut_to_oss
from graph.state import GeneratedShot, ProductCutState, Shot, Voiceover, WinningScript

logger = logging.getLogger("productcut.agents.assembly_agent")

# Canonical output frame rate -- forces true CFR across every segment
# regardless of source fps (Wan's realized fps is never assumed to be 30;
# Ken-Burns is already native CFR30). See module docstring.
FPS = 30

# §5.9's own fixed fallback spec -- used only when every usable clip in the
# job is a Ken-Burns fallback (no "passed" clip to measure a real canvas from).
DEFAULT_CANVAS = (1920, 1080)

# Imperceptible-speed-up ceiling (module docstring, DURATION CONFORMING).
STRETCH_MAX_RATIO = 0.15

# PHASE 3 CTA/ending fix -- module docstring point 2. Short enough to read as
# a deliberate close, not a slow fade that eats into the ad's last real content.
AUDIO_FADE_SEC = 0.4
VIDEO_FADE_SEC = 0.5

# PHASE 3 point 3 -- mirrored from agents/shot_list_agent.py's own
# HUMAN_INTERACTION_SHOT_TYPES (hand-kept in sync, same posture
# video_gen_node.py's own mirrored copy already uses).
_HUMAN_INTERACTION_SHOT_TYPES = frozenset({"product_in_hand", "worn_in_use"})

# Stage-1 intermediate segments favor speed (re-encoded again in Stage 2);
# Stage 2 is the mezzanine deliverable and favors quality.
STAGE1_CRF = 16
STAGE1_PRESET = "veryfast"
STAGE2_CRF = 16
STAGE2_PRESET = "slow"

# Forces identical timebase across every Stage-1 output so the Stage-2 concat
# DEMUXER (no re-encode of the concat step itself) never hits a timebase
# mismatch -- safe here specifically because Stage 1 already normalized every
# segment to identical codec/size/fps/pix_fmt/no-audio.
STAGE1_TRACK_TIMESCALE = 15360

# Explicit fontfile (not a bare font-family lookup) for determinism -- see
# module docstring. Empirically confirmed (a real ffmpeg-python render against
# this exact path succeeded, including the Windows-drive-letter-colon escaping
# that ffmpeg-python's own two-pass `escape_chars` handles correctly). Env-
# overridable per this codebase's "flag, don't hardcode forever" pattern
# (budget_gate.py's DEFAULT_JOB_BUDGET_CAP) since a non-Windows deployment
# target would need a different default.
DEFAULT_CAPTION_FONT_PATH = os.getenv("ASSEMBLY_CAPTION_FONT_PATH", "C:/Windows/Fonts/arialbd.ttf")

_WRAP_WIDTH_LOWER_THIRD = 22
_WRAP_WIDTH_SIDE = 13


class AssemblyError(Exception):
    """A hard, job-level Assembly precondition failed (VO audio/captions
    could not be fetched or parsed) -- there is no meaningful ad without a
    voiceover, so this is raised, not degraded, mirroring
    `voiceover_caption_agent_node`'s own hard-precondition posture on
    `state["winning_script"]`."""


DownloadFn = Callable[[str], str]
ProbeFn = Callable[[str], dict]
UploadFn = Callable[[str], str]


class AssemblyResult(NamedTuple):
    """The richer result `_assemble_master_cut_impl` produces; the public
    `assemble_master_cut` returns only `.master_cut_uri` (see module
    docstring's PUBLIC SIGNATURE NOTE)."""

    master_cut_uri: str
    shot_count: int              # number of REAL segments actually rendered (held-frame gaps don't add one)
    total_duration_sec: float    # ffprobe'd duration of the finished master cut
    degraded_beats: list[dict]   # [{beat_index, shot_id, reason}] -- beats that fell back to a held frame


class _SegmentPlan(NamedTuple):
    """One Stage-1 segment to render: `shot`'s clip, conformed to its own
    beat's real VO window, plus any adjacent orphaned-beat time folded in as
    a held-frame pad (module docstring, BEAT/SHOT MISMATCH POLICY)."""

    beat_index: int
    shot: Shot
    target: float
    pad_before: float
    pad_after: float


class _ConformPlan(NamedTuple):
    mode: str              # "trim" | "stretch" | "freeze" | "keep"
    freeze_pad_sec: float   # >0 only when mode == "freeze"
    # "keep" (PHASE 3, last-segment only): the clip's own full natural length
    # is used verbatim, never trimmed down to `target` -- see module
    # docstring's PHASE 3 point 1.


# ---------------------------------------------------------------------------
# Pure logic (no ffmpeg/network) -- unit-testable without real I/O.
# ---------------------------------------------------------------------------
def _map_shots_by_beat(shot_list: list[Shot]) -> dict[int, Shot]:
    """Map `winning_script.beats` index -> `Shot` via
    `shot.justification.treatment_ref` -- the ONLY reliable mapping (see
    module docstring). A duplicate `treatment_ref` (should not happen; the
    Shot-List Agent is expected to emit one shot per beat_treatment it
    consumes) keeps the FIRST shot encountered and logs, rather than
    crashing or silently overwriting.
    """
    mapping: dict[int, Shot] = {}
    for shot in shot_list:
        beat_index = shot.get("justification", {}).get("treatment_ref")
        if beat_index is None:
            continue
        if beat_index in mapping:
            logger.warning(
                "Assembly: beat_index %s already mapped to shot %s -- shot %s "
                "ignored (duplicate treatment_ref, keeping the first).",
                beat_index, mapping[beat_index].get("shot_id"), shot.get("shot_id"),
            )
            continue
        mapping[beat_index] = shot
    return mapping


def _select_canvas(usable_shots_by_beat: dict[int, Shot], probes: dict[int, dict]) -> tuple[int, int]:
    """Largest-area REAL (`status=="passed"`) clip's dimensions, rounded down
    to even; `DEFAULT_CANVAS` if no real clip is usable (module docstring)."""
    best: Optional[tuple[int, int]] = None
    best_area = -1
    for beat_index, shot in usable_shots_by_beat.items():
        if shot.get("status") != "passed":
            continue
        probe = probes.get(beat_index)
        if not probe:
            continue
        w, h = int(probe["width"]), int(probe["height"])
        area = w * h
        if area > best_area:
            best_area = area
            best = (w, h)
    if best is None:
        return DEFAULT_CANVAS
    w, h = best
    return (w - (w % 2), h - (h % 2))


def _plan_segments(
    beats: list[dict],
    captions: list[dict],
    shots_by_beat: dict[int, Shot],
) -> list[_SegmentPlan]:
    """Build the ordered Stage-1 segment plan, folding any orphaned beat's
    real VO duration into an adjacent real segment's hold-pad (module
    docstring, BEAT/SHOT MISMATCH POLICY). Returns `[]` iff there is no
    usable shot anywhere in the job (caller renders a single placeholder
    spanning the whole VO instead).

    `shots_by_beat` must already be filtered to USABLE shots only (a real
    clip successfully downloaded+probed) -- a beat absent from it is treated
    identically whether the cause is "Budget Gate cut this shot" or "this
    shot's clip failed to fetch"; both are real, both get the same held-frame
    treatment, by design (see module docstring).
    """
    n = min(len(beats), len(captions))
    if len(beats) != len(captions):
        logger.warning(
            "Assembly: winning_script has %d beat(s) but the captions track has "
            "%d entr(y/ies) -- using the shorter length (%d); this should not "
            "happen if voiceover_caption_agent ran against the SAME winning_script.",
            len(beats), len(captions), n,
        )
    targets = [max(0.0, float(captions[i]["end_ts"]) - float(captions[i]["start_ts"])) for i in range(n)]
    real_indices = [i for i in range(n) if i in shots_by_beat]
    if not real_indices:
        return []

    pads: dict[int, dict[str, float]] = {i: {"before": 0.0, "after": 0.0} for i in real_indices}
    i = 0
    while i < n:
        if i in shots_by_beat:
            i += 1
            continue
        run_start = i
        while i < n and i not in shots_by_beat:
            i += 1
        run_end = i  # exclusive
        gap_total = sum(targets[run_start:run_end])
        if run_start > 0:
            # Maximal run -> beat run_start-1, if it exists, is real by construction.
            pads[run_start - 1]["after"] += gap_total
        elif run_end < n:
            # Leading gap run: hold the NEXT real shot's first frame backward.
            pads[run_end]["before"] += gap_total
        # else: unreachable -- real_indices is non-empty, so a gap run that is
        # both leading (run_start==0) and trailing (run_end==n) can't exist.

    return [
        _SegmentPlan(
            beat_index=idx,
            shot=shots_by_beat[idx],
            target=targets[idx],
            pad_before=pads[idx]["before"],
            pad_after=pads[idx]["after"],
        )
        for idx in real_indices
    ]


def _resolve_duration_conform(target: float, actual: float, *, is_last_segment: bool = False) -> _ConformPlan:
    """Trim / imperceptible-stretch / freeze-pad decision (module docstring,
    DURATION CONFORMING), including the 15% stretch-ceiling boundary.

    `is_last_segment` (PHASE 3): the segment that ends the whole master cut
    gets a different policy -- never hard-trimmed (module docstring, PHASE 3
    point 1). An overrun keeps the clip's full natural length ("keep"); a
    shortfall always freezes (never stretches -- a slight speed-up is the
    wrong trade at the one moment the ad should feel most settled).
    """
    if actual <= 1e-6:
        # Degenerate (a near-zero-duration probe) -- hold the whole target as
        # a single frame rather than dividing by ~0. Logged by the caller.
        return _ConformPlan("freeze", target)
    if is_last_segment:
        if actual >= target - 1e-9:
            return _ConformPlan("keep", 0.0)
        return _ConformPlan("freeze", target - actual)
    if actual >= target - 1e-9:
        return _ConformPlan("trim", 0.0)
    deficit = target - actual
    if deficit / actual <= STRETCH_MAX_RATIO:
        return _ConformPlan("stretch", 0.0)
    return _ConformPlan("freeze", deficit)


def _effective_zone(shot: Optional[Shot]) -> str:
    """`"none"` (or no shot at all, for a held/orphaned beat) falls back to
    `lower_third` -- see module docstring, CAPTION TEXT ZONE."""
    if shot is None:
        return "lower_third"
    zone = shot.get("text_overlay_zone", "none")
    return zone if zone != "none" else "lower_third"


def _caption_position_expr(zone: str) -> tuple[str, str]:
    if zone == "left_third":
        return "w*0.06", "(h-text_h)/2"
    if zone == "right_third":
        return "w*0.94-text_w", "(h-text_h)/2"
    return "(w-text_w)/2", "h*0.80-text_h"  # lower_third and any other value


def _wrap_caption_text(text: str, zone: str) -> str:
    """ffmpeg's `drawtext` does not auto-wrap -- pre-wrap ourselves (module
    docstring's researched 20-24 chars/lower_third, 12-14/side widths)."""
    width = _WRAP_WIDTH_SIDE if zone in ("left_third", "right_third") else _WRAP_WIDTH_LOWER_THIRD
    return textwrap.fill(text, width=width) if text.strip() else " "


def _captions_for_render(captions: list[dict], shots_by_beat: dict[int, Shot]) -> list[dict]:
    """Every beat's caption (ALL of them -- including orphaned/held beats,
    which still play their real VO line over a held frame; see module
    docstring's BEAT/SHOT MISMATCH POLICY), each carrying its rendering zone."""
    rendered = []
    for i, cap in enumerate(captions):
        rendered.append(
            {
                "text": cap.get("text", ""),
                "start_ts": float(cap.get("start_ts", 0.0)),
                "end_ts": float(cap.get("end_ts", 0.0)),
                "zone": _effective_zone(shots_by_beat.get(i)),
            }
        )
    return rendered


# ---------------------------------------------------------------------------
# Real ffmpeg (impure) -- probing, Stage 1, Stage 2.
# ---------------------------------------------------------------------------
def _probe_local(local_path: str) -> dict:
    """Default `probe_fn`: real, measured duration/width/height -- never
    `resolution_used`/`duration_sec_used` (module docstring)."""
    info = ffmpeg.probe(local_path)
    video_stream = next((s for s in info.get("streams", []) if s.get("codec_type") == "video"), None)
    duration = float(info["format"]["duration"])
    width = int(video_stream["width"]) if video_stream else 0
    height = int(video_stream["height"]) if video_stream else 0
    return {"duration": duration, "width": width, "height": height}


def _mktemp(suffix: str, prefix: str = "assembly_") -> str:
    fd, path = tempfile.mkstemp(suffix=suffix, prefix=prefix)
    os.close(fd)
    return path


def _safe_remove(path: Optional[str]) -> None:
    if path and os.path.exists(path):
        try:
            os.remove(path)
        except OSError:
            pass


def _render_stage1_segment(
    local_clip_path: str,
    out_path: str,
    canvas_w: int,
    canvas_h: int,
    target: float,
    pad_before: float,
    pad_after: float,
    actual_duration: float,
    fps: int = FPS,
    *,
    is_last_segment: bool = False,
    prefer_start_trim: bool = False,
) -> float:
    """Normalize one shot's clip onto the shared canvas and conform its
    duration to `target` (its own beat's real VO window), plus any held-frame
    pad folded in from an adjacent orphaned beat (module docstring).

    `is_last_segment` (PHASE 3): this segment ends the whole master cut, so
    its own conform never hard-trims (module docstring, PHASE 3 point 1).
    `prefer_start_trim` (PHASE 3 point 3): when a genuine `trim` IS applied
    (never on the last segment), cut from the clip's START instead of its
    end -- for mid-ad human-interaction shots only (caller decides).

    Returns the segment's actual rendered duration -- the caller sums these
    across every segment to know the master cut's real planned total
    duration for the Stage-2 fade-out (module docstring, PHASE 3 point 2),
    since a "keep"-mode last segment's effective length is `actual_duration`,
    not `target`.
    """
    conform = _resolve_duration_conform(target, actual_duration, is_last_segment=is_last_segment)

    stream = ffmpeg.input(local_clip_path).video
    stream = stream.filter(
        "scale", w=canvas_w, h=canvas_h, force_original_aspect_ratio="decrease",
        force_divisible_by=2, flags="lanczos",
    )
    stream = stream.filter("pad", canvas_w, canvas_h, "(ow-iw)/2", "(oh-ih)/2", color="black")
    stream = stream.filter("setsar", 1)

    if conform.mode == "trim":
        if prefer_start_trim:
            # Cut the clip's own weakest, least-in-motion frames (its
            # opening, per the i2v conditional-image-leakage bias) rather
            # than its end -- "enter late" (module docstring, PHASE 3 point 3).
            start = max(0.0, actual_duration - target)
            stream = stream.filter("trim", start=start, duration=target).filter("setpts", "PTS-STARTPTS")
        else:
            stream = stream.filter("trim", duration=target).filter("setpts", "PTS-STARTPTS")
    elif conform.mode == "stretch":
        ratio = target / actual_duration
        stream = stream.filter("setpts", f"{ratio}*PTS")
    # "freeze"/"keep": no trim/stretch filter here -- "freeze"'s pad amount is
    # folded into the tpad stop_duration below; "keep" plays the clip's own
    # full natural length verbatim.

    stream = stream.filter("fps", fps=fps)
    stream = stream.filter("format", "yuv420p")

    stop_pad = pad_after + (conform.freeze_pad_sec if conform.mode == "freeze" else 0.0)
    tpad_kwargs: dict = {}
    if pad_before > 1e-6:
        tpad_kwargs["start_duration"] = pad_before
        tpad_kwargs["start_mode"] = "clone"
    if stop_pad > 1e-6:
        tpad_kwargs["stop_duration"] = stop_pad
        tpad_kwargs["stop_mode"] = "clone"
    if tpad_kwargs:
        stream = stream.filter("tpad", **tpad_kwargs)

    # "keep" mode's effective length is the clip's own actual duration, never
    # `target` -- that's the whole point of PHASE 3 point 1 (module docstring).
    effective_target = actual_duration if conform.mode == "keep" else target
    total_duration = pad_before + effective_target + pad_after
    out = ffmpeg.output(
        stream, out_path,
        an=None,  # bare `-an` -- discard every clip's own baked-in audio (module docstring)
        vcodec="libx264", crf=STAGE1_CRF, preset=STAGE1_PRESET, pix_fmt="yuv420p",
        t=total_duration,
        **{"video_track_timescale": STAGE1_TRACK_TIMESCALE},
    ).overwrite_output()
    ffmpeg.run(out, capture_stdout=True, capture_stderr=True)
    return total_duration


def _render_placeholder_segment(canvas_w: int, canvas_h: int, duration: float, fps: int = FPS) -> str:
    """Extreme fallback: not a single shot in the whole job is usable (every
    fetch failed, or shot_list was empty). Still produces a real, playable
    video spanning the full VO -- mirrors
    `voiceover_caption_agent.py`'s own "the ad can assemble with captions
    only" posture for a wholly-failed VO, applied to the visual side."""
    out_path = _mktemp(".mp4", prefix="assembly_placeholder_")
    src = ffmpeg.input(f"color=c=black:s={canvas_w}x{canvas_h}:d={max(duration, 0.1)}:r={fps}", f="lavfi")
    out = ffmpeg.output(
        src, out_path, an=None, vcodec="libx264", crf=STAGE1_CRF, preset=STAGE1_PRESET,
        pix_fmt="yuv420p", t=max(duration, 0.1),
    ).overwrite_output()
    ffmpeg.run(out, capture_stdout=True, capture_stderr=True)
    return out_path


def _write_concat_list(paths: list[str]) -> str:
    list_path = _mktemp(".txt", prefix="assembly_concat_")
    with open(list_path, "w", encoding="utf-8") as fh:
        for p in paths:
            fh.write(f"file '{p.replace(chr(92), '/')}'\n")
    return list_path


def _render_master_cut(
    segment_paths: list[str],
    canvas_w: int,
    canvas_h: int,
    captions: list[dict],
    audio_local_path: str,
    out_path: str,
    font_path: str = DEFAULT_CAPTION_FONT_PATH,
    *,
    total_duration_hint: Optional[float] = None,
) -> list[str]:
    """Stage 2: concat-demux the Stage-1 segments, burn every caption, map the
    VO audio as the sole audio track, encode the mezzanine (module docstring).
    Returns the list of temp caption textfile paths it wrote (caller cleans
    them up -- ffmpeg reads them at run time, so they must outlive the call).

    `total_duration_hint` (PHASE 3 point 2): the REAL summed duration of the
    Stage-1 segments (caller computes it from each segment's own render
    return value, never `target`, since a "keep"-mode last segment can be
    longer than planned). Used to place the end-of-cut fade at the correct
    absolute timestamp; if omitted or too short to fit a fade, no fade is
    applied (defensive -- e.g. the placeholder-only path doesn't pass this).
    """
    list_path = _write_concat_list(segment_paths)
    text_paths: list[str] = []
    try:
        video_in = ffmpeg.input(list_path, format="concat", safe=0)
        audio_in = ffmpeg.input(audio_local_path)

        vstream = video_in.video
        for i, cap in enumerate(captions):
            wrapped = _wrap_caption_text(cap["text"], cap["zone"])
            text_path = _mktemp(".txt", prefix=f"assembly_caption_{i}_")
            with open(text_path, "w", encoding="utf-8") as fh:
                fh.write(wrapped)
            text_paths.append(text_path)
            x_expr, y_expr = _caption_position_expr(cap["zone"])
            vstream = vstream.filter(
                "drawtext",
                fontfile=font_path, textfile=text_path,
                fontsize="h/18", fontcolor="white",
                box=1, boxcolor="black@0.45", boxborderw=16,
                borderw=2, bordercolor="black@0.6",
                x=x_expr, y=y_expr,
                enable=f"between(t,{cap['start_ts']},{cap['end_ts']})",
            )

        # Loudness-normalize BEFORE padding (padding only adds trailing silence,
        # which loudnorm should never see or factor into its measurement). Real
        # measurement on a live-generated master cut found the raw concatenated
        # TTS output sitting at mean_volume -26.6dB -- quiet enough to read as
        # "no audio" on a device at moderate volume, even though the track is
        # genuinely present and unclipped (max_volume -8.8dB). EBU R128 -16 LUFS
        # integrated / -1.5dBTP true-peak ceiling is a standard streaming/social
        # loudness target for voice-led short-form video -- loud and consistent
        # without clipping on the AAC re-encode.
        astream = audio_in.audio.filter("loudnorm", i=-16, tp=-1.5, lra=11).filter("apad")

        # PHASE 3 point 2: a short end-of-cut fade so the ad visibly/audibly
        # resolves instead of stopping dead the instant the last caption's
        # window closes. Placed at `total_duration_hint - fade_len`, computed
        # from the REAL summed Stage-1 segment durations (not `target`), so it
        # still lands correctly when the last segment's "keep" mode (see
        # _resolve_duration_conform) made it longer than originally planned.
        if total_duration_hint and total_duration_hint > VIDEO_FADE_SEC:
            vstream = vstream.filter(
                "fade", t="out", st=max(0.0, total_duration_hint - VIDEO_FADE_SEC), d=VIDEO_FADE_SEC,
            )
        if total_duration_hint and total_duration_hint > AUDIO_FADE_SEC:
            astream = astream.filter(
                "afade", t="out", st=max(0.0, total_duration_hint - AUDIO_FADE_SEC), d=AUDIO_FADE_SEC,
            )

        out = ffmpeg.output(
            vstream, astream, out_path,
            vcodec="libx264", preset=STAGE2_PRESET, crf=STAGE2_CRF, pix_fmt="yuv420p",
            acodec="aac", audio_bitrate="192k", ar=48000, ac=2,
            movflags="+faststart", shortest=None,
            **{"profile:v": "high", "level": "4.2"},
        ).overwrite_output()
        ffmpeg.run(out, capture_stdout=True, capture_stderr=True)
        return text_paths
    finally:
        _safe_remove(list_path)


# ---------------------------------------------------------------------------
# Orchestration.
# ---------------------------------------------------------------------------
async def _assemble_master_cut_impl(
    shot_list: list[Shot],
    generated_shots: dict[str, GeneratedShot],
    voiceover: Voiceover,
    winning_script: WinningScript,
    job_id: str,
    *,
    download_fn: Optional[DownloadFn] = None,
    probe_fn: Optional[ProbeFn] = None,
    upload_fn: Optional[UploadFn] = None,
) -> AssemblyResult:
    """The real implementation; see module docstring's PUBLIC SIGNATURE NOTE
    for why `assemble_master_cut` (the requested public entry point) is a
    thin wrapper around this instead."""
    dl: DownloadFn = download_fn or _download_to_temp
    probe: ProbeFn = probe_fn or _probe_local
    upload: UploadFn = upload_fn or _make_upload_fn(job_id)

    beats = winning_script.get("beats") or []

    # --- Hard preconditions: no meaningful ad without a real VO track. -----
    captions_local = None
    audio_local = None
    try:
        captions_local = await asyncio.to_thread(dl, voiceover["caption_track_uri"])
        with open(captions_local, "r", encoding="utf-8") as fh:
            captions = json.load(fh)
    except Exception as exc:  # noqa: BLE001 -- reclassified into a clear AssemblyError
        raise AssemblyError(f"could not fetch/parse the captions track: {exc}") from exc
    finally:
        _safe_remove(captions_local)

    try:
        audio_local = await asyncio.to_thread(dl, voiceover["audio_uri"])
    except Exception as exc:  # noqa: BLE001
        raise AssemblyError(f"could not fetch the voiceover audio track: {exc}") from exc

    downloaded: dict[int, str] = {}
    probes: dict[int, dict] = {}
    degraded_beats: list[dict] = []
    segment_paths: list[str] = []
    caption_text_paths: list[str] = []
    out_local: Optional[str] = None

    try:
        shots_by_beat = _map_shots_by_beat(shot_list)

        # --- Per-shot pre-pass: download + probe every candidate clip.
        # A shot with no usable generated entry, or whose clip fails to
        # fetch, is folded into the SAME held-frame gap handling as a beat
        # Budget Gate never gave a shot to at all (module docstring).
        for beat_index, shot in shots_by_beat.items():
            entry = generated_shots.get(shot["shot_id"])
            if entry is None or shot.get("status") not in ("passed", "fallback"):
                logger.warning(
                    "Assembly: beat %d's shot %s has no usable generated clip "
                    "(status=%r) -- held-frame gap.",
                    beat_index, shot.get("shot_id"), shot.get("status"),
                )
                degraded_beats.append(
                    {"beat_index": beat_index, "shot_id": shot.get("shot_id"), "reason": "no usable generated clip"}
                )
                continue
            try:
                local_path = await asyncio.to_thread(dl, entry["video_uri"])
                probe_info = await asyncio.to_thread(probe, local_path)
            except Exception as exc:  # noqa: BLE001 -- one shot's fetch failure must not sink the batch
                logger.warning(
                    "Assembly: beat %d's shot %s clip fetch/probe failed (%s) -- "
                    "held-frame gap, assembly continues.",
                    beat_index, shot["shot_id"], exc,
                )
                degraded_beats.append({"beat_index": beat_index, "shot_id": shot["shot_id"], "reason": str(exc)})
                continue
            downloaded[beat_index] = local_path
            probes[beat_index] = probe_info

        usable_shots_by_beat = {i: s for i, s in shots_by_beat.items() if i in downloaded}
        canvas_w, canvas_h = _select_canvas(usable_shots_by_beat, probes)

        segments = _plan_segments(beats, captions, usable_shots_by_beat)

        total_planned_duration = 0.0
        if not segments:
            total = max((c.get("end_ts", 0.0) for c in captions), default=0.0)
            logger.warning(
                "Assembly: no usable shot anywhere in the job -- rendering a "
                "single black placeholder spanning the full %.2fs VO track.",
                total,
            )
            placeholder = await asyncio.to_thread(_render_placeholder_segment, canvas_w, canvas_h, total)
            segment_paths = [placeholder]
            total_planned_duration = total
        else:
            for seg in segments:
                seg_out = _mktemp(".mp4", prefix="assembly_seg_")
                is_last_segment = seg is segments[-1]
                # PHASE 3 point 3: mid-ad human-interaction shots prefer a
                # start-trim (never the last segment -- it never hard-trims at
                # all, see _resolve_duration_conform's is_last_segment branch;
                # never the hook/cta, which keep the original end-trim default).
                prefer_start_trim = (
                    not is_last_segment
                    and seg.shot.get("shot_type") in _HUMAN_INTERACTION_SHOT_TYPES
                    and seg.shot.get("beat_role") not in ("hook", "cta")
                )
                seg_duration = await asyncio.to_thread(
                    _render_stage1_segment,
                    downloaded[seg.beat_index], seg_out, canvas_w, canvas_h,
                    seg.target, seg.pad_before, seg.pad_after,
                    probes[seg.beat_index]["duration"],
                    is_last_segment=is_last_segment,
                    prefer_start_trim=prefer_start_trim,
                )
                segment_paths.append(seg_out)
                total_planned_duration += seg_duration

        captions_for_render = _captions_for_render(captions, shots_by_beat)
        out_local = _mktemp(".mp4", prefix="assembly_mastercut_")
        caption_text_paths = await asyncio.to_thread(
            _render_master_cut, segment_paths, canvas_w, canvas_h, captions_for_render, audio_local, out_local,
            total_duration_hint=total_planned_duration,
        )

        final_probe = await asyncio.to_thread(probe, out_local)
        master_cut_uri = await asyncio.to_thread(upload, out_local)

        return AssemblyResult(
            master_cut_uri=master_cut_uri,
            shot_count=len(segments),
            total_duration_sec=round(final_probe["duration"], 3),
            degraded_beats=degraded_beats,
        )
    finally:
        _safe_remove(audio_local)
        for p in downloaded.values():
            _safe_remove(p)
        for p in segment_paths:
            _safe_remove(p)
        for p in caption_text_paths:
            _safe_remove(p)
        _safe_remove(out_local)


async def assemble_master_cut(
    shot_list: list[Shot],
    generated_shots: dict[str, GeneratedShot],
    voiceover: Voiceover,
    winning_script: WinningScript,
    job_id: str,
    *,
    download_fn: Optional[DownloadFn] = None,
    probe_fn: Optional[ProbeFn] = None,
    upload_fn: Optional[UploadFn] = None,
) -> str:
    """Public entry point (this task's requested signature) -- stitches every
    real/fallback shot clip, the voiceover audio, and the burned captions
    into one finished master-cut MP4, uploads it to OSS, and returns the
    signed URL. See module docstring for the full design (two-stage ffmpeg
    pipeline, duration conforming, beat/shot mismatch policy).

    `download_fn`/`probe_fn`/`upload_fn` are injectable (this codebase's
    established `client=None` pattern) for credential-free/network-free
    testing; they default to the real OSS download, `ffmpeg.probe`, and the
    real OSS upload respectively.

    Returns only the URI -- see module docstring's PUBLIC SIGNATURE NOTE for
    where the richer `AssemblyResult` (shot_count/total_duration_sec/
    degraded_beats) is available for callers that need it.
    """
    result = await _assemble_master_cut_impl(
        shot_list, generated_shots, voiceover, winning_script, job_id,
        download_fn=download_fn, probe_fn=probe_fn, upload_fn=upload_fn,
    )
    return result.master_cut_uri


def _make_upload_fn(job_id: str) -> UploadFn:
    def _upload(local_path: str) -> str:
        return upload_master_cut_to_oss(local_path, job_id)

    return _upload


# ---------------------------------------------------------------------------
# LangGraph node wrapper.
# ---------------------------------------------------------------------------
async def assembly_agent_node(
    state: ProductCutState,
    config: Optional[RunnableConfig] = None,
) -> dict:
    """LangGraph node wrapper: reads `shot_list`/`generated_shots`/
    `voiceover`/`winning_script`/`job_id` from state, assembles the master
    cut, writes `state["master_cut_uri"]`, and emits the proposed C2
    `master_cut_ready` event (graph/events.py v5).

    NOT a parallel-branch race like `voiceover_caption_agent_node` was (v7
    changelog, graph/state.py) -- by the time this runs, BOTH its upstream
    branches (voiceover_caption_agent, and the continuity retry loop via
    `continuity_gate`'s "end" route) have already joined (see graph/build.py's
    `defer=True` on this node -- LangGraph's `LastValueAfterFinish`/deferred-
    node mechanism, verified against a real compiled-graph test in
    tests/test_continuity_loop_e2e.py). It is therefore a single, sequential
    writer at this point in the graph and safely uses the normal shared
    `reasoning_trace` read-modify-write pattern every other (non-parallel)
    node in this codebase uses.

    `state["winning_script"]`/`state["voiceover"]` are accessed directly
    (KeyError, not `.get(...)`) -- by the time this node can possibly run,
    both are guaranteed to exist (this node's own graph position is
    downstream of both producers); a KeyError here means a wiring bug, not a
    normal runtime-data gap, matching `voiceover_caption_agent_node`'s own
    posture on the same precondition.
    """
    job_id = state.get("job_id", "unknown_job")
    shot_list = state.get("shot_list", [])
    generated_shots = state.get("generated_shots", {})
    voiceover = state["voiceover"]
    winning_script = state["winning_script"]

    result = await _assemble_master_cut_impl(shot_list, generated_shots, voiceover, winning_script, job_id)

    await adispatch_custom_event(
        "master_cut_ready",
        {
            "uri": result.master_cut_uri,
            "shot_count": result.shot_count,
            "total_duration_sec": result.total_duration_sec,
        },
        config=config,
    )

    trace_note = (
        f"\n[assembly_agent] assembled master cut from {result.shot_count} shot(s), "
        f"{result.total_duration_sec:.2f}s total."
    )
    if result.degraded_beats:
        trace_note += (
            f" {len(result.degraded_beats)} beat(s) held a neighboring shot's frame "
            "(no usable clip -- Budget Gate cut, or a fetch failure): "
            + ", ".join(f"beat {b['beat_index']}" for b in result.degraded_beats)
            + "."
        )

    return {
        "master_cut_uri": result.master_cut_uri,
        "reasoning_trace": state.get("reasoning_trace", "") + trace_note,
    }


__all__ = [
    "FPS",
    "DEFAULT_CANVAS",
    "STRETCH_MAX_RATIO",
    "DEFAULT_CAPTION_FONT_PATH",
    "AssemblyError",
    "AssemblyResult",
    "assemble_master_cut",
    "assembly_agent_node",
]
