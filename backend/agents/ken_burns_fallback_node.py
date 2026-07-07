"""
Ken-Burns Fallback Node -- deterministic pan/zoom over the static product photo
(Phase 3, RR). Spec of record: docs/TECHNICAL_DOCUMENTATION.md §5.9.

WHAT THIS IS FOR. When the Video-Gen Node (§5.8, agents/video_gen_node.py) hits
a HARD infrastructure failure on a shot -- an actual timeout / API error /
budget-exceeded, NOT a quality problem -- it does not sink the whole run. It
hands that shot off with `status == "fallback_requested"` and a
`failure_reason`, spending no Continuity retry (the retry budget is reserved
for quality drift, §5.10). This node is the graceful-degradation catch: it
renders a simple Ken-Burns (pan/zoom) animation over the shot's own static
reference photo for the shot's own `duration_sec`, marks the shot
`status = "fallback"`, and produces a `generated_shots[shot_id]` entry so
Assembly (§5.12) treats it like any other clip -- a slightly less dynamic shot
in an otherwise complete ad, instead of a failed job.

HAND-OFF CONTRACT (confirmed against agents/video_gen_node.py's INTERFACE
section and the real C1/C3 schemas -- graph/state.py v6, graph/shot_schema.py
v3 -- not a docs guess):
  * We process every shot whose `status == "fallback_requested"` (there can be
    zero, one, or several in one job); every other shot is passed through
    completely untouched -- same object, same values.
  * `retry_count` is GUARANTEED untouched by Video-Gen on this path and MUST
    stay untouched here too. This module never reads or writes that key. The
    retry budget stays reserved for the Continuity Agent's quality retries.
  * No `generated_shots[shot_id]` entry exists yet for a handed-off shot -- we
    write it (a handed-off shot has no video to reference until we render one).
  * On success a shot's status becomes the pre-existing frozen `"fallback"`
    value (NOT `"fallback_requested"` any more) -- rendering the Ken-Burns clip
    is exactly what makes the fallback real, per §5.9's "marks the shot
    status = fallback".

GENERATEDSHOT SHAPE. We mirror video_gen_node.py's GeneratedShot exactly,
including its three extra, non-C1 keys (`resolution_used` / `duration_sec_used`
/ `budget_clamped` -- see that module's "KNOWN DEPARTURES" #3 for why this is a
deliberate, cost-free extension of the un-validated GeneratedShot dict). A
Ken-Burns clip is always rendered at the full 1080p spec below and never budget-
clamped (ffmpeg is free -- there is no per-second Wan cost to clamp against), so
`resolution_used="1080P"`, `duration_sec_used=shot["duration_sec"]`,
`budget_clamped=False`. `attempt` is 1: this is the first (and only) render on
the fallback path, not a retry of a real generation.

OUTPUT SPEC (matches Wan2.6-i2v's own typical output so Assembly's later concat
is homogeneous): MP4 / H.264 (libx264) / yuv420p / 30fps / 1920x1080 (16:9),
WITH a silent AAC audio track. Wan clips carry a real audio track (confirmed in
de-risk testing); an audio-less clip would desync/fail the concat, so we append
an `anullsrc` silent stereo track rather than shipping a video-only file.

DIRECTORIAL INTENT PRESERVED (not a generic default). The pan/zoom is chosen
from the shot's OWN `camera_move` field (see `ken_burns_expressions`): a
`push_in` shot still reads as a push-in, a `pull_back` as a pull-back, a `pan`
drifts sideways, etc. A fully static fallback reads as broken/frozen video, so
even `static` gets a faint zoom. Any unknown/missing `camera_move` degrades to
the subtlest (`static`) treatment rather than crashing.

FAILURE ISOLATION. One shot's Ken-Burns failure (missing photo, ffmpeg error,
upload error) must NOT block the other fallbacks in the same batch. On any such
failure we log it and leave that one shot's status as `"fallback_requested"`
(still visibly "needs handling", not silently swallowed and not marked done) --
the same graceful-degradation posture every other node in this pipeline takes.

    NOT wired into graph/build.py yet -- deliberately, mirroring
    agents/video_gen_node.py's own precedent at initial merge: every Phase 2/3
    agent in this codebase was built + tested standalone first and wired into
    the graph as its own integration commit (which also updates the shared
    graph end-to-end tests). Wiring happened in graph/build.py after this
    module's standalone test suite landed.
"""
from __future__ import annotations

import logging
import os
import tempfile
from typing import Callable, Optional

import ffmpeg
import httpx
from langchain_core.callbacks.manager import adispatch_custom_event
from langchain_core.runnables import RunnableConfig

# Reuse (never re-implement) Video-Gen's reference-photo resolver and its frozen
# hand-off status constant, so the "photo_N" -> URL convention and the
# "fallback_requested" spelling cannot silently drift between the two modules.
from agents.video_gen_node import FALLBACK_REQUESTED_STATUS, _resolve_reference_image_url
from agents._oss import upload_video_to_oss
from graph.state import GeneratedShot, ProductCutState, Shot

logger = logging.getLogger("productcut.agents.ken_burns_fallback_node")

# The pre-existing, frozen Shot.status value meaning "Ken-Burns has ALREADY
# rendered the clip" -- distinct from FALLBACK_REQUESTED_STATUS ("handed off,
# not rendered yet"). See graph/state.py's v6 note.
FALLBACK_STATUS = "fallback"

# Output spec (see module docstring). Wan-matching container/codec/rate/size.
FPS = 30
WIDTH = 1920
HEIGHT = 1080

# Centered zoom crop-window expressions, reused by several camera moves. zoompan
# evaluates these per-output-frame; `zoom` is the current zoom factor, so these
# keep the crop centered as the zoom changes.
_CENTER_X = "iw/2-(iw/zoom/2)"
_CENTER_Y = "ih/2-(ih/zoom/2)"

# Placeholder job segment for the default (non-node-wrapper) call path only. Real
# pipeline use ALWAYS goes through `ken_burns_fallback_node`, which injects a
# job-bound uploader carrying the true `state["job_id"]`; this constant is only
# ever reached when `generate_ken_burns_fallbacks` is called standalone with no
# injected `upload_fn` (e.g. an ad-hoc script), where no job_id is available.
_DEFAULT_JOB_ID = "unknown_job"

# upload_fn contract: (local_mp4_path, shot_id) -> signed_url. Deliberately does
# NOT take job_id -- `generate_ken_burns_fallbacks`'s signature (per §5.9 task
# spec) carries no job_id, so the node wrapper binds job_id into the uploader it
# injects instead. See `_make_oss_upload_fn`.
UploadFn = Callable[[str, str], str]


# ---------------------------------------------------------------------------
# camera_move -> (z, x, y) zoompan expressions (pure -- no ffmpeg needed).
# ---------------------------------------------------------------------------
def ken_burns_expressions(camera_move: str, frames: int) -> tuple[str, str, str]:
    """Map a shot's `camera_move` to zoompan (z, x, y) expressions.

    Honors the original directorial intent even on the fallback path: the 2D
    pan/zoom analog of each 3D camera move (a `push_in` still pushes in, a
    `pull_back` pulls back, a `pan` drifts sideways, a `tilt_up` drifts upward).
    Moves with no real 2D analog (`orbit`, `rack_focus`) degrade to a gentle
    zoom-in; a genuinely `static` shot still gets a very faint zoom, because a
    perfectly frozen frame reads as broken video. Anything unrecognized/missing
    falls back to the subtlest (`static`) treatment rather than raising -- one
    odd enum value must never crash the whole fallback batch.

    `frames` is the total output frame count (`round(duration_sec * FPS)`); it is
    substituted into the pan expressions so the drift completes over exactly the
    clip's length regardless of duration.

    zoompan expression variables: `on` = current output frame number (1-based),
    `zoom` = current zoom factor, `iw`/`ih` = (upscaled) input width/height.
    """
    # Seeded zoom-out: without the `if(eq(on,1),1.3,...)` seed, frame 1 renders
    # unzoomed and the clip visibly flashes before the pull-back starts.
    pull_back_z = "if(eq(on,1),1.3,max(zoom-0.0015,1.0))"

    if camera_move == "push_in":
        return "min(zoom+0.0015,1.3)", _CENTER_X, _CENTER_Y
    if camera_move == "pull_back":
        return pull_back_z, _CENTER_X, _CENTER_Y
    if camera_move == "pan":
        # Fixed zoom, crop window drifts left -> right across the clip.
        return "1.15", f"(iw-iw/zoom)*on/{frames}", _CENTER_Y
    if camera_move == "tilt_up":
        # Fixed zoom, crop window drifts bottom -> top across the clip.
        return "1.15", _CENTER_X, f"(ih-ih/zoom)*(1-on/{frames})"
    if camera_move in ("orbit", "rack_focus"):
        # No real 2D analog -- a gentle, slower-than-push_in zoom-in.
        return "min(zoom+0.0008,1.3)", _CENTER_X, _CENTER_Y
    # "static" and any unknown/missing value: a very faint zoom so the frame
    # doesn't read as frozen/broken.
    return "min(zoom+0.0004,1.06)", _CENTER_X, _CENTER_Y


# ---------------------------------------------------------------------------
# ffmpeg render (CPU-bound; callers run it via asyncio.to_thread).
# ---------------------------------------------------------------------------
def _run_ffmpeg_ken_burns(
    image_path: str,
    out_path: str,
    duration_sec: float,
    z_expr: str,
    x_expr: str,
    y_expr: str,
) -> None:
    """Render one Ken-Burns MP4 (video + silent audio) via the ffmpeg-python API.

    Gotchas handled (all real, all bite in practice):
      * zoompan `d=` is FRAMES, not seconds -- `round(duration_sec * FPS)`.
      * `-loop 1` with no `-t` cap makes zoompan loop forever -- BOTH the input
        (`t=` on `.input()`) and the output (`t=` on `.output()`) are capped.
      * The upfront `scale=8000:-1` upscale is anti-judder: zoompan rounds x/y to
        integer INPUT pixels, so panning/zooming a small source visibly steps;
        upscaling first gives sub-pixel precision at the 1920x1080 target.
      * `yuv420p` is required for broad player compatibility.
      * MP4's `moov` atom needs a seekable output, so we always render to a real
        file path (never pipe to stdout) and add `+faststart`.
    """
    frames = int(round(duration_sec * FPS))

    video = (
        ffmpeg
        .input(image_path, loop=1, framerate=FPS, t=duration_sec)
        .filter("scale", 8000, -1)  # anti-judder upscale -- see docstring
        .filter("zoompan", z=z_expr, x=x_expr, y=y_expr, d=frames, s=f"{WIDTH}x{HEIGHT}", fps=FPS)
    )
    # Silent stereo track so Assembly's later concat doesn't desync on an
    # audio-less clip (Wan clips carry real audio).
    silent_audio = ffmpeg.input("anullsrc=r=44100:cl=stereo", f="lavfi", t=duration_sec)

    stream = (
        ffmpeg
        .output(
            video,
            silent_audio,
            out_path,
            vcodec="libx264",
            pix_fmt="yuv420p",
            acodec="aac",
            r=FPS,
            t=duration_sec,
            movflags="+faststart",
            shortest=None,  # ffmpeg-python: a None value emits the bare `-shortest` flag
        )
        .overwrite_output()
    )
    ffmpeg.run(stream, capture_stdout=True, capture_stderr=True)


def _download_image_to_temp(url: str) -> str:
    """Download a remote image to a temp file and return its path (caller deletes).

    httpx is already a dependency. `follow_redirects=True` because OSS signed
    URLs / CDNs commonly 3xx.
    """
    resp = httpx.get(url, timeout=30.0, follow_redirects=True)
    resp.raise_for_status()
    suffix = os.path.splitext(url.split("?", 1)[0])[1] or ".img"
    fd, path = tempfile.mkstemp(suffix=suffix, prefix="kenburns_src_")
    with os.fdopen(fd, "wb") as fh:  # takes ownership of fd; closes it even on error
        fh.write(resp.content)
    return path


def render_ken_burns_clip(shot: Shot, product_photos: list[str]) -> str:
    """Render the Ken-Burns fallback clip for one shot; return the local MP4 path.

    Resolves the shot's reference photo (reusing Video-Gen's
    `_resolve_reference_image_url`), fetches it to a temp file if it's a URL
    (handles a plain local path too, defensively), picks the pan/zoom
    expressions from the shot's own `camera_move`, and renders at the shot's own
    `duration_sec`.

    Temp-file lifecycle: the downloaded SOURCE image is always cleaned up here.
    The rendered OUTPUT clip is NOT -- the caller owns it (it must survive until
    upload) and is responsible for deleting it afterward. Raises on any failure
    (missing photo / download error / ffmpeg error); the caller isolates that
    per-shot so it can't sink the batch.
    """
    image_url = _resolve_reference_image_url(shot["reference_image_id"], product_photos)
    if not image_url:
        raise ValueError(
            f"shot {shot['shot_id']}: no reference photo available "
            f"(reference_image_id={shot.get('reference_image_id')!r}, "
            f"{len(product_photos)} product photo(s))"
        )

    downloaded_tmp: Optional[str] = None
    if image_url.startswith(("http://", "https://")):
        downloaded_tmp = _download_image_to_temp(image_url)
        local_image = downloaded_tmp
    else:
        # Defensive: a bare local path (not a URL) is used directly.
        local_image = image_url
        if not os.path.exists(local_image):
            raise FileNotFoundError(f"shot {shot['shot_id']}: reference image not found: {local_image}")

    out_fd, out_path = tempfile.mkstemp(suffix=".mp4", prefix="kenburns_out_")
    os.close(out_fd)  # ffmpeg writes the file itself; we only needed a unique path

    z_expr, x_expr, y_expr = ken_burns_expressions(shot["camera_move"], int(round(shot["duration_sec"] * FPS)))
    try:
        _run_ffmpeg_ken_burns(local_image, out_path, shot["duration_sec"], z_expr, x_expr, y_expr)
    except Exception:
        # Don't leak the half-written output on a render failure.
        if os.path.exists(out_path):
            try:
                os.remove(out_path)
            except OSError:
                pass
        raise
    finally:
        if downloaded_tmp and os.path.exists(downloaded_tmp):
            try:
                os.remove(downloaded_tmp)
            except OSError:
                pass
    return out_path


# ---------------------------------------------------------------------------
# Default OSS uploader binding (see UploadFn contract note above).
# ---------------------------------------------------------------------------
def _make_oss_upload_fn(job_id: str) -> UploadFn:
    """Bind `job_id` into the reusable `_oss.upload_video_to_oss` so the resulting
    callable matches the `(local_path, shot_id) -> url` UploadFn contract.
    """
    def _upload(local_path: str, shot_id: str) -> str:
        return upload_video_to_oss(local_path, job_id, shot_id)

    return _upload


# ---------------------------------------------------------------------------
# Main entry point.
# ---------------------------------------------------------------------------
async def generate_ken_burns_fallbacks(
    shots: list[Shot],
    product_photos: list[str],
    upload_fn: Optional[UploadFn] = None,
) -> tuple[list[Shot], dict[str, GeneratedShot]]:
    """Render + upload a Ken-Burns fallback for every `fallback_requested` shot.

    Returns (updated_shots, new_generated_entries):
      * updated_shots is the FULL input list. Each successfully-rendered fallback
        shot has its `status` flipped to "fallback"; a fallback shot whose render
        or upload failed keeps `status == "fallback_requested"` (visibly still
        needs handling); every non-fallback shot passes through unchanged (same
        object, same values).
      * new_generated_entries has one GeneratedShot per successfully-rendered
        fallback shot only. Merge it into the outer state's `generated_shots`
        (the node wrapper does this) -- it never contains entries for shots the
        real Video-Gen node already produced.

    `upload_fn` is injectable (same `client=None` pattern as every other module
    here) for credential-free testing: a `(local_mp4_path, shot_id) -> url`
    callable. It defaults to the real OSS uploader; see the module-level
    `_DEFAULT_JOB_ID` note for why the standalone default uses a placeholder job
    segment while the node wrapper injects a job-bound uploader.

    `retry_count` is never read or written anywhere in here.
    """
    uploader: UploadFn = upload_fn or _make_oss_upload_fn(_DEFAULT_JOB_ID)

    import asyncio  # local: keeps the pure helpers above importable without asyncio

    updated_shots: list[Shot] = []
    new_entries: dict[str, GeneratedShot] = {}

    for shot in shots:
        if shot.get("status") != FALLBACK_REQUESTED_STATUS:
            updated_shots.append(shot)  # untouched pass-through -- same object
            continue

        shot_id = shot["shot_id"]
        clip_path: Optional[str] = None
        try:
            # ffmpeg render is CPU-bound -- run it off the event loop, matching
            # this codebase's async-everywhere convention.
            clip_path = await asyncio.to_thread(render_ken_burns_clip, shot, product_photos)
            # upload_fn may be a blocking (real oss2) call -- also off-loop.
            video_uri = await asyncio.to_thread(uploader, clip_path, shot_id)

            new_entries[shot_id] = {
                "video_uri": video_uri,
                "attempt": 1,
                # Extra, non-C1 GeneratedShot keys -- mirrors video_gen_node.py
                # exactly (its "KNOWN DEPARTURES" #3). Ken-Burns is always full-
                # spec and never budget-clamped.
                "resolution_used": "1080P",
                "duration_sec_used": shot["duration_sec"],
                "budget_clamped": False,
            }
            updated_shots.append({**shot, "status": FALLBACK_STATUS})
            logger.info("Ken-Burns: rendered + uploaded fallback for shot %s.", shot_id)
        except Exception as exc:  # noqa: BLE001 -- one shot's failure must not sink the batch
            logger.error(
                "Ken-Burns: fallback FAILED for shot %s (%s) -- leaving status "
                "'fallback_requested' (still needs handling), retry_count untouched.",
                shot_id, exc,
            )
            updated_shots.append(shot)  # status stays fallback_requested, unchanged
        finally:
            # The caller (this function) owns the rendered clip -- delete it once
            # it's uploaded (or on failure). The downloaded source image was
            # already cleaned up inside render_ken_burns_clip.
            if clip_path and os.path.exists(clip_path):
                try:
                    os.remove(clip_path)
                except OSError:
                    pass

    return updated_shots, new_entries


# ---------------------------------------------------------------------------
# LangGraph node wrapper.
# ---------------------------------------------------------------------------
async def ken_burns_fallback_node(
    state: ProductCutState,
    config: Optional[RunnableConfig] = None,
) -> dict:
    """LangGraph node wrapper: renders Ken-Burns fallbacks for every
    `fallback_requested` shot in `state["shot_list"]`.

    Injects a job-bound OSS uploader carrying the real `state["job_id"]` (so the
    entry point's job_id-less signature still produces correctly-namespaced OSS
    keys), and merges the new GeneratedShot entries INTO any existing
    `generated_shots` (never clobbering real Video-Gen output for other shots).

    Emits one C2 `shot_generated` event per successfully-rendered fallback clip,
    with `is_fallback=True` and `status="fallback"` -- the fallback counterpart
    to the Video-Gen node's `is_fallback=False` events, so the dashboard can tell
    a real clip from a graceful-degradation one. A shot whose render/upload
    failed (still `fallback_requested`) emits nothing -- it has no clip yet.
    Dispatched via `adispatch_custom_event` (mirrors budget_gate_node); `config`
    defaults to None so the node stays directly callable/testable.
    """
    shots = state.get("shot_list", [])
    product_photos = state.get("product_photos", [])
    job_id = state.get("job_id", _DEFAULT_JOB_ID)

    updated_shots, new_entries = await generate_ken_burns_fallbacks(
        shots, product_photos, upload_fn=_make_oss_upload_fn(job_id)
    )

    # One shot_generated event per rendered fallback clip, in shot-list order.
    for shot in updated_shots:
        shot_id = shot["shot_id"]
        if shot_id in new_entries:
            await adispatch_custom_event(
                "shot_generated",
                {
                    "shot_id": shot_id,
                    "generated": new_entries[shot_id],
                    "status": FALLBACK_STATUS,
                    "is_fallback": True,
                },
                config=config,
            )

    n_rendered = len(new_entries)
    n_still_requested = sum(1 for s in updated_shots if s.get("status") == FALLBACK_REQUESTED_STATUS)
    trace_note = f"\n[ken_burns_fallback] rendered {n_rendered} Ken-Burns fallback clip(s)."
    if n_still_requested:
        trace_note += (
            f" {n_still_requested} shot(s) still 'fallback_requested' after a "
            "render/upload failure (isolated, batch not blocked)."
        )

    return {
        "shot_list": updated_shots,
        "generated_shots": {**state.get("generated_shots", {}), **new_entries},
        "reasoning_trace": state.get("reasoning_trace", "") + trace_note,
    }


__all__ = [
    "FALLBACK_STATUS",
    "FPS",
    "WIDTH",
    "HEIGHT",
    "ken_burns_expressions",
    "render_ken_burns_clip",
    "generate_ken_burns_fallbacks",
    "ken_burns_fallback_node",
]
