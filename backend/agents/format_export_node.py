"""
Format Export Node — Phase 5, after assembly_agent.
Spec: docs/TECHNICAL_DOCUMENTATION.md §5.13.

Recomposes the master cut into three aspect ratios:
  9:16  (1080×1920) — TikTok / Reels / Shorts
  1:1   (1080×1080) — Feed
  16:9  (1920×1080) — YouTube / landscape

Uses FFmpeg center-crop + scale on the already-generated master cut —
no additional LLM or video-gen cost. The reserved negative space baked into
the shot schema means the product is never cropped off by any of these
recompositions (the text-overlay zone is known and safe).

Stores signed OSS URLs in state["exports"] as:
  {"aspect_9x16": url, "aspect_1x1": url, "aspect_16x9": url}
"""
from __future__ import annotations

import asyncio
import logging
import os
import tempfile

import ffmpeg
import psycopg
from psycopg.rows import dict_row
from langchain_core.callbacks.manager import adispatch_custom_event

from agents._oss import _download_to_temp, upload_export_to_oss
from db.jobs import update_job_status
from graph.state import ProductCutState

logger = logging.getLogger("productcut.agents.format_export_node")

# (target_width, target_height) for each export format
EXPORT_FORMATS: dict[str, tuple[int, int]] = {
    "aspect_9x16": (1080, 1920),
    "aspect_1x1":  (1080, 1080),
    "aspect_16x9": (1920, 1080),
}

# OSS filename (including sub-folder) for each format
EXPORT_FILENAMES: dict[str, str] = {
    "aspect_9x16": "exports/9x16.mp4",
    "aspect_1x1":  "exports/1x1.mp4",
    "aspect_16x9": "exports/16x9.mp4",
}


def _crop_params(
    src_w: int, src_h: int, tgt_w: int, tgt_h: int
) -> tuple[int, int, int, int]:
    """Return (crop_w, crop_h, x_offset, y_offset) to center-crop src to tgt ratio.

    The crop preserves as much of the frame as possible before scaling.
    When source is wider than target → crop the sides.
    When source is taller than target → crop top/bottom.
    All values are forced even so libx264 never rejects them.
    """
    src_ratio = src_w / src_h
    tgt_ratio = tgt_w / tgt_h

    if src_ratio > tgt_ratio:
        # Source is wider — crop left and right
        crop_h = src_h
        crop_w = int(src_h * tgt_ratio)
        crop_w -= crop_w % 2
        crop_x = (src_w - crop_w) // 2
        crop_y = 0
    else:
        # Source is taller (or equal) — crop top and bottom
        crop_w = src_w
        crop_h = int(src_w / tgt_ratio)
        crop_h -= crop_h % 2
        crop_x = 0
        crop_y = (src_h - crop_h) // 2

    return crop_w, crop_h, crop_x, crop_y


def _render_export(
    src_path: str,
    out_path: str,
    tgt_w: int,
    tgt_h: int,
    src_w: int,
    src_h: int,
) -> None:
    """Center-crop + scale the master cut to one target format and write to out_path."""
    crop_w, crop_h, crop_x, crop_y = _crop_params(src_w, src_h, tgt_w, tgt_h)
    inp = ffmpeg.input(src_path)
    video = (
        inp.video
        .filter("crop", crop_w, crop_h, crop_x, crop_y)
        .filter("scale", tgt_w, tgt_h, flags="lanczos")
        .filter("setsar", 1)
    )
    (
        ffmpeg.output(
            video, inp.audio, out_path,
            vcodec="libx264", acodec="aac",
            video_bitrate="4M", audio_bitrate="128k",
            preset="fast", crf=18,
        )
        .overwrite_output()
        .run(quiet=True)
    )


async def generate_format_exports(
    master_cut_uri: str,
    job_id: str,
    *,
    bucket=None,
) -> dict[str, str]:
    """Download the master cut and produce all three format exports.

    Returns a dict keyed by aspect ratio name (aspect_9x16, aspect_1x1,
    aspect_16x9) → signed OSS GET URL.
    """
    local_master = await asyncio.to_thread(_download_to_temp, master_cut_uri)
    try:
        probe = ffmpeg.probe(local_master)
        vs = next(s for s in probe["streams"] if s["codec_type"] == "video")
        src_w = int(vs["width"])
        src_h = int(vs["height"])
        logger.info(
            "format_export_node: source %dx%d, generating exports for job %s",
            src_w, src_h, job_id,
        )

        exports: dict[str, str] = {}
        for key, (tgt_w, tgt_h) in EXPORT_FORMATS.items():
            fd, out_path = tempfile.mkstemp(suffix=".mp4", prefix=f"export_{key}_")
            os.close(fd)
            try:
                await asyncio.to_thread(
                    _render_export, local_master, out_path, tgt_w, tgt_h, src_w, src_h
                )
                url = await asyncio.to_thread(
                    upload_export_to_oss, out_path, job_id, EXPORT_FILENAMES[key],
                    bucket=bucket,
                )
                exports[key] = url
                logger.info("format_export_node: %s -> %s", key, EXPORT_FILENAMES[key])
            finally:
                if os.path.exists(out_path):
                    os.remove(out_path)

        return exports

    finally:
        if os.path.exists(local_master):
            os.remove(local_master)


async def format_export_node(state: ProductCutState) -> dict:
    """LangGraph node: runs after assembly_agent, writes state['exports']."""
    master_cut_uri = state.get("master_cut_uri", "")
    job_id = state.get("job_id", "unknown")

    if not master_cut_uri:
        logger.warning("format_export_node: no master_cut_uri in state — skipping")
        return {}

    try:
        exports = await generate_format_exports(master_cut_uri, job_id)
    except Exception as exc:
        logger.error(
            "format_export_node: failed to generate exports for job %s: %s",
            job_id, exc, exc_info=True,
        )
        # Bug 4: re-raise so LangGraph emits run.error and the checkpoint stays
        # resumable at this node. Swallowing sends the graph to END successfully
        # with no job_complete, leaving the frontend in a permanent reconnect loop.
        raise

    # Emit the job_complete C2 event so the frontend Delivery section renders.
    voiceover = state.get("voiceover")
    payload: dict = {"master_cut_uri": master_cut_uri, "exports": exports}
    if voiceover:
        payload["voiceover"] = voiceover
    await adispatch_custom_event("job_complete", payload)

    # Update job status to "complete" in RDS (best-effort; never blocks the return).
    database_url = os.environ.get("DATABASE_URL")
    if database_url:
        try:
            conn = await psycopg.AsyncConnection.connect(database_url, row_factory=dict_row)
            try:
                await update_job_status(conn, job_id, "complete")
            finally:
                await conn.close()
        except Exception as exc:
            logger.warning("format_export_node: could not update job status: %s", exc)

    return {"exports": exports}
