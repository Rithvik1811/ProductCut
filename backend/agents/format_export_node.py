"""
Format Export Node — Phase 5, after assembly_agent.
Spec: docs/TECHNICAL_DOCUMENTATION.md §5.13.

Recomposes the master cut into three aspect ratios:
  9:16  (1080×1920) — TikTok / Reels / Shorts
  1:1   (1080×1080) — Feed
  16:9  (1920×1080) — YouTube / landscape

Uses FFmpeg blurred-background fill on the already-generated master cut —
no additional LLM or video-gen cost. Each format scales the video to fit
entirely within the target frame (no cropping of content) and fills any
letterbox/pillarbox space with a blurred version of the same frame.

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
from langchain_core.runnables import RunnableConfig

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


def _render_export(
    src_path: str,
    out_path: str,
    tgt_w: int,
    tgt_h: int,
    src_w: int,
    src_h: int,
) -> None:
    """Fit the master cut into tgt_w×tgt_h without cropping (blurred-background fill).

    The foreground is scaled to fit entirely inside the target frame.
    The background is the same frame scaled to fill (cropped) then blurred,
    eliminating letterbox/pillarbox black bars while preserving all content.
    """
    inp = ffmpeg.input(src_path)
    split = inp.video.filter_multi_output("split")
    bg = (
        split[0]
        .filter("scale", tgt_w, tgt_h, force_original_aspect_ratio="increase")
        .filter("crop", tgt_w, tgt_h)
        .filter("gblur", sigma=20)
    )
    fg = split[1].filter(
        "scale", tgt_w, tgt_h,
        force_original_aspect_ratio="decrease",
        force_divisible_by=2,
    )
    video = (
        ffmpeg.overlay(bg, fg, x="(W-w)/2", y="(H-h)/2")
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


async def format_export_node(state: ProductCutState, config: RunnableConfig) -> dict:
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
    await adispatch_custom_event("job_complete", payload, config=config)

    # Update job status to "completed" in RDS (best-effort; never blocks the return).
    database_url = os.environ.get("DATABASE_URL")
    if database_url:
        try:
            conn = await psycopg.AsyncConnection.connect(database_url, row_factory=dict_row)
            try:
                await update_job_status(conn, job_id, "completed")
            finally:
                await conn.close()
        except Exception as exc:
            logger.warning("format_export_node: could not update job status: %s", exc)

    return {"exports": exports}
