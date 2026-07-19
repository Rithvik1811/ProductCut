"""
FastAPI app that streams LangGraph events over a WebSocket.

Endpoints (C2-aligned):
  GET  /health                   -> liveness + graph status
  POST /jobs                     -> create a new job (multipart: brief + photos + seller direction)
  GET  /jobs                     -> list jobs (newest first, optional ?seller_id=)
  GET  /jobs/{job_id}/state      -> current graph snapshot for a job
  WS   /ws/{job_id}              -> run (or resume) the graph and stream C2 events

WebSocket behaviour:
  - Fresh run: loads persisted job state from DB, passes it as initial_state to astream_events.
  - Resume (after interrupt): pass ?resolution=approve|retry_with_edit|accept_fallback; the
    handler detects the interrupted checkpoint and calls astream_events(Command(resume=...)).

Two event families are forwarded, both with envelope {type, job_id, ts, payload}:
  - Real C2 business events (graph.events.EventType) -- the frozen contract.
  - Raw LangGraph lifecycle events (on_chain_start, etc.) -- debugging passthrough only,
    NOT part of C2; do not build dashboard panels against these.
"""
from __future__ import annotations

import asyncio
import sys

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import json
import logging
import os
import uuid
from contextlib import AsyncExitStack, asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, get_args

from dotenv import load_dotenv

# Unlike the derisk/ scripts, this app was never loading backend/.env: every
# os.environ.get() below (and DATABASE_URL, DASHSCOPE_*, etc. read deeper in
# graph/build.py and the agents) silently saw nothing unless the caller
# exported vars manually or passed `uvicorn --env-file`.
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from graph.build import build_graph
from graph.events import EventType, build_event

_KNOWN_C2_TYPES = frozenset(get_args(EventType))

# Bug 1: per-job concurrency guard — prevents two simultaneous astream_events
# calls on the same thread_id which would corrupt checkpoint writes.
_run_locks: dict[str, asyncio.Lock] = {}

# Bug 8: node names the compiled graph emits on on_chain_start — used to
# synthesize node_started C2 events so the phase indicator advances.
_KNOWN_NODE_NAMES = frozenset({
    "brand_research_node", "product_truth_extractor", "concept_agent",
    "hook_checker", "pacing_checker", "body_checker", "cta_checker",
    "tone_checker", "meta_critic", "merge_validator", "copy_editor",
    "visual_direction_agent", "treatment_agent", "shot_list_agent",
    "budget_gate", "video_gen", "ken_burns_fallback", "continuity_agent",
    "continuity_gate", "voice_direction_agent", "voiceover_caption_agent",
    "assembly_agent", "format_export_node",
})

# Bug 15: valid ?resolution= values for the WS resume path.
_VALID_RESOLUTIONS = frozenset({"approve", "retry_with_edit", "accept_fallback"})

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("productcut.app")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

UPLOADS_DIR = Path(os.environ.get("UPLOADS_DIR", "./uploads"))
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

# Base URL the graph will use when building absolute photo URLs for DashScope.
# Must be reachable by DashScope's servers in prod (use a public URL).
BACKEND_BASE_URL = os.environ.get("BACKEND_BASE_URL", "http://localhost:8000")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _envelope(event_type: str, job_id: str, payload: Any) -> dict:
    return {"type": event_type, "job_id": job_id, "ts": _now_iso(), "payload": payload}


def _jsonable(obj: Any) -> Any:
    """Best-effort serialise to JSON-safe primitives."""
    try:
        json.dumps(obj)
        return obj
    except (TypeError, ValueError):
        if isinstance(obj, dict):
            return {k: _jsonable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_jsonable(v) for v in obj]
        return repr(obj)


# ---------------------------------------------------------------------------
# App + lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.exit_stack = AsyncExitStack()
    logger.info("Building LangGraph graph...")
    app.state.graph = await build_graph(exit_stack=app.state.exit_stack)
    logger.info("Graph compiled and ready.")
    # Bug 6: ensure jobs/seller_direction tables exist before serving any traffic
    try:
        from db.jobs import connect as db_connect, init_tables
        conn = await db_connect()
        try:
            await init_tables(conn)
            logger.info("DB tables verified/created.")
        finally:
            await conn.close()
    except Exception as exc:
        logger.error("Could not init DB tables: %s — jobs will fail if tables are absent.", exc)
    try:
        yield
    finally:
        await app.state.exit_stack.aclose()
        logger.info("Shutdown: released graph resources.")


app = FastAPI(title="ProductCut Backend", lifespan=lifespan)

# CORS: allow the Next.js dev server and any configured frontend origin
_CORS_ORIGINS = os.environ.get(
    "CORS_ORIGINS", "http://localhost:3000,http://localhost:3001"
).split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _CORS_ORIGINS],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve locally-uploaded photos in dev (in prod, use signed OSS URLs)
app.mount("/uploads", StaticFiles(directory=str(UPLOADS_DIR)), name="uploads")

# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


@app.get("/health")
async def health() -> dict:
    from fastapi.responses import JSONResponse
    graph = getattr(app.state, "graph", None)
    checkpointer_type = "none"
    if graph is not None:
        cp = getattr(graph, "checkpointer", None)
        checkpointer_type = type(cp).__name__ if cp else "none"
    return JSONResponse({
        "status": "ok",
        "service": "productcut-backend",
        "graph_ready": graph is not None,
        "checkpointer": checkpointer_type,
        "ts": _now_iso(),
    })


# ---------------------------------------------------------------------------
# POST /jobs  — create a new job
# ---------------------------------------------------------------------------


@app.post("/jobs")
async def create_job_endpoint(
    brief: str = Form(default=""),
    seller_id: str = Form(default=""),
    brand_name: str = Form(default=""),
    brand_url: str = Form(default=""),
    mood_words: str = Form(default=""),    # JSON array string, e.g. '["bold","warm"]'
    reference_ad: str = Form(default=""),
    never_do: str = Form(default=""),
    notes: str = Form(default=""),
    photos: list[UploadFile] = File(default=[]),
) -> dict:
    """Accept a seller's brief + photos + optional direction; return a job_id.

    Photos are saved to UPLOADS_DIR/{job_id}/ and exposed at /uploads/{job_id}/{name}.
    In production swap _save_photos for OSS upload (agents/_oss.py).
    """
    job_id = str(uuid.uuid4())

    # Save uploaded photos and build absolute URLs for DashScope
    photo_refs = await _save_photos(photos, job_id)

    # Build SellerDirection dict
    sd: dict = {}
    if mood_words:
        try:
            sd["mood_words"] = json.loads(mood_words)
        except json.JSONDecodeError:
            pass
    if reference_ad:
        sd["reference_ad"] = {"url_or_text": reference_ad, "why": ""}
    if never_do:
        sd["never_do"] = never_do
    if notes:
        sd["freeform"] = notes

    # Persist to DB — required: brief/photos reach the graph exclusively via this row.
    try:
        from db.jobs import connect as db_connect, create_job, upsert_seller_direction
        conn = await db_connect()
        try:
            await create_job(conn, job_id, seller_id or None, brief, photo_refs)
            if sd:
                await upsert_seller_direction(conn, job_id, sd)
        finally:
            await conn.close()
    except Exception as exc:
        logger.error("DB write failed for job %s: %s", job_id, exc)
        raise HTTPException(status_code=503, detail="Database unavailable — could not persist job. Please retry.")

    result: dict = {"job_id": job_id}
    if brand_name:
        result["brand_name"] = brand_name
    if brand_url:
        result["brand_url"] = brand_url
    return result


async def _save_photos(photos: list[UploadFile], job_id: str) -> list[str]:
    refs: list[str] = []
    valid = [p for p in photos if p and p.filename]
    if not valid:
        return refs
    job_dir = UPLOADS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    from agents._oss import upload_photo_to_oss

    _CONTENT_TYPES = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}

    for idx, photo in enumerate(valid):
        # Bug 16: prefix with index to prevent filename collisions overwriting product angles
        safe = f"{idx}_{Path(photo.filename).name}"  # type: ignore[arg-type]
        content = await photo.read()
        local_path = job_dir / safe
        local_path.write_bytes(content)
        suffix = Path(safe).suffix.lower()
        content_type = _CONTENT_TYPES.get(suffix, "image/jpeg")
        try:
            oss_url = upload_photo_to_oss(str(local_path), job_id, safe, content_type=content_type)
            refs.append(oss_url)
        except Exception as exc:
            # Fallback to localhost URL if OSS is unavailable (dev without OSS creds)
            logger.warning("OSS photo upload failed for %s — falling back to localhost URL: %s", safe, exc)
            refs.append(f"{BACKEND_BASE_URL}/uploads/{job_id}/{safe}")
    return refs


# ---------------------------------------------------------------------------
# GET /jobs  — list jobs for Library
# ---------------------------------------------------------------------------


@app.get("/jobs")
async def list_jobs_endpoint(
    seller_id: Optional[str] = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
) -> list[dict]:
    """Return job summaries (job_id, brief, status, created_at) newest first."""
    try:
        from db.jobs import connect as db_connect, list_jobs
        conn = await db_connect()
        try:
            return await list_jobs(conn, seller_id=seller_id, limit=limit)
        finally:
            await conn.close()
    except Exception as exc:
        logger.error("list_jobs failed: %s", exc)
        raise HTTPException(status_code=503, detail="Database unavailable")


# ---------------------------------------------------------------------------
# GET /jobs/{job_id}/state  — graph checkpoint snapshot
# ---------------------------------------------------------------------------


@app.get("/jobs/{job_id}/state")
async def get_job_state(job_id: str) -> dict:
    """Return the latest graph checkpoint for a job (values + next nodes).

    Useful for reconnecting a browser tab to an in-progress or interrupted job.
    """
    graph = app.state.graph
    config = {"configurable": {"thread_id": job_id}}
    try:
        snapshot = await graph.aget_state(config)
        if snapshot is None or not snapshot.values:
            raise HTTPException(status_code=404, detail="No checkpoint found for this job")
        return {
            "job_id": job_id,
            "state": _jsonable(dict(snapshot.values)),
            "next": list(snapshot.next) if snapshot.next else [],
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("aget_state failed for %s: %s", job_id, exc)
        raise HTTPException(status_code=500, detail=f"Checkpoint read failed: {exc}")


# ---------------------------------------------------------------------------
# GET /jobs/{job_id}/exports  — re-sign export URLs for a completed job
# ---------------------------------------------------------------------------


@app.get("/jobs/{job_id}/exports")
async def refresh_export_urls(job_id: str) -> dict:
    """Return fresh signed OSS URLs for a job's format exports and master cut.

    Signed URLs expire after 24 h (SIGNED_URL_TTL_SEC). This endpoint re-signs
    them on demand so the Library can play videos from previous sessions.
    """
    import os as _os
    # Skip signing if OSS credentials are absent (local dev without OSS)
    if not _os.environ.get("OSS_ACCESS_KEY_ID"):
        raise HTTPException(status_code=503, detail="OSS not configured")

    from agents._oss import oss_job_asset_key, sign_existing_key

    try:
        result: dict = {}
        for ratio_id, filename in [
            ("aspect_9x16", "exports/9x16.mp4"),
            ("aspect_1x1",  "exports/1x1.mp4"),
            ("aspect_16x9", "exports/16x9.mp4"),
        ]:
            key = oss_job_asset_key(job_id, filename)
            label = filename.split("/")[-1]  # e.g. "9x16.mp4"
            result[ratio_id] = await asyncio.to_thread(
                sign_existing_key, key,
                params={"response-content-disposition": f'attachment;filename="{label}"'},
            )

        master_key = oss_job_asset_key(job_id, "master_cut.mp4")
        result["master_cut"] = await asyncio.to_thread(sign_existing_key, master_key)

        return result
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("refresh_export_urls failed for %s: %s", job_id, exc)
        raise HTTPException(status_code=500, detail=f"Re-sign failed: {exc}")


# ---------------------------------------------------------------------------
# GET /jobs/{job_id}/download/{ratio_id}  — proxy OSS video as attachment
# ---------------------------------------------------------------------------
# Direct browser fetches to signed OSS URLs are blocked by CORS, so the
# download button cannot use fetch()+createObjectURL on them. This endpoint
# fetches from OSS server-side and streams the bytes back with a
# Content-Disposition: attachment header so the browser saves the file.
# ---------------------------------------------------------------------------

_DOWNLOAD_RATIO_MAP = {
    "9x16":   "exports/9x16.mp4",
    "1x1":    "exports/1x1.mp4",
    "16x9":   "exports/16x9.mp4",
    "master": "master_cut.mp4",
}


@app.get("/jobs/{job_id}/download/{ratio_id}")
async def download_export(job_id: str, ratio_id: str):
    """Stream a format-export video as a downloadable attachment.

    Proxies the OSS video server-side to avoid browser CORS restrictions on
    signed URLs. The browser receives the response as a same-origin download.
    """
    import os as _os
    if ratio_id not in _DOWNLOAD_RATIO_MAP:
        raise HTTPException(
            status_code=400,
            detail=f"ratio_id must be one of: {list(_DOWNLOAD_RATIO_MAP)}",
        )
    if not _os.environ.get("OSS_ACCESS_KEY_ID"):
        raise HTTPException(status_code=503, detail="OSS not configured")

    from agents._oss import oss_job_asset_key, sign_existing_key
    import httpx
    from starlette.responses import StreamingResponse

    key = oss_job_asset_key(job_id, _DOWNLOAD_RATIO_MAP[ratio_id])
    try:
        signed_url = await asyncio.to_thread(sign_existing_key, key)
    except Exception as exc:
        logger.error("download_export sign failed for %s/%s: %s", job_id, ratio_id, exc)
        raise HTTPException(status_code=500, detail=f"Re-sign failed: {exc}")

    filename = f"{ratio_id}.mp4"

    async def _stream():
        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream("GET", signed_url) as resp:
                resp.raise_for_status()
                async for chunk in resp.aiter_bytes(65536):
                    yield chunk

    return StreamingResponse(
        _stream(),
        media_type="video/mp4",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# ---------------------------------------------------------------------------
# WS /ws/{job_id}  — run or resume the graph, stream C2 events
# ---------------------------------------------------------------------------


@app.websocket("/ws/{job_id}")
async def ws_run(
    websocket: WebSocket,
    job_id: str,
    brand_name: str = Query(default=""),
    brand_url: str = Query(default=""),
    resolution: str = Query(default=""),
) -> None:
    """Run (or resume after interrupt) the graph and forward every event."""
    await websocket.accept()

    # Bug 15: validate resolution before doing anything else
    if resolution and resolution not in _VALID_RESOLUTIONS:
        await websocket.send_json(_envelope("run.error", job_id, {
            "error": f"Invalid resolution {resolution!r}. Must be one of: {sorted(_VALID_RESOLUTIONS)}"
        }))
        await websocket.close()
        return

    graph = websocket.app.state.graph
    config = {"configurable": {"thread_id": job_id}}

    # Bug 1: per-job concurrency lock — reject a second concurrent run on same thread
    lock = _run_locks.setdefault(job_id, asyncio.Lock())
    if lock.locked():
        await websocket.send_json(_envelope("run.busy", job_id, {
            "message": "Another session is already running this job"
        }))
        await websocket.close()
        return

    async with lock:
        # Decide: interrupt resume, checkpoint resume, or fresh run?
        input_data: Any = None
        if resolution:
            try:
                from langgraph.types import Command
                snapshot = await graph.aget_state(config)
                if snapshot and snapshot.next:
                    input_data = Command(resume={"resolution": resolution})
                    logger.info("Resuming job %s with resolution=%r", job_id, resolution)
                else:
                    logger.warning("Resolution supplied for %s but no interrupt pending", job_id)
            except Exception as exc:
                # Bug 7: aget_state failure is fatal — don't silently proceed
                logger.error("aget_state failed for %s: %s", job_id, exc)
                await websocket.send_json(_envelope("run.error", job_id, {"error": "checkpoint read failed"}))
                try:
                    await websocket.close()
                except Exception:
                    pass
                return

        if input_data is None:
            try:
                snapshot = await graph.aget_state(config)
                has_checkpoint = snapshot is not None and bool(snapshot.values)
            except Exception as exc:
                # Bug 7: treat aget_state failure as fatal, not "no checkpoint"
                logger.error("aget_state failed for job %s: %s", job_id, exc)
                await websocket.send_json(_envelope("run.error", job_id, {"error": "checkpoint read failed"}))
                try:
                    await websocket.close()
                except Exception:
                    pass
                return

            if has_checkpoint:
                logger.info("Reconnect: resuming %s from checkpoint (next=%s)", job_id, list(getattr(snapshot, "next", [])))
                input_data = None  # astream_events resumes from checkpoint when input is None
            else:
                # Fresh run — load brief/photos from the DB row created by POST /jobs.
                # N5: treat DB read failure as fatal; a missing row means a phantom job_id
                # (typo or stale link) that would start a doomed run and waste credits.
                initial_state: dict = {"job_id": job_id}
                try:
                    from db.jobs import connect as db_connect, read_job_state
                    conn = await db_connect()
                    try:
                        db_state = await read_job_state(conn, job_id)
                    finally:
                        await conn.close()
                except Exception as exc:
                    logger.error("DB read failed for job %s: %s", job_id, exc)
                    await websocket.send_json(_envelope("run.error", job_id, {"error": "database unavailable — cannot start run"}))
                    try:
                        await websocket.close()
                    except Exception:
                        pass
                    return
                if not db_state:
                    logger.error("No DB row for job %s — unknown job_id, aborting", job_id)
                    await websocket.send_json(_envelope("run.error", job_id, {"error": f"job {job_id!r} not found"}))
                    try:
                        await websocket.close()
                    except Exception:
                        pass
                    return
                initial_state.update(db_state)
                if brand_name:
                    initial_state["brand_name"] = brand_name
                if brand_url:
                    initial_state["brand_url"] = brand_url
                input_data = initial_state

        # Bug 11: mark job as running so GET /jobs can distinguish in-progress from ingested/failed
        try:
            from db.jobs import connect as db_connect, update_job_status
            _conn = await db_connect()
            try:
                await update_job_status(_conn, job_id, "running")
            finally:
                await _conn.close()
        except Exception as exc:
            logger.warning("Could not mark job %s as running: %s", job_id, exc)

        try:
            await websocket.send_json(_envelope("run.started", job_id, {"message": "graph run starting"}))

            async for event in graph.astream_events(input_data, config=config, version="v2"):
                # Bug 8: synthesize node_started C2 events from on_chain_start lifecycle events
                if event.get("event") == "on_chain_start" and event.get("name") in _KNOWN_NODE_NAMES:
                    await websocket.send_json(
                        build_event("node_started", job_id, {"node": event["name"]})
                    )

                if event.get("event") == "on_custom_event":
                    name = event.get("name")
                    if name in _KNOWN_C2_TYPES:
                        await websocket.send_json(
                            build_event(name, job_id, _jsonable(event.get("data")))
                        )
                        continue  # known C2 event: skip generic passthrough below
                    # Unknown custom event: fall through to generic passthrough so
                    # the frontend sees it rather than silently losing it.

                await websocket.send_json(
                    _envelope(
                        event.get("event", "unknown"),
                        job_id,
                        {"name": event.get("name"), "data": _jsonable(event.get("data"))},
                    )
                )

            # Bug 9: distinguish interrupt-pause from true completion
            try:
                final_snap = await graph.aget_state(config)
                if final_snap and final_snap.next:
                    await websocket.send_json(
                        _envelope("run.interrupted", job_id, {"next": list(final_snap.next)})
                    )
                else:
                    await websocket.send_json(
                        _envelope("run.completed", job_id, {"message": "graph run finished"})
                    )
                    # N2: write "complete" here so the row never stays "running" if
                    # format_export_node's best-effort update already ran or was skipped.
                    if not (final_snap and final_snap.next):
                        try:
                            from db.jobs import connect as db_connect, update_job_status
                            _conn2 = await db_connect()
                            try:
                                await update_job_status(_conn2, job_id, "complete")
                            finally:
                                await _conn2.close()
                        except Exception as _db_exc:
                            logger.warning("Could not mark job %s complete: %s", job_id, _db_exc)
            except Exception:
                await websocket.send_json(
                    _envelope("run.completed", job_id, {"message": "graph run finished"})
                )

        except WebSocketDisconnect:
            logger.info("Client disconnected from job_id=%s", job_id)
        except Exception as exc:
            logger.exception("Error during graph run for job_id=%s", job_id)
            # Bug 5: mark job failed so GET /jobs won't try to auto-reconnect to it
            try:
                from db.jobs import connect as db_connect, update_job_status
                _conn = await db_connect()
                try:
                    await update_job_status(_conn, job_id, "failed")
                finally:
                    await _conn.close()
            except Exception:
                pass
            try:
                await websocket.send_json(_envelope("run.error", job_id, {"error": str(exc)}))
            except Exception:
                pass
        finally:
            try:
                await websocket.close()
            except Exception:
                pass
