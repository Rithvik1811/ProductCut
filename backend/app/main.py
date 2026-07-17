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

from fastapi import FastAPI, File, Form, Query, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from graph.build import build_graph
from graph.events import EventType, build_event

_KNOWN_C2_TYPES = frozenset(get_args(EventType))

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
    return {
        "status": "ok",
        "service": "productcut-backend",
        "graph_ready": getattr(app.state, "graph", None) is not None,
        "ts": _now_iso(),
    }


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

    # Persist to DB (best-effort; graph can run without it if DB is unavailable)
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
        logger.warning("DB unavailable — job %s will run without persistence: %s", job_id, exc)

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

    for photo in valid:
        safe = Path(photo.filename).name  # type: ignore[arg-type]
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
        return []


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
        if snapshot is None:
            return {"job_id": job_id, "state": None, "next": []}
        return {
            "job_id": job_id,
            "state": _jsonable(dict(snapshot.values) if snapshot.values else {}),
            "next": list(snapshot.next) if snapshot.next else [],
        }
    except Exception as exc:
        logger.error("aget_state failed for %s: %s", job_id, exc)
        return {"job_id": job_id, "state": None, "next": [], "error": str(exc)}


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
    """Run (or resume after interrupt) the graph and forward every event.

    Query params:
      brand_name   — passed into initial state (CTA + tone context)
      brand_url    — fetched by brand_research_node
      resolution   — when non-empty, tries to resume an interrupted job:
                     approve | retry_with_edit | accept_fallback
    """
    await websocket.accept()
    graph = websocket.app.state.graph
    config = {"configurable": {"thread_id": job_id}}

    # Decide: fresh run or resume?
    input_data: Any = None
    if resolution:
        try:
            from langgraph.types import Command
            snapshot = await graph.aget_state(config)
            if snapshot and snapshot.next:
                input_data = Command(resume={"resolution": resolution})
                logger.info("Resuming job %s with resolution=%r", job_id, resolution)
        except Exception as exc:
            logger.warning("Could not check state for resume of %s: %s", job_id, exc)

    if input_data is None:
        # Fresh run — load persisted job state from DB
        initial_state: dict = {"job_id": job_id}
        try:
            from db.jobs import connect as db_connect, read_job_state
            conn = await db_connect()
            try:
                db_state = await read_job_state(conn, job_id)
                if db_state:
                    initial_state.update(db_state)
            finally:
                await conn.close()
        except Exception as exc:
            logger.warning("Could not load DB state for job %s: %s", job_id, exc)

        if brand_name:
            initial_state["brand_name"] = brand_name
        if brand_url:
            initial_state["brand_url"] = brand_url
        input_data = initial_state

    try:
        await websocket.send_json(
            _envelope("run.started", job_id, {"message": "graph run starting"})
        )

        async for event in graph.astream_events(
            input_data, config=config, version="v2"
        ):
            if event.get("event") == "on_custom_event" and event.get("name") in _KNOWN_C2_TYPES:
                await websocket.send_json(
                    build_event(event["name"], job_id, _jsonable(event.get("data")))
                )
                continue
            elif event.get("event") == "on_custom_event":
                logger.warning(
                    "Unknown custom event %r — forwarding as passthrough.", event.get("name")
                )

            await websocket.send_json(
                _envelope(
                    event.get("event", "unknown"),
                    job_id,
                    {"name": event.get("name"), "data": _jsonable(event.get("data"))},
                )
            )

        await websocket.send_json(
            _envelope("run.completed", job_id, {"message": "graph run finished"})
        )
    except WebSocketDisconnect:
        logger.info("Client disconnected from job_id=%s", job_id)
    except Exception as exc:
        logger.exception("Error during graph run for job_id=%s", job_id)
        try:
            await websocket.send_json(_envelope("run.error", job_id, {"error": str(exc)}))
        except Exception:
            pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass
