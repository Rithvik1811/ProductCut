"""
FastAPI app that streams LangGraph events over a WebSocket.

Endpoints:
  GET  /health          -> basic liveness + which checkpointer is active
  WS   /ws/{job_id}     -> runs the graph and forwards each astream_events
                           event to the client as a JSON envelope.

Two event families reach the client, both using the same outer envelope
`{"type": ..., "job_id": ..., "ts": ..., "payload": ...}`:
  - Real C2 business events (graph.events.EventType), dispatched by agent
    nodes via adispatch_custom_event and unwrapped here via build_event.
    This IS the frozen contract (docs/C2_EVENT_SCHEMA.md).
  - Everything else (raw LangGraph lifecycle events like on_chain_start) is
    forwarded as an untyped passthrough for debugging -- NOT part of C2,
    do not build dashboard panels against these.
"""
from __future__ import annotations

import logging
from contextlib import AsyncExitStack, asynccontextmanager
from datetime import datetime, timezone
from typing import Any, get_args

from fastapi import FastAPI, Query, WebSocket, WebSocketDisconnect

from graph.build import build_graph
from graph.events import EventType, build_event

_KNOWN_C2_TYPES = frozenset(get_args(EventType))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("productcut.app")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _envelope(event_type: str, job_id: str, payload: Any) -> dict:
    """Build the placeholder C2-style WebSocket event envelope."""
    return {
        "type": event_type,
        "job_id": job_id,
        "ts": _now_iso(),
        "payload": payload,
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Compile the graph once at startup; keep any DB connection open for life."""
    app.state.exit_stack = AsyncExitStack()
    logger.info("Building Phase 0 LangGraph graph...")
    app.state.graph = await build_graph(exit_stack=app.state.exit_stack)
    logger.info("Graph compiled and ready.")
    try:
        yield
    finally:
        await app.state.exit_stack.aclose()
        logger.info("Shutdown: released graph resources.")


app = FastAPI(title="ProductCut Backend (Phase 0 scaffold)", lifespan=lifespan)


@app.get("/health")
async def health() -> dict:
    """Basic health check."""
    return {
        "status": "ok",
        "service": "productcut-backend",
        "phase": "0-scaffold",
        "graph_ready": getattr(app.state, "graph", None) is not None,
        "ts": _now_iso(),
    }


@app.websocket("/ws/{job_id}")
async def ws_run(
    websocket: WebSocket,
    job_id: str,
    brand_name: str = Query(default=""),
    brand_url: str = Query(default=""),
) -> None:
    """Run the graph for `job_id` and stream each event to the client.

    Optional query params:
      ?brand_name=HydroFlask   — included in CTA and tone
      ?brand_url=https://...   — fetched by brand_research_node for full context
    """
    await websocket.accept()
    graph = websocket.app.state.graph
    config = {"configurable": {"thread_id": job_id}}
    initial_state: dict = {"job_id": job_id}
    if brand_name:
        initial_state["brand_name"] = brand_name
    if brand_url:
        initial_state["brand_url"] = brand_url

    try:
        await websocket.send_json(
            _envelope("run.started", job_id, {"message": "graph run starting"})
        )

        # astream_events surfaces the internal event stream of the graph run.
        # on_custom_event entries are C2 business events dispatched by agent
        # nodes (adispatch_custom_event) -- unwrap those into a real C2
        # envelope via graph.events.build_event instead of the generic
        # passthrough used for raw LangGraph lifecycle events.
        async for event in graph.astream_events(
            initial_state, config=config, version="v2"
        ):
            if event.get("event") == "on_custom_event" and event.get("name") in _KNOWN_C2_TYPES:
                c2_type = event["name"]
                c2_payload = _jsonable(event.get("data"))
                await websocket.send_json(build_event(c2_type, job_id, c2_payload))
                continue
            elif event.get("event") == "on_custom_event":
                logger.warning(
                    "Custom event %r is not a known C2 type; "
                    "forwarding as generic passthrough instead.", event.get("name"),
                )

            await websocket.send_json(
                _envelope(
                    event.get("event", "unknown"),
                    job_id,
                    {
                        "name": event.get("name"),
                        "data": _jsonable(event.get("data")),
                    },
                )
            )

        await websocket.send_json(
            _envelope("run.completed", job_id, {"message": "graph run finished"})
        )
    except WebSocketDisconnect:
        logger.info("Client disconnected from job_id=%s", job_id)
    except Exception as exc:  # noqa: BLE001 - scaffold: report, don't crash socket
        logger.exception("Error during graph run for job_id=%s", job_id)
        try:
            await websocket.send_json(
                _envelope("run.error", job_id, {"error": str(exc)})
            )
        except Exception:  # noqa: BLE001
            pass
    finally:
        try:
            await websocket.close()
        except Exception:  # noqa: BLE001
            pass


def _jsonable(obj: Any) -> Any:
    """Best-effort convert event data to something json-serializable.

    astream_events payloads can contain rich objects; for a scaffold we only
    need a safe, lossy string/dict view so the socket never chokes.
    """
    try:
        import json

        json.dumps(obj)
        return obj
    except (TypeError, ValueError):
        if isinstance(obj, dict):
            return {k: _jsonable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_jsonable(v) for v in obj]
        return repr(obj)
