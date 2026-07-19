"""
Test for app/main.py's on_custom_event -> C2 envelope translation, the one
piece test_graph_build.py doesn't cover (that test asserts on the raw
astream_events output, not on what the WebSocket handler does with it).

Replaces app.state.graph with a fake whose astream_events yields synthetic
events, so this exercises main.py's actual branching logic without needing
a real Ingest endpoint to feed real photos through a real run.
"""
from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app


class _FakeSnapshot:
    """Minimal state snapshot: non-empty values so the WS handler treats this
    as a reconnect (skips the DB read path) and no pending work so it routes
    to run.completed rather than run.interrupted."""
    values = {"job_id": "testjob"}
    next = ()
    interrupts = ()


class _FakeGraph:
    def __init__(self, events: list[dict]):
        self._events = events

    async def aget_state(self, *_args, **_kwargs):
        return _FakeSnapshot()

    async def astream_events(self, *_args, **_kwargs):
        for event in self._events:
            yield event


def test_known_c2_event_is_translated_and_unknown_falls_back_to_passthrough():
    fake_events = [
        {"event": "on_custom_event", "name": "truth_extracted", "data": {"truths": [], "count": 0}},
        {"event": "on_custom_event", "name": "not_a_real_c2_type", "data": {"foo": "bar"}},
        {"event": "on_chain_start", "name": "some_node", "data": {}},
    ]

    with TestClient(app) as client:
        # Lifespan already compiled the real graph on startup; swap it for
        # our fake so this run is fully synthetic and instant.
        app.state.graph = _FakeGraph(fake_events)

        with client.websocket_connect("/ws/testjob") as ws:
            started = ws.receive_json()
            assert started["type"] == "run.started"

            known = ws.receive_json()
            assert known["type"] == "truth_extracted", "recognized C2 type must use its real event name"
            assert known["job_id"] == "testjob"
            assert known["payload"] == {"truths": [], "count": 0}, "must be build_event's payload, not the wrapped {name, data} shape"

            unknown = ws.receive_json()
            assert unknown["type"] == "on_custom_event", "unrecognized custom event must fall back to generic passthrough"
            assert unknown["payload"]["name"] == "not_a_real_c2_type"

            lifecycle = ws.receive_json()
            assert lifecycle["type"] == "on_chain_start", "non-custom LangGraph events always use generic passthrough"

            completed = ws.receive_json()
            assert completed["type"] == "run.completed"
