"""
C2 — WebSocket event schema (frozen contract, Phase 0).

The outbound WebSocket stream from FastAPI (fed by LangGraph `astream_events`)
carries a single envelope shape `{type, job_id, ts, payload}` for every message.
This module freezes that envelope plus the payload shape of each of the ten
named business events the pipeline emits. The frontend dashboard renders off
these types; agent nodes construct events against them.

Payloads reuse the C1 state types (see graph/state.py) wherever an event is
"here is a freshly produced piece of ProductCutState" — e.g. `critic_score`
wraps `CriticScore`, `shot_generated` wraps `GeneratedShot`, `interrupt_requested`
wraps `HumanReviewEntry`, `edit_routed` wraps `EditRouterOutput`, `budget_updated`
wraps `BudgetLedger`. We import those types, never redefine them.

Extend additively only: add new event types / payload keys, never rename or
remove an existing one without a sync between KR and RR and a version bump in
this docstring. Spec of record: docs/TECHNICAL_DOCUMENTATION.md §9.5 (realtime)
and §5 (agent-by-agent). Companion review doc: docs/C2_EVENT_SCHEMA.md.

Scope note: the Phase 0 scaffold in app/main.py also emits `run.started` /
`run.completed` / `run.error` and forwards raw `astream_events` internals
(`on_chain_start`, etc.). Those are transport/lifecycle scaffolding, NOT part of
this frozen business-event contract, and are deliberately excluded here.

version: 3
  - v2: added "merge_validated" event type + MergeValidatedPayload (§5.4.7's
        dashed `CV -.-> FE` streaming edge in the architecture diagram) so a
        merge-candidate retry/fallback is visible on the live stream, not hidden.
  - v3: ShotGeneratedPayload.status += "fallback_requested" (Phase 3, mirrors
        graph.state.Shot.status v6 / graph.shot_schema.ShotStatus v3 -- see
        state.py's v6 note for why this is distinct from the existing
        "fallback" value). Formalizes what agents/video_gen_node.py (KR) flagged
        as a self-invented, unconfirmed departure pending a KR/RR sync
        (docs/BUILD_TASKS.md Phase 3).
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal, TypedDict, Union

from typing_extensions import NotRequired

from graph.state import (
    BudgetLedger,
    CriticScore,
    EditRouterOutput,
    Exports,
    GeneratedShot,
    HumanReviewEntry,
    ProductTruth,
    Treatment,
    Voiceover,
)

# ---------------------------------------------------------------------------
# Envelope + event-type strings. The original 10 are named in docs/BUILD_TASKS.md
# (C2); "merge_validated" (v2) is an additive 11th, not part of that original list.
# ---------------------------------------------------------------------------

EventType = Literal[
    "node_started",
    "truth_extracted",
    "critic_score",
    "treatment_ready",
    "budget_updated",
    "drift_scored",
    "shot_generated",
    "interrupt_requested",
    "edit_routed",
    "job_complete",
    "merge_validated",
]


# ---------------------------------------------------------------------------
# Per-event payload shapes. Each is the `payload` of an EventEnvelope whose
# `type` is the matching string above.
# ---------------------------------------------------------------------------


class NodeStartedPayload(TypedDict):
    """A graph node began executing (generic progress heartbeat)."""
    node: str  # LangGraph node name, e.g. "product_truth_extractor"
    label: NotRequired[str]  # human-friendly label for the dashboard timeline
    phase: NotRequired[int]  # pipeline phase number (1-9), if known


class TruthExtractedPayload(TypedDict):
    """Product Truth Extractor produced its grounded facts."""
    truths: list[ProductTruth]
    count: int  # len(truths); handy for the dashboard's "N facts found" badge


class CriticScorePayload(TypedDict):
    """Critic chain / Meta-Critic scored the script variants."""
    scores: dict[str, CriticScore]  # keyed by variant_id, mirrors state.critic_scores
    winning_variant_ids: NotRequired[list[str]]  # source variants of the merged winner


class TreatmentReadyPayload(TypedDict):
    """Treatment Agent produced (and the validator passed) the director's treatment."""
    treatment: Treatment


class BudgetUpdatedPayload(TypedDict):
    """Budget Gate wrote/updated per-shot allocations in the ledger."""
    ledger: BudgetLedger
    over_cap: bool  # True while the gate is looping back to the Shot-List Agent


class ShotGeneratedPayload(TypedDict):
    """A single shot finished generation (real clip or Ken-Burns fallback)."""
    shot_id: str
    generated: GeneratedShot
    # mirrors Shot.status in state.py (pending/generating/passed/fallback/review/fallback_requested)
    status: Literal[
        "pending", "generating", "passed", "fallback", "review", "fallback_requested",
    ]
    is_fallback: bool  # True when routed to the Ken-Burns fallback node


class DriftScoredPayload(TypedDict):
    """Continuity Agent scored a generated shot against the reference/style."""
    shot_id: str
    drift_score: float
    threshold: float  # the pass/fail line, so the UI can color the score
    passed: bool  # drift_score <= threshold
    attempt: int  # which regen attempt this score belongs to (0-based)


class InterruptRequestedPayload(TypedDict):
    """Retry cap exhausted: a real human-review interrupt was raised (C4 payload)."""
    review: HumanReviewEntry
    queue_position: NotRequired[int]  # index in state.human_review_queue


class EditRoutedPayload(TypedDict):
    """Phase 9 Edit Router classified a chat edit request (C5 territory)."""
    edit_id: str
    router_output: EditRouterOutput


class JobCompletePayload(TypedDict):
    """Assembly + Export finished: the final deliverables are ready."""
    master_cut_uri: str
    exports: Exports
    voiceover: NotRequired[Voiceover]


class MergeValidatedPayload(TypedDict):
    """Merge Coherence Validator (5.4.7) scored a merge candidate -- the dashed
    CV -.-> FE streaming edge. Fires once per attempt so a retry/fallback is part
    of the visible live trace, not hidden."""
    result: dict
    attempt_number: int


EventPayload = Union[
    NodeStartedPayload,
    TruthExtractedPayload,
    CriticScorePayload,
    TreatmentReadyPayload,
    BudgetUpdatedPayload,
    ShotGeneratedPayload,
    DriftScoredPayload,
    InterruptRequestedPayload,
    EditRoutedPayload,
    JobCompletePayload,
    MergeValidatedPayload,
]


class EventEnvelope(TypedDict):
    """The single outer shape of every message on the WebSocket stream."""
    type: EventType
    job_id: str
    ts: str  # ISO-8601 UTC timestamp
    payload: EventPayload


# ---------------------------------------------------------------------------
# Helper: construct a well-formed envelope with a real ISO timestamp.
# Mirrors app/main.py's local `_envelope()`, but typed + importable so nodes
# and tests share one builder.
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    """Current time as an ISO-8601 UTC string (matches app/main.py)."""
    return datetime.now(timezone.utc).isoformat()


def build_event(
    event_type: EventType,
    job_id: str,
    payload: EventPayload,
) -> EventEnvelope:
    """Build a C2 event envelope, stamping `ts` with the current UTC time.

    Example:
        build_event("truth_extracted", job_id, {"truths": truths, "count": len(truths)})
    """
    return {
        "type": event_type,
        "job_id": job_id,
        "ts": _now_iso(),
        "payload": payload,
    }


__all__ = [
    "EventType",
    "EventEnvelope",
    "EventPayload",
    "NodeStartedPayload",
    "TruthExtractedPayload",
    "CriticScorePayload",
    "TreatmentReadyPayload",
    "BudgetUpdatedPayload",
    "ShotGeneratedPayload",
    "DriftScoredPayload",
    "InterruptRequestedPayload",
    "EditRoutedPayload",
    "JobCompletePayload",
    "MergeValidatedPayload",
    "build_event",
]
