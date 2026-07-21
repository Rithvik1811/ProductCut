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

version: 5
  - v2: added "merge_validated" event type + MergeValidatedPayload (§5.4.7's
        dashed `CV -.-> FE` streaming edge in the architecture diagram) so a
        merge-candidate retry/fallback is visible on the live stream, not hidden.
  - v3: ShotGeneratedPayload.status += "fallback_requested" (Phase 3, mirrors
        graph.state.Shot.status v6 / graph.shot_schema.ShotStatus v3 -- see
        state.py's v6 note for why this is distinct from the existing
        "fallback" value). Formalizes what agents/video_gen_node.py (KR) flagged
        as a self-invented, unconfirmed departure pending a KR/RR sync
        (docs/BUILD_TASKS.md Phase 3).
  - v4: added "vo_ready" event type + VoReadyPayload (Phase 5, §5.11's
        `MC -> VOX` parallel branch -- the Voiceover + Caption Agent's own
        live-stream signal, the VO analog of "shot_generated"). PROPOSED
        ADDITIVE CHANGE, not confirmed against a body-side dashboard consumer
        yet: no "vo_ready" (or equivalent) event existed before this agent
        needed one (checked: the original 10 named events + v2's
        "merge_validated" cover Ingest through Critic Chain, Treatment,
        Shot-List, Budget, Video-Gen, drift, interrupt, edit-routing, and job
        completion -- none of them fire when VO synthesis itself finishes).
        Flagged here exactly like "fallback_requested" was in v3, pending a
        sync with whoever builds the dashboard's VO panel; see
        agents/voiceover_caption_agent.py's module docstring for the fuller
        rationale.
  - v5: added "master_cut_ready" event type + MasterCutReadyPayload (Phase 5,
        §5.12's Assembly Agent -- the fan-in join of the voiceover branch and
        the continuity retry loop, see graph/build.py's module docstring).
        Fires once per job, only after both upstream branches have settled
        (the Assembly node is registered `defer=True` in graph/build.py, so
        -- unlike `interrupt_requested`'s documented double-fire across a
        pause/resume cycle, continuity_gate.py's own KNOWN LIMITATION -- this
        event fires exactly once, verified against a real compiled-graph
        test). PROPOSED ADDITIVE CHANGE, same posture as v3's
        "fallback_requested" and v4's "vo_ready": no event fired when the
        finished master cut itself became available (the existing
        `job_complete`/JobCompletePayload is Assembly+Export's COMBINED
        signal per its own docstring, and Export, §5.13, is not built yet) --
        flagged here, pending a sync with whoever builds the dashboard's
        Assembly panel. `total_duration_sec` is the ffprobe'd (real, not
        planned) duration of the finished mezzanine.
  - v6: added "job_failed" event type + JobFailedPayload. Fired when every
        script variant is rejected by the critic chain (meta_critic_result.outcome
        == "all_excluded_failure"). meta_critic_node dispatches this event with a
        user-readable reason and graph/build.py routes to END rather than
        continuing with no winning_script. This is the pipeline's only
        terminal-failure signal; every prior "the job stops" path was either
        `job_complete` (success) or an actual unhandled exception.
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
    "research_complete",  # v13: Product Web Research node finished
    "critic_score",
    "treatment_ready",
    "budget_updated",
    "drift_scored",
    "shot_generated",
    "interrupt_requested",
    "edit_routed",
    "job_complete",
    "merge_validated",
    "vo_ready",
    "master_cut_ready",
    "job_failed",
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


class ResearchCompletePayload(TypedDict):
    """Product Web Research node finished (v13). Fires only when research was
    actually performed (TAVILY_API_KEY present and product classified as
    research_needed). Skipped products don't emit this event."""
    fact_count: int  # number of ResearchFacts extracted after grounding check
    product_name: str  # the product name the queries were built around
    queries: list[str]  # the 1-3 Tavily queries that were run


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


class VoReadyPayload(TypedDict):
    """Voiceover + Caption Agent (5.11) finished synthesizing the VO audio track
    and caption-timing JSON for the finalized winning_script -- fires once per
    job on the `MC -> VOX` parallel branch, the VO analog of `shot_generated`."""
    voiceover: Voiceover  # mirrors state.voiceover, C1
    caption_count: int  # number of {text, start_ts, end_ts} caption entries produced
    degraded: bool  # True when >=1 beat's TTS synthesis permanently failed (silent gap, captions-only for that beat)


class MasterCutReadyPayload(TypedDict):
    """Assembly Agent (5.12) finished stitching every real/fallback shot clip,
    the voiceover audio, and the burned captions into one finished master-cut
    MP4 -- fires once per job, after the Assembly fan-in join settles (see
    graph/build.py's module docstring)."""
    uri: str  # mirrors state.master_cut_uri, C1
    shot_count: int  # number of REAL segments actually rendered (held-frame gaps don't add one)
    total_duration_sec: float  # ffprobe'd (real, not planned) duration of the finished mezzanine


class JobFailedPayload(TypedDict):
    """The pipeline hit a genuine terminal failure it cannot recover from --
    fired by meta_critic_node when all script variants fail never_do checks
    (outcome == "all_excluded_failure"). The frontend's only other "the job
    stopped" signal is job_complete (success); this is the failure counterpart,
    meant to be user-visible rather than a dropped WebSocket connection."""
    reason: str    # human-readable, safe to show a seller directly
    stage: str     # which node/phase detected the failure, e.g. "meta_critic"


EventPayload = Union[
    NodeStartedPayload,
    TruthExtractedPayload,
    ResearchCompletePayload,
    CriticScorePayload,
    TreatmentReadyPayload,
    BudgetUpdatedPayload,
    ShotGeneratedPayload,
    DriftScoredPayload,
    InterruptRequestedPayload,
    EditRoutedPayload,
    JobCompletePayload,
    MergeValidatedPayload,
    VoReadyPayload,
    MasterCutReadyPayload,
    JobFailedPayload,
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
    "VoReadyPayload",
    "MasterCutReadyPayload",
    "JobFailedPayload",
    "build_event",
]
