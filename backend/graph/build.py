"""
Phase 1-5: LangGraph graph -- Product Truth Extractor -> Concept Agent -> 5 parallel
Critic Chain checkers -> Meta-Critic -> Merge Coherence Validator -> (Copy Editor
loop-back | Meta-Critic swap retry | fallback) -> winning_script finalized ->
[Treatment Agent -> Shot-List Agent -> Budget Gate -> Video-Gen Node -> Ken-Burns
Fallback Node -> Continuity Agent -> Continuity Gate -> (retry loop back to
Video-Gen | Assembly Agent)] IN PARALLEL WITH [Voiceover + Caption Agent ->
Assembly Agent] -> END.

The full Critic Chain (§5.4) is wired end to end, including the Merge Coherence
Validator (§5.4.7) and Copy Editor (§5.4.8). `winning_script` is set by
merge_validator_node on EITHER a full pass ("finalize") or a terminal fallback
("fallback") -- both are legitimate, usable winning scripts (the fallback is
the single highest composite-scoring original variant, not a degraded/partial
result), so BOTH route into Treatment Agent rather than only "finalize". Phase 2
(Treatment Agent §5.5, Shot-List Agent §5.6, Budget Gate §5.7) and Phase 3
(Video-Gen Node §5.8, Ken-Burns Fallback Node §5.9) are wired in after it.

Phase 4 (§5.10) adds the Continuity Agent (Qwen-VL drift scoring) and the
Continuity Gate (capped retry + human-in-the-loop) after Ken-Burns, plus a
CONDITIONAL LOOP: the Gate's `route_after_continuity_gate` sends the run back to
`video_gen` whenever it reset any drifted shot to "pending" (an automatic retry,
or a human `retry_with_edit`), else to END. This is the pipeline's first real
cycle: `video_gen -> ken_burns_fallback -> continuity_agent -> continuity_gate ->
(loop back to video_gen for retrying shots | finish)`. The Gate is also where the
graph genuinely `interrupt()`s for a retry-exhausted shot -- pause/resume that
REQUIRES the checkpointer below. Unlike prior phases, Phase 4 wires itself into
this file directly: a retry loop is a graph-topology feature that can only be
exercised in the compiled graph.

Phase 5 (§5.11) adds the Voiceover + Caption Agent as a SECOND parallel branch
off `merge_validator` -- it reads only `winning_script`, not `treatment`/
`shot_list`/`generated_shots`, so it does not need to wait behind Video-Gen or
Continuity and starts the same superstep as `treatment_agent`. Wired via
`_route_after_merge_validation_with_vo` (below), which wraps the pure
`route_after_merge_validation` (kept untouched -- it is independently
unit-tested and also called internally by `merge_validator_node`) to fan out
to BOTH `treatment_agent` and `voiceover_caption_agent` on "finalize"/
"fallback", per this node's own documented wiring plan
(agents/voiceover_caption_agent.py, "HOW THIS COMPOSES WITH THE REST OF THE
GRAPH").

Phase 5 (§5.12) adds the Assembly Agent as a genuine FAN-IN JOIN of the two
branches above: `voiceover_caption_agent -> assembly_agent` and
`continuity_gate`'s conditional "end" route -> `assembly_agent` (replacing
both branches' previous independent `-> END` edges). This is a NEW topology
shape for this graph -- the prior fan-in precedent (the 5 Critic-Chain
checkers -> meta_critic) always completes every branch in the SAME superstep
(a static parallel fan-out with no loop), but here one branch
(voiceover_caption_agent) typically finishes in one early superstep while the
other (the `video_gen -> ken_burns_fallback -> continuity_agent ->
continuity_gate` retry cycle) can take an unbounded number of superstep
passes before finally routing to "end". A node with two plain (non-list)
incoming edges into the SAME per-node trigger channel fires on the FIRST
writer by default (LangGraph's `EphemeralValue` trigger semantics) -- which
would run `assembly_agent` immediately after `voiceover_caption_agent`
completes, long before the continuity loop (or a human-review interrupt
inside it) has settled, with `generated_shots`/`shot_list` still mid-flight.
`assembly_agent` is therefore registered with `defer=True` (confirmed against
the installed LangGraph 1.2.7's own `graph/state.py`: `defer=True` swaps that
per-node trigger channel from `EphemeralValue` to `LastValueAfterFinish`,
which only becomes available once the Pregel loop calls `finish()` -- i.e.
once the ENTIRE rest of the graph, including an arbitrarily-long continuity
retry loop and any interrupt/resume pause inside it, has genuinely settled
with no more work scheduled). This is verified against the REAL compiled
graph, not assumed: `tests/test_continuity_loop_e2e.py::test_assembly_agent_runs_exactly_once_after_continuity_loop_resolves`
seeds a 2-pass retry loop (voiceover finishes trivially fast, the drifty
shot's auto-retry takes a second full loop pass) and asserts `assembly_agent`
(and its `master_cut_ready` C2 event) fires EXACTLY ONCE, only after the
loop's FINAL pass resolves to "end", with the fully-final post-retry
`shot_list`/`generated_shots` and the early-finished `voiceover` both visible
to it. `assembly_agent -> END` closes the graph.

Checkpointer selection is graceful:
  - if DATABASE_URL is set  -> AsyncPostgresSaver (real durable checkpoints)
  - otherwise               -> MemorySaver  (standalone, no DB required)

AsyncPostgresSaver, not the sync PostgresSaver: the FastAPI app drives the
graph via `astream_events` (async), and the sync PostgresSaver's async
methods (aget_tuple, etc.) raise NotImplementedError -- confirmed by hitting
that exact error against the real RDS instance. AsyncPostgresSaver implements
the async checkpointer interface properly.

Do NOT import/modify graph.state's schema shape — we only consume it.

NOTE: there is no Ingest node yet, so nothing populates `product_photos` or
`brief` in state before this graph runs. Driving a real run through the
/ws/{job_id} endpoint will fail with a KeyError on `state["product_photos"]`
(then `state["brief"]`) until the job-submission-form -> ingest-endpoint ->
OSS-upload path exists (Phase 1, still open on both KR's and RR's task
lists). Use `derisk/test_truth_extractor.py` and `derisk/test_concept_agent.py`
to exercise each agent standalone until then.
"""
from __future__ import annotations

import logging
import os
from contextlib import AsyncExitStack
from typing import Optional

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from agents.assembly_agent import assembly_agent_node
from agents.body_checker import body_checker_node
from agents.budget_gate import budget_gate_node
from agents.concept_agent import concept_agent_node
from agents.continuity_agent import continuity_agent_node
from agents.continuity_gate import continuity_gate_node, route_after_continuity_gate
from agents.copy_editor import copy_editor_node
from agents.cta_tone_checkers import cta_checker_node, tone_checker_node
from agents.hook_checker import hook_checker_node
from agents.ken_burns_fallback_node import ken_burns_fallback_node
from agents.merge_validator import merge_validator_node, route_after_merge_validation
from agents.meta_critic import meta_critic_node
from agents.pacing_checker import pacing_checker_node
from agents.product_truth_extractor import product_truth_extractor_node
from agents.shot_list_agent import shot_list_agent_node
from agents.treatment_agent import treatment_agent_node
from agents.visual_direction_agent import visual_direction_agent_node
from agents.video_gen_node import video_gen_node
from agents.format_export_node import format_export_node
from agents.voiceover_caption_agent import voiceover_caption_agent_node
from graph.state import ProductCutState

logger = logging.getLogger("productcut.graph")


def _build_uncompiled() -> StateGraph:
    """Construct the (uncompiled) graph."""
    builder = StateGraph(ProductCutState)
    builder.add_node("product_truth_extractor", product_truth_extractor_node)
    builder.add_node("concept_agent", concept_agent_node)
    builder.add_node("hook_checker", hook_checker_node)
    builder.add_node("pacing_checker", pacing_checker_node)
    builder.add_node("body_checker", body_checker_node)
    builder.add_node("cta_checker", cta_checker_node)
    builder.add_node("tone_checker", tone_checker_node)
    builder.add_node("meta_critic", meta_critic_node)

    builder.add_edge(START, "product_truth_extractor")
    builder.add_edge("product_truth_extractor", "concept_agent")

    # Fan-out: concept_agent's 4 script variants get scored by 5 parallel specialists.
    builder.add_edge("concept_agent", "hook_checker")
    builder.add_edge("concept_agent", "pacing_checker")
    builder.add_edge("concept_agent", "body_checker")
    builder.add_edge("concept_agent", "cta_checker")
    builder.add_edge("concept_agent", "tone_checker")

    # Fan-in: Meta-Critic waits for all 5 before reconciling (LangGraph superstep semantics).
    builder.add_edge("hook_checker", "meta_critic")
    builder.add_edge("pacing_checker", "meta_critic")
    builder.add_edge("body_checker", "meta_critic")
    builder.add_edge("cta_checker", "meta_critic")
    builder.add_edge("tone_checker", "meta_critic")

    builder.add_node("merge_validator", merge_validator_node)
    builder.add_node("copy_editor", copy_editor_node)
    builder.add_edge("meta_critic", "merge_validator")
    builder.add_edge("copy_editor", "merge_validator")

    # Phase 2 (§5.5-5.7): both "finalize" and "fallback" set a real, usable
    # winning_script (see module docstring) -- neither is a dead end anymore.
    builder.add_node("visual_direction_agent", visual_direction_agent_node)
    builder.add_edge("visual_direction_agent", "treatment_agent")
    builder.add_node("treatment_agent", treatment_agent_node)
    builder.add_node("shot_list_agent", shot_list_agent_node)
    builder.add_node("budget_gate", budget_gate_node)
    builder.add_node("video_gen", video_gen_node)
    builder.add_node("ken_burns_fallback", ken_burns_fallback_node)
    builder.add_node("continuity_agent", continuity_agent_node)
    builder.add_node("continuity_gate", continuity_gate_node)
    # Phase 5 (§5.11): Voiceover + Caption Agent runs as a PARALLEL branch off
    # winning_script alone -- it depends only on the script, not on the rendered
    # video, so it starts the same superstep as treatment_agent rather than
    # waiting behind Video-Gen/Continuity.
    builder.add_node("voiceover_caption_agent", voiceover_caption_agent_node)
    # Phase 5 (§5.12): Assembly Agent is a genuine fan-in JOIN of the voiceover
    # branch and the (possibly multi-pass) continuity retry loop -- see module
    # docstring's Phase 5 (§5.12) section for why `defer=True` is required here
    # (not merely a style choice) and how it was verified against a real
    # compiled-graph test.
    builder.add_node("assembly_agent", assembly_agent_node, defer=True)
    builder.add_node("format_export_node", format_export_node)
    builder.add_edge("voiceover_caption_agent", "assembly_agent")
    builder.add_edge("assembly_agent", "format_export_node")
    builder.add_edge("format_export_node", END)
    builder.add_edge("treatment_agent", "shot_list_agent")
    builder.add_edge("shot_list_agent", "budget_gate")
    builder.add_edge("budget_gate", "video_gen")
    builder.add_edge("video_gen", "ken_burns_fallback")
    # Phase 4 (§5.10): Continuity scores drift, then the Gate decides retry /
    # human-review / pass. The Gate's conditional edge closes the retry cycle --
    # back to video_gen for any shot still needing a pass ("pending" retry, or
    # "fallback_requested" from a human accept_fallback so it reaches Ken-Burns),
    # else on to assembly_agent (Phase 5, §5.12 -- the join with the voiceover
    # branch described in the module docstring's Phase 5 (§5.12) section). See
    # route_after_continuity_gate for why fallback_requested loops.
    builder.add_edge("ken_burns_fallback", "continuity_agent")
    builder.add_edge("continuity_agent", "continuity_gate")
    builder.add_conditional_edges(
        "continuity_gate",
        route_after_continuity_gate,
        {"video_gen": "video_gen", "end": "assembly_agent"},
    )

    builder.add_conditional_edges(
        "merge_validator",
        _route_after_merge_validation_with_vo,
        {
            "finalize": "visual_direction_agent",      # was "treatment_agent"
            "copy_editor": "copy_editor",
            "meta_critic": "meta_critic",
            "fallback": "visual_direction_agent",      # was "treatment_agent"
            "voiceover_caption_agent": "voiceover_caption_agent",
            "visual_direction_agent": "visual_direction_agent",
        },
    )
    return builder


def _route_after_merge_validation_with_vo(state: ProductCutState) -> list[str]:
    """Conditional-edge path function: wraps `route_after_merge_validation` (kept
    untouched -- it's independently unit-tested in test_merge_validator.py and
    also called internally by merge_validator_node itself, both against its
    original bare-str contract) to additionally fan out to
    `voiceover_caption_agent` whenever winning_script is actually final
    ("finalize" or "fallback" -- both set a real, usable winning_script per the
    module docstring above). LangGraph's `add_conditional_edges` accepts a path
    function returning a *list* of routing keys for exactly this "one edge,
    multiple parallel next-nodes" case (confirmed against the installed
    `StateGraph.add_conditional_edges` signature: `path: Callable[...,
    Hashable | Sequence[Hashable]]`) -- see agents/voiceover_caption_agent.py's
    own "HOW THIS COMPOSES WITH THE REST OF THE GRAPH" docstring section for the
    original wiring plan this implements. The "copy_editor"/"meta_critic" retry
    branches are unaffected -- winning_script isn't final yet on those paths, so
    Voiceover must not fire.
    """
    route = route_after_merge_validation(state)
    if route in ("finalize", "fallback"):
        return ["visual_direction_agent", "voiceover_caption_agent"]
    return [route]


async def build_graph(exit_stack: Optional[AsyncExitStack] = None):
    """Compile the Phase 0 graph with a checkpointer.

    If DATABASE_URL is present, use AsyncPostgresSaver (its async context
    manager is entered via the provided AsyncExitStack so the connection
    lives for the app's lifetime). Otherwise fall back to an in-memory
    MemorySaver so the scaffold runs standalone without a provisioned
    database.

    Args:
        exit_stack: optional AsyncExitStack owned by the caller (e.g. the
            FastAPI lifespan) used to keep the Postgres connection open. If
            None and a DATABASE_URL is set, a module-level stack is used
            instead.

    Returns:
        A compiled LangGraph runnable.
    """
    builder = _build_uncompiled()
    database_url = os.getenv("DATABASE_URL")

    if database_url:
        try:
            from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

            stack = exit_stack if exit_stack is not None else AsyncExitStack()
            checkpointer = await stack.enter_async_context(
                AsyncPostgresSaver.from_conn_string(database_url)
            )
            await checkpointer.setup()
            logger.info("Checkpointer: AsyncPostgresSaver (DATABASE_URL detected)")
            return builder.compile(checkpointer=checkpointer)
        except Exception as exc:  # noqa: BLE001 - scaffold must not hard-fail
            logger.warning(
                "DATABASE_URL set but AsyncPostgresSaver init failed (%s); "
                "falling back to MemorySaver",
                exc,
            )

    logger.info("Checkpointer: MemorySaver (no DATABASE_URL — standalone mode)")
    return builder.compile(checkpointer=MemorySaver())
