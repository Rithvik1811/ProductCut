"""
Phase 1+2: LangGraph graph -- Product Truth Extractor -> Concept Agent -> 5 parallel
Critic Chain checkers -> Meta-Critic -> Merge Coherence Validator -> (Copy Editor
loop-back | Meta-Critic swap retry | fallback) -> winning_script finalized ->
Treatment Agent -> Shot-List Agent -> Budget Gate.

The full Critic Chain (§5.4) is wired end to end, including the Merge Coherence
Validator (§5.4.7) and Copy Editor (§5.4.8). `winning_script` is set by
merge_validator_node on EITHER a full pass ("finalize") or a terminal fallback
("fallback") -- both are legitimate, usable winning scripts (the fallback is
the single highest composite-scoring original variant, not a degraded/partial
result), so BOTH route into Treatment Agent rather than only "finalize". Phase 2
(Treatment Agent §5.5, Shot-List Agent §5.6, Budget Gate §5.7) is now wired in
after it -- nothing downstream of Budget Gate (Video-Gen, Continuity, Voiceover,
Assembly) is wired yet.

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

from agents.body_checker import body_checker_node
from agents.budget_gate import budget_gate_node
from agents.concept_agent import concept_agent_node
from agents.copy_editor import copy_editor_node
from agents.cta_tone_checkers import cta_checker_node, tone_checker_node
from agents.hook_checker import hook_checker_node
from agents.merge_validator import merge_validator_node, route_after_merge_validation
from agents.meta_critic import meta_critic_node
from agents.pacing_checker import pacing_checker_node
from agents.product_truth_extractor import product_truth_extractor_node
from agents.shot_list_agent import shot_list_agent_node
from agents.treatment_agent import treatment_agent_node
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
    builder.add_node("treatment_agent", treatment_agent_node)
    builder.add_node("shot_list_agent", shot_list_agent_node)
    builder.add_node("budget_gate", budget_gate_node)
    builder.add_edge("treatment_agent", "shot_list_agent")
    builder.add_edge("shot_list_agent", "budget_gate")
    builder.add_edge("budget_gate", END)

    builder.add_conditional_edges(
        "merge_validator",
        route_after_merge_validation,
        {
            "finalize": "treatment_agent",
            "copy_editor": "copy_editor",
            "meta_critic": "meta_critic",
            "fallback": "treatment_agent",
        },
    )
    return builder


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
