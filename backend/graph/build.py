"""
Phase 1: LangGraph graph -- Product Truth Extractor -> Concept Agent.

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

from agents.concept_agent import concept_agent_node
from agents.product_truth_extractor import product_truth_extractor_node
from graph.state import ProductCutState

logger = logging.getLogger("productcut.graph")


def _build_uncompiled() -> StateGraph:
    """Construct the (uncompiled) graph."""
    builder = StateGraph(ProductCutState)
    builder.add_node("product_truth_extractor", product_truth_extractor_node)
    builder.add_node("concept_agent", concept_agent_node)
    builder.add_edge(START, "product_truth_extractor")
    builder.add_edge("product_truth_extractor", "concept_agent")
    builder.add_edge("concept_agent", END)
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
