"""
Integration test for the piece the derisk scripts never exercised: do the
Product Truth Extractor AND Concept Agent actually work when run *inside*
the compiled LangGraph graph, chained -- RunnableConfig injection,
adispatch_custom_event, state flowing from one node into the next, the whole
astream_events path -- not just as bare function calls.

Fakes only the network boundary (AsyncOpenAI construction inside each agent
module); everything else (StateGraph, checkpointer, astream_events, custom
event dispatch) is the real thing.
"""
from __future__ import annotations

import json

import pytest

from graph.build import build_graph
from tests._fakes import make_fake_async_openai

GOOD_FACTS = [
    ("a hairline scratch runs diagonally across the lower left corner of the lid", "imperfection"),
    ("the base plate has two asymmetric ventilation slots near the rear edge", "construction_detail"),
    ("a faint discoloration ring marks where a sticker was once removed", "imperfection"),
    ("the power button has a slightly recessed matte texture unlike the glossy housing", "texture"),
    ("the charging port surround shows minor oxidation on the metal contacts", "imperfection"),
    ("a small manufacturer stamp is debossed near the bottom-right hinge", "construction_detail"),
]

FOUR_GOOD_VARIANTS = [
    {
        "variant_id": "v1",
        "text": "Scratched already? This one shrugs it off. Tap to shop.",
        "framework": "hook_problem_product_cta",
        "hook_type": "pattern interrupt",
        "emotional_trigger": "curiosity",
        "grounding_truth_ids": ["t1", "t4"],
        "beats": [{"t_start": 0, "t_end": 3, "line": "Scratched already? Not this one."},
                  {"t_start": 3, "t_end": 12, "line": "This one shrugs it off."},
                  {"t_start": 12, "t_end": 15, "line": "Tap to shop."}],
    },
    {
        "variant_id": "v2",
        "text": "Stickers leave rings. Ours won't. Tap to shop.",
        "framework": "PAS",
        "hook_type": "bold claim",
        "emotional_trigger": "FOMO",
        "grounding_truth_ids": ["t3", "t5"],
        "beats": [{"t_start": 0, "t_end": 3, "line": "Stickers leave rings, not this base."},
                  {"t_start": 3, "t_end": 12, "line": "Ours won't."},
                  {"t_start": 12, "t_end": 15, "line": "Tap to shop."}],
    },
    {
        "variant_id": "v3",
        "text": "Every detail, debossed with care. Tap to shop.",
        "framework": "AIDA",
        "hook_type": "social proof",
        "emotional_trigger": "recognition",
        "grounding_truth_ids": ["t6", "t2"],
        "beats": [{"t_start": 0, "t_end": 3, "line": "Every detail debossed, not printed."},
                  {"t_start": 3, "t_end": 12, "line": "Built to last."},
                  {"t_start": 12, "t_end": 15, "line": "Tap to shop."}],
    },
    {
        "variant_id": "v4",
        "text": "From scratch to spotless. Tap to shop.",
        "framework": "BAB",
        "hook_type": "before/after",
        "emotional_trigger": "relief",
        "grounding_truth_ids": ["t1", "t3"],
        "beats": [{"t_start": 0, "t_end": 3, "line": "From scratch, not spotless -- until now."},
                  {"t_start": 3, "t_end": 12, "line": "Built for real life."},
                  {"t_start": 12, "t_end": 15, "line": "Tap to shop."}],
    },
]


TRUTH_EXTRACTOR_PAYLOAD = json.dumps(
    {
        "same_product": True,
        "mismatch_reason": "",
        "product_truths": [
            {"truth_id": f"t{i}", "fact": fact, "category": category, "source": "photo_1"}
            for i, (fact, category) in enumerate(GOOD_FACTS, start=1)
        ],
    }
)
CONCEPT_AGENT_PAYLOAD = json.dumps({"script_variants": FOUR_GOOD_VARIANTS})


@pytest.mark.asyncio
async def test_truth_extractor_and_concept_agent_run_chained_in_graph(monkeypatch):
    monkeypatch.setattr(
        "agents.product_truth_extractor.AsyncOpenAI",
        make_fake_async_openai([TRUTH_EXTRACTOR_PAYLOAD]),
    )
    monkeypatch.setattr(
        "agents.concept_agent.AsyncOpenAI",
        make_fake_async_openai([CONCEPT_AGENT_PAYLOAD]),
    )

    graph = await build_graph()
    initial_state = {
        "job_id": "test-job-graph",
        "product_photos": ["http://example.com/a.jpg"],
        "brief": "a durable everyday case",
    }
    config = {"configurable": {"thread_id": "test-job-graph"}}

    custom_events = [
        event
        async for event in graph.astream_events(initial_state, config=config, version="v2")
        if event.get("event") == "on_custom_event"
    ]

    assert len(custom_events) == 1, "only product_truth_extractor_node dispatches a custom event"
    assert custom_events[0]["name"] == "truth_extracted"
    assert custom_events[0]["data"]["count"] == len(GOOD_FACTS)

    final_state = await graph.aget_state(config)
    assert len(final_state.values["product_truths"]) == len(GOOD_FACTS)
    assert len(final_state.values["script_variants"]) == 4, (
        "concept_agent_node must have read product_truths from state (written by "
        "the upstream node) and produced 4 variants from them"
    )
