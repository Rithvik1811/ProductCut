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
from tests._fakes import make_content_routed_sync_openai, make_fake_async_openai

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

_VARIANT_IDS = [v["variant_id"] for v in FOUR_GOOD_VARIANTS]  # v1..v4

# Canned checker responses. The graph now fans concept_agent out into the five
# Critic-Chain checkers and back into the Meta-Critic, so even a test that only
# cares about the upstream two agents must mock the whole chain's network
# boundary or the downstream checkers hit the real DashScope endpoint.
HOOK_PAYLOAD = json.dumps(
    {
        "hook_scores": [
            {"variant_id": vid, "hook_score": 4, "justification": f"{vid} hook is fine"}
            for vid in _VARIANT_IDS
        ]
    }
)
BODY_PAYLOAD = json.dumps(
    {
        "results": [
            {
                "variant_id": vid,
                "completion_score": 4,
                "redundant_beat_pairs": [],
                "promise_payoff_match": True,
                "emotional_trigger_landed": True,
                "justification": f"{vid} body pays off the hook",
            }
            for vid in _VARIANT_IDS
        ]
    }
)
CTA_PAYLOAD = json.dumps(
    {
        "results": [
            {"variant_id": vid, "cta_score": 4, "justification": f"{vid} cta is clear"}
            for vid in _VARIANT_IDS
        ]
    }
)
TONE_PAYLOAD = json.dumps(
    {
        "results": [
            {
                "variant_id": vid,
                "tone_score": 4,
                "never_do_violation": False,
                "justification": f"{vid} is on brand",
            }
            for vid in _VARIANT_IDS
        ]
    }
)
# Valid MetaCriticLLMOutput cross-pollinating hook v1 / body v2 / cta v3.
META_PAYLOAD = json.dumps(
    {
        "leaderboards": [
            {"axis": "hook", "ranked_variant_ids": ["v1", "v2", "v3", "v4"], "note": "hook order"},
            {"axis": "completion", "ranked_variant_ids": ["v2", "v1", "v3", "v4"], "note": "body order"},
            {"axis": "cta", "ranked_variant_ids": ["v3", "v1", "v2", "v4"], "note": "cta order"},
        ],
        "audition": {
            "promise_payoff": "v2 body develops v1's hook claim",
            "hook_body_seam": "'v1 hook' -> 'v2 body': continuous",
            "body_cta_seam": "'v2 body' -> 'v3 cta': continuous",
            "trigger_continuity": "curiosity escalates into the ask",
            "passed": True,
            "risks_to_flag_forward": [],
        },
        "substitutions": [],
        "hook_source_variant_id": "v1",
        "body_source_variant_id": "v2",
        "cta_source_variant_id": "v3",
        "no_compatible_merge": False,
        "rationale": [
            {
                "axis": "hook",
                "decision": "hook from v1",
                "quoted_evidence": "'v1 hook' scored 4",
                "steelmanned_runner_up": "v2's hook is punchy too",
                "trade_off": "gave up v2's opener",
                "why_it_holds": "audition passed the hook-body seam",
                "named_risk": "hook-body seam",
            },
            {
                "axis": "body",
                "decision": "body from v2",
                "quoted_evidence": "'v2 body' scored 4",
                "steelmanned_runner_up": "v1's body is coherent",
                "trade_off": "gave up v1's body",
                "why_it_holds": "audition passed promise-payoff",
                "named_risk": "body beat 1",
            },
            {
                "axis": "cta",
                "decision": "cta from v3",
                "quoted_evidence": "'v3 cta' scored 4",
                "steelmanned_runner_up": "v1's cta is clean",
                "trade_off": "gave up v1's cta",
                "why_it_holds": "audition passed the body-cta seam",
                "named_risk": "body-cta seam",
            },
        ],
        "overall_reasoning": "cross-pollinated the strongest hook, body and cta across survivors",
    }
)
# (needle in the checker's SYSTEM prompt, canned response). Shared by the
# full-chain test in test_graph_critic_chain.py.
CHECKER_ROUTES = [
    ("META-CRITIC", META_PAYLOAD),
    ("COLD read of the BODY", BODY_PAYLOAD),
    ("call-to-action (CTA) clarity", CTA_PAYLOAD),
    ("brand-tone critic", TONE_PAYLOAD),
]


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
    # The critic chain is now wired into the graph, so running the graph fans out
    # into all five checkers + the Meta-Critic. Mock their network boundary too,
    # or the downstream checkers hit the real endpoint (Hook uses its own async
    # client; Body/CTA/Tone/Meta share critic_llm's sync OpenAI).
    monkeypatch.setattr(
        "agents.hook_checker.AsyncOpenAI",
        make_fake_async_openai([HOOK_PAYLOAD]),
    )
    monkeypatch.setattr(
        "agents.critic_llm.OpenAI",
        make_content_routed_sync_openai(CHECKER_ROUTES),
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

    # Three custom events now fire across the full chain: truth_extracted (from the
    # truth extractor), critic_score (from the Meta-Critic fan-in node), and
    # merge_validated (from the Merge Coherence Validator, §5.4.7, now wired in
    # after meta_critic). FOUR_GOOD_VARIANTS gives each variant a single body
    # beat, so the merge candidate's pacing re-check fails unrepairably here and
    # merge_validator routes straight to the fallback -- see
    # test_graph_merge_validator.py for that path's dedicated assertions.
    event_names = {e["name"] for e in custom_events}
    assert event_names == {"truth_extracted", "critic_score", "merge_validated"}, event_names
    truth_event = next(e for e in custom_events if e["name"] == "truth_extracted")
    assert truth_event["data"]["count"] == len(GOOD_FACTS)

    final_state = await graph.aget_state(config)
    assert len(final_state.values["product_truths"]) == len(GOOD_FACTS)
    assert len(final_state.values["script_variants"]) == 4, (
        "concept_agent_node must have read product_truths from state (written by "
        "the upstream node) and produced 4 variants from them"
    )
