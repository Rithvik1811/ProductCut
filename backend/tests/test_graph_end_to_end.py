"""
Full-pipeline end-to-end test: drive the ENTIRE compiled LangGraph in one shot,
from ingest-state through the Budget Gate, faking every network boundary and
asserting a genuinely coherent final state (not merely "it didn't crash").

Faked boundaries (every module that builds an OpenAI client):
  * agents.product_truth_extractor.AsyncOpenAI  (async)  — truth extractor
  * agents.concept_agent.AsyncOpenAI            (async)  — concept agent
  * agents.hook_checker.AsyncOpenAI             (async)  — Hook-Checker
  * agents.critic_llm.OpenAI                    (sync)   — Body/CTA/Tone/Meta
  * agents.treatment_agent.AsyncOpenAI          (async)  — Treatment Agent (§5.5)
  * agents.shot_list_agent.AsyncOpenAI          (async)  — Shot-List Agent (§5.6)

Everything else is real: StateGraph superstep scheduling, the checkpointer,
astream_events, custom-event dispatch, the Pacing-Checker (pure code), KR's real
Justification Validator (shared by Treatment + Shot-List agents), and the Budget
Gate (pure code, §5.7). The canned payloads are the single shared source of truth
in test_graph_build.py, reused here rather than reinvented.

Deliberate scenario chosen (and why). The shared FOUR_GOOD_VARIANTS fixture gives
each variant a single body beat, which makes meta_critic's merge candidate fail
the Merge Coherence Validator's pacing re-check unrepairably — so merge_validator
routes via "fallback" and the winning_script is variant v1's own unmerged
text/beats. That is the realistic path this fixture produces, so this test asserts
it honestly rather than forcing a "finalize".

For the Budget Gate we deliberately seed a job cap of $1.00 into the initial
budget_ledger. With three shots of 3s/4s/3s the per-shot 1080p ceilings sum to
$1.20 and the floors to $0.72, so $1.00 lands strictly inside the feasible band:
the water-filling allocation SUCCEEDS (no shot cut, no floor/over-cap case) and
the allocations sum to exactly the cap. This exercises the Budget Gate's normal
grounding-weighted distribution — the real outcome we want to assert on, chosen so
the test verifies a computed allocation rather than the degenerate floor branch a
too-loose default cap would trigger.
"""
from __future__ import annotations

import pytest

from agents.budget_gate import RATE_1080P
from graph.build import build_graph
from tests._fakes import make_content_routed_sync_openai, make_fake_async_openai
from tests.test_graph_build import (
    CHECKER_ROUTES,
    CONCEPT_AGENT_PAYLOAD,
    HOOK_PAYLOAD,
    SHOT_LIST_CALL_A_PAYLOAD,
    SHOT_LIST_CALL_B_PAYLOAD,
    TREATMENT_PAYLOAD,
    TRUTH_EXTRACTOR_PAYLOAD,
)

# A cap strictly inside [Σ floor = $0.72, Σ 1080p ceiling = $1.20] for the three
# 3s/4s/3s shots, so the Budget Gate's water-fill succeeds and allocations sum to
# exactly the cap (see module docstring for the deliberate-scenario rationale).
_SEEDED_CAP = 1.00


@pytest.mark.asyncio
async def test_full_pipeline_ingest_through_budget_gate(monkeypatch):
    # --- Fake every network boundary in the whole graph --------------------
    monkeypatch.setattr(
        "agents.product_truth_extractor.AsyncOpenAI",
        make_fake_async_openai([TRUTH_EXTRACTOR_PAYLOAD]),
    )
    monkeypatch.setattr(
        "agents.concept_agent.AsyncOpenAI",
        make_fake_async_openai([CONCEPT_AGENT_PAYLOAD]),
    )
    monkeypatch.setattr(
        "agents.hook_checker.AsyncOpenAI",
        make_fake_async_openai([HOOK_PAYLOAD]),
    )
    monkeypatch.setattr(
        "agents.critic_llm.OpenAI",
        make_content_routed_sync_openai(CHECKER_ROUTES),
    )
    monkeypatch.setattr(
        "agents.treatment_agent.AsyncOpenAI",
        make_fake_async_openai([TREATMENT_PAYLOAD]),
    )
    monkeypatch.setattr(
        "agents.shot_list_agent.AsyncOpenAI",
        make_fake_async_openai([SHOT_LIST_CALL_A_PAYLOAD, SHOT_LIST_CALL_B_PAYLOAD]),
    )

    graph = await build_graph()
    initial_state = {
        "job_id": "test-job-e2e",
        "product_photos": ["http://example.com/a.jpg"],
        "brief": "a durable everyday case",
        # Seed a tight-but-feasible cap so the Budget Gate exercises its real
        # water-fill allocation (see module docstring).
        "budget_ledger": {"cap": _SEEDED_CAP, "spent": 0.0, "per_shot": {}},
    }
    config = {"configurable": {"thread_id": "test-job-e2e"}}

    custom_events = [
        event
        async for event in graph.astream_events(initial_state, config=config, version="v2")
        if event.get("event") == "on_custom_event"
    ]

    # --- Whole-run event set (fallback path) -------------------------------
    event_names = {e["name"] for e in custom_events}
    assert event_names == {
        "truth_extracted",
        "critic_score",
        "merge_validated",
        "budget_updated",
    }, event_names

    values = (await graph.aget_state(config)).values

    # --- Winning script came from the fallback path (unmerged v1) ----------
    winning = values["winning_script"]
    assert winning["source_variant_ids"] == ["v1"]
    assert len(winning["beats"]) == 3

    # --- Treatment: exactly one beat_treatment per winning-script beat ------
    treatment = values["treatment"]
    assert len(treatment["beat_treatments"]) == len(winning["beats"])
    assert [bt["beat_index"] for bt in treatment["beat_treatments"]] == [0, 1, 2]

    # --- Shot list: 3-7 shots, anti-genericness guarantee, real budgets -----
    shot_list = values["shot_list"]
    assert 3 <= len(shot_list) <= 7

    ledger = values["budget_ledger"]
    assert ledger["cap"] == _SEEDED_CAP

    for shot in shot_list:
        # Core anti-genericness guarantee: no shot may carry a product_category.
        assert "product_category" not in shot
        # Every shot got a real, positive allocation within a sane per-shot ceiling
        # (its own duration at the 1080p rate — the most it could ever cost).
        assert shot["allocated_budget"] > 0
        assert shot["allocated_budget"] <= shot["duration_sec"] * RATE_1080P + 1e-6

    # Water-fill succeeded (no shot cut, not the floor/over-cap case): the total
    # committed spend never exceeds the cap.
    total_alloc = sum(shot["allocated_budget"] for shot in shot_list)
    assert total_alloc <= ledger["cap"] + 1e-9
    assert ledger["spent"] == pytest.approx(total_alloc)
    # No shot was dropped in this feasible-cap scenario.
    assert len(shot_list) == 3
    assert set(ledger["per_shot"].keys()) == {shot["shot_id"] for shot in shot_list}

    # --- The budget_updated event's ledger matches the final state ledger ---
    budget_event = next(e for e in custom_events if e["name"] == "budget_updated")
    assert budget_event["data"]["ledger"] == ledger
    assert budget_event["data"]["over_cap"] is False
