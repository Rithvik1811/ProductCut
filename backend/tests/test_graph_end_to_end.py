"""
Full-pipeline end-to-end test: drive the ENTIRE compiled LangGraph in one shot,
from ingest-state through Ken-Burns Fallback, faking every network boundary and
asserting a genuinely coherent final state (not merely "it didn't crash").

Faked boundaries (every module that builds an OpenAI client):
  * agents.product_truth_extractor.AsyncOpenAI  (async)  — truth extractor
  * agents.concept_agent.AsyncOpenAI            (async)  — concept agent
  * agents.hook_checker.AsyncOpenAI             (async)  — Hook-Checker
  * agents.critic_llm.OpenAI                    (sync)   — Body/CTA/Tone/Meta
  * agents.treatment_agent.AsyncOpenAI          (async)  — Treatment Agent (§5.5)
  * agents.shot_list_agent.AsyncOpenAI          (async)  — Shot-List Agent (§5.6)
  * agents.video_gen_node._call_wan_video_gen   (async)  — Video-Gen Node (§5.8)
  * agents.ken_burns_fallback_node.render_ken_burns_clip + agents._oss.upload_video_to_oss
    — Ken-Burns Fallback Node (§5.9)

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
from agents.video_gen_node import FALLBACK_REQUESTED_STATUS, SUCCESS_STATUS
from graph.build import build_graph
from tests._fakes import make_content_routed_sync_openai, make_fake_async_openai
from tests._phase3_graph import (
    patch_assembly_boundaries,
    patch_continuity_boundaries,
    patch_format_export_boundaries,
    patch_phase3_boundaries,
    patch_visual_direction_boundaries,
    patch_voiceover_boundaries,
)
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
async def test_full_pipeline_ingest_through_ken_burns_fallback(monkeypatch):
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
    patch_visual_direction_boundaries(monkeypatch)
    patch_phase3_boundaries(monkeypatch, fail_shot_s2=False)
    patch_continuity_boundaries(monkeypatch)  # Phase 4 (§5.10): clean drift, no retry loop
    patch_voiceover_boundaries(monkeypatch)  # Phase 5: parallel branch off merge_validator
    patch_assembly_boundaries(monkeypatch)  # Phase 5: fan-in join off voiceover + continuity_gate
    patch_format_export_boundaries(monkeypatch)  # Phase 6: format exports without real ffmpeg/OSS

    graph = await build_graph()
    initial_state = {
        "job_id": "test-job-e2e",
        "product_photos": ["http://example.com/a.jpg"],
        # Positive-Only Truths fix: signals the explicit authentic/well-loved
        # carve-out -- see test_graph_build.py's GOOD_FACTS/FOUR_GOOD_VARIANTS
        # comment for why (this shared fixture's narrative is imperfection-led).
        "brief": "a durable everyday case with authentic, well-loved character",
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
        "treatment_ready",  # Concept Agent re-prompt event
        "budget_updated",
        "shot_generated",
        "drift_scored",  # Phase 4: Continuity Agent scored every real clip
        "vo_ready",  # Phase 5: Voiceover + Caption Agent's parallel branch
        "master_cut_ready",  # Phase 5: Assembly Agent's fan-in join
        "job_complete",  # Phase 6: Format Export node signals delivery
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

    # --- Phase 3: every shot has a generated clip and a terminal status -------
    generated = values["generated_shots"]
    assert set(generated.keys()) == {shot["shot_id"] for shot in shot_list}
    for shot in shot_list:
        assert shot["status"] == SUCCESS_STATUS
        assert shot["retry_count"] == 0
        entry = generated[shot["shot_id"]]
        # Real clips are re-homed in OSS under the job's namespace (§5.8).
        assert entry["video_uri"].startswith("http://oss.example.com/jobs/test-job-e2e/shots/")
        assert entry["attempt"] == 1
    assert "[video_gen]" in values["reasoning_trace"]
    assert "persisted 3 to OSS" in values["reasoning_trace"]
    assert "[ken_burns_fallback]" in values["reasoning_trace"]

    # --- Phase 5: Voiceover + Caption Agent ran as a genuine parallel branch
    # off merge_validator, alongside treatment_agent -- not blocked by it, and
    # its own dedicated trace key landed in final state without corrupting the
    # treatment_agent-and-onward reasoning_trace chain above (the whole reason
    # it's a separate key -- see graph/state.py's v7 changelog).
    voiceover = values["voiceover"]
    assert voiceover["audio_uri"].startswith("http://oss.example.com/jobs/test-job-e2e/")
    assert voiceover["caption_track_uri"].startswith("http://oss.example.com/jobs/test-job-e2e/")
    assert "[voiceover_caption_agent]" in values["voiceover_reasoning_trace"]
    vo_event = next(e for e in custom_events if e["name"] == "vo_ready")
    assert vo_event["data"]["caption_count"] == len(winning["beats"])
    assert vo_event["data"]["degraded"] is False

    # --- Phase 5: Assembly Agent is the fan-in join of the voiceover branch
    # and the continuity retry loop (here a trivial single-pass loop) -- both
    # state["master_cut_uri"] and the master_cut_ready event reflect the fake
    # injected via patch_assembly_boundaries.
    assert values["master_cut_uri"] == f"http://oss.example.com/jobs/test-job-e2e/master_cut.mp4"
    master_cut_event = next(e for e in custom_events if e["name"] == "master_cut_ready")
    assert master_cut_event["data"]["uri"] == values["master_cut_uri"]
    assert master_cut_event["data"]["shot_count"] == len(shot_list)

    # --- shot_generated events: one per shot, all real (is_fallback=False) ----
    shot_events = [e for e in custom_events if e["name"] == "shot_generated"]
    assert len(shot_events) == len(shot_list)
    assert {e["data"]["shot_id"] for e in shot_events} == set(generated.keys())
    assert all(e["data"]["is_fallback"] is False for e in shot_events)
    assert all(e["data"]["status"] == SUCCESS_STATUS for e in shot_events)


@pytest.mark.asyncio
async def test_full_pipeline_video_gen_failure_routes_to_ken_burns_without_blocking(monkeypatch):
    """One shot's Wan failure must not block the others; Ken-Burns recovers it."""
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
    patch_visual_direction_boundaries(monkeypatch)
    patch_phase3_boundaries(monkeypatch, fail_shot_s2=True)
    patch_continuity_boundaries(monkeypatch)  # Phase 4 (§5.10): clean drift, no retry loop
    patch_voiceover_boundaries(monkeypatch)  # Phase 5: parallel branch off merge_validator
    patch_assembly_boundaries(monkeypatch)  # Phase 5: fan-in join off voiceover + continuity_gate
    patch_format_export_boundaries(monkeypatch)  # Phase 6: format exports without real ffmpeg/OSS

    graph = await build_graph()
    initial_state = {
        "job_id": "test-job-e2e-mixed",
        "product_photos": ["http://example.com/a.jpg"],
        # Positive-Only Truths fix: signals the explicit authentic/well-loved
        # carve-out -- see test_graph_build.py's GOOD_FACTS/FOUR_GOOD_VARIANTS
        # comment for why (this shared fixture's narrative is imperfection-led).
        "brief": "a durable everyday case with authentic, well-loved character",
        "budget_ledger": {"cap": _SEEDED_CAP, "spent": 0.0, "per_shot": {}},
    }
    config = {"configurable": {"thread_id": "test-job-e2e-mixed"}}

    custom_events = [
        e
        async for e in graph.astream_events(initial_state, config=config, version="v2")
        if e.get("event") == "on_custom_event"
    ]

    values = (await graph.aget_state(config)).values
    shot_list = values["shot_list"]
    generated = values["generated_shots"]
    by_id = {s["shot_id"]: s for s in shot_list}

    # s2's Call-B description is unique ("asymmetric rear vent") — our fake Wan fails on it.
    assert by_id["s1"]["status"] == SUCCESS_STATUS
    assert by_id["s2"]["status"] == "fallback"
    assert by_id["s3"]["status"] == SUCCESS_STATUS
    assert set(generated.keys()) == {"s1", "s2", "s3"}
    assert generated["s2"]["video_uri"].startswith("http://oss.example.com/jobs/test-job-e2e-mixed/shots/s2/")
    for shot in shot_list:
        assert shot["retry_count"] == 0
    assert FALLBACK_REQUESTED_STATUS not in {s["status"] for s in shot_list}

    # --- shot_generated events: s1/s3 real, s2 fallback — one per shot --------
    shot_events = {e["data"]["shot_id"]: e["data"] for e in custom_events if e["name"] == "shot_generated"}
    assert set(shot_events.keys()) == {"s1", "s2", "s3"}
    assert shot_events["s1"]["is_fallback"] is False
    assert shot_events["s3"]["is_fallback"] is False
    assert shot_events["s2"]["is_fallback"] is True
    assert shot_events["s2"]["status"] == "fallback"
