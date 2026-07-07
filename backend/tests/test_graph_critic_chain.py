"""
Full-chain integration test: does the compiled LangGraph actually drive the
Concept Agent -> 5 parallel checkers (fan-out) -> Meta-Critic (fan-in) wiring,
end to end, when LangGraph itself runs it — not just when the node functions are
called by hand?

This extends tests/test_graph_build.py (which stops asserting at concept_agent)
forward through the whole Critic Chain. Only the network boundary is faked; the
StateGraph, superstep fan-out/fan-in scheduling, checkpointer, astream_events and
custom-event dispatch are all the real thing.

Fakes, at the four module boundaries that actually construct a client:
  * agents.product_truth_extractor.AsyncOpenAI  (async)   — truth extractor
  * agents.concept_agent.AsyncOpenAI            (async)   — concept agent
  * agents.hook_checker.AsyncOpenAI             (async)   — Hook-Checker
  * agents.critic_llm.OpenAI                    (sync)    — Body/CTA/Tone/Meta

Pacing-Checker is pure deterministic code (no client), so it is NOT faked.

Body/CTA/Tone/Meta-Critic all route through `critic_llm.call_qwen_json`, which
builds a fresh `OpenAI()` per call, so each gets its own instance with
call_count==0 — a flat response list would serve them all `responses[0]`. The
shared `make_content_routed_sync_openai` fake instead routes by system-prompt
substring (unique per checker), which is also robust to the non-deterministic
order the parallel branches run in. The canned payloads + routes live in
test_graph_build.py so both graph tests share exactly one source of truth.
"""
from __future__ import annotations

import pytest

from graph.build import build_graph
from tests._fakes import make_content_routed_sync_openai, make_fake_async_openai
from tests.test_graph_build import (
    CHECKER_ROUTES,
    CONCEPT_AGENT_PAYLOAD,
    GOOD_FACTS,
    HOOK_PAYLOAD,
    TRUTH_EXTRACTOR_PAYLOAD,
    _VARIANT_IDS,
)


@pytest.mark.asyncio
async def test_full_critic_chain_runs_fanout_fanin_in_graph(monkeypatch):
    # Upstream agents (async clients).
    monkeypatch.setattr(
        "agents.product_truth_extractor.AsyncOpenAI",
        make_fake_async_openai([TRUTH_EXTRACTOR_PAYLOAD]),
    )
    monkeypatch.setattr(
        "agents.concept_agent.AsyncOpenAI",
        make_fake_async_openai([CONCEPT_AGENT_PAYLOAD]),
    )
    # Hook-Checker has its own async client.
    monkeypatch.setattr(
        "agents.hook_checker.AsyncOpenAI",
        make_fake_async_openai([HOOK_PAYLOAD]),
    )
    # Body/CTA/Tone/Meta-Critic share this one sync target -> content routing.
    monkeypatch.setattr(
        "agents.critic_llm.OpenAI",
        make_content_routed_sync_openai(CHECKER_ROUTES),
    )

    graph = await build_graph()
    initial_state = {
        "job_id": "test-job-critic-chain",
        "product_photos": ["http://example.com/a.jpg"],
        "brief": "a durable everyday case",
    }
    config = {"configurable": {"thread_id": "test-job-critic-chain"}}

    custom_events = [
        event
        async for event in graph.astream_events(initial_state, config=config, version="v2")
        if event.get("event") == "on_custom_event"
    ]

    # Three custom events fire across the whole chain: truth_extracted (from the
    # truth extractor), critic_score (from the meta_critic fan-in node), and
    # merge_validated (from the Merge Coherence Validator, §5.4.7, now wired in
    # after meta_critic -- see test_graph_merge_validator.py for its own assertions).
    names = {e["name"] for e in custom_events}
    assert names == {"truth_extracted", "critic_score", "merge_validated"}, names
    truth_event = next(e for e in custom_events if e["name"] == "truth_extracted")
    assert truth_event["data"]["count"] == len(GOOD_FACTS)
    critic_event = next(e for e in custom_events if e["name"] == "critic_score")
    assert set(critic_event["data"]["scores"].keys()) == set(_VARIANT_IDS)
    assert critic_event["data"]["winning_variant_ids"] == ["v1", "v2", "v3"]

    final = (await graph.aget_state(config)).values

    # Fan-out: every one of the 5 checkers wrote its own per-variant score dict.
    for key in ("hook_scores", "pacing_scores", "body_scores", "cta_scores", "tone_scores"):
        assert key in final, f"{key} missing — a checker branch did not run"
        assert set(final[key].keys()) == set(_VARIANT_IDS), f"{key} did not cover all variants"

    # Fan-in: meta_critic assembled the composite CriticScore per variant.
    critic_scores = final["critic_scores"]
    assert set(critic_scores.keys()) == set(_VARIANT_IDS)
    for vid, cs in critic_scores.items():
        assert cs["composite"] > 0, f"{vid} composite not computed"
        assert cs["justification"], f"{vid} justification not assembled"
        assert cs["never_do_violation"] is False

    # Fan-in: the raw MetaCriticResult was stashed for the downstream validator.
    mcr = final["meta_critic_result"]
    assert mcr["outcome"] == "cross_pollinated"
    assert mcr["survivor_ids"] == _VARIANT_IDS
    mc = mcr["merge_candidate"]
    assert mc is not None
    assert (
        mc["hook_source_variant_id"],
        mc["body_source_variant_id"],
        mc["cta_source_variant_id"],
    ) == ("v1", "v2", "v3")
    # Step 6 re-timing ran and produced a contiguous candidate ending at target.
    assert mc["merged_beats"][-1]["t_end"] == pytest.approx(mc["target_length_sec"])
