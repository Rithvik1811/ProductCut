"""
Adversarial WIRED-TOGETHER edge cases for the Phase 1->Phase 2 pipeline.

The existing end-to-end test (test_graph_end_to_end.py) deliberately seeds a
loose-but-feasible cap ($1.00) so the Budget Gate exercises only its NORMAL
water-fill allocation branch: no shot is ever cut and the over-cap floor branch
never fires. That leaves two of the Budget Gate's three §5.7 branches completely
un-exercised in a real, wired-together graph run (they are only ever hit by
budget_gate's own unit tests, which hand-build Shot dicts directly rather than
letting the REAL Shot-List Agent's assembled output flow through them):

  1. The floor / over-cap branch (§5.7 step 4): cap too small to fit even the
     cheapest render of MIN_SHOTS shots -> every shot pinned to FLOOR_COST, the
     overage ACCEPTED and FLAGGED (over_cap=True), not hidden.
  2. The deterministic cut / reduce branch (§5.7): cap infeasible with N>MIN_SHOTS
     shots -> the single lowest-weight shot is dropped and the whole allocation
     retried, until it fits or the 3-shot floor is reached.

Branch (2) is UNREACHABLE with the shared fixture, because that fixture produces
exactly MIN_SHOTS (3) shots and the reduce loop bails to the floor case the moment
`len(working) <= MIN_SHOTS` -- so no amount of cap-tightening against the 3-shot
fixture can force a genuine cut. This file adds a 5-shot fixture (assembled by the
REAL Shot-List Agent through the REAL graph) specifically to drive that path.

Both tests reuse the real upstream fixtures / real Justification Validator / real
Budget Gate; only the network boundaries are faked, exactly as the shared e2e test
does.

The final test ORIGINALLY characterized a genuine latent bug (not a crash) found
while tracing the fallback path: `merge_validator._winning_script_from_fallback_variant`
preferred a variant's own polished `text` field over joining its beat lines, so
`winning_script["text"]` could diverge from `beats[].line` -- while both the
Treatment Agent and Shot-List Agent PROMPT the model to quote a beat's own line,
and KR's validator checks that quote against `winning_script["text"]`. A
correctly-instructed model response could therefore fail validation for no real
reason. THIS HAS SINCE BEEN FIXED (in `merge_validator.py` and, since the same
`variant.get("text") or ...` pattern existed there too, in `meta_critic.py`'s
single-survivor short-circuit) -- both now always join beat lines, matching the
normal cross-pollinated merge path's pre-existing, always-correct convention. The
test below now verifies the FIX holds against the real function, not a hand-built
dict that never actually exercised the buggy code path.
"""
from __future__ import annotations

import json

import pytest

from agents.budget_gate import FLOOR_COST, RATE_1080P, W_ROLE, W_TYPE, TRUTH_BONUS
from agents.justification_validator import validate_justifications
from agents.merge_validator import _winning_script_from_fallback_variant
from graph.build import build_graph
from tests._fakes import make_content_routed_sync_openai, make_fake_async_openai
from tests._phase3_graph import (
    patch_assembly_boundaries,
    patch_phase3_boundaries,
    patch_voiceover_boundaries,
)
from tests.test_graph_build import (
    CHECKER_ROUTES,
    CONCEPT_AGENT_PAYLOAD,
    FOUR_GOOD_VARIANTS,
    HOOK_PAYLOAD,
    TREATMENT_PAYLOAD,
    TRUTH_EXTRACTOR_PAYLOAD,
    SHOT_LIST_CALL_A_PAYLOAD,
    SHOT_LIST_CALL_B_PAYLOAD,
)


def _patch_all_boundaries(monkeypatch, *, call_a: str, call_b: str) -> None:
    """Fake every network boundary in the whole graph, parameterizing only the
    Shot-List Agent's two responses (so a test can vary shot count/shape)."""
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
        make_fake_async_openai([call_a, call_b]),
    )
    patch_phase3_boundaries(monkeypatch, fail_shot_s2=False)
    patch_voiceover_boundaries(monkeypatch)  # Phase 5: parallel branch off merge_validator
    patch_assembly_boundaries(monkeypatch)  # Phase 5: fan-in join off voiceover + continuity_gate


async def _run_graph(monkeypatch, *, cap: float, call_a: str, call_b: str):
    """Drive the full compiled graph with a seeded cap, returning (values, events)."""
    _patch_all_boundaries(monkeypatch, call_a=call_a, call_b=call_b)
    graph = await build_graph()
    thread = f"edge-{cap}-{hash(call_a) & 0xffff}"
    initial_state = {
        "job_id": thread,
        "product_photos": ["http://example.com/a.jpg"],
        # Positive-Only Truths fix: signals the explicit authentic/well-loved
        # carve-out -- see test_graph_build.py's GOOD_FACTS/FOUR_GOOD_VARIANTS
        # comment for why (this shared fixture's narrative is imperfection-led).
        "brief": "a durable everyday case with authentic, well-loved character",
        "budget_ledger": {"cap": cap, "spent": 0.0, "per_shot": {}},
    }
    config = {"configurable": {"thread_id": thread}}
    events = [
        e
        async for e in graph.astream_events(initial_state, config=config, version="v2")
        if e.get("event") == "on_custom_event"
    ]
    values = (await graph.aget_state(config)).values
    return values, events


# ===========================================================================
# GAP 1 — the Budget Gate's floor / over-cap branch, exercised end-to-end with
# the REAL 3-shot fixtures (only the seeded cap differs from the shared e2e test).
# ===========================================================================
@pytest.mark.asyncio
async def test_full_pipeline_over_cap_floor_case_with_real_fixtures(monkeypatch):
    # Three shots -> Σ FLOOR_COST = 3 * $0.24 = $0.72. A cap BELOW that cannot fit
    # even the cheapest render, and with only MIN_SHOTS shots there is nothing to
    # cut: the §5.7 floor branch must fire (over_cap=True, non-zero overage).
    cap = 0.50
    values, events = await _run_graph(
        monkeypatch, cap=cap,
        call_a=SHOT_LIST_CALL_A_PAYLOAD, call_b=SHOT_LIST_CALL_B_PAYLOAD,
    )

    shot_list = values["shot_list"]
    ledger = values["budget_ledger"]

    # Nothing was cut (can't drop below MIN_SHOTS); all three survive.
    assert len(shot_list) == 3
    assert set(ledger["per_shot"].keys()) == {s["shot_id"] for s in shot_list}

    # Every shot pinned to the honest cheapest-possible spend, FLOOR_COST.
    for shot in shot_list:
        assert shot["allocated_budget"] == pytest.approx(FLOOR_COST)

    # Spent is Σ floor = $0.72 and it is ABOVE the cap: the overage is real, not
    # papered over by pushing shots below their floor.
    expected_spent = 3 * FLOOR_COST
    assert ledger["spent"] == pytest.approx(expected_spent)
    assert ledger["spent"] > ledger["cap"] + 1e-9
    assert ledger["cap"] == cap

    # The event and the trace both surface the over-cap condition honestly.
    budget_event = next(e for e in events if e["name"] == "budget_updated")
    assert budget_event["data"]["over_cap"] is True
    assert budget_event["data"]["ledger"] == ledger
    assert "OVER CAP" in values["reasoning_trace"]
    # Overage surfaced in the trace = Σfloor - cap = 0.72 - 0.50 = 0.22.
    assert f"${expected_spent - cap:.4f}" in values["reasoning_trace"]


# ===========================================================================
# GAP 2 — the Budget Gate's deterministic CUT/reduce branch, exercised end-to-end
# with a REAL, Shot-List-Agent-assembled 5-shot list flowing through it (the exact
# shape budget_gate's own unit tests never see, because they hand-build Shot dicts).
# ===========================================================================

# Five Call-A shots (all quotes verbatim spans of v1's fallback TEXT, real truth
# ids, real treatment_refs 0/1/2), assigned deliberately DISTINCT grounding weights
# so the deterministic lowest-weight cut order is predictable:
#   s3  proof  / macro_detail      / t2 construction_detail -> 1.15*1.30*1.10 = 1.6445  (highest)
#   s1  hook   / hook_hero         / t1 imperfection         -> 1.20*1.15*1.10 = 1.5180
#   s2  cta    / cta_endcard       / t3 imperfection         -> 1.20*1.10*1.10 = 1.4520
#   s5  demo   / hero_reframe      / t6 construction_detail  -> 1.00*1.00*1.10 = 1.1000
#   s4  problem/ lifestyle_context / t4 texture              -> 0.90*0.90*1.10 = 0.8910  (lowest)
# So the two lowest-weight shots cut are s4 then s5; survivors are {s1, s2, s3}.
_FIVE_CALL_A = json.dumps(
    {
        "shots": [
            {"shot_id": "s1", "beat_role": "hook", "script_quote": "Scratched already?", "truth_fact_id": "t1", "treatment_ref": 0},
            {"shot_id": "s2", "beat_role": "cta", "script_quote": "Tap to shop.", "truth_fact_id": "t3", "treatment_ref": 2},
            {"shot_id": "s3", "beat_role": "proof", "script_quote": "This one shrugs it off.", "truth_fact_id": "t2", "treatment_ref": 1},
            {"shot_id": "s4", "beat_role": "problem", "script_quote": "This one shrugs it off.", "truth_fact_id": "t4", "treatment_ref": 1},
            {"shot_id": "s5", "beat_role": "demo", "script_quote": "Tap to shop.", "truth_fact_id": "t6", "treatment_ref": 2},
        ]
    }
)


def _call_b_shot(shot_id: str, shot_type: str, overlay: str = "none") -> dict:
    return {
        "shot_id": shot_id,
        "shot_type": shot_type,
        "camera_move": "static",
        "framing": "fills_frame",
        "text_overlay_zone": overlay,
        "duration_sec": 3,
        "voiceover_line": "line",
        "description": (
            "Matte graphite housing fills the frame under a steady warm key, the cited "
            "detail held in sharp focus with no camera movement. Preserve product shape, "
            "keep label text, keep proportions."
        ),
        "negative_prompt_extra": "",
    }


_FIVE_CALL_B = json.dumps(
    {
        "lighting": "warm tungsten key with a soft graphite falloff, constant across every shot",
        "shots": [
            _call_b_shot("s1", "hook_hero"),
            _call_b_shot("s2", "cta_endcard", overlay="lower_third"),
            _call_b_shot("s3", "macro_detail"),
            _call_b_shot("s4", "lifestyle_context"),
            _call_b_shot("s5", "hero_reframe"),
        ],
    }
)


def _weight(role: str, shot_type: str, truth_category_specific: bool) -> float:
    return W_ROLE[role] * W_TYPE[shot_type] * (TRUTH_BONUS if truth_category_specific else 1.0)


@pytest.mark.asyncio
async def test_full_pipeline_forces_genuine_cut_with_five_shots(monkeypatch):
    # Five 3s shots -> Σ FLOOR_COST = 5 * $0.24 = $1.20 (infeasible at cap $0.90);
    # at 4 shots Σ floor = $0.96 (still infeasible); at 3 shots Σ floor = $0.72 and
    # Σ 1080p ceiling = $1.08, so $0.90 lands inside the feasible band. The Budget
    # Gate must therefore cut EXACTLY the two lowest-weight shots (s4, then s5) and
    # then allocate the surviving three to sum to exactly the cap.
    cap = 0.90
    values, events = await _run_graph(
        monkeypatch, cap=cap, call_a=_FIVE_CALL_A, call_b=_FIVE_CALL_B,
    )

    # Sanity: the Shot-List Agent really did assemble five structurally-valid shots
    # before the Budget Gate ran (the winning_script fell back to v1, 3 beats).
    assert values["winning_script"]["source_variant_ids"] == ["v1"]

    shot_list = values["shot_list"]
    ledger = values["budget_ledger"]

    # Genuine cut path fired: two lowest-weight shots removed, down to MIN_SHOTS.
    surviving_ids = {s["shot_id"] for s in shot_list}
    assert surviving_ids == {"s1", "s2", "s3"}, surviving_ids
    assert "s4" not in surviving_ids and "s5" not in surviving_ids
    # Cut shots are absent from BOTH the shot list and the ledger breakdown.
    assert set(ledger["per_shot"].keys()) == {"s1", "s2", "s3"}
    assert "s4" not in ledger["per_shot"] and "s5" not in ledger["per_shot"]

    # The surviving three are exactly the three HIGHEST-weight shots (the §5.7
    # "never cut the hook/cta/top-proof" protection falling out of the weighting).
    dropped_weight = _weight("problem", "lifestyle_context", True)   # s4
    kept_weights = [
        _weight("hook", "hook_hero", True),      # s1
        _weight("cta", "cta_endcard", True),     # s2
        _weight("proof", "macro_detail", True),  # s3
        _weight("demo", "hero_reframe", True),   # s5 (also dropped)
    ]
    assert dropped_weight < min(kept_weights)  # s4 is unambiguously the first cut

    # This is NOT the floor/over-cap case: allocation succeeded inside the windows.
    budget_event = next(e for e in events if e["name"] == "budget_updated")
    assert budget_event["data"]["over_cap"] is False

    # Allocations sum to exactly the cap and each lands within its feasible window.
    total = sum(s["allocated_budget"] for s in shot_list)
    assert total == pytest.approx(cap)
    assert ledger["spent"] == pytest.approx(total)
    assert total <= ledger["cap"] + 1e-9
    for shot in shot_list:
        assert FLOOR_COST - 1e-9 <= shot["allocated_budget"] <= shot["duration_sec"] * RATE_1080P + 1e-9

    # The reduce is reflected honestly in the trace.
    assert "cut 2" in values["reasoning_trace"]


# ===========================================================================
# LATENT SEAM (characterization, not a crash) — fallback winning_script.text is
# NOT the concatenation of its beat lines, but the Treatment/Shot-List prompts ask
# the model to quote a beat's OWN LINE while KR's validator checks the quote against
# winning_script["text"]. So the exact thing the prompt requests is rejected in the
# fallback path -> forces those beats/shots onto the re-prompt+fallback path, where
# the lifted-from-treatment quote is NOT re-validated against the text and can reach
# the final shot list ungrounded. Model-behaviour-dependent (the hand-authored e2e
# fixtures dodge it by quoting only text-substrings), so this pins the seam
# deterministically at the validator boundary rather than asserting a model output.
# ===========================================================================
@pytest.mark.asyncio
async def test_fallback_winning_text_always_contains_every_beat_line():
    """Regression test for the FIXED bug: `_winning_script_from_fallback_variant`
    now always builds `text` by joining beat lines, never from the variant's own
    (potentially divergent) `text` field. Confirms the invariant Treatment
    Agent/Shot-List Agent's prompts and KR's validator both implicitly depend on
    (every beat's own line the model is asked to quote is a real substring of
    `winning_script["text"]`) now holds for real, via the actual production
    function -- not a hand-built dict that never exercised the fixed code path.
    """
    v1 = FOUR_GOOD_VARIANTS[0]  # the fallback winner in the shared fixtures

    # v1's own `text` field is confirmed still structurally divergent from its
    # beat lines (this is real, upstream Concept Agent output shape) -- if the fix
    # were reverted, `_winning_script_from_fallback_variant` would surface that
    # divergence; the assertions below prove it no longer does.
    assert v1["beats"][0]["line"] not in v1["text"]

    state = {
        "meta_critic_result": {"fallback_variant_id": "v1"},
        "script_variants": FOUR_GOOD_VARIANTS,
    }
    winning_script = _winning_script_from_fallback_variant(state)

    # The invariant: every beat's own line is a real substring of the returned text.
    for beat in winning_script["beats"]:
        assert beat["line"] in winning_script["text"], (
            f"beat line {beat['line']!r} is not a substring of winning_script text "
            f"{winning_script['text']!r} -- the text/beats consistency fix regressed"
        )

    # And the concrete case that used to fail now validates cleanly, quoting the
    # beat's own line exactly as the Treatment/Shot-List prompts instruct.
    product_truths = [{"truth_id": "t1", "fact": "a scratch", "category": "imperfection", "source": "photo_1"}]
    treatment = {
        "director_persona": "x", "color_story": "y", "pacing_philosophy": "z",
        "beat_treatments": [
            {"beat_index": 0, "beat_function": "hook", "script_quote": winning_script["beats"][0]["line"],
             "truth_fact_id": "t1", "visual_approach": "static", "why_not_generic": "specific mark"}
        ],
    }
    results = validate_justifications(
        [{
            "shot_id": "s1",
            "script_quote": winning_script["beats"][0]["line"],
            "truth_fact_id": "t1",
            "treatment_ref": 0,
        }],
        winning_script, product_truths, treatment,
    )
    assert results[0]["passed"] is True, results[0]["violation"]
