"""
Hermetic unit tests for agents/budget_gate.py — the Budget Gate (§5.7).

No LLM calls anywhere in this module (§5.7's "deterministic, not generative"
decision), so unlike the Shot-List Agent's tests, none of this needs a fake
DashScope client — every test here is a pure, synchronous computation except
the node-wrapper tests, which exercise `adispatch_custom_event` (needs a real
LangChain run context, same constraint documented in test_merge_validator.py /
test_graph_wiring.py) via a `RunnableLambda` wrapper, mirroring that precedent.
"""
from __future__ import annotations

import pytest
from langchain_core.runnables import RunnableLambda

from agents.budget_gate import (
    DEFAULT_JOB_BUDGET_CAP,
    FLOOR_COST,
    RATE_1080P,
    RATE_720P,
    SPECIFIC_TRUTH_CATEGORIES,
    TRUTH_BONUS,
    W_ROLE,
    W_TYPE,
    _shot_weight,
    allocate_budget,
    budget_gate_node,
)
from agents.shot_list_agent import MIN_SHOTS

_EPS = 1e-6

# ---------------------------------------------------------------------------
# Fixtures / builders (mirrors the Shot shape used by test_shot_list_agent.py).
# ---------------------------------------------------------------------------
TRUTHS = [
    {"truth_id": "t_material", "fact": "matte black anodized aluminum body", "category": "material", "source": "photo_1"},
    {"truth_id": "t_texture", "fact": "brushed grain finish", "category": "texture", "source": "photo_1"},
    {"truth_id": "t_construction", "fact": "dual knurled hinge with brass end caps", "category": "construction_detail", "source": "photo_2"},
    {"truth_id": "t_imperfection", "fact": "faint scuff on the base plate cutout", "category": "imperfection", "source": "photo_1"},
    {"truth_id": "t_color", "fact": "graphite gray colorway", "category": "color", "source": "photo_1"},
    {"truth_id": "t_brief", "fact": "seller says it's a gift-ready item", "category": "brief_or_intake_fact", "source": "photo_1"},
]


def _shot(shot_id, beat_role, shot_type, duration_sec, truth_fact_id="t_brief"):
    """Build a fully-shaped Shot dict (mirrors the Shot-List Agent's assembly)."""
    return {
        "shot_id": shot_id,
        "t_start": 0.0,
        "t_end": duration_sec,
        "beat_role": beat_role,
        "description": f"a {beat_role} shot",
        "shot_type": shot_type,
        "camera_move": "static",
        "framing": "fills_frame",
        "lighting": "soft key light, neutral background",
        "negative_prompt": "warped label, distorted logo",
        "reference_image_id": "photo_1",
        "text_overlay_zone": "none",
        "duration_sec": duration_sec,
        "allocated_budget": 0.0,  # placeholder, exactly like the Shot-List Agent emits
        "voiceover_line": "line",
        "justification": {
            "script_quote": "a real quoted line from the script",
            "truth_fact_id": truth_fact_id,
            "treatment_ref": 0,
        },
        "status": "pending",
        "retry_count": 0,
    }


def _bounds_ok(shot, alloc):
    lo = FLOOR_COST
    hi = shot["duration_sec"] * RATE_1080P
    return lo - _EPS <= alloc <= hi + _EPS


# ===========================================================================
# Normal case: comfortably under cap, higher-weight shots get more.
# ===========================================================================
def test_normal_case_allocations_sum_to_cap_and_favor_high_weight_shots():
    shots = [
        _shot("s_hook", "hook", "hook_hero", 4.0),
        _shot("s_problem", "problem", "lifestyle_context", 4.0),  # lowest weight
        _shot("s_macro", "demo", "macro_detail", 4.0, truth_fact_id="t_material"),  # truth bonus
        _shot("s_cta", "cta", "cta_endcard", 4.0),
    ]
    cap = 1.5  # chosen so every shot's proportional target lands inside [floor, ceiling]

    result = allocate_budget(shots, TRUTHS, cap)

    assert not result.over_cap
    assert result.overage == 0.0
    assert len(result.shots) == 4  # nothing cut

    per_shot = result.ledger["per_shot"]
    assert sum(per_shot.values()) == pytest.approx(result.ledger["spent"])
    assert result.ledger["spent"] == pytest.approx(cap, abs=1e-3)

    # Higher-weight shots (hook / cta / truth-bonus macro_detail) outspend the
    # lowest-weight shot (problem/lifestyle_context) at the SAME duration.
    assert per_shot["s_hook"] > per_shot["s_problem"]
    assert per_shot["s_cta"] > per_shot["s_problem"]
    assert per_shot["s_macro"] > per_shot["s_problem"]

    for shot in shots:
        assert _bounds_ok(shot, per_shot[shot["shot_id"]])


# ===========================================================================
# Tight-but-feasible: cap forces some shots toward the floor, no violations.
# ===========================================================================
def test_tight_but_feasible_forces_low_weight_shot_toward_floor():
    shots = [
        _shot("s_hook", "hook", "hook_hero", 4.0),
        _shot("s_problem", "problem", "lifestyle_context", 4.0),  # lowest weight
        _shot("s_macro", "demo", "macro_detail", 4.0, truth_fact_id="t_material"),
        _shot("s_cta", "cta", "cta_endcard", 4.0),
    ]
    # sum(lo) = 4*0.24 = 0.96, sum(hi) = 4*0.48 = 1.92 -- pick a cap well inside
    # that range but low enough that the proportional target undershoots the
    # floor for the lowest-weight shot, forcing real clamp-and-redistribute.
    cap = 1.0

    result = allocate_budget(shots, TRUTHS, cap)

    assert not result.over_cap
    per_shot = result.ledger["per_shot"]
    assert sum(per_shot.values()) == pytest.approx(result.ledger["spent"])
    assert result.ledger["spent"] == pytest.approx(cap, abs=1e-3)

    for shot in shots:
        alloc = per_shot[shot["shot_id"]]
        assert _bounds_ok(shot, alloc), f"{shot['shot_id']} alloc {alloc} out of bounds"

    # The lowest-weight shot is pinned at (or very near) its floor while at
    # least one higher-weight shot still clears more than the floor.
    assert per_shot["s_problem"] == pytest.approx(FLOOR_COST, abs=1e-3)
    assert per_shot["s_macro"] > per_shot["s_problem"]
    assert per_shot["s_hook"] > per_shot["s_problem"]


# ===========================================================================
# Infeasible-but-above-MIN_SHOTS: cut the lowest-weight shot(s), retry.
# ===========================================================================
def test_infeasible_above_min_shots_cuts_lowest_weight_first_and_retries():
    # 5 shots, strictly ordered weights hook > macro(truth-bonus) > cta > demo2 > problem.
    shots = [
        _shot("s_hook", "hook", "hook_hero", 4.0),                                     # w=1.20*1.15=1.38
        _shot("s_macro", "demo", "macro_detail", 4.0, truth_fact_id="t_material"),     # w=1.00*1.30*1.10=1.43
        _shot("s_cta", "cta", "cta_endcard", 4.0),                                     # w=1.20*1.10=1.32
        _shot("s_demo2", "demo", "lifestyle_context", 4.0),                            # w=1.00*1.10=1.10
        _shot("s_problem", "problem", "lifestyle_context", 4.0),                       # w=0.90*1.10=0.99 (lowest)
    ]
    # sum(lo) at 5 shots = 1.20 (infeasible), at 4 shots = 0.96 (still infeasible
    # against cap below), at 3 shots = 0.72 (feasible) -- forces exactly TWO cuts,
    # in ascending-weight order (problem first, then demo2), landing at MIN_SHOTS
    # without tripping the floor/over_cap case.
    cap = 0.85

    result = allocate_budget(shots, TRUTHS, cap)

    assert not result.over_cap
    assert result.overage == 0.0
    remaining_ids = {s["shot_id"] for s in result.shots}
    assert remaining_ids == {"s_hook", "s_macro", "s_cta"}
    assert len(result.shots) == MIN_SHOTS

    # Cut shots are absent from BOTH the returned shot list and the ledger.
    per_shot = result.ledger["per_shot"]
    assert set(per_shot.keys()) == remaining_ids
    assert "s_problem" not in per_shot
    assert "s_demo2" not in per_shot

    assert sum(per_shot.values()) == pytest.approx(result.ledger["spent"])
    assert result.ledger["spent"] == pytest.approx(cap, abs=1e-3)
    for shot in result.shots:
        assert _bounds_ok(shot, per_shot[shot["shot_id"]])


def test_infeasible_single_cut_still_retries_the_whole_computation():
    """A milder infeasible case needing exactly one cut -- proves the loop
    actually retries the FULL computation on the smaller list (recomputed
    weights/targets/waterfill), not a partial patch."""
    shots = [
        _shot("s_hook", "hook", "hook_hero", 4.0),
        _shot("s_macro", "demo", "macro_detail", 4.0, truth_fact_id="t_material"),
        _shot("s_cta", "cta", "cta_endcard", 4.0),
        _shot("s_hero", "demo", "hero_reframe", 4.0),         # w = 1.0
        _shot("s_problem", "problem", "lifestyle_context", 4.0),  # lowest weight -> cut
    ]
    # sum(lo) at 5 shots = 1.20 (infeasible); at 4 shots = 0.96 (feasible, since
    # cap=1.05 <= sum(hi)=1.92 and >= 0.96).
    cap = 1.05

    result = allocate_budget(shots, TRUTHS, cap)

    assert not result.over_cap
    remaining_ids = {s["shot_id"] for s in result.shots}
    assert remaining_ids == {"s_hook", "s_macro", "s_cta", "s_hero"}
    assert "s_problem" not in result.ledger["per_shot"]
    assert result.ledger["spent"] == pytest.approx(cap, abs=1e-3)


# ===========================================================================
# Floor case: cap so low even MIN_SHOTS' floors don't fit -- accept + flag.
# ===========================================================================
def test_floor_case_never_cuts_below_min_shots_and_flags_overage():
    shots = [
        _shot("s_hook", "hook", "hook_hero", 4.0),
        _shot("s_macro", "demo", "macro_detail", 4.0, truth_fact_id="t_material"),
        _shot("s_cta", "cta", "cta_endcard", 4.0),
    ]
    assert len(shots) == MIN_SHOTS
    # sum(lo) = 3 * 0.24 = 0.72; pick a cap well below that.
    cap = 0.5

    result = allocate_budget(shots, TRUTHS, cap)

    # Never cut below MIN_SHOTS.
    assert len(result.shots) == MIN_SHOTS
    assert {s["shot_id"] for s in result.shots} == {"s_hook", "s_macro", "s_cta"}

    # Overage is accepted and flagged, not hidden.
    assert result.over_cap is True
    expected_floor_total = MIN_SHOTS * FLOOR_COST  # 0.72
    assert result.overage == pytest.approx(expected_floor_total - cap, abs=1e-6)
    assert result.overage > 0

    # Every shot sits at the honest floor cost, never silently pushed under it.
    per_shot = result.ledger["per_shot"]
    for shot_id, alloc in per_shot.items():
        assert alloc == pytest.approx(FLOOR_COST, abs=1e-9)
    assert result.ledger["spent"] == pytest.approx(expected_floor_total, abs=1e-6)


# ===========================================================================
# product_truths lookup: the "specific" categories earn the truth bonus.
# ===========================================================================
def test_specific_truth_categories_constant_matches_spec():
    # "imperfection" deliberately excluded -- Positive-Only Truths fix
    # (docs/BUILD_TASKS.md "Script Quality (CTA Bridge) + Positive-Only
    # Truths..." workstream, Problem 1): a scratch/wear-citing shot must not
    # get a budget priority bump over a color/style/size shot.
    assert SPECIFIC_TRUTH_CATEGORIES == frozenset(
        {"material", "texture", "construction_detail"}
    )


@pytest.mark.parametrize("truth_id", ["t_material", "t_texture", "t_construction"])
def test_shot_weight_gets_truth_bonus_for_specific_categories(truth_id):
    truths_by_id = {t["truth_id"]: t for t in TRUTHS}
    specific_shot = _shot("s1", "demo", "macro_detail", 4.0, truth_fact_id=truth_id)
    generic_shot = _shot("s2", "demo", "macro_detail", 4.0, truth_fact_id="t_color")

    w_specific = _shot_weight(specific_shot, truths_by_id)
    w_generic = _shot_weight(generic_shot, truths_by_id)

    assert w_specific == pytest.approx(w_generic * TRUTH_BONUS)


def test_shot_weight_gets_no_truth_bonus_for_imperfection_category():
    """Positive-Only Truths fix: imperfection is no longer in the "specific"
    bonus set -- a shot grounded in an imperfection fact gets the same
    (unbonused) weight as a generic-category shot."""
    truths_by_id = {t["truth_id"]: t for t in TRUTHS}
    imperfection_shot = _shot("s1", "demo", "macro_detail", 4.0, truth_fact_id="t_imperfection")
    generic_shot = _shot("s2", "demo", "macro_detail", 4.0, truth_fact_id="t_color")

    w_imperfection = _shot_weight(imperfection_shot, truths_by_id)
    w_generic = _shot_weight(generic_shot, truths_by_id)

    assert w_imperfection == pytest.approx(w_generic)


def test_truth_bonus_shot_gets_more_budget_than_otherwise_identical_generic_shot():
    # Two shots identical in every allocation-relevant field except the cited
    # truth's category -- one specific (material), one generic (color).
    shots = [
        _shot("s_specific", "demo", "macro_detail", 4.0, truth_fact_id="t_material"),
        _shot("s_generic", "demo", "macro_detail", 4.0, truth_fact_id="t_color"),
        _shot("s_third", "hook", "hook_hero", 4.0),  # pad to MIN_SHOTS
    ]
    cap = 1.0  # comfortably inside [sum(lo)=0.72, sum(hi)=1.44]

    result = allocate_budget(shots, TRUTHS, cap)

    per_shot = result.ledger["per_shot"]
    assert per_shot["s_specific"] > per_shot["s_generic"]


def test_truth_bonus_ignored_for_brief_or_intake_fact_and_scale_cue_and_missing_id():
    truths_by_id = {t["truth_id"]: t for t in TRUTHS}
    base = _shot("s1", "demo", "macro_detail", 4.0, truth_fact_id="t_brief")
    unknown = _shot("s2", "demo", "macro_detail", 4.0, truth_fact_id="does_not_exist")
    w_base = _shot_weight(base, truths_by_id)
    w_unknown = _shot_weight(unknown, truths_by_id)
    neutral = W_ROLE["demo"] * W_TYPE["macro_detail"]
    assert w_base == pytest.approx(neutral)
    assert w_unknown == pytest.approx(neutral)


# ===========================================================================
# Ledger shape.
# ===========================================================================
def test_ledger_shape_sum_matches_spent_and_all_shot_ids_present():
    shots = [
        _shot("s1", "hook", "hook_hero", 4.0),
        _shot("s2", "problem", "lifestyle_context", 4.0),
        _shot("s3", "cta", "cta_endcard", 4.0),
    ]
    cap = 1.0
    result = allocate_budget(shots, TRUTHS, cap)

    assert sum(result.ledger["per_shot"].values()) == pytest.approx(result.ledger["spent"])
    for shot in shots:
        assert shot["shot_id"] in result.ledger["per_shot"]
    returned_ids = {s["shot_id"] for s in result.shots}
    assert returned_ids == set(result.ledger["per_shot"].keys())
    assert result.ledger["cap"] == cap


def test_ledger_omits_cut_shots_from_both_shot_list_and_per_shot():
    shots = [
        _shot("s_hook", "hook", "hook_hero", 4.0),
        _shot("s_macro", "demo", "macro_detail", 4.0, truth_fact_id="t_material"),
        _shot("s_cta", "cta", "cta_endcard", 4.0),
        _shot("s_hero", "demo", "hero_reframe", 4.0),
        _shot("s_problem", "problem", "lifestyle_context", 4.0),
    ]
    cap = 1.05  # forces exactly one cut (see test above)

    result = allocate_budget(shots, TRUTHS, cap)

    shot_ids = {s["shot_id"] for s in result.shots}
    assert "s_problem" not in shot_ids
    assert "s_problem" not in result.ledger["per_shot"]


def test_allocate_budget_does_not_mutate_caller_inputs():
    shots = [_shot("s1", "hook", "hook_hero", 4.0), _shot("s2", "cta", "cta_endcard", 4.0), _shot("s3", "demo", "macro_detail", 4.0)]
    snapshot = [dict(s) for s in shots]

    allocate_budget(shots, TRUTHS, 1.0)

    assert shots == snapshot  # allocated_budget still 0.0 on the caller's originals


# ===========================================================================
# Node wrapper.
# ===========================================================================
@pytest.mark.asyncio
async def test_node_wrapper_reads_state_and_appends_trace():
    shots = [
        _shot("s1", "hook", "hook_hero", 4.0),
        _shot("s2", "problem", "lifestyle_context", 4.0),
        _shot("s3", "cta", "cta_endcard", 4.0),
    ]
    state = {
        "shot_list": shots,
        "product_truths": TRUTHS,
        "budget_ledger": {"cap": 1.0, "spent": 0.0, "per_shot": {}},
        "reasoning_trace": "prior.",
    }

    node_runnable = RunnableLambda(budget_gate_node)
    out = await node_runnable.ainvoke(state)

    assert out["reasoning_trace"].startswith("prior.")
    assert "[budget_gate]" in out["reasoning_trace"]
    assert out["budget_ledger"]["cap"] == 1.0
    assert len(out["shot_list"]) == 3
    for shot in out["shot_list"]:
        assert shot["allocated_budget"] > 0.0


@pytest.mark.asyncio
async def test_node_wrapper_defaults_cap_when_budget_ledger_absent():
    shots = [
        _shot("s1", "hook", "hook_hero", 4.0),
        _shot("s2", "problem", "lifestyle_context", 4.0),
        _shot("s3", "cta", "cta_endcard", 4.0),
    ]
    state = {
        "shot_list": shots,
        "product_truths": TRUTHS,
        # no budget_ledger key at all
    }

    node_runnable = RunnableLambda(budget_gate_node)
    out = await node_runnable.ainvoke(state)

    assert out["budget_ledger"]["cap"] == DEFAULT_JOB_BUDGET_CAP
    assert "DEFAULT_JOB_BUDGET_CAP" in out["reasoning_trace"] or "[budget_gate]" in out["reasoning_trace"]
    # reasoning_trace defaults to "" + note when absent from state
    assert out["reasoning_trace"].startswith("\n[budget_gate]")


@pytest.mark.asyncio
async def test_node_wrapper_flags_over_cap_in_trace_on_floor_case():
    shots = [
        _shot("s1", "hook", "hook_hero", 4.0),
        _shot("s2", "demo", "macro_detail", 4.0, truth_fact_id="t_material"),
        _shot("s3", "cta", "cta_endcard", 4.0),
    ]
    state = {
        "shot_list": shots,
        "product_truths": TRUTHS,
        "budget_ledger": {"cap": 0.5, "spent": 0.0, "per_shot": {}},
        "reasoning_trace": "",
    }

    node_runnable = RunnableLambda(budget_gate_node)
    out = await node_runnable.ainvoke(state)

    assert "OVER CAP" in out["reasoning_trace"]
    assert len(out["shot_list"]) == MIN_SHOTS
