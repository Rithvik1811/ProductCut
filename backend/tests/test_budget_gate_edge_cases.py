"""
Adversarial edge-case tests for agents/budget_gate.py — the Budget Gate (§5.7).

These are ADDITIVE to test_budget_gate.py (the builder's own 18 tests); nothing
here duplicates those. They were written by an independent tester whose goal was
to BREAK the module, not confirm it works — same "never a self-grade" posture the
codebase applies to the Merge Coherence Validator.

No LLM/network anywhere in this module, so every test is a pure synchronous
computation except the node-wrapper tests, which use a RunnableLambda wrapper to
provide the LangChain run context `adispatch_custom_event` needs (mirrors the
precedent in test_budget_gate.py / test_merge_validator.py).

CONFIRMED BUG (see report): `test_duplicate_shot_ids_ledger_must_account_for_all_spend`
is marked `xfail(strict=True)` — it asserts the spec-correct ledger invariant and
currently fails because duplicate shot_ids collide in `per_shot`. It is honestly
asserted (not weakened) and flagged; flip the decorator off to see it red.
"""
from __future__ import annotations

import pytest
from langchain_core.runnables import RunnableLambda

from agents.budget_gate import (
    DEFAULT_JOB_BUDGET_CAP,
    FLOOR_COST,
    RATE_1080P,
    W_ROLE,
    W_TYPE,
    _argmin,
    _shot_weight,
    allocate_budget,
    budget_gate_node,
)
from agents.shot_list_agent import MIN_SHOTS

_EPS = 1e-6

TRUTHS = [
    {"truth_id": "t_material", "fact": "matte black anodized aluminum body", "category": "material", "source": "photo_1"},
    {"truth_id": "t_texture", "fact": "brushed grain finish", "category": "texture", "source": "photo_1"},
    {"truth_id": "t_color", "fact": "graphite gray colorway", "category": "color", "source": "photo_1"},
    {"truth_id": "t_brief", "fact": "seller says it's a gift-ready item", "category": "brief_or_intake_fact", "source": "photo_1"},
]


def _shot(shot_id, beat_role, shot_type, duration_sec, truth_fact_id="t_brief", with_justification=True):
    """Build a fully-shaped Shot dict (mirrors the Shot-List Agent's assembly)."""
    shot = {
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
        "allocated_budget": 0.0,
        "voiceover_line": "line",
        "status": "pending",
        "retry_count": 0,
    }
    if with_justification:
        shot["justification"] = {
            "script_quote": "a real quoted line from the script",
            "truth_fact_id": truth_fact_id,
            "treatment_ref": 0,
        }
    return shot


def _bounds_ok(shot, alloc):
    return FLOOR_COST - _EPS <= alloc <= shot["duration_sec"] * RATE_1080P + _EPS


# ===========================================================================
# FIXED BUG (was xfail): duplicate shot_ids used to silently corrupt the ledger.
# `spent` is now summed from `updated_shots` directly, not from `per_shot.values()`,
# so it stays correct even though `per_shot`'s own breakdown still can't represent
# two allocations under one colliding key (see budget_gate.py's comment at the
# ledger-assembly site for the full reasoning).
# ===========================================================================
def test_duplicate_shot_ids_ledger_must_account_for_all_spend():
    """If two shots share a shot_id, the ledger MUST still account for every dollar
    actually assigned to the returned shots. It currently does not: `per_shot` is
    keyed by shot_id, so the duplicate key's allocation is clobbered and
    `ledger['spent']` (= sum of per_shot.values()) is short by one shot's spend."""
    shots = [
        _shot("dup", "hook", "hook_hero", 4.0),
        _shot("dup", "cta", "cta_endcard", 4.0),          # same shot_id!
        _shot("s3", "demo", "macro_detail", 4.0, truth_fact_id="t_material"),
    ]
    cap = 1.0
    result = allocate_budget(shots, TRUTHS, cap)

    # The gate keeps all 3 shots (nothing is over cap), each with a real allocation.
    assert len(result.shots) == 3
    total_assigned = sum(s["allocated_budget"] for s in result.shots)

    # Spec-correct invariant: the ledger's reported spend equals what was actually
    # handed out to the shots. (Fails today: spent ~= 0.666 while total ~= 1.0.)
    assert result.ledger["spent"] == pytest.approx(total_assigned, abs=1e-6)


# ===========================================================================
# Weight edge cases — must fall back to neutral, never crash.
# ===========================================================================
def test_unknown_beat_role_and_shot_type_fall_back_to_neutral_weight():
    truths_by_id = {t["truth_id"]: t for t in TRUTHS}
    weird = _shot("s1", "not_a_real_role", "not_a_real_type", 4.0)
    # Both unknown -> role_w=1.0, type_w=1.0, no bonus -> exactly 1.0.
    assert _shot_weight(weird, truths_by_id) == pytest.approx(1.0)


def test_unknown_enums_do_not_crash_allocation():
    shots = [
        _shot("s1", "mystery_role", "mystery_type", 4.0),
        _shot("s2", "cta", "cta_endcard", 4.0),
        _shot("s3", "demo", "macro_detail", 4.0, truth_fact_id="t_material"),
    ]
    result = allocate_budget(shots, TRUTHS, 1.0)
    assert len(result.shots) == 3
    for s in result.shots:
        assert _bounds_ok(s, result.ledger["per_shot"][s["shot_id"]])


def test_missing_justification_key_does_not_crash():
    truths_by_id = {t["truth_id"]: t for t in TRUTHS}
    no_just = _shot("s1", "demo", "macro_detail", 4.0, with_justification=False)
    assert "justification" not in no_just
    # `.get("justification", {})` guards this — neutral weight, no crash.
    assert _shot_weight(no_just, truths_by_id) == pytest.approx(
        W_ROLE["demo"] * W_TYPE["macro_detail"]
    )
    shots = [no_just, _shot("s2", "cta", "cta_endcard", 4.0), _shot("s3", "hook", "hook_hero", 4.0)]
    result = allocate_budget(shots, TRUTHS, 1.0)
    assert len(result.shots) == 3


def test_empty_product_truths_gives_no_bonus_and_no_crash():
    truths_by_id: dict = {}
    s = _shot("s1", "demo", "macro_detail", 4.0, truth_fact_id="t_material")
    # No truth table -> the "specific" bonus can never apply.
    assert _shot_weight(s, truths_by_id) == pytest.approx(W_ROLE["demo"] * W_TYPE["macro_detail"])
    shots = [s, _shot("s2", "cta", "cta_endcard", 4.0), _shot("s3", "hook", "hook_hero", 4.0)]
    result = allocate_budget(shots, [], 1.0)   # product_truths = []
    assert len(result.shots) == 3
    assert result.ledger["spent"] == pytest.approx(1.0, abs=1e-3)


def test_nonexistent_truth_fact_id_gets_no_bonus():
    truths_by_id = {t["truth_id"]: t for t in TRUTHS}
    s = _shot("s1", "demo", "macro_detail", 4.0, truth_fact_id="does_not_exist")
    assert _shot_weight(s, truths_by_id) == pytest.approx(W_ROLE["demo"] * W_TYPE["macro_detail"])


# ===========================================================================
# Numeric edge cases.
# ===========================================================================
def test_empty_shot_list_returns_total_empty_result():
    result = allocate_budget([], TRUTHS, 1.0)
    assert result.shots == []
    assert result.ledger == {"cap": 1.0, "spent": 0.0, "per_shot": {}}
    assert result.over_cap is False
    assert result.overage == 0.0


def test_cap_zero_triggers_floor_case_and_flags_over_cap():
    shots = [
        _shot("a", "hook", "hook_hero", 4.0),
        _shot("b", "cta", "cta_endcard", 4.0),
        _shot("c", "demo", "macro_detail", 4.0, truth_fact_id="t_material"),
    ]
    result = allocate_budget(shots, TRUTHS, 0.0)
    assert result.over_cap is True
    assert len(result.shots) == MIN_SHOTS
    # Every shot pinned to the honest floor; overage == the whole floor total.
    assert result.overage == pytest.approx(MIN_SHOTS * FLOOR_COST, abs=1e-6)
    for alloc in result.ledger["per_shot"].values():
        assert alloc == pytest.approx(FLOOR_COST, abs=1e-9)


def test_negative_cap_does_not_crash_and_is_flagged_over_cap():
    shots = [
        _shot("a", "hook", "hook_hero", 4.0),
        _shot("b", "cta", "cta_endcard", 4.0),
        _shot("c", "demo", "macro_detail", 4.0, truth_fact_id="t_material"),
    ]
    result = allocate_budget(shots, TRUTHS, -1.0)
    assert result.over_cap is True
    assert result.overage == pytest.approx(MIN_SHOTS * FLOOR_COST - (-1.0), abs=1e-6)
    assert result.ledger["spent"] == pytest.approx(MIN_SHOTS * FLOOR_COST, abs=1e-6)


def test_single_shot_feasible_allocation():
    # allocate_budget does not require MIN_SHOTS on input; a lone feasible shot works.
    shot = _shot("only", "hook", "hook_hero", 3.0)   # window [0.24, 0.36]
    result = allocate_budget([shot], TRUTHS, 0.30)
    assert len(result.shots) == 1
    assert not result.over_cap
    assert result.ledger["per_shot"]["only"] == pytest.approx(0.30, abs=1e-4)


# ===========================================================================
# Determinism — ties must resolve identically across runs.
# ===========================================================================
def test_argmin_returns_first_index_on_ties():
    assert _argmin([1.0, 1.0, 1.0]) == 0
    assert _argmin([2.0, 1.0, 1.0, 0.5, 0.5]) == 3


def test_reduce_is_deterministic_across_repeated_runs_on_weight_ties():
    def build():
        return [_shot(f"s{i}", "demo", "hero_reframe", 4.0) for i in range(5)]  # all identical weights

    cap = 0.8  # infeasible at 5 (sum lo = 1.20), forces cuts
    kept1 = {s["shot_id"] for s in allocate_budget(build(), TRUTHS, cap).shots}
    kept2 = {s["shot_id"] for s in allocate_budget(build(), TRUTHS, cap).shots}
    kept3 = {s["shot_id"] for s in allocate_budget(build(), TRUTHS, cap).shots}
    assert kept1 == kept2 == kept3
    assert len(kept1) == MIN_SHOTS
    # First-on-ties argmin cuts the lowest indices first -> the last MIN_SHOTS survive.
    assert kept1 == {"s2", "s3", "s4"}


# ===========================================================================
# Larger-than-MAX_SHOTS lists — the module must be generic, not 3-7 hardcoded.
# ===========================================================================
def test_ten_shots_feasible_all_allocated():
    shots = [_shot(f"s{i}", "demo", "hero_reframe", 4.0) for i in range(10)]
    result = allocate_budget(shots, TRUTHS, 3.0)   # sum(lo)=2.4, sum(hi)=4.8 -> feasible
    assert len(result.shots) == 10
    assert not result.over_cap
    assert result.ledger["spent"] == pytest.approx(3.0, abs=1e-3)
    for s in result.shots:
        assert _bounds_ok(s, result.ledger["per_shot"][s["shot_id"]])


def test_many_shots_infeasible_cuts_down_to_exactly_min_shots():
    # 8 shots, cap far below even 3-shot floors -> cut all the way to MIN_SHOTS, then floor.
    shots = [_shot("hook", "hook", "hook_hero", 4.0)] + [
        _shot(f"s{i}", "problem", "lifestyle_context", 4.0) for i in range(7)
    ]
    result = allocate_budget(shots, TRUTHS, 0.30)   # < 3 * FLOOR_COST = 0.72
    assert len(result.shots) == MIN_SHOTS
    assert result.over_cap is True
    # The single highest-weight shot (the hook) is never among the cut ones.
    assert "hook" in {s["shot_id"] for s in result.shots}


# ===========================================================================
# Reduce loop: terminates AND recomputes the WHOLE thing from scratch after a cut.
# ===========================================================================
def test_reduce_loop_recomputes_from_scratch_equals_fresh_allocation_on_survivors():
    """After the reduce loop cuts shots, the surviving allocation must be identical
    to allocating the survivors alone — proving base/weights/targets/waterfill are
    fully recomputed on the smaller list, with no stale pre-cut value leaking in."""
    shots = [
        _shot("s_hook", "hook", "hook_hero", 4.0),
        _shot("s_macro", "demo", "macro_detail", 4.0, truth_fact_id="t_material"),
        _shot("s_cta", "cta", "cta_endcard", 4.0),
        _shot("s_demo2", "demo", "lifestyle_context", 4.0),
        _shot("s_problem", "problem", "lifestyle_context", 4.0),
    ]
    cap = 0.85   # forces two cuts down to MIN_SHOTS (see builder's own test)
    result = allocate_budget(shots, TRUTHS, cap)

    survivor_ids = {s["shot_id"] for s in result.shots}
    survivors = [s for s in shots if s["shot_id"] in survivor_ids]
    fresh = allocate_budget(survivors, TRUTHS, cap)

    assert result.ledger["per_shot"] == fresh.ledger["per_shot"]
    assert result.ledger["spent"] == pytest.approx(fresh.ledger["spent"], abs=1e-9)


def test_exactly_min_shots_gets_one_more_allocation_attempt_before_floor_case():
    """Boundary check for `n <= MIN_SHOTS`: a list that is infeasible at MIN_SHOTS+1
    but FEASIBLE once cut to exactly MIN_SHOTS must succeed (not be prematurely
    treated as the floor/over_cap case)."""
    shots = [
        _shot("s_hook", "hook", "hook_hero", 4.0),
        _shot("s_macro", "demo", "macro_detail", 4.0, truth_fact_id="t_material"),
        _shot("s_cta", "cta", "cta_endcard", 4.0),
        _shot("s_problem", "problem", "lifestyle_context", 4.0),  # lowest weight -> cut
    ]
    # sum(lo) at 4 = 0.96 (infeasible), at 3 = 0.72 (feasible), cap in between.
    cap = 0.80
    result = allocate_budget(shots, TRUTHS, cap)

    assert len(result.shots) == MIN_SHOTS
    assert result.over_cap is False   # feasible at exactly MIN_SHOTS, NOT floor case
    assert result.overage == 0.0
    assert "s_problem" not in result.ledger["per_shot"]
    assert result.ledger["spent"] == pytest.approx(cap, abs=1e-3)


# ===========================================================================
# Floor-case behavior (design tension — flagged, but tested against the code's
# stated contract: at the floor everyone is at rock-bottom FLOOR_COST).
# ===========================================================================
def test_floor_case_flattens_all_shots_to_floor_cost_regardless_of_duration():
    """At the floor case every shot is reported at FLOOR_COST — even a 5s shot whose
    own weighted target and 1080p ceiling are well above the floor, and even the
    high-weight hook. This documents that weighting/duration are deliberately moot
    at the floor (§5.7 step 4: 'every shot already at its cheapest resolution')."""
    shots = [
        _shot("s_hook_long", "hook", "hook_hero", 5.0),            # high weight, long
        _shot("s_cta", "cta", "cta_endcard", 3.0),
        _shot("s_macro_long", "demo", "macro_detail", 5.0, truth_fact_id="t_material"),
    ]
    cap = 0.30   # < 3 * FLOOR_COST = 0.72
    result = allocate_budget(shots, TRUTHS, cap)

    assert result.over_cap is True
    assert len(result.shots) == MIN_SHOTS
    # Identical FLOOR_COST for all three despite different durations & weights.
    for alloc in result.ledger["per_shot"].values():
        assert alloc == pytest.approx(FLOOR_COST, abs=1e-9)
    # The protection §5.7 promises is realized as "never CUT the hook", not
    # "allocate the hook more at the floor" — the hook is still present.
    assert "s_hook_long" in {s["shot_id"] for s in result.shots}


def test_floor_over_cap_flag_not_raised_when_floor_total_equals_cap():
    """The floor-case guard is `spent_floor > cap + _EPS`: exactly at the floor
    total must NOT be flagged over_cap."""
    shots = [_shot(f"s{i}", "demo", "hero_reframe", 4.0) for i in range(3)]
    result = allocate_budget(shots, TRUTHS, MIN_SHOTS * FLOOR_COST)  # cap == 0.72 exactly
    assert result.over_cap is False
    assert result.overage == 0.0


# ===========================================================================
# Non-mutation guarantee — including the nested justification dict.
# ===========================================================================
def test_no_mutation_of_caller_list_dicts_or_nested_justification():
    shots = [
        _shot("s1", "hook", "hook_hero", 4.0, truth_fact_id="t_material"),
        _shot("s2", "cta", "cta_endcard", 4.0),
        _shot("s3", "demo", "macro_detail", 4.0),
    ]
    import copy as _copy
    deep_snapshot = _copy.deepcopy(shots)
    original_ids = [id(s) for s in shots]
    original_just_ids = [id(s["justification"]) for s in shots]

    result = allocate_budget(shots, TRUTHS, 1.0)

    # Caller's list contents unchanged, byte-for-byte (allocated_budget still 0.0).
    assert shots == deep_snapshot
    # Caller's dict objects are the SAME objects (list() copies the list, not dicts),
    # but were not mutated.
    assert [id(s) for s in shots] == original_ids
    assert [id(s["justification"]) for s in shots] == original_just_ids

    # Returned shots are NEW dict objects (never the caller's).
    for orig, new in zip(shots, result.shots):
        assert new is not orig
        assert new["allocated_budget"] > 0.0
        # Documented shallow-copy footgun: the nested justification is SHARED by
        # reference (`{**shot}` is shallow). Harmless here because nothing mutates
        # it, but assert it explicitly so a future in-place edit is caught.
        assert new["justification"] is orig["justification"]


# ===========================================================================
# Ledger arithmetic precision.
# ===========================================================================
def test_ledger_spent_stays_within_tight_tolerance_of_cap_success_case():
    shots = [
        _shot("s_hook", "hook", "hook_hero", 4.0),
        _shot("s_problem", "problem", "lifestyle_context", 4.0),
        _shot("s_macro", "demo", "macro_detail", 4.0, truth_fact_id="t_material"),
        _shot("s_cta", "cta", "cta_endcard", 4.0),
    ]
    for cap in (1.0, 1.1, 1.25, 1.5, 1.75):
        result = allocate_budget(shots, TRUTHS, cap)
        assert not result.over_cap
        # per_shot sum and ledger spent are the same summation -> exact agreement.
        assert sum(result.ledger["per_shot"].values()) == pytest.approx(result.ledger["spent"], abs=1e-12)
        # Waterfill's feasibility threshold (1e-4) plus 6dp rounding bounds the
        # divergence well under 2e-4.
        assert result.ledger["spent"] == pytest.approx(cap, abs=2e-4)


# ===========================================================================
# Node wrapper — cap resolution edge cases.
# ===========================================================================
@pytest.mark.asyncio
async def test_node_real_zero_cap_is_honored_not_treated_as_unset():
    """A genuine cap of 0.0 must be used as the cap (the guard is `is not None`),
    not silently replaced by DEFAULT_JOB_BUDGET_CAP because 0.0 is falsy."""
    shots = [
        _shot("a", "hook", "hook_hero", 4.0),
        _shot("b", "cta", "cta_endcard", 4.0),
        _shot("c", "demo", "macro_detail", 4.0, truth_fact_id="t_material"),
    ]
    state = {
        "shot_list": shots,
        "product_truths": TRUTHS,
        "budget_ledger": {"cap": 0.0, "spent": 0.0, "per_shot": {}},
        "reasoning_trace": "",
    }
    out = await RunnableLambda(budget_gate_node).ainvoke(state)
    assert out["budget_ledger"]["cap"] == 0.0            # NOT DEFAULT_JOB_BUDGET_CAP
    assert DEFAULT_JOB_BUDGET_CAP != 0.0                 # guard against a vacuous assertion
    assert "OVER CAP" in out["reasoning_trace"]          # 0.0 cap -> floor/over-cap


@pytest.mark.asyncio
async def test_node_cap_key_missing_from_present_ledger_falls_back_to_default():
    shots = [
        _shot("a", "hook", "hook_hero", 4.0),
        _shot("b", "cta", "cta_endcard", 4.0),
        _shot("c", "demo", "macro_detail", 4.0),
    ]
    state = {
        "shot_list": shots,
        "product_truths": TRUTHS,
        "budget_ledger": {"spent": 0.0, "per_shot": {}},   # ledger present, NO cap key
    }
    out = await RunnableLambda(budget_gate_node).ainvoke(state)
    assert out["budget_ledger"]["cap"] == DEFAULT_JOB_BUDGET_CAP


@pytest.mark.asyncio
async def test_node_cap_none_in_ledger_falls_back_to_default():
    shots = [
        _shot("a", "hook", "hook_hero", 4.0),
        _shot("b", "cta", "cta_endcard", 4.0),
        _shot("c", "demo", "macro_detail", 4.0),
    ]
    state = {
        "shot_list": shots,
        "product_truths": TRUTHS,
        "budget_ledger": {"cap": None, "spent": 0.0, "per_shot": {}},
    }
    out = await RunnableLambda(budget_gate_node).ainvoke(state)
    assert out["budget_ledger"]["cap"] == DEFAULT_JOB_BUDGET_CAP


@pytest.mark.asyncio
async def test_node_empty_budget_ledger_dict_falls_back_to_default():
    shots = [
        _shot("a", "hook", "hook_hero", 4.0),
        _shot("b", "cta", "cta_endcard", 4.0),
        _shot("c", "demo", "macro_detail", 4.0),
    ]
    state = {
        "shot_list": shots,
        "product_truths": TRUTHS,
        "budget_ledger": {},   # falsy -> `... or {}` path, cap None -> default
    }
    out = await RunnableLambda(budget_gate_node).ainvoke(state)
    assert out["budget_ledger"]["cap"] == DEFAULT_JOB_BUDGET_CAP
