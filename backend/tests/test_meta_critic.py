"""
Hermetic unit tests for agents/meta_critic.py — the deterministic/pure-code
layers (Steps 1, 2, 6), the Pydantic validation gates, and the meta_critic()
entry point's short-circuit + LLM paths.

Everything network-touching is mocked at the boundary: the four LLM-backed
checkers (Body/CTA/Tone/Meta-Critic) all route through
`agents.critic_llm.call_qwen_json`, which builds its own sync `OpenAI()` client
internally, so the ONE monkeypatch target `agents.critic_llm.OpenAI` covers the
Meta-Critic's own LLM call. The deterministic layers make no network call at all.

conftest.py's autouse `_fake_dashscope_env` fixture supplies dummy DASHSCOPE_*
env so `critic_llm._client()` does not raise before the fake client is returned.
"""
from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from agents.meta_critic import (
    _AXIS_WEIGHTS,
    _waterfill,
    compute_composites,
    disqualify,
    meta_critic,
    pick_fallback,
    retime_merged_beats,
    Disqualification,
    MetaCriticLLMOutput,
    MetaCriticResult,
)
from tests._fakes import make_fake_sync_openai


# ===========================================================================
# Small builders.
# ===========================================================================


def _mk_variant(vid: str, target: int = 15) -> dict:
    """A 3-beat variant (hook / body / cta) carrying target_length_sec."""
    return {
        "variant_id": vid,
        "text": f"{vid} full text",
        "framework": "PAS",
        "hook_type": "bold claim",
        "emotional_trigger": "curiosity",
        "grounding_truth_ids": ["t1", "t2"],
        "target_length_sec": target,
        "beats": [
            {"t_start": 0, "t_end": 3, "line": f"{vid} hook"},
            {"t_start": 3, "t_end": 12, "line": f"{vid} body"},
            {"t_start": 12, "t_end": 15, "line": f"{vid} cta"},
        ],
    }


def _scores(vids, never_do=None, overrides=None):
    """Five per-variant score dicts on the shared 1-5 rubric.

    `never_do` = set of vids whose tone score flags a never_do violation.
    `overrides` = {vid: {axis_key: value}} to bump specific axis scores.
    """
    never_do = never_do or set()
    overrides = overrides or {}

    def _o(vid, key, default):
        return overrides.get(vid, {}).get(key, default)

    hook = {v: {"hook_score": _o(v, "hook_score", 4), "justification": "h"} for v in vids}
    pacing = {v: {"pacing_score": _o(v, "pacing_score", 4), "violations": []} for v in vids}
    body = {
        v: {
            "completion_score": _o(v, "completion_score", 4),
            "promise_payoff_match": True,
            "emotional_trigger_landed": True,
            "redundant_beat_pairs": [],
            "justification": "b",
        }
        for v in vids
    }
    cta = {v: {"cta_score": _o(v, "cta_score", 4), "justification": "c"} for v in vids}
    tone = {
        v: {
            "tone_score": _o(v, "tone_score", 4),
            "justification": "t",
            "never_do_violation": v in never_do,
        }
        for v in vids
    }
    return hook, pacing, body, cta, tone


def _llm_out_dict(**overrides) -> dict:
    """A fully-valid MetaCriticLLMOutput payload (python objects) to mutate."""
    base = {
        "leaderboards": [
            {"axis": "hook", "ranked_variant_ids": ["v1", "v2"], "note": "tie by justification"},
        ],
        "audition": {
            "promise_payoff": "body develops the hook claim",
            "hook_body_seam": "seam ok",
            "body_cta_seam": "seam ok",
            "trigger_continuity": "escalates together",
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
                "steelmanned_runner_up": "v2 hook is punchy",
                "trade_off": "lost v2's opener",
                "why_it_holds": "audition passed",
                "named_risk": "hook-body seam",
            },
            {
                "axis": "body",
                "decision": "body from v2",
                "quoted_evidence": "'v2 body' scored 4",
                "steelmanned_runner_up": "v1 body is fine",
                "trade_off": "lost v1's body",
                "why_it_holds": "audition passed",
                "named_risk": "body beat 1",
            },
            {
                "axis": "cta",
                "decision": "cta from v3",
                "quoted_evidence": "'v3 cta' scored 4",
                "steelmanned_runner_up": "v2 cta is clear",
                "trade_off": "lost v2's cta",
                "why_it_holds": "audition passed",
                "named_risk": "body-cta seam",
            },
        ],
        "overall_reasoning": "cross-pollinated from three survivors",
    }
    base.update(overrides)
    return base


class _RaiseOnConstruct:
    """Monkeypatch target for `agents.critic_llm.OpenAI` proving no LLM call happens.

    meta_critic()'s short-circuit paths must never reach call_qwen_json, so the
    client class is never constructed. If it is, this raises loudly.
    """

    def __init__(self, *_a, **_k):
        raise AssertionError(
            "meta_critic short-circuit must NOT construct an OpenAI client (no LLM call)"
        )


# ===========================================================================
# Pydantic model gates.
# ===========================================================================


def test_cap_substitutions_allows_two_rejects_three():
    two = [
        {"axis": "body", "from_variant_id": "v1", "to_variant_id": "v2", "reason": "r1"},
        {"axis": "hook", "from_variant_id": "v3", "to_variant_id": "v4", "reason": "r2"},
    ]
    # exactly 2 passes
    ok = MetaCriticLLMOutput.model_validate(_llm_out_dict(substitutions=two))
    assert len(ok.substitutions) == 2

    three = two + [
        {"axis": "cta", "from_variant_id": "v2", "to_variant_id": "v4", "reason": "r3"},
    ]
    with pytest.raises(ValidationError) as exc:
        MetaCriticLLMOutput.model_validate(_llm_out_dict(substitutions=three))
    assert "capped at 2" in str(exc.value)


def test_metacritic_result_outcome_regex():
    for good in (
        "cross_pollinated",
        "unanimous",
        "single_survivor",
        "fallback_no_compatible_merge",
        "all_excluded_failure",
    ):
        assert MetaCriticResult(outcome=good).outcome == good
    with pytest.raises(ValidationError):
        MetaCriticResult(outcome="totally_made_up")


def test_extra_forbid_on_models():
    # Disqualification has extra="forbid"
    with pytest.raises(ValidationError):
        Disqualification(variant_id="v1", note="n", bogus_field=1)
    # MetaCriticLLMOutput too — a hallucinated top-level key is rejected.
    with pytest.raises(ValidationError):
        MetaCriticLLMOutput.model_validate(_llm_out_dict(hallucinated="x"))


# ===========================================================================
# STEP 1 — disqualify().
# ===========================================================================


def test_disqualify_excludes_violator_preserves_order():
    variants = [_mk_variant("v1"), _mk_variant("v2"), _mk_variant("v3")]
    _, _, _, _, tone = _scores(["v1", "v2", "v3"], never_do={"v2"})
    tone["v2"]["justification"] = "mentions a banned discount"

    survivors, disqualified = disqualify(variants, tone)

    assert survivors == ["v1", "v3"]  # input order preserved, v2 removed
    assert len(disqualified) == 1
    assert disqualified[0].variant_id == "v2"
    assert "never_do violation" in disqualified[0].note
    assert "banned discount" in disqualified[0].note


def test_disqualify_missing_tone_score_raises():
    variants = [_mk_variant("v1")]
    with pytest.raises(ValueError) as exc:
        disqualify(variants, {})  # no tone score for v1
    assert "no tone score for variant 'v1'" in str(exc.value)


# ===========================================================================
# STEP 2 — compute_composites() + pick_fallback().
# ===========================================================================


def test_axis_weights_are_the_documented_constants():
    assert _AXIS_WEIGHTS == {
        "hook": 0.25,
        "pacing": 0.20,
        "completion": 0.20,
        "cta": 0.20,
        "tone": 0.15,
    }
    assert abs(sum(_AXIS_WEIGHTS.values()) - 1.0) < 1e-9


def test_compute_composites_weighted_formula():
    hook, pacing, body, cta, tone = _scores(
        ["v1"],
        overrides={
            "v1": {
                "hook_score": 4,
                "pacing_score": 3,
                "completion_score": 5,
                "cta_score": 2,
                "tone_score": 1,
            }
        },
    )
    composites = compute_composites(["v1"], hook, pacing, body, cta, tone)
    # 0.25*4 + 0.20*3 + 0.20*5 + 0.20*2 + 0.15*1 = 3.15
    expected = (
        _AXIS_WEIGHTS["hook"] * 4
        + _AXIS_WEIGHTS["pacing"] * 3
        + _AXIS_WEIGHTS["completion"] * 5
        + _AXIS_WEIGHTS["cta"] * 2
        + _AXIS_WEIGHTS["tone"] * 1
    )
    assert composites["v1"] == pytest.approx(round(expected, 4))
    assert composites["v1"] == 3.15


def test_compute_composites_works_for_arbitrary_id_list():
    # Not special-cased to disqualification survivors: any id list works.
    hook, pacing, body, cta, tone = _scores(["a", "b", "c"])
    composites = compute_composites(["c", "a"], hook, pacing, body, cta, tone)
    assert set(composites) == {"c", "a"}  # only the ids asked for


def test_compute_composites_missing_axis_raises():
    hook, pacing, body, cta, tone = _scores(["v1"])
    del cta["v1"]["cta_score"]
    with pytest.raises(ValueError) as exc:
        compute_composites(["v1"], hook, pacing, body, cta, tone)
    assert "missing 'cta_score'" in str(exc.value)


def test_pick_fallback_highest_composite():
    composites = {"v1": 3.0, "v2": 4.5, "v3": 2.0}
    _, _, _, _, tone = _scores(["v1", "v2", "v3"])
    hook = {v: {"hook_score": 3} for v in composites}
    assert pick_fallback(composites, tone, hook) == "v2"


def test_pick_fallback_tiebreak_tone_then_hook_then_id():
    # Composites tie -> tone desc breaks it.
    composites = {"a": 3.0, "b": 3.0}
    tone = {"a": {"tone_score": 4}, "b": {"tone_score": 5}}
    hook = {"a": {"hook_score": 5}, "b": {"hook_score": 5}}
    assert pick_fallback(composites, tone, hook) == "b"

    # Composite + tone tie -> hook desc breaks it.
    tone2 = {"a": {"tone_score": 4}, "b": {"tone_score": 4}}
    hook2 = {"a": {"hook_score": 5}, "b": {"hook_score": 4}}
    assert pick_fallback(composites, tone2, hook2) == "a"

    # Everything ties -> lexicographically smallest id wins.
    tone3 = {"a": {"tone_score": 4}, "b": {"tone_score": 4}}
    hook3 = {"a": {"hook_score": 4}, "b": {"hook_score": 4}}
    assert pick_fallback(composites, tone3, hook3) == "a"


def test_pick_fallback_empty_returns_none():
    assert pick_fallback({}, {}, {}) is None


# ===========================================================================
# STEP 6 — _waterfill() and retime_merged_beats().
# ===========================================================================


def test_waterfill_fits_cleanly_redistributes_proportionally():
    # raw sums to 6, budget 9: +3 split proportionally 2:4 -> [3, 6].
    dur, infeasible = _waterfill([2.0, 4.0], [(1.0, 10.0), (1.0, 10.0)], 9.0)
    assert infeasible is False
    assert dur == pytest.approx([3.0, 6.0])
    assert sum(dur) == pytest.approx(9.0)


def test_waterfill_clamps_to_window_then_redistributes_remainder():
    # beat0 wants 5 but is capped at 4; the freed 1s flows to beat1 -> [4, 6].
    dur, infeasible = _waterfill([3.0, 3.0], [(2.0, 4.0), (2.0, 20.0)], 10.0)
    assert infeasible is False
    assert dur == pytest.approx([4.0, 6.0])
    assert sum(dur) == pytest.approx(10.0)


def test_waterfill_infeasible_budget_flagged():
    # Max feasible inside both [2,4] windows is 8; budget 20 cannot fit.
    dur, infeasible = _waterfill([3.0, 3.0], [(2.0, 4.0), (2.0, 4.0)], 20.0)
    assert infeasible is True
    # Residual is still spread so the budget/target is met (surfaced, not hidden).
    assert sum(dur) == pytest.approx(20.0)


def test_retime_clean_case_no_flags():
    hook_beats = [{"t_start": 0, "t_end": 3, "line": "hook line", "source_variant_id": "v1"}]
    body_beats = [
        {"t_start": 3, "t_end": 6.5, "line": "body one", "source_variant_id": "v2"},
        {"t_start": 6.5, "t_end": 10, "line": "body two", "source_variant_id": "v2"},
        {"t_start": 10, "t_end": 13.5, "line": "body three", "source_variant_id": "v2"},
        {"t_start": 13.5, "t_end": 17, "line": "body four", "source_variant_id": "v2"},
    ]
    cta_beat = {"t_start": 17, "t_end": 20, "line": "tap now", "source_variant_id": "v3"}

    merged, flags = retime_merged_beats(hook_beats, body_beats, cta_beat, 20)

    assert flags == []
    assert [b["role"] for b in merged] == ["hook", "body", "body", "body", "body", "cta"]
    assert [b["source_variant_id"] for b in merged] == ["v1", "v2", "v2", "v2", "v2", "v3"]
    # contiguity + exact target end
    for i in range(len(merged) - 1):
        assert merged[i]["t_end"] == pytest.approx(merged[i + 1]["t_start"])
    assert merged[-1]["t_end"] == pytest.approx(20.0)
    assert merged[0]["t_start"] == pytest.approx(0.0)


def test_retime_infeasible_window_flagged():
    # 1 hook + 2 body: both body beats land in the early (2-3s) window, but the
    # body_budget is 9s -> cannot fit inside 2*3s, so it is flagged infeasible.
    hook_beats = [{"t_start": 0, "t_end": 3, "line": "hook", "source_variant_id": "v1"}]
    body_beats = [
        {"t_start": 3, "t_end": 8, "line": "body one", "source_variant_id": "v2"},
        {"t_start": 8, "t_end": 12, "line": "body two", "source_variant_id": "v2"},
    ]
    cta_beat = {"t_start": 12, "t_end": 15, "line": "tap", "source_variant_id": "v3"}

    merged, flags = retime_merged_beats(hook_beats, body_beats, cta_beat, 15)

    kinds = {f["kind"] for f in flags}
    assert "window_infeasible" in kinds
    # Still contiguous and hits target despite the infeasibility (surfaced, not hidden).
    assert merged[-1]["t_end"] == pytest.approx(15.0)


def test_retime_no_body_budget_flagged():
    # Hook + CTA already consume the whole target -> no room for body beats.
    hook_beats = [{"t_start": 0, "t_end": 10, "line": "long hook", "source_variant_id": "v1"}]
    body_beats = [{"t_start": 10, "t_end": 12, "line": "body", "source_variant_id": "v2"}]
    cta_beat = {"t_start": 12, "t_end": 15, "line": "tap", "source_variant_id": "v3"}

    merged, flags = retime_merged_beats(hook_beats, body_beats, cta_beat, 12)

    kinds = {f["kind"] for f in flags}
    assert "no_body_budget" in kinds
    assert merged[-1]["t_end"] == pytest.approx(12.0)


# ===========================================================================
# meta_critic() — short-circuit paths (must make NO LLM call).
# ===========================================================================


def test_empty_variants_all_excluded_failure():
    result = meta_critic([], {}, {}, {}, {}, {})
    assert result.outcome == "all_excluded_failure"
    assert "no variants supplied" in result.notes
    assert result.merge_candidate is None


def test_single_survivor_short_circuits_without_llm(monkeypatch):
    monkeypatch.setattr("agents.critic_llm.OpenAI", _RaiseOnConstruct)
    variants = [_mk_variant("v1"), _mk_variant("v2")]
    hook, pacing, body, cta, tone = _scores(["v1", "v2"], never_do={"v2"})

    result = meta_critic(variants, hook, pacing, body, cta, tone)

    assert result.outcome == "single_survivor"
    assert result.survivor_ids == ["v1"]
    assert result.merge_candidate is not None
    assert result.merge_candidate.hook_source_variant_id == "v1"
    assert result.merge_candidate.body_source_variant_id == "v1"
    assert result.merge_candidate.cta_source_variant_id == "v1"
    assert len(result.disqualified) == 1 and result.disqualified[0].variant_id == "v2"


def test_all_excluded_short_circuits_without_llm(monkeypatch):
    monkeypatch.setattr("agents.critic_llm.OpenAI", _RaiseOnConstruct)
    variants = [_mk_variant("v1"), _mk_variant("v2")]
    hook, pacing, body, cta, tone = _scores(["v1", "v2"], never_do={"v1", "v2"})

    result = meta_critic(variants, hook, pacing, body, cta, tone)

    assert result.outcome == "all_excluded_failure"
    assert result.survivor_ids == []
    assert result.merge_candidate is None
    assert len(result.disqualified) == 2


# ===========================================================================
# meta_critic() — the LLM cross-pollination path.
# ===========================================================================


def test_cross_pollination_valid_llm(monkeypatch):
    payload = json.dumps(_llm_out_dict())  # hook v1 / body v2 / cta v3
    monkeypatch.setattr("agents.critic_llm.OpenAI", make_fake_sync_openai([payload]))
    variants = [_mk_variant(v) for v in ("v1", "v2", "v3", "v4")]
    hook, pacing, body, cta, tone = _scores(["v1", "v2", "v3", "v4"])

    result = meta_critic(variants, hook, pacing, body, cta, tone)

    assert result.outcome == "cross_pollinated"
    assert result.survivor_ids == ["v1", "v2", "v3", "v4"]
    mc = result.merge_candidate
    assert mc is not None
    assert (mc.hook_source_variant_id, mc.body_source_variant_id, mc.cta_source_variant_id) == (
        "v1",
        "v2",
        "v3",
    )
    assert mc.rationale is not None and len(mc.rationale) == 3
    assert mc.merged_beats[-1].t_end == pytest.approx(15.0)


def test_unanimous_when_all_pieces_from_one_variant(monkeypatch):
    payload = json.dumps(
        _llm_out_dict(
            hook_source_variant_id="v1",
            body_source_variant_id="v1",
            cta_source_variant_id="v1",
        )
    )
    monkeypatch.setattr("agents.critic_llm.OpenAI", make_fake_sync_openai([payload]))
    variants = [_mk_variant(v) for v in ("v1", "v2", "v3")]
    hook, pacing, body, cta, tone = _scores(["v1", "v2", "v3"])

    result = meta_critic(variants, hook, pacing, body, cta, tone)
    assert result.outcome == "unanimous"


def test_no_compatible_merge_returns_fallback_unmerged(monkeypatch):
    payload = json.dumps(_llm_out_dict(no_compatible_merge=True))
    monkeypatch.setattr("agents.critic_llm.OpenAI", make_fake_sync_openai([payload]))
    variants = [_mk_variant(v) for v in ("v1", "v2", "v3", "v4")]
    # Make v1 the clear highest-composite -> the fallback target.
    hook, pacing, body, cta, tone = _scores(
        ["v1", "v2", "v3", "v4"],
        overrides={"v1": {"hook_score": 5, "pacing_score": 5, "completion_score": 5, "cta_score": 5, "tone_score": 5}},
    )

    result = meta_critic(variants, hook, pacing, body, cta, tone)

    assert result.outcome == "fallback_no_compatible_merge"
    assert result.fallback_variant_id == "v1"
    mc = result.merge_candidate
    assert mc.hook_source_variant_id == mc.body_source_variant_id == mc.cta_source_variant_id == "v1"


def test_llm_output_missing_required_field_raises(monkeypatch):
    bad = _llm_out_dict()
    del bad["overall_reasoning"]  # required field
    monkeypatch.setattr("agents.critic_llm.OpenAI", make_fake_sync_openai([json.dumps(bad)]))
    variants = [_mk_variant(v) for v in ("v1", "v2", "v3")]
    hook, pacing, body, cta, tone = _scores(["v1", "v2", "v3"])

    with pytest.raises(ValueError) as exc:
        meta_critic(variants, hook, pacing, body, cta, tone)
    assert "failed validation" in str(exc.value)


def test_llm_source_not_a_survivor_raises(monkeypatch):
    payload = json.dumps(_llm_out_dict(hook_source_variant_id="v99"))
    monkeypatch.setattr("agents.critic_llm.OpenAI", make_fake_sync_openai([payload]))
    variants = [_mk_variant(v) for v in ("v1", "v2", "v3")]
    hook, pacing, body, cta, tone = _scores(["v1", "v2", "v3"])

    with pytest.raises(ValueError) as exc:
        meta_critic(variants, hook, pacing, body, cta, tone)
    assert "not a surviving" in str(exc.value)


# ---------------------------------------------------------------------------
# Bounded retry-on-validation-failure (call_qwen_json_validated, video-gen-
# fidelity branch fix) -- integration coverage for the Meta-Critic's own call
# site (agents/meta_critic.py's `meta_critic()`, ~line 1081).
# ---------------------------------------------------------------------------
def test_meta_critic_retries_once_on_validation_failure_and_recovers(monkeypatch):
    """First response is missing a required field (fails MetaCriticLLMOutput
    validation); the re-prompted second response is complete and succeeds."""
    bad = _llm_out_dict()
    del bad["overall_reasoning"]
    good = json.dumps(_llm_out_dict())
    monkeypatch.setattr(
        "agents.critic_llm.OpenAI", make_fake_sync_openai([json.dumps(bad), good])
    )
    variants = [_mk_variant(v) for v in ("v1", "v2", "v3", "v4")]
    hook, pacing, body, cta, tone = _scores(["v1", "v2", "v3", "v4"])

    result = meta_critic(variants, hook, pacing, body, cta, tone)

    assert result.outcome == "cross_pollinated"


def test_meta_critic_raises_clearly_after_max_attempts_exhausted(monkeypatch):
    """Every attempt is missing the same required field -> bounded retry
    (max_attempts=2), never hangs, raises the real ValueError clearly."""
    bad = _llm_out_dict()
    del bad["overall_reasoning"]
    monkeypatch.setattr("agents.critic_llm.OpenAI", make_fake_sync_openai([json.dumps(bad)]))
    variants = [_mk_variant(v) for v in ("v1", "v2", "v3")]
    hook, pacing, body, cta, tone = _scores(["v1", "v2", "v3"])

    with pytest.raises(ValueError) as exc:
        meta_critic(variants, hook, pacing, body, cta, tone)
    assert "failed validation" in str(exc.value)
