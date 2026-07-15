"""
Hermetic unit tests for agents/merge_validator.py -- the Merge Coherence
Validator (§5.4.7): the deterministic layers (pacing re-check, the one repair
attempt, seam-flag derivation), the `validate_merge_candidate` entry point
(network faked at `agents.critic_llm.OpenAI`, mirroring test_meta_critic.py's
pattern), and the pure `route_after_merge_validation` routing function.

`merge_validator_node` itself is NOT unit-tested here in isolation: it calls
`adispatch_custom_event`, which raises RuntimeError without a real parent run
id (confirmed against the installed langchain_core), so it can only be safely
exercised inside a real, compiled LangGraph run -- see
test_graph_merge_validator.py for that integration coverage.
"""
from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from agents.merge_validator import (
    CoherenceRead,
    _cta_window_intersection,
    check_pronoun_consistency,
    derive_seam_flags,
    repair_pacing,
    route_after_merge_validation,
    run_pacing_recheck,
    validate_merge_candidate,
)
from agents.meta_critic import MergeCandidate, MergedBeat
from tests._fakes import FakeSyncOpenAIClient, make_fake_sync_openai


# ===========================================================================
# Builders.
# ===========================================================================


def _beat(t_start, t_end, line, role, source="v1"):
    return MergedBeat(t_start=t_start, t_end=t_end, line=line, role=role, source_variant_id=source)


def _mk_clean_candidate() -> MergeCandidate:
    """hook(0), body(1), body(2), cta(3) -- already fully pacing-compliant."""
    beats = [
        _beat(0, 3, "hook line", "hook", "v1"),
        _beat(3, 6, "body one", "body", "v2"),
        _beat(6, 9, "body two", "body", "v2"),
        _beat(9, 13, "tap now", "cta", "v3"),
    ]
    return MergeCandidate(
        hook_source_variant_id="v1",
        body_source_variant_id="v2",
        cta_source_variant_id="v3",
        merged_beats=beats,
        merged_text=" ".join(b.line for b in beats),
        target_length_sec=13,
    )


def _mk_needs_repair_candidate() -> MergeCandidate:
    """1 hook + 4 body + 1 cta, mirroring test_meta_critic.py's known-clean
    retime_merged_beats input shape (test_retime_clean_case_no_flags), but
    presented here as the pre-repair MERGED candidate. body1/body2 sit at
    3.5s each in the early (2-3s) window, which run_pacing_recheck rejects;
    repair_pacing re-derives contiguous timestamps via the exact same
    meta_critic.retime_merged_beats call already proven clean for this shape.
    """
    beats = [
        _beat(0, 3, "hook line", "hook", "v1"),
        _beat(3, 6.5, "body one", "body", "v2"),
        _beat(6.5, 10, "body two", "body", "v2"),
        _beat(10, 13.5, "body three", "body", "v2"),
        _beat(13.5, 17, "body four", "body", "v2"),
        _beat(17, 20, "tap now", "cta", "v3"),
    ]
    return MergeCandidate(
        hook_source_variant_id="v1",
        body_source_variant_id="v2",
        cta_source_variant_id="v3",
        merged_beats=beats,
        merged_text=" ".join(b.line for b in beats),
        target_length_sec=20,
    )


def _mk_unfixable_candidate() -> MergeCandidate:
    """1 hook + 1 body + 1 cta: the single body beat has no OTHER body beat to
    donate to/receive slack from, so a too-long body beat cannot be squeezed
    into its window -- the one deterministic repair genuinely fails here."""
    beats = [
        _beat(0, 3, "hook line", "hook", "v1"),
        _beat(3, 19, "body one", "body", "v2"),
        _beat(19, 20, "tap now", "cta", "v3"),
    ]
    return MergeCandidate(
        hook_source_variant_id="v1",
        body_source_variant_id="v2",
        cta_source_variant_id="v3",
        merged_beats=beats,
        merged_text=" ".join(b.line for b in beats),
        target_length_sec=20,
    )


def _mk_candidate_n_body(n_body: int) -> MergeCandidate:
    """hook + n_body body beats + cta, each beat 3s (all early/late-window
    boundary-safe for the seam-derivation tests, which don't run a pacing
    re-check)."""
    beats = [_beat(0, 3, "hook line", "hook", "v1")]
    t = 3.0
    for i in range(n_body):
        beats.append(_beat(t, t + 3, f"body {i}", "body", "v2"))
        t += 3
    beats.append(_beat(t, t + 3, "tap now", "cta", "v3"))
    t += 3
    return MergeCandidate(
        hook_source_variant_id="v1",
        body_source_variant_id="v2",
        cta_source_variant_id="v3",
        merged_beats=beats,
        merged_text=" ".join(b.line for b in beats),
        target_length_sec=t,
    )


def _coherence_payload(**overrides) -> dict:
    base = {
        "coherence_score": 5,
        "voice_consistency": True,
        "promise_payoff_match": True,
        "register_shift_flags": [],
        "justification": "reads as one voice, payoff lands, seams are smooth",
    }
    base.update(overrides)
    return base


class _RaiseOnConstruct:
    """Monkeypatch target for `agents.critic_llm.OpenAI` proving no LLM call
    happens (mirrors test_meta_critic.py's identical helper)."""

    def __init__(self, *_a, **_k):
        raise AssertionError(
            "validate_merge_candidate must NOT construct an OpenAI client here "
            "(coherence read must be skipped when pacing still fails after repair)"
        )


def _sequential_construction_openai_factory(responses: list[str]):
    """Unlike `make_fake_sync_openai` (whose factory returns a FRESH
    FakeSyncOpenAIClient, call_count reset to 0, on every construction --
    correct for tests with exactly one call_qwen_json invocation), this
    advances by CONSTRUCTION rather than by `.create()` call. That distinction
    matters here because `agents.critic_llm._client()` builds a brand-new
    `OpenAI()` on EVERY `call_qwen_json` invocation, so the re-prompt-once path
    (two separate `run_coherence_read` calls, each triggering its own fresh
    client) needs a factory keyed on "which construction is this", not on
    "which `.create()` call is this within one client"."""
    state = {"i": 0}

    def _factory(*_a, **_k):
        content = responses[min(state["i"], len(responses) - 1)]
        state["i"] += 1
        return FakeSyncOpenAIClient([content])

    return _factory


# ===========================================================================
# run_pacing_recheck
# ===========================================================================


def test_run_pacing_recheck_passes_clean_candidate():
    result = run_pacing_recheck(_mk_clean_candidate())
    assert result.passed is True
    assert result.violations == []
    assert result.repaired is False


def test_run_pacing_recheck_fails_needs_repair_candidate():
    result = run_pacing_recheck(_mk_needs_repair_candidate())
    assert result.passed is False
    assert result.violations != []


def test_run_pacing_recheck_fails_unfixable_candidate():
    result = run_pacing_recheck(_mk_unfixable_candidate())
    assert result.passed is False
    assert result.violations != []


# ===========================================================================
# _cta_window_intersection -- the CTA clamp math.
# ===========================================================================


def test_cta_window_intersection_early_index():
    # index<3 -> early(2,3) intersect retime's CTA window (2.5,4.0) -> (2.5, 3.0)
    assert _cta_window_intersection(0) == (2.5, 3.0)
    assert _cta_window_intersection(2) == (2.5, 3.0)


def test_cta_window_intersection_late_index():
    # index>=3 -> late(3,5) intersect (2.5,4.0) -> (3.0, 4.0)
    assert _cta_window_intersection(3) == (3.0, 4.0)
    assert _cta_window_intersection(10) == (3.0, 4.0)


# ===========================================================================
# repair_pacing
# ===========================================================================


def test_repair_pacing_fixes_a_repairable_candidate():
    before = _mk_needs_repair_candidate()
    assert run_pacing_recheck(before).passed is False

    after = repair_pacing(before)
    recheck = run_pacing_recheck(after)

    assert recheck.passed is True, recheck.violations
    assert after.merged_beats[0].t_start == pytest.approx(0.0)
    assert after.merged_beats[-1].t_end == pytest.approx(20.0)
    for i in range(len(after.merged_beats) - 1):
        assert after.merged_beats[i].t_end == pytest.approx(after.merged_beats[i + 1].t_start)


def test_repair_pacing_clamps_cta_into_window_intersection():
    # cta sits at overall index 5 (>=3) -> late window intersection (3.0, 4.0).
    before = _mk_needs_repair_candidate()
    after = repair_pacing(before)
    cta_beat = after.merged_beats[-1]
    cta_dur = cta_beat.t_end - cta_beat.t_start
    assert 3.0 - 1e-6 <= cta_dur <= 4.0 + 1e-6


def test_repair_pacing_cannot_fix_unfixable_single_body_beat():
    before = _mk_unfixable_candidate()
    assert run_pacing_recheck(before).passed is False

    after = repair_pacing(before)
    recheck = run_pacing_recheck(after)
    assert recheck.passed is False  # the one repair genuinely can't fix this shape


# ===========================================================================
# derive_seam_flags
# ===========================================================================


def test_derive_seam_flags_hook_body_seam():
    candidate = _mk_candidate_n_body(2)  # first_body=1, last_body=2
    read = CoherenceRead(**_coherence_payload(voice_consistency=False, register_shift_flags=[1]))
    flags = derive_seam_flags(candidate, read)
    assert len(flags) == 1
    assert flags[0].seam == "hook_body"
    assert flags[0].flagged_beat_index == 1
    assert flags[0].editable_beat_index == 1


def test_derive_seam_flags_body_cta_seam():
    candidate = _mk_candidate_n_body(2)  # first_body=1, last_body=2
    read = CoherenceRead(**_coherence_payload(voice_consistency=False, register_shift_flags=[2]))
    flags = derive_seam_flags(candidate, read)
    assert len(flags) == 1
    assert flags[0].seam == "body_cta"
    assert flags[0].editable_beat_index == 2


def test_derive_seam_flags_interior_body_not_routable():
    candidate = _mk_candidate_n_body(3)  # first_body=1, last_body=3, interior=2
    read = CoherenceRead(**_coherence_payload(voice_consistency=True, register_shift_flags=[2]))
    flags = derive_seam_flags(candidate, read)
    assert flags == []


def test_derive_seam_flags_voice_failure_no_localizable_seam_flags_both():
    candidate = _mk_candidate_n_body(3)  # first_body=1, last_body=3
    read = CoherenceRead(**_coherence_payload(voice_consistency=False, register_shift_flags=[2]))
    flags = derive_seam_flags(candidate, read)
    seams = {f.seam for f in flags}
    assert seams == {"hook_body", "body_cta"}
    hook_body = next(f for f in flags if f.seam == "hook_body")
    body_cta = next(f for f in flags if f.seam == "body_cta")
    assert hook_body.editable_beat_index == 1
    assert body_cta.editable_beat_index == 3


def test_derive_seam_flags_dedupes_per_seam():
    candidate = _mk_candidate_n_body(2)
    read = CoherenceRead(**_coherence_payload(voice_consistency=False, register_shift_flags=[1, 1]))
    flags = derive_seam_flags(candidate, read)
    assert len(flags) == 1


def test_derive_seam_flags_no_body_beats_returns_empty():
    beats = [_beat(0, 3, "hook", "hook", "v1"), _beat(3, 6, "tap", "cta", "v3")]
    candidate = MergeCandidate(
        hook_source_variant_id="v1",
        body_source_variant_id="v1",
        cta_source_variant_id="v3",
        merged_beats=beats,
        merged_text="hook tap",
        target_length_sec=6,
    )
    read = CoherenceRead(**_coherence_payload(voice_consistency=False, register_shift_flags=[]))
    assert derive_seam_flags(candidate, read) == []


# ===========================================================================
# check_pronoun_consistency -- deterministic pronoun-thread seam check
# (video-gen-fidelity story-arc fix).
# ===========================================================================


def _mk_pronoun_mismatch_hook_body_candidate() -> MergeCandidate:
    # Word counts kept low enough to pass check_pacing's 2.3 words/sec window
    # for each beat's duration (same posture as _mk_clean_candidate's own
    # short lines) -- this fixture is testing the pronoun check, not pacing.
    beats = [
        _beat(0, 3, "She grabs it and leaves.", "hook", "v1"),
        _beat(3, 6, "He never looks back again.", "body", "v2"),
        _beat(6, 9, "The seam holds up well.", "body", "v2"),
        _beat(9, 13, "Tap now.", "cta", "v3"),
    ]
    return MergeCandidate(
        hook_source_variant_id="v1", body_source_variant_id="v2", cta_source_variant_id="v3",
        merged_beats=beats, merged_text=" ".join(b.line for b in beats), target_length_sec=13,
    )


def _mk_pronoun_mismatch_body_cta_candidate() -> MergeCandidate:
    beats = [
        _beat(0, 3, "She grabs it and leaves.", "hook", "v1"),
        _beat(3, 6, "The grip finds the seam.", "body", "v2"),
        _beat(6, 9, "Her hand never lets go.", "body", "v2"),
        _beat(9, 13, "Tap now -- he did too.", "cta", "v3"),
    ]
    return MergeCandidate(
        hook_source_variant_id="v1", body_source_variant_id="v2", cta_source_variant_id="v3",
        merged_beats=beats, merged_text=" ".join(b.line for b in beats), target_length_sec=13,
    )


def _mk_pronoun_consistent_candidate() -> MergeCandidate:
    beats = [
        _beat(0, 3, "She grabs it and leaves.", "hook", "v1"),
        _beat(3, 6, "Her grip finds the seam.", "body", "v2"),
        _beat(6, 9, "Her hand never lets go.", "body", "v2"),
        _beat(9, 13, "Tap now -- she already did.", "cta", "v3"),
    ]
    return MergeCandidate(
        hook_source_variant_id="v1", body_source_variant_id="v2", cta_source_variant_id="v3",
        merged_beats=beats, merged_text=" ".join(b.line for b in beats), target_length_sec=13,
    )


def _mk_pronoun_they_mixed_with_she_candidate() -> MergeCandidate:
    # "they" is treated as gender-neutral and must never be flagged against
    # a specific gendered reference to the same person elsewhere.
    beats = [
        _beat(0, 3, "They grab it and leave.", "hook", "v1"),
        _beat(3, 6, "Her grip finds the seam.", "body", "v2"),
        _beat(6, 9, "The seam holds up well.", "body", "v2"),
        _beat(9, 13, "Tap now.", "cta", "v3"),
    ]
    return MergeCandidate(
        hook_source_variant_id="v1", body_source_variant_id="v2", cta_source_variant_id="v3",
        merged_beats=beats, merged_text=" ".join(b.line for b in beats), target_length_sec=13,
    )


def test_pronoun_check_flags_hook_body_mismatch():
    flags = check_pronoun_consistency(_mk_pronoun_mismatch_hook_body_candidate())
    assert len(flags) == 1
    assert flags[0].seam == "hook_body"
    assert flags[0].editable_beat_index == 1  # first body beat
    assert "she" in flags[0].evidence.lower() or "he" in flags[0].evidence.lower()


def test_pronoun_check_flags_body_cta_mismatch():
    flags = check_pronoun_consistency(_mk_pronoun_mismatch_body_cta_candidate())
    assert len(flags) == 1
    assert flags[0].seam == "body_cta"
    assert flags[0].editable_beat_index == 2  # last body beat


def test_pronoun_check_passes_consistent_thread():
    assert check_pronoun_consistency(_mk_pronoun_consistent_candidate()) == []


def test_pronoun_check_never_flags_they_against_a_gendered_pronoun():
    assert check_pronoun_consistency(_mk_pronoun_they_mixed_with_she_candidate()) == []


def test_pronoun_check_no_body_beats_returns_empty():
    beats = [_beat(0, 3, "She grabs it.", "hook", "v1"), _beat(3, 6, "He taps now.", "cta", "v3")]
    candidate = MergeCandidate(
        hook_source_variant_id="v1", body_source_variant_id="v1", cta_source_variant_id="v3",
        merged_beats=beats, merged_text="she he", target_length_sec=6,
    )
    assert check_pronoun_consistency(candidate) == []


def test_pronoun_check_no_pronouns_anywhere_returns_empty():
    assert check_pronoun_consistency(_mk_clean_candidate()) == []


def test_validate_pronoun_mismatch_fails_without_llm_call(monkeypatch):
    """Mirrors test_validate_pacing_fails_even_after_repair_no_llm_call: a
    deterministic pronoun mismatch must skip the coherence read entirely,
    same as a pacing failure does."""
    monkeypatch.setattr("agents.critic_llm.OpenAI", _RaiseOnConstruct)

    result = validate_merge_candidate(_mk_pronoun_mismatch_hook_body_candidate())

    assert result.passed is False
    assert result.failure_kind == "voice_register"
    assert len(result.seam_flags) == 1
    assert result.seam_flags[0].seam == "hook_body"
    assert result.coherence_score is None
    assert "pronoun" in result.justification.lower()


def test_validate_pronoun_consistent_candidate_still_calls_llm(monkeypatch):
    """Regression: a candidate with a real, consistent pronoun thread must
    proceed to the LLM coherence read exactly as before this fix."""
    payload = json.dumps(_coherence_payload())
    monkeypatch.setattr("agents.critic_llm.OpenAI", make_fake_sync_openai([payload]))

    result = validate_merge_candidate(_mk_pronoun_consistent_candidate())

    assert result.passed is True
    assert result.failure_kind is None


# ===========================================================================
# validate_merge_candidate -- LLM-backed paths, fake sync OpenAI client.
# ===========================================================================


def test_validate_clean_pass(monkeypatch):
    payload = json.dumps(_coherence_payload())
    monkeypatch.setattr("agents.critic_llm.OpenAI", make_fake_sync_openai([payload]))

    result = validate_merge_candidate(_mk_clean_candidate())

    assert result.passed is True
    assert result.failure_kind is None
    assert result.pacing_recheck.repaired is False
    assert result.candidate_after_repair is None
    assert result.seam_flags == []


def test_validate_pacing_repair_then_pass(monkeypatch):
    payload = json.dumps(_coherence_payload())
    monkeypatch.setattr("agents.critic_llm.OpenAI", make_fake_sync_openai([payload]))

    result = validate_merge_candidate(_mk_needs_repair_candidate())

    assert result.passed is True
    assert result.failure_kind is None
    assert result.pacing_recheck.passed is True
    assert result.pacing_recheck.repaired is True
    assert result.candidate_after_repair is not None
    assert result.candidate_after_repair["merged_beats"][-1]["t_end"] == pytest.approx(20.0)


def test_validate_pacing_fails_even_after_repair_no_llm_call(monkeypatch):
    monkeypatch.setattr("agents.critic_llm.OpenAI", _RaiseOnConstruct)

    result = validate_merge_candidate(_mk_unfixable_candidate())

    assert result.passed is False
    assert result.failure_kind == "pacing"
    assert result.pacing_recheck.passed is False
    assert result.pacing_recheck.repaired is True
    assert result.coherence_score is None
    assert result.voice_consistency is None
    assert result.promise_payoff_match is None
    assert result.seam_flags == []
    assert result.candidate_after_repair is not None


def test_validate_promise_payoff_failure(monkeypatch):
    payload = json.dumps(_coherence_payload(promise_payoff_match=False, coherence_score=2))
    monkeypatch.setattr("agents.critic_llm.OpenAI", make_fake_sync_openai([payload]))

    result = validate_merge_candidate(_mk_clean_candidate())

    assert result.passed is False
    assert result.failure_kind == "promise_payoff"
    assert result.seam_flags == []


def test_validate_voice_register_failure_derives_seam_flags(monkeypatch):
    # _mk_clean_candidate: hook(0), body(1), body(2), cta(3) -> first_body=1.
    payload = json.dumps(_coherence_payload(register_shift_flags=[1]))
    monkeypatch.setattr("agents.critic_llm.OpenAI", make_fake_sync_openai([payload]))

    result = validate_merge_candidate(_mk_clean_candidate())

    assert result.passed is False
    assert result.failure_kind == "voice_register"
    assert len(result.seam_flags) == 1
    assert result.seam_flags[0].seam == "hook_body"


def test_validate_voice_consistency_false_with_no_flags_still_derives_seams(monkeypatch):
    payload = json.dumps(
        _coherence_payload(voice_consistency=False, coherence_score=2, register_shift_flags=[])
    )
    monkeypatch.setattr("agents.critic_llm.OpenAI", make_fake_sync_openai([payload]))

    result = validate_merge_candidate(_mk_clean_candidate())

    assert result.passed is False
    assert result.failure_kind == "voice_register"
    seams = {f.seam for f in result.seam_flags}
    assert seams == {"hook_body", "body_cta"}


def test_validate_reprompt_once_then_gives_up(monkeypatch):
    monkeypatch.setattr(
        "agents.critic_llm.OpenAI",
        _sequential_construction_openai_factory(["not valid json", "still not valid json"]),
    )

    result = validate_merge_candidate(_mk_clean_candidate())

    assert result.passed is False
    assert result.failure_kind is None
    assert "re-prompt" in result.justification
    assert result.coherence_score is None


def test_validate_reprompt_once_then_succeeds(monkeypatch):
    good = json.dumps(_coherence_payload())
    monkeypatch.setattr(
        "agents.critic_llm.OpenAI",
        _sequential_construction_openai_factory(["not valid json", good]),
    )

    result = validate_merge_candidate(_mk_clean_candidate())

    assert result.passed is True
    assert result.failure_kind is None


def test_validate_reprompt_once_on_validation_error_then_succeeds(monkeypatch):
    # First response is syntactically valid JSON but fails CoherenceRead's own
    # cross-field validator (score<=2 with an all-clear read is incoherent).
    bad = json.dumps(
        {
            "coherence_score": 1,
            "voice_consistency": True,
            "promise_payoff_match": True,
            "register_shift_flags": [],
            "justification": "no reason given",
        }
    )
    good = json.dumps(_coherence_payload())
    monkeypatch.setattr(
        "agents.critic_llm.OpenAI",
        _sequential_construction_openai_factory([bad, good]),
    )

    result = validate_merge_candidate(_mk_clean_candidate())

    assert result.passed is True


# ===========================================================================
# route_after_merge_validation -- pure function, no LLM/config needed.
# ===========================================================================


def _attempt(passed: bool, failure_kind=None) -> dict:
    return {
        "coherence_check": {
            "passed": passed,
            "failure_kind": failure_kind,
        }
    }


def test_route_raises_on_empty_attempts():
    with pytest.raises(ValueError):
        route_after_merge_validation({"merge_attempts": []})


def test_route_passed_finalizes_even_on_second_attempt():
    state = {"merge_attempts": [_attempt(False, "voice_register"), _attempt(True)]}
    assert route_after_merge_validation(state) == "finalize"


def test_route_voice_register_goes_to_copy_editor():
    state = {"merge_attempts": [_attempt(False, "voice_register")]}
    assert route_after_merge_validation(state) == "copy_editor"


def test_route_promise_payoff_goes_to_meta_critic():
    state = {"merge_attempts": [_attempt(False, "promise_payoff")]}
    assert route_after_merge_validation(state) == "meta_critic"


def test_route_pacing_goes_to_fallback():
    state = {"merge_attempts": [_attempt(False, "pacing")]}
    assert route_after_merge_validation(state) == "fallback"


def test_route_no_verdict_goes_to_fallback():
    state = {"merge_attempts": [_attempt(False, None)]}
    assert route_after_merge_validation(state) == "fallback"


def test_route_hard_cap_forces_fallback_regardless_of_failure_kind():
    state = {
        "merge_attempts": [
            _attempt(False, "voice_register"),
            _attempt(False, "voice_register"),
        ]
    }
    assert route_after_merge_validation(state) == "fallback"
