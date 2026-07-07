"""
Unit tests for the Body-Checker (§5.4.3).

Three layers are covered, all hermetic (no network):
  * the deterministic redundancy PRE-PASS (`redundant_beat_prepass`) — a pure
    function, tested directly with hand-computed Jaccard cases;
  * the `BodyCheckResult` Pydantic gate — extra="forbid", score range, the
    ordered/non-negative/non-self redundant_beat_pairs validator, and the
    StrictBool booleans that a truthy string must NOT satisfy;
  * `_apply_hard_cap` (deterministic score clamp) and the full `check_body`
    path via a monkeypatched sync fake client (`agents.critic_llm.OpenAI`),
    since call_qwen_json builds its own client internally and has no `client=`
    injection param the way Hook-Checker does.
"""
from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from agents.body_checker import (
    BodyCheckResult,
    _apply_hard_cap,
    check_body,
    redundant_beat_prepass,
)
from tests._fakes import FakeSyncOpenAIClient


# ---------------------------------------------------------------------------
# Test data helpers.
# ---------------------------------------------------------------------------


def _beat(line: str) -> dict:
    return {"t_start": 0.0, "t_end": 3.0, "line": line}


def _variant(vid: str, lines: list[str]) -> dict:
    return {
        "variant_id": vid,
        "text": " ".join(lines),
        "framework": "PAS",
        "hook_type": "bold claim",
        "emotional_trigger": "curiosity",
        "grounding_truth_ids": [],
        "beats": [_beat(l) for l in lines],
        "target_length_sec": 15,
    }


def _result_entry(
    vid: str,
    *,
    completion_score: int = 5,
    redundant_beat_pairs: list | None = None,
    promise_payoff_match: bool = True,
    emotional_trigger_landed: bool = True,
    justification: str = "Body pays off the hook with a real mechanism.",
) -> dict:
    return {
        "variant_id": vid,
        "completion_score": completion_score,
        "redundant_beat_pairs": redundant_beat_pairs or [],
        "promise_payoff_match": promise_payoff_match,
        "emotional_trigger_landed": emotional_trigger_landed,
        "justification": justification,
    }


def _payload(entries: list[dict]) -> str:
    return json.dumps({"results": entries})


# ---------------------------------------------------------------------------
# redundant_beat_prepass — pure function, no client.
# ---------------------------------------------------------------------------


def test_prepass_flags_near_identical_body_beats():
    beats = [
        _beat("Your coffee is cold in twelve minutes."),  # hook (index 0)
        _beat("The handmade leather bag is stitched carefully by hand."),  # body 1
        _beat("The handmade leather bag is stitched carefully by hand."),  # body 2 (dup)
        _beat("Tap the link to grab yours today."),  # cta (index 3)
    ]
    assert redundant_beat_prepass(beats) == [[1, 2]]


def test_prepass_does_not_flag_clearly_distinct_body_beats():
    beats = [
        _beat("Your coffee is cold in twelve minutes."),  # hook
        _beat("Handmade leather stitching under a magnifier."),  # body 1
        _beat("Waterproof brass zippers guard against rain."),  # body 2
        _beat("Tap the link to grab yours today."),  # cta
    ]
    assert redundant_beat_prepass(beats) == []


def test_prepass_ignores_hook_and_cta_beats():
    # beats[0] (hook) is lexically identical to beats[1] (body), but the hook is
    # never compared: only the body pair (1, 2) is considered, and 1 vs 2 differ.
    beats = [
        _beat("machined titanium hinge plates lock tight"),  # hook == body 1 text
        _beat("machined titanium hinge plates lock tight"),  # body 1
        _beat("velvet lined pocket protects the screen"),    # body 2 (distinct)
        _beat("shop the stand today"),                       # cta
    ]
    assert redundant_beat_prepass(beats) == []


def test_prepass_empty_and_single_and_short_lists_return_empty():
    assert redundant_beat_prepass([]) == []
    assert redundant_beat_prepass([_beat("only one beat")]) == []
    # n == 3 => hook + exactly one body beat + cta => no within-body pair exists.
    assert redundant_beat_prepass([_beat("hook"), _beat("one body"), _beat("cta")]) == []


def test_prepass_threshold_boundary_is_inclusive():
    # Body beat token sets: A = {apple, banana, cherry, date}, B = {apple, banana}
    # Jaccard = |A∩B| / |A∪B| = 2 / 4 = 0.5 exactly.
    beats = [
        _beat("hook line here"),
        _beat("apple banana cherry date"),
        _beat("apple banana"),
        _beat("cta line here"),
    ]
    # >= threshold, so exactly 0.5 at threshold 0.5 IS flagged.
    assert redundant_beat_prepass(beats, threshold=0.5) == [[1, 2]]
    # A hair above the actual overlap => not flagged.
    assert redundant_beat_prepass(beats, threshold=0.6) == []


def test_prepass_flags_multiple_pairs_across_three_body_beats():
    beats = [
        _beat("hook"),
        _beat("solar panels charge the battery fast"),
        _beat("solar panels charge the battery fast"),
        _beat("solar panels charge the battery fast"),
        _beat("cta"),
    ]
    # Body indices 1,2,3 all identical => pairs (1,2),(1,3),(2,3).
    assert redundant_beat_prepass(beats) == [[1, 2], [1, 3], [2, 3]]


# ---------------------------------------------------------------------------
# BodyCheckResult — the Pydantic gate.
# ---------------------------------------------------------------------------


def test_body_result_valid_construction():
    res = BodyCheckResult(
        completion_score=4,
        redundant_beat_pairs=[[1, 2]],
        promise_payoff_match=True,
        emotional_trigger_landed=False,
        justification="ok",
    )
    assert res.completion_score == 4
    assert res.promise_payoff_match is True
    assert res.emotional_trigger_landed is False


def test_body_result_rejects_unknown_key():
    with pytest.raises(ValidationError):
        BodyCheckResult(
            completion_score=3,
            promise_payoff_match=True,
            emotional_trigger_landed=True,
            justification="ok",
            surprise_field="nope",
        )


@pytest.mark.parametrize("score", [0, 6, -1, 100])
def test_body_result_rejects_out_of_range_score(score):
    with pytest.raises(ValidationError):
        BodyCheckResult(
            completion_score=score,
            promise_payoff_match=True,
            emotional_trigger_landed=True,
            justification="ok",
        )


def test_body_result_normalises_unordered_pair():
    res = BodyCheckResult(
        completion_score=3,
        redundant_beat_pairs=[[2, 1]],
        promise_payoff_match=True,
        emotional_trigger_landed=True,
        justification="ok",
    )
    assert res.redundant_beat_pairs == [[1, 2]]


def test_body_result_rejects_self_pair():
    with pytest.raises(ValidationError):
        BodyCheckResult(
            completion_score=3,
            redundant_beat_pairs=[[1, 1]],
            promise_payoff_match=True,
            emotional_trigger_landed=True,
            justification="ok",
        )


def test_body_result_rejects_negative_index_pair():
    with pytest.raises(ValidationError):
        BodyCheckResult(
            completion_score=3,
            redundant_beat_pairs=[[-1, 2]],
            promise_payoff_match=True,
            emotional_trigger_landed=True,
            justification="ok",
        )


@pytest.mark.parametrize("field", ["promise_payoff_match", "emotional_trigger_landed"])
def test_body_result_strictbool_rejects_truthy_string(field):
    kwargs = dict(
        completion_score=3,
        promise_payoff_match=True,
        emotional_trigger_landed=True,
        justification="ok",
    )
    kwargs[field] = "true"  # a string must NOT be coerced to a real bool
    with pytest.raises(ValidationError):
        BodyCheckResult(**kwargs)


def test_body_result_rejects_empty_justification():
    with pytest.raises(ValidationError):
        BodyCheckResult(
            completion_score=3,
            promise_payoff_match=True,
            emotional_trigger_landed=True,
            justification="",
        )


# ---------------------------------------------------------------------------
# _apply_hard_cap — deterministic score clamp.
# ---------------------------------------------------------------------------


def _res(score, payoff, trigger):
    return BodyCheckResult(
        completion_score=score,
        promise_payoff_match=payoff,
        emotional_trigger_landed=trigger,
        justification="ok",
    )


def test_hard_cap_clamps_when_payoff_false():
    capped = _apply_hard_cap(_res(5, payoff=False, trigger=True))
    assert capped.completion_score == 3


def test_hard_cap_clamps_when_trigger_false():
    capped = _apply_hard_cap(_res(5, payoff=True, trigger=False))
    assert capped.completion_score == 3


def test_hard_cap_leaves_valid_high_score_untouched():
    capped = _apply_hard_cap(_res(5, payoff=True, trigger=True))
    assert capped.completion_score == 5


def test_hard_cap_does_not_raise_score_already_at_or_below_cap():
    # Payoff false but score already 2 => nothing to clamp, left as-is.
    capped = _apply_hard_cap(_res(2, payoff=False, trigger=False))
    assert capped.completion_score == 2


# ---------------------------------------------------------------------------
# check_body — full path via monkeypatched sync fake client.
# ---------------------------------------------------------------------------

VARIANTS = [
    _variant("v1", ["hook one", "body a1", "body a2", "cta one"]),
    _variant("v2", ["hook two", "body b1", "body b2", "cta two"]),
]


def _patch_client(monkeypatch, fake: FakeSyncOpenAIClient) -> None:
    monkeypatch.setattr("agents.critic_llm.OpenAI", lambda *a, **k: fake)


def test_check_body_all_variants_scored_first_try(monkeypatch):
    fake = FakeSyncOpenAIClient([_payload([_result_entry("v1"), _result_entry("v2")])])
    _patch_client(monkeypatch, fake)

    result = check_body(VARIANTS)

    assert fake.call_count == 1
    assert set(result.keys()) == {"v1", "v2"}
    assert result["v1"]["completion_score"] == 5


def test_check_body_empty_variants_returns_empty_without_model_call(monkeypatch):
    def _boom(*_a, **_k):
        raise AssertionError("check_body must not construct a client for empty input")

    monkeypatch.setattr("agents.critic_llm.OpenAI", _boom)

    assert check_body([]) == {}


def test_check_body_missing_variant_id_raises_value_error(monkeypatch):
    # Response only scores v1; v2 is missing => structural mismatch.
    fake = FakeSyncOpenAIClient([_payload([_result_entry("v1")])])
    _patch_client(monkeypatch, fake)

    with pytest.raises(ValueError, match="missing"):
        check_body(VARIANTS)


def test_check_body_extra_variant_id_raises_value_error(monkeypatch):
    entries = [_result_entry("v1"), _result_entry("v2"), _result_entry("v99")]
    fake = FakeSyncOpenAIClient([_payload(entries)])
    _patch_client(monkeypatch, fake)

    with pytest.raises(ValueError, match="extra"):
        check_body(VARIANTS)


def test_check_body_duplicate_variant_id_raises_value_error(monkeypatch):
    entries = [_result_entry("v1"), _result_entry("v1"), _result_entry("v2")]
    fake = FakeSyncOpenAIClient([_payload(entries)])
    _patch_client(monkeypatch, fake)

    with pytest.raises(ValueError, match="duplicate"):
        check_body(VARIANTS)


def test_check_body_applies_hard_cap_end_to_end(monkeypatch):
    entries = [
        _result_entry("v1", completion_score=5, promise_payoff_match=False),
        _result_entry("v2", completion_score=5),
    ]
    fake = FakeSyncOpenAIClient([_payload(entries)])
    _patch_client(monkeypatch, fake)

    result = check_body(VARIANTS)

    # v1 claimed 5 with payoff false => clamped to 3; v2 untouched at 5.
    assert result["v1"]["completion_score"] == 3
    assert result["v2"]["completion_score"] == 5


def test_check_body_normalises_pair_end_to_end(monkeypatch):
    entries = [
        _result_entry("v1", redundant_beat_pairs=[[2, 1]]),
        _result_entry("v2"),
    ]
    fake = FakeSyncOpenAIClient([_payload(entries)])
    _patch_client(monkeypatch, fake)

    result = check_body(VARIANTS)

    assert result["v1"]["redundant_beat_pairs"] == [[1, 2]]


def test_check_body_invalid_entry_raises_value_error(monkeypatch):
    # completion_score out of range for v1 => BodyCheckResult validation fails,
    # surfaced as a ValueError by _validate_results.
    entries = [_result_entry("v1", completion_score=9), _result_entry("v2")]
    fake = FakeSyncOpenAIClient([_payload(entries)])
    _patch_client(monkeypatch, fake)

    with pytest.raises(ValueError, match="v1"):
        check_body(VARIANTS)
