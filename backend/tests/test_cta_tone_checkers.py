"""
Unit tests for the CTA-Checker (§5.4.4) and Tone-Checker (§5.4.5).

All hermetic (no network). Covers:
  * `CtaCheckResult` / `ToneCheckResult` Pydantic gates — extra="forbid", score
    range, and (for Tone) the StrictBool `never_do_violation` that a truthy
    string must NOT satisfy;
  * the shared `_validate_results` helper — unknown/duplicate/missing variant_id
    all raise a descriptive ValueError (this module drops nothing; the Meta-Critic
    gate relies on an exact one-per-variant envelope);
  * `check_cta` / `check_tone` full paths via a monkeypatched sync fake client
    (`agents.critic_llm.OpenAI`), including the never_do hard-gate coming through
    as a real bool and a None seller_direction not crashing.
"""
from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from agents.cta_tone_checkers import (
    CtaCheckResult,
    ToneCheckResult,
    _validate_results,
    check_cta,
    check_tone,
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


VARIANTS = [
    _variant("v1", ["Cold coffee again?", "It stays hot.", "Tap to shop today."]),
    _variant("v2", ["Scratched already?", "Machined plates hold.", "Grab yours now."]),
]


def _cta_entry(vid: str, *, cta_score: int = 4, justification: str = "Clear single ask.") -> dict:
    return {"variant_id": vid, "cta_score": cta_score, "justification": justification}


def _tone_entry(
    vid: str,
    *,
    tone_score: int = 4,
    never_do_violation: bool = False,
    justification: str = "On voice for the brief.",
) -> dict:
    return {
        "variant_id": vid,
        "tone_score": tone_score,
        "never_do_violation": never_do_violation,
        "justification": justification,
    }


def _payload(entries: list[dict]) -> str:
    return json.dumps({"results": entries})


def _patch_client(monkeypatch, fake: FakeSyncOpenAIClient) -> None:
    monkeypatch.setattr("agents.critic_llm.OpenAI", lambda *a, **k: fake)


# ---------------------------------------------------------------------------
# CtaCheckResult — Pydantic gate.
# ---------------------------------------------------------------------------


def test_cta_result_valid_construction():
    res = CtaCheckResult(cta_score=5, justification="Tap to shop the set.")
    assert res.cta_score == 5


def test_cta_result_rejects_unknown_key():
    with pytest.raises(ValidationError):
        CtaCheckResult(cta_score=3, justification="ok", extra_field="nope")


@pytest.mark.parametrize("score", [0, 6, -2])
def test_cta_result_rejects_out_of_range_score(score):
    with pytest.raises(ValidationError):
        CtaCheckResult(cta_score=score, justification="ok")


def test_cta_result_rejects_empty_justification():
    with pytest.raises(ValidationError):
        CtaCheckResult(cta_score=3, justification="")


# ---------------------------------------------------------------------------
# ToneCheckResult — Pydantic gate (incl. StrictBool never_do_violation).
# ---------------------------------------------------------------------------


def test_tone_result_valid_construction():
    res = ToneCheckResult(tone_score=2, justification="Off voice.", never_do_violation=True)
    assert res.tone_score == 2
    assert res.never_do_violation is True


def test_tone_result_rejects_unknown_key():
    with pytest.raises(ValidationError):
        ToneCheckResult(
            tone_score=3, justification="ok", never_do_violation=False, mood="calm"
        )


@pytest.mark.parametrize("score", [0, 6, 42])
def test_tone_result_rejects_out_of_range_score(score):
    with pytest.raises(ValidationError):
        ToneCheckResult(tone_score=score, justification="ok", never_do_violation=False)


def test_tone_result_strictbool_rejects_truthy_string():
    with pytest.raises(ValidationError):
        ToneCheckResult(tone_score=3, justification="ok", never_do_violation="true")


# ---------------------------------------------------------------------------
# _validate_results — shared envelope validator.
# ---------------------------------------------------------------------------


def test_validate_results_happy_path():
    raw = {"results": [_cta_entry("v1"), _cta_entry("v2")]}
    out = _validate_results(raw, CtaCheckResult, {"v1", "v2"}, axis="CTA")
    assert set(out.keys()) == {"v1", "v2"}
    assert isinstance(out["v1"], CtaCheckResult)


def test_validate_results_missing_results_list_raises():
    with pytest.raises(ValueError, match="results"):
        _validate_results({"nope": []}, CtaCheckResult, {"v1"}, axis="CTA")


def test_validate_results_unknown_variant_id_raises_extra():
    raw = {"results": [_cta_entry("v1"), _cta_entry("v99")]}
    with pytest.raises(ValueError, match="extra"):
        _validate_results(raw, CtaCheckResult, {"v1"}, axis="CTA")


def test_validate_results_duplicate_variant_id_raises():
    raw = {"results": [_cta_entry("v1"), _cta_entry("v1")]}
    with pytest.raises(ValueError, match="duplicate"):
        _validate_results(raw, CtaCheckResult, {"v1"}, axis="CTA")


def test_validate_results_missing_variant_id_raises_missing():
    raw = {"results": [_cta_entry("v1")]}
    with pytest.raises(ValueError, match="missing"):
        _validate_results(raw, CtaCheckResult, {"v1", "v2"}, axis="CTA")


def test_validate_results_entry_without_variant_id_raises():
    raw = {"results": [{"cta_score": 4, "justification": "no id"}]}
    with pytest.raises(ValueError, match="variant_id"):
        _validate_results(raw, CtaCheckResult, {"v1"}, axis="CTA")


# ---------------------------------------------------------------------------
# check_cta — full path.
# ---------------------------------------------------------------------------


def test_check_cta_scores_all_variants(monkeypatch):
    fake = FakeSyncOpenAIClient([_payload([_cta_entry("v1", cta_score=5), _cta_entry("v2")])])
    _patch_client(monkeypatch, fake)

    result = check_cta(VARIANTS)

    assert fake.call_count == 1
    assert set(result.keys()) == {"v1", "v2"}
    assert result["v1"]["cta_score"] == 5


def test_check_cta_empty_variants_returns_empty_without_model_call(monkeypatch):
    def _boom(*_a, **_k):
        raise AssertionError("check_cta must not construct a client for empty input")

    monkeypatch.setattr("agents.critic_llm.OpenAI", _boom)

    assert check_cta([]) == {}


def test_check_cta_missing_variant_raises(monkeypatch):
    fake = FakeSyncOpenAIClient([_payload([_cta_entry("v1")])])
    _patch_client(monkeypatch, fake)

    with pytest.raises(ValueError, match="missing"):
        check_cta(VARIANTS)


def test_check_cta_duplicate_variant_raises(monkeypatch):
    fake = FakeSyncOpenAIClient([_payload([_cta_entry("v1"), _cta_entry("v1"), _cta_entry("v2")])])
    _patch_client(monkeypatch, fake)

    with pytest.raises(ValueError, match="duplicate"):
        check_cta(VARIANTS)


# ---------------------------------------------------------------------------
# check_tone — full path, incl. never_do hard gate and None seller_direction.
# ---------------------------------------------------------------------------

BRIEF = "A quiet, tactile, handmade candle brand."
SELLER_DIRECTION = {
    "mood_words": ["calm", "sensory", "understated"],
    "never_do": "never mention discounts or sale pricing",
}


def test_check_tone_scores_all_variants(monkeypatch):
    fake = FakeSyncOpenAIClient([_payload([_tone_entry("v1"), _tone_entry("v2")])])
    _patch_client(monkeypatch, fake)

    result = check_tone(BRIEF, SELLER_DIRECTION, VARIANTS)

    assert fake.call_count == 1
    assert set(result.keys()) == {"v1", "v2"}
    assert result["v1"]["never_do_violation"] is False


def test_check_tone_never_do_violation_comes_through_as_real_bool(monkeypatch):
    entries = [
        _tone_entry("v1", tone_score=2, never_do_violation=True, justification="Says '20% off'."),
        _tone_entry("v2"),
    ]
    fake = FakeSyncOpenAIClient([_payload(entries)])
    _patch_client(monkeypatch, fake)

    result = check_tone(BRIEF, SELLER_DIRECTION, VARIANTS)

    assert result["v1"]["never_do_violation"] is True
    assert result["v2"]["never_do_violation"] is False


def test_check_tone_none_seller_direction_does_not_crash(monkeypatch):
    fake = FakeSyncOpenAIClient([_payload([_tone_entry("v1"), _tone_entry("v2")])])
    _patch_client(monkeypatch, fake)

    result = check_tone(BRIEF, None, VARIANTS)

    assert set(result.keys()) == {"v1", "v2"}


def test_check_tone_empty_variants_returns_empty_without_model_call(monkeypatch):
    def _boom(*_a, **_k):
        raise AssertionError("check_tone must not construct a client for empty input")

    monkeypatch.setattr("agents.critic_llm.OpenAI", _boom)

    assert check_tone(BRIEF, SELLER_DIRECTION, []) == {}


def test_check_tone_invalid_entry_raises(monkeypatch):
    # never_do_violation emitted as a string => StrictBool rejects it => ValueError.
    entries = [
        {"variant_id": "v1", "tone_score": 3, "never_do_violation": "true", "justification": "x"},
        _tone_entry("v2"),
    ]
    fake = FakeSyncOpenAIClient([_payload(entries)])
    _patch_client(monkeypatch, fake)

    with pytest.raises(ValueError, match="v1"):
        check_tone(BRIEF, SELLER_DIRECTION, VARIANTS)
