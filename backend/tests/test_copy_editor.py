"""Hermetic unit tests for agents/copy_editor.py (Copy Editor, Section 5.4.8)."""
from __future__ import annotations

import json

import pytest

from agents.copy_editor import (
    ConstraintCheck,
    CopyEditResult,
    _check_constraints,
    copy_edit_seams,
)
from agents.merge_validator import SeamFlag
from agents.meta_critic import MergeCandidate, MergedBeat
from tests._fakes import FakeSyncOpenAIClient


DEFAULT_LINES = [
    "Your coffee is cold in 12 minutes flat today",
    "Many people find their drinks losing warmth too quickly",
    "The double wall vacuum seal holds heat for 40 minutes",
    "Which means your coffee stays hot all morning long",
    "Grab yours today and taste the difference now",
]

REVISED_BEAT1 = "Yours loses warmth too quickly too but mine never"
REVISED_BEAT3 = "So grab yours before it stops staying this hot"
TOO_LONG_REVISION = (
    "one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen"
)


def _make_candidate(lines=None):
    lines = lines or DEFAULT_LINES
    beats = [
        MergedBeat(t_start=0.0, t_end=3.0, line=lines[0], role="hook", source_variant_id="v1"),
        MergedBeat(t_start=3.0, t_end=6.0, line=lines[1], role="body", source_variant_id="v2"),
        MergedBeat(t_start=6.0, t_end=10.0, line=lines[2], role="body", source_variant_id="v2"),
        MergedBeat(t_start=10.0, t_end=13.0, line=lines[3], role="body", source_variant_id="v2"),
        MergedBeat(t_start=13.0, t_end=16.0, line=lines[4], role="cta", source_variant_id="v3"),
    ]
    return MergeCandidate(
        hook_source_variant_id="v1",
        body_source_variant_id="v2",
        cta_source_variant_id="v3",
        merged_beats=beats,
        merged_text=" ".join(lines),
        target_length_sec=16,
    )


def _hook_body_flag():
    return SeamFlag(
        seam="hook_body",
        flagged_beat_index=1,
        editable_beat_index=1,
        evidence="body beat 1 uses distant third person against the hooks direct address",
    )


def _body_cta_flag():
    return SeamFlag(
        seam="body_cta",
        flagged_beat_index=3,
        editable_beat_index=3,
        evidence="body beat 3 tone does not lead energetically into the CTA",
    )


def _patched_with(candidate, index, line):
    beats = list(candidate.merged_beats)
    beats[index] = beats[index].model_copy(update={"line": line})
    return candidate.model_copy(
        update={"merged_beats": beats, "merged_text": " ".join(b.line for b in beats)}
    )


def _response(revisions, justification="Smoothed the seam."):
    return json.dumps({"revised_lines": revisions, "justification": justification})


def _patch_client(monkeypatch, fake):
    monkeypatch.setattr("agents.critic_llm.OpenAI", lambda *a, **k: fake)


def test_constraints_pass_on_clean_edit():
    original = _make_candidate()
    patched = _patched_with(original, 1, REVISED_BEAT1)

    check = _check_constraints(original, patched, [1])

    assert check.passed is True
    assert check.violations == []


def test_constraints_fail_when_non_edited_beat_changed():
    original = _make_candidate()
    patched = _patched_with(original, 1, REVISED_BEAT1)
    patched = _patched_with(patched, 2, "an unrequested change to body beat 2 entirely")

    check = _check_constraints(original, patched, [1])

    assert check.passed is False
    assert any("beat 2" in v for v in check.violations)


def test_constraints_fail_when_word_count_deviates_more_than_10_percent():
    original = _make_candidate()
    patched = _patched_with(original, 1, TOO_LONG_REVISION)

    check = _check_constraints(original, patched, [1])

    assert check.passed is False
    assert any("word count" in v for v in check.violations)


@pytest.mark.parametrize("hook_or_cta_index", [0, 4])
def test_constraints_fail_when_hook_or_cta_index_is_edited(hook_or_cta_index):
    original = _make_candidate()
    check = _check_constraints(original, original, [hook_or_cta_index])

    assert check.passed is False
    assert any(
        f"beat {hook_or_cta_index}" in v and "hook or CTA" in v for v in check.violations
    )


def test_constraints_fail_when_any_timestamp_changes():
    original = _make_candidate()
    patched = _patched_with(original, 1, REVISED_BEAT1)
    beats = list(patched.merged_beats)
    beats[2] = beats[2].model_copy(update={"t_end": beats[2].t_end + 0.5})
    patched = patched.model_copy(update={"merged_beats": beats})

    check = _check_constraints(original, patched, [1])

    assert check.passed is False
    assert any("timestamp changed" in v for v in check.violations)


def test_constraints_fail_when_numeric_token_dropped():
    original = _make_candidate()
    patched = _patched_with(
        original, 2, "The double wall vacuum seal holds heat for many minutes"
    )

    check = _check_constraints(original, patched, [2])

    assert check.passed is False
    assert any("numeric token" in v and "40" in v for v in check.violations)


def test_constraint_check_model_holds_violation_list():
    check = ConstraintCheck(passed=False, violations=["x", "y"])
    assert check.violations == ["x", "y"]


def test_copy_edit_seams_single_seam_success(monkeypatch):
    fake = FakeSyncOpenAIClient([_response({"1": REVISED_BEAT1})])
    _patch_client(monkeypatch, fake)
    candidate = _make_candidate()

    result = copy_edit_seams(
        candidate,
        [_hook_body_flag()],
        "hook is direct-address, body opener is distant third-person",
        ["t1", "t2"],
    )

    assert fake.call_count == 1
    assert isinstance(result, CopyEditResult)
    assert result.constraint_check.passed is True
    assert result.seams_edited == [1]
    assert result.original_seam_text == DEFAULT_LINES[1]
    assert result.revised_seam_text == REVISED_BEAT1
    assert result.patched_candidate.merged_beats[1].line == REVISED_BEAT1
    expected_text = " ".join(
        [DEFAULT_LINES[0], REVISED_BEAT1, DEFAULT_LINES[2], DEFAULT_LINES[3], DEFAULT_LINES[4]]
    )
    assert result.patched_candidate.merged_text == expected_text
    for i in (0, 2, 3, 4):
        assert result.patched_candidate.merged_beats[i] == candidate.merged_beats[i]


def test_copy_edit_seams_two_seams_success(monkeypatch):
    fake = FakeSyncOpenAIClient(
        [_response({"1": REVISED_BEAT1, "3": REVISED_BEAT3}, justification="Smoothed both seams.")]
    )
    _patch_client(monkeypatch, fake)
    candidate = _make_candidate()

    result = copy_edit_seams(
        candidate,
        [_hook_body_flag(), _body_cta_flag()],
        "both seams read as a jarring register shift",
        ["t1"],
    )

    assert fake.call_count == 1
    assert result.constraint_check.passed is True
    assert result.seams_edited == [1, 3]
    assert result.original_seam_text == f"{DEFAULT_LINES[1]}\n{DEFAULT_LINES[3]}"
    assert result.revised_seam_text == f"{REVISED_BEAT1}\n{REVISED_BEAT3}"
    assert result.patched_candidate.merged_beats[1].line == REVISED_BEAT1
    assert result.patched_candidate.merged_beats[3].line == REVISED_BEAT3
    for i in (0, 2, 4):
        assert result.patched_candidate.merged_beats[i] == candidate.merged_beats[i]


def test_copy_edit_seams_constraint_failure_then_successful_retry(monkeypatch):
    bad = _response({"1": TOO_LONG_REVISION}, justification="attempt 1")
    good = _response({"1": REVISED_BEAT1}, justification="attempt 2, corrected")
    fake = FakeSyncOpenAIClient([bad, good])
    _patch_client(monkeypatch, fake)
    candidate = _make_candidate()

    result = copy_edit_seams(
        candidate, [_hook_body_flag()], "register clash at beat 1", ["t1"]
    )

    assert fake.call_count == 2
    assert result.constraint_check.passed is True
    assert result.patched_candidate.merged_beats[1].line == REVISED_BEAT1
    assert result.justification == "attempt 2, corrected"


def test_copy_edit_seams_fails_both_attempts_returns_original_unmodified(monkeypatch):
    bad = _response({"1": TOO_LONG_REVISION}, justification="still too long")
    fake = FakeSyncOpenAIClient([bad, bad])
    _patch_client(monkeypatch, fake)
    candidate = _make_candidate()

    result = copy_edit_seams(
        candidate, [_hook_body_flag()], "register clash at beat 1", ["t1"]
    )

    assert fake.call_count == 2
    assert result.constraint_check.passed is False
    assert result.seams_edited == []
    assert result.patched_candidate.model_dump() == candidate.model_dump()
    assert result.patched_candidate == candidate


def test_copy_edit_seams_rejects_llm_editing_hook_then_succeeds_on_retry(monkeypatch):
    rogue = _response(
        {"1": REVISED_BEAT1, "0": "A completely different hook nobody asked to touch"},
        justification="rogue attempt",
    )
    clean = _response({"1": REVISED_BEAT1}, justification="clean retry")
    fake = FakeSyncOpenAIClient([rogue, clean])
    _patch_client(monkeypatch, fake)
    candidate = _make_candidate()

    result = copy_edit_seams(
        candidate, [_hook_body_flag()], "register clash at beat 1", ["t1"]
    )

    assert fake.call_count == 2
    assert result.constraint_check.passed is True
    assert result.patched_candidate.merged_beats[0].line == DEFAULT_LINES[0]
    assert result.patched_candidate.merged_beats[1].line == REVISED_BEAT1
    assert 0 not in result.seams_edited


def test_copy_edit_seams_rejects_llm_editing_cta_both_attempts_returns_original(monkeypatch):
    rogue = _response(
        {"3": REVISED_BEAT3, "4": "Buy it now before the rewrite disappears"},
        justification="rogue CTA edit",
    )
    fake = FakeSyncOpenAIClient([rogue, rogue])
    _patch_client(monkeypatch, fake)
    candidate = _make_candidate()

    result = copy_edit_seams(
        candidate, [_body_cta_flag()], "register clash at beat 3", ["t1"]
    )

    assert fake.call_count == 2
    assert result.constraint_check.passed is False
    assert any("hook or CTA" in v for v in result.constraint_check.violations)
    assert result.patched_candidate.model_dump() == candidate.model_dump()


def test_copy_edit_seams_raises_when_seam_flags_empty():
    candidate = _make_candidate()
    with pytest.raises(ValueError, match="non-empty"):
        copy_edit_seams(candidate, [], "no seams", [])


def test_copy_edit_seams_raises_when_seam_names_hook_or_cta_as_editable():
    candidate = _make_candidate()
    bad_flag = SeamFlag(
        seam="hook_body",
        flagged_beat_index=0,
        editable_beat_index=0,
        evidence="bogus upstream bug",
    )
    with pytest.raises(ValueError, match="hook/CTA"):
        copy_edit_seams(candidate, [bad_flag], "n/a", [])

