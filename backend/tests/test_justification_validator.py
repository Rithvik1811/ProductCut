"""
Direct unit tests for the shared Justification Validator
(agents/justification_validator.py) -- the function both Treatment Agent
(beat_treatments[] entries) and Shot-List Agent (ShotJustification objects,
RR's caller, built separately) validate through.

Exercises the function in isolation against plain dicts, independent of
either caller's own module, to confirm it's genuinely field-presence-driven
(not hardcoded to one caller's shape) per the Phase 2 interface handoff.
"""
from __future__ import annotations

from agents.justification_validator import (
    VIOLATION_INVALID_BEAT_FUNCTION,
    VIOLATION_QUOTE_MISMATCH,
    VIOLATION_STOPLIST_HIT,
    VIOLATION_TREATMENT_REF_INVALID,
    VIOLATION_UNKNOWN_TRUTH_ID,
    validate_justifications,
)

WINNING_SCRIPT = {
    "text": "Your coffee is cold in 12 minutes. Mine isn't. Double-wall seam keeps it hot. Grab yours today.",
    "beats": [
        {"t_start": 0, "t_end": 3, "line": "Your coffee is cold in 12 minutes. Mine isn't."},
        {"t_start": 3, "t_end": 9, "line": "Double-wall seam keeps it hot."},
        {"t_start": 9, "t_end": 15, "line": "Grab yours today."},
    ],
    "source_variant_ids": ["v1"],
}

TRUTHS = [
    {"truth_id": "t1", "fact": "double-wall stainless seam", "category": "construction_detail", "source": "photo_1"},
    {"truth_id": "t2", "fact": "matte black finish", "category": "color", "source": "photo_1"},
]

TREATMENT = {
    "director_persona": "quiet, tactile, slow-reveal",
    "color_story": "warm neutrals, single matte-black accent",
    "pacing_philosophy": "let the hook breathe, then a quick punch on the CTA",
    "beat_treatments": [
        {
            "beat_index": 0,
            "beat_function": "hook",
            "script_quote": "Your coffee is cold in 12 minutes.",
            "truth_fact_id": "t1",
            "visual_approach": "static macro push on the seam",
            "why_not_generic": "This seam is the one visible proof of the double-wall claim.",
        },
    ],
}


def test_fully_valid_beat_treatment_style_justification_passes():
    justification = {
        "beat_index": 1,
        "beat_function": "demo",
        "script_quote": "Double-wall seam keeps it hot.",
        "truth_fact_id": "t1",
        "visual_approach": "static macro push on the seam",
        "why_not_generic": "This seam is the one visible proof of the double-wall claim.",
    }

    results = validate_justifications([justification], WINNING_SCRIPT, TRUTHS, treatment=None)

    assert results == [{"shot_id_or_beat_index": 1, "passed": True, "violation": None}]


def test_fully_valid_shot_justification_style_passes_without_beat_function():
    # Shot-List Agent's ShotJustification shape: script_quote/truth_fact_id/
    # treatment_ref, keyed by shot_id, never a beat_function field.
    justification = {
        "shot_id": "s3",
        "script_quote": "Grab yours today.",
        "truth_fact_id": "t2",
        "treatment_ref": 0,
    }

    results = validate_justifications([justification], WINNING_SCRIPT, TRUTHS, treatment=TREATMENT)

    assert results == [{"shot_id_or_beat_index": "s3", "passed": True, "violation": None}]


def test_quote_mismatch():
    justification = {
        "beat_index": 0,
        "script_quote": "this text does not appear anywhere in the script",
        "truth_fact_id": "t1",
    }

    results = validate_justifications([justification], WINNING_SCRIPT, TRUTHS, treatment=None)

    assert results[0]["passed"] is False
    assert results[0]["violation"] == VIOLATION_QUOTE_MISMATCH


def test_unknown_truth_id():
    justification = {
        "beat_index": 0,
        "script_quote": "Your coffee is cold in 12 minutes.",
        "truth_fact_id": "t99",
    }

    results = validate_justifications([justification], WINNING_SCRIPT, TRUTHS, treatment=None)

    assert results[0]["passed"] is False
    assert results[0]["violation"] == VIOLATION_UNKNOWN_TRUTH_ID


def test_treatment_ref_invalid():
    justification = {
        "shot_id": "s1",
        "script_quote": "Grab yours today.",
        "truth_fact_id": "t2",
        "treatment_ref": 7,  # no beat_index 7 in TREATMENT["beat_treatments"]
    }

    results = validate_justifications([justification], WINNING_SCRIPT, TRUTHS, treatment=TREATMENT)

    assert results[0]["passed"] is False
    assert results[0]["violation"] == VIOLATION_TREATMENT_REF_INVALID


def test_stoplist_hit_on_banned_word():
    justification = {
        "beat_index": 0,
        "script_quote": "Your coffee is cold in 12 minutes.",
        "truth_fact_id": "t1",
        "why_not_generic": "This is what every product in this category needs to show.",
    }

    results = validate_justifications([justification], WINNING_SCRIPT, TRUTHS, treatment=None)

    assert results[0]["passed"] is False
    assert results[0]["violation"] == VIOLATION_STOPLIST_HIT


def test_stoplist_hit_on_generic_phrase_without_banned_word():
    justification = {
        "shot_id": "s2",
        "script_quote": "Grab yours today.",
        "truth_fact_id": "t1",
        "justification": "Just show the product clearly here.",
    }

    results = validate_justifications([justification], WINNING_SCRIPT, TRUTHS, treatment=None)

    assert results[0]["passed"] is False
    assert results[0]["violation"] == VIOLATION_STOPLIST_HIT


def test_invalid_beat_function():
    justification = {
        "beat_index": 0,
        "beat_function": "climax",  # not one of the 5-value enum
        "script_quote": "Your coffee is cold in 12 minutes.",
        "truth_fact_id": "t1",
    }

    results = validate_justifications([justification], WINNING_SCRIPT, TRUTHS, treatment=None)

    assert results[0]["passed"] is False
    assert results[0]["violation"] == VIOLATION_INVALID_BEAT_FUNCTION


def test_beat_function_check_is_skipped_when_key_absent():
    # Proves the shot-shaped justification (no beat_function key at all) is
    # safely reusable by Shot-List Agent: an out-of-enum value would fail if
    # checked, but since the key isn't present, no such check ever runs.
    justification = {
        "shot_id": "s4",
        "script_quote": "Grab yours today.",
        "truth_fact_id": "t1",
        "treatment_ref": 0,
    }
    assert "beat_function" not in justification

    results = validate_justifications([justification], WINNING_SCRIPT, TRUTHS, treatment=TREATMENT)

    assert results[0]["passed"] is True
    assert results[0]["violation"] is None


def test_results_preserve_input_order_and_identifiers():
    justifications = [
        {"shot_id": "s1", "script_quote": "Grab yours today.", "truth_fact_id": "t1"},
        {"beat_index": 2, "script_quote": "not in the script at all", "truth_fact_id": "t1"},
    ]

    results = validate_justifications(justifications, WINNING_SCRIPT, TRUTHS, treatment=None)

    assert [r["shot_id_or_beat_index"] for r in results] == ["s1", 2]
    assert results[0]["passed"] is True
    assert results[1]["passed"] is False


def test_empty_justifications_list_returns_empty():
    assert validate_justifications([], WINNING_SCRIPT, TRUTHS, treatment=None) == []
