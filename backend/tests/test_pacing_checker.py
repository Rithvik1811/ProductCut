"""
Unit tests for the deterministic Pacing-Checker. Pure functions, no fakes
needed -- these are direct arithmetic checks, which is exactly why the spec
mandates code here instead of an LLM call.
"""
from __future__ import annotations

from agents.pacing_checker import MAX_SCORE, MIN_SCORE, check_pacing, check_pacing_all


def _variant(beats: list[dict], target_length_sec: int = 15) -> dict:
    return {"variant_id": "v1", "beats": beats, "target_length_sec": target_length_sec}


PERFECT_BEATS = [
    # First 3 beats (indices 0-2) must land in the 2-3s "early" window; the
    # rest in the 3-5s "later" window. All 3s here satisfies both boundaries.
    {"t_start": 0, "t_end": 3, "line": "Scratched already? Not this one."},  # 3s, 5 words
    {"t_start": 3, "t_end": 6, "line": "Machined hinge plates lock tight."},  # 3s, 5 words
    {"t_start": 6, "t_end": 9, "line": "It grips your phone all day."},  # 3s, 6 words
    {"t_start": 9, "t_end": 12, "line": "No more wobbly desk setups."},  # 3s, 5 words
    {"t_start": 12, "t_end": 15, "line": "Shop the Nulaxy stand today."},  # 3s, 5 words
]


def test_perfectly_paced_script_scores_max_with_no_violations():
    result = check_pacing(_variant(PERFECT_BEATS))
    assert result["violations"] == []
    assert result["pacing_score"] == MAX_SCORE


def test_no_beats_scores_minimum():
    result = check_pacing(_variant([]))
    assert result["pacing_score"] == MIN_SCORE
    assert "no beats present" in result["violations"][0]


def test_total_duration_mismatch_is_flagged():
    short_beats = [{"t_start": 0, "t_end": 3, "line": "Not this one."}]  # only sums to 3s, target 15
    result = check_pacing(_variant(short_beats, target_length_sec=15))
    assert any("expected ~15" in v for v in result["violations"])
    assert result["pacing_score"] < MAX_SCORE


def test_beat_outside_pacing_window_is_flagged():
    # First beat is 8s -- way outside the 2-3s early window.
    beats = [
        {"t_start": 0, "t_end": 8, "line": "Not this one, unlike others."},
        {"t_start": 8, "t_end": 15, "line": "Shop today."},
    ]
    result = check_pacing(_variant(beats))
    assert any("outside early pacing window" in v for v in result["violations"])


def test_line_too_long_for_beat_duration_is_flagged():
    # A 2-second beat can't fit a ~20-word line at 2.3 words/sec.
    beats = [
        {
            "t_start": 0,
            "t_end": 2,
            "line": "This incredibly long hook line has way too many words to possibly fit in two seconds of speech",
        },
        {"t_start": 2, "t_end": 15, "line": "Shop today."},
    ]
    result = check_pacing(_variant(beats))
    assert any("needs ~" in v and "only" in v for v in result["violations"])


def test_score_never_drops_below_minimum_regardless_of_violation_count():
    # Two beats, each violating BOTH the window and speech-pacing checks, plus
    # the total-duration check -- 5 violations against a 5-point scale, to
    # confirm the floor holds rather than going negative.
    terrible_beats = [
        {"t_start": 0, "t_end": 1, "line": "This line has way too many words to possibly fit in one second of speech"},
        {"t_start": 1, "t_end": 2, "line": "Another line packed with entirely too many words for such a short beat"},
    ]
    result = check_pacing(_variant(terrible_beats, target_length_sec=15))
    assert len(result["violations"]) >= 4
    assert result["pacing_score"] == MIN_SCORE


def test_check_pacing_all_batches_by_variant_id():
    v1 = _variant(PERFECT_BEATS)
    v2 = dict(v1)
    v2["variant_id"] = "v2"
    v2["beats"] = []

    results = check_pacing_all([v1, v2])

    assert set(results.keys()) == {"v1", "v2"}
    assert results["v1"]["pacing_score"] == MAX_SCORE
    assert results["v2"]["pacing_score"] == MIN_SCORE
