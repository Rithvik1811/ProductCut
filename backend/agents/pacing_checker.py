"""
Pacing-Checker — deterministic code, NOT an LLM call (Phase 1, Critic Chain).
Spec of record: docs/TECHNICAL_DOCUMENTATION.md §5.4.2.

Validates: (1) beat timestamps sum to the variant's target_length_sec,
(2) each beat falls within the pacing window (2-3s for the first few beats,
3-5s after), (3) each beat's line fits its duration at ~2.3 words/second.
Timing correctness is arithmetic, not judgment -- an LLM would be a strictly
worse choice here, per the spec's own reasoning.

Output shape mirrors Hook-Checker's: {variant_id: {pacing_score, violations}},
a lightweight per-variant result Meta-Critic merges into the real C1
CriticScore. WIRED into graph/build.py, same fan-out/fan-in as
hook_checker.py.

pacing_score formula (this module's own design -- the spec specifies what to
check, not how to score it): starts at 5, loses 1 point per violation found,
floors at 1. Kept on the same 1-5 scale as Hook-Checker for Meta-Critic's
eventual weighted composite; adjust the formula in one place if that
composite math ends up wanting something different.
"""
from __future__ import annotations

from typing import TypedDict

from graph.state import ProductCutState, ScriptVariant

MIN_SCORE = 1
MAX_SCORE = 5

WORDS_PER_SECOND = 2.3
EARLY_BEAT_WINDOW = (2, 3)  # seconds -- "first 2-3 beats"
LATER_BEAT_WINDOW = (3, 5)  # seconds -- "thereafter"
NUM_EARLY_BEATS = 3
TOTAL_DURATION_TOLERANCE_SEC = 1
SPEECH_TIME_TOLERANCE_SEC = 0.5  # slack before a line-too-long violation fires


class PacingResult(TypedDict):
    pacing_score: float
    violations: list[str]


def _check_total_duration(beats: list[dict], target_length_sec: float) -> list[str]:
    if not beats:
        return ["no beats present"]
    total = beats[-1].get("t_end", 0) - beats[0].get("t_start", 0)
    if abs(total - target_length_sec) > TOTAL_DURATION_TOLERANCE_SEC:
        return [f"beats span {total}s, expected ~{target_length_sec}s"]
    return []


def _check_beat_windows(beats: list[dict]) -> list[str]:
    violations = []
    for i, beat in enumerate(beats):
        duration = beat.get("t_end", 0) - beat.get("t_start", 0)
        window = EARLY_BEAT_WINDOW if i < NUM_EARLY_BEATS else LATER_BEAT_WINDOW
        if not (window[0] <= duration <= window[1]):
            phase = "early" if i < NUM_EARLY_BEATS else "later"
            violations.append(
                f"beat {i} ({beat.get('t_start')}-{beat.get('t_end')}s, {duration}s) "
                f"outside {phase} pacing window {window[0]}-{window[1]}s"
            )
    return violations


def _check_speech_pacing(beats: list[dict]) -> list[str]:
    violations = []
    for i, beat in enumerate(beats):
        duration = beat.get("t_end", 0) - beat.get("t_start", 0)
        word_count = len(beat.get("line", "").split())
        required = word_count / WORDS_PER_SECOND
        if required > duration + SPEECH_TIME_TOLERANCE_SEC:
            violations.append(
                f"beat {i} line ({word_count} words) needs ~{required:.1f}s to speak "
                f"at {WORDS_PER_SECOND} words/sec but the beat is only {duration}s"
            )
    return violations


def check_pacing(variant: ScriptVariant) -> PacingResult:
    """Score one script variant's pacing. Pure function, no I/O."""
    beats = variant.get("beats") or []
    target = variant.get("target_length_sec", 0)

    if not beats:
        # A completely missing beat list is the worst case, not "one violation
        # among several" -- don't let the generic 5-minus-count formula below
        # under-punish it.
        return PacingResult(pacing_score=float(MIN_SCORE), violations=["no beats present"])

    violations = _check_total_duration(beats, target)
    violations += _check_beat_windows(beats)
    violations += _check_speech_pacing(beats)

    score = max(MIN_SCORE, MAX_SCORE - len(violations))
    return PacingResult(pacing_score=float(score), violations=violations)


def check_pacing_all(script_variants: list[ScriptVariant]) -> dict[str, PacingResult]:
    """Batch form, matching Hook-Checker's {variant_id: result} shape."""
    return {v["variant_id"]: check_pacing(v) for v in script_variants}


async def pacing_checker_node(state: ProductCutState) -> dict:
    """LangGraph node wrapper: deterministic timing check for each variant. Sync/pure, no I/O — runs in parallel with the other 4 checkers."""
    scores = check_pacing_all(state["script_variants"])
    return {"pacing_scores": scores}
