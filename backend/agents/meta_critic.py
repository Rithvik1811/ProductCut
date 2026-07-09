"""
Meta-Critic (§5.4.6) of the Critic Chain.

Spec of record: docs/TECHNICAL_DOCUMENTATION.md §5.4.6. The Meta-Critic is the
*aggregator/reconciler* that sits after the five parallel specialist checkers
(Hook §5.4.1, Pacing §5.4.2, Body §5.4.3, CTA §5.4.4, Tone §5.4.5) and produces a
cross-pollinated **merge CANDIDATE** — the strongest hook + strongest body +
strongest CTA stitched across the surviving variants, with re-derived contiguous
beat timestamps and an ADR-shaped rationale.

Critically (per the doc) the Meta-Critic **no longer has the final word** on
whether its own merge is coherent: this module produces a *candidate + rationale*
only. A separate Merge Coherence Validator (§5.4.7) — NOT built here — independently
re-checks it before anything is written to `winning_script`. Every framing in this
module matches "merge candidate, not yet final".

Architecture note (why it accepts pre-computed score dicts, not the checkers).
The architecture diagram fans each checker's edge INTO the Meta-Critic node — the
Meta-Critic reconciles their outputs, it does not orchestrate the checkers. So this
function takes five pre-computed per-variant score dicts as parameters. (It also
happens to be the only way it *can* work on this branch: Hook-Checker §5.4.1 and
Pacing-Checker §5.4.2 were built by a teammate on an unmerged branch and are not
importable here.)

The deterministic / LLM split (mirrors Body-Checker's "mechanical where mechanical
is possible" posture — §5.4.3 — and the Pacing-Checker's justification for being
pure code):

  * STEP 1  Disqualification gate ............ CODE  (never_do_violation exclusion)
  * STEP 2  Composite table + fallback pick ... CODE  (weighted arithmetic)
  * STEP 3  Axis leaderboards ................. LLM   (reads justifications)
  * STEP 4  Compatibility audition ............ LLM   (promise-payoff / seams / triggers)
  * STEP 5  Swap-down rule (cap 2) ............ LLM   (substitute on offending axis)
  * STEP 6  Re-time the merged beats .......... CODE  (film/VO-editor arithmetic)
  * STEP 7  Merge rationale (ADR-shaped) ...... LLM   (decision/evidence/steelman/...)

The single LLM call (Steps 3/4/5/7) *selects* pieces and *explains*; it never edits
copy and never computes timestamps. Only Step 6's code changes timestamps. Its raw
JSON is validated by a Pydantic model (`extra="forbid"`, same precedent as the
other checkers) BEFORE the code does Step 6's re-timing on top of the selection.

Scope note — what this is NOT (identical posture to body_checker.py):
  * WIRED into the live LangGraph graph (backend/graph/build.py): the fan-in
    join after the 5 parallel checkers, feeding into merge_validator. Was a
    standalone, independently-callable/testable function before the Concept
    Agent existed; that follow-up wiring has since landed.
  * NOT the Merge Coherence Validator (§5.4.7), the Copy Editor (§5.4.8), the
    Concept Agent (§5.3), or any of the five checkers. Meta-Critic only.
  * It does NOT write to `graph.state` (`critic_scores` / `winning_script` /
    `merge_attempts`). It returns a validated result object; wiring that into state
    is a separate task. We import the C1 `ScriptVariant` type for input typing only.

Beat convention (the hook/body/CTA split already implicit in the other checkers):
  hook  = beats[0]        (a single opening beat)
  body  = beats[1:-1]     (the middle)
  CTA   = beats[-1]       (the closing ask)
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Optional

from langchain_core.callbacks.manager import adispatch_custom_event
from langchain_core.runnables import RunnableConfig
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    ValidationError,
    field_validator,
)

from agents.critic_llm import call_qwen_json
from graph.state import ProductCutState, ScriptVariant

logger = logging.getLogger("productcut.critics.meta")


# ---------------------------------------------------------------------------
# Weights (§5.4.6 composite table) and re-timing constants (§Step 6 research).
# ---------------------------------------------------------------------------

# Composite weight per axis. Sums to 1.0. Kept as a module constant so the
# composite math is inspectable and the weights are the single source of truth.
_AXIS_WEIGHTS: dict[str, float] = {
    "hook": 0.25,       # Hook-Checker (§5.4.1)
    "pacing": 0.20,     # Pacing-Checker (§5.4.2)
    "completion": 0.20, # Body-Checker's completion_score (§5.4.3)
    "cta": 0.20,        # CTA-Checker (§5.4.4)
    "tone": 0.15,       # Tone-Checker (§5.4.5)
}

# Spoken rate for the word-fit check (§5.4.2 / Step 6d): ~2.3 words per second.
_WORDS_PER_SEC = 2.3

# CTA beat is locked to this window and ends exactly at target_length_sec (Step 6a).
_CTA_MIN, _CTA_MAX = 2.5, 4.0

# Pacing windows (§5.3): the first 3 beats overall are 2-3s, later beats 3-5s.
_EARLY_MIN, _EARLY_MAX = 2.0, 3.0
_LATE_MIN, _LATE_MAX = 3.0, 5.0
_EARLY_BEAT_COUNT = 3

_EPS = 1e-6


# ===========================================================================
# Pydantic models — output contract + the runtime gate on raw LLM JSON.
# (shot_schema.py / body_checker.py precedent: a TypedDict is a shape hint with
#  NO runtime checking, so anything the model emits gets a real Pydantic gate.)
# ===========================================================================


class AxisLeaderboard(BaseModel):
    """Step 3: one axis's ranked variant order, with the reason ties were broken."""

    model_config = ConfigDict(extra="forbid")

    axis: str = Field(..., pattern="^(hook|completion|cta)$")
    ranked_variant_ids: list[str] = Field(..., min_length=1)
    note: str = Field(..., min_length=1)


class AuditionFinding(BaseModel):
    """Step 4: the compatibility audition on the (provisional or final) assembly."""

    model_config = ConfigDict(extra="forbid")

    promise_payoff: str = Field(..., min_length=1)
    hook_body_seam: str = Field(..., min_length=1)
    body_cta_seam: str = Field(..., min_length=1)
    trigger_continuity: str = Field(..., min_length=1)
    passed: StrictBool
    # Smoothable register shifts are RISKS carried forward (the Copy Editor can
    # polish them downstream) — they are NOT audition failures.
    risks_to_flag_forward: list[str] = Field(default_factory=list)


class Substitution(BaseModel):
    """Step 5: one swap-down on the offending axis (the sequence is capped at 2)."""

    model_config = ConfigDict(extra="forbid")

    axis: str = Field(..., pattern="^(hook|body|cta)$")
    from_variant_id: str = Field(..., min_length=1)
    to_variant_id: str = Field(..., min_length=1)
    reason: str = Field(..., min_length=1)


class AxisRationale(BaseModel):
    """Step 7: the ADR-shaped rationale for one chosen axis (hook / body / cta).

    All six required elements are separate, checkable fields — not one prose blob —
    so a reader (and the downstream validator) can confirm each is present and
    grounded, and 'it scored higher' alone can never satisfy `quoted_evidence`.
    """

    model_config = ConfigDict(extra="forbid")

    axis: str = Field(..., pattern="^(hook|body|cta)$")
    decision: str = Field(..., min_length=1)              # (1) piece + source variant
    quoted_evidence: str = Field(..., min_length=1)       # (2) the actual line/claim + its score
    steelmanned_runner_up: str = Field(..., min_length=1) # (3) runner-up's genuine strength
    trade_off: str = Field(..., min_length=1)             # (4) what was given up
    why_it_holds: str = Field(..., min_length=1)          # (5) cite the Step 4 audition
    named_risk: str = Field(..., min_length=1)            # (6) seam / beat index least confident in


class MetaCriticLLMOutput(BaseModel):
    """The raw LLM judgment (Steps 3/4/5/7), validated before Step 6 runs on it.

    This is the gate between "what the model said" and "what the deterministic
    re-timing trusts". `extra="forbid"` so a hallucinated field can't sneak past;
    the axis-source ids and the one-rationale-per-axis invariant are checked in
    `_validate_llm_output` (they need the surviving-id set, which a field validator
    doesn't have).
    """

    model_config = ConfigDict(extra="forbid")

    leaderboards: list[AxisLeaderboard] = Field(..., min_length=1)
    audition: AuditionFinding
    substitutions: list[Substitution] = Field(default_factory=list)
    hook_source_variant_id: str = Field(..., min_length=1)
    body_source_variant_id: str = Field(..., min_length=1)
    cta_source_variant_id: str = Field(..., min_length=1)
    # True => no compatible cross-variant assembly beats the single best script;
    # the code then returns the fallback variant unmerged (never merge for merging's
    # sake — a merge must beat the fallback or the fallback is what's returned).
    no_compatible_merge: StrictBool
    rationale: list[AxisRationale] = Field(..., min_length=1)
    overall_reasoning: str = Field(..., min_length=1)

    @field_validator("substitutions")
    @classmethod
    def _cap_substitutions(cls, subs):
        if len(subs) > 2:
            raise ValueError(
                f"swap-down is capped at 2 substitutions, got {len(subs)}"
            )
        return subs


class MergedBeat(BaseModel):
    """One re-timed beat in the merge candidate (Step 6 output)."""

    model_config = ConfigDict(extra="forbid")

    t_start: float
    t_end: float
    line: str
    role: str = Field(..., pattern="^(hook|body|cta)$")
    source_variant_id: str


class RetimingFlag(BaseModel):
    """A word-fit / feasibility problem Step 6 could NOT silently resolve.

    Surfaced (not hidden) per this project's error-handling philosophy: a line that
    still overflows its beat after the widen-and-shave adjustment, or a merge whose
    budget is infeasible within the pacing windows, is reported here for the trace
    and for the downstream Merge Coherence Validator (§5.4.7) to see.
    """

    model_config = ConfigDict(extra="forbid")

    beat_index: int
    kind: str = Field(..., pattern="^(word_overflow|window_infeasible|no_body_budget)$")
    overflow_sec: float = 0.0
    detail: str


class MergeCandidate(BaseModel):
    """The Meta-Critic's product: a stitched candidate, its sources, and rationale.

    NOT a finalized `winning_script` — the Merge Coherence Validator (§5.4.7) has
    the final word. `rationale`/`audition`/`substitutions` are optional because the
    single-survivor short-circuit and the fallback outcome produce a candidate with
    no negotiation to report.
    """

    model_config = ConfigDict(extra="forbid")

    hook_source_variant_id: str
    body_source_variant_id: str
    cta_source_variant_id: str
    merged_beats: list[MergedBeat]
    merged_text: str
    target_length_sec: int
    rationale: Optional[list[AxisRationale]] = None
    overall_reasoning: Optional[str] = None
    audition: Optional[AuditionFinding] = None
    substitutions: list[Substitution] = Field(default_factory=list)
    retiming_flags: list[RetimingFlag] = Field(default_factory=list)


class Disqualification(BaseModel):
    """Step 1: a variant excluded by a Tone-Checker never_do violation."""

    model_config = ConfigDict(extra="forbid")

    variant_id: str
    note: str


class MetaCriticResult(BaseModel):
    """The full Meta-Critic result (§5.4.6 output contract).

    `outcome` enumerates every terminal state:
      * cross_pollinated ............ merge draws pieces from >1 surviving variant
      * unanimous ................... all three winning pieces came from one variant
                                       (a valid outcome — 'unanimous, not cross-pollinated')
      * single_survivor ............. exactly one variant survived DQ; it wins
                                       un-negotiated (short-circuit, no merge/audition)
      * fallback_no_compatible_merge  no assembly beat the fallback within 2 swaps;
                                       the highest-composite variant is returned unmerged
      * all_excluded_failure ........ zero variants survived DQ (pipeline failure —
                                       NOT papered over by un-excluding a violator)
    """

    model_config = ConfigDict(extra="forbid")

    outcome: str = Field(
        ...,
        pattern="^(cross_pollinated|unanimous|single_survivor"
        "|fallback_no_compatible_merge|all_excluded_failure)$",
    )
    merge_candidate: Optional[MergeCandidate] = None
    disqualified: list[Disqualification] = Field(default_factory=list)
    composite_scores: dict[str, float] = Field(default_factory=dict)
    fallback_variant_id: Optional[str] = None
    survivor_ids: list[str] = Field(default_factory=list)
    notes: str = ""


# ===========================================================================
# STEP 1 — Disqualification gate (deterministic, CODE).
# ===========================================================================


def disqualify(
    variants: list[ScriptVariant],
    tone_scores: dict[str, dict],
) -> tuple[list[str], list[Disqualification]]:
    """Exclude any variant whose Tone-Checker output has never_do_violation=true.

    Real compliance clearance disqualifies the whole spot, not just the offending
    scene, so an excluded variant is removed entirely — its beats never become
    eligible for piece-borrowing downstream (only `survivor_ids` reach the LLM).

    The recorded note carries the violated-rule evidence: the Tone-Checker's own
    justification (which names the never_do rule + the offending text).

    Returns:
        (survivor_ids in input order, disqualified list in input order).
    """
    survivors: list[str] = []
    disqualified: list[Disqualification] = []
    for v in variants:
        vid = v["variant_id"]
        tone = tone_scores.get(vid)
        if tone is None:
            raise ValueError(f"Meta-Critic: no tone score for variant {vid!r}")
        if tone.get("never_do_violation") is True:
            disqualified.append(
                Disqualification(
                    variant_id=vid,
                    note=(
                        "never_do violation (Tone-Checker §5.4.5): "
                        + str(tone.get("justification", "")).strip()
                    ),
                )
            )
        else:
            survivors.append(vid)
    return survivors, disqualified


# ===========================================================================
# STEP 2 — Composite table + fallback pick (deterministic, CODE).
# ===========================================================================


def _axis_score(scores: dict[str, dict], vid: str, key: str, axis: str) -> float:
    entry = scores.get(vid)
    if entry is None:
        raise ValueError(f"Meta-Critic: no {axis} score for variant {vid!r}")
    if key not in entry:
        raise ValueError(
            f"Meta-Critic: {axis} score for {vid!r} missing {key!r} (got {list(entry)})"
        )
    return float(entry[key])


def compute_composites(
    survivor_ids: list[str],
    hook_scores: dict[str, dict],
    pacing_scores: dict[str, dict],
    body_scores: dict[str, dict],
    cta_scores: dict[str, dict],
    tone_scores: dict[str, dict],
) -> dict[str, float]:
    """Weighted composite per surviving variant (§5.4.6 table).

    composite = 0.25*hook + 0.20*pacing + 0.20*completion + 0.20*cta + 0.15*tone
    (all axes on the checkers' shared 1-5 rubric scale, so the composite is too).
    Excluded variants are not scored at all (they are absent from `survivor_ids`).
    """
    composites: dict[str, float] = {}
    for vid in survivor_ids:
        composite = (
            _AXIS_WEIGHTS["hook"] * _axis_score(hook_scores, vid, "hook_score", "hook")
            + _AXIS_WEIGHTS["pacing"]
            * _axis_score(pacing_scores, vid, "pacing_score", "pacing")
            + _AXIS_WEIGHTS["completion"]
            * _axis_score(body_scores, vid, "completion_score", "body")
            + _AXIS_WEIGHTS["cta"] * _axis_score(cta_scores, vid, "cta_score", "cta")
            + _AXIS_WEIGHTS["tone"] * _axis_score(tone_scores, vid, "tone_score", "tone")
        )
        composites[vid] = round(composite, 4)
    return composites


def pick_fallback(
    composites: dict[str, float],
    tone_scores: dict[str, dict],
    hook_scores: dict[str, dict],
) -> Optional[str]:
    """The highest-composite surviving variant — the pipeline's revert target.

    Ties broken deterministically (composite desc, then tone desc, then hook desc,
    then variant_id asc) so the same inputs always name the same fallback. Naming it
    is all this node owes; the actual revert mechanism is the Merge Coherence
    Validator's job (§5.4.7).
    """
    if not composites:
        return None

    def _key(vid: str):
        return (
            composites[vid],
            float(tone_scores.get(vid, {}).get("tone_score", 0)),
            float(hook_scores.get(vid, {}).get("hook_score", 0)),
            # variant_id ascending as the final tiebreak -> negate via reverse sort
            # handled below by sorting ids ascending first.
        )

    # Sort ids ascending, then pick max by (composite, tone, hook) — Python's max
    # returns the FIRST max on ties, and iterating an ascending-id list makes that
    # the lexicographically-smallest id, giving a fully deterministic result.
    ordered = sorted(composites.keys())
    return max(ordered, key=_key)


# ===========================================================================
# STEP 6 — Re-time the merged beats (deterministic, CODE).
# ===========================================================================


def _word_count(line: str) -> int:
    return len(line.split())


def _window_for_overall_index(idx: int) -> tuple[float, float]:
    """Pacing window for a beat at overall position `idx` (§5.3)."""
    if idx < _EARLY_BEAT_COUNT:
        return _EARLY_MIN, _EARLY_MAX
    return _LATE_MIN, _LATE_MAX


def _waterfill(
    raw: list[float], windows: list[tuple[float, float]], budget: float
) -> tuple[list[float], bool]:
    """Distribute `budget` across beats near their `raw` proportions, honoring
    each beat's [lo, hi] window, redistributing any clamping remainder across the
    still-unclamped beats. Classic iterative clamp-and-redistribute (water-filling).

    Returns (durations, infeasible) where `infeasible` is True iff the budget could
    not be met inside every window (the residual is then spread across all beats,
    breaking a window, and the caller flags it rather than hiding it).
    """
    n = len(raw)
    if n == 0:
        return [], abs(budget) > _EPS
    dur = list(raw)
    fixed = [False] * n
    for _ in range(n + 2):
        for i in range(n):
            lo, hi = windows[i]
            if not fixed[i]:
                if dur[i] < lo - _EPS:
                    dur[i], fixed[i] = lo, True
                elif dur[i] > hi + _EPS:
                    dur[i], fixed[i] = hi, True
        rem = budget - sum(dur)
        unfixed = [i for i in range(n) if not fixed[i]]
        if not unfixed or abs(rem) < _EPS:
            break
        base = sum(dur[i] for i in unfixed)
        for i in unfixed:
            weight = (dur[i] / base) if base > _EPS else (1.0 / len(unfixed))
            dur[i] += rem * weight

    residual = budget - sum(dur)
    infeasible = abs(residual) > 1e-4
    if infeasible:
        # No feasible assignment inside the windows — spread the residual evenly so
        # the budget (and thus exact target length) is still met, and let the caller
        # raise a window_infeasible flag. Meeting target beats hiding the overflow.
        for i in range(n):
            dur[i] += residual / n
    return dur, infeasible


def retime_merged_beats(
    hook_beats: list[dict],
    body_beats: list[dict],
    cta_beat: dict,
    target_length_sec: int,
) -> tuple[list[dict], list[dict]]:
    """Re-derive contiguous beat timestamps for a stitched merge (Step 6).

    The three borrowed pieces arrive with three different original timelines that do
    not line up; this recomputes a single contiguous timeline that ends exactly at
    `target_length_sec`. The model does NOT do this arithmetic — it is code with a
    right answer (Pacing-Checker's own justification for being deterministic).

      (a) Lock the CTA at 2.5-4s, ending exactly at target_length_sec.
      (b) Keep each hook beat at its original 2-3s duration, starting at t=0.
      (c) body_budget = target - hook_total - cta_dur; scale the body's beats
          proportionally to their share of the body's ORIGINAL duration (preserving
          internal rhythm), clamp each to its pacing window (2-3s if among the first
          3 overall beats, else 3-5s), redistribute the clamping remainder.
      (d) Verify each line fits at ~2.3 w/s; widen an overflowing body beat within
          its window and shave the surplus from the longest OTHER body beat; if it
          still overflows, FLAG it (beat index + overflow) rather than ship silently.
      (e) Contiguity + final t_end == target_length_sec exactly.

    Args:
        hook_beats:  the hook_source variant's hook beats (normally [beats[0]]).
        body_beats:  the body_source variant's body beats (beats[1:-1]).
        cta_beat:    the cta_source variant's CTA beat (beats[-1]).
        target_length_sec: the merged script's target length.

    Returns:
        (merged_beats, flags). `merged_beats` are dicts with t_start/t_end/line/
        role/source_variant_id. `flags` are dicts matching RetimingFlag.
    """
    flags: list[dict] = []
    target = float(target_length_sec)

    def _dur(beat: dict) -> float:
        return float(beat["t_end"]) - float(beat["t_start"])

    # (a) CTA duration: clamp original into [2.5, 4.0]; it will end at `target`.
    cta_dur = min(max(_dur(cta_beat), _CTA_MIN), _CTA_MAX)

    # (b) Hook beats keep their original durations, starting at t=0.
    hook_durs = [_dur(b) for b in hook_beats]
    hook_total = sum(hook_durs)

    n_hook = len(hook_beats)
    n_body = len(body_beats)

    # (c) Body budget and proportional scaling.
    body_budget = target - hook_total - cta_dur
    if n_body == 0 or body_budget <= _EPS:
        flags.append(
            {
                "beat_index": n_hook,
                "kind": "no_body_budget",
                "overflow_sec": round(-body_budget, 4) if body_budget <= 0 else 0.0,
                "detail": (
                    f"no room for body beats: target={target}s, hook_total={hook_total}s, "
                    f"cta={cta_dur}s leaves body_budget={round(body_budget, 4)}s "
                    f"for {n_body} body beat(s)"
                ),
            }
        )
        body_durs: list[float] = []
        # Degenerate: fill whatever budget remains equally so target is still hit.
        if n_body:
            body_durs = [max(body_budget, 0.0) / n_body] * n_body
    else:
        orig_body_durs = [_dur(b) for b in body_beats]
        orig_body_total = sum(orig_body_durs) or float(n_body)
        raw_scaled = [body_budget * (d / orig_body_total) for d in orig_body_durs]
        windows = [
            _window_for_overall_index(n_hook + k) for k in range(n_body)
        ]
        body_durs, infeasible = _waterfill(raw_scaled, windows, body_budget)
        if infeasible:
            flags.append(
                {
                    "beat_index": n_hook,
                    "kind": "window_infeasible",
                    "overflow_sec": 0.0,
                    "detail": (
                        f"body_budget={round(body_budget, 4)}s cannot be met inside the "
                        f"pacing windows for {n_body} body beat(s); durations were spread "
                        "to still hit target_length_sec but a window is broken"
                    ),
                }
            )

    # (d) Word-fit: widen an overflowing body beat within its window, shaving the
    # surplus from the longest OTHER body beat (keeps the body sum == body_budget).
    if n_body:
        windows = [_window_for_overall_index(n_hook + k) for k in range(n_body)]
        for k in range(n_body):
            need = _word_count(body_beats[k]["line"]) / _WORDS_PER_SEC
            if body_durs[k] + _EPS >= need:
                continue
            lo_k, hi_k = windows[k]
            want = need - body_durs[k]
            room_k = max(hi_k - body_durs[k], 0.0)
            take = min(want, room_k)
            # Shave from the longest OTHER body beat that has room above its floor.
            donor = None
            donor_room = 0.0
            for j in range(n_body):
                if j == k:
                    continue
                lo_j, _ = windows[j]
                room_j = body_durs[j] - lo_j
                if room_j > donor_room:
                    donor, donor_room = j, room_j
            actual = min(take, donor_room) if donor is not None else 0.0
            if actual > _EPS and donor is not None:
                body_durs[k] += actual
                body_durs[donor] -= actual
            if body_durs[k] + 1e-4 < need:
                flags.append(
                    {
                        "beat_index": n_hook + k,
                        "kind": "word_overflow",
                        "overflow_sec": round(need - body_durs[k], 4),
                        "detail": (
                            f"body beat {n_hook + k} has "
                            f"{_word_count(body_beats[k]['line'])} words "
                            f"(needs {round(need, 4)}s at {_WORDS_PER_SEC} w/s) but only "
                            f"{round(body_durs[k], 4)}s fits within its pacing window"
                        ),
                    }
                )

    # Also surface (but do not adjust) hook/CTA overflow — those beats are fixed by
    # rule (b)/(a), so a widen-and-shave doesn't apply; a flag makes it visible.
    for hi, hb in enumerate(hook_beats):
        need = _word_count(hb["line"]) / _WORDS_PER_SEC
        if hook_durs[hi] + 1e-4 < need:
            flags.append(
                {
                    "beat_index": hi,
                    "kind": "word_overflow",
                    "overflow_sec": round(need - hook_durs[hi], 4),
                    "detail": (
                        f"hook beat {hi} line needs {round(need, 4)}s but keeps its "
                        f"original {round(hook_durs[hi], 4)}s (hook duration is fixed)"
                    ),
                }
            )
    cta_need = _word_count(cta_beat["line"]) / _WORDS_PER_SEC
    if cta_dur + 1e-4 < cta_need:
        flags.append(
            {
                "beat_index": n_hook + n_body,
                "kind": "word_overflow",
                "overflow_sec": round(cta_need - cta_dur, 4),
                "detail": (
                    f"CTA line needs {round(cta_need, 4)}s but is locked to "
                    f"{round(cta_dur, 4)}s within [2.5, 4.0]s"
                ),
            }
        )

    # (e) Build contiguous timestamps from the durations; snap the final t_end to
    # target exactly so floating error can never leave the ad the wrong length.
    durations = hook_durs + body_durs + [cta_dur]
    roles = (
        ["hook"] * n_hook + ["body"] * n_body + ["cta"]
    )
    sources = (
        [b["source_variant_id"] for b in hook_beats]
        + [b["source_variant_id"] for b in body_beats]
        + [cta_beat["source_variant_id"]]
    )
    lines = (
        [b["line"] for b in hook_beats]
        + [b["line"] for b in body_beats]
        + [cta_beat["line"]]
    )

    boundaries = [0.0]
    for d in durations:
        boundaries.append(boundaries[-1] + d)
    # Snap the final boundary to target exactly (absorbs sub-eps float drift).
    boundaries[-1] = target
    boundaries = [round(b, 4) for b in boundaries]
    boundaries[-1] = target  # round() could nudge it; re-snap.

    merged_beats: list[dict] = []
    for i in range(len(durations)):
        merged_beats.append(
            {
                "t_start": boundaries[i],
                "t_end": boundaries[i + 1],
                "line": lines[i],
                "role": roles[i],
                "source_variant_id": sources[i],
            }
        )

    # Hard invariants — arithmetic with a right answer, so assert it (contiguity +
    # exact target). A breach is a bug in this function, not a data problem.
    for i in range(len(merged_beats) - 1):
        assert abs(merged_beats[i]["t_end"] - merged_beats[i + 1]["t_start"]) < 1e-9, (
            f"non-contiguous beats at index {i}"
        )
    assert abs(merged_beats[-1]["t_end"] - target) < 1e-9, "final beat must end at target"

    return merged_beats, flags


# ===========================================================================
# LLM layer (Steps 3/4/5/7) — selection + rationale, validated before Step 6.
# ===========================================================================

_META_SYSTEM_PROMPT = """\
You are the META-CRITIC of an ad-script critic chain. Five specialist checkers have \
already scored each surviving script variant on five axes (hook, pacing, completion/body, \
cta, tone), each with a numeric score AND a justification. Your job is to build ONE \
cross-pollinated MERGE CANDIDATE by selecting the single best HOOK, the single best BODY, \
and the single best CTA across the surviving variants, and to explain the choice.

ABSOLUTE RULES:
- You SELECT pieces and EXPLAIN. You NEVER edit, rewrite, or re-word any copy. You NEVER \
compute or change timestamps (a separate deterministic step re-times the merge).
- You may only select from the surviving variants you are given. Excluded variants are \
absent from your input and must NEVER be resurrected.
- Read the JUSTIFICATIONS, not just the numbers: a 3/5 hook noted as "generic" is worse \
than a 3/5 hook noted as "strong claim, slightly long". Severity lives in the words.
- A merge must BEAT the provided fallback (the single highest-composite variant). Do not \
merge for merging's sake. If no compatible cross-variant assembly is better than that \
single best script, set no_compatible_merge=true and return the fallback's own hook/body/cta.

Walk these steps IN ORDER:

STEP 3 — AXIS LEADERBOARDS. Rank the variants by hook_score, by completion_score (body), \
and by cta_score. Break near-ties by reading justifications first, then tone_score, then \
composite. Emit one leaderboard per axis (axis one of: hook, completion, cta).

STEP 4 — COMPATIBILITY AUDITION (do this BEFORE committing to any assembly). Take the \
provisional #1-hook + #1-body + #1-CTA. Judge:
 (a) promise-payoff: does the chosen BODY beat actually develop the chosen HOOK's SPECIFIC \
     claim (mechanism/evidence), not a different easier claim, not mere restatement?
 (b) seam read: quote the two literal seam junctions (hook->body, body->CTA) and judge \
     voice/register/energy continuity across each.
 (c) trigger continuity: do the hook's and body's emotional triggers escalate together or \
     collide?
A NOTICEABLE-BUT-SMOOTHABLE register shift is a RISK to flag forward (put it in \
risks_to_flag_forward; the downstream Copy Editor can polish it) — it is NOT a failure. \
A PROMISE-PAYOFF failure or a FRAMING CONTRADICTION IS a failure (passed=false).

STEP 5 — SWAP-DOWN (capped at 2 substitutions total). If the audition fails, substitute \
the NEXT-ranked piece on the OFFENDING axis only. Prefer swapping the BODY before the HOOK \
(protect the scarcest resource — the opening/attention). Re-run the audition. If nothing \
passes after 2 substitutions, set no_compatible_merge=true and fall back to the single \
highest-composite variant unmerged, explaining why ("no compatible cross-variant assembly \
beats the single best script"). Record each swap in `substitutions`. If all three winning \
pieces come from the SAME variant, that is a VALID outcome (unanimous, not cross-pollinated) \
— it is not an error and needs no swap.

STEP 7 — MERGE RATIONALE (ADR-shaped), one entry per chosen axis (hook, body, cta). For \
each, ALL SIX of: (1) decision = the piece + its source variant; (2) quoted_evidence = the \
ACTUAL line/claim quoted verbatim + its checker score (NEVER "it scored higher" alone); \
(3) steelmanned_runner_up = the runner-up's GENUINE strength stated fairly (not a strawman); \
(4) trade_off = what was given up by not choosing it; (5) why_it_holds = why this piece \
combines coherently with the others, CITING your Step 4 audition findings; (6) named_risk = \
the seam or beat index you are least confident in, for the downstream validator to check. \
Every claim must cite a checker's output or quote the script — no free-floating adjectives.

Return ONLY a JSON object of EXACTLY this shape (no prose outside it):
{
  "leaderboards": [
    {"axis": "hook|completion|cta", "ranked_variant_ids": ["...","..."], "note": "how ties were broken"}
  ],
  "audition": {
    "promise_payoff": "...", "hook_body_seam": "quote the seam + verdict",
    "body_cta_seam": "quote the seam + verdict", "trigger_continuity": "...",
    "passed": true|false, "risks_to_flag_forward": ["..."]
  },
  "substitutions": [
    {"axis": "hook|body|cta", "from_variant_id": "...", "to_variant_id": "...", "reason": "..."}
  ],
  "hook_source_variant_id": "...",
  "body_source_variant_id": "...",
  "cta_source_variant_id": "...",
  "no_compatible_merge": false,
  "rationale": [
    {"axis": "hook", "decision": "...", "quoted_evidence": "...", "steelmanned_runner_up": "...",
     "trade_off": "...", "why_it_holds": "...", "named_risk": "..."},
    {"axis": "body", ...},
    {"axis": "cta", ...}
  ],
  "overall_reasoning": "2-4 sentences summarising the negotiation"
}
Booleans MUST be real JSON booleans. `substitutions` may be empty. Exactly one rationale \
entry per axis (hook, body, cta).\
"""


def _variant_index(variants: list[ScriptVariant]) -> dict[str, ScriptVariant]:
    return {v["variant_id"]: v for v in variants}


def _llm_payload_for_variant(
    variant: ScriptVariant,
    hook_scores: dict[str, dict],
    pacing_scores: dict[str, dict],
    body_scores: dict[str, dict],
    cta_scores: dict[str, dict],
    tone_scores: dict[str, dict],
) -> dict:
    """Full per-variant payload for the LLM: the script/beats + every checker's
    score AND justification (severity lives in the justification, not the number)."""
    vid = variant["variant_id"]
    beats = variant.get("beats") or []
    return {
        "variant_id": vid,
        "framework": variant.get("framework"),
        "hook_type": variant.get("hook_type"),
        "emotional_trigger": variant.get("emotional_trigger"),
        "beats": [
            {"index": i, "line": b["line"]} for i, b in enumerate(beats)
        ],
        "scores": {
            "hook": hook_scores.get(vid),
            "pacing": pacing_scores.get(vid),
            "completion_body": body_scores.get(vid),
            "cta": cta_scores.get(vid),
            "tone": tone_scores.get(vid),
        },
    }


def _validate_llm_output(
    raw: dict,
    survivor_ids: set[str],
) -> MetaCriticLLMOutput:
    """Gate the raw LLM JSON, then enforce the cross-checks a field validator can't
    (they need the surviving-id set): every source id is a real survivor, and there
    is exactly one rationale per axis (hook, body, cta)."""
    try:
        out = MetaCriticLLMOutput.model_validate(raw)
    except ValidationError as exc:
        raise ValueError(f"Meta-Critic: LLM output failed validation: {exc}") from exc

    for axis, sid in (
        ("hook", out.hook_source_variant_id),
        ("body", out.body_source_variant_id),
        ("cta", out.cta_source_variant_id),
    ):
        if sid not in survivor_ids:
            raise ValueError(
                f"Meta-Critic: {axis}_source_variant_id={sid!r} is not a surviving "
                f"variant (survivors={sorted(survivor_ids)}) — a merge may never "
                "select an excluded or unknown variant"
            )
    for sub in out.substitutions:
        for sid in (sub.from_variant_id, sub.to_variant_id):
            if sid not in survivor_ids:
                raise ValueError(
                    f"Meta-Critic: substitution references non-survivor {sid!r}"
                )

    axes = [r.axis for r in out.rationale]
    if sorted(axes) != ["body", "cta", "hook"]:
        raise ValueError(
            f"Meta-Critic: rationale must have exactly one entry per axis "
            f"(hook, body, cta); got {axes}"
        )
    return out


# ---------------------------------------------------------------------------
# Merge assembly helpers (turn the validated selection into a candidate).
# ---------------------------------------------------------------------------


def _hook_beat_dicts(variant: ScriptVariant) -> list[dict]:
    beats = variant.get("beats") or []
    vid = variant["variant_id"]
    return [{**beats[0], "source_variant_id": vid}] if beats else []


def _body_beat_dicts(variant: ScriptVariant) -> list[dict]:
    beats = variant.get("beats") or []
    vid = variant["variant_id"]
    return [{**beats[i], "source_variant_id": vid} for i in range(1, len(beats) - 1)]


def _cta_beat_dict(variant: ScriptVariant) -> dict:
    beats = variant.get("beats") or []
    vid = variant["variant_id"]
    if not beats:
        raise ValueError(f"Meta-Critic: variant {vid!r} has no beats to take a CTA from")
    return {**beats[-1], "source_variant_id": vid}


def _build_merge_candidate(
    hook_src: ScriptVariant,
    body_src: ScriptVariant,
    cta_src: ScriptVariant,
    target_length_sec: int,
    *,
    rationale: Optional[list[AxisRationale]] = None,
    overall_reasoning: Optional[str] = None,
    audition: Optional[AuditionFinding] = None,
    substitutions: Optional[list[Substitution]] = None,
) -> MergeCandidate:
    """Assemble + re-time (Step 6) a candidate from the three chosen source variants."""
    hook_beats = _hook_beat_dicts(hook_src)
    body_beats = _body_beat_dicts(body_src)
    cta_beat = _cta_beat_dict(cta_src)
    merged_beats, flags = retime_merged_beats(
        hook_beats, body_beats, cta_beat, target_length_sec
    )
    merged_text = " ".join(b["line"] for b in merged_beats)
    return MergeCandidate(
        hook_source_variant_id=hook_src["variant_id"],
        body_source_variant_id=body_src["variant_id"],
        cta_source_variant_id=cta_src["variant_id"],
        merged_beats=[MergedBeat.model_validate(b) for b in merged_beats],
        merged_text=merged_text,
        target_length_sec=target_length_sec,
        rationale=rationale,
        overall_reasoning=overall_reasoning,
        audition=audition,
        substitutions=substitutions or [],
        retiming_flags=[RetimingFlag.model_validate(f) for f in flags],
    )


def _target_length(variants: list[ScriptVariant], survivor_ids: list[str]) -> int:
    """The merged script's target length. All variants share the job's target; if a
    stray variant disagrees we take the surviving majority and it is a data smell,
    not something this node invents around — we just pick the first survivor's value
    (they are expected identical) and trust the Pacing re-check in §5.4.7."""
    idx = _variant_index(variants)
    return int(idx[survivor_ids[0]]["target_length_sec"])


# ===========================================================================
# Public API.
# ===========================================================================


def meta_critic(
    variants: list[ScriptVariant],
    hook_scores: dict[str, dict],
    pacing_scores: dict[str, dict],
    body_scores: dict[str, dict],
    cta_scores: dict[str, dict],
    tone_scores: dict[str, dict],
    *,
    model: Optional[str] = None,
    validator_feedback: Optional[str] = None,
) -> MetaCriticResult:
    """Meta-Critic (§5.4.6): reconcile the five checkers into a merge CANDIDATE.

    Accepts pre-computed per-variant score dicts for all five checkers (it is a
    reconciler, not an orchestrator — each checker fans its own edge in). Produces a
    cross-pollinated merge candidate + ADR rationale; it does NOT decide whether the
    merge is coherent (that is the independent Merge Coherence Validator, §5.4.7) and
    does NOT write to graph state.

    Deterministic vs. LLM: Steps 1 (disqualify), 2 (composite + fallback), and 6
    (re-timing) are pure code; Steps 3/4/5/7 (leaderboards, audition, swap-down,
    rationale) are one Qwen call whose raw JSON is Pydantic-validated before the
    re-timing runs on the selection. Copy is never edited; only Step 6 touches times.

    Args:
        variants:      the surviving-or-not ScriptVariants (typically four).
        hook_scores:   {variant_id: {hook_score, justification}}      (§5.4.1)
        pacing_scores: {variant_id: {pacing_score, violations[]}}     (§5.4.2)
        body_scores:   {variant_id: {completion_score, promise_payoff_match,
                       emotional_trigger_landed, redundant_beat_pairs, justification}} (§5.4.3)
        cta_scores:    {variant_id: {cta_score, justification}}       (§5.4.4)
        tone_scores:   {variant_id: {tone_score, justification, never_do_violation}} (§5.4.5)
        model:         optional model-id override; defaults to MODEL_TEXT (.env).
        validator_feedback: optional. Set by meta_critic_node on the Merge Coherence
                       Validator's (§5.4.7) promise-payoff swap retry -- the CV's own
                       cold-read justification naming the specific clash it found,
                       so this retry doesn't re-select the same failing assembly.
                       None on a first attempt (the common case).

    Returns:
        MetaCriticResult — outcome, merge_candidate (or None), disqualified list,
        composite table, fallback_variant_id, survivor_ids.

    Raises:
        ValueError:    a score dict is missing a variant/field, or the LLM output
                       fails structural/cross validation.
        QwenJSONError: the model returned non-JSON (from the shared helper).
    """
    if not variants:
        return MetaCriticResult(outcome="all_excluded_failure", notes="no variants supplied")

    idx = _variant_index(variants)

    # STEP 1 — Disqualification gate (CODE).
    survivor_ids, disqualified = disqualify(variants, tone_scores)

    # Degenerate: zero survivors is a genuine pipeline failure — do NOT un-exclude
    # the least-bad violator to paper over it.
    if not survivor_ids:
        return MetaCriticResult(
            outcome="all_excluded_failure",
            disqualified=disqualified,
            notes=(
                "every variant violated seller_direction.never_do; no compliant script "
                "exists to build from — this is a pipeline failure, surfaced not hidden"
            ),
        )

    # STEP 2 — Composite table + fallback (CODE). Computed for all survivors even in
    # the short-circuit case, so the trace always carries the table.
    composites = compute_composites(
        survivor_ids, hook_scores, pacing_scores, body_scores, cta_scores, tone_scores
    )
    fallback_id = pick_fallback(composites, tone_scores, hook_scores)

    # Degenerate: exactly one survivor wins un-negotiated (short-circuit, flagged).
    if len(survivor_ids) == 1:
        only = idx[survivor_ids[0]]
        target = int(only["target_length_sec"])
        beats = only.get("beats") or []
        merged_beats = [
            MergedBeat(
                t_start=float(b["t_start"]),
                t_end=float(b["t_end"]),
                line=b["line"],
                role=("hook" if i == 0 else "cta" if i == len(beats) - 1 else "body"),
                source_variant_id=only["variant_id"],
            )
            for i, b in enumerate(beats)
        ]
        candidate = MergeCandidate(
            hook_source_variant_id=only["variant_id"],
            body_source_variant_id=only["variant_id"],
            cta_source_variant_id=only["variant_id"],
            merged_beats=merged_beats,
            # Always join the beat lines, never prefer `only["text"]` -- the Concept
            # Agent writes `text` and `beats[].line` in the same call but nothing
            # guarantees they're byte-identical (contractions, dropped connective
            # words), and downstream (Treatment Agent, Shot-List Agent) prompt the
            # model to quote a BEAT'S OWN LINE while the Justification Validator
            # checks that quote against `winning_script["text"]`. Any divergence
            # here would make a correctly-instructed model response fail validation
            # for no real reason -- confirmed as a real bug via an adversarial
            # integration test pass (Phase 2, see docs/BUILD_TASKS.md). Matches the
            # normal cross-pollinated merge path's `merged_text` (build_merge_candidate,
            # this module), which already always joins beat lines and was never wrong.
            merged_text=" ".join(b["line"] for b in beats),
            target_length_sec=target,
            overall_reasoning=(
                "Single surviving variant after never_do disqualification — it wins "
                "un-negotiated; no merge or audition is needed (short-circuit §5.4.6)."
            ),
        )
        return MetaCriticResult(
            outcome="single_survivor",
            merge_candidate=candidate,
            disqualified=disqualified,
            composite_scores=composites,
            fallback_variant_id=fallback_id,
            survivor_ids=survivor_ids,
            notes="short-circuit: exactly one compliant variant survived",
        )

    # STEPS 3/4/5/7 — the single LLM call over the SURVIVORS only.
    survivor_variants = [idx[vid] for vid in survivor_ids]
    payload = {
        "surviving_variants": [
            _llm_payload_for_variant(
                v, hook_scores, pacing_scores, body_scores, cta_scores, tone_scores
            )
            for v in survivor_variants
        ],
        "composite_scores": composites,
        "fallback_variant_id": fallback_id,
        "target_length_sec": _target_length(variants, survivor_ids),
    }
    user_prompt = (
        "Reconcile these surviving ad-script variants into ONE cross-pollinated merge "
        "candidate. Walk Steps 3 (leaderboards), 4 (audition), 5 (swap-down, cap 2), "
        "7 (ADR rationale). Only select from these variants. The merge must beat the "
        "named fallback or you must set no_compatible_merge=true.\n\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
    )
    if validator_feedback:
        user_prompt += (
            "\n\nThe previous merge candidate from these same survivors failed "
            "independent validation (Merge Coherence Validator, §5.4.7): "
            f"{validator_feedback}\n"
            "Do NOT return the same hook/body/cta assembly again. Swap the flagged "
            "axis to its next-ranked piece among these survivors (or, if no swap "
            "resolves the flagged clash, set no_compatible_merge=true)."
        )
    logger.info(
        "Meta-Critic reconciling %d survivor(s) (excluded %d); fallback=%s",
        len(survivor_ids),
        len(disqualified),
        fallback_id,
    )

    raw = call_qwen_json(_META_SYSTEM_PROMPT, user_prompt, model=model)
    out = _validate_llm_output(raw, set(survivor_ids))

    target = _target_length(variants, survivor_ids)

    # No compatible merge beats the fallback -> return the fallback variant unmerged.
    if out.no_compatible_merge:
        fb = idx[fallback_id]
        candidate = _build_merge_candidate(
            fb, fb, fb, target,
            rationale=out.rationale,
            overall_reasoning=out.overall_reasoning,
            audition=out.audition,
            substitutions=out.substitutions,
        )
        return MetaCriticResult(
            outcome="fallback_no_compatible_merge",
            merge_candidate=candidate,
            disqualified=disqualified,
            composite_scores=composites,
            fallback_variant_id=fallback_id,
            survivor_ids=survivor_ids,
            notes=(
                "no compatible cross-variant assembly beat the single best script "
                f"({fallback_id}) within the 2-substitution cap; returned it unmerged"
            ),
        )

    hook_src = idx[out.hook_source_variant_id]
    body_src = idx[out.body_source_variant_id]
    cta_src = idx[out.cta_source_variant_id]
    candidate = _build_merge_candidate(
        hook_src, body_src, cta_src, target,
        rationale=out.rationale,
        overall_reasoning=out.overall_reasoning,
        audition=out.audition,
        substitutions=out.substitutions,
    )

    unanimous = (
        out.hook_source_variant_id
        == out.body_source_variant_id
        == out.cta_source_variant_id
    )
    return MetaCriticResult(
        outcome="unanimous" if unanimous else "cross_pollinated",
        merge_candidate=candidate,
        disqualified=disqualified,
        composite_scores=composites,
        fallback_variant_id=fallback_id,
        survivor_ids=survivor_ids,
        notes=(
            "all three winning pieces came from one variant (unanimous, not "
            "cross-pollinated)"
            if unanimous
            else "cross-pollinated merge from more than one surviving variant"
        ),
    )


# ===========================================================================
# LangGraph node wrapper — the fan-in join after the 5 parallel checkers.
# ===========================================================================


def _build_justification(vid: str, hook_scores, pacing_scores, body_scores, cta_scores, tone_scores) -> str:
    """Combine each checker's own justification/violations into the single justification
    string CriticScore's frozen schema has room for (one string field, five checkers)."""
    hook = hook_scores.get(vid, {})
    pacing = pacing_scores.get(vid, {})
    body = body_scores.get(vid, {})
    cta = cta_scores.get(vid, {})
    tone = tone_scores.get(vid, {})
    parts = [
        f"Hook: {hook.get('justification', 'n/a')}",
        f"Pacing: {'; '.join(pacing.get('violations', [])) or 'no violations'}",
        f"Body: {body.get('justification', 'n/a')}",
        f"CTA: {cta.get('justification', 'n/a')}",
        f"Tone: {tone.get('justification', 'n/a')}",
    ]
    return " | ".join(parts)


def _source_variant_ids(result: MetaCriticResult) -> list[str]:
    """The variant(s) the merge candidate's pieces came from, for the C2 event payload."""
    if result.merge_candidate is None:
        return [result.fallback_variant_id] if result.fallback_variant_id else []
    mc = result.merge_candidate
    return sorted({mc.hook_source_variant_id, mc.body_source_variant_id, mc.cta_source_variant_id})


async def meta_critic_node(state: ProductCutState, config: RunnableConfig) -> dict:
    """LangGraph node wrapper: the fan-in join after the 5 parallel checkers.

    Assembles the full CriticScore per variant (every variant, not just survivors —
    disqualification is a merge-selection concern, not a reason to withhold a
    variant's own scores from the trace), calls meta_critic() for the actual
    cross-pollinated merge candidate, and stashes the raw MetaCriticResult for
    merge_validator_node (wired downstream, graph/build.py) to consume. Does NOT set
    winning_script — per docs/TECHNICAL_DOCUMENTATION.md §5.4.7, that only happens
    once an independent validator passes.
    """
    variants = state["script_variants"]
    hook_scores = state.get("hook_scores", {})
    pacing_scores = state.get("pacing_scores", {})
    body_scores = state.get("body_scores", {})
    cta_scores = state.get("cta_scores", {})
    tone_scores = state.get("tone_scores", {})

    all_ids = [v["variant_id"] for v in variants]
    composites = compute_composites(
        all_ids, hook_scores, pacing_scores, body_scores, cta_scores, tone_scores
    )

    critic_scores = {}
    for vid in all_ids:
        body = body_scores.get(vid, {})
        critic_scores[vid] = {
            "hook": hook_scores.get(vid, {}).get("hook_score", 0.0),
            "pacing": pacing_scores.get(vid, {}).get("pacing_score", 0.0),
            "completion": body.get("completion_score", 0.0),
            "completion_detail": {
                "redundant_beat_pairs": body.get("redundant_beat_pairs", []),
                "promise_payoff_match": body.get("promise_payoff_match", False),
                "emotional_trigger_landed": body.get("emotional_trigger_landed", False),
            },
            "cta": cta_scores.get(vid, {}).get("cta_score", 0.0),
            "tone": tone_scores.get(vid, {}).get("tone_score", 0.0),
            "composite": composites.get(vid, 0.0),
            "justification": _build_justification(
                vid, hook_scores, pacing_scores, body_scores, cta_scores, tone_scores
            ),
            "never_do_violation": tone_scores.get(vid, {}).get("never_do_violation", False),
        }

    # If the Merge Coherence Validator (§5.4.7) routed back here after a
    # promise-payoff failure, its cold-read justification (naming the specific
    # clash) is the last merge_attempts entry's coherence_check.justification --
    # pass it through so this retry doesn't re-select the same failing assembly.
    prior_attempts = state.get("merge_attempts") or []
    validator_feedback = None
    if prior_attempts:
        last_check = prior_attempts[-1].get("coherence_check") or {}
        if last_check.get("failure_kind") == "promise_payoff":
            validator_feedback = last_check.get("justification")

    result = await asyncio.to_thread(
        meta_critic,
        variants,
        hook_scores,
        pacing_scores,
        body_scores,
        cta_scores,
        tone_scores,
        validator_feedback=validator_feedback,
    )
    result_dict = result.model_dump()

    await adispatch_custom_event(
        "critic_score",
        {"scores": critic_scores, "winning_variant_ids": _source_variant_ids(result)},
        config=config,
    )

    trace_note = (
        f"\n[meta_critic] outcome={result.outcome}; "
        f"disqualified={[d.variant_id for d in result.disqualified]}; "
        f"composites={composites}."
    )
    return {
        "critic_scores": critic_scores,
        "meta_critic_result": result_dict,
        "reasoning_trace": state.get("reasoning_trace", "") + trace_note,
    }


__all__ = [
    # Pydantic models
    "AxisLeaderboard",
    "AuditionFinding",
    "Substitution",
    "AxisRationale",
    "MetaCriticLLMOutput",
    "MergedBeat",
    "RetimingFlag",
    "MergeCandidate",
    "Disqualification",
    "MetaCriticResult",
    # deterministic layers (independently testable)
    "disqualify",
    "compute_composites",
    "pick_fallback",
    "retime_merged_beats",
    # LLM entry point
    "meta_critic",
    # LangGraph node wrapper
    "meta_critic_node",
]
