"""
Merge Coherence Validator ("CV") — independent post-merge check (§5.4.7).
Spec of record: docs/TECHNICAL_DOCUMENTATION.md §5.4.7. Interface locked by
Fable research (see conversation/task history) so the Copy Editor (§5.4.8,
backend/agents/copy_editor.py) can be built in parallel against a fixed
contract rather than guessing this module's shape.

Scope: this module is deliberately NOT the Meta-Critic. It receives a
`MergeCandidate` (agents.meta_critic's cross-pollinated merge) and shares no
call/context with the reasoning that produced it — that separation is what
makes its pass/fail judgment an actual second opinion. Routing to the Copy
Editor on a voice/register failure, or back to the Meta-Critic on a
promise-payoff failure, is a LangGraph conditional edge, NOT a call CV makes
itself — this file never imports agents.copy_editor.

The four Pydantic models below are the locked contract the Copy Editor
(agents/copy_editor.py) was built against in parallel. The entry point
(`validate_merge_candidate`), its internal steps (`run_pacing_recheck`,
`repair_pacing`, `run_coherence_read`, `derive_seam_flags`), and the LangGraph
node wrapper (`merge_validator_node` / `route_after_merge_validation`) are
implemented below.
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
    model_validator,
)

from agents.critic_llm import QwenJSONError, call_qwen_json
from agents.meta_critic import (
    MergeCandidate,
    MergedBeat,
    RetimingFlag,
    _CTA_MAX,
    _CTA_MIN,
    retime_merged_beats,
)
from agents.pacing_checker import (
    EARLY_BEAT_WINDOW,
    LATER_BEAT_WINDOW,
    NUM_EARLY_BEATS,
    check_pacing,
)
from graph.state import ProductCutState

logger = logging.getLogger("productcut.critics.merge_validator")


class PacingRecheck(BaseModel):
    """Sub-check 1 (§5.4.7): re-runs Pacing-Checker logic against merged beats."""

    model_config = ConfigDict(extra="forbid")

    passed: StrictBool
    violations: list[str] = Field(default_factory=list)
    repaired: StrictBool = False  # True iff the one deterministic re-time repair ran


class CoherenceRead(BaseModel):
    """Raw LLM cold-read output (sub-check 2), validated before anything trusts it."""

    model_config = ConfigDict(extra="forbid")

    coherence_score: int = Field(..., ge=1, le=5)
    voice_consistency: StrictBool
    promise_payoff_match: StrictBool
    register_shift_flags: list[int] = Field(default_factory=list)
    justification: str = Field(..., min_length=1)

    @model_validator(mode="after")
    def _consistent(self):
        # A low score with all-clear booleans/no flags is incoherent model output --
        # caller re-prompts once (Concept-Agent re-prompt convention), then hard-fails.
        if (
            self.coherence_score <= 2
            and self.voice_consistency
            and self.promise_payoff_match
            and not self.register_shift_flags
        ):
            raise ValueError("score<=2 requires at least one named failure")
        return self


class SeamFlag(BaseModel):
    """One CV -> Copy Editor handoff unit: a single constrained-repair target.

    `editable_beat_index` is always the adjacent BODY beat -- the hook line and
    the CTA line are never editable by the Copy Editor (§5.4.8).
    """

    model_config = ConfigDict(extra="forbid")

    seam: str = Field(..., pattern="^(hook_body|body_cta)$")
    flagged_beat_index: int
    editable_beat_index: int
    evidence: str = Field(..., min_length=1)


class CoherenceValidationResult(BaseModel):
    """CV's full output contract (§5.4.7)."""

    model_config = ConfigDict(extra="forbid")

    passed: StrictBool
    pacing_recheck: PacingRecheck
    # None when the coherence read never ran (pacing failed even after repair):
    coherence_score: Optional[int] = Field(None, ge=1, le=5)
    voice_consistency: Optional[StrictBool] = None
    promise_payoff_match: Optional[StrictBool] = None
    register_shift_flags: list[int] = Field(default_factory=list)
    justification: str = ""
    # Routing key for the LangGraph conditional edge -- the ONLY field routing reads:
    failure_kind: Optional[str] = Field(None, pattern="^(pacing|promise_payoff|voice_register)$")
    seam_flags: list[SeamFlag] = Field(default_factory=list)  # non-empty iff failure_kind == "voice_register"
    candidate_after_repair: Optional[dict] = None  # MergeCandidate.model_dump(), set iff repair ran


# ===========================================================================
# SUB-CHECK 1 — Deterministic pacing re-check + one repair attempt (CODE).
# ===========================================================================


def run_pacing_recheck(candidate: MergeCandidate) -> PacingRecheck:
    """Re-runs pacing_checker.check_pacing's exact logic against candidate.merged_beats.

    check_pacing takes a dict-like variant (`.get("beats")` / `.get("target_length_sec")`),
    not a Pydantic model, so MergedBeats are model_dump()'d into plain dicts first.
    The gate is `len(violations) == 0`, NOT the informational pacing_score.
    """
    variant_like = {
        "beats": [b.model_dump() for b in candidate.merged_beats],
        "target_length_sec": candidate.target_length_sec,
    }
    result = check_pacing(variant_like)
    violations = result["violations"]
    return PacingRecheck(passed=len(violations) == 0, violations=violations, repaired=False)


def _cta_window_intersection(cta_overall_index: int) -> tuple[float, float]:
    """The CTA beat's ALLOWED duration range: the intersection of retime's own
    CTA window (meta_critic._CTA_MIN/_CTA_MAX, 2.5-4.0s) and check_pacing's window
    for the CTA's overall beat index (index<3 -> early (2,3)s, else late (3,5)s).

    Without this clamp, retime_merged_beats can re-emit a CTA duration (e.g. 3.5s
    for an early-window CTA) that satisfies its own 2.5-4.0s rule but still fails
    check_pacing's stricter early-beat window -- an endless repair/re-check loop
    on a single deterministic repair attempt. Read live off both modules' real
    constants (not hard-coded numbers) so the two stay in sync if either changes.
    """
    if cta_overall_index < NUM_EARLY_BEATS:
        lo, hi = EARLY_BEAT_WINDOW
    else:
        lo, hi = LATER_BEAT_WINDOW
    return max(lo, _CTA_MIN), min(hi, _CTA_MAX)


def repair_pacing(candidate: MergeCandidate) -> MergeCandidate:
    """One deterministic repair attempt via meta_critic.retime_merged_beats.

    Splits the candidate's already-merged beats back into hook/body/cta dicts by
    `role`, pre-clamps the CTA beat's duration into `_cta_window_intersection` (see
    above), then re-derives contiguous timestamps exactly like the Meta-Critic's
    own Step 6. Returns a NEW MergeCandidate (merged_beats/merged_text/retiming_flags
    replaced); the caller re-runs `run_pacing_recheck` on the result.
    """
    beats = candidate.merged_beats
    hook_beats = [b.model_dump() for b in beats if b.role == "hook"]
    body_beats = [b.model_dump() for b in beats if b.role == "body"]
    cta_beats = [b for b in beats if b.role == "cta"]
    if not cta_beats:
        raise ValueError("merge_validator.repair_pacing: candidate has no CTA beat")
    cta_beat = cta_beats[-1].model_dump()
    cta_overall_index = len(beats) - 1

    lo, hi = _cta_window_intersection(cta_overall_index)
    dur = cta_beat["t_end"] - cta_beat["t_start"]
    clamped_dur = min(max(dur, lo), hi)
    if abs(clamped_dur - dur) > 1e-9:
        cta_beat = {**cta_beat, "t_end": cta_beat["t_start"] + clamped_dur}

    merged_beats, flags = retime_merged_beats(
        hook_beats, body_beats, cta_beat, candidate.target_length_sec
    )
    return candidate.model_copy(
        update={
            "merged_beats": [MergedBeat.model_validate(b) for b in merged_beats],
            "merged_text": " ".join(b["line"] for b in merged_beats),
            "retiming_flags": [RetimingFlag.model_validate(f) for f in flags],
        }
    )


# ===========================================================================
# SUB-CHECK 2 — Independent blind LLM coherence read (LLM).
# ===========================================================================

_COHERENCE_SYSTEM_PROMPT = """\
You are an independent editorial reviewer doing a BLIND COLD READ of a stitched ad-script \
voiceover. You were not involved in writing or assembling it, and you have NOT been told \
which piece came from where, why it was chosen, or what any other reviewer thinks. Read it \
exactly as a listener would hear it for the first time, beat by beat, in order.

You are given ONLY the beats (each with its index, its line, and its narrative role: hook, \
body, or cta) and the target_length_sec. Judge THREE things:

1. VOICE / POV CONSISTENCY. Does the whole script sound like it was written by ONE voice --
consistent point of view (direct address vs third person), consistent register (casual vs \
formal), consistent energy? A stitched script pulled from independently-written pieces often \
has a seam where the voice audibly changes. Set voice_consistency=false if you can hear a \
voice change anywhere in the script.

2. PROMISE-PAYOFF MATCH. Does the BODY (the beats with role="body") actually pay off the \
SPECIFIC claim the HOOK (role="hook") makes -- the same concrete claim, not a restatement of \
a generic benefit and not a different, easier claim? Set promise_payoff_match=false if the \
body does not substantively develop the hook's own specific promise.

3. REGISTER / TRANSITION SMOOTHNESS AT THE TWO SEAMS. Read the hook->body transition (the \
last hook beat into the first body beat) and the body->cta transition (the last body beat \
into the cta beat). For EACH seam, judge whether the transition is smooth or jarring -- a \
sudden register/voice/energy shift right at that junction. For every beat index where a \
jarring shift is audible AT THAT BEAT (the beat where the shift becomes noticeable), add \
that beat's index to register_shift_flags. A script with no jarring transitions gets an \
empty register_shift_flags list.

Score coherence_score 1-5 (5 = reads as one seamless voice, real payoff, no jarring seams; \
1 = feels like three different scripts taped together). If coherence_score is 1 or 2, you \
MUST also set voice_consistency=false OR promise_payoff_match=false OR include at least one \
register_shift_flags entry -- a low score with no named failure is not an acceptable verdict.

Return ONLY a JSON object of this exact shape:
{"coherence_score": <int 1-5>, "voice_consistency": <true|false>, \
"promise_payoff_match": <true|false>, "register_shift_flags": [<beat_index>, ...], \
"justification": "<2-4 sentences: the voice verdict, the promise-payoff verdict, and what \
(if anything) you heard at each seam>"}
Booleans MUST be real JSON booleans. register_shift_flags MUST use the beat index values \
given (an empty list if none). No prose outside the JSON.\
"""


def run_coherence_read(candidate: MergeCandidate, *, model: Optional[str] = None) -> CoherenceRead:
    """The one LLM call -- a BLIND cold read.

    Strips source_variant_id from every beat (and everything else about how the
    candidate was assembled -- rationale/audition/substitutions/overall_reasoning/
    retiming_flags/any checker score/framework/hook_type/emotional_trigger/
    product_truths never enter this payload). Gives ONLY beats [{index, line, role}]
    and target_length_sec, so the read is genuinely blind to the merge's own reasoning.
    """
    payload = {
        "beats": [
            {"index": i, "line": b.line, "role": b.role}
            for i, b in enumerate(candidate.merged_beats)
        ],
        "target_length_sec": candidate.target_length_sec,
    }
    user_prompt = (
        "Cold-read this stitched ad-script voiceover blind. Judge voice/POV consistency, "
        "promise-payoff match, and register smoothness at the hook->body and body->cta "
        "seams.\n\n" + json.dumps(payload, ensure_ascii=False, indent=2)
    )
    raw = call_qwen_json(_COHERENCE_SYSTEM_PROMPT, user_prompt, model=model)
    return CoherenceRead.model_validate(raw)


# ===========================================================================
# Deterministic seam-flag derivation (CODE) -- routes a voice/register failure
# to the Copy Editor's constrained repair target.
# ===========================================================================


def derive_seam_flags(candidate: MergeCandidate, read: CoherenceRead) -> list[SeamFlag]:
    """Deterministic. Maps each flagged register_shift index to a routable seam.

    first_body/last_body = first/last index with role=='body' in candidate.merged_beats.
    For each flagged index i: i<=first_body -> seam='hook_body', editable_beat_index=
    first_body; i>=last_body -> seam='body_cta', editable_beat_index=last_body; a
    strictly interior body index is not seam-routable (no SeamFlag emitted -- the
    evidence still lives in read.justification). If voice_consistency is False and
    this yields zero SeamFlags, both seams get a SeamFlag (a voice failure with no
    localizable seam still gets the constrained repair). Dedupe by seam (at most one
    SeamFlag per seam) -- first match wins.

    Edge case (not explicit in the original spec): if the candidate has NO body
    beats at all (e.g. a degenerate single-beat-per-role candidate), there is no
    editable beat to route to, so no SeamFlag is emitted at all -- see final report.
    """
    beats = candidate.merged_beats
    body_indices = [i for i, b in enumerate(beats) if b.role == "body"]
    if not body_indices:
        return []
    first_body = body_indices[0]
    last_body = body_indices[-1]

    flags_by_seam: dict[str, SeamFlag] = {}

    def _add(seam: str, flagged_index: int, editable_index: int, evidence: str) -> None:
        if seam not in flags_by_seam:
            flags_by_seam[seam] = SeamFlag(
                seam=seam,
                flagged_beat_index=flagged_index,
                editable_beat_index=editable_index,
                evidence=evidence,
            )

    for i in read.register_shift_flags:
        if i <= first_body:
            _add(
                "hook_body", i, first_body,
                f"register_shift_flags includes beat {i} (hook->body seam): {read.justification}",
            )
        elif i >= last_body:
            _add(
                "body_cta", i, last_body,
                f"register_shift_flags includes beat {i} (body->cta seam): {read.justification}",
            )
        # else: strictly interior body beat -- not seam-routable; the evidence
        # stays in read.justification only, no SeamFlag is emitted for it.

    if not read.voice_consistency and not flags_by_seam:
        _add(
            "hook_body", first_body, first_body,
            f"voice_consistency failed with no localizable seam flag: {read.justification}",
        )
        _add(
            "body_cta", last_body, last_body,
            f"voice_consistency failed with no localizable seam flag: {read.justification}",
        )

    return list(flags_by_seam.values())


# ===========================================================================
# Public entry point.
# ===========================================================================

# One initial attempt + one re-prompt (the "re-prompt once, then give up"
# convention used by the Concept Agent / Hook-Checker elsewhere in this codebase).
_MAX_COHERENCE_ATTEMPTS = 2


def validate_merge_candidate(
    candidate: MergeCandidate, *, model: Optional[str] = None
) -> CoherenceValidationResult:
    """The public entry point (§5.4.7). See module docstring / task brief for the
    full flow. Summary:

    1. Pacing re-check; on violation, one deterministic repair + re-check.
    2. Still failing -> failed result, failure_kind='pacing', coherence read SKIPPED.
    3. Pacing passes (possibly after repair) -> blind coherence read (one re-prompt
       on QwenJSONError/ValidationError, then give up).
    4. No coherence verdict obtainable even after the re-prompt -> failed result,
       failure_kind=None (this exact case is not one of the three named failure
       kinds -- see the docstring note below and the final report), justification
       names why.
    5. Gate = voice_consistency AND promise_payoff_match AND not register_shift_flags.
       Pass -> passed=True. Fail -> failure_kind='promise_payoff' (precedence over
       'voice_register') if promise_payoff_match is False, else 'voice_register'
       (register_shift_flags non-empty or voice_consistency False with payoff intact);
       seam_flags derived only for the voice_register case.
    """
    working = candidate
    repaired = False

    pacing = run_pacing_recheck(working)
    if not pacing.passed:
        working = repair_pacing(working)
        repaired = True
        pacing = run_pacing_recheck(working)
        pacing = pacing.model_copy(update={"repaired": True})

    if not pacing.passed:
        return CoherenceValidationResult(
            passed=False,
            pacing_recheck=pacing,
            coherence_score=None,
            voice_consistency=None,
            promise_payoff_match=None,
            register_shift_flags=[],
            justification=(
                "pacing re-check failed even after the one deterministic repair "
                "attempt; coherence read was skipped entirely (no LLM call)."
            ),
            failure_kind="pacing",
            seam_flags=[],
            candidate_after_repair=working.model_dump(),
        )

    read: Optional[CoherenceRead] = None
    last_error: Optional[Exception] = None
    for attempt in range(_MAX_COHERENCE_ATTEMPTS):
        try:
            read = run_coherence_read(working, model=model)
            break
        except (QwenJSONError, ValidationError) as exc:
            last_error = exc
            logger.warning(
                "Merge Coherence Validator: coherence read attempt %d failed (%s); %s",
                attempt + 1,
                exc,
                "re-prompting once" if attempt == 0 else "giving up after re-prompt",
            )

    if read is None:
        # No coherence verdict was obtainable even after the one re-prompt. None of
        # the three named failure_kinds ("pacing"/"promise_payoff"/"voice_register")
        # fit this case -- they all presuppose a verdict to route on. Represented as
        # failed + failure_kind=None (routing falls through to the fallback default,
        # since there is nothing to retry against) with the reason named in
        # `justification` for the trace. See final report for this design choice.
        return CoherenceValidationResult(
            passed=False,
            pacing_recheck=pacing,
            coherence_score=None,
            voice_consistency=None,
            promise_payoff_match=None,
            register_shift_flags=[],
            justification=(
                "coherence read could not be obtained after one re-prompt "
                f"({last_error}); no coherence verdict was reached -- routed to fallback."
            ),
            failure_kind=None,
            seam_flags=[],
            candidate_after_repair=working.model_dump() if repaired else None,
        )

    gate = (
        read.voice_consistency
        and read.promise_payoff_match
        and not read.register_shift_flags
    )

    if gate:
        return CoherenceValidationResult(
            passed=True,
            pacing_recheck=pacing,
            coherence_score=read.coherence_score,
            voice_consistency=read.voice_consistency,
            promise_payoff_match=read.promise_payoff_match,
            register_shift_flags=read.register_shift_flags,
            justification=read.justification,
            failure_kind=None,
            seam_flags=[],
            candidate_after_repair=working.model_dump() if repaired else None,
        )

    if not read.promise_payoff_match:
        failure_kind = "promise_payoff"
        seam_flags: list[SeamFlag] = []
    else:
        failure_kind = "voice_register"
        seam_flags = derive_seam_flags(working, read)

    return CoherenceValidationResult(
        passed=False,
        pacing_recheck=pacing,
        coherence_score=read.coherence_score,
        voice_consistency=read.voice_consistency,
        promise_payoff_match=read.promise_payoff_match,
        register_shift_flags=read.register_shift_flags,
        justification=read.justification,
        failure_kind=failure_kind,
        seam_flags=seam_flags,
        candidate_after_repair=working.model_dump() if repaired else None,
    )


# ===========================================================================
# LangGraph integration.
# ===========================================================================


def _candidate_from_state(state: ProductCutState) -> MergeCandidate:
    """Reads state["pending_merge_candidate"] if present (a repair/next-attempt
    scratch candidate), else falls back to state["meta_critic_result"]["merge_candidate"]
    (the original, first-attempt candidate)."""
    pending = state.get("pending_merge_candidate")
    if pending:
        return MergeCandidate.model_validate(pending)
    mcr = state.get("meta_critic_result")
    if not mcr or not mcr.get("merge_candidate"):
        raise ValueError(
            "merge_validator_node: no pending_merge_candidate and no "
            "meta_critic_result.merge_candidate in state"
        )
    return MergeCandidate.model_validate(mcr["merge_candidate"])


def _winning_script_from_candidate(candidate: MergeCandidate) -> dict:
    """Build WinningScript from a PASSED candidate -- strip role/source_variant_id
    from each beat (WinningScript.beats is typed list[ScriptBeat], t_start/t_end/line
    only)."""
    return {
        "text": candidate.merged_text,
        "beats": [
            {"t_start": b.t_start, "t_end": b.t_end, "line": b.line}
            for b in candidate.merged_beats
        ],
        "source_variant_ids": sorted(
            {
                candidate.hook_source_variant_id,
                candidate.body_source_variant_id,
                candidate.cta_source_variant_id,
            }
        ),
    }


def _winning_script_from_fallback_variant(state: ProductCutState) -> dict:
    """Build WinningScript directly from the fallback variant's OWN beats/text,
    unmerged (§5.4.7: "already fully valid ... no re-validation is needed")."""
    mcr = state.get("meta_critic_result") or {}
    fallback_id = mcr.get("fallback_variant_id")
    variants = state.get("script_variants") or []
    match = next((v for v in variants if v.get("variant_id") == fallback_id), None)
    if match is None:
        raise ValueError(
            f"merge_validator_node: fallback_variant_id={fallback_id!r} not found in "
            "state['script_variants']"
        )
    return {
        "text": match.get("text") or " ".join(b["line"] for b in match.get("beats") or []),
        "beats": [
            {"t_start": b["t_start"], "t_end": b["t_end"], "line": b["line"]}
            for b in match.get("beats") or []
        ],
        "source_variant_ids": [fallback_id],
    }


def route_after_merge_validation(state: ProductCutState) -> str:
    """Pure function reading ONLY the last state["merge_attempts"] entry.

    passed -> 'finalize'. Else, if the single retry slot is already spent
    (len(merge_attempts) >= 2) -> 'fallback' regardless of failure_kind. Else,
    routes by failure_kind: 'voice_register' -> 'copy_editor'; 'promise_payoff' ->
    'meta_critic'; 'pacing' (or None -- no coherence verdict obtainable) -> 'fallback'.
    """
    attempts = state.get("merge_attempts") or []
    if not attempts:
        raise ValueError(
            "route_after_merge_validation: state['merge_attempts'] is empty -- "
            "merge_validator_node must run before this conditional edge is evaluated"
        )
    last = attempts[-1]
    coherence_check = last.get("coherence_check") or {}

    if coherence_check.get("passed"):
        return "finalize"
    if len(attempts) >= 2:
        return "fallback"

    failure_kind = coherence_check.get("failure_kind")
    if failure_kind == "voice_register":
        return "copy_editor"
    if failure_kind == "promise_payoff":
        return "meta_critic"
    return "fallback"  # 'pacing', or None (no coherence verdict was obtainable)


async def merge_validator_node(state: ProductCutState, config: RunnableConfig) -> dict:
    """LangGraph node wrapper (§5.4.7): validates the pending merge candidate,
    appends a §6-shaped merge_attempts[] entry, and — only on a pass or the
    terminal fallback — sets winning_script. Does NOT decide routing itself;
    `route_after_merge_validation` (a separate conditional-edge function) does that,
    though this node calls it internally too, purely to decide whether IT should
    do the fallback assembly / set winning_script this turn (single source of truth,
    not a re-implementation).
    """
    candidate = _candidate_from_state(state)
    prior_attempts = state.get("merge_attempts") or []
    attempt_number = len(prior_attempts) + 1
    # agents.copy_editor.copy_editor_node (built in parallel) writes its result to
    # state["last_copy_edit"] -- reconciled here since that module now exists and
    # documents its own state contract in its docstring (this file never imports
    # agents.copy_editor; the key name is just data-plumbing).
    prior_was_copy_edit = state.get("last_copy_edit") is not None

    result = await asyncio.to_thread(validate_merge_candidate, candidate)
    result_dict = result.model_dump()

    if result.passed:
        # NOTE: 'copy_edited_then_accepted' is conceptually the Copy Editor's outcome
        # to declare (per the task brief), but this node is the only place that
        # actually learns whether the post-copy-edit re-validation passed, so it is
        # set here when state["last_copy_edit"] (written by agents.copy_editor.
        # copy_editor_node, confirmed against its actual source) is present.
        outcome = "copy_edited_then_accepted" if prior_was_copy_edit else "accepted"
    else:
        outcome = "retried"  # overwritten to "fell_back_to_variant" below if terminal

    attempt_entry = {
        "attempt_number": attempt_number,
        "hook_source_variant": candidate.hook_source_variant_id,
        "body_source_variant": candidate.body_source_variant_id,
        "cta_source_variant": candidate.cta_source_variant_id,
        "merged_script": candidate.merged_text,
        "pacing_recheck": result_dict["pacing_recheck"],
        "coherence_check": {
            "passed": result_dict["passed"],
            "coherence_score": result_dict["coherence_score"],
            "voice_consistency": result_dict["voice_consistency"],
            "promise_payoff_match": result_dict["promise_payoff_match"],
            "register_shift_flags": result_dict["register_shift_flags"],
            "justification": result_dict["justification"],
            # additive beyond §6's literal listed sub-keys, per state.py's own
            # "extend additively only" convention -- routing needs failure_kind,
            # and the Copy Editor needs seam_flags.
            "failure_kind": result_dict["failure_kind"],
            "seam_flags": result_dict["seam_flags"],
        },
        # Populated only when the Copy Editor ran on this attempt (§6). This node
        # never writes state["last_copy_edit"] itself -- agents.copy_editor.
        # copy_editor_node does, per its own docstring's state contract -- read
        # here so the CE round-trip's before/after record lands in this attempt's
        # §6 copy_edit sub-object.
        "copy_edit": state.get("last_copy_edit"),
        "outcome": outcome,
    }

    merge_attempts = prior_attempts + [attempt_entry]
    route = route_after_merge_validation({**state, "merge_attempts": merge_attempts})

    updates: dict = {"merge_attempts": merge_attempts}

    if route == "finalize":
        updates["winning_script"] = _winning_script_from_candidate(candidate)
    elif route == "fallback":
        attempt_entry["outcome"] = "fell_back_to_variant"
        updates["winning_script"] = _winning_script_from_fallback_variant(state)
    elif route == "copy_editor":
        # Hand the (possibly pacing-repaired) candidate to the Copy Editor via the
        # scratch key so it has something concrete to patch.
        updates["pending_merge_candidate"] = result.candidate_after_repair or candidate.model_dump()
        # agents.copy_editor.copy_editor_node reads state["coherence_validation_result"]
        # (its own documented assumption) for seam_flags + justification -- write the
        # full result here so that reconciles without copy_editor.py needing any change.
        updates["coherence_validation_result"] = result_dict
        # Clear any stale copy-edit record from an earlier round so this round's
        # `prior_was_copy_edit` check (on the NEXT merge_validator_node call) reflects
        # only the copy-edit the Copy Editor is about to perform now, not a leftover.
        updates["last_copy_edit"] = None
    elif route == "meta_critic":
        # meta_critic_node recomputes wholly from the checker scores (it does not
        # consume pending_merge_candidate), so clear any stale scratch candidate --
        # otherwise a later attempt could wrongly prefer a leftover copy-edited
        # candidate over the fresh meta_critic_result this loop-back produces.
        updates["pending_merge_candidate"] = None

    await adispatch_custom_event(
        "merge_validated",
        {"result": result_dict, "attempt_number": attempt_number},
        config=config,
    )

    trace_note = (
        f"\n[merge_validator] attempt={attempt_number} passed={result.passed} "
        f"failure_kind={result.failure_kind} route={route}."
    )
    updates["reasoning_trace"] = state.get("reasoning_trace", "") + trace_note
    return updates


__all__ = [
    "PacingRecheck",
    "CoherenceRead",
    "SeamFlag",
    "CoherenceValidationResult",
    # deterministic layers (independently testable)
    "run_pacing_recheck",
    "repair_pacing",
    "derive_seam_flags",
    # LLM layer
    "run_coherence_read",
    # public entry point
    "validate_merge_candidate",
    # LangGraph integration
    "merge_validator_node",
    "route_after_merge_validation",
]
