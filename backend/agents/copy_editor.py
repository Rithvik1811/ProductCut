"""
Copy Editor ("CE", §5.4.8) of the Critic Chain's merge-repair path.

Spec of record: docs/TECHNICAL_DOCUMENTATION.md §5.4.8 (this node), §5.4.7 (the
Merge Coherence Validator, "CV" -- this node's only caller), §4 (architecture
diagram: `CV -->|voice/register fail, attempts<1| CE`, `CE --> CV`), §6
(`merge_attempts[].copy_edit` shape).

Scope. A cross-pollinated merge (agents.meta_critic's `MergeCandidate`) stitches
a hook, body, and CTA that were each written for a *different* script. When the
independent Merge Coherence Validator's blind cold read flags a voice/register
clash at one of the two stitch points (hook->body or body->CTA), the honest
repair is a professional copy editor's constrained polish of that single seam
line -- not a rewrite, not a swap to a different pre-written piece (that piece
also wasn't written for this hook), and never a self-grade (the patched
candidate always goes back through the same CV for a full re-check; this node
never marks its own work sufficient).

This module is deliberately NOT the Merge Coherence Validator: it depends on
exactly one thing from `agents.merge_validator` -- the locked `SeamFlag` model,
which names WHICH beat is editable (always the adjacent BODY beat; the CV has
already ruled out the hook beat and the CTA beat as edit targets before this
node ever sees a `SeamFlag`). It does not import, call, or duplicate any of
that module's actual validation logic.

Scope note -- what this is NOT:
  * WIRED into `backend/graph/build.py` -- `copy_editor -> merge_validator`
    loop-back, alongside `merge_validator`'s own conditional routing to it.
    Was left for a follow-up integration once both this file and
    merge_validator.py were complete (to avoid two agents colliding on the
    same shared file); that wiring has since landed.
  * NOT an editor of `backend/graph/state.py` -- the CV agent owns the
    additive state-schema changes (`merge_attempts`, `pending_merge_candidate`,
    `winning_script` finalization). This node's `copy_editor_node` wrapper
    documents, in its own docstring, exactly which state keys it *assumes* for
    reading/writing, so the orchestrating session can reconcile them against
    whatever `merge_validator_node` actually lands on.
  * NOT the Meta-Critic (§5.4.6) or the Concept Agent (§5.3) -- this node never
    generates new copy or selects among pre-written pieces; it only polishes
    the literal text already sitting at one or two named beat indices.

The deterministic / LLM split (same "mechanical where mechanical is possible"
posture as the Pacing-Checker and Body-Checker):
  * The ONE LLM call (Qwen-Plus via the shared `critic_llm.call_qwen_json`)
    proposes a revised line for each flagged seam, grounded in the CV's own
    cold-read justification and the seam's neighboring line.
  * `_check_constraints` is pure code, no LLM: it deterministically verifies
    the model actually stayed inside its license (no beat outside the named
    indices changed, no timestamp anywhere moved, the hook/CTA beats were
    never touched, word count stayed within ~10%, and no numeric token present
    in the original line silently vanished). A prompt alone is necessary but
    not sufficient to guarantee this -- exactly the reasoning behind
    Body-Checker's deterministic hard-cap and the Meta-Critic's deterministic
    re-timing step.
  * On a constraint-check failure, `copy_edit_seams` re-prompts ONCE, naming
    the specific violation(s) (the same "re-prompt once, then fall back"
    convention as the Concept Agent / other checkers). A second failure ships
    the ORIGINAL, unmodified candidate with `constraint_check.passed=False` --
    a failed `ConstraintCheck` IS the signal to the caller (CV), not an
    exception; CE never partially or silently ships a bad edit.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Optional

from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, ConfigDict, Field, StrictBool

from agents.critic_llm import call_qwen_json
from agents.merge_validator import SeamFlag
from agents.meta_critic import MergeCandidate
from graph.state import ProductCutState

logger = logging.getLogger("productcut.critics.copy_editor")


# ===========================================================================
# Pydantic models — output contract (§5.4.8 + §6's merge_attempts[].copy_edit).
# ===========================================================================


class ConstraintCheck(BaseModel):
    """Deterministic, pure-code verdict on whether a proposed edit stayed
    inside the Copy Editor's license (see `_check_constraints`)."""

    model_config = ConfigDict(extra="forbid")

    passed: StrictBool
    violations: list[str] = Field(default_factory=list)


class CopyEditResult(BaseModel):
    """Copy Editor's full output (§5.4.8 output contract + §6's copy_edit sub-object).

    `patched_candidate` is the FULL candidate; on a successful repair, ONLY the
    `line` field at the edited beat indices and the derived `merged_text`
    differ from the input candidate -- every timestamp, role, and
    source_variant_id is byte-identical. On a failed repair (constraint check
    still failing after the one re-prompt), `patched_candidate` is the
    ORIGINAL, unmodified candidate -- no partial/bad edit ever ships.
    """

    model_config = ConfigDict(extra="forbid")

    patched_candidate: MergeCandidate
    seams_edited: list[int] = Field(default_factory=list)
    original_seam_text: str
    revised_seam_text: str
    justification: str
    constraint_check: ConstraintCheck


# ===========================================================================
# System prompt — matches this codebase's established voice (Body-Checker):
# numbered/bulleted hard rules + calibrated strong/weak exemplar pairs.
# ===========================================================================

_COPY_EDITOR_SYSTEM_PROMPT = """\
You are a COPY EDITOR performing a constrained seam repair on an ad script stitched \
from pieces by different writers (a cross-pollinated merge). A validator has \
independently cold-read the stitched script and flagged a voice/register clash at one \
or more specific seams. Your license is a copy editor's, not a writer's: you smooth HOW \
it is said at the flagged transition; you never change WHAT is said, anywhere.

YOU MAY rewrite ONLY the line(s) at the beat index(es) given to you below. Return a \
revision for EACH of those indices and nothing else -- do not return any other beat's \
line, edited or not.

YOU MAY NOT, under any circumstance:
- introduce any new claim, number, benefit, or product detail not already in the \
original line;
- drop any claim, number, or detail the original line already carried;
- touch the hook line or the CTA line, even if you think they would read better;
- move content from one line to another;
- change a line's length by more than roughly 10% of its original word count.

If a seam cannot be smoothed inside these rules, return the ORIGINAL line UNCHANGED for \
that beat index and say so plainly in your justification -- exceeding your license is a \
worse failure than an unfixed seam.

Ground yourself in the validator's own justification for WHY the seam clashes (given \
below as `coherence_justification`) and the seam's neighboring line (the hook line for \
a hook_body seam, the CTA line for a body_cta seam) -- match register, energy, and \
address to that neighbor, and nothing more.

CALIBRATION:
- STRONG repair: body opener "Many people find their drinks losing warmth too quickly" \
sitting after the hook "Your coffee is cold in 12 minutes. Mine isn't." becomes "Yours \
does too -- here's why mine doesn't:". This picks up the hook's direct address and \
energy, and every downstream claim in the body is left completely untouched.
- WEAK (out of scope, do NOT do this): rewriting the whole body paragraph in the hook's \
voice, adding a new stat to make the transition punchier, or cutting a sentence for \
pace. Any of these is a rewrite, not a copy-edit, and it destroys the guarantees \
(grounding, non-redundancy, pacing) that got this piece chosen by the checkers in the \
first place.

Return ONLY a JSON object of exactly this shape (no prose outside it):
{"revised_lines": {"<beat_index>": "<revised line>", ...}, \
"justification": "<1-2 sentences: what register clash you smoothed, at which beat(s), \
and how -- or why you left a line unchanged>"}
Keys of "revised_lines" MUST be exactly the beat index(es) given to you, as strings. Do \
not add, omit, or renumber keys. No prose outside the JSON.\
"""


# ===========================================================================
# Deterministic constraint check (pure code, no LLM).
# ===========================================================================

_NUM_RE = re.compile(r"\d+")


def _check_constraints(
    original_candidate: MergeCandidate,
    patched_candidate: MergeCandidate,
    edited_indices: list[int],
) -> ConstraintCheck:
    """Deterministically verify a proposed edit stayed inside the Copy Editor's
    license. See the module docstring / class docstrings for the five checks;
    named here again for the reader following the code:

      (a) every beat NOT in `edited_indices` is byte-identical (line, t_start,
          t_end, role, source_variant_id) between original and patched.
      (b) each edited line's word count is within +/-10% of the original's.
      (c) the hook beat (role=='hook') and the CTA beat (role=='cta') are
          NEVER in `edited_indices` -- an immediate, hard violation regardless
          of what the LLM tried to return.
      (d) every t_start/t_end across ALL beats (edited or not) is
          byte-identical pre/post -- CE must never touch timing.
      (e) every numeric token in each ORIGINAL edited line still appears
          somewhere in the REVISED line (cheap proxy for "no dropped claim").

    Returns violations as clear, human-readable strings naming which check
    failed and for which beat index, so `copy_edit_seams`'s one re-prompt can
    name the specific problem(s).
    """
    violations: list[str] = []
    orig_beats = original_candidate.merged_beats
    patched_beats = patched_candidate.merged_beats
    edited_set = set(edited_indices)

    if len(orig_beats) != len(patched_beats):
        violations.append(
            f"beat count changed: original had {len(orig_beats)} beats, "
            f"patched has {len(patched_beats)}"
        )
        return ConstraintCheck(passed=False, violations=violations)

    for idx in edited_set:
        if idx < 0 or idx >= len(orig_beats):
            violations.append(
                f"edited index {idx} is out of range (beats has {len(orig_beats)} entries)"
            )
            continue
        role = orig_beats[idx].role
        if role in ("hook", "cta"):
            violations.append(
                f"beat {idx} has role={role!r} -- the Copy Editor may never edit the "
                "hook or CTA beat, regardless of what the model returned"
            )

    for i, (ob, pb) in enumerate(zip(orig_beats, patched_beats)):
        # (d) timestamps must NEVER change, edited beat or not.
        if ob.t_start != pb.t_start or ob.t_end != pb.t_end:
            violations.append(
                f"beat {i} timestamp changed: original ({ob.t_start}, {ob.t_end}) -> "
                f"patched ({pb.t_start}, {pb.t_end}); CE must never touch timing"
            )

        if i in edited_set:
            if ob.role != pb.role:
                violations.append(f"beat {i} role changed from {ob.role!r} to {pb.role!r}")
            if ob.source_variant_id != pb.source_variant_id:
                violations.append(
                    f"beat {i} source_variant_id changed from {ob.source_variant_id!r} "
                    f"to {pb.source_variant_id!r}"
                )

            orig_wc = len(ob.line.split())
            new_wc = len(pb.line.split())
            lo, hi = orig_wc * 0.9, orig_wc * 1.1
            if not (lo - 1e-9 <= new_wc <= hi + 1e-9):
                violations.append(
                    f"beat {i} revised word count {new_wc} is outside +/-10% of the "
                    f"original's {orig_wc} words (allowed [{lo:.2f}, {hi:.2f}])"
                )

            orig_nums = set(_NUM_RE.findall(ob.line))
            new_nums = set(_NUM_RE.findall(pb.line))
            missing = sorted(orig_nums - new_nums)
            if missing:
                violations.append(
                    f"beat {i} dropped numeric token(s) {missing} present in the "
                    "original line -- a possible dropped claim"
                )
        else:
            # (a) non-edited beats must be byte-identical in every field.
            if (
                ob.line != pb.line
                or ob.role != pb.role
                or ob.source_variant_id != pb.source_variant_id
            ):
                violations.append(
                    f"beat {i} was not named as edited but changed anyway "
                    f"(original={ob.model_dump()!r}, patched={pb.model_dump()!r})"
                )

    return ConstraintCheck(passed=(len(violations) == 0), violations=violations)


# ===========================================================================
# LLM payload + response parsing.
# ===========================================================================


def _hook_and_cta_indices(beats) -> tuple[int, int]:
    """Locate the hook beat and the CTA beat by role (defensive: convention is
    beats[0]/beats[-1], but this reads the actual role field rather than
    assuming position)."""
    hook_idx = next((i for i, b in enumerate(beats) if b.role == "hook"), 0)
    cta_idx = next(
        (i for i, b in enumerate(beats) if b.role == "cta"), len(beats) - 1
    )
    return hook_idx, cta_idx


def _seam_payload(
    candidate: MergeCandidate,
    seam_flags: list[SeamFlag],
    hook_idx: int,
    cta_idx: int,
) -> list[dict]:
    beats = candidate.merged_beats
    payload = []
    for f in seam_flags:
        neighbor_line = beats[hook_idx].line if f.seam == "hook_body" else beats[cta_idx].line
        payload.append(
            {
                "seam": f.seam,
                "editable_beat_index": f.editable_beat_index,
                "current_line": beats[f.editable_beat_index].line,
                "neighbor_line": neighbor_line,
                "evidence": f.evidence,
            }
        )
    return payload


def _build_user_prompt(
    seam_payload: list[dict],
    coherence_justification: str,
    grounding_truth_ids: list[str],
    violation_note: Optional[str],
) -> str:
    payload = {
        "seams": seam_payload,
        "coherence_justification": coherence_justification,
        "grounding_truth_ids": grounding_truth_ids,
    }
    prompt = (
        "Repair ONLY the seam line(s) named below. Return a revised line for EACH "
        "beat index listed under \"seams\" and nothing else.\n\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
    )
    if violation_note:
        prompt += (
            "\n\nYour previous attempt violated your license in the following way(s); "
            "fix them and stay strictly within your license this time:\n" + violation_note
        )
    return prompt


def _parse_llm_output(
    raw: dict,
    requested_indices: list[int],
    n_beats: int,
) -> tuple[dict[int, str], str]:
    """Validate the raw LLM JSON into {beat_index: revised_line}, justification.

    Raises ValueError on any structural problem (missing keys, non-integer or
    out-of-range beat index, empty line/justification, a requested index with
    no revision) -- the caller treats this the same as a constraint-check
    failure for retry purposes.
    """
    revised_lines = raw.get("revised_lines")
    if not isinstance(revised_lines, dict):
        raise ValueError("copy editor response is missing a 'revised_lines' object")

    justification = raw.get("justification")
    if not isinstance(justification, str) or not justification.strip():
        raise ValueError("copy editor response is missing a non-empty 'justification' string")

    revised_map: dict[int, str] = {}
    for key, value in revised_lines.items():
        try:
            idx = int(key)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"copy editor response has a non-integer beat index key {key!r}"
            ) from exc
        if idx < 0 or idx >= n_beats:
            raise ValueError(
                f"copy editor response names beat index {idx}, out of range for "
                f"{n_beats} beats"
            )
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"copy editor response has an empty/non-string revised line for beat {idx}"
            )
        revised_map[idx] = value

    missing = [i for i in requested_indices if i not in revised_map]
    if missing:
        raise ValueError(
            f"copy editor response is missing a revision for requested beat index(es) {missing}"
        )
    return revised_map, justification


# ===========================================================================
# Public API.
# ===========================================================================


def copy_edit_seams(
    candidate: MergeCandidate,
    seam_flags: list[SeamFlag],
    coherence_justification: str,
    grounding_truth_ids: list[str],
    *,
    model: Optional[str] = None,
) -> CopyEditResult:
    """Copy Editor (§5.4.8): constrained, seam-only repair of a merge candidate.

    Args:
        candidate:                the merge candidate CV flagged.
        seam_flags:                CV's SeamFlag(s) naming which beat(s) to repair
                                   (always the adjacent body beat, never hook/CTA).
        coherence_justification:  CV's cold-read justification naming the clash --
                                   this is CE's brief for what to fix.
        grounding_truth_ids:      union of grounding truth ids across the 3 source
                                   variants, for context only -- CE must not invent
                                   new claims even seeing these.
        model:                    optional model-id override; defaults to MODEL_TEXT.

    Returns:
        CopyEditResult. On success, `patched_candidate` differs from `candidate`
        only in the edited beats' `line` field and the derived `merged_text`; on
        a repair that still fails the deterministic constraint check after one
        re-prompt, `patched_candidate` is `candidate` itself, unmodified, with
        `constraint_check.passed=False` -- the caller (CV) is expected to treat
        that as a failed repair and route to fallback. This function does not
        raise for that case; a failed ConstraintCheck IS the signal.

    Raises:
        ValueError:    seam_flags is empty, or a seam names an out-of-range or
                       hook/CTA beat as editable (CV should never do this --
                       this is a defensive guard, not the expected path).
        QwenJSONError: the model returned non-JSON (from the shared helper);
                       this is a transport/content failure, not a repair
                       failure, so it is not swallowed into a ConstraintCheck.
    """
    if not seam_flags:
        raise ValueError("copy_edit_seams: seam_flags must be non-empty")

    beats = candidate.merged_beats
    n_beats = len(beats)
    for f in seam_flags:
        idx = f.editable_beat_index
        if idx < 0 or idx >= n_beats:
            raise ValueError(
                f"copy_edit_seams: seam names editable_beat_index={idx}, out of range "
                f"for {n_beats} beats"
            )
        if beats[idx].role in ("hook", "cta"):
            raise ValueError(
                f"copy_edit_seams: seam names beat {idx} (role={beats[idx].role!r}) as "
                "editable, but the hook/CTA beat may never be a Copy Editor target -- "
                "this indicates a bug upstream in the Merge Coherence Validator"
            )

    hook_idx, cta_idx = _hook_and_cta_indices(beats)
    requested_indices = [f.editable_beat_index for f in seam_flags]
    original_lines = {idx: beats[idx].line for idx in set(requested_indices)}
    original_seam_text = "\n".join(original_lines[i] for i in requested_indices)

    seam_payload = _seam_payload(candidate, seam_flags, hook_idx, cta_idx)

    violation_note: Optional[str] = None
    last_justification = ""
    last_revised_map: dict[int, str] = {}
    last_check = ConstraintCheck(passed=False, violations=["copy edit never attempted"])
    last_patched_candidate = candidate

    for attempt in range(2):  # initial attempt + one re-prompt, per §5.4.7/§5.4.8 convention
        user_prompt = _build_user_prompt(
            seam_payload, coherence_justification, grounding_truth_ids, violation_note
        )
        try:
            raw = call_qwen_json(_COPY_EDITOR_SYSTEM_PROMPT, user_prompt, model=model)
            revised_map, justification = _parse_llm_output(raw, requested_indices, n_beats)
        except ValueError as exc:
            logger.warning("Copy Editor: attempt %d produced malformed output: %s", attempt + 1, exc)
            last_justification = str(exc)
            last_revised_map = {}
            last_patched_candidate = candidate
            last_check = ConstraintCheck(passed=False, violations=[f"attempt {attempt + 1}: {exc}"])
            violation_note = str(exc)
            continue

        patched_beats = list(beats)
        for idx, line in revised_map.items():
            patched_beats[idx] = patched_beats[idx].model_copy(update={"line": line})
        merged_text = " ".join(b.line for b in patched_beats)
        patched_candidate = candidate.model_copy(
            update={"merged_beats": patched_beats, "merged_text": merged_text}
        )
        check = _check_constraints(candidate, patched_candidate, sorted(revised_map.keys()))

        last_justification = justification
        last_revised_map = revised_map
        last_patched_candidate = patched_candidate
        last_check = check

        if check.passed:
            break
        logger.info(
            "Copy Editor: attempt %d failed constraint check: %s", attempt + 1, check.violations
        )
        violation_note = "; ".join(check.violations)

    revised_seam_text = "\n".join(
        last_revised_map.get(i, original_lines[i]) for i in requested_indices
    )

    if not last_check.passed:
        return CopyEditResult(
            patched_candidate=candidate,
            seams_edited=[],
            original_seam_text=original_seam_text,
            revised_seam_text=revised_seam_text,
            justification=last_justification or "copy edit failed to produce a valid repair",
            constraint_check=last_check,
        )

    return CopyEditResult(
        patched_candidate=last_patched_candidate,
        seams_edited=sorted(last_revised_map.keys()),
        original_seam_text=original_seam_text,
        revised_seam_text=revised_seam_text,
        justification=last_justification,
        constraint_check=last_check,
    )


# ===========================================================================
# LangGraph node wrapper.
# ===========================================================================


async def copy_editor_node(state: ProductCutState, config: RunnableConfig) -> dict:  # noqa: ARG001
    """LangGraph node wrapper for the Copy Editor (§5.4.8), following
    meta_critic_node's / body_checker_node's style (asyncio.to_thread around
    the sync/blocking `copy_edit_seams` call).

    STATE-KEY ASSUMPTIONS -- merge_validator_node was not finished (only the
    four locked Pydantic models existed) when this file was written, so the
    exact keys `merge_validator.py` reads/writes are unconfirmed. This node
    assumes:

      READS:
        - state["pending_merge_candidate"]: dict shape of a `MergeCandidate` --
          the merge candidate CV just evaluated and flagged voice/register on.
        - state["coherence_validation_result"]: dict shape of a
          `CoherenceValidationResult` -- specifically its `seam_flags` and
          `justification` fields.
        - state["script_variants"]: to recover `grounding_truth_ids` for the
          three source variants (hook/body/cta) referenced by the candidate.

      WRITES:
        - state["pending_merge_candidate"]: REPLACED with the patched candidate
          (`result.patched_candidate.model_dump()`) so CV's next call
          re-validates the repair, per the `CE --> CV` edge in §4.
        - state["last_copy_edit"]: the full `CopyEditResult.model_dump()`. This
          is a dedicated, clearly-documented key rather than a guessed append
          into `merge_attempts[...].copy_edit` (§6) -- the orchestrating
          session should fold this into the right `merge_attempts` entry
          during final graph wiring, once merge_validator.py's actual list-
          append mechanics are known, rather than risk silently corrupting a
          list another node also appends to.

    If `merge_validator_node`'s real key names differ, only the state.get(...)
    calls below and the two dict keys in the return value need reconciling --
    `copy_edit_seams` itself has no state coupling at all.
    """
    candidate_raw = state.get("pending_merge_candidate")
    if candidate_raw is None:
        raise ValueError(
            "copy_editor_node: no state['pending_merge_candidate'] to repair -- this "
            "node must run only after the Merge Coherence Validator flags a "
            "voice/register seam failure on a merge candidate"
        )
    candidate = (
        candidate_raw
        if isinstance(candidate_raw, MergeCandidate)
        else MergeCandidate.model_validate(candidate_raw)
    )

    cv_result_raw = state.get("coherence_validation_result") or {}
    seam_flags_raw = cv_result_raw.get("seam_flags") or []
    seam_flags = [
        f if isinstance(f, SeamFlag) else SeamFlag.model_validate(f) for f in seam_flags_raw
    ]
    if not seam_flags:
        raise ValueError(
            "copy_editor_node: state['coherence_validation_result']['seam_flags'] is "
            "empty -- Copy Editor was invoked without a voice/register seam to repair"
        )
    coherence_justification = cv_result_raw.get("justification", "")

    variants = state.get("script_variants") or []
    source_ids = {
        candidate.hook_source_variant_id,
        candidate.body_source_variant_id,
        candidate.cta_source_variant_id,
    }
    grounding_truth_ids = sorted(
        {
            tid
            for v in variants
            if v.get("variant_id") in source_ids
            for tid in (v.get("grounding_truth_ids") or [])
        }
    )

    result = await asyncio.to_thread(
        copy_edit_seams,
        candidate,
        seam_flags,
        coherence_justification,
        grounding_truth_ids,
    )

    trace_note = (
        f"\n[copy_editor] seams_edited={result.seams_edited}; "
        f"constraint_check.passed={result.constraint_check.passed}"
        + (
            f"; violations={result.constraint_check.violations}"
            if not result.constraint_check.passed
            else ""
        )
        + f"; justification={result.justification}"
    )

    return {
        "pending_merge_candidate": result.patched_candidate.model_dump(),
        "last_copy_edit": result.model_dump(),
        "reasoning_trace": state.get("reasoning_trace", "") + trace_note,
    }


__all__ = [
    "ConstraintCheck",
    "CopyEditResult",
    "copy_edit_seams",
    "copy_editor_node",
]
