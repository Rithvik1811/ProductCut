"""
Justification Validator — deterministic, shared (Phase 2).
Spec of record: docs/TECHNICAL_DOCUMENTATION.md §5.5 (Treatment Agent's
per-beat validation, extracted out of agents/treatment_agent.py here) and
§5.6's "Interface handoff to the Justification Validator" (Shot-List Agent's
per-shot ShotJustification validation, RR's caller, built separately).

Extracted so both callers share one implementation instead of each
duplicating the same grounding checks. Deliberately field-presence-driven,
not caller-specific: a justification dict is validated on whichever of
{script_quote, truth_fact_id, treatment_ref, beat_function} keys it actually
carries, so a Treatment Agent `BeatTreatment` entry (has script_quote/
truth_fact_id/beat_function, never treatment_ref) and a Shot-List Agent
`ShotJustification` (has script_quote/truth_fact_id/treatment_ref, never
beat_function) both validate correctly through the same function without
hardcoding either caller's exact shape.

Each justification gets ONE `violation` (a fixed-vocabulary string or None),
not an exhaustive list of every problem wrong with it -- checks run in a
fixed priority order (quote -> truth_id -> treatment_ref -> beat_function ->
stoplist) and the first failure wins. That's narrower than what
treatment_agent.py's superseded internal `_validate_beat` used to return (a
list of every problem string for a beat); a caller that wants a richer,
multi-issue re-prompt message builds one itself from the single violation
code plus its own knowledge of the justification's contents (see
treatment_agent.py's `_reprompt_message` for the pattern) -- re-validating
after a fix will surface the next problem, if any, on the following call.

version: 1 (new module, not yet part of the C1/C2 frozen-schema versioning --
this is a plain deterministic helper, not shared LangGraph state).
"""
from __future__ import annotations

from typing import Optional, TypedDict

from graph.state import ProductTruth, Treatment, WinningScript

# Fixed violation vocabulary -- callers branch on these exact strings for their
# own re-prompt/fallback logic, so this must never change silently; adding a
# new violation type is a contract change requiring a KR/RR sync, like any
# other shared-interface change in this codebase.
VIOLATION_QUOTE_MISMATCH = "quote_mismatch"
VIOLATION_UNKNOWN_TRUTH_ID = "unknown_truth_id"
VIOLATION_TREATMENT_REF_INVALID = "treatment_ref_invalid"
VIOLATION_STOPLIST_HIT = "stoplist_hit"
VIOLATION_INVALID_BEAT_FUNCTION = "invalid_beat_function"

# C1's frozen enum (graph.state.BeatTreatment.beat_function / Shot.beat_role).
# Only checked when a justification dict actually carries a beat_function key
# -- Shot-List Agent's ShotJustification objects never do.
BEAT_FUNCTIONS = ("hook", "problem", "demo", "proof", "cta")

# Spec: "the word 'category' ... disallowed" (§5.5), checked against any
# free-text field present. Public (not underscore-prefixed) so callers with
# their own adjacent banned-word checks on OTHER fields (e.g. Treatment
# Agent's top-level director_persona/color_story/pacing_philosophy, which
# aren't per-justification fields and so aren't covered by this function)
# can reuse the same literal instead of re-declaring it.
BANNED_WORD = "category"

# Doc's own two example generic phrases for the Shot-List Agent's
# Justification Validator (§5.6): "show the product clearly," "highlight
# quality." Cheap deterministic reject, same idea as
# product_truth_extractor.py's _GENERIC_STOPLIST, scoped to justification
# free text rather than product-truth facts.
_GENERIC_STOPLIST = (
    "show the product clearly",
    "highlight quality",
)

# Free-text fields checked for the banned word / stoplist, whichever is
# present. "justification" covers a hypothetical future caller that names its
# free-text field that generically; Treatment Agent uses why_not_generic and
# visual_approach.
_FREE_TEXT_FIELDS = ("why_not_generic", "visual_approach", "justification")


class ValidationResult(TypedDict):
    shot_id_or_beat_index: object  # whatever identifier the input dict carried (str shot_id or int beat_index), echoed back unchanged
    passed: bool
    violation: Optional[str]


def _is_verbatim_substring(quote: str, haystack: str) -> bool:
    return bool(quote) and quote.lower() in haystack.lower()


def _stoplist_hit(text: str) -> bool:
    lowered = text.lower()
    if BANNED_WORD in lowered:
        return True
    return any(phrase in lowered for phrase in _GENERIC_STOPLIST)


def validate_justifications(
    justifications: list[dict],
    winning_script: WinningScript,
    product_truths: list[ProductTruth],
    treatment: Optional[Treatment],
) -> list["ValidationResult"]:
    """Validate a list of justification-shaped dicts against shared job context.

    Each dict is checked only on the fields it actually carries:
    - `script_quote` (if present): must be a verbatim, case-insensitive
      substring of `winning_script["text"]`.
    - `truth_fact_id` (if present): must exist in `product_truths`.
    - `treatment_ref` (if present): must match a `beat_index` in
      `treatment["beat_treatments"]`. `treatment` may be None (e.g. the
      Treatment Agent calling this on its own beats, before any Treatment
      exists yet) -- harmless as long as none of the input dicts carry a
      treatment_ref key in that case, which is always true for BeatTreatment
      entries.
    - `beat_function` (if present): must be one of the 5-value BEAT_FUNCTIONS
      enum.
    - any of `why_not_generic` / `visual_approach` / `justification` (if
      present): run the banned-word/generic-phrase stoplist.

    Checks run in that fixed order; the first failing check wins as the
    reported `violation`. Returns one ValidationResult per input
    justification, in the same order, with `shot_id_or_beat_index` read from
    either a `shot_id` or `beat_index` key on the input dict (whichever is
    present) so callers can match results back to their own inputs by
    identifier as well as by position.
    """
    truth_ids = {t["truth_id"] for t in product_truths}
    treatment_refs = {bt["beat_index"] for bt in treatment["beat_treatments"]} if treatment else set()
    script_text = winning_script["text"]

    results: list[ValidationResult] = []
    for j in justifications:
        identifier = j.get("shot_id", j.get("beat_index"))
        violation: Optional[str] = None

        if violation is None and "script_quote" in j:
            if not _is_verbatim_substring(j.get("script_quote") or "", script_text):
                violation = VIOLATION_QUOTE_MISMATCH

        if violation is None and "truth_fact_id" in j:
            if j.get("truth_fact_id") not in truth_ids:
                violation = VIOLATION_UNKNOWN_TRUTH_ID

        if violation is None and "treatment_ref" in j:
            if j.get("treatment_ref") not in treatment_refs:
                violation = VIOLATION_TREATMENT_REF_INVALID

        if violation is None and "beat_function" in j:
            if j.get("beat_function") not in BEAT_FUNCTIONS:
                violation = VIOLATION_INVALID_BEAT_FUNCTION

        if violation is None:
            for field in _FREE_TEXT_FIELDS:
                text = j.get(field)
                if text and _stoplist_hit(text):
                    violation = VIOLATION_STOPLIST_HIT
                    break

        results.append(
            ValidationResult(
                shot_id_or_beat_index=identifier,
                passed=violation is None,
                violation=violation,
            )
        )
    return results
