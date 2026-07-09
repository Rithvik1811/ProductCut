"""
CTA-Checker (§5.4.4) and Tone-Checker (§5.4.5) of the Critic Chain.

Spec of record: docs/TECHNICAL_DOCUMENTATION.md §5.4.4 / §5.4.5. These are two of
the five parallel specialist checkers that score the Concept Agent's four
structurally-distinct `ScriptVariant`s (§5.3) along orthogonal axes, before the
Meta-Critic (§5.4.6) reconciles them into a cross-pollinated merge candidate.

Both checkers live in one module because they share *all* of their calling and
validation infrastructure (see `critic_llm.call_qwen_json`) and differ only in
rubric and output shape — but they remain two independent public functions
(`check_cta`, `check_tone`) because they are conceptually distinct scoring axes
that the Meta-Critic weights separately (CTA 20%, Tone 15%) and that run in
parallel, not in sequence.

Scope note — what these are NOT:
  * WIRED into the live LangGraph graph (backend/graph/build.py): fan out from
    concept_agent in parallel with Hook-/Pacing-/Body-Checker, fanning back in
    to meta_critic. Were standalone, independently-callable functions before
    the Concept Agent existed; that follow-up wiring has since landed.
  * NOT the other checkers. Hook (§5.4.1), Pacing (§5.4.2), Body (§5.4.3),
    Meta-Critic (§5.4.6), Merge Coherence Validator (§5.4.7) and Copy Editor
    (§5.4.8) are separate tasks and are deliberately out of scope here.
  * They do NOT populate `graph.state.CriticScore` directly. Each checker returns
    only its own axis (cta / tone + never_do_violation); the Meta-Critic is what
    assembles the full CriticScore composite. We import the C1 `ScriptVariant` /
    `SellerDirection` types (never redefine them) for input typing only.

Design of the output validation: following the precedent set by
`graph.shot_schema` (a `TypedDict` is a shape hint with *no* runtime checking, so
raw LLM JSON gets a real Pydantic model gate before it is trusted), each checker's
per-variant result is validated by a small Pydantic model here — score in the 1-5
range, justification non-empty, and (critically for Tone) `never_do_violation` an
actual `bool`, not a truthy string the LLM happened to emit. The hard `never_do`
gate that the Meta-Critic relies on (§5.4.6: "Any variant with never_do_violation
= true is excluded ... regardless of composite score") is only as trustworthy as
this bool being a real bool, which is exactly why it is validated rather than
passed through raw.
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, StrictBool, ValidationError

from agents.critic_llm import call_qwen_json
from graph.state import ProductCutState, ScriptVariant, SellerDirection

logger = logging.getLogger("productcut.critics.cta_tone")


# ---------------------------------------------------------------------------
# Output schemas — the runtime gate on raw LLM JSON (shot_schema.py precedent).
# ---------------------------------------------------------------------------


class CtaCheckResult(BaseModel):
    """Validated CTA-Checker output for a single variant (§5.4.4).

    `cta_score` is the raw 1-5 rubric integer. It maps onto the Meta-Critic's
    `CriticScore.cta` axis (a float there) — the Meta-Critic does the cast/weight,
    this checker only produces the discrete rubric score + its justification.
    """

    model_config = ConfigDict(extra="forbid")

    cta_score: int = Field(..., ge=1, le=5)
    justification: str = Field(..., min_length=1)


class ToneCheckResult(BaseModel):
    """Validated Tone-Checker output for a single variant (§5.4.5).

    `never_do_violation` is the enforceable half of the `seller_direction.never_do`
    constraint: this flag is what the Meta-Critic (§5.4.6) reads to hard-exclude a
    variant *before* weighting, regardless of composite score. It is typed as a
    strict `bool` so a stray "true"/"yes" string can never masquerade as the gate.
    """

    model_config = ConfigDict(extra="forbid")

    tone_score: int = Field(..., ge=1, le=5)
    justification: str = Field(..., min_length=1)
    # StrictBool, not plain `bool`: Pydantic v2's lax mode coerces the strings
    # "true"/"yes"/"1" to True, which would let a string masquerade as the hard
    # gate this docstring promises is a real JSON boolean. StrictBool rejects
    # anything that is not an actual bool.
    never_do_violation: StrictBool


# ---------------------------------------------------------------------------
# Prompts — example-anchored rubrics, matching the Hook-Checker's stated style
# ("the rubric ships example-anchored (weak vs. strong exemplars) so scores are
# calibrated rather than arbitrary", §5.4.1).
# ---------------------------------------------------------------------------

_CTA_SYSTEM_PROMPT = """\
You are a direct-response advertising critic. Score ONLY the call-to-action (CTA) \
clarity of each short-form video ad script, on an integer scale of 1 to 5.

A CTA is the closing ask that tells the viewer exactly what to do next. Judge \
clarity and singularity, NOT how exciting the product is.

Rubric (calibrated with exemplars):
- 5 — One concrete action verb + one specific destination, unmistakable.
      STRONG: "Tap to shop the autumn set."  /  "Swipe up to grab yours today."
- 4 — Single clear ask, slightly generic destination or verb.
      e.g. "Shop now at the link below." (clear verb, vague-ish destination)
- 3 — Present but soft/vague; the viewer knows there is an ask but not a crisp
      next step. e.g. "Check it out." / "Learn more." (no destination, weak verb)
- 2 — Vague AND weak, or buried so the ask is barely a CTA at all.
- 1 — MISSING (script just ends), OR two or more COMPETING calls-to-action that
      split the viewer's attention.
      WEAK (competing): "Tap to shop the set — or visit our site to book a
      styling call, and don't forget to follow us!" (three asks = split intent)

HARD RULE: multiple competing CTAs must be penalised heavily (score 1-2), even if
each individual ask is well-phrased. Split calls-to-action measurably depress
conversion — one ad, one ask. Say so in the justification when you see it.

You are given, per variant: the CTA line (the last beat) and the closing beats
around it for context. Score the CTA, not the whole script.

Return ONLY a JSON object of this exact shape:
{"results": [{"variant_id": "<id>", "cta_score": <int 1-5>, "justification": "<one or two sentences naming the concrete reason>"}]}
Include exactly one entry per variant you were given. No prose outside the JSON.\
"""

_TONE_SYSTEM_PROMPT = """\
You are a brand-tone critic for short-form video ads. Score ONLY brand/tone FIT \
of each script against the seller's brief and stated direction, on an integer \
scale of 1 to 5, and separately flag any hard "never do" violation.

Judge whether the script's VOICE, register, and energy match what the brief and \
the seller_direction (especially its mood words) ask for. A technically strong ad \
can still score LOW on tone if it is tonally wrong for THIS brand.

Rubric (calibrated with exemplars). Suppose the brief asks for a "quiet, tactile, \
handmade feel" with mood words like "calm, sensory, understated":
- 5 — Fully on-voice. STRONG: "Run your thumb along the ridge where the potter's
      hand pressed the clay. It stays warm long after the last sip." (calm,
      sensory, unhurried — matches the mood words)
- 3 — Mixed: mostly on-voice but with an off-register line or two.
- 1 — Tonally opposite. WEAK: "LIMITED TIME MEGA SALE!! 50% OFF — BUY NOW BEFORE
      IT'S GONE!!!" (hard-sell, high-urgency, shouty — the exact opposite of a
      quiet, tactile, handmade feel), even if its hook would grab attention.

The mismatch direction matters: a hard-sell / high-urgency script scores LOW when
the brief/mood words call for something calmer, and vice-versa. Explain the
specific mismatch (or match) in the justification.

NEVER-DO HARD GATE:
- If seller_direction includes a "never_do" constraint, set "never_do_violation":
  true for ANY script that violates it — in letter OR in substance. Example: if
  never_do = "never mention discounts or sale pricing", then a script that says
  "20% off this week" or "grab it before the sale ends" violates it -> true.
- If there is no never_do constraint, or the script complies with it, set
  "never_do_violation": false.
- never_do_violation is INDEPENDENT of the tone_score: judge tone on its own
  merits, and set the violation flag on its own merits. (A script can be perfectly
  on-tone and still violate never_do, or be off-tone yet compliant.)
- This flag is a hard gate downstream: a true value removes the variant from
  consideration entirely, so only flag a genuine violation.

You are given: the brief, the seller_direction (may be null / may omit fields),
and each full script's text.

Return ONLY a JSON object of this exact shape:
{"results": [{"variant_id": "<id>", "tone_score": <int 1-5>, "never_do_violation": <true|false>, "justification": "<one or two sentences naming the concrete tone match/mismatch and, if flagged, the exact never_do violation>"}]}
Include exactly one entry per variant. "never_do_violation" MUST be a JSON boolean
(true/false), never a string. No prose outside the JSON.\
"""


# ---------------------------------------------------------------------------
# Input serialisation helpers.
# ---------------------------------------------------------------------------


def _cta_payload_for_variant(variant: ScriptVariant, n_closing: int = 3) -> dict:
    """Extract the CTA line (last beat) + surrounding closing beats for one variant.

    Per §5.4.4 the CTA-Checker's input is "each script's CTA line (last beat) and
    surrounding closing beats" — not the whole script. We send the last
    `n_closing` beats so the model can judge the CTA in its closing context (and
    catch a second competing ask hiding one beat earlier) without being distracted
    by the hook/body.
    """
    beats = variant.get("beats") or []
    closing = beats[-n_closing:] if beats else []
    return {
        "variant_id": variant["variant_id"],
        "cta_line": (beats[-1]["line"] if beats else ""),
        "closing_beats": [b["line"] for b in closing],
    }


def _tone_payload_for_variant(variant: ScriptVariant) -> dict:
    """Give the Tone-Checker the FULL script text (§5.4.5 input: 'each full script').

    Tone/register is a whole-script property (and a never_do violation can appear
    anywhere), so unlike the CTA-Checker this one does not trim to the close.
    Falls back to joining beat lines if `text` is empty.
    """
    text = variant.get("text") or ""
    if not text:
        text = " ".join(b["line"] for b in (variant.get("beats") or []))
    return {"variant_id": variant["variant_id"], "script_text": text}


def _validate_results(
    raw: dict,
    model_cls: type[BaseModel],
    expected_ids: set[str],
    axis: str,
) -> dict[str, BaseModel]:
    """Validate the LLM's {"results": [...]} envelope into {variant_id: model}.

    Enforces: one result per expected variant, no extras, and each entry passing
    its Pydantic model (score in range, fields present, correct types). Raises a
    ValueError describing the exact structural problem — the caller decides
    whether to re-prompt (the pipeline's standard re-prompt-once policy) or crash.
    """
    results = raw.get("results")
    if not isinstance(results, list):
        raise ValueError(
            f"{axis}-Checker: expected a JSON object with a 'results' list, got keys "
            f"{list(raw.keys())!r}"
        )

    validated: dict[str, BaseModel] = {}
    for entry in results:
        if not isinstance(entry, dict) or "variant_id" not in entry:
            raise ValueError(f"{axis}-Checker: result entry missing variant_id: {entry!r}")
        vid = entry["variant_id"]
        if vid in validated:
            raise ValueError(
                f"{axis}-Checker: duplicate result entry for variant {vid!r} "
                "(expected exactly one entry per variant)"
            )
        payload = {k: v for k, v in entry.items() if k != "variant_id"}
        try:
            validated[vid] = model_cls.model_validate(payload)
        except ValidationError as exc:
            raise ValueError(
                f"{axis}-Checker: result for variant {vid!r} failed validation: {exc}"
            ) from exc

    got_ids = set(validated.keys())
    if got_ids != expected_ids:
        missing = expected_ids - got_ids
        extra = got_ids - expected_ids
        raise ValueError(
            f"{axis}-Checker: variant id mismatch. missing={sorted(missing)} "
            f"extra={sorted(extra)}"
        )
    return validated


# ---------------------------------------------------------------------------
# Public API — the two checkers.
# ---------------------------------------------------------------------------


def check_cta(
    variants: list[ScriptVariant],
    *,
    model: Optional[str] = None,
) -> dict[str, dict]:
    """CTA-Checker (§5.4.4): score call-to-action clarity for each variant.

    Args:
        variants: the Concept Agent's script variants (typically four).
        model:    optional model-id override; defaults to MODEL_TEXT (.env).

    Returns:
        {variant_id: {"cta_score": int 1-5, "justification": str}} — validated.

    Raises:
        ValueError:    if the model output fails structural validation.
        QwenJSONError: if the model returns non-JSON (from the shared helper).
    """
    if not variants:
        return {}

    payload = [_cta_payload_for_variant(v) for v in variants]
    user_prompt = (
        "Score the CTA of each of these ad-script variants.\n\n"
        + json.dumps({"variants": payload}, ensure_ascii=False, indent=2)
    )
    logger.info("CTA-Checker scoring %d variant(s)", len(variants))

    raw = call_qwen_json(_CTA_SYSTEM_PROMPT, user_prompt, model=model)
    expected = {v["variant_id"] for v in variants}
    validated = _validate_results(raw, CtaCheckResult, expected, axis="CTA")
    return {vid: res.model_dump() for vid, res in validated.items()}


def check_tone(
    brief: str,
    seller_direction: Optional[SellerDirection],
    variants: list[ScriptVariant],
    *,
    model: Optional[str] = None,
) -> dict[str, dict]:
    """Tone-Checker (§5.4.5): score brand/tone fit + enforce the never_do hard gate.

    Args:
        brief:            the seller's one-line brief.
        seller_direction: optional direction (mood_words, never_do, ...). May be
                          None; may omit any field.
        variants:         the Concept Agent's script variants (typically four).
        model:            optional model-id override; defaults to MODEL_TEXT.

    Returns:
        {variant_id: {"tone_score": int 1-5, "justification": str,
                      "never_do_violation": bool}} — validated.

    Raises:
        ValueError:    if the model output fails structural validation.
        QwenJSONError: if the model returns non-JSON (from the shared helper).
    """
    if not variants:
        return {}

    payload = [_tone_payload_for_variant(v) for v in variants]
    # seller_direction is a TypedDict (plain dict at runtime) or None; pass it
    # through as-is so the model sees exactly the mood_words / never_do / freeform
    # the seller supplied (or explicit null when there is no direction).
    user_prompt = (
        "Score the brand/tone fit of each ad-script variant against the brief and "
        "seller_direction, and flag any never_do violation.\n\n"
        + json.dumps(
            {
                "brief": brief,
                "seller_direction": seller_direction,
                "variants": payload,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    logger.info(
        "Tone-Checker scoring %d variant(s); never_do set=%s",
        len(variants),
        bool(seller_direction and seller_direction.get("never_do")),
    )

    raw = call_qwen_json(_TONE_SYSTEM_PROMPT, user_prompt, model=model)
    expected = {v["variant_id"] for v in variants}
    validated = _validate_results(raw, ToneCheckResult, expected, axis="Tone")
    return {vid: res.model_dump() for vid, res in validated.items()}


async def cta_checker_node(state: ProductCutState) -> dict:
    """LangGraph node wrapper. check_cta is sync + blocking network I/O -> asyncio.to_thread."""
    scores = await asyncio.to_thread(check_cta, state["script_variants"])
    return {"cta_scores": scores}


async def tone_checker_node(state: ProductCutState) -> dict:
    """LangGraph node wrapper. check_tone is sync + blocking network I/O -> asyncio.to_thread."""
    scores = await asyncio.to_thread(
        check_tone, state["brief"], state.get("seller_direction"), state["script_variants"]
    )
    return {"tone_scores": scores}


__all__ = [
    "CtaCheckResult",
    "ToneCheckResult",
    "check_cta",
    "check_tone",
]
