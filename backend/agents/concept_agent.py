"""
Concept Agent — Qwen-Max via DashScope (Phase 1).
Spec of record: docs/TECHNICAL_DOCUMENTATION.md §5.3.

Output shape is constrained to the frozen C1 contract (`graph.state.ScriptVariant`):
{variant_id, text, framework, hook_type, emotional_trigger, grounding_truth_ids,
beats, target_length_sec}. `framework` is C1's 4-value enum, not a free-text
list -- if that enum ever needs a new value, that's a C1 change requiring a
KR/RR sync and a version bump in graph/state.py, not a unilateral addition here.

KNOWN GAP: C1 has no top-level `target_length_sec` job field (the spec assumes
the job carries a 15s/30s preference, but that was never added to the frozen
schema). Defaulting to 15s here rather than inventing a new state field
unilaterally -- raise with RR if a real per-job length preference is wanted.
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Optional

from openai import AsyncOpenAI

from agents._retry import create_completion
from graph.state import ProductCutState, ProductTruth, ScriptVariant

logger = logging.getLogger("productcut.agents.concept_agent")

DEFAULT_TARGET_LENGTH_SEC = 15
REQUIRED_VARIANT_COUNT = 4
MIN_VARIANTS_AFTER_DEGRADE = 2
MIN_GROUNDING_TRUTH_IDS = 2
MAX_HOOK_WORDS = 10

# The two graph.state.ProductTruth categories that are near-impossible to
# guess without actually looking at the photos -- a real anti-genericness
# lever, not just "cite 2 facts, any 2 facts." See module docstring.
SPECIFIC_CATEGORIES = frozenset({"imperfection", "construction_detail"})

# C1's frozen enum (graph.state.ScriptVariant.framework) -- exactly 4 values,
# exactly 4 variants required, so "distinct framework per variant" reduces to
# "each of these 4 used exactly once."
FRAMEWORKS = ("hook_problem_product_cta", "PAS", "AIDA", "BAB")

HOOK_TYPES = (
    "pattern interrupt", "bold claim", "curiosity gap", "direct address",
    "contrarian / myth-busting", "social proof", "POV", "before/after",
    "price anchor", "FOMO / urgency", "how-to",
)

EMOTIONAL_TRIGGERS = (
    "curiosity", "recognition", "FOMO", "tribal identity",
    "transformation / aspiration", "relief",
)

# Cheap proxy for "does this hook actually land a number or a contrast,"
# per the spec's own strong-hook exemplar ("Your coffee is cold in 12
# minutes. Mine isn't." -- a number, then a contrast). Not a full rhetorical
# quality scorer -- that's the Hook-Checker's job -- just a floor that
# rejects a pure curiosity tease with no payoff at all.
_CONTRAST_MARKERS = (
    "but", "not", "unlike", "instead", "isn't", "won't", "never", "yet ",
    "while others", "no more", "without",
)


def _format_truths(truths: list[ProductTruth]) -> str:
    return "\n".join(f"- [{t['truth_id']}] ({t['category']}) {t['fact']}" for t in truths)


def _build_system_prompt(target_length_sec: int) -> str:
    return f"""You are a creative director writing short-form ad video scripts ({target_length_sec}
seconds) for e-commerce product ads.

You will receive:
- A one-line seller brief
- A list of specific product truths (grounded facts about the product), each with a truth_id
- Optional seller direction (mood words, a reference ad link, "never do"
  constraints, freeform notes)

Write exactly 4 DISTINCT script variants. Each of the 4 must use a DIFFERENT one
of these exact framework values (use each exactly once, spelled exactly as shown):
{', '.join(FRAMEWORKS)}

Each variant must also use a distinct hook type (the first 2-3 seconds' angle)
and a distinct emotional trigger -- no two variants may share either. Prefer
these vocabularies (you may pick something else if it genuinely fits better,
but it must still be distinct across variants):
- Hook types: {', '.join(HOOK_TYPES)}
- Emotional triggers: {', '.join(EMOTIONAL_TRIGGERS)}

Grounding (mandatory):
- Every claim or visual detail in the script must trace back to a specific
  truth_id from the list you were given. Do not invent details not present
  in the provided truths.
- Each variant must cite AT LEAST 2 different truth_ids in "grounding_truth_ids".
- Each variant must cite AT LEAST ONE truth whose category is
  "imperfection" or "construction_detail" -- these are the specific,
  idiosyncratic details a generic description of this kind of product could
  never predict (a scuff mark, an odd cutout, a specific hinge mechanism).
  Do not build a variant only from generic material/color/dimension facts;
  those are true of many similar products and produce generic-sounding copy.

Per-variant requirements:
- The HOOK LINE must directly reference or paraphrase a specific, unusual
  truth -- not a generic industry pain point that could be reskinned onto
  any similar product by swapping the brand name. AVOID stock ad openers
  like "tired of flimsy X," "X is ruining your Y," "say goodbye to X" unless
  the sentence that follows immediately anchors it in a concrete visual
  detail from the truths. If your hook could apply to a *different* product
  in the same rough category with only the brand name changed, rewrite it.
- HOOK STRENGTH: a strong hook does at least one of these -- (a) names a
  specific pain tied to a concrete detail, (b) cites an exact number or
  measurement, (c) makes a contrarian or surprising claim that resolves in
  the same line via contrast (a "but"/"not"/"unlike"/"instead" turn, not a
  dangling question with no payoff). Calibration example: "Check out this
  amazing mug" is a WEAK hook (generic, no claim). "Your coffee is cold in
  12 minutes. Mine isn't." is a STRONG hook (a number, then a contrast that
  resolves it). A mere curiosity tease with no number and no contrast (e.g.
  "Look closely at this detail...") is weaker than either of the above --
  prefer a hook that also lands a number or a contrast, not just a tease.
- A single named pain point (not a vague benefit).
- A hook line of {MAX_HOOK_WORDS} words or fewer.
- Exactly one CTA verb -- never two competing calls to action.
- Beat-level timestamps: break the script into small beats (NOT 3 coarse
  hook/body/cta buckets) -- a new beat roughly every 2-3 seconds for the first
  2-3 beats, then every 3-5 seconds after that. Beats must be contiguous and
  sum to exactly {target_length_sec} seconds. Each beat's "line" is the actual
  script text spoken/shown during that beat.

If seller_direction includes "never_do" constraints, do not violate them in
any variant. If mood words are present, let them bias framework/tone choice.

Return ONLY valid JSON in this exact shape, no preamble or commentary:

{{
  "script_variants": [
    {{
      "variant_id": "v1",
      "text": "full script text",
      "framework": "one of: {' | '.join(FRAMEWORKS)}",
      "hook_type": "the hook angle",
      "emotional_trigger": "the primary emotional trigger",
      "grounding_truth_ids": ["t1", "t3"],
      "beats": [
        {{"t_start": 0, "t_end": 3, "line": "the hook line"}},
        {{"t_start": 3, "t_end": 6, "line": "next beat's line"}}
      ],
      "target_length_sec": {target_length_sec}
    }}
  ]
}}"""


def _build_user_content(
    brief: str,
    product_truths: list[ProductTruth],
    seller_direction: Optional[dict],
) -> str:
    parts = [f"Seller's one-line brief: {brief}", "", "Product truths:", _format_truths(product_truths)]
    if seller_direction:
        parts.append("")
        parts.append("Seller direction:")
        if seller_direction.get("mood_words"):
            parts.append(f"- Mood words: {', '.join(seller_direction['mood_words'])}")
        if seller_direction.get("never_do"):
            parts.append(f"- Never do: {seller_direction['never_do']}")
        if seller_direction.get("freeform"):
            parts.append(f"- Freeform notes: {seller_direction['freeform']}")
        ref_ad = seller_direction.get("reference_ad")
        if ref_ad:
            parts.append(f"- Reference ad: {ref_ad.get('url_or_text', '')} ({ref_ad.get('why', '')})")
    return "\n".join(parts)


def _parse_json_response(raw: str) -> dict:
    stripped = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.IGNORECASE)
    return json.loads(stripped)


def _hook_line(variant: dict) -> str:
    beats = variant.get("beats") or []
    return beats[0].get("line", "") if beats else ""


def _hook_has_number_or_contrast(hook: str) -> bool:
    lowered = hook.lower()
    if any(ch.isdigit() for ch in hook):
        return True
    return any(marker in lowered for marker in _CONTRAST_MARKERS)


def _validate_variant(variant: dict, truth_categories: dict[str, str], target_length_sec: int) -> list[str]:
    """Return a list of violation strings (empty = structurally valid).

    `truth_categories` maps truth_id -> category, for both the "does this
    truth_id actually exist" check and the "did you ground in something
    idiosyncratic, not just generic material/color facts" check below.
    """
    problems = []
    required = ("variant_id", "text", "framework", "hook_type", "emotional_trigger", "grounding_truth_ids", "beats")
    for key in required:
        if key not in variant:
            problems.append(f"missing field '{key}'")
    if problems:
        return problems  # can't check the rest meaningfully without the fields

    if variant["framework"] not in FRAMEWORKS:
        problems.append(f"framework '{variant['framework']}' is not one of {FRAMEWORKS}")

    gti = variant.get("grounding_truth_ids") or []
    if len(gti) < MIN_GROUNDING_TRUTH_IDS:
        problems.append(f"only {len(gti)} grounding_truth_ids, needs >= {MIN_GROUNDING_TRUTH_IDS}")
    unknown = [t for t in gti if t not in truth_categories]
    if unknown:
        problems.append(f"grounding_truth_ids references unknown truth_id(s): {unknown}")
    elif not any(truth_categories[t] in SPECIFIC_CATEGORIES for t in gti):
        problems.append(
            f"grounding_truth_ids {gti} are all generic categories "
            f"(none is {SPECIFIC_CATEGORIES}) -- cite at least one idiosyncratic detail"
        )

    beats = variant.get("beats") or []
    if not beats:
        problems.append("no beats")
    else:
        total = beats[-1].get("t_end", 0) - beats[0].get("t_start", 0)
        if abs(total - target_length_sec) > 1:
            problems.append(f"beats span {total}s, expected ~{target_length_sec}s")
        hook_words = len(_hook_line(variant).split())
        if hook_words > MAX_HOOK_WORDS:
            problems.append(f"hook line is {hook_words} words, expected <= {MAX_HOOK_WORDS}")
        if not _hook_has_number_or_contrast(_hook_line(variant)):
            problems.append(
                f"hook line '{_hook_line(variant)}' has no number and no contrast marker "
                f"({', '.join(_CONTRAST_MARKERS)}) -- reads as a bare tease, not a claim"
            )

    return problems


def _split_valid_invalid(
    variants: list[dict], truth_categories: dict[str, str], target_length_sec: int
) -> tuple[list[ScriptVariant], list[tuple[dict, list[str]]]]:
    """Per-variant structural check, THEN cross-variant dedup, in one pass.

    A variant that duplicates an already-accepted framework/hook_type is
    demoted to `invalid` (not silently kept in `valid`) -- first-seen wins,
    later duplicates are what get named in the re-prompt.
    """
    valid: list[ScriptVariant] = []
    invalid: list[tuple[dict, list[str]]] = []
    seen_frameworks: set[str] = set()
    seen_hooks: set[str] = set()

    for v in variants:
        problems = _validate_variant(v, truth_categories, target_length_sec)
        if problems:
            invalid.append((v, problems))
            continue

        framework = v["framework"]
        hook = str(v.get("hook_type", "")).lower()
        dup_problems = []
        if framework in seen_frameworks:
            dup_problems.append(f"duplicate framework '{framework}' (another variant already used it)")
        if hook in seen_hooks:
            dup_problems.append(f"duplicate hook_type '{hook}' (another variant already used it)")
        if dup_problems:
            invalid.append((v, dup_problems))
            continue

        seen_frameworks.add(framework)
        seen_hooks.add(hook)
        valid.append(
            ScriptVariant(
                variant_id=v["variant_id"],
                text=v["text"],
                framework=framework,
                hook_type=v["hook_type"],
                emotional_trigger=v["emotional_trigger"],
                grounding_truth_ids=v["grounding_truth_ids"],
                beats=v["beats"],
                target_length_sec=target_length_sec,
            )
        )
    return valid, invalid


def _reprompt_message(invalid: list[tuple[dict, list[str]]], valid_count: int) -> str:
    if not invalid:
        return (
            f"You returned only {valid_count} script variant(s), but exactly "
            f"{REQUIRED_VARIANT_COUNT} are required. Return the full corrected "
            f"JSON object with all {REQUIRED_VARIANT_COUNT} variants, still "
            "distinct in framework/hook_type/emotional_trigger."
        )
    lines = []
    for v, problems in invalid:
        vid = v.get("variant_id", "?")
        for p in problems:
            lines.append(f"- {vid}: {p}")
    return (
        "The following problems were found in your response:\n"
        + "\n".join(lines)
        + "\n\nFix ONLY these specific issues and return the full corrected JSON "
        "object in the same shape, still with exactly 4 variants."
    )


async def generate_script_variants(
    brief: str,
    product_truths: list[ProductTruth],
    seller_direction: Optional[dict] = None,
    target_length_sec: int = DEFAULT_TARGET_LENGTH_SEC,
    client: Optional[AsyncOpenAI] = None,
) -> list[ScriptVariant]:
    """Run the Concept Agent: one Qwen-Max call, one bounded re-prompt on failure.

    Degrades to the best N valid variants (minimum 2) rather than blocking the
    job; a single surviving variant is a legitimate degrade state too (it just
    means the Critic Chain has nothing to cross-pollinate against -- flagged
    by the caller, not here, since the "un-negotiated" flag belongs in the
    reasoning_trace the graph node writes, not in this pure function).
    """
    model = os.environ["MODEL_TEXT"]
    own_client = client is None
    if own_client:
        # Explicit timeout: the SDK's default (~10 min) turns a hung connection
        # into an apparent freeze rather than a fast, retryable failure. 45s
        # was too tight for this call specifically -- generating 4 full script
        # variants against a constrained JSON schema is a much bigger ask than
        # a trivial completion, and qwen3.7-plus may run an extended
        # "thinking" pass before producing output at all.
        client = AsyncOpenAI(
            api_key=os.environ["DASHSCOPE_API_KEY"],
            base_url=os.environ["DASHSCOPE_BASE_URL"],
            timeout=120.0,
        )
    truth_categories = {t["truth_id"]: t["category"] for t in product_truths}

    try:
        messages = [
            {"role": "system", "content": _build_system_prompt(target_length_sec)},
            {"role": "user", "content": _build_user_content(brief, product_truths, seller_direction)},
        ]

        response_text = await create_completion(client, model=model, messages=messages)
        parsed = _parse_json_response(response_text)
        raw_variants = parsed.get("script_variants", [])
        valid, invalid = _split_valid_invalid(raw_variants, truth_categories, target_length_sec)

        # Per spec: fewer than 4 variants is its own re-prompt trigger, even
        # when nothing else was individually wrong (the model just under-delivered).
        needs_reprompt = len(valid) < REQUIRED_VARIANT_COUNT
        if needs_reprompt:
            logger.info(
                "Concept Agent: %d/%d variants valid, re-prompting once (%d problems)",
                len(valid), REQUIRED_VARIANT_COUNT, len(invalid),
            )
            messages.append({"role": "assistant", "content": response_text})
            messages.append({"role": "user", "content": _reprompt_message(invalid, len(valid))})
            retry_text = await create_completion(client, model=model, messages=messages)
            retry_parsed = _parse_json_response(retry_text)
            retry_valid, _ = _split_valid_invalid(
                retry_parsed.get("script_variants", []), truth_categories, target_length_sec
            )
            if len(retry_valid) > len(valid):
                valid = retry_valid

        if len(valid) < MIN_VARIANTS_AFTER_DEGRADE:
            logger.warning(
                "Concept Agent: only %d valid variant(s) after re-prompt (spec wants %d) -- "
                "proceeding degraded, do not block the job.", len(valid), REQUIRED_VARIANT_COUNT,
            )
        elif len(valid) < REQUIRED_VARIANT_COUNT:
            logger.info(
                "Concept Agent: proceeding with %d/%d variants after degrade.",
                len(valid), REQUIRED_VARIANT_COUNT,
            )
        return valid
    finally:
        if own_client:
            await client.close()


async def concept_agent_node(state: ProductCutState) -> dict:
    """LangGraph node wrapper: reads brief/product_truths/seller_direction from state.

    No C2 custom event dispatched here -- per docs/BUILD_TASKS.md Phase 1, the
    `critic_score` event covers the whole Concept+Critic+Meta-Critic chain's
    result (RR's task), not a separate event per node.
    """
    variants = await generate_script_variants(
        brief=state["brief"],
        product_truths=state.get("product_truths", []),
        seller_direction=state.get("seller_direction"),
    )
    trace_note = f"\n[concept_agent] produced {len(variants)} script variant(s)."
    if len(variants) == 1:
        trace_note += " Only 1 survived validation -- un-negotiated, Critic Chain has nothing to cross-pollinate."
    return {
        "script_variants": variants,
        "reasoning_trace": state.get("reasoning_trace", "") + trace_note,
    }
