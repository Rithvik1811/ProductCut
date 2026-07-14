"""
Product Truth Extractor — Qwen-VL via DashScope (Phase 1).
Spec of record: docs/TECHNICAL_DOCUMENTATION.md §5.2.

Output shape is constrained to the frozen C1 contract (`graph.state.ProductTruth`):
{truth_id, fact, category, source}. The category enum is C1's, not a new one --
if that enum ever needs a new bucket, that's a C1 change requiring a KR/RR sync
and a version bump in graph/state.py, not a unilateral addition here.

Requires MODEL_VISION in the environment (DashScope Qwen-VL model id). This is
region/account-scoped like MODEL_VIDEO was -- see docs/DERISK_VIDEO_GEN_RESULT.md
§5 for why we don't hardcode a default and instead fail loudly if it's unset.
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Optional

from langchain_core.callbacks.manager import adispatch_custom_event
from langchain_core.runnables import RunnableConfig
from openai import AsyncOpenAI

from agents._retry import create_completion
from graph.state import ProductCutState, ProductTruth

logger = logging.getLogger("productcut.agents.product_truth_extractor")

MIN_FACTS = 6
MAX_FACTS = 10
MIN_VALID_FACTS_TO_SKIP_REPROMPT = 4

# Categories are C1's, not this prompt's own vocabulary -- see module docstring.
# "form_factor" (v8) is the one HOLISTIC whole-object anchor fact among an
# otherwise all-micro-fact vocabulary -- see FORM-FACTOR ANCHOR note in
# _build_system_prompt for why it exists (the Meta Quest -> "phone on a stand"
# wrong-object bug).
_CATEGORIES = (
    "color", "material", "texture",
    "construction_detail", "material_character", "scale_cue", "brief_or_intake_fact",
    "form_factor",
)

# Cheap heuristic for "generic enough to apply to any product" -- catches
# facts like "the mug is ceramic" without needing a second model call.
_GENERIC_STOPLIST = (
    "good quality", "well made", "nice product", "looks great",
    "high quality", "sturdy", "durable", "functional", "practical",
    "attractive", "modern design", "classic design",
)


def _build_system_prompt() -> str:
    return f"""You are an ad copywriter's assistant. Your job is to extract positive, sellable facts
from product photos — facts that would make a buyer MORE interested in this product.
You are NOT a quality inspector or defect detector. Every fact you extract should pass
this test: "Would this make a buyer more likely to want this product?" If not, exclude it.

You are analyzing product photos for an ad video pipeline. You will be given
2-3 photos, numbered in the order given (photo_1, photo_2, ...).

STEP 1 -- MANDATORY, before anything else: decide whether all photos show the
SAME physical product (same item, different angles/distances is fine; a
different item or different brand is NOT). You must answer this explicitly --
"same_product" is a REQUIRED boolean field in your response, always present,
never omitted, whether the answer is true or false. If false, you must also
fill "mismatch_reason" explaining what differs, and "product_truths" must
contain facts about photo_1's product ONLY -- do not silently pick a subset
of photos or blend facts from more than one item.

STEP 2 -- only if same_product is true: extract {MIN_FACTS}-{MAX_FACTS} specific,
non-generic facts about this exact product. Focus on: COLOR, STYLE/SILHOUETTE,
SIZE, and positive material/texture quality -- these are the sellable,
positive features a viewer should come away knowing. These are the facts that
make this exact item specific and desirable, not a generic listing photo.

POSITIVE-ONLY, BY DEFAULT (mandatory): describe what the product IS and what
makes it good -- never go hunting for scratches, scuffs, wear, damage, or
other flaws. If a flaw is so obvious it fills the frame, you may note it, but
do not seek it out, and do not let it crowd out the positive facts above --
{MIN_FACTS}-{MAX_FACTS} facts should be overwhelmingly color/style/size/
construction/material observations, not a defect inventory. Only actively look
for wear/damage/imperfection detail if the seller's brief or freeform notes
below explicitly ask for an authentic/well-loved/imperfection-forward angle --
absent that, a flawless-reading product is the correct, desired outcome, not
a gap to fill.

De-emphasize logos/brand marks/debossed insignia specifically: a logo is a
minor supporting detail, never the product's most important or most specific
feature. Only note one if it is genuinely load-bearing for identifying this
exact item (per the FORM-FACTOR ANCHOR below); do not let it crowd out color,
style, size, and construction facts.

Rules for each fact:
- Every fact must be something a person could ONLY know by looking at THIS
  specific item, not generic category knowledge (e.g. "a deep burgundy with a
  matte, slightly waxy finish" is good, "this is a mug" is not) -- specific
  does not mean negative; a precise color/style/size observation is exactly as
  specific and checkable as a flaw, and is the preferred way to satisfy this
  rule.
- If you are not confident about a detail (e.g. small text you can't fully
  read), either omit it or explicitly flag it as uncertain -- never state a
  guess as a confirmed fact.
- Do not use the word "category" or describe what type of product this is
  in general terms.
- Use category "brief_or_intake_fact" ONLY for a fact drawn from the seller's
  brief/notes text provided below (if any) -- never for something you observe
  directly in a photo. Visible logos, debossed/etched brand names, and printed
  text on the product itself belong under "construction_detail", not
  "brief_or_intake_fact".
- If you cannot find {MIN_FACTS} genuinely specific facts, look harder before giving up
  -- check surface material finish and texture, construction quality indicators (seams, joints,
  hardware, stitching, assembly precision), distinctive design details, material-specific
  quality signals that a buyer would value (grain in natural materials, glaze quality in
  ceramics, machining precision in metal, weave density in textiles), color and light
  behavior (matte/gloss/sheen), and brand/maker marks and their execution quality
  before settling for fewer. Do not resort to hunting for wear/damage to hit the count.
- Every fact must cite which photo it came from.

FORM-FACTOR ANCHOR -- exactly ONE of your facts must use category "form_factor".
This is a single sentence describing the ENTIRE product's physical gestalt,
synthesized across ALL the photos, written so that a stranger could pick this
exact object out of a lineup of unrelated products by shape alone. It must
include, in roughly this order:
  (1) the overall three-dimensional silhouette in plain shape words -- state
      whether it is flat or deep, boxy or curved, and its rough proportions
      (e.g. "a deep, rounded block, wider than it is tall, with a smoothly
      curved front face"). Describe the shape you SEE; never say what kind of
      product it is or what it is used for.
  (2) approximate real-world size, stated observationally relative to a
      familiar anchor visible or implied in the photos (e.g. "spans about two
      hand-widths", "small enough to sit in one palm"). An estimate from
      visual cues, not a spec-sheet number.
  (3) the dominant color(s) and surface finish (matte/gloss/fabric/metal).
  (4) how its major parts physically connect -- straps, hinges, cables,
      handles, pads, detachable pieces -- and where they attach.
Hard rules for this fact: no category or type words of any kind (never
"headset", "device", "gadget", "wearable", "appliance"); describe only what
IS present -- never contrast with other objects or say what it is NOT (do not
write "not a phone"); one sentence, 30-60 words; list this fact FIRST in
product_truths; set its "source" to the one photo showing the product alone
and filling the frame most completely (prefer that photo over one with props/
context, if you must choose).

CATEGORY DEFINITIONS (special-case categories that need explicit framing):
"material_character" — a natural variation in the product's material that signals
authenticity or quality of the underlying material. Examples: grain variation in genuine
leather (proof it is not synthetic), glaze drips on hand-thrown ceramics (proof of kiln
firing), hammer marks on hand-forged metal (proof of artisan process), knots or color
variation in solid wood (proof it is not MDF or veneer), slub texture in natural-fiber
textiles (proof of natural fiber content). ALWAYS describe these as the buyer would value
them — as proof of quality and authenticity — never as flaws or damage.

Return ONLY valid JSON in this exact shape, no preamble or commentary. The
first two keys are REQUIRED in every response, even when there is no mismatch:

{{
  "same_product": true,
  "mismatch_reason": "",
  "product_truths": [
    {{
      "truth_id": "t1",
      "category": "{' | '.join(_CATEGORIES)}",
      "fact": "the specific fact",
      "source": "photo_1"
    }}
  ]
}}"""


def _build_user_content(
    photo_urls: list[str], brief: Optional[str], freeform: Optional[str]
) -> list[dict]:
    """Assemble the multimodal user message: numbered photos + optional text context.

    Photos are passed by URL (product_photos in state are OSS URIs). DashScope's
    OpenAI-compatible endpoint accepts a fetchable image URL directly -- if the
    OSS bucket is private, these need to be signed URLs, not raw object paths.
    """
    parts: list[dict] = []
    context_bits = []
    if brief:
        context_bits.append(f"Seller's one-line brief: {brief}")
    if freeform:
        context_bits.append(f"Seller's freeform creative notes: {freeform}")
    if context_bits:
        parts.append({"type": "text", "text": "\n".join(context_bits)})

    for i, url in enumerate(photo_urls, start=1):
        parts.append({"type": "text", "text": f"photo_{i}:"})
        parts.append({"type": "image_url", "image_url": {"url": url}})

    parts.append(
        {
            "type": "text",
            "text": "Extract the product truths per the system instructions. "
            "Return only the JSON object.",
        }
    )
    return parts


def _parse_json_response(raw: str) -> dict:
    """Strip markdown code fences (models often wrap JSON in ```json ... ```) and parse."""
    stripped = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.IGNORECASE)
    return json.loads(stripped)


def _is_generic(fact: str) -> bool:
    """Cheap reject heuristic: too short, or matches a known generic phrase."""
    if len(fact.split()) < 6:
        return True
    lowered = fact.lower()
    return any(phrase in lowered for phrase in _GENERIC_STOPLIST)


def _validate_and_filter(raw_truths: list[dict]) -> tuple[list[ProductTruth], list[dict]]:
    """Split model output into (facts that pass the generic-heuristic bar, rejected raw entries)."""
    valid: list[ProductTruth] = []
    rejected: list[dict] = []
    for entry in raw_truths:
        fact = entry.get("fact", "")
        category = entry.get("category")
        if category not in _CATEGORIES or not fact or _is_generic(fact):
            rejected.append(entry)
            continue
        valid.append(
            ProductTruth(
                truth_id=entry.get("truth_id", f"t{len(valid) + 1}"),
                fact=fact,
                category=category,
                source=entry.get("source", "unknown"),
            )
        )
    return valid, rejected


def _has_form_factor(facts: list[ProductTruth]) -> bool:
    """Whether a valid "form_factor" anchor fact survived filtering (see the
    FORM-FACTOR ANCHOR instructions in _build_system_prompt)."""
    return any(f.get("category") == "form_factor" for f in facts)


# Positive-Only Truths fix (docs/BUILD_TASKS.md "Script Quality (CTA Bridge) +
# Positive-Only Truths..." workstream, Problem 1). Owner's words: "we mainly
# need to still focus on only the positive factors, the vl model should not
# capture the negatives at all like scratches and all." The prompt-level bias
# above (POSITIVE-ONLY, BY DEFAULT) is not trusted alone -- same "prompt +
# deterministic backstop" posture as every other gate in this codebase
# (_rhyme_problems, _flaw_led_hook_problem, etc.): an "imperfection" fact is
# dropped from the returned truths by default, full stop, UNLESS the seller's
# own direction explicitly asks for an authentic/well-worn/character angle.
# Category stays in the enum (real observational data that might genuinely
# matter for that explicit ask -- BUILD_TASKS.md's own leaning) but the
# default path downstream (Concept Agent, Budget Gate) never sees one unless
# asked for.
_IMPERFECTION_ANGLE_KEYWORDS = (
    "authentic", "imperfect", "imperfection", "well-loved", "well loved",
    "worn-in", "worn in", "lived-in", "lived in", "distressed", "patina",
    "character", "vintage", "weathered", "broken-in", "broken in",
)


def _wants_imperfection_angle(brief: Optional[str], freeform: Optional[str]) -> bool:
    """Crude keyword proxy (same posture as agents/_affordance.py) for "the
    seller explicitly asked for an authentic/imperfection-forward angle" --
    errs toward false negatives (missing an implied ask just means the
    default positive-only behavior applies, the safe direction)."""
    combined = f"{brief or ''} {freeform or ''}".lower()
    return any(kw in combined for kw in _IMPERFECTION_ANGLE_KEYWORDS)


def _filter_imperfection_by_default(
    facts: list[ProductTruth], wants_imperfection: bool
) -> list[ProductTruth]:
    """Drop category="material_character" facts unless the seller asked for that
    angle (see module note above) -- the deterministic gate behind the
    POSITIVE-ONLY, BY DEFAULT prompt rule."""
    if wants_imperfection:
        return facts
    return [f for f in facts if f.get("category") != "material_character"]


def _reprompt_message(rejected: list[dict], missing_form_factor: bool = False) -> str:
    """Name the exact violating facts so the re-prompt is targeted, not a generic retry.

    `missing_form_factor=True` adds a targeted, distinct instruction naming that
    specific failure -- a missing category is a different problem from a rejected
    (too-generic/too-short) fact, so it gets its own clear call-out rather than
    being folded into the generic "too generic" wording.
    """
    parts: list[str] = []
    if missing_form_factor:
        parts.append(
            "You did NOT include a required \"form_factor\" fact at all. Exactly "
            "one fact MUST have category=\"form_factor\": one 30-60 word sentence "
            "describing the ENTIRE product's whole-object shape/silhouette, "
            "approximate size, color/finish, and how its parts connect (see the "
            "FORM-FACTOR ANCHOR instructions above). Add it now -- do not omit it "
            "again."
        )
    if rejected:
        lines = [
            f'- "{r.get("fact", "")}" (category={r.get("category")}) -- too generic or too short'
            for r in rejected
        ]
        parts.append(
            "The following facts were rejected as too generic or too short "
            "(under 6 words, or a stock phrase like 'good quality'):\n"
            + "\n".join(lines)
        )
    parts.append(
        "Look again at the photos and replace/add facts as needed with more specific, "
        "checkable details (exact wear marks, precise color/texture, construction "
        "details). Return the full corrected JSON object in the same shape."
    )
    return "\n\n".join(parts)


async def extract_product_truths(
    photo_urls: list[str],
    brief: Optional[str] = None,
    freeform: Optional[str] = None,
    client: Optional[AsyncOpenAI] = None,
) -> list[ProductTruth]:
    """Run the Product Truth Extractor: one Qwen-VL call, one bounded re-prompt on failure.

    Mirrors the pattern used elsewhere in the pipeline (Shot-List Agent, Treatment
    Agent): one targeted re-prompt naming the exact violation, then proceed with
    whatever passed rather than blocking the job (docs/TECHNICAL_DOCUMENTATION.md §5.2).
    """
    model = os.environ["MODEL_VISION"]  # KeyError is intentional -- see module docstring
    own_client = client is None
    if own_client:
        # Explicit timeout: the SDK's default (~10 min) turns a hung connection
        # into an apparent freeze rather than a fast, retryable failure.
        client = AsyncOpenAI(
            api_key=os.environ["DASHSCOPE_API_KEY"],
            base_url=os.environ["DASHSCOPE_BASE_URL"],
            timeout=60.0,
        )

    try:
        messages = [
            {"role": "system", "content": _build_system_prompt()},
            {"role": "user", "content": _build_user_content(photo_urls, brief, freeform)},
        ]

        response_text = await create_completion(client, model=model, messages=messages)
        parsed = _parse_json_response(response_text)

        if "same_product" not in parsed:
            # The model skipped the required field entirely -- prompt compliance
            # failure, not a "no mismatch" signal. Do not silently treat as fine.
            logger.warning(
                "Product Truth Extractor: model omitted the required 'same_product' "
                "field entirely -- cannot confirm it actually checked for a mismatch. "
                "Raw response: %s", response_text[:500],
            )
        same_product = parsed.get("same_product")
        if same_product is False:
            logger.warning(
                "Product Truth Extractor: model flagged a product mismatch across "
                "the input photos: %s. Facts (if any) should be photo_1-only -- this "
                "job's photos should be reviewed, they may not be the same item.",
                parsed.get("mismatch_reason", "(no reason given)"),
            )

        raw_truths = parsed.get("product_truths", [])
        if same_product is False:
            # Deterministic backstop, not just a prompt instruction: don't trust
            # the model to have actually restricted itself to photo_1 -- the same
            # "don't trust self-report for something safety-relevant" lesson as
            # the missing-field check above. Applied again to the retry's output
            # below (v8's new form_factor trigger means a reprompt can now fire
            # even when this mismatch backstop is the reason valid facts are few,
            # so the retry path must re-apply the SAME backstop, not just the
            # first pass).
            raw_truths = [t for t in raw_truths if t.get("source") == "photo_1"]

        valid, rejected = _validate_and_filter(raw_truths)

        # Re-prompt on either of two independent triggers (§ v8): too few valid
        # facts survived filtering, OR the required form_factor anchor is simply
        # missing (a compliance failure distinct from "too generic" -- the model
        # never attempted one at all).
        missing_form_factor = not _has_form_factor(valid)
        too_few_valid = len(valid) < MIN_VALID_FACTS_TO_SKIP_REPROMPT and bool(rejected)
        if too_few_valid or missing_form_factor:
            logger.info(
                "Product Truth Extractor: re-prompting once (valid=%d, rejected=%d, "
                "missing_form_factor=%s)",
                len(valid), len(rejected), missing_form_factor,
            )
            messages.append({"role": "assistant", "content": response_text})
            messages.append(
                {"role": "user", "content": _reprompt_message(rejected, missing_form_factor)}
            )
            retry_text = await create_completion(client, model=model, messages=messages)
            retry_parsed = _parse_json_response(retry_text)
            retry_raw_truths = retry_parsed.get("product_truths", [])
            if same_product is False:
                # Same photo_1-only backstop as the first pass -- see the comment
                # at the first application above.
                retry_raw_truths = [t for t in retry_raw_truths if t.get("source") == "photo_1"]
            retry_valid, _ = _validate_and_filter(retry_raw_truths)
            # Take the retry's facts only if it actually did better; never go backwards.
            # For the missing-form_factor trigger specifically, "better" means the
            # retry gained a valid form_factor fact the original didn't have, even
            # if the total count is only similar -- the anchor mattering more than
            # raw count is the whole point of this trigger.
            gained_form_factor = (
                missing_form_factor and _has_form_factor(retry_valid) and not _has_form_factor(valid)
            )
            if gained_form_factor or len(retry_valid) > len(valid):
                valid = retry_valid

        # Positive-Only Truths fix -- applied AFTER the re-prompt/retry decision
        # above (which must judge the model's genuine compliance, unaffected by
        # this filter) and BEFORE the count warnings/MAX_FACTS truncation below
        # (which should reflect what's actually being returned downstream).
        wants_imperfection = _wants_imperfection_angle(brief, freeform)
        before_filter = len(valid)
        valid = _filter_imperfection_by_default(valid, wants_imperfection)
        if len(valid) < before_filter:
            logger.info(
                "Product Truth Extractor: dropped %d material_character-category fact(s) "
                "by default (positive-only truths) -- seller_direction did not ask "
                "for an authentic/material_character angle.", before_filter - len(valid),
            )

        if len(valid) < MIN_VALID_FACTS_TO_SKIP_REPROMPT:
            logger.warning(
                "Product Truth Extractor: proceeding with only %d facts after re-prompt "
                "(spec wants %d-%d) -- flag in reasoning trace, do not block the job.",
                len(valid), MIN_FACTS, MAX_FACTS,
            )
        if not _has_form_factor(valid):
            logger.warning(
                "Product Truth Extractor: proceeding with NO form_factor anchor fact "
                "after the bounded re-prompt -- the Video-Gen Node's Subject line will "
                "fall back to its per-shot micro-fact only (see video_gen_node.py "
                "_build_prompt), which is exactly the under-specified-subject failure "
                "mode this category exists to prevent."
            )
        if len(valid) > MAX_FACTS:
            logger.info(
                "Product Truth Extractor: model returned %d facts, truncating to the "
                "spec's cap of %d.", len(valid), MAX_FACTS,
            )
            # form_factor-aware truncation: the anchor must never be silently
            # dropped just because the model listed it last -- partition it out,
            # keep it unconditionally, and only truncate the rest.
            ff_fact = next((f for f in valid if f["category"] == "form_factor"), None)
            if ff_fact is not None:
                rest = [f for f in valid if f is not ff_fact]
                valid = [ff_fact] + rest[: MAX_FACTS - 1]
            else:
                valid = valid[:MAX_FACTS]
        return valid
    finally:
        if own_client:
            await client.close()


async def product_truth_extractor_node(state: ProductCutState, config: RunnableConfig) -> dict:
    """LangGraph node wrapper: reads product_photos/brief/seller_direction from state.

    Dispatches the C2 `truth_extracted` custom event via `adispatch_custom_event`,
    which surfaces in `astream_events` as `on_custom_event` -- app/main.py unwraps
    that into a proper C2 envelope (graph.events.build_event) rather than the
    generic passthrough it uses for raw LangGraph lifecycle events.
    """
    seller_direction = state.get("seller_direction") or {}
    truths = await extract_product_truths(
        photo_urls=state["product_photos"],
        brief=state.get("brief"),
        freeform=seller_direction.get("freeform"),
    )
    await adispatch_custom_event(
        "truth_extracted", {"truths": truths, "count": len(truths)}, config=config
    )
    return {
        "product_truths": truths,
        "reasoning_trace": state.get("reasoning_trace", "")
        + f"\n[product_truth_extractor] extracted {len(truths)} facts.",
    }
