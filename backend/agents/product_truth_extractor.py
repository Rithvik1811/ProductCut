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
_CATEGORIES = (
    "color", "material", "texture",
    "construction_detail", "imperfection", "scale_cue", "brief_or_intake_fact",
)

# Cheap heuristic for "generic enough to apply to any product" -- catches
# facts like "the mug is ceramic" without needing a second model call.
_GENERIC_STOPLIST = (
    "good quality", "well made", "nice product", "looks great",
    "high quality", "sturdy", "durable", "functional", "practical",
    "attractive", "modern design", "classic design",
)


def _build_system_prompt() -> str:
    return f"""You are analyzing product photos for an ad video pipeline. You will be given
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
non-generic facts about this exact product. Focus on: material, color,
texture, distinguishing marks, size cues, and shape details.

Rules for each fact:
- Every fact must be something a person could ONLY know by looking at THIS
  specific item, not generic category knowledge (e.g. "hairline crack near
  the handle base" is good, "this is a mug" is not).
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
  -- check texture, reflections, wear patterns, proportions, and any text/logo
  detail before settling for fewer.
- Every fact must cite which photo it came from.

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


def _reprompt_message(rejected: list[dict]) -> str:
    """Name the exact violating facts so the re-prompt is targeted, not a generic retry."""
    lines = [
        f'- "{r.get("fact", "")}" (category={r.get("category")}) -- too generic or too short'
        for r in rejected
    ]
    return (
        "The following facts were rejected as too generic or too short "
        f"(under 6 words, or a stock phrase like 'good quality'):\n"
        + "\n".join(lines)
        + "\n\nLook again at the photos and replace these with more specific, checkable "
        "details (exact wear marks, precise color/texture, construction details). "
        "Return the full corrected JSON object in the same shape."
    )


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
            # the missing-field check above.
            raw_truths = [t for t in raw_truths if t.get("source") == "photo_1"]

        valid, rejected = _validate_and_filter(raw_truths)

        if len(valid) < MIN_VALID_FACTS_TO_SKIP_REPROMPT and rejected:
            logger.info(
                "Product Truth Extractor: only %d valid facts, re-prompting once (%d rejected)",
                len(valid), len(rejected),
            )
            messages.append({"role": "assistant", "content": response_text})
            messages.append({"role": "user", "content": _reprompt_message(rejected)})
            retry_text = await create_completion(client, model=model, messages=messages)
            retry_parsed = _parse_json_response(retry_text)
            retry_valid, _ = _validate_and_filter(retry_parsed.get("product_truths", []))
            # Take the retry's facts only if it actually did better; never go backwards.
            if len(retry_valid) > len(valid):
                valid = retry_valid

        if len(valid) < MIN_VALID_FACTS_TO_SKIP_REPROMPT:
            logger.warning(
                "Product Truth Extractor: proceeding with only %d facts after re-prompt "
                "(spec wants %d-%d) -- flag in reasoning trace, do not block the job.",
                len(valid), MIN_FACTS, MAX_FACTS,
            )
        if len(valid) > MAX_FACTS:
            logger.info(
                "Product Truth Extractor: model returned %d facts, truncating to the "
                "spec's cap of %d.", len(valid), MAX_FACTS,
            )
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
