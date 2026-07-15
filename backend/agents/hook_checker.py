"""
Hook-Checker — Qwen via DashScope (Phase 1, Critic Chain).
Spec of record: docs/TECHNICAL_DOCUMENTATION.md §5.4.1.

Scores each script variant's hook for specificity/strength, 1-5, with a
written justification -- one axis only, not full quality scoring (that's the
whole Critic Chain + Meta-Critic's job). Output is NOT a C1 CriticScore --
that's Meta-Critic's merged structure (hook+pacing+cta+tone+composite+
never_do_violation). This returns a lightweight per-variant
{hook_score, justification} dict that Meta-Critic (agents/meta_critic.py)
merges with Pacing/CTA/Tone into the real CriticScore.

Score scale is 1-5, matching the spec's explicit language, not an arbitrary
0-100 -- consistency matters because Meta-Critic computes a weighted
composite across all 4 checkers; a scale mismatch would silently break that
math unless every checker uses the same range.

WIRED into graph/build.py: fans out from concept_agent in parallel with
Pacing-/Body-/CTA-/Tone-Checker, fanning back in to meta_critic (see
graph/build.py). Was standalone-and-tested before Meta-Critic and its
siblings existed; that follow-up wiring has since landed.
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Optional, TypedDict

from openai import AsyncOpenAI

from agents._affordance import human_use_suits_product
from agents._retry import create_completion
from graph.state import ProductCutState, ScriptVariant

logger = logging.getLogger("productcut.agents.hook_checker")

MIN_SCORE = 1
MAX_SCORE = 5


class HookScore(TypedDict):
    hook_score: float
    justification: str


def _build_system_prompt(human_use_bias: bool = False) -> str:
    # Human-Centric Bias fix (video-gen-fidelity, 2026-07-11): a deliberate,
    # PRODUCT-CONDITIONAL scoring edge for human-moment hooks, per the owner's
    # reversal of the earlier "all three hook paths weigh equally" decision
    # (docs/BUILD_TASKS.md, Backstory-First section, issue #2). Only rendered
    # when the product's own facts establish a human-use affordance
    # (agents/_affordance.py) -- a product nobody wears/carries/holds keeps
    # the original equal-weighting calibration untouched. Deliberately a
    # TIEBREAK, not a blanket bonus, so it can never outrank the flaw-led/
    # sing-song SCORE DOWN rules or rescue a weak, vague human hook.
    human_bias_block = (
        f"""

PRODUCT-SUITABILITY TIEBREAK (this specific product is one a person wears,
carries, or holds in daily life -- its facts name real human-contact parts):
when a HUMAN-MOMENT hook and a claim-led or curiosity-gap hook are otherwise
comparable in specificity and craft, score the human-moment one ONE POINT
higher (capped at {MAX_SCORE}). Human presence in the first second is the
single strongest retention lever on short-form feeds. This tiebreak NEVER
overrides the SCORE DOWN rules below -- a flaw-led or sing-song hook still
caps at 2 even with a person in it -- and a vague, unspecific human hook
("someone loves this bag") still scores low on its own (lack of) merits."""
        if human_use_bias else ""
    )
    return f"""You are scoring the opening hook (first 2-3 seconds) of ad video scripts.{human_bias_block}

You will receive several script variants, each with its full text and stated hook_type.

Score EACH hook {MIN_SCORE}-{MAX_SCORE} on how well it would stop a scroll on a
short-form video feed (TikTok/Reels/Shorts). Consider: does it create curiosity,
tension, or an immediate visual/emotional pull within the first line?

Calibration -- anchor your scores against these. THREE DIFFERENT paths can all
reach {MAX_SCORE}/{MAX_SCORE} -- a human-moment or curiosity-gap hook is NEVER
weaker by default than a claim-led one just because it lacks a digit:
- {MIN_SCORE} (weakest): generic, could open an ad for almost any similar
  product. Example: "Check out this amazing mug." No claim, no tension, no
  specific detail, no person, nothing to resolve.
- {MAX_SCORE} (strongest) -- CLAIM PATH: a concrete, specific claim -- names a
  pain tied to a real detail, cites an exact number, or makes a contrarian
  claim resolved by contrast. Example: "Your coffee is cold in 12 minutes.
  Mine isn't." A number, then a contrast that resolves it, in one breath.
- {MAX_SCORE} (strongest) -- HUMAN-MOMENT PATH: drops the viewer into a
  specific person's specific instant, product visible in-scene but not yet
  pitched. Example: "She's out the door before sunrise, bag already on one
  shoulder." No number, no contrast marker -- and no weaker for it.
- {MAX_SCORE} (strongest) -- CURIOSITY-GAP PATH: a genuine curiosity gap
  grounded in a real, specific detail that the rest of the script actually
  pays off (not a bare unpaid tease). Example: "There's one part of this bag
  nobody notices for the first six months." A concrete promise, resolved
  later, not just a vague "watch till the end."
- Middling scores (2-4): specific but no payoff (a tease with no resolution
  anywhere in the script), or a contrast/number not tied to anything concrete.
- SCORE DOWN (2 max), regardless of specificity, in EITHER of these cases:
  (a) the hook is built on the PRODUCT'S OWN flaw or a competitor-flaw
      comparison ("Other bags hide this scuff, not ours.") -- specific detail
      does not redeem an opener that leads with a defect instead of a
      strength; imperfection-category material belongs later in the script,
      never the hook.
  (b) the hook line is sing-song/rhyming (its ending chimes with a nearby
      line's ending, or it scans like it could be sung/rapped) -- a jingle
      cadence reads as an ad, not a person talking, no matter how specific
      the claim inside it is.

Score each variant independently, but you may compare them to calibrate --
if all variants are similar strength, they can receive similar scores. Don't
force artificial spread just to look decisive.

Return ONLY valid JSON in this exact shape, no preamble or commentary:

{{
  "hook_scores": [
    {{"variant_id": "v1", "hook_score": 4, "justification": "one to two sentence justification"}}
  ]
}}"""


def _build_user_content(script_variants: list[ScriptVariant]) -> str:
    blocks = [
        f"variant_id: {v['variant_id']}\nhook_type: {v['hook_type']}\nfull text: {v['text']}"
        for v in script_variants
    ]
    return "\n---\n".join(blocks)


def _parse_json_response(raw: str) -> dict:
    stripped = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.IGNORECASE)
    return json.loads(stripped)


def _validate_scores(
    raw_scores: list[dict], expected_ids: set[str]
) -> tuple[dict[str, HookScore], list[str]]:
    """Return (valid scores keyed by variant_id, problem strings)."""
    valid: dict[str, HookScore] = {}
    problems: list[str] = []

    for entry in raw_scores:
        vid = entry.get("variant_id")
        score = entry.get("hook_score")
        justification = entry.get("justification", "")
        if vid not in expected_ids:
            problems.append(f"unknown variant_id '{vid}'")
            continue
        if vid in valid:
            problems.append(f"duplicate score for variant_id '{vid}'")
            continue
        if not isinstance(score, (int, float)) or isinstance(score, bool) or not (MIN_SCORE <= score <= MAX_SCORE):
            problems.append(f"{vid}: hook_score {score!r} out of range [{MIN_SCORE}, {MAX_SCORE}]")
            continue
        if not justification:
            problems.append(f"{vid}: missing justification")
            continue
        valid[vid] = HookScore(hook_score=score, justification=justification)

    for vid in expected_ids - set(valid.keys()):
        problems.append(f"missing score for variant_id '{vid}'")

    return valid, problems


def _reprompt_message(problems: list[str]) -> str:
    return (
        "The following problems were found in your response:\n"
        + "\n".join(f"- {p}" for p in problems)
        + "\n\nFix ONLY these specific issues and return the full corrected JSON "
        "object, with exactly one score for every variant_id listed."
    )


async def score_hooks(
    script_variants: list[ScriptVariant],
    client: Optional[AsyncOpenAI] = None,
    *,
    human_use_bias: bool = False,
) -> dict[str, HookScore]:
    """Run the Hook-Checker: one Qwen call scoring all variants, one bounded re-prompt.

    Degrades to a neutral fallback score (flagged in its own justification) for
    any variant still unscored after the retry, rather than blocking the
    Critic Chain -- same bounded-retry-then-safe-fallback shape as every other
    loop in this pipeline.

    `human_use_bias` renders the product-conditional human-moment scoring
    tiebreak into the rubric (see _build_system_prompt) -- the node wrapper
    derives it from the job's own product truths; default False keeps the
    original calibration for products with no human-use affordance.
    """
    if not script_variants:
        return {}

    model = os.environ["MODEL_TEXT"]
    own_client = client is None
    if own_client:
        client = AsyncOpenAI(
            api_key=os.environ["DASHSCOPE_API_KEY"],
            base_url=os.environ["DASHSCOPE_BASE_URL"],
            timeout=60.0,
        )
    expected_ids = {v["variant_id"] for v in script_variants}

    try:
        messages = [
            {"role": "system", "content": _build_system_prompt(human_use_bias)},
            {"role": "user", "content": _build_user_content(script_variants)},
        ]

        response_text = await create_completion(client, model=model, messages=messages)
        parsed = _parse_json_response(response_text)
        valid, problems = _validate_scores(parsed.get("hook_scores", []), expected_ids)

        if problems:
            logger.info(
                "Hook-Checker: %d/%d variants scored, re-prompting once (%d problems)",
                len(valid), len(expected_ids), len(problems),
            )
            messages.append({"role": "assistant", "content": response_text})
            messages.append({"role": "user", "content": _reprompt_message(problems)})
            retry_text = await create_completion(client, model=model, messages=messages)
            retry_parsed = _parse_json_response(retry_text)
            retry_valid, _ = _validate_scores(retry_parsed.get("hook_scores", []), expected_ids)
            # Retry can fill in what the first attempt missed; keep first-attempt
            # scores for anything the retry doesn't also provide.
            valid = {**valid, **retry_valid}

        still_missing = expected_ids - set(valid.keys())
        if still_missing:
            logger.warning(
                "Hook-Checker: %d variant(s) still unscored after re-prompt (%s) -- "
                "assigning neutral fallback score, not blocking the Critic Chain.",
                len(still_missing), sorted(still_missing),
            )
            mid = (MIN_SCORE + MAX_SCORE) / 2
            for vid in still_missing:
                valid[vid] = HookScore(
                    hook_score=mid,
                    justification="Hook-Checker failed to score this variant after a retry; neutral fallback assigned.",
                )
        return valid
    finally:
        if own_client:
            await client.close()


async def hook_checker_node(state: ProductCutState) -> dict:
    """LangGraph node wrapper: scores each variant's hook. Runs in parallel with
    the other 4 checkers (see graph/build.py's fan-out from concept_agent).
    Derives the product-conditional human-moment tiebreak from the job's own
    product truths (Human-Centric Bias fix)."""
    scores = await score_hooks(
        state["script_variants"],
        human_use_bias=human_use_suits_product(state.get("product_truths", [])),
    )
    return {"hook_scores": scores}
