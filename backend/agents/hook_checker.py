"""
Hook-Checker — Qwen via DashScope (Phase 1, Critic Chain).
Spec of record: docs/TECHNICAL_DOCUMENTATION.md §5.4.1.

Scores each script variant's hook for specificity/strength, 1-5, with a
written justification -- one axis only, not full quality scoring (that's the
whole Critic Chain + Meta-Critic's job). Output is NOT a C1 CriticScore --
that's Meta-Critic's merged structure (hook+pacing+cta+tone+composite+
never_do_violation). This returns a lightweight per-variant
{hook_score, justification} dict that Meta-Critic (RR's task, not built yet)
will merge with Pacing/CTA/Tone into the real CriticScore.

Score scale is 1-5, matching the spec's explicit language, not an arbitrary
0-100 -- consistency matters because Meta-Critic computes a weighted
composite across all 4 checkers; a scale mismatch would silently break that
math unless every checker uses the same range.

NOT wired into graph/build.py yet: there's no Meta-Critic to merge this
into state["critic_scores"], and CTA-Checker/Tone-Checker (the siblings this
is meant to run in parallel with) don't exist yet either. Standalone and
tested so RR can wire it in once those exist.
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Optional, TypedDict

from openai import AsyncOpenAI

from agents._retry import create_completion
from graph.state import ProductCutState, ScriptVariant

logger = logging.getLogger("productcut.agents.hook_checker")

MIN_SCORE = 1
MAX_SCORE = 5


class HookScore(TypedDict):
    hook_score: float
    justification: str


def _build_system_prompt() -> str:
    return f"""You are scoring the opening hook (first 2-3 seconds) of ad video scripts.

You will receive several script variants, each with its full text and stated hook_type.

Score EACH hook {MIN_SCORE}-{MAX_SCORE} on how well it would stop a scroll on a
short-form video feed (TikTok/Reels/Shorts). Consider: does it create curiosity,
tension, or an immediate visual/emotional pull within the first line?

Calibration -- anchor your scores against these:
- {MIN_SCORE} (weakest): generic, could open an ad for almost any similar
  product. Example: "Check out this amazing mug." No claim, no tension, no
  specific detail.
- {MAX_SCORE} (strongest): a concrete, specific claim -- names a pain tied to
  a real detail, cites an exact number, or makes a contrarian claim resolved
  by contrast. Example: "Your coffee is cold in 12 minutes. Mine isn't." A
  number, then a contrast that resolves it, in one breath.
- Middling scores (2-4): specific but no contrast/number (a tease with no
  payoff), or a contrast/number not tied to anything concrete.

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
    script_variants: list[ScriptVariant], client: Optional[AsyncOpenAI] = None
) -> dict[str, HookScore]:
    """Run the Hook-Checker: one Qwen call scoring all variants, one bounded re-prompt.

    Degrades to a neutral fallback score (flagged in its own justification) for
    any variant still unscored after the retry, rather than blocking the
    Critic Chain -- same bounded-retry-then-safe-fallback shape as every other
    loop in this pipeline.
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
            {"role": "system", "content": _build_system_prompt()},
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
    """LangGraph node wrapper: scores each variant's hook. Runs in parallel with the other 4 checkers (see graph/build.py's fan-out from concept_agent)."""
    scores = await score_hooks(state["script_variants"])
    return {"hook_scores": scores}
