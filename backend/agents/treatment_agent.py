"""
Treatment Agent — Qwen-Plus via DashScope (Phase 2).
Spec of record: docs/TECHNICAL_DOCUMENTATION.md §5.5.

Output shape is constrained to the frozen C1 contract (`graph.state.Treatment`
/ `BeatTreatment`): {director_persona, color_story, pacing_philosophy,
beat_treatments: [{beat_index, beat_function, script_quote, truth_fact_id,
visual_approach, why_not_generic}]}. `beat_function` is C1's 5-value enum
(hook/problem/demo/proof/cta, shared with `Shot.beat_role`) -- if that enum
ever needs a new value, that's a C1 change requiring a KR/RR sync and a
version bump in graph/state.py, not a unilateral addition here.

One beat_treatments[] entry per winning_script.beats[] entry, same index and
order -- beat_index is that list position, not an independent numbering.

Per-beat grounding checks (verbatim quote, real truth_fact_id, valid
beat_function, banned-word/stoplist) are delegated to
agents.justification_validator.validate_justifications -- the same function
Shot-List Agent's per-shot ShotJustification validation uses (Phase 2
interface handoff, docs/TECHNICAL_DOCUMENTATION.md §5.6). This module owns
only what's specific to Treatment Agent: prompting, matching returned entries
back to expected beat slots, the missing-entry-entirely case (the shared
validator only judges justifications that exist), the re-prompt/fallback
control flow, and the top-level (non-per-beat) director_persona/color_story/
pacing_philosophy banned-word check.

Failure handling deviates slightly from the doc's literal "re-prompts once
for that beat specifically" framing: this reuses the same whole-response,
name-the-violation retry shape as every other agent in this codebase
(Concept Agent, Product Truth Extractor, Hook-Checker) -- one bounded
re-prompt naming exactly which beat_index failed and why, not a separate
isolated API call per beat. Cheaper, and still surgical (violations are
named per beat, only the bad beats are called out), just not literally one
call per beat. A beat still failing after that retry falls back
deterministically to the doc's literal lowest-risk treatment (static
framing, shared lighting only) rather than blocking the job.

WIRED into graph/build.py: `treatment_agent -> shot_list_agent -> budget_gate
-> video_gen`, downstream of `winning_script` (set by merge_validator_node).
Was standalone and tested before that, same posture hook_checker.py was in
before its siblings existed; that follow-up wiring has since landed.
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Optional

from openai import AsyncOpenAI

from agents._retry import create_completion
from agents.justification_validator import BANNED_WORD, BEAT_FUNCTIONS, validate_justifications
from graph.state import BeatTreatment, ProductTruth, ScriptBeat, Treatment, WinningScript

logger = logging.getLogger("productcut.agents.treatment_agent")

# Doc's literal lowest-risk fallback for a beat still invalid after the retry
# (docs/TECHNICAL_DOCUMENTATION.md §5.5: "falls back to the single most
# literal, lowest-risk treatment for that beat (static framing, shared
# lighting only)").
_FALLBACK_VISUAL_APPROACH = "static framing, shared lighting only"
_FALLBACK_WHY_NOT_GENERIC = (
    "Fallback treatment after repeated validation failure -- not a reasoned creative choice."
)

# Human-readable messages for the shared validator's fixed violation
# vocabulary, used to build a targeted re-prompt. `treatment_ref_invalid` is
# intentionally omitted: a BeatTreatment entry never carries a treatment_ref
# field (that's ShotJustification's field, not this module's), so
# validate_justifications never emits that violation for calls from here.
_VIOLATION_MESSAGES = {
    "quote_mismatch": "script_quote is not a verbatim substring of the winning script text",
    "unknown_truth_id": "truth_fact_id does not exist in product_truths",
    "invalid_beat_function": f"beat_function is not one of {BEAT_FUNCTIONS}",
    "stoplist_hit": f"why_not_generic/visual_approach uses the banned word '{BANNED_WORD}' or a generic stock phrase",
}


def _format_truths(truths: list[ProductTruth]) -> str:
    return "\n".join(f"- [{t['truth_id']}] ({t['category']}) {t['fact']}" for t in truths)


def _format_beats(beats: list[ScriptBeat]) -> str:
    return "\n".join(f'{i}: [{b["t_start"]}s-{b["t_end"]}s] "{b["line"]}"' for i, b in enumerate(beats))


def _build_system_prompt(beat_count: int) -> str:
    last = beat_count - 1
    return f"""You are a director's assistant creating a visual treatment for a short-form
product ad (15-30s), grounded in the winning script and specific product
facts -- never in generic category knowledge.

You will receive:
- The winning script's full text and its beats as a numbered list (numbered
  0 to {last}, in order)
- product_truths: a list of {{truth_id, category, fact}}

Produce a director's treatment with:
1. director_persona: a short description of the visual/directorial voice for
   this specific ad (e.g. "intimate, handheld warmth" vs "crisp, editorial
   precision") -- must be justified by something in the script's tone, not a
   generic "modern and clean" default.
2. color_story: the intended color palette/mood, grounded in the product's
   actual visible colors/materials from product_truths.
3. pacing_philosophy: how the ad should feel to move through (e.g. "slow
   build then a quick punch on the CTA") tied to the script's beat structure.
4. beat_treatments: EXACTLY {beat_count} entries, one per script beat listed
   above, in the same order (beat_index 0 through {last}). For each:
   - beat_index: the beat's position in the numbered list above (integer)
   - beat_function: this beat's narrative role -- one of exactly:
     {", ".join(BEAT_FUNCTIONS)}. Reason from what the beat is DOING
     narratively (hook = shock/tension, problem = discomfort, demo/proof =
     trust/clarity, cta = urgency/clarity), never from the product's category.
   - script_quote: a VERBATIM quote from that beat's own line (must match
     exactly, word for word -- this will be validated)
   - truth_fact_id: the specific product_truths[] truth_id this beat's
     visual choice is grounded in
   - visual_approach: the specific camera/framing/lighting approach for this
     beat (e.g. "static macro push on the seam, no camera movement")
   - why_not_generic: 1-2 sentences explaining why this visual choice is
     specific to THIS product, not a generic stock-footage choice any
     similar product could use

HARD RULES:
- Never use the word "{BANNED_WORD}" anywhere in your output, especially not
  in why_not_generic or visual_approach.
- Every beat_treatment must cite a real truth_id that exists in the provided
  product_truths list -- never invent or hallucinate a fact.
- script_quote must be an exact, verbatim substring of that beat's own line --
  no paraphrasing, no partial-match approximations.
- Every visual_approach/why_not_generic must reason from the beat's
  narrative function and the cited fact, not from what kind of product this is.

Return ONLY valid JSON in this exact shape, no preamble or commentary:

{{
  "director_persona": "...",
  "color_story": "...",
  "pacing_philosophy": "...",
  "beat_treatments": [
    {{
      "beat_index": 0,
      "beat_function": "one of: {' | '.join(BEAT_FUNCTIONS)}",
      "script_quote": "exact verbatim quote from that beat's line",
      "truth_fact_id": "t3",
      "visual_approach": "...",
      "why_not_generic": "..."
    }}
  ]
}}"""


def _build_user_content(winning_script: WinningScript, product_truths: list[ProductTruth]) -> str:
    return "\n".join(
        [
            "Winning script (full text):",
            winning_script["text"],
            "",
            "Winning script beats (numbered, in order):",
            _format_beats(winning_script["beats"]),
            "",
            "Product truths:",
            _format_truths(product_truths),
        ]
    )


def _parse_json_response(raw: str) -> dict:
    stripped = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.IGNORECASE)
    return json.loads(stripped)


def _contains_banned_word(*texts: str) -> bool:
    """Top-level (non-per-beat) banned-word check for director_persona/
    color_story/pacing_philosophy -- these aren't justification dicts, so
    they're outside validate_justifications' scope; flagged only, not
    re-prompted (see generate_treatment)."""
    return any(BANNED_WORD in t.lower() for t in texts if t)


def _default_beat_function(index: int, total: int) -> str:
    """Deterministic beat_function guess -- used only in the literal fallback path."""
    if total <= 1 or index == 0:
        return "hook"
    if index == total - 1:
        return "cta"
    middle = ("problem", "demo", "proof")
    return middle[(index - 1) % len(middle)]


def _fallback_beat_treatment(
    beat_index: int, beat: ScriptBeat, total_beats: int, truth_ids: list[str]
) -> BeatTreatment:
    return BeatTreatment(
        beat_index=beat_index,
        beat_function=_default_beat_function(beat_index, total_beats),
        script_quote=beat["line"],
        truth_fact_id=truth_ids[0] if truth_ids else "",
        visual_approach=_FALLBACK_VISUAL_APPROACH,
        why_not_generic=_FALLBACK_WHY_NOT_GENERIC,
    )


def _extract_by_index(parsed: dict) -> dict[int, dict]:
    return {
        e["beat_index"]: e
        for e in parsed.get("beat_treatments", [])
        if isinstance(e, dict) and isinstance(e.get("beat_index"), int)
    }


def _validate_entries(
    indices: list[int],
    by_index: dict[int, dict],
    winning_script: WinningScript,
    product_truths: list[ProductTruth],
) -> tuple[dict[int, dict], dict[int, str]]:
    """Run the shared validator on whichever of `indices` the model actually
    returned; anything missing entirely is this module's own concern (the
    shared validator only judges justifications that exist -- it has no
    notion of "this one is absent").
    """
    present = [i for i in indices if i in by_index]
    missing = [i for i in indices if i not in by_index]

    justifications = [by_index[i] for i in present]
    # treatment=None: this module is producing the Treatment, so there is no
    # existing Treatment object to check a treatment_ref against yet -- moot
    # anyway, since BeatTreatment entries never carry a treatment_ref field.
    results = validate_justifications(justifications, winning_script, product_truths, treatment=None)

    valid: dict[int, dict] = {}
    problems: dict[int, str] = {i: f"missing beat_treatments entry for beat_index {i}" for i in missing}
    for i, result in zip(present, results):
        if result["passed"]:
            valid[i] = by_index[i]
        else:
            violation = result["violation"] or "invalid"
            problems[i] = _VIOLATION_MESSAGES.get(violation, violation)
    return valid, problems


def _reprompt_message(problems_by_index: dict[int, str]) -> str:
    lines = [f"- beat_index {i}: {p}" for i, p in sorted(problems_by_index.items())]
    return (
        "The following problems were found in your beat_treatments:\n"
        + "\n".join(lines)
        + "\n\nFix ONLY these specific beats and return the full corrected JSON "
        "object in the same shape, with every beat_treatments entry present "
        "(not just the ones you're fixing)."
    )


async def generate_treatment(
    winning_script: WinningScript,
    product_truths: list[ProductTruth],
    client: Optional[AsyncOpenAI] = None,
) -> Treatment:
    """Run the Treatment Agent: one Qwen-Plus call, one bounded re-prompt naming
    exactly which beats failed and why (per the shared validator), then a
    deterministic per-beat fallback for anything still invalid
    (docs/TECHNICAL_DOCUMENTATION.md §5.5).
    """
    model = os.environ["MODEL_TEXT"]
    own_client = client is None
    if own_client:
        # Explicit timeout -- see product_truth_extractor.py's module docstring
        # for why the SDK default is unsafe here.
        client = AsyncOpenAI(
            api_key=os.environ["DASHSCOPE_API_KEY"],
            base_url=os.environ["DASHSCOPE_BASE_URL"],
            timeout=90.0,
        )

    beats = winning_script["beats"]
    beat_count = len(beats)
    truth_id_list = [t["truth_id"] for t in product_truths]
    all_indices = list(range(beat_count))

    try:
        messages = [
            {"role": "system", "content": _build_system_prompt(beat_count)},
            {"role": "user", "content": _build_user_content(winning_script, product_truths)},
        ]

        response_text = await create_completion(client, model=model, messages=messages)
        parsed = _parse_json_response(response_text)

        director_persona = parsed.get("director_persona") or ""
        color_story = parsed.get("color_story") or ""
        pacing_philosophy = parsed.get("pacing_philosophy") or ""
        if _contains_banned_word(director_persona, color_story, pacing_philosophy):
            logger.warning(
                "Treatment Agent: top-level fields used the banned word '%s' -- "
                "not re-prompted (only beat_treatments are), flagging only.",
                BANNED_WORD,
            )

        by_index = _extract_by_index(parsed)
        valid_entries, problems_by_index = _validate_entries(all_indices, by_index, winning_script, product_truths)

        if problems_by_index:
            logger.info(
                "Treatment Agent: %d/%d beats valid, re-prompting once (bad beats: %s)",
                beat_count - len(problems_by_index), beat_count, sorted(problems_by_index),
            )
            messages.append({"role": "assistant", "content": response_text})
            messages.append({"role": "user", "content": _reprompt_message(problems_by_index)})
            retry_text = await create_completion(client, model=model, messages=messages)
            retry_parsed = _parse_json_response(retry_text)
            retry_by_index = _extract_by_index(retry_parsed)

            retry_valid, still_problems = _validate_entries(
                list(problems_by_index.keys()), retry_by_index, winning_script, product_truths
            )
            valid_entries.update(retry_valid)
            problems_by_index = still_problems

        if problems_by_index:
            logger.warning(
                "Treatment Agent: %d beat(s) still invalid after re-prompt (%s) -- "
                "falling back to the literal lowest-risk treatment, not blocking the job.",
                len(problems_by_index), sorted(problems_by_index),
            )

        beat_treatments: list[BeatTreatment] = []
        for i, beat in enumerate(beats):
            if i in valid_entries:
                e = valid_entries[i]
                beat_treatments.append(
                    BeatTreatment(
                        beat_index=i,
                        beat_function=e["beat_function"],
                        script_quote=e["script_quote"],
                        truth_fact_id=e["truth_fact_id"],
                        visual_approach=e.get("visual_approach") or "",
                        why_not_generic=e.get("why_not_generic") or "",
                    )
                )
            else:
                beat_treatments.append(_fallback_beat_treatment(i, beat, beat_count, truth_id_list))

        return Treatment(
            director_persona=director_persona or "understated, product-forward",
            color_story=color_story or "neutral, true-to-photo color",
            pacing_philosophy=pacing_philosophy or "even pacing, no beat shorter than its script timing",
            beat_treatments=beat_treatments,
        )
    finally:
        if own_client:
            await client.close()


async def treatment_agent_node(state: dict) -> dict:
    """LangGraph node wrapper: reads winning_script/product_truths from state.

    Typed as `dict` rather than `ProductCutState` for the parameter -- this
    node IS wired into graph/build.py now (see module docstring), so the
    compiled graph does guarantee its input matches that shape; the looser
    `dict` annotation is a pre-existing minor inconsistency with every other
    node wrapper in this codebase (which take the real `ProductCutState`),
    left as-is here since correcting it is a type-hint-only change outside
    this pass's documentation-accuracy scope, not a functional one.
    """
    treatment = await generate_treatment(
        winning_script=state["winning_script"],
        product_truths=state.get("product_truths", []),
    )
    fallback_count = sum(
        1 for bt in treatment["beat_treatments"] if bt["why_not_generic"] == _FALLBACK_WHY_NOT_GENERIC
    )
    trace_note = f"\n[treatment_agent] produced treatment with {len(treatment['beat_treatments'])} beat treatment(s)."
    if fallback_count:
        trace_note += f" {fallback_count} beat(s) used the literal lowest-risk fallback after failed validation."
    return {
        "treatment": treatment,
        "reasoning_trace": state.get("reasoning_trace", "") + trace_note,
    }
