"""
Visual Direction Agent — Phase 2 (between merge_validator and treatment_agent).
Spec of record: task description, VDA section.

Bridges VO scripts → concrete visual shot decisions: which feature/truth to
highlight per beat, whether a human is present, what they do, what shot type
and camera move to use. Runs BEFORE treatment_agent, so treatment_agent
receives pre-decided visual choices as fixed constraints rather than
re-deciding them independently.

OUTPUT: VisualDirection (graph.state) keyed into ProductCutState as
`visual_direction`. Downstream agents (treatment_agent, shot_list_agent)
consume it via state.get("visual_direction") — graceful degradation when
absent.
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Optional

from openai import AsyncOpenAI

from agents._affordance import human_use_suits_product
from agents._retry import create_completion
from graph.state import BeatVisualDirection, ProductCutState, ProductTruth, VisualDirection, WinningScript

logger = logging.getLogger("productcut.agents.visual_direction_agent")

_VALID_SHOT_TYPES = frozenset({
    "hook_hero", "macro_detail", "lifestyle_context", "hero_reframe",
    "cta_endcard", "product_in_hand", "worn_in_use",
})
_VALID_CAMERA_MOVES = frozenset({
    "push_in", "orbit", "static", "pan", "tilt_up", "pull_back", "rack_focus",
})


def _format_truths(truths: list[ProductTruth]) -> str:
    return "\n".join(f"- [{t['truth_id']}] ({t['category']}) {t['fact']}" for t in truths)


def _parse_json_response(raw: str) -> dict:
    stripped = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.IGNORECASE)
    return json.loads(stripped)


def _scene_inspiration_block(research_facts: list) -> str:
    """SCENE INSPIRATION prompt block built from web-researched use_case /
    visual_moment facts. Returns "" when empty so the facts-absent prompt is
    byte-identical to the pre-injection prompt (regression safety)."""
    if not research_facts:
        return ""
    lines = [
        "─────────────────────────────────────────────────────",
        "SCENE INSPIRATION FROM WEB RESEARCH",
        "─────────────────────────────────────────────────────",
        "Researched real-world scenes and use contexts for this product (from public",
        "web sources — NOT visible in the product photos):",
    ]
    for f in research_facts:
        lines.append(
            f"- [{f.get('fact_id', '?')}] ({f.get('category', '')}) {f.get('claim', '')}"
        )
    lines += [
        "",
        "How to use these:",
        "- They are SCENE and SETTING inspiration: where a beat takes place, what a person",
        "  does with the product, which specific cinematic moment a beat stages.",
        "- When a beat's VO line evokes one of these scenes, DIRECT that exact moment: pick",
        "  the setting, human_action, shot type, camera move, and framing that stage it.",
        "- story_context may draw its setting and arc from these scenes.",
        "",
        "FIREWALL — what these facts may NEVER do:",
        "- They NEVER describe the product's appearance. Every visible product detail (color,",
        "  material, shape, parts, markings) comes ONLY from the photo truth table.",
        "- focus_feature_truth_id must ALWAYS be a real truth_id from the truth table — NEVER",
        "  an r* research id. Research facts inspire the scene AROUND the truth, not the truth.",
        "- Never stage a spec, number, or claim as on-screen text; these inspire SCENES, not copy.",
        "- If a scene contradicts what the photos show the product to be, the photos win — drop it.",
    ]
    return "\n".join(lines) + "\n"


def _build_system_prompt(
    beat_count: int,
    truth_ids: list[str],
    human_affordance: bool,
    research_facts: Optional[list] = None,
) -> str:
    last = beat_count - 1
    human_target_note = (
        """HUMAN PRESENCE GUIDANCE (this product has real human-contact parts — handle, straps, etc.):
- Demo/proof beats (beats 1 through {second_last}): PREFER human_presence "yes" when the cited truth
  names a handle, strap, pocket, or scale fact. The whole reason people buy a carryable product is
  the moment of use — show it.
- Target for a {beat_count}-beat ad: 1-2 human shots concentrated in beats 1-{second_last}.
- The opening hook (beat 0) and the closing CTA (beat {last}) are product-alone by default
  (see rules below — the CTA rule is non-negotiable; the hook rule can bend only if the VO
  explicitly opens on a person).
""".format(beat_count=beat_count, last=last, second_last=max(0, last - 1))
        if human_affordance else
        """HUMAN PRESENCE GUIDANCE (this product's truths do not name explicit human-contact parts):
- IMPERATIVE BEATS: When a non-CTA beat's VO line is an imperative action involving the
  product ("Toss it in your bag", "Fill it up", "Hold it", "Pour it out", "Press the lid",
  "Squeeze it", "Drop it in", "Clip it on", "Drink from it", etc.), human_presence MUST be
  "yes" — show someone performing that exact action. The video must embody what the VO
  asks the viewer to imagine doing.
- DESCRIPTIVE BEATS: When the VO describes a product quality without implying physical use
  (color, form, texture, a material property), human_presence "no" is appropriate.
- CTA beat: always "no" (non-negotiable, see rule 1).
"""
    )
    return f"""You are a visual director bridging a voiceover script into concrete shot decisions
for a short-form product video ad ({beat_count} beats, 0 to {last}).

You will receive the VO beats (numbered 0–{last}) and a product truth table
(truth_ids: {', '.join(truth_ids)}).

Your output has two parts:

─────────────────────────────────────────────────────
PART 1 — story_context (2-3 sentences)
─────────────────────────────────────────────────────
Write a film synopsis describing the physical reality of this ad: what setting,
who appears (or doesn't), how the product is revealed, what arc the viewer watches.
This is a DIRECTOR'S FILM NOTE — not marketing language, not a product description,
not a tagline. Write it as if briefing a DP on set.

Example (leather bag): "A clean table surface. The bag sits in frame as we push
in on its stitching. A hand lifts it by the handle in the middle of the ad before
we return to the clean product for the close."

─────────────────────────────────────────────────────
PART 2 — beat_visual_directions (EXACTLY {beat_count} entries, one per beat)
─────────────────────────────────────────────────────
For each beat output:
- beat_index: integer (0 through {last})
- focus_feature_truth_id: which truth_id this beat highlights — MUST be a real
  truth_id from the provided table. Never invent one.
- focus_moment: what the viewer notices — sensory and specific
  (e.g. "waxed brown thread caught in warm sidelight", not "the stitching").
- human_presence: "yes" or "no" ONLY. No other values.
- human_action: ONLY include this field when human_presence is "yes".
  Write a 15-30 word action description using a decisive action verb, naming
  the EXACT contact part from the cited truth verbatim.
  Example: "A hand grips the top handle and lifts the bag off a surface,
  the stitching briefly catching the light."
  When human_presence is "no", DO NOT include human_action in the output at all.
- suggested_shot_type: one of exactly: hook_hero, macro_detail, lifestyle_context,
  hero_reframe, cta_endcard, product_in_hand, worn_in_use
- suggested_camera_move: one of exactly: push_in, orbit, static, pan, tilt_up,
  pull_back, rack_focus
- framing_notes: 10-20 word framing/composition note

─────────────────────────────────────────────────────
HARD RULES — non-negotiable
─────────────────────────────────────────────────────
1. CTA beat (beat_index {last}, ALWAYS the last beat): human_presence MUST be "no".
   Non-negotiable. The CTA closes on the product alone. shot_type → cta_endcard.
2. Hook beat (beat_index 0): human_presence is usually "no" — establish the product
   first. Only use "yes" if the VO line for beat 0 explicitly sets up a person.
3. When human_presence is "yes", human_action is REQUIRED and must name the exact
   contact part from the cited truth.
4. When human_presence is "no", do NOT include human_action in the JSON at all.
5. IMPERATIVE ACTION RULE: When any non-CTA beat's VO line is an imperative
   describing a physical action with the product ("Toss it in your bag", "Fill it
   up", "Hold it", "Pour it out", "Press the lid", "Squeeze it", "Drop it",
   "Clip it on", "Carry it", "Drink from it"), human_presence MUST be "yes" —
   the video shows someone doing exactly what the VO commands, regardless of
   whether the product's truths name explicit contact parts.

─────────────────────────────────────────────────────
SHOT TYPE GUIDANCE
─────────────────────────────────────────────────────
- hook, product-alone → hook_hero + push_in
- hook, person present → lifestyle_context + static
- construction/texture detail → macro_detail + push_in
- person carrying/wearing (full body/shoulder) → worn_in_use + static
- person hand-contact with product (grip/lift/press) → product_in_hand + static
- CTA → cta_endcard + static

{human_target_note}

{_scene_inspiration_block(research_facts or [])}─────────────────────────────────────────────────────
OUTPUT FORMAT
─────────────────────────────────────────────────────
Return ONLY valid JSON, no preamble or commentary:

{{
  "story_context": "...",
  "beat_visual_directions": [
    {{
      "beat_index": 0,
      "focus_feature_truth_id": "t1",
      "focus_moment": "the block form emerging from soft dark",
      "human_presence": "no",
      "suggested_shot_type": "hook_hero",
      "suggested_camera_move": "push_in",
      "framing_notes": "fills frame, product center, no negative space"
    }},
    {{
      "beat_index": 2,
      "focus_feature_truth_id": "t4",
      "focus_moment": "waxed thread seams under fingertip pressure",
      "human_presence": "yes",
      "human_action": "A hand presses the front pocket flap, thumb tracing the bar-tack stitching along the top edge.",
      "suggested_shot_type": "product_in_hand",
      "suggested_camera_move": "static",
      "framing_notes": "tight enough to see thread texture and individual stitch spacing"
    }}
  ]
}}"""


def _validate_vda_output(
    result: dict,
    beat_count: int,
    truth_ids: list[str],
) -> list[str]:
    """Validate the parsed VDA JSON. Returns a list of problem strings (empty = valid)."""
    problems: list[str] = []
    bvds = result.get("beat_visual_directions")
    if not isinstance(bvds, list):
        problems.append("beat_visual_directions is missing or not a list")
        return problems

    if len(bvds) != beat_count:
        problems.append(
            f"beat_visual_directions has {len(bvds)} entries but expected exactly {beat_count}"
        )

    seen_indices: set[int] = set()
    for bvd in bvds:
        if not isinstance(bvd, dict):
            problems.append("a beat_visual_directions entry is not a dict")
            continue
        idx = bvd.get("beat_index")
        if not isinstance(idx, int):
            problems.append(f"beat_index is not an int: {idx!r}")
        else:
            seen_indices.add(idx)

        fid = bvd.get("focus_feature_truth_id", "")
        if fid not in truth_ids:
            problems.append(
                f"beat {idx}: focus_feature_truth_id {fid!r} is not in the truth table "
                f"(valid: {', '.join(truth_ids)})"
            )

        stype = bvd.get("suggested_shot_type", "")
        if stype not in _VALID_SHOT_TYPES:
            problems.append(
                f"beat {idx}: suggested_shot_type {stype!r} is not valid "
                f"(valid: {', '.join(sorted(_VALID_SHOT_TYPES))})"
            )

        cmove = bvd.get("suggested_camera_move", "")
        if cmove not in _VALID_CAMERA_MOVES:
            problems.append(
                f"beat {idx}: suggested_camera_move {cmove!r} is not valid "
                f"(valid: {', '.join(sorted(_VALID_CAMERA_MOVES))})"
            )

        hp = bvd.get("human_presence", "")
        if hp not in ("yes", "no"):
            problems.append(f"beat {idx}: human_presence must be 'yes' or 'no', got {hp!r}")
        elif hp == "yes":
            action = bvd.get("human_action", "")
            if not action or not action.strip():
                problems.append(
                    f"beat {idx}: human_presence is 'yes' but human_action is missing or empty"
                )

    # Last beat must be human_presence: "no"
    if bvds:
        last_bvd = bvds[-1]
        if isinstance(last_bvd, dict) and last_bvd.get("human_presence") != "no":
            problems.append(
                f"The last beat (beat_index {last_bvd.get('beat_index')}) must have "
                f"human_presence 'no' (CTA rule) but got {last_bvd.get('human_presence')!r}"
            )

    return problems


def _fallback_bvd(beat_index: int, truth_ids: list[str], is_last: bool) -> BeatVisualDirection:
    return BeatVisualDirection(
        beat_index=beat_index,
        focus_feature_truth_id=truth_ids[0] if truth_ids else "t1",
        focus_moment="product clearly in frame",
        human_presence="no",
        suggested_shot_type="cta_endcard" if is_last else "macro_detail",
        suggested_camera_move="static",
        framing_notes="fills_frame, neutral",
    )


async def generate_visual_direction(
    winning_script: WinningScript,
    product_truths: list[ProductTruth],
    client: Optional[AsyncOpenAI] = None,
    research_facts: Optional[list] = None,
) -> VisualDirection:
    """Run the Visual Direction Agent: one LLM call, one bounded re-prompt on
    validation failure, then per-beat fallback for anything still invalid.
    """
    model = os.environ["MODEL_TEXT"]
    own_client = client is None
    if own_client:
        client = AsyncOpenAI(
            api_key=os.environ["DASHSCOPE_API_KEY"],
            base_url=os.environ["DASHSCOPE_BASE_URL"],
            timeout=90.0,
        )

    beats = winning_script.get("beats", [])
    beat_count = len(beats)
    truth_ids = [t["truth_id"] for t in product_truths]
    human_affordance = human_use_suits_product(product_truths)

    # Build user content: numbered beats + truths table
    beat_lines = "\n".join(
        f"  beat {i} [{b.get('t_start', 0)}-{b.get('t_end', 0)}s]: {b.get('line', '')}"
        for i, b in enumerate(beats)
    )
    user_content = (
        f"VO beats (numbered 0–{beat_count - 1}):\n{beat_lines}\n\n"
        f"Product truths:\n{_format_truths(product_truths)}"
    )

    system_prompt = _build_system_prompt(beat_count, truth_ids, human_affordance, research_facts)

    try:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        response_text = await create_completion(client, model=model, messages=messages, enable_thinking=True)

        try:
            parsed = _parse_json_response(response_text)
        except (json.JSONDecodeError, ValueError) as exc:
            logger.warning("VDA: failed to parse JSON response (%s) -- using full fallback.", exc)
            parsed = {}

        problems = _validate_vda_output(parsed, beat_count, truth_ids)

        if problems:
            logger.info(
                "VDA: %d validation problem(s) on first attempt, re-prompting once: %s",
                len(problems), problems,
            )
            reprompt = (
                "Your response had these problems:\n"
                + "\n".join(f"- {p}" for p in problems)
                + "\n\nFix ALL of these and return the complete corrected JSON "
                "with ALL beats present. Remember:\n"
                "- beat_visual_directions must have EXACTLY "
                f"{beat_count} entries (beat_index 0 through {beat_count - 1})\n"
                f"- The last beat (beat_index {beat_count - 1}) MUST have human_presence: \"no\"\n"
                "- focus_feature_truth_id must be one of: "
                + ", ".join(truth_ids)
                + "\n- suggested_shot_type must be one of: "
                + ", ".join(sorted(_VALID_SHOT_TYPES))
                + "\n- suggested_camera_move must be one of: "
                + ", ".join(sorted(_VALID_CAMERA_MOVES))
                + "\n- When human_presence is \"yes\", human_action is required and must name "
                "the exact contact part from the cited truth."
                + "\n- When human_presence is \"no\", do NOT include human_action at all."
            )
            messages.append({"role": "assistant", "content": response_text})
            messages.append({"role": "user", "content": reprompt})

            retry_text = await create_completion(client, model=model, messages=messages, enable_thinking=True)
            try:
                parsed = _parse_json_response(retry_text)
            except (json.JSONDecodeError, ValueError) as exc:
                logger.warning("VDA: re-prompt JSON parse failed (%s) -- using fallback.", exc)
                parsed = {}

            problems = _validate_vda_output(parsed, beat_count, truth_ids)
            if problems:
                logger.warning(
                    "VDA: still %d problem(s) after re-prompt (%s) -- falling back per-beat.",
                    len(problems), problems,
                )

        # Build the final beat_visual_directions list, using fallback for any bad/missing entry
        raw_bvds: list[dict] = parsed.get("beat_visual_directions", []) if isinstance(
            parsed.get("beat_visual_directions"), list
        ) else []
        bvd_by_index: dict[int, dict] = {}
        for entry in raw_bvds:
            if isinstance(entry, dict) and isinstance(entry.get("beat_index"), int):
                bvd_by_index[entry["beat_index"]] = entry

        # Re-validate each entry individually and fall back as needed
        final_bvds: list[BeatVisualDirection] = []
        for i in range(beat_count):
            is_last = (i == beat_count - 1)
            entry = bvd_by_index.get(i)
            if entry is None:
                logger.info("VDA: missing entry for beat %d -- using fallback.", i)
                final_bvds.append(_fallback_bvd(i, truth_ids, is_last))
                continue

            # Per-entry validity checks
            entry_ok = True
            fid = entry.get("focus_feature_truth_id", "")
            if fid not in truth_ids:
                logger.info("VDA: beat %d has invalid focus_feature_truth_id %r -- fallback.", i, fid)
                entry_ok = False
            stype = entry.get("suggested_shot_type", "")
            if stype not in _VALID_SHOT_TYPES:
                logger.info("VDA: beat %d has invalid suggested_shot_type %r -- fallback.", i, stype)
                entry_ok = False
            cmove = entry.get("suggested_camera_move", "")
            if cmove not in _VALID_CAMERA_MOVES:
                logger.info("VDA: beat %d has invalid suggested_camera_move %r -- fallback.", i, cmove)
                entry_ok = False
            hp = entry.get("human_presence", "")
            if hp not in ("yes", "no"):
                logger.info("VDA: beat %d has invalid human_presence %r -- fallback.", i, hp)
                entry_ok = False
            elif hp == "yes" and not (entry.get("human_action") or "").strip():
                logger.info("VDA: beat %d human_presence=yes but human_action missing -- fallback.", i)
                entry_ok = False
            if is_last and entry.get("human_presence") != "no":
                logger.info("VDA: last beat %d has human_presence != 'no' -- forcing fallback.", i)
                entry_ok = False

            if not entry_ok:
                final_bvds.append(_fallback_bvd(i, truth_ids, is_last))
                continue

            bvd = BeatVisualDirection(
                beat_index=i,
                focus_feature_truth_id=fid,
                focus_moment=(entry.get("focus_moment") or "").strip() or "product clearly in frame",
                human_presence=hp,  # type: ignore[arg-type]
                suggested_shot_type=stype,
                suggested_camera_move=cmove,
                framing_notes=(entry.get("framing_notes") or "").strip() or "fills_frame, neutral",
            )
            if hp == "yes":
                bvd["human_action"] = entry["human_action"].strip()
            final_bvds.append(bvd)

        story_context = (parsed.get("story_context") or "").strip()
        if not story_context:
            story_context = "Product shown in a clean, direct reveal across the ad's beats."

        return VisualDirection(
            story_context=story_context,
            beat_visual_directions=final_bvds,
        )

    finally:
        if own_client:
            await client.close()


async def visual_direction_agent_node(state: ProductCutState) -> dict:
    """LangGraph node wrapper: runs between merge_validator and treatment_agent."""
    _research = state.get("product_research") or {}
    research_facts = _research.get("facts", []) if _research.get("performed") else []
    scene_facts = [
        f for f in research_facts
        if f.get("category") in ("use_case", "visual_moment")
    ]
    vd = await generate_visual_direction(
        winning_script=state["winning_script"],
        product_truths=state.get("product_truths", []),
        research_facts=scene_facts,
    )
    human_beats = sum(1 for b in vd["beat_visual_directions"] if b["human_presence"] == "yes")
    trace_note = (
        f"\n[visual_direction_agent] produced visual direction: "
        f"{human_beats} human beat(s), "
        f"{len(vd['beat_visual_directions'])} total beats."
    )
    return {
        "visual_direction": vd,
        "reasoning_trace": state.get("reasoning_trace", "") + trace_note,
    }
