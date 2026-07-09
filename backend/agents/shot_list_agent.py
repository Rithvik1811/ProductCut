"""
Shot-List Agent — Qwen via DashScope (Phase 2).
Spec of record: docs/TECHNICAL_DOCUMENTATION.md §5.6.

Converts the winning script + the Treatment Agent's document into a
camera-literate shot list of 3-7 fully-specified `graph.state.Shot` briefs for
the Video-Gen Node. Its whole reason for existing (§2.4) is that this is where
genericness is *structurally blocked* rather than merely discouraged.

TWO SEQUENTIAL QWEN CALLS, not one (Phase 2 research decision, §5.6). The
original single-call design ordered `justification` before the camera fields on
the theory that key-emission order alone would force citation-before-composition.
That guarantee only holds under OpenAI's strict `json_schema` grammar mode;
DashScope's plain `json_object` mode does not grammar-force key order, so on Qwen
the ordering benefit is merely probabilistic. We therefore split the agent:

  * Call A — "Justify" (low temperature, extraction): produces ONLY
    `[{shot_id, beat_role, script_quote, truth_fact_id, treatment_ref}]` per
    shot. Sources are presented to the model as numbered/ID'd menus (script beats
    as a numbered list of exact quotable lines, product_truths as an id->fact
    table, treatment.beat_treatments as a beat_index->visual_approach table), so
    the model can only *select* real quotes/IDs, not invent them.
  * Justification Validator runs on Call A's output BEFORE Call B ever runs
    (see `validate_justifications` below).
  * Call B — "Realize" (warmer, grounded-creative): given each shot's now-
    VALIDATED justification, produces the camera/composition fields. Because the
    justification is a validated fact in the prompt rather than a hoped-for
    token-order effect, each camera choice is provably conditioned on a real
    quote + real truth, not merely correlated with one.

This is the same "validate the small thing, then build on it" shape already used
by the Body-Checker (deterministic pre-pass before its LLM ruling) and the Merge
Coherence Validator (pacing re-check before its blind coherence read) — and
re-prompting a Call A failure re-runs only the tiny justification object, not the
whole shot list, so it converges faster and cheaper.

JUSTIFICATION VALIDATOR OWNERSHIP. KR's real, production Justification Validator
(`agents.justification_validator.validate_justifications`, shared with Treatment
Agent) is now WIRED IN as this module's default -- the swap flagged as a one-line
change below has happened. `_default_validate_justifications` in THIS file is
RR's LOCAL STAND-IN, kept (not deleted) because it's still directly unit-tested
by this file's own test suite as a reference implementation of the same contract,
and remains available via the `validate_justifications` injection point for
anyone testing this module without KR's dependency. It is NOT the active default
anymore and is NOT a claim of ownership over KR's task.

Confirmed compatible on integration: KR's `validate_justifications` takes the
identical positional signature `(justifications, winning_script, product_truths,
treatment)` this module always called with. The one real difference is the result
key -- KR's `ValidationResult` uses `shot_id_or_beat_index` (shared with Treatment
Agent's beat validation), not this module's own stand-in's `shot_id` -- so
`_build_call_a_reprompt` below reads either key (see its docstring). No other
code path in this module reads a validator result's identifier field.

ANTI-GENERICNESS. There is deliberately no `product_category` field anywhere in
the Shot contract (a category field is the seam a lookup-table shortcut would
re-enter through). This module NEVER emits such a field and NEVER lets the model
justify a choice by "category" — the word is banned from generated content, and
`graph.shot_schema.validate_shot_list` (extra="forbid") mechanically rejects any
shot that smuggles one in.

Scope note — what this is NOT (identical posture to body_checker.py):
  * WIRED into the live LangGraph graph (backend/graph/build.py):
    `treatment_agent -> shot_list_agent -> budget_gate`. Was a standalone,
    independently-callable/testable function before the Treatment Agent
    existed; that follow-up wiring has since landed.
  * NOT the Budget Gate (§5.7). `allocated_budget` is emitted as an explicit
    0.0 placeholder here; the Budget Gate node (wired downstream, graph/build.py)
    computes and overwrites the real grounding-weighted allocation. See the
    comment at the assembly site — 0.0 here is a placeholder, never a real cap.
  * NOT the Justification Validator itself (KR) — see the ownership note above.
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Callable, Optional, get_args

from openai import AsyncOpenAI
from pydantic import ValidationError

from agents._retry import create_completion
from agents.justification_validator import (
    validate_justifications as _kr_validate_justifications,
)
from graph.shot_schema import (
    BeatRole,
    CameraMove,
    Framing,
    ShotType,
    TextOverlayZone,
    validate_shot_list,
)
from graph.state import (
    ProductCutState,
    ProductTruth,
    Shot,
    Treatment,
    WinningScript,
)

logger = logging.getLogger("productcut.agents.shot_list_agent")

MIN_SHOTS = 3
MAX_SHOTS = 7

# Call A is structured extraction (§5.6: "not ideation") -> low temperature.
# Call B is a grounded-but-creative decision (which real camera move a real
# fact motivates), so it runs warmer -- the justification is already pinned by
# construction, so the extra sampling variance only affects phrasing/framing,
# not grounding.
CALL_A_TEMPERATURE = 0.2
CALL_B_TEMPERATURE = 0.45

# Shots are short by design (§5.6): drift compounds over longer single-shot
# durations, so every shot is clamped into this window regardless of model output.
MIN_SHOT_DURATION_SEC = 3.0
MAX_SHOT_DURATION_SEC = 5.0
DEFAULT_SHOT_DURATION_SEC = 4.0

# Justification Validator check-4 knobs (§5.6): a quote under 4 words, or one
# that matches a category-generic phrase, is rejected even if it is technically
# verbatim -- those are the "plausible-sounding but says nothing specific" quotes.
MIN_QUOTE_WORDS = 4
_GENERIC_QUOTE_STOPLIST = frozenset(
    {
        "show the product clearly",
        "show the product",
        "highlight quality",
        "highlight the quality",
        "showcase the product",
        "highlight the product",
    }
)

# Enum vocabularies pulled straight off graph.shot_schema's Literals via
# get_args -- so they can never silently drift out of sync with the runtime
# validator (including the v2 additions rack_focus / product_in_hand). These
# feed both the Call B prompt menus and the defensive enum-snapping in assembly.
_SHOT_TYPES: tuple[str, ...] = get_args(ShotType)
_CAMERA_MOVES: tuple[str, ...] = get_args(CameraMove)
_FRAMINGS: tuple[str, ...] = get_args(Framing)
_TEXT_ZONES: tuple[str, ...] = get_args(TextOverlayZone)
_BEAT_ROLES: tuple[str, ...] = get_args(BeatRole)

# Expanded, identity-first negative prompt (§5.6). Earlier terms are weighted
# more heavily by Wan/Kling-family models, so geometry/label/texture-stability
# terms LEAD. Kept as ONE shared boilerplate string (like `lighting`) and only
# extended per-shot for a shot's specific risk -- never reordered, so the
# identity terms always stay in the high-weight leading positions.
NEGATIVE_PROMPT_BOILERPLATE = (
    "warped label, distorted logo, morphing text, melted edges, deformed "
    "packaging, changing product shape, geometry warp, color shift, texture "
    "flicker, warbling surface, extra logos, extra text, watermark, subtitles, "
    "floating product, duplicated product, product leaving frame, background "
    "warping, flickering, jitter, deformed hands, fused fingers, low quality"
)

# Affordance-binding rubric (§5.6): which camera_move is MOTIVATED by which kind
# of cited fact, so Call B's choice is not merely grounded but actually
# motivated. Rendered verbatim into Call B's system prompt.
_AFFORDANCE_RUBRIC = """\
camera_move affordance rubric — a move is MOTIVATED only when the cited fact names the thing in column 2:
- orbit       -> a 3-D form worth circling (a bezel, a faceted surface, a sculpted silhouette). Reject "it's this kind of product, orbit it". Keep orbit a SHORT/PARTIAL 15-30 arc, never a full rotation, never over tight text.
- push_in     -> one arriving detail (a seam, an engraving, a texture). Reject pushing in on nothing in particular.
- tilt_up     -> a vertical geometry worth revealing. Reject when the product isn't tall.
- rack_focus  -> TWO facts on different planes worth linking (e.g. label -> double-wall seam). Reject when only one plane is of interest.
- pull_back   -> a scale/context reveal the product's size makes meaningful. Reject when it reveals nothing new.
- static      -> a claim proven BY stillness. Reject as default laziness.
- pan         -> lateral geometry worth traversing.
NEVER compound/stacked moves — stacked camera moves visibly break current text-to-video models. static/push_in are safest (may run full length); orbit is highest-risk (keep short)."""


# ---------------------------------------------------------------------------
# JSON parsing (shared shape with concept_agent._parse_json_response).
# ---------------------------------------------------------------------------
def _parse_json_response(raw: str) -> dict:
    stripped = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.IGNORECASE)
    return json.loads(stripped)


# ---------------------------------------------------------------------------
# Justification Validator — RR's LOCAL STAND-IN for KR's module.
# ---------------------------------------------------------------------------
def _norm_for_match(text: str) -> str:
    """Lowercase, straighten smart quotes, collapse whitespace -- the 'fuzzy'
    normalization behind the verbatim-substring check (§5.6 check 1 is a
    fuzzy-matched, case-insensitive substring, not a byte-exact compare)."""
    text = (
        text.replace("’", "'")
        .replace("‘", "'")
        .replace("“", '"')
        .replace("”", '"')
    )
    return re.sub(r"\s+", " ", text.lower()).strip()


def _default_validate_justifications(
    justifications: list[dict],
    winning_script: WinningScript,
    product_truths: list[ProductTruth],
    treatment: Treatment,
) -> list[dict]:
    """RR's LOCAL STAND-IN for KR's Justification Validator — NOT RR's deliverable.

    This is the placeholder that lets the Shot-List Agent be fully functional and
    testable today without waiting on KR's real deterministic module. Swapping in
    KR's module later is a one-line change: pass it as the `validate_justifications`
    argument to `generate_shot_list` (or change that parameter's default). It is
    NOT a claim of ownership over KR's task — see this module's docstring.

    Interface contract (agreed in docs/BUILD_TASKS.md Phase 2 "Interface handoff",
    mirrors §5.6):

        validate_justifications(
            justifications: list[dict],   # each {shot_id, beat_role, script_quote,
                                          #       truth_fact_id, treatment_ref}
            winning_script: WinningScript,
            product_truths: list[ProductTruth],
            treatment: Treatment,
        ) -> list[dict]                   # one per shot, IN ORDER:
                                          #   {shot_id, passed: bool, violation: Optional[str]}

    Implements §5.6's four checks, returning the FIRST failing one as `violation`
    so RR's re-prompt can name the exact failure type:
      1. `script_quote` is a verbatim (fuzzy, case-insensitive) substring of
         `winning_script["text"]`.
      2. `truth_fact_id` exists in `product_truths[]`.
      3. `treatment_ref` exists in `treatment["beat_treatments"][].beat_index`.
      4. `script_quote` is >= 4 words AND not a category-generic stoplist phrase.
    """
    script_text_norm = _norm_for_match(winning_script.get("text", ""))
    truth_ids = {t["truth_id"] for t in product_truths}
    beat_indices = {bt["beat_index"] for bt in treatment.get("beat_treatments", [])}

    results: list[dict] = []
    for j in justifications:
        shot_id = str(j.get("shot_id", "?"))
        quote = str(j.get("script_quote", ""))
        truth_fact_id = str(j.get("truth_fact_id", ""))
        treatment_ref = j.get("treatment_ref")
        violation: Optional[str] = None

        # 1. verbatim (fuzzy) substring of the script.
        quote_core = _norm_for_match(quote).strip(" \"'.,!?;:-")
        if not quote_core or quote_core not in script_text_norm:
            violation = "script_quote is not a verbatim span of the winning script text"
        # 2. truth_fact_id exists.
        elif truth_fact_id not in truth_ids:
            violation = f"truth_fact_id '{truth_fact_id}' does not exist in product_truths"
        # 3. treatment_ref exists (accept an int or an int-like string).
        elif _as_beat_index(treatment_ref) not in beat_indices:
            violation = f"treatment_ref '{treatment_ref}' does not exist in treatment.beat_treatments"
        # 4. not too short, not a category-generic phrase.
        elif len(quote.split()) < MIN_QUOTE_WORDS:
            violation = f"script_quote is under {MIN_QUOTE_WORDS} words -- cite a longer specific span"
        elif any(phrase in quote_core for phrase in _GENERIC_QUOTE_STOPLIST):
            violation = "script_quote matches a banned category-generic phrase"

        results.append({"shot_id": shot_id, "passed": violation is None, "violation": violation})
    return results


def _as_beat_index(value) -> Optional[int]:
    """Coerce a treatment_ref to an int beat_index, or None if it isn't one."""
    if isinstance(value, bool):  # bool is an int subclass -- exclude it explicitly
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().lstrip("-").isdigit():
        return int(value.strip())
    return None


# ---------------------------------------------------------------------------
# Source menus for Call A (numbered/ID'd, never prose — §5.6).
# ---------------------------------------------------------------------------
def _beat_menu(winning_script: WinningScript) -> str:
    """Numbered list of the exact script lines Call A must quote from."""
    beats = winning_script.get("beats", [])
    if beats:
        return "\n".join(f"  {i}. \"{b.get('line', '')}\"" for i, b in enumerate(beats))
    # Fallback: no beat breakdown -> offer the raw text as line 0 so there is
    # always at least one quotable span presented as a menu, not prose.
    return f"  0. \"{winning_script.get('text', '')}\""


def _truth_menu(product_truths: list[ProductTruth]) -> str:
    return "\n".join(f"  {t['truth_id']} -> ({t['category']}) {t['fact']}" for t in product_truths)


def _treatment_menu(treatment: Treatment) -> str:
    return "\n".join(
        f"  {bt['beat_index']} -> [{bt['beat_function']}] {bt['visual_approach']}"
        for bt in treatment.get("beat_treatments", [])
    )


def _build_call_a_system_prompt() -> str:
    return f"""You are a shot-list producer breaking a finished ad script into {MIN_SHOTS}-{MAX_SHOTS} shots.

In THIS step you only justify each shot -- you decide nothing about camera,
composition, or wording yet. For every shot output exactly these five fields:
- shot_id: a stable id like "s1", "s2", ...
- beat_role: one of {', '.join(_BEAT_ROLES)}
- script_quote: a span COPIED CHARACTER-FOR-CHARACTER from ONE of the numbered
  script lines you are given. Do not paraphrase, trim to a fragment, or stitch
  two lines together. It must be at least {MIN_QUOTE_WORDS} words and must not be
  a generic filler phrase (e.g. "show the product clearly", "highlight quality").
- truth_fact_id: EXACTLY one of the truth ids from the id -> fact table.
- treatment_ref: EXACTLY one of the beat_index integers from the treatment table.

Every shot must cite a real quote, a real truth id, and a real treatment beat
index -- these are how the shot is grounded in the actual product and script, not
in a template. Never justify a shot by the KIND of product it is.

Return ONLY valid JSON in this exact shape, no preamble or commentary:

{{
  "shots": [
    {{
      "shot_id": "s1",
      "beat_role": "hook",
      "script_quote": "an exact span copied from a numbered line",
      "truth_fact_id": "t2",
      "treatment_ref": 0
    }}
  ]
}}"""


def _build_call_a_user_content(
    winning_script: WinningScript,
    product_truths: list[ProductTruth],
    treatment: Treatment,
) -> str:
    return (
        "Numbered script lines (quote one of these verbatim per shot):\n"
        f"{_beat_menu(winning_script)}\n\n"
        "Product truths (truth_id -> fact):\n"
        f"{_truth_menu(product_truths)}\n\n"
        "Treatment beats (beat_index -> visual_approach):\n"
        f"{_treatment_menu(treatment)}\n\n"
        f"Produce {MIN_SHOTS}-{MAX_SHOTS} shots as JSON."
    )


def _build_call_a_reprompt(
    failures: list[dict],
    winning_script: WinningScript,
    product_truths: list[ProductTruth],
    treatment: Treatment,
) -> str:
    """Surgical re-prompt (§5.6): name each violating shot and failure, then
    re-list the valid menus so the fix is a selection, not another guess.

    Reads `shot_id` OR `shot_id_or_beat_index` from each failure dict: KR's real
    Justification Validator (backend/agents/justification_validator.py, p2kr)
    keys its results `shot_id_or_beat_index` (made generic on purpose so
    Treatment Agent's own beats validate through the same function) rather than
    the `shot_id` this module's own local stand-in (`_default_validate_justifications`
    above) uses. Accepting either here means swapping in KR's real validator
    later via the `validate_justifications` injection point needs zero further
    changes in this file -- the alternative was renaming KR's key back to match
    ours, which is a worse trade (their generic name is the right call for their
    module's dual use)."""
    lines = [
        f"- {f.get('shot_id') or f.get('shot_id_or_beat_index') or '?'}: "
        f"{f.get('violation') or 'failed validation'}"
        for f in failures
    ]
    return (
        "These shots failed grounding validation:\n"
        + "\n".join(lines)
        + "\n\nFix ONLY these shots. Copy script_quote character-for-character from "
        "one of these numbered lines:\n"
        + _beat_menu(winning_script)
        + "\n\nValid truth_fact_id values: "
        + ", ".join(t["truth_id"] for t in product_truths)
        + "\nValid treatment_ref values: "
        + ", ".join(str(bt["beat_index"]) for bt in treatment.get("beat_treatments", []))
        + "\n\nReturn the full corrected JSON with all shots in the same shape."
    )


# ---------------------------------------------------------------------------
# Fallback + assembly helpers.
# ---------------------------------------------------------------------------
def _fallback_justification(shot: dict, treatment: Treatment, index: int) -> dict:
    """Second-failure fallback (§5.6): lift the justification wholesale from the
    corresponding treatment beat, which is already quote-grounded by construction
    (the Treatment Agent validated its own script_quote/truth_fact_id). Preferred
    match is the beat the model *tried* to reference; otherwise the beat at the
    same ordinal position; otherwise the first beat. Never drops the shot."""
    beats = treatment.get("beat_treatments", [])
    if not beats:
        return shot  # nothing to fall back to; leave as-is (defensive, shouldn't happen in Phase 2)

    by_index = {bt["beat_index"]: bt for bt in beats}
    chosen = by_index.get(_as_beat_index(shot.get("treatment_ref")))
    if chosen is None:
        chosen = beats[min(index, len(beats) - 1)]

    return {
        "shot_id": shot["shot_id"],
        # Adopt the treatment beat's own function so beat_role stays consistent
        # with the justification we just lifted from it.
        "beat_role": chosen.get("beat_function", shot.get("beat_role", "demo")),
        "script_quote": chosen["script_quote"],
        "truth_fact_id": chosen["truth_fact_id"],
        "treatment_ref": chosen["beat_index"],
    }


def _coerce_enum(value, allowed: tuple[str, ...], default: str) -> str:
    """Snap an out-of-enum value to `default` (§5.6 repair policy: snap to nearest
    valid enum rather than block). Structural validation is the backstop; this
    just keeps a stray model value from tripping it."""
    return value if value in allowed else default


def _default_shot_type(beat_role: str) -> str:
    return {"hook": "hook_hero", "cta": "cta_endcard"}.get(beat_role, "macro_detail")


def _clamp_duration(value) -> float:
    try:
        d = float(value)
    except (TypeError, ValueError):
        return DEFAULT_SHOT_DURATION_SEC
    return max(MIN_SHOT_DURATION_SEC, min(MAX_SHOT_DURATION_SEC, d))


def _reference_image_id(truth_fact_id: str, truths_by_id: dict[str, ProductTruth]) -> str:
    """A shot references the same seller photo its cited truth came from -- the
    `photo_N` convention `ProductTruth.source` and `jobs.product_photo_refs` use
    elsewhere. Defaults to photo_1 when the truth carries no usable source."""
    truth = truths_by_id.get(truth_fact_id)
    source = (truth or {}).get("source", "")
    return source if source else "photo_1"


def _build_call_b_system_prompt() -> str:
    return f"""You are a cinematographer turning already-justified shots into concrete
video-generation briefs. Each shot's grounding (its exact script quote + the real
product fact + the treatment beat it realizes) is FIXED and given to you -- do not
change it. Your job is only to choose how the camera realizes it.

For every shot produce these fields:
- shot_type: one of {', '.join(_SHOT_TYPES)}
- camera_move: one of {', '.join(_CAMERA_MOVES)}. NEVER compound/stacked moves.
- framing: one of {', '.join(_FRAMINGS)}
- text_overlay_zone: one of {', '.join(_TEXT_ZONES)} -- reserved empty space for a
  caption/CTA composited later. On-screen text is NEVER generated; reserve a zone
  when the shot carries a caption/CTA, otherwise "none".
- duration_sec: a number in [{MIN_SHOT_DURATION_SEC}, {MAX_SHOT_DURATION_SEC}]. Keep
  static/push_in at full length; keep orbit/rack_focus short.
- voiceover_line: the script line spoken over this shot.
- description: the video-gen prompt text, {80}-{120} words, ordered
  Subject -> Action/Motion -> Camera -> Lighting -> Composition -> Mood -> Quality.
  Name the product's REAL color/material/logo (from the cited truth) in the FIRST
  20-30 words -- front-loaded terms carry the most weight -- then spend the rest on
  motion/camera, not on re-describing the static scene the reference photo fixes.
  End with a positive identity-preservation clause ("preserve product shape, keep
  label text, keep proportions"). On any shot with human interaction or a
  transition-adjacent move, ALSO add an anti-cut clause ("product stays centered,
  never leaves frame, no scene cut") -- this positive instruction, not the negative
  prompt, is what stops the product vanishing.
- negative_prompt_extra: OPTIONAL short extra risk terms for THIS shot only (a
  shared identity-first negative prompt is already applied; only add per-shot
  risk). Leave "" if none.

{_AFFORDANCE_RUBRIC}

SWAP TEST — before finalizing each shot ask: if this product were replaced by a
category competitor, would this exact shot still work? If yes, the shot is too
generic -- change the camera_move/framing/description until it is specific to THIS
product's cited fact.

Also return one top-level "lighting" string: a SINGLE shared lighting/style
sentence derived from the treatment's color_story, reused across every shot (do
not vary it per shot).

Never mention a product "category" or justify anything by the kind of product it
is. Never output a product_category field.

Return ONLY valid JSON in this exact shape, no preamble or commentary:

{{
  "lighting": "one shared lighting sentence from the color story",
  "shots": [
    {{
      "shot_id": "s1",
      "shot_type": "hook_hero",
      "camera_move": "push_in",
      "framing": "fills_frame",
      "text_overlay_zone": "none",
      "duration_sec": 4,
      "voiceover_line": "the spoken line",
      "description": "80-120 word prompt ...",
      "negative_prompt_extra": ""
    }}
  ]
}}"""


def _build_call_b_user_content(
    justifications: list[dict],
    truths_by_id: dict[str, ProductTruth],
    treatment: Treatment,
) -> str:
    beats_by_index = {bt["beat_index"]: bt for bt in treatment.get("beat_treatments", [])}
    blocks = []
    for j in justifications:
        truth = truths_by_id.get(j["truth_fact_id"], {})
        beat = beats_by_index.get(_as_beat_index(j["treatment_ref"]), {})
        blocks.append(
            f"- {j['shot_id']} ({j['beat_role']}):\n"
            f"    quote: \"{j['script_quote']}\"\n"
            f"    cited fact: ({truth.get('category', '?')}) {truth.get('fact', '')}\n"
            f"    treatment visual_approach: {beat.get('visual_approach', '')}"
        )
    return (
        f"Director persona: {treatment.get('director_persona', '')}\n"
        f"Color story (derive the shared lighting from this): {treatment.get('color_story', '')}\n"
        f"Pacing philosophy: {treatment.get('pacing_philosophy', '')}\n\n"
        "Realize each of these validated shots:\n" + "\n".join(blocks)
    )


def _assemble_shots(
    justifications: list[dict],
    call_b_by_id: dict[str, dict],
    shared_lighting: str,
    truths_by_id: dict[str, ProductTruth],
) -> list[dict]:
    """Combine each shot's validated justification (Call A) with its camera fields
    (Call B) into a full Shot dict.

    t_start/t_end tile the timeline contiguously by each shot's own duration_sec
    (t_end - t_start == duration_sec), starting at 0 -- a simple, honest mapping
    that keeps position and clip length consistent; finer voiceover sync is the
    Video-Gen / Assembly nodes' job, not this one's.
    """
    shots: list[dict] = []
    cursor = 0.0
    for j in justifications:
        b = call_b_by_id.get(j["shot_id"], {})
        beat_role = _coerce_enum(j.get("beat_role"), _BEAT_ROLES, "demo")
        duration = _clamp_duration(b.get("duration_sec"))
        t_start = cursor
        t_end = round(cursor + duration, 3)
        cursor = t_end

        extra = (b.get("negative_prompt_extra") or "").strip()
        negative_prompt = f"{NEGATIVE_PROMPT_BOILERPLATE}, {extra}" if extra else NEGATIVE_PROMPT_BOILERPLATE

        shots.append(
            {
                "shot_id": j["shot_id"],
                "t_start": t_start,
                "t_end": t_end,
                "beat_role": beat_role,
                "description": (b.get("description") or "").strip()
                or f"{truths_by_id.get(j['truth_fact_id'], {}).get('fact', 'the product')}",
                "shot_type": _coerce_enum(b.get("shot_type"), _SHOT_TYPES, _default_shot_type(beat_role)),
                "camera_move": _coerce_enum(b.get("camera_move"), _CAMERA_MOVES, "static"),
                "framing": _coerce_enum(b.get("framing"), _FRAMINGS, "fills_frame"),
                "lighting": shared_lighting,
                "negative_prompt": negative_prompt,
                "reference_image_id": _reference_image_id(j["truth_fact_id"], truths_by_id),
                "text_overlay_zone": _coerce_enum(b.get("text_overlay_zone"), _TEXT_ZONES, "none"),
                "duration_sec": duration,
                # PLACEHOLDER, NOT a real allocation. The grounding-weighted budget
                # formula (§5.7) belongs to the separate Budget Gate node, which
                # overwrites this field before any money is spent. A future reader
                # must NOT mistake this 0.0 for a computed per-shot cap.
                "allocated_budget": 0.0,
                "voiceover_line": (b.get("voiceover_line") or j["script_quote"]).strip(),
                "justification": {
                    "script_quote": j["script_quote"],
                    "truth_fact_id": j["truth_fact_id"],
                    "treatment_ref": _as_beat_index(j["treatment_ref"]) or 0,
                },
                "status": "pending",
                "retry_count": 0,
            }
        )
    return shots


# ---------------------------------------------------------------------------
# Main entry point.
# ---------------------------------------------------------------------------
async def generate_shot_list(
    winning_script: WinningScript,
    treatment: Treatment,
    product_truths: list[ProductTruth],
    validate_justifications: Callable[..., list[dict]] = _kr_validate_justifications,
    client: Optional[AsyncOpenAI] = None,
) -> list[Shot]:
    """Run the Shot-List Agent: Call A -> validate (-> one bounded Call A re-prompt
    -> per-shot treatment fallback) -> Call B -> assemble -> structural validate.

    `validate_justifications` defaults to KR's real, shared Justification
    Validator (`agents.justification_validator.validate_justifications`) but stays
    injectable -- tests pass RR's local stand-in (`_default_validate_justifications`
    above) or any other conforming callable without touching this function. The
    job is never blocked: a shot that can't be validated after one re-prompt falls
    back to its treatment beat's grounded justification rather than being dropped.
    """
    model = os.environ["MODEL_TEXT"]
    own_client = client is None
    if own_client:
        # Explicit timeout, same rationale as the other agents: the SDK default
        # (~10 min) turns a hung connection into a freeze rather than a fast,
        # retryable failure. 120s covers two full generation calls comfortably.
        client = AsyncOpenAI(
            api_key=os.environ["DASHSCOPE_API_KEY"],
            base_url=os.environ["DASHSCOPE_BASE_URL"],
            timeout=120.0,
        )
    truths_by_id = {t["truth_id"]: t for t in product_truths}

    try:
        # --- Call A: Justify -------------------------------------------------
        messages = [
            {"role": "system", "content": _build_call_a_system_prompt()},
            {"role": "user", "content": _build_call_a_user_content(winning_script, product_truths, treatment)},
        ]
        raw_a = await create_completion(client, model=model, messages=messages, temperature=CALL_A_TEMPERATURE)
        justifications = _parse_json_response(raw_a).get("shots", [])[:MAX_SHOTS]

        results = validate_justifications(justifications, winning_script, product_truths, treatment)
        failures = [r for r in results if not r.get("passed")]

        # --- One bounded Call A re-prompt on any failure ---------------------
        if failures:
            logger.info("Shot-List Agent: %d/%d justifications failed, re-prompting Call A once.",
                        len(failures), len(justifications))
            messages.append({"role": "assistant", "content": raw_a})
            messages.append({
                "role": "user",
                "content": _build_call_a_reprompt(failures, winning_script, product_truths, treatment),
            })
            raw_a2 = await create_completion(client, model=model, messages=messages, temperature=CALL_A_TEMPERATURE)
            retry_justifications = _parse_json_response(raw_a2).get("shots", [])[:MAX_SHOTS]
            retry_by_id = {str(r.get("shot_id")): r for r in retry_justifications}

            # Merge by shot_id rather than replacing the whole list: a re-prompt
            # reply may legitimately contain only the corrected shot(s) ("here is
            # the fixed shot"), not the full set. Every already-valid shot is kept
            # untouched; only originally-failing shot_ids are swapped for the
            # retry's entry (when the model actually returned one) -- wholesale
            # replacement would silently drop shots that already passed.
            justifications = [
                j if r.get("passed") else retry_by_id.get(str(j.get("shot_id")), j)
                for j, r in zip(justifications, results)
            ]
            results = validate_justifications(justifications, winning_script, product_truths, treatment)

            # Second failure for a given shot -> fall back to its treatment beat
            # (grounded by construction) rather than blocking the job.
            repaired: list[dict] = []
            for i, (j, r) in enumerate(zip(justifications, results)):
                if r.get("passed"):
                    repaired.append(j)
                else:
                    logger.info("Shot-List Agent: shot %s still failing (%s) -> treatment-beat fallback.",
                                j.get("shot_id"), r.get("violation"))
                    repaired.append(_fallback_justification(j, treatment, i))
            justifications = repaired

        if not justifications:
            logger.warning("Shot-List Agent: Call A produced no shots; returning empty shot list.")
            return []
        if len(justifications) < MIN_SHOTS:
            logger.warning("Shot-List Agent: only %d shot(s) (< %d) -- proceeding degraded, not blocking.",
                           len(justifications), MIN_SHOTS)

        # --- Call B: Realize -------------------------------------------------
        shots = await _run_call_b(client, model, justifications, truths_by_id, treatment)
        return shots
    finally:
        if own_client:
            await client.close()


async def _run_call_b(
    client: AsyncOpenAI,
    model: str,
    justifications: list[dict],
    truths_by_id: dict[str, ProductTruth],
    treatment: Treatment,
) -> list[Shot]:
    """Call B + assembly + structural validation, with one bounded Call B retry
    if the assembled list fails structural validation (§5.6 repair posture)."""
    messages = [
        {"role": "system", "content": _build_call_b_system_prompt()},
        {"role": "user", "content": _build_call_b_user_content(justifications, truths_by_id, treatment)},
    ]

    for attempt in range(2):  # first try + one bounded retry
        raw_b = await create_completion(client, model=model, messages=messages, temperature=CALL_B_TEMPERATURE)
        parsed_b = _parse_json_response(raw_b)
        call_b_by_id = {s.get("shot_id"): s for s in parsed_b.get("shots", [])}
        shared_lighting = (parsed_b.get("lighting") or "").strip() or (
            treatment.get("color_story") or "soft key light, neutral background, clean commercial look"
        )
        assembled = _assemble_shots(justifications, call_b_by_id, shared_lighting, truths_by_id)
        try:
            validate_shot_list(assembled)
            return assembled  # type: ignore[return-value]
        except ValidationError as exc:
            if attempt == 0:
                logger.warning("Shot-List Agent: Call B output failed structural validation, retrying once: %s", exc)
                messages.append({"role": "assistant", "content": raw_b})
                messages.append({
                    "role": "user",
                    "content": (
                        "Your shot fields failed schema validation:\n"
                        f"{exc}\n\nReturn the full corrected JSON in the same shape. Use ONLY the "
                        "listed enum values and never include a product_category field."
                    ),
                })
            else:
                # Should not happen given the enum-snapping in assembly; surfacing
                # rather than emitting a structurally-invalid shot list into typed
                # state that every downstream node trusts.
                logger.error("Shot-List Agent: Call B still structurally invalid after retry: %s", exc)
                raise
    return assembled  # type: ignore[return-value]  # unreachable; loop returns or raises


async def shot_list_agent_node(state: ProductCutState) -> dict:
    """LangGraph node wrapper: reads winning_script/treatment/product_truths from state.

    WIRED into backend/graph/build.py, downstream of the Treatment Agent
    (`treatment_agent -> shot_list_agent -> budget_gate`). Was standalone and
    independently testable before that; that follow-up wiring has since landed.
    """
    shots = await generate_shot_list(
        winning_script=state["winning_script"],
        treatment=state["treatment"],
        product_truths=state.get("product_truths", []),
    )
    trace_note = f"\n[shot_list_agent] produced {len(shots)} shot(s) via two-call justify->realize flow."
    if len(shots) < MIN_SHOTS:
        trace_note += f" Only {len(shots)} survived (< {MIN_SHOTS}) -- degraded, not blocked."
    return {
        "shot_list": shots,
        "reasoning_trace": state.get("reasoning_trace", "") + trace_note,
    }
