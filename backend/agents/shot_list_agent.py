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

from agents._affordance import human_use_suits_product
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
    VisualDirection,
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

# Single-Detail Fixation fix (video-gen-fidelity, 2026-07-11, owner-flagged
# issue #1 in docs/BUILD_TASKS.md's Backstory-First section): the live
# leather-bag run's 3-shot list cited the SAME truth (the debossed shield
# logo) for every single shot, cascading the script's own fixation into the
# visuals. Deterministic list-level floor: across the whole shot list, at
# least this many DISTINCT truth_fact_ids must be cited (enforced via the
# existing bounded Call-A re-prompt; degrades with a logged warning, never
# blocks -- and is skipped entirely when the job has fewer distinct truths
# than the floor, or fewer than 2 shots).
MIN_DISTINCT_TRUTHS_ACROSS_SHOTS = 2

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
    # v8 fix (Meta Quest -> "phone on a stand" wrong-object bug). Negative
    # prompts work by "deletion through neutralization" of concrete visual
    # concepts -- a vague abstract phrase like "different object" alone is
    # weak, but naming the SPECIFIC observed failure mode gives it a concrete
    # visual referent to cancel. Deliberately NOT expanded into a long list of
    # every conceivable wrong object (that would burn char budget legitimate
    # per-shot negative terms need) -- just the one empirically observed
    # attractor plus one generic-but-concrete substitution phrase.
    ", object substitution, product transforming into a different object"
)

# Phone-related wrong-object terms are only safe to use as negatives when the
# product itself is NOT phone-related. A phone case, phone stand, or any other
# phone accessory would be hurt by these terms (the main product IS a phone
# or phone-adjacent object). Added conditionally via _build_negative_prompt.
_PHONE_WORDS = frozenset({"phone", "smartphone", "iphone", "android", "mobile"})


def _is_phone_product(product_truths: list) -> bool:
    """True if any form_factor/scale_cue/brief_or_intake_fact mentions phone words."""
    for t in product_truths or []:
        if t.get("category") in ("form_factor", "scale_cue", "brief_or_intake_fact"):
            words = set(t.get("fact", "").lower().split())
            if words & _PHONE_WORDS:
                return True
    return False


def _build_negative_prompt(product_truths: list = None, extra: str = "") -> str:
    """Build the full negative prompt, appending phone-attractor terms only for non-phone products."""
    parts = [NEGATIVE_PROMPT_BOILERPLATE]
    if not _is_phone_product(product_truths):
        parts.append(", smartphone, phone on a stand")
    if extra:
        parts.append(f", {extra}")
    return "".join(parts)

# shot_type values naming the human-interaction composition (C3 v4 addition of
# "worn_in_use" alongside the existing "product_in_hand"). Hand-kept in sync
# with agents/video_gen_node.py's own `_HUMAN_INTERACTION_SHOT_TYPES` (same
# "hand-kept in sync" posture graph/shot_schema.py's docstring already uses for
# its own enum mirroring) -- if this set changes, update that one too.
HUMAN_INTERACTION_SHOT_TYPES = frozenset({"product_in_hand", "worn_in_use"})

# Tighter duration window than the general [MIN_SHOT_DURATION_SEC,
# MAX_SHOT_DURATION_SEC] range -- occlusion and identity drift accumulate
# faster on a shot with a moving human hand/body than on a still-life shot, so
# the safe-length ceiling is lower (research point 4: "duration_sec hard-
# clamped to [3, 4]").
HUMAN_SHOT_MIN_DURATION_SEC = 3.0
HUMAN_SHOT_MAX_DURATION_SEC = 4.0

# Only one motion source at a time on a human-interaction shot: the human's own
# motion is already one source, so stacking a second one (orbit/rack_focus/
# pull_back/tilt_up) compounds drift exactly the way the affordance rubric
# below already warns against for stacked camera moves generally.
HUMAN_SHOT_ALLOWED_CAMERA_MOVES = frozenset({"static", "push_in"})

# Extra negative-prompt terms specific to the human-interaction risk tier,
# appended (never replacing) NEGATIVE_PROMPT_BOILERPLATE -- deterministic, not
# left to Call B's optional negative_prompt_extra, since this risk applies to
# EVERY shot of this shot_type, not just the ones the model happens to flag.
HUMAN_SHOT_NEGATIVE_EXTRA = (
    "deformed hands, extra fingers, fused fingers, product changing size, "
    "product changing color, duplicate product, warped product silhouette, "
    "scene cut"
    # video-gen-fidelity PHASE 1 fix: a real failed job's rendered clip showed
    # almost no motion (a near-static hand resting on the bag against a prompt
    # describing a multi-stage lifting/adjusting action) -- these anti-static
    # terms directly target that failure mode, same "name the specific
    # observed failure mode" posture as the v8 object-substitution terms in
    # NEGATIVE_PROMPT_BOILERPLATE above.
    ", static scene, motionless person, frozen pose, still image"
)

# ---------------------------------------------------------------------------
# HERO SHOT mechanism (video-gen-fidelity story-arc fix).
#
# Text-only i2v prompting cannot lock FACIAL identity across independent Wan
# generations -- each generation is a fully independent call (no video-to-
# video chaining, and per Alibaba's own docs "the same seed does not
# guarantee identical results"). What it CAN reliably hold across independent
# calls is wardrobe color / hair / a named setting, pinned once via
# Treatment.character_anchor and reused verbatim (agents/video_gen_node.py's
# new Cast section). The winning strategy: concentrate the ad's ONE real
# face-visible emotional beat into a SINGLE shot -- the "hero" -- where a
# single continuous generation is already proven to hold identity well
# (this project's own Phase 0 de-risk testing and a later 10s real test both
# stayed identity-consistent across one continuous clip). Every OTHER
# human-interaction shot is faceless (hands/over-shoulder/insert), which
# sidesteps the cross-shot face problem entirely instead of trying to solve it.
#
# HERO IDENTIFICATION MECHANISM. Deliberately NOT a new Shot field -- C1
# (graph/state.py) stays additive-minimal for this fix (only
# Treatment.character_anchor was added, see its v10 changelog note); adding a
# Shot field would also require a graph/shot_schema.py ShotModel change
# (`extra="forbid"`) for no real gain. Instead: a shot IS the hero iff it is
# human-interaction-typed (shot_type in HUMAN_INTERACTION_SHOT_TYPES) AND its
# duration_sec exceeds HUMAN_SHOT_MAX_DURATION_SEC (the ceiling every OTHER
# human-interaction shot is hard-clamped under, below in `_assemble_shots`).
# No other code path in this module can ever produce a human-interaction shot
# above that ceiling except the one hero-assembly branch, so the condition is
# structurally unambiguous -- and it composes for free with every downstream
# reader that already inspects duration_sec (agents/budget_gate.py's
# allocation windows, agents/video_gen_node.py's Cast-section gate) without
# needing a new field threaded through three modules.
HERO_SHOT_MIN_DURATION_SEC = float(os.getenv("HERO_SHOT_MIN_DURATION_SEC", "10.0"))
HERO_SHOT_MAX_DURATION_SEC = float(os.getenv("HERO_SHOT_MAX_DURATION_SEC", "15.0"))

# Backstory-First fix (video-gen-fidelity, 2026-07-11): the flat [10, 15]s
# hero window above was sized for a 30s ad and, applied unscaled to a 15s ad,
# could alone eat the ad's ENTIRE budget even before a ~3s backstory-opening
# shot and a ~2.5-4s CTA claim their own room. Scale BOTH ends of the window
# by the target ad length so a 15s ad's hero gets real-but-proportionate room
# (roughly [5, 7.5]s) while a 30s-target ad keeps the original [10, 15]s
# range exactly (these ratios were chosen so 30 * ratio reproduces the
# original constants precisely -- a no-op at 30s and above).
HERO_SHOT_MIN_DURATION_RATIO = 1.0 / 3.0  # 30 * (1/3) == HERO_SHOT_MIN_DURATION_SEC
HERO_SHOT_MAX_DURATION_RATIO = 0.5        # 30 * 0.5 == HERO_SHOT_MAX_DURATION_SEC

# Fallback target length when the winning script carries no derivable length
# (defensive only -- see `_target_duration_sec` below). Mirrors
# concept_agent.DEFAULT_TARGET_LENGTH_SEC; not imported from there to avoid a
# cross-module dependency for one constant (same posture as this module's
# other small duplicated proxies, e.g. its own `_IMPLIED_PERSON_RE` below).
DEFAULT_TARGET_LENGTH_SEC = 18.0


def _target_duration_sec(winning_script: WinningScript) -> float:
    """The ad's target length in seconds, derived from the winning script.

    C1's `WinningScript` (graph/state.py) carries no `target_length_sec`
    field of its own (same "known gap" posture as concept_agent.py's and
    budget_gate.py's own target_length_sec/budget_cap gaps -- see their
    module docstrings) -- but `winning_script["beats"][-1]["t_end"]` is
    already a reliable proxy: Meta-Critic's `retime_merged_beats` (§Step 6)
    guarantees the merged beats end EXACTLY at target_length_sec, and
    Concept Agent's own `_validate_variant` already rejects any variant whose
    beats don't sum to within 1s of target_length_sec (so even the
    single-survivor short-circuit path, which skips re-timing, is within 1s).
    Falls back to `DEFAULT_TARGET_LENGTH_SEC` only if beats are somehow empty.
    """
    beats = winning_script.get("beats") or []
    if beats:
        try:
            return float(beats[-1].get("t_end", DEFAULT_TARGET_LENGTH_SEC))
        except (TypeError, ValueError):
            pass
    return DEFAULT_TARGET_LENGTH_SEC


def _scaled_hero_window(target_duration_sec: float) -> tuple[float, float]:
    """The [min, max] hero-shot duration window, scaled to `target_duration_sec`
    (see the ratio constants' comment above). Never lets the window invert
    (min > max), and never scales the floor below what `is_hero_shot` itself
    needs to keep recognizing this shot as the hero (duration_sec strictly
    greater than HUMAN_SHOT_MAX_DURATION_SEC) even for a very short target ad.
    """
    scaled_max = min(HERO_SHOT_MAX_DURATION_SEC, target_duration_sec * HERO_SHOT_MAX_DURATION_RATIO)
    scaled_min = min(HERO_SHOT_MIN_DURATION_SEC, target_duration_sec * HERO_SHOT_MIN_DURATION_RATIO)
    scaled_min = min(scaled_min, scaled_max)
    floor = min(HUMAN_SHOT_MAX_DURATION_SEC + 0.5, scaled_max)
    scaled_min = max(scaled_min, floor)
    return scaled_min, scaled_max


# Backstory-First fix: which shot_type the OPENING (hook beat_role) shot
# should default to. Deliberately duplicated (not imported) from
# agents/treatment_agent.py's own `_IMPLIED_PERSON_RE` -- same "each module
# owns its own small text-matching helpers" posture that module's docstring
# already documents for its own duplication from agents/concept_agent.py.
_IMPLIED_PERSON_RE = re.compile(
    r"\b(she|he|her|him|his|hers|they|them|their|theirs|a person|someone|a man|a woman|a hand)\b",
    re.IGNORECASE,
)


def _hook_beat_implies_person(
    winning_script: WinningScript,
    visual_direction: Optional[VisualDirection] = None,
) -> bool:
    """True iff the hook beat (beat 0) should open on a person.

    When VDA is present, uses its authoritative human_presence decision for beat 0
    rather than scanning the VO text (which is now product-focused and rarely
    contains pronouns after the concept-agent VO focus fix). Falls back to the
    old pronoun scan when VDA is absent.
    """
    if visual_direction:
        bvds = visual_direction.get("beat_visual_directions", [])
        if bvds:
            return bvds[0].get("human_presence") == "yes"
    beats = winning_script.get("beats") or []
    if not beats:
        return False
    return bool(_IMPLIED_PERSON_RE.search(beats[0].get("line", "")))


def is_hero_shot(shot: dict) -> bool:
    """True iff `shot` is THE hero shot (see the mechanism note above).

    Imported by agents/budget_gate.py (per-shot allocation floor/ceiling) and
    agents/video_gen_node.py (Cast-section face-visible gate) -- both already
    import other constants from this module (MIN_SHOTS, MIN_SHOT_DURATION_SEC),
    so this follows the same established cross-module dependency direction.
    """
    return (
        shot.get("shot_type") in HUMAN_INTERACTION_SHOT_TYPES
        and shot.get("duration_sec", 0.0) > HUMAN_SHOT_MAX_DURATION_SEC
    )


# Structural facelessness reinforcement for every human-interaction shot
# EXCEPT the hero (see mechanism note above) -- a negative-prompt nudge
# alongside Call B's own faceless-framing instruction, same "never trust an
# LLM instruction alone for a hard constraint" posture as
# HUMAN_SHOT_NEGATIVE_EXTRA above. This is a reinforcing signal, not the
# primary lever -- the primary lever is the Call B system prompt's own
# ONE HERO SHOT, ALL OTHERS FACELESS instruction (positive instructions are
# what empirically move this model family, per
# docs/DERISK_VIDEO_GEN_RESULT.md SS6 -- the negative prompt alone was never
# sufficient for the analogous product-vanishing fix either).
NON_HERO_HUMAN_SHOT_NEGATIVE_EXTRA = "clearly visible face, camera-facing face, facial close-up"

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
NEVER compound/stacked moves — stacked camera moves visibly break current text-to-video models. static/push_in are safest (may run full length); orbit is highest-risk (keep short).

human-contact affordance rubric — a HUMAN-INTERACTION shot_type (product_in_hand / worn_in_use) is MOTIVATED only when a cited fact names a part whose function is human contact — a handle, strap, grip, rim, spout, clasp, zipper pull, button, drawstring, band, trigger, or an equivalent named part — or a scale_cue/form_factor fact establishing the object is hand- or body-scale. When such a fact exists, the shot's contact point MUST be that fact's text, verbatim — same grounding discipline as every other shot. Reject "it's this kind of product, show someone using it" with no such fact cited. When at least one such fact exists among the product's truths, the shot list MUST include at least one human-interaction beat — people buy the moment of use, not the object; a wearable/carryable product shown only as still-life surfaces is a demo reel, not an ad. Omit human-interaction shots ONLY when no such fact exists."""


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

TRUTH DIVERSITY (mandatory): across the WHOLE shot list, cite at least
{MIN_DISTINCT_TRUTHS_ACROSS_SHOTS} DIFFERENT truth_fact_id values -- prefer 3
when the truth table offers them. A shot list where every shot cites the same
single truth is a demo reel of one feature, not an ad for the product: the
shots must collectively show the product's different real aspects (its overall
form, its material/texture, its specific details), not the same detail from
three angles.

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


def _truth_diversity_failures(
    justifications: list[dict], product_truths: list[ProductTruth]
) -> list[dict]:
    """List-level TRUTH DIVERSITY floor (Single-Detail Fixation fix -- see
    MIN_DISTINCT_TRUTHS_ACROSS_SHOTS's comment). Returns validator-shaped
    failure dicts ({shot_id, passed, violation}) for every shot AFTER the
    first when the whole list cites a single truth_fact_id, so the existing
    Call-A re-prompt machinery can name the exact shots to fix (the first
    shot keeps its citation -- only the duplicates are asked to move).
    Empty when the list is already diverse, has < 2 shots, or the job's
    truths can't support diversity in the first place.
    """
    if len(justifications) < 2:
        return []
    available = {t["truth_id"] for t in product_truths}
    if len(available) < MIN_DISTINCT_TRUTHS_ACROSS_SHOTS:
        return []
    cited = {str(j.get("truth_fact_id", "")) for j in justifications}
    if len(cited) >= MIN_DISTINCT_TRUTHS_ACROSS_SHOTS:
        return []
    dup = next(iter(cited))
    alternatives = sorted(available - cited)
    return [
        {
            "shot_id": str(j.get("shot_id", "?")),
            "passed": False,
            "violation": (
                f"every shot in the list cites the same truth_fact_id '{dup}' -- "
                "the shot list must show the product's different aspects, not one "
                f"detail from several angles; re-justify this shot on a DIFFERENT "
                f"truth (e.g. {', '.join(alternatives[:3])}) with a matching "
                "script_quote"
            ),
        }
        for j in justifications[1:]
    ]


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


def _default_shot_type(beat_role: str, hook_implies_person: bool = False) -> str:
    """Deterministic shot_type default -- used as the enum-snap fallback when
    Call B doesn't return a valid shot_type (`_coerce_enum`), or when
    assembling a shot the fallback-justification path produced.

    Backstory-First fix: the "hook" default is now conditional on whether the
    winning script's own beat 0 implies a person (`hook_implies_person`,
    computed once from the winning script in `generate_shot_list` and threaded
    through to `_assemble_shots`) -- a person-establishing hook beat defaults
    to the faceless `lifestyle_context` open instead of the product-alone
    `hook_hero`, matching Call B's own STRUCTURE instruction. Deliberately
    never `product_in_hand`/`worn_in_use` here: those are
    HUMAN_INTERACTION_SHOT_TYPES, and the hero-assignment logic in
    `_assemble_shots` would mistake an opening shot of that type for the
    mid-ad hero and hijack its duration budget.
    """
    if beat_role == "hook":
        return "lifestyle_context" if hook_implies_person else "hook_hero"
    return {"cta": "cta_endcard"}.get(beat_role, "macro_detail")


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


def _build_call_b_system_prompt(
    hero_max_duration_sec: float = HERO_SHOT_MAX_DURATION_SEC,
    hook_implies_person: bool = False,
    has_visual_direction: bool = False,
) -> str:
    hook_structure_note = (
        """This script's hook beat establishes a person in a specific moment (the
winning script's own opening line names/implies them). Open on shot_type
`lifestyle_context`: a FACELESS, scene-establishing shot -- from-behind,
over-the-shoulder, or waist-down framing -- with the product clearly visible
and identifiable within the first frames, but NOT yet the subject of a
close-up. Do NOT use `product_in_hand` or `worn_in_use` for this opening
shot: those are HUMAN-INTERACTION shot_types reserved for the mid-ad hero
below, and using one here would make the hero-assignment logic mistake this
opening shot for the hero and hijack its whole duration budget."""
        if hook_implies_person else
        """This script's hook beat is a claim or curiosity beat about the product
itself (no person established in the opening line). Open on shot_type
`hook_hero`, product ALONE -- never a human-interaction shot_type for this
kind of hook."""
    )
    vda_note = (
        """VISUAL DIRECTION IS GIVEN: each shot's shot_type and camera_move have been
pre-decided by the Visual Direction Agent and are listed per-shot in the user
content below. Use them exactly unless a hard structural rule forces a change
(CTA must always be cta_endcard; human shots must use static/push_in only).
For human-interaction shots, use the provided human_action as the core of the
description's action -- expand it to the full description word count, but keep
that action as the central motion.
For every shot, use the provided focus_moment as the sensory focal point the
description builds toward -- what the viewer's eye should land on at the shot's
peak. Lead the description with a brief ambient setting phrase drawn from the
film context (if given in the user content), so the action is grounded in a
real space rather than floating against white.

"""
        if has_visual_direction else ""
    )
    return f"""{vda_note}You are a cinematographer turning already-justified shots into concrete
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
- duration_sec: a number in [{MIN_SHOT_DURATION_SEC}, {MAX_SHOT_DURATION_SEC}] for
  every ordinary shot. Keep static/push_in at full length; keep orbit/rack_focus
  short. EXCEPTION: if this shot is THE HERO (see STRUCTURE below), duration_sec
  may extend up to {hero_max_duration_sec}s for this ad's target length -- give the
  arc real room, but do not assume the full 15s ceiling applies to every ad length.
- voiceover_line: the script line spoken over this shot.
- description: the Action/Motion text ONLY -- what the subject physically DOES
  in this shot: what starts, what happens, what it ends on. Word count depends
  on the shot (see STRUCTURE/HUMAN-INTERACTION SHOTS below for the human cases);
  an ordinary product-alone shot runs 40-70 words. Camera, lighting,
  framing/composition, and identity-protection (product shape/color/material
  staying constant, hands having five fingers, no scene cut) are ALL added
  separately downstream by the Video-Gen Node's own prompt builder -- do NOT mention camera moves, lighting, framing, composition, or
  any identity-preservation/anti-cut clause here. Writing that language into description
  duplicates content the downstream builder already adds (costing real
  character budget a video-gen prompt has very little of) for zero extra
  benefit. Name the product's REAL color/material (from the cited truth) in
  the FIRST 10-15 words if it naturally belongs in the action, but do not pad
  with static appearance description the reference photo already fixes -- spend
  the words on the actual motion. Use DECISIVE action verbs ("lifts," "settles,"
  "adjusts") -- avoid hedging/softening adverbs ("slowly," "gently,"
  "carefully") unless the beat's own narrative genuinely calls for a
  deliberately slow, tender moment (rare); a hedged verb compounds i2v models'
  documented bias toward under-motion early in a clip into a shot that barely
  reads as moving at all.
- negative_prompt_extra: OPTIONAL short extra risk terms for THIS shot only (a
  shared identity-first negative prompt is already applied; only add per-shot
  risk). Leave "" if none.

STRUCTURE (bookend rule): the OPENING shot must match what the hook beat is
actually doing (Backstory-First fix). {hook_structure_note}
Always close (cta beat_role) on the product ALONE -- never a human-interaction
shot_type (product_in_hand / worn_in_use) for the CTA. This is also the
technically safest choice for the close: an i2v clip's last frames stay
closest to the reference photo, and a clean product-alone shot is the natural
fallback anchor if a riskier human shot elsewhere fails. Concentrate any
human-interaction shots in the demo/proof beats instead. Cap human-interaction
shots at 1-3 per ad -- never zero when the human-contact affordance rubric
below supports one, never every shot. Avoid `context_wide` framing more than
once per ad on a worn_in_use shot -- wide framing makes the product smallest
and hardest to identity-check.

ONE HERO SHOT, ALL OTHERS FACELESS (structural rule, video-gen-fidelity
story-arc fix). Text-only i2v prompting cannot lock FACIAL identity across
independent Wan generations -- each shot is an independent call, no video
chaining, no seed guarantee. So: of all the human-interaction shots you write
(1-3 per the cap above), AT MOST ONE may be face-visible -- this is THE HERO.
Concentrate the ad's real emotional moment there, since a SINGLE continuous
generation holds character identity well. EVERY OTHER human-interaction shot
MUST be faceless framing: hands, over-the-shoulder, from-behind, waist-down,
or an insert on the contact point -- never a visible face, eyes, or
expression. This sidesteps the cross-shot face-consistency problem entirely
(no face in one shot to contradict a different-looking face in another).
- The hero shot gets the extended duration (up to {hero_max_duration_sec}s,
  see duration_sec above) so a real arc can complete -- e.g. lift, travel,
  turn, settle -- not a truncated snippet. Its description should scale up to
  roughly 120-180 words to give that whole arc real content (still Action/
  Motion only, same rule as above -- do not spend the extra words on
  appearance; a separate Cast section is added downstream for that).
- Every OTHER (non-hero) human-interaction shot stays faceless, keeps
  duration_sec inside the ordinary tight human window (hard-clamped downstream
  regardless of what you write here), and keeps description to 40-55 words --
  there is less to show without a face in frame, so there is less to write.

HUMAN-INTERACTION SHOTS (shot_type product_in_hand / worn_in_use) -- extra
phrasing discipline, on top of the description rule above:
- Describe the human MINIMALLY and generically ("a person" / "a hand" + one
  clear action) -- never spend words on human appearance, it steals
  subject-weight from the product in the front-loaded first 20-30 words. This
  applies to the hero shot too -- its extra word budget goes to the ACTION
  arc, never to appearance (a separate Cast section, grounded in the
  treatment's character_anchor, is added downstream and would otherwise be
  duplicated here).
- Name the EXACT contact point using the human-contact fact cited above,
  verbatim: e.g. "a hand enters frame and grips the {{that fact's exact
  text}}" -- not a vague "holds it."
- Do NOT add a scale-lock clause or an occlusion-continuity clause here -- the
  Video-Gen Node's own prompt builder already appends one compressed
  identity-protection sentence covering size/proportion/silhouette/occlusion/
  color/material/no-scene-cut downstream; writing it again in description
  duplicates the same protection twice and burns character budget the action
  text needs.
- Use a clear, DECISIVE action verb for the human action ("lifts," "settles,"
  "adjusts") -- NOT a hedged/softened one ("slowly lifts," "gently slides"),
  unless the beat's own narrative genuinely calls for a deliberately slow,
  tender moment (rare). A real failed shot showed almost no motion because a
  hedged verb ("slowly slings... gently adjusts") combined with i2v models'
  documented bias toward under-motion early in a clip left a 3-4s clip with
  no budget left to render a real action -- default to decisive.
- camera_move for these shot_types must be "static" or "push_in" ONLY (also
  hard-enforced downstream) -- one motion source at a time; a moving human
  plus a moving camera compounds drift.

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


def _format_vda_for_call_b(
    visual_direction: VisualDirection,
    justifications: list[dict],
) -> str:
    """Format VDA per-beat decisions as Call B context. Maps via treatment_ref."""
    bvd_by_beat: dict[int, dict] = {
        bvd["beat_index"]: bvd
        for bvd in visual_direction.get("beat_visual_directions", [])
    }
    lines = ["\nVisual direction per shot (shot_type and camera_move are GIVEN — use them):"]
    for j in justifications:
        beat_idx = _as_beat_index(j.get("treatment_ref"))
        bvd = bvd_by_beat.get(beat_idx) if beat_idx is not None else None
        if not bvd:
            continue
        action_line = (
            f"\n    human_action: {bvd['human_action']}"
            if bvd.get("human_action") else ""
        )
        focus_line = (
            f"\n    focus_moment: {bvd['focus_moment']}"
            if bvd.get("focus_moment") else ""
        )
        lines.append(
            f"  {j['shot_id']}: shot_type={bvd['suggested_shot_type']}, "
            f"camera_move={bvd['suggested_camera_move']}{action_line}{focus_line}\n"
            f"    framing_notes: {bvd.get('framing_notes', '')}"
        )
    return "\n".join(lines)


def _build_call_b_user_content(
    justifications: list[dict],
    truths_by_id: dict[str, ProductTruth],
    treatment: Treatment,
    visual_direction: Optional[VisualDirection] = None,
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
    film_context_line = ""
    if visual_direction:
        story = (visual_direction.get("story_context") or "").strip()
        if story:
            film_context_line = f"Film context (Visual Director's briefing — use as ambient setting for descriptions): {story}\n"

    content = (
        film_context_line
        + f"Director persona: {treatment.get('director_persona', '')}\n"
        f"Color story (derive the shared lighting from this): {treatment.get('color_story', '')}\n"
        f"Pacing philosophy: {treatment.get('pacing_philosophy', '')}\n\n"
        "Realize each of these validated shots:\n" + "\n".join(blocks)
    )
    if visual_direction:
        content += "\n" + _format_vda_for_call_b(visual_direction, justifications)
    return content


def _assemble_shots(
    justifications: list[dict],
    call_b_by_id: dict[str, dict],
    shared_lighting: str,
    truths_by_id: dict[str, ProductTruth],
    target_duration_sec: float = DEFAULT_TARGET_LENGTH_SEC,
    hook_implies_person: bool = False,
) -> list[dict]:
    """Combine each shot's validated justification (Call A) with its camera fields
    (Call B) into a full Shot dict.

    t_start/t_end tile the timeline contiguously by each shot's own duration_sec
    (t_end - t_start == duration_sec), starting at 0 -- a simple, honest mapping
    that keeps position and clip length consistent; finer voiceover sync is the
    Video-Gen / Assembly nodes' job, not this one's.

    HERO ASSIGNMENT (video-gen-fidelity story-arc fix, see the HERO SHOT
    mechanism comment above `is_hero_shot`): the FIRST human-interaction shot
    encountered in `justifications` order becomes THE hero -- deterministically,
    regardless of what Call B's own shot_type/duration choices "intended" --
    and is clamped into the extended hero window instead of the tight human
    window. That window is now SCALED to `target_duration_sec`
    (Backstory-First fix, `_scaled_hero_window`) rather than the flat
    [HERO_SHOT_MIN_DURATION_SEC, HERO_SHOT_MAX_DURATION_SEC]=[10, 15]s window,
    which could alone consume an entire 15s-target ad's budget. EVERY
    subsequent human-interaction shot is force-clamped into the ordinary
    tight window regardless of what Call B proposed, which is what makes
    "at most one hero" a structural guarantee rather than a hoped-for prompt
    outcome: only one shot per assembled list can ever have duration_sec above
    HUMAN_SHOT_MAX_DURATION_SEC, by construction.
    """
    hero_min, hero_max = _scaled_hero_window(target_duration_sec)
    shots: list[dict] = []
    cursor = 0.0
    hero_assigned = False
    for j in justifications:
        b = call_b_by_id.get(j["shot_id"], {})
        beat_role = _coerce_enum(j.get("beat_role"), _BEAT_ROLES, "demo")

        # shot_type and camera_move are resolved BEFORE duration/negative_prompt
        # below -- the human-interaction risk tier (research point 4: never trust
        # an LLM instruction alone for a hard constraint, same posture as the
        # enum-snapping this function already does) clamps duration and
        # camera_move, and extends the negative prompt, based on shot_type.
        default_shot_type = _default_shot_type(
            beat_role, hook_implies_person if beat_role == "hook" else False
        )
        shot_type = _coerce_enum(b.get("shot_type"), _SHOT_TYPES, default_shot_type)
        camera_move = _coerce_enum(b.get("camera_move"), _CAMERA_MOVES, "static")
        is_human_shot = shot_type in HUMAN_INTERACTION_SHOT_TYPES
        is_hero = False

        duration = _clamp_duration(b.get("duration_sec"))
        if is_human_shot:
            if not hero_assigned:
                is_hero = True
                hero_assigned = True
                duration = max(hero_min, min(hero_max, duration))
            else:
                duration = max(HUMAN_SHOT_MIN_DURATION_SEC, min(HUMAN_SHOT_MAX_DURATION_SEC, duration))
        if is_human_shot and camera_move not in HUMAN_SHOT_ALLOWED_CAMERA_MOVES:
            camera_move = "static"

        t_start = cursor
        t_end = round(cursor + duration, 3)
        cursor = t_end

        extra_parts = []
        call_b_extra = (b.get("negative_prompt_extra") or "").strip()
        if call_b_extra:
            extra_parts.append(call_b_extra)
        if is_human_shot:
            extra_parts.append(HUMAN_SHOT_NEGATIVE_EXTRA)
            if not is_hero:
                extra_parts.append(NON_HERO_HUMAN_SHOT_NEGATIVE_EXTRA)
        extra = ", ".join(extra_parts)
        negative_prompt = _build_negative_prompt(list(truths_by_id.values()), extra=extra)
        # The hosted API truncates negative_prompt at 500 chars server-side. We
        # never truncate it ourselves -- flag it so a shot's specific extra risk
        # terms silently lost to server-side truncation are at least visible in
        # logs (same "flag, don't silently hide" posture as video_gen_node.py's
        # prompt-length guard).
        if len(negative_prompt) > 500:
            logger.warning(
                "Shot-List Agent: shot %s negative_prompt is %d chars, over the "
                "hosted API's 500-char truncation limit -- some of its "
                "negative_prompt_extra terms may be silently dropped server-side.",
                j["shot_id"], len(negative_prompt),
            )

        shots.append(
            {
                "shot_id": j["shot_id"],
                "t_start": t_start,
                "t_end": t_end,
                "beat_role": beat_role,
                "description": (b.get("description") or "").strip()
                or f"{truths_by_id.get(j['truth_fact_id'], {}).get('fact', 'the product')}",
                "shot_type": shot_type,
                "camera_move": camera_move,
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
    visual_direction: Optional[VisualDirection] = None,
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
    target_duration_sec = _target_duration_sec(winning_script)
    hook_implies_person = _hook_beat_implies_person(winning_script, visual_direction)

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
        # TRUTH DIVERSITY floor (Single-Detail Fixation fix): a list where every
        # shot cites one truth re-prompts through the same bounded loop, with
        # the duplicated shots named individually so the fix is surgical.
        diversity_failures = _truth_diversity_failures(justifications, product_truths)

        # --- One bounded Call A re-prompt on any failure ---------------------
        if failures or diversity_failures:
            logger.info(
                "Shot-List Agent: %d/%d justifications failed (+%d truth-diversity "
                "flags), re-prompting Call A once.",
                len(failures), len(justifications), len(diversity_failures),
            )
            messages.append({"role": "assistant", "content": raw_a})
            messages.append({
                "role": "user",
                "content": _build_call_a_reprompt(
                    failures + diversity_failures, winning_script, product_truths, treatment
                ),
            })
            raw_a2 = await create_completion(client, model=model, messages=messages, temperature=CALL_A_TEMPERATURE)
            retry_justifications = _parse_json_response(raw_a2).get("shots", [])[:MAX_SHOTS]
            # Only accept retry entries that are actually justification-shaped:
            # KR's validator is deliberately field-presence-driven (a dict with
            # NO script_quote/truth_fact_id keys passes vacuously), so merging a
            # malformed retry entry in would let it slip through re-validation
            # and crash Call B assembly downstream instead of falling back.
            retry_by_id = {
                str(r.get("shot_id")): r
                for r in retry_justifications
                if "script_quote" in r and "truth_fact_id" in r
            }

            # Merge by shot_id rather than replacing the whole list: a re-prompt
            # reply may legitimately contain only the corrected shot(s) ("here is
            # the fixed shot"), not the full set. Every already-valid shot is kept
            # untouched; only originally-failing (or diversity-flagged) shot_ids
            # are swapped for the retry's entry (when the model actually returned
            # one) -- wholesale replacement would silently drop shots that
            # already passed.
            diversity_flagged_ids = {f["shot_id"] for f in diversity_failures}
            justifications = [
                retry_by_id.get(str(j.get("shot_id")), j)
                if (not r.get("passed") or str(j.get("shot_id")) in diversity_flagged_ids)
                else j
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

            if _truth_diversity_failures(justifications, product_truths):
                logger.warning(
                    "Shot-List Agent: shot list still cites a single truth_fact_id "
                    "after the re-prompt -- proceeding degraded (bounded loop, "
                    "never blocks), the fixated list ships rather than nothing."
                )

        if not justifications:
            logger.warning("Shot-List Agent: Call A produced no shots; returning empty shot list.")
            return []
        if len(justifications) < MIN_SHOTS:
            logger.warning("Shot-List Agent: only %d shot(s) (< %d) -- proceeding degraded, not blocking.",
                           len(justifications), MIN_SHOTS)

        # --- Call B: Realize -------------------------------------------------
        shots = await _run_call_b(
            client, model, justifications, truths_by_id, treatment,
            target_duration_sec, hook_implies_person,
            human_affordance=human_use_suits_product(product_truths),
            visual_direction=visual_direction,
        )
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
    target_duration_sec: float = DEFAULT_TARGET_LENGTH_SEC,
    hook_implies_person: bool = False,
    human_affordance: bool = False,
    visual_direction: Optional[VisualDirection] = None,
) -> list[Shot]:
    """Call B + assembly + structural validation, with one bounded Call B retry
    if the assembled list fails structural validation (§5.6 repair posture) OR
    -- Human-Centric Bias fix -- if a product whose truths establish a
    human-use affordance came back with ZERO human-interaction shots (the
    affordance rubric's own "MUST include at least one" rule, enforced
    deterministically rather than trusted to the prompt alone; the live
    leather-bag run produced zero human shots despite strap facts). Both
    triggers share the same single bounded retry; a second miss degrades with
    a logged warning rather than blocking."""
    _, hero_max = _scaled_hero_window(target_duration_sec)
    messages = [
        {
            "role": "system",
            "content": _build_call_b_system_prompt(hero_max, hook_implies_person, has_visual_direction=visual_direction is not None),
        },
        {"role": "user", "content": _build_call_b_user_content(justifications, truths_by_id, treatment, visual_direction)},
    ]

    for attempt in range(2):  # first try + one bounded retry
        raw_b = await create_completion(client, model=model, messages=messages, temperature=CALL_B_TEMPERATURE)
        parsed_b = _parse_json_response(raw_b)
        call_b_by_id = {s.get("shot_id"): s for s in parsed_b.get("shots", [])}
        shared_lighting = (parsed_b.get("lighting") or "").strip() or (
            treatment.get("color_story") or "soft key light, neutral background, clean commercial look"
        )
        assembled = _assemble_shots(
            justifications, call_b_by_id, shared_lighting, truths_by_id,
            target_duration_sec, hook_implies_person,
        )
        try:
            validate_shot_list(assembled)
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
                continue
            # Should not happen given the enum-snapping in assembly; surfacing
            # rather than emitting a structurally-invalid shot list into typed
            # state that every downstream node trusts.
            logger.error("Shot-List Agent: Call B still structurally invalid after retry: %s", exc)
            raise

        has_human_shot = any(
            s.get("shot_type") in HUMAN_INTERACTION_SHOT_TYPES for s in assembled
        )
        # If VDA is present and explicitly assigned human_presence="no" to every
        # beat, respect that deliberate decision -- don't second-guess it with a
        # human-affordance re-prompt that would contradict the VDA's intent.
        vda_suppresses_humans = visual_direction is not None and not any(
            b.get("human_presence") == "yes"
            for b in visual_direction.get("beat_visual_directions", [])
        )
        if human_affordance and not has_human_shot and not vda_suppresses_humans:
            if attempt == 0:
                logger.info(
                    "Shot-List Agent: product truths establish a human-use affordance "
                    "but Call B returned zero human-interaction shots -- re-prompting "
                    "Call B once (human-contact affordance rubric)."
                )
                messages.append({"role": "assistant", "content": raw_b})
                messages.append({
                    "role": "user",
                    "content": (
                        "Your shot list contains ZERO human-interaction shots, but this "
                        "product's own truths name real human-contact parts -- per the "
                        "human-contact affordance rubric, the shot list MUST include at "
                        "least one. Change exactly one demo/proof shot's shot_type to "
                        "product_in_hand or worn_in_use, grounding its contact point in "
                        "the human-contact fact per the rubric (never the opening hook "
                        "shot, never the CTA shot). Return the full corrected JSON in "
                        "the same shape with all shots."
                    ),
                })
                continue
            logger.warning(
                "Shot-List Agent: still zero human-interaction shots after the "
                "re-prompt for a human-suited product -- proceeding degraded, "
                "not blocking."
            )
        return assembled  # type: ignore[return-value]
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
        visual_direction=state.get("visual_direction"),
    )
    trace_note = f"\n[shot_list_agent] produced {len(shots)} shot(s) via two-call justify->realize flow."
    if len(shots) < MIN_SHOTS:
        trace_note += f" Only {len(shots)} survived (< {MIN_SHOTS}) -- degraded, not blocked."
    return {
        "shot_list": shots,
        "reasoning_trace": state.get("reasoning_trace", "") + trace_note,
    }
