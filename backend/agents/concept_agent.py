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

import pronouncing
from openai import AsyncOpenAI

from agents._affordance import human_use_suits_product
from agents._retry import create_completion
from agents.product_truth_extractor import _wants_imperfection_angle
from graph.state import ProductCutState, ProductTruth, ScriptVariant

logger = logging.getLogger("productcut.agents.concept_agent")

DEFAULT_TARGET_LENGTH_SEC = 15
REQUIRED_VARIANT_COUNT = 4
MIN_VARIANTS_AFTER_DEGRADE = 2
MIN_GROUNDING_TRUTH_IDS = 2
MAX_HOOK_WORDS = 10

# The graph.state.ProductTruth categories that are near-impossible to guess
# without actually looking at the photos -- a real anti-genericness lever, not
# just "cite 2 facts, any 2 facts." See module docstring.
#
# Positive-Only Truths fix (docs/BUILD_TASKS.md "Script Quality (CTA Bridge) +
# Positive-Only Truths + Video-Gen Fidelity Fix" workstream, Problem 1):
# "imperfection" removed -- a flaw must never be the thing that satisfies this
# rule by default (see IMPERFECTION_ANGLE_KEYWORDS / _imperfection_citation_
# problem below for the hard ban on citing one at all). "texture"/"form_factor"
# added in its place: both are genuinely photo-specific, hard-to-guess-
# generically facts per the owner's explicit redirect toward color/style/size/
# positive material-texture as the preferred grounding pool, and narrowing
# this set to "construction_detail" alone (now that imperfection can no longer
# backstop it) would risk making every variant structurally unvalidatable for
# a product with zero construction_detail facts.
SPECIFIC_CATEGORIES = frozenset({"construction_detail", "texture", "form_factor"})

# Positive-Only Truths fix: imperfection-category truths are banned from
# grounding_truth_ids BY DEFAULT -- reuses the identical keyword proxy
# agents/product_truth_extractor.py already applies at extraction time (same
# "seller explicitly asked for an authentic/imperfection angle" signal), so
# the two gates agree rather than drifting independently. In the common case
# the extractor has already filtered these out of product_truths entirely;
# this is the defense-in-depth backstop for any caller that hands
# concept_agent a truth list some other way (e.g. an explicit authentic-angle
# ask, or a future caller that doesn't route through the extractor).
IMPERFECTION_CATEGORY = "imperfection"

# Single-Detail Fixation fix (video-gen-fidelity, 2026-07-11, owner-flagged
# issue #1 in docs/BUILD_TASKS.md's Backstory-First section): the live
# leather-bag run's winning script cited/discussed ONLY the debossed shield
# logo across every beat, even though 7 truths spanning form_factor/texture/
# color/scale_cue/construction/imperfection had been extracted. Root cause:
# the grounding rule above only requires >= 2 truth_ids total plus one
# SPECIFIC one -- it never required a SPREAD, so one vivid detail could
# legally carry a whole script. Two new deterministic levers:
#   1. cited grounding_truth_ids must span at least this many DISTINCT
#      categories (enforced only when the extracted truths themselves span
#      that many -- a degenerate all-one-category truth list degrades
#      gracefully rather than making every variant unvalidatable);
#   2. `_single_truth_fixation_problems` below flags a script whose beats
#      overwhelmingly revolve around ONE truth with no other product truth
#      mentioned anywhere.
MIN_DISTINCT_TRUTH_CATEGORIES = 2

# A variant is "fixated" when >= this many of its beats mention the same
# single truth AND no beat mentions any other truth at all. 3 (not 2) so a
# legitimate hook+payoff pair about one detail never false-positives -- only
# a script that spends essentially its whole runtime on one micro-detail
# trips this, which is exactly the observed live failure.
FIXATION_MIN_BEAT_MENTIONS = 3

# Human-Centric Bias fix (owner-flagged issue #2, same section): when the
# product's own facts establish a human-use affordance (see
# agents/_affordance.py), at least this many of the 4 variants must actually
# commit to an implied person (deterministic proxy: a pronoun-thread beat,
# reusing _first_pronoun_beat_index below). Product-conditional by explicit
# owner requirement -- a product with no human-use affordance keeps the
# original equal-weighting behavior untouched.
MIN_HUMAN_PRESENCE_VARIANTS = 2

# C1's frozen enum (graph.state.ScriptVariant.framework) -- exactly 4 values,
# exactly 4 variants required, so "distinct framework per variant" reduces to
# "each of these 4 used exactly once."
FRAMEWORKS = ("hook_problem_product_cta", "PAS", "AIDA", "BAB")

HOOK_TYPES = (
    "pattern interrupt", "bold claim", "curiosity gap", "direct address",
    "contrarian / myth-busting", "social proof", "POV", "before/after",
    "price anchor", "FOMO / urgency", "how-to", "relatable moment / in-media-res",
)

# Backstory-First fix (video-gen-fidelity, 2026-07-11): which hook_type values
# are "claim-led" (keep the existing number/contrast floor below) vs.
# "story/curiosity" (get a human-moment-marker floor instead -- a bare human
# moment like "She's out the door before sunrise, bag on one shoulder" has no
# digit and no contrast marker, and SHOULDN'T need one; forcing every hook
# through the number/contrast check was mechanism #2 of the flaw-led-hook bug,
# see docs/BUILD_TASKS.md "Backstory-First Script Restructuring" section).
# Anything NOT in either set (direct address / social proof / FOMO-urgency /
# how-to, or a hook_type the model invents outside HOOK_TYPES) keeps the
# existing number/contrast floor -- conservative default, unchanged behavior.
_CLAIM_LED_HOOK_TYPES = frozenset({
    "bold claim", "contrarian / myth-busting", "price anchor", "before/after",
})
_STORY_HOOK_TYPES = frozenset({
    "pov", "curiosity gap", "pattern interrupt", "relatable moment / in-media-res",
})

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


def _build_system_prompt(target_length_sec: int, human_use_suits: bool = False) -> str:
    human_centric_block = (
        f"""HUMAN-CENTRIC BIAS (this product's own facts name real human-contact parts --
a strap, a handle, a body-scale form -- so it is one people wear, carry, or
hold in daily life):
- At least {MIN_HUMAN_PRESENCE_VARIANTS} of the 4 variants must commit to an
  implied person -- a real backstory moment of someone's life WITH the
  product (per the PRONOUN THREAD rule below), not just product surfaces
  narrated at the camera. Human presence in a short-form ad's first second is
  the single strongest scroll-stop lever there is; a wearable/carryable
  product that never shows a person wearing or carrying it is leaving that
  on the table.
- Prefer the HUMAN MOMENT / IN-MEDIAS-RES hook path (HOOK STRENGTH (a)) for
  those variants: open on the person mid-moment, product visible in-scene.
- The remaining variants may stay product-led -- this is a bias, not a
  uniform template; the best-written script still wins.
"""
        if human_use_suits else ""
    )
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

FEATURE SPREAD (mandatory -- sell the WHOLE product, not one detail):
- Each variant's cited truth_ids must span AT LEAST {MIN_DISTINCT_TRUTH_CATEGORIES}
  DIFFERENT categories -- one idiosyncratic detail (imperfection /
  construction_detail, per the rule above) PLUS at least one truth from a
  different category (form_factor, texture, material, color, scale_cue).
  The form_factor truth exists precisely for this: it describes the object
  as a whole and grounds "what this thing IS" before any close-up detail.
- Across a variant's beats, the script must talk about the PRODUCT, not
  orbit one micro-detail. Never spend every beat on the same single feature
  (e.g. an entire script about one logo) -- a viewer who watches the whole
  ad should come away knowing 2-3 different things about this product, not
  one thing said three ways.

VOICE -- SPOKEN, NOT CATALOG (mandatory, every line of every variant):
The product truths are deliberately written as clinical visual observations so
they can be checked against the photos. That phrasing is for CHECKING, never
for SPEAKING. A truth_id is something you POINT AT, not something you QUOTE:
citing a truth means a viewer could verify your line against the photo -- it
does not mean reusing the truth's words. Truths are raw material; every line
you write must sound like one person talking to another on camera.

Hard rules:
1. Never reuse 4 or more consecutive words from any truth's text. Translate
   the observation into what a person would SAY or FEEL about it.
2. Never stack two or more adjectives or hyphenated compound modifiers before
   a noun ("brass-zippered, dome-topped bag" is a photo caption, not speech).
   In hook and CTA lines, allow at most ONE adjective before any noun. Put
   detail AFTER the noun instead: point at it ("that zipper? actual brass")
   or state its consequence ("so the pull won't snap off").
3. Never write an attribute-inventory sentence -- a sentence whose only
   content is listing color, material, or shape. Every mention of a physical
   detail must carry what it MEANS for the person: what it survives, saves,
   signals, or feels like.
4. Never coin abstract compound nouns no one says out loud ("first-mark
   anxiety"). Say the human version ("that pit in your stomach when you
   scratch something new").
5. Use contractions and second person. Vary rhythm: mix short fragments with
   an occasional longer sentence -- uniform medium-length descriptive
   sentences read as a spec sheet.
6. READ-ALOUD TEST: before returning, imagine saying each line to a friend
   while holding the product. If a line could only appear in a product
   listing or an image caption, rewrite it. The camera shows the product;
   your words add the meaning, not a second description.
7. END CONSECUTIVE SENTENCES/BEATS ON UNRELATED SOUNDS. If two nearby lines
   end on rhyming or near-rhyming words, rewrite one so the endings don't
   chime. Read it aloud -- if it could be sung, flatten it.
   WEAK: "Built to last, made to move fast." -- "last"/"fast" rhyme; it
   scans like a jingle, not a person talking.
   STRONG: "It's built to take a beating. And it moves when you do." -- same
   claim, no rhyme, still lands.

CALIBRATION (same truths cited either way -- grounding_truth_ids identical;
only the voice differs):
- WEAK (lifted phrasing, do NOT do this): "This brass-zippered, dome-topped
  bag only gets better." -- three attributes copied out of a visual
  description and stacked in front of the noun. Nobody says this sentence.
- STRONG (same two truths, transformed): "That zipper's real brass. And the
  scuffs? They're the point -- this leather looks better beat up." -- points
  at the zipper, gives the finish a meaning, still checkable against the
  photos, and survives being said out loud.
- WEAK: "Matte, lightly distressed russet-brown leather ages gracefully."
- STRONG: "It's already a little scuffed -- on purpose. So it never looks
  ruined. Just worn in."
None of this loosens grounding: the STRONG lines are MORE specific, because a
consequence can only be written from the real detail, while copied phrasing
is just the truth list read back.

Per-variant requirements:
- The HOOK LINE must directly reference or paraphrase a specific, unusual
  truth -- not a generic industry pain point that could be reskinned onto
  any similar product by swapping the brand name. AVOID stock ad openers
  like "tired of flimsy X," "X is ruining your Y," "say goodbye to X" unless
  the sentence that follows immediately anchors it in a concrete visual
  detail from the truths. If your hook could apply to a *different* product
  in the same rough category with only the brand name changed, rewrite it.
- HOOK STRENGTH: these three hook approaches score EQUALLY -- never treat a
  human-moment opening as weaker than a claim-led one just because it has no
  number:
  (a) HUMAN MOMENT / IN-MEDIAS-RES -- drop the viewer into a specific
      person's specific instant, product visible in-scene but not yet
      pitched. Example: "She's out the door before sunrise, bag already on
      one shoulder."
  (b) CURIOSITY GAP grounded in a real detail, resolved LATER in the script
      -- never a bare, unpaid tease with nothing to cash in.
  (c) CONCRETE CLAIM -- an exact number or measurement, or a contrarian/
      surprising claim resolved in the same line via contrast (a
      "but"/"not"/"unlike"/"instead" turn). Example: "Your coffee is cold in
      12 minutes. Mine isn't." -- a number, then a contrast that resolves it.
  HARD RULE: NEVER open on the product's OWN imperfection or a
  competitor-flaw comparison ("Other bags hide this scuff, not ours.") --
  imperfection-category truths are late-middle material only, framed as
  earned character, never as the hook. At most ONE of the 4 variants may use
  a pain-point opening (PAS-style), and even then the pain must be the
  VIEWER's own situation ("you're always digging for your keys at the
  door"), never the product's own flaw.
- A hook line of {MAX_HOOK_WORDS} words or fewer.
- Exactly one CTA verb -- never two competing calls to action.
- Beat-level timestamps: break the script into small beats (NOT 3 coarse
  hook/body/cta buckets) -- a new beat roughly every 2-3 seconds for the first
  2-3 beats, then every 3-5 seconds after that. Beats must be contiguous and
  sum to exactly {target_length_sec} seconds. Each beat's "line" is the actual
  script text spoken/shown during that beat.

STORY STRUCTURE (beat-order rule, applies to every variant): open on the
person (beat 1: the human-moment or curiosity-gap opening from HOOK STRENGTH
above -- no feature, no flaw, no spec) -> arrive at the product (beats 2-3:
the product enters by name or a clear reference, its features framed as the
PAYOFF of the opening moment, never as an inventory list) -> close on exactly
one CTA verb (the rule above -- unchanged), EARNED not tacked on (see CTA
BRIDGE below).

POSITIVE-ONLY GROUNDING (mandatory, reverses the earlier "imperfection as
earned character" framing): never cite an imperfection-category truth (a
scratch, scuff, wear mark) in ANY variant -- flaws are off by default. Ground
every variant in color, style/silhouette, size, and positive
material/construction facts instead; these are what make the product
desirable and specific, not a defect inventory. Do not treat a debossed
logo/brand mark as the star of the script either -- it is a minor supporting
detail, never the headline truth a variant is built around.

CTA BRIDGE (mandatory): the CTA must land as an earned close, not a
disconnected command. The beat immediately before the CTA must set up the ask
-- name the specific feeling, benefit, or moment just established -- and the
CTA line itself should pick up that thread (a connective like "so"/"that's
why"/"now", or a direct reference back to what was just said) rather than
jump-cutting straight into a bare imperative with no bridge.
  WEAK (abrupt, do NOT do this): "...It's already getting darker right where
  her hands grab it. Grab yours before the next batch sells out." -- the CTA
  shares no thread with the line before it; it just starts a new, unrelated
  imperative.
  STRONG (same idea, bridged): "...It's already getting darker right where
  her hands grab it -- that's the mark of a bag that's actually yours. Go
  make it yours too." -- the CTA explicitly picks up "that's the mark of..."
  before asking.

{human_centric_block}
STORY / REAL-WORLD USE (a narrative/visual choice, distinct from the grounding
rule above -- this framing choice needs no truth_id of its own, the same way a
camera angle or pacing decision doesn't need one either):
- At least 2 of the 4 variants must include at least one beat depicting the
  product in genuine real-world use -- someone wearing/carrying/reaching for/
  using it in a specific daily moment (e.g. "she slings it over one shoulder
  on her way out the door," not "the product sits on a table"). A script that
  never leaves material/construction description reads as a demo reel, not an
  ad that sells an experience -- the moment of someone's life with the product
  is usually the actual sales pitch, not another angle on its surfaces.
- This is a STORY choice, not a factual claim: describing that a moment of use
  happens needs no truth_id. But any physical product detail you mention
  WITHIN that moment (its color, a named part, a texture) still must trace
  back to a real truth_id per the grounding rule above -- invent the moment,
  never invent the product.
- PRONOUN THREAD (one implied person per variant, never a second one): if a
  variant commits to an implied person at all, establish them ONCE -- a
  pronoun plus a minimal moment marker (a time, a state, a place: "on her way
  out the door," "mid-run," "at the door") -- the first time they appear, then
  refer to them with that SAME pronoun in every later beat that mentions them.
  Never reintroduce them a second time as "a person"/"someone"/"a man"/
  "a woman"/"a hand" later in the same variant -- that reads as a second,
  different person, not a continuing story. One implied person per variant,
  one pronoun thread, start to finish.

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


# Backstory-First fix, mechanism #2's replacement floor for story/curiosity
# hook_types (see _STORY_HOOK_TYPES above): a personal pronoun (reusing
# _PRONOUN_RE below -- same file, same "cheap proxy" posture, not duplicated)
# or a second-person address, PLUS a concrete noun beyond it, so a hook
# actually drops the viewer into a specific moment rather than staying a
# content-free address ("You'll love this" has a pronoun but no concrete noun
# and should still fail).
_SECOND_PERSON_RE = re.compile(r"\b(you|your|yours)\b", re.IGNORECASE)
_PLAIN_WORD_RE = re.compile(r"[a-z]+")
_HUMAN_MOMENT_STOPWORDS = frozenset({
    "the", "a", "an", "and", "or", "but", "to", "of", "in", "on", "at", "is",
    "it", "its", "that", "this", "are", "was", "were", "be", "been", "not",
    "so", "as", "with", "from", "she", "he", "her", "him", "his", "hers",
    "they", "them", "their", "theirs", "you", "your", "yours", "one", "out",
    "up", "before", "after", "for",
})


def _has_human_moment_marker(hook: str) -> bool:
    """True iff `hook` drops the viewer into a person's moment: a personal
    pronoun or second-person address, plus a concrete noun/detail beyond it.
    Used as the story/curiosity-type hook floor in place of
    `_hook_has_number_or_contrast` (see `_STORY_HOOK_TYPES`) -- a bare human
    moment like "She's out the door before sunrise, bag on one shoulder" has
    no digit and no contrast marker and should never need one.
    """
    has_person = bool(_PRONOUN_RE.search(hook)) or bool(_SECOND_PERSON_RE.search(hook))
    if not has_person:
        return False
    concrete = [
        w for w in _PLAIN_WORD_RE.findall(hook.lower())
        if len(w) >= 4 and w not in _HUMAN_MOMENT_STOPWORDS
    ]
    return len(concrete) >= 1


# Flaw-led-hook deterministic backstop (Backstory-First fix): catches the
# exact failure mode from the real winning script that triggered this fix
# ("Other bags hide this 2cm scuff, not ours.") even if the prompt's HOOK
# STRENGTH rule above is ignored. Two independent triggers: a literal
# competitor-flaw-comparison pattern, or a hook that lifts language straight
# from an imperfection-category truth (imperfection truths are late-middle
# "earned character" material per STORY STRUCTURE, never hook material).
_FLAW_LED_HOOK_RE = re.compile(
    r"\bother\s+\w+s?\b[^.!?]{0,40}\b(hide|hides|hiding|can't|cant|won't|wont)\b",
    re.IGNORECASE,
)
HOOK_FLAW_LIFT_MIN_RUN = 3  # shorter than LIFT_MIN_RUN -- hook lines are capped at MAX_HOOK_WORDS


def _flaw_led_hook_problem(
    hook: str, cited_truths: list[tuple[str, str]], truth_categories: dict[str, str]
) -> Optional[str]:
    """Flag a hook that opens on the product's own flaw or a competitor-flaw
    comparison -- see the module docstring's mechanism #2/#3 and the STORY
    STRUCTURE prompt rule above (imperfection truths are late-middle only).
    """
    if _FLAW_LED_HOOK_RE.search(hook):
        return (
            f"hook line '{hook}' opens on a competitor-flaw comparison "
            "('other X hide/can't/won't...') -- imperfection-category truths "
            "are late-middle material only, framed as earned character, "
            "never the hook; open on a human moment, curiosity gap, or a "
            "concrete claim about what the product DOES instead"
        )
    for truth_id, fact in cited_truths:
        if truth_categories.get(truth_id) != "imperfection":
            continue
        if _lifted_run(hook, fact, min_run=HOOK_FLAW_LIFT_MIN_RUN):
            return (
                f"hook line '{hook}' directly echoes imperfection-category "
                f"truth {truth_id} -- imperfection truths are late-middle "
                "'earned character' material, never hook material; open on "
                "a human moment, curiosity gap, or a concrete claim instead"
            )
    return None


def _imperfection_citation_problem(
    gti: list[str], truth_categories: dict[str, str], wants_imperfection: bool
) -> Optional[str]:
    """Positive-Only Truths fix: flag ANY citation of an imperfection-category
    truth when the seller didn't ask for an authentic/imperfection angle --
    a harder rule than the hook-only flaw check above (that one only stops a
    flaw from being the OPENING; this stops it being cited anywhere at all,
    matching the extractor-level default of not surfacing negatives in the
    first place, see agents/product_truth_extractor.py).
    """
    if wants_imperfection:
        return None
    imperfection_ids = [t for t in gti if truth_categories.get(t) == IMPERFECTION_CATEGORY]
    if not imperfection_ids:
        return None
    return (
        f"grounding_truth_ids cites imperfection-category truth(s) {imperfection_ids} -- "
        "flaws/wear are off by default (positive-only truths); drop this citation and "
        "ground the variant in a color/style/size/construction/material fact instead, "
        "unless seller_direction explicitly asks for an authentic/well-loved angle"
    )


# CTA Bridge fix (docs/BUILD_TASKS.md "Script Quality (CTA Bridge)..."
# workstream, Problem 2). Owner's words: "even though its human centric, its
# not properly directed, at the end it just says directly shop it. its an
# abrupt end, not proper." Deterministic backstop for the CTA BRIDGE prompt
# rule above, same "crude proxy, errs toward false negatives" posture as
# every other check in this file: only fires on the clearest shape of the
# documented failure (a SHORT bare imperative with no bridging connective and
# no back-reference to the prior beat) -- a real bridge can be phrased many
# ways this can't enumerate, so this only catches the unambiguous case rather
# than risk rejecting a genuinely earned close it doesn't recognize.
_CTA_BRIDGE_CONNECTIVES = (
    "so ", "so,", "so-", "that's why", "that's the", "that's what",
    "which is why", "which means", "now ", "now,", "no wonder", "and that's",
)
_CTA_BRIDGE_REFERENCE_RE = re.compile(r"\b(it|that|this|these|those|one)\b", re.IGNORECASE)
ABRUPT_CTA_MAX_WORDS = 8  # a bare command this short with no bridge reads as tacked-on


def _abrupt_cta_problem(beats: list[dict]) -> Optional[str]:
    """Flag a CTA beat that has neither a bridging connective nor any
    back-reference to the immediately preceding beat -- the exact shape of
    the live-observed failure ("...her hands grab it. Grab yours before the
    next batch sells out."). Only fires when the CTA is also short (a longer
    CTA line has more room to be judged a false positive by this crude a
    proxy); a longer or clearly-bridged CTA is left to the CTA-Checker's own
    scoring rubric instead."""
    if len(beats) < 2:
        return None
    cta_line = beats[-1].get("line", "")
    if not cta_line or len(cta_line.split()) > ABRUPT_CTA_MAX_WORDS:
        return None
    lowered = cta_line.lower()
    has_connective = any(c in lowered for c in _CTA_BRIDGE_CONNECTIVES)
    has_reference = bool(_CTA_BRIDGE_REFERENCE_RE.search(cta_line))
    if has_connective or has_reference:
        return None
    return (
        f"CTA line '{cta_line}' has no bridging connective (so/that's why/now/...) "
        "and no reference back to the prior beat -- reads as a disconnected command "
        "tacked onto the script rather than an earned close; tie it to what the "
        "preceding beat just established (see the CTA BRIDGE rule)"
    )


_WORD_RE = re.compile(r"[a-z0-9']+")
LIFT_MIN_RUN = 4  # matches the VOICE prompt block's "never reuse 4+ consecutive words" rule


def _words(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def _lifted_run(line: str, fact: str, min_run: int = LIFT_MIN_RUN) -> Optional[str]:
    """Return the shared 4+-word run (as a string) if `line` reuses one from `fact`, else None.

    Deliberately crude bag-of-ngrams matching (not a diff/LCS algorithm) --
    a false negative here just means the re-prompt loop misses a subtler lift
    and the STRONG/WEAK prompt calibration has to carry the rest; a false
    positive would wrongly reject grounded copy, which is the worse failure.
    """
    line_words, fact_words = _words(line), _words(fact)
    if len(line_words) < min_run or len(fact_words) < min_run:
        return None
    fact_ngrams = {
        tuple(fact_words[i:i + min_run]) for i in range(len(fact_words) - min_run + 1)
    }
    for i in range(len(line_words) - min_run + 1):
        run = tuple(line_words[i:i + min_run])
        if run in fact_ngrams:
            return " ".join(run)
    return None


# Crude, deliberately not a real grammar parser (per the task's own framing):
# two-or-more hyphenated modifier tokens in a row, immediately followed by a
# bare word -- "brass-zippered, dome-topped bag" is exactly this shape. Good
# enough to name the specific construction in the re-prompt; the model does
# the actual rewriting.
_STACKED_COMPOUND_RE = re.compile(
    r"\b(\w+-\w+),?\s+(\w+-\w+)\s+(\w+)\b", re.IGNORECASE
)


def _stacked_compound_match(line: str) -> Optional[re.Match]:
    return _STACKED_COMPOUND_RE.search(line)


# PRONOUN THREAD backstop (see the STORY/REAL-WORLD USE prompt block above).
# Deliberately crude -- same "cheap proxy, not a real coreference resolver"
# posture as the lift/stacked-compound checks: a personal pronoun's presence
# is treated as "this beat establishes/continues an implied person," and a
# LATER beat's generic indefinite reference is treated as "a second person
# was (re-)introduced." A false negative just means a subtler inconsistency
# slips through to the prompt's own instruction to carry the rest; a false
# positive would wrongly reject a legitimately pronoun-free variant, which is
# why this only fires at all once a pronoun has actually appeared.
_PRONOUN_RE = re.compile(r"\b(she|he|her|him|his|hers|they|them|their|theirs)\b", re.IGNORECASE)
_GENERIC_PERSON_REINTRO_RE = re.compile(r"\b(a person|someone|a man|a woman|a hand)\b", re.IGNORECASE)


def _first_pronoun_beat_index(beats: list[dict]) -> Optional[int]:
    """Index of the first beat whose line uses a personal pronoun -- the point
    at which this variant is treated as having committed to an implied person.
    None if no beat ever uses one (nothing to check -- most variants are
    legitimately person-free)."""
    for i, beat in enumerate(beats):
        if _PRONOUN_RE.search(beat.get("line", "")):
            return i
    return None


def _reintroduction_problems(beats: list[dict]) -> list[str]:
    """Flag any beat AFTER the first pronoun-establishing beat that reverts to
    a generic indefinite reference ("a person"/"someone"/"a man"/"a woman"/
    "a hand") -- per the STORY block's pronoun-thread rule, that reads as a
    second, different implied person rather than a continuation of the first.
    """
    first_idx = _first_pronoun_beat_index(beats)
    if first_idx is None:
        return []
    problems = []
    for i, beat in enumerate(beats):
        if i <= first_idx:
            continue
        line = beat.get("line", "")
        match = _GENERIC_PERSON_REINTRO_RE.search(line)
        if match:
            problems.append(
                f"beat {i} line '{line}' reintroduces the story's person generically "
                f"('{match.group(0)}') after beat {first_idx} already established a "
                "pronoun thread -- refer to the same person with the same pronoun "
                "instead of a second generic introduction"
            )
    return problems


def _voice_problems_for_line(line: str, cited_truths: list[tuple[str, str]]) -> list[str]:
    """Lift + stacked-compound checks for one beat line -- the deterministic
    backstop for the VOICE prompt block above (system prompt is the primary
    lever; this catches what slips through it, per generate_script_variants'
    existing bounded re-prompt-once loop).
    """
    problems = []
    for truth_id, fact in cited_truths:
        run = _lifted_run(line, fact)
        if run:
            problems.append(
                f"beat line '{line}' reuses 4+ consecutive words ('{run}') from truth "
                f"{truth_id} -- translate the observation into spoken language, don't quote it"
            )
            break  # one lift finding per line is enough to name the fix
    match = _stacked_compound_match(line)
    if match:
        problems.append(
            f"beat line '{line}' stacks hyphenated compound modifiers "
            f"'{match.group(1)}' and '{match.group(2)}' before '{match.group(3)}' -- that reads "
            "as a photo caption; keep at most one adjective before a noun and move the rest "
            "after it instead"
        )
    return problems


# ---------------------------------------------------------------------------
# ANTI-RHYME check (Backstory-First fix, VOICE rule 7 above). LLM
# self-assessment of its own rhymes is unreliable (the research behind this
# fix cites ~54% accuracy), so this is a deterministic phonetic check via the
# `pronouncing` package (wraps the CMU Pronouncing Dictionary), wired into the
# same re-prompt loop `_lifted_run`/`_stacked_compound_match` already use.
#
# Deliberately NOT naive same-last-syllable string matching -- that wrongly
# matches e.g. "lessen"/"strengthen" (different rhyme, same trailing letters).
# `pronouncing.rhyming_part()` gives the linguistically correct rhyme unit
# (phones from the last stressed vowel onward), so "lessen" (EH1 S AH0 N) and
# "strengthen" (EH1 NG TH AH0 N) correctly do NOT match.
# ---------------------------------------------------------------------------
RHYME_WINDOW = 3  # compare each clause-final word against the next ~3 clause-final words
_CLAUSE_SPLIT_RE = re.compile(r"[.!?;:,]+|--|—")
_RHYME_STOPWORDS = frozenset({
    "the", "a", "an", "and", "or", "but", "to", "of", "in", "on", "at", "is",
    "it", "its", "that", "this", "are", "was", "were", "be", "been", "not",
    "so", "as", "with", "from", "for", "you", "your", "she", "he", "her",
    "him", "his", "they", "them", "their", "one", "out", "up",
})
# Fallback rhyme key for a word with no CMU dictionary entry: an approximate
# "final vowel cluster + trailing consonants" regex, so an out-of-vocabulary
# word degrades gracefully to a cruder same-ending check rather than crashing
# or silently never matching.
_VOWEL_CODA_RE = re.compile(r"([aeiouy]+[^aeiouy]*)$")


def _clause_final_words(line: str) -> list[str]:
    """The last content word (skipping stopwords/short words) of each clause
    in `line` -- the unit VOICE rule 7 judges rhyme on ("end consecutive
    sentences/beats on unrelated sounds"), not every word in the line.
    """
    finals = []
    for clause in _CLAUSE_SPLIT_RE.split(line):
        words = _PLAIN_WORD_RE.findall(clause.lower())
        for w in reversed(words):
            if len(w) >= 3 and w not in _RHYME_STOPWORDS:
                finals.append(w)
                break
    return finals


def _normalize_rhyme_phones(rhyming_part: str) -> str:
    """Strip stress digits, and canonicalize the documented CMUdict IH/IY
    inconsistency immediately before a coda "R" (e.g. "gear" -> G IH1 R vs.
    "here"/"hear" -> HH IY1 R -- the same NEAR vowel in most dialects, but
    CMUdict transcribes it inconsistently word-to-word) so genuine near-rhymes
    like gear/here are still caught.
    """
    tokens = [re.sub(r"\d", "", t) for t in rhyming_part.split()]
    if len(tokens) >= 2 and tokens[-1] == "R" and tokens[-2] in ("IH", "IY"):
        tokens[-2] = "IH"
    return " ".join(tokens)


def _rhyme_key(word: str) -> Optional[str]:
    """The comparable rhyme unit for `word`, or None if it can't be judged at
    all (empty string). Prefers the real CMU pronunciation via `pronouncing`;
    falls back to a crude vowel-cluster+coda regex for out-of-vocabulary
    words rather than raising or silently never matching.
    """
    if not word:
        return None
    phones = pronouncing.phones_for_word(word)
    if phones:
        return _normalize_rhyme_phones(pronouncing.rhyming_part(phones[0]))
    match = _VOWEL_CODA_RE.search(word)
    return match.group(1) if match else None


def _rhyme_problems(beats: list[dict]) -> list[str]:
    """Flag adjacent clause-final words (within RHYME_WINDOW of each other,
    across beat boundaries too -- a rhyme between the end of beat N and the
    start of beat N+1 reads exactly as sing-song as one within a beat) that
    rhyme per `_rhyme_key`. Identical words are repetition, not rhyme, and are
    skipped. One finding per rhyming pair, naming both beats and both words so
    the re-prompt can name the exact fix.
    """
    entries: list[tuple[int, str]] = []
    for i, beat in enumerate(beats):
        for w in _clause_final_words(beat.get("line", "")):
            entries.append((i, w))

    problems: list[str] = []
    for idx, (beat_a, word_a) in enumerate(entries):
        key_a = _rhyme_key(word_a)
        if key_a is None:
            continue
        for beat_b, word_b in entries[idx + 1: idx + 1 + RHYME_WINDOW]:
            if word_a == word_b:
                continue  # repetition, not rhyme
            key_b = _rhyme_key(word_b)
            if key_b is not None and key_a == key_b:
                problems.append(
                    f"beat {beat_a} ends on '{word_a}' and beat {beat_b} ends on "
                    f"'{word_b}' -- these rhyme (sing-song); rewrite one so "
                    "consecutive lines don't chime"
                )
    return problems


# ---------------------------------------------------------------------------
# SINGLE-TRUTH FIXATION check (Single-Detail Fixation fix -- see the
# MIN_DISTINCT_TRUTH_CATEGORIES comment block near the top of this module).
# Deterministic post-generation backstop in the same re-prompt-loop pattern as
# _rhyme_problems/_flaw_led_hook_problem: a beat "mentions" a truth when it
# shares at least one distinctive content word with that truth's fact text --
# a deliberately crude bag-of-words proxy (NOT semantic matching), which errs
# toward false negatives (a stray shared word with a second truth suppresses
# the flag), the safe direction: a false positive would wrongly bounce a
# legitimately-varied script into the re-prompt loop.
# ---------------------------------------------------------------------------
def _truth_mention_beats(beats: list[dict], truth_facts: dict[str, str]) -> dict[str, set[int]]:
    """Map truth_id -> indices of beats sharing a distinctive content word
    with that truth's fact text (len >= 4, non-stopword)."""
    beat_words: list[set[str]] = [
        {
            w for w in _PLAIN_WORD_RE.findall(beat.get("line", "").lower())
            if len(w) >= 4 and w not in _HUMAN_MOMENT_STOPWORDS
        }
        for beat in beats
    ]
    mentions: dict[str, set[int]] = {}
    for tid, fact in truth_facts.items():
        fact_words = {
            w for w in _PLAIN_WORD_RE.findall(fact.lower())
            if len(w) >= 4 and w not in _HUMAN_MOMENT_STOPWORDS
        }
        hit_beats = {i for i, bw in enumerate(beat_words) if bw & fact_words}
        if hit_beats:
            mentions[tid] = hit_beats
    return mentions


def _single_truth_fixation_problems(beats: list[dict], truth_facts: dict[str, str]) -> list[str]:
    """Flag a script whose beats overwhelmingly revolve around ONE truth with
    no other product truth mentioned anywhere -- the exact live failure this
    fix targets (a whole leather-bag script about nothing but its logo).
    Fires only when exactly one truth is mentioned at all AND it's mentioned
    by >= FIXATION_MIN_BEAT_MENTIONS beats."""
    if not truth_facts or len(truth_facts) < 2:
        return []  # with 0-1 truths there is nothing else the script COULD mention
    mentions = _truth_mention_beats(beats, truth_facts)
    if len(mentions) != 1:
        return []
    (tid, hit_beats), = mentions.items()
    if len(hit_beats) < FIXATION_MIN_BEAT_MENTIONS:
        return []
    return [
        f"beats {sorted(hit_beats)} all revolve around truth {tid} and no other "
        "product truth is mentioned anywhere in the script -- the whole script is "
        "about one micro-detail; rework it so at least one other real product "
        "truth (its overall form, material, texture, another detail) genuinely "
        "features in the copy"
    ]


def _validate_variant(
    variant: dict,
    truth_categories: dict[str, str],
    target_length_sec: int,
    truth_facts: Optional[dict[str, str]] = None,
    wants_imperfection: bool = False,
) -> list[str]:
    """Return a list of violation strings (empty = structurally valid).

    `truth_categories` maps truth_id -> category, for both the "does this
    truth_id actually exist" check and the "did you ground in something
    idiosyncratic, not just generic material/color facts" check below.
    `truth_facts` maps truth_id -> fact text, used only by the VOICE lift
    check; optional (defaults to no lift-check) so existing callers that
    don't have fact text handy don't need to change.
    `wants_imperfection` gates the Positive-Only Truths hard ban (see
    _imperfection_citation_problem) -- defaults to False (the common case:
    flaws are off) so existing callers that don't pass it get the new,
    stricter behavior rather than silently keeping the old one.
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
    else:
        imperfection_problem = _imperfection_citation_problem(gti, truth_categories, wants_imperfection)
        if imperfection_problem:
            problems.append(imperfection_problem)
        # imperfection only counts toward the "cite something idiosyncratic"
        # bar when the seller explicitly asked for that angle -- otherwise the
        # hard ban above already fires, and letting a banned citation ALSO
        # satisfy this bar would be a contradiction (a flaw the script isn't
        # even allowed to use "unlocking" the specificity requirement).
        effective_specific_categories = (
            SPECIFIC_CATEGORIES | {IMPERFECTION_CATEGORY} if wants_imperfection else SPECIFIC_CATEGORIES
        )
        if not any(truth_categories[t] in effective_specific_categories for t in gti):
            problems.append(
                f"grounding_truth_ids {gti} are all generic categories "
                f"(none is {SPECIFIC_CATEGORIES}) -- cite at least one idiosyncratic detail"
            )
        # FEATURE SPREAD (Single-Detail Fixation fix): cited truths must span
        # multiple categories -- but only when the extracted truths themselves
        # do; a degenerate all-one-category truth list degrades gracefully
        # rather than making every variant structurally unvalidatable.
        cited_categories = {truth_categories[t] for t in gti}
        available_categories = set(truth_categories.values())
        if (
            gti
            and len(cited_categories) < MIN_DISTINCT_TRUTH_CATEGORIES
            and len(available_categories) >= MIN_DISTINCT_TRUTH_CATEGORIES
        ):
            problems.append(
                f"grounding_truth_ids {gti} all share one category "
                f"('{next(iter(cited_categories))}') -- cite truths from at least "
                f"{MIN_DISTINCT_TRUTH_CATEGORIES} different categories so the script "
                "sells the whole product, not one detail"
            )

    beats = variant.get("beats") or []
    if not beats:
        problems.append("no beats")
    else:
        total = beats[-1].get("t_end", 0) - beats[0].get("t_start", 0)
        if abs(total - target_length_sec) > 1:
            problems.append(f"beats span {total}s, expected ~{target_length_sec}s")
        hook_line = _hook_line(variant)
        hook_words = len(hook_line.split())
        if hook_words > MAX_HOOK_WORDS:
            problems.append(f"hook line is {hook_words} words, expected <= {MAX_HOOK_WORDS}")

        cited_truths = [
            (tid, truth_facts[tid]) for tid in gti if truth_facts and tid in truth_facts
        ]

        # Backstory-First fix, mechanism #2: gate the hook floor by hook_type
        # instead of applying the same number/contrast requirement to every
        # hook regardless of its declared angle -- a story/curiosity hook is
        # judged on whether it actually drops the viewer into a moment, a
        # claim-led hook keeps the original number/contrast floor.
        hook_type_lower = str(variant.get("hook_type", "")).strip().lower()
        if hook_type_lower in _STORY_HOOK_TYPES:
            if not _has_human_moment_marker(hook_line):
                problems.append(
                    f"hook line '{hook_line}' (hook_type '{variant.get('hook_type')}') has no "
                    "personal pronoun/second-person address plus a concrete noun -- a "
                    "story/curiosity hook must drop the viewer into a specific moment, not "
                    "stay abstract"
                )
        elif not _hook_has_number_or_contrast(hook_line):
            problems.append(
                f"hook line '{hook_line}' has no number and no contrast marker "
                f"({', '.join(_CONTRAST_MARKERS)}) -- reads as a bare tease, not a claim"
            )

        flaw_problem = _flaw_led_hook_problem(hook_line, cited_truths, truth_categories)
        if flaw_problem:
            problems.append(flaw_problem)

        for beat in beats:
            line = beat.get("line", "")
            if line:
                problems.extend(_voice_problems_for_line(line, cited_truths))

        problems.extend(_reintroduction_problems(beats))
        problems.extend(_rhyme_problems(beats))
        if truth_facts:
            problems.extend(_single_truth_fixation_problems(beats, truth_facts))

        abrupt_cta_problem = _abrupt_cta_problem(beats)
        if abrupt_cta_problem:
            problems.append(abrupt_cta_problem)

    return problems


def _split_valid_invalid(
    variants: list[dict],
    truth_categories: dict[str, str],
    target_length_sec: int,
    truth_facts: Optional[dict[str, str]] = None,
    wants_imperfection: bool = False,
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
        problems = _validate_variant(
            v, truth_categories, target_length_sec, truth_facts, wants_imperfection
        )
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


def _human_presence_count(variants: list) -> int:
    """How many variants commit to an implied person -- the deterministic
    proxy (a pronoun-thread beat, reusing `_first_pronoun_beat_index`) behind
    the HUMAN-CENTRIC BIAS rule's floor. Works on both raw variant dicts and
    validated ScriptVariants (both carry `beats`)."""
    return sum(
        1 for v in variants
        if _first_pronoun_beat_index(v.get("beats") or []) is not None
    )


def _reprompt_message(
    invalid: list[tuple[dict, list[str]]],
    valid_count: int,
    human_shortfall: bool = False,
) -> str:
    parts: list[str] = []
    if invalid:
        lines = []
        for v, problems in invalid:
            vid = v.get("variant_id", "?")
            for p in problems:
                lines.append(f"- {vid}: {p}")
        parts.append(
            "The following problems were found in your response:\n" + "\n".join(lines)
        )
    elif valid_count < REQUIRED_VARIANT_COUNT:
        parts.append(
            f"You returned only {valid_count} script variant(s), but exactly "
            f"{REQUIRED_VARIANT_COUNT} are required, still distinct in "
            "framework/hook_type/emotional_trigger."
        )
    if human_shortfall:
        parts.append(
            f"Fewer than {MIN_HUMAN_PRESENCE_VARIANTS} of your variants commit to "
            "an implied person, but this product's own facts name real "
            "human-contact parts (see the HUMAN-CENTRIC BIAS rule): rewrite enough "
            f"variants so at least {MIN_HUMAN_PRESENCE_VARIANTS} of the 4 open on "
            "or carry a real moment of someone wearing/carrying/using the product "
            "(one pronoun thread per variant, per the PRONOUN THREAD rule), while "
            "keeping every other rule intact."
        )
    parts.append(
        "Fix ONLY these specific issues and return the full corrected JSON "
        "object in the same shape, still with exactly 4 variants."
    )
    return "\n\n".join(parts)


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
    truth_facts = {t["truth_id"]: t["fact"] for t in product_truths}
    human_bias = human_use_suits_product(product_truths)
    # Positive-Only Truths fix: same seller_direction-derived signal
    # agents/product_truth_extractor.py already applies at extraction time
    # (see _imperfection_citation_problem's docstring on why the two gates
    # must agree).
    wants_imperfection = _wants_imperfection_angle(
        brief, (seller_direction or {}).get("freeform")
    )

    try:
        messages = [
            {"role": "system", "content": _build_system_prompt(target_length_sec, human_bias)},
            {"role": "user", "content": _build_user_content(brief, product_truths, seller_direction)},
        ]

        response_text = await create_completion(client, model=model, messages=messages)
        parsed = _parse_json_response(response_text)
        raw_variants = parsed.get("script_variants", [])
        valid, invalid = _split_valid_invalid(
            raw_variants, truth_categories, target_length_sec, truth_facts, wants_imperfection
        )

        # Per spec: fewer than 4 variants is its own re-prompt trigger, even
        # when nothing else was individually wrong (the model just under-delivered).
        # Human-Centric Bias fix: for a product whose facts establish a human-use
        # affordance, too few person-committed variants is ALSO a re-prompt
        # trigger (deterministic floor behind the HUMAN-CENTRIC BIAS prompt rule).
        human_shortfall = (
            human_bias and _human_presence_count(valid) < MIN_HUMAN_PRESENCE_VARIANTS
        )
        needs_reprompt = len(valid) < REQUIRED_VARIANT_COUNT or human_shortfall
        if needs_reprompt:
            logger.info(
                "Concept Agent: %d/%d variants valid (human-presence shortfall: %s), "
                "re-prompting once (%d problems)",
                len(valid), REQUIRED_VARIANT_COUNT, human_shortfall, len(invalid),
            )
            messages.append({"role": "assistant", "content": response_text})
            messages.append({
                "role": "user",
                "content": _reprompt_message(invalid, len(valid), human_shortfall=human_shortfall),
            })
            retry_text = await create_completion(client, model=model, messages=messages)
            retry_parsed = _parse_json_response(retry_text)
            retry_valid, _ = _split_valid_invalid(
                retry_parsed.get("script_variants", []),
                truth_categories, target_length_sec, truth_facts, wants_imperfection,
            )
            retry_better = len(retry_valid) > len(valid) or (
                human_shortfall
                and len(retry_valid) >= len(valid)
                and _human_presence_count(retry_valid) > _human_presence_count(valid)
            )
            if retry_better:
                valid = retry_valid
            if human_shortfall and _human_presence_count(valid) < MIN_HUMAN_PRESENCE_VARIANTS:
                logger.warning(
                    "Concept Agent: still only %d/%d person-committed variant(s) after "
                    "re-prompt for a human-suited product -- proceeding degraded, the "
                    "Hook-Checker's human-moment tiebreak is the remaining lever.",
                    _human_presence_count(valid), MIN_HUMAN_PRESENCE_VARIANTS,
                )

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
