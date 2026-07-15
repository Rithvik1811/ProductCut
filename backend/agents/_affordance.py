"""
Shared deterministic "does human use suit this product?" signal.

Human-Centric Bias fix (branch `video-gen-fidelity`, 2026-07-11, second
owner-flagged issue in docs/BUILD_TASKS.md's "Backstory-First" section): the
project owner wants ads to lean human-centric (a person visibly using the
product, a human-moment opening) MORE OFTEN — but explicitly conditioned on
the product actually suiting it ("its not like every product should have the
same human generic way"), never as a universal template.

This module is that condition, computed deterministically from the extracted
product truths rather than trusted to any LLM's own judgment (same posture as
every other hard gate in this codebase — e.g. treatment_agent's
`_script_implies_person` guard). A product "suits" human-centric framing iff
its own observed facts establish a real human-use affordance:

  1. any fact names a part whose function IS human contact (a strap, handle,
     grip, clasp, zipper pull, drawstring, buckle, ...) — the same part
     vocabulary agents/shot_list_agent.py's human-contact affordance rubric
     already uses for Call B, minus the parts that exist on nearly every
     manufactured object (a button/rim alone doesn't make something wearable
     or carryable — that would turn this into an always-true signal and
     defeat the owner's "product-dependent" requirement); or
  2. a `scale_cue`/`form_factor` fact explicitly places the object on a human
     body or in a hand (mentions a shoulder, wrist, palm, pocket, ...).

Cheap keyword proxy, deliberately NOT semantic analysis — identical "cheap
proxy, crude on purpose" posture as concept_agent's `_PRONOUN_RE`/
`_lifted_run` and treatment_agent's `_IMPLIED_PERSON_RE`. A false negative
here just means a genuinely-wearable product misses the extra human-centric
bias (the pre-fix status quo, not a new failure); a false positive would
force human framing onto a product it doesn't fit, which is the worse
failure — hence the conservative part list.

Shared as its own module (same precedent as agents/_oss.py) because three
agents consume the SAME signal and it must never drift between them:
concept_agent (prompt bias + re-prompt trigger), hook_checker (scoring
tiebreak), shot_list_agent (zero-human-shot Call B retry).
"""
from __future__ import annotations

import re

from graph.state import ProductTruth

# Parts whose function is human contact — the wearable/carryable/graspable
# subset of the Call B rubric's vocabulary (see module docstring for why
# button/rim/trigger are deliberately excluded here despite appearing there).
_HUMAN_CONTACT_PART_WORDS = frozenset({
    "handle", "strap", "grip", "clasp", "drawstring", "buckle", "lanyard",
    "sling", "harness", "lace", "toggle", "wristband", "armband", "headband",
    "earcup", "earbud", "nosepad", "waistband", "shoulder",
    # Direct hand-use goods (pans, kitchen tools, cutting boards, etc.) — no
    # "carry" part needed; these objects are inherently hand-operated.
    "pan", "pot", "lid", "knob", "spatula", "utensil", "blade", "knife",
    "tong", "ladle", "whisk", "board", "rack", "tray", "platter",
})

# Truth categories that describe the object's overall size/shape — the only
# categories where a body-scale word is read as "this object lives on a
# person" rather than an incidental mention.
_SCALE_CATEGORIES = frozenset({"scale_cue", "form_factor"})

_BODY_SCALE_WORDS = frozenset({
    "hand", "palm", "wrist", "shoulder", "waist", "hip", "neck", "torso",
    "back", "chest", "pocket", "arm", "leg", "ankle", "head", "ear", "body",
    "wearable", "handheld",
    # Action verbs that appear in scale_cue/form_factor for hand-operated goods
    "held", "lifted", "gripped", "poured", "tilted", "maneuvered", "grasped",
})

_WORD_RE = re.compile(r"[a-z]+")


def _fact_words(fact: str) -> set[str]:
    """Lowercased words with a crude plural strip, so 'straps' matches 'strap'."""
    words = set()
    for w in _WORD_RE.findall(fact.lower()):
        words.add(w)
        if len(w) > 3 and w.endswith("s"):
            words.add(w[:-1])
    return words


def human_use_suits_product(product_truths: list[ProductTruth]) -> bool:
    """True iff the product's own observed facts establish a human-use
    affordance (see module docstring for the two triggers). Deterministic —
    the gate for every product-conditional human-centric bias downstream.
    """
    for truth in product_truths or []:
        words = _fact_words(truth.get("fact", ""))
        if words & _HUMAN_CONTACT_PART_WORDS:
            return True
        if truth.get("category") in _SCALE_CATEGORIES and words & _BODY_SCALE_WORDS:
            return True
    return False
