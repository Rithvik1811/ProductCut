"""
Unit tests for the Concept Agent's validation/re-prompt/degrade logic --
same rationale as test_product_truth_extractor.py: these code paths (malformed
variant, duplicate framework/hook, degrade to best-N, single-survivor
short-circuit) are unlikely to fire reliably against the real model on demand,
so they're covered with a fake client instead.

Product coverage (3+ product types):
  ELECTRONICS_TRUTHS  -- a Bluetooth speaker (no human-use affordance)
  BAG_TRUTHS          -- a leather shoulder bag (has strap -> human-use affordance)
  SHOE_TRUTHS         -- a running shoe (has foot/sole contact -> human-use affordance)
  SKINCARE_TRUTHS     -- a face serum (no ergonomic affordance)

Tests verify BEHAVIORAL PROPERTIES ("does the validator reject rhymes?") not
implementation strings ("does the prompt say 'brass-zippered'?").
"""
from __future__ import annotations

import json

import pytest

from agents.concept_agent import (
    FRAMEWORKS,
    MIN_VARIANTS_AFTER_DEGRADE,
    REQUIRED_VARIANT_COUNT,
    _abrupt_cta_problem,
    _build_system_prompt,
    _flaw_led_hook_problem,
    _has_human_moment_marker,
    _human_presence_count,
    _imperfection_citation_problem,
    _rhyme_problems,
    _single_truth_fixation_problems,
    _validate_variant,
    generate_script_variants,
)
from tests._fakes import FakeOpenAIClient


# ---------------------------------------------------------------------------
# Product-type fixtures — 4 different product categories
# ---------------------------------------------------------------------------

# Electronics: Bluetooth portable speaker.  No strap/handle -> no human-use
# affordance, so the HUMAN-CENTRIC BIAS block must NOT appear in the prompt.
ELECTRONICS_TRUTHS = [
    # e0: form_factor — required by MANDATORY TIER 1 (WHOLE-OBJECT IDENTITY)
    {"truth_id": "e0", "fact": "compact cylindrical body roughly the size of a tall can", "category": "form_factor", "source": "photo_1"},
    {"truth_id": "e1", "fact": "matte black anodized aluminum grille", "category": "material", "source": "photo_1"},
    {"truth_id": "e2", "fact": "dual passive radiator ports on each end cap", "category": "construction_detail", "source": "photo_1"},
    # e3: imperfection truths are stored as "material_character" (matching IMPERFECTION_CATEGORY)
    {"truth_id": "e3", "fact": "hairline scratch across the lower left grille panel", "category": "material_character", "source": "photo_2"},
    {"truth_id": "e4", "fact": "deep navy blue colorway", "category": "color", "source": "photo_1"},
    # Second construction_detail truth so the FEATURE SPREAD check can be tested.
    {"truth_id": "e5", "fact": "recessed rubber USB-C port with a hinged waterproof flap", "category": "construction_detail", "source": "photo_3"},
]

# Leather shoulder bag — carries a strap -> human_use_suits_product returns True.
BAG_TRUTHS = [
    {"truth_id": "b0", "fact": "structured rectangular tote with a flat reinforced base", "category": "form_factor", "source": "photo_1"},
    {"truth_id": "b1", "fact": "pebbled russet-brown leather front panel", "category": "material", "source": "photo_1"},
    {"truth_id": "b2", "fact": "adjustable shoulder strap with a stitched pad", "category": "construction_detail", "source": "photo_2"},
    {"truth_id": "b3", "fact": "pale compression halo around the debossed mark", "category": "material_character", "source": "photo_1"},
]

# Running shoe — has a sole/foot contact point -> human_use_suits_product
# returns True (strap/handle/scale_cue facts).
SHOE_TRUTHS = [
    {"truth_id": "s0", "fact": "low-top lace-up silhouette with a chunky foam sole", "category": "form_factor", "source": "photo_1"},
    {"truth_id": "s1", "fact": "high-rebound foam midsole with visible bubble cell pattern", "category": "material", "source": "photo_1"},
    {"truth_id": "s2", "fact": "carbon-fibre tension plate embedded mid-stack", "category": "construction_detail", "source": "photo_2"},
    {"truth_id": "s3", "fact": "small lateral scuff on the heel counter from a prior run", "category": "material_character", "source": "photo_3"},
    {"truth_id": "s4", "fact": "neon volt upper mesh with a reflective heel tab", "category": "color", "source": "photo_1"},
    {"truth_id": "s5", "fact": "asymmetric lace eyelet row with reinforced brass grommets", "category": "construction_detail", "source": "photo_4"},
]

# Skincare face serum — no ergonomic human-use affordance.
SKINCARE_TRUTHS = [
    {"truth_id": "k0", "fact": "slim 30ml dropper bottle with a rounded shoulder", "category": "form_factor", "source": "photo_1"},
    {"truth_id": "k1", "fact": "translucent amber glass dropper bottle", "category": "material", "source": "photo_1"},
    {"truth_id": "k2", "fact": "pharmaceutical-grade silicone dropper tip with a precision hole", "category": "construction_detail", "source": "photo_2"},
    {"truth_id": "k3", "fact": "faint residue ring at the neck of the bottle", "category": "material_character", "source": "photo_3"},
    {"truth_id": "k4", "fact": "pale gold fluid with no visible sediment", "category": "color", "source": "photo_1"},
    {"truth_id": "k5", "fact": "tactile vertical ribbing around the lower third of the bottle", "category": "texture", "source": "photo_4"},
]

# Default truths used for most structural tests (electronics product).
TRUTHS = ELECTRONICS_TRUTHS

# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------

def _beats(target_length_sec: int = 30) -> list[dict]:
    return [
        {"t_start": 0, "t_end": 3, "line": "Your phone slides off every stand -- not this one."},
        {"t_start": 3, "t_end": 6, "line": "This one grips with a dual radiator port design."},
        {"t_start": 6, "t_end": 9, "line": "It even charges at full speed with any USB-C cable."},
        {"t_start": 9, "t_end": 14, "line": "The matte finish stays clean through daily drops."},
        # CTA Bridge fix: "so" is a recognized bridging connective (see
        # _CTA_BRIDGE_CONNECTIVES) -- this default beat set is meant to read as
        # a clean baseline that passes every check.
        {"t_start": 14, "t_end": target_length_sec, "line": "So tap to get yours."},
    ]


def _variant(variant_id: str, framework: str, hook_type: str, trigger: str, gti=("e0", "e1", "e2"), hook_line=None) -> dict:
    beats = _beats()
    if hook_line is not None:
        beats[0]["line"] = hook_line
    return {
        "variant_id": variant_id,
        "text": "Your phone slides off every stand -- not this one. This one grips with a dual radiator port design. It even charges at full speed with any USB-C cable. The matte finish stays clean through daily drops. So tap to get yours.",
        "framework": framework,
        "hook_type": hook_type,
        "emotional_trigger": trigger,
        "grounding_truth_ids": list(gti),
        "beats": beats,
    }


def _payload(variants: list[dict]) -> str:
    return json.dumps({"script_variants": variants})


FOUR_GOOD_VARIANTS = [
    _variant("v1", "hook_problem_product_cta", "pattern interrupt", "curiosity"),
    _variant("v2", "PAS", "bold claim", "FOMO"),
    _variant("v3", "AIDA", "social proof", "recognition"),
    _variant("v4", "BAB", "how-to", "relief"),
]

_TRUTH_CATEGORIES = {t["truth_id"]: t["category"] for t in TRUTHS}
_TRUTH_FACTS = {t["truth_id"]: t["fact"] for t in TRUTHS}


# ---------------------------------------------------------------------------
# Basic structural / flow tests (product-agnostic)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_four_good_variants_pass_without_reprompt():
    client = FakeOpenAIClient([_payload(FOUR_GOOD_VARIANTS)])

    result = await generate_script_variants("a brief", TRUTHS, client=client)

    assert client.call_count == 1
    assert len(result) == REQUIRED_VARIANT_COUNT
    assert {v["framework"] for v in result} == set(FRAMEWORKS)


@pytest.mark.asyncio
async def test_too_few_grounding_ids_triggers_reprompt_and_fix_is_accepted():
    bad = _variant("v4", "BAB", "how-to", "relief", gti=("e1",))  # only 1, needs 2
    first = _payload(FOUR_GOOD_VARIANTS[:3] + [bad])
    fixed_v4 = _variant("v4", "BAB", "how-to", "relief", gti=("e0", "e1", "e2"))
    retry = _payload(FOUR_GOOD_VARIANTS[:3] + [fixed_v4])
    client = FakeOpenAIClient([first, retry])

    result = await generate_script_variants("a brief", TRUTHS, client=client)

    assert client.call_count == 2
    assert len(result) == REQUIRED_VARIANT_COUNT


@pytest.mark.asyncio
async def test_unknown_truth_id_is_rejected():
    bad = _variant("v4", "BAB", "how-to", "relief", gti=("e1", "e99"))  # e99 doesn't exist
    only_attempt = _payload(FOUR_GOOD_VARIANTS[:3] + [bad])
    client = FakeOpenAIClient([only_attempt, only_attempt])  # retry would repeat the same mistake

    result = await generate_script_variants("a brief", TRUTHS, client=client)

    assert len(result) == 3, "the variant citing a nonexistent truth_id must be dropped"
    assert {v["variant_id"] for v in result} == {"v1", "v2", "v3"}


@pytest.mark.asyncio
async def test_duplicate_frameworks_across_variants_triggers_reprompt():
    dup = [
        _variant("v1", "PAS", "pattern interrupt", "curiosity"),
        _variant("v2", "PAS", "bold claim", "FOMO"),  # duplicate framework with v1
        _variant("v3", "AIDA", "social proof", "recognition"),
        _variant("v4", "BAB", "how-to", "relief"),
    ]
    first = _payload(dup)
    client = FakeOpenAIClient([first, _payload(FOUR_GOOD_VARIANTS)])

    result = await generate_script_variants("a brief", TRUTHS, client=client)

    assert client.call_count == 2, "duplicate framework across variants must trigger a re-prompt"
    assert len(result) == REQUIRED_VARIANT_COUNT
    assert {v["framework"] for v in result} == set(FRAMEWORKS)


@pytest.mark.asyncio
async def test_grounding_in_only_generic_categories_triggers_reprompt():
    # e1 (material) + e4 (color) -- 2 ids, satisfies the count bar, but neither
    # is a "construction_detail"/"texture"/"form_factor" -- must still be rejected.
    generic_only = _variant("v4", "BAB", "how-to", "relief", gti=("e1", "e4"))
    first = _payload(FOUR_GOOD_VARIANTS[:3] + [generic_only])
    fixed = _variant("v4", "BAB", "how-to", "relief", gti=("e0", "e1", "e2"))  # e0=form_factor, e2=construction_detail
    retry = _payload(FOUR_GOOD_VARIANTS[:3] + [fixed])
    client = FakeOpenAIClient([first, retry])

    result = await generate_script_variants("a brief", TRUTHS, client=client)

    assert client.call_count == 2, "grounding in only generic categories must trigger a re-prompt"
    assert len(result) == REQUIRED_VARIANT_COUNT


@pytest.mark.asyncio
async def test_weak_hook_with_no_number_or_contrast_triggers_reprompt():
    weak_hook = _variant(
        "v4", "BAB", "how-to", "relief", hook_line="Look closely at this detail here."
    )
    first = _payload(FOUR_GOOD_VARIANTS[:3] + [weak_hook])
    fixed_hook = _variant(
        "v4", "BAB", "how-to", "relief", hook_line="Scratched in 2 uses. Not this one."
    )
    retry = _payload(FOUR_GOOD_VARIANTS[:3] + [fixed_hook])
    client = FakeOpenAIClient([first, retry])

    result = await generate_script_variants("a brief", TRUTHS, client=client)

    assert client.call_count == 2, "a hook with no number and no contrast marker must trigger a re-prompt"
    assert len(result) == REQUIRED_VARIANT_COUNT


@pytest.mark.asyncio
async def test_degrades_to_best_n_when_retry_still_bad(caplog):
    bad = _variant("v4", "BAB", "how-to", "relief", gti=("e1",))  # invalid: only 1 grounding id
    first = _payload(FOUR_GOOD_VARIANTS[:3] + [bad])
    still_bad = _payload(FOUR_GOOD_VARIANTS[:3] + [bad])  # retry repeats the same mistake
    client = FakeOpenAIClient([first, still_bad])

    with caplog.at_level("INFO"):
        result = await generate_script_variants("a brief", TRUTHS, client=client)

    assert len(result) == 3, "must degrade to the 3 valid variants rather than block the job"
    assert len(result) >= MIN_VARIANTS_AFTER_DEGRADE


@pytest.mark.asyncio
async def test_single_surviving_variant_is_accepted_as_a_degrade_state(caplog):
    only_one_valid = _payload([FOUR_GOOD_VARIANTS[0]])
    client = FakeOpenAIClient([only_one_valid, only_one_valid])

    with caplog.at_level("WARNING"):
        result = await generate_script_variants("a brief", TRUTHS, client=client)

    assert len(result) == 1
    assert any("proceeding degraded" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# VOICE backstop: lift detection + stacked-compound detection (_validate_variant).
# Exercised directly against _validate_variant since the thing under test is
# the exact violation string, not just whether a re-prompt fires.
# ---------------------------------------------------------------------------


def test_lifted_phrase_from_cited_truth_is_flagged():
    # e2's fact is "dual passive radiator ports on each end cap" -- this line
    # quotes "dual passive radiator ports on" (5 consecutive words) verbatim.
    v = _variant("v1", "hook_problem_product_cta", "pattern interrupt", "curiosity", gti=("e0", "e1", "e2"))
    v["beats"][1]["line"] = "This speaker has dual passive radiator ports on each end."

    problems = _validate_variant(v, _TRUTH_CATEGORIES, 15, _TRUTH_FACTS)

    assert any(
        "reuses 4+ consecutive words" in p and "e2" in p for p in problems
    ), f"expected a lift violation naming e2, got: {problems}"


def test_lifted_phrase_from_shoe_truth_is_flagged():
    # s2's fact is "carbon-fibre tension plate embedded mid-stack" -- lifting
    # "carbon-fibre tension plate embedded" (4 words) should fire the check.
    shoe_cats = {t["truth_id"]: t["category"] for t in SHOE_TRUTHS}
    shoe_facts = {t["truth_id"]: t["fact"] for t in SHOE_TRUTHS}
    v = _variant("v1", "hook_problem_product_cta", "pattern interrupt", "curiosity", gti=("s1", "s2"))
    v["grounding_truth_ids"] = ["s1", "s2"]
    v["beats"][1]["line"] = "It has a carbon-fibre tension plate embedded mid-stack for propulsion."

    problems = _validate_variant(v, shoe_cats, 15, shoe_facts)

    assert any(
        "reuses 4+ consecutive words" in p and "s2" in p for p in problems
    ), f"expected a lift violation naming s2 for shoe truths, got: {problems}"


def test_stacked_hyphenated_compounds_before_noun_is_flagged():
    # Tests the principle: two hyphenated compound modifiers stacked before a noun
    # must be rejected regardless of what noun/product is involved.
    v = _variant("v1", "hook_problem_product_cta", "pattern interrupt", "curiosity", gti=("e0", "e1", "e2"))
    v["beats"][1]["line"] = "This matte-finished, anodized-panel speaker only gets better."

    problems = _validate_variant(v, _TRUTH_CATEGORIES, 15, _TRUTH_FACTS)

    assert any(
        "stacks hyphenated compound modifiers" in p for p in problems
    ), f"expected a stacked-compound violation, got: {problems}"


def test_stacked_hyphenated_compounds_flagged_for_skincare_product():
    # Same principle, different product type: skincare.
    skin_cats = {t["truth_id"]: t["category"] for t in SKINCARE_TRUTHS}
    skin_facts = {t["truth_id"]: t["fact"] for t in SKINCARE_TRUTHS}
    v = _variant("v1", "hook_problem_product_cta", "pattern interrupt", "curiosity", gti=("k1", "k2"))
    v["grounding_truth_ids"] = ["k1", "k2"]
    v["beats"][1]["line"] = "This amber-tinted, precision-tipped serum won't drip."

    problems = _validate_variant(v, skin_cats, 15, skin_facts)

    assert any(
        "stacks hyphenated compound modifiers" in p for p in problems
    ), f"expected a stacked-compound violation for skincare product, got: {problems}"


def test_natural_transformed_line_passes_voice_checks_cleanly():
    # Same truths (e1/e2) cited, but transformed into spoken language rather
    # than lifted -- must NOT false-positive on ordinary grounded copy.
    v = _variant("v1", "hook_problem_product_cta", "pattern interrupt", "curiosity", gti=("e0", "e1", "e2"))
    v["beats"][1]["line"] = "That port? Actual rubber seal -- it won't ever leak."

    problems = _validate_variant(v, _TRUTH_CATEGORIES, 30, _TRUTH_FACTS)

    assert problems == [], f"natural transformed line should pass cleanly, got: {problems}"


@pytest.mark.asyncio
async def test_lifted_phrasing_triggers_reprompt_through_full_flow():
    lifted = _variant("v4", "BAB", "how-to", "relief", gti=("e0", "e1", "e2"))
    lifted["beats"][1]["line"] = "It has dual passive radiator ports on each end cap."
    first = _payload(FOUR_GOOD_VARIANTS[:3] + [lifted])
    client = FakeOpenAIClient([first, _payload(FOUR_GOOD_VARIANTS)])

    result = await generate_script_variants("a brief", TRUTHS, client=client)

    assert client.call_count == 2, "a beat line lifting 4+ words from a cited truth must trigger a re-prompt"
    assert len(result) == REQUIRED_VARIANT_COUNT


# ---------------------------------------------------------------------------
# System prompt structural properties (check section headings, not examples)
# ---------------------------------------------------------------------------


def test_voice_prompt_block_present_in_system_prompt():
    prompt = _build_system_prompt(15)

    # Check the structural section marker and the principle, not a product example.
    assert "VOICE -- SPOKEN, NOT CATALOG" in prompt
    assert "Never reuse 4 or more consecutive words" in prompt
    # The system prompt contains an example of the stacked-compound anti-pattern.
    # We verify the PRINCIPLE exists (the words "stacks" or "compound" or
    # "hyphenated compound modifiers") and the CALIBRATION section, not the exact
    # example string (which could change when the prompt is refined).
    assert "hyphenated compound modifiers" in prompt or "stacked" in prompt.lower()


def test_voice_mandate_block_contains_stacked_modifier_rule():
    prompt = _build_system_prompt(15)
    # The VOICE block must contain both the lift rule and the stacked-modifier rule.
    assert "4 or more consecutive words" in prompt
    assert "adjective" in prompt.lower() or "modifier" in prompt.lower()


def test_pronoun_thread_rule_present_in_system_prompt():
    # The reintroduction check is enforced via VO FOCUS MANDATE (not a
    # "PRONOUN THREAD" section); verify the mandate and its third-person ban.
    prompt = _build_system_prompt(15)

    assert "VO FOCUS MANDATE" in prompt
    assert "third-person" in prompt


def test_feature_spread_block_present_in_system_prompt():
    prompt = _build_system_prompt(15)

    assert "FEATURE SPREAD" in prompt
    assert "sell the WHOLE product" in prompt


def test_json_schema_omits_grounding_research_ids_when_no_research_facts():
    """Real bug, confirmed on a live charcoal-briquettes run: the JSON schema
    example previously showed only "grounding_truth_ids", even though the
    prose (_research_facts_block) separately instructs the model to cite
    research facts in "grounding_research_ids" -- with no schema slot ever
    demonstrated for it, the model sometimes crammed research fact ids into
    grounding_truth_ids instead, which then failed validation as an "unknown
    truth_id" (grounding_truth_ids is checked against the photo-truth id
    namespace only). No research facts means nothing to cite, so the field is
    correctly omitted from the example in that case.
    """
    prompt = _build_system_prompt(15, research_facts=None)
    schema = prompt.split("Return ONLY valid JSON")[1]
    assert '"grounding_truth_ids"' in schema
    assert '"grounding_research_ids"' not in schema


def test_json_schema_includes_grounding_research_ids_when_research_facts_present():
    prompt = _build_system_prompt(
        15,
        research_facts=[
            {"fact_id": "r1", "claim": "x", "category": "spec", "source_url": "", "confidence": "medium"},
        ],
    )
    schema = prompt.split("Return ONLY valid JSON")[1]
    assert '"grounding_truth_ids": ["t1", "t3"]' in schema
    # The example must show an r-prefixed id (not a t-prefixed one) so the
    # model has a concrete, correctly-namespaced example to follow.
    assert '"grounding_research_ids": ["r1"]' in schema
    # grounding_research_ids must appear on the same object as
    # grounding_truth_ids, not floating elsewhere in the prompt.
    truth_idx = schema.index('"grounding_truth_ids"')
    research_idx = schema.index('"grounding_research_ids"')
    beats_idx = schema.index('"beats"')
    assert truth_idx < research_idx < beats_idx


# ---------------------------------------------------------------------------
# PRONOUN THREAD backstop (_reintroduction_problems, wired into _validate_variant).
# ---------------------------------------------------------------------------


def test_reintroduction_of_implied_person_is_flagged():
    v = _variant("v1", "hook_problem_product_cta", "pattern interrupt", "curiosity", gti=("e0", "e1", "e2"))
    v["beats"] = [
        {"t_start": 0, "t_end": 3, "line": "She clips it onto her belt loop mid-run."},
        {"t_start": 3, "t_end": 8, "line": "A person checks the charge indicator without breaking stride."},
        {"t_start": 8, "t_end": 15, "line": "Tap to get yours."},
    ]

    problems = _validate_variant(v, _TRUTH_CATEGORIES, 15, _TRUTH_FACTS)

    assert any(
        "reintroduces the story's person generically" in p and "beat 1" in p for p in problems
    ), f"expected a reintroduction violation naming beat 1, got: {problems}"


def test_reintroduction_flagged_for_shoe_product():
    shoe_cats = {t["truth_id"]: t["category"] for t in SHOE_TRUTHS}
    shoe_facts = {t["truth_id"]: t["fact"] for t in SHOE_TRUTHS}
    v = _variant("v1", "hook_problem_product_cta", "pattern interrupt", "curiosity", gti=("s1", "s2"))
    v["grounding_truth_ids"] = ["s1", "s2"]
    v["beats"] = [
        {"t_start": 0, "t_end": 3, "line": "He laces them up before a 5am run."},
        {"t_start": 3, "t_end": 8, "line": "A man hits the carbon plate at mile three and surges."},
        {"t_start": 8, "t_end": 15, "line": "So grab a pair now."},
    ]

    problems = _validate_variant(v, shoe_cats, 15, shoe_facts)

    assert any(
        "reintroduces the story's person generically" in p for p in problems
    ), f"expected a reintroduction violation for shoe product, got: {problems}"


def test_consistent_pronoun_thread_passes_cleanly():
    v = _variant("v1", "hook_problem_product_cta", "pattern interrupt", "curiosity", gti=("e0", "e1", "e2"))
    v["beats"] = [
        {"t_start": 0, "t_end": 3, "line": "She clips it onto her belt loop mid-run."},
        {"t_start": 3, "t_end": 8, "line": "Her thumb finds the volume button without looking."},
        {"t_start": 8, "t_end": 15, "line": "So tap to get yours."},
    ]

    problems = _validate_variant(v, _TRUTH_CATEGORIES, 15, _TRUTH_FACTS)

    assert not any("reintroduces the story's person" in p for p in problems), (
        f"a consistent pronoun thread must not be flagged, got: {problems}"
    )


def test_no_pronoun_at_all_is_never_flagged():
    # A legitimately person-free variant (pure product/material beats) must
    # never trip the reintroduction check.
    v = _variant("v1", "hook_problem_product_cta", "pattern interrupt", "curiosity", gti=("e0", "e1", "e2"))
    v["beats"] = [
        {"t_start": 0, "t_end": 3, "line": "That grille? Actual matte aluminum."},
        {"t_start": 3, "t_end": 8, "line": "A hand tests the grip before it ships."},
        {"t_start": 8, "t_end": 15, "line": "So tap to get yours."},
    ]

    problems = _validate_variant(v, _TRUTH_CATEGORIES, 15, _TRUTH_FACTS)

    assert not any("reintroduces the story's person" in p for p in problems), (
        f"no pronoun ever appeared, so nothing should be flagged, got: {problems}"
    )


# ---------------------------------------------------------------------------
# Anti-rhyme check (_rhyme_problems) -- tested with multiple product contexts.
# ---------------------------------------------------------------------------


def test_rhyme_problems_flags_genuine_rhyme_pair():
    # "gear"/"here" -- CMUdict transcribes them inconsistently (G IH1 R vs.
    # HH IY1 R) even though they're the same NEAR vowel in most dialects;
    # _normalize_rhyme_phones' IH/IY-before-R canonicalization is what makes
    # this genuinely-rhyming pair actually match.
    beats = [
        {"line": "Stop babying your gear."},
        {"line": "It belongs right here."},
    ]

    problems = _rhyme_problems(beats)

    assert any("gear" in p and "here" in p for p in problems), (
        f"expected a flagged rhyme between 'gear' and 'here', got: {problems}"
    )


def test_rhyme_problems_flags_rhyme_in_electronics_context():
    # Verifies the anti-rhyme check is product-agnostic.
    beats = [
        {"line": "Play it loud."},
        {"line": "Any crowd."},
    ]

    problems = _rhyme_problems(beats)

    assert len(problems) > 0, (
        f"expected a rhyme flag for 'loud'/'crowd' in electronics context, got: {problems}"
    )


def test_rhyme_problems_does_not_flag_non_rhyming_pair():
    beats = [
        {"line": "Stop babying your backpack."},
        {"line": "It belongs on the shelf."},
    ]

    assert _rhyme_problems(beats) == []


def test_rhyme_problems_does_not_flag_identical_word_repetition():
    # Repeating the same word is not a rhyme -- it's repetition.
    beats = [
        {"line": "She loves her gear."},
        {"line": "She loves her gear."},
    ]

    assert _rhyme_problems(beats) == []


def test_rhyme_problems_does_not_flag_short_or_stopword_final_words():
    # Both clause-final candidates here are stopwords/too short ("so", "go").
    beats = [
        {"line": "It is a speaker, so."},
        {"line": "Grab yours, go."},
    ]

    assert _rhyme_problems(beats) == []


def test_rhyme_problems_does_not_false_positive_on_natural_lessen_strengthen_pair():
    # The exact false-positive case naive same-last-syllable matching produces --
    # "lessen" and "strengthen" do NOT share a rhyming_part.
    beats = [
        {"line": "Time will lessen the shine."},
        {"line": "It only makes it strengthen."},
    ]

    assert _rhyme_problems(beats) == []


# ---------------------------------------------------------------------------
# Hook-type-gated hook floors (Backstory-First fix).
# ---------------------------------------------------------------------------


def test_human_moment_hook_passes_for_story_curiosity_hook_type():
    # No digit, no contrast marker -- would fail the OLD floor outright.
    # hook_type "curiosity gap" is story-type so judged on _has_human_moment_marker.
    hook_line = "She's out the door before sunrise, bag on one shoulder."
    assert _has_human_moment_marker(hook_line)
    v = _variant(
        "v1", "hook_problem_product_cta", "curiosity gap", "curiosity",
        gti=("e0", "e1", "e2"), hook_line=hook_line,
    )

    problems = _validate_variant(v, _TRUTH_CATEGORIES, 15, _TRUTH_FACTS)

    assert not any("has no number and no contrast marker" in p for p in problems), (
        f"a story/curiosity hook_type must not be judged on the number/contrast floor, got: {problems}"
    )
    assert not any("personal pronoun/second-person address" in p for p in problems), (
        f"this hook line DOES carry a human-moment marker and must pass, got: {problems}"
    )


def test_human_moment_hook_still_fails_number_contrast_floor_for_claim_led_hook_type():
    # Same hook line, but hook_type is claim-led ("bold claim") -- proves the
    # hook_type gating is real.
    hook_line = "She's out the door before sunrise, bag on one shoulder."
    v = _variant(
        "v1", "hook_problem_product_cta", "bold claim", "curiosity",
        gti=("e0", "e1", "e2"), hook_line=hook_line,
    )

    problems = _validate_variant(v, _TRUTH_CATEGORIES, 15, _TRUTH_FACTS)

    assert any("has no number and no contrast marker" in p for p in problems), (
        f"a claim-led hook_type must still enforce the number/contrast floor, got: {problems}"
    )


# ---------------------------------------------------------------------------
# Flaw-led-hook backstop (_flaw_led_hook_problem).
# ---------------------------------------------------------------------------


def test_flaw_led_hook_backstop_rejects_competitor_flaw_comparison():
    # Any product type: a "Other X hide Y, not ours" hook is always rejected.
    problem = _flaw_led_hook_problem(
        "Other speakers hide this scratch, not ours.", cited_truths=[], truth_categories={}
    )

    assert problem is not None
    assert "competitor-flaw comparison" in problem


def test_flaw_led_hook_backstop_rejects_competitor_flaw_for_shoe():
    problem = _flaw_led_hook_problem(
        "Other shoes hide this scuff, not ours.", cited_truths=[], truth_categories={}
    )

    assert problem is not None
    assert "competitor-flaw comparison" in problem


def test_flaw_led_hook_backstop_rejects_via_validate_variant():
    hook_line = "Other speakers hide this scratch, not ours."
    v = _variant(
        "v1", "hook_problem_product_cta", "bold claim", "curiosity",
        gti=("e1", "e3"), hook_line=hook_line,
    )

    problems = _validate_variant(v, _TRUTH_CATEGORIES, 15, _TRUTH_FACTS)

    assert any("competitor-flaw comparison" in p for p in problems), (
        f"expected the flaw-led-hook backstop to fire through _validate_variant, got: {problems}"
    )


def test_flaw_led_hook_backstop_does_not_flag_an_ordinary_claim_hook():
    problem = _flaw_led_hook_problem(
        "Your coffee is cold in 12 minutes. Mine isn't.", cited_truths=[], truth_categories={}
    )

    assert problem is None


def test_flaw_led_hook_product_agnostic_no_false_positive_for_positive_claim():
    # A positive concrete claim about any product type must never be flagged.
    for hook in [
        "This serum absorbs in 8 seconds. Not 30.",
        "Most shoes lose cushion by mile 200. Not these.",
        "Your speaker dies in 6 hours. This one doesn't.",
    ]:
        problem = _flaw_led_hook_problem(hook, cited_truths=[], truth_categories={})
        assert problem is None, f"unexpected flag for hook: {hook!r} -- got: {problem}"


# ---------------------------------------------------------------------------
# FEATURE SPREAD check + single-truth fixation backstop.
# ---------------------------------------------------------------------------


def test_same_category_grounding_ids_are_flagged_for_missing_spread():
    # e2 and e5 are BOTH construction_detail -- spans only ONE category.
    v = _variant("v1", "hook_problem_product_cta", "pattern interrupt", "curiosity", gti=("e2", "e5"))

    problems = _validate_variant(v, _TRUTH_CATEGORIES, 15, _TRUTH_FACTS)

    assert any("all share one category" in p for p in problems), (
        f"expected a category-spread violation, got: {problems}"
    )


def test_same_category_spread_violation_for_shoe_truths():
    # s2 and s5 are BOTH construction_detail in the shoe truth set.
    shoe_cats = {t["truth_id"]: t["category"] for t in SHOE_TRUTHS}
    shoe_facts = {t["truth_id"]: t["fact"] for t in SHOE_TRUTHS}
    v = _variant("v1", "hook_problem_product_cta", "pattern interrupt", "curiosity", gti=("s2", "s5"))
    v["grounding_truth_ids"] = ["s2", "s5"]

    problems = _validate_variant(v, shoe_cats, 15, shoe_facts)

    assert any("all share one category" in p for p in problems), (
        f"expected a category-spread violation for shoe truths, got: {problems}"
    )


def test_cross_category_grounding_ids_pass_the_spread_check():
    # e1 (material) + e2 (construction_detail) = two different categories -> passes.
    v = _variant("v1", "hook_problem_product_cta", "pattern interrupt", "curiosity", gti=("e0", "e1", "e2"))

    problems = _validate_variant(v, _TRUTH_CATEGORIES, 15, _TRUTH_FACTS)

    assert not any("all share one category" in p for p in problems)


def test_cross_category_spread_passes_for_skincare_truths():
    skin_cats = {t["truth_id"]: t["category"] for t in SKINCARE_TRUTHS}
    skin_facts = {t["truth_id"]: t["fact"] for t in SKINCARE_TRUTHS}
    v = _variant("v1", "hook_problem_product_cta", "pattern interrupt", "curiosity", gti=("k1", "k5"))
    v["grounding_truth_ids"] = ["k1", "k5"]  # material + texture

    problems = _validate_variant(v, skin_cats, 15, skin_facts)

    assert not any("all share one category" in p for p in problems), (
        f"material + texture cross-category should pass spread check, got: {problems}"
    )


def test_spread_check_degrades_when_all_available_truths_share_one_category():
    # When every extracted truth is the same category, every variant is forced
    # into that single category -- the check must degrade gracefully.
    single_cat = {"e2": "construction_detail", "e5": "construction_detail"}
    facts = {"e2": _TRUTH_FACTS["e2"], "e5": _TRUTH_FACTS["e5"]}
    v = _variant("v1", "hook_problem_product_cta", "pattern interrupt", "curiosity", gti=("e2", "e5"))

    problems = _validate_variant(v, single_cat, 15, facts)

    assert not any("all share one category" in p for p in problems), (
        f"spread check must degrade gracefully with a single-category truth list, got: {problems}"
    )


@pytest.mark.asyncio
async def test_same_category_spread_violation_triggers_reprompt_through_full_flow():
    fixated = _variant("v4", "BAB", "how-to", "relief", gti=("e2", "e5"))
    first = _payload(FOUR_GOOD_VARIANTS[:3] + [fixated])
    fixed = _variant("v4", "BAB", "how-to", "relief", gti=("e0", "e1", "e2"))
    retry = _payload(FOUR_GOOD_VARIANTS[:3] + [fixed])
    client = FakeOpenAIClient([first, retry])

    result = await generate_script_variants("a brief", TRUTHS, client=client)

    assert client.call_count == 2, "same-category grounding must trigger a re-prompt"
    assert len(result) == REQUIRED_VARIANT_COUNT


# ---------------------------------------------------------------------------
# _single_truth_fixation_problems -- tested with 3 product types.
# ---------------------------------------------------------------------------


def test_single_truth_fixation_is_flagged_on_a_one_detail_script_electronics():
    # An entire script about the radiator port and nothing else.
    beats = [
        {"line": "Look at this radiator port -- a hidden detail."},
        {"line": "Dual passive radiator ports on each end, that's what sets it apart."},
        {"line": "The radiator port keeps the bass clean and tight."},
        {"line": "Buy it today."},
    ]
    truth_facts = {
        "e2": "dual passive radiator ports on each end cap",
        "e1": "matte black anodized aluminum grille",
    }

    problems = _single_truth_fixation_problems(beats, truth_facts)

    assert len(problems) == 1
    assert "revolve around truth e2" in problems[0]


def test_single_truth_fixation_is_flagged_on_a_one_detail_script_skincare():
    # An entire script orbiting only the silicone tip (k2). Words exclusive to
    # k2: "pharmaceutical", "silicone", "precision". The word "dropper" appears
    # in both k1 and k2, so beats deliberately avoid it to ensure only k2 is
    # matched (otherwise len(mentions)==2 and the fixation check correctly stays
    # silent). "bottle" / "amber" / "glass" / "translucent" are k1-exclusive
    # and are also kept out of the beats.
    beats = [
        {"line": "That silicone tip -- it never drips."},
        {"line": "Pharmaceutical precision means one exact dose every time."},
        {"line": "The silicone holds tight, the precision never changes."},
        {"line": "Order yours now."},
    ]
    truth_facts = {
        "k2": "pharmaceutical-grade silicone dropper tip with a precision hole",
        "k1": "translucent amber glass dropper bottle",
    }

    problems = _single_truth_fixation_problems(beats, truth_facts)

    assert len(problems) == 1
    assert "revolve around truth k2" in problems[0]


def test_single_truth_fixation_not_flagged_when_a_second_truth_features():
    # When a second truth appears in the copy the fixation check must not fire.
    beats = [
        {"line": "Look at this radiator -- a hidden detail."},
        {"line": "Dual passive radiator ports on each end, that's what sets it apart."},
        {"line": "And that matte grille? Scratch-resistant aluminum."},
        {"line": "Buy it today."},
    ]
    truth_facts = {
        "e2": "dual passive radiator ports on each end cap",
        "e1": "matte black anodized aluminum grille",
    }

    assert _single_truth_fixation_problems(beats, truth_facts) == []


def test_single_truth_fixation_not_flagged_below_the_beat_threshold():
    # A legitimate hook+payoff pair about one detail (2 beats) must never trip.
    beats = [
        {"line": "Look at this radiator port -- a hidden detail."},
        {"line": "Dual passive radiator ports on each end."},
        {"line": "Grab it now."},
    ]
    truth_facts = {
        "e2": "dual passive radiator ports on each end cap",
        "e1": "matte black anodized aluminum grille",
    }

    assert _single_truth_fixation_problems(beats, truth_facts) == []


def test_single_truth_fixation_skipped_with_fewer_than_two_truths():
    beats = [
        {"line": "Look at this radiator port -- a hidden detail."},
        {"line": "Dual passive radiator ports on each end."},
        {"line": "The radiator port keeps the bass tight."},
    ]
    assert _single_truth_fixation_problems(
        beats, {"e2": "dual passive radiator ports on each end cap"}
    ) == []


def test_single_truth_fixation_not_flagged_for_shoe_with_spread():
    # Shoe script that mentions both midsole and carbon plate -- no fixation.
    beats = [
        {"line": "That midsole foam? It bounces back harder than anything else out there."},
        {"line": "The carbon plate stiffens the push-off at mile three."},
        {"line": "So lace up and go."},
    ]
    truth_facts = {
        "s1": "high-rebound foam midsole with visible bubble cell pattern",
        "s2": "carbon-fibre tension plate embedded mid-stack",
    }

    assert _single_truth_fixation_problems(beats, truth_facts) == []


# ---------------------------------------------------------------------------
# Human-Centric Bias fix: product-conditional HUMAN-CENTRIC BIAS prompt block
# + deterministic person-committed-variant floor.
# ---------------------------------------------------------------------------


def _bag_variant(variant_id: str, framework: str, hook_type: str, trigger: str, human: bool = False) -> dict:
    v = _variant(variant_id, framework, hook_type, trigger, gti=("b0", "b1", "b2"))
    v["grounding_truth_ids"] = ["b0", "b1", "b2"]
    if human:
        v["beats"][1]["line"] = "She grips it tight on her walk to work."
    return v


def _shoe_variant(variant_id: str, framework: str, hook_type: str, trigger: str, human: bool = False) -> dict:
    v = _variant(variant_id, framework, hook_type, trigger, gti=("s0", "s1", "s2"))
    v["grounding_truth_ids"] = ["s0", "s1", "s2"]
    if human:
        v["beats"][1]["line"] = "He lands on that midsole at mile four and doesn't feel a thing."
    return v


def test_human_centric_block_rendered_only_when_product_suits_it():
    # The old HUMAN-CENTRIC BIAS block is replaced by the universal VO FOCUS
    # MANDATE (MIN_HUMAN_PRESENCE_VARIANTS=0). Both prompts are now identical;
    # verify the mandate is present and no stale HUMAN-CENTRIC BIAS text leaks in.
    with_bias = _build_system_prompt(15, human_use_suits=True)
    without_bias = _build_system_prompt(15, human_use_suits=False)

    assert "VO FOCUS MANDATE" in with_bias
    assert "VO FOCUS MANDATE" in without_bias
    assert "HUMAN-CENTRIC BIAS" not in with_bias
    assert "HUMAN-CENTRIC BIAS" not in without_bias


def test_human_presence_count_uses_pronoun_thread():
    humans = [_bag_variant("v1", "PAS", "bold claim", "FOMO", human=True)]
    no_humans = [_bag_variant("v2", "AIDA", "social proof", "recognition")]

    assert _human_presence_count(humans) == 1
    assert _human_presence_count(no_humans) == 0


def test_human_presence_count_works_for_shoe_product():
    humans = [_shoe_variant("v1", "PAS", "bold claim", "FOMO", human=True)]
    no_humans = [_shoe_variant("v2", "AIDA", "social proof", "recognition")]

    assert _human_presence_count(humans) == 1
    assert _human_presence_count(no_humans) == 0


@pytest.mark.asyncio
async def test_human_shortfall_triggers_reprompt_for_human_suited_product():
    # MIN_HUMAN_PRESENCE_VARIANTS=0: human-presence floor is disabled, so 4
    # valid variants with zero person-committed beats must NOT trigger a re-prompt.
    no_human_variants = [
        _bag_variant("v1", "hook_problem_product_cta", "pattern interrupt", "curiosity"),
        _bag_variant("v2", "PAS", "bold claim", "FOMO"),
        _bag_variant("v3", "AIDA", "social proof", "recognition"),
        _bag_variant("v4", "BAB", "how-to", "relief"),
    ]
    client = FakeOpenAIClient([_payload(no_human_variants)])

    result = await generate_script_variants("a brief", BAG_TRUTHS, client=client)

    assert client.call_count == 1, (
        "MIN_HUMAN_PRESENCE_VARIANTS=0 means no human-presence re-prompt fires, "
        "even when the product suits human use"
    )
    assert len(result) == REQUIRED_VARIANT_COUNT


@pytest.mark.asyncio
async def test_human_shortfall_triggers_reprompt_for_shoe_product():
    # MIN_HUMAN_PRESENCE_VARIANTS=0: human-presence floor is disabled for shoe
    # products too -- a single-call pass is correct behavior.
    shoe_truths = SHOE_TRUTHS
    no_human_variants = [
        _shoe_variant("v1", "hook_problem_product_cta", "pattern interrupt", "curiosity"),
        _shoe_variant("v2", "PAS", "bold claim", "FOMO"),
        _shoe_variant("v3", "AIDA", "social proof", "recognition"),
        _shoe_variant("v4", "BAB", "how-to", "relief"),
    ]
    client = FakeOpenAIClient([_payload(no_human_variants)])

    result = await generate_script_variants("a brief", shoe_truths, client=client)

    assert client.call_count == 1, (
        "MIN_HUMAN_PRESENCE_VARIANTS=0 means no human-presence re-prompt fires "
        "even for shoe products with strap/sole affordance facts"
    )
    assert len(result) == REQUIRED_VARIANT_COUNT


@pytest.mark.asyncio
async def test_no_human_reprompt_for_product_without_affordance():
    # Electronics truths (TRUTHS): no strap/handle/scale_cue -> no human-use
    # affordance -> the product-conditional gate must keep single-call behavior.
    client = FakeOpenAIClient([_payload(FOUR_GOOD_VARIANTS)])

    result = await generate_script_variants("a brief", TRUTHS, client=client)

    assert client.call_count == 1
    assert len(result) == REQUIRED_VARIANT_COUNT


@pytest.mark.asyncio
async def test_no_human_reprompt_for_skincare_product():
    # Skincare truths have no ergonomic affordance either -- same single-call
    # behavior as the electronics test above.
    # Build a fake payload with skincare-valid variants.
    skin_good_variants = [
        {
            "variant_id": "v1",
            "text": "One drop. That amber bottle. Yours.",
            "framework": "hook_problem_product_cta",
            "hook_type": "FOMO / urgency",
            "emotional_trigger": "curiosity",
            "grounding_truth_ids": ["k0", "k1", "k2"],
            "beats": [
                {"t_start": 0, "t_end": 3, "line": "Most serums waste half the bottle. Not this one."},
                {"t_start": 3, "t_end": 8, "line": "That silicone tip gives you exactly one measured dose."},
                {"t_start": 8, "t_end": 30, "line": "So tap to get yours."},
            ],
        },
        {
            "variant_id": "v2",
            "text": "Two drops of gold every morning.",
            "framework": "PAS",
            "hook_type": "bold claim",
            "emotional_trigger": "FOMO",
            "grounding_truth_ids": ["k0", "k1", "k2"],  # k2=construction_detail satisfies TIER 3
            "beats": [
                {"t_start": 0, "t_end": 3, "line": "Your serum costs 3x this and drips everywhere."},
                {"t_start": 3, "t_end": 8, "line": "That precision tip gives you exactly one dose."},
                {"t_start": 8, "t_end": 30, "line": "So tap to get yours."},
            ],
        },
        {
            "variant_id": "v3",
            "text": "The gold fluid is real.",
            "framework": "AIDA",
            "hook_type": "social proof",
            "emotional_trigger": "recognition",
            "grounding_truth_ids": ["k0", "k4", "k2"],
            "beats": [
                {"t_start": 0, "t_end": 3, "line": "That gold color? Not dye."},
                {"t_start": 3, "t_end": 8, "line": "The silicone tip keeps it pure from bottle to skin."},
                {"t_start": 8, "t_end": 30, "line": "So tap to get yours."},
            ],
        },
        {
            "variant_id": "v4",
            "text": "Amber glass, silicone tip -- yours.",
            "framework": "BAB",
            "hook_type": "how-to",
            "emotional_trigger": "relief",
            "grounding_truth_ids": ["k0", "k2", "k1"],  # k2=construction_detail satisfies TIER 3
            "beats": [
                {"t_start": 0, "t_end": 3, "line": "Two drops every morning -- not 10, not 20."},
                {"t_start": 3, "t_end": 8, "line": "That silicone tip means you never fumble it half-asleep."},
                {"t_start": 8, "t_end": 30, "line": "So tap to get yours."},
            ],
        },
    ]
    client = FakeOpenAIClient([_payload(skin_good_variants)])

    result = await generate_script_variants("a face serum", SKINCARE_TRUTHS, client=client)

    assert client.call_count == 1, (
        "skincare product with no human-use affordance must not trigger a human-presence re-prompt"
    )
    assert len(result) == REQUIRED_VARIANT_COUNT


@pytest.mark.asyncio
async def test_human_shortfall_degrades_when_retry_still_has_no_people(caplog):
    # MIN_HUMAN_PRESENCE_VARIANTS=0: the human-presence degrade path no longer
    # fires. 4 valid bag variants pass in a single call with no warning logged.
    no_human_variants = [
        _bag_variant("v1", "hook_problem_product_cta", "pattern interrupt", "curiosity"),
        _bag_variant("v2", "PAS", "bold claim", "FOMO"),
        _bag_variant("v3", "AIDA", "social proof", "recognition"),
        _bag_variant("v4", "BAB", "how-to", "relief"),
    ]
    client = FakeOpenAIClient([_payload(no_human_variants)])

    with caplog.at_level("WARNING"):
        result = await generate_script_variants("a brief", BAG_TRUTHS, client=client)

    assert client.call_count == 1, "no human-presence reprompt with MIN_HUMAN_PRESENCE_VARIANTS=0"
    assert len(result) == REQUIRED_VARIANT_COUNT
    assert not any("person-committed" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Positive-Only Truths fix: imperfection-category citation ban.
# ---------------------------------------------------------------------------


def test_imperfection_citation_banned_by_default():
    problem = _imperfection_citation_problem(["e1", "e3"], _TRUTH_CATEGORIES, wants_imperfection=False)
    assert problem is not None
    assert "e3" in problem
    assert "off by default" in problem


def test_imperfection_citation_allowed_when_seller_wants_it():
    problem = _imperfection_citation_problem(["e1", "e3"], _TRUTH_CATEGORIES, wants_imperfection=True)
    assert problem is None


def test_imperfection_citation_check_ignores_non_imperfection_ids():
    problem = _imperfection_citation_problem(["e1", "e2"], _TRUTH_CATEGORIES, wants_imperfection=False)
    assert problem is None


def test_imperfection_citation_banned_through_validate_variant():
    v = _variant("v1", "hook_problem_product_cta", "pattern interrupt", "curiosity", gti=("e1", "e3"))

    problems = _validate_variant(v, _TRUTH_CATEGORIES, 15, _TRUTH_FACTS, wants_imperfection=False)

    assert any("off by default" in p for p in problems), (
        f"expected the imperfection-citation ban to fire, got: {problems}"
    )


def test_imperfection_citation_allowed_through_validate_variant_when_requested():
    v = _variant("v1", "hook_problem_product_cta", "pattern interrupt", "curiosity", gti=("e1", "e3"))

    problems = _validate_variant(v, _TRUTH_CATEGORIES, 15, _TRUTH_FACTS, wants_imperfection=True)

    assert not any("off by default" in p for p in problems)


def test_imperfection_citation_banned_for_shoe_product():
    shoe_cats = {t["truth_id"]: t["category"] for t in SHOE_TRUTHS}
    problem = _imperfection_citation_problem(["s1", "s3"], shoe_cats, wants_imperfection=False)
    assert problem is not None, "imperfection ban must fire for shoe truths too"
    assert "s3" in problem


def test_imperfection_citation_banned_for_skincare_product():
    skin_cats = {t["truth_id"]: t["category"] for t in SKINCARE_TRUTHS}
    problem = _imperfection_citation_problem(["k1", "k3"], skin_cats, wants_imperfection=False)
    assert problem is not None, "imperfection ban must fire for skincare truths too"
    assert "k3" in problem


@pytest.mark.asyncio
async def test_generate_script_variants_bans_imperfection_by_default_and_recovers():
    imperfection_variant = _variant(
        "v4", "BAB", "how-to", "relief", gti=("e1", "e3")
    )  # e3 is material_character (imperfection) -- banned by default
    first = _payload(FOUR_GOOD_VARIANTS[:3] + [imperfection_variant])
    fixed_v4 = _variant("v4", "BAB", "how-to", "relief", gti=("e0", "e1", "e2"))
    retry = _payload(FOUR_GOOD_VARIANTS[:3] + [fixed_v4])
    client = FakeOpenAIClient([first, retry])

    result = await generate_script_variants("a durable portable speaker", TRUTHS, client=client)

    assert client.call_count == 2, "citing an imperfection by default must trigger a re-prompt"
    assert len(result) == REQUIRED_VARIANT_COUNT
    assert all(
        not any(_TRUTH_CATEGORIES.get(t) == "imperfection" for t in v["grounding_truth_ids"])
        for v in result
    )


@pytest.mark.asyncio
async def test_generate_script_variants_allows_imperfection_when_brief_asks_for_authentic_angle():
    # e0=form_factor (TIER 1), e1=material (TIER 2), e3=material_character (TIER 3 imperfection)
    imperfection_variants = [
        _variant("v1", "hook_problem_product_cta", "pattern interrupt", "curiosity", gti=("e0", "e1", "e3")),
        _variant("v2", "PAS", "bold claim", "FOMO", gti=("e0", "e1", "e3")),
        _variant("v3", "AIDA", "social proof", "recognition", gti=("e0", "e1", "e3")),
        _variant("v4", "BAB", "how-to", "relief", gti=("e0", "e1", "e3")),
    ]
    client = FakeOpenAIClient([_payload(imperfection_variants)])

    result = await generate_script_variants(
        "an authentic, well-used speaker with character", TRUTHS, client=client
    )

    assert client.call_count == 1, "an explicit authentic-angle ask must not trigger a re-prompt"
    assert len(result) == REQUIRED_VARIANT_COUNT


# ---------------------------------------------------------------------------
# CTA Bridge fix: abrupt-CTA backstop (_abrupt_cta_problem).
# ---------------------------------------------------------------------------


def test_abrupt_cta_backstop_fires_on_the_documented_failure_shape():
    # The exact real winning-script failure mode that triggered this fix.
    beats = [
        {"t_start": 0, "t_end": 3, "line": "She's out the door, bag already on one shoulder."},
        {"t_start": 3, "t_end": 10, "line": "And that leather grain? It's already getting darker right where her hands grab it."},
        {"t_start": 10, "t_end": 15, "line": "Grab yours before the next batch sells out."},
    ]

    problem = _abrupt_cta_problem(beats)

    assert problem is not None
    assert "disconnected command" in problem


def test_abrupt_cta_fires_for_electronics_context():
    beats = [
        {"t_start": 0, "t_end": 3, "line": "She clips it to her pack before the trailhead."},
        {"t_start": 3, "t_end": 10, "line": "The bass hits clean at full volume even in the wind."},
        {"t_start": 10, "t_end": 15, "line": "Order yours now."},
    ]

    problem = _abrupt_cta_problem(beats)

    assert problem is not None, "abrupt CTA must be caught regardless of product type"
    assert "disconnected command" in problem


def test_abrupt_cta_backstop_does_not_fire_with_bridging_connective():
    beats = [
        {"t_start": 0, "t_end": 3, "line": "She's out the door, bag already on one shoulder."},
        {"t_start": 3, "t_end": 10, "line": "That grain gets darker right where her hands grab it."},
        {"t_start": 10, "t_end": 15, "line": "So grab yours before the next batch sells out."},
    ]

    assert _abrupt_cta_problem(beats) is None


def test_abrupt_cta_backstop_does_not_fire_with_back_reference():
    beats = [
        {"t_start": 0, "t_end": 3, "line": "She's out the door, bag already on one shoulder."},
        {"t_start": 3, "t_end": 10, "line": "That grain gets darker right where her hands grab it."},
        {"t_start": 10, "t_end": 15, "line": "That's the mark of a bag that's actually yours -- go get it."},
    ]

    assert _abrupt_cta_problem(beats) is None


def test_abrupt_cta_backstop_skips_longer_cta_lines():
    # Long CTAs are left to the CTA-Checker's own scoring rubric.
    beats = [
        {"t_start": 0, "t_end": 3, "line": "She's out the door, bag already on one shoulder."},
        {"t_start": 3, "t_end": 10, "line": "That grain gets darker right where her hands grab it."},
        {"t_start": 10, "t_end": 15, "line": "Grab yours now before the very next production batch quietly sells out for good."},
    ]

    assert _abrupt_cta_problem(beats) is None


def test_abrupt_cta_backstop_fires_through_validate_variant():
    v = _variant(
        "v1", "hook_problem_product_cta", "pattern interrupt", "curiosity",
        gti=("e0", "e1", "e2"),
    )
    v["beats"][-1]["line"] = "Grab yours before the next batch sells out."

    problems = _validate_variant(v, _TRUTH_CATEGORIES, 15, _TRUTH_FACTS)

    assert any("disconnected command" in p for p in problems), (
        f"expected the abrupt-CTA backstop to fire, got: {problems}"
    )


# ---------------------------------------------------------------------------
# _missing_required_tiers: conditional tier validation — added 2026-07-15.
# ---------------------------------------------------------------------------

def test_missing_required_tiers_does_not_report_tier1_when_no_tier1_exists():
    """Tier 1 (form_factor) must not fire when the truth table has no form_factor truths."""
    from agents.concept_agent import _missing_required_tiers
    truths_by_id = {
        "t1": {"category": "color", "fact": "deep matte navy"},
        "t2": {"category": "construction_detail", "fact": "double-stitched reinforced seams at stress points"},
    }
    variant = {"grounding_truth_ids": ["t1"]}  # cites color only
    problems = _missing_required_tiers(variant, truths_by_id)
    assert not any("TIER 1" in p for p in problems), (
        "Tier 1 should not be required when no form_factor truth exists in the table"
    )


def test_missing_required_tiers_reports_all_three_when_all_tiers_present_but_none_cited():
    """All tiers present in table but variant cites none → all 3 fire."""
    from agents.concept_agent import _missing_required_tiers
    truths_by_id = {
        "t1": {"category": "form_factor", "fact": "compact 1-litre flask with matte finish and screw cap"},
        "t2": {"category": "color", "fact": "midnight-blue powder coat"},
        "t3": {"category": "construction_detail", "fact": "double-wall vacuum insulation with laser-welded seam"},
    }
    variant = {"grounding_truth_ids": []}  # cites nothing
    problems = _missing_required_tiers(variant, truths_by_id)
    assert any("TIER 1" in p for p in problems)
    assert any("TIER 2" in p for p in problems)
    assert any("TIER 3" in p for p in problems)


def test_missing_required_tiers_reports_nothing_when_only_brief_or_intake_fact():
    """Truth table with only brief_or_intake_fact: no tier categories exist, so no tier fires."""
    from agents.concept_agent import _missing_required_tiers
    truths_by_id = {
        "t1": {"category": "brief_or_intake_fact", "fact": "minimal packaging, ships plastic-free in recycled cardboard"},
        "t2": {"category": "brief_or_intake_fact", "fact": "certified B-Corp, 1% donated to reforestation per sale"},
    }
    variant = {"grounding_truth_ids": ["t1", "t2"]}
    problems = _missing_required_tiers(variant, truths_by_id)
    assert problems == [], f"No tier should fire when no tier categories exist: {problems}"


def test_specific_categories_includes_brief_or_intake_fact():
    """brief_or_intake_fact must be in SPECIFIC_CATEGORIES as a valid fallback."""
    from agents.concept_agent import SPECIFIC_CATEGORIES
    assert "brief_or_intake_fact" in SPECIFIC_CATEGORIES
