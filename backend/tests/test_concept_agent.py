"""
Unit tests for the Concept Agent's validation/re-prompt/degrade logic --
same rationale as test_product_truth_extractor.py: these code paths (malformed
variant, duplicate framework/hook, degrade to best-N, single-survivor
short-circuit) are unlikely to fire reliably against the real model on demand,
so they're covered with a fake client instead.
"""
from __future__ import annotations

import json

import pytest

from agents.concept_agent import (
    FRAMEWORKS,
    MIN_VARIANTS_AFTER_DEGRADE,
    REQUIRED_VARIANT_COUNT,
    generate_script_variants,
)
from tests._fakes import FakeOpenAIClient


TRUTHS = [
    {"truth_id": "t1", "fact": "matte black anodized aluminum finish", "category": "material", "source": "photo_1"},
    {"truth_id": "t2", "fact": "dual cylindrical hinge with knurled end caps", "category": "construction_detail", "source": "photo_1"},
    {"truth_id": "t3", "fact": "faint scuff on the base plate's right cutout", "category": "imperfection", "source": "photo_1"},
    {"truth_id": "t4", "fact": "deep navy blue colorway", "category": "color", "source": "photo_1"},
]


def _beats(target_length_sec: int = 15) -> list[dict]:
    return [
        {"t_start": 0, "t_end": 3, "line": "Your phone slides off every stand -- not this one."},
        {"t_start": 3, "t_end": 8, "line": "This one grips with a dual knurled hinge."},
        {"t_start": 8, "t_end": target_length_sec, "line": "Tap to get yours."},
    ]


def _variant(variant_id: str, framework: str, hook_type: str, trigger: str, gti=("t1", "t2"), hook_line=None) -> dict:
    beats = _beats()
    if hook_line is not None:
        beats[0]["line"] = hook_line
    return {
        "variant_id": variant_id,
        "text": "Your phone keeps sliding off every stand you own. This one grips with a dual knurled hinge. Tap to get yours.",
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


@pytest.mark.asyncio
async def test_four_good_variants_pass_without_reprompt():
    client = FakeOpenAIClient([_payload(FOUR_GOOD_VARIANTS)])

    result = await generate_script_variants("a brief", TRUTHS, client=client)

    assert client.call_count == 1
    assert len(result) == REQUIRED_VARIANT_COUNT
    assert {v["framework"] for v in result} == set(FRAMEWORKS)


@pytest.mark.asyncio
async def test_too_few_grounding_ids_triggers_reprompt_and_fix_is_accepted():
    bad = _variant("v4", "BAB", "how-to", "relief", gti=("t1",))  # only 1, needs 2
    first = _payload(FOUR_GOOD_VARIANTS[:3] + [bad])
    fixed_v4 = _variant("v4", "BAB", "how-to", "relief", gti=("t1", "t3"))
    retry = _payload(FOUR_GOOD_VARIANTS[:3] + [fixed_v4])
    client = FakeOpenAIClient([first, retry])

    result = await generate_script_variants("a brief", TRUTHS, client=client)

    assert client.call_count == 2
    assert len(result) == REQUIRED_VARIANT_COUNT


@pytest.mark.asyncio
async def test_unknown_truth_id_is_rejected():
    bad = _variant("v4", "BAB", "how-to", "relief", gti=("t1", "t99"))  # t99 doesn't exist
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
    # t1 (material) + t4 (color) -- 2 ids, satisfies the count bar, but neither
    # is an "imperfection"/"construction_detail" -- must still be rejected.
    generic_only = _variant("v4", "BAB", "how-to", "relief", gti=("t1", "t4"))
    first = _payload(FOUR_GOOD_VARIANTS[:3] + [generic_only])
    fixed = _variant("v4", "BAB", "how-to", "relief", gti=("t1", "t2"))  # t2 is construction_detail
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
    bad = _variant("v4", "BAB", "how-to", "relief", gti=("t1",))  # invalid: only 1 grounding id
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
