"""
Unit tests for the Hook-Checker's validation/re-prompt/degrade logic --
same rationale as the Concept Agent tests: these code paths (missing score,
out-of-range score, unknown variant_id, fallback-after-retry) are unlikely
to fire reliably against the real model on demand, so they're covered with
a fake client instead.
"""
from __future__ import annotations

import json

import pytest

from agents.hook_checker import MAX_SCORE, MIN_SCORE, _build_system_prompt, score_hooks
from tests._fakes import FakeOpenAIClient

VARIANTS = [
    {"variant_id": "v1", "hook_type": "pattern interrupt", "text": "Scratched already? Not this one."},
    {"variant_id": "v2", "hook_type": "bold claim", "text": "2 metal plates outlast plastic."},
    {"variant_id": "v3", "hook_type": "curiosity gap", "text": "Machining lines prove it's real."},
]


def _payload(scores: list[dict]) -> str:
    return json.dumps({"hook_scores": scores})


GOOD_SCORES = [
    {"variant_id": "v1", "hook_score": 4, "justification": "Specific detail with contrast."},
    {"variant_id": "v2", "hook_score": 5, "justification": "Number plus a clear contrarian claim."},
    {"variant_id": "v3", "hook_score": 3, "justification": "Specific but no real tension."},
]


@pytest.mark.asyncio
async def test_all_variants_scored_without_reprompt():
    client = FakeOpenAIClient([_payload(GOOD_SCORES)])

    result = await score_hooks(VARIANTS, client=client)

    assert client.call_count == 1
    assert set(result.keys()) == {"v1", "v2", "v3"}
    assert result["v2"]["hook_score"] == 5


@pytest.mark.asyncio
async def test_missing_variant_triggers_reprompt_and_fix_is_accepted():
    incomplete = [GOOD_SCORES[0], GOOD_SCORES[1]]  # v3 missing
    retry = GOOD_SCORES  # all 3 present this time
    client = FakeOpenAIClient([_payload(incomplete), _payload(retry)])

    result = await score_hooks(VARIANTS, client=client)

    assert client.call_count == 2
    assert set(result.keys()) == {"v1", "v2", "v3"}


@pytest.mark.asyncio
async def test_out_of_range_score_is_rejected():
    bad = [
        {"variant_id": "v1", "hook_score": 7, "justification": "way out of range"},  # > MAX_SCORE
        GOOD_SCORES[1],
        GOOD_SCORES[2],
    ]
    client = FakeOpenAIClient([_payload(bad), _payload(GOOD_SCORES)])

    result = await score_hooks(VARIANTS, client=client)

    assert client.call_count == 2
    assert result["v1"]["hook_score"] == 4  # fixed value from the retry


@pytest.mark.asyncio
async def test_unknown_variant_id_is_dropped_not_crashed_on():
    with_phantom = GOOD_SCORES + [{"variant_id": "v99", "hook_score": 3, "justification": "doesn't exist"}]
    client = FakeOpenAIClient([_payload(with_phantom)])

    result = await score_hooks(VARIANTS, client=client)

    assert "v99" not in result
    assert set(result.keys()) == {"v1", "v2", "v3"}


@pytest.mark.asyncio
async def test_degrades_to_neutral_fallback_when_still_missing_after_retry(caplog):
    incomplete = [GOOD_SCORES[0], GOOD_SCORES[1]]  # v3 missing both times
    client = FakeOpenAIClient([_payload(incomplete), _payload(incomplete)])

    with caplog.at_level("WARNING"):
        result = await score_hooks(VARIANTS, client=client)

    assert set(result.keys()) == {"v1", "v2", "v3"}
    assert result["v3"]["hook_score"] == (MIN_SCORE + MAX_SCORE) / 2
    assert any("neutral fallback" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_empty_variant_list_returns_empty_without_calling_the_model():
    client = FakeOpenAIClient([])

    result = await score_hooks([], client=client)

    assert result == {}
    assert client.call_count == 0


# ---------------------------------------------------------------------------
# Human-Centric Bias fix (video-gen-fidelity, 2026-07-11): product-conditional
# human-moment scoring tiebreak. LLM scoring behavior can't be asserted
# deterministically here, so these test the rendered rubric text (same posture
# as the shot-list suite's system-prompt-content tests) and that the flag
# plumbs through score_hooks.
# ---------------------------------------------------------------------------


def test_human_bias_tiebreak_rendered_only_when_flag_set():
    with_bias = _build_system_prompt(human_use_bias=True)
    without_bias = _build_system_prompt(human_use_bias=False)

    assert "PRODUCT-SUITABILITY TIEBREAK" in with_bias
    assert "This tiebreak NEVER" in with_bias
    assert "PRODUCT-SUITABILITY TIEBREAK" not in without_bias


def test_human_bias_block_keeps_score_down_rules_intact():
    # The tiebreak must be additive: the flaw-led/sing-song SCORE DOWN
    # calibration stays present in BOTH prompt variants.
    for flag in (True, False):
        prompt = _build_system_prompt(human_use_bias=flag)
        assert "SCORE DOWN (2 max)" in prompt
        assert "HUMAN-MOMENT PATH" in prompt


@pytest.mark.asyncio
async def test_score_hooks_accepts_human_bias_flag():
    client = FakeOpenAIClient([_payload(GOOD_SCORES)])

    result = await score_hooks(VARIANTS, client=client, human_use_bias=True)

    assert set(result.keys()) == {"v1", "v2", "v3"}
