"""
Unit tests for the two code paths that real de-risk runs never actually
exercised: the bounded re-prompt loop, and the MAX_FACTS truncation. Also
covers the same_product mismatch handling (missing-field detection + the
deterministic photo_1-only backstop), since those are cheap to regression-test
even though they were already validated manually against the real model.

Uses a fake OpenAI-shaped client (extract_product_truths already accepts an
injected `client`) so these are fast, deterministic, and don't depend on
DashScope being reachable or the model's mood that day.
"""
from __future__ import annotations

import json

import pytest

from agents.product_truth_extractor import MAX_FACTS, extract_product_truths
from tests._fakes import FakeOpenAIClient


def _truth(truth_id: str, fact: str, category: str = "material", source: str = "photo_1") -> dict:
    return {"truth_id": truth_id, "fact": fact, "category": category, "source": source}


def _payload(truths: list[dict], same_product: bool = True, mismatch_reason: str = "") -> str:
    return json.dumps(
        {"same_product": same_product, "mismatch_reason": mismatch_reason, "product_truths": truths}
    )


GOOD_FACT_1 = "a hairline scratch runs diagonally across the lower left corner of the lid"
GOOD_FACT_2 = "the base plate has two asymmetric ventilation slots near the rear edge"
GOOD_FACT_3 = "a faint discoloration ring marks where a sticker was once removed"
GOOD_FACT_4 = "the power button has a slightly recessed matte texture unlike the glossy housing"
GOOD_FACT_5 = "the charging port surround shows minor oxidation on the metal contacts"


@pytest.mark.asyncio
async def test_reprompt_fires_and_takes_the_better_retry():
    first = _payload([_truth("t1", GOOD_FACT_1), _truth("t2", "nice product")])  # 1 valid, 1 rejected
    retry = _payload(
        [_truth(f"t{i}", fact) for i, fact in enumerate(
            [GOOD_FACT_1, GOOD_FACT_2, GOOD_FACT_3, GOOD_FACT_4, GOOD_FACT_5], start=1
        )]
    )
    client = FakeOpenAIClient([first, retry])

    result = await extract_product_truths(["http://example.com/a.jpg"], client=client)

    assert client.call_count == 2, "re-prompt should fire when valid facts < 4 and something was rejected"
    assert len(result) == 5


@pytest.mark.asyncio
async def test_reprompt_never_regresses_below_the_first_attempt():
    first = _payload(
        [_truth("t1", GOOD_FACT_1), _truth("t2", GOOD_FACT_2), _truth("t3", "too short"), _truth("t4", "bad")]
    )  # 2 valid, 2 rejected -> still < 4, reprompt fires
    worse_retry = _payload([_truth("t1", GOOD_FACT_1)])  # only 1 valid -- worse than first attempt
    client = FakeOpenAIClient([first, worse_retry])

    result = await extract_product_truths(["http://example.com/a.jpg"], client=client)

    assert client.call_count == 2
    assert len(result) == 2, "must keep the original 2 facts, not regress to the retry's 1"


@pytest.mark.asyncio
async def test_no_reprompt_when_enough_valid_facts_survive():
    enough = _payload(
        [_truth(f"t{i}", fact) for i, fact in enumerate([GOOD_FACT_1, GOOD_FACT_2, GOOD_FACT_3, GOOD_FACT_4], start=1)]
    )
    client = FakeOpenAIClient([enough])

    result = await extract_product_truths(["http://example.com/a.jpg"], client=client)

    assert client.call_count == 1, "4 valid facts meets the skip-reprompt bar -- must not re-prompt"
    assert len(result) == 4


@pytest.mark.asyncio
async def test_max_facts_truncation():
    facts = [GOOD_FACT_1, GOOD_FACT_2, GOOD_FACT_3, GOOD_FACT_4, GOOD_FACT_5] * 3  # 15, all distinct enough
    truths = [_truth(f"t{i}", f"{fact} (variant {i})") for i, fact in enumerate(facts, start=1)]
    client = FakeOpenAIClient([_payload(truths)])

    result = await extract_product_truths(["http://example.com/a.jpg"], client=client)

    assert len(result) == MAX_FACTS
    assert [t["truth_id"] for t in result] == [f"t{i}" for i in range(1, MAX_FACTS + 1)]


@pytest.mark.asyncio
async def test_missing_same_product_field_logs_compliance_warning(caplog):
    payload = json.dumps({"product_truths": [_truth("t1", GOOD_FACT_1), _truth("t2", GOOD_FACT_2)]})
    client = FakeOpenAIClient([payload])

    with caplog.at_level("WARNING"):
        await extract_product_truths(["http://example.com/a.jpg"], client=client)

    assert any("omitted the required" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_same_product_false_applies_deterministic_photo1_backstop(caplog):
    truths = [
        _truth("t1", GOOD_FACT_1, source="photo_1"),
        _truth("t2", GOOD_FACT_2, source="photo_2"),  # model didn't self-restrict -- code must catch this
        _truth("t3", GOOD_FACT_3, source="photo_1"),
    ]
    payload = _payload(truths, same_product=False, mismatch_reason="photo_2 shows an unrelated item")
    client = FakeOpenAIClient([payload])

    with caplog.at_level("WARNING"):
        result = await extract_product_truths(["http://example.com/a.jpg"], client=client)

    assert {t["source"] for t in result} == {"photo_1"}
    assert len(result) == 2
    assert any("flagged a product mismatch" in r.message for r in caplog.records)
