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
GOOD_FORM_FACTOR_FACT = (
    "a deep, rounded block, wider than it is tall, with a smoothly curved front "
    "face, spanning about two hand-widths, in matte charcoal with a soft-touch "
    "finish, with a fabric strap stitched into slots on either side"
)


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
    # Includes a valid form_factor fact -- otherwise the v8 missing-form_factor
    # trigger would fire a re-prompt regardless of count (see the dedicated
    # missing-form_factor tests below).
    enough = _payload(
        [_truth("t0", GOOD_FORM_FACTOR_FACT, category="form_factor")]
        + [_truth(f"t{i}", fact) for i, fact in enumerate([GOOD_FACT_1, GOOD_FACT_2, GOOD_FACT_3, GOOD_FACT_4], start=1)]
    )
    client = FakeOpenAIClient([enough])

    result = await extract_product_truths(["http://example.com/a.jpg"], client=client)

    assert client.call_count == 1, "4 valid facts + a form_factor anchor meets the skip-reprompt bar"
    assert len(result) == 5


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



# ---------------------------------------------------------------------------
# v8: form_factor anchor fact (Meta Quest -> "phone on a stand" bug fix).
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_form_factor_fact_present_and_validated():
    payload = _payload(
        [_truth("t0", GOOD_FORM_FACTOR_FACT, category="form_factor")]
        + [_truth(f"t{i}", fact) for i, fact in enumerate([GOOD_FACT_1, GOOD_FACT_2, GOOD_FACT_3, GOOD_FACT_4], start=1)]
    )
    client = FakeOpenAIClient([payload])

    result = await extract_product_truths(["http://example.com/a.jpg"], client=client)

    assert client.call_count == 1, "a valid form_factor fact present should not trigger a re-prompt"
    form_factor_facts = [t for t in result if t["category"] == "form_factor"]
    assert len(form_factor_facts) == 1
    assert form_factor_facts[0]["fact"] == GOOD_FORM_FACTOR_FACT


@pytest.mark.asyncio
async def test_missing_form_factor_fires_targeted_reprompt_and_recovers():
    # First attempt: 4 valid facts, no form_factor at all.
    first = _payload(
        [_truth(f"t{i}", fact) for i, fact in enumerate([GOOD_FACT_1, GOOD_FACT_2, GOOD_FACT_3, GOOD_FACT_4], start=1)]
    )
    # Retry: same 4 facts, now WITH a form_factor fact added.
    retry = _payload(
        [_truth("t0", GOOD_FORM_FACTOR_FACT, category="form_factor")]
        + [_truth(f"t{i}", fact) for i, fact in enumerate([GOOD_FACT_1, GOOD_FACT_2, GOOD_FACT_3, GOOD_FACT_4], start=1)]
    )
    client = FakeOpenAIClient([first, retry])

    result = await extract_product_truths(["http://example.com/a.jpg"], client=client)

    assert client.call_count == 2, "a missing form_factor fact must trigger the bounded re-prompt"
    form_factor_facts = [t for t in result if t["category"] == "form_factor"]
    assert len(form_factor_facts) == 1, "the retry's gained form_factor fact must be adopted"
    assert len(result) == 5


@pytest.mark.asyncio
async def test_missing_form_factor_reprompt_message_is_targeted():
    first = _payload([_truth("t1", GOOD_FACT_1)])  # no form_factor, and few facts too

    class _CapturingClient(FakeOpenAIClient):
        def __init__(self, responses):
            super().__init__(responses)
            self.seen_messages: list[list[dict]] = []

        async def create(self, model, messages, stream=False, **kwargs):
            self.seen_messages.append(messages)
            return await super().create(model, messages, stream=stream, **kwargs)

    retry = _payload(
        [_truth("t0", GOOD_FORM_FACTOR_FACT, category="form_factor"), _truth("t1", GOOD_FACT_1)]
    )
    client = _CapturingClient([first, retry])

    await extract_product_truths(["http://example.com/a.jpg"], client=client)

    assert client.call_count == 2
    reprompt_user_msg = client.seen_messages[1][-1]["content"]
    assert "form_factor" in reprompt_user_msg
    assert "did NOT include" in reprompt_user_msg


@pytest.mark.asyncio
async def test_max_facts_truncation_never_drops_form_factor_even_when_listed_last():
    # 10 non-form-factor facts (== MAX_FACTS) listed FIRST, form_factor fact LAST.
    filler = [
        _truth(f"t{i}", f"{[GOOD_FACT_1, GOOD_FACT_2, GOOD_FACT_3, GOOD_FACT_4, GOOD_FACT_5][i % 5]} (variant {i})")
        for i in range(1, MAX_FACTS + 1)
    ]
    truths = filler + [_truth("t_ff", GOOD_FORM_FACTOR_FACT, category="form_factor")]
    client = FakeOpenAIClient([_payload(truths)])

    result = await extract_product_truths(["http://example.com/a.jpg"], client=client)

    assert len(result) == MAX_FACTS
    form_factor_facts = [t for t in result if t["category"] == "form_factor"]
    assert len(form_factor_facts) == 1, "the form_factor fact must survive truncation even when listed last"
    assert form_factor_facts[0]["truth_id"] == "t_ff"


def test_generic_heuristic_does_not_reject_a_compliant_form_factor_sentence():
    from agents.product_truth_extractor import _is_generic

    assert len(GOOD_FORM_FACTOR_FACT.split()) >= 30  # sanity: genuinely in the 30-60 word range
    assert _is_generic(GOOD_FORM_FACTOR_FACT) is False


# ---------------------------------------------------------------------------
# Positive-Only Truths fix (docs/BUILD_TASKS.md "Script Quality (CTA Bridge) +
# Positive-Only Truths + Video-Gen Fidelity Fix" workstream, Problem 1):
# imperfection-category facts are dropped by default, kept only when the
# seller's brief/freeform notes explicitly ask for an authentic/imperfection
# angle.
# ---------------------------------------------------------------------------
GOOD_IMPERFECTION_FACT = "a faint hairline crease runs diagonally across the lower right corner of the flap"


@pytest.mark.asyncio
async def test_imperfection_fact_dropped_by_default(caplog):
    payload = _payload(
        [_truth("t0", GOOD_FORM_FACTOR_FACT, category="form_factor")]
        + [_truth(f"t{i}", fact) for i, fact in enumerate([GOOD_FACT_1, GOOD_FACT_2, GOOD_FACT_3], start=1)]
        + [_truth("t_imp", GOOD_IMPERFECTION_FACT, category="imperfection")]
    )
    client = FakeOpenAIClient([payload])

    with caplog.at_level("INFO"):
        result = await extract_product_truths(["http://example.com/a.jpg"], client=client)

    assert all(t["category"] != "imperfection" for t in result)
    assert not any(t["truth_id"] == "t_imp" for t in result)
    assert any("dropped 1 imperfection-category fact" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_imperfection_fact_kept_when_brief_asks_for_authentic_angle():
    payload = _payload(
        [_truth("t0", GOOD_FORM_FACTOR_FACT, category="form_factor")]
        + [_truth(f"t{i}", fact) for i, fact in enumerate([GOOD_FACT_1, GOOD_FACT_2, GOOD_FACT_3], start=1)]
        + [_truth("t_imp", GOOD_IMPERFECTION_FACT, category="imperfection")]
    )
    client = FakeOpenAIClient([payload])

    result = await extract_product_truths(
        ["http://example.com/a.jpg"], brief="a well-loved, authentic leather bag", client=client
    )

    assert any(t["truth_id"] == "t_imp" for t in result)


@pytest.mark.asyncio
async def test_imperfection_fact_kept_when_freeform_asks_for_character():
    payload = _payload(
        [_truth("t0", GOOD_FORM_FACTOR_FACT, category="form_factor")]
        + [_truth(f"t{i}", fact) for i, fact in enumerate([GOOD_FACT_1, GOOD_FACT_2, GOOD_FACT_3], start=1)]
        + [_truth("t_imp", GOOD_IMPERFECTION_FACT, category="imperfection")]
    )
    client = FakeOpenAIClient([payload])

    result = await extract_product_truths(
        ["http://example.com/a.jpg"], freeform="lean into the vintage character of this piece", client=client
    )

    assert any(t["truth_id"] == "t_imp" for t in result)


def test_wants_imperfection_angle_keyword_proxy():
    from agents.product_truth_extractor import _wants_imperfection_angle

    assert _wants_imperfection_angle("a durable everyday backpack", None) is False
    assert _wants_imperfection_angle("an authentic, well-loved leather bag", None) is True
    assert _wants_imperfection_angle(None, "show off its patina and character") is True


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
