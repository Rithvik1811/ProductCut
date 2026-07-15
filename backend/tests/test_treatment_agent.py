"""
Unit tests for the Treatment Agent's re-prompt/degrade control flow -- same
rationale as test_concept_agent.py / test_hook_checker.py: these code paths
(missing beat entry, non-verbatim quote, unknown truth_id, banned word,
persistent-failure fallback) are unlikely to fire reliably against the real
model on demand, so they're covered with a fake client instead.

The per-beat grounding checks themselves (verbatim quote, real truth_fact_id,
valid beat_function, stoplist) now live in
agents.justification_validator.validate_justifications, shared with
Shot-List Agent -- see tests/test_justification_validator.py for direct unit
tests of that function. These tests stay at the generate_treatment() level to
confirm the re-prompt-then-fallback behavior built on top of it is unchanged.
"""
from __future__ import annotations

import json

import pytest

from agents.treatment_agent import (
    _FALLBACK_VISUAL_APPROACH,
    _FALLBACK_WHY_NOT_GENERIC,
    _build_system_prompt,
    _script_implies_person,
    generate_treatment,
)
from tests._fakes import FakeOpenAIClient

WINNING_SCRIPT = {
    "text": "Your coffee is cold in 12 minutes. Mine isn't. Double-wall seam keeps it hot. Grab yours today.",
    "beats": [
        {"t_start": 0, "t_end": 3, "line": "Your coffee is cold in 12 minutes. Mine isn't."},
        {"t_start": 3, "t_end": 9, "line": "Double-wall seam keeps it hot."},
        {"t_start": 9, "t_end": 15, "line": "Grab yours today."},
    ],
    "source_variant_ids": ["v1"],
}

TRUTHS = [
    {"truth_id": "t1", "fact": "double-wall stainless seam", "category": "construction_detail", "source": "photo_1"},
    {"truth_id": "t2", "fact": "matte black finish", "category": "color", "source": "photo_1"},
]


def _beat_treatment(
    beat_index: int,
    beat_function: str,
    script_quote: str,
    truth_fact_id: str = "t1",
    visual_approach: str = "static macro push on the seam",
    why_not_generic: str = "This seam is the one visible proof of the double-wall claim.",
) -> dict:
    return {
        "beat_index": beat_index,
        "beat_function": beat_function,
        "script_quote": script_quote,
        "truth_fact_id": truth_fact_id,
        "visual_approach": visual_approach,
        "why_not_generic": why_not_generic,
    }


GOOD_BEAT_TREATMENTS = [
    _beat_treatment(0, "hook", "Your coffee is cold in 12 minutes."),
    _beat_treatment(1, "demo", "Double-wall seam keeps it hot."),
    _beat_treatment(2, "cta", "Grab yours today."),
]


def _payload(beat_treatments: list[dict], **top_level) -> str:
    body = {
        "director_persona": "quiet, tactile, slow-reveal",
        "color_story": "warm neutrals, single matte-black accent",
        "pacing_philosophy": "let the hook breathe, then a quick punch on the CTA",
        "beat_treatments": beat_treatments,
    }
    body.update(top_level)
    return json.dumps(body)


@pytest.mark.asyncio
async def test_all_beats_valid_without_reprompt():
    client = FakeOpenAIClient([_payload(GOOD_BEAT_TREATMENTS)])

    treatment = await generate_treatment(WINNING_SCRIPT, TRUTHS, client=client)

    assert client.call_count == 1
    assert len(treatment["beat_treatments"]) == 3
    assert [bt["beat_function"] for bt in treatment["beat_treatments"]] == ["hook", "demo", "cta"]
    assert treatment["director_persona"] == "quiet, tactile, slow-reveal"


@pytest.mark.asyncio
async def test_missing_beat_entry_triggers_reprompt_and_fix_is_accepted():
    incomplete = [GOOD_BEAT_TREATMENTS[0], GOOD_BEAT_TREATMENTS[2]]  # beat 1 missing
    client = FakeOpenAIClient([_payload(incomplete), _payload(GOOD_BEAT_TREATMENTS)])

    treatment = await generate_treatment(WINNING_SCRIPT, TRUTHS, client=client)

    assert client.call_count == 2
    beat1 = treatment["beat_treatments"][1]
    assert beat1["script_quote"] == "Double-wall seam keeps it hot."
    assert beat1["why_not_generic"] != _FALLBACK_WHY_NOT_GENERIC


@pytest.mark.asyncio
async def test_non_verbatim_quote_is_rejected_then_fixed():
    bad = [
        GOOD_BEAT_TREATMENTS[0],
        _beat_treatment(1, "demo", "totally made up text not in the script"),
        GOOD_BEAT_TREATMENTS[2],
    ]
    client = FakeOpenAIClient([_payload(bad), _payload(GOOD_BEAT_TREATMENTS)])

    treatment = await generate_treatment(WINNING_SCRIPT, TRUTHS, client=client)

    assert client.call_count == 2
    assert treatment["beat_treatments"][1]["script_quote"] == "Double-wall seam keeps it hot."


@pytest.mark.asyncio
async def test_unknown_truth_id_is_rejected_then_fixed():
    bad = [
        _beat_treatment(0, "hook", "Your coffee is cold in 12 minutes.", truth_fact_id="t99"),
        GOOD_BEAT_TREATMENTS[1],
        GOOD_BEAT_TREATMENTS[2],
    ]
    client = FakeOpenAIClient([_payload(bad), _payload(GOOD_BEAT_TREATMENTS)])

    treatment = await generate_treatment(WINNING_SCRIPT, TRUTHS, client=client)

    assert client.call_count == 2
    assert treatment["beat_treatments"][0]["truth_fact_id"] == "t1"


@pytest.mark.asyncio
async def test_banned_word_category_is_rejected_then_fixed():
    bad = [
        GOOD_BEAT_TREATMENTS[0],
        GOOD_BEAT_TREATMENTS[1],
        _beat_treatment(
            2, "cta", "Grab yours today.",
            why_not_generic="This is what every product in this category needs to show.",
        ),
    ]
    client = FakeOpenAIClient([_payload(bad), _payload(GOOD_BEAT_TREATMENTS)])

    treatment = await generate_treatment(WINNING_SCRIPT, TRUTHS, client=client)

    assert client.call_count == 2
    assert "category" not in treatment["beat_treatments"][2]["why_not_generic"].lower()


@pytest.mark.asyncio
async def test_degrades_to_literal_fallback_when_still_invalid_after_retry(caplog):
    # beat 2's truth_fact_id is bad both times -- never fixed.
    bad = [
        GOOD_BEAT_TREATMENTS[0],
        GOOD_BEAT_TREATMENTS[1],
        _beat_treatment(2, "cta", "Grab yours today.", truth_fact_id="t99"),
    ]
    client = FakeOpenAIClient([_payload(bad), _payload(bad)])

    with caplog.at_level("WARNING"):
        treatment = await generate_treatment(WINNING_SCRIPT, TRUTHS, client=client)

    assert client.call_count == 2
    fallback_beat = treatment["beat_treatments"][2]
    assert fallback_beat["visual_approach"] == _FALLBACK_VISUAL_APPROACH
    assert fallback_beat["why_not_generic"] == _FALLBACK_WHY_NOT_GENERIC
    assert fallback_beat["script_quote"] == "Grab yours today."  # beat's own line, guaranteed verbatim
    assert fallback_beat["truth_fact_id"] == "t1"  # first available truth, deterministic
    assert fallback_beat["beat_function"] == "cta"  # last beat -> deterministic default
    assert any("falling back to the literal lowest-risk treatment" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# CHARACTER ANCHOR (v10, graph/state.py Treatment.character_anchor).
# ---------------------------------------------------------------------------

SCRIPT_WITH_IMPLIED_PERSON = {
    "text": "She slings it over one shoulder on her way out the door. Double-wall seam keeps it hot. Grab yours today.",
    "beats": [
        {"t_start": 0, "t_end": 3, "line": "She slings it over one shoulder on her way out the door."},
        {"t_start": 3, "t_end": 9, "line": "Double-wall seam keeps it hot."},
        {"t_start": 9, "t_end": 15, "line": "Grab yours today."},
    ],
    "source_variant_ids": ["v1"],
}

IMPLIED_PERSON_BEAT_TREATMENTS = [
    _beat_treatment(0, "hook", "She slings it over one shoulder on her way out the door."),
    _beat_treatment(1, "demo", "Double-wall seam keeps it hot."),
    _beat_treatment(2, "cta", "Grab yours today."),
]


def test_script_implies_person_detects_pronoun():
    assert _script_implies_person(SCRIPT_WITH_IMPLIED_PERSON) is True


def test_script_implies_person_false_for_product_only_script():
    assert _script_implies_person(WINNING_SCRIPT) is False


def test_system_prompt_requests_character_anchor_when_person_implied():
    prompt = _build_system_prompt(3, implies_person=True)

    assert "character_anchor" in prompt
    assert "hair color, length, and texture" in prompt
    assert "color_story" in prompt


def test_system_prompt_omits_character_anchor_field_when_no_person():
    prompt = _build_system_prompt(3, implies_person=False)

    assert "does not\n   imply a recurring person" in prompt or "does not imply a recurring person" in prompt


@pytest.mark.asyncio
async def test_character_anchor_set_when_script_implies_person():
    payload = _payload(IMPLIED_PERSON_BEAT_TREATMENTS, character_anchor=(
        "A woman in her late 20s with shoulder-length dark hair wears a "
        "rust-orange canvas jacket in a sunlit kitchen with an open window "
        "and a wooden counter, mid-morning."
    ))
    client = FakeOpenAIClient([payload])

    treatment = await generate_treatment(SCRIPT_WITH_IMPLIED_PERSON, TRUTHS, client=client)

    assert treatment.get("character_anchor", "").startswith("A woman in her late 20s")


@pytest.mark.asyncio
async def test_character_anchor_absent_when_script_has_no_person():
    client = FakeOpenAIClient([_payload(GOOD_BEAT_TREATMENTS)])

    treatment = await generate_treatment(WINNING_SCRIPT, TRUTHS, client=client)

    assert not treatment.get("character_anchor")


@pytest.mark.asyncio
async def test_fabricated_character_anchor_is_discarded_on_product_only_script(caplog):
    # Model returns a character_anchor anyway even though the script never
    # implies a person -- the deterministic guard must strip it.
    payload = _payload(GOOD_BEAT_TREATMENTS, character_anchor="A man in a blue jacket in a garage.")
    client = FakeOpenAIClient([payload])

    with caplog.at_level("INFO"):
        treatment = await generate_treatment(WINNING_SCRIPT, TRUTHS, client=client)

    assert not treatment.get("character_anchor")
    assert any("discarding it" in r.message for r in caplog.records)
