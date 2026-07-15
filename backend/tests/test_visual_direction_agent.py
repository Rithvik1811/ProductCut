"""
Tests for visual_direction_agent — covers:
  1. Happy path: valid VDA JSON accepted, no re-prompt
  2. Re-prompt on invalid first response: second call returns valid JSON
  3. Fallback on double failure: both calls invalid → fallback beats used
  4. _validate_vda_output: missing required field triggers rejection
  5. _validate_vda_output: wrong type (beat_visual_directions as string) is rejected
  6. CTA beat (last) must have human_presence "no"
  7. human_presence "yes" requires human_action
  8. State I/O: node reads winning_script, writes visual_direction
  9. No winning_script in state: documents KeyError behavior
 10. JSON parse failure on first call: re-prompt triggered
"""
from __future__ import annotations

import json

import pytest

from agents.visual_direction_agent import (
    _validate_vda_output,
    generate_visual_direction,
    visual_direction_agent_node,
)
from tests._fakes import FakeOpenAIClient

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_TRUTHS = [
    {"truth_id": "t1", "fact": "1-litre stainless steel flask with matte finish", "category": "form_factor", "source": "photo_1"},
    {"truth_id": "t2", "fact": "midnight-blue powder coat on outer shell", "category": "color", "source": "photo_1"},
]

_WINNING_SCRIPT = {
    "text": "Built for your commute. Stay cold all day. Get yours.",
    "beats": [
        {"t_start": 0, "t_end": 5, "line": "Built for your commute."},
        {"t_start": 5, "t_end": 12, "line": "Stay cold all day."},
        {"t_start": 12, "t_end": 18, "line": "Get yours."},
    ],
    "source_variant_ids": ["v1"],
}


def _valid_bvd_list() -> list[dict]:
    return [
        {
            "beat_index": 0,
            "focus_feature_truth_id": "t1",
            "focus_moment": "flask emerging from dark background",
            "human_presence": "no",
            "suggested_shot_type": "hook_hero",
            "suggested_camera_move": "push_in",
            "framing_notes": "fills frame center",
        },
        {
            "beat_index": 1,
            "focus_feature_truth_id": "t2",
            "focus_moment": "matte blue surface in sidelight",
            "human_presence": "no",
            "suggested_shot_type": "macro_detail",
            "suggested_camera_move": "static",
            "framing_notes": "tight on surface texture",
        },
        {
            "beat_index": 2,
            "focus_feature_truth_id": "t1",
            "focus_moment": "flask on minimal surface, center frame",
            "human_presence": "no",
            "suggested_shot_type": "cta_endcard",
            "suggested_camera_move": "static",
            "framing_notes": "clean product hero, fade to black",
        },
    ]


def _valid_vda_json() -> str:
    return json.dumps({
        "story_context": "Clean minimal reveal across three beats.",
        "beat_visual_directions": _valid_bvd_list(),
    })


# ---------------------------------------------------------------------------
# 1. Happy path
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_happy_path_valid_vda_no_reprompt():
    client = FakeOpenAIClient([_valid_vda_json()])
    result = await generate_visual_direction(_WINNING_SCRIPT, _TRUTHS, client=client)

    assert client.call_count == 1, "Only one LLM call expected on valid response"
    assert result["story_context"] == "Clean minimal reveal across three beats."
    assert len(result["beat_visual_directions"]) == 3
    assert result["beat_visual_directions"][2]["suggested_shot_type"] == "cta_endcard"


# ---------------------------------------------------------------------------
# 2. Re-prompt on invalid first response
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_reprompt_on_invalid_first_call():
    invalid_first = json.dumps({
        "story_context": "bad",
        "beat_visual_directions": [],  # wrong: 0 entries instead of 3
    })
    client = FakeOpenAIClient([invalid_first, _valid_vda_json()])
    result = await generate_visual_direction(_WINNING_SCRIPT, _TRUTHS, client=client)

    assert client.call_count == 2, "Re-prompt should trigger a second LLM call"
    assert len(result["beat_visual_directions"]) == 3


# ---------------------------------------------------------------------------
# 3. Fallback on double failure
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_fallback_on_double_failure():
    bad_json = json.dumps({"story_context": "bad", "beat_visual_directions": []})
    client = FakeOpenAIClient([bad_json, bad_json])
    result = await generate_visual_direction(_WINNING_SCRIPT, _TRUTHS, client=client)

    assert client.call_count == 2
    bvds = result["beat_visual_directions"]
    assert len(bvds) == 3
    for bvd in bvds:
        assert bvd["human_presence"] == "no"
    assert bvds[-1]["suggested_shot_type"] == "cta_endcard"
    assert bvds[0]["suggested_shot_type"] == "macro_detail"


# ---------------------------------------------------------------------------
# 4. _validate_vda_output — missing required fields
# ---------------------------------------------------------------------------

def test_validate_missing_beat_visual_directions():
    problems = _validate_vda_output({}, beat_count=3, truth_ids=["t1", "t2"])
    assert any("beat_visual_directions" in p for p in problems)


def test_validate_wrong_beat_count():
    result = {"beat_visual_directions": _valid_bvd_list()[:2]}
    problems = _validate_vda_output(result, beat_count=3, truth_ids=["t1", "t2"])
    assert any("3" in p for p in problems)


def test_validate_missing_focus_feature_truth_id():
    bvds = _valid_bvd_list()
    bvds[0]["focus_feature_truth_id"] = "t99"  # not in truth_ids
    problems = _validate_vda_output(
        {"beat_visual_directions": bvds}, beat_count=3, truth_ids=["t1", "t2"]
    )
    assert any("focus_feature_truth_id" in p or "t99" in p for p in problems)


def test_validate_invalid_shot_type():
    bvds = _valid_bvd_list()
    bvds[0]["suggested_shot_type"] = "not_a_real_type"
    problems = _validate_vda_output(
        {"beat_visual_directions": bvds}, beat_count=3, truth_ids=["t1", "t2"]
    )
    assert any("suggested_shot_type" in p for p in problems)


def test_validate_invalid_camera_move():
    bvds = _valid_bvd_list()
    bvds[1]["suggested_camera_move"] = "helicopter"
    problems = _validate_vda_output(
        {"beat_visual_directions": bvds}, beat_count=3, truth_ids=["t1", "t2"]
    )
    assert any("suggested_camera_move" in p for p in problems)


# ---------------------------------------------------------------------------
# 5. _validate_vda_output — wrong type
# ---------------------------------------------------------------------------

def test_validate_beat_visual_directions_not_a_list():
    result = {"beat_visual_directions": "should be a list"}
    problems = _validate_vda_output(result, beat_count=3, truth_ids=["t1", "t2"])
    assert any("not a list" in p for p in problems)


def test_validate_beat_index_not_int():
    bvds = _valid_bvd_list()
    bvds[0]["beat_index"] = "zero"
    problems = _validate_vda_output(
        {"beat_visual_directions": bvds}, beat_count=3, truth_ids=["t1", "t2"]
    )
    assert any("beat_index" in p for p in problems)


# ---------------------------------------------------------------------------
# 6. CTA beat (last) must have human_presence "no"
# ---------------------------------------------------------------------------

def test_validate_cta_beat_human_presence_must_be_no():
    bvds = _valid_bvd_list()
    bvds[-1]["human_presence"] = "yes"
    bvds[-1]["human_action"] = "A hand grips the flask."
    problems = _validate_vda_output(
        {"beat_visual_directions": bvds}, beat_count=3, truth_ids=["t1", "t2"]
    )
    assert any("human_presence" in p and ("no" in p or "CTA" in p or "last" in p) for p in problems)


def test_validate_all_valid_produces_no_problems():
    problems = _validate_vda_output(
        {"beat_visual_directions": _valid_bvd_list()}, beat_count=3, truth_ids=["t1", "t2"]
    )
    assert problems == []


# ---------------------------------------------------------------------------
# 7. human_presence "yes" requires human_action
# ---------------------------------------------------------------------------

def test_validate_human_presence_yes_requires_human_action():
    bvds = _valid_bvd_list()
    bvds[0]["human_presence"] = "yes"
    # Deliberately omit human_action
    problems = _validate_vda_output(
        {"beat_visual_directions": bvds}, beat_count=3, truth_ids=["t1", "t2"]
    )
    assert any("human_action" in p for p in problems)


def test_validate_human_presence_yes_with_action_is_valid():
    bvds = _valid_bvd_list()
    bvds[1]["human_presence"] = "yes"
    bvds[1]["human_action"] = "A hand lifts the flask from a wooden surface."
    problems = _validate_vda_output(
        {"beat_visual_directions": bvds}, beat_count=3, truth_ids=["t1", "t2"]
    )
    assert problems == [], f"Unexpected problems: {problems}"


# ---------------------------------------------------------------------------
# 8. State I/O: node reads winning_script, writes visual_direction
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_node_reads_winning_script_writes_visual_direction(monkeypatch):
    monkeypatch.setattr(
        "agents.visual_direction_agent.AsyncOpenAI",
        lambda *a, **kw: FakeOpenAIClient([_valid_vda_json()]),
    )
    state = {
        "winning_script": _WINNING_SCRIPT,
        "product_truths": _TRUTHS,
        "reasoning_trace": "",
    }
    result = await visual_direction_agent_node(state)

    assert "visual_direction" in result
    vd = result["visual_direction"]
    assert len(vd["beat_visual_directions"]) == 3
    assert "reasoning_trace" in result
    assert "visual_direction_agent" in result["reasoning_trace"]


# ---------------------------------------------------------------------------
# 9. No winning_script in state: documents current behavior (KeyError)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_node_missing_winning_script_raises():
    with pytest.raises(KeyError):
        await visual_direction_agent_node({"product_truths": _TRUTHS, "reasoning_trace": ""})


# ---------------------------------------------------------------------------
# 10. JSON parse failure on first call triggers re-prompt
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_json_parse_failure_triggers_reprompt():
    not_json = "Sorry, I cannot help with that request."
    client = FakeOpenAIClient([not_json, _valid_vda_json()])
    result = await generate_visual_direction(_WINNING_SCRIPT, _TRUTHS, client=client)

    assert client.call_count == 2
    assert len(result["beat_visual_directions"]) == 3
