"""
Unit tests for the C3 shot-list Pydantic schema (graph/shot_schema.py).

Covers the things this module exists to guarantee mechanically: (1) the v2
additive enum values (rack_focus, product_in_hand) added ahead of the
Shot-List Agent build validate correctly, (2) the hard no-`product_category`
rule (extra="forbid") actually rejects a shot carrying that field, since that
is the concrete anti-genericness mechanism the schema is built around, and
(3) the v3 fallback_requested/failure_reason fields (Phase 3 KR/RR sync,
formalizing agents/video_gen_node.py's failure hand-off contract) validate
correctly and reject a malformed failure_reason.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from graph.shot_schema import validate_shot, validate_shot_list


def _shot(**overrides) -> dict:
    base = {
        "shot_id": "s1",
        "t_start": 0.0,
        "t_end": 4.0,
        "beat_role": "hook",
        "description": "A macro push-in on the double-wall seam.",
        "shot_type": "macro_detail",
        "camera_move": "push_in",
        "framing": "fills_frame",
        "lighting": "soft key light, neutral background, clean commercial look",
        "negative_prompt": "warped label, distorted logo",
        "reference_image_id": "photo_1",
        "text_overlay_zone": "none",
        "duration_sec": 4.0,
        "allocated_budget": 0.5,
        "voiceover_line": "Your coffee is cold in 12 minutes. Mine isn't.",
        "justification": {
            "script_quote": "Your coffee is cold in 12 minutes. Mine isn't.",
            "truth_fact_id": "t3",
            "treatment_ref": 0,
        },
        "status": "pending",
        "retry_count": 0,
    }
    base.update(overrides)
    return base


def test_valid_shot_passes():
    validate_shot(_shot())


def test_rack_focus_camera_move_is_valid():
    shot = validate_shot(_shot(camera_move="rack_focus"))
    assert shot.camera_move == "rack_focus"


def test_product_in_hand_shot_type_is_valid():
    shot = validate_shot(_shot(shot_type="product_in_hand"))
    assert shot.shot_type == "product_in_hand"


def test_worn_in_use_shot_type_is_valid():
    """C3 v4 addition (video-gen-fidelity branch): the wider, person-in-motion
    human-interaction composition, distinct from the static/close
    product_in_hand composition."""
    shot = validate_shot(_shot(shot_type="worn_in_use"))
    assert shot.shot_type == "worn_in_use"


def test_unknown_camera_move_is_rejected():
    with pytest.raises(ValidationError):
        validate_shot(_shot(camera_move="dolly_zoom"))


def test_unknown_shot_type_is_rejected():
    with pytest.raises(ValidationError):
        validate_shot(_shot(shot_type="explainer_diagram"))


def test_product_category_field_is_hard_rejected():
    """The concrete anti-genericness mechanism: extra="forbid" means a shot
    carrying `product_category` fails validation outright, not on a prompt-level
    guard the LLM could ignore."""
    with pytest.raises(ValidationError):
        validate_shot(_shot(product_category="running_shoe"))


def test_missing_justification_is_rejected():
    shot = _shot()
    del shot["justification"]
    with pytest.raises(ValidationError):
        validate_shot(shot)


def test_validate_shot_list_validates_every_shot():
    shots = validate_shot_list([_shot(shot_id="s1"), _shot(shot_id="s2", camera_move="orbit")])
    assert [s.shot_id for s in shots] == ["s1", "s2"]


def test_validate_shot_list_raises_on_first_invalid_shot():
    with pytest.raises(ValidationError):
        validate_shot_list([_shot(shot_id="s1"), _shot(shot_id="s2", camera_move="bad_move")])


def test_fallback_requested_status_is_valid():
    shot = validate_shot(_shot(status="fallback_requested"))
    assert shot.status == "fallback_requested"


def test_shot_without_failure_reason_defaults_to_none():
    shot = validate_shot(_shot())
    assert shot.failure_reason is None


def test_shot_with_valid_failure_reason_passes():
    shot = validate_shot(_shot(
        status="fallback_requested",
        failure_reason={"type": "timeout", "detail": "Wan exceeded the 180s wait timeout"},
    ))
    assert shot.failure_reason.type == "timeout"
    assert shot.failure_reason.detail == "Wan exceeded the 180s wait timeout"


@pytest.mark.parametrize("failure_type", ["timeout", "api_error", "budget_exceeded"])
def test_all_three_failure_types_are_valid(failure_type):
    shot = validate_shot(_shot(
        status="fallback_requested",
        failure_reason={"type": failure_type, "detail": "some detail"},
    ))
    assert shot.failure_reason.type == failure_type


def test_unknown_failure_type_is_rejected():
    with pytest.raises(ValidationError):
        validate_shot(_shot(
            status="fallback_requested",
            failure_reason={"type": "gremlins", "detail": "the API just vanished"},
        ))


def test_failure_reason_missing_detail_is_rejected():
    with pytest.raises(ValidationError):
        validate_shot(_shot(
            status="fallback_requested",
            failure_reason={"type": "timeout"},
        ))
