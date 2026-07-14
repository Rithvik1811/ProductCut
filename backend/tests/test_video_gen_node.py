"""
Unit tests for the Video-Gen Node -- parallel Send() fan-out, the budget-clamp
policy, and the failure hand-off contract (status/failure_reason, retry_count
guaranteed untouched). Uses a fixture shot list matching the REAL merged C3
schema (graph.state.Shot / graph.shot_schema.ShotModel), not an earlier-docs
guess -- see agents/shot_list_agent.py's own `_assemble_shots` for the shape
this mirrors.

Every test injects a fake `generate_fn` (same injection pattern as every
other agent's `client=`/`validate_justifications=` parameter) -- no real
DashScope call is ever made here.
"""
from __future__ import annotations

import asyncio
import time

import pytest
from langchain_core.runnables import RunnableLambda

from agents.budget_gate import RATE_720P, RATE_1080P
from agents.shot_list_agent import MIN_SHOT_DURATION_SEC
from agents.video_gen_node import (
    FAILURE_TYPE_API_ERROR,
    FAILURE_TYPE_BUDGET_EXCEEDED,
    FAILURE_TYPE_TIMEOUT,
    FALLBACK_REQUESTED_STATUS,
    PROMPT_CHAR_BUDGET,
    SUCCESS_STATUS,
    VideoGenAPIError,
    VideoGenTimeoutError,
    _build_prompt,
    _call_wan_video_gen,
    _resolve_generation_params,
    _resolve_reference_image_url,
    generate_videos,
    video_gen_node,
)

FORM_FACTOR_TRUTH = {
    "truth_id": "tff",
    "fact": "a deep, rounded block, wider than it is tall, matte charcoal finish",
    "category": "form_factor",
    "source": "photo_1",
}

TRUTHS = [
    {"truth_id": "t1", "fact": "double-wall stainless seam", "category": "construction_detail", "source": "photo_1"},
    {"truth_id": "t2", "fact": "matte black finish", "category": "color", "source": "photo_1"},
]
TRUTHS_WITH_FORM_FACTOR = TRUTHS + [FORM_FACTOR_TRUTH]

TREATMENT = {
    "director_persona": "quiet, tactile, slow-reveal",
    "color_story": "warm neutrals, single matte-black accent",
    "pacing_philosophy": "let the hook breathe, then a quick punch on the CTA",
    "beat_treatments": [
        {
            "beat_index": 0,
            "beat_function": "hook",
            "script_quote": "Your coffee is cold in 12 minutes.",
            "truth_fact_id": "t1",
            "visual_approach": "static macro push on the seam",
            "why_not_generic": "This seam is the one visible proof of the double-wall claim.",
        },
    ],
}

PRODUCT_PHOTOS = ["http://example.com/photo1.jpg", "http://example.com/photo2.jpg"]


def _shot(
    shot_id: str,
    *,
    duration_sec: float = 4.0,
    allocated_budget: float = 1.0,
    shot_type: str = "macro_detail",
    camera_move: str = "push_in",
    reference_image_id: str = "photo_1",
    truth_fact_id: str = "t1",
    text_overlay_zone: str = "none",
    retry_count: int = 0,
) -> dict:
    return {
        "shot_id": shot_id,
        "t_start": 0.0,
        "t_end": duration_sec,
        "beat_role": "hook",
        "description": "The seam catches the morning light as the mug slowly rotates into frame.",
        "shot_type": shot_type,
        "camera_move": camera_move,
        "framing": "fills_frame",
        "lighting": "soft key light, neutral background, clean commercial look",
        "negative_prompt": "warped label, distorted logo, morphing text, deformed hands, fused fingers, low quality",
        "reference_image_id": reference_image_id,
        "text_overlay_zone": text_overlay_zone,
        "duration_sec": duration_sec,
        "allocated_budget": allocated_budget,
        "voiceover_line": "Your coffee is cold in 12 minutes.",
        "justification": {
            "script_quote": "Your coffee is cold in 12 minutes.",
            "truth_fact_id": truth_fact_id,
            "treatment_ref": 0,
        },
        "status": "pending",
        "retry_count": retry_count,
    }


# ---------------------------------------------------------------------------
# _resolve_reference_image_url
# ---------------------------------------------------------------------------
def test_reference_image_maps_photo_n_to_1_indexed_url():
    assert _resolve_reference_image_url("photo_1", PRODUCT_PHOTOS) == PRODUCT_PHOTOS[0]
    assert _resolve_reference_image_url("photo_2", PRODUCT_PHOTOS) == PRODUCT_PHOTOS[1]


def test_reference_image_defaults_to_first_photo_when_out_of_range_or_malformed():
    assert _resolve_reference_image_url("photo_99", PRODUCT_PHOTOS) == PRODUCT_PHOTOS[0]
    assert _resolve_reference_image_url("not-a-photo-id", PRODUCT_PHOTOS) == PRODUCT_PHOTOS[0]
    assert _resolve_reference_image_url("photo_1", []) == ""


# ---------------------------------------------------------------------------
# _resolve_generation_params (budget clamp, requirement 5)
# ---------------------------------------------------------------------------
def test_budget_covers_full_1080p():
    shot = _shot("s1", duration_sec=4.0, allocated_budget=4.0 * RATE_1080P)
    duration, resolution, failure = _resolve_generation_params(shot)
    assert (duration, resolution, failure) == (4.0, "1080P", None)


def test_budget_clamps_resolution_to_720p_only():
    shot = _shot("s1", duration_sec=4.0, allocated_budget=4.0 * RATE_720P)  # < 1080p cost, >= 720p cost
    duration, resolution, failure = _resolve_generation_params(shot)
    assert duration == 4.0  # duration untouched
    assert resolution == "720P"
    assert failure is None


def test_budget_clamps_duration_down_at_720p():
    # allocated affords < full duration at 720p but still >= the MIN_SHOT_DURATION_SEC floor.
    allocated = (MIN_SHOT_DURATION_SEC + 0.5) * RATE_720P
    shot = _shot("s1", duration_sec=5.0, allocated_budget=allocated)
    duration, resolution, failure = _resolve_generation_params(shot)
    assert resolution == "720P"
    assert failure is None
    assert duration == pytest.approx(allocated / RATE_720P)
    assert duration < shot["duration_sec"]
    assert duration >= MIN_SHOT_DURATION_SEC


def test_budget_exceeded_below_floor_fails_without_calling_api():
    # Budget Gate's §5.7 floor case: allocated_budget pinned below what even the
    # cheapest (MIN_SHOT_DURATION_SEC @ 720p) shot costs.
    allocated = (MIN_SHOT_DURATION_SEC * RATE_720P) - 0.01
    shot = _shot("s1", duration_sec=5.0, allocated_budget=allocated)
    duration, resolution, failure = _resolve_generation_params(shot)
    assert duration is None
    assert resolution is None
    assert failure["type"] == FAILURE_TYPE_BUDGET_EXCEEDED
    assert "floor" in failure["detail"]


# ---------------------------------------------------------------------------
# _build_prompt (requirement 2 mapping)
# ---------------------------------------------------------------------------
def test_prompt_sections_present_in_order():
    shot = _shot("s1")
    prompt = _build_prompt(shot, TRUTHS, TREATMENT)
    for section in ["Subject:", "Action/Motion:", "Camera:", "Lighting:", "Composition:", "Mood:", "Quality:"]:
        assert section in prompt
    assert prompt.index("Subject:") < prompt.index("Action/Motion:") < prompt.index("Camera:")
    assert prompt.index("Camera:") < prompt.index("Lighting:") < prompt.index("Composition:")
    assert prompt.index("Composition:") < prompt.index("Mood:") < prompt.index("Quality:")
    assert "double-wall stainless seam" in prompt  # cited truth grounds Subject
    assert shot["description"] in prompt  # Action/Motion reuses it verbatim
    assert TREATMENT["director_persona"] in prompt  # Mood


def test_prompt_adds_hand_continuity_clause_for_product_in_hand():
    shot = _shot("s1", shot_type="product_in_hand")
    prompt = _build_prompt(shot, TRUTHS, TREATMENT)
    assert "five natural fingers" in prompt
    assert "no scene cut" in prompt


def test_prompt_omits_hand_continuity_clause_for_non_human_shot_types():
    shot = _shot("s1", shot_type="macro_detail")
    prompt = _build_prompt(shot, TRUTHS, TREATMENT)
    assert "five natural fingers" not in prompt


# ---------------------------------------------------------------------------
# video-gen-fidelity branch: worn_in_use expansion of the human-interaction
# tier, plus the scale-lock/occlusion-continuity clause extension.
# ---------------------------------------------------------------------------
def test_prompt_adds_hand_continuity_clause_for_worn_in_use():
    """worn_in_use (C3 v4) must get the exact same human-interaction treatment
    as product_in_hand -- the whole point of expanding
    _HUMAN_INTERACTION_SHOT_TYPES was to stop this composition falling through
    the old lifestyle_context overload with no safety clause at all."""
    shot = _shot("s1", shot_type="worn_in_use")
    prompt = _build_prompt(shot, TRUTHS, TREATMENT)
    assert "five natural fingers" in prompt
    assert "no scene cut" in prompt


# ---------------------------------------------------------------------------
# video-gen-fidelity PHASE 1: the four identity-protection clauses (scale-lock,
# occlusion-continuity, anatomy, anti-cut) are now ONE compressed sentence
# (_IDENTITY_PROTECTION_CLAUSE), not four separate clauses -- same four
# protections, far fewer characters.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("shot_type", ["product_in_hand", "worn_in_use"])
def test_prompt_adds_compressed_identity_protection_clause_for_human_shot_types(shot_type):
    shot = _shot("s1", shot_type=shot_type)
    prompt = _build_prompt(shot, TRUTHS, TREATMENT)
    assert "keeps its exact shape, size, color and material" in prompt
    assert "stays fully recognizable when partially covered" in prompt
    assert "stays in frame" in prompt
    assert "five natural fingers" in prompt
    assert "no scene cut" in prompt


def test_prompt_omits_identity_protection_clause_for_non_human_shot_types():
    shot = _shot("s1", shot_type="lifestyle_context")
    prompt = _build_prompt(shot, TRUTHS, TREATMENT)
    assert "keeps its exact shape, size, color and material" not in prompt
    assert "stays fully recognizable when partially covered" not in prompt


def test_prompt_old_multi_clause_identity_wording_is_gone():
    """Regression: the OLD four-clause wording (pre-PHASE-1) must not appear
    anywhere -- it has been replaced wholesale by the compressed sentence, not
    added alongside it (that would cost MORE chars, the opposite of the fix)."""
    shot = _shot("s1", shot_type="product_in_hand")
    prompt = _build_prompt(shot, TRUTHS, TREATMENT)
    assert "anatomically correct fingers" not in prompt
    assert "exact size and proportions relative" not in prompt
    assert "silhouette unchanged" not in prompt
    assert "remains fully recognizable when partially covered" not in prompt
    assert "color and material never change" not in prompt


# ---------------------------------------------------------------------------
# video-gen-fidelity PHASE 1: fixed-camera phrasing, action-urgency clause,
# CTA terminal-stillness clause.
# ---------------------------------------------------------------------------
def test_static_camera_move_renders_fixed_camera_phrasing_not_ambiguous_static():
    shot = _shot("s1", camera_move="static")
    prompt = _build_prompt(shot, TRUTHS, TREATMENT)
    assert "Fixed camera on a tripod; the camera does not move." in prompt
    assert "The subject moves naturally and completes the described action fully within the clip." in prompt
    # The old ambiguous phrasing must be gone -- "Camera: static" alone could
    # read as "the whole scene is static", which is exactly the failure mode
    # this fix targets.
    assert "Camera: static" not in prompt


def test_non_static_camera_move_keeps_ordinary_phrasing():
    shot = _shot("s1", camera_move="push_in")
    prompt = _build_prompt(shot, TRUTHS, TREATMENT)
    assert "Camera: push in" in prompt
    assert "Fixed camera on a tripod" not in prompt


@pytest.mark.parametrize("shot_type", ["product_in_hand", "worn_in_use"])
def test_human_interaction_shots_get_action_urgency_clause(shot_type):
    shot = _shot("s1", shot_type=shot_type)
    prompt = _build_prompt(shot, TRUTHS, TREATMENT)
    assert "The action begins immediately and completes before the clip ends." in prompt


def test_non_human_shots_omit_action_urgency_clause():
    shot = _shot("s1", shot_type="macro_detail")
    prompt = _build_prompt(shot, TRUTHS, TREATMENT)
    assert "begins immediately and completes before the clip ends" not in prompt


def test_cta_endcard_shots_get_terminal_stillness_clause():
    shot = _shot("s1", shot_type="cta_endcard")
    prompt = _build_prompt(shot, TRUTHS, TREATMENT)
    assert "All motion resolves in the final second" in prompt
    assert "posing for an end card" in prompt
    # cta_endcard is never a human-interaction shot_type -- the urgency clause
    # is a distinct mechanism and must not also fire here.
    assert "begins immediately and completes before the clip ends" not in prompt


def test_non_cta_shots_omit_terminal_stillness_clause():
    shot = _shot("s1", shot_type="macro_detail")
    prompt = _build_prompt(shot, TRUTHS, TREATMENT)
    assert "All motion resolves in the final second" not in prompt


# ---------------------------------------------------------------------------
# v8 fix: form_factor anchor threaded into the Subject line (Meta Quest ->
# "phone on a stand" wrong-object bug).
# ---------------------------------------------------------------------------
def test_prompt_leads_with_form_factor_anchor_before_micro_fact_when_present():
    shot = _shot("s1")
    prompt = _build_prompt(shot, TRUTHS_WITH_FORM_FACTOR, TREATMENT)
    assert FORM_FACTOR_TRUTH["fact"] in prompt
    assert "double-wall stainless seam" in prompt
    # The anchor is the whole-object identity and must lead; the per-shot
    # micro-fact follows it (see _build_prompt's own docstring comment on why
    # leading tokens carry the most weight).
    assert prompt.index(FORM_FACTOR_TRUTH["fact"]) < prompt.index("double-wall stainless seam")
    assert prompt.index("Subject:") < prompt.index(FORM_FACTOR_TRUTH["fact"])


def test_prompt_omits_form_factor_anchor_when_absent_regression():
    """Regression: with no form_factor truth present (the pre-fix world), the
    Subject line is unchanged from before this fix -- no leading "The product:"
    clause, just the per-shot micro-fact."""
    shot = _shot("s1")
    prompt = _build_prompt(shot, TRUTHS, TREATMENT)
    assert "The product:" not in prompt
    assert "Subject: Product detail: double-wall stainless seam." in prompt


def test_prompt_falls_back_to_form_factor_anchor_alone_when_no_cited_truth_matches():
    shot = _shot("s1", truth_fact_id="does-not-exist")
    prompt = _build_prompt(shot, TRUTHS_WITH_FORM_FACTOR, TREATMENT)
    assert FORM_FACTOR_TRUTH["fact"] in prompt
    assert "Product detail:" not in prompt


# ---------------------------------------------------------------------------
# video-gen-fidelity PHASE 4 fix: drop the redundant per-shot "Product detail"
# clause from Subject on human-interaction shots (confirmed redundant with
# Action/Motion's own required verbatim contact-fact mention -- see
# agents/shot_list_agent.py's HUMAN-INTERACTION SHOTS rule -- and confirmed as
# the single largest section pushing real shots past Wan's 1,500-char hard
# truncation ceiling in a live run; see _build_prompt's own comment).
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("shot_type", ["product_in_hand", "worn_in_use"])
def test_human_interaction_shots_omit_redundant_product_detail_clause(shot_type):
    shot = _shot("s1", shot_type=shot_type)
    prompt = _build_prompt(shot, TRUTHS_WITH_FORM_FACTOR, TREATMENT)
    # The form_factor anchor (the actual identity-fix content) still leads.
    assert FORM_FACTOR_TRUTH["fact"] in prompt
    # But the redundant per-shot micro-fact restatement is gone.
    assert "Product detail:" not in prompt
    assert "double-wall stainless seam" not in prompt


def test_non_human_shots_keep_product_detail_clause_regression():
    """Regression: this fix is scoped to human-interaction shot_types only --
    an ordinary shot's Subject line is unchanged."""
    shot = _shot("s1", shot_type="macro_detail")
    prompt = _build_prompt(shot, TRUTHS_WITH_FORM_FACTOR, TREATMENT)
    assert "Product detail: double-wall stainless seam." in prompt


def test_human_interaction_shot_with_no_form_factor_falls_back_to_generic_subject():
    """With no form_factor truth present at all, a human-interaction shot's
    Subject falls back to the generic reference-photo line rather than an
    empty string -- never silently drop Subject to nothing."""
    shot = _shot("s1", shot_type="product_in_hand")
    prompt = _build_prompt(shot, TRUTHS, TREATMENT)
    assert "Subject: The product shown in the reference photo." in prompt


# ---------------------------------------------------------------------------
# video-gen-fidelity PHASE 1: hard 1,400-char prompt budget, enforced (not just
# warned about) -- cuts Quality, then compresses Mood, then trims Lighting, in
# that order, stopping as soon as it fits; never lets the server silently
# truncate an unknown tail.
# ---------------------------------------------------------------------------
def test_prompt_length_no_warning_and_all_sections_present_under_budget(caplog):
    shot = _shot("s1")
    with caplog.at_level("WARNING"):
        prompt = _build_prompt(shot, TRUTHS, TREATMENT)
    assert len(prompt) <= PROMPT_CHAR_BUDGET
    assert not any("exceeded the" in r.message or "approaching Wan's" in r.message for r in caplog.records)
    for section in ["Subject:", "Action/Motion:", "Camera:", "Lighting:", "Composition:", "Mood:", "Quality:"]:
        assert section in prompt


def test_prompt_budget_drops_quality_only_when_that_alone_suffices(caplog):
    """Just over budget from a longer-than-usual Mood dump -- dropping Quality
    alone is enough; Mood must survive UNCOMPRESSED and Lighting untouched."""
    shot = _shot("s1")
    treatment = {**TREATMENT, "director_persona": "quiet mood word " * 119, "pacing_philosophy": "p"}
    with caplog.at_level("WARNING"):
        prompt = _build_prompt(shot, TRUTHS, treatment)
    assert len(prompt) <= PROMPT_CHAR_BUDGET
    assert "Quality:" not in prompt
    assert treatment["director_persona"] in prompt  # full, UNcompressed mood text survives
    assert shot["lighting"] in prompt  # lighting untouched
    warnings = [r.message for r in caplog.records]
    assert any("exceeded the" in w and "Quality." in w for w in warnings)  # ONLY Quality was cut
    assert not any("Mood (compressed)" in w for w in warnings)
    assert not any("Lighting (trimmed)" in w for w in warnings)


def test_prompt_budget_drops_quality_then_compresses_mood(caplog):
    """Further over budget -- Quality alone isn't enough, so Mood is ALSO
    compressed to a short clause. Lighting must still survive untouched."""
    shot = _shot("s1")
    treatment = {**TREATMENT, "director_persona": "quiet mood word " * 120, "pacing_philosophy": "p"}
    with caplog.at_level("WARNING"):
        prompt = _build_prompt(shot, TRUTHS, treatment)
    assert len(prompt) <= PROMPT_CHAR_BUDGET
    assert "Quality:" not in prompt
    assert "quiet mood word " * 120 not in prompt  # the full dump is gone
    assert shot["lighting"] in prompt  # lighting still untouched
    warnings = [r.message for r in caplog.records]
    assert any("Quality, Mood (compressed)" in w for w in warnings)
    assert not any("Lighting (trimmed)" in w for w in warnings)


def test_prompt_budget_drops_all_three_sections_in_order_when_needed(caplog):
    """Even after dropping Quality and compressing Mood, a sufficiently long
    shared Lighting string still pushes it over -- Lighting gets trimmed too,
    in the same fixed order (Quality, then Mood, then Lighting)."""
    base_lighting = (
        "a warm, deliberate practical lighting setup with layered rim key and "
        "fill sources chosen to flatter the material finish. "
    )
    shot = _shot("s1")
    shot["lighting"] = base_lighting * 17
    treatment = {**TREATMENT, "director_persona": "quiet mood word " * 200, "pacing_philosophy": "p"}
    with caplog.at_level("WARNING"):
        prompt = _build_prompt(shot, TRUTHS, treatment)
    assert len(prompt) <= PROMPT_CHAR_BUDGET
    assert "Quality:" not in prompt
    assert shot["lighting"] not in prompt  # the full lighting dump is gone -- trimmed
    assert base_lighting.split(",")[0].split(".")[0].strip() in prompt  # first clause survives
    warnings = [r.message for r in caplog.records]
    assert any("Quality, Mood (compressed), Lighting (trimmed)" in w for w in warnings)


def test_prompt_action_tail_gets_trimmed_when_it_is_the_overflow_source(caplog):
    """video-gen-fidelity PHASE 4 fix: when the shot's own description is so
    long that dropping every cuttable section (Quality/Mood/Lighting) still
    isn't enough, Action/Motion's own tail is now trimmed (last resort) rather
    than left whole and silently handed to Wan's server-side truncation --
    confirmed against a real live overflow (derisk/outputs/
    full_pipeline_live_vikr_postfix.log, shot s3, a hero shot at 1999 chars
    pre-trim)."""
    shot = _shot("s1")
    shot["description"] = "A very long action description. " * 80  # pushes well over budget alone
    with caplog.at_level("WARNING"):
        prompt = _build_prompt(shot, TRUTHS, TREATMENT)
    assert len(prompt) <= PROMPT_CHAR_BUDGET
    assert shot["description"] not in prompt  # trimmed, not kept whole
    assert "A very long action description." in prompt  # at least the first sentence survives
    assert any(
        "Action/Motion (trimmed tail)" in r.message and "s1" in r.message for r in caplog.records
    )


def test_prompt_budget_never_silently_exceeds_when_even_first_sentence_is_huge(caplog):
    """Genuine last-resort case: _trim_description_tail always keeps at least
    the description's first "sentence" -- if that alone (no sentence-ending
    punctuation to trim at) is still enormous, the prompt can still exceed
    budget. Confirms this is flagged loudly, never silently truncated or
    crashed on."""
    shot = _shot("s1")
    shot["description"] = "an enormous run-on action description with no sentence breaks at all " * 30
    with caplog.at_level("WARNING"):
        prompt = _build_prompt(shot, TRUTHS, TREATMENT)
    assert len(prompt) > PROMPT_CHAR_BUDGET
    assert shot["description"] in prompt  # single unsplittable "sentence" -- never cut
    assert any(
        "approaching Wan's" in r.message and "s1" in r.message for r in caplog.records
    )


def test_prompt_budget_drops_are_logged_with_shot_id_and_final_length(caplog):
    shot = _shot("s1")
    treatment = {**TREATMENT, "director_persona": "quiet mood word " * 119, "pacing_philosophy": "p"}
    with caplog.at_level("WARNING"):
        prompt = _build_prompt(shot, TRUTHS, treatment)
    warnings = [r.message for r in caplog.records]
    assert any("s1" in m and "Final length" in m for m in warnings)
    assert len(prompt) <= PROMPT_CHAR_BUDGET


# ---------------------------------------------------------------------------
# video-gen-fidelity story-arc fix: Cast section (graph/state.py Treatment v10
# character_anchor, rendered verbatim on every human-interaction shot).
# ---------------------------------------------------------------------------
CHARACTER_ANCHOR = (
    "A woman in her late 20s with shoulder-length dark hair wears a "
    "rust-orange canvas jacket in a sunlit kitchen with an open window "
    "and a wooden counter, mid-morning."
)
TREATMENT_WITH_ANCHOR = {**TREATMENT, "character_anchor": CHARACTER_ANCHOR}


@pytest.mark.parametrize("shot_type", ["product_in_hand", "worn_in_use"])
def test_cast_section_present_verbatim_for_human_shot_with_anchor(shot_type):
    shot = _shot("s1", shot_type=shot_type)
    prompt = _build_prompt(shot, TRUTHS, TREATMENT_WITH_ANCHOR)
    assert f"Cast: {CHARACTER_ANCHOR}" in prompt
    assert prompt.index("Subject:") < prompt.index("Cast:") < prompt.index("Action/Motion:")


def test_cast_section_absent_for_non_human_shot_even_with_anchor():
    shot = _shot("s1", shot_type="macro_detail")
    prompt = _build_prompt(shot, TRUTHS, TREATMENT_WITH_ANCHOR)
    assert "Cast:" not in prompt


def test_cast_section_absent_when_no_character_anchor():
    shot = _shot("s1", shot_type="product_in_hand")
    prompt = _build_prompt(shot, TRUTHS, TREATMENT)  # TREATMENT has no character_anchor key
    assert "Cast:" not in prompt


def test_cast_section_absent_when_treatment_is_none():
    shot = _shot("s1", shot_type="worn_in_use")
    prompt = _build_prompt(shot, TRUTHS, None)
    assert "Cast:" not in prompt


def test_cast_section_never_cut_even_under_severe_overflow():
    """Cast must survive the budget cutter exactly like the identity-protection
    clause -- even when the shot's own description is so long the prompt
    can't be brought under budget by cutting every cuttable section."""
    shot = _shot("s1", shot_type="product_in_hand")
    shot["description"] = "A very long action description. " * 60
    prompt = _build_prompt(shot, TRUTHS, TREATMENT_WITH_ANCHOR)
    assert f"Cast: {CHARACTER_ANCHOR}" in prompt


def test_human_shot_drops_quality_and_compresses_mood_unconditionally():
    """Funds the Cast section without raising PROMPT_CHAR_BUDGET: Quality is
    dropped and Mood compressed on EVERY human-interaction shot, even one
    that's nowhere near the char budget on its own."""
    shot = _shot("s1", shot_type="product_in_hand")
    long_persona_treatment = {
        **TREATMENT_WITH_ANCHOR,
        "director_persona": "a long, wandering directorial voice with many clauses describing mood",
    }
    prompt = _build_prompt(shot, TRUTHS, long_persona_treatment)
    assert "Quality:" not in prompt
    assert long_persona_treatment["director_persona"] not in prompt  # full Mood dump is gone
    assert "Mood:" in prompt  # still present, just compressed


def test_human_shot_without_anchor_still_gets_unconditional_funding():
    """The Quality-drop/Mood-compress funding applies to every human-
    interaction shot, not only ones that actually have a Cast section --
    matches the research synthesis's literal instruction (funds the
    mechanism generally, not per-shot conditionally)."""
    shot = _shot("s1", shot_type="worn_in_use")
    prompt = _build_prompt(shot, TRUTHS, TREATMENT)  # no character_anchor
    assert "Cast:" not in prompt
    assert "Quality:" not in prompt


def test_non_human_shot_keeps_quality_and_full_mood_when_under_budget():
    """Regression: the unconditional human-shot funding must not leak onto
    ordinary product-alone shots -- Quality and the full Mood dump survive
    exactly as before this fix."""
    shot = _shot("s1", shot_type="macro_detail")
    prompt = _build_prompt(shot, TRUTHS, TREATMENT)
    assert "Quality:" in prompt
    assert TREATMENT["director_persona"] in prompt


def test_human_shot_funding_is_logged_at_info_not_as_an_overflow_warning(caplog):
    shot = _shot("s1", shot_type="product_in_hand")
    with caplog.at_level("INFO"):
        prompt = _build_prompt(shot, TRUTHS, TREATMENT_WITH_ANCHOR)
    assert len(prompt) <= PROMPT_CHAR_BUDGET
    info_records = [r for r in caplog.records if r.levelname == "INFO"]
    assert any("fund the Cast section" in r.message and "s1" in r.message for r in info_records)
    warning_records = [r for r in caplog.records if r.levelname == "WARNING"]
    assert not any("exceeded the" in r.message for r in warning_records)


# ---------------------------------------------------------------------------
# v8 fix: `seed` (env-driven, optional) and `prompt_extend` (default OFF) wiring
# into the real Wan call.
# ---------------------------------------------------------------------------
class _FakeWanOutput:
    task_status = "SUCCEEDED"
    video_url = "http://example.com/clip.mp4"


class _FakeWanResponse:
    status_code = 200
    output = _FakeWanOutput()


@pytest.mark.asyncio
async def test_call_wan_omits_seed_when_not_provided(monkeypatch):
    captured: dict = {}

    async def fake_call(**kwargs):
        captured.update(kwargs)
        return _FakeWanResponse()

    monkeypatch.setattr("agents.video_gen_node.AioVideoSynthesis.call", fake_call)
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")
    monkeypatch.setenv("MODEL_VIDEO", "wan-test")

    await _call_wan_video_gen(
        image_url="http://x/img.jpg", prompt="p", negative_prompt="n",
        duration_sec=4.0, resolution="1080P",
    )
    assert "seed" not in captured, "omitting seed must preserve today's random-seed production behavior"


@pytest.mark.asyncio
async def test_call_wan_passes_seed_when_provided(monkeypatch):
    captured: dict = {}

    async def fake_call(**kwargs):
        captured.update(kwargs)
        return _FakeWanResponse()

    monkeypatch.setattr("agents.video_gen_node.AioVideoSynthesis.call", fake_call)
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")
    monkeypatch.setenv("MODEL_VIDEO", "wan-test")

    await _call_wan_video_gen(
        image_url="http://x/img.jpg", prompt="p", negative_prompt="n",
        duration_sec=4.0, resolution="1080P", seed=42,
    )
    assert captured["seed"] == 42


@pytest.mark.asyncio
async def test_call_wan_prompt_extend_defaults_false_and_env_flippable(monkeypatch):
    captured: dict = {}

    async def fake_call(**kwargs):
        captured.update(kwargs)
        return _FakeWanResponse()

    monkeypatch.setattr("agents.video_gen_node.AioVideoSynthesis.call", fake_call)
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")
    monkeypatch.setenv("MODEL_VIDEO", "wan-test")

    await _call_wan_video_gen(
        image_url="http://x/img.jpg", prompt="p", negative_prompt="n",
        duration_sec=4.0, resolution="1080P",
    )
    assert captured["prompt_extend"] is False  # explicit default, not DashScope's silent true

    monkeypatch.setattr("agents.video_gen_node.DEFAULT_PROMPT_EXTEND", True)
    captured.clear()
    await _call_wan_video_gen(
        image_url="http://x/img.jpg", prompt="p", negative_prompt="n",
        duration_sec=4.0, resolution="1080P",
    )
    assert captured["prompt_extend"] is True


@pytest.mark.asyncio
async def test_generate_videos_threads_seed_env_var_into_generate_fn_only_when_set(monkeypatch):
    captured: list[dict] = []

    async def fake_generate(**kwargs):
        captured.append(kwargs)
        return "http://oss.example.com/clip.mp4"

    monkeypatch.delenv("VIDEO_GEN_SEED", raising=False)
    await generate_videos([_shot("s1")], TRUTHS, TREATMENT, PRODUCT_PHOTOS, generate_fn=fake_generate)
    assert "seed" not in captured[0]

    captured.clear()
    monkeypatch.setenv("VIDEO_GEN_SEED", "1234")
    await generate_videos([_shot("s2")], TRUTHS, TREATMENT, PRODUCT_PHOTOS, generate_fn=fake_generate)
    assert captured[0]["seed"] == 1234


# ---------------------------------------------------------------------------
# generate_videos -- happy path, real parallel Send() fan-out
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_happy_path_all_shots_succeed_in_parallel():
    per_call_delay = 0.05
    call_order: list[str] = []

    async def fake_generate(*, image_url, prompt, negative_prompt, duration_sec, resolution):
        call_order.append(image_url)
        await asyncio.sleep(per_call_delay)
        return f"http://oss.example.com/{image_url.rsplit('/', 1)[-1]}.mp4"

    shots = [_shot("s1"), _shot("s2"), _shot("s3")]

    started = time.monotonic()
    updated_shots, generated = await generate_videos(shots, TRUTHS, TREATMENT, PRODUCT_PHOTOS, generate_fn=fake_generate)
    elapsed = time.monotonic() - started

    assert len(call_order) == 3
    assert set(generated.keys()) == {"s1", "s2", "s3"}
    for shot in updated_shots:
        assert shot["status"] == SUCCESS_STATUS
        assert "failure_reason" not in shot
        assert shot["retry_count"] == 0  # untouched

    # Genuinely parallel: three 0.05s calls finish in well under their 0.15s sum.
    assert elapsed < per_call_delay * 3


@pytest.mark.asyncio
async def test_generated_shot_records_budget_clamp_info():
    allocated = 4.0 * RATE_720P  # affords 720p full duration, not 1080p
    shot = _shot("s1", duration_sec=4.0, allocated_budget=allocated)

    async def fake_generate(**kwargs):
        return "http://oss.example.com/clip.mp4"

    _, generated = await generate_videos([shot], TRUTHS, TREATMENT, PRODUCT_PHOTOS, generate_fn=fake_generate)

    entry = generated["s1"]
    assert entry["resolution_used"] == "720P"
    assert entry["duration_sec_used"] == 4.0
    assert entry["budget_clamped"] is True  # resolution was clamped even though duration wasn't
    assert entry["attempt"] == 1


# ---------------------------------------------------------------------------
# Failure hand-off (requirement 6)
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_timeout_hands_off_without_touching_retry_count():
    async def fake_generate(**kwargs):
        raise VideoGenTimeoutError("simulated timeout")

    shot = _shot("s1", retry_count=2)
    updated_shots, generated = await generate_videos([shot], TRUTHS, TREATMENT, PRODUCT_PHOTOS, generate_fn=fake_generate)

    assert generated == {}
    result = updated_shots[0]
    assert result["status"] == FALLBACK_REQUESTED_STATUS
    assert result["failure_reason"]["type"] == FAILURE_TYPE_TIMEOUT
    assert result["retry_count"] == 2  # guaranteed untouched


@pytest.mark.asyncio
async def test_api_error_hands_off_without_touching_retry_count():
    async def fake_generate(**kwargs):
        raise VideoGenAPIError("simulated 400 from Wan")

    shot = _shot("s1", retry_count=1)
    updated_shots, generated = await generate_videos([shot], TRUTHS, TREATMENT, PRODUCT_PHOTOS, generate_fn=fake_generate)

    assert generated == {}
    result = updated_shots[0]
    assert result["status"] == FALLBACK_REQUESTED_STATUS
    assert result["failure_reason"]["type"] == FAILURE_TYPE_API_ERROR
    assert result["retry_count"] == 1  # guaranteed untouched


@pytest.mark.asyncio
async def test_budget_exceeded_hands_off_without_calling_generate_fn():
    calls = []

    async def fake_generate(**kwargs):
        calls.append(kwargs)
        return "http://oss.example.com/clip.mp4"

    allocated = (MIN_SHOT_DURATION_SEC * RATE_720P) - 0.01  # below even the cheapest floor
    shot = _shot("s1", duration_sec=5.0, allocated_budget=allocated, retry_count=0)

    updated_shots, generated = await generate_videos([shot], TRUTHS, TREATMENT, PRODUCT_PHOTOS, generate_fn=fake_generate)

    assert calls == []  # no API call made -- nothing spent on an unaffordable shot
    assert generated == {}
    result = updated_shots[0]
    assert result["status"] == FALLBACK_REQUESTED_STATUS
    assert result["failure_reason"]["type"] == FAILURE_TYPE_BUDGET_EXCEEDED
    assert result["retry_count"] == 0


@pytest.mark.asyncio
async def test_mixed_success_and_failure_across_parallel_shots():
    async def fake_generate(*, image_url, prompt, negative_prompt, duration_sec, resolution):
        if "photo_1" in image_url or image_url.endswith("photo1.jpg"):
            raise VideoGenAPIError("simulated failure for this shot only")
        return "http://oss.example.com/clip.mp4"

    ok_shot = _shot("s_ok", reference_image_id="photo_2")
    bad_shot = _shot("s_bad", reference_image_id="photo_1")

    updated_shots, generated = await generate_videos(
        [ok_shot, bad_shot], TRUTHS, TREATMENT, PRODUCT_PHOTOS, generate_fn=fake_generate
    )

    by_id = {s["shot_id"]: s for s in updated_shots}
    assert by_id["s_ok"]["status"] == SUCCESS_STATUS
    assert "s_ok" in generated
    assert by_id["s_bad"]["status"] == FALLBACK_REQUESTED_STATUS
    assert "s_bad" not in generated
    assert by_id["s_bad"]["failure_reason"]["type"] == FAILURE_TYPE_API_ERROR
    # One shot's failure never blocks/affects the other's success.
    assert by_id["s_ok"]["retry_count"] == 0
    assert by_id["s_bad"]["retry_count"] == 0


# ---------------------------------------------------------------------------
# Phase 4 retry-loop filter: only "pending" shots are (re-)sent to Wan.
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_only_pending_shots_are_dispatched_to_wan():
    sent: list[str] = []

    async def fake_generate(*, image_url, prompt, negative_prompt, duration_sec, resolution):
        sent.append(image_url)
        return "http://oss.example.com/clip.mp4"

    # A already-passed shot (from a prior pass) alongside one pending retry shot.
    passed_shot = {**_shot("s_passed"), "status": SUCCESS_STATUS}
    pending_shot = _shot("s_pending")  # _shot defaults status="pending"

    updated_shots, generated = await generate_videos(
        [passed_shot, pending_shot], TRUTHS, TREATMENT, PRODUCT_PHOTOS, generate_fn=fake_generate
    )

    # Only the pending shot hit the Wan API; the passed shot was NOT re-sent.
    assert len(sent) == 1
    assert set(generated.keys()) == {"s_pending"}
    by_id = {s["shot_id"]: s for s in updated_shots}
    # The passed shot passes through the join step completely untouched.
    assert by_id["s_passed"]["status"] == SUCCESS_STATUS
    assert by_id["s_pending"]["status"] == SUCCESS_STATUS


@pytest.mark.asyncio
async def test_node_merges_new_clips_into_existing_generated_shots(monkeypatch):
    """A retry pass must not wipe already-generated shots' entries."""
    async def fake_generate(**kwargs):
        return "http://wan.example.com/fresh.mp4"

    monkeypatch.setattr("agents.video_gen_node._call_wan_video_gen", fake_generate)
    monkeypatch.setattr(
        "agents.video_gen_node.persist_remote_video_to_oss",
        lambda url, job_id, shot_id, filename="shot.mp4", *, bucket=None, download_fn=None: f"http://oss/{shot_id}.mp4",
    )

    # State already carries s1's clip (passed on a prior pass); s2 is a pending retry.
    passed_shot = {**_shot("s1"), "status": SUCCESS_STATUS}
    pending_shot = _shot("s2")
    state = _base_state([passed_shot, pending_shot])
    state["generated_shots"] = {"s1": {"video_uri": "http://oss/s1_old.mp4", "attempt": 1, "drift_score": 0.9}}

    result = await RunnableLambda(video_gen_node).ainvoke(state)

    gen = result["generated_shots"]
    # s1's prior entry is preserved (not clobbered); s2's fresh entry is added.
    assert gen["s1"]["video_uri"] == "http://oss/s1_old.mp4"
    assert gen["s1"]["drift_score"] == 0.9
    assert gen["s2"]["video_uri"] == "http://oss/s2.mp4"


# ---------------------------------------------------------------------------
# video_gen_node wrapper: OSS persistence + shot_generated events
# ---------------------------------------------------------------------------
def _base_state(shots):
    return {
        "job_id": "job-vg",
        "product_photos": PRODUCT_PHOTOS,
        "product_truths": TRUTHS,
        "treatment": TREATMENT,
        "shot_list": shots,
        "reasoning_trace": "",
    }


@pytest.mark.asyncio
async def test_node_persists_real_clips_to_oss_and_rewrites_uri(monkeypatch):
    async def fake_generate(**kwargs):
        return "http://wan.example.com/ephemeral/clip.mp4?token=xyz"

    persisted: list[tuple] = []

    def fake_persist(remote_url, job_id, shot_id, filename="shot.mp4", *, bucket=None, download_fn=None):
        persisted.append((remote_url, job_id, shot_id))
        return f"http://oss.example.com/jobs/{job_id}/shots/{shot_id}/{filename}"

    monkeypatch.setattr("agents.video_gen_node._call_wan_video_gen", fake_generate)
    monkeypatch.setattr("agents.video_gen_node.persist_remote_video_to_oss", fake_persist)

    result = await RunnableLambda(video_gen_node).ainvoke(_base_state([_shot("s1"), _shot("s2")]))

    gen = result["generated_shots"]
    assert set(gen.keys()) == {"s1", "s2"}
    for sid in ("s1", "s2"):
        assert gen[sid]["video_uri"] == f"http://oss.example.com/jobs/job-vg/shots/{sid}/shot.mp4"
    assert {p[2] for p in persisted} == {"s1", "s2"}
    assert "persisted 2 to OSS" in result["reasoning_trace"]


@pytest.mark.asyncio
async def test_node_keeps_provider_url_when_oss_persist_fails(monkeypatch):
    async def fake_generate(**kwargs):
        return "http://wan.example.com/keep-me.mp4"

    def boom_persist(*a, **k):
        raise OSError("OSS down")

    monkeypatch.setattr("agents.video_gen_node._call_wan_video_gen", fake_generate)
    monkeypatch.setattr("agents.video_gen_node.persist_remote_video_to_oss", boom_persist)

    result = await RunnableLambda(video_gen_node).ainvoke(_base_state([_shot("s1")]))

    # Persist failure must not sink a real clip: keep the still-valid provider URL.
    assert result["generated_shots"]["s1"]["video_uri"] == "http://wan.example.com/keep-me.mp4"
    assert result["shot_list"][0]["status"] == SUCCESS_STATUS
    assert "persisted 0 to OSS" in result["reasoning_trace"]


@pytest.mark.asyncio
async def test_node_emits_shot_generated_real_events(monkeypatch):
    async def fake_generate(**kwargs):
        return "http://wan.example.com/clip.mp4"

    monkeypatch.setattr("agents.video_gen_node._call_wan_video_gen", fake_generate)
    monkeypatch.setattr(
        "agents.video_gen_node.persist_remote_video_to_oss",
        lambda url, job_id, shot_id, filename="shot.mp4", *, bucket=None, download_fn=None: f"http://oss/{shot_id}.mp4",
    )

    events = [
        e
        async for e in RunnableLambda(video_gen_node).astream_events(_base_state([_shot("s1"), _shot("s2")]), version="v2")
        if e.get("event") == "on_custom_event" and e.get("name") == "shot_generated"
    ]

    by_id = {e["data"]["shot_id"]: e["data"] for e in events}
    assert set(by_id.keys()) == {"s1", "s2"}
    assert all(d["is_fallback"] is False for d in by_id.values())
    assert all(d["status"] == SUCCESS_STATUS for d in by_id.values())


@pytest.mark.asyncio
async def test_node_does_not_emit_for_handed_off_shot(monkeypatch):
    async def fake_generate(**kwargs):
        raise VideoGenAPIError("hard failure")

    monkeypatch.setattr("agents.video_gen_node._call_wan_video_gen", fake_generate)
    monkeypatch.setattr(
        "agents.video_gen_node.persist_remote_video_to_oss",
        lambda *a, **k: "http://oss/x.mp4",
    )

    events = [
        e
        async for e in RunnableLambda(video_gen_node).astream_events(_base_state([_shot("s1")]), version="v2")
        if e.get("event") == "on_custom_event" and e.get("name") == "shot_generated"
    ]
    # The shot was handed off (fallback_requested) — no clip, so no event here.
    assert events == []
