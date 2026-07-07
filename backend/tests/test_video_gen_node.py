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
    SUCCESS_STATUS,
    VideoGenAPIError,
    VideoGenTimeoutError,
    _build_prompt,
    _resolve_generation_params,
    _resolve_reference_image_url,
    generate_videos,
    video_gen_node,
)

TRUTHS = [
    {"truth_id": "t1", "fact": "double-wall stainless seam", "category": "construction_detail", "source": "photo_1"},
    {"truth_id": "t2", "fact": "matte black finish", "category": "color", "source": "photo_1"},
]

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
    assert "anatomically correct fingers" in prompt
    assert "no scene cut" in prompt


def test_prompt_omits_hand_continuity_clause_for_non_human_shot_types():
    shot = _shot("s1", shot_type="macro_detail")
    prompt = _build_prompt(shot, TRUTHS, TREATMENT)
    assert "anatomically correct fingers" not in prompt


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
