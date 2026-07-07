"""
Unit tests for the Ken-Burns Fallback Node (§5.9).

Covers camera_move -> zoompan expression mapping, the Video-Gen hand-off contract
(fallback_requested in, fallback out, retry_count never touched), batch failure
isolation, OSS upload injection, and the LangGraph node wrapper.

No real ffmpeg, httpx, or OSS calls — render and upload are injected/mocked.
"""
from __future__ import annotations

import os
import tempfile

import pytest
from langchain_core.runnables import RunnableLambda

from agents.ken_burns_fallback_node import (
    FALLBACK_STATUS,
    FPS,
    HEIGHT,
    WIDTH,
    generate_ken_burns_fallbacks,
    ken_burns_expressions,
    ken_burns_fallback_node,
    render_ken_burns_clip,
)
from agents.video_gen_node import FALLBACK_REQUESTED_STATUS

PRODUCT_PHOTOS = ["http://example.com/photo1.jpg", "http://example.com/photo2.jpg"]


def _shot(
    shot_id: str,
    *,
    status: str = "pending",
    camera_move: str = "push_in",
    duration_sec: float = 4.0,
    reference_image_id: str = "photo_1",
    retry_count: int = 0,
) -> dict:
    return {
        "shot_id": shot_id,
        "t_start": 0.0,
        "t_end": duration_sec,
        "beat_role": "hook",
        "description": "Product detail catches the light.",
        "shot_type": "macro_detail",
        "camera_move": camera_move,
        "framing": "fills_frame",
        "lighting": "soft key light",
        "negative_prompt": "warped label",
        "reference_image_id": reference_image_id,
        "text_overlay_zone": "none",
        "duration_sec": duration_sec,
        "allocated_budget": 1.0,
        "voiceover_line": "line",
        "justification": {
            "script_quote": "a quoted line",
            "truth_fact_id": "t1",
            "treatment_ref": 0,
        },
        "status": status,
        "retry_count": retry_count,
    }


def _fallback_requested_shot(shot_id: str, **kwargs) -> dict:
    shot = _shot(shot_id, status=FALLBACK_REQUESTED_STATUS, **kwargs)
    shot["failure_reason"] = {"type": "api_error", "detail": "simulated Wan failure"}
    return shot


@pytest.fixture
def fake_render(monkeypatch, tmp_path):
    """Patch render_ken_burns_clip to write a tiny temp MP4 without ffmpeg."""

    def _fake(shot, product_photos):
        path = tmp_path / f"{shot['shot_id']}.mp4"
        path.write_bytes(b"fake-mp4")
        return str(path)

    monkeypatch.setattr("agents.ken_burns_fallback_node.render_ken_burns_clip", _fake)
    return _fake


@pytest.fixture
def fake_upload():
    def _upload(local_path: str, shot_id: str) -> str:
        return f"http://oss.example.com/fallback/{shot_id}.mp4"

    return _upload


# ---------------------------------------------------------------------------
# ken_burns_expressions (pure mapping)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "camera_move, expected_z_fragment",
    [
        ("push_in", "min(zoom+0.0015"),
        ("pull_back", "if(eq(on,1),1.3"),
        ("pan", "1.15"),
        ("tilt_up", "1.15"),
        ("orbit", "min(zoom+0.0008"),
        ("rack_focus", "min(zoom+0.0008"),
        ("static", "min(zoom+0.0004"),
        ("totally_unknown", "min(zoom+0.0004"),
    ],
)
def test_ken_burns_expressions_maps_camera_move(camera_move, expected_z_fragment):
    frames = 120
    z, x, y = ken_burns_expressions(camera_move, frames)
    assert expected_z_fragment in z
    assert isinstance(x, str) and isinstance(y, str)


def test_pan_expression_uses_frame_count():
    frames = 90
    z, x, y = ken_burns_expressions("pan", frames)
    assert z == "1.15"
    assert f"/{frames}" in x


# ---------------------------------------------------------------------------
# generate_ken_burns_fallbacks
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_non_fallback_shots_pass_through_unchanged(fake_render, fake_upload):
    original = _shot("s_ok", status="passed")
    updated, entries = await generate_ken_burns_fallbacks(
        [original], PRODUCT_PHOTOS, upload_fn=fake_upload
    )
    assert updated[0] is original
    assert entries == {}


@pytest.mark.asyncio
async def test_fallback_requested_shot_becomes_fallback_with_generated_entry(fake_render, fake_upload):
    shot = _fallback_requested_shot("s1", retry_count=2)
    updated, entries = await generate_ken_burns_fallbacks(
        [shot], PRODUCT_PHOTOS, upload_fn=fake_upload
    )

    result = updated[0]
    assert result["status"] == FALLBACK_STATUS
    assert result["retry_count"] == 2  # guaranteed untouched
    assert result["failure_reason"]["type"] == "api_error"
    assert "s1" in entries
    assert entries["s1"]["video_uri"] == "http://oss.example.com/fallback/s1.mp4"
    assert entries["s1"]["attempt"] == 1
    assert entries["s1"]["resolution_used"] == "1080P"
    assert entries["s1"]["duration_sec_used"] == shot["duration_sec"]
    assert entries["s1"]["budget_clamped"] is False


@pytest.mark.asyncio
async def test_mixed_batch_success_and_pass_through(fake_render, fake_upload):
    ok = _shot("s_ok", status="passed")
    fb = _fallback_requested_shot("s_fb")
    updated, entries = await generate_ken_burns_fallbacks(
        [ok, fb], PRODUCT_PHOTOS, upload_fn=fake_upload
    )

    by_id = {s["shot_id"]: s for s in updated}
    assert by_id["s_ok"] is ok
    assert by_id["s_fb"]["status"] == FALLBACK_STATUS
    assert set(entries.keys()) == {"s_fb"}


@pytest.mark.asyncio
async def test_render_failure_leaves_fallback_requested(fake_upload, monkeypatch):
    def _boom(shot, product_photos):
        raise RuntimeError("ffmpeg exploded")

    monkeypatch.setattr("agents.ken_burns_fallback_node.render_ken_burns_clip", _boom)
    shot = _fallback_requested_shot("s_bad", retry_count=1)
    updated, entries = await generate_ken_burns_fallbacks(
        [shot], PRODUCT_PHOTOS, upload_fn=fake_upload
    )

    assert updated[0]["status"] == FALLBACK_REQUESTED_STATUS
    assert updated[0]["retry_count"] == 1
    assert entries == {}


@pytest.mark.asyncio
async def test_upload_failure_leaves_fallback_requested(fake_render):
    def _fail_upload(local_path, shot_id):
        raise OSError("OSS unavailable")

    shot = _fallback_requested_shot("s_up")
    updated, entries = await generate_ken_burns_fallbacks(
        [shot], PRODUCT_PHOTOS, upload_fn=_fail_upload
    )

    assert updated[0]["status"] == FALLBACK_REQUESTED_STATUS
    assert entries == {}


@pytest.mark.asyncio
async def test_one_shots_failure_does_not_block_sibling(fake_render):
    calls: list[str] = []

    def _upload(local_path, shot_id):
        calls.append(shot_id)
        if shot_id == "s_bad":
            raise OSError("upload failed for s_bad only")
        return f"http://oss.example.com/{shot_id}.mp4"

    shots = [_fallback_requested_shot("s_ok"), _fallback_requested_shot("s_bad")]
    updated, entries = await generate_ken_burns_fallbacks(
        shots, PRODUCT_PHOTOS, upload_fn=_upload
    )

    by_id = {s["shot_id"]: s for s in updated}
    assert by_id["s_ok"]["status"] == FALLBACK_STATUS
    assert by_id["s_bad"]["status"] == FALLBACK_REQUESTED_STATUS
    assert set(entries.keys()) == {"s_ok"}


# ---------------------------------------------------------------------------
# render_ken_burns_clip (local path + mocked ffmpeg)
# ---------------------------------------------------------------------------
def test_render_ken_burns_clip_uses_local_image_without_download(monkeypatch, tmp_path):
    image = tmp_path / "product.jpg"
    image.write_bytes(b"fake-jpeg")
    shot = _shot("s1", reference_image_id="photo_1")

    captured: dict = {}

    def _fake_ffmpeg(image_path, out_path, duration_sec, z_expr, x_expr, y_expr):
        captured.update(
            {
                "image_path": image_path,
                "out_path": out_path,
                "duration_sec": duration_sec,
                "z": z_expr,
                "x": x_expr,
                "y": y_expr,
            }
        )
        with open(out_path, "wb") as fh:
            fh.write(b"mp4")

    monkeypatch.setattr("agents.ken_burns_fallback_node._run_ffmpeg_ken_burns", _fake_ffmpeg)
    # Force resolver to return a local path (only one photo in list).
    monkeypatch.setattr(
        "agents.ken_burns_fallback_node._resolve_reference_image_url",
        lambda ref, photos: str(image),
    )

    out = render_ken_burns_clip(shot, [str(image)])
    assert os.path.exists(out)
    assert captured["duration_sec"] == shot["duration_sec"]
    assert "min(zoom+0.0015" in captured["z"]  # push_in
    os.remove(out)


def test_render_ken_burns_clip_raises_when_no_reference_photo():
    shot = _shot("s1", reference_image_id="photo_99")
    with pytest.raises(ValueError, match="no reference photo"):
        render_ken_burns_clip(shot, [])


def test_render_ken_burns_clip_downloads_remote_image(monkeypatch, tmp_path):
    shot = _shot("s1")
    downloaded = tmp_path / "downloaded.jpg"
    downloaded.write_bytes(b"img")

    class _Resp:
        content = b"img"
        def raise_for_status(self):
            return None

    monkeypatch.setattr("agents.ken_burns_fallback_node.httpx.get", lambda *a, **k: _Resp())
    monkeypatch.setattr(
        "agents.ken_burns_fallback_node._download_image_to_temp",
        lambda url: str(downloaded),
    )
    monkeypatch.setattr(
        "agents.ken_burns_fallback_node._run_ffmpeg_ken_burns",
        lambda *a, **k: open(a[1], "wb").write(b"mp4"),
    )

    out = render_ken_burns_clip(shot, PRODUCT_PHOTOS)
    assert out.endswith(".mp4")
    os.remove(out)


# ---------------------------------------------------------------------------
# ken_burns_fallback_node wrapper
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_node_wrapper_merges_generated_shots_and_updates_trace(fake_render, monkeypatch):
    def _upload(local_path, job_id, shot_id, filename="fallback_kenburns.mp4", *, bucket=None):
        return f"http://oss.example.com/jobs/{job_id}/shots/{shot_id}/clip.mp4"

    monkeypatch.setattr("agents.ken_burns_fallback_node.upload_video_to_oss", _upload)

    state = {
        "job_id": "job-99",
        "product_photos": PRODUCT_PHOTOS,
        "shot_list": [
            _shot("s_ok", status="passed"),
            _fallback_requested_shot("s_fb"),
        ],
        "generated_shots": {
            "s_ok": {"video_uri": "http://oss.example.com/wan/s_ok.mp4", "attempt": 1},
        },
        "reasoning_trace": "prior",
    }

    # RunnableLambda provides the LangChain run context adispatch_custom_event
    # needs (same precedent as test_budget_gate.py's node-wrapper tests).
    result = await RunnableLambda(ken_burns_fallback_node).ainvoke(state)

    assert result["shot_list"][1]["status"] == FALLBACK_STATUS
    assert result["generated_shots"]["s_ok"]["video_uri"].endswith("s_ok.mp4")
    assert result["generated_shots"]["s_fb"]["video_uri"] == "http://oss.example.com/jobs/job-99/shots/s_fb/clip.mp4"
    assert "[ken_burns_fallback] rendered 1 Ken-Burns fallback clip(s)." in result["reasoning_trace"]


@pytest.mark.asyncio
async def test_node_wrapper_emits_shot_generated_for_rendered_fallback_only(fake_render, monkeypatch):
    def _upload(local_path, job_id, shot_id, filename="fallback_kenburns.mp4", *, bucket=None):
        return f"http://oss.example.com/jobs/{job_id}/shots/{shot_id}/clip.mp4"

    monkeypatch.setattr("agents.ken_burns_fallback_node.upload_video_to_oss", _upload)

    state = {
        "job_id": "job-evt",
        "product_photos": PRODUCT_PHOTOS,
        "shot_list": [
            _shot("s_ok", status="passed"),  # already a real clip, not ours to emit
            _fallback_requested_shot("s_fb"),
        ],
        "generated_shots": {"s_ok": {"video_uri": "http://oss/x.mp4", "attempt": 1}},
        "reasoning_trace": "",
    }

    events = [
        e
        async for e in RunnableLambda(ken_burns_fallback_node).astream_events(state, version="v2")
        if e.get("event") == "on_custom_event" and e.get("name") == "shot_generated"
    ]

    assert len(events) == 1
    payload = events[0]["data"]
    assert payload["shot_id"] == "s_fb"
    assert payload["is_fallback"] is True
    assert payload["status"] == FALLBACK_STATUS
    assert payload["generated"]["video_uri"].endswith("/shots/s_fb/clip.mp4")


@pytest.mark.asyncio
async def test_node_wrapper_emits_nothing_when_render_fails(fake_upload, monkeypatch):
    def _boom(shot, product_photos):
        raise RuntimeError("ffmpeg exploded")

    monkeypatch.setattr("agents.ken_burns_fallback_node.render_ken_burns_clip", _boom)
    state = {
        "job_id": "job-fail",
        "product_photos": PRODUCT_PHOTOS,
        "shot_list": [_fallback_requested_shot("s_fb")],
        "generated_shots": {},
        "reasoning_trace": "",
    }

    events = [
        e
        async for e in RunnableLambda(ken_burns_fallback_node).astream_events(state, version="v2")
        if e.get("event") == "on_custom_event" and e.get("name") == "shot_generated"
    ]
    assert events == []


def test_output_spec_constants_match_wan_target():
    assert FPS == 30
    assert WIDTH == 1920
    assert HEIGHT == 1080
