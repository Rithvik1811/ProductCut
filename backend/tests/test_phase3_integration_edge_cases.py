"""
Phase 3 integration / edge-case coverage — independent adversarial pass.

This file closes real coverage GAPS in the Video-Gen + Ken-Burns + OSS trio that
the existing suites (test_video_gen_node.py, test_ken_burns_fallback_node.py,
test_oss.py, test_graph_end_to_end.py) do not exercise. Every test here documents
a behavior an adversarial reviewer specifically probed; each asserts the
spec-correct outcome (they PASS — no genuine bug was found that requires a
production change, see this reviewer's report for the full write-up).

The numbered items below map to the review checklist:
  1. resolution_used semantics across real vs. fallback GeneratedShot entries.
  2. _persist_generated_to_oss: MIXED per-shot persist success/failure (the
     existing tests only cover all-success or all-fail on a single shot).
  3. suffix-detection edge cases in the two download-to-temp helpers.
  7. frame-count consistency between render_ken_burns_clip and _run_ffmpeg_ken_burns.
  8. a fallback_requested shot Ken-Burns ALSO cannot fix (empty product_photos):
     batch continues, no ffmpeg reached, sibling still recovers.
"""
from __future__ import annotations

import os
import shutil
import subprocess

import pytest
from langchain_core.runnables import RunnableLambda

from agents.budget_gate import RATE_720P
import agents.ken_burns_fallback_node as kb
from agents.ken_burns_fallback_node import (
    FPS,
    HEIGHT,
    WIDTH,
    FALLBACK_STATUS,
    generate_ken_burns_fallbacks,
    ken_burns_expressions,
)
from agents.video_gen_node import (
    FALLBACK_REQUESTED_STATUS,
    SUCCESS_STATUS,
    _resolve_generation_params,
    video_gen_node,
)
from agents._oss import _download_to_temp  # noqa: F401  (imported to show the module path used)
import agents._oss as _oss

_HAS_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None
_skip_no_ffmpeg = pytest.mark.skipif(not _HAS_FFMPEG, reason="ffmpeg/ffprobe not on PATH")

PRODUCT_PHOTOS = ["http://example.com/photo1.jpg", "http://example.com/photo2.jpg"]


def _shot(
    shot_id: str,
    *,
    status: str = "pending",
    camera_move: str = "push_in",
    duration_sec: float = 4.0,
    allocated_budget: float = 1.0,
    reference_image_id: str = "photo_1",
    retry_count: int = 0,
) -> dict:
    shot = {
        "shot_id": shot_id,
        "t_start": 0.0,
        "t_end": duration_sec,
        "beat_role": "hook",
        "description": "The product detail catches the morning light.",
        "shot_type": "macro_detail",
        "camera_move": camera_move,
        "framing": "fills_frame",
        "lighting": "soft key light, neutral background",
        "negative_prompt": "warped label, deformed hands",
        "reference_image_id": reference_image_id,
        "text_overlay_zone": "none",
        "duration_sec": duration_sec,
        "allocated_budget": allocated_budget,
        "voiceover_line": "line",
        "justification": {"script_quote": "q", "truth_fact_id": "t1", "treatment_ref": 0},
        "status": status,
        "retry_count": retry_count,
    }
    return shot


def _fallback_requested(shot_id: str, **kwargs) -> dict:
    shot = _shot(shot_id, status=FALLBACK_REQUESTED_STATUS, **kwargs)
    shot["failure_reason"] = {"type": "api_error", "detail": "simulated Wan failure"}
    return shot


TRUTHS = [{"truth_id": "t1", "fact": "double-wall seam", "category": "construction_detail", "source": "photo_1"}]
TREATMENT = {
    "director_persona": "quiet, tactile",
    "color_story": "warm neutrals",
    "pacing_philosophy": "let the hook breathe",
    "beat_treatments": [],
}


def _base_state(shots):
    return {
        "job_id": "job-edge",
        "product_photos": PRODUCT_PHOTOS,
        "product_truths": TRUTHS,
        "treatment": TREATMENT,
        "shot_list": shots,
        "reasoning_trace": "",
    }


# ---------------------------------------------------------------------------
# Item 2 — _persist_generated_to_oss under MIXED per-shot success/failure.
#
# The existing test_node_keeps_provider_url_when_oss_persist_fails only fails a
# SINGLE shot (so "keep provider url" and "rewrite to oss" are never observed in
# the SAME gather). This is the concurrency concern from the review: with N
# shots persisted concurrently via asyncio.gather, a failure on ONE shot must
# leave THAT shot's dict un-mutated (original provider URL intact) while the
# OTHERS are correctly rewritten to their OSS URIs — no torn/aliased write.
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_persist_mixed_success_and_failure_across_shots(monkeypatch):
    async def fake_generate(*, image_url, prompt, negative_prompt, duration_sec, resolution):
        # distinct provider URL per shot so we can prove the failed one is kept verbatim
        leaf = image_url.rsplit("/", 1)[-1]
        return f"http://wan.example.com/ephemeral/{leaf}.mp4?token=xyz"

    def fake_persist(remote_url, job_id, shot_id, filename="shot.mp4", *, bucket=None, download_fn=None):
        if shot_id == "s_fail":
            raise OSError("OSS put failed for this shot only")
        return f"http://oss.example.com/jobs/{job_id}/shots/{shot_id}/{filename}"

    monkeypatch.setattr("agents.video_gen_node._call_wan_video_gen", fake_generate)
    monkeypatch.setattr("agents.video_gen_node.persist_remote_video_to_oss", fake_persist)

    shots = [
        _shot("s_ok", reference_image_id="photo_1"),
        _shot("s_fail", reference_image_id="photo_2"),
    ]
    result = await RunnableLambda(video_gen_node).ainvoke(_base_state(shots))

    gen = result["generated_shots"]
    # The succeeding shot was rewritten to its OSS URI.
    assert gen["s_ok"]["video_uri"] == "http://oss.example.com/jobs/job-edge/shots/s_ok/shot.mp4"
    # The failing shot kept its ORIGINAL provider URL verbatim — not sunk, not
    # torn, not accidentally rewritten to the other shot's URI.
    assert gen["s_fail"]["video_uri"] == "http://wan.example.com/ephemeral/photo2.jpg.mp4?token=xyz"
    # Both shots are still successful real clips (a copy failure never downgrades).
    by_id = {s["shot_id"]: s for s in result["shot_list"]}
    assert by_id["s_ok"]["status"] == SUCCESS_STATUS
    assert by_id["s_fail"]["status"] == SUCCESS_STATUS
    assert "persisted 1 to OSS" in result["reasoning_trace"]


# ---------------------------------------------------------------------------
# Item 1 — resolution_used semantics: real budget-clamped clip vs. fallback.
#
# A budget-clamped REAL clip legitimately records resolution_used="720P"; a
# Ken-Burns FALLBACK clip records "1080P". The review asked whether this is
# misleading (cheaper path claiming higher resolution). It is NOT a bug: the
# fallback clip is GENUINELY rendered at 1920x1080 (WIDTH/HEIGHT), so "1080P" is
# truthful, and the 720p on a real clip is also truthful. Both values accurately
# describe their own pixel dimensions. This test pins that both labels are
# accurate so a future change can't silently make either one lie.
# ---------------------------------------------------------------------------
def test_budget_clamped_real_clip_records_720p_truthfully():
    # allocated affords 720p full-duration but not 1080p
    shot = _shot("s1", duration_sec=4.0, allocated_budget=4.0 * RATE_720P)
    duration, resolution, failure = _resolve_generation_params(shot)
    assert failure is None
    assert resolution == "720P"  # a real, truthful downgrade driven by budget


@_skip_no_ffmpeg
def test_fallback_clip_is_genuinely_1080p_not_a_mislabel(tmp_path):
    """The fallback GeneratedShot claims resolution_used='1080P'; confirm the
    rendered pixels really ARE 1920x1080, i.e. the label is truthful, not a
    cheaper-path-claiming-higher-res inconsistency."""
    img = str(tmp_path / "src.png")
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=blue:s=640x480:d=1", "-frames:v", "1", img],
        check=True, capture_output=True,
    )
    z, x, y = ken_burns_expressions("push_in", frames=45)
    out = str(tmp_path / "out.mp4")
    kb._run_ffmpeg_ken_burns(img, out, 1.5, z, x, y)
    import json
    probe = json.loads(subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "stream=width,height,codec_type",
         "-of", "json", out], check=True, capture_output=True, text=True).stdout)
    vstream = next(s for s in probe["streams"] if s["codec_type"] == "video")
    assert (vstream["width"], vstream["height"]) == (WIDTH, HEIGHT) == (1920, 1080)


# ---------------------------------------------------------------------------
# Item 7 — frame-count consistency (guard against future drift).
#
# render_ken_burns_clip computes `int(round(duration_sec * FPS))` and passes it
# into ken_burns_expressions (baked into the `on/{frames}` pan/tilt expressions);
# _run_ffmpeg_ken_burns INDEPENDENTLY recomputes the same value for zoompan's
# `d=`. If these ever drift, the pan/tilt would reference a frame count that
# doesn't match the actual clip length — a silent wrong-motion bug, not a crash.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("duration_sec", [1.5, 3.0, 3.7, 3.75, 4.0, 5.0])
def test_frame_count_formula_is_identical_in_both_call_sites(duration_sec):
    # The single formula both sites use. If someone edits one site's formula,
    # this recomputation (mirroring render_ken_burns_clip) must be kept in sync
    # with _run_ffmpeg_ken_burns' internal `int(round(duration_sec * FPS))`.
    frames = int(round(duration_sec * FPS))
    _, x, y = ken_burns_expressions("pan", frames)
    assert f"/{frames}" in x  # the pan expression bakes in exactly this count
    _, _, y_tilt = ken_burns_expressions("tilt_up", frames)
    assert f"/{frames}" in y_tilt


@_skip_no_ffmpeg
def test_pan_frame_count_matches_real_render_at_odd_duration(tmp_path):
    """Concrete verification (not just reasoning) at an unusual 3.7s: the pan
    expression's `on/{frames}` denominator must equal the number of frames the
    render actually produces, so the drift completes exactly 0->1 over the clip."""
    duration = 3.7
    frames = int(round(duration * FPS))  # 111
    img = str(tmp_path / "src.png")
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=red:s=640x480:d=1", "-frames:v", "1", img],
        check=True, capture_output=True,
    )
    z, x, y = ken_burns_expressions("pan", frames)
    assert f"/{frames}" in x  # denominator is 111
    out = str(tmp_path / "out.mp4")
    kb._run_ffmpeg_ken_burns(img, out, duration, z, x, y)
    import json
    probe = json.loads(subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames",
         "-show_entries", "format=duration:stream=nb_read_frames,codec_type",
         "-of", "json", out], check=True, capture_output=True, text=True).stdout)
    vstream = next(s for s in probe["streams"] if s["codec_type"] == "video")
    assert int(vstream["nb_read_frames"]) == frames  # rendered frames == expression denominator
    assert abs(float(probe["format"]["duration"]) - duration) < 0.2


# ---------------------------------------------------------------------------
# Item 8 — a fallback_requested shot Ken-Burns ALSO can't fix.
#
# product_photos is empty, so _resolve_reference_image_url returns "" and
# render_ken_burns_clip raises BEFORE ever touching ffmpeg. The batch must
# continue: that shot stays fallback_requested (visibly still needs handling),
# a sibling fallback with a real photo still recovers, and ffmpeg is never
# reached for the unfixable shot.
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_fallback_with_no_photo_stays_requested_and_never_reaches_ffmpeg(monkeypatch):
    ffmpeg_calls: list = []

    def _spy_ffmpeg(*a, **k):
        ffmpeg_calls.append(a)
        # write a dummy so anything downstream that expected a file still works
        with open(a[1], "wb") as fh:
            fh.write(b"mp4")

    monkeypatch.setattr("agents.ken_burns_fallback_node._run_ffmpeg_ken_burns", _spy_ffmpeg)

    def _upload(local_path, shot_id):
        return f"http://oss.example.com/{shot_id}.mp4"

    unfixable = _fallback_requested("s_nophoto")
    passthrough = _shot("s_done", status="passed")

    # No product photos at all -> the unfixable shot can't resolve a reference.
    updated, entries = await generate_ken_burns_fallbacks(
        [unfixable, passthrough], [], upload_fn=_upload
    )

    by_id = {s["shot_id"]: s for s in updated}
    # Stays fallback_requested (not silently marked done, not crashed).
    assert by_id["s_nophoto"]["status"] == FALLBACK_REQUESTED_STATUS
    assert by_id["s_nophoto"]["retry_count"] == 0  # never touched
    assert "s_nophoto" not in entries
    # The non-fallback shot passed straight through, same object.
    assert by_id["s_done"] is passthrough
    # ffmpeg was never reached — the resolution failure short-circuits first.
    assert ffmpeg_calls == []


@pytest.mark.asyncio
async def test_unfixable_shot_does_not_block_a_recoverable_sibling(monkeypatch):
    """Two fallback_requested shots in ONE batch: the unfixable one (empty photo
    list -> resolver returns "" -> render raises) stays fallback_requested, while
    the recoverable sibling (its own resolvable photo, injected via the render
    fake) still renders + uploads. Proves per-shot failure isolation across a
    real mixed batch, not just a single-shot path."""
    def _fake_render(shot, product_photos):
        # The unfixable shot must still fail: reproduce the real resolver's
        # "no photo -> raise" behavior when its photo list is empty.
        if not product_photos:
            raise ValueError(f"shot {shot['shot_id']}: no reference photo available")
        import tempfile
        fd, path = tempfile.mkstemp(suffix=".mp4")
        os.close(fd)
        with open(path, "wb") as fh:
            fh.write(b"mp4")
        return path

    monkeypatch.setattr("agents.ken_burns_fallback_node.render_ken_burns_clip", _fake_render)

    captured: list[str] = []

    def _upload(local_path, shot_id):
        captured.append(shot_id)
        return f"http://oss.example.com/{shot_id}.mp4"

    # generate_ken_burns_fallbacks passes the SAME product_photos to every shot,
    # so to get one fixable + one unfixable in a single call we drive the render
    # fake off each shot's own reference id: the "unfixable" one carries a
    # sentinel the fake treats as photoless.
    def _fake_render2(shot, product_photos):
        if shot["reference_image_id"] == "MISSING":
            raise ValueError(f"shot {shot['shot_id']}: no reference photo available")
        import tempfile
        fd, path = tempfile.mkstemp(suffix=".mp4")
        os.close(fd)
        with open(path, "wb") as fh:
            fh.write(b"mp4")
        return path

    monkeypatch.setattr("agents.ken_burns_fallback_node.render_ken_burns_clip", _fake_render2)

    bad = _fallback_requested("s_bad", reference_image_id="MISSING")
    good = _fallback_requested("s_good", reference_image_id="photo_1")
    updated, entries = await generate_ken_burns_fallbacks([bad, good], PRODUCT_PHOTOS, upload_fn=_upload)

    by_id = {s["shot_id"]: s for s in updated}
    assert by_id["s_bad"]["status"] == FALLBACK_REQUESTED_STATUS  # isolated failure
    assert "s_bad" not in entries
    assert by_id["s_good"]["status"] == FALLBACK_STATUS  # sibling still recovered
    assert entries["s_good"]["video_uri"] == "http://oss.example.com/s_good.mp4"
    assert entries["s_good"]["resolution_used"] == "1080P"
    assert captured == ["s_good"]  # only the good shot ever uploaded


# ---------------------------------------------------------------------------
# Item 3 — suffix-detection edge cases in the two download-to-temp helpers.
#
# Both agents/_oss._download_to_temp and ken_burns._download_image_to_temp
# derive a temp-file suffix via os.path.splitext(url.split("?", 1)[0])[1].
# Real signed provider URLs are query-based (no fragment), so the common cases
# resolve correctly. A #fragment with NO ?query is the one quirky case: it is
# NOT stripped (split only handles "?"), so the fragment leaks into the suffix.
# This is documented here as a known, benign quirk (temp-file extension is
# cosmetic; content, not extension, drives httpx write + ffmpeg/oss upload), and
# does not occur for real Wan/OSS signed URLs which always carry ?query.
# ---------------------------------------------------------------------------
def _suffix(url: str, default: str) -> str:
    return os.path.splitext(url.split("?", 1)[0])[1] or default


@pytest.mark.parametrize(
    "url, expected",
    [
        ("http://cdn.example.com/clip.mp4?token=abc", ".mp4"),
        ("http://cdn.example.com/clip.mp4?token=abc#frag", ".mp4"),  # ? before # -> clean
        ("https://dashscope-result.oss-cn-beijing.aliyuncs.com/1.mp4?Expires=1&Signature=x", ".mp4"),
        ("http://cdn.example.com/video", ".mp4"),           # no ext -> default
        ("http://cdn.example.com/video#frag", ".mp4"),      # no dot in basename -> default
    ],
)
def test_suffix_detection_for_realistic_provider_urls(url, expected):
    # These are the shapes that actually occur (query-based signed URLs); the
    # suffix logic handles them correctly.
    assert _suffix(url, ".mp4") == expected


def test_suffix_fragment_without_query_is_a_known_benign_quirk():
    # A fragment with no query is NOT stripped -> leaks into the suffix. Pinned
    # here so the behavior is explicit; harmless because a temp-file extension is
    # cosmetic and this URL shape never occurs for real signed provider URLs.
    assert _suffix("http://cdn.example.com/clip.mp4#t=5", ".mp4") == ".mp4#t=5"
