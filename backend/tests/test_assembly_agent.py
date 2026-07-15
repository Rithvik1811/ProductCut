"""
Unit tests for the Assembly Agent (§5.12).

Covers: pure logic (beat-to-shot mapping, canvas selection, the beat/shot
mismatch hold-frame policy, the trim/stretch/freeze-pad duration-conform
decision including the 15% boundary, caption zone/positioning) with no
ffmpeg/network; the orchestration (`_assemble_master_cut_impl`) with
injected fakes and the real render functions monkeypatched out (mirrors
`test_ken_burns_fallback_node.py`'s posture: fake the render step for fast
tests, exercise it for real separately); and a REAL-ffmpeg execution suite
(gated on ffmpeg/ffprobe on PATH, same `_skip_no_ffmpeg` convention as
`test_ken_burns_fallback_node.py` / `test_voiceover_caption_agent.py`) that
runs the actual two-stage pipeline against real synthetic lavfi clips at
mixed resolution/orientation and mixed duration-vs-target cases (forcing a
real trim, a real stretch, and a real freeze-pad in one batch), verifying the
output via ffprobe and a burned-caption pixel-variance check.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess

import pytest

from agents.assembly_agent import (
    AUDIO_FADE_SEC,
    STRETCH_MAX_RATIO,
    VIDEO_FADE_SEC,
    AssemblyError,
    AssemblyResult,
    _caption_position_expr,
    _captions_for_render,
    _effective_zone,
    _map_shots_by_beat,
    _plan_segments,
    _render_stage1_segment,
    _resolve_duration_conform,
    _select_canvas,
    _wrap_caption_text,
    assemble_master_cut,
    _assemble_master_cut_impl,
)

_HAS_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None
_skip_no_ffmpeg = pytest.mark.skipif(not _HAS_FFMPEG, reason="ffmpeg/ffprobe not on PATH")


# ---------------------------------------------------------------------------
# Fixtures / helpers.
# ---------------------------------------------------------------------------
def _shot(shot_id: str, treatment_ref: int, *, status: str = "passed", zone: str = "none") -> dict:
    return {
        "shot_id": shot_id,
        "status": status,
        "text_overlay_zone": zone,
        "justification": {"script_quote": "q", "truth_fact_id": "t1", "treatment_ref": treatment_ref},
    }


def _beats(n: int, beat_len: float = 3.0) -> list[dict]:
    return [{"t_start": i * beat_len, "t_end": (i + 1) * beat_len, "line": f"line {i}"} for i in range(n)]


def _captions(n: int, beat_len: float = 3.0) -> list[dict]:
    return [
        {"text": f"line {i}", "start_ts": round(i * beat_len, 3), "end_ts": round((i + 1) * beat_len, 3)}
        for i in range(n)
    ]


def _winning_script(n: int, beat_len: float = 3.0) -> dict:
    return {"text": " ".join(f"line {i}" for i in range(n)), "beats": _beats(n, beat_len), "source_variant_ids": ["v1"]}


# ---------------------------------------------------------------------------
# _map_shots_by_beat
# ---------------------------------------------------------------------------
def test_map_shots_by_beat_uses_treatment_ref_not_position():
    # Deliberately out-of-order shot_list -- treatment_ref, not list position, wins.
    shots = [_shot("s2", 1), _shot("s1", 0), _shot("s3", 2)]
    mapping = _map_shots_by_beat(shots)
    assert mapping[0]["shot_id"] == "s1"
    assert mapping[1]["shot_id"] == "s2"
    assert mapping[2]["shot_id"] == "s3"


def test_map_shots_by_beat_ignores_missing_treatment_ref():
    shots = [_shot("s1", 0), {"shot_id": "s_bad", "justification": {}}]
    mapping = _map_shots_by_beat(shots)
    assert set(mapping.keys()) == {0}


def test_map_shots_by_beat_keeps_first_on_duplicate_treatment_ref():
    shots = [_shot("s1", 0), _shot("s2", 0)]
    mapping = _map_shots_by_beat(shots)
    assert mapping[0]["shot_id"] == "s1"


# ---------------------------------------------------------------------------
# _select_canvas
# ---------------------------------------------------------------------------
def test_select_canvas_picks_largest_area_passed_clip():
    shots_by_beat = {
        0: _shot("s1", 0, status="passed"),
        1: _shot("s2", 1, status="passed"),
        2: _shot("s3", 2, status="fallback"),  # excluded even though it's the largest probe
    }
    probes = {0: {"width": 720, "height": 1280}, 1: {"width": 1920, "height": 1080}, 2: {"width": 3840, "height": 2160}}
    assert _select_canvas(shots_by_beat, probes) == (1920, 1080)


def test_select_canvas_rounds_dimensions_down_to_even():
    shots_by_beat = {0: _shot("s1", 0, status="passed")}
    probes = {0: {"width": 721, "height": 1281}}
    assert _select_canvas(shots_by_beat, probes) == (720, 1280)


def test_select_canvas_defaults_when_only_fallback_clips_present():
    shots_by_beat = {0: _shot("s1", 0, status="fallback")}
    probes = {0: {"width": 1920, "height": 1080}}
    assert _select_canvas(shots_by_beat, probes) == (1920, 1080)


def test_select_canvas_defaults_when_nothing_usable():
    assert _select_canvas({}, {}) == (1920, 1080)


# ---------------------------------------------------------------------------
# _resolve_duration_conform -- trim / stretch / freeze, including the 15%
# stretch-ceiling boundary.
# ---------------------------------------------------------------------------
def test_conform_trims_when_actual_meets_or_exceeds_target():
    assert _resolve_duration_conform(target=3.0, actual=3.5).mode == "trim"
    assert _resolve_duration_conform(target=3.0, actual=3.0).mode == "trim"


def test_conform_stretches_within_15_percent_deficit():
    # deficit/actual = 0.6/4.0 = 0.15 exactly -> boundary is inclusive ("<=").
    plan = _resolve_duration_conform(target=4.6, actual=4.0)
    assert plan.mode == "stretch"
    assert plan.freeze_pad_sec == 0.0


def test_conform_freezes_just_past_the_15_percent_boundary():
    # deficit/actual = 0.61/4.0 = 0.1525 > 0.15 -> freeze, not stretch.
    plan = _resolve_duration_conform(target=4.61, actual=4.0)
    assert plan.mode == "freeze"
    assert plan.freeze_pad_sec == pytest.approx(0.61)


def test_conform_freeze_pad_amount_is_exact_deficit():
    plan = _resolve_duration_conform(target=5.0, actual=2.0)
    assert plan.mode == "freeze"
    assert plan.freeze_pad_sec == pytest.approx(3.0)


def test_conform_handles_near_zero_actual_duration_without_crashing():
    plan = _resolve_duration_conform(target=3.0, actual=0.0)
    assert plan.mode == "freeze"
    assert plan.freeze_pad_sec == pytest.approx(3.0)


def test_stretch_max_ratio_constant_is_15_percent():
    assert STRETCH_MAX_RATIO == 0.15


# ---------------------------------------------------------------------------
# video-gen-fidelity PHASE 3: the LAST segment's own conform NEVER hard-trims
# -- an overrun keeps the clip's full natural length ("keep"); a shortfall
# always freezes, never stretches.
# ---------------------------------------------------------------------------
def test_last_segment_overrun_keeps_full_length_instead_of_trimming():
    plan = _resolve_duration_conform(target=3.0, actual=4.0, is_last_segment=True)
    assert plan.mode == "keep"
    assert plan.freeze_pad_sec == 0.0


def test_last_segment_exact_match_keeps_not_trims():
    plan = _resolve_duration_conform(target=3.0, actual=3.0, is_last_segment=True)
    assert plan.mode == "keep"


def test_last_segment_shortfall_always_freezes_never_stretches():
    """A deficit well within the 15% stretch ceiling would normally stretch
    for a non-last segment -- for the last segment it must freeze instead."""
    plan = _resolve_duration_conform(target=3.0, actual=2.9, is_last_segment=True)  # ~3.4% deficit
    assert plan.mode == "freeze"
    assert plan.freeze_pad_sec == pytest.approx(0.1)


def test_last_segment_near_zero_actual_still_freezes_defensively():
    plan = _resolve_duration_conform(target=3.0, actual=0.0, is_last_segment=True)
    assert plan.mode == "freeze"
    assert plan.freeze_pad_sec == pytest.approx(3.0)


def test_non_last_segment_behavior_is_unchanged_regression():
    """Regression: is_last_segment defaults to False, and explicit False
    behaves identically to the original trim/stretch/freeze logic."""
    assert _resolve_duration_conform(target=3.0, actual=3.5, is_last_segment=False).mode == "trim"
    assert _resolve_duration_conform(target=4.6, actual=4.0, is_last_segment=False).mode == "stretch"
    assert _resolve_duration_conform(target=4.61, actual=4.0, is_last_segment=False).mode == "freeze"


# ---------------------------------------------------------------------------
# _plan_segments -- the beat/shot MISMATCH POLICY (hold a neighboring shot's
# frame across an orphaned beat's real VO window; never drop VO audio/captions).
# ---------------------------------------------------------------------------
def test_plan_segments_clean_1to1_has_no_padding():
    beats, caps = _beats(3), _captions(3)
    shots_by_beat = {0: _shot("s1", 0), 1: _shot("s2", 1), 2: _shot("s3", 2)}
    segs = _plan_segments(beats, caps, shots_by_beat)
    assert [s.beat_index for s in segs] == [0, 1, 2]
    assert all(s.pad_before == 0.0 and s.pad_after == 0.0 for s in segs)
    assert all(s.target == 3.0 for s in segs)


def test_plan_segments_trailing_gap_holds_previous_shots_last_frame():
    """Real-world shape (confirmed against derisk/outputs/full_pipeline_live_result.json:
    a real live pipeline run produced a 4-beat winning_script with only 3 shots,
    the CTA beat (index 3) orphaned) -- the preceding shot (index 2) absorbs the
    orphaned beat's real VO duration as a trailing hold-pad."""
    beats, caps = _beats(4), _captions(4)
    shots_by_beat = {0: _shot("s1", 0), 1: _shot("s2", 1), 2: _shot("s3", 2)}  # beat 3 orphaned
    segs = _plan_segments(beats, caps, shots_by_beat)
    assert [s.beat_index for s in segs] == [0, 1, 2]
    last = segs[-1]
    assert last.beat_index == 2
    assert last.pad_before == 0.0
    assert last.pad_after == pytest.approx(3.0)  # beat 3's own 3s VO window
    assert last.target == pytest.approx(3.0)  # beat 2's own target is unaffected


def test_plan_segments_leading_gap_holds_next_shots_first_frame():
    beats, caps = _beats(4), _captions(4)
    shots_by_beat = {1: _shot("s2", 1), 2: _shot("s3", 2), 3: _shot("s4", 3)}  # beat 0 orphaned
    segs = _plan_segments(beats, caps, shots_by_beat)
    assert [s.beat_index for s in segs] == [1, 2, 3]
    first = segs[0]
    assert first.beat_index == 1
    assert first.pad_before == pytest.approx(3.0)
    assert first.pad_after == 0.0


def test_plan_segments_multi_beat_gap_run_folds_into_one_hold():
    """Two CONSECUTIVE orphaned beats (1 and 2) both fold into shot 0's trailing pad,
    as a single combined hold -- not two separate holds."""
    beats, caps = _beats(4), _captions(4)
    shots_by_beat = {0: _shot("s1", 0), 3: _shot("s4", 3)}
    segs = _plan_segments(beats, caps, shots_by_beat)
    assert [s.beat_index for s in segs] == [0, 3]
    assert segs[0].pad_after == pytest.approx(6.0)  # beats 1 + 2's combined 3s+3s
    assert segs[1].pad_before == 0.0


def test_plan_segments_no_usable_shot_anywhere_returns_empty():
    beats, caps = _beats(3), _captions(3)
    assert _plan_segments(beats, caps, {}) == []


def test_plan_segments_mismatched_beats_and_captions_lengths_uses_shorter_without_crashing():
    beats = _beats(4)
    caps = _captions(3)  # one short -- should never happen, but must not crash
    shots_by_beat = {0: _shot("s1", 0), 1: _shot("s2", 1), 2: _shot("s3", 2)}
    segs = _plan_segments(beats, caps, shots_by_beat)
    assert [s.beat_index for s in segs] == [0, 1, 2]


# ---------------------------------------------------------------------------
# Caption zone / positioning / wrapping.
# ---------------------------------------------------------------------------
def test_effective_zone_falls_back_to_lower_third_for_none_and_missing_shot():
    assert _effective_zone(None) == "lower_third"
    assert _effective_zone(_shot("s1", 0, zone="none")) == "lower_third"
    assert _effective_zone(_shot("s1", 0, zone="left_third")) == "left_third"


def test_caption_position_expr_for_each_zone():
    assert _caption_position_expr("left_third") == ("w*0.06", "(h-text_h)/2")
    assert _caption_position_expr("right_third") == ("w*0.94-text_w", "(h-text_h)/2")
    assert _caption_position_expr("lower_third") == ("(w-text_w)/2", "h*0.80-text_h")
    assert _caption_position_expr("none") == ("(w-text_w)/2", "h*0.80-text_h")  # falls back


def test_wrap_caption_text_wraps_long_lines():
    text = "one two three four five six seven eight nine ten eleven twelve"
    wrapped = _wrap_caption_text(text, "lower_third")
    assert "\n" in wrapped
    for line in wrapped.split("\n"):
        assert len(line) <= 22 + 10  # textwrap.fill breaks on words, allow slack for one long word


def test_wrap_caption_text_never_empty_for_blank_line():
    assert _wrap_caption_text("", "lower_third").strip() != "" or _wrap_caption_text("", "lower_third") == " "


def test_captions_for_render_carries_zone_and_falls_back_for_orphaned_beat():
    caps = _captions(2)
    shots_by_beat = {0: _shot("s1", 0, zone="right_third")}  # beat 1 has no shot
    rendered = _captions_for_render(caps, shots_by_beat)
    assert rendered[0]["zone"] == "right_third"
    assert rendered[1]["zone"] == "lower_third"  # orphaned beat still gets a caption + a zone


# ---------------------------------------------------------------------------
# _assemble_master_cut_impl orchestration -- fake I/O boundaries, real renders
# monkeypatched out (mirrors test_ken_burns_fallback_node.py's fast-test posture).
# ---------------------------------------------------------------------------
@pytest.fixture
def fake_renders(monkeypatch, tmp_path):
    """Patch the three real-ffmpeg render functions with cheap stand-ins that
    just write a placeholder file and record their call args."""
    calls = {"stage1": [], "placeholder": [], "stage2": []}

    def _fake_stage1(
        local_clip_path, out_path, canvas_w, canvas_h, target, pad_before, pad_after, actual_duration, fps=30,
        *, is_last_segment=False, prefer_start_trim=False,
    ):
        calls["stage1"].append(
            dict(
                local_clip_path=local_clip_path, out_path=out_path, canvas_w=canvas_w, canvas_h=canvas_h,
                target=target, pad_before=pad_before, pad_after=pad_after, actual_duration=actual_duration,
                is_last_segment=is_last_segment, prefer_start_trim=prefer_start_trim,
            )
        )
        with open(out_path, "wb") as fh:
            fh.write(b"fake-segment")
        # "keep" mode (is_last_segment + actual >= target) uses actual_duration
        # as its effective length -- mirror the real function's return value so
        # callers summing this into total_planned_duration see the same shape.
        if is_last_segment and actual_duration >= target:
            return pad_before + actual_duration + pad_after
        return pad_before + target + pad_after

    def _fake_placeholder(canvas_w, canvas_h, duration, fps=30):
        calls["placeholder"].append(dict(canvas_w=canvas_w, canvas_h=canvas_h, duration=duration))
        path = str(tmp_path / "placeholder.mp4")
        with open(path, "wb") as fh:
            fh.write(b"fake-placeholder")
        return path

    def _fake_stage2(
        segment_paths, canvas_w, canvas_h, captions, audio_local_path, out_path, font_path=None,
        *, total_duration_hint=None,
    ):
        calls["stage2"].append(
            dict(
                segment_paths=list(segment_paths), canvas_w=canvas_w, canvas_h=canvas_h, captions=captions,
                audio_local_path=audio_local_path, total_duration_hint=total_duration_hint,
            )
        )
        with open(out_path, "wb") as fh:
            fh.write(b"fake-mastercut")
        return []  # no caption text paths to clean up

    monkeypatch.setattr("agents.assembly_agent._render_stage1_segment", _fake_stage1)
    monkeypatch.setattr("agents.assembly_agent._render_placeholder_segment", _fake_placeholder)
    monkeypatch.setattr("agents.assembly_agent._render_master_cut", _fake_stage2)
    return calls


def _fake_probe_fn(_local_path: str) -> dict:
    """Canned probe -- FAST tests never touch real ffmpeg.probe."""
    return {"duration": 3.0, "width": 1920, "height": 1080}


def _make_download_fn(job_id: str, tmp_path, *, fail_shot_ids: frozenset = frozenset()) -> callable:
    """Fake download_fn: writes a real captions JSON for the captions URL (the
    orchestration genuinely `json.load`s it), a dummy file for the audio URL,
    and a dummy file per shot -- unless that shot_id is in `fail_shot_ids`,
    which raises (simulating a per-shot fetch failure)."""

    def _dl(url: str) -> str:
        if url.endswith("captions.json"):
            path = tmp_path / "captions.json"
            path.write_text(
                json.dumps([{"text": f"line {i}", "start_ts": i * 3.0, "end_ts": (i + 1) * 3.0} for i in range(3)]),
                encoding="utf-8",
            )
            return str(path)
        if url.endswith("voiceover.mp3"):
            path = tmp_path / "voiceover.mp3"
            path.write_bytes(b"fake-audio")
            return str(path)
        # shot clip URL, e.g. ".../shots/s2/shot.mp4"
        for sid in fail_shot_ids:
            if f"/{sid}/" in url:
                raise RuntimeError(f"simulated fetch failure for {sid}")
        leaf = url.rsplit("/", 1)[-1]
        path = tmp_path / f"clip_{leaf}"
        path.write_bytes(b"fake-clip")
        return str(path)

    return _dl


def _voiceover(job_id: str) -> dict:
    return {
        "audio_uri": f"http://oss.example.com/jobs/{job_id}/voiceover.mp3",
        "caption_track_uri": f"http://oss.example.com/jobs/{job_id}/captions.json",
    }


@pytest.mark.asyncio
async def test_impl_happy_path_renders_one_segment_per_shot_and_uploads(fake_renders, tmp_path):
    shot_list = [_shot("s1", 0), _shot("s2", 1), _shot("s3", 2)]
    generated_shots = {sid: {"video_uri": f"http://oss.example.com/jobs/job-1/shots/{sid}/shot.mp4"} for sid in ("s1", "s2", "s3")}

    uploaded = {}

    def _upload(local_path):
        uploaded["path"] = local_path
        return "http://oss.example.com/jobs/job-1/master_cut.mp4"

    result = await _assemble_master_cut_impl(
        shot_list, generated_shots, _voiceover("job-1"), _winning_script(3), "job-1",
        download_fn=_make_download_fn("job-1", tmp_path), probe_fn=_fake_probe_fn, upload_fn=_upload,
    )

    assert result.master_cut_uri == "http://oss.example.com/jobs/job-1/master_cut.mp4"
    assert result.shot_count == 3
    assert result.degraded_beats == []
    assert len(fake_renders["stage1"]) == 3
    assert fake_renders["stage2"][0]["canvas_w"] == 1920 and fake_renders["stage2"][0]["canvas_h"] == 1080
    # the uploaded local file no longer exists afterward -- cleaned up.
    assert not os.path.exists(uploaded["path"])


@pytest.mark.asyncio
async def test_impl_marks_only_the_last_segment_and_passes_total_duration_hint(fake_renders, tmp_path):
    """PHASE 3 wiring: only the LAST rendered segment gets is_last_segment=True,
    and Stage 2 receives the real summed durations as total_duration_hint."""
    shot_list = [_shot("s1", 0), _shot("s2", 1), _shot("s3", 2)]
    generated_shots = {sid: {"video_uri": f"http://oss.example.com/jobs/job-9/shots/{sid}/shot.mp4"} for sid in ("s1", "s2", "s3")}

    result = await _assemble_master_cut_impl(
        shot_list, generated_shots, _voiceover("job-9"), _winning_script(3), "job-9",
        download_fn=_make_download_fn("job-9", tmp_path), probe_fn=_fake_probe_fn,
        upload_fn=lambda p: "http://oss.example.com/jobs/job-9/master_cut.mp4",
    )
    assert result.shot_count == 3

    stage1_calls = fake_renders["stage1"]
    assert [c["is_last_segment"] for c in stage1_calls] == [False, False, True]
    # _fake_probe_fn returns duration=3.0 == target=3.0 for every beat -> the
    # last segment's "keep" branch (actual >= target) still sums to 3.0.
    assert fake_renders["stage2"][0]["total_duration_hint"] == pytest.approx(9.0)


@pytest.mark.asyncio
async def test_impl_prefers_start_trim_for_mid_ad_human_interaction_shots_only(fake_renders, tmp_path):
    """PHASE 3 point 3: a worn_in_use/product_in_hand MID-ad shot (not the
    last segment, not hook/cta) gets prefer_start_trim=True; every other shot
    type/position keeps the original end-trim default."""
    shot_list = [
        {**_shot("s1", 0), "shot_type": "hook_hero", "beat_role": "hook"},
        {**_shot("s2", 1), "shot_type": "worn_in_use", "beat_role": "proof"},
        {**_shot("s3", 2), "shot_type": "cta_endcard", "beat_role": "cta"},
    ]
    generated_shots = {sid: {"video_uri": f"http://oss.example.com/jobs/job-10/shots/{sid}/shot.mp4"} for sid in ("s1", "s2", "s3")}

    await _assemble_master_cut_impl(
        shot_list, generated_shots, _voiceover("job-10"), _winning_script(3), "job-10",
        download_fn=_make_download_fn("job-10", tmp_path), probe_fn=_fake_probe_fn,
        upload_fn=lambda p: "http://oss.example.com/jobs/job-10/master_cut.mp4",
    )
    stage1_calls = fake_renders["stage1"]
    assert [c["prefer_start_trim"] for c in stage1_calls] == [False, True, False]


@pytest.mark.asyncio
async def test_impl_human_interaction_shot_as_the_last_segment_never_prefers_start_trim(fake_renders, tmp_path):
    """A worn_in_use shot that happens to BE the last segment must not get
    prefer_start_trim=True -- it never hard-trims at all (is_last_segment
    already governs its conform), so the flag would be meaningless/misleading."""
    shot_list = [_shot("s1", 0), {**_shot("s2", 1), "shot_type": "worn_in_use", "beat_role": "proof"}]
    generated_shots = {sid: {"video_uri": f"http://oss.example.com/jobs/job-11/shots/{sid}/shot.mp4"} for sid in ("s1", "s2")}

    await _assemble_master_cut_impl(
        shot_list, generated_shots, _voiceover("job-11"), _winning_script(2), "job-11",
        download_fn=_make_download_fn("job-11", tmp_path), probe_fn=_fake_probe_fn,
        upload_fn=lambda p: "http://oss.example.com/jobs/job-11/master_cut.mp4",
    )
    stage1_calls = fake_renders["stage1"]
    last_call = stage1_calls[-1]
    assert last_call["is_last_segment"] is True
    assert last_call["prefer_start_trim"] is False


@pytest.mark.asyncio
async def test_public_assemble_master_cut_returns_only_the_uri(fake_renders, tmp_path):
    shot_list = [_shot("s1", 0)]
    generated_shots = {"s1": {"video_uri": "http://oss.example.com/jobs/job-2/shots/s1/shot.mp4"}}

    uri = await assemble_master_cut(
        shot_list, generated_shots, _voiceover("job-2"), _winning_script(1), "job-2",
        download_fn=_make_download_fn("job-2", tmp_path), probe_fn=_fake_probe_fn,
        upload_fn=lambda p: "http://oss.example.com/jobs/job-2/master_cut.mp4",
    )
    assert isinstance(uri, str)
    assert uri == "http://oss.example.com/jobs/job-2/master_cut.mp4"


@pytest.mark.asyncio
async def test_impl_beat_shot_mismatch_folds_orphaned_beat_into_neighbor(fake_renders, tmp_path):
    """The real shape confirmed in derisk/outputs/full_pipeline_live_result.json:
    4 beats, only 3 shots (treatment_refs 0/1/2) -- beat 3 has no shot."""
    shot_list = [_shot("s1", 0), _shot("s2", 1), _shot("s3", 2)]
    generated_shots = {sid: {"video_uri": f"http://oss.example.com/jobs/job-3/shots/{sid}/shot.mp4"} for sid in ("s1", "s2", "s3")}

    def _dl(url):
        if url.endswith("captions.json"):
            path = tmp_path / "captions.json"
            path.write_text(json.dumps(_captions(4)), encoding="utf-8")
            return str(path)
        if url.endswith("voiceover.mp3"):
            path = tmp_path / "voiceover.mp3"
            path.write_bytes(b"fake-audio")
            return str(path)
        leaf = url.rsplit("/", 1)[-1]
        path = tmp_path / f"clip_{leaf}_{url.rsplit('/', 2)[-2]}"
        path.write_bytes(b"fake-clip")
        return str(path)

    result = await _assemble_master_cut_impl(
        shot_list, generated_shots, _voiceover("job-3"), _winning_script(4), "job-3",
        download_fn=_dl, probe_fn=_fake_probe_fn,
        upload_fn=lambda p: "http://oss.example.com/jobs/job-3/master_cut.mp4",
    )

    assert result.shot_count == 3  # 3 real segments; the orphaned beat added no new segment
    assert result.degraded_beats == []  # no FETCH failure -- Budget Gate simply never gave beat 3 a shot
    last_seg = fake_renders["stage1"][-1]
    assert last_seg["pad_after"] == pytest.approx(3.0)  # beat 3's real VO window held onto s3's last frame


@pytest.mark.asyncio
async def test_impl_shot_download_failure_is_isolated_as_a_held_frame_gap(fake_renders, tmp_path):
    """One shot's clip fetch failure must not sink the whole assembly -- it
    degrades to the SAME held-frame handling as a Budget-Gate-cut beat
    (module docstring's BEAT/SHOT MISMATCH POLICY), logged/traced, not crashed."""
    shot_list = [_shot("s1", 0), _shot("s2", 1), _shot("s3", 2)]
    generated_shots = {sid: {"video_uri": f"http://oss.example.com/jobs/job-4/shots/{sid}/shot.mp4"} for sid in ("s1", "s2", "s3")}

    result = await _assemble_master_cut_impl(
        shot_list, generated_shots, _voiceover("job-4"), _winning_script(3), "job-4",
        download_fn=_make_download_fn("job-4", tmp_path, fail_shot_ids=frozenset({"s2"})),
        probe_fn=_fake_probe_fn,
        upload_fn=lambda p: "http://oss.example.com/jobs/job-4/master_cut.mp4",
    )

    assert result.shot_count == 2  # only s1 and s3 rendered
    assert len(result.degraded_beats) == 1
    assert result.degraded_beats[0]["beat_index"] == 1
    assert result.degraded_beats[0]["shot_id"] == "s2"
    # s1 (beat 0, leading real segment before the failed one) absorbs the hold.
    stage1_calls = fake_renders["stage1"]
    assert len(stage1_calls) == 2
    holder = next(c for c in stage1_calls if c["pad_after"] > 0 or c["pad_before"] > 0)
    assert holder["pad_after"] == pytest.approx(3.0)  # beat 0 (s1) holds forward into beat 1's gap


@pytest.mark.asyncio
async def test_impl_no_usable_shot_anywhere_renders_placeholder(fake_renders, tmp_path):
    shot_list = [_shot("s1", 0)]
    generated_shots = {}  # no generated entry at all -> s1 is unusable

    def _dl(url):
        if url.endswith("captions.json"):
            path = tmp_path / "captions.json"
            path.write_text(json.dumps(_captions(1)), encoding="utf-8")  # single 3.0s beat
            return str(path)
        if url.endswith("voiceover.mp3"):
            path = tmp_path / "voiceover.mp3"
            path.write_bytes(b"fake-audio")
            return str(path)
        raise AssertionError("no shot clip should ever be fetched -- s1 has no generated_shots entry")

    result = await _assemble_master_cut_impl(
        shot_list, generated_shots, _voiceover("job-5"), _winning_script(1), "job-5",
        download_fn=_dl, probe_fn=_fake_probe_fn,
        upload_fn=lambda p: "http://oss.example.com/jobs/job-5/master_cut.mp4",
    )
    assert result.shot_count == 0
    assert len(result.degraded_beats) == 1
    assert len(fake_renders["placeholder"]) == 1
    assert fake_renders["placeholder"][0]["duration"] == pytest.approx(3.0)


@pytest.mark.asyncio
async def test_impl_raises_assembly_error_when_captions_track_fetch_fails(fake_renders, tmp_path):
    def _dl(url):
        raise RuntimeError("OSS unavailable")

    with pytest.raises(AssemblyError, match="captions"):
        await _assemble_master_cut_impl(
            [_shot("s1", 0)], {"s1": {"video_uri": "http://x/s1.mp4"}}, _voiceover("job-6"),
            _winning_script(1), "job-6", download_fn=_dl, probe_fn=_fake_probe_fn,
        )


@pytest.mark.asyncio
async def test_impl_raises_assembly_error_when_audio_fetch_fails(fake_renders, tmp_path):
    def _dl(url):
        if url.endswith("captions.json"):
            path = tmp_path / "captions.json"
            path.write_text(json.dumps(_captions(1)), encoding="utf-8")
            return str(path)
        raise RuntimeError("OSS unavailable")

    with pytest.raises(AssemblyError, match="voiceover audio"):
        await _assemble_master_cut_impl(
            [_shot("s1", 0)], {"s1": {"video_uri": "http://x/s1.mp4"}}, _voiceover("job-7"),
            _winning_script(1), "job-7", download_fn=_dl, probe_fn=_fake_probe_fn,
        )


@pytest.mark.asyncio
async def test_impl_shot_with_non_terminal_status_is_treated_as_unusable(fake_renders, tmp_path):
    """A shot that somehow reaches Assembly still 'pending'/'review'/etc.
    (should not happen post continuity_gate, but defensively) is treated as
    no-clip-available, not a crash."""
    shot_list = [_shot("s1", 0, status="pending")]
    generated_shots = {"s1": {"video_uri": "http://oss.example.com/jobs/job-8/shots/s1/shot.mp4"}}

    result = await _assemble_master_cut_impl(
        shot_list, generated_shots, _voiceover("job-8"), _winning_script(1), "job-8",
        download_fn=_make_download_fn("job-8", tmp_path), probe_fn=_fake_probe_fn,
        upload_fn=lambda p: "http://oss.example.com/jobs/job-8/master_cut.mp4",
    )
    assert result.shot_count == 0
    assert result.degraded_beats[0]["reason"] == "no usable generated clip"


# ---------------------------------------------------------------------------
# REAL ffmpeg execution -- everything above fakes the render step entirely.
# This suite actually runs the two-stage pipeline (Stage 1 normalize/conform
# per shot, Stage 2 concat+captions+audio) against real synthetic clips, gated
# on ffmpeg/ffprobe being on PATH (matches test_ken_burns_fallback_node.py /
# test_voiceover_caption_agent.py's established convention).
# ---------------------------------------------------------------------------
def _make_color_clip(tmp_path, name: str, size: str, duration: float, fps: int = 30, color: str = "red") -> str:
    path = str(tmp_path / name)
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", f"color=c={color}:s={size}:d={duration}:r={fps}",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", path],
        check=True, capture_output=True,
    )
    return path


def _make_sine_audio(tmp_path, name: str, duration: float) -> str:
    path = str(tmp_path / name)
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", f"sine=frequency=440:duration={duration}",
         "-ac", "2", "-ar", "44100", path],
        check=True, capture_output=True,
    )
    return path


def _ffprobe(path: str) -> dict:
    result = subprocess.run(
        ["ffprobe", "-v", "error",
         "-show_entries", "format=duration:stream=codec_type,codec_name,width,height,r_frame_rate,pix_fmt",
         "-of", "json", path],
        check=True, capture_output=True, text=True,
    )
    return json.loads(result.stdout)


def _frame_luma_range(path: str, timestamp: float) -> tuple[int, int]:
    """Extract one frame at `timestamp` and return its (Y-min, Y-max) via
    ffmpeg's own `signalstats` -- a burned drawtext box+text produces a much
    wider Y range than a flat color-source background (empirically confirmed:
    a plain color frame has YMIN==YMAX; a captioned one spans black-box to
    white-text). No OCR needed, per this task's own stated bar."""
    result = subprocess.run(
        ["ffmpeg", "-y", "-ss", str(timestamp), "-i", path, "-frames:v", "1",
         "-vf", "signalstats,metadata=print", "-f", "null", "-"],
        capture_output=True, text=True,
    )
    out = result.stderr
    ymin = ymax = None
    for line in out.splitlines():
        if "signalstats.YMIN=" in line:
            ymin = int(line.rsplit("=", 1)[-1])
        if "signalstats.YMAX=" in line:
            ymax = int(line.rsplit("=", 1)[-1])
    assert ymin is not None and ymax is not None, f"signalstats did not report YMIN/YMAX:\n{out}"
    return ymin, ymax


@_skip_no_ffmpeg
@pytest.mark.asyncio
async def test_real_two_stage_pipeline_mixed_resolution_orientation_and_duration(tmp_path):
    """The real gap this suite closes: every test above fakes Stage 1/Stage 2
    entirely. This runs the ACTUAL ffmpeg filter graphs (scale/pad/setsar/fps/
    format, trim/setpts/tpad, concat demuxer, drawtext chain, apad+audio map)
    against three real synthetic clips deliberately chosen to force all three
    duration-conform branches in ONE batch:
      s1: 720x1280 portrait, 4.0s actual vs. 3.0s target -> TRIM.
      s2: 1920x1080 landscape (largest area -> becomes the canvas), 2.7s vs
          3.0s target (10% short) -> STRETCH.
      s3: 480x854 portrait, 1.0s vs. 4.0s target (75% short) -> FREEZE-PAD.
    """
    clip_s1 = _make_color_clip(tmp_path, "s1.mp4", "720x1280", duration=4.0, fps=24, color="red")
    clip_s2 = _make_color_clip(tmp_path, "s2.mp4", "1920x1080", duration=2.7, fps=30, color="blue")
    clip_s3 = _make_color_clip(tmp_path, "s3.mp4", "480x854", duration=1.0, fps=25, color="green")
    audio_path = _make_sine_audio(tmp_path, "vo.wav", duration=10.0)  # 3.0 + 3.0 + 4.0

    beats = [
        {"t_start": 0.0, "t_end": 3.0, "line": "Hook line here for the top third."},
        {"t_start": 3.0, "t_end": 6.0, "line": "Middle demo line goes here now."},
        {"t_start": 6.0, "t_end": 10.0, "line": "Final call to action, buy today."},
    ]
    captions_path = tmp_path / "captions.json"
    captions_path.write_text(
        json.dumps([
            {"text": beats[0]["line"], "start_ts": 0.0, "end_ts": 3.0},
            {"text": beats[1]["line"], "start_ts": 3.0, "end_ts": 6.0},
            {"text": beats[2]["line"], "start_ts": 6.0, "end_ts": 10.0},
        ]),
        encoding="utf-8",
    )

    shot_list = [
        _shot("s1", 0, status="passed", zone="lower_third"),
        _shot("s2", 1, status="passed", zone="left_third"),
        _shot("s3", 2, status="fallback", zone="none"),  # excluded from canvas choice, still rendered
    ]
    generated_shots = {
        "s1": {"video_uri": "local://s1"},
        "s2": {"video_uri": "local://s2"},
        "s3": {"video_uri": "local://s3"},
    }
    voiceover = {"audio_uri": "local://audio", "caption_track_uri": "local://captions"}
    winning_script = {"text": " ".join(b["line"] for b in beats), "beats": beats, "source_variant_ids": ["v1"]}

    local_map = {"local://s1": clip_s1, "local://s2": clip_s2, "local://s3": clip_s3,
                 "local://audio": audio_path, "local://captions": str(captions_path)}

    captured: dict = {}

    def _upload(local_path: str) -> str:
        dst = str(tmp_path / "captured_master_cut.mp4")
        shutil.copy(local_path, dst)
        captured["path"] = dst
        return "http://fake.example.com/master_cut.mp4"

    result = await _assemble_master_cut_impl(
        shot_list, generated_shots, voiceover, winning_script, "real-job-1",
        download_fn=lambda url: local_map[url], upload_fn=_upload,
    )

    assert result.master_cut_uri == "http://fake.example.com/master_cut.mp4"
    assert result.shot_count == 3
    assert result.degraded_beats == []

    probe = _ffprobe(captured["path"])
    # Total duration matches the sum of the caption windows (10s), within a
    # couple frames of tolerance for encoder rounding.
    assert abs(float(probe["format"]["duration"]) - 10.0) < 0.3
    assert result.total_duration_sec == pytest.approx(float(probe["format"]["duration"]), abs=0.05)

    video_stream = next(s for s in probe["streams"] if s["codec_type"] == "video")
    audio_stream = next((s for s in probe["streams"] if s["codec_type"] == "audio"), None)
    # Canvas: s2 (1920x1080=2,073,600) beats s1 (720x1280=921,600); s3 (fallback) excluded.
    assert video_stream["width"] == 1920 and video_stream["height"] == 1080
    assert video_stream["pix_fmt"] == "yuv420p"
    assert audio_stream is not None, "the voiceover audio must be mapped as the sole audio track"
    assert audio_stream["codec_name"] == "aac"

    # Burned captions: sample one frame inside each of the 3 caption windows
    # and confirm the Y range is far wider than a flat color source alone
    # would produce (a real box+text overlay, not a blank/silent frame).
    for t in (1.5, 4.5, 8.0):
        ymin, ymax = _frame_luma_range(captured["path"], t)
        assert ymax - ymin > 60, f"frame at t={t}s does not look like it has a burned caption (Y range {ymin}-{ymax})"


@_skip_no_ffmpeg
@pytest.mark.asyncio
async def test_real_pipeline_holds_frame_across_orphaned_beat(tmp_path):
    """Real-ffmpeg counterpart of the fake-boundary mismatch-policy tests
    above: 3 beats, only 2 shots (the real derisk shape, scaled down) -- the
    LAST beat has no shot, so the total rendered duration must still cover
    the full VO window by holding shot s2's last frame."""
    clip_s1 = _make_color_clip(tmp_path, "s1.mp4", "1280x720", duration=3.0, fps=30, color="red")
    clip_s2 = _make_color_clip(tmp_path, "s2.mp4", "1280x720", duration=3.0, fps=30, color="blue")
    audio_path = _make_sine_audio(tmp_path, "vo.wav", duration=9.0)  # 3.0 * 3 beats

    captions = [
        {"text": "Beat zero.", "start_ts": 0.0, "end_ts": 3.0},
        {"text": "Beat one.", "start_ts": 3.0, "end_ts": 6.0},
        {"text": "Orphaned beat two -- no shot for this one.", "start_ts": 6.0, "end_ts": 9.0},
    ]
    captions_path = tmp_path / "captions.json"
    captions_path.write_text(json.dumps(captions), encoding="utf-8")

    beats = [{"t_start": c["start_ts"], "t_end": c["end_ts"], "line": c["text"]} for c in captions]
    winning_script = {"text": " ".join(b["line"] for b in beats), "beats": beats, "source_variant_ids": ["v1"]}

    shot_list = [_shot("s1", 0, status="passed"), _shot("s2", 1, status="passed")]  # beat 2 orphaned
    generated_shots = {"s1": {"video_uri": "local://s1"}, "s2": {"video_uri": "local://s2"}}
    voiceover = {"audio_uri": "local://audio", "caption_track_uri": "local://captions"}
    local_map = {"local://s1": clip_s1, "local://s2": clip_s2, "local://audio": audio_path, "local://captions": str(captions_path)}

    captured: dict = {}

    def _upload(local_path: str) -> str:
        dst = str(tmp_path / "captured.mp4")
        shutil.copy(local_path, dst)
        captured["path"] = dst
        return "http://fake.example.com/master_cut.mp4"

    result = await _assemble_master_cut_impl(
        shot_list, generated_shots, voiceover, winning_script, "real-job-2",
        download_fn=lambda url: local_map[url], upload_fn=_upload,
    )

    assert result.shot_count == 2  # only 2 real segments -- the orphaned beat added no third
    assert result.degraded_beats == []  # no fetch failure; Budget Gate simply never had a shot for beat 2

    probe = _ffprobe(captured["path"])
    assert abs(float(probe["format"]["duration"]) - 9.0) < 0.3  # full VO window still covered

    # The held frame is genuinely s2's own content extended, not a black gap:
    # sample well inside the orphaned beat's window (t=7.5s) and confirm it is
    # NOT just flat black (a real render failure/gap would be near-zero Y range
    # even accounting for the caption box).
    ymin, ymax = _frame_luma_range(captured["path"], 7.5)
    assert ymax > 40, f"orphaned-beat frame at t=7.5s looks blank (Y range {ymin}-{ymax})"


# ---------------------------------------------------------------------------
# video-gen-fidelity PHASE 3, real ffmpeg -- the CTA/ending fix (never
# hard-trim the last segment's tail; burn a short end-of-cut fade) and the
# mid-ad trim-direction policy (prefer cutting a human-interaction shot's
# START, not its end).
# ---------------------------------------------------------------------------
def _avg_pixel_rgb(path: str, timestamp: float) -> tuple[int, int, int]:
    """Downscale one frame to a single pixel and read its raw RGB -- a cheap,
    dependency-free way to tell "which half of a two-tone clip survived a
    trim" without OCR or a vision model."""
    result = subprocess.run(
        ["ffmpeg", "-y", "-ss", str(timestamp), "-i", path, "-frames:v", "1",
         "-vf", "scale=1:1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        capture_output=True, check=True,
    )
    data = result.stdout
    return data[0], data[1], data[2]


def _make_two_tone_clip(tmp_path, name: str, size: str, first_color: str, first_dur: float,
                         second_color: str, second_dur: float, fps: int = 30) -> str:
    """A single clip whose first `first_dur` seconds are `first_color` and
    remaining `second_dur` seconds are `second_color` -- lets a real-ffmpeg
    test tell which end of a trim was kept by sampling pixel color."""
    p1 = _make_color_clip(tmp_path, f"{name}_a.mp4", size, first_dur, fps, first_color)
    p2 = _make_color_clip(tmp_path, f"{name}_b.mp4", size, second_dur, fps, second_color)
    out = str(tmp_path / name)
    subprocess.run(
        ["ffmpeg", "-y", "-i", p1, "-i", p2, "-filter_complex",
         "[0:v][1:v]concat=n=2:v=1:a=0[outv]", "-map", "[outv]",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", out],
        check=True, capture_output=True,
    )
    return out


@_skip_no_ffmpeg
@pytest.mark.asyncio
async def test_real_last_segment_overrun_is_never_trimmed_and_gets_a_fade(tmp_path):
    """The real gap PHASE 3 closes: a final beat whose clip runs LONGER than
    its VO window used to be hard-trimmed down to the window, cutting off the
    clip's own natural ending. Now the full clip plays (video overruns its
    nominal 3.0s target to its real 5.0s length), the VO audio is silently
    extended to match (apad+shortest), and the very end fades out."""
    clip_s1 = _make_color_clip(tmp_path, "s1.mp4", "640x480", duration=3.0, fps=30, color="red")
    clip_s2 = _make_color_clip(tmp_path, "s2.mp4", "640x480", duration=5.0, fps=30, color="blue")  # overruns its 3.0s target
    audio_path = _make_sine_audio(tmp_path, "vo.wav", duration=6.0)  # 3.0 + 3.0 nominal VO

    captions = [
        {"text": "Beat zero.", "start_ts": 0.0, "end_ts": 3.0},
        {"text": "Beat one, the CTA close.", "start_ts": 3.0, "end_ts": 6.0},
    ]
    captions_path = tmp_path / "captions.json"
    captions_path.write_text(json.dumps(captions), encoding="utf-8")
    beats = [{"t_start": c["start_ts"], "t_end": c["end_ts"], "line": c["text"]} for c in captions]
    winning_script = {"text": " ".join(b["line"] for b in beats), "beats": beats, "source_variant_ids": ["v1"]}

    shot_list = [_shot("s1", 0, status="passed"), _shot("s2", 1, status="passed")]
    generated_shots = {"s1": {"video_uri": "local://s1"}, "s2": {"video_uri": "local://s2"}}
    voiceover = {"audio_uri": "local://audio", "caption_track_uri": "local://captions"}
    local_map = {"local://s1": clip_s1, "local://s2": clip_s2, "local://audio": audio_path, "local://captions": str(captions_path)}

    captured: dict = {}

    def _upload(local_path: str) -> str:
        dst = str(tmp_path / "captured_ending.mp4")
        shutil.copy(local_path, dst)
        captured["path"] = dst
        return "http://fake.example.com/master_cut.mp4"

    result = await _assemble_master_cut_impl(
        shot_list, generated_shots, voiceover, winning_script, "real-job-3",
        download_fn=lambda url: local_map[url], upload_fn=_upload,
    )
    assert result.shot_count == 2

    probe = _ffprobe(captured["path"])
    # 3.0 (s1) + 5.0 (s2's FULL natural length, not trimmed to its 3.0s
    # target) = 8.0s total -- proves the tail was never hard-trimmed.
    assert abs(float(probe["format"]["duration"]) - 8.0) < 0.3
    audio_stream = next((s for s in probe["streams"] if s["codec_type"] == "audio"), None)
    assert audio_stream is not None  # VO audio still present (auto-extended by apad+shortest)

    # The tail is genuinely blue content (s2's own real footage), not a
    # trimmed-away black gap -- sample well before the fade kicks in.
    r, g, b = _avg_pixel_rgb(captured["path"], 6.5)
    assert b > r and b > 100, f"expected s2's own blue content at t=6.5s, got RGB=({r},{g},{b})"

    # The very end fades to black (VIDEO_FADE_SEC=0.5s fade-out) -- sample
    # right at the tail and confirm the frame has visibly dimmed relative to
    # the mid-clip blue sampled above.
    r_end, g_end, b_end = _avg_pixel_rgb(captured["path"], 7.9)
    assert b_end < b, f"expected the final frame to have faded toward black (mid={b}, end={b_end})"


@_skip_no_ffmpeg
@pytest.mark.asyncio
async def test_real_mid_ad_human_interaction_shot_trims_from_the_start(tmp_path):
    """PHASE 3 point 3: a worn_in_use MID-ad shot (not last, not hook/cta)
    whose clip overruns its VO window gets trimmed from the START (discarding
    its own weakest/least-in-motion opening), not the end."""
    # s1 (hook, hook_hero): 1s red then 3s blue, target 3.0s -> DEFAULT end-trim
    # keeps the FIRST 3.0s (red 0-1 + blue 1-3) -- some red survives.
    clip_hook = _make_two_tone_clip(tmp_path, "hook.mp4", "640x480", "red", 1.0, "blue", 3.0)
    # s2 (proof, worn_in_use): 1s green then 3s yellow, target 3.0s -> START-trim
    # discards the first 1.0s (green), keeping ONLY the last 3.0s (yellow).
    clip_worn = _make_two_tone_clip(tmp_path, "worn.mp4", "640x480", "green", 1.0, "yellow", 3.0)
    # s3 (cta, cta_endcard, the LAST segment): plain clip matching its target
    # exactly so this test isolates the mid-ad trim-direction behavior.
    clip_cta = _make_color_clip(tmp_path, "cta.mp4", "640x480", duration=3.0, fps=30, color="white")

    audio_path = _make_sine_audio(tmp_path, "vo.wav", duration=9.0)
    captions = [
        {"text": "Hook.", "start_ts": 0.0, "end_ts": 3.0},
        {"text": "Proof, worn in use.", "start_ts": 3.0, "end_ts": 6.0},
        {"text": "CTA.", "start_ts": 6.0, "end_ts": 9.0},
    ]
    captions_path = tmp_path / "captions.json"
    captions_path.write_text(json.dumps(captions), encoding="utf-8")
    beats = [{"t_start": c["start_ts"], "t_end": c["end_ts"], "line": c["text"]} for c in captions]
    winning_script = {"text": " ".join(b["line"] for b in beats), "beats": beats, "source_variant_ids": ["v1"]}

    shot_list = [
        {**_shot("s1", 0, status="passed"), "shot_type": "hook_hero", "beat_role": "hook"},
        {**_shot("s2", 1, status="passed"), "shot_type": "worn_in_use", "beat_role": "proof"},
        {**_shot("s3", 2, status="passed"), "shot_type": "cta_endcard", "beat_role": "cta"},
    ]
    generated_shots = {"s1": {"video_uri": "local://s1"}, "s2": {"video_uri": "local://s2"}, "s3": {"video_uri": "local://s3"}}
    voiceover = {"audio_uri": "local://audio", "caption_track_uri": "local://captions"}
    local_map = {
        "local://s1": clip_hook, "local://s2": clip_worn, "local://s3": clip_cta,
        "local://audio": audio_path, "local://captions": str(captions_path),
    }

    captured: dict = {}

    def _upload(local_path: str) -> str:
        dst = str(tmp_path / "captured_trimdir.mp4")
        shutil.copy(local_path, dst)
        captured["path"] = dst
        return "http://fake.example.com/master_cut.mp4"

    await _assemble_master_cut_impl(
        shot_list, generated_shots, voiceover, winning_script, "real-job-4",
        download_fn=lambda url: local_map[url], upload_fn=_upload,
    )

    # s2's rendered window is [3.0, 6.0)s in the concatenated timeline --
    # sample just inside it: START-trim means it is ENTIRELY yellow (green
    # discarded), not a green/yellow mix.
    r, g, b = _avg_pixel_rgb(captured["path"], 3.2)
    assert r > 100 and g > 100 and b < 80, f"expected yellow (start-trimmed) at t=3.2s, got RGB=({r},{g},{b})"

    # s1's rendered window is [0.0, 3.0)s -- DEFAULT end-trim keeps the first
    # 3.0s of a 1s-red+3s-blue clip, so sampling right at its start is red
    # (the opening was NOT discarded, unlike the human-interaction shot above).
    r0, g0, b0 = _avg_pixel_rgb(captured["path"], 0.2)
    assert r0 > 100 and b0 < 80, f"expected red (opening kept, end-trim default) at t=0.2s, got RGB=({r0},{g0},{b0})"
