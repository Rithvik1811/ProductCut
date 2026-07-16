"""
Shared fakes for Phase 3 graph tests (Video-Gen + Ken-Burns Fallback).

Every graph test that runs past the Budget Gate should monkeypatch these
boundaries so no real DashScope/Wan/ffmpeg/OSS calls are made.
"""
from __future__ import annotations

import json
import os
import tempfile
from typing import Callable

from agents.video_gen_node import VideoGenAPIError

# Canned Visual Direction Agent response for the shared 3-beat winning script
# (beats: hook/demo/cta). t7 is the form_factor truth (compact aluminum housing).
_VDA_PAYLOAD = json.dumps({
    "story_context": (
        "The compact aluminum housing sits alone on a clean surface. "
        "A slow push reveals the diagonal scratch before we return to the "
        "full form for the closing ask."
    ),
    "beat_visual_directions": [
        {
            "beat_index": 0,
            "focus_feature_truth_id": "t7",
            "focus_moment": "compact charcoal block emerging from dark background",
            "human_presence": "no",
            "suggested_shot_type": "hook_hero",
            "suggested_camera_move": "push_in",
            "framing_notes": "product fills frame, matte finish catching warm key",
        },
        {
            "beat_index": 1,
            "focus_feature_truth_id": "t1",
            "focus_moment": "hairline diagonal scratch catching warm sidelight",
            "human_presence": "no",
            "suggested_shot_type": "macro_detail",
            "suggested_camera_move": "static",
            "framing_notes": "extreme macro, scratch bisects frame diagonally",
        },
        {
            "beat_index": 2,
            "focus_feature_truth_id": "t7",
            "focus_moment": "full form resolves to endcard, proportions preserved",
            "human_presence": "no",
            "suggested_shot_type": "cta_endcard",
            "suggested_camera_move": "static",
            "framing_notes": "lower third reserved for CTA overlay",
        },
    ],
})


def patch_visual_direction_boundaries(monkeypatch) -> None:
    """Fake the Visual Direction Agent's AsyncOpenAI boundary (sits between
    merge_validator and treatment_agent in the graph). Every graph test that
    runs past merge_validator's "finalize"/"fallback" routes needs this, or
    the real agent hits the DashScope endpoint."""
    from tests._fakes import make_fake_async_openai
    monkeypatch.setattr(
        "agents.visual_direction_agent.AsyncOpenAI",
        make_fake_async_openai([_VDA_PAYLOAD]),
    )


def make_fake_wan_generate(*, fail_if_prompt_contains: str | None = None) -> Callable:
    """Return an async generate_fn that succeeds unless the prompt matches."""

    async def _fake(**kwargs):
        prompt = kwargs.get("prompt", "")
        if fail_if_prompt_contains and fail_if_prompt_contains in prompt:
            raise VideoGenAPIError(f"simulated failure for prompt containing {fail_if_prompt_contains!r}")
        image_url = kwargs.get("image_url", "clip")
        leaf = image_url.rsplit("/", 1)[-1].split("?", 1)[0]
        return f"http://oss.example.com/wan/{leaf}.mp4"

    return _fake


def patch_phase3_boundaries(monkeypatch, *, fail_shot_s2: bool = False) -> None:
    """Monkeypatch Video-Gen Wan calls + OSS persist + Ken-Burns render/upload."""
    fail_marker = "asymmetric rear vent" if fail_shot_s2 else None
    monkeypatch.setattr(
        "agents.video_gen_node._call_wan_video_gen",
        make_fake_wan_generate(fail_if_prompt_contains=fail_marker),
    )

    def _fake_persist(remote_url, job_id, shot_id, filename="shot.mp4", *, bucket=None, download_fn=None):
        return f"http://oss.example.com/jobs/{job_id}/shots/{shot_id}/{filename}"

    monkeypatch.setattr("agents.video_gen_node.persist_remote_video_to_oss", _fake_persist)

    def _fake_render(shot, product_photos):
        fd, path = tempfile.mkstemp(suffix=".mp4", prefix="kenburns_test_")
        os.close(fd)
        with open(path, "wb") as fh:
            fh.write(b"fake-kenburns-mp4")
        return path

    monkeypatch.setattr("agents.ken_burns_fallback_node.render_ken_burns_clip", _fake_render)

    def _fake_upload(local_path, job_id, shot_id, filename="fallback_kenburns.mp4", *, bucket=None):
        return f"http://oss.example.com/jobs/{job_id}/shots/{shot_id}/{filename}"

    monkeypatch.setattr("agents.ken_burns_fallback_node.upload_video_to_oss", _fake_upload)


def patch_continuity_boundaries(
    monkeypatch, *, drift_score: float = 0.0, same_object: bool = True
) -> None:
    """Fake the Continuity Agent's per-shot scoring units (§5.10 -- ffmpeg frame
    extraction + the Qwen-VL call for BOTH the drift score and the v8 frame-0(ish)
    identity check), so a full-graph test can run through the Continuity
    Agent/Gate without real network or ffmpeg.

    Defaults to a clean, within-threshold `drift_score` AND a clean
    `same_object=True` identity verdict, so every scored shot passes both
    checks, the Continuity Gate leaves them all "passed", and the retry loop
    ends immediately with no interrupt/hard-identity-retry -- i.e. the full
    pipeline reaches END exactly as it did before Phase 4 wired Continuity in.
    Tests that want to exercise the drift retry/interrupt path fake
    `_score_one_shot` themselves (see tests/test_continuity_loop_e2e.py); tests
    exercising the identity hard-failure path fake `_score_one_shot_identity`
    themselves (see tests/test_continuity_agent.py / test_continuity_gate.py)."""
    from agents.continuity_agent import IdentityCheckResult

    async def _fake_score_one_shot(shot, entry, product_photos, client, extract):  # noqa: ARG001
        return drift_score, "clean match (test fake)"

    async def _fake_score_one_shot_identity(shot, entry, product_photos, client, extract):  # noqa: ARG001
        return IdentityCheckResult(
            matching_features=["clean match (test fake)"],
            mismatching_features=[],
            same_object=same_object,
            confidence="high",
        )

    monkeypatch.setattr("agents.continuity_agent._score_one_shot", _fake_score_one_shot)
    monkeypatch.setattr("agents.continuity_agent._score_one_shot_identity", _fake_score_one_shot_identity)


def patch_voiceover_boundaries(monkeypatch) -> None:
    """Fake the Voiceover + Caption Agent's TTS synth + OSS upload boundary AND
    the Voice Direction Agent's LLM boundary (Phase 5 -- both now sit in a
    parallel sub-branch off merge_validator: voice_direction_agent ->
    voiceover_caption_agent), so a full-graph test doesn't hit real CosyVoice
    TTS, OSS, or the DashScope LLM. Mirrors patch_phase3_boundaries/
    patch_continuity_boundaries above -- every graph test that runs past
    merge_validator's "finalize"/"fallback" routes should call this too, now
    that both nodes fan out on the same edge."""

    async def _fake_generate_voiceover(winning_script, job_id, **kwargs):  # noqa: ARG001
        beats = winning_script.get("beats") or []
        captions = [
            {"text": b.get("line", ""), "start_ts": float(b.get("t_start", 0.0)), "end_ts": float(b.get("t_end", 0.0))}
            for b in beats
        ]
        voiceover = {
            "audio_uri": f"http://oss.example.com/jobs/{job_id}/voiceover.mp3",
            "caption_track_uri": f"http://oss.example.com/jobs/{job_id}/captions.json",
        }
        return voiceover, captions

    monkeypatch.setattr("agents.voiceover_caption_agent.generate_voiceover", _fake_generate_voiceover)

    async def _fake_generate_directed_beats(winning_script, **kwargs):  # noqa: ARG001
        beats = winning_script.get("beats") or []
        return [
            {
                "beat_index": i,
                "spoken_text": b.get("line", ""),
                "emotion": "conversational",
                "pacing": "normal",
            }
            for i, b in enumerate(beats)
        ]

    monkeypatch.setattr(
        "agents.voice_direction_agent.generate_directed_beats",
        _fake_generate_directed_beats,
    )


def patch_assembly_boundaries(monkeypatch) -> None:
    """Fake the Assembly Agent's ffmpeg/OSS boundary (Phase 5, §5.12 -- the
    fan-in join of the voiceover branch and the continuity retry loop, see
    graph/build.py's module docstring), so a full-graph test doesn't shell
    out to real ffmpeg or hit real OSS. Mirrors patch_voiceover_boundaries
    above -- every graph test that runs past merge_validator's "finalize"/
    "fallback" routes (i.e. every test that already calls
    patch_voiceover_boundaries) should call this too, now that both branches
    converge on assembly_agent instead of independently terminating at END.

    Patches `_assemble_master_cut_impl` (the same "core function" boundary
    patch_voiceover_boundaries patches `generate_voiceover` at) rather than
    the thin public `assemble_master_cut` wrapper, so the node wrapper's own
    orchestration (event dispatch, trace note, `state["master_cut_uri"]`
    write) still runs for real against this fake result.
    """
    from agents.assembly_agent import AssemblyResult

    async def _fake_assemble_impl(shot_list, generated_shots, voiceover, winning_script, job_id, **kwargs):  # noqa: ARG001
        real_shots = [s for s in shot_list if s.get("status") in ("passed", "fallback")]
        total = sum(
            float(b.get("t_end", 0.0)) - float(b.get("t_start", 0.0))
            for b in (winning_script.get("beats") or [])
        )
        return AssemblyResult(
            master_cut_uri=f"http://oss.example.com/jobs/{job_id}/master_cut.mp4",
            shot_count=len(real_shots),
            total_duration_sec=round(total, 3),
            degraded_beats=[],
        )

    monkeypatch.setattr("agents.assembly_agent._assemble_master_cut_impl", _fake_assemble_impl)
