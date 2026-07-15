"""
Real-asset live verification of the Assembly Agent (§5.12) -- costs nothing
new. Reuses `derisk/outputs/full_pipeline_live_result.json` (a genuinely real
full-pipeline run's final state: real shot_list, real generated_shots with
real Wan video_uris, real winning_script, real voiceover) and the already-
downloaded `derisk/outputs/shot_s1.mp4` / `shot_s2.mp4` / `shot_s3.mp4` clips
sitting on disk from that same run.

This is the single most valuable Assembly verification available without a
new API call: it proves the real two-stage ffmpeg pipeline against a
genuinely real, already-validated 3-shot narrative ad (hook/demo/proof roles,
varied camera moves, all `same_object=True`) -- AND it happens to be the real
beat/shot MISMATCH case (the real winning_script has 4 beats; the real
shot_list only covers treatment_ref 0/1/2 -- beat 3, the CTA line "Stop
fighting your bag. Grab yours today.", has no shot), so this run exercises
the hold-frame policy for real, not just in a synthetic unit test.

Shot clips are read directly from the local `shot_s{1,2,3}.mp4` files (NOT
re-downloaded from the possibly-now-expired 24h signed OSS URLs baked into
the saved JSON) via a custom `download_fn`. The voiceover audio + captions
track were NOT saved locally by the original run, so those ARE downloaded
fresh from their signed URLs -- if those have also expired since, this script
reports that clearly rather than silently producing a bad/empty result.

The finished master cut is saved locally (never re-uploaded to real OSS --
`upload_fn` just copies the local file into `derisk/outputs/`), then
ffprobe'd and a representative frame is extracted for direct visual
inspection (burned captions, held-frame content).

Usage (from backend/):
    python -m derisk.test_assembly_live
"""
from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from agents._oss import _download_to_temp  # noqa: E402
from agents.assembly_agent import _assemble_master_cut_impl  # noqa: E402

OUTPUTS_DIR = Path(__file__).resolve().parent / "outputs"
RESULT_JSON = OUTPUTS_DIR / "full_pipeline_live_result.json"
LOCAL_CLIPS = {"s1": OUTPUTS_DIR / "shot_s1.mp4", "s2": OUTPUTS_DIR / "shot_s2.mp4", "s3": OUTPUTS_DIR / "shot_s3.mp4"}
JOB_ID = "livetest-fullpipeline"


def _ffprobe(path: str) -> dict:
    result = subprocess.run(
        ["ffprobe", "-v", "error",
         "-show_entries", "format=duration:stream=codec_type,codec_name,width,height,r_frame_rate,pix_fmt",
         "-of", "json", path],
        check=True, capture_output=True, text=True,
    )
    return json.loads(result.stdout)


def _make_download_fn():
    """Local shot clips for s1/s2/s3 (already on disk, no re-download); a
    REAL fresh download for anything else (voiceover audio, captions JSON) --
    those signed URLs are attempted for real, per this script's own docstring."""

    def _dl(url: str) -> str:
        for shot_id, local_path in LOCAL_CLIPS.items():
            if f"/shots/{shot_id}/" in url:
                print(f"  [download_fn] {shot_id}: using LOCAL {local_path} (no re-download)")
                return str(local_path)
        print(f"  [download_fn] fetching fresh: {url[:100]}...")
        return _download_to_temp(url)

    return _dl


def main() -> int:
    if not RESULT_JSON.exists():
        print(f"ERROR: {RESULT_JSON} not found -- nothing to verify against.")
        return 1
    for shot_id, path in LOCAL_CLIPS.items():
        if not path.exists():
            print(f"ERROR: expected local clip {path} not found.")
            return 1

    state = json.loads(RESULT_JSON.read_text(encoding="utf-8"))
    shot_list = state["shot_list"]
    generated_shots = state["generated_shots"]
    voiceover = state["voiceover"]
    winning_script = state["winning_script"]

    print("=== Real state loaded from a genuine prior live pipeline run ===")
    print(f"  winning_script beats: {len(winning_script['beats'])}")
    for b in winning_script["beats"]:
        print(f"    [{b['t_start']}-{b['t_end']}] {b['line']!r}")
    print(f"  shot_list: {[s['shot_id'] for s in shot_list]} "
          f"(treatment_refs: {[s['justification']['treatment_ref'] for s in shot_list]})")
    if len(shot_list) < len(winning_script["beats"]):
        orphaned = set(range(len(winning_script["beats"]))) - {
            s["justification"]["treatment_ref"] for s in shot_list
        }
        print(f"  *** REAL beat/shot mismatch: beat(s) {sorted(orphaned)} have NO shot -- "
              "this run genuinely exercises the hold-frame policy. ***")

    print("\n=== Fetching voiceover audio + captions (real, fresh download attempt) ===")
    try:
        # Cheap pre-flight so a signed-URL expiry fails LOUDLY and early, with a
        # clear message, rather than deep inside the assembly pipeline.
        probe_dl = _download_to_temp(voiceover["caption_track_uri"])
        print(f"  captions track fetched OK ({probe_dl})")
    except Exception as exc:  # noqa: BLE001
        print(f"  *** voiceover/captions signed URL appears EXPIRED or unreachable: {exc}")
        print("  Cannot proceed -- Assembly has a hard precondition on a real VO track (AssemblyError).")
        return 1

    print("\n=== Running the REAL Assembly two-stage ffmpeg pipeline ===")
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    captured: dict = {}

    def _upload(local_path: str) -> str:
        dst = OUTPUTS_DIR / "master_cut_live.mp4"
        shutil.copy(local_path, dst)
        captured["path"] = str(dst)
        return f"file://{dst}"

    result = asyncio.run(
        _assemble_master_cut_impl(
            shot_list, generated_shots, voiceover, winning_script, JOB_ID,
            download_fn=_make_download_fn(), upload_fn=_upload,
        )
    )

    print(f"\n=== DONE ===")
    print(f"  master_cut_uri (local file:// stand-in): {result.master_cut_uri}")
    print(f"  shot_count (real segments rendered): {result.shot_count}")
    print(f"  total_duration_sec (ffprobe'd): {result.total_duration_sec}")
    print(f"  degraded_beats: {result.degraded_beats}")

    out_path = captured["path"]
    probe = _ffprobe(out_path)
    print(f"\n=== ffprobe of {out_path} ===")
    print(json.dumps(probe, indent=2))

    video_stream = next(s for s in probe["streams"] if s["codec_type"] == "video")
    audio_stream = next((s for s in probe["streams"] if s["codec_type"] == "audio"), None)
    print("\n=== Verification summary ===")
    print(f"  duration: {probe['format']['duration']}s (expected ~= {winning_script['beats'][-1]['t_end']}s of script, "
          f"actual VO-conformed duration may differ slightly -- real measured TTS durations, not script t_end)")
    print(f"  resolution: {video_stream['width']}x{video_stream['height']} @ {video_stream['r_frame_rate']} fps, pix_fmt={video_stream['pix_fmt']}")
    print(f"  video codec: {video_stream['codec_name']}")
    print(f"  audio present: {audio_stream is not None} ({audio_stream.get('codec_name') if audio_stream else 'N/A'})")

    # Extract a handful of representative frames for direct visual inspection
    # (burned captions + the held-frame content over the orphaned CTA beat).
    frame_times = []
    running = 0.0
    # Sample near the start of each beat's window using the REAL captions
    # timing this run actually produced isn't available here (only the
    # planned script beats are) -- sample evenly across the real total
    # duration instead, which still lands inside every beat given contiguous
    # coverage.
    total = float(probe["format"]["duration"])
    for frac in (0.1, 0.4, 0.65, 0.9):
        frame_times.append(round(total * frac, 2))

    for i, t in enumerate(frame_times):
        frame_path = OUTPUTS_DIR / f"master_cut_live_frame_{i}_t{t}s.jpg"
        subprocess.run(
            ["ffmpeg", "-y", "-ss", str(t), "-i", out_path, "-frames:v", "1", str(frame_path)],
            check=True, capture_output=True,
        )
        print(f"  saved frame at t={t}s -> {frame_path}")

    print(f"\nFinished master cut saved locally at: {out_path}")
    print("Extracted frames saved for direct visual inspection (see printed paths above).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
