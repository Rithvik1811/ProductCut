"""
PHASE 1 CHECKPOINT (branch `video-gen-fidelity`) -- live re-generation of the
real failed shot s4 (derisk/outputs/full_pipeline_live_result.json) under the
NEW prompt-construction logic, to confirm the near-static-hand failure is
fixed before Phase 2/3 are attempted.

Mirrors backend/derisk/test_identity_fix_live.py's established convention
(real product truths/treatment, a hand-built Shot fixture, the real
_build_prompt + _call_wan_video_gen path) but targets the SPECIFIC confirmed
failure: shot s4's prompt was silently truncated at Wan's 1,500-char
server-side limit, with the surviving ~1,500 chars ~90% static appearance/
identity/camera/lighting text and only ~7% action -- the rendered clip showed
almost no motion (a static hand resting on the bag) despite the prompt
describing "a person slowly slings the bag, and a hand gently adjusts the
3 cm wide black shoulder strap across her collarbone."

Real product truths/treatment reused from that saved job's own JSON (the
saved job does not persist `product_truths`, so t6/t1 are reconstructed
VERBATIM from the treatment's own visual_approach/why_not_generic text for
beat_index 3, matching what the real Product Truth Extractor would have
produced for this exact job). Photos: derisk/photos/backpack_1.jpg,
backpack_2.jpg (the same real leather-backpack photo set already in this
repo from test_identity_fix_live.py; reference_image_id "photo_2" matches
the original failed shot's own reference).

The Shot's `description` is hand-written in the NEW post-fix style (Action/
Motion ONLY, decisive verbs, no camera/lighting/identity duplication) --
i.e. what agents/shot_list_agent.py's fixed Call B system prompt now asks
the model to produce -- so this checkpoint tests the FULL new prompt
pipeline (shot_list_agent's leaner description + video_gen_node's budget
enforcement + compressed identity clause + fixed-camera phrasing + action-
urgency clause), not just one half of it.

COST WARNING: ONE real, billed Wan2.6-i2v-us generation (~$0.35-0.50 for a
4s 720p clip per agents/budget_gate.py's RATE_720P). Deliberately
single-shot -- do not loop this script.

Usage (from backend/):
    python -m derisk.test_phase1_checkpoint_s4_live
"""
from __future__ import annotations

import asyncio
import json
import subprocess
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from agents._oss import _download_to_temp, _put_and_sign, oss_job_asset_key  # noqa: E402
from agents.shot_list_agent import HUMAN_SHOT_NEGATIVE_EXTRA, NEGATIVE_PROMPT_BOILERPLATE  # noqa: E402
from agents.video_gen_node import (  # noqa: E402
    VideoGenAPIError,
    VideoGenTimeoutError,
    _build_prompt,
    _call_wan_video_gen,
)

PHOTOS_DIR = Path(__file__).resolve().parent / "photos"
OUTPUTS_DIR = Path(__file__).resolve().parent / "outputs"
JOB_ID = "phase1-checkpoint-s4"

DURATION_SEC = 4.0
RESOLUTION = "720P"

# Reconstructed verbatim from full_pipeline_live_result.json's own
# treatment.beat_treatments[3] (the "proof" beat s4 realizes) -- the real
# job's own words for the strap fact, not an invented one.
TRUTHS = [
    {
        "truth_id": "t1",
        "fact": (
            "a softly rounded rectangular block, roughly two hand-widths wide, "
            "with two gusseted side pockets, rich medium-brown matte leather finish"
        ),
        "category": "form_factor",
        "source": "photo_1",
    },
    {
        "truth_id": "t6",
        "fact": "a 3 cm wide black shoulder strap that stretches across the collarbone",
        "category": "construction_detail",
        "source": "photo_2",
    },
]

TREATMENT = {
    "director_persona": (
        "Intimate, tactile realism. Grounded in the script's focus on the physical "
        "reality of wear and tear, the camera feels close and observational, almost "
        "like the viewer's own hands examining the bag."
    ),
    "color_story": (
        "Warm, earthy, and high-contrast. Deep medium-browns and rich tans of the "
        "vegetable-tanned leather are punctuated by the sharp metallic glint of brass "
        "and silver rivets."
    ),
    "pacing_philosophy": (
        "A contemplative, deliberate build that accelerates into confident momentum."
    ),
    "beat_treatments": [
        {
            "beat_index": 3,
            "beat_function": "proof",
            "script_quote": "She slings it over her shoulder, embracing the wear.",
            "truth_fact_id": "t6",
            "visual_approach": (
                "A dynamic, over-the-shoulder mid-shot in natural daylight showing the "
                "bag settling against her back as she walks, with the 3 cm wide black "
                "shoulder strap stretching comfortably across her collarbone."
            ),
            "why_not_generic": (
                "This grounds the proof in a real daily-life moment of carrying the bag, "
                "specifically highlighting the ergonomic 3 cm black shoulder strap."
            ),
        },
    ],
}

# NEW-STYLE description (post-fix): Action/Motion ONLY, decisive verbs, no
# camera/lighting/identity duplication -- what agents/shot_list_agent.py's
# fixed Call B system prompt now asks the model to produce, in place of the
# ORIGINAL failed shot's hedged "A person slowly slings the bag, and a hand
# gently adjusts the 3 cm wide black shoulder strap across her collarbone"
# (which also had camera/lighting/identity text duplicated into it).
NEW_DESCRIPTION = (
    "A hand grips the 3 cm wide black shoulder strap and lifts the bag up onto "
    "the shoulder in one continuous motion. The strap settles across the "
    "collarbone as the bag comes to rest snugly against the back, and the hand "
    "makes one final adjustment to the strap."
)

# ORIGINAL failed shot's description, verbatim from
# derisk/outputs/full_pipeline_live_result.json's shot s4 -- kept here only
# for the printed side-by-side comparison, never sent to Wan.
ORIGINAL_FAILED_DESCRIPTION = (
    "Rich medium-brown vegetable-tanned leather backpack. A person slowly slings "
    "the bag, and a hand gently adjusts the 3 cm wide black shoulder strap across "
    "her collarbone. The camera executes a slow push-in. Warm, earthy, "
    "high-contrast lighting with deep medium-browns and rich tans punctuated by "
    "sharp metallic glints, utilizing soft but present shadows to emphasize the "
    "topography of the grain and subtle scuffs. The composition frames the bag "
    "on the right third. Cinematic 8k. The product keeps its exact size and "
    "proportions relative to the body; silhouette unchanged. The product remains "
    "fully recognizable when partially covered; color and material never change. "
    "Product stays centered, never leaves frame, no scene cut. Preserve product "
    "shape, keep label text, keep proportions."
)


def _ffprobe_duration(path: str) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
        capture_output=True, text=True, check=True,
    )
    return float(out.stdout.strip())


def _extract_frames(video_path: str, out_prefix: Path, n: int = 4) -> list[str]:
    duration = _ffprobe_duration(video_path)
    frames = []
    for i in range(n):
        t = round(duration * (i / max(n - 1, 1)) * 0.95 + 0.05, 3)  # spread across, avoid exact 0/end
        out_path = f"{out_prefix}_{i}_t{t:.2f}s.jpg"
        subprocess.run(
            ["ffmpeg", "-y", "-ss", str(t), "-i", video_path, "-frames:v", "1", out_path],
            check=True, capture_output=True,
        )
        frames.append(out_path)
    return frames


async def main() -> int:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    photo_paths = [PHOTOS_DIR / "backpack_1.jpg", PHOTOS_DIR / "backpack_2.jpg"]
    for p in photo_paths:
        if not p.exists():
            print(f"ERROR: missing {p}")
            return 1

    print("=== Step 1: upload real photos to OSS ===")
    photo_urls = []
    for i, p in enumerate(photo_paths, start=1):
        key = oss_job_asset_key(JOB_ID, f"photo_{i}.jpg")
        url = _put_and_sign(key, str(p), "image/jpeg")
        photo_urls.append(url)
        print(f"  photo_{i} -> {url[:80]}...")

    print("\n=== Step 2: build the real s4-shaped Shot fixture (NEW-style description) ===")
    shot = {
        "shot_id": "s4",
        "t_start": 12.0,
        "t_end": 16.0,
        "beat_role": "proof",
        "description": NEW_DESCRIPTION,
        "shot_type": "worn_in_use",
        "camera_move": "push_in",
        "framing": "rule_of_thirds_right",
        "lighting": (
            "Warm, earthy, high-contrast lighting with deep medium-browns and rich "
            "tans punctuated by sharp metallic glints, utilizing soft but present "
            "shadows to emphasize the topography of the grain and subtle scuffs."
        ),
        "negative_prompt": f"{NEGATIVE_PROMPT_BOILERPLATE}, {HUMAN_SHOT_NEGATIVE_EXTRA}",
        "reference_image_id": "photo_2",
        "text_overlay_zone": "none",
        "duration_sec": DURATION_SEC,
        "allocated_budget": 5.0,
        "voiceover_line": "She slings it over her shoulder, embracing the wear.",
        "justification": {
            "script_quote": "She slings it over her shoulder, embracing the wear.",
            "truth_fact_id": "t6",
            "treatment_ref": 3,
        },
        "status": "pending",
        "retry_count": 0,
    }

    prompt = _build_prompt(shot, TRUTHS, TREATMENT)
    print(f"  NEW prompt ({len(prompt)} chars):\n{'-' * 60}\n{prompt}\n{'-' * 60}")
    print(f"\n  For comparison, the ORIGINAL failed description was {len(ORIGINAL_FAILED_DESCRIPTION)} chars")
    print("  (and, per the real job's logged warning, the ASSEMBLED prompt was 2388 chars")
    print("  total -- silently truncated by Wan's 1,500-char server-side limit).")

    print("\n=== Step 3: REAL Wan2.6-i2v-us generation (BILLED, ~$0.35-0.50) ===")
    start = time.monotonic()
    try:
        video_url = await _call_wan_video_gen(
            image_url=photo_urls[1],  # photo_2, matches reference_image_id
            prompt=prompt,
            negative_prompt=shot["negative_prompt"],
            duration_sec=DURATION_SEC,
            resolution=RESOLUTION,
        )
    except (VideoGenTimeoutError, VideoGenAPIError) as exc:
        print(f"  *** FAIL: Wan generation failed: {exc} ***")
        (OUTPUTS_DIR / "phase1_checkpoint_s4_result.json").write_text(
            json.dumps({"outcome": "wan_generation_failed", "error": str(exc), "prompt": prompt}, indent=2)
        )
        return 1
    elapsed = time.monotonic() - start
    print(f"  SUCCEEDED in {elapsed:.1f}s -> {video_url}")

    print("\n=== Step 4: download clip + extract frames across its duration ===")
    local_path = _download_to_temp(video_url)
    duration = _ffprobe_duration(local_path)
    print(f"  downloaded to {local_path}, real duration {duration:.2f}s")

    frame_prefix = OUTPUTS_DIR / "phase1_checkpoint_s4"
    frames = _extract_frames(local_path, frame_prefix, n=4)
    for f in frames:
        print(f"  frame -> {f}")

    # Keep a permanent copy of the clip itself alongside the frames.
    clip_copy = OUTPUTS_DIR / "phase1_checkpoint_s4.mp4"
    clip_copy.write_bytes(Path(local_path).read_bytes())

    result = {
        "outcome": "ran_end_to_end",
        "photo_urls": photo_urls,
        "prompt": prompt,
        "prompt_len": len(prompt),
        "video_url": video_url,
        "local_clip": str(clip_copy),
        "real_duration_sec": duration,
        "generation_elapsed_sec": round(elapsed, 1),
        "frames": frames,
        "new_description": NEW_DESCRIPTION,
        "original_failed_description": ORIGINAL_FAILED_DESCRIPTION,
    }
    out_path = OUTPUTS_DIR / "phase1_checkpoint_s4_result.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\nFull result saved to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
