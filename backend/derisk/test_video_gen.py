"""
De-risk / reproduction script for Wan2.6-i2v-us image-to-video generation
(Phase 0, RR). docs/DERISK_VIDEO_GEN_RESULT.md references this exact path
(`backend/derisk/test_video_gen.py`) and `backend/derisk/outputs/results.json`
as the reproduction evidence behind its written GO verdict, but neither was
ever actually committed to the repo (confirmed via `git ls-files` during the
BUILD_TASKS.md audit) -- this is that script, reconstructed.

Reuses the real code path -- agents/video_gen_node.py's `_call_wan_video_gen`
-- not a hand-rolled second Wan call, and the real product photos already
present under backend/derisk/photos/. Uploads the chosen photo to OSS first
(reusing agents/_oss.py's shared put+sign primitive, no new upload logic):
the native video-synthesis endpoint needs a fetchable HTTP(S) URL, confirmed
live, unlike the OpenAI-compatible vision chat endpoint
derisk/test_truth_extractor.py uses (which accepts an inline data: URI).

COST WARNING: each clip is a REAL, BILLED Wan2.6-i2v-us generation
(duration_sec x RATE_720P, per agents/budget_gate.py -- ~$0.40 for one 5s
720p clip). Defaults to one clip; raise --max-clips deliberately, not by
accident.

Usage (from backend/):
    python -m derisk.test_video_gen
    python -m derisk.test_video_gen --max-clips 2 --camera-moves push_in orbit
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from agents._oss import _download_to_temp, _put_and_sign, oss_job_asset_key  # noqa: E402
from agents.budget_gate import RATE_720P  # noqa: E402
from agents.video_gen_node import (  # noqa: E402
    VideoGenAPIError,
    VideoGenTimeoutError,
    _call_wan_video_gen,
)

PHOTOS_DIR = Path(__file__).resolve().parent / "photos"
OUTPUTS_DIR = Path(__file__).resolve().parent / "outputs"

DURATION_SEC = 5.0
RESOLUTION = "720P"

_MOVE_DESCRIPTIONS = {
    "push_in": "a slow push-in toward the product",
    "orbit": "a slow orbit around the product",
    "static": "a held static shot with a faint breathing zoom",
    "pan": "a slow sideways pan across the product",
}

_NEGATIVE_PROMPT = (
    "blurry, distorted, morphing, melting, flickering, deformed shape, "
    "warped product, text/logo distortion, extra limbs"
)


def _build_prompt(camera_move: str) -> str:
    """Simple, ad-hoc Subject/Action/Camera/... prompt -- this script is a raw
    quality/reachability check of the MODEL, matching the original de-risk
    test's own scope (docs/DERISK_VIDEO_GEN_RESULT.md), not a test of
    video_gen_node.py's own structured `_build_prompt` (already covered by
    test_video_gen_node.py), which needs full Shot/Treatment/ProductTruth
    fixtures this standalone script has no reason to construct.
    """
    move_desc = _MOVE_DESCRIPTIONS.get(camera_move, "gentle motion")
    return (
        "Subject: the product shown in the reference photo.\n"
        f"Action/Motion: static product presentation, camera performs {move_desc}.\n"
        f"Camera: {camera_move}.\n"
        "Lighting: soft, even studio lighting.\n"
        "Composition: product centered, fills frame, no scene cut.\n"
        "Mood: clean, minimal, professional commercial.\n"
        "Quality: photorealistic, professional commercial cinematography, "
        "sharp focus, high detail, natural color, no artifacts."
    )


async def _upload_photo_to_oss(photo_path: Path) -> str:
    """Upload the chosen local photo to a job-level OSS namespace and return
    a signed, fetchable URL -- reuses agents/_oss.py's shared `_put_and_sign`
    primitive (same one agents/voiceover_caption_agent.py's OSS wrappers are
    built on), no new upload/signing code here.
    """
    key = oss_job_asset_key("derisk-videogen", photo_path.name)
    return await asyncio.to_thread(_put_and_sign, key, str(photo_path), "image/jpeg")


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-clips", type=int, default=1,
        help="hard cap on real, billed Wan calls this run makes (default 1)",
    )
    parser.add_argument(
        "--camera-moves", nargs="+", default=["push_in"],
        help="camera moves to test, one real clip generated per move, capped by --max-clips",
    )
    args = parser.parse_args()

    photo_paths = sorted(
        p for p in PHOTOS_DIR.glob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")
    )
    if not photo_paths:
        print(
            f"No photos found in {PHOTOS_DIR}. Drop a real product photo there and re-run.",
            file=sys.stderr,
        )
        return 1
    photo_path = photo_paths[0]

    moves = args.camera_moves[: args.max_clips]
    est_cost = len(moves) * DURATION_SEC * RATE_720P
    print(
        f"About to generate {len(moves)} REAL, BILLED Wan2.6-i2v-us clip(s) "
        f"(~${est_cost:.2f} total estimated) using {photo_path.name}."
    )
    print(f"Camera moves: {moves}\n")

    image_url = await _upload_photo_to_oss(photo_path)
    print(f"Uploaded reference photo to OSS: {image_url}\n")

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    for move in moves:
        prompt = _build_prompt(move)
        print(f"--- Generating clip: camera_move={move!r} ---")
        start = time.monotonic()
        try:
            video_url = await _call_wan_video_gen(
                image_url=image_url,
                prompt=prompt,
                negative_prompt=_NEGATIVE_PROMPT,
                duration_sec=DURATION_SEC,
                resolution=RESOLUTION,
            )
            elapsed = time.monotonic() - start
            print(f"OK in {elapsed:.1f}s: {video_url}")

            local_clip = await asyncio.to_thread(_download_to_temp, video_url)
            saved_path = OUTPUTS_DIR / f"videogen_{move}.mp4"
            # shutil.move, not os.replace/os.rename -- the source temp file and
            # OUTPUTS_DIR can be on different drives on Windows, which those
            # would reject with "cannot move the file to a different disk drive".
            shutil.move(local_clip, saved_path)
            print(f"Saved clip to {saved_path}\n")

            results.append(
                {
                    "camera_move": move,
                    "status": "success",
                    "latency_sec": round(elapsed, 1),
                    "provider_video_url": video_url,
                    "saved_clip": saved_path.name,
                }
            )
        except (VideoGenTimeoutError, VideoGenAPIError) as exc:
            elapsed = time.monotonic() - start
            print(f"FAILED after {elapsed:.1f}s: {type(exc).__name__}: {exc}\n", file=sys.stderr)
            results.append(
                {
                    "camera_move": move,
                    "status": "failed",
                    "latency_sec": round(elapsed, 1),
                    "error_type": type(exc).__name__,
                    "error_detail": str(exc),
                }
            )

    summary = {
        "model": os.environ.get("MODEL_VIDEO", "<unset>"),
        "reference_photo": photo_path.name,
        "resolution": RESOLUTION,
        "duration_sec": DURATION_SEC,
        "results": results,
    }
    out_path = OUTPUTS_DIR / "results.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"Saved summary to {out_path}")

    n_ok = sum(1 for r in results if r["status"] == "success")
    print(f"\n{n_ok}/{len(results)} clip(s) succeeded.")
    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
