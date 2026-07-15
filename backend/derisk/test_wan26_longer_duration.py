"""
Real, one-shot test: does `wan2.6-i2v-us` (the model this pipeline actually
uses, on the proven-working `dashscope-us` config) accept a LONGER duration
than the 3-5s range `agents/shot_list_agent.py`'s MAX_SHOT_DURATION_SEC=5.0
currently clamps every shot to?

That clamp is a deliberate PROJECT choice ("drift compounds over longer
single-shot durations" -- shot_list_agent.py's own comment), not a discovered
hard API ceiling. Alibaba's own docs (fetched live) state duration accepts
"5, 10, 15" for this exact model id -- this test asks for 10s and measures
the REAL resulting clip via ffprobe, resolving whether that's accurate for
real generations (our own session already has many successful REAL
sub-5s/non-enum generations, e.g. 3.01s/4.03s measured clips, which
contradicts a strict 5/10/15-only reading of the docs -- so this test checks
the OTHER direction: can we get MORE than 5s if asked).

Reuses the real, unmodified `agents.video_gen_node._call_wan_video_gen` --
not a hand-rolled duplicate -- so this is a direct test of the actual
production code path.

Usage (from backend/):
    python -m derisk.test_wan26_longer_duration
"""
from __future__ import annotations

import asyncio
import json
import subprocess
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from agents._oss import _put_and_sign, oss_job_asset_key  # noqa: E402
from agents.shot_list_agent import NEGATIVE_PROMPT_BOILERPLATE  # noqa: E402
from agents.video_gen_node import (  # noqa: E402
    VideoGenAPIError,
    VideoGenTimeoutError,
    _call_wan_video_gen,
)

PHOTOS_DIR = Path(__file__).resolve().parent / "photos"
OUTPUTS_DIR = Path(__file__).resolve().parent / "outputs"
JOB_ID = "wan26-duration-spike"

REQUESTED_DURATION_SEC = 10
RESOLUTION = "720P"

PROMPT = (
    "Subject: The product: a deep, rounded rectangular block with a rich "
    "medium-brown matte leather finish and a debossed logo on the front panel.\n"
    "Action/Motion: A person slowly walks toward the camera, the backpack worn "
    "over one shoulder, swaying gently with each step; the camera holds steady "
    "as they approach and come to a stop, then turn slightly to show the side "
    "profile of the bag before the clip ends.\n"
    "Camera: static.\n"
    "Lighting: soft, even studio lighting.\n"
    "Composition: worn in use, rule of thirds framing.\n"
    "Mood: clean, confident, everyday commercial.\n"
    "Quality: photorealistic, sharp focus, no artifacts."
)


def _probe_duration(path: str) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", path],
        check=True, capture_output=True, text=True,
    )
    return float(out.stdout.strip())


async def main() -> int:
    photo_path = PHOTOS_DIR / "backpack_1.jpg"
    if not photo_path.exists():
        print(f"ERROR: {photo_path} not found")
        return 1

    key = oss_job_asset_key(JOB_ID, "photo_1.jpg")
    image_url = _put_and_sign(key, str(photo_path), "image/jpeg")
    print(f"Reference photo: {image_url[:80]}...")
    print(f"Requesting duration_sec={REQUESTED_DURATION_SEC} (vs this pipeline's normal 3-5s clamp) "
          f"on the REAL production _call_wan_video_gen path.\n")

    result: dict = {"requested_duration_sec": REQUESTED_DURATION_SEC, "resolution": RESOLUTION}
    start = time.monotonic()
    try:
        video_url = await _call_wan_video_gen(
            image_url=image_url,
            prompt=PROMPT,
            negative_prompt=NEGATIVE_PROMPT_BOILERPLATE,
            duration_sec=REQUESTED_DURATION_SEC,
            resolution=RESOLUTION,
        )
        elapsed = time.monotonic() - start
        print(f"SUCCEEDED in {elapsed:.1f}s -> {video_url}")
        result.update({"outcome": "succeeded", "elapsed_sec": round(elapsed, 1), "video_url": video_url})
    except (VideoGenTimeoutError, VideoGenAPIError) as exc:
        elapsed = time.monotonic() - start
        print(f"FAILED after {elapsed:.1f}s: {exc}")
        result.update({"outcome": "failed", "elapsed_sec": round(elapsed, 1), "error": str(exc)})
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        (OUTPUTS_DIR / "wan26_longer_duration_result.json").write_text(json.dumps(result, indent=2))
        return 1

    import urllib.request
    local_path = str(OUTPUTS_DIR / "wan26_10s_test.mp4")
    urllib.request.urlretrieve(video_url, local_path)
    real_duration = _probe_duration(local_path)
    print(f"\nREAL measured clip duration (ffprobe): {real_duration:.2f}s "
          f"(requested {REQUESTED_DURATION_SEC}s)")
    result["real_measured_duration_sec"] = round(real_duration, 2)
    result["local_path"] = local_path

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUTS_DIR / "wan26_longer_duration_result.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"Saved result to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
