"""
Fresh targeted i2v verification for wan2.7-i2v on Singapore (DASHSCOPE_INTL).

Previous test (derisk/test_wan27_i2v_intl.py) got task_status=FAILED -- most
likely because the reference image URL was from oss-us-east-1 (regional US
endpoint), which Singapore's Wan service may not be able to fetch. This test
re-uploads the photo fresh and uses a longer poll timeout.

Also validates: does Wan 2.7 actually have no prompt-length truncation ceiling?
Tests with a deliberately long prompt (~1800 chars, well over 2.6's ~1400 limit)
and checks whether the output quality/content matches.

Usage (from backend/):
    python -m derisk.test_wan27_i2v_verify
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path

import httpx
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from agents._oss import oss_job_asset_key  # noqa: E402

OUTPUTS_DIR = Path(__file__).resolve().parent / "outputs"
PHOTOS_DIR = Path(__file__).resolve().parent / "photos"
JOB_ID = "wan27-i2v-verify"

# Deliberately long prompt (~1800 chars) to test the no-char-limit claim.
# Wan 2.6 would silently truncate this; if Wan 2.7 reproduces the full
# prompt content in the output, the claim is confirmed.
LONG_PROMPT = (
    "Subject: A deep saddle-brown full-grain leather backpack with a smooth, "
    "slightly waxy surface finish and a debossed geometric shield logo centered "
    "on the upper front panel. The leather shows a rich, multi-tonal warm brown "
    "with natural pull-up highlights where the material is under tension.\n\n"
    "Action/Motion: The camera holds a slow, deliberate push-in toward the front "
    "face of the backpack. The bag rests on a clean wooden surface. As the camera "
    "moves closer, the natural grain of the leather becomes increasingly visible -- "
    "the surface texture resolves from a smooth uniform brown into a rich, pebbled "
    "grain with micro-variations in tone. The debossed logo catches a side light, "
    "its shadow deepening as the camera advances. The leather's natural luster "
    "shifts subtly as the viewing angle changes. No movement in the bag itself -- "
    "only the camera advances.\n\n"
    "Camera: Slow, deliberate push-in. Start on a medium shot with the full bag "
    "visible, end on a close-up of the front panel with the logo and upper pocket "
    "filling the frame. Speed: approximately 8% scale increase over the full clip "
    "duration. No shake, no rack focus, no cut.\n\n"
    "Lighting: Warm, directional studio lighting from camera-left at roughly "
    "45 degrees, creating a soft shadow that falls diagonally across the front "
    "face and emphasizes the leather's three-dimensional surface texture. A subtle "
    "fill light from camera-right prevents the shadow side from losing detail. "
    "The overall light temperature is warm (approximately 3200K), giving the "
    "leather a rich, amber-gold tone in the highlights.\n\n"
    "Composition: The backpack is centered in the frame and fills approximately "
    "70% of the frame height on the opening shot, growing to fill 90% by the end "
    "of the push-in. The bag is slightly angled (about 10 degrees off-center) to "
    "show both the front panel and a hint of the left side profile. The clean "
    "wooden surface provides a warm, neutral base. No text, no watermarks, no "
    "other objects in frame.\n\n"
    "Mood: Premium, considered, confident. The pace is unhurried -- this is a "
    "product that rewards close attention. The feel is closer to a craftsmanship "
    "documentary than a commercial: clean, real, no artificial perfection.\n\n"
    "Quality: Photorealistic, ultra-sharp focus across the leather surface, "
    "cinematic color grading, no motion blur, no compression artifacts, no "
    "lens distortion. The leather texture must be clearly legible in the final "
    "frame -- grain, pebbling, and highlight variation all visible."
)

NEGATIVE_PROMPT = (
    "text, watermark, logo overlay, person, face, hands, ugly, deformed, "
    "blurry, low quality, pixelated, compression artifacts, distorted geometry, "
    "morphing, dissolving, color shifting, overexposed, underexposed, "
    "motionless, near-static, timelapse, stop motion, choppy motion, frame stutter"
)

DURATION_SEC = 5
RESOLUTION = "720P"
POLL_TIMEOUT_SEC = 300  # Wan 2.7 may take longer than 2.6


async def main() -> int:
    photo_path = PHOTOS_DIR / "backpack_1.jpg"
    if not photo_path.exists():
        print(f"ERROR: {photo_path} not found")
        return 1

    # Upload to the Singapore OSS bucket so Wan 2.7 (Singapore endpoint) can
    # fetch the reference image without cross-region access failure.
    import oss2
    auth = oss2.Auth(os.environ["OSS_ACCESS_KEY_ID"], os.environ["OSS_ACCESS_KEY_SECRET"])
    sg_bucket = oss2.Bucket(auth, os.environ["OSS_INTL_ENDPOINT"], os.environ["OSS_INTL_BUCKET"])
    key = oss_job_asset_key(JOB_ID, "photo_1.jpg")
    sg_bucket.put_object_from_file(key, str(photo_path), headers={"Content-Type": "image/jpeg"})
    image_url = sg_bucket.sign_url("GET", key, 3600, slash_safe=True)
    print(f"Photo uploaded to SG bucket, signed URL: {image_url[:80]}...")
    print(f"Prompt length: {len(LONG_PROMPT)} chars (Wan 2.6 limit was ~1400)")

    api_key = os.environ["DASHSCOPE_VIDEO_INTL_API_KEY"]
    base_url = os.environ["DASHSCOPE_INTL_VIDEO_BASE_URL"]
    model = os.environ.get("MODEL_VIDEO_INTL", "wan2.7-i2v-2026-04-25")
    # Force HTTPS -- Wan 2.7 rejects HTTP image URLs
    image_url = image_url.replace("http://", "https://", 1)

    print(f"\nModel: {model}  |  Endpoint: {base_url}")
    print(f"Credential: DASHSCOPE_VIDEO_INTL_API_KEY (Singapore)")
    print(f"Poll timeout: {POLL_TIMEOUT_SEC}s\n")

    result: dict = {
        "model": model,
        "base_http_api_url": base_url,
        "credential": "DASHSCOPE_VIDEO_INTL_API_KEY",
        "prompt_length": len(LONG_PROMPT),
        "image_url": image_url,
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "X-DashScope-Async": "enable",
    }

    start = time.monotonic()
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{base_url}/services/aigc/video-generation/video-synthesis",
                headers=headers,
                json={
                    "model": model,
                    "input": {
                        "prompt": LONG_PROMPT,
                        "negative_prompt": NEGATIVE_PROMPT,
                        "media": [{"type": "first_frame", "url": image_url}],
                    },
                    "parameters": {"duration": DURATION_SEC, "resolution": RESOLUTION},
                },
                timeout=30,
            )
            if resp.status_code != 200:
                raise RuntimeError(f"Submit failed: HTTP {resp.status_code}: {resp.text[:300]}")

            task_id = resp.json()["output"]["task_id"]
            print(f"Task submitted: {task_id}")

            video_url = None
            task_status = "PENDING"
            deadline = time.monotonic() + POLL_TIMEOUT_SEC
            while time.monotonic() < deadline:
                await asyncio.sleep(10)
                poll = await client.get(
                    f"{base_url}/tasks/{task_id}",
                    headers={"Authorization": f"Bearer {api_key}"},
                    timeout=30,
                )
                data = poll.json()
                task_status = data.get("output", {}).get("task_status", "?")
                print(f"  [{round(time.monotonic() - start)}s] {task_status}")
                if task_status == "SUCCEEDED":
                    video_url = data["output"].get("video_url", "")
                    break
                if task_status == "FAILED":
                    code = data.get("output", {}).get("code", "")
                    message = data.get("output", {}).get("message", "")
                    raise RuntimeError(f"Task FAILED: code={code!r}, message={message!r}")

        elapsed = time.monotonic() - start
        result.update({
            "elapsed_sec": round(elapsed, 1),
            "task_status": task_status,
            "video_url": video_url or "",
        })

        if task_status == "SUCCEEDED" and video_url:
            print(f"SUCCEEDED in {elapsed:.1f}s")
            print(f"   video_url: {video_url}")
            outcome = "works"

            import urllib.request
            out_path = OUTPUTS_DIR / "wan27_i2v_verify.mp4"
            print(f"   Downloading to {out_path}...")
            urllib.request.urlretrieve(video_url, out_path)
            print(f"   Downloaded: {out_path.stat().st_size / 1024:.0f} KB")
            result["local_path"] = str(out_path)
        else:
            print(f"FAILED -- task_status={task_status!r}")
            outcome = "reachable_no_success"

    except asyncio.TimeoutError:
        elapsed = time.monotonic() - start
        result.update({"elapsed_sec": round(elapsed, 1), "error": "timeout"})
        print(f"TIMEOUT after {elapsed:.1f}s")
        outcome = "timeout"
    except Exception as exc:  # noqa: BLE001
        elapsed = time.monotonic() - start
        result.update({"elapsed_sec": round(elapsed, 1), "error_type": type(exc).__name__, "error_detail": str(exc)})
        print(f"ERROR: {type(exc).__name__}: {exc}")
        outcome = "error"

    result["outcome"] = outcome
    out_json = OUTPUTS_DIR / "wan27_i2v_verify_result.json"
    out_json.write_text(json.dumps(result, indent=2))
    print(f"\nResult saved to {out_json}")
    return 0 if outcome == "works" else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
