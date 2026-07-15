"""
De-risk / availability spike for `wan2.7-i2v` (IMAGE-to-video) on the exact
proven-working configuration from `derisk/test_wan27_t2v_intl.py` (standard
Singapore gateway, native `/api/v1` path, TTS workspace key) -- that spike
confirmed `wan2.7-t2v` reaches SUCCEEDED on this exact key+host, so this test
targets the one remaining unknown: does the SAME auth/routing accept the i2v
model id + a real img_url, or is i2v specifically gated differently from t2v
on this workspace?

Uses a real backpack photo already in `derisk/photos/` (uploaded fresh to OSS
for a fetchable URL, matching every other real i2v test this session).

Time-boxed, best-effort -- AT MOST ONE real, billed-if-it-succeeds API call.

Usage (from backend/):
    python -m derisk.test_wan27_i2v_intl
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

import dashscope  # noqa: E402
from dashscope.aigc.video_synthesis import AioVideoSynthesis  # noqa: E402

from agents._oss import _put_and_sign, oss_job_asset_key  # noqa: E402

OUTPUTS_DIR = Path(__file__).resolve().parent / "outputs"
PHOTOS_DIR = Path(__file__).resolve().parent / "photos"
JOB_ID = "wan27-i2v-spike"

MODEL = os.getenv("MODEL_VIDEO", "wan2.7-i2v")
PROMPT = (
    "Subject: a rich medium-brown leather backpack with a debossed logo.\n"
    "Action/Motion: the camera holds a static shot with a faint breathing zoom.\n"
    "Camera: static.\n"
    "Lighting: soft, even studio lighting.\n"
    "Composition: backpack centered, fills frame.\n"
    "Mood: clean, minimal, professional commercial.\n"
    "Quality: photorealistic, sharp focus, no artifacts."
)
NEGATIVE_PROMPT = "blurry, distorted, morphing, deformed shape, low quality"
DURATION_SEC = 5
RESOLUTION = "720P"


async def main() -> int:
    photo_path = PHOTOS_DIR / "backpack_1.jpg"
    if not photo_path.exists():
        print(f"ERROR: {photo_path} not found")
        return 1

    key = oss_job_asset_key(JOB_ID, "photo_1.jpg")
    image_url = _put_and_sign(key, str(photo_path), "image/jpeg")
    print(f"Reference photo uploaded: {image_url[:80]}...")

    api_key = os.environ["DASHSCOPE_TTS_API_KEY"]
    base_url = os.environ["DASHSCOPE_INTL_VIDEO_BASE_URL"]
    dashscope.api_key = api_key
    dashscope.base_http_api_url = base_url

    print(f"Probing model={MODEL!r} on base_http_api_url={base_url!r} (TTS workspace key, Singapore, real i2v with img_url)")
    print("This is AT MOST ONE real API call. If it reaches SUCCEEDED, it is a "
          "real, billed generation.\n")

    result: dict = {"model": MODEL, "base_http_api_url": base_url, "credential": "DASHSCOPE_TTS_API_KEY", "mode": "i2v_real_photo", "image_url": image_url}
    start = time.monotonic()
    try:
        response = await asyncio.wait_for(
            AioVideoSynthesis.call(
                model=MODEL,
                prompt=PROMPT,
                negative_prompt=NEGATIVE_PROMPT,
                img_url=image_url,
                duration=DURATION_SEC,
                resolution=RESOLUTION,
                prompt_extend=False,
            ),
            timeout=180.0,
        )
        elapsed = time.monotonic() - start
        task_status = response.output.task_status if response.output else None
        result.update(
            {
                "elapsed_sec": round(elapsed, 1),
                "status_code": response.status_code,
                "code": getattr(response, "code", None),
                "message": getattr(response, "message", None),
                "task_status": task_status,
                "video_url": getattr(response.output, "video_url", None) if response.output else None,
                "raw_output_repr": repr(response.output) if response.output else None,
            }
        )
        if response.status_code == 200 and task_status == "SUCCEEDED":
            print(f"RESULT: works -- SUCCEEDED in {elapsed:.1f}s, video_url={result['video_url']}")
            outcome = "works"
        else:
            print(
                f"RESULT: reachable but did not succeed -- status_code={response.status_code}, "
                f"code={result['code']!r}, message={result['message']!r}, task_status={task_status!r}"
            )
            print(f"raw output: {result['raw_output_repr']}")
            outcome = "reachable_no_success"
    except asyncio.TimeoutError:
        elapsed = time.monotonic() - start
        result.update({"elapsed_sec": round(elapsed, 1), "error": "timeout"})
        print(f"RESULT: timeout after {elapsed:.1f}s -- inconclusive")
        outcome = "timeout"
    except Exception as exc:  # noqa: BLE001 -- this script's whole job is to report whatever comes back
        elapsed = time.monotonic() - start
        msg = str(exc)
        result.update({"elapsed_sec": round(elapsed, 1), "error_type": type(exc).__name__, "error_detail": msg})
        code_match = None
        for marker in ('"code": "', "'code': '"):
            idx = msg.find(marker)
            if idx != -1:
                start_i = idx + len(marker)
                end_i = msg.find(marker[-1], start_i)
                if end_i != -1:
                    code_match = msg[start_i:end_i]
                break
        lowered = msg.lower()
        if code_match == "AccessDenied":
            outcome = "access_denied"
        elif code_match and "model" in code_match.lower():
            outcome = "model_not_found"
        elif "region" in lowered or "unsupported" in lowered:
            outcome = "region_error"
        elif "workspace" in lowered:
            outcome = "workspace_error"
        elif "path" in lowered:
            outcome = "path_error"
        else:
            outcome = "other_error"
        print(f"RESULT: {outcome} (code={code_match!r}) -- {type(exc).__name__}: {msg}")

    result["outcome"] = outcome
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUTS_DIR / "wan27_i2v_intl_result.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\nSaved raw result to {out_path}")
    return 0 if outcome == "works" else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
