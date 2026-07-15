"""
De-risk / availability spike for `wan2.7-i2v` (Task 5, video-gen fidelity fix
branch). docs cite wan2.7-i2v as offering `first_frame`/`last_frame`
conditioning (per the official Alibaba DashScope docs), which is a plausible
FUTURE improvement for the same identity-drift bug this branch fixes (an
explicit first/last-frame anchor is a stronger identity constraint than a
single `img_url` + text prompt) -- but that is a follow-up idea, NOT something
this branch implements. This script only answers one narrow question: is
`wan2.7-i2v` reachable at all on this account's `dashscope-us` region/
credentials right now?

Time-boxed, best-effort, LOW priority relative to Tasks 1-4 (per this branch's
own task list) -- this script makes AT MOST ONE real, billed-if-it-succeeds
API call and reports the raw outcome. It deliberately does NOT loop/retry
against the paid API; if it fails, the exact status_code/code/message is
printed and saved so a human can decide whether to investigate further.

Reuses agents/video_gen_node.py's already-verified `dashscope-us` region/
base-URL configuration (`DASHSCOPE_VIDEO_BASE_URL`) rather than re-deriving
it -- see that module's docstring / docs/DERISK_VIDEO_GEN_RESULT.md §5 for why
the native SDK needs the *native* `/api/v1` base, not the OpenAI-compatible one.

Does NOT use a local product photo (this repo's derisk/photos/ directory does
not exist in this environment) -- a stable, public, fetchable JPEG URL is used
instead, since this spike only cares about MODEL reachability, not output
quality (that's the job of derisk/test_video_gen.py, already covered).

Usage (from backend/):
    python -m derisk.test_wan27_availability
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

import dashscope  # noqa: E402
from dashscope.aigc.video_synthesis import AioVideoSynthesis  # noqa: E402

OUTPUTS_DIR = Path(__file__).resolve().parent / "outputs"

MODEL = "wan2.7-i2v"
# A stable, public, fetchable JPEG -- no local photo asset needed, and no OSS
# upload step, since this spike only checks model/region REACHABILITY, not
# output quality (already covered by derisk/test_video_gen.py).
IMAGE_URL = "https://httpbin.org/image/jpeg"
PROMPT = (
    "Subject: a plain product sitting on a neutral surface.\n"
    "Action/Motion: the camera holds a static shot with a faint breathing zoom.\n"
    "Camera: static.\n"
    "Lighting: soft, even studio lighting.\n"
    "Composition: product centered, fills frame.\n"
    "Mood: clean, minimal, professional commercial.\n"
    "Quality: photorealistic, sharp focus, no artifacts."
)
NEGATIVE_PROMPT = "blurry, distorted, morphing, deformed shape, low quality"
DURATION_SEC = 5
RESOLUTION = "720P"


async def main() -> int:
    dashscope.api_key = os.environ["DASHSCOPE_API_KEY"]
    video_base_url = os.getenv("DASHSCOPE_VIDEO_BASE_URL")
    if video_base_url:
        dashscope.base_http_api_url = video_base_url

    print(f"Probing model={MODEL!r} on base_http_api_url={dashscope.base_http_api_url!r}")
    print("This is AT MOST ONE real API call. If it reaches SUCCEEDED, it is a "
          "real, billed generation (same cost class as wan2.6-i2v-us).\n")

    result: dict = {"model": MODEL, "base_http_api_url": dashscope.base_http_api_url}
    start = time.monotonic()
    try:
        response = await asyncio.wait_for(
            AioVideoSynthesis.call(
                model=MODEL,
                prompt=PROMPT,
                negative_prompt=NEGATIVE_PROMPT,
                img_url=IMAGE_URL,
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
            outcome = "reachable_no_success"
    except asyncio.TimeoutError:
        elapsed = time.monotonic() - start
        result.update({"elapsed_sec": round(elapsed, 1), "error": "timeout"})
        print(f"RESULT: timeout after {elapsed:.1f}s -- inconclusive (network/queue latency, not necessarily model_not_found)")
        outcome = "timeout"
    except Exception as exc:  # noqa: BLE001 -- this script's whole job is to report whatever comes back
        elapsed = time.monotonic() - start
        msg = str(exc)
        result.update({"elapsed_sec": round(elapsed, 1), "error_type": type(exc).__name__, "error_detail": msg})
        # Check the STRUCTURED "code" field the DashScope error body carries
        # (e.g. "AccessDenied", "InvalidParameter", "ModelNotFound") FIRST --
        # far more reliable than substring-sniffing the free-text message,
        # which can incidentally contain misleading words (e.g. a help-URL
        # fragment like ".../model-studio/error-code" makes "model" appear in
        # an AccessDenied message that has nothing to do with the model name).
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
        else:
            outcome = "other_error"
        print(f"RESULT: {outcome} (code={code_match!r}) -- {type(exc).__name__}: {msg}")

    result["outcome"] = outcome
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUTS_DIR / "wan27_availability_result.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\nSaved raw result to {out_path}")
    return 0 if outcome == "works" else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
