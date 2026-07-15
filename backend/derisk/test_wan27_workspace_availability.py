"""
De-risk / availability spike for `wan2.7-i2v` via a Model Studio WORKSPACE-scoped
endpoint (`DASHSCOPE_VIDEO27_API_KEY`/`DASHSCOPE_VIDEO27_BASE_URL`), distinct
from `derisk/test_wan27_availability.py` (which tested the shared `dashscope-us`
region and got a real, confirmed AccessDenied -- that result stands, this is a
SEPARATE credential class: an `sk-ws-H.` workspace session key on a per-workspace
`ws-<id>.<region>.maas.aliyuncs.com` host, the same credential family already
confirmed working for TTS in `agents/voiceover_caption_agent.py`).

Time-boxed, best-effort -- AT MOST ONE real, billed-if-it-succeeds API call.
Does not loop/retry. If it fails, the raw status_code/code/message is printed
and saved so a human can decide whether to investigate further (e.g. whether
the native video-synthesis task API even exists on this workspace gateway, vs
only an OpenAI-chat-compatible surface -- video synthesis is NOT a chat-shaped
API, so a workspace gateway that only proxies chat completions would not
support this call at all, which is a real, live open question this script
answers empirically rather than assuming).

Usage (from backend/):
    python -m derisk.test_wan27_workspace_availability
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

OUTPUTS_DIR = Path(__file__).resolve().parent / "outputs"

MODEL = "wan2.7-i2v"
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
    api_key = os.environ["DASHSCOPE_VIDEO27_API_KEY"]
    base_url = os.environ["DASHSCOPE_VIDEO27_BASE_URL"]
    dashscope.api_key = api_key
    dashscope.base_http_api_url = base_url

    print(f"Probing model={MODEL!r} on base_http_api_url={base_url!r} (workspace-scoped key)")
    print("This is AT MOST ONE real API call. If it reaches SUCCEEDED, it is a "
          "real, billed generation.\n")

    result: dict = {"model": MODEL, "base_http_api_url": base_url, "credential_class": "workspace_sk_ws"}
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
        else:
            outcome = "other_error"
        print(f"RESULT: {outcome} (code={code_match!r}) -- {type(exc).__name__}: {msg}")

    result["outcome"] = outcome
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUTS_DIR / "wan27_workspace_availability_result.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\nSaved raw result to {out_path}")
    return 0 if outcome == "works" else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
