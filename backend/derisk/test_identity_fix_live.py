"""
Live end-to-end validation of the video-gen product-identity fidelity fix
(branch `video-gen-fidelity`) against a REAL product photo set.

Unlike derisk/test_video_gen.py (which uses an ad-hoc hand-written prompt to
raw-test Wan reachability/quality), this script exercises the ACTUAL fixed
code path end to end:

  1. Upload the real photos to OSS (Wan needs a fetchable URL).
  2. agents.product_truth_extractor.extract_product_truths -- REAL Qwen-VL
     call, must return a valid "form_factor" anchor fact (the core of the
     fix) alongside the usual micro-facts.
  3. Build a real, minimal Shot fixture citing one of those facts.
  4. agents.video_gen_node._build_prompt -- REAL prompt builder, now leading
     the Subject line with the form_factor anchor.
  5. agents.video_gen_node._call_wan_video_gen -- REAL, BILLED Wan2.6-i2v-us
     generation using that prompt.
  6. agents.continuity_agent._score_one_shot_identity -- REAL Qwen-VL
     same-object verification on the actual generated clip, proving the
     fix's own safety net runs correctly on genuine output.

Test subject: two real Unsplash product photos of a brown leather backpack
(backend/derisk/photos/backpack_1.jpg, backpack_2.jpg) -- a clean, well-lit,
non-adversarial "does the fixed pipeline still work well on an easy case"
sanity check. This is NOT a reproduction of the original Meta Quest failure
(those photos aren't available in this environment) -- it's a regression +
safety-net check on real generation, using a real product a Shopify seller
would plausibly list.

COST WARNING: this makes ONE real, billed Qwen-VL call (Truth Extractor),
ONE real, billed Wan2.6-i2v-us generation (~$0.40 for a 5s 720p clip per
agents/budget_gate.py's RATE_720P), and ONE real, billed Qwen-VL call
(identity check). Deliberately single-shot -- do not loop this script.

Usage (from backend/):
    python -m derisk.test_identity_fix_live
"""
from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from agents._oss import _put_and_sign, oss_job_asset_key  # noqa: E402
from agents.continuity_agent import _score_one_shot_identity, extract_frame  # noqa: E402
from agents.product_truth_extractor import extract_product_truths  # noqa: E402
from agents.shot_list_agent import NEGATIVE_PROMPT_BOILERPLATE  # noqa: E402
from agents.video_gen_node import (  # noqa: E402
    VideoGenAPIError,
    VideoGenTimeoutError,
    _build_prompt,
    _call_wan_video_gen,
)

PHOTOS_DIR = Path(__file__).resolve().parent / "photos"
OUTPUTS_DIR = Path(__file__).resolve().parent / "outputs"
JOB_ID = "livetest-backpack"

DURATION_SEC = 5.0
RESOLUTION = "720P"

_DESCRIPTION = (
    "The backpack sits centered and motionless on a seamless neutral backdrop. "
    "The camera performs a slow, steady push-in toward the front pocket and "
    "embossed logo tab, holding sharp focus on the leather's natural grain and "
    "stitching the entire time. No hands, no other objects, no scene cuts -- "
    "the backpack itself never leaves frame and never changes shape."
)


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

    print("\n=== Step 2: REAL Product Truth Extractor call (Qwen-VL) ===")
    truths = await extract_product_truths(
        photo_urls=photo_urls,
        brief="a durable everyday leather backpack",
    )
    print(f"  {len(truths)} facts extracted:")
    for t in truths:
        print(f"  - [{t['category']}] {t['fact']}")

    form_factor = next((t for t in truths if t["category"] == "form_factor"), None)
    if form_factor is None:
        print("\n  *** FAIL: no form_factor anchor fact was returned. ***")
        print("  This is the core claim of the fix -- stopping here.")
        (OUTPUTS_DIR / "identity_fix_live_result.json").write_text(
            json.dumps({"outcome": "no_form_factor_fact", "truths": truths}, indent=2)
        )
        return 1
    print(f"\n  FORM-FACTOR ANCHOR: {form_factor['fact']}")

    micro_fact = next((t for t in truths if t["category"] != "form_factor"), form_factor)

    print("\n=== Step 3: build a real Shot fixture ===")
    shot = {
        "shot_id": "s1",
        "t_start": 0.0,
        "t_end": 5.0,
        "beat_role": "hook",
        "description": _DESCRIPTION,
        "shot_type": "hook_hero",
        "camera_move": "push_in",
        "framing": "fills_frame",
        "lighting": "soft, even studio lighting from the upper left, neutral white balance",
        "negative_prompt": f"{NEGATIVE_PROMPT_BOILERPLATE}, backpack strap detaching",
        "reference_image_id": "photo_1",
        "text_overlay_zone": "none",
        "duration_sec": DURATION_SEC,
        "allocated_budget": 5.0,
        "voiceover_line": "",
        "justification": {
            "script_quote": "built to hold its shape, day after day",
            "truth_fact_id": micro_fact["truth_id"],
            "treatment_ref": 0,
        },
        "status": "pending",
        "retry_count": 0,
    }

    prompt = _build_prompt(shot, truths, treatment=None)
    print(f"  Prompt ({len(prompt)} chars):\n{'-' * 60}\n{prompt}\n{'-' * 60}")

    print("\n=== Step 4: REAL Wan2.6-i2v-us generation (BILLED, ~$0.40) ===")
    start = time.monotonic()
    try:
        video_url = await _call_wan_video_gen(
            image_url=photo_urls[0],
            prompt=prompt,
            negative_prompt=shot["negative_prompt"],
            duration_sec=DURATION_SEC,
            resolution=RESOLUTION,
        )
    except (VideoGenTimeoutError, VideoGenAPIError) as exc:
        print(f"  *** FAIL: Wan generation failed: {exc} ***")
        (OUTPUTS_DIR / "identity_fix_live_result.json").write_text(
            json.dumps(
                {"outcome": "wan_generation_failed", "error": str(exc), "truths": truths, "prompt": prompt},
                indent=2,
            )
        )
        return 1
    elapsed = time.monotonic() - start
    print(f"  SUCCEEDED in {elapsed:.1f}s -> {video_url}")

    print("\n=== Step 5: REAL identity check (Qwen-VL, same_object verdict) ===")
    entry = {"video_uri": video_url, "attempt": 1, "duration_sec_used": DURATION_SEC}
    identity = await _score_one_shot_identity(
        shot=shot, entry=entry, product_photos=photo_urls, client=None, extract_frame_fn=extract_frame
    )
    print(f"  same_object={identity.same_object}  confidence={identity.confidence}")
    print(f"  matching_features: {identity.matching_features}")
    print(f"  mismatching_features: {identity.mismatching_features}")

    result = {
        "outcome": "ran_end_to_end",
        "photo_urls": photo_urls,
        "truths": truths,
        "form_factor_fact": form_factor["fact"],
        "prompt": prompt,
        "prompt_len": len(prompt),
        "video_url": video_url,
        "generation_elapsed_sec": round(elapsed, 1),
        "identity_check": identity.model_dump(),
    }
    out_path = OUTPUTS_DIR / "identity_fix_live_result.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\nFull result saved to {out_path}")
    print(
        "\n=== VERDICT ==="
        f"\n  form_factor anchor present: YES"
        f"\n  Wan generation: SUCCEEDED"
        f"\n  identity check same_object: {identity.same_object} (confidence={identity.confidence})"
    )
    return 0 if identity.same_object else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
