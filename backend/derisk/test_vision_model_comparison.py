"""
Real, live side-by-side comparison of MODEL_VISION candidates for the Product
Truth Extractor: the current `qwen3-vl-plus` (dedicated VL line) vs the
candidate `qwen3.7-plus` (newer unified text+vision architecture, already
used elsewhere in this pipeline as MODEL_TEXT) -- run against the SAME real
photos so fact quality is directly comparable, not just claimed from search
results.

Two REAL, billed Qwen-VL calls (one per model). Does not touch backend/.env;
overrides os.environ["MODEL_VISION"] in-process only, per call.

Usage (from backend/):
    python -m derisk.test_vision_model_comparison
"""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from agents._oss import _put_and_sign, oss_job_asset_key  # noqa: E402
from agents.product_truth_extractor import extract_product_truths  # noqa: E402

PHOTOS_DIR = Path(__file__).resolve().parent / "photos"
OUTPUTS_DIR = Path(__file__).resolve().parent / "outputs"
JOB_ID = "vision-model-compare"
BRIEF = "a durable everyday leather backpack built to age beautifully"

CANDIDATES = ["qwen3-vl-plus", "qwen3.7-plus"]


async def main() -> int:
    photo_paths = [PHOTOS_DIR / "backpack_1.jpg", PHOTOS_DIR / "backpack_2.jpg"]
    for p in photo_paths:
        if not p.exists():
            print(f"ERROR: missing {p}")
            return 1

    print("=== Uploading real photos to OSS (shared across both model calls) ===")
    photo_urls = []
    for i, p in enumerate(photo_paths, start=1):
        key = oss_job_asset_key(JOB_ID, f"photo_{i}.jpg")
        url = _put_and_sign(key, str(p), "image/jpeg")
        photo_urls.append(url)
        print(f"  photo_{i} -> {url[:80]}...")

    results: dict[str, list[dict]] = {}
    for model in CANDIDATES:
        print(f"\n=== REAL call: MODEL_VISION={model!r} ===")
        os.environ["MODEL_VISION"] = model
        truths = await extract_product_truths(photo_urls=photo_urls, brief=BRIEF)
        results[model] = truths
        print(f"  {len(truths)} facts:")
        for t in truths:
            print(f"  - [{t['category']}] {t['fact']}")

    out_path = OUTPUTS_DIR / "vision_model_comparison_result.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved full comparison to {out_path}")

    print("\n=== SIDE BY SIDE ===")
    for model in CANDIDATES:
        print(f"\n--- {model} ({len(results[model])} facts) ---")
        for t in results[model]:
            print(f"  [{t['category']}] {t['fact']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
