"""
De-risk / sanity-check script for the Product Truth Extractor (Phase 1, KR).

Usage:
    Drop 2-3 real product photos into backend/derisk/photos/, then from
    backend/ run:
        python -m derisk.test_truth_extractor
        python -m derisk.test_truth_extractor --brief "handmade ceramic mugs, cozy autumn vibe"

Encodes local photos as base64 data URIs (no OSS upload needed for this
sanity check) and calls the real extract_product_truths() -- same code path
the graph node uses, just fed local files instead of OSS URIs. Prints the
result and saves it to backend/derisk/outputs/truth_extractor_result.json.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import mimetypes
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from agents.product_truth_extractor import MAX_FACTS, MIN_FACTS, extract_product_truths  # noqa: E402

PHOTOS_DIR = Path(__file__).resolve().parent / "photos"
OUTPUTS_DIR = Path(__file__).resolve().parent / "outputs"


def _photo_to_data_uri(path: Path) -> str:
    mime, _ = mimetypes.guess_type(str(path))
    mime = mime or "image/jpeg"
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{b64}"


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--brief", default=None, help="optional one-line seller brief")
    parser.add_argument("--freeform", default=None, help="optional seller_direction.freeform text")
    args = parser.parse_args()

    photo_paths = sorted(
        p for p in PHOTOS_DIR.glob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")
    )
    if not (2 <= len(photo_paths) <= 3):
        print(
            f"Expected 2-3 photos in {PHOTOS_DIR}, found {len(photo_paths)}. "
            "Drop real product photos there and re-run.",
            file=sys.stderr,
        )
        return 1

    print(f"Using {len(photo_paths)} photo(s): {[p.name for p in photo_paths]}")
    data_uris = [_photo_to_data_uri(p) for p in photo_paths]

    truths = await extract_product_truths(
        photo_urls=data_uris, brief=args.brief, freeform=args.freeform
    )

    print(f"\nExtracted {len(truths)} facts (spec wants {MIN_FACTS}-{MAX_FACTS}):\n")
    for t in truths:
        print(f"  [{t['truth_id']}] ({t['category']}, {t['source']}) {t['fact']}")

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUTS_DIR / "truth_extractor_result.json"
    out_path.write_text(json.dumps(truths, indent=2))
    print(f"\nSaved raw result to {out_path}")

    if len(truths) < MIN_FACTS:
        print(
            f"\n⚠ Only {len(truths)} facts survived the generic-heuristic filter "
            f"(wanted {MIN_FACTS}+). Eyeball whether that's the model under-delivering "
            "or the heuristic being too strict.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
