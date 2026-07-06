"""
De-risk / sanity-check script for the Concept Agent (Phase 1, KR).

Usage (from backend/, after running test_truth_extractor.py at least once so
outputs/truth_extractor_result.json exists):
    python -m derisk.test_concept_agent --brief "handmade ceramic mugs, cozy autumn vibe"
    python -m derisk.test_concept_agent --brief "..." --length 30
    python -m derisk.test_concept_agent --brief "..." --truths-file path/to/other.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from agents.concept_agent import REQUIRED_VARIANT_COUNT, generate_script_variants  # noqa: E402

OUTPUTS_DIR = Path(__file__).resolve().parent / "outputs"
DEFAULT_TRUTHS_FILE = OUTPUTS_DIR / "truth_extractor_result.json"


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--brief", required=True, help="one-line seller brief")
    parser.add_argument("--length", type=int, default=15, choices=(15, 30), help="target ad length in seconds")
    parser.add_argument("--truths-file", default=str(DEFAULT_TRUTHS_FILE), help="path to a saved product_truths JSON array")
    args = parser.parse_args()

    truths_path = Path(args.truths_file)
    if not truths_path.exists():
        print(
            f"No truths file at {truths_path}. Run test_truth_extractor.py first, "
            "or pass --truths-file pointing at a saved product_truths array.",
            file=sys.stderr,
        )
        return 1
    product_truths = json.loads(truths_path.read_text())
    print(f"Using {len(product_truths)} product truths from {truths_path}")

    variants = await generate_script_variants(
        brief=args.brief, product_truths=product_truths, target_length_sec=args.length
    )

    print(f"\nProduced {len(variants)} variant(s) (spec wants {REQUIRED_VARIANT_COUNT}):\n")
    for v in variants:
        print(f"=== {v['variant_id']} — {v['framework']} / {v['hook_type']} / {v['emotional_trigger']} ===")
        print(f"grounding_truth_ids: {v['grounding_truth_ids']}")
        print(f"text: {v['text']}")
        print("beats:")
        for b in v["beats"]:
            print(f"  [{b['t_start']:>4}-{b['t_end']:>4}s] {b['line']}")
        print()

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUTS_DIR / "concept_agent_result.json"
    out_path.write_text(json.dumps(variants, indent=2))
    print(f"Saved raw result to {out_path}")

    if len(variants) < REQUIRED_VARIANT_COUNT:
        print(
            f"\n⚠ Only {len(variants)} variant(s) survived validation "
            f"(wanted {REQUIRED_VARIANT_COUNT}). Check the terminal above for "
            "re-prompt/degrade log lines.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
