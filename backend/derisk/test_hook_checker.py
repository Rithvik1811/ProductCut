"""
De-risk / sanity-check script for the Hook-Checker (Phase 1, KR).

Usage (from backend/, after running test_concept_agent.py at least once so
outputs/concept_agent_result.json exists):
    python -m derisk.test_hook_checker
    python -m derisk.test_hook_checker --variants-file path/to/other.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from agents.hook_checker import score_hooks  # noqa: E402
from agents.pacing_checker import check_pacing_all  # noqa: E402

OUTPUTS_DIR = Path(__file__).resolve().parent / "outputs"
DEFAULT_VARIANTS_FILE = OUTPUTS_DIR / "concept_agent_result.json"


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants-file", default=str(DEFAULT_VARIANTS_FILE))
    args = parser.parse_args()

    variants_path = Path(args.variants_file)
    if not variants_path.exists():
        print(
            f"No variants file at {variants_path}. Run test_concept_agent.py first, "
            "or pass --variants-file pointing at a saved script_variants array.",
            file=sys.stderr,
        )
        return 1
    script_variants = json.loads(variants_path.read_text())
    print(f"Scoring {len(script_variants)} variant(s) from {variants_path}\n")

    hook_scores = await score_hooks(script_variants)
    pacing_scores = check_pacing_all(script_variants)

    for v in script_variants:
        vid = v["variant_id"]
        hook = hook_scores.get(vid, {})
        pacing = pacing_scores.get(vid, {})
        print(f"=== {vid} ({v['hook_type']}) ===")
        print(f"  hook text: {v['beats'][0]['line'] if v.get('beats') else '(no beats)'}")
        print(f"  hook_score:   {hook.get('hook_score')} -- {hook.get('justification')}")
        print(f"  pacing_score: {pacing.get('pacing_score')}")
        if pacing.get("violations"):
            for viol in pacing["violations"]:
                print(f"    ! {viol}")
        print()

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUTS_DIR / "hook_pacing_result.json"
    out_path.write_text(json.dumps({"hook_scores": hook_scores, "pacing_scores": pacing_scores}, indent=2))
    print(f"Saved raw result to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
