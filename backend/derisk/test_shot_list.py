"""
Live test for the Shot-List Agent with VDA + treatment input.
Requires treatment_result.json (run test_treatment.py first).
Usage (from backend/): python -m derisk.test_shot_list
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from agents.shot_list_agent import generate_shot_list  # noqa: E402

OUTPUTS = Path(__file__).resolve().parent / "outputs"


async def main() -> int:
    concept = json.loads((OUTPUTS / "concept_agent_result.json").read_text(encoding="utf-8"))
    truths = json.loads((OUTPUTS / "truth_extractor_result.json").read_text(encoding="utf-8"))
    treatment = json.loads((OUTPUTS / "treatment_result.json").read_text(encoding="utf-8"))
    vda = json.loads((OUTPUTS / "vda_result.json").read_text(encoding="utf-8"))

    v1 = concept[0]
    winning_script = {
        "text": v1["text"],
        "beats": v1["beats"],
        "source_variant_ids": [v1["variant_id"]],
    }

    print(f"Running shot-list agent for {len(winning_script['beats'])} beats ...")
    print()

    shots = await generate_shot_list(
        winning_script=winning_script,
        treatment=treatment,
        product_truths=truths,
        visual_direction=vda,
    )

    print(f"Got {len(shots)} shot(s):")
    print()
    for s in shots:
        human_marker = " [HUMAN]" if s["shot_type"] in ("product_in_hand", "worn_in_use") else ""
        print(
            f"  {s['shot_id']} [{s['beat_role']}]{human_marker}"
            f"  {s['shot_type']} / {s['camera_move']}  {s['duration_sec']}s"
        )
        print(f"    truth: {s['justification']['truth_fact_id']}  "
              f"quote: {s['justification']['script_quote']!r}")
        print(f"    desc:  {s['description'][:120]}...")
        np = s["negative_prompt"]
        print(f"    neg:   {np[:100]}{'...' if len(np) > 100 else ''}  ({len(np)} chars)")
        print()

    out = OUTPUTS / "shot_list_result.json"
    out.write_text(json.dumps(shots, indent=2), encoding="utf-8")
    print(f"Saved to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
