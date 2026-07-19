"""
Live test for the Visual Direction Agent.
Usage (from backend/): python -m derisk.test_vda
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from agents.visual_direction_agent import generate_visual_direction  # noqa: E402

OUTPUTS = Path(__file__).resolve().parent / "outputs"


async def main() -> int:
    concept = json.loads((OUTPUTS / "concept_agent_result.json").read_text(encoding="utf-8"))
    truths = json.loads((OUTPUTS / "truth_extractor_result.json").read_text(encoding="utf-8"))

    v1 = concept[0]
    winning_script = {
        "text": v1["text"],
        "beats": v1["beats"],
        "source_variant_ids": [v1["variant_id"]],
    }

    print("VO script:")
    for i, b in enumerate(winning_script["beats"]):
        print(f"  beat {i} [{b['t_start']}-{b['t_end']}s]: {b['line']}")
    print()

    result = await generate_visual_direction(winning_script, truths)

    print(f"Story context: {result['story_context']}")
    print()
    for bvd in result["beat_visual_directions"]:
        hp = bvd["human_presence"]
        print(
            f"Beat {bvd['beat_index']} | truth={bvd['focus_feature_truth_id']} | "
            f"human={hp} | {bvd['suggested_shot_type']} / {bvd['suggested_camera_move']}"
        )
        print(f"  focus: {bvd['focus_moment']}")
        if bvd.get("human_action"):
            print(f"  action: {bvd['human_action']}")
        print(f"  framing: {bvd['framing_notes']}")
        print()

    out = OUTPUTS / "vda_result.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Saved to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
