"""
Live test for the Treatment Agent with VDA input.
Usage (from backend/): python -m derisk.test_treatment
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from agents.treatment_agent import generate_treatment  # noqa: E402

OUTPUTS = Path(__file__).resolve().parent / "outputs"


async def main() -> int:
    concept = json.loads((OUTPUTS / "concept_agent_result.json").read_text(encoding="utf-8"))
    truths = json.loads((OUTPUTS / "truth_extractor_result.json").read_text(encoding="utf-8"))
    vda = json.loads((OUTPUTS / "vda_result.json").read_text(encoding="utf-8"))

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

    result = await generate_treatment(winning_script, truths, visual_direction=vda)

    print(f"Director persona: {result['director_persona']}")
    print(f"Color story:      {result['color_story']}")
    print(f"Pacing:           {result['pacing_philosophy']}")
    anchor = result.get("character_anchor", "")
    if anchor:
        print(f"Character anchor: {anchor}")
    else:
        print("Character anchor: (none — product-only)")
    print()

    for bt in result["beat_treatments"]:
        print(
            f"Beat {bt['beat_index']} [{bt['beat_function']}] truth={bt['truth_fact_id']}"
        )
        print(f"  quote:    {bt['script_quote']!r}")
        print(f"  approach: {bt['visual_approach']}")
        print(f"  why:      {bt['why_not_generic']}")
        print()

    out = OUTPUTS / "treatment_result.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Saved to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
