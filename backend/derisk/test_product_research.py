"""
Derisk test for product_research_node — comprehensive real-world coverage.

Tests 8 product types spanning the full classification spectrum:
  SHOULD classify research_needed:
    1. Windproof lighter (generic, no brand — function not visible in image)
    2. Meta Quest 3S (named tech, VR/MR headset with software features)
    3. AirPods Pro 2 (earbuds with software/AI features: ANC, Transparency, EQ)
    4. Nike Air Zoom Pegasus 41 (running shoes with performance cushioning tech)
    5. Dyson V15 Detect (smart vacuum, laser sensor + AI suction mode)
    6. GoPro Hero 12 Black (action camera, stabilisation + software features)
    7. Scented soy candle (function — scent, burn time — not visible in photo)
  SHOULD classify skip:
    8. Unlabelled artisan ceramic bowl (visual product, no brand, no function beyond aesthetics)

Usage (from backend/):
  .venv/Scripts/python.exe -m derisk.test_product_research
"""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

CASES = [
    {
        "label": "1. Windproof lighter (generic, no brand)",
        "expect": "research_needed",
        "state": {
            "brief": "A slim windproof lighter — everyday carry piece.",
            "brand_name": "",
            "product_truths": [
                {"truth_id": "t1", "fact": "Matte black metal body with brushed finish"},
                {"truth_id": "t2", "fact": "Slim rectangular form, fits in shirt pocket"},
                {"truth_id": "t3", "fact": "Flip-open lid with a chrome hinge"},
            ],
        },
    },
    {
        "label": "2. Meta Quest 3S (VR/MR headset, named tech)",
        "expect": "research_needed",
        "state": {
            "brief": "Meta Quest 3S — next-gen mixed reality headset for everyone.",
            "brand_name": "Meta",
            "product_truths": [
                {"truth_id": "t1", "fact": "White plastic headset with two controller wands"},
                {"truth_id": "t2", "fact": "Fabric forehead cushion, adjustable strap"},
                {"truth_id": "t3", "fact": "Two front-facing cameras for passthrough"},
            ],
        },
    },
    {
        "label": "3. AirPods Pro 2 (earbuds with ANC + software/AI features)",
        "expect": "research_needed",
        "state": {
            "brief": "Apple AirPods Pro (2nd generation) — premium wireless earbuds.",
            "brand_name": "Apple",
            "product_truths": [
                {"truth_id": "t1", "fact": "White silicone ear tips with stem design"},
                {"truth_id": "t2", "fact": "Matte white case with Lightning port"},
                {"truth_id": "t3", "fact": "Small physical button on the stem"},
            ],
        },
    },
    {
        "label": "4. Nike Air Zoom Pegasus 41 (running shoes, cushioning tech)",
        "expect": "research_needed",
        "state": {
            "brief": "Nike Air Zoom Pegasus 41 — daily training running shoe.",
            "brand_name": "Nike",
            "product_truths": [
                {"truth_id": "t1", "fact": "Blue and white mesh upper with reflective Swoosh"},
                {"truth_id": "t2", "fact": "Thick white foam midsole with visible Air unit at heel"},
                {"truth_id": "t3", "fact": "Rubber outsole with waffle traction pattern"},
            ],
        },
    },
    {
        "label": "5. Dyson V15 Detect (smart vacuum, laser + AI suction)",
        "expect": "research_needed",
        "state": {
            "brief": "Dyson V15 Detect — cordless vacuum with intelligent suction.",
            "brand_name": "Dyson",
            "product_truths": [
                {"truth_id": "t1", "fact": "Purple and silver cordless stick vacuum"},
                {"truth_id": "t2", "fact": "LCD screen on the body"},
                {"truth_id": "t3", "fact": "Elongated floor nozzle with green LED strip"},
            ],
        },
    },
    {
        "label": "6. GoPro Hero 12 Black (action camera, software stabilisation)",
        "expect": "research_needed",
        "state": {
            "brief": "GoPro HERO12 Black — flagship action camera.",
            "brand_name": "GoPro",
            "product_truths": [
                {"truth_id": "t1", "fact": "Compact black rectangular body with lens"},
                {"truth_id": "t2", "fact": "Front LCD status screen, rear touch display"},
                {"truth_id": "t3", "fact": "Side door for battery and SD card access"},
            ],
        },
    },
    {
        "label": "7. Scented soy candle (function not visible in photo)",
        "expect": "research_needed",
        "state": {
            "brief": "Hand-poured soy candle with a cedarwood and vanilla scent.",
            "brand_name": "Wax & Wick",
            "product_truths": [
                {"truth_id": "t1", "fact": "Amber glass vessel with a cream label"},
                {"truth_id": "t2", "fact": "Single cotton wick, clean wax surface"},
                {"truth_id": "t3", "fact": "Kraft paper lid with twine tied around it"},
            ],
        },
    },
    {
        "label": "8. Unlabelled artisan ceramic bowl (should SKIP)",
        "expect": "skip",
        "state": {
            "brief": "A hand-thrown ceramic bowl in a natural glaze.",
            "brand_name": "",
            "product_truths": [
                {"truth_id": "t1", "fact": "Earthy terracotta body with irregular rim"},
                {"truth_id": "t2", "fact": "Speckled grey-green ash glaze on the interior"},
                {"truth_id": "t3", "fact": "Foot ring with visible finger-pull marks"},
            ],
        },
    },
]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def _print_result(label: str, result: dict, expect: str) -> None:
    pr = result.get("product_research", {})
    classification = pr.get("classification", "unknown")
    passed = (
        (expect == "research_needed" and classification == "research_needed" and pr.get("performed"))
        or (expect == "skip" and classification == "skipped" and not pr.get("performed"))
    )
    status = "PASS" if passed else "FAIL"

    print(f"\n{'='*64}")
    print(f"  {label}")
    print(f"  expected={expect}  got={classification}  [{status}]")
    print(f"{'='*64}")

    if not pr.get("performed"):
        print("  (skipped — no facts)")
        return passed

    print(f"  product_name : {pr.get('product_name', '—')}")
    queries = pr.get("queries_used") or []
    for q in queries:
        print(f"  query  : {q}")

    facts = pr.get("facts") or []
    by_cat: dict[str, list] = {}
    for f in facts:
        by_cat.setdefault(f.get("category", "?"), []).append(f)

    for cat in ["feature", "spec", "differentiator", "compatibility", "use_case", "visual_moment"]:
        items = by_cat.get(cat, [])
        if not items:
            continue
        print(f"\n  [{cat}]  ({len(items)})")
        for f in items:
            conf = f.get("confidence", "?")
            print(f"    [{f.get('fact_id')}] ({conf}) {f.get('claim')}")

    unknown_cats = set(by_cat) - {"feature", "spec", "differentiator", "compatibility", "use_case", "visual_moment"}
    if unknown_cats:
        print(f"\n  [UNEXPECTED CATEGORIES]: {unknown_cats}")
        for cat in unknown_cats:
            for f in by_cat[cat]:
                print(f"    [{f.get('fact_id')}] ({f.get('category')}) {f.get('claim')}")

    print(f"\n  Total facts: {len(facts)}")
    return passed


async def run_all() -> None:
    from agents.product_research_node import product_research_node

    results: list[dict] = []
    for case in CASES:
        print(f"\nRunning: {case['label']} ...")
        result = await product_research_node(case["state"])
        passed = _print_result(case["label"], result, case["expect"])
        results.append({"label": case["label"], "passed": passed, "expect": case["expect"]})

    print(f"\n{'='*64}")
    print("  SUMMARY")
    print(f"{'='*64}")
    failed = 0
    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        print(f"  [{status}] {r['label']}")
        if not r["passed"]:
            failed += 1

    print(f"\n  {len(results) - failed}/{len(results)} passed")
    if failed:
        print(f"  {failed} FAILED")
        sys.exit(1)
    else:
        print("  All passed.")


async def main() -> None:
    missing = [k for k in ("TAVILY_API_KEY", "DASHSCOPE_API_KEY") if not os.environ.get(k)]
    if missing:
        print(f"SKIP: missing env vars: {', '.join(missing)}")
        sys.exit(0)
    await run_all()


if __name__ == "__main__":
    asyncio.run(main())
