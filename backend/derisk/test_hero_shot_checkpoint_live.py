"""
REAL hero-shot checkpoint (video-gen-fidelity story-arc fix).

This is the checkpoint that was skipped once already this session (a hand-
written prompt WITHOUT this branch's real identity-protection clauses, Cast
section, or 15s duration was used to review a 10s clip -- a real
methodological gap the task brief called out explicitly). This script does
NOT repeat that mistake: it drives the REAL, production
`agents.video_gen_node._build_prompt()` path end to end --

  1. Upload real product photos to OSS.
  2. REAL agents.product_truth_extractor.extract_product_truths call
     (Qwen-VL) -- real form_factor anchor + micro-facts, not fabricated.
  3. A hand-built Treatment carrying a real character_anchor (grounded in the
     same color_story a real Treatment Agent call would produce -- skipping
     the LLM call here only because that call is already independently unit-
     tested in tests/test_treatment_agent.py; the thing THIS script needs to
     prove is what _build_prompt does with a real character_anchor, not
     whether the LLM can write one).
  4. A hand-built hero Shot fixture: shot_type="worn_in_use",
     duration_sec=15.0 (HERO_SHOT_MAX_DURATION_SEC), so
     agents.shot_list_agent.is_hero_shot() is True on it -- matching exactly
     what the real Shot-List Agent would assemble for a hero shot.
  5. The REAL agents.video_gen_node._build_prompt() -- Cast section,
     compressed identity-protection clause, action-urgency clause, all
     present, all from the real production code path.
  6. REAL, BILLED Wan2.6-i2v-us generation at the full 15s duration
     (~$1.20 at 720p).
  7. ffmpeg frame extraction at 0.5s intervals across the full clip (31
     frames) -- NOT the sparse 5-frames-across-10s sampling that missed the
     color-bleed/interpenetration issue earlier this session.

Usage (from backend/):
    python -m derisk.test_hero_shot_checkpoint_live
"""
from __future__ import annotations

import asyncio
import json
import subprocess
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from agents._oss import _put_and_sign, oss_job_asset_key  # noqa: E402
from agents.product_truth_extractor import extract_product_truths  # noqa: E402
from agents.shot_list_agent import (  # noqa: E402
    HERO_SHOT_MAX_DURATION_SEC,
    NEGATIVE_PROMPT_BOILERPLATE,
    HUMAN_SHOT_NEGATIVE_EXTRA,
    is_hero_shot,
)
from agents.video_gen_node import (  # noqa: E402
    VideoGenAPIError,
    VideoGenTimeoutError,
    _build_prompt,
    _call_wan_video_gen,
)

PHOTOS_DIR = Path(__file__).resolve().parent / "photos"
OUTPUTS_DIR = Path(__file__).resolve().parent / "outputs"
JOB_ID = "hero-shot-checkpoint"

RESOLUTION = "720P"

_HERO_DESCRIPTION = (
    "A woman lifts the backpack by its top handle in one decisive motion and swings it onto "
    "her shoulder. She turns and walks toward the camera through a sunlit kitchen, the bag "
    "riding naturally against her side with each stride. Midway she pauses beside a wooden "
    "counter, resettles the strap higher on her shoulder, and rotates her body to show the "
    "bag's side profile catching the window light. She continues her stride toward the "
    "camera, slows as she nears, and comes to a full stop facing camera, one hand resting on "
    "the top handle as the motion settles into a final still pose."
)

TREATMENT = {
    "director_persona": "warm, intimate handheld realism",
    "color_story": "warm neutrals -- rust-orange and cream -- under natural daylight",
    "pacing_philosophy": "a slow, confident build that lets the moment breathe",
    "beat_treatments": [],
    "character_anchor": (
        "A woman in her late 20s with shoulder-length dark brown hair wears a rust-orange "
        "canvas jacket in a sunlit kitchen with an open window and a wooden counter, "
        "mid-morning."
    ),
}


def _probe_duration(path: str) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", path],
        check=True, capture_output=True, text=True,
    )
    return float(out.stdout.strip())


def _extract_frames(video_path: str, out_dir: Path, interval_sec: float, duration_sec: float) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    t = 0.0
    while t <= duration_sec + 1e-6:
        out_path = out_dir / f"hero_checkpoint_t{t:.1f}.jpg"
        subprocess.run(
            ["ffmpeg", "-y", "-ss", f"{t:.2f}", "-i", video_path,
             "-frames:v", "1", "-q:v", "2", str(out_path)],
            check=True, capture_output=True,
        )
        paths.append(str(out_path))
        t += interval_sec
    return paths


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
        brief="a durable everyday leather backpack built to age beautifully",
    )
    print(f"  {len(truths)} facts extracted:")
    for t in truths:
        print(f"  - [{t['category']}] {t['fact']}")

    # Prefer a human-contact fact (strap/handle/etc.) per the human-contact
    # affordance rubric shot_list_agent.py's real Call B would follow; fall
    # back to any construction_detail/material fact if none names contact.
    # NEVER consider the form_factor truth itself here -- it is already the
    # Subject-line anchor (_build_prompt leads with it separately), and its
    # own fact text often happens to mention a contact part in passing (e.g.
    # "a single top carry handle..."), which would make this selection
    # accidentally duplicate the ENTIRE anchor sentence a second time as the
    # "cited micro-fact" -- a real bug caught in this checkpoint's first
    # real run (2150-char prompt, Lighting trimmed down to one word as a
    # result). Excluding form_factor here is the fix.
    contact_kw = ("strap", "handle", "grip", "clasp", "zip", "buckle")
    non_anchor_truths = [t for t in truths if t["category"] != "form_factor"]
    contact_truth = next(
        (t for t in non_anchor_truths if any(k in t["fact"].lower() for k in contact_kw)),
        None,
    )
    micro_fact = contact_truth or (non_anchor_truths[0] if non_anchor_truths else truths[0])
    print(f"\n  Using cited fact for the hero shot: [{micro_fact['category']}] {micro_fact['fact']}")

    print("\n=== Step 3: build the REAL hero Shot fixture ===")
    shot = {
        "shot_id": "hero1",
        "t_start": 0.0,
        "t_end": HERO_SHOT_MAX_DURATION_SEC,
        "beat_role": "demo",
        "description": _HERO_DESCRIPTION,
        "shot_type": "worn_in_use",
        "camera_move": "static",
        "framing": "rule_of_thirds_left",
        "lighting": "soft, even daylight from a window camera-left, warm neutral white balance",
        "negative_prompt": f"{NEGATIVE_PROMPT_BOILERPLATE}, {HUMAN_SHOT_NEGATIVE_EXTRA}",
        "reference_image_id": "photo_1",
        "text_overlay_zone": "none",
        "duration_sec": HERO_SHOT_MAX_DURATION_SEC,
        "allocated_budget": HERO_SHOT_MAX_DURATION_SEC * 0.08,
        "voiceover_line": "",
        "justification": {
            "script_quote": "she slings it over one shoulder on her way out the door",
            "truth_fact_id": micro_fact["truth_id"],
            "treatment_ref": 0,
        },
        "status": "pending",
        "retry_count": 0,
    }
    assert is_hero_shot(shot), "fixture must satisfy the real is_hero_shot() predicate"
    print(f"  is_hero_shot(shot) = {is_hero_shot(shot)} (duration_sec={shot['duration_sec']})")

    print("\n=== Step 4: REAL _build_prompt() (production path) ===")
    prompt = _build_prompt(shot, truths, TREATMENT)
    print(f"  Prompt ({len(prompt)} chars):\n{'-' * 70}\n{prompt}\n{'-' * 70}")
    has_cast = "Cast:" in prompt
    print(f"  Cast section present: {has_cast}")

    print(f"\n=== Step 5: REAL Wan2.6-i2v-us generation at {shot['duration_sec']}s (BILLED, ~$1.20) ===")
    start = time.monotonic()
    try:
        video_url = await _call_wan_video_gen(
            image_url=photo_urls[0],
            prompt=prompt,
            negative_prompt=shot["negative_prompt"],
            duration_sec=shot["duration_sec"],
            resolution=RESOLUTION,
        )
    except (VideoGenTimeoutError, VideoGenAPIError) as exc:
        print(f"  *** FAIL: Wan generation failed: {exc} ***")
        (OUTPUTS_DIR / "hero_shot_checkpoint_result.json").write_text(
            json.dumps(
                {"outcome": "wan_generation_failed", "error": str(exc), "truths": truths, "prompt": prompt},
                indent=2,
            )
        )
        return 1
    elapsed = time.monotonic() - start
    print(f"  SUCCEEDED in {elapsed:.1f}s -> {video_url}")

    import urllib.request
    local_path = str(OUTPUTS_DIR / "hero_shot_checkpoint.mp4")
    urllib.request.urlretrieve(video_url, local_path)
    real_duration = _probe_duration(local_path)
    print(f"  REAL measured clip duration (ffprobe): {real_duration:.2f}s (requested {shot['duration_sec']}s)")

    print("\n=== Step 6: extract frames every 0.5s across the full clip ===")
    frame_dir = OUTPUTS_DIR / "hero_checkpoint_frames"
    frames = _extract_frames(local_path, frame_dir, interval_sec=0.5, duration_sec=real_duration)
    print(f"  Extracted {len(frames)} frames to {frame_dir}")

    result = {
        "outcome": "ran_end_to_end",
        "photo_urls": photo_urls,
        "truths": truths,
        "cited_fact": micro_fact,
        "shot": shot,
        "treatment": TREATMENT,
        "prompt": prompt,
        "prompt_len": len(prompt),
        "cast_section_present": has_cast,
        "video_url": video_url,
        "local_path": local_path,
        "generation_elapsed_sec": round(elapsed, 1),
        "real_measured_duration_sec": round(real_duration, 2),
        "frame_paths": frames,
    }
    out_path = OUTPUTS_DIR / "hero_shot_checkpoint_result.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\nFull result saved to {out_path}")
    print(f"\n=== NEXT: manually inspect every frame in {frame_dir} for color-bleed / interpenetration ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
