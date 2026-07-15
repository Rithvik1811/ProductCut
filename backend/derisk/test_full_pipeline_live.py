"""
Full, REAL, live end-to-end pipeline run -- no faked LLM/network boundaries
anywhere. This is the test derisk/test_identity_fix_live.py deliberately was
NOT: it drives the actual compiled graph (graph.build.build_graph()) through
every real stage --

  Product Truth Extractor -> Concept Agent -> 5 parallel Critics -> Meta-Critic
  -> Merge Coherence Validator -> (Copy Editor / swap-retry loop if needed) ->
  Treatment Agent + Voiceover/Caption Agent (parallel) -> Shot-List Agent ->
  Budget Gate -> Video-Gen (real Wan2.6-i2v-us per shot, parallel fan-out) ->
  Ken-Burns Fallback (only if a shot hard-fails) -> Continuity Agent (real
  drift + identity Qwen-VL checks per shot) -> Continuity Gate (real capped
  retry loop; auto-resolves any human-review interrupt with "accept_fallback"
  since this is an unattended script, not a UI) -> END.

Uses the real DATABASE_URL checkpointer if reachable (graph.build.build_graph
gracefully falls back to MemorySaver if Postgres init fails -- this script
does not special-case that, it just reports which one was actually used).

COST WARNING: this is the expensive one. Real calls for every text/vision LLM
stage above, PLUS one real, billed Wan2.6-i2v-us generation PER SHOT (3-7
shots, ~$0.40-0.60 each), PLUS real TTS calls per script beat, PLUS 2 real
Qwen-VL calls per shot for Continuity (drift + identity). A retry loop can
multiply the Video-Gen/Continuity cost further. Expect several dollars and
several minutes of wall time, not the ~$0.40 single-shot spike this branch's
other derisk scripts run.

Usage (from backend/):
    python -m derisk.test_full_pipeline_live
"""
from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from langgraph.types import Command  # noqa: E402

from agents._oss import _put_and_sign, oss_job_asset_key  # noqa: E402
from graph.build import build_graph  # noqa: E402

PHOTOS_DIR = Path(__file__).resolve().parent / "photos"
OUTPUTS_DIR = Path(__file__).resolve().parent / "outputs"
JOB_ID = "livetest-fullpipeline"
THREAD_ID = "livetest-fullpipeline"

BRIEF = "a durable everyday leather backpack built to age beautifully"

MAX_LOOP_PASSES = 10  # guard against an unexpected infinite interrupt/resume cycle


def _summarize_event(e: dict) -> str:
    name = e.get("name")
    data = e.get("data", {})
    if name == "truth_extracted":
        return f"truth_extracted: {data.get('count')} facts"
    if name == "critic_score":
        return f"critic_score: winning_variant_ids={data.get('winning_variant_ids')}"
    if name == "merge_validated":
        r = data.get("result", {})
        return f"merge_validated: attempt={data.get('attempt_number')} passed={r.get('passed')} failure_kind={r.get('failure_kind')}"
    if name == "budget_updated":
        ledger = data.get("ledger", {})
        return f"budget_updated: spent={ledger.get('spent')} cap={ledger.get('cap')} over_cap={data.get('over_cap')}"
    if name == "shot_generated":
        return f"shot_generated: shot_id={data.get('shot_id')} is_fallback={data.get('is_fallback')} status={data.get('status')}"
    if name == "drift_scored":
        return f"drift_scored: shot_id={data.get('shot_id')} passed={data.get('passed')}"
    if name == "vo_ready":
        return f"vo_ready: caption_count={data.get('caption_count')} degraded={data.get('degraded')}"
    if name == "interrupt_requested":
        return f"interrupt_requested: {data}"
    return f"{name}: {data}"


async def _drain_and_log(graph, arg, cfg) -> list[dict]:
    events = []
    async for e in graph.astream_events(arg, config=cfg, version="v2"):
        if e.get("event") == "on_custom_event":
            events.append(e)
            print(f"  [{time.strftime('%H:%M:%S')}] {_summarize_event(e)}")
    return events


async def main() -> int:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    photo_paths = [PHOTOS_DIR / "backpack_1.jpg", PHOTOS_DIR / "backpack_2.jpg"]

    print("=== Step 0: upload photos to OSS ===")
    photo_urls = []
    for i, p in enumerate(photo_paths, start=1):
        key = oss_job_asset_key(JOB_ID, f"photo_{i}.jpg")
        url = _put_and_sign(key, str(p), "image/jpeg")
        photo_urls.append(url)
        print(f"  photo_{i} -> {url[:80]}...")

    print("\n=== Building real compiled graph (checkpointer: real DATABASE_URL if reachable) ===")
    graph = await build_graph()

    initial_state = {
        "job_id": JOB_ID,
        "product_photos": photo_urls,
        "brief": BRIEF,
    }
    cfg = {"configurable": {"thread_id": THREAD_ID}, "recursion_limit": 100}

    all_events: list[dict] = []
    overall_start = time.monotonic()

    print(f"\n=== Driving the REAL graph (job_id={JOB_ID}) ===")
    all_events += await _drain_and_log(graph, initial_state, cfg)

    # Auto-resolve any human-review interrupt (unattended script -- no real UI).
    passes = 0
    while passes < MAX_LOOP_PASSES:
        st = await graph.aget_state(cfg)
        if not st.interrupts:
            break
        passes += 1
        entry = st.interrupts[0].value
        print(f"\n  *** interrupt raised for shot {entry.get('shot_id')} (drift_score={entry.get('drift_score')}) "
              "-- auto-resolving with accept_fallback (unattended script) ***")
        all_events += await _drain_and_log(graph, Command(resume={"resolution": "accept_fallback"}), cfg)

    overall_elapsed = time.monotonic() - overall_start
    final = await graph.aget_state(cfg)
    values = final.values

    print(f"\n=== DONE in {overall_elapsed:.1f}s ({overall_elapsed / 60:.1f} min), {passes} interrupt-resolve pass(es) ===")

    # ---- Human-readable summary ----
    winning = values.get("winning_script", {})
    treatment = values.get("treatment", {})
    shot_list = values.get("shot_list", [])
    generated = values.get("generated_shots", {})
    ledger = values.get("budget_ledger", {})
    voiceover = values.get("voiceover", {})
    master_cut_uri = values.get("master_cut_uri", "")

    print("\n--- WINNING SCRIPT ---")
    print(f"  source_variant_ids: {winning.get('source_variant_ids')}")
    print(f"  text: {winning.get('text')}")

    print("\n--- TREATMENT ---")
    print(f"  director_persona: {treatment.get('director_persona')}")
    print(f"  color_story: {treatment.get('color_story')}")
    print(f"  pacing_philosophy: {treatment.get('pacing_philosophy')}")

    print(f"\n--- SHOT LIST ({len(shot_list)} shots) ---")
    for s in shot_list:
        g = generated.get(s["shot_id"], {})
        print(
            f"  {s['shot_id']} [{s.get('beat_role')}/{s.get('shot_type')}] "
            f"camera={s.get('camera_move')} status={s.get('status')} "
            f"retry_count={s.get('retry_count')}"
        )
        print(f"      description: {s.get('description', '')[:160]}...")
        print(f"      video_uri: {g.get('video_uri', '(none)')}")
        if "drift_score" in g:
            print(f"      drift_score={g.get('drift_score')} identity={g.get('identity_check')}")

    print("\n--- BUDGET LEDGER ---")
    print(f"  cap={ledger.get('cap')} spent={ledger.get('spent')}")

    print("\n--- VOICEOVER ---")
    print(f"  audio_uri: {voiceover.get('audio_uri', '(none)')}")
    print(f"  caption_track_uri: {voiceover.get('caption_track_uri', '(none)')}")

    print("\n--- MASTER CUT (agents.assembly_agent, video-gen-fidelity PHASE 3) ---")
    print(f"  master_cut_uri: {master_cut_uri or '(none -- assembly_agent did not run or produced nothing)'}")

    result = {
        "elapsed_sec": round(overall_elapsed, 1),
        "interrupt_passes": passes,
        "winning_script": winning,
        "treatment": treatment,
        "shot_list": shot_list,
        "generated_shots": generated,
        "budget_ledger": ledger,
        "voiceover": voiceover,
        "master_cut_uri": master_cut_uri,
        "reasoning_trace": values.get("reasoning_trace", ""),
        "voiceover_reasoning_trace": values.get("voiceover_reasoning_trace", ""),
        "custom_event_log": [{"name": e["name"], "data": e.get("data")} for e in all_events],
    }
    out_path = OUTPUTS_DIR / "full_pipeline_live_result.json"
    out_path.write_text(json.dumps(result, indent=2, default=str))
    print(f"\nFull result saved to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
