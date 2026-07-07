"""
De-risk / sanity-check script for the Critic Chain graph wiring (Phase 1, RR).

Two checks:
  1. Structural (free, no API calls): build + compile the graph and assert all
     eight nodes are present (product_truth_extractor, concept_agent, the five
     parallel checkers, meta_critic).
  2. Live functional smoke (real DashScope calls, kept cheap -- 2 hand-built
     script variants): drive each of the five new checker node-wrappers directly,
     merge their outputs the way LangGraph's parallel fan-in would, then run
     meta_critic_node and assert every variant gets a full 9-key CriticScore and
     the Meta-Critic returns a valid outcome.

Usage (from backend/):
    python -m derisk.test_graph_wiring
    python -m derisk.test_graph_wiring --structural-only   # skip live API calls
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

# On Windows, stdout defaults to cp1252 when redirected to a file, which blows up
# printing LLM justifications that contain Unicode (arrows, degree signs, etc.).
# Force UTF-8 so this de-risk script's output survives a `> file` redirect.
for _stream in (sys.stdout, sys.stderr):
    _reconfigure = getattr(_stream, "reconfigure", None)
    if _reconfigure is not None:
        _reconfigure(encoding="utf-8")

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from langgraph.checkpoint.memory import MemorySaver  # noqa: E402
from langchain_core.runnables import RunnableLambda  # noqa: E402

from graph.build import _build_uncompiled  # noqa: E402
from agents.hook_checker import hook_checker_node  # noqa: E402
from agents.pacing_checker import pacing_checker_node  # noqa: E402
from agents.body_checker import body_checker_node  # noqa: E402
from agents.cta_tone_checkers import cta_checker_node, tone_checker_node  # noqa: E402
from agents.meta_critic import meta_critic_node  # noqa: E402
from agents.merge_validator import merge_validator_node, route_after_merge_validation  # noqa: E402
from agents.copy_editor import copy_editor_node  # noqa: E402

EXPECTED_NODES = {
    "product_truth_extractor",
    "concept_agent",
    "hook_checker",
    "pacing_checker",
    "body_checker",
    "cta_checker",
    "tone_checker",
    "meta_critic",
    "merge_validator",
    "copy_editor",
}

CRITIC_SCORE_KEYS = {
    "hook",
    "pacing",
    "completion",
    "completion_detail",
    "cta",
    "tone",
    "composite",
    "justification",
    "never_do_violation",
}


def structural_check() -> int:
    """Compile the graph and assert all eight nodes are wired. No API calls."""
    print("=== 1. STRUCTURAL CHECK (no API calls) ===")
    graph = _build_uncompiled().compile(checkpointer=MemorySaver())
    nodes = set(graph.get_graph().nodes) - {"__start__", "__end__"}
    print(f"compiled graph nodes: {sorted(nodes)}")
    missing = EXPECTED_NODES - nodes
    extra = nodes - EXPECTED_NODES
    assert not missing, f"missing nodes: {missing}"
    assert not extra, f"unexpected nodes: {extra}"
    print(f"OK: all {len(EXPECTED_NODES)} expected nodes present.\n")
    return 0


def _fixture_variants() -> list[dict]:
    """Two hand-built, valid ScriptVariants with real short ad-script beats."""
    return [
        {
            "variant_id": "v1",
            "text": (
                "Your coffee is cold in 12 minutes. This mug keeps it hot for two "
                "hours. Double-walled ceramic, fired at 1200 degrees. Grab yours today."
            ),
            "framework": "hook_problem_product_cta",
            "hook_type": "problem_callout",
            "emotional_trigger": "frustration_relief",
            "grounding_truth_ids": ["t1", "t2"],
            "beats": [
                {"t_start": 0.0, "t_end": 3.0, "line": "Your coffee is cold in 12 minutes."},
                {"t_start": 3.0, "t_end": 7.0, "line": "This mug keeps it hot for two full hours."},
                {"t_start": 7.0, "t_end": 11.0, "line": "Double-walled ceramic, fired at 1200 degrees."},
                {"t_start": 11.0, "t_end": 15.0, "line": "Grab yours today before they sell out."},
            ],
            "target_length_sec": 15,
        },
        {
            "variant_id": "v2",
            "text": (
                "Most mugs are boring. This one is hand-glazed with a reactive finish, "
                "so no two are alike. Feel the weight of real stoneware. Order now."
            ),
            "framework": "PAS",
            "hook_type": "bold_claim",
            "emotional_trigger": "desire_uniqueness",
            "grounding_truth_ids": ["t3"],
            "beats": [
                {"t_start": 0.0, "t_end": 3.0, "line": "Most mugs are boring and forgettable."},
                {"t_start": 3.0, "t_end": 7.0, "line": "This one is hand-glazed with a reactive finish."},
                {"t_start": 7.0, "t_end": 11.0, "line": "No two mugs are ever exactly alike."},
                {"t_start": 11.0, "t_end": 15.0, "line": "Feel real stoneware. Order yours now."},
            ],
            "target_length_sec": 15,
        },
    ]


def _fixture_truths() -> list[dict]:
    return [
        {"truth_id": "t1", "fact": "Double-walled ceramic construction", "category": "construction_detail", "source": "photo"},
        {"truth_id": "t2", "fact": "Fired at 1200 degrees", "category": "material", "source": "brief_or_intake_fact"},
        {"truth_id": "t3", "fact": "Reactive glaze, each piece unique", "category": "texture", "source": "photo"},
    ]


async def functional_smoke() -> int:
    """Drive the five checker nodes + meta_critic_node with fixtures. Live API."""
    print("=== 2. LIVE FUNCTIONAL SMOKE (real DashScope calls) ===")
    variants = _fixture_variants()
    base_state = {
        "brief": "handmade double-walled ceramic coffee mugs, cozy premium vibe",
        "seller_direction": {"mood_words": ["cozy", "premium"], "never_do": "do not use fake urgency countdowns"},
        "product_truths": _fixture_truths(),
        "script_variants": variants,
    }

    print("Running the 5 checker node-wrappers (hook async; pacing pure; body/cta/tone via to_thread)...")
    hook_out, pacing_out, body_out, cta_out, tone_out = await asyncio.gather(
        hook_checker_node(base_state),
        pacing_checker_node(base_state),
        body_checker_node(base_state),
        cta_checker_node(base_state),
        tone_checker_node(base_state),
    )

    # Merge the parallel branch outputs the way LangGraph's superstep fan-in would.
    merged_state = {**base_state, **hook_out, **pacing_out, **body_out, **cta_out, **tone_out}
    print("  hook_scores:  ", merged_state["hook_scores"])
    print("  pacing_scores:", merged_state["pacing_scores"])
    print("  body_scores:  ", merged_state["body_scores"])
    print("  cta_scores:   ", merged_state["cta_scores"])
    print("  tone_scores:  ", merged_state["tone_scores"])

    # meta_critic_node dispatches a C2 custom event, which requires an active
    # LangChain run context. Wrapping it in a RunnableLambda establishes that
    # context (same one astream_events provides in the real graph) so
    # adispatch_custom_event finds a parent run id. We also capture the event to
    # prove it fires correctly.
    print("\nRunning meta_critic_node (fan-in join; capturing its critic_score C2 event)...")
    node_runnable = RunnableLambda(meta_critic_node)
    captured_events = []
    async for ev in node_runnable.astream_events(merged_state, version="v2"):
        if ev.get("event") == "on_custom_event":
            captured_events.append(ev)
    meta_out = await node_runnable.ainvoke(merged_state)

    critic_scores = meta_out["critic_scores"]
    meta_result = meta_out["meta_critic_result"]

    print("\n--- RESULTS ---")
    for vid in [v["variant_id"] for v in variants]:
        cs = critic_scores[vid]
        print(
            f"  {vid}: hook={cs['hook']} pacing={cs['pacing']} completion={cs['completion']} "
            f"cta={cs['cta']} tone={cs['tone']} composite={cs['composite']:.3f} "
            f"never_do_violation={cs['never_do_violation']}"
        )
        print(f"       completion_detail={cs['completion_detail']}")
        print(f"       justification={cs['justification']}")
    print(f"\n  meta_critic outcome:       {meta_result['outcome']}")
    print(f"  survivor_ids:              {meta_result['survivor_ids']}")
    print(f"  disqualified:              {[d['variant_id'] for d in meta_result['disqualified']]}")
    print(f"  composite_scores:          {meta_result['composite_scores']}")
    if meta_result.get("merge_candidate"):
        mc = meta_result["merge_candidate"]
        print(
            f"  merge_candidate sources:   hook={mc['hook_source_variant_id']} "
            f"body={mc['body_source_variant_id']} cta={mc['cta_source_variant_id']}"
        )
    print(f"  captured C2 critic_score events: {[e['name'] for e in captured_events]}")
    if captured_events:
        print(f"       payload winning_variant_ids: {captured_events[0]['data'].get('winning_variant_ids')}")

    # ---- Assertions ----
    valid_outcomes = {
        "cross_pollinated", "unanimous", "single_survivor",
        "fallback_no_compatible_merge", "all_excluded_failure",
    }
    assert set(critic_scores.keys()) == {v["variant_id"] for v in variants}, "missing a variant in critic_scores"
    for vid, cs in critic_scores.items():
        assert set(cs.keys()) == CRITIC_SCORE_KEYS, f"{vid}: CriticScore keys mismatch: {set(cs.keys())}"
        assert set(cs["completion_detail"].keys()) == {
            "redundant_beat_pairs", "promise_payoff_match", "emotional_trigger_landed"
        }, f"{vid}: completion_detail keys mismatch"
        assert isinstance(cs["justification"], str) and cs["justification"], f"{vid}: empty justification"
        assert cs["composite"] > 0, f"{vid}: composite not computed (still 0)"
    assert meta_result["outcome"] in valid_outcomes, f"bad outcome: {meta_result['outcome']}"
    assert len(captured_events) == 1, f"expected exactly 1 critic_score event, got {len(captured_events)}"
    assert captured_events[0]["name"] == "critic_score"

    print("\nOK: every variant has a full 9-key CriticScore with real values; "
          "meta_critic_result has a valid outcome; C2 critic_score event fired.\n")

    if meta_result["outcome"] == "all_excluded_failure" or not meta_result.get("merge_candidate"):
        print("meta_critic produced no merge_candidate (outcome="
              f"{meta_result['outcome']}) -- skipping the merge-validation chain.\n")
        return 0

    # meta_critic_node (like every LangGraph node) returns only its OWN partial
    # state update (critic_scores/meta_critic_result/reasoning_trace) -- merge it
    # back into the full state (script_variants, product_truths, etc.) the same
    # way LangGraph's own fan-in would, so merge_validator_node has everything
    # it needs (e.g. script_variants, for the fallback-to-unmerged-variant path).
    return await merge_chain_smoke({**merged_state, **meta_out})


async def merge_chain_smoke(state: dict, max_rounds: int = 4) -> int:
    """=== 3. LIVE MERGE-VALIDATION CHAIN (real DashScope calls) ===

    Drives merge_validator_node (and copy_editor_node / meta_critic_node loop-backs,
    exactly as graph/build.py's conditional edges would route) against the REAL
    meta_critic_result produced above, until a terminal route (finalize/fallback) is
    hit or `max_rounds` is exceeded (a real bug -- the graph itself caps at 2 attempts
    via route_after_merge_validation, so this should never actually trip).
    """
    print("=== 3. LIVE MERGE-VALIDATION CHAIN (real DashScope calls) ===")
    merge_validator_runnable = RunnableLambda(merge_validator_node)
    copy_editor_runnable = RunnableLambda(copy_editor_node)

    for round_num in range(1, max_rounds + 1):
        print(f"\n-- round {round_num}: merge_validator_node --")
        cv_events = []
        async for ev in merge_validator_runnable.astream_events(state, version="v2"):
            if ev.get("event") == "on_custom_event":
                cv_events.append(ev)
        cv_out = await merge_validator_runnable.ainvoke(state)
        state = {**state, **cv_out}

        last_attempt = state["merge_attempts"][-1]
        cc = last_attempt["coherence_check"]
        print(f"   attempt={last_attempt['attempt_number']} outcome={last_attempt['outcome']}")
        print(f"   pacing_recheck={last_attempt['pacing_recheck']}")
        print(f"   coherence: passed={cc['passed']} failure_kind={cc['failure_kind']} "
              f"voice_consistency={cc['voice_consistency']} promise_payoff_match={cc['promise_payoff_match']} "
              f"register_shift_flags={cc['register_shift_flags']}")
        print(f"   justification={cc['justification']}")
        print(f"   captured merge_validated events: {[e['name'] for e in cv_events]}")

        route = route_after_merge_validation(state)
        print(f"   route_after_merge_validation -> {route!r}")

        if route == "finalize" or route == "fallback":
            ws = state["winning_script"]
            print(f"\n--- WINNING SCRIPT ({route}) ---")
            print(f"  text: {ws['text']}")
            print(f"  source_variant_ids: {ws['source_variant_ids']}")
            print(f"  beats: {len(ws['beats'])}")
            assert state.get("winning_script"), "route was terminal but winning_script was never set"
            assert {"text", "beats", "source_variant_ids"} <= set(ws.keys())
            print(f"\nOK: merge chain reached a terminal route ({route}) after "
                  f"{round_num} validator round(s); winning_script is set.\n")
            return 0

        if route == "copy_editor":
            print("   -- running copy_editor_node --")
            ce_out = await copy_editor_runnable.ainvoke(state)
            state = {**state, **ce_out}
            last_ce = state.get("last_copy_edit")
            print(f"   copy_edit constraint_check: {last_ce['constraint_check'] if last_ce else None}")
        elif route == "meta_critic":
            print("   -- re-running meta_critic_node (swap retry) --")
            mc_runnable = RunnableLambda(meta_critic_node)
            mc_out = await mc_runnable.ainvoke(state)
            state = {**state, **mc_out}
        else:
            raise AssertionError(f"unexpected route: {route!r}")

    raise AssertionError(f"merge-validation chain did not reach a terminal route within {max_rounds} rounds")


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--structural-only", action="store_true", help="skip the live API smoke test")
    args = parser.parse_args()

    rc = structural_check()
    if rc != 0 or args.structural_only:
        return rc
    return await functional_smoke()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
