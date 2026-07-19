"""
Phase 4 adversarial integration edge cases (independent review of the Continuity
retry loop). These target intersections the existing Phase 4 suite does NOT cover:

  * multi-round human review on the SAME shot (retry_with_edit that still drifts);
  * a shot that hard-FAILS Video-Gen during a Continuity-triggered retry, then
    falls back through Ken-Burns and terminates cleanly;
  * MAX_AUTO_RETRIES=0 (first drift goes straight to human review);
  * the "review" dead-end for an unknown resume value (clean termination);
  * DURABILITY of a human "approve" across LATER loop passes driven by a DIFFERENT
    shot -- this is the bug this file's fix addresses;
  * the Continuity Agent's real frame-extraction failure path through the FULL
    compiled graph (not just the isolated unit).

The full-graph tests drive graph.build.build_graph with every network boundary
faked exactly as tests/test_continuity_loop_e2e.py does; the gate-level tests
reuse tests/test_continuity_gate.py's small single-node harness.
"""
from __future__ import annotations

import os
import tempfile

import pytest

from agents.continuity_agent import DRIFT_THRESHOLD
from agents.continuity_gate import (
    CONTINUITY_APPROVED_KEY,
    MAX_AUTO_RETRIES,
    continuity_gate_node,
    route_after_continuity_gate,
)
from agents.video_gen_node import VideoGenAPIError
from graph.build import build_graph

from tests._fakes import FakeOpenAIClient
from tests._phase3_graph import (
    patch_assembly_boundaries,
    patch_format_export_boundaries,
    patch_phase3_boundaries,
    patch_voiceover_boundaries,
)
from tests.test_continuity_gate import _build_gate_graph, _gen, _shot, _state
from tests.test_continuity_loop_e2e import _initial_state, _patch_upstream

_OVER = DRIFT_THRESHOLD + 0.25
_WITHIN = DRIFT_THRESHOLD - 0.2
_S2_MARKER = "asymmetric rear vent"  # unique to s2's description (test_graph_build)


def _drift_json(score: float, justification: str = "j") -> str:
    return f'{{"drift_score": {score}, "justification": "{justification}"}}'


def _counting_wan(monkeypatch) -> dict:
    counts: dict[str, int] = {}

    async def _wan(*, image_url, prompt, negative_prompt, duration_sec, resolution):
        leaf = image_url.rsplit("/", 1)[-1].split("?", 1)[0]
        counts[leaf] = counts.get(leaf, 0) + 1
        return f"http://wan.example.com/{leaf}/attempt{counts[leaf]}.mp4"

    monkeypatch.setattr("agents.video_gen_node._call_wan_video_gen", _wan)
    return counts


def _patch_clean_identity(monkeypatch) -> None:
    """Fake the v8 identity check clean (`same_object=True`) for every shot.

    None of the tests in this file are exercising the identity check itself
    (that's test_continuity_agent.py / test_continuity_gate.py's job) -- they
    drive the full graph to test the DRIFT retry/interrupt/durability machinery.
    Without this, the real `_score_one_shot_identity` would run against a fake
    `http://...` clip URL, fail at the network boundary (worst-case
    `same_object=False`), and continuity_gate.py's hard-identity-failure routing
    would fire and steal these tests' outcomes out from under the drift-focused
    assertions they're actually making.
    """
    from agents.continuity_agent import IdentityCheckResult

    async def _clean_identity(shot, entry, product_photos, client, extract):
        return IdentityCheckResult(
            matching_features=["clean match"], mismatching_features=[], same_object=True, confidence="high",
        )

    monkeypatch.setattr("agents.continuity_agent._score_one_shot_identity", _clean_identity)


def _cfg(thread_id: str) -> dict:
    # Generous recursion_limit: several loop passes + the ~9 upstream supersteps
    # can exceed LangGraph's default of 25.
    return {"configurable": {"thread_id": thread_id}, "recursion_limit": 80}


async def _drain(graph, arg, cfg):
    async for _ in graph.astream_events(arg, config=cfg, version="v2"):
        pass


# ===========================================================================
# ITEM 6 (the bug): a human "approve" must be DURABLE across a later loop pass
# that a DIFFERENT shot forces. Before the fix, the approved shot's clip entry
# still carried an over-threshold drift_score, so the Gate re-raised a review
# interrupt for it every subsequent pass.
# ===========================================================================
@pytest.mark.asyncio
async def test_human_approved_shot_not_rereviewed_on_later_loop_pass(monkeypatch):
    _patch_upstream(monkeypatch)
    _patch_clean_identity(monkeypatch)
    patch_phase3_boundaries(monkeypatch, fail_shot_s2=False)
    patch_voiceover_boundaries(monkeypatch)  # Phase 5: parallel branch off merge_validator
    patch_assembly_boundaries(monkeypatch)  # Phase 5: fan-in join off voiceover + continuity_gate
    _counting_wan(monkeypatch)

    # s1 and s2 both drift forever; s3 is always clean. s1 will be APPROVED; s2 is
    # kept looping via repeated retry_with_edit so the graph keeps running gate
    # passes AFTER s1 is approved -- exactly the condition that re-reviews s1.
    async def _oracle(shot, entry, product_photos, client, extract):
        if shot["shot_id"] in ("s1", "s2"):
            return _OVER, "drifted"
        return _WITHIN, "clean"

    monkeypatch.setattr("agents.continuity_agent._score_one_shot", _oracle)

    graph = await build_graph()
    cfg = _cfg("p4-approve-durable")
    await _drain(graph, _initial_state("p4-approve-durable"), cfg)

    s1_reviews = 0
    s2_edits = 0
    guard = 0
    while guard < 40:
        guard += 1
        st = await graph.aget_state(cfg)
        if not st.interrupts:
            break
        sid = st.interrupts[0].value["shot_id"]
        if sid == "s1":
            s1_reviews += 1
            res = "approve"
        else:  # s2: keep it looping, then finally approve to let the run finish
            s2_edits += 1
            res = "retry_with_edit" if s2_edits <= 2 else "approve"
        await _drain(graph, Command(resume={"resolution": res}), cfg)

    final = await graph.aget_state(cfg)
    assert not final.interrupts
    assert final.next == ()
    # The whole point: s1 was surfaced for human review EXACTLY once.
    assert s1_reviews == 1, f"approved shot s1 was re-reviewed {s1_reviews}x (approval not durable)"
    s1_queue = [e for e in final.values["human_review_queue"] if e["shot_id"] == "s1"]
    assert len(s1_queue) == 1  # not one duplicate queue entry per re-review
    by_id = {s["shot_id"]: s for s in final.values["shot_list"]}
    assert by_id["s1"]["status"] == "passed"
    assert by_id["s2"]["status"] == "passed"
    assert by_id["s3"]["status"] == "passed"


# ===========================================================================
# ITEM 1: multi-round review on the SAME shot. retry_with_edit is uncapped and
# routes back to Video-Gen; if the regenerated clip STILL drifts, a SECOND
# interrupt must fire for the same shot with a fresh, correct HumanReviewEntry.
# ===========================================================================
@pytest.mark.asyncio
async def test_retry_with_edit_that_still_drifts_reinterrupts_same_shot(monkeypatch):
    _patch_upstream(monkeypatch)
    _patch_clean_identity(monkeypatch)
    patch_phase3_boundaries(monkeypatch, fail_shot_s2=False)
    patch_voiceover_boundaries(monkeypatch)  # Phase 5: parallel branch off merge_validator
    patch_assembly_boundaries(monkeypatch)  # Phase 5: fan-in join off voiceover + continuity_gate

    # Count generation calls FOR s2 specifically (via its unique prompt marker) --
    # NOT via the final persisted video_uri, which the fake OSS persist step
    # (patch_phase3_boundaries) deliberately collapses to the same job/shot-scoped
    # path regardless of which Wan attempt produced it (mirrors the real OSS-key
    # convention: a shot's clip lives at one key, overwritten on regeneration --
    # so the URI is NOT a valid "was this regenerated" signal).
    s2_calls = {"n": 0}

    async def _wan(*, image_url, prompt, negative_prompt, duration_sec, resolution):
        if _S2_MARKER in prompt:
            s2_calls["n"] += 1
        leaf = image_url.rsplit("/", 1)[-1].split("?", 1)[0]
        return f"http://wan.example.com/{leaf}.mp4"

    monkeypatch.setattr("agents.video_gen_node._call_wan_video_gen", _wan)

    async def _oracle(shot, entry, product_photos, client, extract):
        # s2 drifts on EVERY generation (including the human-authorized retry).
        if shot["shot_id"] == "s2":
            return _OVER, "still drifting"
        return _WITHIN, "clean"

    monkeypatch.setattr("agents.continuity_agent._score_one_shot", _oracle)

    graph = await build_graph()
    cfg = _cfg("p4-multi-round")
    await _drain(graph, _initial_state("p4-multi-round"), cfg)

    # First review interrupt: s2 exhausted its 2 automatic retries -- 1 initial
    # generation + 2 auto-retries = 3 Wan calls for s2 by this point.
    st = await graph.aget_state(cfg)
    assert st.interrupts[0].value["shot_id"] == "s2"
    paused = {s["shot_id"]: s for s in st.values["shot_list"]}
    assert paused["s2"]["retry_count"] == MAX_AUTO_RETRIES
    assert s2_calls["n"] == 1 + MAX_AUTO_RETRIES
    calls_before_edit = s2_calls["n"]

    # Human-authorized retry (uncapped) -- pushes retry_count past the cap.
    await _drain(graph, Command(resume={"resolution": "retry_with_edit"}), cfg)

    # The regenerated clip STILL drifts -> a SECOND interrupt fires for the SAME
    # shot with a FRESH review entry, and exactly ONE new Wan call was made for it
    # (proving this round genuinely regenerated the clip rather than reusing the
    # stale, already-exhausted entry).
    st2 = await graph.aget_state(cfg)
    assert len(st2.interrupts) == 1
    surfaced2 = st2.interrupts[0].value
    assert surfaced2["shot_id"] == "s2"
    assert surfaced2["drift_score"] > DRIFT_THRESHOLD
    paused2 = {s["shot_id"]: s for s in st2.values["shot_list"]}
    assert paused2["s2"]["retry_count"] == MAX_AUTO_RETRIES + 1  # uncapped bump
    assert s2_calls["n"] == calls_before_edit + 1  # exactly one fresh regeneration
    # While paused on the SECOND interrupt, only the FIRST review entry is
    # committed -- `review_queue.append()` for round 2 runs before `interrupt()`,
    # but nothing this node returns commits until it finishes WITHOUT pausing
    # (see continuity_gate.py's module docstring). So exactly 1 entry is visible
    # here; the second commits only once this round is resolved below.
    assert len(st2.values["human_review_queue"]) == 1

    # Resolve the second round -> now BOTH review rounds are committed.
    await _drain(graph, Command(resume={"resolution": "accept_fallback"}), cfg)
    final = await graph.aget_state(cfg)
    assert not final.interrupts
    assert final.next == ()
    assert len(final.values["human_review_queue"]) == 2  # two distinct review rounds
    by_id = {s["shot_id"]: s for s in final.values["shot_list"]}
    assert by_id["s2"]["status"] == "fallback"


# ===========================================================================
# ITEM 2: a shot that hard-FAILS Video-Gen during a Continuity-triggered retry
# must fall back through Ken-Burns (fallback_requested -> fallback), skip
# Continuity scoring, and let the graph terminate -- no loop, no crash.
# ===========================================================================
@pytest.mark.asyncio
async def test_hard_fail_during_continuity_retry_falls_back_and_terminates(monkeypatch):
    _patch_upstream(monkeypatch)
    _patch_clean_identity(monkeypatch)
    patch_phase3_boundaries(monkeypatch, fail_shot_s2=False)  # provides OSS + Ken-Burns fakes
    patch_voiceover_boundaries(monkeypatch)  # Phase 5: parallel branch off merge_validator
    patch_assembly_boundaries(monkeypatch)  # Phase 5: fan-in join off voiceover + continuity_gate

    # s2 SUCCEEDS on its first generation (so it can drift + auto-retry), then the
    # Wan call FAILS on the retry regeneration -- a genuine VideoGenAPIError.
    s2_calls = {"n": 0}

    async def _wan(*, image_url, prompt, negative_prompt, duration_sec, resolution):
        if _S2_MARKER in prompt:
            s2_calls["n"] += 1
            if s2_calls["n"] >= 2:
                raise VideoGenAPIError("simulated hard failure on the continuity retry")
        leaf = image_url.rsplit("/", 1)[-1].split("?", 1)[0]
        return f"http://wan.example.com/{leaf}/{s2_calls['n']}.mp4"

    monkeypatch.setattr("agents.video_gen_node._call_wan_video_gen", _wan)

    async def _oracle(shot, entry, product_photos, client, extract):
        if shot["shot_id"] == "s2":  # only ever scored on pass 1 (passed then)
            return _OVER, "drift"
        return _WITHIN, "clean"

    monkeypatch.setattr("agents.continuity_agent._score_one_shot", _oracle)

    graph = await build_graph()
    cfg = _cfg("p4-hardfail-retry")
    await _drain(graph, _initial_state("p4-hardfail-retry"), cfg)

    final = await graph.aget_state(cfg)
    assert not final.interrupts  # never went to human review
    assert final.next == ()  # terminated cleanly
    by_id = {s["shot_id"]: s for s in final.values["shot_list"]}
    # The retry hard-failed -> handed to Ken-Burns -> real fallback clip + status.
    assert by_id["s2"]["status"] == "fallback"
    assert s2_calls["n"] == 2  # generated once, retry attempted once (then failed)
    gen = final.values["generated_shots"]
    assert gen["s2"]["video_uri"].startswith("http://oss.example.com/jobs/p4-hardfail-retry/shots/s2/")
    assert "drift_score" not in gen["s2"]  # the fallback clip was NOT drift-scored
    # retry_count records the one auto-retry; the infra failure did not touch it.
    assert by_id["s2"]["retry_count"] == 1
    for sid in ("s1", "s3"):
        assert by_id[sid]["status"] == "passed"
        assert by_id[sid]["retry_count"] == 0


# ===========================================================================
# ITEM 3: MAX_AUTO_RETRIES = 0 -> the very first over-threshold drift goes
# STRAIGHT to human review (retry_count(0) < 0 is false), no auto-retry, no crash.
# ===========================================================================
@pytest.mark.asyncio
async def test_max_auto_retries_zero_goes_straight_to_review(monkeypatch):
    monkeypatch.setattr("agents.continuity_gate.MAX_AUTO_RETRIES", 0)
    graph = _build_gate_graph()
    cfg = {"configurable": {"thread_id": "p4-cap-zero"}}
    await graph.ainvoke(_state([_shot("s1", retry_count=0)], {"s1": _gen(_OVER)}), config=cfg)

    st = await graph.aget_state(cfg)
    assert st.next == ("continuity_gate",)  # paused
    assert len(st.interrupts) == 1
    assert st.interrupts[0].value["shot_id"] == "s1"
    # It went to review, NOT an auto-retry (would have set status "pending").
    assert st.values["shot_list"][0]["status"] == "passed"
    assert st.values["shot_list"][0]["retry_count"] == 0

    # Resume approve -> resolves cleanly.
    result = await graph.ainvoke(Command(resume={"resolution": "approve"}), config=cfg)
    st2 = await graph.aget_state(cfg)
    assert st2.next == ()
    assert result["shot_list"][0]["status"] == "passed"


# ===========================================================================
# ITEM 5: an unknown resume value leaves the shot in "review"; the router sends
# the graph to END with that shot still visibly unresolved -- clean, not a hang.
# ===========================================================================
@pytest.mark.asyncio
@pytest.mark.parametrize("bad", [{"resolution": "frobnicate"}, "not_a_real_choice", {"nope": 1}])
async def test_unknown_resolution_leaves_shot_in_review_and_terminates(monkeypatch, bad):
    graph = _build_gate_graph()
    cfg = {"configurable": {"thread_id": f"p4-review-deadend-{hash(str(bad))}"}}
    await graph.ainvoke(
        _state([_shot("s1", retry_count=MAX_AUTO_RETRIES)], {"s1": _gen(_OVER)}), config=cfg
    )
    result = await graph.ainvoke(Command(resume=bad), config=cfg)

    st = await graph.aget_state(cfg)
    assert st.next == ()  # terminated, no hang
    assert not st.interrupts
    shot = result["shot_list"][0]
    assert shot["status"] == "review"  # not silently passed
    # The router treats "review" as a terminal dead-end (neither pending nor
    # fallback_requested), so the graph ends -- the pipeline does NOT pretend the
    # shot is fine.
    assert route_after_continuity_gate(result) == "end"


# ===========================================================================
# ITEM 7: the Continuity Agent's REAL frame-extraction failure path, run through
# the FULL compiled graph (not just the isolated unit). One shot's ffmpeg failure
# -> worst-case drift 1.0 -> flows into the retry/review path; siblings score
# normally via the real Qwen-VL call; the astream_events run never crashes.
# ===========================================================================
@pytest.mark.asyncio
async def test_frame_extraction_failure_through_compiled_graph(monkeypatch):
    _patch_upstream(monkeypatch)
    _patch_clean_identity(monkeypatch)
    patch_phase3_boundaries(monkeypatch, fail_shot_s2=False)
    patch_voiceover_boundaries(monkeypatch)  # Phase 5: parallel branch off merge_validator
    patch_assembly_boundaries(monkeypatch)  # Phase 5: fan-in join off voiceover + continuity_gate
    _counting_wan(monkeypatch)

    def _extract(video_uri: str, duration_sec: float) -> str:
        if "/shots/s2/" in video_uri:  # s2's persisted clip -> ffmpeg boundary blows up
            raise RuntimeError("simulated ffmpeg extraction failure")
        fd, path = tempfile.mkstemp(suffix=".jpg", prefix="p4_extract_")
        os.close(fd)
        with open(path, "wb") as fh:
            fh.write(b"\xff\xd8\xff\xe0fake")
        return path

    # Real score_continuity / _score_one_shot / _call_qwen_vl_drift are used; only
    # the ffmpeg boundary and the vision client are faked.
    monkeypatch.setattr("agents.continuity_agent.extract_midpoint_frame", _extract)
    monkeypatch.setattr(
        "agents.continuity_agent.AsyncOpenAI",
        lambda *a, **k: FakeOpenAIClient([_drift_json(_WITHIN, "clean sibling")]),
    )
    monkeypatch.setenv("MODEL_VISION", "qwen-vl-test")
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")
    monkeypatch.setenv("DASHSCOPE_BASE_URL", "http://test")

    graph = await build_graph()
    cfg = _cfg("p4-extract-fail")
    await _drain(graph, _initial_state("p4-extract-fail"), cfg)

    st = await graph.aget_state(cfg)
    # s2 could not be checked -> worst-case 1.0 every pass -> exhausted -> interrupt.
    assert len(st.interrupts) == 1
    surfaced = st.interrupts[0].value
    assert surfaced["shot_id"] == "s2"
    assert surfaced["drift_score"] == 1.0
    paused = {s["shot_id"]: s for s in st.values["shot_list"]}
    assert paused["s2"]["retry_count"] == MAX_AUTO_RETRIES
    gen = st.values["generated_shots"]
    # Siblings scored normally via the real vision path, NOT corrupted to 1.0.
    for sid in ("s1", "s3"):
        assert paused[sid]["status"] == "passed"
        assert gen[sid]["drift_score"] == pytest.approx(_WITHIN)
    assert gen["s2"]["drift_score"] == 1.0

    # Resolve so the run terminates cleanly (proving no crash/hang on this path).
    await _drain(graph, Command(resume={"resolution": "accept_fallback"}), cfg)
    final = await graph.aget_state(cfg)
    assert not final.interrupts
    assert final.next == ()
    assert {s["shot_id"]: s for s in final.values["shot_list"]}["s2"]["status"] == "fallback"


# ===========================================================================
# ITEM 4 (fact-pattern check, not a live bug): `{**shot, ...}` copies in the Gate
# are shallow, so the ORIGINAL and the resolved shot share the SAME nested
# `justification` dict object. This is only a latent hazard IF something later
# mutates a justification in place -- nothing in the wired pipeline does (grep
# confirms read-only use in video_gen_node._build_prompt). Pin the fact pattern.
# ===========================================================================
@pytest.mark.asyncio
async def test_gate_resolved_shot_shares_justification_object_but_it_is_never_mutated():
    graph = _build_gate_graph()
    cfg = {"configurable": {"thread_id": "p4-aliasing"}}
    original = _shot("s1", retry_count=MAX_AUTO_RETRIES)
    orig_just = original["justification"]
    await graph.ainvoke(_state([original], {"s1": _gen(_OVER)}), config=cfg)
    result = await graph.ainvoke(Command(resume={"resolution": "retry_with_edit"}), config=cfg)

    resolved = result["shot_list"][0]
    # Shallow copy: same nested justification object is aliased. Documented here so
    # a future in-place mutation of a justification dict is a KNOWN risk, not a
    # surprise. The value is unchanged either way.
    assert resolved["justification"] == orig_just
    assert resolved["status"] == "pending"  # a new top-level value, not aliased
