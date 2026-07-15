"""
Full retry-loop integration tests (Phase 4 exit criteria, §5.10).

These drive the ENTIRE compiled LangGraph (graph/build.py), faking every network
boundary, to prove the Continuity retry CYCLE works as a real graph topology --
not just the nodes in isolation:

  Scenario A -- a shot drifts on the first pass, is AUTOMATICALLY retried once,
    the drift resolves, and the shot ends "passed" with retry_count incremented
    exactly once and a FRESH generated_shots entry from the re-generation.

  Scenario B -- a shot drifts and never resolves, EXHAUSTS its automatic retries,
    raises a REAL interrupt(), and a resume with "accept_fallback" routes it back
    through the (real, already-built) Ken-Burns Fallback Node to a final
    "fallback" status.

Upstream LLM boundaries are faked exactly as tests/test_graph_end_to_end.py does
(shared canned payloads from tests/test_graph_build.py). The Phase 3 boundaries
(Wan / OSS / Ken-Burns render+upload) are faked via tests/_phase3_graph.py, with
a COUNTING Wan fake layered on so we can assert re-generation actually happened.
The per-shot Continuity scoring unit (`_score_one_shot`, i.e. ffmpeg frame
extraction + the Qwen-VL call) is faked deterministically by shot_id/retry_count
-- the real vision+ffmpeg path is already covered by test_continuity_agent.py;
here the point is the LOOP, so a deterministic drift oracle keeps the topology
assertions crisp.
"""
from __future__ import annotations

import pytest
from langgraph.types import Command

from agents.continuity_agent import DRIFT_THRESHOLD
from graph.build import build_graph
from tests._fakes import make_content_routed_sync_openai, make_fake_async_openai
from tests._phase3_graph import (
    patch_assembly_boundaries,
    patch_phase3_boundaries,
    patch_voiceover_boundaries,
)
from tests.test_graph_build import (
    CHECKER_ROUTES,
    CONCEPT_AGENT_PAYLOAD,
    HOOK_PAYLOAD,
    SHOT_LIST_CALL_A_PAYLOAD,
    SHOT_LIST_CALL_B_PAYLOAD,
    TREATMENT_PAYLOAD,
    TRUTH_EXTRACTOR_PAYLOAD,
)

_SEEDED_CAP = 1.00  # same feasible cap as test_graph_end_to_end.py
_DRIFTY_SHOT = "s2"
_OVER = DRIFT_THRESHOLD + 0.3
_WITHIN = DRIFT_THRESHOLD - 0.2


def _patch_upstream(monkeypatch):
    """Fake every upstream LLM boundary (identical to test_graph_end_to_end.py)."""
    monkeypatch.setattr("agents.product_truth_extractor.AsyncOpenAI", make_fake_async_openai([TRUTH_EXTRACTOR_PAYLOAD]))
    monkeypatch.setattr("agents.concept_agent.AsyncOpenAI", make_fake_async_openai([CONCEPT_AGENT_PAYLOAD]))
    monkeypatch.setattr("agents.hook_checker.AsyncOpenAI", make_fake_async_openai([HOOK_PAYLOAD]))
    monkeypatch.setattr("agents.critic_llm.OpenAI", make_content_routed_sync_openai(CHECKER_ROUTES))
    monkeypatch.setattr("agents.treatment_agent.AsyncOpenAI", make_fake_async_openai([TREATMENT_PAYLOAD]))
    monkeypatch.setattr(
        "agents.shot_list_agent.AsyncOpenAI",
        make_fake_async_openai([SHOT_LIST_CALL_A_PAYLOAD, SHOT_LIST_CALL_B_PAYLOAD]),
    )


def _patch_counting_wan(monkeypatch) -> dict:
    """Layer a per-shot COUNTING Wan fake so we can assert re-generation happened.

    Returns a mutable dict mapping reference-photo leaf -> call count. Each call
    returns a URL tagged with its attempt number so a re-generated clip is
    distinguishable from the original.
    """
    counts: dict[str, int] = {}

    async def _counting_wan(*, image_url, prompt, negative_prompt, duration_sec, resolution):
        leaf = image_url.rsplit("/", 1)[-1].split("?", 1)[0]
        counts[leaf] = counts.get(leaf, 0) + 1
        return f"http://wan.example.com/{leaf}/attempt{counts[leaf]}.mp4"

    monkeypatch.setattr("agents.video_gen_node._call_wan_video_gen", _counting_wan)
    return counts


def _patch_continuity_scoring(monkeypatch, *, always_drift: bool):
    """Fake the per-shot scoring unit deterministically by shot_id/retry_count.

    `always_drift=False` -> the drifty shot drifts only until it has been retried
    once (models "the retry fixed it"); `always_drift=True` -> it drifts forever
    (models "retries can't fix it" -> exhaustion -> human review).

    This module's tests are exercising the DRIFT retry cycle specifically, not
    the v8 identity check -- so the identity check is ALSO faked here, always
    clean (`same_object=True`). Without this, the real `_score_one_shot_identity`
    would run against a fake `http://...` clip URL, fail at the network boundary,
    and the NEW hard-identity-failure routing in continuity_gate.py would kick
    in and steal these tests' outcomes; see test_continuity_agent.py /
    test_continuity_gate.py for the identity check's own dedicated tests.
    """
    from agents.continuity_agent import IdentityCheckResult

    async def _fake_score_one_shot(shot, entry, product_photos, client, extract):
        if shot["shot_id"] == _DRIFTY_SHOT and (always_drift or shot.get("retry_count", 0) < 1):
            return _OVER, "product identity drifted"
        return _WITHIN, "clean match"

    async def _fake_score_one_shot_identity(shot, entry, product_photos, client, extract):
        return IdentityCheckResult(
            matching_features=["clean match"], mismatching_features=[], same_object=True, confidence="high",
        )

    monkeypatch.setattr("agents.continuity_agent._score_one_shot", _fake_score_one_shot)
    monkeypatch.setattr("agents.continuity_agent._score_one_shot_identity", _fake_score_one_shot_identity)


def _initial_state(job_id: str) -> dict:
    return {
        "job_id": job_id,
        "product_photos": ["http://example.com/a.jpg"],
        # Positive-Only Truths fix: signals the explicit authentic/well-loved
        # carve-out -- see test_graph_build.py's GOOD_FACTS/FOUR_GOOD_VARIANTS
        # comment for why (this shared fixture's narrative is imperfection-led).
        "brief": "a durable everyday case with authentic, well-loved character",
        "budget_ledger": {"cap": _SEEDED_CAP, "spent": 0.0, "per_shot": {}},
    }


# ---------------------------------------------------------------------------
# Scenario A: automatic retry resolves the drift.
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_drifted_shot_auto_retried_once_then_passes(monkeypatch):
    _patch_upstream(monkeypatch)
    patch_phase3_boundaries(monkeypatch, fail_shot_s2=False)
    patch_voiceover_boundaries(monkeypatch)  # Phase 5: parallel branch off merge_validator
    patch_assembly_boundaries(monkeypatch)  # Phase 5: fan-in join off voiceover + continuity_gate
    wan_counts = _patch_counting_wan(monkeypatch)
    _patch_continuity_scoring(monkeypatch, always_drift=False)

    graph = await build_graph()
    cfg = {"configurable": {"thread_id": "loop-auto-retry"}}
    custom_events = [
        e
        async for e in graph.astream_events(_initial_state("loop-auto-retry"), config=cfg, version="v2")
        if e.get("event") == "on_custom_event"
    ]

    values = (await graph.aget_state(cfg)).values
    by_id = {s["shot_id"]: s for s in values["shot_list"]}
    generated = values["generated_shots"]

    # The drifty shot was retried exactly once and now passes.
    assert by_id[_DRIFTY_SHOT]["status"] == "passed"
    assert by_id[_DRIFTY_SHOT]["retry_count"] == 1
    # Its non-drifty siblings never retried.
    for sid, shot in by_id.items():
        if sid != _DRIFTY_SHOT:
            assert shot["retry_count"] == 0
            assert shot["status"] == "passed"

    # Its final drift_score is within threshold now (the re-generation resolved it).
    assert generated[_DRIFTY_SHOT]["drift_score"] <= DRIFT_THRESHOLD
    # Wan was hit once per shot on pass 1 plus exactly one re-generation for the
    # drifty shot -- proving the loop re-ran Video-Gen for ONLY the retried shot.
    assert sum(wan_counts.values()) == len(by_id) + 1  # 3 shots + 1 re-gen

    # No human review was needed; no interrupt fired.
    assert values.get("human_review_queue", []) == []
    assert not (await graph.aget_state(cfg)).interrupts
    # drift_scored fired for the retried shot on BOTH passes (over, then within).
    drift_events = [e for e in custom_events if e["name"] == "drift_scored"]
    drifty_events = [e for e in drift_events if e["data"]["shot_id"] == _DRIFTY_SHOT]
    assert [e["data"]["passed"] for e in drifty_events] == [False, True]


# ---------------------------------------------------------------------------
# Scenario D (Phase 5, §5.12): the Assembly Agent is a genuine fan-in JOIN of
# TWO branches with very different completion latency -- voiceover_caption_agent
# (one early superstep) and the continuity retry loop (>=2 passes here). This
# does NOT just assume LangGraph's join semantics handle a late/looping branch
# correctly (a node with two plain incoming edges into the SAME per-node
# trigger channel fires on the FIRST writer by default -- see graph/build.py's
# module docstring on why `defer=True` is required) -- it verifies against the
# REAL compiled graph that assembly_agent runs EXACTLY ONCE, only after the
# loop's FINAL pass resolves to "end", with both the early-finished voiceover
# AND the fully-final post-retry shot_list/generated_shots visible to it.
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_assembly_agent_runs_exactly_once_after_continuity_loop_resolves(monkeypatch):
    _patch_upstream(monkeypatch)
    patch_phase3_boundaries(monkeypatch, fail_shot_s2=False)
    patch_voiceover_boundaries(monkeypatch)  # finishes trivially fast (one early superstep)
    wan_counts = _patch_counting_wan(monkeypatch)
    _patch_continuity_scoring(monkeypatch, always_drift=False)  # 2-pass loop: drift, then resolve

    from agents.assembly_agent import AssemblyResult

    calls: list[dict] = []

    async def _counting_assemble_impl(shot_list, generated_shots, voiceover, winning_script, job_id, **kwargs):
        calls.append(
            {
                "shot_list": shot_list,
                "generated_shots": generated_shots,
                "voiceover": voiceover,
                "job_id": job_id,
            }
        )
        return AssemblyResult(
            master_cut_uri=f"http://oss.example.com/jobs/{job_id}/master_cut.mp4",
            shot_count=len(shot_list),
            total_duration_sec=15.0,
            degraded_beats=[],
        )

    monkeypatch.setattr("agents.assembly_agent._assemble_master_cut_impl", _counting_assemble_impl)

    graph = await build_graph()
    cfg = {"configurable": {"thread_id": "loop-assembly-join"}}
    custom_events = [
        e
        async for e in graph.astream_events(_initial_state("loop-assembly-join"), config=cfg, version="v2")
        if e.get("event") == "on_custom_event"
    ]

    # The continuity loop genuinely took 2 passes (same proof as Scenario A):
    # 3 shots on pass 1 + exactly 1 re-generation for the drifty shot.
    assert sum(wan_counts.values()) == 4

    # --- The join fired EXACTLY ONCE. ---------------------------------------
    assert len(calls) == 1, f"assembly_agent ran {len(calls)} time(s), expected exactly 1"
    master_cut_events = [e for e in custom_events if e["name"] == "master_cut_ready"]
    assert len(master_cut_events) == 1, f"master_cut_ready fired {len(master_cut_events)} time(s), expected exactly 1"

    # --- It ran with the FULLY-FINAL, post-retry shot_list/generated_shots --
    # (not a stale mid-loop snapshot from before the drifty shot's re-generation).
    call = calls[0]
    by_id = {s["shot_id"]: s for s in call["shot_list"]}
    assert by_id[_DRIFTY_SHOT]["status"] == "passed"
    assert by_id[_DRIFTY_SHOT]["retry_count"] == 1  # the retry already happened
    assert call["generated_shots"][_DRIFTY_SHOT]["drift_score"] <= DRIFT_THRESHOLD

    # --- ...AND the early-finished voiceover branch's output. ---------------
    assert call["voiceover"]["audio_uri"] == "http://oss.example.com/jobs/loop-assembly-join/voiceover.mp3"
    assert call["voiceover"]["caption_track_uri"] == "http://oss.example.com/jobs/loop-assembly-join/captions.json"

    # --- state["master_cut_uri"] reflects the (single) join's own output. ---
    final_values = (await graph.aget_state(cfg)).values
    assert final_values["master_cut_uri"] == "http://oss.example.com/jobs/loop-assembly-join/master_cut.mp4"
    assert master_cut_events[0]["data"]["uri"] == final_values["master_cut_uri"]
    assert "[assembly_agent]" in final_values["reasoning_trace"]

    # --- The graph is fully terminated (assembly_agent -> END), no dangling
    # interrupt/pending work.
    final_state = await graph.aget_state(cfg)
    assert final_state.next == ()
    assert not final_state.interrupts


# ---------------------------------------------------------------------------
# Scenario B: retries exhausted -> real interrupt -> accept_fallback -> Ken-Burns.
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_exhausted_retries_interrupt_then_accept_fallback_routes_to_ken_burns(monkeypatch):
    _patch_upstream(monkeypatch)
    patch_phase3_boundaries(monkeypatch, fail_shot_s2=False)
    patch_voiceover_boundaries(monkeypatch)  # Phase 5: parallel branch off merge_validator
    patch_assembly_boundaries(monkeypatch)  # Phase 5: fan-in join off voiceover + continuity_gate
    _patch_counting_wan(monkeypatch)
    _patch_continuity_scoring(monkeypatch, always_drift=True)

    graph = await build_graph()
    cfg = {"configurable": {"thread_id": "loop-exhaust"}}

    # Drive until the graph pauses at the real interrupt for the drifty shot.
    async for _ in graph.astream_events(_initial_state("loop-exhaust"), config=cfg, version="v2"):
        pass

    st = await graph.aget_state(cfg)
    assert len(st.interrupts) == 1
    surfaced = st.interrupts[0].value
    assert surfaced["shot_id"] == _DRIFTY_SHOT
    assert surfaced["drift_score"] > DRIFT_THRESHOLD

    # The drifty shot exhausted its automatic retries before the interrupt.
    paused_by_id = {s["shot_id"]: s for s in st.values["shot_list"]}
    assert paused_by_id[_DRIFTY_SHOT]["retry_count"] == 2

    # Resume with accept_fallback -> routes back through Ken-Burns to "fallback".
    async for _ in graph.astream_events(
        Command(resume={"resolution": "accept_fallback"}), config=cfg, version="v2"
    ):
        pass

    final = await graph.aget_state(cfg)
    assert not final.interrupts  # fully resolved
    assert final.next == ()
    by_id = {s["shot_id"]: s for s in final.values["shot_list"]}
    # The drifty shot got a REAL Ken-Burns fallback clip and terminal status.
    assert by_id[_DRIFTY_SHOT]["status"] == "fallback"
    generated = final.values["generated_shots"]
    assert _DRIFTY_SHOT in generated
    assert generated[_DRIFTY_SHOT]["video_uri"].startswith("http://oss.example.com/jobs/loop-exhaust/shots/s2/")
    # The review entry was recorded in the queue.
    queue = final.values["human_review_queue"]
    assert any(e["shot_id"] == _DRIFTY_SHOT for e in queue)
    # Siblings are untouched, still real passes.
    for sid, shot in by_id.items():
        if sid != _DRIFTY_SHOT:
            assert shot["status"] == "passed"
            assert shot["retry_count"] == 0


# ---------------------------------------------------------------------------
# Scenario C (v8 fix): a shot that reproducibly fails the IDENTITY check (not
# drift) gets exactly ONE automatic retry, then a SECOND consecutive identity
# failure routes it straight to a REAL Ken-Burns fallback clip -- no interrupt,
# no human review, and the normal drift retry/interrupt machinery (Scenario
# A/B above) is completely unaffected for a sibling shot.
# ---------------------------------------------------------------------------
def _patch_identity_only_failure(monkeypatch, *, always_fail_identity: bool):
    """Drift is always clean for every shot (isolates the identity check as the
    ONLY thing that can trigger a retry/fallback here); the drifty shot's
    IDENTITY check fails on every generation."""
    from agents.continuity_agent import IdentityCheckResult

    async def _fake_score_one_shot(shot, entry, product_photos, client, extract):
        return _WITHIN, "clean match"

    async def _fake_score_one_shot_identity(shot, entry, product_photos, client, extract):
        if shot["shot_id"] == _DRIFTY_SHOT and always_fail_identity:
            return IdentityCheckResult(
                matching_features=[],
                mismatching_features=["flat silhouette vs. reference's deep block"],
                same_object=False,
                confidence="high",
            )
        return IdentityCheckResult(
            matching_features=["deep rounded block"], mismatching_features=[], same_object=True, confidence="high",
        )

    monkeypatch.setattr("agents.continuity_agent._score_one_shot", _fake_score_one_shot)
    monkeypatch.setattr("agents.continuity_agent._score_one_shot_identity", _fake_score_one_shot_identity)


@pytest.mark.asyncio
async def test_reproducible_identity_failure_retries_once_then_real_ken_burns_fallback(monkeypatch):
    _patch_upstream(monkeypatch)
    patch_phase3_boundaries(monkeypatch, fail_shot_s2=False)
    patch_voiceover_boundaries(monkeypatch)  # Phase 5: parallel branch off merge_validator
    patch_assembly_boundaries(monkeypatch)  # Phase 5: fan-in join off voiceover + continuity_gate
    wan_counts = _patch_counting_wan(monkeypatch)
    _patch_identity_only_failure(monkeypatch, always_fail_identity=True)

    graph = await build_graph()
    cfg = {"configurable": {"thread_id": "loop-identity-fallback"}}

    async for _ in graph.astream_events(_initial_state("loop-identity-fallback"), config=cfg, version="v2"):
        pass

    final = await graph.aget_state(cfg)
    # A hard identity failure never raises a human-review interrupt, even after
    # its one retry also fails -- it routes straight to Ken-Burns instead.
    assert not final.interrupts
    assert final.next == ()

    by_id = {s["shot_id"]: s for s in final.values["shot_list"]}
    assert by_id[_DRIFTY_SHOT]["status"] == "fallback"
    assert by_id[_DRIFTY_SHOT]["retry_count"] == 1  # exactly one auto-retry, never MAX_AUTO_RETRIES worth
    assert by_id[_DRIFTY_SHOT]["identity_hard_fail_streak"] == 2

    generated = final.values["generated_shots"]
    assert _DRIFTY_SHOT in generated
    # A REAL Ken-Burns fallback clip -- the real ken_burns_fallback_node ran,
    # not a stub. Its OSS key convention (jobs/<job_id>/shots/<shot_id>/...)
    # is only ever produced by that node's real upload path.
    assert generated[_DRIFTY_SHOT]["video_uri"].startswith(
        "http://oss.example.com/jobs/loop-identity-fallback/shots/s2/"
    )
    assert "drift_score" not in generated[_DRIFTY_SHOT], "a Ken-Burns fallback clip is never drift-scored"

    # No human review was ever needed for this shot.
    assert final.values.get("human_review_queue", []) == []

    # Wan was hit once per shot on pass 1 plus exactly ONE re-generation for the
    # drifty shot's single identity retry -- proving the loop regenerated ONLY
    # once before giving up, not MAX_AUTO_RETRIES times.
    assert sum(wan_counts.values()) == len(by_id) + 1

    # The sibling shots are completely unaffected by s2's identity failures --
    # normal drift/retry machinery never even engaged for them.
    for sid, shot in by_id.items():
        if sid != _DRIFTY_SHOT:
            assert shot["status"] == "passed"
            assert shot["retry_count"] == 0
            assert "identity_hard_fail_streak" not in shot
