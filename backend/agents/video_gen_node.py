"""
Video-Gen Node — orchestration around Wan2.6-i2v-us (Phase 3).
Spec of record: docs/TECHNICAL_DOCUMENTATION.md §5.8. Ken-Burns Fallback Node
(§5.9) is RR's separate Phase 3 task (agents/ken_burns_fallback_node.py, now
built and wired) -- see the INTERFACE section below for the hand-off this
module assumes.

Confirmed against the ACTUAL merged code before writing anything (per this
task's own instruction not to assume field names from docs):
  * `graph.state.Shot` / `graph.shot_schema.ShotModel` (C1/C3, frozen) --
    fields used here: shot_id, beat_role, description, shot_type, camera_move,
    framing, lighting, negative_prompt, reference_image_id, text_overlay_zone,
    duration_sec, allocated_budget, voiceover_line, justification
    ({script_quote, truth_fact_id, treatment_ref}), status, retry_count.
  * `graph.state.GeneratedShot` (C1, frozen): {video_uri, drift_score
    (NotRequired, set later by Continuity), attempt}.
  * `agents.shot_list_agent`: MIN_SHOT_DURATION_SEC/MAX_SHOT_DURATION_SEC (3-5s
    per shot), NEGATIVE_PROMPT_BOILERPLATE already baked into every shot's own
    `negative_prompt` field -- reused here, not rebuilt (see requirement 3 note
    below).
  * `agents.budget_gate`: RATE_720P/RATE_1080P ($/sec Wan pricing) and its own
    documented expectation ("A future Video-Gen Node derives resolution/retry
    affordability by comparing allocated_budget against duration_sec ×
    RATE_1080P, not by reading a categorical field") -- this module is that
    future node; the budget-clamp policy below is that comparison, made real.
  * `.env.example`: MODEL_VIDEO=wan2.6-i2v-us (image-to-video only, confirmed;
    never falls back to text-to-video -- img_url is always supplied).

PROMPT-CONSTRUCTION MAPPING (requirement 2; §5.8's formula is "Subject →
Action/Motion → Camera → Lighting → Composition → Mood → Quality"). Each
section is built from a real, already-typed field -- nothing is re-derived
from free text except where noted:
  * Subject      <- the shot's cited product_truths[] fact (justification.
                    truth_fact_id lookup). A concrete factual anchor, not an
                    invented product name (ProductCutState has no product-name
                    field at all -- only brief/product_photos).
  * Action/Motion <- shot.description VERBATIM. The Shot-List Agent (§5.6 Call
                    B) already wrote this as the shot's 80-120 word action/
                    motion narrative, itself grounded in the same cited fact
                    and reasoned through the camera-affordance rubric --
                    rebuilding it from scratch here would just be a worse
                    rewrite of already-grounded, already-validated content.
  * Camera        <- shot.camera_move (push_in/orbit/static/pan/tilt_up/
                    pull_back/rack_focus -- the v2 C3 addition, handled
                    generically since it's just another enum value here).
  * Lighting      <- shot.lighting (the one shared lighting/style string
                    reused across every shot in the job, per C1).
  * Composition   <- shot.shot_type (hook_hero/macro_detail/lifestyle_context/
                    hero_reframe/cta_endcard/product_in_hand -- the v2 C3
                    addition) + shot.framing + a text_overlay_zone reservation
                    note. When shot_type is "product_in_hand" (the
                    human-interaction composition), an explicit POSITIVE
                    continuity clause is appended ("natural hand interaction
                    with five anatomically correct fingers... product stays
                    centered, no scene cut") -- per
                    docs/DERISK_VIDEO_GEN_RESULT.md §6, this positive
                    instruction (not the negative prompt alone) is what
                    empirically stopped the product-vanishing failure mode
                    during human-interaction shots in the real de-risk test.
  * Mood          <- treatment.director_persona + treatment.pacing_philosophy
                    (the whole ad's directorial voice, applied per shot).
  * Quality       <- a fixed quality/fidelity boilerplate clause (this is
                    genuinely generic across every shot in every job -- unlike
                    every other section, there is no real per-shot/per-product
                    field this could be grounded in).

NEGATIVE PROMPT (requirement 3). Reused verbatim from `shot.negative_prompt`
-- the Shot-List Agent already built this (its own
NEGATIVE_PROMPT_BOILERPLATE, identity-first ordering, plus any per-shot
extra risk terms) covering exactly the guardrail list requirement 3 asks
for (morphing/melting, flickering, deformed hands/fused fingers, warped
product shape, text/logo distortion). Rebuilding a second copy here would
risk the two silently drifting apart; passed straight through to the
DashScope call's own `negative_prompt` parameter (a first-class, separate
parameter from `prompt` in the native SDK -- not string-concatenated).

BUDGET CLAMP (requirement 5, `_resolve_generation_params`). Per
agents/budget_gate.py's own documented contract: compare `allocated_budget`
against `duration_sec x RATE_1080P` / `x RATE_720P`.
  * allocated_budget >= 1080p cost  -> generate at full duration_sec, 1080p.
  * allocated_budget >= 720p cost   -> generate at full duration_sec, 720p
                                       (resolution clamped, duration untouched).
  * otherwise                       -> clamp DURATION down to what
                                       allocated_budget affords at 720p (the
                                       cheapest rate). If even
                                       MIN_SHOT_DURATION_SEC at 720p doesn't
                                       fit (can genuinely happen: the Budget
                                       Gate's §5.7 floor case sets EVERY
                                       shot's allocated_budget to a flat
                                       FLOOR_COST regardless of that shot's
                                       own duration_sec when the cap can't fit
                                       even at floor), the shot is handed off
                                       as a "budget_exceeded" failure WITHOUT
                                       ever calling the video-gen API --
                                       nothing is spent generating something
                                       that structurally cannot be afforded.
The realized resolution/duration and whether a clamp occurred are recorded on
the *GeneratedShot* entry (resolution_used/duration_sec_used/budget_clamped),
not on the Shot itself -- see "KNOWN DEPARTURES" below for why.

PARALLEL FAN-OUT (requirement 1) -- SELF-CONTAINED, NO C1 CHANGE NEEDED. This
module builds and runs its OWN small internal LangGraph StateGraph (dispatch
-> Send() one branch per shot -> per-shot node -> join) inside
`generate_videos()`, entirely private to this module. Two accumulator fields
in that INTERNAL graph's own state type (`_VideoGenGraphState`,
`generated_shots` / `failures`) use `Annotated[..., reducer]` so N parallel
branches merge instead of colliding -- this is required (LangGraph raises
InvalidUpdateError if two branches in the same superstep write the same
un-reduced key), but it is entirely private to this module's internal
subgraph. The OUTER `graph.state.ProductCutState` is NEVER touched: from the
outer graph's point of view (once this is wired in), `video_gen_node` is one
ordinary node that happens to parallelize its own work internally and returns
ONE consolidated update, exactly as if it had used `asyncio.gather` -- no
reducer annotation needed on the frozen `ProductCutState.generated_shots`
field. `Send()` is genuinely exercised (verified by the graph-level test in
test_video_gen_node.py), it is just not the OUTER pipeline graph.

WIRED INTO graph/build.py -- this section previously said "NOT WIRED... YET"
even after the wiring commit landed (a stale claim, corrected here; the
file's own later "PHASE 4" section below already contradicted it, saying
"When this node was wired into graph/build.py..."). Real status:
`budget_gate -> video_gen -> ken_burns_fallback -> continuity_agent ->
continuity_gate`, with a conditional loop back to `video_gen` for retries
(Phase 4). Historical note, for context: this was originally left unwired as
a deliberate scope boundary (every prior phase's agents were built+merged
standalone first, then wired into graph/build.py as a separate, later
integration commit, matching this codebase's established rhythm) -- that
follow-up step has since happened.

RESOLVED -- native DashScope SDK region/base-URL configuration (previously a
KNOWN GAP, confirmed live during the BUILD_TASKS.md audit's de-risk pass).
docs/DERISK_VIDEO_GEN_RESULT.md §5 found that the native `dashscope` SDK (not
the OpenAI-compatible client every other agent in this codebase uses) needs
the *native* API base (`.../api/v1`, region-scoped to `dashscope-us` for this
account), a DIFFERENT path than `DASHSCOPE_BASE_URL` (.env.example's
documented OpenAI-*compatible* base for chat/vision). This module sets
`dashscope.api_key` from `DASHSCOPE_API_KEY` and, if present,
`dashscope.base_http_api_url` from an optional `DASHSCOPE_VIDEO_BASE_URL` env
var. CONFIRMED live: setting `DASHSCOPE_VIDEO_BASE_URL=https://dashscope-us.aliyuncs.com/api/v1`
(same host as `DASHSCOPE_BASE_URL`, native `/api/v1` path instead of
`/compatible-mode/v1`) against the real account produced two real,
ffprobe-verified Wan2.6-i2v-us clips (latency 41.7s/92.3s, both within the
originally documented 42-99s range). `backend/derisk/test_video_gen.py` (the
reproduction script this KNOWN GAP note said was never committed) now exists
and was used to produce this confirmation.

SCHEMA STATUS -- RESOLVED (Phase 3 KR/RR sync). Items 1-2 below were originally
flagged as self-invented, unconfirmed departures from the frozen schemas,
pending a sync with RR before the Ken-Burns Fallback Node got built against
them. That sync happened: RR formalized both into C1/C2/C3 (graph/state.py v6,
graph/shot_schema.py v3, graph/events.py v3) rather than changing the shape --
this module's runtime behavior is unchanged, the schemas simply now declare
what this module already produced. Item 3 remains a deliberate, permanent
design choice (not something pending resolution):
  1. `status = "fallback_requested"` is now a real value in
     `graph.state.Shot.status`'s Literal, `graph.shot_schema.ShotModel`'s
     `ShotStatus`, and `graph.events.ShotGeneratedPayload.status`'s Literal.
     Still deliberately distinct from the existing "fallback" value: per §5.9,
     "fallback" is what the Ken-Burns node (agents/ken_burns_fallback_node.py,
     now built and wired) sets once it has actually produced the pan/zoom
     clip -- "fallback_requested" means "handed off, Ken-Burns hasn't run yet."
  2. `failure_reason: {type, detail}` is now a declared, optional field on
     `graph.shot_schema.ShotModel` (`Optional[FailureReasonModel] = None`) and
     `graph.state.Shot` (`NotRequired[FailureReason]`) -- a shot carrying it no
     longer risks failing a future re-validation pass through that Pydantic
     model (`extra="forbid"` no longer applies to this field).
  3. `resolution_used` / `duration_sec_used` / `budget_clamped` are extra keys
     on the *GeneratedShot* dict beyond its frozen `{video_uri, drift_score,
     attempt}` shape (requirement 5's "record that clamp in the shot's
     state"). Chosen deliberately over extending Shot itself: unlike Shot,
     GeneratedShot has no Pydantic/`extra="forbid"` validator anywhere in this
     codebase, so this costs nothing today.
  4. Event dispatch (`shot_generated` per C2, graph/events.py) IS now wired up
     in the node wrapper (Phase 3 RR task 2): one `shot_generated` event per
     REAL clip (`is_fallback=False`); handed-off shots get their
     `is_fallback=True` event from the Ken-Burns node once it renders them. Real
     clips are also copied from the provider's ephemeral URL into OSS here (§5.8
     "clip persisted to OSS"). See `video_gen_node` below.

INTERFACE FOR KEN-BURNS FALLBACK NODE (not yet built -- RR). CONFIRMED, per
the Phase 3 KR/RR sync above -- this shape is now the real, formalized C1/C3
contract, not an assumption pending agreement:
  * Watch for `shot["status"] == "fallback_requested"` on any shot in
    `state["shot_list"]` this node returns.
  * That shot will have `shot["failure_reason"] = {"type": "timeout" |
    "api_error" | "budget_exceeded", "detail": "<human-readable string>"}`.
  * `shot["retry_count"]` is GUARANTEED untouched on this path -- this module
    never reads or writes that key anywhere (grep confirms: the only per-shot
    output keys this module ever returns are `status`, `failure_reason`, and
    the entries under `generated_shots`). The retry budget stays reserved for
    the Continuity Agent's quality-driven retries (§5.10), never consumed by
    an infrastructure failure, per §5.8's own explicit requirement.
  * No `generated_shots[shot_id]` entry is written for a handed-off shot --
    there is no video to reference yet. Ken-Burns should write that entry
    itself once its own pan/zoom clip exists.

PHASE 4 (Continuity retry loop -- two small, surgical production changes here).
When this node was wired into graph/build.py it ran exactly once per job. Phase 4
adds a Continuity retry cycle (agents/continuity_gate.py:
`... -> continuity_gate -> [loop back to video_gen if any shot is now "pending"]`),
which means this node can now run MORE than once, and the second pass must NOT
blindly regenerate every shot. Two minimal changes make it loop-safe:
  A. `_dispatch` now fans out `Send()` ONLY for shots whose `status == "pending"`
     (the value shots arrive with, and the value the Continuity Gate resets a
     retrying shot to). Every other shot ("passed"/"fallback"/"fallback_requested"
     /"review") is passed through unchanged by `_join`'s existing else-branch --
     the same "pass through non-matching shots untouched" posture
     ken_burns_fallback_node uses. Without this, a retry loop would re-hit the Wan
     API (real money) for shots that already passed.
  B. The node wrapper now MERGES its new GeneratedShot entries INTO the incoming
     `state["generated_shots"]` instead of returning only the newly-generated
     ones. `generated_shots` is an overwrite-semantics field; on a retry pass
     `generate_videos` only produces entries for the one(s) being regenerated, so
     returning just those would clobber every already-generated shot's entry.
     Merging preserves them (mirrors ken_burns_fallback_node). First-pass state
     has no entries, so this is a no-op there and existing tests are unaffected.
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
from typing import Annotated, Awaitable, Callable, Optional, TypedDict

import dashscope
from dashscope.aigc.video_synthesis import AioVideoSynthesis
from langchain_core.callbacks.manager import adispatch_custom_event
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from agents._oss import persist_remote_video_to_oss
from agents.budget_gate import RATE_720P, RATE_1080P
from agents.shot_list_agent import MIN_SHOT_DURATION_SEC
from graph.state import GeneratedShot, ProductCutState, ProductTruth, Shot, Treatment

logger = logging.getLogger("productcut.agents.video_gen_node")

# --- Failure-hand-off vocabulary (requirement 6; see INTERFACE section above) ---
FAILURE_TYPE_TIMEOUT = "timeout"
FAILURE_TYPE_API_ERROR = "api_error"
FAILURE_TYPE_BUDGET_EXCEEDED = "budget_exceeded"

# Deliberately NOT one of graph.state.Shot's frozen ShotStatus values -- see
# module docstring's "KNOWN DEPARTURES" #1.
FALLBACK_REQUESTED_STATUS = "fallback_requested"
# Already a real, frozen Shot.status value -- used on success.
SUCCESS_STATUS = "passed"
# The frozen "(re-)generate this shot" status -- the value shots arrive with from
# the Shot-List Agent AND the value the Continuity Gate resets a retrying shot to.
# Used by the fan-out filter (Phase 4, see module docstring's PHASE 4 note).
PENDING_STATUS = "pending"

# Typical observed latency was 42-99s per 5s clip (docs/DERISK_VIDEO_GEN_RESULT.md
# §2); this gives generous margin above that range. Env-overridable per the
# "flag genuine gaps, don't hardcode a guess forever" pattern used elsewhere
# (e.g. budget_gate.py's DEFAULT_JOB_BUDGET_CAP).
DEFAULT_WAIT_TIMEOUT_SEC = float(os.getenv("VIDEO_GEN_WAIT_TIMEOUT_SEC", "180"))

_QUALITY_BOILERPLATE = (
    "photorealistic, professional commercial cinematography, sharp focus, "
    "high detail, natural color, no artifacts"
)

# shot_type value naming the human-interaction composition (C3 v2 addition).
_HUMAN_INTERACTION_SHOT_TYPES = frozenset({"product_in_hand"})


class VideoGenTimeoutError(Exception):
    """The Wan task never reached a terminal status within the wait timeout."""


class VideoGenAPIError(Exception):
    """The Wan API returned a hard failure (non-200, error code, or a FAILED task)."""


# ---------------------------------------------------------------------------
# Budget clamp (requirement 5).
# ---------------------------------------------------------------------------
def _resolve_generation_params(shot: Shot) -> tuple[Optional[float], Optional[str], Optional[dict]]:
    """Returns (duration_sec, resolution, failure_reason).

    If `failure_reason` is not None, this shot cannot be generated at all
    within its allocated_budget even at the cheapest feasible duration/
    resolution -- the caller must skip the API call entirely (no point
    spending anything generating something that will be over budget).
    """
    allocated = shot["allocated_budget"]
    duration = shot["duration_sec"]
    cost_1080p = duration * RATE_1080P
    cost_720p = duration * RATE_720P

    if allocated >= cost_1080p:
        return duration, "1080P", None
    if allocated >= cost_720p:
        return duration, "720P", None  # resolution clamped, duration untouched

    feasible_duration = (allocated / RATE_720P) if RATE_720P > 0 else 0.0
    if feasible_duration < MIN_SHOT_DURATION_SEC:
        return (
            None,
            None,
            {
                "type": FAILURE_TYPE_BUDGET_EXCEEDED,
                "detail": (
                    f"allocated_budget ${allocated:.4f} cannot cover even the "
                    f"{MIN_SHOT_DURATION_SEC}s floor at 720p "
                    f"(${MIN_SHOT_DURATION_SEC * RATE_720P:.4f})"
                ),
            },
        )
    return round(feasible_duration, 3), "720P", None


# ---------------------------------------------------------------------------
# Reference photo resolution.
# ---------------------------------------------------------------------------
def _resolve_reference_image_url(reference_image_id: str, product_photos: list[str]) -> str:
    """Maps a shot's `reference_image_id` ("photo_1", "photo_2", ...) to the
    matching URL in `state["product_photos"]` -- the same 1-indexed "photo_N"
    convention `agents/shot_list_agent.py`'s own `_reference_image_id` helper
    and `ProductTruth.source` already use. Defaults to the first photo when
    the id is malformed/out of range rather than failing the shot over a
    naming edge case -- same defensive posture as that helper.
    """
    if not product_photos:
        return ""
    match = re.match(r"^photo_(\d+)$", reference_image_id or "")
    if match:
        idx = int(match.group(1)) - 1
        if 0 <= idx < len(product_photos):
            return product_photos[idx]
    return product_photos[0]


# ---------------------------------------------------------------------------
# Structured prompt (requirement 2; full mapping table in module docstring).
# ---------------------------------------------------------------------------
def _build_prompt(shot: Shot, product_truths: list[ProductTruth], treatment: Optional[Treatment]) -> str:
    truths_by_id = {t["truth_id"]: t for t in product_truths}
    truth = truths_by_id.get(shot["justification"]["truth_fact_id"])
    subject = f"Product detail: {truth['fact']}." if truth else "The product shown in the reference photo."

    action = shot["description"]
    camera = shot["camera_move"].replace("_", " ")
    lighting = shot["lighting"]

    composition = f"{shot['shot_type'].replace('_', ' ')}, {shot['framing'].replace('_', ' ')} framing"
    if shot["text_overlay_zone"] != "none":
        composition += (
            f", reserve the {shot['text_overlay_zone'].replace('_', ' ')} empty "
            "for a composited caption/CTA"
        )
    if shot["shot_type"] in _HUMAN_INTERACTION_SHOT_TYPES:
        # Positive continuity clause -- docs/DERISK_VIDEO_GEN_RESULT.md §6 found
        # this (not the negative prompt alone) is what actually prevents the
        # product vanishing during human-interaction shots.
        composition += (
            ". Natural hand interaction with five anatomically correct fingers "
            "per hand; product stays centered in frame, no scene cut"
        )

    mood = f"{treatment['director_persona']}; {treatment['pacing_philosophy']}" if treatment else "understated, product-forward"

    return (
        f"Subject: {subject}\n"
        f"Action/Motion: {action}\n"
        f"Camera: {camera}\n"
        f"Lighting: {lighting}\n"
        f"Composition: {composition}\n"
        f"Mood: {mood}\n"
        f"Quality: {_QUALITY_BOILERPLATE}"
    )


# ---------------------------------------------------------------------------
# Real Wan2.6-i2v-us call (native dashscope SDK -- see module docstring's
# KNOWN GAP on region/base-URL configuration).
# ---------------------------------------------------------------------------
async def _call_wan_video_gen(
    *,
    image_url: str,
    prompt: str,
    negative_prompt: str,
    duration_sec: float,
    resolution: str,
) -> str:
    """Submit + wait for one Wan2.6-i2v-us generation. Returns the video URL on
    success. Raises VideoGenTimeoutError / VideoGenAPIError on failure -- never
    retries (that policy lives in the caller / Continuity Agent later, not here).
    """
    dashscope.api_key = os.environ["DASHSCOPE_API_KEY"]
    video_base_url = os.getenv("DASHSCOPE_VIDEO_BASE_URL")
    if video_base_url:
        dashscope.base_http_api_url = video_base_url
    model = os.environ["MODEL_VIDEO"]

    try:
        response = await asyncio.wait_for(
            AioVideoSynthesis.call(
                model=model,
                prompt=prompt,
                negative_prompt=negative_prompt,
                img_url=image_url,
                duration=int(round(duration_sec)),
                resolution=resolution,
            ),
            timeout=DEFAULT_WAIT_TIMEOUT_SEC,
        )
    except asyncio.TimeoutError as exc:
        raise VideoGenTimeoutError(
            f"Wan generation exceeded the {DEFAULT_WAIT_TIMEOUT_SEC:.0f}s wait timeout"
        ) from exc
    except (VideoGenTimeoutError, VideoGenAPIError):
        raise
    except Exception as exc:  # noqa: BLE001 -- any transport/SDK error is a classified api_error, never crashes the fan-out
        raise VideoGenAPIError(str(exc)) from exc

    task_status = response.output.task_status if response.output else None
    if response.status_code != 200 or task_status != "SUCCEEDED":
        code = getattr(response, "code", "") or ""
        message = getattr(response, "message", "") or task_status or "no output"
        raise VideoGenAPIError(f"Wan task did not succeed (code={code!r}, status/message={message!r})")

    return response.output.video_url


GenerateFn = Callable[..., Awaitable[str]]


# ---------------------------------------------------------------------------
# Internal, self-contained Send()-based fan-out (requirement 1). Private to
# this module -- see module docstring's "PARALLEL FAN-OUT" section for why
# this needs no change to the frozen ProductCutState.
# ---------------------------------------------------------------------------
def _merge_dicts(a: dict, b: dict) -> dict:
    return {**a, **b}


def _concat_lists(a: list, b: list) -> list:
    return [*a, *b]


class _VideoGenGraphState(TypedDict, total=False):
    shots: list[Shot]
    product_truths: list[ProductTruth]
    treatment: Optional[Treatment]
    product_photos: list[str]
    generated_shots: Annotated[dict[str, GeneratedShot], _merge_dicts]
    failures: Annotated[list[dict], _concat_lists]


async def generate_videos(
    shots: list[Shot],
    product_truths: list[ProductTruth],
    treatment: Optional[Treatment],
    product_photos: list[str],
    generate_fn: Optional[GenerateFn] = None,
) -> tuple[list[Shot], dict[str, GeneratedShot]]:
    """Run the Video-Gen Node: one independent Wan call per shot, fanned out in
    parallel via LangGraph `Send()`, never sequential and never retried here.

    Returns (updated_shots, generated_shots) -- updated_shots is the full input
    list with each shot's `status` (and `failure_reason` on a hand-off)
    patched in; generated_shots contains one entry per shot that actually
    produced a clip (never one for a handed-off shot -- see INTERFACE section).

    `generate_fn` defaults to the real `_call_wan_video_gen` but stays
    injectable for tests, the same `client=None` / `validate_justifications=...`
    injection pattern every other agent module in this codebase already uses.
    """
    fn: GenerateFn = generate_fn or _call_wan_video_gen

    async def _single_shot_node(payload: dict) -> dict:
        shot: Shot = payload["shot"]
        shot_id = shot["shot_id"]

        duration, resolution, budget_failure = _resolve_generation_params(shot)
        if budget_failure is not None:
            logger.warning(
                "Video-Gen Node: shot %s cannot be generated within budget (%s) -- "
                "handing off, no API call made, retry_count left untouched.",
                shot_id, budget_failure["detail"],
            )
            return {"failures": [{"shot_id": shot_id, "failure_reason": budget_failure}]}

        image_url = _resolve_reference_image_url(shot["reference_image_id"], payload.get("product_photos", []))
        prompt = _build_prompt(shot, payload.get("product_truths", []), payload.get("treatment"))

        try:
            video_url = await fn(
                image_url=image_url,
                prompt=prompt,
                negative_prompt=shot["negative_prompt"],
                duration_sec=duration,
                resolution=resolution,
            )
        except VideoGenTimeoutError as exc:
            logger.warning("Video-Gen Node: shot %s timed out -- handing off, retry_count untouched: %s", shot_id, exc)
            return {"failures": [{"shot_id": shot_id, "failure_reason": {"type": FAILURE_TYPE_TIMEOUT, "detail": str(exc)}}]}
        except VideoGenAPIError as exc:
            logger.warning("Video-Gen Node: shot %s API error -- handing off, retry_count untouched: %s", shot_id, exc)
            return {"failures": [{"shot_id": shot_id, "failure_reason": {"type": FAILURE_TYPE_API_ERROR, "detail": str(exc)}}]}
        except Exception as exc:  # noqa: BLE001 -- any other unexpected failure still hands off cleanly, never crashes the fan-out
            logger.error("Video-Gen Node: shot %s unexpected failure -- handing off, retry_count untouched: %s", shot_id, exc)
            return {"failures": [{"shot_id": shot_id, "failure_reason": {"type": FAILURE_TYPE_API_ERROR, "detail": str(exc)}}]}

        clamped = resolution != "1080P" or abs(duration - shot["duration_sec"]) > 1e-6
        generated: dict = {
            "video_uri": video_url,
            "attempt": 1,
            # Extra, non-C1 fields -- see module docstring's "KNOWN DEPARTURES" #3.
            "resolution_used": resolution,
            "duration_sec_used": duration,
            "budget_clamped": clamped,
        }
        return {"generated_shots": {shot_id: generated}}

    def _dispatch(state: _VideoGenGraphState) -> list[Send]:
        # PHASE 4 retry-loop filter (see module docstring's PHASE 4 note): only
        # fan out Send() for shots that still need (re-)generation -- status
        # "pending". Every other shot ("passed"/"fallback"/"fallback_requested"/
        # "review") is left for `_join` to pass through untouched, so a Continuity
        # retry loop regenerates ONLY the shot(s) the Gate reset to "pending"
        # instead of blindly re-hitting Wan for every shot on every pass.
        return [
            Send(
                "single_shot",
                {
                    "shot": shot,
                    "product_truths": state.get("product_truths", []),
                    "treatment": state.get("treatment"),
                    "product_photos": state.get("product_photos", []),
                },
            )
            for shot in state.get("shots", [])
            if shot.get("status") == PENDING_STATUS
        ]

    def _join(state: _VideoGenGraphState) -> dict:
        generated = state.get("generated_shots", {})
        failures_by_id = {f["shot_id"]: f["failure_reason"] for f in state.get("failures", [])}
        updated: list[Shot] = []
        for shot in state.get("shots", []):
            shot_id = shot["shot_id"]
            if shot_id in generated:
                updated.append({**shot, "status": SUCCESS_STATUS})
            elif shot_id in failures_by_id:
                updated.append(
                    {**shot, "status": FALLBACK_REQUESTED_STATUS, "failure_reason": failures_by_id[shot_id]}
                )
            else:
                updated.append(shot)  # defensive: shouldn't happen, every shot dispatches exactly one branch
        return {"shots": updated}

    builder = StateGraph(_VideoGenGraphState)
    builder.add_node("single_shot", _single_shot_node)
    builder.add_node("join", _join)
    builder.add_conditional_edges(START, _dispatch, ["single_shot"])
    builder.add_edge("single_shot", "join")
    builder.add_edge("join", END)
    compiled = builder.compile()

    result = await compiled.ainvoke(
        {
            "shots": shots,
            "product_truths": product_truths,
            "treatment": treatment,
            "product_photos": product_photos,
        }
    )
    return result.get("shots", []), result.get("generated_shots", {})


async def _persist_generated_to_oss(
    generated: dict[str, GeneratedShot], job_id: str
) -> int:
    """Copy each real clip from its ephemeral provider URL into the job's OSS
    namespace, rewriting `video_uri` in place. Returns the count persisted.

    Best-effort per shot: a persistence failure logs and KEEPS the still-valid
    (if short-lived) provider URL rather than sinking the shot -- the clip was
    generated successfully, so a copy failure must not downgrade it to a
    fallback. Runs the blocking download+upload off the event loop and all
    shots concurrently (matching this module's async-everywhere posture).
    """
    async def _persist_one(shot_id: str, entry: GeneratedShot) -> bool:
        try:
            oss_uri = await asyncio.to_thread(
                persist_remote_video_to_oss, entry["video_uri"], job_id, shot_id
            )
            entry["video_uri"] = oss_uri
            return True
        except Exception as exc:  # noqa: BLE001 -- a copy failure never sinks a real clip
            logger.warning(
                "Video-Gen Node: OSS persist failed for shot %s (%s) -- keeping the "
                "provider URL (clip still valid, just not re-homed in OSS).",
                shot_id, exc,
            )
            return False

    results = await asyncio.gather(
        *(_persist_one(sid, entry) for sid, entry in generated.items())
    )
    return sum(results)


async def video_gen_node(
    state: ProductCutState,
    config: Optional[RunnableConfig] = None,
) -> dict:
    """LangGraph node wrapper: reads shot_list/product_truths/treatment/
    product_photos from state, generates every shot, persists each real clip to
    OSS, records per-shot status, and emits a C2 `shot_generated` event per real
    clip (Phase 3 RR task -- §5.8 output contract + C2 realtime).

    Event semantics (real vs. fallback): this node emits `shot_generated` with
    `is_fallback=False` for every shot that produced a REAL clip. Handed-off
    shots (`fallback_requested`) emit nothing here -- they have no clip yet;
    the Ken-Burns Fallback Node (§5.9) emits their `is_fallback=True` event once
    it renders them. So exactly one `shot_generated` fires per shot that ends up
    with a clip, correctly labelled.

    Dispatches via `adispatch_custom_event` (surfaces in `astream_events` as
    `on_custom_event`, which app/main.py unwraps into a C2 envelope), mirroring
    budget_gate_node's precedent. `config` defaults to None so the node stays
    directly callable/testable; LangGraph injects the real RunnableConfig.
    """
    job_id = state.get("job_id", "unknown_job")
    shots, generated = await generate_videos(
        shots=state.get("shot_list", []),
        product_truths=state.get("product_truths", []),
        treatment=state.get("treatment"),
        product_photos=state.get("product_photos", []),
    )

    n_persisted = await _persist_generated_to_oss(generated, job_id)

    # One shot_generated event per real clip, in shot-list order (deterministic).
    for shot in shots:
        shot_id = shot["shot_id"]
        if shot_id in generated:
            await adispatch_custom_event(
                "shot_generated",
                {
                    "shot_id": shot_id,
                    "generated": generated[shot_id],
                    "status": SUCCESS_STATUS,
                    "is_fallback": False,
                },
                config=config,
            )

    n_handed_off = sum(1 for s in shots if s.get("status") == FALLBACK_REQUESTED_STATUS)
    trace_note = (
        f"\n[video_gen] generated {len(generated)}/{len(shots)} shot(s); "
        f"persisted {n_persisted} to OSS."
    )
    if n_handed_off:
        trace_note += f" {n_handed_off} shot(s) handed off for Ken-Burns fallback."
    return {
        "shot_list": shots,
        # PHASE 4: merge INTO existing generated_shots rather than replacing it.
        # With the retry-loop filter above, a re-generation pass only produces
        # entries for the shot(s) being retried; `generated_shots` is an
        # overwrite-semantics field, so returning just those would wipe every
        # already-generated shot's entry. Merging preserves them (same posture as
        # ken_burns_fallback_node). On the first pass state has none, so this is a
        # no-op there -- existing tests are unaffected.
        "generated_shots": {**state.get("generated_shots", {}), **generated},
        "reasoning_trace": state.get("reasoning_trace", "") + trace_note,
    }
