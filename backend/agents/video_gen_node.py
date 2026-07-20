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
  * Cast          <- treatment.character_anchor VERBATIM (video-gen-fidelity
                    story-arc fix, graph/state.py Treatment v10) -- ONLY on
                    human-interaction shots (shot_type product_in_hand /
                    worn_in_use), ONLY when the treatment actually carries one
                    (a script implying no person never gets a fabricated Cast
                    section). Placed immediately after Subject, before
                    Action/Motion -- never paraphrased, never trimmed by the
                    char-budget cutter below (same never-cut status as
                    _IDENTITY_PROTECTION_CLAUSE). Funded without raising the
                    overall PROMPT_CHAR_BUDGET: Quality is dropped and Mood is
                    compressed UNCONDITIONALLY on every human-interaction
                    shot (not just as the overflow fallback) -- see
                    `_build_prompt`'s own comments for the exact mechanism.
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
                    hero_reframe/cta_endcard/product_in_hand/worn_in_use --
                    product_in_hand is the v2 C3 addition, worn_in_use the v4
                    C3 addition) + shot.framing + a text_overlay_zone
                    reservation note. When shot_type is "product_in_hand" or
                    "worn_in_use" (the human-interaction compositions), an
                    explicit POSITIVE continuity clause is appended ("natural
                    hand interaction with five anatomically correct fingers...
                    product stays centered, no scene cut", extended with a
                    scale-lock and an occlusion-continuity clause) -- per
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
import mimetypes
import os
import re
import tempfile
from typing import Annotated, Awaitable, Callable, Optional, TypedDict

import httpx

import dashscope
from dashscope.aigc.video_synthesis import AioVideoSynthesis
from langchain_core.callbacks.manager import adispatch_custom_event
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from agents._oss import SIGNED_URL_TTL_SEC, persist_remote_video_to_oss
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
    "high detail, natural color, no artifacts, "
    "constant exposure throughout, stable studio lighting, no light flicker"
)

# video-gen-fidelity PHASE 1 fix. Root cause (confirmed against a real failed
# job, derisk/outputs/full_pipeline_live_result.json + our own logged warning
# "shot s4 prompt is 2388 chars, approaching Wan's 1,500-char server-side
# truncation limit"): the old code only WARNED above 1,400 chars and then let
# Wan's server silently truncate the tail -- which was routinely
# Mood/Quality/part of Composition. A silent server-side truncation gives zero
# visibility into what was lost. This module now enforces the budget itself,
# cutting deliberately (and logging exactly what got cut) instead of leaving
# that to an opaque server-side cutoff.
PROMPT_CHAR_BUDGET = 2200

# Compressed identity-protection clause -- the SAME four protections the old
# multi-clause version had (scale-lock, occlusion-continuity, anatomy,
# anti-cut), compressed to ~30 words. The old version cost ~330 chars by
# itself; per-shot it was ALSO being duplicated by the Shot-List Agent's Call B
# writing near-identical language into shot.description (see
# agents/shot_list_agent.py's Call B system prompt fix, same branch) -- so a
# human-interaction shot was paying for this protection twice. This one
# instance, in Composition, is now the only copy.
_IDENTITY_PROTECTION_CLAUSE = (
    "Throughout: the product keeps its exact shape, size, color and "
    "material, stays fully recognizable when partially covered and stays "
    "in frame; hands have five natural fingers; no scene cut."
)

# Counters i2v models' documented bias toward under-motion early in a clip
# (the rendered motion "leaks" from the static reference image and takes time
# to depart from it -- NeurIPS 2024 "Conditional Image Leakage in I2V
# Diffusion," arXiv:2406.15735) plus mid-motion endings -- appended to every
# human-interaction shot's Action/Motion text.
_ACTION_URGENCY_CLAUSE = " The action begins immediately and completes before the clip ends."

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _trim_description_tail(description: str, max_len: int) -> str:
    """video-gen-fidelity PHASE 4 fix, last-resort tier: drop trailing
    sentences from `description` (the shot's own Action/Motion content, e.g.
    a hero shot's legitimately-longer 120-180 word arc) until it fits within
    `max_len` -- always keeps at least the first sentence, never returns an
    empty string. Only reached when Quality/Mood/Lighting cuts and the
    redundant-Subject-clause drop (is_human_shot) still weren't enough --
    confirmed a real, live-occurring gap: a hero shot's Cast + full-length
    Action/Motion + Composition's identity clause can exceed even Wan's
    1,500-char hard truncation ceiling on their own (derisk/outputs/
    full_pipeline_live_vikr_postfix.log: shot s3, 1999 chars pre-trim).
    Trimming HERE (Action/Motion sits before Composition/Mood in render
    order) is strictly better than leaving it to Wan's own silent truncation,
    which would otherwise cut into -- or entirely drop -- the never-cut
    Composition identity-protection clause instead.
    """
    sentences = _SENTENCE_SPLIT_RE.split(description.strip())
    if len(sentences) <= 1:
        return description  # nothing to drop -- return verbatim, not just .strip()'d
    kept = sentences[0]
    for sentence in sentences[1:]:
        candidate = f"{kept} {sentence}"
        if len(candidate) > max_len:
            break
        kept = candidate
    return kept

# The mirror-image fix for the CTA close: here we WANT the motion to resolve
# into stillness by the end, not keep moving into an abrupt trim (Phase 3
# fixes the trim-side half of that same failure in assembly_agent.py).
_CTA_STILLNESS_CLAUSE = (
    " All motion resolves in the final second: the product comes to rest "
    "perfectly still on a clean, centered composition and holds, as if "
    "posing for an end card."
)

# Alibaba's own Wan prompt guide uses "fixed camera" as its term for camera
# stillness, and separately warns that a weak/empty prompt produces "a static
# camera in an undefined void" -- i.e. this model family plausibly conflates
# SCENE stillness with CAMERA stillness. The old phrasing ("static") was
# ambiguous between the two; this is explicit that only the camera is meant to
# hold still.
_FIXED_CAMERA_PHRASING = (
    "Fixed camera on a tripod; the camera does not move. The subject moves "
    "naturally and completes the described action fully within the clip."
)

# Rich motion descriptions for every named camera_move. Bare two-word names
# ("push in", "orbit") give Wan almost no signal about magnitude or intent,
# causing it to default to its safest training-data pattern: a barely-perceptible
# zoom. Each description names: what moves, in which direction, by how much, and
# that the motion must be visible and complete within the clip duration.
_CAMERA_MOVE_DESCRIPTIONS: dict[str, str] = {
    "push_in": (
        "Push in: the camera advances steadily and noticeably toward the subject, "
        "closing roughly a third of the original distance by the clip's end. "
        "The motion is smooth and continuous — a real travel, not a subtle drift."
    ),
    "pull_back": (
        "Pull back: the camera retreats steadily from a tight frame to a wider view, "
        "visibly revealing more of the environment as it moves. "
        "The subject shrinks in frame measurably from start to end."
    ),
    "orbit": (
        "Orbital arc: the camera sweeps in a smooth lateral arc around the subject, "
        "shifting the angle of view by 20–35 degrees. The subject stays centered "
        "while the background perspective visibly rotates behind it."
    ),
    "pan": (
        "Pan: the camera pivots horizontally in a steady sweep, traversing the subject "
        "or environment from one side to the other. The frame moves decisively — "
        "a half-second of movement, not a nearly-imperceptible drift."
    ),
    "tilt_up": (
        "Tilt up: the camera rotates smoothly upward from base to apex, "
        "revealing the subject's full vertical extent as it rises. "
        "The motion begins at the bottom of the subject and completes at the top."
    ),
    "rack_focus": (
        "Rack focus: the focal plane shifts mid-clip — one element blurs as another "
        "sharpens. The transition is visible and deliberate: begin with foreground "
        "in sharp focus, then smoothly rack to background (or reverse), with both "
        "states held for at least one second each."
    ),
}

# Appended to Action/Motion for every product-alone shot (non-human, non-static)
# to counter Wan's documented bias toward under-motion (arXiv:2406.15735 —
# the same static-start bias that drives _ACTION_URGENCY_CLAUSE on human shots).
# Without an explicit urgency mandate, product shots tend to produce a barely-
# perceptible zoom on the reference image rather than the specified camera move.
_PRODUCT_MOTION_URGENCY_CLAUSE = (
    " The camera movement is deliberate and clearly visible throughout — "
    "not a subtle drift or barely-perceptible zoom. Motion begins within "
    "the first second and continues noticeably to the clip's end."
)

# v8 fix: DashScope's `prompt_extend` silently defaults to `true` when omitted,
# letting an internal, opaque LLM prompt-rewriter re-describe the reference
# image in text before generation -- a plausible amplifier of the Meta Quest ->
# "phone on a stand" bug (an image-aware rewriter can mis-describe an ambiguous
# photo, and we'd never see what it wrote). Now that the prompt carries a real
# form_factor subject anchor (_build_prompt above), the prompt is self-
# sufficient, so the default flips to OFF. Env-overridable per this codebase's
# "flag, don't hardcode forever" pattern (budget_gate.py's DEFAULT_JOB_BUDGET_CAP).
DEFAULT_PROMPT_EXTEND = os.getenv("VIDEO_GEN_PROMPT_EXTEND", "false").lower() in ("1", "true")

# shot_type values naming the human-interaction composition (C3 v2 addition of
# "product_in_hand", C3 v4 addition of "worn_in_use" -- the wider, person-in-
# motion composition). Hand-kept in sync with agents/shot_list_agent.py's own
# `HUMAN_INTERACTION_SHOT_TYPES` (that module's docstring notes the same).
_HUMAN_INTERACTION_SHOT_TYPES = frozenset({"product_in_hand", "worn_in_use"})


def _use_intl_video() -> bool:
    """True when all three Singapore Wan 2.7 env vars are present -- activates
    INTL mode for API credentials, model, and OSS bucket selection."""
    return bool(
        os.getenv("DASHSCOPE_VIDEO_INTL_API_KEY")
        and os.getenv("DASHSCOPE_INTL_VIDEO_BASE_URL")
        and os.getenv("MODEL_VIDEO_INTL")
    )


def _reupload_photos_to_intl_oss(photo_urls: list[str], job_id: str) -> list[str]:
    """Download each reference photo and re-upload to the Singapore OSS bucket
    so Wan 2.7 (Singapore endpoint) can fetch them without cross-region failure.

    The output clips are still persisted to the primary US bucket as normal --
    only the *input* reference images need to live in the SG bucket.
    Returns signed Singapore URLs in the same order as the input list.
    """
    import httpx
    import oss2

    auth = oss2.Auth(os.environ["OSS_ACCESS_KEY_ID"], os.environ["OSS_ACCESS_KEY_SECRET"])
    bucket = oss2.Bucket(auth, os.environ["OSS_INTL_ENDPOINT"], os.environ["OSS_INTL_BUCKET"])

    result: list[str] = []
    for idx, url in enumerate(photo_urls):
        key = f"jobs/{job_id}/photo_{idx + 1}.jpg"
        content_type = mimetypes.guess_type(url.split("?")[0])[0] or "image/jpeg"
        resp = httpx.get(url, timeout=60.0, follow_redirects=True)
        resp.raise_for_status()
        fd, local_path = tempfile.mkstemp(suffix=".jpg", prefix="intl_photo_")
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(resp.content)
            bucket.put_object_from_file(key, local_path, headers={"Content-Type": content_type})
        finally:
            if os.path.exists(local_path):
                os.remove(local_path)
        # Force HTTPS -- Wan 2.7 rejects HTTP image URLs
        signed_url = bucket.sign_url("GET", key, SIGNED_URL_TTL_SEC, slash_safe=True)
        result.append(signed_url.replace("http://", "https://", 1))
        logger.info("Video-Gen Node: re-uploaded photo_%d to SG OSS bucket -> %s", idx + 1, key)
    return result



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
    is_human_shot = shot["shot_type"] in _HUMAN_INTERACTION_SHOT_TYPES

    # v8 fix (Meta Quest -> "phone on a stand" wrong-object bug): lead the
    # Subject line with the holistic whole-object form_factor anchor fact
    # (agents/product_truth_extractor.py's FORM-FACTOR ANCHOR), BEFORE the
    # per-shot micro-fact. Leading tokens carry the most weight in this
    # module's own documented prompt-construction posture (see the
    # human-interaction positive-clause precedent above), and a single
    # isolated micro-fact alone under-specifies the subject as a whole object
    # -- exactly what let the i2v model collapse toward a common
    # training-data composition instead of the actual product shape.
    form = next((t for t in product_truths if t["category"] == "form_factor"), None)
    anchor = f"The product: {form['fact']} " if form else ""
    if is_human_shot:
        # video-gen-fidelity PHASE 4 fix: on human-interaction shots the
        # per-shot micro-fact is already required verbatim in Action/Motion by
        # the Shot-List Agent's HUMAN-INTERACTION SHOTS rule -- restating it in
        # Subject too is redundant and was confirmed as the single largest
        # single section pushing real shots past Wan's 1,500-char hard
        # truncation ceiling. Drop it: use only the form_factor anchor (the
        # object-identity content that actually helps), or fall back to the
        # generic reference-photo line if no form_factor truth is present.
        subject = anchor.strip() or "The product shown in the reference photo."
    else:
        subject = (
            f"{anchor}Product detail: {truth['fact']}." if truth
            else (anchor.strip() or "The product shown in the reference photo.")
        )

    # Cast section (video-gen-fidelity story-arc fix). Text-only i2v prompting
    # cannot lock FACIAL identity across independent Wan generations, but it
    # CAN reliably hold wardrobe color / hair / a named setting when pinned
    # once (Treatment.character_anchor, agents/treatment_agent.py v10) and
    # reused VERBATIM here -- never paraphrased, never trimmed by the
    # char-budget cutter below (same never-cut status as
    # _IDENTITY_PROTECTION_CLAUSE, for the identical reason: this is the one
    # lever that keeps a human-interaction shot visually consistent with every
    # OTHER independently-generated human-interaction shot in the same ad).
    # Applies to every human-interaction shot, hero or faceless -- a faceless
    # shot still benefits from matching wardrobe/setting/palette, per the
    # research synthesis's point that Cast text "carries perceived identity
    # through clothing/palette" even without a face in frame. Absent (no
    # section at all) when the script implies no person -- never fabricated.
    cast_line = ""
    if is_human_shot and treatment:
        cast_line = (treatment.get("character_anchor") or "").strip()

    # `_action_suffix` tracked separately from the base description (not just
    # concatenated straight into `action`) so the video-gen-fidelity PHASE 4
    # last-resort trim below can shorten the shot's own (possibly
    # hero-length) description tail without ever eating into this short,
    # never-cut motion-completion/stillness clause.
    description = shot["description"]

    # Script-sync fix: prepend the narration line as the primary action
    # directive. Wan weights the first tokens of each section most heavily —
    # leading with "Narrator says: X. Show this literally." forces the motion
    # to match what is being said rather than drifting to generic product
    # footage. Placed at the FRONT of description (not appended) so it
    # survives the last-resort tail-trim and is never silently dropped.
    # script_quote is always present (ShotJustification contract).
    script_quote = (shot.get("justification") or {}).get("script_quote", "").strip()
    if script_quote:
        description = f'Narrator says: "{script_quote}". Show this action literally. {description}'
    _action_suffix = ""
    if is_human_shot:
        # Counters both the i2v static-start bias and a clip that's still
        # mid-motion when it cuts off (see _ACTION_URGENCY_CLAUSE's own comment).
        _action_suffix += _ACTION_URGENCY_CLAUSE
    elif shot["shot_type"] != "cta_endcard" and shot["camera_move"] != "static":
        # Same static-start bias affects product-alone shots: Wan defaults to a
        # barely-perceptible zoom when the camera move instruction is vague.
        # Append the product urgency clause for every moving, non-human shot.
        _action_suffix += _PRODUCT_MOTION_URGENCY_CLAUSE
    if shot["shot_type"] == "cta_endcard":
        # Opposite problem, same mechanism -- here the clip should visibly
        # SETTLE by the end rather than just stop (see _CTA_STILLNESS_CLAUSE).
        _action_suffix += _CTA_STILLNESS_CLAUSE
    action = description + _action_suffix

    move = shot["camera_move"]
    if move == "static":
        camera = _FIXED_CAMERA_PHRASING
    else:
        camera = _CAMERA_MOVE_DESCRIPTIONS.get(move, move.replace("_", " "))

    lighting = shot["lighting"]

    composition = f"{shot['shot_type'].replace('_', ' ')}, {shot['framing'].replace('_', ' ')} framing"
    if shot["text_overlay_zone"] != "none":
        composition += (
            f", reserve the {shot['text_overlay_zone'].replace('_', ' ')} empty "
            "for a composited caption/CTA"
        )
    if is_human_shot:
        # Positive continuity clause -- docs/DERISK_VIDEO_GEN_RESULT.md §6 found
        # this (not the negative prompt alone) is what actually prevents the
        # product vanishing during human-interaction shots. Compressed (PHASE 1
        # fix) to the single _IDENTITY_PROTECTION_CLAUSE -- same four
        # protections, far fewer characters; deliberately placed in Composition
        # (added before the budget-cut pass below, and Composition is never one
        # of the sections that pass cuts) so it survives even if Quality/Mood/
        # Lighting all get cut.
        composition += f". {_IDENTITY_PROTECTION_CLAUSE}"

    mood_full = (
        f"{treatment['director_persona']}; {treatment['pacing_philosophy']}"
        if treatment else "understated, product-forward"
    )

    sections: list[list[str]] = [["Subject", subject]]
    if cast_line:
        # Immediately after Subject, before Action/Motion -- high in the
        # prompt per Wan's documented front-token-weighting, but after the
        # product anchor (product identity is the harder, already-solved
        # constraint and must keep the leading position).
        sections.append(["Cast", cast_line])
    sections += [
        ["Action/Motion", action],
        ["Camera", camera],
        ["Lighting", lighting],
        ["Composition", composition],
        ["Mood", mood_full],
        ["Quality", _QUALITY_BOILERPLATE],
    ]

    def _render(secs: list[list[str]]) -> str:
        return "\n".join(f"{name}: {value}" for name, value in secs)

    dropped: list[str] = []
    quality_dropped = False
    mood_compressed = False

    def _compress_mood() -> None:
        # Compress Mood to a single short clause -- not the full
        # director_persona + pacing_philosophy dump. First sentence/clause of
        # director_persona alone, further capped to 8 words: a persona with no
        # early "." or ";" (free-form prose) must still shrink meaningfully,
        # not just drop the (already-absent) pacing_philosophy half.
        persona = (treatment or {}).get("director_persona", "") if treatment else ""
        short_mood = persona.split(".")[0].split(";")[0].strip()
        short_mood = " ".join(short_mood.split()[:8]) or "understated, product-forward"
        for s in sections:
            if s[0] == "Mood":
                s[1] = short_mood

    # Fund the new Cast section WITHOUT raising PROMPT_CHAR_BUDGET
    # (video-gen-fidelity story-arc fix): on every human-interaction shot,
    # Quality is dropped and Mood is compressed UNCONDITIONALLY here, not
    # merely as the overflow fallback below. Both are already usually the
    # first things the overflow path cuts on a human shot's typically-longer
    # prompt anyway (Quality is pure boilerplate; Mood is the whole ad's
    # generic directorial voice, not this shot's specific content) -- making
    # that deterministic rather than incidental is what pays for Cast.
    if is_human_shot:
        if any(s[0] == "Quality" for s in sections):
            sections = [s for s in sections if s[0] != "Quality"]
            quality_dropped = True
        _compress_mood()
        mood_compressed = True
        logger.info(
            "Video-Gen Node: shot %s is a human-interaction shot -- Quality "
            "dropped and Mood compressed unconditionally to fund the Cast "
            "section (not an overflow cut; see module docstring).",
            shot["shot_id"],
        )

    prompt = _render(sections)

    # Hard budget enforcement (PHASE 1 fix) -- Subject/Cast/Action/Camera/
    # Composition are never touched here (they carry the shot's actual
    # grounded content); only the three genuinely-generic-across-every-shot
    # sections are cut, in this order, stopping as soon as it fits. Quality/
    # Mood are skipped here if the human-interaction funding above already
    # handled them -- `quality_dropped`/`mood_compressed` prevent this from
    # double-logging (or, for Quality, a harmless but misleading no-op filter)
    # the same cut twice under two different reasons.
    if len(prompt) > PROMPT_CHAR_BUDGET and not quality_dropped:
        sections = [s for s in sections if s[0] != "Quality"]
        dropped.append("Quality")
        prompt = _render(sections)

    if len(prompt) > PROMPT_CHAR_BUDGET and not mood_compressed:
        _compress_mood()
        dropped.append("Mood (compressed)")
        prompt = _render(sections)

    if len(prompt) > PROMPT_CHAR_BUDGET:
        # Trim Lighting to its first clause -- still names the light quality,
        # drops the rest of the shared, per-job-identical detail.
        trimmed_lighting = lighting.split(",")[0].split(".")[0].strip()
        for s in sections:
            if s[0] == "Lighting":
                s[1] = trimmed_lighting
        dropped.append("Lighting (trimmed)")
        prompt = _render(sections)

    if len(prompt) > PROMPT_CHAR_BUDGET:
        # video-gen-fidelity PHASE 4, last-resort tier: every genuinely-generic
        # section is already gone/compressed, and this shot's own description
        # is still too long to fit -- confirmed live-occurring for a hero shot
        # (120-180 word allowance by design) and for a long cited-fact Subject
        # line alike (see _trim_description_tail's docstring). Trim
        # Action/Motion's OWN tail (never the short, never-cut urgency/
        # stillness suffix) rather than leave the overflow to Wan's silent
        # server-side truncation, which would otherwise cut into whatever
        # comes after Action/Motion in render order -- including the
        # never-cut Composition identity-protection clause.
        overage = len(prompt) - PROMPT_CHAR_BUDGET
        target_len = max(len(description) - overage, 1)
        trimmed_description = _trim_description_tail(description, target_len)
        if len(trimmed_description) < len(description):
            for s in sections:
                if s[0] == "Action/Motion":
                    s[1] = trimmed_description + _action_suffix
            dropped.append("Action/Motion (trimmed tail)")
            prompt = _render(sections)

    if dropped:
        logger.warning(
            "Video-Gen Node: shot %s prompt exceeded the 2200-char budget -- cut "
            "deliberately (never left to Wan's server-side truncation): %s. "
            "Final length %d chars.",
            shot["shot_id"], ", ".join(dropped), len(prompt),
        )
    if len(prompt) > PROMPT_CHAR_BUDGET:
        # Even after every cuttable section (including Action/Motion's own
        # tail above) is gone, Subject/Cast/Camera/Composition alone are still
        # too long -- these are never cut (they carry the shot's non-
        # negotiable identity-fix content) so flag it: the tail may still be
        # silently dropped by Wan's server-side 1,500-char truncation, and
        # this is the one case we can't prevent that ourselves.
        logger.warning(
            "Video-Gen Node: shot %s prompt is still %d chars after all budget "
            "cuts -- approaching Wan's 1,500-char server-side truncation limit.",
            shot["shot_id"], len(prompt),
        )
    return prompt


# ---------------------------------------------------------------------------
# Wan 2.7 (Singapore INTL) raw REST caller.
# Wan 2.7 uses a different input schema than Wan 2.6:
#   input.media = [{"type": "first_frame", "url": <image_url>}]
# instead of the SDK's img_url= kwarg which maps to input.img_url (Wan 2.6).
# The dashscope Python SDK does not yet support this field, so we call the
# REST API directly with httpx and do our own polling loop.
# ---------------------------------------------------------------------------
async def _call_wan_video_gen_intl(
    *,
    image_url: str,
    prompt: str,
    negative_prompt: str,
    duration_sec: float,
    resolution: str,
    seed: Optional[int] = None,
) -> str:
    api_key = os.environ["DASHSCOPE_VIDEO_INTL_API_KEY"]
    base_url = os.environ["DASHSCOPE_INTL_VIDEO_BASE_URL"]
    model = os.environ["MODEL_VIDEO_INTL"]

    logger.info("Video-Gen INTL: submitting model=%s to %s", model, base_url)

    body: dict = {
        "model": model,
        "input": {
            "prompt": prompt,
            "media": [{"type": "first_frame", "url": image_url}],
        },
        "parameters": {
            "duration": int(round(duration_sec)),
            "resolution": resolution,
        },
    }
    if negative_prompt:
        body["input"]["negative_prompt"] = negative_prompt
    if seed is not None:
        body["parameters"]["seed"] = seed

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "X-DashScope-Async": "enable",
    }

    try:
        async with httpx.AsyncClient() as client:
            resp = await asyncio.wait_for(
                client.post(
                    f"{base_url}/services/aigc/video-generation/video-synthesis",
                    headers=headers,
                    json=body,
                    timeout=30,
                ),
                timeout=DEFAULT_WAIT_TIMEOUT_SEC,
            )
            if resp.status_code != 200:
                raise VideoGenAPIError(f"Wan 2.7 submit failed: HTTP {resp.status_code}: {resp.text[:200]}")

            task_id = resp.json()["output"]["task_id"]
            deadline = asyncio.get_event_loop().time() + DEFAULT_WAIT_TIMEOUT_SEC

            while asyncio.get_event_loop().time() < deadline:
                await asyncio.sleep(10)
                poll = await client.get(
                    f"{base_url}/tasks/{task_id}",
                    headers={"Authorization": f"Bearer {api_key}"},
                    timeout=30,
                )
                data = poll.json()
                status = data.get("output", {}).get("task_status")
                if status == "SUCCEEDED":
                    return data["output"]["video_url"]
                if status == "FAILED":
                    code = data.get("output", {}).get("code", "")
                    message = data.get("output", {}).get("message", "")
                    raise VideoGenAPIError(
                        f"Wan 2.7 task FAILED (code={code!r}, message={message!r})"
                    )

            raise VideoGenTimeoutError(
                f"Wan 2.7 task did not complete within {DEFAULT_WAIT_TIMEOUT_SEC:.0f}s"
            )
    except asyncio.TimeoutError as exc:
        raise VideoGenTimeoutError(
            f"Wan 2.7 generation exceeded the {DEFAULT_WAIT_TIMEOUT_SEC:.0f}s wait timeout"
        ) from exc
    except (VideoGenTimeoutError, VideoGenAPIError):
        raise
    except Exception as exc:  # noqa: BLE001
        raise VideoGenAPIError(str(exc)) from exc


# ---------------------------------------------------------------------------
# HappyHorse 1.1-i2v (US Virginia) raw REST caller.
# Same input schema as Wan 2.7 intl: media = [{"type": "first_frame", "url": ...}]
# Uses the primary US API key + DASHSCOPE_VIDEO_BASE_URL.
# ---------------------------------------------------------------------------
async def _call_happyhorse_video_gen(
    *,
    image_url: str,
    prompt: str,
    negative_prompt: str,
    duration_sec: float,
    resolution: str,
    seed: Optional[int] = None,
) -> str:
    api_key = os.environ["DASHSCOPE_API_KEY"]
    base_url = os.environ.get("DASHSCOPE_VIDEO_BASE_URL", "https://dashscope-us.aliyuncs.com/api/v1")
    model = os.environ["MODEL_VIDEO"]

    body: dict = {
        "model": model,
        "input": {
            "prompt": prompt,
            "media": [{"type": "first_frame", "url": image_url}],
        },
        "parameters": {
            "duration": int(round(duration_sec)),
            "resolution": resolution,
        },
    }
    if negative_prompt:
        body["input"]["negative_prompt"] = negative_prompt
    if seed is not None:
        body["parameters"]["seed"] = seed

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "X-DashScope-Async": "enable",
    }

    try:
        async with httpx.AsyncClient() as client:
            resp = await asyncio.wait_for(
                client.post(
                    f"{base_url}/services/aigc/video-generation/video-synthesis",
                    headers=headers,
                    json=body,
                    timeout=30,
                ),
                timeout=DEFAULT_WAIT_TIMEOUT_SEC,
            )
            if resp.status_code != 200:
                raise VideoGenAPIError(f"HappyHorse submit failed: HTTP {resp.status_code}: {resp.text[:200]}")

            task_id = resp.json()["output"]["task_id"]
            deadline = asyncio.get_event_loop().time() + DEFAULT_WAIT_TIMEOUT_SEC

            while asyncio.get_event_loop().time() < deadline:
                await asyncio.sleep(10)
                poll = await client.get(
                    f"{base_url}/tasks/{task_id}",
                    headers={"Authorization": f"Bearer {api_key}"},
                    timeout=30,
                )
                data = poll.json()
                status = data.get("output", {}).get("task_status")
                if status == "SUCCEEDED":
                    return data["output"]["video_url"]
                if status == "FAILED":
                    code = data.get("output", {}).get("code", "")
                    message = data.get("output", {}).get("message", "")
                    raise VideoGenAPIError(
                        f"HappyHorse task FAILED (code={code!r}, message={message!r})"
                    )

            raise VideoGenTimeoutError(
                f"HappyHorse task did not complete within {DEFAULT_WAIT_TIMEOUT_SEC:.0f}s"
            )
    except asyncio.TimeoutError as exc:
        raise VideoGenTimeoutError(
            f"HappyHorse generation exceeded the {DEFAULT_WAIT_TIMEOUT_SEC:.0f}s wait timeout"
        ) from exc
    except (VideoGenTimeoutError, VideoGenAPIError):
        raise
    except Exception as exc:  # noqa: BLE001
        raise VideoGenAPIError(str(exc)) from exc


# ---------------------------------------------------------------------------
# Primary video-gen dispatcher.
# Routes: HappyHorse (US REST) → Wan 2.7 INTL (SG REST) → Wan 2.6 (US SDK).
# ---------------------------------------------------------------------------
async def _call_wan_video_gen(
    *,
    image_url: str,
    prompt: str,
    negative_prompt: str,
    duration_sec: float,
    resolution: str,
    seed: Optional[int] = None,
) -> str:
    """Submit + wait for one video generation. Returns the video URL on success.
    Raises VideoGenTimeoutError / VideoGenAPIError on failure -- never retries.

    Routing priority:
      1. MODEL_VIDEO starts with "happyhorse" → _call_happyhorse_video_gen (US REST)
      2. INTL env vars all present         → _call_wan_video_gen_intl (SG REST)
      3. Otherwise                         → dashscope SDK (Wan 2.6 US)
    """
    model = os.environ.get("MODEL_VIDEO", "")
    if model.startswith("happyhorse"):
        return await _call_happyhorse_video_gen(
            image_url=image_url,
            prompt=prompt,
            negative_prompt=negative_prompt,
            duration_sec=duration_sec,
            resolution=resolution,
            seed=seed,
        )

    if _use_intl_video():
        return await _call_wan_video_gen_intl(
            image_url=image_url,
            prompt=prompt,
            negative_prompt=negative_prompt,
            duration_sec=duration_sec,
            resolution=resolution,
            seed=seed,
        )

    dashscope.api_key = os.environ["DASHSCOPE_API_KEY"]
    video_base_url = os.getenv("DASHSCOPE_VIDEO_BASE_URL")
    if video_base_url:
        dashscope.base_http_api_url = video_base_url

    call_kwargs = dict(
        model=model,
        prompt=prompt,
        negative_prompt=negative_prompt,
        img_url=image_url,
        duration=int(round(duration_sec)),
        resolution=resolution,
        # v8 fix: explicit, not DashScope's silent default -- see
        # DEFAULT_PROMPT_EXTEND's module-level comment for why.
        prompt_extend=DEFAULT_PROMPT_EXTEND,
    )
    if seed is not None:
        call_kwargs["seed"] = seed

    try:
        response = await asyncio.wait_for(
            AioVideoSynthesis.call(**call_kwargs),
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
    job_id: str
    generated_shots: Annotated[dict[str, GeneratedShot], _merge_dicts]
    failures: Annotated[list[dict], _concat_lists]


async def generate_videos(
    shots: list[Shot],
    product_truths: list[ProductTruth],
    treatment: Optional[Treatment],
    product_photos: list[str],
    generate_fn: Optional[GenerateFn] = None,
    job_id: str = "unknown_job",
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

    # Optional, for future A/B testing and this fix's own reproducibility during
    # manual verification -- unset by default, which preserves today's random-
    # seed production behavior exactly (see `seed` note on _call_wan_video_gen).
    _seed_env = os.getenv("VIDEO_GEN_SEED")
    seed: Optional[int] = int(_seed_env) if _seed_env else None

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

        first_frame_url = image_url

        try:
            call_kwargs = dict(
                image_url=first_frame_url,
                prompt=prompt,
                negative_prompt=shot["negative_prompt"],
                duration_sec=duration,
                resolution=resolution,
            )
            if seed is not None:
                call_kwargs["seed"] = seed
            video_url = await fn(**call_kwargs)
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
                    "job_id": state.get("job_id", "unknown_job"),
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
            "job_id": job_id,
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
    product_photos = state.get("product_photos", [])
    if _use_intl_video() and product_photos:
        logger.info(
            "Video-Gen Node: INTL mode (Wan 2.7 SG) -- re-uploading %d photo(s) to Singapore OSS bucket.",
            len(product_photos),
        )
        product_photos = await asyncio.to_thread(_reupload_photos_to_intl_oss, product_photos, job_id)

    shots, generated = await generate_videos(
        shots=state.get("shot_list", []),
        product_truths=state.get("product_truths", []),
        treatment=state.get("treatment"),
        product_photos=product_photos,
        job_id=job_id,
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
