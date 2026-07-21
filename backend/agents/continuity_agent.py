"""
Continuity Agent -- Qwen-VL drift scoring (Phase 4).
Spec of record: docs/TECHNICAL_DOCUMENTATION.md §5.10.

WHAT THIS IS FOR. Guard visual fidelity of every REAL generated clip. For each
`status == "passed"` shot it extracts one representative frame from the generated
clip and, in ONE Qwen-VL call, compares it against (a) the shot's source
reference photo -- product-identity drift (color/shape/label) -- and (b) the
shot's shared `lighting`/style string -- cross-shot style consistency (§5.10's
two checks). The result is a single continuous `drift_score` written onto
`generated_shots[shot_id]`.

DRIFT SCALE (documented, single source of truth). `drift_score` is a float in
[0.0, 1.0] where:
  * 0.0  = the generated frame is a perfect identity/style match to the reference.
  * 1.0  = a totally different product or style (worst case).
The pass/fail line is `DRIFT_THRESHOLD` (default 0.35, env-overridable via
CONTINUITY_DRIFT_THRESHOLD, same "tunable constant, flag-don't-hardcode-forever"
pattern as agents/budget_gate.py's DEFAULT_JOB_BUDGET_CAP). "lower is more
similar" -- a shot PASSES when `drift_score <= DRIFT_THRESHOLD`. This constant
lives HERE (the scorer owns the scale) and is imported by the Continuity Gate
(agents/continuity_gate.py) so the score, the emitted `drift_scored` event, and
the retry/review decision all read ONE definition rather than two that could
drift apart.

WHY SCORE ONLY "passed" SHOTS. This mirrors ken_burns_fallback_node.py's own
status-filtering shape:
  * `status == "passed"` -> a real Wan-generated clip. Score it.
  * `status == "fallback"` -> a Ken-Burns pan/zoom over the SAME reference photo
    it would be compared against; drift is definitionally near-zero and not a
    meaningful check, so SKIP.
  * `status == "fallback_requested"` -> handed off, broken, no clip yet; nothing
    to score, SKIP.
  * anything else (pending/generating/review) -> not a finished real clip, SKIP.

WHY SCORE ONLY UN-SCORED CLIPS (retry-loop efficiency). A passed shot whose
GeneratedShot entry ALREADY carries a `drift_score` is skipped. On a Continuity
retry loop (Gate resets a drifted shot to "pending" -> Video-Gen regenerates
ONLY that shot and writes a FRESH GeneratedShot entry with no `drift_score` ->
back here), this means we re-score exactly the freshly-(re)generated clip and
leave every already-scored, unchanged clip alone -- no redundant Qwen-VL calls
and no duplicate `drift_scored` events for shots that didn't change. On the
first pass, no entry has a score yet, so every passed shot is scored.

SEPARATION OF CONCERNS -- "never a self-grade", and the interrupt-rerun gotcha.
This node ONLY scores; it NEVER changes a shot's `status`. The decision (retry /
human-review / pass) is the Continuity Gate's job (§5.10 control flow), which is
a SEPARATE downstream node. That split is not cosmetic: the Gate is the node
that calls LangGraph `interrupt()`, and on resume LangGraph re-executes that
whole node from the top. Keeping the expensive, non-deterministic Qwen-VL
scoring in THIS node means resuming a human decision never pointlessly re-fires
the vision model -- the drift scores are already committed to state by the time
the Gate runs.

FAILURE ISOLATION (conservative). One shot's Qwen-VL/ffmpeg failure must not
block the others (same posture as every node in this pipeline). On such a
failure we do NOT silently pass the shot Continuity couldn't actually check:
we write the WORST-CASE score (`_FAILED_DRIFT_SCORE = 1.0`, definitively over
any sane threshold) rather than a passing one, so the shot flows into the Gate's
retry/review path instead of sneaking through unverified. The `drift_scored`
event for that shot therefore carries `passed=False`.

FRAME EXTRACTION. We download the generated clip to a temp file (httpx, same
pattern as agents/_oss.py's `_download_to_temp`, which we reuse directly) and
pull ONE frame near the clip's midpoint via ffmpeg
(`ffmpeg -ss <duration/2> -i <clip> -frames:v 1 <out>.jpg`), cleaning up both
temp files afterward. The midpoint (not frame 0) is deliberate: a clip's opening
frame is often still settling from the reference still, so mid-clip is the more
honest test of whether the product held its identity through the motion.

`candidate_frame_uris` NOTE (deliberately simple, per the Phase 4 scope call).
This node does NOT separately persist the extracted still to OSS. The Continuity
Gate populates a review entry's `candidate_frame_uris` with the CLIP's own
`video_uri` (a reviewer can scrub it), which avoids adding an image-upload path
to agents/_oss.py (whose helper is video/mp4-shaped) for a still we only needed
transiently to score. Building the actual review UI is explicitly out of scope
here, so the simpler option is the right one -- see agents/continuity_gate.py.

INJECTABLE DEPENDENCIES (same `client=None` testability pattern as every other
agent here). `client` (an AsyncOpenAI-compatible Qwen-VL client) and
`extract_frame_fn` (the ffmpeg-boundary frame extractor) are both injectable, so
tests fake the vision call and the ffmpeg step without touching real DashScope
or ffmpeg -- matching how tests/_phase3_graph.py fakes Ken-Burns's render step.

FRAME-0(ISH) IDENTITY CHECK (v8 fix -- Meta Quest -> "phone on a stand"
wrong-object bug). This is a SECOND, INDEPENDENT check alongside the drift
score above, not a stricter threshold on the same [0,1] scale. Drift asks a
CONTINUOUS question ("how similar is this frame to the reference,
color/shape/style-wise"); identity asks a CATEGORICAL one ("is this even the
same physical object class at all"). A wrong-object generation can plausibly
score a misleadingly moderate drift (lighting/composition can still roughly
match), so a purely continuous check is not guaranteed to catch it -- hence a
dedicated categorical verdict, `DRIFT_THRESHOLD` never involved.

Deliberately NOT frame 0 -- i2v models keep frame 0 tightly conditioned on the
reference image (often near-identical), so an identity check AT t=0 can
trivially pass even when the clip drifts to the wrong object by t=1s. We check
an EARLY frame instead: `at_sec = max(0.4, 0.1 * duration_sec_used)`. This
reuses the frame-extraction boundary generalized below (`extract_frame`, of
which `extract_midpoint_frame` is now a thin wrapper) with a different seek
target, and a SEPARATE Qwen-VL call (`_call_qwen_vl_identity`) with its own
system prompt (`_IDENTITY_SYSTEM_PROMPT`) forcing feature-by-feature evidence
BEFORE a same_object verdict -- deliberate field ordering (mirrors this
module's BaseModel-validation posture below) that measurably reduces blind-
approval bias in multimodal judging.

Runs ALONGSIDE drift scoring in `score_continuity`, same scope restriction
(only un-scored `status == "passed"` shots -- see WHY SCORE ONLY "passed"
SHOTS / WHY SCORE ONLY UN-SCORED CLIPS above) and the SAME conservative
failure posture: an identity-check failure (bad JSON, API error, ffmpeg
failure) is recorded as the worst case, `same_object=False`, never silently
passed -- exactly mirroring `_FAILED_DRIFT_SCORE`'s posture for drift. Written
onto `generated_shots[shot_id]["identity_check"]`, ADDITIVE alongside
`drift_score` (graph/state.py v8), never replacing it. This node still only
SCORES -- `agents/continuity_gate.py` owns the routing decision on a hard
identity failure, exactly as it already owns the drift-retry/review decision.
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import re
import tempfile
from typing import Callable, Literal, Optional

import ffmpeg
from langchain_core.callbacks.manager import adispatch_custom_event
from langchain_core.runnables import RunnableConfig
from openai import AsyncOpenAI
from pydantic import BaseModel, ConfigDict, Field, StrictBool

from agents._oss import _download_to_temp
from agents._retry import create_completion
# Reuse (never re-implement) Video-Gen's reference-photo resolver and its frozen
# "passed" status constant, so the "photo_N" -> URL convention and the "passed"
# spelling cannot silently drift between modules (same reuse posture Ken-Burns
# takes toward video_gen_node.py).
from agents.video_gen_node import SUCCESS_STATUS, _resolve_reference_image_url
from graph.state import GeneratedShot, ProductCutState, Shot

logger = logging.getLogger("productcut.agents.continuity_agent")

# Pass/fail line on the [0,1] drift scale (see module docstring). Lower = more
# similar; a shot passes when drift_score <= DRIFT_THRESHOLD. Env-overridable
# like every other tunable constant in this codebase (budget_gate.py precedent).
# Owned HERE and imported by continuity_gate.py -- single source of truth.
DRIFT_THRESHOLD = float(os.getenv("CONTINUITY_DRIFT_THRESHOLD", "0.35"))

# The worst-case score written when Continuity could NOT actually check a shot
# (Qwen-VL or ffmpeg failure). Deliberately > any sane threshold so the shot is
# routed to the Gate's retry/review path, never silently passed (see docstring).
_FAILED_DRIFT_SCORE = 1.0

# ExtractFrameFn contract: (video_uri, seek_seconds) -> local_jpg_path. The
# caller deletes the returned path. Injectable so tests fake the ffmpeg boundary.
ExtractFrameFn = Callable[[str, float], str]

_DRIFT_SYSTEM_PROMPT = """You are a strict visual-continuity checker for an ad-video pipeline. You are
given a REFERENCE product photo and one FRAME extracted from a generated video
clip that is supposed to depict the SAME physical product, plus the ad's shared
lighting/style description.

Judge TWO things:
  (a) product-identity drift: does the product in the generated frame still match
      the reference in color, shape, proportions, materials, and any label/logo/
      text? A different product, a warped/morphed shape, a wrong color, or a
      distorted label is HIGH drift.
  (b) style consistency: does the frame's lighting/mood match the shared
      lighting/style string given below?

Return ONLY a JSON object in this exact shape, no preamble or commentary:
{
  "drift_score": 0.0,
  "justification": "one short sentence citing the biggest identity/style difference (or lack of one)"
}

drift_score is a single number in [0.0, 1.0]:
  0.0  = perfect match to the reference product and style.
  1.0  = a totally different product or completely inconsistent style.
Use the middle of the range for partial/subtle drift. Be honest and specific."""

# v8 fix -- see module docstring's "FRAME-0(ISH) IDENTITY CHECK" section. Field
# order in the required JSON (features/evidence BEFORE the same_object verdict)
# is deliberate: it forces the model to generate comparison evidence before
# committing to a yes/no, mirrored in IdentityCheckResult's own field order below.
_IDENTITY_SYSTEM_PROMPT = """You are a strict product-identity verifier for an ad-video pipeline. You are
given a REFERENCE product photo and one FRAME from the very start of a
generated video clip that is REQUIRED to depict the same physical product.

Your job is to catch TWO failure modes:
  (A) WRONG OBJECT: the video shows a different kind of object entirely.
  (B) PHANTOM PARTS / DISAPPEARING PARTS: the frame shows physical components
      that DO NOT EXIST in the reference photo (e.g. a second lid, a phantom
      hinge, a duplicate piece), OR parts that ARE in the reference have
      completely vanished. NOTE: it is NOT a failure for an existing mechanism
      to operate (a lighter lid may open, a bottle cap may be off) as long as
      the product's part count and overall physical structure remain consistent
      with the reference.

IGNORE background, scene, camera angle, lighting, and mechanical state of
existing parts. Judge ONLY whether the object is the same type AND has the
same set of physical components.

Work in this exact order:
1. List the distinct physical parts/components visible in the REFERENCE photo
   (e.g. "hinged lid", "rectangular body", "hinge bar", "wheel striker").
2. For each part, state whether it is present in the FRAME, absent, or
   cannot be determined.
3. Check if the FRAME shows any parts that do NOT appear anywhere in the
   reference (phantom additions).
4. Only then decide: same_object is false if (a) the object is a different
   product type, (b) a phantom component appears that has no counterpart in
   the reference, or (c) a clearly present reference component has completely
   disappeared. Do not flag mechanism operation (open vs closed lid) as a
   failure — only flag structural additions or structural disappearances.

Return ONLY this JSON, keys in this exact order:
{
  "matching_features": ["..."],
  "mismatching_features": ["..."],
  "same_object": true,
  "confidence": "high" | "medium" | "low"
}"""


class IdentityCheckResult(BaseModel):
    """Validated frame-0(ish) identity-check output (see module docstring).

    Field order mirrors the exact order the prompt demands (evidence BEFORE the
    verdict) -- Pydantic doesn't enforce input JSON key order, but keeping the
    model's own field order the same documents the intended reasoning sequence
    for a future reader. `extra="forbid"` / StrictBool mirror this codebase's
    other raw-LLM-JSON validation gates (see agents/body_checker.py's
    BodyCheckResult for the precedent this is modeled on).
    """

    model_config = ConfigDict(extra="forbid")

    matching_features: list[str] = Field(default_factory=list)
    mismatching_features: list[str] = Field(default_factory=list)
    same_object: StrictBool
    confidence: Literal["high", "medium", "low"]


# ---------------------------------------------------------------------------
# ffmpeg frame extraction (boundary -- faked in tests, run off-loop in prod).
# ---------------------------------------------------------------------------
def extract_frame(
    video_uri: str,
    at_sec: float,
    download_fn: Optional[Callable[[str], str]] = None,
) -> str:
    """Download the generated clip and extract ONE frame at `at_sec` seconds in.

    Returns the local .jpg path (the CALLER deletes it). The downloaded clip is
    always cleaned up here. `download_fn` reuses agents/_oss.py's
    `_download_to_temp` by default (httpx, follow_redirects for OSS/CDN 3xx);
    injectable only so a caller could swap the transport.

    ffmpeg invocation matches the spec: `-ss <at_sec>` (seek BEFORE `-i` for a
    fast keyframe seek) then `-frames:v 1` to grab a single frame. Generalizes
    what was `extract_midpoint_frame`'s own inline ffmpeg call (see that
    function, now a thin wrapper around this one, for the midpoint-specific
    call site; the v8 identity check below calls this directly with an EARLY
    seek target instead).
    """
    dl = download_fn or _download_to_temp
    clip_path = dl(video_uri)
    try:
        out_fd, out_path = tempfile.mkstemp(suffix=".jpg", prefix="continuity_frame_")
        os.close(out_fd)  # ffmpeg writes the file itself; we only needed a unique path
        seek = max(at_sec, 0.0)
        try:
            (
                ffmpeg
                .input(clip_path, ss=seek)
                .output(out_path, vframes=1)
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True)
            )
        except Exception:
            if os.path.exists(out_path):
                try:
                    os.remove(out_path)
                except OSError:
                    pass
            raise
        return out_path
    finally:
        if os.path.exists(clip_path):
            try:
                os.remove(clip_path)
            except OSError:
                pass


def extract_midpoint_frame(
    video_uri: str,
    duration_sec: float,
    download_fn: Optional[Callable[[str], str]] = None,
) -> str:
    """Thin wrapper over `extract_frame`, preserved so every existing caller
    (score_continuity's drift path, and every test that imports this name)
    keeps working unchanged. Seeks to the clip's midpoint -- the drift check's
    own rationale (a clip's opening frame is often still settling from the
    reference still, so mid-clip is the more honest drift test) is unaffected
    by this refactor; only the ffmpeg mechanics moved into `extract_frame`.
    """
    return extract_frame(video_uri, at_sec=duration_sec / 2.0, download_fn=download_fn)


def _encode_image_to_data_uri(path: str) -> str:
    """Base64-encode a LOCAL image as a data: URI.

    The extracted frame lives only on local disk, so (unlike the reference photo,
    which is already a fetchable OSS URL) it must be inlined as a data URI for
    DashScope's OpenAI-compatible vision endpoint -- the same `image_url` block
    shape agents/product_truth_extractor.py uses, just with a data URI instead of
    a remote URL.
    """
    with open(path, "rb") as fh:
        b64 = base64.b64encode(fh.read()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def _parse_json_response(raw: str) -> dict:
    """Strip markdown code fences (models wrap JSON in ```json ... ```) and parse.

    Mirrors agents/product_truth_extractor.py's `_parse_json_response`.
    """
    stripped = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.IGNORECASE)
    return json.loads(stripped)


# ---------------------------------------------------------------------------
# One Qwen-VL drift call (reuses product_truth_extractor's vision message shape).
# ---------------------------------------------------------------------------
async def _call_qwen_vl_drift(
    reference_url: str,
    frame_path: str,
    lighting: str,
    client: Optional[AsyncOpenAI] = None,
) -> tuple[float, str]:
    """One Qwen-VL call comparing the extracted frame to the reference + style.

    Returns (drift_score clamped to [0,1], justification). `client` is injectable
    (same pattern as product_truth_extractor.py); when None we build our own from
    the DashScope OpenAI-compatible env vars and close it after.
    """
    model = os.environ["MODEL_VISION"]  # KeyError intentional -- see product_truth_extractor.py
    own_client = client is None
    if own_client:
        client = AsyncOpenAI(
            api_key=os.environ["DASHSCOPE_API_KEY"],
            base_url=os.environ["DASHSCOPE_BASE_URL"],
            timeout=60.0,
        )
    try:
        frame_data_uri = _encode_image_to_data_uri(frame_path)
        user_content = [
            {"type": "text", "text": "REFERENCE product photo:"},
            {"type": "image_url", "image_url": {"url": reference_url}},
            {"type": "text", "text": "FRAME extracted from the generated clip (check this):"},
            {"type": "image_url", "image_url": {"url": frame_data_uri}},
            {
                "type": "text",
                "text": f"Shared lighting/style string for the ad: {lighting}\n"
                "Return only the JSON object.",
            },
        ]
        messages = [
            {"role": "system", "content": _DRIFT_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        raw = await create_completion(client, model=model, messages=messages)
        parsed = _parse_json_response(raw)
        drift = float(parsed.get("drift_score"))
        drift = min(max(drift, 0.0), 1.0)  # clamp defensively to the documented scale
        return drift, str(parsed.get("justification", ""))
    finally:
        if own_client:
            await client.close()


async def _score_one_shot(
    shot: Shot,
    entry: GeneratedShot,
    product_photos: list[str],
    client: Optional[AsyncOpenAI],
    extract_frame_fn: ExtractFrameFn,
) -> tuple[float, str]:
    """Extract the midpoint frame, score it, and always clean the frame up."""
    reference_url = _resolve_reference_image_url(shot["reference_image_id"], product_photos)
    # Prefer the clip's realized duration (Video-Gen may have budget-clamped it)
    # over the shot's nominal duration, so the midpoint seek lands inside the clip.
    duration = float(entry.get("duration_sec_used") or shot["duration_sec"])
    # ffmpeg + httpx are blocking/CPU-bound -- run off the event loop (matches this
    # codebase's async-everywhere convention, e.g. ken_burns_fallback_node.py).
    frame_path = await asyncio.to_thread(extract_frame_fn, entry["video_uri"], duration)
    try:
        return await _call_qwen_vl_drift(reference_url, frame_path, shot["lighting"], client=client)
    finally:
        if frame_path and os.path.exists(frame_path):
            try:
                os.remove(frame_path)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# One Qwen-VL identity call (v8 fix -- see module docstring's "FRAME-0(ISH)
# IDENTITY CHECK" section). Separate call, separate frame, separate prompt from
# the drift call above -- a categorical verdict, not a continuous score.
# ---------------------------------------------------------------------------
async def _call_qwen_vl_identity(
    reference_url: str,
    frame_path: str,
    client: Optional[AsyncOpenAI] = None,
) -> IdentityCheckResult:
    """One Qwen-VL call judging whether the FRAME is even the same physical
    object as the REFERENCE photo. Returns a validated IdentityCheckResult;
    raises (ValidationError / JSONDecodeError / etc.) on any malformed output --
    the caller (`_score_one_shot_identity`'s caller, `score_continuity`) is
    responsible for the worst-case fallback, mirroring `_call_qwen_vl_drift`'s
    contract with its own caller.
    """
    model = os.environ["MODEL_VISION"]  # KeyError intentional -- see product_truth_extractor.py
    own_client = client is None
    if own_client:
        client = AsyncOpenAI(
            api_key=os.environ["DASHSCOPE_API_KEY"],
            base_url=os.environ["DASHSCOPE_BASE_URL"],
            timeout=60.0,
        )
    try:
        frame_data_uri = _encode_image_to_data_uri(frame_path)
        user_content = [
            {"type": "text", "text": "REFERENCE product photo:"},
            {"type": "image_url", "image_url": {"url": reference_url}},
            {
                "type": "text",
                "text": "FRAME extracted from the very start of the generated clip (check this):",
            },
            {"type": "image_url", "image_url": {"url": frame_data_uri}},
            {"type": "text", "text": "Return only the JSON object."},
        ]
        messages = [
            {"role": "system", "content": _IDENTITY_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        raw = await create_completion(client, model=model, messages=messages)
        parsed = _parse_json_response(raw)
        return IdentityCheckResult.model_validate(parsed)
    finally:
        if own_client:
            await client.close()


async def _score_one_shot_identity(
    shot: Shot,
    entry: GeneratedShot,
    product_photos: list[str],
    client: Optional[AsyncOpenAI],
    extract_frame_fn: ExtractFrameFn,
) -> IdentityCheckResult:
    """Extract an EARLY frame (not frame 0 -- see module docstring for why) and
    run the identity check, always cleaning the frame up.

    `at_sec = max(0.4, 0.1 * duration_sec_used)`: a small absolute floor so a
    very short clip still seeks a moment past the tightly-conditioned opening
    frame, otherwise 10% into the clip's REALIZED duration (Video-Gen may have
    budget-clamped it, same "prefer duration_sec_used" precedent as
    `_score_one_shot`'s drift midpoint above).
    """
    reference_url = _resolve_reference_image_url(shot["reference_image_id"], product_photos)
    duration = float(entry.get("duration_sec_used") or shot["duration_sec"])
    at_sec = max(0.4, 0.1 * duration)
    frame_path = await asyncio.to_thread(extract_frame_fn, entry["video_uri"], at_sec)
    try:
        return await _call_qwen_vl_identity(reference_url, frame_path, client=client)
    finally:
        if frame_path and os.path.exists(frame_path):
            try:
                os.remove(frame_path)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Batch scorer (status-filtered, per-shot isolated) -- the reusable core.
# ---------------------------------------------------------------------------
async def score_continuity(
    shots: list[Shot],
    generated_shots: dict[str, GeneratedShot],
    product_photos: list[str],
    client: Optional[AsyncOpenAI] = None,
    extract_frame_fn: Optional[ExtractFrameFn] = None,
    identity_extract_frame_fn: Optional[ExtractFrameFn] = None,
) -> tuple[dict[str, GeneratedShot], list[dict]]:
    """Score every un-scored `passed` shot's drift AND identity; leave everything
    else alone.

    Returns (updated_entries, scored_records):
      * updated_entries: one GeneratedShot per shot we scored, each a copy of the
        existing entry with `drift_score` AND `identity_check` written in. Merge
        it INTO the outer state's `generated_shots` (the node wrapper does this)
        -- it never contains entries for shots we skipped.
      * scored_records: one dict per scored shot for event emission --
        {shot_id, drift_score, passed, attempt, identity_check}.

    Never raises for a single shot's failure: a Qwen-VL/ffmpeg error on EITHER
    check is caught and isolated from the other (drift and identity are two
    independent checks against two independently-extracted frames -- one
    failing must not silently skip or corrupt the other). A failed drift check
    is recorded as the worst-case `_FAILED_DRIFT_SCORE`; a failed identity
    check is recorded as the worst-case `same_object=False` -- both mirror the
    same "never silently pass" posture (see module docstring). Does NOT change
    any shot's `status` -- that is the Continuity Gate's job.
    """
    extract = extract_frame_fn or extract_midpoint_frame
    # Defaults to whatever `extract_frame_fn` was given (if any) before falling
    # back to the real `extract_frame`: a caller that only fakes ONE frame-
    # extraction boundary (the common case -- every existing test here does
    # this) gets that same fake reused for the identity check too, rather than
    # the identity path silently falling through to a REAL httpx download +
    # ffmpeg call. Production (`continuity_agent_node`, neither arg given)
    # is unaffected -- both still resolve to real extraction, just at
    # different seek points (midpoint vs. the early identity frame).
    identity_extract = identity_extract_frame_fn or extract_frame_fn or extract_frame
    updated: dict[str, GeneratedShot] = {}
    scored_records: list[dict] = []

    for shot in shots:
        if shot.get("status") != SUCCESS_STATUS:
            continue  # only real Wan clips are scored (see docstring's status table)
        shot_id = shot["shot_id"]
        entry = generated_shots.get(shot_id)
        if entry is None:
            logger.warning(
                "Continuity Agent: shot %s is 'passed' but has no generated_shots "
                "entry -- skipping (nothing to score).", shot_id,
            )
            continue
        if "drift_score" in entry:
            continue  # already scored (unchanged clip on a retry loop) -- see docstring

        try:
            drift, justification = await _score_one_shot(shot, entry, product_photos, client, extract)
            logger.info(
                "Continuity Agent: shot %s drift_score=%.3f (threshold %.3f) -- %s",
                shot_id, drift, DRIFT_THRESHOLD, justification,
            )
        except Exception as exc:  # noqa: BLE001 -- one shot's failure must not block the batch
            drift = _FAILED_DRIFT_SCORE
            logger.error(
                "Continuity Agent: scoring FAILED for shot %s (%s) -- recording "
                "worst-case drift_score=%.1f so it is NOT silently passed; it will "
                "flow into the Gate's retry/review path.",
                shot_id, exc, _FAILED_DRIFT_SCORE,
            )

        try:
            identity_result = await _score_one_shot_identity(
                shot, entry, product_photos, client, identity_extract
            )
            identity_check = identity_result.model_dump()
            logger.info(
                "Continuity Agent: shot %s identity_check same_object=%s confidence=%s",
                shot_id, identity_check["same_object"], identity_check["confidence"],
            )
        except Exception as exc:  # noqa: BLE001 -- an identity failure must not block drift or the batch
            identity_check = {
                "matching_features": [],
                "mismatching_features": [],
                "same_object": False,
                "confidence": "low",
            }
            logger.error(
                "Continuity Agent: identity check FAILED for shot %s (%s) -- "
                "recording worst-case same_object=False so it is NOT silently "
                "passed; it will flow into the Gate's hard-identity-failure path.",
                shot_id, exc,
            )

        updated[shot_id] = {**entry, "drift_score": drift, "identity_check": identity_check}
        scored_records.append(
            {
                "shot_id": shot_id,
                "drift_score": drift,
                "passed": drift <= DRIFT_THRESHOLD,
                "attempt": shot.get("retry_count", 0),
                "identity_check": identity_check,
            }
        )

    return updated, scored_records


# ---------------------------------------------------------------------------
# LangGraph node wrapper.
# ---------------------------------------------------------------------------
async def continuity_agent_node(
    state: ProductCutState,
    config: Optional[RunnableConfig] = None,
) -> dict:
    """LangGraph node wrapper: scores drift for every un-scored `passed` shot and
    writes `drift_score` onto its `generated_shots` entry (§5.10 output contract).

    Emits one C2 `drift_scored` event per scored shot (the dashed `CTY -.-> FE`
    streaming edge), carrying {shot_id, drift_score, threshold, passed, attempt},
    dispatched via `adispatch_custom_event` (mirrors budget_gate_node /
    video_gen_node). Does NOT change any shot's `status` -- the Continuity Gate
    decides retry/review/pass. `config` defaults to None so the node stays
    directly callable/testable; LangGraph injects the real RunnableConfig.
    """
    shots = state.get("shot_list", [])
    generated = state.get("generated_shots", {})
    product_photos = state.get("product_photos", [])

    updated, scored_records = await score_continuity(shots, generated, product_photos)

    for rec in scored_records:
        await adispatch_custom_event(
            "drift_scored",
            {
                "shot_id": rec["shot_id"],
                "drift_score": rec["drift_score"],
                "threshold": DRIFT_THRESHOLD,
                "passed": rec["passed"],
                "attempt": rec["attempt"],
            },
            config=config,
        )

    n_scored = len(scored_records)
    n_flagged = sum(1 for r in scored_records if not r["passed"])
    trace_note = (
        f"\n[continuity_agent] scored {n_scored} shot(s) against drift threshold "
        f"{DRIFT_THRESHOLD}; {n_flagged} over threshold."
    )
    return {
        # Merge new drift_score entries INTO existing generated_shots so other
        # shots' entries are never clobbered (overwrite-semantics field).
        "generated_shots": {**generated, **updated},
        "reasoning_trace": state.get("reasoning_trace", "") + trace_note,
    }


__all__ = [
    "DRIFT_THRESHOLD",
    "IdentityCheckResult",
    "extract_frame",
    "extract_midpoint_frame",
    "score_continuity",
    "continuity_agent_node",
]
