"""
Continuity Gate -- capped retry + autonomous auto-accept (Phase 4).
Spec of record: docs/TECHNICAL_DOCUMENTATION.md §5.10 (control flow / failure
handling).

AUTONOMY NOTE (current behavior). This pipeline is FULLY AUTONOMOUS: there is
no human-in-the-loop. The retry-exhausted branch below no longer calls
LangGraph's `interrupt()` -- it auto-accepts the best-available generated clip
instead. Ken-Burns static-image fallback is reserved for HARD failures (a
Video-Gen infra/API failure, or two consecutive hard-identity failures), NOT
for continuity-quality drift. The `interrupt()`/human-review machinery once
described here (and the `_apply_resolution` helper / `HumanReviewEntry` type,
kept below only for backward reference) is no longer wired into the node.

WHAT THIS IS FOR. Turn the drift scores the Continuity Agent
(agents/continuity_agent.py) already wrote into DECISIONS. For every real
(`status == "passed"`) shot that carries a `drift_score`, this node applies
§5.10's three-way control flow:

  1. Drift within threshold (`drift_score <= DRIFT_THRESHOLD`)
     -> leave the shot exactly as-is (stays "passed"). It passes to Assembly.

  2. Drift over threshold AND `retry_count < MAX_AUTO_RETRIES`
     -> automatic re-generation. Set `status = "pending"` (the existing frozen
     value Video-Gen's filter treats as "(re-)generate this") and increment
     `retry_count` by 1. THIS NODE IS THE ONLY PLACE IN THE PIPELINE THAT EVER
     INCREMENTS `retry_count` -- Video-Gen and Ken-Burns both explicitly never
     touch it (their own docstrings + a grep confirm this), reserving the whole
     retry budget for quality-driven drift retries per §5.8/§5.9.

  3. Drift over threshold AND retries exhausted (`retry_count >= MAX_AUTO_RETRIES`)
     -> AUTO-ACCEPT the best-available clip. Keep `status="passed"` and stamp the
     clip's generated_shots entry with `CONTINUITY_APPROVED_KEY` so a later loop
     pass (driven by a different shot) doesn't re-evaluate this still-over-
     threshold clip. No `interrupt()`, no human review, no Ken-Burns -- the shot
     flows straight to Assembly with the closest generation we could produce.

THE INTERRUPT RE-RUN CONTRACT (why the scoring is in the PREVIOUS node). On
resume, LangGraph re-executes this ENTIRE node from the top; each prior
`interrupt()` call returns its stored resume value instead of pausing again, and
the next un-resumed one pauses. This node is therefore written to be cheap and
deterministic to re-run: it does NO model calls and NO I/O before/around the
interrupts -- it only re-reads state (drift scores already committed by the
Continuity Agent) and re-derives the same decisions. Nothing this node returns
is committed until it finishes WITHOUT pausing, so `retry_count` increments,
status changes, and `human_review_queue` all commit atomically on the final
resumed pass -- never doubled in COMMITTED STATE by a pause. (Verified against
LangGraph 1.2.7's real pause/resume mechanics, including empirically: a
paused-then-resumed run leaves exactly one entry in `human_review_queue` per
reviewed shot, not two.)

KNOWN, VERIFIED LIMITATION -- the `interrupt_requested` EVENT (not state) fires
TWICE per reviewed shot across a pause+resume cycle. The event dispatch sits
BEFORE `interrupt()` (so a live watcher learns *why* the pause happened, not
just that it did) -- but per the re-run contract above, that same dispatch line
re-executes on resume too, before this call's cached value is returned. Verified
directly: two `astream_events` passes (initial pause, then resume) each yield one
`interrupt_requested` event for the same shot_id -- committed STATE stays correct
(one queue entry), only the live stream double-announces. This is a known class
of LangGraph gotcha (side effects before `interrupt()` are not exactly-once) with
no clean IN-NODE fix: there is no way to distinguish "pausing now" from "resuming
now" from inside this node without help from whatever actually drives the graph
(detecting a genuinely-new pause by diffing `graph.get_state(config)` between
calls, and synthesizing the live notification from THERE instead of from inside
the node). That's WS/dashboard-driving-layer work, explicitly out of this phase's
scope (see docs/BUILD_TASKS.md's Phase 4 deferred-follow-up note) -- flagged
here, not silently left for someone to rediscover the hard way, and locked in by
a test (`test_interrupt_requested_event_fires_twice_across_pause_resume_known_limitation`
in test_continuity_gate.py) so a future accidental "fix" that makes it stop
firing AT ALL (worse: no live notification at all) would be caught.

MULTI-SHOT REVIEW IN ONE BATCH. If several shots exhaust retries in the same
run, we call `interrupt()` once per shot, IN SHOT-LIST ORDER. LangGraph pauses
at the first, and a resumer supplies one resume value per pause
(`Command(resume=<resolution>)`), matched back to the interrupts in call order --
so resolutions route to the correct shot deterministically.

RESUME PAYLOAD SHAPE. The resume value is the human's choice, shaped to mirror
`HumanReviewEntry.resolution`'s frozen Literal so applying it is trivial:
`{"resolution": "approve" | "retry_with_edit" | "accept_fallback"}` (a bare
string is also accepted defensively). Applied by `_apply_resolution`:
  * "approve"         -> accept as generated despite the flag: `status` back to
                         "passed". The clip's drift_score STAYS over threshold
                         (it genuinely drifted; the human just accepted it), so
                         we ALSO stamp the clip's generated_shots entry with a
                         durable `continuity_approved` marker. Without it, a LATER
                         retry-loop pass driven by a DIFFERENT shot would re-read
                         this shot's still-over-threshold, retries-exhausted entry
                         and re-raise a human-review interrupt for an already-
                         approved shot (approval would not be durable across
                         passes). The Gate skips any entry carrying that marker.
  * "retry_with_edit" -> a human-authorized one-off retry. `status = "pending"`
                         so Video-Gen picks it up again. Deliberately NOT gated
                         by the `retry_count < MAX_AUTO_RETRIES` cap -- the cap
                         bounds UNSUPERVISED spend, not an informed human choice
                         (§5.10). We still increment `retry_count` for trace
                         visibility, but a value >= the cap does not block it.
  * "accept_fallback" -> `status = "fallback_requested"` -- the EXACT existing
                         Phase 3 hand-off contract, so the shot flows back through
                         the already-built ken_burns_fallback_node on the next
                         pass and gets a working Ken-Burns clip for free (no new
                         fallback logic here).
  * anything else     -> logged; the shot is left in "review" (still visibly
                         needs handling, never silently passed).

WHY THIS NODE OWNS NO THRESHOLD OF ITS OWN. `DRIFT_THRESHOLD` is imported from
agents/continuity_agent.py (the scorer defines the scale). MAX_AUTO_RETRIES is
the local, env-overridable cap (§5.10's "retry_count < 2"), same
flag-don't-hardcode-forever pattern as budget_gate.py's DEFAULT_JOB_BUDGET_CAP.

HARD IDENTITY FAILURE (v8 fix -- Meta Quest -> "phone on a stand" wrong-object
bug). `agents/continuity_agent.py` now writes a SECOND, independent verdict
onto a scored clip's entry: `identity_check.same_object` (a categorical "is
this even the same physical object class" check on an early frame, NOT a
stricter threshold on `drift_score`'s continuous scale -- see that module's
own docstring). `same_object == false` -- regardless of `confidence`, per the
identity prompt's own "do not give the benefit of the doubt" instruction -- is
treated as a HARD failure, routed by a SEPARATE, ADDITIVE decision layered
BEFORE the existing drift-threshold branch below:
  * 1st hard identity failure for a shot -> exactly ONE automatic retry (same
    mechanism as an over-threshold drift auto-retry -- `status="pending"`,
    `retry_count` incremented -- but NOT counted against `MAX_AUTO_RETRIES`,
    and logged/traced distinctly for observability). A fresh sample can escape
    a bad generation mode a drift-style retry would also fix, so it gets its
    own one-shot budget rather than being folded into the drift retry cap.
  * 2nd CONSECUTIVE hard identity failure for the same shot -> routes STRAIGHT
    to `fallback_requested` (Ken-Burns), skipping both the remaining
    `MAX_AUTO_RETRIES` budget and human review entirely -- two wrong-object
    generations in a row means the prompt/reference combination itself is
    unfixable by resampling, so burning the normal retry/review budget
    chasing it would just delay an inevitable fallback.
  * A CONSECUTIVE-failure streak is tracked via `IDENTITY_HARD_FAIL_STREAK_KEY`,
    an extra, undeclared key written directly onto the Shot dict (not a C1
    schema field -- `graph.shot_schema.validate_shot_list`'s `extra="forbid"`
    Pydantic gate only ever runs ONCE, in `agents/shot_list_agent.py`'s own
    assembly step, before a shot ever reaches this node; every downstream
    node's `{**shot, ...}` spread already preserves arbitrary extra keys
    unvalidated, exactly the same "costs nothing" posture video_gen_node.py's
    "KNOWN DEPARTURES" #3 already established for GeneratedShot's
    resolution_used/duration_sec_used/budget_clamped). It resets to 0 the
    moment a pass's identity check comes back `same_object=true` again, so a
    much LATER, unrelated hard failure is never mistaken for a second
    consecutive one.
  * A shot that never hard-fails identity (the overwhelming common case: no
    `identity_check` on the entry, or `same_object=true`) falls straight
    through to the existing drift-threshold branch below, completely
    untouched by any of this -- this is why every pre-existing drift/retry/
    interrupt test in this suite needed zero changes for this fix.

THE RETRY CYCLE (wired in graph/build.py). After this node,
`route_after_continuity_gate` sends the run back to `video_gen` iff any shot
still needs a pass -- "pending" (an auto-retry or a human retry_with_edit) OR
"fallback_requested" (a human accept_fallback, which must reach Ken-Burns) --
else to END. That closes the loop `video_gen -> ken_burns_fallback ->
continuity_agent -> continuity_gate -> (loop | END)`, which is exactly why Phase 4
wires the graph here rather than leaving this standalone: a retry loop is a
graph-topology feature that cannot be meaningfully exercised outside the compiled
graph. See `route_after_continuity_gate` for why fallback_requested must loop.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

from langchain_core.runnables import RunnableConfig

# DRIFT_THRESHOLD is owned by the scorer (single source of truth). SUCCESS_STATUS
# ("passed") and FALLBACK_REQUESTED_STATUS ("fallback_requested") are reused from
# Video-Gen so the frozen status spellings can't drift between modules.
from agents.continuity_agent import DRIFT_THRESHOLD
from agents.video_gen_node import FALLBACK_REQUESTED_STATUS, SUCCESS_STATUS
from graph.state import HumanReviewEntry, ProductCutState, Shot

logger = logging.getLogger("productcut.agents.continuity_gate")

# §5.10's "retry_count < 2" automatic-retry cap. Env-overridable (budget_gate.py
# DEFAULT_JOB_BUDGET_CAP pattern) -- bounds UNSUPERVISED regeneration spend.
MAX_AUTO_RETRIES = int(os.getenv("CONTINUITY_MAX_AUTO_RETRIES", "3"))

# Existing frozen Shot.status values (graph/state.py). "pending" is what
# Video-Gen's filter treats as "(re-)generate"; "review" is the previously-unused
# human-review-pending value this phase finally uses.
PENDING_STATUS = "pending"
REVIEW_STATUS = "review"

# Durable "a human accepted this clip's drift" marker, written onto the clip's
# (un-validated, extra-keys-allowed -- see video_gen_node.py "KNOWN DEPARTURES" #3)
# generated_shots entry when an over-threshold shot is human-APPROVED. The Gate
# skips any entry carrying it, so a later retry-loop pass (driven by a different
# shot) never re-surfaces an already-approved shot for review. Binds to the CLIP
# (the entry), not the Shot: it is exactly this rendered clip's drift the human
# accepted, and if the clip were ever regenerated its entry (and this marker) is
# overwritten, correctly forcing a fresh decision.
CONTINUITY_APPROVED_KEY = "continuity_approved"

# v8 fix: extra, undeclared Shot key tracking CONSECUTIVE same_object=false
# identity verdicts for a given shot across regeneration passes -- see module
# docstring's "HARD IDENTITY FAILURE" section for why this is safe (Shot is
# never re-validated after agents/shot_list_agent.py's own assembly step) and
# why it must exist (a fresh GeneratedShot entry carries no memory of a prior
# pass's identity_check once the clip is regenerated, so the Shot itself is the
# only place a cross-pass streak can live without a C1 schema change).
IDENTITY_HARD_FAIL_STREAK_KEY = "identity_hard_fail_streak"

# Resume-value resolutions -- mirror HumanReviewEntry.resolution's Literal.
RESOLUTION_APPROVE = "approve"
RESOLUTION_RETRY_WITH_EDIT = "retry_with_edit"
RESOLUTION_ACCEPT_FALLBACK = "accept_fallback"


def _apply_resolution(shot: Shot, resolution) -> Shot:
    """Turn a human resume value into the resolved shot (see module docstring).

    Accepts the documented `{"resolution": <literal>}` dict or a bare string.
    An unknown/malformed value leaves the shot in "review" (never silently
    passed).
    """
    res = resolution.get("resolution") if isinstance(resolution, dict) else resolution
    shot_id = shot["shot_id"]

    if res == RESOLUTION_APPROVE:
        logger.info("Continuity Gate: shot %s human-APPROVED as-is -> passed.", shot_id)
        return {**shot, "status": SUCCESS_STATUS}
    if res == RESOLUTION_RETRY_WITH_EDIT:
        # Human-authorized retry -- NOT capped by MAX_AUTO_RETRIES (see docstring).
        logger.info(
            "Continuity Gate: shot %s human-RETRY_WITH_EDIT -> pending (uncapped, "
            "human-authorized).", shot_id,
        )
        return {**shot, "status": PENDING_STATUS, "retry_count": shot.get("retry_count", 0) + 1}
    if res == RESOLUTION_ACCEPT_FALLBACK:
        logger.info(
            "Continuity Gate: shot %s human-ACCEPT_FALLBACK -> fallback_requested "
            "(routes through Ken-Burns next pass).", shot_id,
        )
        return {**shot, "status": FALLBACK_REQUESTED_STATUS}

    logger.warning(
        "Continuity Gate: shot %s got an unknown resume resolution %r -- leaving in "
        "'review' (still needs handling, not silently passed).", shot_id, res,
    )
    return {**shot, "status": REVIEW_STATUS}


# ---------------------------------------------------------------------------
# LangGraph node wrapper (fully autonomous -- no interrupt / human review).
# ---------------------------------------------------------------------------
async def continuity_gate_node(
    state: ProductCutState,
    config: Optional[RunnableConfig] = None,
) -> dict:
    """LangGraph node: apply the drift decision to every scored `passed` shot.

    Fully autonomous -- there is NO human-in-the-loop `interrupt()`. Over-threshold
    shots are auto-retried up to MAX_AUTO_RETRIES times; once those retries are
    exhausted the best-available generated clip is AUTO-ACCEPTED as-is (stamped
    with CONTINUITY_APPROVED_KEY, kept `status="passed"`), never routed to the
    Ken-Burns static-image fallback. (Ken-Burns remains reserved for hard infra
    failures raised by Video-Gen itself, and for the two-consecutive hard-identity
    failure path below.)

    Returns updated `shot_list` (statuses/retry_counts patched), the (unchanged)
    `human_review_queue` passed through, and `reasoning_trace`.
    """
    shots = state.get("shot_list", [])
    generated = state.get("generated_shots", {})
    # Rebuilt from the ENTRY state on every (re-)execution -- overwrite-semantics
    # field, so re-running on resume reproduces the same queue rather than
    # doubling it.
    review_queue: list[HumanReviewEntry] = list(state.get("human_review_queue", []))

    updated_shots: list[Shot] = []
    # generated_shots entries this node stamps (currently only the durable
    # `continuity_approved` marker on a human-approved clip). Merged into the
    # returned generated_shots; empty on any pass with no approvals, so this is a
    # no-op for the common path.
    generated_updates: dict = {}
    n_retry = 0
    n_review = 0
    n_within = 0
    n_identity_retry = 0
    n_identity_fallback = 0

    for shot in shots:
        shot_id = shot["shot_id"]
        entry = generated.get(shot_id)
        # Only decide on real, scored clips. Everything else (fallback,
        # fallback_requested, pending, unscored) passes through untouched -- same
        # object -- exactly like ken_burns_fallback_node's non-matching shots.
        if shot.get("status") != SUCCESS_STATUS or entry is None or "drift_score" not in entry:
            updated_shots.append(shot)
            continue

        # A clip a human already ACCEPTED (over threshold, but approved) must not
        # be re-surfaced for review on a later loop pass driven by another shot --
        # see CONTINUITY_APPROVED_KEY. Leave it exactly as-is (stays "passed").
        if entry.get(CONTINUITY_APPROVED_KEY):
            updated_shots.append(shot)
            n_within += 1
            continue

        # v8 fix -- HARD IDENTITY FAILURE, a separate, categorical check ahead of
        # the continuous drift-threshold decision below (see module docstring).
        identity = entry.get("identity_check")
        if identity and identity.get("same_object") is True and shot.get(IDENTITY_HARD_FAIL_STREAK_KEY):
            # A genuine same_object=true clears any stale streak from an earlier
            # round so a much LATER, unrelated hard failure is never mistaken
            # for a second consecutive one.
            shot = {**shot, IDENTITY_HARD_FAIL_STREAK_KEY: 0}

        if identity and identity.get("same_object") is False:
            streak = shot.get(IDENTITY_HARD_FAIL_STREAK_KEY, 0) + 1
            if streak >= 2:
                updated_shots.append(
                    {**shot, "status": FALLBACK_REQUESTED_STATUS, IDENTITY_HARD_FAIL_STREAK_KEY: streak}
                )
                n_identity_fallback += 1
                logger.warning(
                    "Continuity Gate: shot %s failed the IDENTITY check TWICE in a "
                    "row (same_object=false, confidence=%s) -- routing straight to "
                    "Ken-Burns fallback instead of exhausting the normal drift "
                    "retry/review budget on an unfixable prompt/reference combo.",
                    shot_id, identity.get("confidence"),
                )
            else:
                updated_shots.append(
                    {**shot, "status": PENDING_STATUS, "retry_count": shot.get("retry_count", 0) + 1,
                     IDENTITY_HARD_FAIL_STREAK_KEY: streak}
                )
                n_identity_retry += 1
                logger.warning(
                    "Continuity Gate: shot %s failed the IDENTITY check (same_object="
                    "false, confidence=%s) -- distinct from a normal drift retry: one "
                    "automatic re-sample allowed (not counted against MAX_AUTO_RETRIES) "
                    "-> pending.",
                    shot_id, identity.get("confidence"),
                )
            continue

        drift = float(entry["drift_score"])
        if drift <= DRIFT_THRESHOLD:
            updated_shots.append(shot)  # within threshold -- leave passed
            n_within += 1
            continue

        retry_count = shot.get("retry_count", 0)
        if retry_count < MAX_AUTO_RETRIES:
            # Automatic capped drift retry (the hard-identity-failure branch
            # above is this node's only OTHER retry_count-incrementing path).
            updated_shots.append({**shot, "status": PENDING_STATUS, "retry_count": retry_count + 1})
            n_retry += 1
            logger.info(
                "Continuity Gate: shot %s drift %.3f > %.3f, retry %d/%d -> pending.",
                shot_id, drift, DRIFT_THRESHOLD, retry_count + 1, MAX_AUTO_RETRIES,
            )
            continue

        # Retries exhausted -> auto-accept the best available clip.
        # The pipeline is fully autonomous; Ken Burns is only for hard infra failures.
        # Stamp the clip's entry with CONTINUITY_APPROVED_KEY so later loop passes
        # (driven by another shot) don't re-evaluate this still-over-threshold clip,
        # and keep status="passed" so it flows straight to Assembly.
        updated_shots.append({**shot, "status": SUCCESS_STATUS})
        generated_updates[shot_id] = {**entry, CONTINUITY_APPROVED_KEY: True}
        n_review += 1
        logger.warning(
            "Continuity Gate: shot %s drift %.3f > %.3f, retries exhausted "
            "(retry_count=%d) -> auto-accepting best available clip.",
            shot_id, drift, DRIFT_THRESHOLD, retry_count,
        )

    trace_note = (
        f"\n[continuity_gate] {n_within} shot(s) within drift threshold; "
        f"{n_retry} auto-retried; {n_review} auto-accepted (retries exhausted); "
        f"{n_identity_retry} auto-retried for a HARD IDENTITY failure; "
        f"{n_identity_fallback} routed straight to Ken-Burns after a second "
        "consecutive identity failure."
    )
    return {
        "shot_list": updated_shots,
        # Merge any approval markers INTO existing generated_shots (overwrite-
        # semantics field) so no other shot's entry is clobbered; a no-op merge
        # when generated_updates is empty (every non-approval pass).
        "generated_shots": {**generated, **generated_updates},
        "human_review_queue": review_queue,
        "reasoning_trace": state.get("reasoning_trace", "") + trace_note,
    }


# ---------------------------------------------------------------------------
# Conditional-edge router: loop back to Video-Gen for retrying shots, else END.
# ---------------------------------------------------------------------------
def route_after_continuity_gate(state: ProductCutState) -> str:
    """Route the compiled graph after the Gate: back to `video_gen` iff any shot
    still needs another loop pass, else `end`.

    "Needs another pass" means status `"pending"` (an automatic retry or a human
    `retry_with_edit` -- Video-Gen must (re-)generate it) OR `"fallback_requested"`
    (a human `accept_fallback`, or any prior hand-off -- Ken-Burns must render it).

    DELIBERATE SUPERSET of the §5.10 routing snippet (which names only "pending").
    The `accept_fallback` resolution sets a shot to `"fallback_requested"`, and the
    only node that turns that into a real `"fallback"` clip is
    ken_burns_fallback_node -- which sits UPSTREAM of Continuity on the loop
    (`video_gen -> ken_burns_fallback -> continuity_agent -> continuity_gate`). So
    a `fallback_requested` shot can ONLY reach Ken-Burns by looping back to
    `video_gen` (which passes it straight through its "pending"-only fan-out
    filter, untouched) and on to Ken-Burns. Routing only on "pending" would send an
    `accept_fallback` shot to END with no clip -- breaking the very contract that
    resolution promises ("flows back through ken_burns_fallback_node"). Including
    `fallback_requested` here is what makes `accept_fallback` actually work, and it
    terminates cleanly: once Ken-Burns renders the shot it becomes `"fallback"`
    (neither pending nor fallback_requested), so the next Gate pass routes to END.

    Reads the just-returned `shot_list`, same primitive as
    agents/merge_validator.py's `route_after_merge_validation`. The build.py edge
    map wires "video_gen" -> the video_gen node and "end" -> END.
    """
    shots = state.get("shot_list", [])
    if any(s.get("status") in (PENDING_STATUS, FALLBACK_REQUESTED_STATUS) for s in shots):
        return "video_gen"
    return "end"


__all__ = [
    "MAX_AUTO_RETRIES",
    "CONTINUITY_APPROVED_KEY",
    "IDENTITY_HARD_FAIL_STREAK_KEY",
    "RESOLUTION_APPROVE",
    "RESOLUTION_RETRY_WITH_EDIT",
    "RESOLUTION_ACCEPT_FALLBACK",
    "continuity_gate_node",
    "route_after_continuity_gate",
]
