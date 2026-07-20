"""
Budget Gate — deterministic cost-cap enforcement (Phase 2).
Spec of record: docs/TECHNICAL_DOCUMENTATION.md §5.7.

Enforces the hard cost cap BEFORE any money is spent on video generation — the
"quality under a limited budget" guarantee made concrete. It runs after the
Shot-List Agent (§5.6), which emits every shot's `allocated_budget` as an
explicit `0.0` PLACEHOLDER (see the assembly-site comment in
`agents/shot_list_agent.py`); this module is what computes and overwrites those
placeholders with real, grounding-weighted dollar allocations.

NO LLM ANYWHERE IN THIS MODULE. This is a pure, deterministic computation over
already-validated content — the "Reduce is deterministic, not generative"
decision settled in §5.7. Concretely, the reduce path:
  * NEVER calls an LLM and NEVER re-invokes the Shot-List Agent — cutting a shot
    is a plain list removal on content that already passed the Justification
    Validator, so nothing new is generated and nothing new needs re-validating
    (the reduced list is always a strict subset of already-validated shots).
  * ONLY cuts, never merges — merging two shots into one coherent shot would be
    a genuine creative operation and is explicitly OUT of scope for now (§5.7).

`allocated_budget` semantics (§5.7). It is REAL generation cost,
`duration_sec × rate(resolution)`, not an abstract proxy — Wan pricing is flat
per-second-by-resolution, so a real-dollar ledger makes both the hard cap and the
live dashboard ledger literal rather than illustrative. Deliberately there is NO
separate `resolution_tier` / `retry_reserve` field on the Shot (§5.7 design
decision — avoid schema churn): the single `allocated_budget` figure implicitly
encodes both "how much this shot may spend" and "does it clear the 1080p ceiling".
A future Video-Gen Node derives resolution/retry affordability by comparing
`allocated_budget` against `duration_sec × RATE_1080P`, not by reading a
categorical field.

Allocation reuses meta_critic's clamp-and-redistribute `_waterfill()` — the same
algorithm family, second use in this codebase (§5.7 step 4 calls for exactly this
reuse). Its "spread the residual evenly when even the floor doesn't fit" fallback
IS the spec's "uniform trim, last resort" behavior — we get it for free by reusing
`_waterfill` correctly, so there is no separate uniform-trim code here.

KNOWN GAP (same posture as concept_agent.py's `target_length_sec` gap). C1 has NO
job-level budget-cap field: `ProductCutState` carries no `budget_cap`, and while
`BudgetLedger.cap` exists, nothing upstream currently populates it. Rather than
unilaterally invent a permanent-feeling new required C1 field, the node reads
`state["budget_ledger"]["cap"]` first (in case something later sets it) and
otherwise defaults to `DEFAULT_JOB_BUDGET_CAP` (env-overridable). Raise with the
team if a real per-job cap belongs in the frozen schema.

Scope note — what this is NOT (identical posture to body_checker.py /
shot_list_agent.py): WIRED into the live LangGraph graph
(backend/graph/build.py): `shot_list_agent -> budget_gate -> video_gen`. Was a
standalone, independently-callable/testable node before the Shot-List Agent
it consumes was wired in; that follow-up wiring has since landed.
"""
from __future__ import annotations

import logging
import os
from typing import NamedTuple, Optional

from langchain_core.callbacks.manager import adispatch_custom_event
from langchain_core.runnables import RunnableConfig

# Reuse (never reimplement) the existing clamp-and-redistribute water-filling
# routine — §5.7 step 4 explicitly calls for a second use of this exact function.
from agents.meta_critic import _waterfill
# Reuse the Shot-List Agent's own constants so they cannot silently drift: the
# 3-7 shot contract's floor (MIN_SHOTS), the per-shot minimum duration that
# defines the cheapest-possible 720p render (§5.6), and the structural hero-
# identification predicate -- see `_shot_floor_cost` below for why the hero
# needs its OWN floor rather than the flat per-shot FLOOR_COST. NOTE: this
# module deliberately does NOT import HERO_SHOT_MIN_DURATION_SEC (anymore) --
# the Backstory-First fix (video-gen-fidelity, 2026-07-11) made the hero
# floor derive from each shot's own (already target-length-scaled)
# duration_sec instead of that flat constant; see `_shot_floor_cost`.
from agents.shot_list_agent import (
    MIN_SHOTS,
    MIN_SHOT_DURATION_SEC,
    is_hero_shot,
)
from graph.state import BudgetLedger, ProductCutState, ProductTruth, Shot

logger = logging.getLogger("productcut.agents.budget_gate")

# Wan 2.6 I2V flat per-second rates by resolution (Phase 2 research finding).
# Treat as approximate — verify against the real DashScope console later. Camera
# move / shot complexity do NOT change cost (pricing is flat per second by
# resolution), which is exactly why a real-dollar ledger is accurate (§5.7).
RATE_720P = 0.08   # $/sec @ 720p
RATE_1080P = 0.12  # $/sec @ 1080p

# Per-shot floor: the cheapest a shot can be rendered — MIN_SHOT_DURATION_SEC (3s)
# at 720p. A constant across shots (§5.7): it is the "trim this shot to the minimum
# and render it cheapest" lower bound, not the shot's own current 720p cost.
FLOOR_COST = MIN_SHOT_DURATION_SEC * RATE_720P  # 3.0 * 0.08 = 0.24

# KNOWN GAP default (see module docstring). A sensible whole-job cap for a
# ~15-30s, 3-7 shot ad; env-overridable. NOT a permanent C1 field — flagged, not
# silently invented.
DEFAULT_JOB_BUDGET_CAP = float(os.getenv("DEFAULT_JOB_BUDGET_CAP", "2.00"))

# --- Grounding-weight table (§5.7 allocation formula) ----------------------
# Already-researched defaults; kept as inspectable module-level dicts so the
# weights are the single tunable source of truth. Only the RATIOS between weights
# matter — the allocation is normalized to the cap regardless of absolute scale.
#
# w_role favors attention (hook) and conversion (cta); w_type favors the
# specificity-carriers (macro_detail / hook_hero) AND, since the video-gen-
# fidelity creative-direction redesign, human-interaction shots
# (product_in_hand / worn_in_use) comparably — a real person using the product
# is now a deliberate, load-bearing part of the ad's story, not an afterthought
# the original research pass treated as generic filler; truth_bonus rewards a
# shot that cites a truth in one of the four "specific" categories — the facts
# that make the product SPECIFIC rather than generic, which is the whole point.
W_ROLE: dict[str, float] = {
    "hook": 1.20,     # opening attention — the scarcest resource
    "cta": 1.20,      # the conversion ask
    "proof": 1.15,    # evidence that earns the claim
    "demo": 1.00,     # baseline
    "problem": 0.90,  # setup; cheaper to render generically
}
W_TYPE: dict[str, float] = {
    "macro_detail": 1.30,       # the extreme close-up that proves texture/construction
    "hook_hero": 1.15,          # the identity-forward opener
    # product_in_hand is new in this project's Phase 2 C3 addition and was not
    # covered by the original research pass. Weighted alongside hook_hero since it
    # is usually a meaningful demo/proof composition (a real human-interaction
    # shot), not the extreme close-up macro_detail is. Reasoned default, tune later.
    "product_in_hand": 1.15,
    # worn_in_use is the video-gen-fidelity redesign's C3 v4 addition (the wider,
    # person-in-motion human-interaction composition) — weighted identically to
    # product_in_hand, deliberately, not left to the 1.0 unknown-type fallback:
    # both are the human-story payoff shot the redesign exists to protect, they
    # just differ in framing/motion, not in narrative value.
    "worn_in_use": 1.15,
    "cta_endcard": 1.10,        # the closing card
    "hero_reframe": 1.00,       # baseline
    # Raised from the pre-redesign 0.90 ("the generic establishing/context
    # shot" — a framing that predates the video-gen-fidelity creative-direction
    # redesign and is now outdated: lifestyle_context is a real narrative
    # composition, not filler). 1.10 sits between hero_reframe's 1.00 baseline
    # and product_in_hand/worn_in_use's 1.15 — competitive with the rest of the
    # table rather than the table's structural bottom, which previously made a
    # human/lifestyle shot the first thing cut under a tight cap regardless of
    # how load-bearing it was to the story.
    "lifestyle_context": 1.10,
}
TRUTH_BONUS = 1.10  # applied when the cited truth is one of the "specific" categories below

# Shot-type composition set naming the human-interaction shots the reduce path
# below protects from being the first thing cut under a tight cap (§5.7 reduce,
# extended for the video-gen-fidelity redesign). MUST exactly match
# agents/video_gen_node.py's own `_HUMAN_INTERACTION_SHOT_TYPES` — kept as a
# separate local constant rather than imported, since video_gen_node.py already
# imports RATE_720P/RATE_1080P FROM this module, and importing back the other
# direction would create a circular dependency. If this set changes, update
# that one too.
HUMAN_INTERACTION_SHOT_TYPES = frozenset({"product_in_hand", "worn_in_use"})

# The ProductTruth categories that make a product SPECIFIC, not generic — the
# facts you cannot guess without actually looking at the photos (§5.7).
# "imperfection" deliberately excluded (Positive-Only Truths fix,
# docs/BUILD_TASKS.md "Script Quality (CTA Bridge) + Positive-Only Truths..."
# workstream, Problem 1): a scratch/wear-citing shot must not get a budget
# priority bump over a color/style/size shot — the whole point of that fix is
# that a flaw is no longer a preferred grounding detail by default.
SPECIFIC_TRUTH_CATEGORIES = frozenset(
    {"material", "texture", "construction_detail"}
)

_EPS = 1e-9


class BudgetResult(NamedTuple):
    """The Budget Gate's product.

    A NamedTuple (also a plain tuple) so the node wrapper can build the
    `budget_updated` event payload from `over_cap`, and surface `overage` in the
    trace, without a second pass over the ledger. `overage` is 0.0 unless
    `over_cap` is True (the §5.7 floor case).
    """

    shots: list[Shot]        # updated shots with real allocated_budget (cut shots removed)
    ledger: BudgetLedger     # {cap, spent, per_shot}
    over_cap: bool           # True only in the §5.7 floor case (can't fit even at MIN_SHOTS)
    overage: float           # dollars over cap when over_cap; else 0.0


# ---------------------------------------------------------------------------
# Weighting (§5.7 allocation formula, w_i = w_role * w_type * truth_bonus).
# ---------------------------------------------------------------------------
def _shot_weight(shot: Shot, truths_by_id: dict[str, ProductTruth]) -> float:
    """Grounding weight for one shot (§5.7): role × type × truth-specificity bonus.

    The truth bonus looks up the shot's cited truth via
    `justification.truth_fact_id` and applies only when that truth's category is
    one of the four "specific" categories — an unknown/missing truth id or a
    generic category (color / scale_cue / brief_or_intake_fact) gets no bonus.
    Unknown roles/types fall back to 1.0 (neutral) rather than raising, so a
    still-valid-but-unexpected enum value can never crash the budget pass.
    """
    role_w = W_ROLE.get(shot["beat_role"], 1.0)
    type_w = W_TYPE.get(shot["shot_type"], 1.0)
    truth_id = shot.get("justification", {}).get("truth_fact_id", "")
    category = truths_by_id.get(truth_id, {}).get("category")
    truth_w = TRUTH_BONUS if category in SPECIFIC_TRUTH_CATEGORIES else 1.0
    return role_w * type_w * truth_w


def _shot_floor_cost(shot: Shot) -> float:
    """Per-shot floor for the waterfill window's lower bound AND for the §5.7
    floor case's `final_alloc = list(lo)` (video-gen-fidelity story-arc fix).

    FLOOR_COST (MIN_SHOT_DURATION_SEC @ 720p, ~3s) is the right floor for
    every ORDINARY shot. It is the WRONG floor for the hero shot (see
    agents/shot_list_agent.py's HERO SHOT mechanism note, `is_hero_shot`):
    that constant assumes a render length no hero shot is ever clamped to, so
    using it as the hero's floor would let the §5.7 floor case (cap can't fit
    even at floor -> every shot pinned to its floor) silently clamp a
    15s-planned hero shot down to a ~3s render -- defeating the entire reason
    the extended duration ceiling exists, exactly the "silently clamped back
    down" failure mode this fix was asked to prevent (see
    agents/video_gen_node.py's `_resolve_generation_params`, which derives the
    ACTUAL generated duration from allocated_budget / RATE_720P -- an
    under-floored hero would generate at a few seconds regardless of what
    shot_list_agent.py planned).

    The hero's floor is THIS SHOT's own `duration_sec` @ 720p -- NOT the flat
    HERO_SHOT_MIN_DURATION_SEC constant (Backstory-First fix,
    video-gen-fidelity, 2026-07-11). The hero window itself now SCALES to the
    ad's target length (`agents.shot_list_agent._scaled_hero_window`) -- a
    15s-target ad's hero can be clamped to ~5-7.5s, well under the flat 10s
    constant. Using that flat constant as the floor would then set floor
    ($0.80 @ 10s*720p) ABOVE the ceiling (`duration_sec * RATE_1080P`, e.g.
    $0.60 @ 5s*1080p) -- an inverted, infeasible window on every 15s-target
    ad. Deriving the floor from the shot's own (already target-scaled)
    duration_sec instead guarantees floor < ceiling always (RATE_720P <
    RATE_1080P on the same duration basis) at ANY target ad length, while
    still being "the cheapest resolution tier for what this hero was actually
    planned to render" -- the same semantics the flat constant approximated
    for the 30s case it was originally sized for.
    """
    if is_hero_shot(shot):
        return shot["duration_sec"] * RATE_720P
    return FLOOR_COST


def _argmin(values: list[float]) -> int:
    """Index of the smallest value (first on ties → fully deterministic)."""
    return min(range(len(values)), key=lambda i: values[i])


def _choose_drop_index(working: list[Shot], weights: list[float]) -> int:
    """Pick which shot the deterministic cut-only reduce (§5.7) removes next.

    Normally this is just the lowest-weight shot (`_argmin`). But the
    video-gen-fidelity creative-direction redesign added a deliberate priority
    override: if the lowest-weight shot is the SOLE remaining human-interaction
    shot (`HUMAN_INTERACTION_SHOT_TYPES` — product_in_hand / worn_in_use) in
    `working`, cutting it would silently erase the ad's entire human-usage
    story beat. This is not hypothetical — a real live pipeline run caught
    exactly this: the human-usage shot was the cheapest-weighted survivor and
    got cut first under the default $2.00 cap, even though the Shot-List
    Agent had correctly written it and the affordance rubric had correctly
    motivated it. Re-weighting `lifestyle_context`/`worn_in_use` upward (see
    W_TYPE above) reduces how OFTEN this happens, but does not guarantee it —
    a cheap short duration or a `problem`/`demo` beat_role can still leave a
    human shot the argmin under a tight enough cap, so this is a real,
    deterministic floor protection, not just a weighting nudge. Mirrors the
    existing MIN_SHOTS floor protection's spirit (never let a mechanical cut
    silently erase something structurally load-bearing) without touching that
    floor logic at all — this only changes WHICH shot gets cut, never whether
    one does, and never overrides the MIN_SHOTS floor case itself.

    If the argmin shot is NOT the sole human-interaction shot in `working`
    (there is another one left, or it isn't human-interaction-typed at all),
    behavior is exactly the pre-existing plain argmin — no regression for the
    common case.

    Pathological edge case: if EVERY remaining shot in `working` is human-
    interaction-typed, there is no non-human shot to redirect the cut to.
    Falls through to plain argmin rather than refuse to reduce or hang — a cut
    still has to happen for the cap to ever be met above the MIN_SHOTS floor,
    and this case is rare/pathological (it needs almost the whole shot list to
    already be human-interaction-typed), not a scenario worth blocking on.
    """
    drop_index = _argmin(weights)
    if working[drop_index].get("shot_type") not in HUMAN_INTERACTION_SHOT_TYPES:
        return drop_index

    human_indices = [
        i for i, s in enumerate(working) if s.get("shot_type") in HUMAN_INTERACTION_SHOT_TYPES
    ]
    if len(human_indices) > 1:
        return drop_index  # another human-interaction shot survives either way

    non_human_indices = [i for i in range(len(working)) if i not in human_indices]
    if not non_human_indices:
        return drop_index  # pathological: nothing non-human left to redirect to

    return min(non_human_indices, key=lambda i: weights[i])


# ---------------------------------------------------------------------------
# Core allocation (§5.7) — pure, deterministic, no LLM.
# ---------------------------------------------------------------------------
def allocate_budget(
    shots: list[Shot],
    product_truths: list[ProductTruth],
    cap: float,
) -> BudgetResult:
    """Grounding-weighted, cap-normalized per-shot allocation (§5.7).

    For each shot: `base_i = duration_sec_i × RATE_720P`, `w_i = _shot_weight(shot)`,
    and a proportional target normalized to the cap `alloc_i = (base_i·w_i)·(cap/Σ)`.
    Targets are then clamped to each shot's feasible window
    `[_shot_floor_cost(shot), duration_sec_i × RATE_1080P]` (FLOOR_COST for an
    ordinary shot; this shot's own duration_sec @ 720p for the hero shot --
    video-gen-fidelity story-arc fix, see `_shot_floor_cost`'s own docstring)
    and the clamping remainder is redistributed by the reused `_waterfill()`
    routine.

    Over-cap reduce path (deterministic, cut-only — §5.7): if `_waterfill` reports
    the cap cannot be met inside every window, the single LOWEST-WEIGHT shot is cut
    and the WHOLE computation retried from scratch on the smaller list. Because
    hook/cta carry the highest role weight, the lowest-weight argmin is never the
    hook/cta/top-weighted shot — the §5.7 "never cut the hook/cta/top proof"
    protection falls out of the weighting for free.

    Human-interaction protection (video-gen-fidelity redesign, see
    `_choose_drop_index`): WHICH shot the argmin picks is further overridden
    when it would cut the SOLE remaining human-interaction shot
    (product_in_hand / worn_in_use) — the next-lowest-weight NON-human shot is
    cut instead. This is a real, deliberate priority override discovered from a
    real live pipeline run (the human-usage story beat was consistently the
    cheapest survivor and got cut first under the default cap), not merely a
    side-effect of the W_TYPE re-weighting above.

    Floor case (§5.7 step 4): once the list is down to MIN_SHOTS (3) and the cap
    STILL cannot fit, there is nothing left to cut without breaking the 3-shot
    contract. Every shot is set to its cheapest resolution (FLOOR_COST) and the
    over-cap total is ACCEPTED and FLAGGED (`over_cap=True`, non-zero `overage`)
    rather than silently pretended away — a visible overage beats a hidden one.

    NOTE on `_waterfill` reuse for the floor case: `_waterfill`'s own infeasible
    fallback would spread the residual evenly, pushing shots BELOW the floor so the
    sum hits the (too-small) cap exactly. That is dishonest for a budget ledger — a
    shot cannot actually render below its floor. So the floor case reports each
    shot at FLOOR_COST (the true cheapest realizable spend) and derives the overage
    as `Σ(floor) − cap`, which is precisely the "non-zero overage" §5.7 requires.

    Args:
        shots:          the shot list from the Shot-List Agent (allocated_budget 0.0).
        product_truths: the job's grounded facts, for the truth-specificity bonus.
        cap:            the job's hard dollar cost cap.

    Returns:
        BudgetResult(shots, ledger, over_cap, overage). `shots` are NEW dicts (the
        caller's list and dicts are never mutated in place) with real
        `allocated_budget`; any shot cut in the reduce path is absent from both
        `shots` and `ledger["per_shot"]`.
    """
    if not shots:
        # Defensive: nothing to allocate. Not expected in Phase 2 (the Shot-List
        # Agent yields 3-7 shots) but keeps the function total.
        return BudgetResult([], {"cap": cap, "spent": 0.0, "per_shot": {}}, False, 0.0)

    truths_by_id = {t["truth_id"]: t for t in product_truths}
    working = list(shots)  # copy the list — never mutate the caller's list in place

    over_cap = False
    overage = 0.0
    while True:
        n = len(working)
        base = [s["duration_sec"] * RATE_720P for s in working]
        weights = [_shot_weight(s, truths_by_id) for s in working]
        raw = [b * w for b, w in zip(base, weights)]
        total_raw = sum(raw)

        # Proportional target per shot, normalized toward the cap (§5.7 step 3).
        if total_raw > _EPS:
            targets = [r * (cap / total_raw) for r in raw]
        else:  # defensive — durations>0 and weights>0 make this unreachable in practice
            targets = [cap / n] * n

        # Per-shot floor (video-gen-fidelity story-arc fix): FLOOR_COST for
        # every ordinary shot, this shot's own duration_sec @ 720p for the
        # hero -- see `_shot_floor_cost`'s own docstring for why a flat floor
        # would silently starve a hero shot's real generation duration.
        lo = [_shot_floor_cost(s) for s in working]
        hi = [s["duration_sec"] * RATE_1080P for s in working]  # ceiling: THIS shot @ 1080p

        # Fast path: cap already covers every shot at 1080p — no cuts needed.
        # The dynamic cap (sum × 1.20) always exceeds sum(hi) (sum × 1.0), so
        # without this the waterfill always reports infeasible (residual > 0
        # because budget > sum(hi)) and triggers unnecessary shot cuts.
        if cap >= sum(hi) - _EPS:
            final_alloc = hi
            break

        allocations, infeasible = _waterfill(targets, list(zip(lo, hi)), cap)

        if not infeasible:
            # Success: the cap fits inside every window. `allocations` sum to the
            # cap and each is within [floor, own-1080p-ceiling].
            final_alloc = allocations
            break

        if n <= MIN_SHOTS:
            # Floor case (§5.7 step 4): can't cut further without breaking the
            # 3-shot contract. Report the honest cheapest-possible spend and flag.
            final_alloc = list(lo)
            spent_floor = sum(final_alloc)
            over_cap = spent_floor > cap + _EPS
            overage = max(0.0, spent_floor - cap)
            if over_cap:
                logger.info(
                    "Budget Gate: floor case at %d shot(s) — cap $%.4f cannot fit even "
                    "cheapest render ($%.4f); accepting over_cap overage $%.4f (§5.7).",
                    n, cap, spent_floor, overage,
                )
            break

        # Deterministic cut-only reduce (§5.7): drop the single lowest-weight shot
        # and retry the whole computation from scratch on the smaller list. No LLM,
        # no Shot-List re-invocation — a plain removal of already-validated content.
        # `_choose_drop_index` (not plain `_argmin`) additionally protects the sole
        # remaining human-interaction shot from being that cut — see its docstring.
        drop_index = _choose_drop_index(working, weights)
        dropped = working.pop(drop_index)
        logger.info(
            "Budget Gate: over cap with %d shots — cutting lowest-weight shot %s "
            "(weight %.4f) and retrying (§5.7).",
            n, dropped.get("shot_id"), weights[drop_index],
        )

    # Assign allocations onto NEW shot dicts (never mutate the caller's) and build
    # the ledger. `spent` is summed from `updated_shots` directly (the authoritative
    # per-shot allocation), NOT from `per_shot.values()`: `per_shot` is a dict keyed
    # by `shot_id`, which silently collapses two shots sharing the same id into one
    # entry (an upstream-invariant violation this module doesn't control -- the
    # Shot-List Agent is expected to emit unique shot_ids, same implicit uniqueness
    # every other id-keyed structure in this codebase assumes). Deriving `spent`
    # from `per_shot` in that case would under-report true committed spend by a
    # full shot's allocation -- a real ledger-integrity bug caught by an
    # independent adversarial test pass. Summing `updated_shots` directly keeps
    # `spent` correct regardless of any such collision, even though `per_shot`'s
    # breakdown itself still can't represent two allocations under one key.
    updated_shots: list[Shot] = []
    per_shot: dict[str, float] = {}
    for shot, alloc in zip(working, final_alloc):
        rounded = round(float(alloc), 6)
        updated_shots.append({**shot, "allocated_budget": rounded})
        per_shot[shot["shot_id"]] = rounded

    ledger: BudgetLedger = {
        "cap": cap,
        "spent": sum(s["allocated_budget"] for s in updated_shots),
        "per_shot": per_shot,
    }
    return BudgetResult(updated_shots, ledger, over_cap, round(overage, 6))


# ---------------------------------------------------------------------------
# LangGraph node wrapper.
# ---------------------------------------------------------------------------
async def budget_gate_node(
    state: ProductCutState,
    config: Optional[RunnableConfig] = None,
) -> dict:
    """LangGraph node wrapper: reads shot_list/product_truths, resolves the cap,
    allocates, dispatches the C2 `budget_updated` event, and returns state updates.

    Cap resolution follows the KNOWN GAP handling (see module docstring): prefer an
    already-set `budget_ledger.cap`, else fall back to `DEFAULT_JOB_BUDGET_CAP`.

    Dispatches `budget_updated` via `adispatch_custom_event`, mirroring
    meta_critic.py's precedent (it surfaces in `astream_events` as `on_custom_event`,
    which app/main.py unwraps into a proper C2 envelope). The frozen
    `BudgetUpdatedPayload` (graph/events.py) carries `{ledger, over_cap}` only, so
    the dollar `overage` is surfaced in the reasoning trace instead of the event.

    `config` defaults to None so the node is directly callable/testable outside a
    compiled graph; LangGraph injects the real RunnableConfig either way.

    WIRED into backend/graph/build.py (`shot_list_agent -> budget_gate ->
    video_gen`); was standalone before that, same posture shot_list_agent.py
    was in before it.
    """
    shots = state.get("shot_list", [])
    product_truths = state.get("product_truths", [])

    # Cap resolution: prefer an already-set ledger cap; otherwise DERIVE the cap
    # from the actual planned shots rather than a fixed default. Guard on
    # `is not None` (a real cap of 0.0 is falsy).
    existing_ledger = state.get("budget_ledger") or {}
    cap = existing_ledger.get("cap")
    cap_source = "state.budget_ledger.cap"
    if cap is None:
        if shots:
            # Size the cap to cover all planned shots at 1080p with a 20% retry
            # buffer. This is the natural ceiling: every shot rendered at maximum
            # quality + retries, so the waterfill never makes arbitrary cuts from
            # an undersized fixed default (e.g. a 30s / 5-shot ad needs $3.60 at
            # 1080p, which the old flat $2.00 default could never fit).
            raw_cost = sum(s["duration_sec"] * RATE_1080P for s in shots)
            cap = round(raw_cost * 1.20, 4)
            cap_source = f"dynamic ({len(shots)} shots × 1080p × 1.20 retry buffer = ${cap:.4f})"
        else:
            cap = DEFAULT_JOB_BUDGET_CAP
            cap_source = "DEFAULT_JOB_BUDGET_CAP (no shots to size from)"

    result = allocate_budget(shots, product_truths, cap)

    await adispatch_custom_event(
        "budget_updated",
        {"ledger": result.ledger, "over_cap": result.over_cap},
        config=config,
    )

    n_cut = len(shots) - len(result.shots)
    trace_note = (
        f"\n[budget_gate] cap=${cap:.4f} ({cap_source}); "
        f"allocated {len(result.shots)} shot(s), spent=${result.ledger['spent']:.4f}"
    )
    if n_cut:
        trace_note += f"; cut {n_cut} lowest-weight shot(s) to fit (§5.7 deterministic reduce)"
    if result.over_cap:
        trace_note += (
            f"; OVER CAP by ${result.overage:.4f} at floor case — accepted and flagged, "
            "not hidden (§5.7 step 4)"
        )

    return {
        "shot_list": result.shots,
        "budget_ledger": result.ledger,
        "reasoning_trace": state.get("reasoning_trace", "") + trace_note,
    }


__all__ = [
    "RATE_720P",
    "RATE_1080P",
    "FLOOR_COST",
    "DEFAULT_JOB_BUDGET_CAP",
    "W_ROLE",
    "W_TYPE",
    "TRUTH_BONUS",
    "SPECIFIC_TRUTH_CATEGORIES",
    "HUMAN_INTERACTION_SHOT_TYPES",
    "BudgetResult",
    "allocate_budget",
    "budget_gate_node",
]
