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
# 3-7 shot contract's floor (MIN_SHOTS) and the per-shot minimum duration that
# defines the cheapest-possible 720p render (§5.6).
from agents.shot_list_agent import MIN_SHOTS, MIN_SHOT_DURATION_SEC
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
# specificity-carriers (macro_detail / hook_hero); truth_bonus rewards a shot
# that cites a truth in one of the four "specific" categories — the facts that
# make the product SPECIFIC rather than generic, which is the whole point.
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
    "cta_endcard": 1.10,        # the closing card
    "hero_reframe": 1.00,       # baseline
    "lifestyle_context": 0.90,  # the generic establishing/context shot
}
TRUTH_BONUS = 1.10  # applied when the cited truth is one of the "specific" categories below

# The four ProductTruth categories that make a product SPECIFIC, not generic —
# the facts you cannot guess without actually looking at the photos (§5.7).
SPECIFIC_TRUTH_CATEGORIES = frozenset(
    {"material", "texture", "construction_detail", "imperfection"}
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


def _argmin(values: list[float]) -> int:
    """Index of the smallest value (first on ties → fully deterministic)."""
    return min(range(len(values)), key=lambda i: values[i])


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
    `[FLOOR_COST, duration_sec_i × RATE_1080P]` and the clamping remainder is
    redistributed by the reused `_waterfill()` routine.

    Over-cap reduce path (deterministic, cut-only — §5.7): if `_waterfill` reports
    the cap cannot be met inside every window, the single LOWEST-WEIGHT shot is cut
    and the WHOLE computation retried from scratch on the smaller list. Because
    hook/cta carry the highest role weight, the lowest-weight argmin is never the
    hook/cta/top-weighted shot — the §5.7 "never cut the hook/cta/top proof"
    protection falls out of the weighting for free.

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

        lo = [FLOOR_COST] * n                          # constant floor: 3s @ 720p
        hi = [s["duration_sec"] * RATE_1080P for s in working]  # ceiling: THIS shot @ 1080p

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
        drop_index = _argmin(weights)
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

    # KNOWN GAP: no job-level cap field in C1 — read an existing ledger cap if set,
    # else the module default. Guard on `is not None` (a real cap of 0.0 is falsy).
    existing_ledger = state.get("budget_ledger") or {}
    cap = existing_ledger.get("cap")
    cap_source = "state.budget_ledger.cap"
    if cap is None:
        cap = DEFAULT_JOB_BUDGET_CAP
        cap_source = "DEFAULT_JOB_BUDGET_CAP (no job cap in C1 — see KNOWN GAP)"

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
    "BudgetResult",
    "allocate_budget",
    "budget_gate_node",
]
