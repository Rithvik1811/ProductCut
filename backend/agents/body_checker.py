"""
Body-Checker (§5.4.3) of the Critic Chain.

Spec of record: docs/TECHNICAL_DOCUMENTATION.md §5.4.3. This is the fifth of the
specialist checkers that score the Concept Agent's four structurally-distinct
`ScriptVariant`s (§5.3) along orthogonal axes, before the Meta-Critic (§5.4.6)
reconciles them into a cross-pollinated merge candidate. It fills the previously
undefined "Completion / structural fit" axis: nothing else in the chain scores
the *body* — the beats between the hook (`beats[0]`) and the CTA (`beats[-1]`) —
the part actually responsible for paying off the hook's promise, escalating (not
repeating) the case, carrying one throughline, and landing the declared
`emotional_trigger`.

It scores four things a creative director reads the middle of a script for:
  (1) promise-payoff match  — does the body substantively develop the SPECIFIC
      claim/pain the hook named (mechanism/evidence), not just restate it?
  (2) non-redundancy        — does each beat add new information, or repeat a
      point an earlier beat already made (even in different words)?
  (3) throughline           — one problem + one product promise carried through,
      or does a competing second benefit sneak in?
  (4) emotional-trigger fidelity — do the beats actually EARN the declared
      trigger, or is the trigger merely an asserted label?

Two-layer design, mirroring the Pacing-Checker's "mechanical where mechanical is
possible" philosophy:
  * A deterministic redundancy PRE-PASS (pure code, no LLM) flags candidate
    redundant beat pairs by content-word overlap. *Literal* repetition is
    mechanically detectable and should never be left to LLM sampling variance.
  * The LLM JUDGMENT call (via the shared `critic_llm.call_qwen_json`) is handed
    those flags as evidence it must explicitly rule on — it must not silently
    overrule an unambiguous overlap flag — and additionally catches the
    redundancy the lexical pre-pass structurally *cannot* (two beats making the
    same argument with zero shared words), plus the three non-lexical axes.

Scope note — what this is NOT (identical posture to cta_tone_checkers.py):
  * WIRED into the live LangGraph graph (backend/graph/build.py): fans out
    from concept_agent in parallel with Hook-/Pacing-/CTA-/Tone-Checker,
    fanning back in to meta_critic. Was a standalone, independently-callable/
    testable function before the Concept Agent existed; that follow-up wiring
    has since landed.
  * NOT the other checkers (Hook §5.4.1, Pacing §5.4.2, CTA §5.4.4, Tone §5.4.5,
    Meta-Critic §5.4.6, ...). Body-Checker only.
  * It does NOT populate `graph.state.CriticScore` directly — the Meta-Critic
    assembles the composite. We import the C1 `ScriptVariant` / `ProductTruth`
    types (never redefine them) for input typing only.

Index convention: `redundant_beat_pairs` are pairs of indices into the variant's
FULL `beats` array (so `[1, 2]` means `beats[1]` and `beats[2]`), matching the
Pacing-Checker's "name the exact offending beat" convention and keeping the
indices meaningful to a human reading the whole variant. Body beats are
`beats[1:-1]`, i.e. original indices `1 .. len(beats)-2`.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, StrictBool, ValidationError, conlist, field_validator

from agents.critic_llm import call_qwen_json_validated
from graph.state import ProductCutState, ProductTruth, ScriptVariant

logger = logging.getLogger("productcut.critics.body")


# ---------------------------------------------------------------------------
# Deterministic redundancy pre-pass (pure code — no LLM, no network).
#
# Method: content-word Jaccard similarity on each pair of body beat lines.
#
# Why Jaccard on stopword-filtered token SETS, and not embeddings:
#   * Dependency-light — the spec (§5.4.3) explicitly allows "a cheap
#     token-overlap ratio" as an alternative to "a single embedding-similarity
#     pass", and requirements.txt pins no embedding model / sentence-transformers.
#     Adding one for a *pre-pass whose only job is catching obvious literal
#     repetition* would be unjustified weight; this stays pure-stdlib.
#   * Filtering stopwords before comparing is what makes the ratio meaningful:
#     two unrelated beats share "the/a/is/it" heavily, so raw-token Jaccard
#     floors around a nonzero baseline for every pair. Comparing CONTENT words
#     only means the score reflects shared *subject matter*, so near-restatements
#     ("keeps drinks warmer, way longer" vs "stays warm way longer") land high
#     while genuinely distinct beats land near zero.
#
# Known and intended blind spot: this catches LEXICAL restatement, not SEMANTIC
# restatement. Two beats arguing the identical point with disjoint vocabulary
# ("done entirely by hand" vs "no machine ever touches this") score ~0 here — by
# construction, since they share no content words. That is exactly the case the
# LLM judgment layer exists to catch (reverse-outline: same label => redundant).
# The pre-pass guarantees the *mechanical* cases; it does not pretend to catch the
# semantic ones. See test_body_checker.py for the honest report of this behaviour.
# ---------------------------------------------------------------------------

# Small, self-contained English stopword set — enough to strip the function words
# that would otherwise inflate every pair's overlap. Deliberately not a full NLP
# stoplist (no dependency); tuned for short ad-copy beats.
_STOPWORDS = frozenset(
    """
    a an the this that these those it its it's is are was were be been being am
    and or but so then than as of to in on at by for with from into onto out up
    down off over under again more most very just only own same too also not no
    yes do does did done doing have has had having i you he she we they them him
    her his hers your yours my mine our ours their theirs me us who whom which
    what when where why how all any both each few some such can could will would
    shall should may might must here there about after before because while if
    """.split()
)

_WORD_RE = re.compile(r"[a-z0-9]+")

# Default overlap threshold above which a pair is flagged a candidate redundancy.
# 0.4 content-word Jaccard: near-restatements (most content words shared) clear
# it comfortably, while distinct beats that merely share one or two topic words
# stay below it. It is a tunable knob, not a claim of perfect calibration — the
# LLM makes the final ruling on every flag (accept, or explain the false-positive).
_DEFAULT_THRESHOLD = 0.4


def _content_tokens(line: str) -> set[str]:
    """Lowercase, split to alphanumeric word tokens, drop stopwords -> a set."""
    return {tok for tok in _WORD_RE.findall(line.lower()) if tok not in _STOPWORDS}


def _jaccard(a: set[str], b: set[str]) -> float:
    """Jaccard similarity of two token sets. 0.0 when either is empty."""
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def redundant_beat_prepass(
    beats: list[dict],
    threshold: float = _DEFAULT_THRESHOLD,
) -> list[list[int]]:
    """Flag candidate redundant BODY beat pairs by content-word Jaccard overlap.

    Operates on the body only — `beats[1:-1]`, i.e. every pair of original beat
    indices in `1 .. len(beats)-2` — never comparing against the hook (`beats[0]`)
    or CTA (`beats[-1]`).

    Args:
        beats:     the variant's FULL beats list (each a dict with a "line").
        threshold: content-word Jaccard above which a pair is flagged (default 0.4).

    Returns:
        A list of `[i, j]` pairs (i < j), indices into the FULL beats array,
        for every body-beat pair whose overlap exceeds `threshold`. Empty when
        there are fewer than two body beats or nothing crosses the threshold.
    """
    n = len(beats)
    if n < 4:
        # Need at least hook + 2 body beats + CTA for a within-body pair to exist.
        return []

    body_indices = list(range(1, n - 1))
    token_sets = {i: _content_tokens(beats[i].get("line", "")) for i in body_indices}

    flagged: list[list[int]] = []
    for a_pos in range(len(body_indices)):
        for b_pos in range(a_pos + 1, len(body_indices)):
            i, j = body_indices[a_pos], body_indices[b_pos]
            if _jaccard(token_sets[i], token_sets[j]) >= threshold:
                flagged.append([i, j])
    return flagged


# ---------------------------------------------------------------------------
# Output schema — the runtime gate on raw LLM JSON (shot_schema.py precedent,
# same posture as CtaCheckResult / ToneCheckResult).
# ---------------------------------------------------------------------------


class BodyCheckResult(BaseModel):
    """Validated Body-Checker output for a single variant (§5.4.3 contract).

    Mirrors the §5.4.3 output contract exactly:
      `{completion_score, redundant_beat_pairs, promise_payoff_match,
        emotional_trigger_landed, justification}`.

    `completion_score` is the raw 1-5 rubric integer that fills the Meta-Critic's
    "Completion / structural fit" axis (cast/weighted there, not here).
    `promise_payoff_match` and `emotional_trigger_landed` are StrictBool — a hard
    gate downstream reads them, so a stray "true"/"yes" string must never
    masquerade as a real JSON boolean (the CtaCheckResult never_do precedent).
    """

    model_config = ConfigDict(extra="forbid")

    completion_score: int = Field(..., ge=1, le=5)
    # Each pair is exactly two integers — indices into the full beats array.
    redundant_beat_pairs: list[conlist(int, min_length=2, max_length=2)] = Field(
        default_factory=list
    )
    promise_payoff_match: StrictBool
    emotional_trigger_landed: StrictBool
    justification: str = Field(..., min_length=1)

    @field_validator("redundant_beat_pairs")
    @classmethod
    def _pairs_are_ordered_nonnegative(cls, pairs):
        """Reject negative indices / self-pairs; normalise each pair to (min, max).

        The LLM occasionally emits `[j, i]` (unordered) or `[i, i]`; ordering and
        de-selfing here keeps downstream de-duplication trivial without a second
        pass. Negative indices can never be valid beat positions, so they fail
        the structural gate rather than being silently dropped.
        """
        normalised: list[list[int]] = []
        for pair in pairs:
            i, j = pair[0], pair[1]
            if i < 0 or j < 0:
                raise ValueError(f"redundant_beat_pairs contains a negative index: {pair}")
            if i == j:
                raise ValueError(f"redundant_beat_pairs contains a self-pair: {pair}")
            normalised.append([min(i, j), max(i, j)])
        return normalised


# ---------------------------------------------------------------------------
# System prompt — example-anchored rubric embedding the editorial diagnostics
# and calibrated exemplar pairs from the research pass (§5.4.3).
# ---------------------------------------------------------------------------

_BODY_SYSTEM_PROMPT = """\
You are a creative director doing a COLD read of the BODY of a short-form video ad \
script — the beats between the opening hook and the closing CTA. You did not write \
it. Your job is to judge whether the middle of the script is actually doing its \
work, and return an integer completion_score from 1 to 5 per variant.

You score FOUR things. Walk through these diagnostic steps explicitly for each variant \
before you score:

STEP 1 — REVERSE-OUTLINE every body beat.
Label each beat by WHAT IT ARGUES, not what it says (e.g. "handmade", "keeps heat", \
"price is worth it"). Two beats with the SAME label are redundant even with ZERO shared \
words. Apply the Deletion Test: if a beat were deleted, does the argument still hold? If \
yes, that beat adds nothing. Apply Sugarman's slippery-slide: each beat's only job is to \
earn the next one being watched.

STEP 2 — NON-REDUNDANCY.
You are given `candidate_redundant_pairs`: pairs of beat indices a deterministic \
lexical pre-pass flagged as likely repetition. You MUST rule on each one — either \
confirm it is redundant (keep it) or explain in your justification why it is NOT actually \
redundant (e.g. beat 3 converts beat 2's fact into a genuinely new argument). You must \
NOT silently ignore a flag. You must ALSO add pairs the lexical pre-pass could not catch: \
two beats with the same reverse-outline label but different words ARE redundant — include \
them. Report every genuinely-redundant pair in `redundant_beat_pairs`, using the beat \
`index` values shown. Also check ESCALATION DIRECTION: specificity/stakes should rise \
beat-over-beat; a beat that de-escalates (gets vaguer/lower-stakes) is an anticlimax — \
note it.

STEP 3 — PROMISE-PAYOFF MATCH.
Extract the hook's specific promissory element — the concrete noun / number / claim it \
puts on the table (given as `hook_line` for context; do NOT score the hook itself). Does \
a body beat supply the MECHANISM or EVIDENCE for THAT EXACT element? \
Restatement (re-asserting the claim in other words) is NOT payoff. Substitution (proving \
a different, easier claim) is NOT payoff. A real payoff traces to a specific product \
truth (check `grounding_truth_ids` / `product_truths`) where possible. Set \
`promise_payoff_match` true only if the body substantively develops the hook's actual \
promise.

STEP 4 — THROUGHLINE (Rule of One).
Write the script's one-sentence argument: one pain + one product + one promise. Any beat \
that needs a SECOND promise to justify its presence is a competing claim (a "smuggled \
second benefit") — a throughline violation. CALIBRATION: a beat that REFRAMES the same \
promise (e.g. the same price argument expressed as per-hour math) is throughline-CONSISTENT \
and is NOT a violation — do not flag legitimate reframes.

STEP 5 — EMOTIONAL-TRIGGER FIDELITY.
The declared trigger is one of: curiosity, recognition, FOMO, tribal identity, \
transformation/aspiration, relief. Mentally STRIP all emotion-labeling words ("amazing", \
"you'll love", "don't miss out", "finally"). If the declared trigger disappears once the \
labels are gone, it was ASSERTED, not earned -> `emotional_trigger_landed` false. Also \
check the trigger MATCHES: if the body earns a DIFFERENT emotion than the one declared \
(e.g. it earns relief when curiosity was declared), set `emotional_trigger_landed` false \
and NAME the mismatch in the justification — do not give credit for earning the wrong \
emotion well.

SCORING ANCHORS (1-5):
- 5 = payoff with a mechanism traced to a real truth id; all beats escalate (no \
same-label pairs); one clear argument; trigger earned WITHOUT relying on labels.
- 3 = payoff present but partially just restated, OR exactly one redundant pair, OR the \
trigger is earned only generically (not specific to this product).
- 1 = no payoff / unclosed loop, OR two or more redundant pairs, OR a competing claim, OR \
the trigger is merely asserted/labeled.
HARD CAP: if `promise_payoff_match` is false OR `emotional_trigger_landed` is false, the \
completion_score MUST NOT exceed 3, regardless of everything else.

CALIBRATED EXEMPLARS (learn the boundary from these):
- Promise-payoff. Hook "Your coffee is cold in 12 minutes. Mine isn't."
  WEAK body "This mug keeps drinks warmer, way longer than a normal one" — restatement, \
no mechanism -> promise_payoff_match FALSE.
  STRONG body "Twelve minutes is when ceramic gives up. This one's double-wall vacuum \
seal (t3) is still holding 140 degrees at minute 40" — mechanism + number, traced to \
truth t3 -> promise_payoff_match TRUE.
- Non-redundancy (ZERO lexical overlap case — the pre-pass will NOT flag this; YOU must).
  WEAK: beat "The stitching is done entirely by hand" + beat "No machine ever touches \
this bag" — different words, identical argument (reverse-outline both as "handmade") -> \
redundant pair.
  STRONG: second beat instead "Which is why no two bags match — yours has stitch spacing \
nobody else's will" — converts the same fact into a NEW argument (uniqueness) -> NOT \
redundant.
- Throughline. Hook about cold coffee; beat 2 proves heat retention.
  WEAK: beat 3 "Plus it's dishwasher safe and comes in six colors!" — smuggled second \
benefit -> throughline violation, score <= depressed.
  NOT-A-VIOLATION reframe: a 38-dollar candle ad, beat 2 "big-brand candles burn out in \
20 hours", beat 3 "this one burns 60 — per hour of light it's the cheapest thing on your \
shelf" — same price argument reframed as math -> throughline INTACT, NOT competing.
- Emotional-trigger mismatch. Declared trigger "curiosity". Body "Finally — no more \
re-brewing, no more microwave laps. Just sit down and drink your coffee." — this \
well-earns RELIEF but opens/closes no curiosity loop -> emotional_trigger_landed FALSE, \
name the curiosity/relief mismatch, even though the writing is good.

Return ONLY a JSON object of this exact shape:
{"results": [{"variant_id": "<id>", "completion_score": <int 1-5>, \
"redundant_beat_pairs": [[<i>, <j>], ...], "promise_payoff_match": <true|false>, \
"emotional_trigger_landed": <true|false>, "justification": "<2-4 sentences: the \
one-sentence argument you extracted, the hook promise and whether the body paid it off, \
any redundant pairs and your ruling on each flagged candidate, and the trigger verdict>"}]}
Include exactly one entry per variant. Booleans MUST be real JSON booleans, never \
strings. `redundant_beat_pairs` MUST use the beat index values given (an empty list if \
none). No prose outside the JSON.\
"""


# ---------------------------------------------------------------------------
# Input serialisation.
# ---------------------------------------------------------------------------


def _body_payload_for_variant(variant: ScriptVariant, threshold: float) -> dict:
    """Build the per-variant model payload: body beats (with original indices),
    the hook/CTA lines as context, the framework/trigger/grounding metadata, and
    the deterministic pre-pass's candidate redundant pairs for the model to rule on.
    """
    beats = variant.get("beats") or []
    hook_line = beats[0]["line"] if beats else ""
    cta_line = beats[-1]["line"] if len(beats) >= 1 else ""
    # Body = beats[1:-1], carrying each beat's ORIGINAL index so the model's
    # redundant_beat_pairs speak the same index language as the pre-pass + output.
    body_beats = [
        {"index": i, "line": beats[i]["line"]} for i in range(1, len(beats) - 1)
    ]
    return {
        "variant_id": variant["variant_id"],
        "framework": variant.get("framework"),
        "emotional_trigger": variant.get("emotional_trigger"),
        "grounding_truth_ids": variant.get("grounding_truth_ids") or [],
        "hook_line_context_only": hook_line,
        "cta_line_context_only": cta_line,
        "body_beats": body_beats,
        "candidate_redundant_pairs": redundant_beat_prepass(beats, threshold),
    }


def _validate_results(
    raw: dict,
    expected_ids: set[str],
) -> dict[str, BodyCheckResult]:
    """Validate the LLM's {"results": [...]} envelope into {variant_id: model}.

    Same structural contract as cta_tone_checkers._validate_results (one result
    per expected variant, no extras/dupes, each entry passing BodyCheckResult).
    Kept local rather than imported so the Body-Checker has no cross-module
    coupling to a sibling checker's private helper.
    """
    results = raw.get("results")
    if not isinstance(results, list):
        raise ValueError(
            "Body-Checker: expected a JSON object with a 'results' list, got keys "
            f"{list(raw.keys())!r}"
        )

    validated: dict[str, BodyCheckResult] = {}
    for entry in results:
        if not isinstance(entry, dict) or "variant_id" not in entry:
            raise ValueError(f"Body-Checker: result entry missing variant_id: {entry!r}")
        vid = entry["variant_id"]
        if vid in validated:
            raise ValueError(
                f"Body-Checker: duplicate result entry for variant {vid!r} "
                "(expected exactly one entry per variant)"
            )
        payload = {k: v for k, v in entry.items() if k != "variant_id"}
        try:
            validated[vid] = BodyCheckResult.model_validate(payload)
        except ValidationError as exc:
            raise ValueError(
                f"Body-Checker: result for variant {vid!r} failed validation: {exc}"
            ) from exc

    got_ids = set(validated.keys())
    if got_ids != expected_ids:
        missing = expected_ids - got_ids
        extra = got_ids - expected_ids
        raise ValueError(
            f"Body-Checker: variant id mismatch. missing={sorted(missing)} "
            f"extra={sorted(extra)}"
        )
    return validated


def _apply_hard_cap(result: BodyCheckResult) -> BodyCheckResult:
    """Enforce the §5.4.3 hard cap deterministically, not just via the prompt.

    "promise_payoff_match=false OR emotional_trigger_landed=false caps the score
    at 3." Instructing the model is necessary but not sufficient — an LLM can
    still emit the self-contradiction (score 5 with promise_payoff_match false).
    Because the cap is a *rule*, not a judgment, we clamp it in code (Pacing-Checker
    philosophy: mechanical guarantees over sampling variance) and log when we do.
    Only the score is touched; the model's justification is left as written.
    """
    if (not result.promise_payoff_match or not result.emotional_trigger_landed) and (
        result.completion_score > 3
    ):
        logger.info(
            "Body-Checker: clamping completion_score %d->3 (payoff=%s trigger=%s)",
            result.completion_score,
            result.promise_payoff_match,
            result.emotional_trigger_landed,
        )
        return result.model_copy(update={"completion_score": 3})
    return result


# ---------------------------------------------------------------------------
# Public API.
# ---------------------------------------------------------------------------


def check_body(
    variants: list[ScriptVariant],
    product_truths: Optional[list[ProductTruth]] = None,
    *,
    model: Optional[str] = None,
    redundancy_threshold: float = _DEFAULT_THRESHOLD,
) -> dict[str, dict]:
    """Body-Checker (§5.4.3): score the body (beats[1:-1]) of each variant.

    Two layers: a deterministic content-word-overlap pre-pass flags candidate
    redundant beat pairs, then a single Qwen judgment call scores the four axes
    (promise-payoff, non-redundancy, throughline, emotional-trigger fidelity),
    ruling on the flagged pairs and catching the semantic redundancy the lexical
    pre-pass cannot. A deterministic hard cap (score <= 3 when payoff or trigger
    is false) is enforced in code after validation.

    Args:
        variants:            the Concept Agent's script variants (typically four).
        product_truths:      the job's ProductTruth list, for judging whether a
                             body beat's proof traces to a real cited truth. The
                             Concept Agent's variants don't carry this themselves,
                             so it is passed in alongside them (as the Tone-Checker
                             takes brief/seller_direction). Optional — omitting it
                             just removes the truth-traceability context.
        model:               optional model-id override; defaults to MODEL_TEXT (.env).
        redundancy_threshold: content-word Jaccard threshold for the pre-pass.

    Returns:
        {variant_id: {"completion_score": int 1-5, "redundant_beat_pairs": [[i,j]],
                      "promise_payoff_match": bool, "emotional_trigger_landed": bool,
                      "justification": str}} — validated, with the hard cap applied.

    Raises:
        ValueError:    if the model output fails structural validation.
        QwenJSONError: if the model returns non-JSON (from the shared helper).
    """
    if not variants:
        return {}

    payload = [_body_payload_for_variant(v, redundancy_threshold) for v in variants]
    user_prompt = (
        "Score the BODY of each of these ad-script variants. Rule on every "
        "candidate_redundant_pair. product_truths are shared across all variants.\n\n"
        + json.dumps(
            {
                "product_truths": product_truths or [],
                "variants": payload,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    logger.info(
        "Body-Checker scoring %d variant(s); %d have pre-pass redundancy flags",
        len(variants),
        sum(1 for p in payload if p["candidate_redundant_pairs"]),
    )

    expected = {v["variant_id"] for v in variants}
    # call_qwen_json_validated (not the bare call_qwen_json) -- a real live run
    # (video-gen-fidelity branch) found the model can return a plausible-but-
    # wrong field name (`redundant_pairs` instead of `redundant_beat_pairs`),
    # which BodyCheckResult's extra="forbid" rejects outright; with no retry
    # path that killed the whole graph run on a one-off phrasing slip. This
    # gives _validate_results one bounded re-prompt naming the exact error
    # before surfacing it, mirroring every other content-failure retry in this
    # codebase (see agents/critic_llm.py's call_qwen_json_validated docstring).
    validated = call_qwen_json_validated(
        _BODY_SYSTEM_PROMPT,
        user_prompt,
        lambda raw: _validate_results(raw, expected),
        model=model,
    )
    return {vid: _apply_hard_cap(res).model_dump() for vid, res in validated.items()}


async def body_checker_node(state: ProductCutState) -> dict:
    """LangGraph node wrapper. check_body is sync + blocking network I/O, so it runs via asyncio.to_thread to avoid blocking the event loop during the parallel fan-out."""
    scores = await asyncio.to_thread(
        check_body, state["script_variants"], state.get("product_truths")
    )
    return {"body_scores": scores}


__all__ = [
    "BodyCheckResult",
    "check_body",
    "redundant_beat_prepass",
]
