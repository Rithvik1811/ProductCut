"""
C1 — LangGraph shared state schema (frozen contract, Phase 0).
Extend additively only: add new keys, never rename/remove an existing one
without a sync between KR and RR and a version bump in this docstring.
Spec of record: docs/TECHNICAL_DOCUMENTATION.md section 6.

version: 5
  - v2: added CompletionDetail + two CriticScore keys (completion, completion_detail)
        and six NotRequired Critic-Chain scratch keys (hook/pacing/body/cta/tone_scores,
        meta_critic_result) to plumb the 5 parallel checkers into the Meta-Critic join.
  - v3: added two NotRequired Merge Coherence Validator scratch keys (§5.4.7):
        merge_attempts (list[dict], §6-shaped: one entry per merge attempt, appended
        by merge_validator_node) and pending_merge_candidate (dict, a
        MergeCandidate.model_dump() scratch key written when repairing/setting up
        the next attempt). winning_script is unchanged in shape -- it is now only
        ever written by merge_validator_node, on a pass or the terminal fallback.
  - v4: added two NotRequired scratch keys reconciling merge_validator_node with
        agents/copy_editor.py's copy_editor_node (built in parallel, documents its
        own state assumptions in its module docstring): coherence_validation_result
        (dict, a CoherenceValidationResult.model_dump() -- written by
        merge_validator_node when routing to the Copy Editor, read by
        copy_editor_node for seam_flags/justification) and last_copy_edit (dict, a
        CopyEditResult.model_dump() -- written by copy_editor_node, read by
        merge_validator_node's next call to populate that attempt's §6 copy_edit
        sub-object and the "copy_edited_then_accepted" outcome).
  - v5: Phase 2 research (docs/TECHNICAL_DOCUMENTATION.md §5.6) added two
        additive enum values to Shot ahead of the Shot-List Agent build:
        camera_move += "rack_focus", shot_type += "product_in_hand". Mirrored
        in graph/shot_schema.py's runtime-validated CameraMove/ShotType (C3
        freeze, docs/BUILD_TASKS.md Phase 2). No field additions/removals.
"""
from typing import Literal, TypedDict
from typing_extensions import NotRequired


class ReferenceAd(TypedDict):
    url_or_text: str
    why: str


class SellerDirection(TypedDict, total=False):
    mood_words: list[str]
    reference_ad: ReferenceAd
    never_do: str
    freeform: str


class ProductTruth(TypedDict):
    truth_id: str
    fact: str
    category: Literal[
        "color", "material", "texture", "construction_detail",
        "imperfection", "scale_cue", "brief_or_intake_fact",
    ]
    source: str


class ScriptBeat(TypedDict):
    t_start: float
    t_end: float
    line: str


class ScriptVariant(TypedDict):
    variant_id: str
    text: str
    framework: Literal["hook_problem_product_cta", "PAS", "AIDA", "BAB"]
    hook_type: str
    emotional_trigger: str
    grounding_truth_ids: list[str]
    beats: list[ScriptBeat]
    target_length_sec: int


class CompletionDetail(TypedDict):
    redundant_beat_pairs: list[list[int]]
    promise_payoff_match: bool
    emotional_trigger_landed: bool


class CriticScore(TypedDict):
    hook: float
    pacing: float
    completion: float                   # NEW — Body-Checker's completion_score (§5.4.3)
    completion_detail: CompletionDetail  # NEW — Body-Checker's redundancy/promise-payoff/trigger detail
    cta: float
    tone: float
    composite: float
    justification: str
    never_do_violation: bool


class WinningScript(TypedDict):
    text: str
    beats: list[ScriptBeat]
    source_variant_ids: list[str]


class BeatTreatment(TypedDict):
    beat_index: int
    beat_function: Literal["hook", "problem", "demo", "proof", "cta"]
    script_quote: str
    truth_fact_id: str
    visual_approach: str
    why_not_generic: str


class Treatment(TypedDict):
    director_persona: str
    color_story: str
    pacing_philosophy: str
    beat_treatments: list[BeatTreatment]


class ShotJustification(TypedDict):
    script_quote: str
    truth_fact_id: str
    treatment_ref: int  # matches a Treatment.beat_treatments[].beat_index


class Shot(TypedDict):
    shot_id: str
    t_start: float
    t_end: float
    beat_role: Literal["hook", "problem", "demo", "proof", "cta"]
    description: str
    shot_type: Literal[
        "hook_hero", "macro_detail", "lifestyle_context",
        "hero_reframe", "cta_endcard", "product_in_hand",
    ]
    camera_move: Literal[
        "push_in", "orbit", "static", "pan", "tilt_up", "pull_back", "rack_focus",
    ]
    framing: Literal[
        "fills_frame", "rule_of_thirds_left", "rule_of_thirds_right", "context_wide",
    ]
    lighting: str  # one shared string reused across every shot in the job
    negative_prompt: str
    reference_image_id: str
    text_overlay_zone: Literal["none", "left_third", "right_third", "lower_third"]
    duration_sec: float
    allocated_budget: float
    voiceover_line: str
    justification: ShotJustification
    status: Literal["pending", "generating", "passed", "fallback", "review"]
    retry_count: int
    # NOTE: no `product_category` field — omission is deliberate, see TECHNICAL_DOCUMENTATION.md §5.6


class BudgetLedger(TypedDict):
    cap: float
    spent: float
    per_shot: dict[str, float]


class GeneratedShot(TypedDict):
    video_uri: str
    drift_score: NotRequired[float]
    attempt: int


class Voiceover(TypedDict):
    audio_uri: str
    caption_track_uri: str


class Exports(TypedDict):
    aspect_9x16: str
    aspect_1x1: str
    aspect_16x9: str


class HumanReviewEntry(TypedDict):
    shot_id: str
    drift_score: float
    candidate_frame_uris: list[str]
    resolution: NotRequired[Literal["approve", "retry_with_edit", "accept_fallback"]]


class ChatMessage(TypedDict):
    role: Literal["seller", "system"]
    message: str
    ts: str


class EditRouterOutput(TypedDict):
    scope: Literal["shot_visual", "copy_tone", "pacing_length", "cta_text", "global"]
    target_shot_ids: list[str]
    entry_node: str
    confidence: float
    rationale: str


class EditInterpreterPatch(TypedDict, total=False):
    treatment_patch: dict
    shot_patches: list[dict]
    justification: str


class EditRequest(TypedDict):
    edit_id: str
    message: str
    router_output: EditRouterOutput
    interpreter_patch: NotRequired[EditInterpreterPatch]
    status: Literal["pending_preview", "confirmed", "rejected", "applied", "failed"]
    fork_branch_id: NotRequired[str]
    estimated_cost: NotRequired[float]
    actual_cost: NotRequired[float]


class VersionEntry(TypedDict):
    branch_id: str
    parent_branch_id: NotRequired[str]
    created_at: str
    summary: str


class ProductCutState(TypedDict, total=False):
    # populated at Ingest (Phase 1) — required from the start of the job
    job_id: str
    brief: str
    product_photos: list[str]
    seller_direction: SellerDirection

    # populated by Phase 1 (Product Truth Extractor, Concept Agent, Critic Chain)
    product_truths: list[ProductTruth]
    script_variants: list[ScriptVariant]
    hook_scores: NotRequired[dict[str, dict]]     # raw Hook-Checker output, consumed by meta_critic_node
    pacing_scores: NotRequired[dict[str, dict]]   # raw Pacing-Checker output
    body_scores: NotRequired[dict[str, dict]]     # raw Body-Checker output
    cta_scores: NotRequired[dict[str, dict]]      # raw CTA-Checker output
    tone_scores: NotRequired[dict[str, dict]]     # raw Tone-Checker output
    meta_critic_result: NotRequired[dict]         # full MetaCriticResult.model_dump(); the next task (Merge Coherence Validator) consumes this. NOT winning_script — that is only set once an independent validator passes (not built yet).
    critic_scores: dict[str, CriticScore]
    merge_attempts: NotRequired[list[dict]]
    pending_merge_candidate: NotRequired[dict]
    coherence_validation_result: NotRequired[dict]
    last_copy_edit: NotRequired[dict]
    winning_script: WinningScript
    reasoning_trace: str

    # populated by Phase 2 (Treatment Agent, Shot-List Agent, Budget Gate)
    treatment: Treatment
    shot_list: list[Shot]
    budget_ledger: BudgetLedger

    # populated by Phase 3/4 (Video-Gen, Continuity)
    generated_shots: dict[str, GeneratedShot]
    human_review_queue: list[HumanReviewEntry]

    # populated by Phase 5 (Voiceover, Assembly, Export)
    voiceover: Voiceover
    master_cut_uri: str
    exports: Exports

    # populated only by Phase 9 (chat-based revision)
    chat_thread: list[ChatMessage]
    edit_requests: list[EditRequest]
    version_history: list[VersionEntry]
