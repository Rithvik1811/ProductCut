"""
C1 — LangGraph shared state schema (frozen contract, Phase 0).
Extend additively only: add new keys, never rename/remove an existing one
without a sync between KR and RR and a version bump in this docstring.
Spec of record: docs/TECHNICAL_DOCUMENTATION.md section 6.

version: 1
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


class CriticScore(TypedDict):
    hook: float
    pacing: float
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
        "hero_reframe", "cta_endcard",
    ]
    camera_move: Literal["push_in", "orbit", "static", "pan", "tilt_up", "pull_back"]
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
    critic_scores: dict[str, CriticScore]
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
