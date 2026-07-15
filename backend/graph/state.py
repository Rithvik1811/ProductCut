"""
C1 — LangGraph shared state schema (frozen contract, Phase 0).
Extend additively only: add new keys, never rename/remove an existing one
without a sync between KR and RR and a version bump in this docstring.
Spec of record: docs/TECHNICAL_DOCUMENTATION.md section 6.

version: 8
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
  - v6: Phase 3 (agents/video_gen_node.py, KR) hands a shot off to the
        not-yet-built Ken-Burns Fallback Node (§5.9, RR) on a hard API
        failure/timeout/budget-exceeded by setting status to a new value and
        attaching a failure_reason -- formalizing what KR's module flagged as
        a self-invented, unvalidated "known departure" pending a KR/RR sync
        (docs/BUILD_TASKS.md Phase 3). Adds: Shot.status += "fallback_requested"
        (deliberately distinct from the existing "fallback" value, which per
        §5.9 means Ken-Burns has ALREADY rendered the clip -- "fallback_requested"
        means "handed off, not rendered yet"); new FailureReason TypedDict
        ({type: "timeout"|"api_error"|"budget_exceeded", detail: str}) and
        Shot.failure_reason: NotRequired[FailureReason]. Mirrored in
        graph/shot_schema.py's runtime-validated ShotStatus/ShotModel.
  - v7: Phase 5 wiring (agents/voiceover_caption_agent.py) made
        voiceover_caption_agent a genuine parallel branch off merge_validator's
        "finalize"/"fallback" routes, alongside treatment_agent (graph/build.py).
        Both branches used to independently read-modify-write the single
        `reasoning_trace: str` key (every agent module's shared "read current +
        append this node's note" convention) -- fine when sequential, but two
        nodes in the SAME superstep both writing a plain (LastValue) channel is
        rejected outright by LangGraph (InvalidUpdateError: "Can receive only
        one value per step"). Rather than retrofit that read-modify-write
        convention across all dozen existing agent modules (and their already-
        passing unit tests that assert on it) into a proper concurrent-safe
        reducer, voiceover_caption_agent_node was given its OWN dedicated key,
        `voiceover_reasoning_trace` -- a single-writer channel, so no reducer is
        needed and every other node's existing contract/tests are untouched.
        Additive only; `reasoning_trace` is unchanged in shape and still the
        trace for the treatment_agent-and-onward chain.
  - v8: Video-gen fidelity fix (root cause: a Meta Quest VR headset's photos
        produced a generated clip showing "a phone on a phone stand" --
        completely the wrong object). Two additive changes, both used only by
        agents/product_truth_extractor.py, agents/video_gen_node.py,
        agents/continuity_agent.py and agents/continuity_gate.py -- no
        existing field renamed/removed:
          (a) ProductTruth.category += "form_factor" -- exactly one fact per
              job, a single holistic whole-object shape/silhouette/size
              anchor sentence synthesized across ALL product photos, distinct
              from every other category (which are each one isolated
              micro-fact). Root cause: an i2v prompt built from only a single
              isolated micro-fact ("matte-black textured strap") under-
              specifies the subject as a WHOLE object, and i2v models suffer
              documented "identity drift" where an ambiguous subject resolves
              toward a common training-data composition ("phone on a stand")
              instead of the actual, rarer product shape. Consumed by
              video_gen_node.py's Subject line construction, which now leads
              with this anchor before the per-shot micro-fact.
          (b) GeneratedShot.identity_check: NotRequired[IdentityCheck] -- a
              CATEGORICAL "is this even the same physical object class"
              verdict from a SEPARATE Qwen-VL call on an early frame of the
              generated clip (agents/continuity_agent.py's new frame-0(ish)
              identity check). Additive alongside the existing continuous
              `drift_score` -- a different question on a different scale, not
              a stricter threshold on the same one. Consumed by
              agents/continuity_gate.py for a hard-identity-failure routing
              path (one automatic re-sample, then straight to the Ken-Burns
              fallback on a second consecutive failure) distinct from the
              existing drift-retry/human-review path.
  - v9: Video-gen creative-direction fix (video-gen-fidelity branch, RR) added
        one additive enum value to Shot ahead of a Shot-List Agent/Video-Gen
        Node prompt-phrasing rework: shot_type += "worn_in_use" (product worn/
        carried/operated by a visible person at medium-to-wide framing, person
        moves, product rides along) -- distinct from the existing
        "lifestyle_context" (now strictly a NO-human styled scene, resolving a
        prior dual-meaning overload) and "product_in_hand" (a static/close
        hand-contact composition). Mirrored in graph/shot_schema.py's
        runtime-validated ShotType (C3 v4). No field additions/removals.
  - v10: Story-arc/character-consistency fix (video-gen-fidelity branch).
        Adds ONE additive field: Treatment.character_anchor: NotRequired[str].
        Root cause: text-only i2v prompting cannot lock FACIAL identity across
        independent Wan generations (no video-to-video chaining, no seed
        guarantee -- confirmed against Alibaba's own docs), so a script
        implying a recurring person with no anchored physical description let
        each human-interaction shot's Call B independently invent hair/
        wardrobe/setting, producing a different-looking "different person" in
        every shot rather than one consistent story. `character_anchor` is
        ONE sentence, produced by the Treatment Agent (not Concept Agent --
        Concept Agent still produces 4 competing variants pre-selection, so
        synthesizing a character per variant would be wasted work; Treatment
        Agent already runs once on the winning script and already owns other
        whole-ad global fields of identical shape) ONLY when the winning
        script's STORY/REAL-WORLD-USE beat actually implies a person -- never
        forced. Gives hair color/length/texture, one distinctively-colored
        wardrobe item, an age band, and a named setting with 1-2 fixed
        landmarks + time-of-day, drawn from the same palette as `color_story`
        so human shots and product-alone shots visually cohere. Consumed by
        agents/video_gen_node.py's new verbatim, never-cut `Cast:` prompt
        section on every human-interaction shot (see that module's docstring).
        NotRequired + empty-string-safe: a script with no implied person
        (or a Treatment Agent fallback) simply omits/blanks it, and
        video_gen_node.py's Cast section renders nothing in that case.
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
        "imperfection", "scale_cue", "brief_or_intake_fact", "form_factor",
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
    # v10: ONE sentence anchoring a recurring human character's hair, one
    # distinctively-colored wardrobe item, age band, and named setting (1-2
    # fixed landmarks + time-of-day) -- present only when the winning script
    # implies a person; see the v10 changelog note above for why this lives
    # here and not on ScriptVariant/Shot.
    character_anchor: NotRequired[str]


class ShotJustification(TypedDict):
    script_quote: str
    truth_fact_id: str
    treatment_ref: int  # matches a Treatment.beat_treatments[].beat_index


class FailureReason(TypedDict):
    type: Literal["timeout", "api_error", "budget_exceeded"]
    detail: str


class Shot(TypedDict):
    shot_id: str
    t_start: float
    t_end: float
    beat_role: Literal["hook", "problem", "demo", "proof", "cta"]
    description: str
    shot_type: Literal[
        "hook_hero", "macro_detail", "lifestyle_context",
        "hero_reframe", "cta_endcard", "product_in_hand", "worn_in_use",
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
    status: Literal[
        "pending", "generating", "passed", "fallback", "review", "fallback_requested",
    ]
    retry_count: int
    failure_reason: NotRequired[FailureReason]
    # NOTE: no `product_category` field — omission is deliberate, see TECHNICAL_DOCUMENTATION.md §5.6


class BudgetLedger(TypedDict):
    cap: float
    spent: float
    per_shot: dict[str, float]


class IdentityCheck(TypedDict):
    matching_features: list[str]
    mismatching_features: list[str]
    same_object: bool
    confidence: Literal["high", "medium", "low"]


class GeneratedShot(TypedDict):
    video_uri: str
    drift_score: NotRequired[float]
    identity_check: NotRequired[IdentityCheck]
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
    meta_critic_result: NotRequired[dict]         # full MetaCriticResult.model_dump(); merge_validator_node (agents/merge_validator.py) consumes this. NOT winning_script — that is only set once an independent validator passes.
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
    # v7: voiceover_caption_agent_node's OWN trace key -- see v7 changelog note
    # above for why it's separate from `reasoning_trace` (single-writer channel,
    # avoids a same-superstep concurrent-write conflict with treatment_agent).
    voiceover_reasoning_trace: NotRequired[str]

    # populated only by Phase 9 (chat-based revision)
    chat_thread: list[ChatMessage]
    edit_requests: list[EditRequest]
    version_history: list[VersionEntry]
