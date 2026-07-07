"""
C3 — Shot-list JSON schema (first cut, Phase 0; frozen at end of Phase 2).

`graph.state.Shot` / `graph.state.ShotJustification` (C1) define the *shape*
of a shot for typing purposes, but a `TypedDict` performs no runtime checks --
anything can be assigned to it. C3 exists because the Shot-List Agent's raw
LLM output needs actual mechanical validation before it's trusted, per
docs/TECHNICAL_DOCUMENTATION.md §2.2 ("each reasoning node's output schema is
a Pydantic model; a post-call deterministic validator function ... checks
the justification fields"). This module is that Pydantic model.

Scope, deliberately narrow: this is the *structural* schema check only --
right field types, right enum values, required fields present, and no
forbidden fields (see below). It is NOT the semantic Justification Validator
(verbatim script_quote check, truth_fact_id existence check, treatment_ref
cross-reference, stoplist rejection) -- that is a separate Phase 2 deliverable
owned by KR (docs/BUILD_TASKS.md, Phase 2), since it requires cross-referencing
the actual script/truths/treatment, not just validating one shot's shape.

The single hard rule this DOES mechanically enforce: **no `product_category`
field, ever** -- `model_config = ConfigDict(extra="forbid")` means a shot
carrying that field (or any other undeclared field) fails validation outright,
rather than relying on a prompt instruction the LLM might ignore. This is the
concrete mechanism behind the project's core anti-genericness design pillar.

The enum literals below are hand-kept in sync with `graph.state.Shot`'s
inline Literals (TypedDict Literals aren't separately importable without
refactoring state.py, which is out of scope here -- state.py is C1's frozen
contract). If C1's Shot literals change, update these to match and bump the
version below.

version: 2
  - v2: Phase 2 research (docs/TECHNICAL_DOCUMENTATION.md SS5.6) added two
        additive enum values ahead of the Shot-List Agent build: `rack_focus`
        (CameraMove) and `product_in_hand` (ShotType). Both are structurally
        hard to justify generically -- a rack focus requires naming two real
        product referents on different focal planes, and product_in_hand
        gives `demo`/`proof` beats a real human-interaction composition
        instead of being forced into `lifestyle_context` or `macro_detail`.
        This is the C3 freeze (docs/BUILD_TASKS.md, Phase 2, RR).
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

BeatRole = Literal["hook", "problem", "demo", "proof", "cta"]

ShotType = Literal[
    "hook_hero", "macro_detail", "lifestyle_context",
    "hero_reframe", "cta_endcard", "product_in_hand",
]

CameraMove = Literal[
    "push_in", "orbit", "static", "pan", "tilt_up", "pull_back", "rack_focus",
]

Framing = Literal[
    "fills_frame", "rule_of_thirds_left", "rule_of_thirds_right", "context_wide",
]

TextOverlayZone = Literal["none", "left_third", "right_third", "lower_third"]

ShotStatus = Literal["pending", "generating", "passed", "fallback", "review"]


class ShotJustificationModel(BaseModel):
    """Mirrors graph.state.ShotJustification -- runtime-validated."""

    model_config = ConfigDict(extra="forbid")

    script_quote: str = Field(..., min_length=1)
    truth_fact_id: str = Field(..., min_length=1)
    treatment_ref: int = Field(..., ge=0)


class ShotModel(BaseModel):
    """Mirrors graph.state.Shot -- runtime-validated.

    `extra="forbid"` is what makes the `product_category` exclusion a real,
    mechanical rule rather than a prompt-level suggestion: any shot JSON that
    includes a `product_category` key (or any other field not declared here)
    fails Pydantic validation immediately, before it ever reaches a human or
    a downstream node.
    """

    model_config = ConfigDict(extra="forbid")

    shot_id: str = Field(..., min_length=1)
    t_start: float = Field(..., ge=0)
    t_end: float = Field(..., gt=0)
    beat_role: BeatRole
    description: str = Field(..., min_length=1)
    shot_type: ShotType
    camera_move: CameraMove
    framing: Framing
    lighting: str = Field(..., min_length=1)
    negative_prompt: str
    reference_image_id: str = Field(..., min_length=1)
    text_overlay_zone: TextOverlayZone
    duration_sec: float = Field(..., gt=0)
    allocated_budget: float = Field(..., ge=0)
    voiceover_line: str
    justification: ShotJustificationModel
    status: ShotStatus
    retry_count: int = Field(..., ge=0)


def validate_shot(raw: dict) -> ShotModel:
    """Validate one raw shot dict (e.g. parsed from LLM JSON output).

    Raises pydantic.ValidationError on any structural problem: wrong type,
    missing required field, bad enum value, or a forbidden field like
    `product_category`. Callers should catch ValidationError and re-prompt
    (per the Shot-List Agent's re-prompt-once-then-fallback policy in
    docs/TECHNICAL_DOCUMENTATION.md §5.6), not crash the pipeline.
    """
    return ShotModel.model_validate(raw)


def validate_shot_list(raw_shots: list[dict]) -> list[ShotModel]:
    """Validate a full shot list (3-7 shots per the architecture spec).

    Structural validation only -- see module docstring for what this
    deliberately does NOT check (the semantic Justification Validator is
    separate, later, Phase 2 work).
    """
    return [validate_shot(shot) for shot in raw_shots]


__all__ = [
    "BeatRole",
    "ShotType",
    "CameraMove",
    "Framing",
    "TextOverlayZone",
    "ShotStatus",
    "ShotJustificationModel",
    "ShotModel",
    "validate_shot",
    "validate_shot_list",
]
