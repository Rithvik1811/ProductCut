"""
Sanity-level coverage for the Shot-List Agent (§5.6). Same rationale as
test_concept_agent.py: the two-call justify->realize flow and its Call-A
re-prompt/fallback paths won't fire reliably against the real model on demand,
so they're exercised with a fake streaming client and pre-programmed JSON.

Deliberately not exhaustive: per docs/BUILD_TASKS.md this module gets an
independent second stress pass (the codebase's "never self-grade" posture), so
these prove the core flow is wired correctly, not every adversarial edge.
"""
from __future__ import annotations

import json

import pytest

from agents.shot_list_agent import (
    NEGATIVE_PROMPT_BOILERPLATE,
    _default_validate_justifications,
    generate_shot_list,
)
from graph.shot_schema import validate_shot_list
from tests._fakes import FakeOpenAIClient

TRUTHS = [
    {"truth_id": "t1", "fact": "matte black anodized aluminum body", "category": "material", "source": "photo_1"},
    {"truth_id": "t2", "fact": "dual knurled hinge with brass end caps", "category": "construction_detail", "source": "photo_2"},
    {"truth_id": "t3", "fact": "faint scuff on the base plate cutout", "category": "imperfection", "source": "photo_1"},
]

WINNING_SCRIPT = {
    "text": "Your phone slides off every stand you own. This one grips with a dual knurled hinge. Tap the link to grab yours today.",
    "beats": [
        {"t_start": 0, "t_end": 3, "line": "Your phone slides off every stand you own."},
        {"t_start": 3, "t_end": 8, "line": "This one grips with a dual knurled hinge."},
        {"t_start": 8, "t_end": 15, "line": "Tap the link to grab yours today."},
    ],
    "source_variant_ids": ["v1"],
}

TREATMENT = {
    "director_persona": "precise product minimalist",
    "color_story": "cool graphite tones, soft key light, seamless neutral backdrop",
    "pacing_philosophy": "quick hook, one clean proof, decisive cta",
    "beat_treatments": [
        {"beat_index": 0, "beat_function": "hook", "script_quote": "Your phone slides off every stand you own.",
         "truth_fact_id": "t1", "visual_approach": "tight hero on the matte body as a phone slips", "why_not_generic": "names the real matte body"},
        {"beat_index": 1, "beat_function": "proof", "script_quote": "This one grips with a dual knurled hinge.",
         "truth_fact_id": "t2", "visual_approach": "macro push on the knurled hinge gripping", "why_not_generic": "the specific hinge"},
        {"beat_index": 2, "beat_function": "cta", "script_quote": "Tap the link to grab yours today.",
         "truth_fact_id": "t1", "visual_approach": "endcard with product centered", "why_not_generic": "real product endcard"},
    ],
}


def _justif(shot_id, beat_role, quote, tid, ref):
    return {"shot_id": shot_id, "beat_role": beat_role, "script_quote": quote, "truth_fact_id": tid, "treatment_ref": ref}


THREE_GOOD_JUSTIFS = [
    _justif("s1", "hook", "Your phone slides off every stand you own.", "t1", 0),
    _justif("s2", "proof", "This one grips with a dual knurled hinge.", "t2", 1),
    _justif("s3", "cta", "Tap the link to grab yours today.", "t1", 2),
]


def _call_a(justifs):
    return json.dumps({"shots": justifs})


def _call_b(shot_ids):
    return json.dumps(
        {
            "lighting": "cool graphite tones, soft key light, seamless neutral backdrop",
            "shots": [
                {
                    "shot_id": sid,
                    "shot_type": "macro_detail",
                    "camera_move": "push_in",
                    "framing": "fills_frame",
                    "text_overlay_zone": "none",
                    "duration_sec": 4,
                    "voiceover_line": "line for " + sid,
                    "description": (
                        "Matte black anodized aluminum body fills the frame as a slow push-in arrives on the "
                        "dual knurled hinge with brass end caps. The camera eases forward over the graphite "
                        "surface, soft key light raking across the knurling, seamless neutral backdrop behind. "
                        "Composition centered, calm and premium mood, crisp commercial quality. Preserve product "
                        "shape, keep label text, keep proportions, product stays centered, never leaves frame."
                    ),
                    "negative_prompt_extra": "" if sid != "s2" else "smudged brass",
                }
                for sid in shot_ids
            ],
        }
    )


# ---------------------------------------------------------------------------
# The stand-in validator, exercised directly (no network).
# ---------------------------------------------------------------------------
def test_default_validator_passes_grounded_justifications():
    results = _default_validate_justifications(THREE_GOOD_JUSTIFS, WINNING_SCRIPT, TRUTHS, TREATMENT)
    assert all(r["passed"] for r in results)
    assert [r["shot_id"] for r in results] == ["s1", "s2", "s3"]


def test_default_validator_flags_each_failure_type():
    bad = [
        _justif("s1", "hook", "a line never in the script at all", "t1", 0),   # non-verbatim
        _justif("s2", "proof", "This one grips with a dual knurled hinge.", "t9", 1),  # unknown truth
        _justif("s3", "cta", "Tap the link to grab yours today.", "t1", 99),   # unknown treatment_ref
        _justif("s4", "demo", "show the product clearly", "t1", 0),            # stoplist/too-short
    ]
    results = _default_validate_justifications(bad, WINNING_SCRIPT, TRUTHS, TREATMENT)
    assert [r["passed"] for r in results] == [False, False, False, False]
    assert "verbatim" in results[0]["violation"]
    assert "t9" in results[1]["violation"]
    assert "treatment_ref" in results[2]["violation"]
    assert results[3]["violation"]  # generic phrase / too short


# ---------------------------------------------------------------------------
# Full two-call flow.
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_happy_path_produces_valid_shot_list():
    client = FakeOpenAIClient([_call_a(THREE_GOOD_JUSTIFS), _call_b(["s1", "s2", "s3"])])

    shots = await generate_shot_list(WINNING_SCRIPT, TREATMENT, TRUTHS, client=client)

    assert client.call_count == 2, "happy path = exactly Call A + Call B, no re-prompt"
    validate_shot_list(shots)  # must pass the frozen C3 schema unmodified
    assert [s["shot_id"] for s in shots] == ["s1", "s2", "s3"]
    # grounding carried through from Call A's justification
    assert shots[0]["justification"]["truth_fact_id"] == "t1"
    # reference_image_id follows the cited truth's photo (t2 came from photo_2)
    assert shots[1]["reference_image_id"] == "photo_2"
    # shared identity-first negative prompt applied, per-shot extra appended
    assert shots[0]["negative_prompt"] == NEGATIVE_PROMPT_BOILERPLATE
    assert shots[1]["negative_prompt"].startswith(NEGATIVE_PROMPT_BOILERPLATE)
    assert "smudged brass" in shots[1]["negative_prompt"]
    # allocated_budget is the explicit placeholder the Budget Gate overwrites
    assert all(s["allocated_budget"] == 0.0 for s in shots)
    # timeline tiles contiguously by duration
    assert shots[0]["t_start"] == 0.0
    assert shots[1]["t_start"] == shots[0]["t_end"]
    # no product_category smuggled in anywhere
    assert all("product_category" not in s for s in shots)


@pytest.mark.asyncio
async def test_call_a_reprompt_fires_and_repairs_bad_justification():
    bad = [
        THREE_GOOD_JUSTIFS[0],
        _justif("s2", "proof", "This one grips with a dual knurled hinge.", "t9", 1),  # unknown truth id
        THREE_GOOD_JUSTIFS[2],
    ]
    fixed = THREE_GOOD_JUSTIFS  # retry returns the corrected full list
    client = FakeOpenAIClient([_call_a(bad), _call_a(fixed), _call_b(["s1", "s2", "s3"])])

    shots = await generate_shot_list(WINNING_SCRIPT, TREATMENT, TRUTHS, client=client)

    assert client.call_count == 3, "one Call-A re-prompt then Call B"
    validate_shot_list(shots)
    assert shots[1]["justification"]["truth_fact_id"] == "t2"


@pytest.mark.asyncio
async def test_persistent_bad_justification_falls_back_to_treatment_beat():
    # Both Call A attempts return the same unfixable shot -> second failure must
    # fall back to the treatment beat (grounded by construction), never dropped.
    bad_shot = _justif("s2", "proof", "a completely invented line not in the script", "t9", 1)
    bad = [THREE_GOOD_JUSTIFS[0], bad_shot, THREE_GOOD_JUSTIFS[2]]
    client = FakeOpenAIClient([_call_a(bad), _call_a(bad), _call_b(["s1", "s2", "s3"])])

    shots = await generate_shot_list(WINNING_SCRIPT, TREATMENT, TRUTHS, client=client)

    validate_shot_list(shots)
    assert len(shots) == 3, "the failing shot is repaired, not dropped"
    # fallback lifted beat_index-1's grounded justification verbatim
    fb = shots[1]["justification"]
    assert fb["script_quote"] == "This one grips with a dual knurled hinge."
    assert fb["truth_fact_id"] == "t2"
    assert fb["treatment_ref"] == 1


@pytest.mark.asyncio
async def test_out_of_enum_camera_move_is_snapped_not_rejected():
    call_b = json.loads(_call_b(["s1", "s2", "s3"]))
    call_b["shots"][0]["camera_move"] = "zoom_and_spin"  # not a valid CameraMove
    client = FakeOpenAIClient([_call_a(THREE_GOOD_JUSTIFS), json.dumps(call_b)])

    shots = await generate_shot_list(WINNING_SCRIPT, TREATMENT, TRUTHS, client=client)

    validate_shot_list(shots)  # would raise if the bad enum leaked through
    assert shots[0]["camera_move"] == "static", "invalid camera_move snaps to the safe default"


@pytest.mark.asyncio
async def test_node_wrapper_reads_state_and_appends_trace(monkeypatch):
    # The node builds its own AsyncOpenAI internally, so patch the constructor
    # with the shared fake factory (own_client path), like the Phase 1 node tests.
    import agents.shot_list_agent as mod
    from tests._fakes import make_fake_async_openai

    monkeypatch.setattr(
        mod, "AsyncOpenAI", make_fake_async_openai([_call_a(THREE_GOOD_JUSTIFS), _call_b(["s1", "s2", "s3"])])
    )
    state = {
        "winning_script": WINNING_SCRIPT,
        "treatment": TREATMENT,
        "product_truths": TRUTHS,
        "reasoning_trace": "prior.",
    }

    out = await mod.shot_list_agent_node(state)

    assert "shot_list" in out and len(out["shot_list"]) == 3
    assert out["reasoning_trace"].startswith("prior.")
    assert "[shot_list_agent]" in out["reasoning_trace"]
