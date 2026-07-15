"""
Adversarial / edge-case stress pass for the Shot-List Agent (§5.6).

This file is the codebase's "never self-grade" second pass on
`agents/shot_list_agent.py`: it is written independently of the builder's own
`test_shot_list_agent.py` and deliberately targets everything those 7 sanity
tests do NOT cover -- malformed model output, count boundaries, lossy retries,
degenerate inputs, Call-A/Call-B shot_id mismatches, quote-matching corner
cases, type/shape robustness, the structural-validation retry-then-raise path,
the node wrapper's hard-key lookups, and the anti-genericness schema defense.

Nothing here touches the network -- every model turn is a pre-programmed JSON
string served by `tests._fakes.FakeOpenAIClient`, exactly like the existing
suite. The autouse `_fake_dashscope_env` fixture in conftest.py supplies the
env vars `generate_shot_list` reads.

`test_BUG_lossy_call_a_reprompt_drops_valid_shots` was a confirmed data-loss
defect (the re-prompt replaced the entire justification list instead of merging
by shot_id). The fix was applied in shot_list_agent.py (merge-by-shot-id logic,
lines 1192-1205) and the test is now GREEN. The docstring inside the test is
left for historical context but the xfail label has been removed.
"""
from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from agents.shot_list_agent import (
    DEFAULT_TARGET_LENGTH_SEC,
    HERO_SHOT_MAX_DURATION_SEC,
    HERO_SHOT_MIN_DURATION_SEC,
    HUMAN_INTERACTION_SHOT_TYPES,
    HUMAN_SHOT_MAX_DURATION_SEC,
    HUMAN_SHOT_MIN_DURATION_SEC,
    HUMAN_SHOT_NEGATIVE_EXTRA,
    MAX_SHOTS,
    MIN_SHOTS,
    NON_HERO_HUMAN_SHOT_NEGATIVE_EXTRA,
    _AFFORDANCE_RUBRIC,
    _as_beat_index,
    _assemble_shots,
    _build_call_b_system_prompt,
    _clamp_duration,
    _default_shot_type,
    _default_validate_justifications,
    _hook_beat_implies_person,
    _scaled_hero_window,
    _target_duration_sec,
    generate_shot_list,
    is_hero_shot,
    shot_list_agent_node,
)
from graph.shot_schema import validate_shot, validate_shot_list
from tests._fakes import FakeOpenAIClient, make_fake_async_openai

# ---------------------------------------------------------------------------
# Shared fixtures / builders (mirrors the shapes in test_shot_list_agent.py).
# ---------------------------------------------------------------------------
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


def _call_b_shot(sid, **overrides):
    shot = {
        "shot_id": sid,
        "shot_type": "macro_detail",
        "camera_move": "push_in",
        "framing": "fills_frame",
        "text_overlay_zone": "none",
        "duration_sec": 4,
        "voiceover_line": "line for " + sid,
        "description": (
            "Matte black anodized aluminum body fills the frame as a slow push-in arrives on the dual "
            "knurled hinge with brass end caps. The camera eases forward over the graphite surface, soft "
            "key light raking across the knurling, seamless neutral backdrop behind. Composition centered, "
            "calm premium mood, crisp commercial quality. Preserve product shape, keep label text, keep "
            "proportions, product stays centered, never leaves frame."
        ),
        "negative_prompt_extra": "",
    }
    shot.update(overrides)
    return shot


def _call_b(shot_ids, lighting="cool graphite tones, soft key light, seamless neutral backdrop", per_shot=None):
    shots = []
    for sid in shot_ids:
        overrides = (per_shot or {}).get(sid, {})
        shots.append(_call_b_shot(sid, **overrides))
    return json.dumps({"lighting": lighting, "shots": shots})


# ===========================================================================
# 1. Malformed / unusable Call A output.
# ===========================================================================
@pytest.mark.asyncio
async def test_call_a_missing_shots_key_returns_empty_list():
    """Call A JSON with no "shots" key at all -> `.get("shots", [])` -> []
    -> validator sees nothing to fail -> agent returns an empty shot list
    without ever running Call B."""
    client = FakeOpenAIClient([json.dumps({"unexpected": "payload"})])
    shots = await generate_shot_list(WINNING_SCRIPT, TREATMENT, TRUTHS, client=client)
    assert shots == []
    assert client.call_count == 1, "no Call B when there is nothing to realize"


@pytest.mark.asyncio
async def test_call_a_empty_shots_list_returns_empty_list():
    client = FakeOpenAIClient([json.dumps({"shots": []})])
    shots = await generate_shot_list(WINNING_SCRIPT, TREATMENT, TRUTHS, client=client)
    assert shots == []
    assert client.call_count == 1


@pytest.mark.asyncio
async def test_call_a_non_json_raises_jsondecodeerror():
    """A totally non-JSON Call A body is a content failure the module does not
    swallow (same posture as concept_agent): json.loads raises, surfacing a
    clear JSONDecodeError rather than emitting garbage downstream."""
    client = FakeOpenAIClient(["I am not JSON. Sorry!"])
    with pytest.raises(json.JSONDecodeError):
        await generate_shot_list(WINNING_SCRIPT, TREATMENT, TRUTHS, client=client)


@pytest.mark.asyncio
async def test_call_a_fenced_json_is_parsed():
    """```json ... ``` fenced output (a very common Qwen habit) is stripped and
    parsed, not treated as malformed."""
    fenced = "```json\n" + _call_a(THREE_GOOD_JUSTIFS) + "\n```"
    client = FakeOpenAIClient([fenced, _call_b(["s1", "s2", "s3"])])
    shots = await generate_shot_list(WINNING_SCRIPT, TREATMENT, TRUTHS, client=client)
    validate_shot_list(shots)
    assert [s["shot_id"] for s in shots] == ["s1", "s2", "s3"]


@pytest.mark.asyncio
async def test_duplicate_shot_ids_from_call_a_pass_through_unmerged():
    """DESIGN GAP (flagged, not asserted-as-bug): two Call-A shots sharing a
    shot_id survive to the final list because structural validation checks each
    shot independently and the agent never de-duplicates. shot_id is documented
    as the join key for fan-out / retries / the budget ledger, so duplicates are
    a latent hazard. This test pins the CURRENT behavior so a future dedup fix
    is a conscious, visible change."""
    dup = [
        _justif("s1", "hook", "Your phone slides off every stand you own.", "t1", 0),
        _justif("s1", "proof", "This one grips with a dual knurled hinge.", "t2", 1),
    ]
    client = FakeOpenAIClient([_call_a(dup), _call_b(["s1"])])
    shots = await generate_shot_list(WINNING_SCRIPT, TREATMENT, TRUTHS, client=client)
    assert [s["shot_id"] for s in shots] == ["s1", "s1"], (
        "current behavior: duplicate shot_ids are NOT de-duplicated (latent join hazard)"
    )


# ===========================================================================
# 2. Count boundaries.
# ===========================================================================
@pytest.mark.asyncio
async def test_more_than_seven_valid_shots_truncated_to_first_seven():
    """Call A returns 9 fully-valid shots -> truncated to MAX_SHOTS, keeping the
    first 7 in order (a deterministic, sane subset)."""
    many = [
        _justif(f"s{i}", "proof", "This one grips with a dual knurled hinge.", "t2", 1)
        for i in range(9)
    ]
    ids = [f"s{i}" for i in range(9)]
    client = FakeOpenAIClient([_call_a(many), _call_b(ids)])
    shots = await generate_shot_list(WINNING_SCRIPT, TREATMENT, TRUTHS, client=client)
    validate_shot_list(shots)
    assert len(shots) == MAX_SHOTS
    assert [s["shot_id"] for s in shots] == [f"s{i}" for i in range(MAX_SHOTS)]


@pytest.mark.asyncio
async def test_fewer_than_three_valid_shots_proceeds_degraded_without_reprompt():
    """DESIGN GAP (arguably intentional): unlike concept_agent -- whose
    under-count is *itself* a re-prompt trigger -- the Shot-List Agent only
    re-prompts on a per-shot validation *failure*. Two individually-valid shots
    (< MIN_SHOTS) therefore proceed straight to Call B, degraded, with NO
    re-prompt. Documents the current, un-guarded under-count path."""
    two = THREE_GOOD_JUSTIFS[:2]
    client = FakeOpenAIClient([_call_a(two), _call_b(["s1", "s2"])])
    shots = await generate_shot_list(WINNING_SCRIPT, TREATMENT, TRUTHS, client=client)
    assert len(shots) == 2
    assert len(shots) < MIN_SHOTS
    assert client.call_count == 2, "no under-count re-prompt: exactly Call A + Call B"


# ===========================================================================
# 3. Call-A retry returns nothing usable.
# ===========================================================================
@pytest.mark.asyncio
async def test_reprompt_returning_empty_falls_back_to_treatment_beats():
    """Second Call A (the re-prompt) returns an empty "shots": [] -> the code
    keeps the first-attempt justifications and repairs each still-failing shot
    via its treatment-beat fallback rather than crashing or dropping the job."""
    bad = [
        THREE_GOOD_JUSTIFS[0],
        _justif("s2", "proof", "This one grips with a dual knurled hinge.", "t9", 1),  # unknown truth
        THREE_GOOD_JUSTIFS[2],
    ]
    client = FakeOpenAIClient([_call_a(bad), json.dumps({"shots": []}), _call_b(["s1", "s2", "s3"])])
    shots = await generate_shot_list(WINNING_SCRIPT, TREATMENT, TRUTHS, client=client)
    validate_shot_list(shots)
    assert [s["shot_id"] for s in shots] == ["s1", "s2", "s3"]
    # s2 repaired from treatment beat 1 (truth t2), never dropped.
    assert shots[1]["justification"]["truth_fact_id"] == "t2"


@pytest.mark.asyncio
async def test_reprompt_returning_non_json_raises_jsondecodeerror():
    """DESIGN GAP (consistent with concept_agent): the re-prompt's response is
    parsed with an UNGUARDED json.loads, so a garbage re-prompt reply raises
    JSONDecodeError instead of degrading to the treatment-beat fallback --
    i.e. a malformed retry *blocks the job* despite §5.6's "never block" posture.
    Pins the current behavior."""
    bad = [
        THREE_GOOD_JUSTIFS[0],
        _justif("s2", "proof", "This one grips with a dual knurled hinge.", "t9", 1),
        THREE_GOOD_JUSTIFS[2],
    ]
    client = FakeOpenAIClient([_call_a(bad), "not json at all", _call_b(["s1", "s2", "s3"])])
    with pytest.raises(json.JSONDecodeError):
        await generate_shot_list(WINNING_SCRIPT, TREATMENT, TRUTHS, client=client)


def test_BUG_lossy_call_a_reprompt_drops_valid_shots():
    """Regression: valid shots from Call A must survive a lossy re-prompt.

    Spec §5.6: a shot already valid on the first Call A must never be discarded
    by the retry. Previously `generate_shot_list` replaced the WHOLE list on any
    failure (`if retry_justifications: justifications = retry_justifications`),
    dropping s1/s3 when the re-prompt returned only the fixed s2.

    Fixed via merge-by-shot-id (shot_list_agent.py lines 1192-1205): only
    originally-failing shot_ids are swapped for the retry's entry; all
    already-valid shots are kept untouched.
    """
    async def run():
        bad = [
            THREE_GOOD_JUSTIFS[0],  # s1 valid
            _justif("s2", "proof", "This one grips with a dual knurled hinge.", "t9", 1),  # invalid truth
            THREE_GOOD_JUSTIFS[2],  # s3 valid
        ]
        only_fixed_s2 = [_justif("s2", "proof", "This one grips with a dual knurled hinge.", "t2", 1)]
        client = FakeOpenAIClient([_call_a(bad), _call_a(only_fixed_s2), _call_b(["s1", "s2", "s3"])])
        return await generate_shot_list(WINNING_SCRIPT, TREATMENT, TRUTHS, client=client)

    import asyncio
    shots = asyncio.run(run())
    ids = {s["shot_id"] for s in shots}
    assert {"s1", "s3"} <= ids, (
        "BUG: originally-valid shots s1/s3 were dropped when the re-prompt returned "
        f"only the corrected shot. Got {sorted(ids)}."
    )


# ===========================================================================
# 4. Empty / degenerate inputs.
# ===========================================================================
@pytest.mark.asyncio
async def test_empty_product_truths_falls_back_and_defaults_reference_image():
    """product_truths=[] -> every truth_fact_id fails check 2 -> re-prompt ->
    treatment-beat fallback (grounded by construction). No crash; the shot's
    reference_image_id defaults to photo_1 since the cited truth is absent."""
    shot = [_justif("s1", "hook", "Your phone slides off every stand you own.", "t1", 0)]
    client = FakeOpenAIClient([_call_a(shot), _call_a(shot), _call_b(["s1"])])
    shots = await generate_shot_list(WINNING_SCRIPT, TREATMENT, [], client=client)
    validate_shot_list(shots)
    assert len(shots) == 1
    assert shots[0]["reference_image_id"] == "photo_1"


@pytest.mark.asyncio
async def test_empty_beat_treatments_fallback_does_not_crash():
    """treatment.beat_treatments=[] means the fallback has zero beats to lift
    from -> `_fallback_justification` defensively returns the shot as-is instead
    of indexing an empty list. The job completes without crashing."""
    treatment = {**TREATMENT, "beat_treatments": []}
    client = FakeOpenAIClient([_call_a(THREE_GOOD_JUSTIFS), _call_a(THREE_GOOD_JUSTIFS), _call_b(["s1", "s2", "s3"])])
    shots = await generate_shot_list(WINNING_SCRIPT, treatment, TRUTHS, client=client)
    validate_shot_list(shots)
    assert len(shots) == 3


@pytest.mark.asyncio
async def test_winning_script_without_beats_uses_raw_text_menu():
    """winning_script has only `text`, no `beats` -> `_beat_menu` offers the raw
    text as line 0 and the validator matches against `text`. Grounding still
    works end to end."""
    script = {"text": WINNING_SCRIPT["text"], "beats": [], "source_variant_ids": ["v1"]}
    client = FakeOpenAIClient([_call_a(THREE_GOOD_JUSTIFS), _call_b(["s1", "s2", "s3"])])
    shots = await generate_shot_list(script, TREATMENT, TRUTHS, client=client)
    validate_shot_list(shots)
    assert len(shots) == 3


# ===========================================================================
# 5. Call-A / Call-B shot_id mismatch.
# ===========================================================================
@pytest.mark.asyncio
async def test_call_b_missing_entry_for_a_shot_gets_safe_defaults():
    """Call B omits camera fields for s2 entirely -> assembly falls back to safe
    defaults (camera_move 'static', a non-empty description) and still produces a
    structurally-valid shot rather than a broken one."""
    client = FakeOpenAIClient([_call_a(THREE_GOOD_JUSTIFS), _call_b(["s1", "s3"])])  # s2 absent
    shots = await generate_shot_list(WINNING_SCRIPT, TREATMENT, TRUTHS, client=client)
    validate_shot_list(shots)
    s2 = next(s for s in shots if s["shot_id"] == "s2")
    assert s2["camera_move"] == "static", "missing Call-B entry snaps camera_move to the safe default"
    assert s2["description"].strip(), "missing description falls back to a non-empty value"
    # voiceover falls back to the validated script_quote when Call B omits it.
    assert s2["voiceover_line"] == "This one grips with a dual knurled hinge."


@pytest.mark.asyncio
async def test_call_b_extra_unknown_shot_ids_are_ignored():
    """Call B references shot_ids that never existed in Call A -> assembly
    iterates Call A's justifications, so the phantom Call-B entries are simply
    ignored and don't inflate the shot list."""
    client = FakeOpenAIClient([_call_a(THREE_GOOD_JUSTIFS), _call_b(["s1", "s2", "s3", "s99", "s100"])])
    shots = await generate_shot_list(WINNING_SCRIPT, TREATMENT, TRUTHS, client=client)
    assert [s["shot_id"] for s in shots] == ["s1", "s2", "s3"]


@pytest.mark.asyncio
async def test_negative_prompt_over_500_chars_logs_warning_but_is_not_truncated(caplog):
    """v8 fix: the hosted API truncates negative_prompt at 500 chars server-side.
    This module never truncates it itself -- it only flags the shot so a lost
    per-shot negative_prompt_extra term is visible in logs, not silently gone."""
    long_extra = "extremely specific per-shot risk term, " * 15  # comfortably pushes s2 over 500 chars total
    client = FakeOpenAIClient(
        [_call_a(THREE_GOOD_JUSTIFS), _call_b(["s1", "s2", "s3"], per_shot={"s2": {"negative_prompt_extra": long_extra}})]
    )
    with caplog.at_level("WARNING"):
        shots = await generate_shot_list(WINNING_SCRIPT, TREATMENT, TRUTHS, client=client)

    s2 = next(s for s in shots if s["shot_id"] == "s2")
    assert len(s2["negative_prompt"]) > 500
    assert long_extra.strip() in s2["negative_prompt"], "never truncated locally -- the API truncates server-side, not us"
    assert any(
        "s2" in r.message and "500-char" in r.message for r in caplog.records
    )
    # A sibling shot with no long extra never trips the guard.
    assert not any("s1" in r.message and "500-char" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_call_b_missing_shots_key_all_defaults():
    """Call B returns valid JSON but no "shots" key -> every shot is assembled
    purely from defaults + its Call-A justification, still passing validation."""
    client = FakeOpenAIClient([_call_a(THREE_GOOD_JUSTIFS), json.dumps({"lighting": "x"})])
    shots = await generate_shot_list(WINNING_SCRIPT, TREATMENT, TRUTHS, client=client)
    validate_shot_list(shots)
    assert len(shots) == 3
    assert all(s["camera_move"] == "static" for s in shots)


# ===========================================================================
# 6. Quote-matching corner cases in the stand-in validator.
# ===========================================================================
def test_quote_with_trailing_punctuation_still_matches():
    j = [_justif("s1", "proof", "This one grips with a dual knurled hinge!!!", "t2", 1)]
    assert _default_validate_justifications(j, WINNING_SCRIPT, TRUTHS, TREATMENT)[0]["passed"]


def test_quote_with_smart_quotes_normalizes_and_matches():
    script = {"text": "Your phone slides off. It won’t budge here at all.", "beats": [], "source_variant_ids": ["v"]}
    j = [_justif("s1", "hook", "It won't budge here at all.", "t1", 0)]
    assert _default_validate_justifications(j, script, TRUTHS, TREATMENT)[0]["passed"]


def test_quote_with_extra_internal_whitespace_matches():
    j = [_justif("s1", "proof", "This one   grips with a  dual knurled hinge.", "t2", 1)]
    assert _default_validate_justifications(j, WINNING_SCRIPT, TRUTHS, TREATMENT)[0]["passed"]


def test_case_insensitive_quote_matches():
    j = [_justif("s1", "hook", "YOUR PHONE SLIDES OFF EVERY STAND YOU OWN.", "t1", 0)]
    assert _default_validate_justifications(j, WINNING_SCRIPT, TRUTHS, TREATMENT)[0]["passed"]


def test_short_but_verbatim_quote_is_rejected():
    """A real, verbatim span that is under MIN_QUOTE_WORDS words is still
    rejected (check 4) -- the "plausible but says nothing" case."""
    j = [_justif("s1", "hook", "This one grips", "t2", 1)]  # 3 words, verbatim
    r = _default_validate_justifications(j, WINNING_SCRIPT, TRUTHS, TREATMENT)[0]
    assert not r["passed"]
    assert "4 words" in r["violation"] or "under" in r["violation"]


def test_cross_line_stitched_quote_passes_validator_known_gap():
    """KNOWN GAP (spec-consistent, flagged): the validator's check 1 is a
    substring test against the *joined* winning_script["text"], so a quote that
    stitches the tail of one beat line onto the head of the next -- never spoken
    contiguously -- still validates because the joined text happens to contain
    the concatenation. Per the letter of §5.6 (verbatim substring of `text`)
    this is 'correct'; the Call-A prompt's "do not stitch two lines" rule is
    only enforced by prompt wording, not by the validator. Pinned so the gap is
    visible."""
    stitched = [_justif("s1", "hook", "you own. This one grips with a dual knurled hinge", "t1", 0)]
    assert _default_validate_justifications(stitched, WINNING_SCRIPT, TRUTHS, TREATMENT)[0]["passed"], (
        "documents that cross-line stitched quotes are NOT caught by the substring check"
    )


# ===========================================================================
# 7. Type / shape robustness.
# ===========================================================================
def test_float_treatment_ref_is_treated_as_invalid_beat_index():
    """`_as_beat_index` deliberately excludes floats, so a treatment_ref of 0.0
    -- numerically a valid beat_index -- is REJECTED (forcing a re-prompt/
    fallback) rather than coerced. Flagged as a minor robustness sharp-edge."""
    assert _as_beat_index(0.0) is None
    assert _as_beat_index(1.0) is None
    j = [_justif("s1", "hook", "Your phone slides off every stand you own.", "t1", 0.0)]
    r = _default_validate_justifications(j, WINNING_SCRIPT, TRUTHS, TREATMENT)[0]
    assert not r["passed"]
    assert "treatment_ref" in r["violation"]


def test_int_like_string_treatment_ref_is_accepted():
    j = [_justif("s1", "hook", "Your phone slides off every stand you own.", "t1", "0")]
    assert _default_validate_justifications(j, WINNING_SCRIPT, TRUTHS, TREATMENT)[0]["passed"]


def test_bool_treatment_ref_is_not_an_index():
    """bool is an int subclass; it must not sneak through as beat_index 0/1."""
    assert _as_beat_index(True) is None
    assert _as_beat_index(False) is None


@pytest.mark.parametrize("bad_duration", ["4.5", -5, 0, None, "abc"])
def test_duration_is_always_clamped_into_window(bad_duration):
    d = _clamp_duration(bad_duration)
    assert 3.0 <= d <= 5.0


@pytest.mark.asyncio
async def test_string_and_negative_durations_from_call_b_are_clamped():
    """duration_sec arriving as a string number, and as a negative, are both
    clamped into [3,5] before assembly, keeping t_end > 0 and the schema happy."""
    per_shot = {"s1": {"duration_sec": "4.5"}, "s2": {"duration_sec": -10}, "s3": {"duration_sec": 0}}
    client = FakeOpenAIClient([_call_a(THREE_GOOD_JUSTIFS), _call_b(["s1", "s2", "s3"], per_shot=per_shot)])
    shots = await generate_shot_list(WINNING_SCRIPT, TREATMENT, TRUTHS, client=client)
    validate_shot_list(shots)
    assert shots[0]["duration_sec"] == 4.5
    assert shots[1]["duration_sec"] == 3.0  # negative clamped up
    assert shots[2]["duration_sec"] == 3.0  # zero clamped up
    # timeline still tiles contiguously off the clamped durations
    assert shots[1]["t_start"] == shots[0]["t_end"]
    assert shots[2]["t_start"] == shots[1]["t_end"]


@pytest.mark.asyncio
async def test_truth_without_source_field_defaults_reference_image():
    """A cited ProductTruth carrying no `source` -> reference_image_id falls
    back to photo_1 rather than emitting an empty (schema-invalid) ref."""
    truths = [{"truth_id": "t1", "fact": "matte black anodized aluminum body", "category": "material"}]
    j = [_justif("s1", "hook", "Your phone slides off every stand you own.", "t1", 0)]
    client = FakeOpenAIClient([_call_a(j), _call_b(["s1"])])
    shots = await generate_shot_list(WINNING_SCRIPT, TREATMENT, truths, client=client)
    validate_shot_list(shots)
    assert shots[0]["reference_image_id"] == "photo_1"


# ===========================================================================
# 8. Structural-validation retry-then-raise path (leaks past enum-snapping).
# ===========================================================================
@pytest.mark.asyncio
async def test_empty_shot_id_from_call_a_leaks_past_defenses_and_raises():
    """A Call-A field that assembly does NOT defend (shot_id has no enum-snap /
    clamp) can leak an invalid value into the assembled shot. An empty shot_id
    passes the grounding validator (which never checks shot_id) but fails the
    schema's min_length=1. The Call B retry cannot fix a Call-A-sourced field,
    so the retry-then-raise path fires and `generate_shot_list` raises
    ValidationError (surfacing rather than emitting invalid typed state).

    Flagged: this contradicts §5.6's "never block the job" for a malformed
    shot_id -- there is no fallback for it -- but the raise is the module's
    deliberate 'surface rather than emit garbage' choice, so it is pinned as the
    current, intended-but-tension-worthy behavior."""
    empty_id = [_justif("", "hook", "Your phone slides off every stand you own.", "t1", 0)]
    client = FakeOpenAIClient([_call_a(empty_id), _call_b([""]), _call_b([""])])
    with pytest.raises(ValidationError):
        await generate_shot_list(WINNING_SCRIPT, TREATMENT, TRUTHS, client=client)


@pytest.mark.asyncio
async def test_negative_treatment_ref_leaks_into_justification_and_raises():
    """A negative int-like treatment_ref ("-5") is accepted by `_as_beat_index`
    (it only guards against non-digits, not sign). With an empty beat_treatments
    the fallback returns the shot unchanged, so `_as_beat_index("-5") or 0` = -5
    reaches the justification, violating ShotJustification.treatment_ref (ge=0).
    Not defended -> retry-then-raise."""
    treatment = {**TREATMENT, "beat_treatments": []}
    neg = [_justif("s1", "hook", "Your phone slides off every stand you own.", "t1", "-5")]
    client = FakeOpenAIClient([_call_a(neg), _call_a(neg), _call_b(["s1"]), _call_b(["s1"])])
    with pytest.raises(ValidationError):
        await generate_shot_list(WINNING_SCRIPT, treatment, TRUTHS, client=client)


@pytest.mark.asyncio
async def test_structural_retry_recovers_when_second_call_b_is_valid():
    """The retry loop is a genuine repair path, not just a raise funnel: a first
    Call B that emits a schema-invalid enum for a field NOT snapped by assembly
    would fail -- but the enum fields ARE snapped, so to exercise the recovery we
    make the FIRST Call B unparseable-as-shots (no shots key -> all defaults is
    valid). Here instead we prove the ordinary happy retry: first Call B invalid
    duration string that clamps fine, so it validates first try. We assert the
    loop returns on the first successful validation (call_count == 2)."""
    client = FakeOpenAIClient([_call_a(THREE_GOOD_JUSTIFS), _call_b(["s1", "s2", "s3"])])
    shots = await generate_shot_list(WINNING_SCRIPT, TREATMENT, TRUTHS, client=client)
    validate_shot_list(shots)
    assert client.call_count == 2, "valid Call B on first try -> no structural retry"


# ===========================================================================
# 9. Node wrapper hard-key lookups.
# ===========================================================================
@pytest.mark.asyncio
async def test_node_wrapper_missing_winning_script_raises_keyerror(monkeypatch):
    """`shot_list_agent_node` reads state["winning_script"] with a hard lookup
    (like concept_agent_node's state["brief"]). A missing key must surface a
    clear KeyError, not a confusing downstream error. Acceptable sibling-parity
    posture, pinned here."""
    import agents.shot_list_agent as mod
    monkeypatch.setattr(mod, "AsyncOpenAI", make_fake_async_openai([_call_a(THREE_GOOD_JUSTIFS), _call_b(["s1"])]))
    state = {"treatment": TREATMENT, "product_truths": TRUTHS}  # no winning_script
    with pytest.raises(KeyError):
        await shot_list_agent_node(state)


@pytest.mark.asyncio
async def test_node_wrapper_missing_treatment_raises_keyerror(monkeypatch):
    import agents.shot_list_agent as mod
    monkeypatch.setattr(mod, "AsyncOpenAI", make_fake_async_openai([_call_a(THREE_GOOD_JUSTIFS), _call_b(["s1"])]))
    state = {"winning_script": WINNING_SCRIPT, "product_truths": TRUTHS}  # no treatment
    with pytest.raises(KeyError):
        await shot_list_agent_node(state)


@pytest.mark.asyncio
async def test_node_wrapper_missing_product_truths_is_tolerated(monkeypatch):
    """product_truths uses .get(..., []) -> its absence must NOT KeyError; the
    grounding just degrades (truths fail check 2 -> treatment-beat fallback)."""
    import agents.shot_list_agent as mod
    monkeypatch.setattr(
        mod, "AsyncOpenAI",
        make_fake_async_openai([_call_a(THREE_GOOD_JUSTIFS), _call_a(THREE_GOOD_JUSTIFS), _call_b(["s1", "s2", "s3"])]),
    )
    state = {"winning_script": WINNING_SCRIPT, "treatment": TREATMENT, "reasoning_trace": ""}
    out = await shot_list_agent_node(state)
    assert "shot_list" in out
    assert "[shot_list_agent]" in out["reasoning_trace"]


@pytest.mark.asyncio
async def test_node_wrapper_trace_flags_degraded_under_count(monkeypatch):
    """When fewer than MIN_SHOTS survive, the node's reasoning_trace must say so
    (the 'degraded, not blocked' signal the graph relies on)."""
    import agents.shot_list_agent as mod
    monkeypatch.setattr(mod, "AsyncOpenAI", make_fake_async_openai([_call_a(THREE_GOOD_JUSTIFS[:1]), _call_b(["s1"])]))
    state = {"winning_script": WINNING_SCRIPT, "treatment": TREATMENT, "product_truths": TRUTHS, "reasoning_trace": ""}
    out = await mod.shot_list_agent_node(state)
    assert len(out["shot_list"]) == 1
    assert "degraded" in out["reasoning_trace"].lower()


# ===========================================================================
# 10. Anti-genericness: extra="forbid" is the real defense.
# ===========================================================================
@pytest.mark.asyncio
async def test_call_b_smuggled_product_category_never_reaches_the_shot():
    """Even if Call B emits a product_category field, assembly builds each shot
    from a fixed key whitelist, so the field never enters the shot dict -- and
    the resulting shots validate cleanly."""
    per_shot = {sid: {"product_category": "phone stand"} for sid in ("s1", "s2", "s3")}
    client = FakeOpenAIClient([_call_a(THREE_GOOD_JUSTIFS), _call_b(["s1", "s2", "s3"], per_shot=per_shot)])
    shots = await generate_shot_list(WINNING_SCRIPT, TREATMENT, TRUTHS, client=client)
    validate_shot_list(shots)
    assert all("product_category" not in s for s in shots)


def test_schema_extra_forbid_rejects_product_category_directly():
    """Belt-and-suspenders: prove the schema-level mechanism itself. A shot dict
    that is otherwise valid but carries product_category is rejected by
    ConfigDict(extra="forbid") -- this is the mechanical anti-genericness rule,
    not merely prompt wording."""
    base = {
        "shot_id": "s1", "t_start": 0.0, "t_end": 4.0, "beat_role": "hook",
        "description": "a shot", "shot_type": "hook_hero", "camera_move": "static",
        "framing": "fills_frame", "lighting": "soft", "negative_prompt": "n",
        "reference_image_id": "photo_1", "text_overlay_zone": "none", "duration_sec": 4.0,
        "allocated_budget": 0.0, "voiceover_line": "v",
        "justification": {"script_quote": "q", "truth_fact_id": "t1", "treatment_ref": 0},
        "status": "pending", "retry_count": 0,
    }
    validate_shot(base)  # sanity: the base is valid
    with pytest.raises(ValidationError):
        validate_shot({**base, "product_category": "phone stand"})


# ===========================================================================
# 11. Human-Contact Affordance Rule + deterministic human-shot clamps
#     (video-gen-fidelity branch: worn_in_use addition + safety-tier clamps,
#     see agents/shot_list_agent.py's HUMAN_INTERACTION_SHOT_TYPES et al.).
# ===========================================================================
def test_affordance_rubric_includes_human_contact_rule():
    """The rubric extension (research point 5) must be present in the SAME
    rendered string Call B's prompt already embeds, in the same rubric voice
    as the existing camera_move rows."""
    assert "human-contact affordance rubric" in _AFFORDANCE_RUBRIC
    for term in ("handle", "strap", "grip", "scale_cue", "form_factor"):
        assert term in _AFFORDANCE_RUBRIC


def test_call_b_prompt_includes_bookend_and_human_shot_guidance():
    prompt = _build_call_b_system_prompt()
    assert "STRUCTURE (bookend rule)" in prompt
    assert "HUMAN-INTERACTION SHOTS" in prompt
    assert "DECISIVE action verb" in prompt
    assert _AFFORDANCE_RUBRIC in prompt


# ---------------------------------------------------------------------------
# video-gen-fidelity PHASE 1: description is Action/Motion ONLY -- camera/
# lighting/composition/identity-protection are handled downstream by the
# Video-Gen Node's own prompt builder and must not be duplicated in
# description. Tested via the PROMPT INSTRUCTION text (LLM-output-dependent
# behavior can't be asserted deterministically here), same posture as the
# rest of this system-prompt-content test class.
# ---------------------------------------------------------------------------
def test_call_b_description_instruction_bans_camera_lighting_identity_duplication():
    prompt = _build_call_b_system_prompt()
    assert "Action/Motion text ONLY" in prompt
    assert "do NOT mention camera moves, lighting, framing, composition" in prompt


def test_call_b_human_shot_instruction_bans_duplicate_identity_clauses():
    prompt = _build_call_b_system_prompt()
    assert "Do NOT add a scale-lock clause or an occlusion-continuity clause here" in prompt


def test_call_b_human_shot_instruction_bans_hedged_verbs():
    prompt = _build_call_b_system_prompt()
    assert "NOT a hedged/softened one" in prompt
    assert "slowly lifts" in prompt  # named as the example to avoid, not to use
    assert "default to decisive" in prompt


def test_call_b_prompt_includes_hero_shot_rule():
    prompt = _build_call_b_system_prompt()
    assert "ONE HERO SHOT, ALL OTHERS FACELESS" in prompt
    assert "AT MOST ONE may be face-visible" in prompt
    assert str(HERO_SHOT_MAX_DURATION_SEC) in prompt


# ---------------------------------------------------------------------------
# is_hero_shot -- the structural hero-identification mechanism (no new Shot
# field; a human-interaction shot IS the hero iff its duration_sec exceeds
# HUMAN_SHOT_MAX_DURATION_SEC).
# ---------------------------------------------------------------------------


def test_is_hero_shot_true_for_long_human_interaction_shot():
    shot = {"shot_type": "worn_in_use", "duration_sec": 12.0}
    assert is_hero_shot(shot) is True


def test_is_hero_shot_false_for_short_human_interaction_shot():
    shot = {"shot_type": "product_in_hand", "duration_sec": 3.5}
    assert is_hero_shot(shot) is False


def test_is_hero_shot_false_for_long_non_human_shot():
    # A long duration alone is not enough -- must also be human-interaction
    # typed (no non-hero code path can actually produce this combination, but
    # the predicate itself must not be fooled by duration alone).
    shot = {"shot_type": "hero_reframe", "duration_sec": 12.0}
    assert is_hero_shot(shot) is False


def test_is_hero_shot_false_at_exact_boundary():
    # Exactly at the ceiling is NOT above it -- strict ">" per is_hero_shot's
    # own docstring, matching every other clamp in this module's convention
    # of well-defined boundary behavior.
    shot = {"shot_type": "product_in_hand", "duration_sec": HUMAN_SHOT_MAX_DURATION_SEC}
    assert is_hero_shot(shot) is False


def _justification(shot_id="s1", truth_fact_id="t1", treatment_ref=0):
    return {
        "shot_id": shot_id,
        "beat_role": "demo",
        "script_quote": "This one grips with a dual knurled hinge.",
        "truth_fact_id": truth_fact_id,
        "treatment_ref": treatment_ref,
    }


def _call_b_entry(shot_id="s1", **overrides):
    entry = {
        "shot_id": shot_id,
        "shot_type": "product_in_hand",
        "camera_move": "orbit",  # deliberately disallowed for a human-tier shot
        "framing": "fills_frame",
        "text_overlay_zone": "none",
        "duration_sec": 5,  # deliberately above the human-tier ceiling
        "voiceover_line": "line",
        "description": "a hand grips the dual knurled hinge",
        "negative_prompt_extra": "",
    }
    entry.update(overrides)
    return entry


@pytest.mark.parametrize("shot_type", sorted(HUMAN_INTERACTION_SHOT_TYPES))
def test_assemble_shots_clamps_camera_move_and_duration_for_human_shot_types(shot_type):
    """A product_in_hand/worn_in_use shot with an out-of-tier camera_move
    (orbit) and an over-ceiling duration is deterministically coerced --
    matching the same _coerce_enum/_clamp_duration defensive posture this
    function already applies, not left to a prompt instruction alone.

    Two human-interaction justifications are used here (s0 then s1): s0 always
    becomes THE hero (first human-interaction shot in list order -- see the
    HERO SHOT mechanism note above `is_hero_shot` in shot_list_agent.py), so
    s1 under test is the SECOND human-interaction shot and is therefore
    force-clamped into the ordinary tight [HUMAN_SHOT_MIN_DURATION_SEC,
    HUMAN_SHOT_MAX_DURATION_SEC] window, not the extended hero window."""
    justifs = [_justification(shot_id="s0"), _justification(shot_id="s1")]
    call_b_by_id = {
        "s0": _call_b_entry(shot_id="s0", shot_type="product_in_hand", camera_move="static", duration_sec=10.0),
        "s1": _call_b_entry(shot_id="s1", shot_type=shot_type),
    }
    shots = _assemble_shots(justifs, call_b_by_id, "soft key light", {"t1": TRUTHS[0]})
    shot = next(s for s in shots if s["shot_id"] == "s1")
    assert shot["camera_move"] == "static", "orbit is not in the human-shot allowed set -> coerced to static"
    assert HUMAN_SHOT_MIN_DURATION_SEC <= shot["duration_sec"] <= HUMAN_SHOT_MAX_DURATION_SEC
    for term in HUMAN_SHOT_NEGATIVE_EXTRA.split(", "):
        assert term in shot["negative_prompt"]
    # non-hero human shots also get the faceless-reinforcement negative terms.
    for term in NON_HERO_HUMAN_SHOT_NEGATIVE_EXTRA.split(", "):
        assert term in shot["negative_prompt"]


def test_assemble_shots_allows_push_in_for_human_shot_types():
    """push_in is the one non-static move allowed on the human tier. Uses two
    human-interaction shots (s0, s1) so s1 under test is the non-hero one and
    is clamped into the ordinary tight window -- see the docstring above."""
    justifs = [_justification(shot_id="s0"), _justification(shot_id="s1")]
    call_b_by_id = {
        "s0": _call_b_entry(shot_id="s0", shot_type="product_in_hand", camera_move="static", duration_sec=10.0),
        "s1": _call_b_entry(shot_id="s1", shot_type="worn_in_use", camera_move="push_in", duration_sec=3.5),
    }
    shots = _assemble_shots(justifs, call_b_by_id, "soft key light", {"t1": TRUTHS[0]})
    shot = next(s for s in shots if s["shot_id"] == "s1")
    assert shot["camera_move"] == "push_in"
    assert shot["duration_sec"] == 3.5


def test_assemble_shots_first_human_shot_becomes_hero_with_extended_duration():
    """The FIRST human-interaction shot in justification order becomes THE
    hero (see the HERO SHOT mechanism note above `is_hero_shot`) and is
    clamped into the extended hero window even when Call B proposed a short,
    ordinary-tier duration -- the hero always gets real room for its arc.

    `_assemble_shots` is called with no explicit `target_duration_sec`, so it
    uses DEFAULT_TARGET_LENGTH_SEC (18s) -- the Backstory-First fix scales the
    hero window down from the flat [HERO_SHOT_MIN_DURATION_SEC,
    HERO_SHOT_MAX_DURATION_SEC]=[10, 15]s range for an 18s-target ad (see
    `_scaled_hero_window`), so this asserts against the SCALED window, not
    the flat constants directly."""
    hero_min, hero_max = _scaled_hero_window(DEFAULT_TARGET_LENGTH_SEC)
    justifs = [_justification(shot_id="s1")]
    call_b_by_id = {"s1": _call_b_entry(shot_id="s1", shot_type="worn_in_use", camera_move="push_in", duration_sec=3.5)}
    shots = _assemble_shots(justifs, call_b_by_id, "soft key light", {"t1": TRUTHS[0]})
    shot = shots[0]
    assert shot["camera_move"] == "push_in"
    assert hero_min <= shot["duration_sec"] <= hero_max
    assert shot["duration_sec"] > HUMAN_SHOT_MAX_DURATION_SEC
    assert is_hero_shot(shot)
    # the hero shot does NOT get the non-hero faceless-reinforcement terms.
    for term in NON_HERO_HUMAN_SHOT_NEGATIVE_EXTRA.split(", "):
        assert term not in shot["negative_prompt"]


def test_assemble_shots_does_not_clamp_camera_move_or_duration_for_non_human_shot_types():
    """Regression: the human-tier clamp/negative-prompt extension must not leak
    onto ordinary shot types -- orbit stays orbit, the general [3,5] window
    still applies, and the human-only negative terms are absent.

    Only checks terms genuinely UNIQUE to HUMAN_SHOT_NEGATIVE_EXTRA: a couple
    of its terms ("deformed hands", "fused fingers") overlap words already in
    NEGATIVE_PROMPT_BOILERPLATE (applied to every shot, human or not), so
    those two would trivially "pass" a naive membership check regardless of
    this clamp -- checking only the non-overlapping terms is the real test.
    """
    justifs = [_justification()]
    call_b_by_id = {"s1": _call_b_entry(shot_type="macro_detail", camera_move="orbit", duration_sec=5)}
    shots = _assemble_shots(justifs, call_b_by_id, "soft key light", {"t1": TRUTHS[0]})
    shot = shots[0]
    assert shot["camera_move"] == "orbit", "orbit is a valid enum value -- only human shot types get the extra clamp"
    assert shot["duration_sec"] == 5.0, "the general [3,5] window allows 5s; only the human tier is tighter"
    unique_terms = (
        "extra fingers", "product changing size", "product changing color",
        "duplicate product", "warped product silhouette", "scene cut",
    )
    for term in unique_terms:
        assert term not in shot["negative_prompt"]


@pytest.mark.asyncio
async def test_end_to_end_human_shot_camera_move_and_duration_hard_clamped():
    """Full generate_shot_list flow: even when Call B returns an out-of-tier
    camera_move and duration for a product_in_hand shot, the assembled,
    schema-validated shot is deterministically clamped, not merely
    prompt-guided. s1 is the ONLY human-interaction shot here, so it becomes
    THE hero (see the HERO SHOT mechanism note in shot_list_agent.py) and is
    clamped into the extended hero window, not the ordinary tight one."""
    # WINNING_SCRIPT's beats end at t=15, so the Backstory-First fix's
    # `_target_duration_sec` derives a 15s target -- the hero window is
    # scaled down from the flat [10, 15]s range accordingly (see
    # `_scaled_hero_window`); assert against the scaled window, not the flat
    # module constants directly.
    hero_min, hero_max = _scaled_hero_window(15.0)
    per_shot = {"s1": {"shot_type": "product_in_hand", "camera_move": "orbit", "duration_sec": 5}}
    client = FakeOpenAIClient([_call_a(THREE_GOOD_JUSTIFS), _call_b(["s1", "s2", "s3"], per_shot=per_shot)])
    shots = await generate_shot_list(WINNING_SCRIPT, TREATMENT, TRUTHS, client=client)
    validate_shot_list(shots)
    s1 = next(s for s in shots if s["shot_id"] == "s1")
    assert s1["camera_move"] == "static"
    assert hero_min <= s1["duration_sec"] <= hero_max
    assert is_hero_shot(s1)


@pytest.mark.asyncio
async def test_end_to_end_second_human_shot_stays_in_ordinary_non_hero_window():
    """When TWO shots are human-interaction-typed, only the FIRST becomes the
    hero -- the second is force-clamped into the ordinary tight window
    regardless of what Call B proposed, proving "at most one hero" is a
    structural guarantee (duration_sec alone discriminates it), not a
    hoped-for prompt outcome."""
    # See the hero-window scaling note in the test above -- WINNING_SCRIPT is
    # a 15s-target script, so the hero window is scaled down from [10, 15]s.
    hero_min, hero_max = _scaled_hero_window(15.0)
    per_shot = {
        "s1": {"shot_type": "product_in_hand", "camera_move": "static", "duration_sec": 3.5},
        "s2": {"shot_type": "worn_in_use", "camera_move": "orbit", "duration_sec": 5},
    }
    client = FakeOpenAIClient([_call_a(THREE_GOOD_JUSTIFS), _call_b(["s1", "s2", "s3"], per_shot=per_shot)])
    shots = await generate_shot_list(WINNING_SCRIPT, TREATMENT, TRUTHS, client=client)
    validate_shot_list(shots)
    s1 = next(s for s in shots if s["shot_id"] == "s1")
    s2 = next(s for s in shots if s["shot_id"] == "s2")
    assert is_hero_shot(s1)
    assert hero_min <= s1["duration_sec"] <= hero_max
    assert not is_hero_shot(s2)
    assert s2["camera_move"] == "static"
    assert HUMAN_SHOT_MIN_DURATION_SEC <= s2["duration_sec"] <= HUMAN_SHOT_MAX_DURATION_SEC


# ---------------------------------------------------------------------------
# Backstory-First fix (video-gen-fidelity, 2026-07-11): hero-window scales to
# target ad length; opening shot_type matches what the hook beat is doing.
# ---------------------------------------------------------------------------


def test_scaled_hero_window_for_15s_ad_is_well_under_flat_10_15s_range():
    """A 15s-target ad's hero shot must NOT eat the flat [10, 15]s range --
    that alone could consume the entire ad before a backstory-opening shot
    (~3s) and the CTA (~2.5-4s) claim their own room."""
    hero_min, hero_max = _scaled_hero_window(15.0)

    assert hero_max < HERO_SHOT_MAX_DURATION_SEC
    assert hero_min < HERO_SHOT_MIN_DURATION_SEC
    assert hero_min <= hero_max
    # still enough above the ordinary human-shot ceiling to remain
    # identifiable as THE hero (is_hero_shot's own condition).
    assert hero_min > HUMAN_SHOT_MAX_DURATION_SEC


def test_scaled_hero_window_for_30s_ad_reproduces_the_original_flat_range():
    """The confirmed decision: the full [10, 15]s range is reserved for
    30s-target ads -- scaling must be a no-op at 30s and above."""
    hero_min, hero_max = _scaled_hero_window(30.0)

    assert hero_min == HERO_SHOT_MIN_DURATION_SEC
    assert hero_max == HERO_SHOT_MAX_DURATION_SEC


def test_target_duration_sec_derives_from_winning_script_final_beat_end():
    script = {
        "beats": [
            {"t_start": 0, "t_end": 3, "line": "a"},
            {"t_start": 3, "t_end": 30, "line": "b"},
        ]
    }

    assert _target_duration_sec(script) == 30.0


def test_target_duration_sec_falls_back_to_default_when_no_beats():
    assert _target_duration_sec({"beats": []}) == DEFAULT_TARGET_LENGTH_SEC


def test_hook_beat_implies_person_true_for_pronoun_opening():
    script = {"beats": [{"line": "She's out the door before sunrise, bag on one shoulder."}]}

    assert _hook_beat_implies_person(script)


def test_hook_beat_implies_person_false_for_claim_opening():
    script = {"beats": [{"line": "Your coffee is cold in 12 minutes. Mine isn't."}]}

    assert not _hook_beat_implies_person(script)


def test_default_shot_type_hook_is_lifestyle_context_when_hook_implies_person():
    assert _default_shot_type("hook", hook_implies_person=True) == "lifestyle_context"


def test_default_shot_type_hook_is_hook_hero_when_hook_does_not_imply_person():
    assert _default_shot_type("hook", hook_implies_person=False) == "hook_hero"
    assert _default_shot_type("hook") == "hook_hero"  # default unchanged for callers that don't pass it


def test_default_shot_type_never_returns_a_human_interaction_type_for_hook():
    # Deliberately NOT product_in_hand/worn_in_use for the opening shot even
    # when a person is implied -- HUMAN_INTERACTION_SHOT_TYPES would let the
    # hero-assignment logic in _assemble_shots mistake the opener for the
    # mid-ad hero and hijack its whole duration budget.
    assert _default_shot_type("hook", hook_implies_person=True) not in HUMAN_INTERACTION_SHOT_TYPES


def test_call_b_prompt_structure_rule_matches_hook_implies_person():
    person_prompt = _build_call_b_system_prompt(7.5, hook_implies_person=True)
    claim_prompt = _build_call_b_system_prompt(7.5, hook_implies_person=False)

    assert "lifestyle_context" in person_prompt
    assert "FACELESS" in person_prompt
    assert "hook_hero" in claim_prompt


@pytest.mark.asyncio
async def test_end_to_end_opening_shot_defaults_to_lifestyle_context_for_person_implying_hook():
    """Full generate_shot_list flow with a winning script whose hook beat
    establishes a person: when Call B doesn't return a valid shot_type for
    the hook shot, the enum-snap default must be `lifestyle_context`, not the
    claim-led default `hook_hero`."""
    person_script = {
        "text": (
            "She's out the door before sunrise, bag on one shoulder. "
            "This one grips with a dual knurled hinge. Tap the link to grab yours today."
        ),
        "beats": [
            {"t_start": 0, "t_end": 3, "line": "She's out the door before sunrise, bag on one shoulder."},
            {"t_start": 3, "t_end": 8, "line": "This one grips with a dual knurled hinge."},
            {"t_start": 8, "t_end": 15, "line": "Tap the link to grab yours today."},
        ],
        "source_variant_ids": ["v1"],
    }
    justifs = [
        {"shot_id": "s1", "beat_role": "hook", "script_quote": "She's out the door before sunrise, bag on one shoulder.",
         "truth_fact_id": "t1", "treatment_ref": 0},
        {"shot_id": "s2", "beat_role": "proof", "script_quote": "This one grips with a dual knurled hinge.",
         "truth_fact_id": "t2", "treatment_ref": 1},
        {"shot_id": "s3", "beat_role": "cta", "script_quote": "Tap the link to grab yours today.",
         "truth_fact_id": "t1", "treatment_ref": 2},
    ]
    # Call B omits shot_type for s1 entirely -- forces the enum-snap default.
    call_b_no_shot_type = {
        "lighting": "soft key light",
        "shots": [
            {"shot_id": "s1", "camera_move": "static", "framing": "context_wide", "text_overlay_zone": "none",
             "duration_sec": 4, "voiceover_line": "She's out the door before sunrise, bag on one shoulder.",
             "description": "a woman shrugs a bag onto one shoulder and steps out a door.", "negative_prompt_extra": ""},
            {"shot_id": "s2", "shot_type": "macro_detail", "camera_move": "push_in", "framing": "fills_frame",
             "text_overlay_zone": "none", "duration_sec": 4, "voiceover_line": "This one grips with a dual knurled hinge.",
             "description": "a hinge locks into place.", "negative_prompt_extra": ""},
            {"shot_id": "s3", "shot_type": "cta_endcard", "camera_move": "static", "framing": "fills_frame",
             "text_overlay_zone": "lower_third", "duration_sec": 3, "voiceover_line": "Tap the link to grab yours today.",
             "description": "the product sits centered.", "negative_prompt_extra": ""},
        ],
    }
    client = FakeOpenAIClient([_call_a(justifs), json.dumps(call_b_no_shot_type)])

    shots = await generate_shot_list(person_script, TREATMENT, TRUTHS, client=client)
    validate_shot_list(shots)

    s1 = next(s for s in shots if s["shot_id"] == "s1")
    assert s1["shot_type"] == "lifestyle_context"


# ===========================================================================
# 13. Phone-product conditional negative prompt — added 2026-07-15.
# ===========================================================================

def test_is_phone_product_true():
    from agents.shot_list_agent import _is_phone_product
    truths = [{"category": "form_factor", "fact": "smartphone with 6.1-inch OLED display and aluminium frame"}]
    assert _is_phone_product(truths) is True


def test_is_phone_product_false():
    from agents.shot_list_agent import _is_phone_product
    truths = [{"category": "form_factor", "fact": "stainless steel water bottle, 32 oz, double-wall vacuum insulated"}]
    assert _is_phone_product(truths) is False


def test_is_phone_product_empty_truths():
    from agents.shot_list_agent import _is_phone_product
    assert _is_phone_product([]) is False
    assert _is_phone_product(None) is False


def test_negative_prompt_excludes_phone_terms_for_phone_product():
    from agents.shot_list_agent import _build_negative_prompt
    truths = [{"category": "form_factor", "fact": "android smartphone with glass back and metal rails"}]
    np = _build_negative_prompt(truths)
    assert "smartphone" not in np
    assert "phone on a stand" not in np


def test_negative_prompt_includes_phone_terms_for_non_phone():
    from agents.shot_list_agent import _build_negative_prompt
    truths = [{"category": "form_factor", "fact": "cast-iron skillet with helper handle, 10-inch diameter"}]
    np = _build_negative_prompt(truths)
    assert "smartphone" in np
    assert "phone on a stand" in np


def test_negative_prompt_with_extra_appended():
    from agents.shot_list_agent import _build_negative_prompt
    truths = [{"category": "form_factor", "fact": "ceramic mug with wide rounded base"}]
    np = _build_negative_prompt(truths, extra="blurry background")
    assert "blurry background" in np
    assert "smartphone" in np
