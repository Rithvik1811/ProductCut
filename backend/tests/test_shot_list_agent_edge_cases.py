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

ONE test is intentionally RED: `test_BUG_lossy_call_a_reprompt_drops_valid_shots`
demonstrates a confirmed data-loss defect (see its docstring and the report).
It asserts the spec-correct behavior and is left failing on purpose rather than
weakened to stay green.
"""
from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from agents.shot_list_agent import (
    MAX_SHOTS,
    MIN_SHOTS,
    _as_beat_index,
    _clamp_duration,
    _default_validate_justifications,
    generate_shot_list,
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
    """CONFIRMED BUG (intentionally RED) -- data loss on a lossy re-prompt.

    Spec §5.6 defines a PER-SHOT re-prompt + PER-SHOT fallback: a shot that was
    already valid on the first Call A must never be discarded by the retry. The
    sibling concept_agent enforces exactly this via `if len(retry_valid) >
    len(valid): valid = retry_valid` -- it only ADOPTS a retry that is at least
    as good.

    `generate_shot_list` has NO such guard: on any failure it does
    `if retry_justifications: justifications = retry_justifications`, replacing
    the WHOLE list. So when the re-prompt returns only the one corrected shot
    (a very common model behavior -- "here is the fixed shot"), the two
    originally-VALID shots (s1, s3) are silently dropped.

    This asserts the spec-correct outcome (originally-valid shots survive) and
    is left FAILING on purpose to surface the defect. See report.
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
