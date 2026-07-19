"""
Integration test for the piece the derisk scripts never exercised: do the
Product Truth Extractor AND Concept Agent actually work when run *inside*
the compiled LangGraph graph, chained -- RunnableConfig injection,
adispatch_custom_event, state flowing from one node into the next, the whole
astream_events path -- not just as bare function calls.

Fakes only the network boundary (AsyncOpenAI construction inside each agent
module); everything else (StateGraph, checkpointer, astream_events, custom
event dispatch) is the real thing.
"""
from __future__ import annotations

import json

import pytest

from graph.build import build_graph
from tests._fakes import make_content_routed_sync_openai, make_fake_async_openai
from tests._phase3_graph import (
    patch_assembly_boundaries,
    patch_continuity_boundaries,
    patch_format_export_boundaries,
    patch_phase3_boundaries,
    patch_visual_direction_boundaries,
    patch_voiceover_boundaries,
)

GOOD_FACTS = [
    ("a hairline scratch runs diagonally across the lower left corner of the lid", "material_character"),
    ("the base plate has two asymmetric ventilation slots near the rear edge", "construction_detail"),
    ("a faint discoloration ring marks where a sticker was once removed", "material_character"),
    ("the power button has a slightly recessed matte texture unlike the glossy housing", "texture"),
    ("the charging port surround shows minor oxidation on the metal contacts", "material_character"),
    ("a small manufacturer stamp is debossed near the bottom-right hinge", "construction_detail"),
    ("compact aluminum housing, 130mm wide × 80mm tall, matte charcoal finish", "form_factor"),
]

FOUR_GOOD_VARIANTS = [
    {
        "variant_id": "v1",
        "text": "Scratched already? This one shrugs it off. So tap to shop.",
        "framework": "hook_problem_product_cta",
        # Backstory-First fix (video-gen-fidelity): "pattern interrupt" is now a
        # story/curiosity hook_type judged on a human-moment marker, not a
        # number/contrast marker (agents/concept_agent.py's _STORY_HOOK_TYPES).
        # This hook line has no person in it -- it earns its pass via the
        # contrast marker "not", so "contrarian / myth-busting" (a claim-led
        # type) is the accurate label, not a re-engineered fixture.
        "hook_type": "contrarian / myth-busting",
        "emotional_trigger": "curiosity",
        "grounding_truth_ids": ["t1", "t4", "t7"],
        "beats": [{"t_start": 0, "t_end": 3, "line": "Scratched already? Not this one."},
                  {"t_start": 3, "t_end": 12, "line": "This one shrugs it off."},
                  {"t_start": 12, "t_end": 30, "line": "So tap to shop."}],
    },
    {
        "variant_id": "v2",
        "text": "Stickers leave rings. Ours won't. So tap to shop.",
        "framework": "PAS",
        "hook_type": "bold claim",
        "emotional_trigger": "FOMO",
        # Feature-Spread fix (Single-Detail Fixation, video-gen-fidelity): cited
        # truths must now span >= 2 distinct categories, so v2/v3/v4 each pair
        # an imperfection/construction truth with the texture truth (t4) instead
        # of two same-category truths.
        "grounding_truth_ids": ["t3", "t4", "t7"],
        "beats": [{"t_start": 0, "t_end": 3, "line": "Stickers leave rings, not this base."},
                  {"t_start": 3, "t_end": 12, "line": "Ours won't."},
                  {"t_start": 12, "t_end": 30, "line": "So tap to shop."}],
    },
    {
        "variant_id": "v3",
        "text": "Every detail, debossed with care. So tap to shop.",
        "framework": "AIDA",
        "hook_type": "social proof",
        "emotional_trigger": "recognition",
        "grounding_truth_ids": ["t6", "t4", "t7"],
        "beats": [{"t_start": 0, "t_end": 3, "line": "Every detail debossed, not printed."},
                  {"t_start": 3, "t_end": 12, "line": "Built to last."},
                  {"t_start": 12, "t_end": 30, "line": "So tap to shop."}],
    },
    {
        "variant_id": "v4",
        "text": "From scratch to spotless. So tap to shop.",
        "framework": "BAB",
        "hook_type": "before/after",
        "emotional_trigger": "relief",
        "grounding_truth_ids": ["t1", "t2", "t4", "t7"],
        "beats": [{"t_start": 0, "t_end": 3, "line": "From scratch, not spotless -- until now."},
                  {"t_start": 3, "t_end": 12, "line": "Built for real life."},
                  {"t_start": 12, "t_end": 30, "line": "So tap to shop."}],
    },
]


TRUTH_EXTRACTOR_PAYLOAD = json.dumps(
    {
        "same_product": True,
        "mismatch_reason": "",
        "product_truths": [
            {"truth_id": f"t{i}", "fact": fact, "category": category, "source": "photo_1"}
            for i, (fact, category) in enumerate(GOOD_FACTS, start=1)
        ],
    }
)
CONCEPT_AGENT_PAYLOAD = json.dumps({"script_variants": FOUR_GOOD_VARIANTS})

_VARIANT_IDS = [v["variant_id"] for v in FOUR_GOOD_VARIANTS]  # v1..v4

# Canned checker responses. The graph now fans concept_agent out into the five
# Critic-Chain checkers and back into the Meta-Critic, so even a test that only
# cares about the upstream two agents must mock the whole chain's network
# boundary or the downstream checkers hit the real DashScope endpoint.
HOOK_PAYLOAD = json.dumps(
    {
        "hook_scores": [
            {"variant_id": vid, "hook_score": 4, "justification": f"{vid} hook is fine"}
            for vid in _VARIANT_IDS
        ]
    }
)
BODY_PAYLOAD = json.dumps(
    {
        "results": [
            {
                "variant_id": vid,
                "completion_score": 4,
                "redundant_beat_pairs": [],
                "promise_payoff_match": True,
                "emotional_trigger_landed": True,
                "justification": f"{vid} body pays off the hook",
            }
            for vid in _VARIANT_IDS
        ]
    }
)
CTA_PAYLOAD = json.dumps(
    {
        "results": [
            {"variant_id": vid, "cta_score": 4, "justification": f"{vid} cta is clear"}
            for vid in _VARIANT_IDS
        ]
    }
)
TONE_PAYLOAD = json.dumps(
    {
        "results": [
            {
                "variant_id": vid,
                "tone_score": 4,
                "never_do_violation": False,
                "justification": f"{vid} is on brand",
            }
            for vid in _VARIANT_IDS
        ]
    }
)
# Valid MetaCriticLLMOutput cross-pollinating hook v1 / body v2 / cta v3.
META_PAYLOAD = json.dumps(
    {
        "leaderboards": [
            {"axis": "hook", "ranked_variant_ids": ["v1", "v2", "v3", "v4"], "note": "hook order"},
            {"axis": "completion", "ranked_variant_ids": ["v2", "v1", "v3", "v4"], "note": "body order"},
            {"axis": "cta", "ranked_variant_ids": ["v3", "v1", "v2", "v4"], "note": "cta order"},
        ],
        "audition": {
            "promise_payoff": "v2 body develops v1's hook claim",
            "hook_body_seam": "'v1 hook' -> 'v2 body': continuous",
            "body_cta_seam": "'v2 body' -> 'v3 cta': continuous",
            "trigger_continuity": "curiosity escalates into the ask",
            "passed": True,
            "risks_to_flag_forward": [],
        },
        "substitutions": [],
        "hook_source_variant_id": "v1",
        "body_source_variant_id": "v2",
        "cta_source_variant_id": "v3",
        "no_compatible_merge": False,
        "rationale": [
            {
                "axis": "hook",
                "decision": "hook from v1",
                "quoted_evidence": "'v1 hook' scored 4",
                "steelmanned_runner_up": "v2's hook is punchy too",
                "trade_off": "gave up v2's opener",
                "why_it_holds": "audition passed the hook-body seam",
                "named_risk": "hook-body seam",
            },
            {
                "axis": "body",
                "decision": "body from v2",
                "quoted_evidence": "'v2 body' scored 4",
                "steelmanned_runner_up": "v1's body is coherent",
                "trade_off": "gave up v1's body",
                "why_it_holds": "audition passed promise-payoff",
                "named_risk": "body beat 1",
            },
            {
                "axis": "cta",
                "decision": "cta from v3",
                "quoted_evidence": "'v3 cta' scored 4",
                "steelmanned_runner_up": "v1's cta is clean",
                "trade_off": "gave up v1's cta",
                "why_it_holds": "audition passed the body-cta seam",
                "named_risk": "body-cta seam",
            },
        ],
        "overall_reasoning": "cross-pollinated the strongest hook, body and cta across survivors",
    }
)
# (needle in the checker's SYSTEM prompt, canned response). Shared by the
# full-chain test in test_graph_critic_chain.py.
CHECKER_ROUTES = [
    ("META-CRITIC", META_PAYLOAD),
    ("COLD read of the BODY", BODY_PAYLOAD),
    # CTA Bridge fix: the CTA-Checker's prompt opening no longer ends on
    # "...clarity" verbatim (it now scores clarity AND earned-close/bridge
    # quality) -- "call-to-action (CTA)" alone is still a unique, stable needle
    # (the Tone-Checker's prompt never mentions "call-to-action").
    ("call-to-action (CTA)", CTA_PAYLOAD),
    ("brand-tone critic", TONE_PAYLOAD),
]

# ---------------------------------------------------------------------------
# Phase 2 fakes (Treatment Agent §5.5, Shot-List Agent §5.6). merge_validator
# now routes BOTH "finalize" and "fallback" into treatment_agent -> shot_list_agent
# -> budget_gate, so every graph test that drives the chain that far must fake
# these two agents' network boundary or they hit the real DashScope endpoint.
#
# WINNING-SCRIPT NOTE. FOUR_GOOD_VARIANTS gives each variant a single body beat,
# so meta_critic's merge candidate fails the Merge Coherence Validator's pacing
# re-check unrepairably and merge_validator routes via "fallback" -- the winning
# script is then variant v1's OWN, unmerged text/beats (fallback = the single
# highest-composite variant; all four tie on composite here, so the lexicographic
# tiebreak picks v1). The Justification Validator (KR's real one, shared by both
# agents) checks each script_quote against winning_script["text"], NOT against a
# beat's own line -- and v1's text ("Scratched already? This one shrugs it off.
# So tap to shop.") is NOT the concatenation of its beat lines. So every quote below
# is a verbatim substring of v1's TEXT, which is what actually gets validated.
_WINNING_TEXT = FOUR_GOOD_VARIANTS[0]["text"]  # v1's text (the fallback winner)

# Treatment: exactly one beat_treatment per winning-script beat (3), beat_index in
# order, valid 5-value beat_function, a real truth_id (t1..t6), and no banned
# "category" word / generic stoplist phrase in the free-text fields.
TREATMENT_PAYLOAD = json.dumps(
    {
        "director_persona": "intimate, handheld warmth that lingers on worn detail",
        "color_story": "matte graphite housing under warm tungsten highlights",
        "pacing_philosophy": "a slow, curious build that snaps shut on the ask",
        "beat_treatments": [
            {
                "beat_index": 0,
                "beat_function": "hook",
                "script_quote": "Scratched already?",
                "truth_fact_id": "t1",
                "visual_approach": "static macro hold on the diagonal hairline scratch, no camera movement",
                "why_not_generic": "This exact diagonal scratch lives on THIS lid; no stock opener could show it.",
            },
            {
                "beat_index": 1,
                "beat_function": "demo",
                "script_quote": "This one shrugs it off.",
                "truth_fact_id": "t2",
                "visual_approach": "slow partial orbit revealing the two asymmetric rear vent slots",
                "why_not_generic": "The asymmetric slot placement is specific to this base plate, not a generic vent.",
            },
            {
                "beat_index": 2,
                "beat_function": "cta",
                "script_quote": "So tap to shop.",
                "truth_fact_id": "t3",
                "visual_approach": "quick push onto the debossed hinge stamp, then hold on the endcard",
                "why_not_generic": "The debossed stamp is a real mark on this unit, grounding the closing ask.",
            },
        ],
    }
)

# Shot-List Call A ("Justify"): 3 shots, each quoting a verbatim span of v1's TEXT,
# a real truth_id, and a treatment_ref that is a real beat_index (0/1/2) from the
# TREATMENT_PAYLOAD above -- so KR's real Justification Validator passes them all.
SHOT_LIST_CALL_A_PAYLOAD = json.dumps(
    {
        "shots": [
            {
                "shot_id": "s1",
                "beat_role": "hook",
                "script_quote": "Scratched already?",
                "truth_fact_id": "t1",
                "treatment_ref": 0,
            },
            {
                "shot_id": "s2",
                "beat_role": "demo",
                "script_quote": "This one shrugs it off.",
                "truth_fact_id": "t2",
                "treatment_ref": 1,
            },
            {
                "shot_id": "s3",
                "beat_role": "cta",
                "script_quote": "So tap to shop.",
                "truth_fact_id": "t3",
                "treatment_ref": 2,
            },
        ]
    }
)

# Shot-List Call B ("Realize"): one entry per Call-A shot_id, all enum values drawn
# from graph.shot_schema, duration_sec in [3, 5]. These assemble into full Shots
# that pass the structural validate_shot_list (no product_category field anywhere).
SHOT_LIST_CALL_B_PAYLOAD = json.dumps(
    {
        "lighting": "warm tungsten key with a soft graphite falloff, held constant across every shot",
        "shots": [
            {
                "shot_id": "s1",
                "shot_type": "hook_hero",
                "camera_move": "push_in",
                "framing": "fills_frame",
                "text_overlay_zone": "none",
                "duration_sec": 3,
                "voiceover_line": "Scratched already?",
                "description": (
                    "Matte graphite lid fills the frame, the diagonal hairline scratch catching "
                    "warm tungsten light. Slow push-in on the mark, no cuts. Preserve product "
                    "shape, keep label text, keep proportions."
                ),
                "negative_prompt_extra": "",
            },
            {
                "shot_id": "s2",
                "shot_type": "macro_detail",
                "camera_move": "static",
                "framing": "fills_frame",
                "text_overlay_zone": "none",
                "duration_sec": 4,
                "voiceover_line": "This one shrugs it off.",
                "description": (
                    "Extreme macro on the two asymmetric rear vent slots of the base plate, held "
                    "steady under the same warm key. Preserve product shape, keep label text, keep "
                    "proportions."
                ),
                "negative_prompt_extra": "",
            },
            {
                "shot_id": "s3",
                "shot_type": "cta_endcard",
                "camera_move": "static",
                "framing": "fills_frame",
                "text_overlay_zone": "lower_third",
                "duration_sec": 3,
                "voiceover_line": "So tap to shop.",
                "description": (
                    "The debossed hinge stamp resolves into a clean endcard, lower third reserved "
                    "for the composited CTA. Preserve product shape, keep label text, keep "
                    "proportions."
                ),
                "negative_prompt_extra": "",
            },
        ],
    }
)


@pytest.mark.asyncio
async def test_truth_extractor_and_concept_agent_run_chained_in_graph(monkeypatch):
    monkeypatch.setattr(
        "agents.product_truth_extractor.AsyncOpenAI",
        make_fake_async_openai([TRUTH_EXTRACTOR_PAYLOAD]),
    )
    monkeypatch.setattr(
        "agents.concept_agent.AsyncOpenAI",
        make_fake_async_openai([CONCEPT_AGENT_PAYLOAD]),
    )
    # The critic chain is now wired into the graph, so running the graph fans out
    # into all five checkers + the Meta-Critic. Mock their network boundary too,
    # or the downstream checkers hit the real endpoint (Hook uses its own async
    # client; Body/CTA/Tone/Meta share critic_llm's sync OpenAI).
    monkeypatch.setattr(
        "agents.hook_checker.AsyncOpenAI",
        make_fake_async_openai([HOOK_PAYLOAD]),
    )
    monkeypatch.setattr(
        "agents.critic_llm.OpenAI",
        make_content_routed_sync_openai(CHECKER_ROUTES),
    )
    # Phase 2 is now wired in after merge_validator (both "finalize" and "fallback"
    # route into it), so the graph runs on past winning_script into the Treatment
    # Agent + Shot-List Agent, which each build their own AsyncOpenAI. Fake both or
    # they hit the real endpoint. Budget Gate is pure code (no client to fake), and
    # the Shot-List Agent's justification validator is KR's real deterministic one
    # (runs for real, not faked).
    monkeypatch.setattr(
        "agents.treatment_agent.AsyncOpenAI",
        make_fake_async_openai([TREATMENT_PAYLOAD]),
    )
    monkeypatch.setattr(
        "agents.shot_list_agent.AsyncOpenAI",
        make_fake_async_openai([SHOT_LIST_CALL_A_PAYLOAD, SHOT_LIST_CALL_B_PAYLOAD]),
    )
    patch_visual_direction_boundaries(monkeypatch)
    patch_phase3_boundaries(monkeypatch, fail_shot_s2=False)
    patch_continuity_boundaries(monkeypatch)  # Phase 4: clean drift, loop ends at once
    patch_voiceover_boundaries(monkeypatch)  # Phase 5: parallel branch off merge_validator
    patch_assembly_boundaries(monkeypatch)  # Phase 5: fan-in join off voiceover + continuity_gate
    patch_format_export_boundaries(monkeypatch)  # Phase 6: format exports without real ffmpeg/OSS

    graph = await build_graph()
    initial_state = {
        "job_id": "test-job-graph",
        "product_photos": ["http://example.com/a.jpg"],
        # Positive-Only Truths fix: this fixture chain's whole narrative is
        # built around GOOD_FACTS' imperfection-category facts (a scratch, a
        # discoloration ring) cited all the way through treatment/shot-list --
        # signal the explicit "authentic/well-loved" carve-out so the new
        # imperfection-citation ban doesn't strip them, rather than rewriting
        # this entire interlinked fixture chain.
        "brief": "a durable everyday case with authentic, well-loved character",
    }
    config = {"configurable": {"thread_id": "test-job-graph"}}

    custom_events = [
        event
        async for event in graph.astream_events(initial_state, config=config, version="v2")
        if event.get("event") == "on_custom_event"
    ]

    # Four custom events now fire across the full chain: truth_extracted (truth
    # extractor), critic_score (Meta-Critic fan-in), merge_validated (Merge
    # Coherence Validator, §5.4.7), and budget_updated (Budget Gate, §5.7 -- the
    # only Phase 2 node that dispatches an event; treatment_agent/shot_list_agent
    # dispatch none). FOUR_GOOD_VARIANTS gives each variant a single body beat, so
    # the merge candidate's pacing re-check fails unrepairably here and
    # merge_validator routes straight to the fallback, which still sets a real,
    # usable winning_script -- see test_graph_merge_validator.py for that path's
    # dedicated assertions. vo_ready (Phase 5) now also fires -- Voiceover runs as
    # a parallel branch off the same merge_validator "fallback" route.
    event_names = {e["name"] for e in custom_events}
    assert event_names == {
        "truth_extracted",
        "critic_score",
        "merge_validated",
        "treatment_ready",  # Concept Agent re-prompt event (new)
        "budget_updated",
        "shot_generated",
        "drift_scored",  # Phase 4: Continuity Agent scored every real clip
        "vo_ready",  # Phase 5: Voiceover + Caption Agent's parallel branch
        "master_cut_ready",  # Phase 5: Assembly Agent's fan-in join
        "job_complete",  # Phase 6: Format Export node signals delivery
    }, event_names
    truth_event = next(e for e in custom_events if e["name"] == "truth_extracted")
    assert truth_event["data"]["count"] == len(GOOD_FACTS)

    final_state = await graph.aget_state(config)
    assert len(final_state.values["product_truths"]) == len(GOOD_FACTS)
    assert len(final_state.values["script_variants"]) == 4, (
        "concept_agent_node must have read product_truths from state (written by "
        "the upstream node) and produced 4 variants from them"
    )
    # Phase 2+3 ran to the end: treatment, shot_list, budget_ledger, and generated clips.
    values = final_state.values
    assert values["treatment"]["beat_treatments"], "treatment_agent did not run"
    assert 3 <= len(values["shot_list"]) <= 7, "shot_list_agent did not produce 3-7 shots"
    assert values["budget_ledger"]["cap"] > 0, "budget_gate did not build a ledger"
    assert len(values["generated_shots"]) == len(values["shot_list"]), "video_gen + ken_burns did not produce one clip per shot"
    # Phase 5: the Assembly Agent fan-in join ran exactly once, after both the
    # voiceover branch and the (trivial, single-pass here) continuity loop settled.
    assert "master_cut_uri" in values, "assembly_agent did not run"
    assert all(s["status"] in ("passed", "fallback") for s in values["shot_list"])
