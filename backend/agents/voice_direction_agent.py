"""
Voice Direction Agent — LLM pre-processing of winning_script beats for
natural spoken-language TTS delivery.

Runs serially before voiceover_caption_agent (both are in a sub-branch off
merge_validator, running in parallel with visual_direction_agent).

Rewrites each ScriptBeat.line for:
  - Natural spoken English (contractions, breath phrasing, no formal register)
  - Per-beat emotion for CosyVoice instruct_text (excited/warm/authoritative/
    conversational/urgent)
  - Pacing guidance (slow/normal/fast)

Writes directed_script_beats: list[DirectedBeat] to state.

DESIGN NOTES
  * Single batch LLM call for ALL beats (not one call per beat) — cheaper, and
    gives the model cross-beat context so the rewrites flow as one spoken script
    rather than N independently-rewritten lines.
  * Uses the same AsyncOpenAI → DashScope OpenAI-compatible pattern every other
    text agent here uses (MODEL_TEXT / DASHSCOPE_API_KEY / DASHSCOPE_BASE_URL),
    with an injectable `client` for test mocking (same posture as
    treatment_agent.py / visual_direction_agent.py).
  * Beat role is inferred from POSITION, not from Treatment.beat_treatments —
    this node runs in a parallel branch off winning_script alone, so the
    Treatment Agent has not run yet and there is no beat_function available.
    index 0 = hook, last index = cta, middle beats = demo/problem.
  * Temperature 0.3 (low): we want consistent, faithful spoken rewrites, not
    creative variation on the copy the Critic Chain already validated.
  * Graceful fallback: if the LLM call fails, or the JSON can't be parsed, or a
    per-beat entry is missing/invalid, that beat falls back to its original line
    with a sensible position-derived emotion/pacing default — the node never
    blocks the job over a bad rewrite.
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Optional

from openai import AsyncOpenAI
from langchain_core.runnables import RunnableConfig

from agents._retry import create_completion
from graph.state import DirectedBeat, ProductCutState, WinningScript

logger = logging.getLogger("productcut.agents.voice_direction_agent")

_VALID_EMOTIONS = frozenset(
    {"warm", "excited", "authoritative", "conversational", "urgent"}
)
_VALID_PACING = frozenset({"slow", "normal", "fast"})

# Low — consistent, faithful spoken rewrites, not creative variation.
_TEMPERATURE = 0.3


def _infer_beat_role(index: int, total: int) -> tuple[str, str, str]:
    """Position-derived (role, default emotion, default pacing) for a beat.

    No Treatment/beat_function is available here (parallel branch off
    winning_script — see module docstring), so beat role is inferred purely from
    position:
      - index 0            -> hook    (excited, fast)
      - index 1 (total>2)  -> problem (conversational, normal)
      - last index         -> cta     (warm, normal)
      - middle beats       -> demo    (conversational, normal)
    """
    if total <= 1:
        return "hook", "excited", "fast"
    if index == 0:
        return "hook", "excited", "fast"
    if index == total - 1:
        return "cta", "warm", "normal"
    if index == 1:
        return "problem", "conversational", "normal"
    return "demo", "conversational", "normal"


def _format_beats_for_prompt(winning_script: WinningScript) -> str:
    beats = winning_script.get("beats") or []
    total = len(beats)
    lines = []
    for i, b in enumerate(beats):
        role, _, _ = _infer_beat_role(i, total)
        lines.append(f'  beat {i} (role: {role}): "{b.get("line", "")}"')
    return "\n".join(lines)


def _build_system_prompt(beat_count: int) -> str:
    last = beat_count - 1
    return f"""You are a voice director preparing a short-form product ad script
({beat_count} beats, numbered 0 to {last}) for text-to-speech narration by a
professional voice actor.

For EACH beat you receive, produce three things:

1. spoken_text: rewrite the beat's line as NATURAL SPOKEN ENGLISH — the way a
   real person would actually say it out loud, not the way it reads on a page.
   - Use contractions ("it's", "you'll", "don't").
   - Break long sentences into shorter breath-sized phrases.
   - Drop stiff/formal register; keep it warm and conversational.
   - PRESERVE the meaning, the product claims, and any brand/CTA wording — this
     is a delivery rewrite, NOT a rewrite of what the ad says. Never invent new
     claims or drop the call to action.
   - Do NOT add stage directions, emojis, or SSML tags — plain spoken words only.

2. emotion: exactly one of: warm, excited, authoritative, conversational, urgent.
   Choose from what the beat is DOING:
   - hook (beat 0): usually "excited" — grab attention.
   - problem beats: usually "conversational" — relatable, empathetic.
   - demo/proof beats: "conversational" or "authoritative" — build trust.
   - cta (last beat, beat {last}): usually "warm" or "urgent" — close the ask.

3. pacing: exactly one of: slow, normal, fast.
   - fast for energetic hooks, urgent for time-pressured CTAs, slow for
     emphasis/gravitas, normal otherwise.

Return ONLY valid JSON in this exact shape, no preamble or commentary:

{{
  "directed_beats": [
    {{
      "beat_index": 0,
      "spoken_text": "the natural spoken rewrite of beat 0's line",
      "emotion": "excited",
      "pacing": "fast"
    }}
  ]
}}

Return EXACTLY {beat_count} entries, one per beat, beat_index 0 through {last},
in order."""


def _parse_json_response(raw: str) -> dict:
    stripped = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.IGNORECASE)
    return json.loads(stripped)


def _fallback_directed_beats(winning_script: WinningScript) -> list[DirectedBeat]:
    """Original lines + position-derived emotion/pacing defaults — used when the
    LLM call fails, the response can't be parsed, or an individual beat entry is
    missing/invalid. Never blocks the job over a bad rewrite."""
    beats = winning_script.get("beats") or []
    total = len(beats)
    out: list[DirectedBeat] = []
    for i, b in enumerate(beats):
        _, emotion, pacing = _infer_beat_role(i, total)
        out.append(
            DirectedBeat(
                beat_index=i,
                spoken_text=b.get("line", ""),
                emotion=emotion,  # type: ignore[arg-type]
                pacing=pacing,    # type: ignore[arg-type]
            )
        )
    return out


async def generate_directed_beats(
    winning_script: WinningScript,
    client: Optional[AsyncOpenAI] = None,
) -> list[DirectedBeat]:
    """Run the Voice Direction Agent: ONE batch LLM call rewriting every beat's
    line into natural spoken English + assigning emotion/pacing. Falls back to
    the original lines (with position-derived emotion/pacing) on any failure.
    """
    beats = winning_script.get("beats") or []
    beat_count = len(beats)
    if beat_count == 0:
        return []

    model = os.environ["MODEL_TEXT"]
    own_client = client is None
    if own_client:
        client = AsyncOpenAI(
            api_key=os.environ["DASHSCOPE_API_KEY"],
            base_url=os.environ["DASHSCOPE_BASE_URL"],
            timeout=90.0,
        )

    try:
        messages = [
            {"role": "system", "content": _build_system_prompt(beat_count)},
            {
                "role": "user",
                "content": (
                    "Rewrite these beats for spoken delivery:\n"
                    + _format_beats_for_prompt(winning_script)
                ),
            },
        ]

        try:
            response_text = await create_completion(
                client, model=model, messages=messages, temperature=_TEMPERATURE
            )
            parsed = _parse_json_response(response_text)
        except Exception as exc:  # noqa: BLE001 -- any failure degrades to originals
            logger.warning(
                "Voice Direction Agent: LLM call/parse failed (%s) -- "
                "falling back to original lines with default emotion/pacing.",
                exc,
            )
            return _fallback_directed_beats(winning_script)

        raw_entries = parsed.get("directed_beats")
        by_index: dict[int, dict] = {}
        if isinstance(raw_entries, list):
            for e in raw_entries:
                if isinstance(e, dict) and isinstance(e.get("beat_index"), int):
                    by_index[e["beat_index"]] = e

        directed: list[DirectedBeat] = []
        for i, beat in enumerate(beats):
            _, default_emotion, default_pacing = _infer_beat_role(i, beat_count)
            entry = by_index.get(i)
            if entry is None:
                logger.info(
                    "Voice Direction Agent: missing entry for beat %d -- using original line.",
                    i,
                )
                directed.append(
                    DirectedBeat(
                        beat_index=i,
                        spoken_text=beat.get("line", ""),
                        emotion=default_emotion,  # type: ignore[arg-type]
                        pacing=default_pacing,    # type: ignore[arg-type]
                    )
                )
                continue

            spoken_text = str(entry.get("spoken_text") or "").strip() or beat.get("line", "")
            emotion = entry.get("emotion")
            if emotion not in _VALID_EMOTIONS:
                emotion = default_emotion
            pacing = entry.get("pacing")
            if pacing not in _VALID_PACING:
                pacing = default_pacing

            directed.append(
                DirectedBeat(
                    beat_index=i,
                    spoken_text=spoken_text,
                    emotion=emotion,  # type: ignore[arg-type]
                    pacing=pacing,    # type: ignore[arg-type]
                )
            )
        return directed
    finally:
        if own_client:
            await client.close()


async def voice_direction_agent_node(
    state: ProductCutState,
    config: Optional[RunnableConfig] = None,
) -> dict:
    """LangGraph node wrapper: reads winning_script, produces directed_script_beats.

    Runs as a serial pre-step before voiceover_caption_agent (in a sub-branch off
    merge_validator, parallel with visual_direction_agent). Writes only
    `directed_script_beats` -- no shared `reasoning_trace` write, since this node
    runs in the same superstep as visual_direction_agent (which owns the shared
    trace on this branch) and a second read-modify-write of that plain-string
    channel would raise LangGraph's InvalidUpdateError (same reasoning as the
    voiceover node's dedicated `voiceover_reasoning_trace`, graph/state.py v7).
    """
    winning_script = state["winning_script"]
    directed_beats = await generate_directed_beats(winning_script)
    return {"directed_script_beats": directed_beats}


__all__ = ["generate_directed_beats", "voice_direction_agent_node"]
