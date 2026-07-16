"""
De-risk / smoke-test script for CosyVoice v3-flash (voice-direction-tts-upgrade).

Mirrors derisk/test_text_model_smoke.py's scope one level up (audio instead
of text) -- "is this model/endpoint even reachable and does it return real
audio," not a test of the Voiceover + Caption Agent's own beat/retry/concat
logic (that's covered by tests/test_voiceover_caption_agent.py). Confirms the
COSYVOICE_MODEL_ID / COSYVOICE_VOICE_ID in .env.example actually works against
a real account.

Calls the real code path -- agents.voiceover_caption_agent._call_cosyvoice --
not a hand-rolled second SDK call, same "same code path the graph node uses"
precedent as derisk/test_truth_extractor.py.

Usage (from backend/):
    python -m derisk.test_tts_smoke
"""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from agents.voiceover_caption_agent import _call_cosyvoice, _probe_duration_sec  # noqa: E402

OUTPUTS_DIR = Path(__file__).resolve().parent / "outputs"
SAMPLE_LINE = "Your coffee stays hot for twelve hours, guaranteed."


async def main() -> int:
    model = os.environ.get("COSYVOICE_MODEL_ID", "cosyvoice-v3-flash")
    voice = os.environ.get("COSYVOICE_VOICE_ID", "longanyang")
    print(f"Testing model={model!r} voice={voice!r} with sample line: {SAMPLE_LINE!r}")

    try:
        local_path = await _call_cosyvoice(SAMPLE_LINE, pacing="fast")
    except Exception as exc:  # noqa: BLE001 -- this script's whole job is to report the failure
        print(f"FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    try:
        size_bytes = os.path.getsize(local_path)
        duration_sec = _probe_duration_sec(local_path)
        print(f"OK: received {size_bytes} bytes, probed duration {duration_sec:.2f}s")

        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        saved_audio_path = OUTPUTS_DIR / f"tts_smoke_sample{Path(local_path).suffix or '.mp3'}"
        shutil.copyfile(local_path, saved_audio_path)

        result = {
            "model": model,
            "voice": voice,
            "sample_line": SAMPLE_LINE,
            "duration_sec": round(duration_sec, 3),
            "size_bytes": size_bytes,
            "saved_audio_file": saved_audio_path.name,
        }
        result_path = OUTPUTS_DIR / "tts_smoke_result.json"
        result_path.write_text(json.dumps(result, indent=2))
        print(f"Saved result to {result_path} and audio sample to {saved_audio_path}")

        if duration_sec <= 0:
            print(
                "\n⚠ Probed duration is zero/invalid -- response may not be real, "
                "audible audio. Investigate before trusting this model/voice id.",
                file=sys.stderr,
            )
            return 1
        return 0
    finally:
        if os.path.exists(local_path):
            os.remove(local_path)


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
