"""
Trivial smoke test for MODEL_TEXT -- isolates "is this model/endpoint even
reachable" from "is our Concept Agent prompt too big/complex." One tiny
call, short timeout, clear pass/fail. Mirrors the Phase 0 de-risk approach
RR used for the video model (region/account-scoped model IDs have already
been wrong once in this project -- rule that out here too).

Usage (from backend/):
    python -m derisk.test_text_model_smoke
"""
from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from openai import AsyncOpenAI  # noqa: E402


async def main() -> int:
    model = os.environ["MODEL_TEXT"]
    base_url = os.environ["DASHSCOPE_BASE_URL"]
    print(f"Testing MODEL_TEXT={model!r} against {base_url!r} (15s timeout)...")

    client = AsyncOpenAI(
        api_key=os.environ["DASHSCOPE_API_KEY"], base_url=base_url, timeout=15.0
    )
    start = time.monotonic()
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Reply with exactly one word: hello"}],
        )
        elapsed = time.monotonic() - start
        print(f"OK in {elapsed:.1f}s: {response.choices[0].message.content!r}")
        return 0
    except Exception as exc:  # noqa: BLE001 - this script's whole job is to report the failure
        elapsed = time.monotonic() - start
        print(f"FAILED after {elapsed:.1f}s: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    finally:
        await client.close()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
