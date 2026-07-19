"""
Re-assembles master cut from full_pipeline_live_result.json with the updated
subtitle wrap_width=45 (all captions → exactly 2 lines, consistent layout).

Usage (from backend/):
    python -m derisk.test_subtitle_v5
"""
from __future__ import annotations

import asyncio
import json
import shutil
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from agents.assembly_agent import _assemble_master_cut_impl  # noqa: E402

OUTPUTS_DIR = Path(__file__).resolve().parent / "outputs"
RESULT_JSON = OUTPUTS_DIR / "full_pipeline_live_result.json"
OUT_PATH = OUTPUTS_DIR / "master_cut_subtitle_v6.mp4"


async def main() -> int:
    data = json.loads(RESULT_JSON.read_text(encoding="utf-8"))
    shot_list = data["shot_list"]
    generated_shots = data["generated_shots"]
    voiceover = data["voiceover"]
    winning_script = data["winning_script"]

    print(f"Shots: {[s['shot_id'] for s in shot_list]}")
    print(f"Captions: {len(winning_script['beats'])} beats")
    print(f"Output: {OUT_PATH}")

    def _upload(local_path: str) -> str:
        shutil.copy(local_path, OUT_PATH)
        return f"file://{OUT_PATH}"

    result = await _assemble_master_cut_impl(
        shot_list, generated_shots, voiceover, winning_script,
        "subtitle-v5-render", upload_fn=_upload,
    )
    print(f"shot_count={result.shot_count}  total_duration_sec={result.total_duration_sec}")
    print(f"Saved: derisk/outputs/{OUT_PATH.name} ({OUT_PATH.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
