"""
Tests for format_export_node — covers:
  1. _crop_params: wider-than-target source → crops sides
  2. _crop_params: taller-than-target source → crops top/bottom
  3. _crop_params: equal aspect ratio → no crop (offset 0)
  4. Even number enforcement: all returned w/h values are even
  5. format_export_node: missing/empty master_cut_uri → returns {}
  6. format_export_node: writes 'exports' key with 3 format keys (mocked)
  7. generate_format_exports: mocked ffmpeg/OSS → all 3 formats produced
  8. format_export_node: exception in generate_format_exports → returns {}
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from agents.format_export_node import (
    EXPORT_FORMATS,
    _crop_params,
    format_export_node,
    generate_format_exports,
)


# ---------------------------------------------------------------------------
# 1. _crop_params: wider-than-target (landscape → portrait)
# ---------------------------------------------------------------------------

def test_crop_params_wider_than_target_crops_sides():
    crop_w, crop_h, crop_x, crop_y = _crop_params(1920, 1080, 1080, 1920)

    assert crop_h == 1080
    assert crop_w < 1920
    assert crop_y == 0
    assert crop_x > 0
    assert crop_x == (1920 - crop_w) // 2


# ---------------------------------------------------------------------------
# 2. _crop_params: taller-than-target (portrait → landscape)
# ---------------------------------------------------------------------------

def test_crop_params_taller_than_target_crops_top_bottom():
    crop_w, crop_h, crop_x, crop_y = _crop_params(1080, 1920, 1920, 1080)

    assert crop_w == 1080
    assert crop_h < 1920
    assert crop_x == 0
    assert crop_y > 0
    assert crop_y == (1920 - crop_h) // 2


# ---------------------------------------------------------------------------
# 3. _crop_params: equal aspect ratio
# ---------------------------------------------------------------------------

def test_crop_params_equal_aspect_ratio_square():
    crop_w, crop_h, crop_x, crop_y = _crop_params(1080, 1080, 1080, 1080)

    assert crop_w == 1080
    assert crop_h == 1080
    assert crop_x == 0
    assert crop_y == 0


def test_crop_params_equal_aspect_ratio_landscape():
    crop_w, crop_h, crop_x, crop_y = _crop_params(1920, 1080, 1920, 1080)

    assert crop_x == 0
    assert crop_y == 0
    assert crop_w == 1920
    assert crop_h == 1080


# ---------------------------------------------------------------------------
# 4. Even number enforcement (libx264 rejects odd dimensions)
# ---------------------------------------------------------------------------

def test_crop_params_even_dimensions_wider_path():
    crop_w, crop_h, crop_x, crop_y = _crop_params(1920, 1080, 1080, 1920)
    assert crop_w % 2 == 0, f"crop_w={crop_w} is odd"
    assert crop_h % 2 == 0, f"crop_h={crop_h} is odd"


def test_crop_params_even_dimensions_taller_path():
    crop_w, crop_h, crop_x, crop_y = _crop_params(1080, 1920, 1920, 1080)
    assert crop_w % 2 == 0, f"crop_w={crop_w} is odd"
    assert crop_h % 2 == 0, f"crop_h={crop_h} is odd"


def test_crop_params_even_for_all_export_formats():
    src_w, src_h = 1920, 1080
    for key, (tgt_w, tgt_h) in EXPORT_FORMATS.items():
        crop_w, crop_h, crop_x, crop_y = _crop_params(src_w, src_h, tgt_w, tgt_h)
        assert crop_w % 2 == 0, f"{key}: crop_w={crop_w} is odd"
        assert crop_h % 2 == 0, f"{key}: crop_h={crop_h} is odd"


# ---------------------------------------------------------------------------
# 5. format_export_node: missing/empty master_cut_uri → returns {}
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_format_export_node_no_master_cut_uri_returns_empty():
    result = await format_export_node({"job_id": "j1"})
    assert result == {}


@pytest.mark.asyncio
async def test_format_export_node_empty_master_cut_uri_returns_empty():
    result = await format_export_node({"job_id": "j1", "master_cut_uri": ""})
    assert result == {}


# ---------------------------------------------------------------------------
# 6. format_export_node: writes 'exports' with 3 format keys (mocked inner call)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_format_export_node_writes_exports_key(monkeypatch):
    fake_exports = {
        "aspect_9x16": "https://oss.example/jobs/j1/exports/9x16.mp4",
        "aspect_1x1":  "https://oss.example/jobs/j1/exports/1x1.mp4",
        "aspect_16x9": "https://oss.example/jobs/j1/exports/16x9.mp4",
    }

    async def _fake_generate(master_cut_uri, job_id, **kw):
        return fake_exports

    monkeypatch.setattr("agents.format_export_node.generate_format_exports", _fake_generate)

    state = {"job_id": "j1", "master_cut_uri": "https://oss.example/jobs/j1/master_cut.mp4"}
    result = await format_export_node(state)

    assert "exports" in result
    exports = result["exports"]
    assert set(exports.keys()) == {"aspect_9x16", "aspect_1x1", "aspect_16x9"}
    assert exports["aspect_9x16"] == "https://oss.example/jobs/j1/exports/9x16.mp4"


# ---------------------------------------------------------------------------
# 7. generate_format_exports: mocked ffmpeg/OSS → all 3 formats produced
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_generate_format_exports_mocked_ffmpeg(tmp_path):
    fake_local = str(tmp_path / "master.mp4")
    with open(fake_local, "wb") as f:
        f.write(b"fake video bytes")

    upload_count = {"n": 0}

    def _fake_download(url: str) -> str:
        return fake_local

    def _fake_upload(local_path, job_id, filename, *, bucket=None) -> str:
        upload_count["n"] += 1
        return f"https://oss.example/{filename}"

    def _fake_render(src_path, out_path, tgt_w, tgt_h, src_w, src_h):
        with open(out_path, "wb") as f:
            f.write(b"rendered")

    fake_probe = {
        "streams": [{"codec_type": "video", "width": 1920, "height": 1080}]
    }

    with (
        patch("agents.format_export_node._download_to_temp", _fake_download),
        patch("agents.format_export_node.upload_export_to_oss", _fake_upload),
        patch("agents.format_export_node._render_export", _fake_render),
        patch("agents.format_export_node.ffmpeg") as mock_ffmpeg,
    ):
        mock_ffmpeg.probe.return_value = fake_probe
        exports = await generate_format_exports(
            "https://oss.example/master.mp4", "j1"
        )

    assert set(exports.keys()) == {"aspect_9x16", "aspect_1x1", "aspect_16x9"}
    assert upload_count["n"] == 3


# ---------------------------------------------------------------------------
# 8. format_export_node: exception in generate_format_exports → returns {}
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_format_export_node_exception_degrades_gracefully(monkeypatch):
    async def _failing(*a, **kw):
        raise RuntimeError("ffmpeg not found")

    monkeypatch.setattr("agents.format_export_node.generate_format_exports", _failing)

    state = {"job_id": "j1", "master_cut_uri": "https://oss.example/jobs/j1/master_cut.mp4"}
    result = await format_export_node(state)

    assert result == {}
