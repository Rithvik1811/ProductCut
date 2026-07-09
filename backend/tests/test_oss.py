"""
Unit tests for agents/_oss.py — shared OSS upload helper.

Uses an injected fake bucket (same injectable-client pattern as every other
agent module) so no real credentials or network are needed.
"""
from __future__ import annotations

import pytest

from agents._oss import (
    SIGNED_URL_TTL_SEC,
    oss_job_asset_key,
    oss_object_key,
    persist_remote_video_to_oss,
    upload_audio_to_oss,
    upload_json_to_oss,
    upload_video_to_oss,
)


class _FakeBucket:
    def __init__(self) -> None:
        self.uploads: list[tuple[str, str, dict]] = []

    def put_object_from_file(self, key: str, local_path: str, headers: dict | None = None) -> None:
        self.uploads.append((key, local_path, headers or {}))

    def sign_url(self, method: str, key: str, expires: int, slash_safe: bool = False) -> str:
        return f"https://oss.example.invalid/{key}?expires={expires}&method={method}&slash_safe={slash_safe}"


def test_oss_object_key_namespace():
    assert oss_object_key("job-42", "s1", "clip.mp4") == "jobs/job-42/shots/s1/clip.mp4"


def test_upload_video_to_oss_puts_file_and_returns_signed_url(tmp_path):
    local = tmp_path / "clip.mp4"
    local.write_bytes(b"fake-mp4")
    bucket = _FakeBucket()

    url = upload_video_to_oss(str(local), "job-42", "s1", bucket=bucket)

    assert len(bucket.uploads) == 1
    key, path, headers = bucket.uploads[0]
    assert key == "jobs/job-42/shots/s1/fallback_kenburns.mp4"
    assert path == str(local)
    assert headers["Content-Type"] == "video/mp4"
    assert url.startswith("https://oss.example.invalid/jobs/job-42/shots/s1/fallback_kenburns.mp4")
    assert f"expires={SIGNED_URL_TTL_SEC}" in url


def test_upload_video_to_oss_custom_filename(tmp_path):
    local = tmp_path / "shot.mp4"
    local.write_bytes(b"x")
    bucket = _FakeBucket()

    upload_video_to_oss(str(local), "j1", "s2", filename="shot.mp4", bucket=bucket)

    assert bucket.uploads[0][0] == "jobs/j1/shots/s2/shot.mp4"


def test_persist_remote_video_downloads_uploads_and_cleans_up(tmp_path):
    """A remote clip is downloaded, uploaded under the shot namespace, and the
    temp download is deleted afterwards."""
    downloaded = tmp_path / "wan_clip.mp4"
    downloaded.write_bytes(b"wan-bytes")
    bucket = _FakeBucket()
    seen_urls: list[str] = []

    def _fake_download(url: str) -> str:
        seen_urls.append(url)
        return str(downloaded)

    url = persist_remote_video_to_oss(
        "http://wan.example.com/ephemeral/clip.mp4?token=abc",
        "job-7",
        "s2",
        bucket=bucket,
        download_fn=_fake_download,
    )

    assert seen_urls == ["http://wan.example.com/ephemeral/clip.mp4?token=abc"]
    assert bucket.uploads[0][0] == "jobs/job-7/shots/s2/shot.mp4"
    assert url.startswith("https://oss.example.invalid/jobs/job-7/shots/s2/shot.mp4")
    assert not downloaded.exists()  # temp download cleaned up


def test_oss_job_asset_key_namespace_has_no_shot_segment():
    """Job-level assets (Voiceover + Caption Agent's VO track/captions) live
    under jobs/{job_id}/, not jobs/{job_id}/shots/{...}/ -- there is no shot_id
    for an asset produced once per job."""
    assert oss_job_asset_key("job-42", "voiceover.mp3") == "jobs/job-42/voiceover.mp3"


def test_upload_audio_to_oss_puts_file_with_audio_content_type(tmp_path):
    local = tmp_path / "voiceover.mp3"
    local.write_bytes(b"fake-mp3")
    bucket = _FakeBucket()

    url = upload_audio_to_oss(str(local), "job-7", bucket=bucket)

    key, path, headers = bucket.uploads[0]
    assert key == "jobs/job-7/voiceover.mp3"
    assert headers["Content-Type"] == "audio/mpeg"
    assert url.startswith("https://oss.example.invalid/jobs/job-7/voiceover.mp3")


def test_upload_json_to_oss_puts_file_with_json_content_type(tmp_path):
    local = tmp_path / "captions.json"
    local.write_text("[]")
    bucket = _FakeBucket()

    url = upload_json_to_oss(str(local), "job-7", "captions.json", bucket=bucket)

    key, path, headers = bucket.uploads[0]
    assert key == "jobs/job-7/captions.json"
    assert headers["Content-Type"] == "application/json"
    assert url.startswith("https://oss.example.invalid/jobs/job-7/captions.json")


def test_persist_remote_video_cleans_up_temp_even_on_upload_failure(tmp_path):
    downloaded = tmp_path / "wan_clip.mp4"
    downloaded.write_bytes(b"x")

    class _BoomBucket(_FakeBucket):
        def put_object_from_file(self, key, local_path, headers=None):
            raise OSError("OSS put failed")

    with pytest.raises(OSError):
        persist_remote_video_to_oss(
            "http://wan.example.com/clip.mp4",
            "job-7",
            "s2",
            bucket=_BoomBucket(),
            download_fn=lambda url: str(downloaded),
        )

    assert not downloaded.exists()  # temp still cleaned up despite the failure
