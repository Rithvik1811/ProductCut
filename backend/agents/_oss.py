"""
Shared OSS (Alibaba Cloud Object Storage Service) upload helper.

INTENTIONALLY REUSABLE -- NOT KEN-BURNS-SPECIFIC. This lives in
`agents/_oss.py` (the same underscore-prefixed shared-helper convention as
`agents/_retry.py`), NOT buried as a private function inside
`ken_burns_fallback_node.py`, because the exact same capability -- "take a
locally-produced shot asset and persist it to the job's OSS namespace, then
hand back a signed URL other nodes/the dashboard can read" -- is needed by more
than one node. The Ken-Burns Fallback Node (§5.9) is the FIRST caller; a
second RR task ("upload generated shot assets to OSS", the real Video-Gen
persistence path per §5.8's "with the clip persisted to OSS", now built as
`persist_remote_video_to_oss` below) needs the identical operation. Building
it once here keeps the OSS key
convention (`jobs/{job_id}/shots/{shot_id}/...`) and the signed-URL lifetime in
a single source of truth instead of two hand-rolled copies that could drift.

FIRST REAL OSS IMPLEMENTATION IN THIS CODEBASE. `.env.example` already declares
`OSS_ENDPOINT` / `OSS_BUCKET` / `OSS_ACCESS_KEY_ID` / `OSS_ACCESS_KEY_SECRET`
(the four values `oss2.Auth`/`oss2.Bucket` need), and `oss2>=2.19.1` is already
a requirement, but nothing had actually called them yet -- this is that code.

INJECTABLE CLIENT (same `client=None` pattern every agent module in this
codebase uses for testability): pass a `bucket` object exposing
`put_object_from_file(key, local_path, headers=...)` and
`sign_url(method, key, expires, slash_safe=...)` to fake OSS entirely in tests
with no real credentials/network. The real `oss2.Bucket` is constructed from
the environment only when `bucket` is not supplied.

ADDITIVE (Phase 5, Voiceover + Caption Agent, agents/voiceover_caption_agent.py):
that node needs to persist two NEW asset kinds -- a synthesized VO audio track
and a caption-timing JSON file -- neither a "shot asset" (there is no shot_id;
VO is produced once per job, not once per shot). Rather than hand-rolling a
second put_object_from_file/sign_url pair (exactly the "two hand-rolled copies
that could drift" this module's own docstring above warns against), the
shared put+sign mechanics are factored into the private `_put_and_sign` helper
and reused by THREE thin wrappers: the pre-existing `upload_video_to_oss`
(refactored to call it, behavior unchanged -- see test_oss.py, unmodified and
still green) and the two new ones below. A parallel `oss_job_asset_key` sits
next to `oss_object_key` for these job-level (non-shot) assets --
`jobs/{job_id}/{filename}`, dropping the `shots/{shot_id}` segment that would
otherwise misname a job-wide asset as if it belonged to one shot.
"""
from __future__ import annotations

import logging
import os
import tempfile
from typing import Callable, Optional

import httpx

logger = logging.getLogger("productcut.agents.oss")

# Wan's own returned `video_url` is valid for ~24h (docs/DERISK_VIDEO_GEN_RESULT.md);
# matching that lifetime here keeps a fallback clip's signed URL live for exactly
# as long as a real Video-Gen clip's would be, so nothing downstream has to special-
# case "this uri is a fallback" for expiry purposes.
SIGNED_URL_TTL_SEC = 24 * 3600


def _build_bucket():
    """Construct the real `oss2.Bucket` from the four `.env.example` OSS vars.

    Imported lazily (inside the function, not at module top) so that merely
    importing this module -- or injecting a fake `bucket` in a test -- never
    requires `oss2` credentials or even the `oss2` package to be import-time
    reachable. Only an actual real upload touches `oss2`/`os.environ`.
    """
    import oss2  # local import: real OSS is only needed on the non-injected path

    auth = oss2.Auth(os.environ["OSS_ACCESS_KEY_ID"], os.environ["OSS_ACCESS_KEY_SECRET"])
    return oss2.Bucket(auth, os.environ["OSS_ENDPOINT"], os.environ["OSS_BUCKET"])


def oss_object_key(job_id: str, shot_id: str, filename: str) -> str:
    """The single source of truth for a shot asset's OSS object key.

    `jobs/{job_id}/shots/{shot_id}/{filename}` -- a stable, human-inspectable
    namespace so every asset for one shot (fallback clip today, real Video-Gen
    clip / continuity frames later) lands under one predictable prefix.
    """
    return f"jobs/{job_id}/shots/{shot_id}/{filename}"


def oss_job_asset_key(job_id: str, filename: str) -> str:
    """The OSS object key for a job-level (not per-shot) asset.

    `jobs/{job_id}/{filename}` -- same `jobs/{job_id}/` root as `oss_object_key`,
    without a `shots/{shot_id}` segment, for assets produced once per job rather
    than once per shot (the Voiceover + Caption Agent's VO audio track and
    caption-timing JSON today; a future Assembly `master_cut_uri` / Format
    Export `exports` would land here too).
    """
    return f"jobs/{job_id}/{filename}"


def _put_and_sign(key: str, local_path: str, content_type: str, *, bucket: Optional[object] = None) -> str:
    """Shared put-object-then-sign-a-GET-URL mechanics behind every `upload_*_to_oss`
    wrapper below -- the one place that touches `bucket.put_object_from_file` /
    `bucket.sign_url`, so the three asset-kind wrappers can never drift apart.
    """
    b = bucket if bucket is not None else _build_bucket()
    b.put_object_from_file(key, local_path, headers={"Content-Type": content_type})
    url = b.sign_url("GET", key, SIGNED_URL_TTL_SEC, slash_safe=True)
    logger.info("OSS: uploaded %s -> %s (signed for %ds)", local_path, key, SIGNED_URL_TTL_SEC)
    return url


def upload_video_to_oss(
    local_path: str,
    job_id: str,
    shot_id: str,
    filename: str = "fallback_kenburns.mp4",
    *,
    bucket: Optional[object] = None,
) -> str:
    """Upload one local MP4 to the shot's OSS namespace and return a signed GET URL.

    Args:
        local_path: path to the finished local MP4 to upload.
        job_id / shot_id: identify the shot; together they form the object key
            via `oss_object_key` (`jobs/{job_id}/shots/{shot_id}/{filename}`).
        filename: the object's leaf name; defaults to the Ken-Burns fallback
            clip name, overridable so the future Video-Gen persistence caller can
            pass its own (e.g. `shot.mp4`) without a second copy of this code.
        bucket: an injected `oss2.Bucket`-like object (for tests / alternate
            backends). When None, the real bucket is built from the environment.

    Returns:
        A time-limited signed GET URL (valid `SIGNED_URL_TTL_SEC`), suitable for
        Assembly/the dashboard to read the clip without public-read ACLs.
    """
    key = oss_object_key(job_id, shot_id, filename)
    return _put_and_sign(key, local_path, "video/mp4", bucket=bucket)


def upload_audio_to_oss(
    local_path: str,
    job_id: str,
    filename: str = "voiceover.mp3",
    *,
    bucket: Optional[object] = None,
) -> str:
    """Upload one local audio file to the job's (not a shot's) OSS namespace.

    Same shape as `upload_video_to_oss` but keyed via `oss_job_asset_key` (no
    shot_id -- the Voiceover + Caption Agent produces exactly one VO track per
    job) and `Content-Type: audio/mpeg`. First caller:
    agents/voiceover_caption_agent.py.
    """
    key = oss_job_asset_key(job_id, filename)
    return _put_and_sign(key, local_path, "audio/mpeg", bucket=bucket)


def upload_master_cut_to_oss(
    local_path: str,
    job_id: str,
    filename: str = "master_cut.mp4",
    *,
    bucket: Optional[object] = None,
) -> str:
    """Upload the finished Assembly master-cut MP4 to the job's OSS namespace.

    Same shape as `upload_audio_to_oss`/`upload_json_to_oss` -- keyed via
    `oss_job_asset_key` (one master cut per job, not per shot) and
    `Content-Type: video/mp4`. First caller: agents/assembly_agent.py.
    """
    key = oss_job_asset_key(job_id, filename)
    return _put_and_sign(key, local_path, "video/mp4", bucket=bucket)


def upload_json_to_oss(
    local_path: str,
    job_id: str,
    filename: str,
    *,
    bucket: Optional[object] = None,
) -> str:
    """Upload one local JSON file to the job's OSS namespace, `Content-Type: application/json`.

    Generic (not audio/video-specific) -- first caller is
    agents/voiceover_caption_agent.py's caption-timing track
    (`{text, start_ts, end_ts}` entries), but any future job-level JSON artifact
    can reuse this instead of a fourth hand-rolled wrapper.
    """
    key = oss_job_asset_key(job_id, filename)
    return _put_and_sign(key, local_path, "application/json", bucket=bucket)


def _download_to_temp(url: str) -> str:
    """Download a remote video to a temp file and return its path (caller deletes).

    `follow_redirects=True` because a video-gen provider's returned URL / CDN
    commonly 3xx. The suffix is derived from the URL (query string stripped) so
    the temp file keeps a sensible extension, defaulting to `.mp4`.
    """
    resp = httpx.get(url, timeout=60.0, follow_redirects=True)
    resp.raise_for_status()
    suffix = os.path.splitext(url.split("?", 1)[0])[1] or ".mp4"
    fd, path = tempfile.mkstemp(suffix=suffix, prefix="oss_persist_")
    with os.fdopen(fd, "wb") as fh:  # takes ownership of fd; closes it even on error
        fh.write(resp.content)
    return path


def persist_remote_video_to_oss(
    remote_url: str,
    job_id: str,
    shot_id: str,
    filename: str = "shot.mp4",
    *,
    bucket: Optional[object] = None,
    download_fn: Optional[Callable[[str], str]] = None,
) -> str:
    """Download a remote (e.g. Wan) clip and re-upload it to the shot's OSS
    namespace, returning a signed OSS GET URL.

    Why persist at all: the Video-Gen provider's own returned URL is ephemeral
    (Wan's is ~24h, docs/DERISK_VIDEO_GEN_RESULT.md). Copying the clip into OSS
    gives the deliverable a stable home under the SAME
    `jobs/{job_id}/shots/{shot_id}/` prefix every other shot asset (Ken-Burns
    fallback clips, continuity frames) already uses, so Assembly and the
    dashboard read one consistent namespace with one consistent expiry (§5.8
    output contract: "with the clip persisted to OSS").

    `download_fn` and `bucket` are injectable for credential-free / network-free
    tests, the same `client=None` pattern the rest of this module uses. The
    downloaded temp file is always cleaned up, even on upload failure.
    """
    dl = download_fn or _download_to_temp
    local_path = dl(remote_url)
    try:
        return upload_video_to_oss(local_path, job_id, shot_id, filename, bucket=bucket)
    finally:
        if os.path.exists(local_path):
            try:
                os.remove(local_path)
            except OSError:
                pass


__all__ = [
    "SIGNED_URL_TTL_SEC",
    "oss_object_key",
    "oss_job_asset_key",
    "upload_video_to_oss",
    "upload_audio_to_oss",
    "upload_master_cut_to_oss",
    "upload_json_to_oss",
    "persist_remote_video_to_oss",
]
