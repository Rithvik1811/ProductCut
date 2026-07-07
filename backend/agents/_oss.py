"""
Shared OSS (Alibaba Cloud Object Storage Service) upload helper.

INTENTIONALLY REUSABLE -- NOT KEN-BURNS-SPECIFIC. This lives in
`agents/_oss.py` (the same underscore-prefixed shared-helper convention as
`agents/_retry.py`), NOT buried as a private function inside
`ken_burns_fallback_node.py`, because the exact same capability -- "take a
locally-produced shot asset and persist it to the job's OSS namespace, then
hand back a signed URL other nodes/the dashboard can read" -- is needed by more
than one node. The Ken-Burns Fallback Node (§5.9) is the FIRST caller, but a
separate, not-yet-built RR task ("upload generated shot assets to OSS", the
real Video-Gen persistence path per §5.8's "with the clip persisted to OSS")
needs the identical operation. Building it once here keeps the OSS key
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
    b = bucket if bucket is not None else _build_bucket()
    key = oss_object_key(job_id, shot_id, filename)
    b.put_object_from_file(key, local_path, headers={"Content-Type": "video/mp4"})
    url = b.sign_url("GET", key, SIGNED_URL_TTL_SEC, slash_safe=True)
    logger.info("OSS: uploaded %s -> %s (signed for %ds)", local_path, key, SIGNED_URL_TTL_SEC)
    return url


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
    "upload_video_to_oss",
    "persist_remote_video_to_oss",
]
