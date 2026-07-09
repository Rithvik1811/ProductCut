"""
Unit tests for db/jobs.py -- `jobs`/`seller_direction` persistence (Phase 1, RR).

Uses a hand-rolled in-memory fake `psycopg.AsyncConnection` (`_FakeConnection`/
`_FakeCursor` below), NOT a real Postgres instance -- this test suite has an
existing, deliberate policy (conftest.py's autouse fixture always clears
`DATABASE_URL`: "these tests must never touch a real Postgres instance") that
a live-DB integration test would violate. The fake is tailored to the exact,
small, fixed set of queries db/jobs.py issues (two CREATE TABLE statements,
two upserts, two selects, one delete) -- it is NOT a general SQL engine and
does not verify real Postgres grammar (e.g. whether `ON CONFLICT ... DO
UPDATE` is itself valid SQL); it simulates that semantic in Python so this
module's OWN logic (row reconstruction, the skip-write-when-empty behavior,
nullable-field round-tripping) gets real, direct test coverage -- which is
exactly where a real bug is most likely to live, versus in Postgres's own
well-tested SQL engine.
"""
from __future__ import annotations

import pytest

from db.jobs import (
    _conninfo,
    create_job,
    delete_job,
    init_tables,
    read_job_state,
    upsert_seller_direction,
)


class _FakeCursor:
    def __init__(self, conn: "_FakeConnection") -> None:
        self._conn = conn
        self._result = None

    async def __aenter__(self) -> "_FakeCursor":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return False

    async def execute(self, query: str, params: tuple = ()) -> None:
        q = " ".join(query.split())
        self._conn.executed.append(q)

        if q.startswith("CREATE TABLE"):
            return
        if q.startswith("INSERT INTO jobs"):
            job_id, seller_id, brief, status, photo_refs = params
            self._conn.jobs[job_id] = {
                "job_id": job_id,
                "seller_id": seller_id,
                "brief": brief,
                "status": status,
                "product_photo_refs": list(photo_refs),
            }
            return
        if q.startswith("INSERT INTO seller_direction"):
            job_id, mood_words, ref_url, ref_why, never_do, freeform = params
            self._conn.seller_direction[job_id] = {
                "mood_words": mood_words,
                "reference_ad_url_or_text": ref_url,
                "reference_ad_why": ref_why,
                "never_do": never_do,
                "freeform": freeform,
            }
            return
        if q.startswith("SELECT job_id, brief, product_photo_refs FROM jobs"):
            (job_id,) = params
            job = self._conn.jobs.get(job_id)
            self._result = dict(job) if job is not None else None
            return
        if q.startswith("SELECT mood_words"):
            (job_id,) = params
            row = self._conn.seller_direction.get(job_id)
            self._result = dict(row) if row is not None else None
            return
        if q.startswith("DELETE FROM jobs"):
            (job_id,) = params
            self._conn.jobs.pop(job_id, None)
            self._conn.seller_direction.pop(job_id, None)  # simulates ON DELETE CASCADE
            return
        raise AssertionError(f"Unexpected query in fake cursor: {q!r}")

    async def fetchone(self):
        return self._result


class _FakeConnection:
    """In-memory stand-in for `psycopg.AsyncConnection` -- see module docstring."""

    def __init__(self) -> None:
        self.jobs: dict[str, dict] = {}
        self.seller_direction: dict[str, dict] = {}
        self.executed: list[str] = []
        self.commits = 0

    def cursor(self) -> _FakeCursor:
        return _FakeCursor(self)

    async def commit(self) -> None:
        self.commits += 1


@pytest.fixture
def conn() -> _FakeConnection:
    return _FakeConnection()


# ---------------------------------------------------------------------------
# _conninfo -- pure resolution logic, no connection needed.
# ---------------------------------------------------------------------------
def test_conninfo_prefers_explicit_arg_over_env(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://env-value")
    assert _conninfo("postgresql://explicit-arg") == "postgresql://explicit-arg"


def test_conninfo_falls_back_to_env(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://env-value")
    assert _conninfo(None) == "postgresql://env-value"


def test_conninfo_raises_without_arg_or_env(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    with pytest.raises(RuntimeError, match="No Postgres connection string"):
        _conninfo(None)


# ---------------------------------------------------------------------------
# init_tables
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_init_tables_creates_jobs_before_seller_direction(conn):
    await init_tables(conn)
    assert len(conn.executed) == 2
    assert conn.executed[0].startswith("CREATE TABLE IF NOT EXISTS jobs")
    assert conn.executed[1].startswith("CREATE TABLE IF NOT EXISTS seller_direction")
    assert conn.commits == 1


# ---------------------------------------------------------------------------
# create_job + read_job_state: basic round trip, no seller_direction.
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_create_job_and_read_back_round_trip(conn):
    await create_job(
        conn,
        job_id="job-1",
        seller_id="seller-9",
        brief="handmade ceramic mugs, cozy autumn vibe",
        product_photo_refs=["oss://photo1.jpg", "oss://photo2.jpg"],
    )

    state = await read_job_state(conn, "job-1")

    assert state == {
        "job_id": "job-1",
        "brief": "handmade ceramic mugs, cozy autumn vibe",
        "product_photos": ["oss://photo1.jpg", "oss://photo2.jpg"],
    }
    assert "seller_direction" not in state  # no direction row exists at all


@pytest.mark.asyncio
async def test_read_job_state_returns_none_for_missing_job(conn):
    assert await read_job_state(conn, "does-not-exist") is None


@pytest.mark.asyncio
async def test_create_job_defaults_status_and_empty_photo_refs(conn):
    await create_job(conn, job_id="job-2", seller_id=None, brief=None)
    assert conn.jobs["job-2"]["status"] == "ingested"
    assert conn.jobs["job-2"]["product_photo_refs"] == []


@pytest.mark.asyncio
async def test_create_job_upsert_overwrites_on_conflict(conn):
    """Same job_id inserted twice -- the second call's values win (ON CONFLICT
    DO UPDATE semantics), matching the real DDL's upsert intent."""
    await create_job(conn, job_id="job-3", seller_id="s1", brief="first brief")
    await create_job(conn, job_id="job-3", seller_id="s1", brief="revised brief")

    state = await read_job_state(conn, "job-3")
    assert state["brief"] == "revised brief"


# ---------------------------------------------------------------------------
# upsert_seller_direction: nullability + skip-when-empty + round trip.
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_seller_direction_with_all_fields_populated_round_trips(conn):
    await create_job(conn, job_id="job-4", seller_id="s1", brief="brief")
    wrote = await upsert_seller_direction(
        conn,
        "job-4",
        {
            "mood_words": ["cozy", "warm"],
            "reference_ad": {"url_or_text": "https://example.com/ad", "why": "similar vibe"},
            "never_do": "no loud music",
            "freeform": "keep it understated",
        },
    )
    assert wrote is True

    state = await read_job_state(conn, "job-4")
    assert state["seller_direction"] == {
        "mood_words": ["cozy", "warm"],
        "reference_ad": {"url_or_text": "https://example.com/ad", "why": "similar vibe"},
        "never_do": "no loud music",
        "freeform": "keep it understated",
    }


@pytest.mark.asyncio
async def test_seller_direction_all_fields_null_is_skipped_entirely(conn):
    """C1's SellerDirection is `total=False` (every field optional) -- a job
    with no intake at all must round-trip as a clean ABSENCE of the
    seller_direction key, not a row of all-nulls (module's own documented
    contract)."""
    await create_job(conn, job_id="job-5", seller_id="s1", brief="brief")

    wrote_none = await upsert_seller_direction(conn, "job-5", None)
    wrote_empty = await upsert_seller_direction(conn, "job-5", {})

    assert wrote_none is False
    assert wrote_empty is False
    assert "job-5" not in conn.seller_direction  # no row was ever written

    state = await read_job_state(conn, "job-5")
    assert "seller_direction" not in state


@pytest.mark.asyncio
async def test_seller_direction_partial_fields_only_populated_keys_present(conn):
    """Nullability per-column, not just per-row: only `never_do` set -- the
    other three columns are genuinely NULL in the DB, and the read-back must
    omit them as keys entirely (not include them as `None`)."""
    await create_job(conn, job_id="job-6", seller_id="s1", brief="brief")
    wrote = await upsert_seller_direction(conn, "job-6", {"never_do": "no clichés"})
    assert wrote is True

    state = await read_job_state(conn, "job-6")
    assert state["seller_direction"] == {"never_do": "no clichés"}
    assert "mood_words" not in state["seller_direction"]
    assert "reference_ad" not in state["seller_direction"]
    assert "freeform" not in state["seller_direction"]


@pytest.mark.asyncio
async def test_seller_direction_reference_ad_only_reconstructed_when_url_present(conn):
    """`reference_ad` is a nested object rebuilt from two flat columns -- must
    only reappear when the URL/text column is actually populated, matching
    `_row_to_seller_direction`'s own guard."""
    await create_job(conn, job_id="job-7", seller_id="s1", brief="brief")
    await upsert_seller_direction(conn, "job-7", {"mood_words": ["bold"]})

    state = await read_job_state(conn, "job-7")
    assert "reference_ad" not in state["seller_direction"]
    assert state["seller_direction"]["mood_words"] == ["bold"]


@pytest.mark.asyncio
async def test_seller_direction_upsert_overwrites_previous_values(conn):
    await create_job(conn, job_id="job-8", seller_id="s1", brief="brief")
    await upsert_seller_direction(conn, "job-8", {"never_do": "first version"})
    await upsert_seller_direction(conn, "job-8", {"never_do": "second version", "freeform": "added later"})

    state = await read_job_state(conn, "job-8")
    assert state["seller_direction"] == {"never_do": "second version", "freeform": "added later"}


# ---------------------------------------------------------------------------
# delete_job (+ simulated ON DELETE CASCADE)
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_delete_job_removes_job_and_seller_direction(conn):
    await create_job(conn, job_id="job-9", seller_id="s1", brief="brief")
    await upsert_seller_direction(conn, "job-9", {"never_do": "x"})

    await delete_job(conn, "job-9")

    assert await read_job_state(conn, "job-9") is None
    assert "job-9" not in conn.seller_direction
