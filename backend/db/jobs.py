"""
Persistence for the `jobs` and `seller_direction` tables (RR, Phase 1).

Spec of record: docs/TECHNICAL_DOCUMENTATION.md §7.2 (the table listing) and
§5.1 (Ingest Node — "a new `jobs` row ... with status `ingested`; a
`seller_direction` row if any intake field was filled"). This module is the
persistence layer only: raw DDL + async read/write helpers wired to the C1
`ProductCutState` shape. It is NOT the Ingest Node, not an LLM agent, and not
the intake form — those are separate tasks.

Design choices
--------------
* **Async psycopg (psycopg 3).** Matches the established DB-access style in
  `graph/build.py`, which drives the real RDS instance via
  `AsyncPostgresSaver` (async, psycopg-backed). No SQLAlchemy ORM / Core layer
  and no migration framework (Alembic) is introduced — for two small tables in
  a hackathon Phase 1 an idempotent `CREATE TABLE IF NOT EXISTS` is the right
  amount of machinery, and staying on raw async psycopg keeps one DB idiom in
  the codebase rather than two.
* **C1 types are imported, never redefined.** `SellerDirection` and
  `ReferenceAd` come from `graph.state`, the same reuse pattern already used by
  `graph/events.py` and `graph/shot_schema.py`.
* **Nullability.** Every `seller_direction` column except the `job_id` PK/FK is
  nullable, matching `SellerDirection`'s `total=False` (all fields optional). A
  job with no intake at all simply has **no** `seller_direction` row (we skip
  the write); reads then omit the `seller_direction` key entirely, which is the
  faithful round-trip of "no direction given".

version: 1
"""
from __future__ import annotations

import os
from typing import Any, Optional

import psycopg
from psycopg.rows import dict_row

from graph.state import ReferenceAd, SellerDirection

# ---------------------------------------------------------------------------
# DDL — idempotent "create if not exists". Run once against the real DB.
# ---------------------------------------------------------------------------

# `jobs`: the root record for every submission (§7.2). `product_photo_refs`
# holds OSS URIs (binary assets live in OSS, never in the DB — §7.1), stored as
# a text array. `status` defaults to 'ingested' (§5.1). `created_at` is DB-side
# so the row's birth time is authoritative regardless of the caller's clock.
_CREATE_JOBS = """
CREATE TABLE IF NOT EXISTS jobs (
    job_id             TEXT PRIMARY KEY,
    seller_id          TEXT,
    brief              TEXT,
    status             TEXT NOT NULL DEFAULT 'ingested',
    created_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    product_photo_refs TEXT[] NOT NULL DEFAULT '{}'
);
"""

# `seller_direction`: the optional intake, persisted so any later chat edit can
# re-read it (§7.2). `job_id` is BOTH the PK and the FK to `jobs` — a 1:1
# relationship (one direction row per job) that also makes upsert-on-conflict
# natural. EVERY other column is nullable, matching SellerDirection(total=False).
# ON DELETE CASCADE so cleaning up a job removes its direction row too.
_CREATE_SELLER_DIRECTION = """
CREATE TABLE IF NOT EXISTS seller_direction (
    job_id                   TEXT PRIMARY KEY REFERENCES jobs(job_id) ON DELETE CASCADE,
    mood_words               TEXT[],
    reference_ad_url_or_text TEXT,
    reference_ad_why         TEXT,
    never_do                 TEXT,
    freeform                 TEXT
);
"""


# ---------------------------------------------------------------------------
# Connection helper
# ---------------------------------------------------------------------------

def _conninfo(conninfo: Optional[str] = None) -> str:
    """Resolve the Postgres connection string.

    Prefers an explicit argument, else falls back to DATABASE_URL from the
    environment (the same var `graph/build.py` reads). Raises if neither is
    available so callers fail loudly instead of silently degrading.
    """
    resolved = conninfo or os.getenv("DATABASE_URL")
    if not resolved:
        raise RuntimeError(
            "No Postgres connection string: pass conninfo= or set DATABASE_URL."
        )
    return resolved


async def connect(conninfo: Optional[str] = None) -> psycopg.AsyncConnection:
    """Open an async psycopg connection with dict-row results.

    Caller owns the connection lifecycle (use `async with`). `dict_row` makes
    reads return column-keyed dicts, which the reconstruction helpers rely on.
    """
    return await psycopg.AsyncConnection.connect(_conninfo(conninfo), row_factory=dict_row)


# ---------------------------------------------------------------------------
# Schema setup
# ---------------------------------------------------------------------------

async def init_tables(conn: psycopg.AsyncConnection) -> None:
    """Create the `jobs` and `seller_direction` tables if they don't exist.

    Idempotent — safe to call on every app startup. `jobs` is created first so
    the `seller_direction` FK target exists.
    """
    async with conn.cursor() as cur:
        await cur.execute(_CREATE_JOBS)
        await cur.execute(_CREATE_SELLER_DIRECTION)
    await conn.commit()


# ---------------------------------------------------------------------------
# Writes
# ---------------------------------------------------------------------------

async def create_job(
    conn: psycopg.AsyncConnection,
    job_id: str,
    seller_id: Optional[str],
    brief: Optional[str],
    product_photo_refs: Optional[list[str]] = None,
    status: str = "ingested",
) -> None:
    """Insert a new `jobs` row (status defaults to 'ingested', per §5.1).

    Upserts on `job_id` so a re-ingest of the same job is not a hard error;
    `created_at` is left to the DB default on first insert and preserved on
    conflict (we don't overwrite it).
    """
    async with conn.cursor() as cur:
        await cur.execute(
            """
            INSERT INTO jobs (job_id, seller_id, brief, status, product_photo_refs)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (job_id) DO UPDATE SET
                seller_id          = EXCLUDED.seller_id,
                brief              = EXCLUDED.brief,
                status             = EXCLUDED.status,
                product_photo_refs = EXCLUDED.product_photo_refs;
            """,
            (job_id, seller_id, brief, status, list(product_photo_refs or [])),
        )
    await conn.commit()


async def upsert_seller_direction(
    conn: psycopg.AsyncConnection,
    job_id: str,
    seller_direction: Optional[SellerDirection],
) -> bool:
    """Create/update the `seller_direction` row for a job.

    Takes a `SellerDirection`-shaped dict (all keys optional). Behaviour when
    there is nothing to persist — `seller_direction` is None or an empty dict,
    i.e. the seller filled in no intake field at all — is to **skip the write
    entirely** (no row is created). This matches §5.1 ("a `seller_direction`
    row *if any intake field was filled*") and keeps the "no direction" case as
    a clean absence rather than a row of all-nulls.

    Returns True if a row was written, False if the write was skipped.
    """
    if not seller_direction:
        return False

    # Flatten the nested ReferenceAd (C1) into the two flat columns from §7.2.
    reference_ad = seller_direction.get("reference_ad") or {}
    ref_url_or_text = reference_ad.get("url_or_text")
    ref_why = reference_ad.get("why")

    mood_words = seller_direction.get("mood_words")
    never_do = seller_direction.get("never_do")
    freeform = seller_direction.get("freeform")

    async with conn.cursor() as cur:
        await cur.execute(
            """
            INSERT INTO seller_direction (
                job_id, mood_words, reference_ad_url_or_text,
                reference_ad_why, never_do, freeform
            )
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (job_id) DO UPDATE SET
                mood_words               = EXCLUDED.mood_words,
                reference_ad_url_or_text = EXCLUDED.reference_ad_url_or_text,
                reference_ad_why         = EXCLUDED.reference_ad_why,
                never_do                 = EXCLUDED.never_do,
                freeform                 = EXCLUDED.freeform;
            """,
            (
                job_id,
                list(mood_words) if mood_words is not None else None,
                ref_url_or_text,
                ref_why,
                never_do,
                freeform,
            ),
        )
    await conn.commit()
    return True


# ---------------------------------------------------------------------------
# Reads / reconstruction into the C1 ProductCutState slice
# ---------------------------------------------------------------------------

def _row_to_seller_direction(row: dict[str, Any]) -> SellerDirection:
    """Rebuild a `SellerDirection` from a `seller_direction` DB row.

    Only non-null columns become keys, honouring `total=False` (an absent field
    is absent, not present-and-null). The nested `reference_ad` is rebuilt only
    when a reference was actually stored.
    """
    sd: SellerDirection = {}
    if row.get("mood_words") is not None:
        sd["mood_words"] = list(row["mood_words"])
    if row.get("reference_ad_url_or_text") is not None:
        ref: ReferenceAd = {
            "url_or_text": row["reference_ad_url_or_text"],
            "why": row.get("reference_ad_why") or "",
        }
        sd["reference_ad"] = ref
    if row.get("never_do") is not None:
        sd["never_do"] = row["never_do"]
    if row.get("freeform") is not None:
        sd["freeform"] = row["freeform"]
    return sd


async def read_job_state(
    conn: psycopg.AsyncConnection,
    job_id: str,
) -> Optional[dict[str, Any]]:
    """Read `jobs` (+ its optional `seller_direction`) back into the C1 shape.

    Returns the Ingest-phase slice of `ProductCutState`:
        {job_id, brief, product_photos, [seller_direction]}
    where `product_photos` is populated from `jobs.product_photo_refs` (the OSS
    URIs) and `seller_direction` is included **only** when a direction row with
    at least one populated field exists. Returns None if the job doesn't exist.
    """
    async with conn.cursor() as cur:
        await cur.execute(
            "SELECT job_id, brief, product_photo_refs FROM jobs WHERE job_id = %s",
            (job_id,),
        )
        job = await cur.fetchone()
        if job is None:
            return None

        await cur.execute(
            """
            SELECT mood_words, reference_ad_url_or_text, reference_ad_why,
                   never_do, freeform
            FROM seller_direction WHERE job_id = %s
            """,
            (job_id,),
        )
        direction = await cur.fetchone()

    state: dict[str, Any] = {
        "job_id": job["job_id"],
        "brief": job["brief"],
        "product_photos": list(job["product_photo_refs"] or []),
    }
    if direction is not None:
        sd = _row_to_seller_direction(direction)
        if sd:  # skip an all-null row (defensive; we never write one)
            state["seller_direction"] = sd
    return state


async def update_job_status(
    conn: psycopg.AsyncConnection,
    job_id: str,
    status: str,
) -> None:
    """Update the status column for an existing job row."""
    async with conn.cursor() as cur:
        await cur.execute(
            "UPDATE jobs SET status = %s WHERE job_id = %s",
            (status, job_id),
        )
    await conn.commit()


async def delete_job(conn: psycopg.AsyncConnection, job_id: str) -> None:
    """Delete a job and (via ON DELETE CASCADE) its `seller_direction` row.

    Used by tests/cleanup; the cascade means we only touch `jobs`.
    """
    async with conn.cursor() as cur:
        await cur.execute("DELETE FROM jobs WHERE job_id = %s", (job_id,))
    await conn.commit()


async def list_jobs(
    conn: psycopg.AsyncConnection,
    seller_id: Optional[str] = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """List jobs (newest first), optionally filtered by seller_id.

    Returns summary rows: job_id, brief, status, created_at. Intended for the
    Library panel -- not full state reconstruction (use read_job_state for that).
    """
    async with conn.cursor() as cur:
        if seller_id is not None:
            await cur.execute(
                """SELECT job_id, brief, status, created_at
                   FROM jobs WHERE seller_id = %s
                   ORDER BY created_at DESC LIMIT %s""",
                (seller_id, limit),
            )
        else:
            await cur.execute(
                """SELECT job_id, brief, status, created_at
                   FROM jobs ORDER BY created_at DESC LIMIT %s""",
                (limit,),
            )
        rows = await cur.fetchall()
    return [dict(r) for r in rows]


__all__ = [
    "connect",
    "init_tables",
    "create_job",
    "upsert_seller_direction",
    "read_job_state",
    "update_job_status",
    "delete_job",
    "list_jobs",
]
