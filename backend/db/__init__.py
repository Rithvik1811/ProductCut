"""
Application persistence layer (Phase 1).

Structured relational state for ProductCut lives in the managed
Postgres-compatible RDS instance (see docs/TECHNICAL_DOCUMENTATION.md §7).
This package holds the hand-written table DDL + async read/write helpers for
those application tables. It is deliberately separate from `graph/` (which
owns the frozen C1/C2/C3 *schemas*) — this package owns *persistence of* those
schemas, not the schemas themselves.

The LangGraph checkpoint tables are NOT managed here; those are owned by
`langgraph-checkpoint-postgres` and set up in `graph/build.py`.
"""
