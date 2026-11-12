"""The opt-in retrieval trace records how the response was ranked.

Without these fields the dashboard can show a policy label but never explains
that a retrieval was served by a fallback rule.
"""

from __future__ import annotations

import sqlite3

from slowave.core import retrieval_trace

_OLD_RUNS = (
    "CREATE TABLE retrieval_trace_runs ("
    " retrieval_id TEXT PRIMARY KEY, policy_version TEXT NOT NULL, mode TEXT, limit_n INTEGER,"
    " catalog_truncated INTEGER NOT NULL DEFAULT 0, selected_total INTEGER NOT NULL,"
    " scored_total INTEGER NOT NULL, gate_rejections_json TEXT NOT NULL DEFAULT '{}',"
    " created_at INTEGER NOT NULL);"
)
_OLD_CANDIDATES = (
    "CREATE TABLE retrieval_trace_candidates ("
    " retrieval_id TEXT NOT NULL, candidate_kind TEXT NOT NULL, candidate_id TEXT NOT NULL,"
    " decision TEXT NOT NULL, reason_code TEXT NOT NULL, selected_position INTEGER,"
    " dense_cosine REAL, lexical_rank INTEGER, rrf_score REAL, applicability REAL,"
    " threshold REAL, e5_cosine REAL, redundant_with TEXT,"
    " PRIMARY KEY (retrieval_id, candidate_kind, candidate_id)) WITHOUT ROWID;"
)


def _db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(
        "CREATE TABLE context_recall_events (context_id TEXT PRIMARY KEY);"
        "INSERT INTO context_recall_events(context_id) VALUES ('ctx_1');"
    )
    return conn


def _record(conn: sqlite3.Connection, **kwargs: object) -> None:
    retrieval_trace.record(
        conn,
        retrieval_id="ctx_1",
        policy_version="activation-calibrated-v2",
        mode="strict_scope",
        limit=8,
        catalog_truncated=False,
        selected=[],
        filtered_items=[],
        decision_candidates=[],
        **kwargs,  # type: ignore[arg-type]
    )


def test_record_persists_applicability_and_fallback() -> None:
    conn = _db()
    _record(conn, applicability_status="no_model_encoder", policy_fallback="no_model_encoder")
    row = conn.execute(
        "SELECT policy_version, applicability_status, policy_fallback FROM retrieval_trace_runs"
    ).fetchone()
    assert row["policy_version"] == "activation-calibrated-v2"
    assert row["applicability_status"] == "no_model_encoder"
    assert row["policy_fallback"] == "no_model_encoder"


def test_healthy_runs_record_no_fallback() -> None:
    conn = _db()
    _record(conn, applicability_status="applied")
    row = conn.execute(
        "SELECT applicability_status, policy_fallback FROM retrieval_trace_runs"
    ).fetchone()
    assert row["applicability_status"] == "applied"
    assert row["policy_fallback"] is None


def test_trace_tables_predating_the_columns_are_upgraded_in_place() -> None:
    conn = _db()
    conn.executescript(_OLD_RUNS + _OLD_CANDIDATES)
    _record(conn, applicability_status="disabled", policy_fallback="disabled")
    row = conn.execute(
        "SELECT applicability_status, policy_fallback FROM retrieval_trace_runs"
    ).fetchone()
    assert row["applicability_status"] == "disabled"
    assert row["policy_fallback"] == "disabled"
