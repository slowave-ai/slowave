"""Ordered per-session call log (``session_events``).

The table is telemetry/index only: content stays in the domain tables
(``raw_events``, ``context_recall_events``, ``feedback_events``) and is
referenced by ID. ``id`` (AUTOINCREMENT) is the ordering key.
"""

from __future__ import annotations

import logging
import sqlite3
import time

log = logging.getLogger(__name__)

BACKFILL_VERSION = "session-events-backfill-v2"
_RETIRED_BACKFILL_VERSIONS = ("session-events-backfill-v1",)

# Legacy ledger ``retrieval_type`` -> session event kind.
LEDGER_KINDS = {
    "context": "activate",
    "activate": "activate",
    "recall": "recall",
    "recall_page": "recall_page",
    "remember": "remember",
    "feedback": "feedback",
    "commit": "commit",
}
_KIND_RANK = {
    "activate": 0,
    "recall": 1,
    "recall_page": 1,
    "remember": 2,
    "feedback": 3,
    "commit": 4,
    "session_reaped": 5,
}


def record(
    conn: sqlite3.Connection,
    *,
    session_id: str | None,
    kind: str,
    retrieval_id: str | None = None,
    raw_event_id: int | None = None,
    status: str = "ok",
    error_code: str | None = None,
    duration_ms: int | None = None,
    response_chars: int | None = None,
    estimated_tokens: int | None = None,
    summary: str | None = None,
    ts: int | None = None,
) -> None:
    """Insert one live event. Caller owns the transaction. Never raises.

    Skipped when the session does not exist, so unknown/ended-before-creation
    sessions cannot violate the FK.
    """
    if not session_id or session_id == "unknown":
        return
    try:
        conn.execute(
            "INSERT INTO session_events (session_id, ts, kind, status, error_code, duration_ms,"
            " response_chars, estimated_tokens, retrieval_id, raw_event_id, summary, origin)"
            " SELECT ?,?,?,?,?,?,?,?,?,?,?, 'live' WHERE EXISTS"
            " (SELECT 1 FROM sessions WHERE id=?)",
            (
                session_id,
                int(time.time()) if ts is None else int(ts),
                kind,
                status,
                error_code,
                duration_ms,
                response_chars,
                estimated_tokens,
                retrieval_id,
                raw_event_id,
                (summary[:200] if summary else None),
                session_id,
            ),
        )
    except sqlite3.Error:
        log.warning("Could not record session event %s", kind, exc_info=True)


def _candidates(conn: sqlite3.Connection, session_ids: list[str]) -> list[tuple]:
    """One tuple per *call*: (session, ts, kind, retrieval_id, raw_event_id, chars, tokens, summary, key).

    Each call is derived from exactly one source so nothing is counted twice:
    activate/recall from the retrieval ledger, feedback from ``feedback_events``
    (one per assessed retrieval), remember/commit from ``raw_events``. The thin
    remember/feedback/commit ledger rows only duplicate those and are ignored.
    """
    marks = ",".join("?" * len(session_ids))
    rows: list[tuple] = []
    for r in conn.execute(
        f"SELECT session_id, created_at, retrieval_type, context_id, response_chars, estimated_tokens"
        f" FROM context_recall_events WHERE session_id IN ({marks})"
        f" AND retrieval_type IN ('context','activate','recall','recall_page')",
        session_ids,
    ):
        kind = LEDGER_KINDS[r[2]]
        rows.append((r[0], r[1], kind, r[3], None, r[4], r[5], None, r[3]))
    for r in conn.execute(
        f"SELECT session_id, MIN(created_at), retrieval_id FROM feedback_events"
        f" WHERE session_id IN ({marks}) AND retrieval_id IS NOT NULL"
        f" GROUP BY session_id, retrieval_id",
        session_ids,
    ):
        rows.append((r[0], r[1], "feedback", r[2], None, None, None, None, r[2]))
    for r in conn.execute(
        f"SELECT session_id, ts, type, id, substr(content, 1, 200) FROM raw_events"
        f" WHERE session_id IN ({marks}) AND (type LIKE 'remember:%' OR type='task_complete')",
        session_ids,
    ):
        kind = "commit" if r[2] == "task_complete" else "remember"
        rows.append(
            (
                r[0],
                r[1],
                kind,
                None,
                r[3],
                None,
                None,
                None if kind == "commit" else r[4],
                f"raw{r[3]}",
            )
        )
    return rows


def _retire_old_backfills(conn: sqlite3.Connection) -> None:
    """Drop rows written by earlier backfill versions so they are rebuilt correctly."""
    marks = ",".join("?" * len(_RETIRED_BACKFILL_VERSIONS))
    if not conn.execute(
        f"SELECT 1 FROM storage_migrations WHERE version IN ({marks})", _RETIRED_BACKFILL_VERSIONS
    ).fetchone():
        return
    conn.execute("BEGIN IMMEDIATE")
    try:
        conn.execute("DELETE FROM session_events WHERE origin='backfill'")
        conn.execute(
            f"DELETE FROM storage_migrations WHERE version IN ({marks})", _RETIRED_BACKFILL_VERSIONS
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def backfill_batch(conn: sqlite3.Connection, batch_sessions: int = 200) -> int:
    """Backfill ``session_events`` for one batch of sessions. Returns sessions handled.

    Idempotent and safe across concurrent processes: each batch runs inside
    ``BEGIN IMMEDIATE`` (single writer) and the partial unique indexes dedupe
    ``origin='backfill'`` rows. Sessions that already have *live* rows are
    skipped so backfilled history can never be ordered after live events.
    Order within a session: ts, then call-kind rank, then source id
    (second-resolution; the UI flags backfilled sessions as approximate).
    """
    if conn.execute(
        "SELECT 1 FROM storage_migrations WHERE version=?", (BACKFILL_VERSION,)
    ).fetchone():
        return 0
    conn.execute("BEGIN IMMEDIATE")
    try:
        ids = [
            r[0]
            for r in conn.execute(
                "SELECT s.id FROM sessions s WHERE NOT EXISTS"
                " (SELECT 1 FROM session_events e WHERE e.session_id=s.id)"
                " ORDER BY s.started_ts, s.id LIMIT ?",
                (batch_sessions,),
            )
        ]
        if not ids:
            conn.execute(
                "INSERT OR IGNORE INTO storage_migrations(version, applied_at) VALUES (?, ?)",
                (BACKFILL_VERSION, int(time.time())),
            )
            conn.commit()
            return 0
        rows = _candidates(conn, ids)
        rows.sort(key=lambda x: (x[0], x[1] or 0, _KIND_RANK.get(x[2], 9), str(x[8])))
        for sid, ts, kind, rid, raw_id, chars, tokens, summary, _ in rows:
            conn.execute(
                "INSERT OR IGNORE INTO session_events (session_id, ts, kind, retrieval_id,"
                " raw_event_id, response_chars, estimated_tokens, summary, origin)"
                " VALUES (?,?,?,?,?,?,?,?, 'backfill')",
                (sid, int(ts or 0), kind, rid, raw_id, chars, tokens, summary),
            )
        # Sessions with no recoverable events still need a marker to avoid
        # being re-selected forever.
        for sid in set(ids) - {r[0] for r in rows}:
            conn.execute(
                "INSERT INTO session_events (session_id, ts, kind, origin)"
                " SELECT id, COALESCE(started_ts, 0), 'session_start', 'backfill'"
                " FROM sessions WHERE id=?",
                (sid,),
            )
        conn.commit()
        return len(ids)
    except Exception:
        conn.rollback()
        raise


def backfill_evidence_sessions(conn: sqlite3.Connection, batch: int = 5000) -> int:
    """Populate ``schema_evidence.session_id`` for legacy rows. Returns rows updated."""
    cur = conn.execute(
        "UPDATE schema_evidence SET session_id = COALESCE("
        " (SELECT session_id FROM raw_events WHERE id = schema_evidence.raw_event_id),"
        " (SELECT session_id FROM episode_text WHERE episode_id = schema_evidence.episode_id))"
        " WHERE rowid IN (SELECT rowid FROM schema_evidence WHERE session_id IS NULL LIMIT ?)"
        " AND COALESCE("
        " (SELECT session_id FROM raw_events WHERE id = schema_evidence.raw_event_id),"
        " (SELECT session_id FROM episode_text WHERE episode_id = schema_evidence.episode_id))"
        " IS NOT NULL",
        (batch,),
    )
    conn.commit()
    return cur.rowcount


def run_backfill(conn: sqlite3.Connection) -> dict[str, int]:
    """Run all backfills to completion in bounded batches (background job)."""
    _retire_old_backfills(conn)
    sessions = 0
    while True:
        n = backfill_batch(conn)
        if not n:
            break
        sessions += n
    evidence = 0
    while True:
        n = backfill_evidence_sessions(conn)
        if not n:
            break
        evidence += n
    return {"sessions": sessions, "evidence": evidence}


def ensure_schema(db_path: str):
    """Apply the (additive, idempotent) schema to *db_path*; return the open DB wrapper.

    Used by processes that only read (dashboard) or do not build an engine at
    start-up, so they never query a pre-migration database.
    """
    from slowave.core.config import SlowaveConfig
    from slowave.storage.sqlite_db import SQLiteConfig, SQLiteDB

    db = SQLiteDB(SQLiteConfig(path=db_path))
    db.init_schema(SlowaveConfig.default_schema_path())
    return db


def start_backfill_thread(db_path: str):
    """Run :func:`run_backfill` once in a daemon thread. Never raises into the caller."""
    import threading

    def _run() -> None:
        db = None
        try:
            db = ensure_schema(db_path)
            stats = run_backfill(db.connect())
            if stats["sessions"] or stats["evidence"]:
                log.info("session_events backfill: %s", stats)
        except Exception:
            log.warning("session_events backfill failed; will retry on next start", exc_info=True)
        finally:
            if db is not None:
                db.close()

    thread = threading.Thread(target=_run, daemon=True, name="slowave-session-backfill")
    thread.start()
    return thread
