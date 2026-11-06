"""Opt-in retrieval trace for maintainers (off by default, never shipped on).

A trace answers "why was this candidate returned or skipped?" with the real
channel scores, so retrieval quality can be analysed offline. It is a
maintainer-only diagnostic: public installs write nothing and the tables do
not exist unless tracing is enabled.

Enable either way (no restart needed for the marker file):

* ``touch "$SLOWAVE_HOME/.retrieval_trace"`` next to ``slowave.db`` (and
  ``rm`` it to stop), or
* set ``SLOWAVE_RETRIEVAL_TRACE=1`` in the daemon's environment.

Tables, created lazily on the first traced retrieval:

``retrieval_trace_runs``
    One row per retrieval: policy, mode, how many candidates were scored or
    selected, and ``gate_rejections_json`` -- counts of candidates the
    eligibility gate rejected before scoring, by reason (those carry no
    scores, so they are counted instead of stored row by row).

``retrieval_trace_candidates``
    One row per candidate that was selected, suppressed, or reached scoring.
    Join ``retrieval_id``/``candidate_id`` to ``context_recall_items`` and
    ``context_feedback_events`` for what was delivered and used.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

MARKER_FILENAME = ".retrieval_trace"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS retrieval_trace_runs (
  retrieval_id         TEXT PRIMARY KEY,
  policy_version       TEXT NOT NULL,
  mode                 TEXT,
  limit_n              INTEGER,
  catalog_truncated    INTEGER NOT NULL DEFAULT 0,
  selected_total       INTEGER NOT NULL,
  scored_total         INTEGER NOT NULL,
  gate_rejections_json TEXT NOT NULL DEFAULT '{}',
  created_at           INTEGER NOT NULL,
  FOREIGN KEY (retrieval_id) REFERENCES context_recall_events(context_id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS retrieval_trace_candidates (
  retrieval_id       TEXT NOT NULL,
  candidate_kind     TEXT NOT NULL,
  candidate_id       TEXT NOT NULL,
  decision           TEXT NOT NULL,
  reason_code        TEXT NOT NULL,
  selected_position  INTEGER,
  dense_cosine       REAL,
  lexical_rank       INTEGER,
  rrf_score          REAL,
  applicability      REAL,
  threshold          REAL,
  e5_cosine          REAL,
  redundant_with     TEXT,
  PRIMARY KEY (retrieval_id, candidate_kind, candidate_id),
  FOREIGN KEY (retrieval_id) REFERENCES retrieval_trace_runs(retrieval_id) ON DELETE CASCADE
) WITHOUT ROWID;
"""

_INSERT = (
    "INSERT OR REPLACE INTO retrieval_trace_candidates "
    "(retrieval_id, candidate_kind, candidate_id, decision, reason_code, selected_position, "
    "dense_cosine, lexical_rank, rrf_score, applicability, threshold, e5_cosine, redundant_with) "
    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
)


def enabled(db_path: str | None) -> bool:
    """True when tracing is switched on via env var or the marker file."""
    if os.environ.get("SLOWAVE_RETRIEVAL_TRACE", "").strip().lower() in {"1", "true", "yes", "on"}:
        return True
    if not db_path or db_path == ":memory:":
        return False
    return (Path(db_path).parent / MARKER_FILENAME).exists()


def _evidence_scores(evidence: list[dict[str, Any]] | tuple[Any, ...] | None) -> dict[str, Any]:
    """Collapse per-query channel evidence to the best value per signal."""
    best: dict[str, Any] = {}
    for entry in evidence or ():
        for key, pick in (
            ("dense_cosine", max),
            ("rrf_score", max),
            ("applicability_score", max),
            ("e5_cosine", max),
            ("lexical_rank", min),
        ):
            value = entry.get(key)
            if value is not None:
                best[key] = pick(best[key], value) if key in best else value
        if entry.get("threshold") is not None:
            best["threshold"] = entry["threshold"]
    return best


def record(
    conn: sqlite3.Connection,
    *,
    retrieval_id: str,
    policy_version: str,
    mode: str,
    limit: int,
    catalog_truncated: bool,
    selected: list[tuple[str, str, dict[str, Any]]],
    filtered_items: list[dict[str, Any]],
    decision_candidates: list[dict[str, Any]],
) -> None:
    """Persist one retrieval's trace. Caller wraps this fail-open."""
    rows: dict[tuple[str, str], tuple[Any, ...]] = {}
    gate_rejections: dict[str, int] = {}

    for position, (kind, candidate_id, item) in enumerate(selected):
        rows[(kind, candidate_id)] = (
            retrieval_id,
            kind,
            candidate_id,
            "selected",
            str(item.get("reason") or "selected"),
            position,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )
    for item in decision_candidates:
        candidate_id = str(item.get("candidate_id") or "")
        if not candidate_id:
            continue
        kind = str(item.get("candidate_kind") or "memory")
        if (kind, candidate_id) in rows and item.get("decision") != "selected":
            continue
        decision = str(item.get("decision") or "rejected")
        reason = str(item.get("reason_code") or "not_selected")
        scores = _evidence_scores(item.get("channel_evidence"))
        if decision == "rejected" and not scores:
            gate_rejections[reason] = gate_rejections.get(reason, 0) + 1
            continue
        rows[(kind, candidate_id)] = (
            retrieval_id,
            kind,
            candidate_id,
            decision,
            reason,
            item.get("selected_position"),
            scores.get("dense_cosine"),
            scores.get("lexical_rank"),
            scores.get("rrf_score"),
            scores.get("applicability_score"),
            scores.get("threshold"),
            scores.get("e5_cosine"),
            item.get("redundant_with"),
        )
    for item in filtered_items:
        candidate_id = str(item.get("memory_id") or "")
        if not candidate_id or ("memory", candidate_id) in rows:
            continue
        rows[("memory", candidate_id)] = (
            retrieval_id,
            "memory",
            candidate_id,
            "rejected",
            str(item.get("reason") or "filtered"),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )

    scored_total = sum(1 for row in rows.values() if row[6] is not None or row[9] is not None)
    if not _tables_exist(conn):
        conn.executescript(_SCHEMA)
    with conn:
        conn.execute(
            "INSERT OR REPLACE INTO retrieval_trace_runs "
            "(retrieval_id, policy_version, mode, limit_n, catalog_truncated, selected_total, "
            "scored_total, gate_rejections_json, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                retrieval_id,
                policy_version,
                mode,
                int(limit),
                int(bool(catalog_truncated)),
                len(selected),
                scored_total,
                json.dumps(gate_rejections, sort_keys=True),
                int(time.time()),
            ),
        )
        conn.executemany(_INSERT, list(rows.values()))


def _tables_exist(conn: sqlite3.Connection) -> bool:
    row = conn.execute(
        "SELECT COUNT(*) FROM sqlite_master WHERE type='table' "
        "AND name IN ('retrieval_trace_runs', 'retrieval_trace_candidates')"
    ).fetchone()
    return int(row[0]) == 2
