"""Tests for the offline, audit-preserving Phase-1 backfill comparison."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from private.experiments.compare_retrieval_access_backfill import compare
from slowave.core.config import SlowaveConfig
from slowave.storage.sqlite_db import SQLiteConfig, SQLiteDB


def _db(path):
    db = SQLiteDB(SQLiteConfig(path=str(path)))
    db.init_schema(SlowaveConfig.default_schema_path())
    return db


def _snapshot(conn, context_id: str, *, cue: bytes | None, dim: int | None) -> None:
    conn.execute(
        "INSERT INTO context_recall_events (context_id, cue_embedding, cue_dim, created_at) VALUES (?, ?, ?, 1)",
        (context_id, cue, dim),
    )


def _item(
    conn, context_id: str, memory_id: str, *, pathway: str = "direct", admitted: int = 1
) -> None:
    conn.execute(
        """INSERT INTO context_recall_items
           (context_id, memory_id, memory_type, rank, admitted, pathway, created_at)
           VALUES (?, ?, 'schema', 0, ?, ?, 1)""",
        (context_id, memory_id, admitted, pathway),
    )


def _feedback(conn, context_id: str, *, used: list[str] = [], irrelevant: list[str] = []) -> None:
    conn.execute(
        """INSERT INTO context_feedback_events
           (context_id, feedback, feedback_signal_json, used_memory_ids_json,
            irrelevant_memory_ids_json, created_at)
           VALUES (?, 'useful', '{}', ?, ?, 1)""",
        (context_id, json.dumps(used), json.dumps(irrelevant)),
    )


def test_comparison_uses_only_trusted_snapshot_linked_explicit_marks(tmp_path) -> None:
    path = tmp_path / "comparison.db"
    db = _db(path)
    conn = db.connect()
    cue = b"\0" * 8
    _snapshot(conn, "trusted", cue=cue, dim=2)
    _item(conn, "trusted", "sch_1", pathway="direct")
    _feedback(conn, "trusted", used=["sch_1"])
    _snapshot(conn, "no_cue", cue=None, dim=None)
    _item(conn, "no_cue", "sch_2")
    _feedback(conn, "no_cue", irrelevant=["sch_2"])
    _snapshot(conn, "not_admitted", cue=cue, dim=2)
    _item(conn, "not_admitted", "sch_3", admitted=0)
    _feedback(conn, "not_admitted", irrelevant=["sch_3"])
    _snapshot(conn, "conflict", cue=cue, dim=2)
    _item(conn, "conflict", "sch_4")
    _feedback(conn, "conflict", used=["sch_4"], irrelevant=["sch_4"])
    conn.commit()
    db.close()

    report = compare(path)

    assert report["source_read_only"] is True
    assert report["source_sha256_unchanged"] is True
    assert report["retrieval_admission_changed"] is False
    assert report["zero_start"]["evidence_rows"] == 0
    assert report["trusted_snapshot_linked_backfill"] == {
        "feedback_events_examined": 4,
        "trusted_explicit_marks": 1,
        "trusted_marks_by_label": {"useful": 1},
        "trusted_marks_by_pathway": {"direct": 1},
        "candidate_evidence_rows": 1,
        "useful_marks": 1,
        "irrelevant_marks": 0,
    }
    assert report["audit_only"] == {
        "ambiguous_marks": 3,
        "reason_counts": {
            "conflicting_explicit_labels": 1,
            "missing_or_invalid_snapshot_cue": 1,
            "not_admitted_in_snapshot": 1,
        },
    }
