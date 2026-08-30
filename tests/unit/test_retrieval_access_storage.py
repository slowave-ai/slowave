"""Storage contract tests for retrieval-access lifecycle evidence."""

from __future__ import annotations

import sqlite3

from slowave.core.config import SlowaveConfig
from slowave.storage.sqlite_db import SQLiteConfig, SQLiteDB


def _init_db(path) -> SQLiteDB:
    db = SQLiteDB(SQLiteConfig(path=str(path)))
    db.init_schema(SlowaveConfig.default_schema_path())
    return db


def _table_names(conn: sqlite3.Connection) -> set[str]:
    return {
        row["name"] for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
    }


def _index_names(conn: sqlite3.Connection, table: str) -> set[str]:
    return {row["name"] for row in conn.execute(f"PRAGMA index_list({table})")}


def test_fresh_database_creates_retrieval_access_tables_and_indexes(tmp_path) -> None:
    db = _init_db(tmp_path / "fresh.db")
    conn = db.connect()

    assert {"retrieval_cue_prototypes", "schema_retrieval_evidence"} <= _table_names(conn)
    assert _index_names(conn, "retrieval_cue_prototypes") >= {
        "idx_retrieval_cue_prototypes_scope",
    }
    assert _index_names(conn, "schema_retrieval_evidence") >= {
        "idx_schema_retrieval_evidence_cue_pathway",
        "idx_schema_retrieval_evidence_schema",
        "idx_schema_retrieval_evidence_access_state",
    }

    cue_columns = {
        row["name"]: row for row in conn.execute("PRAGMA table_info(retrieval_cue_prototypes)")
    }
    assert set(cue_columns) == {
        "id",
        "embedding",
        "dim",
        "scope_id",
        "scope_kind",
        "task_type",
        "support_count",
        "first_seen_ts",
        "last_seen_ts",
    }
    assert cue_columns["embedding"]["notnull"] == 1

    evidence_columns = {
        row["name"]: row for row in conn.execute("PRAGMA table_info(schema_retrieval_evidence)")
    }
    assert set(evidence_columns) == {
        "schema_id",
        "cue_prototype_id",
        "pathway",
        "useful_count",
        "irrelevant_count",
        "last_useful_ts",
        "last_irrelevant_ts",
        "inhibition_strength",
        "access_state",
        "updated_at",
    }
    assert evidence_columns["schema_id"]["pk"] == 1
    assert evidence_columns["cue_prototype_id"]["pk"] == 2
    assert evidence_columns["pathway"]["pk"] == 3


def test_upgraded_database_converges_without_backfilling_access_evidence(
    tmp_path,
) -> None:
    path = tmp_path / "legacy.db"
    db = _init_db(path)
    legacy = db.connect()
    legacy.execute("DROP TABLE schema_retrieval_evidence")
    legacy.execute("DROP TABLE retrieval_cue_prototypes")
    legacy.execute(
        "INSERT INTO schemas (content_text, first_formed_ts, last_updated_ts) "
        "VALUES ('legacy schema', 1, 1)"
    )
    legacy.execute(
        "INSERT INTO context_recall_events (context_id, created_at) VALUES ('ctx_legacy', 1)"
    )
    legacy.execute(
        "INSERT INTO context_feedback_events ("
        "context_id, feedback, feedback_signal_json, irrelevant_memory_ids_json, created_at"
        ") VALUES ('ctx_legacy', 'irrelevant', '{}', '[\"sch_1\"]', 1)"
    )
    legacy.commit()
    db.close()

    db = _init_db(path)
    conn = db.connect()

    assert {"retrieval_cue_prototypes", "schema_retrieval_evidence"} <= _table_names(conn)
    assert conn.execute("SELECT COUNT(*) FROM retrieval_cue_prototypes").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM schema_retrieval_evidence").fetchone()[0] == 0
    assert (
        conn.execute("SELECT content_text FROM schemas WHERE id = 1").fetchone()[0]
        == "legacy schema"
    )
    assert (
        conn.execute(
            "SELECT irrelevant_memory_ids_json FROM context_feedback_events WHERE id = 1"
        ).fetchone()[0]
        == '["sch_1"]'
    )

    db.init_schema(SlowaveConfig.default_schema_path())
    assert conn.execute("SELECT COUNT(*) FROM retrieval_cue_prototypes").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM schema_retrieval_evidence").fetchone()[0] == 0


def test_access_evidence_defaults_and_cascades_with_its_derived_rows(tmp_path) -> None:
    db = _init_db(tmp_path / "evidence.db")
    conn = db.connect()
    conn.execute(
        "INSERT INTO schemas (content_text, first_formed_ts, last_updated_ts) "
        "VALUES ('schema with access evidence', 1, 1)"
    )
    schema_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    conn.execute(
        "INSERT INTO retrieval_cue_prototypes "
        "(embedding, dim, scope_id, first_seen_ts, last_seen_ts) "
        "VALUES (?, 2, 'project:test', 1, 1)",
        (b"\x00\x00\x00\x00\x00\x00\x00\x00",),
    )
    cue_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    conn.execute(
        "INSERT INTO schema_retrieval_evidence "
        "(schema_id, cue_prototype_id, pathway, updated_at) VALUES (?, ?, 'direct', 1)",
        (schema_id, cue_id),
    )
    conn.commit()

    evidence = conn.execute("SELECT * FROM schema_retrieval_evidence").fetchone()
    assert evidence["useful_count"] == 0
    assert evidence["irrelevant_count"] == 0
    assert evidence["inhibition_strength"] == 0.0
    assert evidence["access_state"] == "eligible"

    conn.execute("DELETE FROM retrieval_cue_prototypes WHERE id = ?", (cue_id,))
    conn.commit()
    assert conn.execute("SELECT COUNT(*) FROM schema_retrieval_evidence").fetchone()[0] == 0
