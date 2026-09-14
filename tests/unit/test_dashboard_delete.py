"""Hard-delete previews and cascading cleanup for dashboard records."""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.request import urlopen

from slowave import ops
from slowave.core.config import SlowaveConfig
from slowave.core.engine import SlowaveEngine
from slowave.dashboard.app import (
    _delete_procedure_action,
    _delete_schema_action,
    _make_handler,
    _procedure_delete_preview,
    _schema_delete_preview,
)


def _engine() -> tuple[SlowaveEngine, str]:
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    return SlowaveEngine(SlowaveConfig(db_path=tmp.name, dim=8, disable_encoder=True)), tmp.name


def _cleanup(path: str) -> None:
    for suffix in ("", "-wal", "-shm"):
        try:
            os.remove(path + suffix)
        except FileNotFoundError:
            pass


def _retrieval_reference(conn, entity_id: str) -> str:
    context_id = f"ctx_delete_{entity_id}"
    now = int(time.time())
    response = {
        "memory_ids": [entity_id],
        "procedure_ids": [entity_id],
        "memories": [{"id": entity_id, "content": "dependent preview"}],
    }
    conn.execute(
        "INSERT INTO context_recall_events (context_id, memory_ids_json, response_json, created_at) VALUES (?, ?, ?, ?)",
        (context_id, json.dumps([entity_id]), json.dumps(response), now),
    )
    conn.execute(
        "INSERT INTO context_recall_items (context_id, memory_id, retrieval_type, memory_type, rank, created_at) VALUES (?, ?, 'recall', 'memory', 1, ?)",
        (context_id, entity_id, now),
    )
    conn.execute(
        "INSERT INTO context_feedback_events (context_id, retrieval_type, feedback, feedback_signal_json, used_memory_ids_json, used_procedure_ids_json, created_at) VALUES (?, 'recall', 'useful', '{}', ?, ?, ?)",
        (context_id, json.dumps([entity_id]), json.dumps([entity_id]), now),
    )
    conn.execute(
        "INSERT INTO feedback_events (event_id, retrieval_id, target_kind, target_id, coverage, source_contract, created_at) VALUES (?, ?, 'memory', ?, 'complete', 'test', ?)",
        (f"fbe_delete_{entity_id}", context_id, entity_id, now),
    )
    conn.commit()
    return context_id


def test_schema_hard_delete_previews_and_removes_dependents() -> None:
    eng, path = _engine()
    try:
        schema_id = eng.schemas.create(
            content_text="delete this durable memory",
            facets={},
            tags=[],
            embedding=None,
            dedupe=False,
        )
        conn = eng.db.connect()
        context_id = _retrieval_reference(conn, f"sch_{schema_id}")
        preview = _schema_delete_preview(path, schema_id)
        assert preview["entity"]["id"] == f"sch_{schema_id}"
        assert any(item["kind"] == "retrieval item" for item in preview["affected"])
        assert any(item["action"] == "reference removed" for item in preview["affected"])

        result = _delete_schema_action(path, schema_id)
        assert result["deleted"] is True
        assert conn.execute("SELECT 1 FROM schemas WHERE id=?", (schema_id,)).fetchone() is None
        assert (
            conn.execute(
                "SELECT 1 FROM context_recall_items WHERE context_id=?", (context_id,)
            ).fetchone()
            is None
        )
        snapshot = json.loads(
            conn.execute(
                "SELECT response_json FROM context_recall_events WHERE context_id=?", (context_id,)
            ).fetchone()[0]
        )
        assert f"sch_{schema_id}" not in json.dumps(snapshot)
        feedback = conn.execute(
            "SELECT used_memory_ids_json FROM context_feedback_events WHERE context_id=?",
            (context_id,),
        ).fetchone()[0]
        assert f"sch_{schema_id}" not in feedback
    finally:
        eng.close()
        _cleanup(path)


def test_schema_delete_preview_is_dispatched_before_generic_detail_route() -> None:
    """The preview suffix must never be parsed as a numeric schema ID."""
    eng, path = _engine()
    server = None
    try:
        schema_id = eng.schemas.create(
            content_text="preview this durable memory",
            facets={},
            tags=[],
            embedding=None,
            dedupe=False,
        )
        handler = _make_handler(db_path=path, refresh_ms=1000, allow_actions=True)
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()

        url = f"http://127.0.0.1:{server.server_port}/api/schemas/{schema_id}/delete-preview"
        with urlopen(url) as response:
            payload = json.loads(response.read())

        assert payload["entity"]["id"] == f"sch_{schema_id}"
        assert "invalid request" not in payload
    finally:
        if server is not None:
            server.shutdown()
            server.server_close()
        eng.close()
        _cleanup(path)


def test_procedure_hard_delete_removes_embedded_definition_and_references() -> None:
    eng, path = _engine()
    try:
        started = ops.activate(
            eng,
            query="delete procedure",
            scope="project:test",
            goal="delete procedure",
            agent="test",
        )
        ops.commit(
            eng,
            session_id=started["session_id"],
            outcome="success",
            procedure={
                "summary": "Delete procedure preview",
                "context": {},
                "steps": [{"summary": "Inspect"}],
                "caveats": [],
            },
        )
        procedure_id = f"proc_{started['session_id']}"
        conn = eng.db.connect()
        context_id = _retrieval_reference(conn, procedure_id)
        preview = _procedure_delete_preview(path, procedure_id)
        assert preview["entity"]["id"] == procedure_id
        assert any(item["kind"] == "procedure definition" for item in preview["affected"])

        result = _delete_procedure_action(path, procedure_id)
        assert result["deleted"] is True
        source = conn.execute(
            "SELECT metadata_json FROM raw_events WHERE session_id=? AND type='task_complete'",
            (started["session_id"],),
        ).fetchone()[0]
        assert "procedure" not in json.loads(source)
        assert (
            conn.execute(
                "SELECT 1 FROM context_recall_items WHERE context_id=?", (context_id,)
            ).fetchone()
            is None
        )
        snapshot = conn.execute(
            "SELECT response_json FROM context_recall_events WHERE context_id=?", (context_id,)
        ).fetchone()[0]
        assert procedure_id not in snapshot
    finally:
        eng.close()
        _cleanup(path)


def test_legacy_forgotten_status_is_restored_and_audit_table_removed() -> None:
    eng, path = _engine()
    try:
        schema_id = eng.schemas.create(
            content_text="legacy suppressed memory",
            facets={},
            tags=[],
            embedding=None,
            dedupe=False,
        )
        conn = eng.db.connect()
        conn.execute(
            "CREATE TABLE schema_forget_log (id INTEGER PRIMARY KEY, schema_id INTEGER, action TEXT, prior_status TEXT, reason TEXT, created_ts INTEGER)"
        )
        conn.execute("UPDATE schemas SET status='forgotten' WHERE id=?", (schema_id,))
        conn.execute(
            "INSERT INTO schema_forget_log VALUES (1, ?, 'forget', 'needs_review', NULL, ?)",
            (schema_id, int(time.time())),
        )
        conn.commit()

        eng.db.init_schema(str(Path(__file__).parents[2] / "slowave/storage/schema.sql"))

        assert (
            conn.execute("SELECT status FROM schemas WHERE id=?", (schema_id,)).fetchone()[0]
            == "needs_review"
        )
        assert (
            conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='schema_forget_log'"
            ).fetchone()
            is None
        )
    finally:
        eng.close()
        _cleanup(path)
