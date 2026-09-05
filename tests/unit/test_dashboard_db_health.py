"""Coverage for the database-health dashboard payload and presentation."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from slowave.dashboard.app import _db_health


def test_db_health_payload_describes_storage_configuration_and_objects(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "health.sqlite3"
    conn = sqlite3.connect(db_path)
    conn.executescript("""
        PRAGMA foreign_keys = ON;
        CREATE TABLE parent (id INTEGER PRIMARY KEY);
        CREATE TABLE child (
            id INTEGER PRIMARY KEY,
            parent_id INTEGER REFERENCES parent(id)
        );
        CREATE INDEX child_parent_idx ON child(parent_id);
        CREATE VIEW child_view AS SELECT * FROM child;
        INSERT INTO parent VALUES (1);
        INSERT INTO child VALUES (1, 1);
        """)
    conn.commit()
    conn.close()

    payload = _db_health(str(db_path))

    assert payload["db_exists"] is True
    assert payload["integrity_check"] == ["ok"]
    assert payload["foreign_key_check"] == []
    assert payload["file_size_bytes"] > 0
    assert payload["sqlite_version"]
    assert payload["storage"]["allocated_bytes"] >= payload["storage"]["used_bytes"]
    assert 0 <= payload["storage"]["utilization_percent"] <= 100
    assert payload["object_counts"] == {"index": 1, "table": 2, "view": 1}
    assert {
        item["name"]: item["count"] for item in payload["tables"] if item["type"] == "table"
    } == {
        "child": 1,
        "parent": 1,
    }


def test_db_health_is_progressively_disclosed_in_diagnostics() -> None:
    source = Path(__file__).parents[2] / "slowave/dashboard/ui/src/pages.tsx"
    app = source.read_text()
    assert "Diagnostics" in app
    assert 'title="Storage"' in app
    assert "/api/db/health" in app
    assert "integrity_status" in app
    assert "formatBytes(status.data.db_size_bytes)" in app
    assert "formatBytes(status.data.wal_size_bytes)" in app
    assert "object_counts" not in app
    assert "Database details" not in app
    assert "LoadingRows" in app and "InlineError" in app and "EmptyState" in app
    assert "Checking…" in app
    assert "Stale · refresh failed" in app
    assert "Health check failed" in app
    assert '"Unavailable"' in app
    assert "database.loading" in app
    assert "database.error" in app
