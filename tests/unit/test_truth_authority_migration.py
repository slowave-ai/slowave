from pathlib import Path

import numpy as np

from slowave.core.config import SlowaveConfig
from slowave.core.engine import SlowaveEngine
from slowave.storage.sqlite_db import SQLiteConfig, SQLiteDB


def test_legacy_geometric_truth_columns_are_removed(tmp_path: Path) -> None:
    current_schema = Path("slowave/storage/schema.sql").resolve()
    legacy_sql = (
        current_schema.read_text(encoding="utf-8")
        .replace(
            "  supporting_episode_ids    TEXT NOT NULL DEFAULT '[]',    -- JSON array\n",
            "  supporting_episode_ids    TEXT NOT NULL DEFAULT '[]',    -- JSON array\n"
            "  contradicting_episode_ids TEXT NOT NULL DEFAULT '[]',\n",
        )
        .replace(
            "  schemas_reinforced    INTEGER NOT NULL DEFAULT 0,\n",
            "  schemas_reinforced    INTEGER NOT NULL DEFAULT 0,\n"
            "  schemas_contradicted  INTEGER NOT NULL DEFAULT 0,\n",
        )
    )
    legacy_schema = tmp_path / "legacy.sql"
    legacy_schema.write_text(legacy_sql, encoding="utf-8")
    db_path = tmp_path / "legacy.db"
    db = SQLiteDB(SQLiteConfig(path=str(db_path)))
    try:
        db.init_schema(str(legacy_schema))
        db.init_schema(str(current_schema))
        conn = db.connect()
        schema_columns = {
            row["name"] for row in conn.execute("PRAGMA table_info(schemas)").fetchall()
        }
        worker_columns = {
            row["name"] for row in conn.execute("PRAGMA table_info(worker_runs)").fetchall()
        }
        assert "contradicting_episode_ids" not in schema_columns
        assert "schemas_contradicted" not in worker_columns
    finally:
        db.close()


def test_legacy_terminal_statuses_migrate_to_stale_with_reasons(tmp_path: Path) -> None:
    path = tmp_path / "statuses.db"
    engine = SlowaveEngine(SlowaveConfig(db_path=str(path), dim=8, disable_encoder=True))
    vector = np.ones(8, dtype=np.float32)
    first = engine.schemas.create(
        content_text="legacy superseded", facets={}, tags=[], embedding=vector
    )
    second = engine.schemas.create(
        content_text="legacy contradicted", facets={}, tags=[], embedding=vector
    )
    conn = engine.db.connect()
    conn.execute("UPDATE schemas SET status='superseded' WHERE id=?", (first,))
    conn.execute("UPDATE schemas SET status='contradicted' WHERE id=?", (second,))
    conn.commit()
    engine.close()

    migrated = SlowaveEngine(SlowaveConfig(db_path=str(path), dim=8, disable_encoder=True))
    try:
        assert migrated.schemas.get(first).status == "stale"
        assert migrated.schemas.get(first).stale_reason == "superseded"
        assert migrated.schemas.get(second).status == "stale"
        assert migrated.schemas.get(second).stale_reason == "contradicted"
    finally:
        migrated.close()
