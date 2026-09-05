from __future__ import annotations

import sqlite3
from pathlib import Path

import click
import pytest

from slowave.cli import migrate_data
from slowave.cli.migrate_data import MigrationPlan, execute_migration, plan_migration


def _legacy_fixture(home: Path, destination: Path) -> MigrationPlan:
    source = home / ".slowave"
    source.mkdir(parents=True)
    conn = sqlite3.connect(source / "slowave.db")
    try:
        conn.execute("CREATE TABLE memories (value TEXT)")
        conn.execute("INSERT INTO memories VALUES ('preserved')")
        conn.commit()
    finally:
        conn.close()
    (source / "logs").mkdir()
    (source / "logs" / "mcp.log").write_text("log")
    (source / "backups").mkdir()
    (source / "backups" / "old.db.gz").write_bytes(b"backup")
    (source / ".setup_done").touch()
    return plan_migration(home=home, destination=destination)


def test_successful_migration_validates_and_preserves_source(tmp_path):
    home = tmp_path / "home with spaces"
    destination = tmp_path / "native" / "slowave data"
    plan = _legacy_fixture(home, destination)

    result = execute_migration(plan)

    assert result["migrated"] is True
    assert plan.source_database.exists()
    conn = sqlite3.connect(plan.destination_database)
    try:
        assert conn.execute("SELECT value FROM memories").fetchone() == ("preserved",)
        assert conn.execute("PRAGMA integrity_check").fetchone() == ("ok",)
    finally:
        conn.close()
    assert (destination / "logs" / "mcp.log").read_text() == "log"
    assert (destination / "backups" / "old.db.gz").read_bytes() == b"backup"
    assert (destination / ".setup_done").exists()


def test_nonempty_destination_is_refused(tmp_path):
    plan = _legacy_fixture(tmp_path / "home", tmp_path / "native" / "slowave")
    plan.destination_root.mkdir(parents=True)
    (plan.destination_root / "existing").write_text("keep")

    with pytest.raises(click.ClickException, match="not empty"):
        execute_migration(plan)
    assert (plan.destination_root / "existing").read_text() == "keep"


def test_integrity_failure_cleans_staging_and_leaves_source(tmp_path):
    destination = tmp_path / "native" / "slowave"
    plan = _legacy_fixture(tmp_path / "home", destination)

    def fail_validation(path: Path) -> None:
        raise click.ClickException("bad integrity")

    with pytest.raises(click.ClickException, match="bad integrity"):
        execute_migration(plan, validate=fail_validation)

    assert plan.source_database.exists()
    assert not destination.exists()
    assert list(destination.parent.glob(".slowave-migrate-*")) == []


def test_live_daemon_refusal_prevents_copy(monkeypatch, tmp_path):
    plan = _legacy_fixture(tmp_path / "home", tmp_path / "native" / "slowave")

    def refuse(pid_file: Path, timeout_s: float = 5.0) -> bool:
        raise click.ClickException("still running")

    monkeypatch.setattr(migrate_data, "_stop_legacy_daemon", refuse)
    with pytest.raises(click.ClickException, match="still running"):
        execute_migration(plan)
    assert not plan.destination_root.exists()
