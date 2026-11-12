from __future__ import annotations

import gzip
import shutil
import sqlite3

import click
import pytest

from slowave.cli import setup as cli_setup
from slowave.cli.setup_output import SetupView
from slowave.storage import embedding_migration as migration
from tests.unit.test_embedding_migration import FakeEncoder, _cleanup, _database


@pytest.fixture(autouse=True)
def no_registered_services(monkeypatch, tmp_path):
    """The OS service supervisor is the boundary: nothing is registered in these tests."""
    monkeypatch.setattr("slowave.cli.services.stop_registered_services", lambda: [])
    monkeypatch.setattr("slowave.cli.services.require_stopped_runtime", lambda operation: None)
    monkeypatch.setenv("SLOWAVE_BACKUP_DIR", str(tmp_path / "backups"))


def _vector_width(path: str) -> int:
    conn = sqlite3.connect(path)
    try:
        return int(conn.execute("SELECT dim FROM schemas LIMIT 1").fetchone()[0])
    finally:
        conn.close()


def _run(path: str, *, dry_run: bool = False) -> None:
    cli_setup._migrate_legacy_embeddings(
        SetupView(), path, dry_run=dry_run, restart_after=False, encoder=FakeEncoder()
    )


def test_setup_backs_up_then_migrates_a_database_from_another_model(tmp_path) -> None:
    db, path = _database()
    try:
        _run(path)

        assert _vector_width(path) == 6
        [backup] = list((tmp_path / "backups").glob("slowave-pre-embedding-migration-*.db.gz"))
        restored = tmp_path / "restored.db"
        with gzip.open(backup, "rb") as source, open(restored, "wb") as target:
            shutil.copyfileobj(source, target)
        assert _vector_width(str(restored)) == 4
    finally:
        _cleanup(path)


def test_setup_dry_run_reports_the_migration_and_writes_nothing(tmp_path, capsys) -> None:
    db, path = _database()
    try:
        _run(path, dry_run=True)

        assert _vector_width(path) == 4
        assert not (tmp_path / "backups").exists()
        assert "would be migrated" in capsys.readouterr().out
    finally:
        _cleanup(path)


def test_setup_leaves_a_database_in_the_current_space_alone(tmp_path) -> None:
    db, path = _database()
    try:
        encoder = FakeEncoder()
        migration.migrate_embeddings(db, formation_encoder=encoder, search_encoder=encoder)

        _run(path)

        assert not (tmp_path / "backups").exists()
    finally:
        _cleanup(path)


def test_a_failed_migration_stops_setup_and_names_the_backup(monkeypatch, tmp_path) -> None:
    db, path = _database()
    try:

        def explode(conn, encoder, batch):
            raise RuntimeError("encoder crashed")

        monkeypatch.setattr(migration, "_migrate_schemas", explode)

        with pytest.raises(click.ClickException) as failure:
            _run(path)

        [backup] = list((tmp_path / "backups").glob("slowave-pre-embedding-migration-*.db.gz"))
        assert str(backup) in failure.value.message
        assert "slowave restore" in failure.value.message
    finally:
        _cleanup(path)
