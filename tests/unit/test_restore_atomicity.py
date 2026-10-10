"""Restore must preserve current data and refuse active supervised writers."""

from __future__ import annotations

import gzip
import sqlite3
from pathlib import Path

from click.testing import CliRunner

from slowave.cli.main import cli


def _make_sqlite_db(path: Path, marker: str) -> None:
    conn = sqlite3.connect(str(path))
    conn.execute("CREATE TABLE marker (value TEXT)")
    conn.execute("INSERT INTO marker VALUES (?)", (marker,))
    conn.commit()
    conn.close()


def _marker_value(path: Path) -> str:
    conn = sqlite3.connect(str(path))
    row = conn.execute("SELECT value FROM marker").fetchone()
    conn.close()
    return row[0]


def _make_backup_gz(tmp_path: Path, marker: str) -> Path:
    raw = tmp_path / "raw_backup.db"
    _make_sqlite_db(raw, marker)
    gz_path = tmp_path / "slowave-20260101_000000.db.gz"
    with open(raw, "rb") as f_in, gzip.open(gz_path, "wb") as f_out:
        f_out.write(f_in.read())
    return gz_path


def test_restore_swaps_file_atomically_and_leaves_no_temp_file(tmp_path, monkeypatch):
    monkeypatch.setenv("SLOWAVE_DAEMON_PID", str(tmp_path / "no_daemon.pid"))
    monkeypatch.setattr("slowave.cli.main._slowave_processes", lambda: [])

    db_path = tmp_path / "slowave.db"
    _make_sqlite_db(db_path, "old-content")
    backup_gz = _make_backup_gz(tmp_path, "new-content")

    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["--db", str(db_path), "restore", str(backup_gz), "--yes", "--json"],
    )

    assert result.exit_code == 0, result.output
    assert _marker_value(db_path) == "new-content"
    assert _marker_value(tmp_path / "slowave.db.bak") == "old-content"
    leftover_tmp = list(tmp_path.glob(".slowave-restore-*"))
    assert leftover_tmp == []


def test_restore_refuses_detected_worker_without_signalling_it(tmp_path, monkeypatch):
    monkeypatch.setenv("SLOWAVE_DAEMON_PID", str(tmp_path / "no_daemon.pid"))

    fake_pid = 999_999_999  # implausible real PID; only ever touched via the mock below
    monkeypatch.setattr(
        "slowave.cli.main._slowave_processes",
        lambda: [{"pid": fake_pid, "command": "python -m slowave worker --interval 600"}],
    )

    killed: list[int] = []
    monkeypatch.setattr("os.kill", lambda pid, sig: killed.append(pid))

    db_path = tmp_path / "slowave.db"
    _make_sqlite_db(db_path, "old-content")
    backup_gz = _make_backup_gz(tmp_path, "new-content")

    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["--db", str(db_path), "restore", str(backup_gz), "--yes", "--json"],
    )

    assert result.exit_code != 0, result.output
    assert killed == []
    assert "slowave stop" in result.output
    assert _marker_value(db_path) == "old-content"


def test_invalid_restore_preserves_committed_wal_data(tmp_path, monkeypatch):
    monkeypatch.setenv("SLOWAVE_DAEMON_PID", str(tmp_path / "no_daemon.pid"))
    monkeypatch.setattr("slowave.cli.main._slowave_processes", lambda: [])
    db_path = tmp_path / "slowave.db"
    _make_sqlite_db(db_path, "old-content")
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("UPDATE marker SET value='latest-wal-content'")
        conn.commit()
        backup = tmp_path / "invalid.db.gz"
        with gzip.open(backup, "wb") as stream:
            stream.write(b"not a database")
        result = CliRunner().invoke(cli, ["--db", str(db_path), "restore", str(backup), "--yes"])
        assert result.exit_code != 0
        assert _marker_value(db_path) == "latest-wal-content"
        assert not list(tmp_path.glob(".slowave-restore-*"))
    finally:
        conn.close()


def test_restore_rollback_snapshot_includes_committed_wal(tmp_path, monkeypatch):
    monkeypatch.setenv("SLOWAVE_DAEMON_PID", str(tmp_path / "no_daemon.pid"))
    monkeypatch.setattr("slowave.cli.main._slowave_processes", lambda: [])
    db_path = tmp_path / "slowave.db"
    _make_sqlite_db(db_path, "old-content")
    backup = _make_backup_gz(tmp_path, "replacement")
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("UPDATE marker SET value='latest-wal-content'")
        conn.commit()
        result = CliRunner().invoke(cli, ["--db", str(db_path), "restore", str(backup), "--yes"])
        assert result.exit_code == 0, result.output
        assert _marker_value(db_path) == "replacement"
        assert _marker_value(Path(str(db_path) + ".bak")) == "latest-wal-content"
    finally:
        conn.close()
