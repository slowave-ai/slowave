"""Coverage for the Overview dashboard timeline and content."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import slowave.dashboard.app as dashboard_app


def test_pulse_uses_exact_window_even_when_first_bucket_starts_earlier(
    tmp_path: Path, monkeypatch
) -> None:
    db_path = tmp_path / "pulse.sqlite3"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE raw_events (ts INTEGER NOT NULL)")
    conn.execute("CREATE TABLE episodic_memories (ts INTEGER NOT NULL)")
    conn.execute("CREATE TABLE schemas (first_formed_ts INTEGER NOT NULL)")
    conn.executemany("INSERT INTO raw_events VALUES (?)", [(6350,), (6450,)])
    conn.commit()
    conn.close()
    monkeypatch.setattr(dashboard_app.time, "time", lambda: 10_000)

    payload = dashboard_app._pulse_payload(str(db_path), {"hours": ["1"], "bucket_m": ["5"]})

    assert payload["window_start"] == 6400
    assert payload["now_ts"] == 10_000
    assert sum(bucket["n"] for bucket in payload["channels"]["raw_events"]) == 1
    assert payload["channels"]["raw_events"][0] == {"ts": 6300, "n": 1}


def test_status_payload_reports_the_running_slowave_version(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        dashboard_app,
        "_daemon_health",
        lambda: {"running": False, "version": None, "active_sessions": 0, "engines_loaded": []},
    )
    payload = dashboard_app._status_payload(str(tmp_path / "missing.sqlite3"))
    assert payload["slowave_version"] == dashboard_app.__version__


def test_react_home_separates_service_observations_and_activity_lanes() -> None:
    source_dir = Path(__file__).parents[2] / "slowave/dashboard/ui/src"
    app = "\n".join(path.read_text() for path in source_dir.glob("*.tsx"))
    assert "/api/home" in app
    assert "MCP daemon" in app
    assert "Database" in app
    assert "Maintenance" in app
    assert "Activity captured" in app
    assert "Episodes" in app
    assert "Memories" in app
    assert "Since you last looked" not in app
