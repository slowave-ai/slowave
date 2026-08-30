from __future__ import annotations

import sqlite3

from slowave.cli.main import _feedback_health, _session_lifecycle_health


def test_v9_health_counts_sessions_and_feedback(tmp_path) -> None:
    path = tmp_path / "health.db"
    conn = sqlite3.connect(path)
    conn.executescript("""
        CREATE TABLE sessions (id TEXT PRIMARY KEY, started_ts INTEGER, ended_ts INTEGER);
        INSERT INTO sessions VALUES ('open', 1, NULL), ('closed', 1, 2);
        CREATE TABLE context_feedback_events (id INTEGER PRIMARY KEY, created_at INTEGER);
        INSERT INTO context_feedback_events VALUES (1, 10);
        CREATE TABLE feedback_events (event_id TEXT PRIMARY KEY, created_at INTEGER);
        INSERT INTO feedback_events VALUES ('fbe_1', 20);
        """)
    conn.commit()
    conn.close()

    sessions = _session_lifecycle_health(str(path))
    assert sessions["sessions_started"] == 2
    assert sessions["sessions_committed"] == 1

    feedback = _feedback_health(str(path))
    assert feedback["feedback_or_reinforcement_calls"] == 2
    assert feedback["last_feedback_ts"] == 20
