"""Test: session-idle reaper (60-min default).

Tests verify:
1. An old open session with no events is closed by _reap_once(timeout_s=0)
2. A session with a fresh event is NOT closed by _reap_once(timeout_s=3600)
"""

import tempfile
import time

from slowave.core.config import SlowaveConfig
from slowave.core.engine import SlowaveEngine
from slowave.mcp.session_reaper import _reap_once, get_idle_timeout_s


def _build_test_engine(db_path: str) -> SlowaveEngine:
    """Create an engine with a temp DB."""
    cfg = SlowaveConfig(db_path=db_path, disable_encoder=True)
    return SlowaveEngine(cfg=cfg)


def test_reap_old_open_session():
    """An old open session with no events is closed by _reap_once(timeout_s=0)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = f"{tmpdir}/test.db"

        def build():
            return _build_test_engine(db_path)

        # Create an engine and open an old session manually
        eng = build()
        now_ts = int(time.time())

        # Insert a session with started_ts = 1 hour ago, no events
        conn = eng.db.connect()
        old_ts = now_ts - 3700  # 1+ hours ago
        conn.execute(
            "INSERT INTO sessions (id, agent, scope_id, scope_kind, started_ts, ended_ts) VALUES (?, ?, ?, ?, ?, NULL)",
            ("sess_old", "test", "project:test", "project", old_ts),
        )
        conn.commit()
        conn.close()

        # Run reaper with timeout_s=0 (reap everything old)
        closed = _reap_once(build, timeout_s=0)

        # Verify session was closed
        assert len(closed) > 0, "Should have closed at least one session"

        # Verify in DB
        eng = build()
        conn = eng.db.connect()
        open_sessions = conn.execute(
            "SELECT COUNT(*) as cnt FROM sessions WHERE ended_ts IS NULL"
        ).fetchone()
        assert open_sessions["cnt"] == 0, "No sessions should be open after reaping with timeout=0"
        conn.close()


def test_preserve_fresh_session():
    """A session with a fresh event is NOT closed by _reap_once(timeout_s=3600)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = f"{tmpdir}/test.db"

        def build():
            return _build_test_engine(db_path)

        # Create an engine
        eng = build()
        now_ts = int(time.time())

        # Insert an open session and a fresh event
        conn = eng.db.connect()
        session_started = now_ts - 100  # 100 seconds ago
        conn.execute(
            "INSERT INTO sessions (id, agent, scope_id, scope_kind, started_ts, ended_ts) VALUES (?, ?, ?, ?, ?, NULL)",
            ("sess_fresh", "test", "project:test", "project", session_started),
        )
        conn.commit()

        # Get the session ID
        sid = conn.execute("SELECT MAX(id) as id FROM sessions").fetchone()["id"]

        # Insert a fresh event (just now)
        conn.execute(
            "INSERT INTO raw_events (session_id, ts, type, content) VALUES (?, ?, ?, ?)",
            (sid, now_ts, "task_complete", "{}"),
        )
        conn.commit()
        conn.close()

        # Run reaper with 1-hour timeout (should NOT close this session)
        closed = _reap_once(build, timeout_s=3600)

        # Verify session was NOT closed
        assert sid not in closed, f"Fresh session {sid} should not be reaped"

        # Verify in DB
        eng = build()
        conn = eng.db.connect()
        open_sessions = conn.execute(
            "SELECT COUNT(*) as cnt FROM sessions WHERE ended_ts IS NULL"
        ).fetchone()
        assert open_sessions["cnt"] >= 1, "Fresh session should still be open after reaping"
        conn.close()


def test_get_idle_timeout_s_default():
    """get_idle_timeout_s() returns 3600 by default."""
    import os

    # Make sure env var is not set
    old_val = os.environ.pop("SLOWAVE_SESSION_IDLE_TIMEOUT", None)
    try:
        result = get_idle_timeout_s()
        assert result == 3600, f"Expected default 3600, got {result}"
    finally:
        if old_val is not None:
            os.environ["SLOWAVE_SESSION_IDLE_TIMEOUT"] = old_val


def test_reap_twice_reuses_same_engine_no_closed_db():
    """Regression: reusing the SAME engine across reap passes must not crash.

    SQLiteDB.connect() caches one connection per thread (threading.local).
    The reaper used to call conn.close() in finally, which left the stale,
    closed handle in the thread-local cache, so the NEXT pass reacquired the
    closed connection and threw "Cannot operate on a closed database". The
    fix releases via eng.db.close(), which clears the thread-local cache.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = f"{tmpdir}/test.db"
        eng = _build_test_engine(db_path)

        def build():
            return eng  # SAME engine instance reused across passes

        # Insert an old open session so timeout_s=0 reaps it. Do NOT call
        # conn.close() here: SQLiteDB caches this per-thread connection and
        # would hand the reaper a closed handle. The reaper owns the release.
        conn = eng.db.connect()
        old_ts = int(time.time()) - 3700
        conn.execute(
            "INSERT INTO sessions (id, agent, scope_id, scope_kind, started_ts, ended_ts) "
            "VALUES (?, ?, ?, ?, ?, NULL)",
            ("sess_same_eng", "test", "project:test", "project", old_ts),
        )
        conn.commit()

        # First pass (would leave a stale closed conn in the cache under the old fix).
        first = _reap_once(build, timeout_s=0)
        assert "sess_same_eng" in first

        # Second pass against the SAME engine/thread must NOT raise
        # "Cannot operate on a closed database".
        closed = _reap_once(build, timeout_s=0)
        assert isinstance(closed, list)


if __name__ == "__main__":
    test_reap_old_open_session()
    test_preserve_fresh_session()
    test_get_idle_timeout_s_default()
    test_reap_twice_reuses_same_engine_no_closed_db()
    print("All tests passed!")
