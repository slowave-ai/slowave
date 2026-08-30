"""Session-idle reaper: closes sessions older than SLOWAVE_SESSION_IDLE_TIMEOUT.

Real session-level reaper, distinct from the existing process watchdog.
Closes sessions whose last event is older than SLOWAVE_SESSION_IDLE_TIMEOUT
(default 3600 seconds / 1 hour) via session_end(consolidate=False).

Can be disabled with SLOWAVE_SESSION_IDLE_TIMEOUT=0.
"""

import logging
import os
import threading
import time
from typing import Callable, Optional

from slowave import ops

log = logging.getLogger(__name__)


def get_idle_timeout_s() -> int:
    """Get configured idle timeout from env or default (3600 = 1 hour)."""
    try:
        val = os.environ.get("SLOWAVE_SESSION_IDLE_TIMEOUT", "3600")
        return int(val)
    except (ValueError, TypeError):
        log.warning("Invalid SLOWAVE_SESSION_IDLE_TIMEOUT, using default 3600")
        return 3600


def _reap_once(build_engine: Callable, timeout_s: int) -> list[str]:
    """Reap idle sessions in a single pass.

    Args:
        build_engine: Callable that creates a SlowaveEngine instance.
        timeout_s: Idle timeout in seconds. If 0, only used for testing (timeout_s > 0 to actually reap).

    Returns:
        List of closed session IDs.

    Implementation notes:
    - Sessions with ended_ts IS NULL (still open)
    - AND last event is older than now - timeout_s
    - (LEFT JOIN raw_events, use COALESCE(MAX(ts), started_ts))
    """
    if timeout_s < 0:
        # Disabled
        return []

    eng = build_engine()
    conn = eng.db.connect()

    try:
        now_ts = int(time.time())
        cutoff_ts = now_ts - timeout_s if timeout_s > 0 else now_ts

        # Find idle open sessions
        rows = conn.execute(
            """
            SELECT sessions.id, MAX(raw_events.ts) as last_event_ts
            FROM sessions
            LEFT JOIN raw_events ON raw_events.session_id = sessions.id
            WHERE sessions.ended_ts IS NULL
            GROUP BY sessions.id
            HAVING COALESCE(MAX(raw_events.ts), sessions.started_ts) < ?
            """,
            (cutoff_ts,),
        ).fetchall()

        closed_ids = []
        for row in rows:
            sid = row["id"]
            try:
                ops.commit(
                    eng,
                    session_id=sid,
                    outcome="unknown",
                    verification={"status": "unverified", "summary": "Idle reaper closure"},
                    enforce_feedback=False,
                )
                closed_ids.append(sid)
                log.info(f"Reaped idle session {sid}")
            except Exception as e:
                log.warning(f"Failed to reap session {sid}: {e}")

        return closed_ids
    finally:
        # Do NOT close `conn` directly: SQLiteDB.connect() caches the
        # connection per thread (threading.local) and returns that cached
        # handle from connect(). Calling conn.close() here leaves the stale,
        # closed handle in the thread-local cache, so the NEXT reap pass
        # reuses a closed connection and throws "Cannot operate on a closed
        # database". Release via SQLiteDB.close(), which closes the current
        # thread's connection AND clears the thread-local cache.
        eng.db.close()


def start(build_engine: Callable, poll_interval_s: int = 120) -> Optional[threading.Thread]:
    """Start the session reaper as a background thread.

    Args:
        build_engine: Callable that creates a SlowaveEngine instance.
        poll_interval_s: How often to check for idle sessions (default 120 = 2 min).

    Returns:
        The reaper thread (daemon), or None if disabled (timeout_s == 0).

    Usage:
        reaper_thread = session_reaper.start(
            build_engine=lambda: SlowaveEngine(),
            poll_interval_s=120
        )
        # ... MCP server runs ...
        # Thread exits gracefully on process shutdown (daemon=True)
    """
    timeout_s = get_idle_timeout_s()
    if timeout_s <= 0:
        log.info("Session reaper disabled (SLOWAVE_SESSION_IDLE_TIMEOUT=0)")
        return None

    log.info(f"Starting session reaper (timeout={timeout_s}s, poll={poll_interval_s}s)")

    def _reaper_loop():
        while True:
            try:
                time.sleep(poll_interval_s)
                _reap_once(build_engine, timeout_s)
            except Exception as e:
                log.error(f"Session reaper error: {e}", exc_info=True)

    thread = threading.Thread(target=_reaper_loop, daemon=True, name="slowave-reaper")
    thread.start()
    return thread
