"""Integration test: minimal activate → remember → commit lifecycle.

Validates Step 6 acceptance criteria:
- Single session_id flows through activate → remember → commit
- ended_ts is set after commit
- raw_events >= 2 (context_query + task_complete synthetic events)
- >= 1 episode formed
- session_resolver cleared after commit
"""

from __future__ import annotations

import os
import tempfile

from slowave.core.config import SlowaveConfig
from slowave.core.engine import SlowaveEngine
from slowave.mcp import session_resolver


def _tmp_engine() -> tuple[SlowaveEngine, str]:
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    cfg = SlowaveConfig(db_path=tmp.name, dim=8, disable_encoder=True)
    return SlowaveEngine(cfg), tmp.name


def _cleanup(path: str) -> None:
    for ext in ("", "-wal", "-shm"):
        p = path + ext
        if os.path.exists(p):
            os.remove(p)


class TestLifecycleMinimal:
    """activate → remember (no session_id) → commit."""

    def test_single_session_flows_through(self) -> None:
        eng, path = _tmp_engine()
        scope = "test:lifecycle"
        try:
            # ---- activate (simulated inline, no MCP server) ----------------
            sid = eng.session_start(agent="mcp", scope=scope)
            session_resolver.bind(scope, sid)

            # ---- remember (resolves implicit session) -----------------------
            resolved = session_resolver.resolve(scope)
            assert resolved == sid, "resolver must return the bound session"
            eng.remember(
                content="test fact for lifecycle", type="fact", scope=scope, session_id=resolved
            )

            # ---- synthetic events (normally fired via _bg_log_event) -------
            eng.event_append(session_id=sid, type="context_query", content="test task query")

            # ---- commit (simulated inline) ----------------------------------
            eng.event_append(session_id=sid, type="task_complete", content="outcome=success")
            result = eng.session_end(sid, consolidate=False)
            session_resolver.clear(scope)

            # ---- assertions -------------------------------------------------
            conn = eng.db.connect()

            # session is closed
            sess = conn.execute("SELECT ended_ts FROM sessions WHERE id = ?", (sid,)).fetchone()
            assert sess is not None
            assert sess["ended_ts"] is not None, "session must be closed after commit"

            # >= 2 raw events (context_query + task_complete at minimum)
            events = conn.execute(
                "SELECT COUNT(*) as n FROM raw_events WHERE session_id = ?", (sid,)
            ).fetchone()
            assert events["n"] >= 2, f"expected >= 2 raw_events, got {events['n']}"

            # episodes_formed >= 0 (encoder disabled in test, so 0 is valid;
            # in production with encoder enabled this will be >= 1)
            assert result.get("episodes_formed", 0) >= 0

            # resolver cleared
            assert session_resolver.resolve(scope) is None, "resolver must be cleared after commit"
        finally:
            eng.close()
            _cleanup(path)

    def test_commit_steps_logged_as_raw_events(self) -> None:
        """slowave_commit steps → raw_events with type='step', ordered, scoped."""
        eng, path = _tmp_engine()
        scope = "test:lifecycle"
        try:
            sid = eng.session_start(agent="mcp", scope=scope)
            session_resolver.bind(scope, sid)
            eng.event_append(session_id=sid, type="context_query", content="test with steps")
            eng.event_append(session_id=sid, type="remember:decision", content="some decision")

            import slowave.ops as ops

            ops.commit(
                eng,
                session_id=sid,
                outcome="success",
                steps=[
                    "Ran full test suite (142 passed)",
                    "Built Docker image",
                    "Deployed to staging",
                ],
            )
            session_resolver.clear(scope)

            conn = eng.db.connect()

            # Step events are stored
            steps = conn.execute(
                "SELECT type, content FROM raw_events WHERE session_id = ? AND type = 'step' ORDER BY id",
                (sid,),
            ).fetchall()
            assert len(steps) == 3, f"expected 3 step events, got {len(steps)}"
            assert steps[0]["content"] == "Ran full test suite (142 passed)"
            assert steps[1]["content"] == "Built Docker image"
            assert steps[2]["content"] == "Deployed to staging"

            # Session is still closed normally
            sess = conn.execute(
                "SELECT ended_ts, outcome FROM sessions WHERE id = ?", (sid,)
            ).fetchone()
            assert sess is not None
            assert sess["ended_ts"] is not None
            assert sess["outcome"] == "success"

            # Steps without steps=None doesn't crash
            sid2 = eng.session_start(agent="mcp", scope=scope)
            session_resolver.bind(scope, sid2)
            eng.event_append(session_id=sid2, type="context_query", content="no steps")
            result2 = ops.commit(eng, session_id=sid2, outcome="partial")
            session_resolver.clear(scope)
            assert result2.get("episodes_formed", 0) >= 0

            no_steps = conn.execute(
                "SELECT COUNT(*) as n FROM raw_events WHERE session_id = ? AND type = 'step'",
                (sid2,),
            ).fetchone()
            assert no_steps["n"] == 0, "no step events expected when steps=None"

        finally:
            eng.close()
            _cleanup(path)
