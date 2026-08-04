"""Tests for the two-path behaviour of engine.remember().

Verifies:
- Standalone (no session_id): ad-hoc session created, ended, episodes formed
  immediately.
- Live session (session_id passed): event appended to caller's session, session
  NOT ended, no episodes formed yet.  Episodes are formed exactly once when the
  caller's session_end runs.
- Calling remember() twice with the same content inside a live session produces
  no duplicate episodes.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from slowave.core.config import SlowaveConfig
from slowave.core.engine import SlowaveEngine

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fake_encoder_engine(db_path: str, dim: int = 32) -> SlowaveEngine:
    """Build an engine with a deterministic stub encoder (no model download)."""
    cfg = SlowaveConfig(db_path=db_path, dim=dim, disable_encoder=True)
    eng = SlowaveEngine(cfg)

    class _StubEncoder:
        def encode(self, text: str) -> np.ndarray:
            seed = int(abs(hash(text)) % (2**31))
            r = np.random.default_rng(seed)
            v = r.standard_normal(dim).astype(np.float32)
            return v / (np.linalg.norm(v) + 1e-12)

    eng.encoder = _StubEncoder()
    return eng


@pytest.fixture()
def eng(tmp_path):
    db = str(tmp_path / "test.db")
    engine = _fake_encoder_engine(db)
    yield engine
    engine.close()
    for ext in ("", "-wal", "-shm"):
        p = db + ext
        if os.path.exists(p):
            os.remove(p)


def _episode_count(eng: SlowaveEngine) -> int:
    conn = eng.db.connect()
    return int(conn.execute("SELECT COUNT(*) FROM episodic_memories").fetchone()[0])


def _session_ended(eng: SlowaveEngine, session_id: str) -> bool:
    conn = eng.db.connect()
    row = conn.execute("SELECT ended_ts FROM sessions WHERE id = ?", (session_id,)).fetchone()
    return row is not None and row["ended_ts"] is not None


def _episodes_for_session(eng: SlowaveEngine, session_id: str) -> int:
    conn = eng.db.connect()
    return int(
        conn.execute(
            "SELECT COUNT(*) FROM episode_text WHERE session_id = ?", (session_id,)
        ).fetchone()[0]
    )


# ---------------------------------------------------------------------------
# Standalone path (no session_id)
# ---------------------------------------------------------------------------


class TestRememberStandalone:
    def test_episodes_formed_immediately(self, eng):
        assert _episode_count(eng) == 0
        eng.remember(content="I prefer Python over Ruby", type="preference")
        assert _episode_count(eng) > 0

    def test_schema_created(self, eng):
        eng.remember(content="Always write tests first", type="fact")
        assert eng.schemas.count() == 1

    def test_live_session_not_contaminated(self, eng):
        """A pre-existing live session must not be ended by a standalone remember."""
        sid = eng.session_start(agent="test")
        eng.remember(content="standalone fact")
        assert not _session_ended(eng, sid)
        assert _episodes_for_session(eng, sid) == 0


# ---------------------------------------------------------------------------
# Live-session path (session_id passed)
# ---------------------------------------------------------------------------


class TestRememberLiveSession:
    def test_session_not_ended(self, eng):
        sid = eng.session_start(agent="test")
        eng.remember(content="I use neovim", type="preference", session_id=sid)
        assert not _session_ended(eng, sid)

    def test_no_episodes_before_session_end(self, eng):
        sid = eng.session_start(agent="test")
        eng.remember(content="I use neovim", type="preference", session_id=sid)
        assert _episode_count(eng) == 0

    def test_schema_created_immediately(self, eng):
        sid = eng.session_start(agent="test")
        eng.remember(content="I use neovim", type="preference", session_id=sid)
        assert eng.schemas.count() == 1

    def test_episodes_formed_once_on_session_end(self, eng):
        sid = eng.session_start(agent="test")
        eng.remember(content="I use neovim", type="preference", session_id=sid)
        assert _episode_count(eng) == 0
        eng.session_end(sid)
        assert _episode_count(eng) > 0

    def test_no_duplicate_episodes_two_remembers(self, eng):
        """Two remember() calls in one session → session_end forms episodes once, not twice."""
        sid = eng.session_start(agent="test")
        eng.remember(content="I prefer dark mode", type="preference", session_id=sid)
        eng.remember(content="I prefer dark mode", type="preference", session_id=sid)
        assert _episode_count(eng) == 0
        eng.session_end(sid)
        count_two_events = _episode_count(eng)

        # Baseline: fresh session with a single remember.
        sid2 = eng.session_start(agent="test")
        eng.remember(content="I prefer light mode", type="preference", session_id=sid2)
        eng.session_end(sid2)
        count_one_event = _episode_count(eng) - count_two_events

        assert count_two_events > 0
        assert count_one_event > 0
        # All episodes for sid came from exactly one session_end call.
        assert _episodes_for_session(eng, sid) == count_two_events
        assert _episodes_for_session(eng, sid2) == count_one_event

    def test_second_session_end_call_does_not_duplicate_episodes(self, eng):
        """Regression (2026-07-24 Tier-0 audit): the idle-session reaper calls
        session_end(consolidate=False) to close an idle session; if the client
        later calls slowave_commit on that same session_id (normal for any
        real task that runs longer than the idle timeout), session_end used
        to re-run form_episodes() from scratch and insert duplicate episode
        rows for every raw event already processed the first time."""
        sid = eng.session_start(agent="test")
        eng.remember(content="I prefer tabs over spaces", type="preference", session_id=sid)

        first = eng.session_end(sid)
        assert first["episodes_formed"] > 0
        assert "already_ended" not in first
        count_after_first = _episode_count(eng)

        second = eng.session_end(sid, outcome="success")
        assert second["episodes_formed"] == 0
        assert second["already_ended"] is True
        assert _episode_count(eng) == count_after_first
        assert _episodes_for_session(eng, sid) == count_after_first
