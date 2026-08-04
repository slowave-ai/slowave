"""Tests for WP-8 (Lifecycle rollout & telemetry, see
private/docs/iterations/20260728_retrieval_quality_execution_progress.md):

  - slowave.core.engine.SlowaveEngine.session_start() stamps the current
    lifecycle-instructions contract version (slowave.lifecycle.LIFECYCLE_VERSION)
    onto every new session, unless overridden.
  - slowave.cli.main._lifecycle_version_health() groups activate/recall/
    feedback counts by that stamped version.
  - slowave.cli.clients.summarize_client_status() flags a client whose
    installed lifecycle block is a stale (pre-current) version.
  - slowave.cli.clients.get_client_statuses() actually extracts the
    installed version, not just presence/absence.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from slowave.cli.clients import ClientStatus, get_client_statuses, summarize_client_status
from slowave.cli.main import _lifecycle_version_health
from slowave.cli.output import Status
from slowave.core.config import SlowaveConfig
from slowave.core.engine import SlowaveEngine
from slowave.lifecycle import LIFECYCLE_VERSION
from slowave.ops import activate
from slowave.ops import recall as ops_recall
from slowave.ops import reinforce


def _fake_encoder_engine(db_path: str, dim: int = 16) -> SlowaveEngine:
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


def _cleanup(path: str) -> None:
    for ext in ("", "-wal", "-shm"):
        p = path + ext
        if os.path.exists(p):
            os.remove(p)


# ===========================================================================
# session_start() lifecycle_version stamping
# ===========================================================================


class TestSessionStartLifecycleVersionStamping:
    def test_defaults_to_current_lifecycle_version(self, tmp_path):
        db = str(tmp_path / "test.db")
        eng = _fake_encoder_engine(db)
        try:
            sid = eng.session_start(agent="mcp", scope="project:x")
            row = (
                eng.db.connect()
                .execute("SELECT lifecycle_version FROM sessions WHERE id = ?", (sid,))
                .fetchone()
            )
            assert row["lifecycle_version"] == LIFECYCLE_VERSION
        finally:
            eng.close()
            _cleanup(db)

    def test_explicit_override_is_honored(self, tmp_path):
        db = str(tmp_path / "test.db")
        eng = _fake_encoder_engine(db)
        try:
            sid = eng.session_start(agent="mcp", scope="project:x", lifecycle_version="v1")
            row = (
                eng.db.connect()
                .execute("SELECT lifecycle_version FROM sessions WHERE id = ?", (sid,))
                .fetchone()
            )
            assert row["lifecycle_version"] == "v1"
        finally:
            eng.close()
            _cleanup(db)

    def test_empty_string_override_clears_to_null(self, tmp_path):
        db = str(tmp_path / "test.db")
        eng = _fake_encoder_engine(db)
        try:
            sid = eng.session_start(agent="mcp", scope="project:x", lifecycle_version="")
            row = (
                eng.db.connect()
                .execute("SELECT lifecycle_version FROM sessions WHERE id = ?", (sid,))
                .fetchone()
            )
            assert row["lifecycle_version"] is None
        finally:
            eng.close()
            _cleanup(db)


# ===========================================================================
# _lifecycle_version_health()
# ===========================================================================


class TestLifecycleVersionHealth:
    def test_no_db_reports_unavailable(self, tmp_path):
        result = _lifecycle_version_health(str(tmp_path / "missing.db"))
        assert result["available"] is False
        assert result["warnings"]

    def test_groups_activate_recall_feedback_by_version(self, tmp_path):
        db = str(tmp_path / "test.db")
        eng = _fake_encoder_engine(db)
        try:
            scope = "project:alpha"

            # ops.activate()/ops.recall()/ops.reinforce() are stamped with the
            # current contract version automatically (no caller override
            # needed) -- this is the production path.
            r1 = activate(eng, query="how does auth work", scope=scope)
            ops_recall(eng, query="auth token expiry", scope=scope)
            reinforce(eng, retrieval_id=r1["retrieval_id"], feedback="useful", outcome="success")

            # A row explicitly stamped as a stale (legacy) version, simulating
            # a retrieval call recorded before the current contract shipped --
            # proves buckets don't mix. recall() has no session concept to
            # thread a version through (ops.recall() never sets session_id),
            # so per-call stamping (not a session join) must be what's tested.
            eng.record_context_recall(
                context_id="ctx_legacy_test",
                scope_id=scope,
                query="unrelated legacy query",
                response={"memory_ids": [], "schemas": []},
                lifecycle_version="v1",
            )

            eng.db.connect().commit()
            result = _lifecycle_version_health(db)

            assert result["available"] is True
            assert result["current_version"] == LIFECYCLE_VERSION
            by_version = result["by_version"]

            current = by_version[LIFECYCLE_VERSION]
            assert current["activate_calls"] == 1
            assert current["recall_calls"] == 1
            assert current["feedback_calls"] == 1
            assert current["recall_activate_ratio"] == pytest.approx(0.5)

            legacy = by_version["v1"]
            assert legacy["activate_calls"] == 1
            assert legacy["recall_calls"] == 0
            assert legacy["feedback_calls"] == 0
            assert legacy["recall_activate_ratio"] == 0.0
        finally:
            eng.close()
            _cleanup(db)

    def test_row_with_no_lifecycle_version_buckets_as_unknown(self, tmp_path):
        db = str(tmp_path / "test.db")
        eng = _fake_encoder_engine(db)
        try:
            eng.record_context_recall(
                context_id="ctx_unknown_test",
                scope_id="project:beta",
                query="some query",
                response={"memory_ids": [], "schemas": []},
                lifecycle_version="",
            )
            eng.db.connect().commit()

            result = _lifecycle_version_health(db)
            assert "unknown" in result["by_version"]
            assert result["by_version"]["unknown"]["activate_calls"] == 1
        finally:
            eng.close()
            _cleanup(db)


# ===========================================================================
# clients.py: stale-version detection
# ===========================================================================


class TestSummarizeClientStatusVersionDrift:
    def test_warns_when_installed_version_is_stale(self):
        client = ClientStatus(
            name="Claude Code",
            mcp_configured=True,
            lifecycle_enabled=True,
            lifecycle_version="v2",
        )
        status, detail = summarize_client_status(client)
        assert status == Status.WARN
        assert "stale" in detail.lower()
        assert "v2" in detail
        assert LIFECYCLE_VERSION in detail

    def test_ok_when_installed_version_matches_current(self):
        client = ClientStatus(
            name="Claude Code",
            mcp_configured=True,
            lifecycle_enabled=True,
            lifecycle_version=LIFECYCLE_VERSION,
        )
        status, detail = summarize_client_status(client)
        assert status == Status.OK
        assert "lifecycle" in detail

    def test_no_false_positive_when_version_unknown(self):
        """A pre-marker legacy install (or a detection miss) has
        lifecycle_version=None -- must not be treated as stale."""
        client = ClientStatus(
            name="Claude Code",
            mcp_configured=True,
            lifecycle_enabled=True,
            lifecycle_version=None,
        )
        status, detail = summarize_client_status(client)
        assert status == Status.OK


class TestGetClientStatusesDetectsInstalledVersion:
    @pytest.fixture()
    def fake_home(self, tmp_path, monkeypatch):
        import slowave.cli.setup as _setup_mod

        monkeypatch.setattr(_setup_mod, "_home", lambda: tmp_path)
        return tmp_path

    def test_claude_code_reports_current_version(self, fake_home):
        from slowave.cli.setup import _lifecycle_block

        claude_dir = fake_home / ".claude"
        claude_dir.mkdir(parents=True)
        (claude_dir / "CLAUDE.md").write_text(_lifecycle_block("claude-code"), encoding="utf-8")

        statuses = get_client_statuses()
        assert statuses["claude_code"].lifecycle_version == LIFECYCLE_VERSION

    def test_claude_code_reports_stale_version(self, fake_home):
        import json

        claude_dir = fake_home / ".claude"
        claude_dir.mkdir(parents=True)
        stale = "<!-- slowave-lifecycle-start v2 -->\nold\n<!-- slowave-lifecycle-end v2 -->\n"
        (claude_dir / "CLAUDE.md").write_text(stale, encoding="utf-8")
        # mcp_configured must be True for summarize_client_status to reach the
        # lifecycle-version check instead of short-circuiting to SKIP.
        (fake_home / ".claude.json").write_text(
            json.dumps({"mcpServers": {"slowave": {"url": "http://127.0.0.1:8766/mcp"}}}),
            encoding="utf-8",
        )

        statuses = get_client_statuses()
        assert statuses["claude_code"].lifecycle_version == "v2"
        status, detail = summarize_client_status(statuses["claude_code"])
        assert status == Status.WARN
