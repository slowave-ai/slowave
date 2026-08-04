"""Regression tests for the 2026-07-27 Procedures tab rewrite.

`_procedures_payload` used to cluster sessions by raw event-TYPE signature,
which private/docs/iterations/20260725_procedural_signal_commit_steps.md's
Phase 1 validation proved carries zero procedural signal. It now uses the
same embedding + alignment + average-linkage method as
scripts/analyze_procedural_signal.py (slowave/symbolic/procedural.py) --
there was no test coverage of this endpoint before this rewrite.
"""

from __future__ import annotations

import os
import tempfile

from slowave import ops
from slowave.core.config import SlowaveConfig
from slowave.core.engine import SlowaveEngine
from slowave.dashboard.app import _procedures_payload


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


def _commit_session(
    eng: SlowaveEngine, *, goal: str, steps: list[str], outcome: str, scope: str
) -> None:
    result = ops.activate(eng, query=goal, scope=scope, goal=goal, agent="test")
    ops.commit(eng, session_id=result["session_id"], outcome=outcome, steps=steps)


def test_insufficient_data_gate_when_no_step_sessions() -> None:
    eng, path = _tmp_engine()
    try:
        # A session with goal+outcome but no commit(steps=...) data.
        result = ops.activate(
            eng, query="do something", scope="project:x", goal="do something", agent="test"
        )
        ops.commit(eng, session_id=result["session_id"], outcome="success")

        payload = _procedures_payload(path, {})
        assert payload["gate"] == "insufficient_data"
        assert payload["clusters"] == []
        assert payload["step_sessions"] == 0
    finally:
        eng.close()
        _cleanup(path)


def test_similar_step_sessions_form_a_cluster() -> None:
    eng, path = _tmp_engine()
    try:
        scope = "project:x"
        deploy_steps = [
            (
                "deploy auth service to staging",
                [
                    "Ran full regression test suite",
                    "Built Docker image",
                    "Pushed image to registry",
                    "Applied rolling update",
                    "Verified health endpoint",
                ],
            ),
            (
                "deploy billing service to staging",
                [
                    "Ran full regression test suite",
                    "Built Docker image",
                    "Pushed image to registry",
                    "Applied rolling update",
                    "Verified health endpoint",
                ],
            ),
            (
                "deploy notifications service to staging",
                [
                    "Ran full regression test suite",
                    "Built Docker image",
                    "Pushed image to registry",
                    "Applied rolling update",
                    "Verified health endpoint",
                ],
            ),
        ]
        for goal, steps in deploy_steps:
            _commit_session(eng, goal=goal, steps=steps, outcome="success", scope=scope)

        payload = _procedures_payload(path, {"min_sessions": ["2"]})

        assert payload["gate"] != "insufficient_data"
        assert payload["step_sessions"] == 3
        assert payload["total_clusters_found"] >= 1
        cluster = payload["clusters"][0]
        assert cluster["session_count"] == 3
        assert cluster["success_rate"] == 1.0
        assert cluster["total_session_ids"] == 3
        assert len(cluster["example_steps"]) > 0
    finally:
        eng.close()
        _cleanup(path)


def test_unrelated_sessions_do_not_cluster_together() -> None:
    eng, path = _tmp_engine()
    try:
        scope = "project:x"
        _commit_session(
            eng,
            goal="deploy service to staging",
            steps=[
                "Ran tests",
                "Built image",
                "Pushed to registry",
                "Rolled out",
                "Checked health",
            ],
            outcome="success",
            scope=scope,
        )
        _commit_session(
            eng,
            goal="renew SSL certificate for docs site",
            steps=[
                "Checked cert expiry",
                "Requested new certificate",
                "Uploaded to CDN",
                "Verified HTTPS",
            ],
            outcome="success",
            scope=scope,
        )

        payload = _procedures_payload(path, {"min_sessions": ["2"]})

        assert payload["total_clusters_found"] == 0
    finally:
        eng.close()
        _cleanup(path)
