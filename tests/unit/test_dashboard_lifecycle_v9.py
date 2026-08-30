from __future__ import annotations

import json
import os
import tempfile

from slowave import ops
from slowave.core.config import SlowaveConfig
from slowave.core.engine import SlowaveEngine
from slowave.dashboard.app import _session_timeline, _status_payload


def test_status_separates_feedback_generations_and_reports_lifecycle_health() -> None:
    handle = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    handle.close()
    eng = SlowaveEngine(SlowaveConfig(db_path=handle.name, dim=8, disable_encoder=True))
    try:
        activated = ops.activate(
            eng,
            query="verify lifecycle dashboard",
            task="verify lifecycle dashboard",
            initial_goal="verify lifecycle dashboard",
            scope="project:test",
            include_peripheral=False,
        )
        eng._feedback.feedback_events.record(
            retrieval_id=activated["retrieval_id"], coverage="partial", mutation_mode="active"
        )
        ops.reinforce(
            eng,
            retrieval_id=activated["retrieval_id"],
            feedback="missing",
        )
        ops.commit(
            eng,
            session_id=activated["session_id"],
            outcome="success",
            final_goal="verified lifecycle dashboard",
            outcome_summary="Dashboard fixture completed.",
            verification={"status": "verified", "summary": "Fixture passed"},
            trajectory=[
                {"kind": "action", "summary": "Ran fixture", "status": "succeeded"},
                {"kind": "observation", "summary": "Fixture passed", "status": "succeeded"},
            ],
            provenance={
                "source_kind": "integration",
                "integration": "test-client",
                "request_id": "internal-secret",
            },
        )

        payload = _status_payload(handle.name)
        assert payload["stats"]["legacy_feedback_events"] == 1
        assert payload["stats"]["v9_feedback_events"] >= 1
        assert payload["stats"]["feedback_events"] == (
            payload["stats"]["legacy_feedback_events"] + payload["stats"]["v9_feedback_events"]
        )
        health = payload["lifecycle_health"]
        assert health["feedback_events"]["accepted"] >= 1
        assert health["trajectory"]["events"] == 2
        assert health["trajectory"]["sessions"] == 1
        assert health["trajectory"]["episodes_formed"] == 0

        timeline = _session_timeline(handle.name, activated["session_id"])
        assert timeline["session"]["initial_goal"] == "verify lifecycle dashboard"
        assert timeline["session"]["final_goal"] == "verified lifecycle dashboard"
        assert timeline["session"]["feedback_status"] == "incomplete"
        assert timeline["session"]["verification"]["status"] == "verified"
        trajectory = [e for e in timeline["events"] if e["type"].startswith("trajectory:")]
        assert [e["status"] for e in trajectory] == ["succeeded", "succeeded"]
        assert trajectory[0]["provenance"]["source_kind"] == "agent_inference"
        assert "request_id" not in json.dumps(timeline)
    finally:
        eng.close()
        for suffix in ("", "-wal", "-shm"):
            if os.path.exists(handle.name + suffix):
                os.remove(handle.name + suffix)
