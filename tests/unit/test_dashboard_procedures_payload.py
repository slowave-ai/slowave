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
from slowave.dashboard.app import _procedural_memory_payload, _procedures_payload
from slowave.lifecycle import LIFECYCLE_VERSION


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


def _commit_structured_session(eng: SlowaveEngine, *, goal: str, outcome: str, scope: str) -> None:
    result = ops.activate(eng, query=goal, scope=scope, goal=goal, agent="test")
    ops.commit(
        eng,
        session_id=result["session_id"],
        outcome=outcome,
        final_goal=goal,
        outcome_summary=f"{outcome}: {goal}",
        procedure={
            "summary": "Inspect, repair, and validate service configuration",
            "context": {"platform": "test", "artifact": "service_config"},
            "steps": [
                {"summary": "Inspect configuration"},
                {"summary": "Repair configuration"},
                {"summary": "Validate service"},
            ],
            "caveats": ["Inspect dependencies before changing configuration."],
        },
    )


def test_structured_dogfood_payload_covers_precedents_retrieval_and_influence() -> None:
    eng, path = _tmp_engine()
    try:
        scope = "project:x"
        _commit_structured_session(
            eng, goal="repair auth configuration", outcome="success", scope=scope
        )
        _commit_structured_session(
            eng, goal="repair billing configuration", outcome="success", scope=scope
        )
        retrieval = ops.activate(
            eng,
            query="repair service configuration",
            scope=scope,
            goal="repair service configuration",
            retrieval_context={"platform": "test", "artifact": "service_config"},
            agent="test",
        )
        procedure_id = retrieval["procedures"][0]["id"]
        ops.reinforce(
            eng,
            retrieval_id=retrieval["retrieval_id"],
            feedback="useful",
            outcome="success",
            used_procedure_ids=[procedure_id],
        )
        eng._feedback.feedback_events.record(
            retrieval_id=retrieval["retrieval_id"],
            procedure_feedback=[
                {
                    "procedure_id": procedure_id,
                    "use": "used",
                    "effect": "helped",
                    "contribution": "The inspection-first ordering transferred.",
                }
            ],
            coverage="partial",
            mutation_mode="active",
        )
        influenced = ops.activate(
            eng,
            query="repair another service configuration",
            scope=scope,
            goal="repair another service configuration",
            agent="test",
        )
        ops.commit(
            eng,
            session_id=influenced["session_id"],
            outcome="success",
            procedure_uses=[
                {
                    "procedure_id": procedure_id,
                    "use": "used",
                    "effect": "helped",
                    "contribution": "The inspection-first ordering exposed the faulty setting.",
                }
            ],
        )

        payload = _procedural_memory_payload(path, {"scope": [scope]})

        assert payload["status"] == "dogfooding"
        assert payload["structured_attempts"] == 2
        assert payload["completed_sessions"] == 3
        assert payload["capture_rate"] == 2 / 3
        assert len(payload["procedures"]) == 2
        precedent = next(item for item in payload["procedures"] if item["id"] == procedure_id)
        # The current v9 payload joins canonical feedback_events as well as
        # legacy commit-time procedure_uses.  These are two distinct
        # retrievals, so both influences remain visible on the precedent.
        assert precedent["evidence"]["used"] == 2
        assert precedent["evidence"]["helped"] == 2
        assert precedent["evidence"]["not_used"] == 0
        assert precedent["contributions"][0]["downstream_goal"] == (
            "repair another service configuration"
        )
        assert precedent["contributions"][0]["downstream_session_id"] == (influenced["session_id"])
        assert payload["influence_counts"]["helped"] == 2
        assert payload["procedure_retrievals"] >= 2
        assert payload["feedback_counts"]["used"] == 1
        assert payload["feedback_counts"]["legacy"]["used"] == 1
        assert payload["feedback_counts"]["v9"]["accepted"] == 1
        assert payload["feedback_counts"]["v9"]["used"] == 1
        assert payload["feedback_counts"]["v9"]["helped"] == 1
        assert any(procedure_id in item["procedure_ids"] for item in payload["recent_retrievals"])
        retrieval_history = precedent["retrievals"]
        assert any(item["session_id"] == retrieval["session_id"] for item in retrieval_history)
        assert all(procedure_id in item["procedure_ids"] for item in retrieval_history)
        influenced_retrieval = next(
            item for item in retrieval_history if item["session_id"] == influenced["session_id"]
        )
        assert influenced_retrieval["procedure_assessment"] == {
            "procedure_id": procedure_id,
            "use": "used",
            "effect": "helped",
            "contribution": "The inspection-first ordering exposed the faulty setting.",
        }
        uncommitted_retrieval = next(
            item for item in retrieval_history if item["session_id"] == retrieval["session_id"]
        )
        assert uncommitted_retrieval["procedure_assessment"]["source"] == "v9"
        assert uncommitted_retrieval["procedure_assessment"]["mutation_mode"] == "active"

        most_used = _procedural_memory_payload(path, {"scope": [scope], "sort": ["used"]})
        assert most_used["sort"] == "used"
        assert most_used["procedures"][0]["id"] == procedure_id

        most_retrieved = _procedural_memory_payload(path, {"scope": [scope], "sort": ["retrieved"]})
        assert most_retrieved["sort"] == "retrieved"
        retrieved_counts = [item["evidence"]["retrieved"] for item in most_retrieved["procedures"]]
        assert retrieved_counts == sorted(retrieved_counts, reverse=True)
        # Retrieved counts are exact snapshot exposures. Explicit use feedback
        # remains a separate field and must not inflate the exposure count.
        assert precedent["evidence"]["retrieved"] == len(retrieval_history)

        by_summary = _procedural_memory_payload(
            path, {"scope": [scope], "sort": ["summary"], "dir": ["asc"]}
        )
        assert by_summary["sort"] == "summary"
        assert by_summary["sort_direction"] == "asc"
        assert [item["summary"] for item in by_summary["procedures"]] == sorted(
            item["summary"] for item in by_summary["procedures"]
        )
    finally:
        eng.close()
        _cleanup(path)


def test_procedural_payload_defaults_to_current_lifecycle_and_attributes_feedback() -> None:
    eng, path = _tmp_engine()
    try:
        scope = "project:x"
        _commit_structured_session(
            eng, goal="repair v9 configuration", outcome="success", scope=scope
        )
        legacy_session = eng.session_start(
            agent="test", scope=scope, goal="repair legacy configuration", lifecycle_version="v8"
        )
        ops.commit(
            eng,
            session_id=legacy_session,
            outcome="success",
            final_goal="repair legacy configuration",
            outcome_summary="legacy procedure",
            procedure={
                "summary": "Legacy repair procedure",
                "context": {},
                "steps": [{"summary": "Inspect legacy configuration"}],
                "caveats": [],
            },
        )
        retrieval = ops.activate(
            eng,
            query="repair v9 configuration",
            scope=scope,
            goal="repair v9 configuration",
            agent="test",
        )
        procedure_id = retrieval["procedures"][0]["id"]
        eng._feedback.feedback_events.record(
            retrieval_id=retrieval["retrieval_id"],
            procedure_feedback=[
                {
                    "procedure_id": procedure_id,
                    "use": "used",
                    "effect": "helped",
                    "contribution": "The v9 precedent provided the repair order.",
                }
            ],
            coverage="complete",
            mutation_mode="active",
        )

        current = _procedural_memory_payload(path, {"scope": [scope]})
        all_versions = _procedural_memory_payload(path, {"scope": [scope], "cohort": ["all"]})

        assert current["cohort"] == LIFECYCLE_VERSION
        assert current["structured_attempts"] == 1
        assert current["procedures"][0]["id"] == procedure_id
        assert current["procedures"][0]["evidence"] == {
            "retrieved": 1,
            "used": 1,
            "not_used": 0,
            "helped": 1,
            "no_effect": 0,
            "harmed": 0,
            "unknown": 0,
        }
        assert all_versions["cohort"] == "all"
        assert all_versions["structured_attempts"] == 2
    finally:
        eng.close()
        _cleanup(path)


def test_insufficient_data_gate_when_no_step_sessions() -> None:
    eng, path = _tmp_engine()
    try:
        # A session with goal+outcome but no commit(steps=...) data.
        result = ops.activate(
            eng,
            query="do something",
            scope="project:x",
            goal="do something",
            agent="test",
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
