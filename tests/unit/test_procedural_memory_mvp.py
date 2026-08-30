from __future__ import annotations

import json
import os
import sqlite3
import tempfile

import pytest

from slowave import ops
from slowave.core.config import SlowaveConfig
from slowave.core.engine import SlowaveEngine
from slowave.mcp.session_reaper import _reap_once
from slowave.symbolic.procedural_memory import flatten_facets, retrieve_procedures


def _engine() -> tuple[SlowaveEngine, str]:
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    return (
        SlowaveEngine(SlowaveConfig(db_path=tmp.name, dim=8, disable_encoder=True)),
        tmp.name,
    )


def _cleanup(path: str) -> None:
    for suffix in ("", "-wal", "-shm"):
        if os.path.exists(path + suffix):
            os.remove(path + suffix)


def _procedure(service: str, *, failure: str = "crashloop") -> dict:
    return {
        "summary": "Diagnose and repair a failing orchestrated service",
        "context": {
            "aiops": {
                "platform": "kubernetes",
                "failure": failure,
                "service_id": service,
            }
        },
        "steps": [
            {
                "summary": "Inspect service evidence",
            },
            {
                "summary": "Repair the deployment",
            },
            {
                "summary": "Validate service health",
            },
        ],
        "caveats": ["Dependency failures may require an indirect repair."],
    }


def _attempt(eng: SlowaveEngine, service: str, outcome: str = "success") -> str:
    activated = ops.activate(
        eng,
        query=f"repair {service} crashloop",
        initial_goal=f"repair {service}",
        scope="project:aiops",
        retrieval_context={"aiops": {"platform": "kubernetes", "service_id": service}},
    )
    ops.commit(
        eng,
        session_id=activated["session_id"],
        final_goal=f"repair {service} deployment",
        outcome=outcome,
        outcome_summary=("Service recovered" if outcome == "success" else "Repair did not recover"),
        procedure=_procedure(service),
    )
    return activated["session_id"]


class _SimilarityEncoder:
    def __init__(self, similarity: float) -> None:
        self.similarity = similarity

    def encode(self, text: str):
        class _Vector:
            def __init__(self, similarity: float) -> None:
                self.similarity = similarity

            def dot(self, other: object) -> float:
                return self.similarity

        return _Vector(self.similarity)


def test_procedure_retrieval_requires_point_five_raw_similarity() -> None:
    procedure = {
        "id": "proc_example",
        "goal": "repair service",
        "summary": "Repair a failing service",
        "context": {},
        "steps": [{"summary": "Inspect and repair"}],
        "caveats": [],
        "outcome_summary": "Recovered",
        "contributions": [],
        "evidence": {
            "helped": 10,
            "harmed": 0,
        },
    }

    assert not retrieve_procedures([procedure], query="repair", encoder=_SimilarityEncoder(0.4999))
    admitted = retrieve_procedures([procedure], query="repair", encoder=_SimilarityEncoder(0.5))
    assert [item["id"] for item in admitted] == ["proc_example"]
    assert admitted[0]["score"] == 0.59


def test_activate_can_omit_duplicate_schemas_and_diagnostics() -> None:
    eng, path = _engine()
    try:
        lean = ops.activate(
            eng,
            query="inspect retrieval",
            scope="project:x",
            include_schemas=False,
            include_diagnostics=False,
        )
        assert "rendered" in lean
        assert "schemas" not in lean
        assert "cue_terms" not in lean
        assert "suppressed" not in lean

        debug = ops.activate(
            eng,
            query="inspect retrieval diagnostics",
            scope="project:x",
            mode="debug",
            include_schemas=False,
            include_diagnostics=False,
        )
        assert "schemas" in debug
        assert "cue_terms" in debug
        assert "suppressed" in debug
        assert "activation_trace" in debug
    finally:
        eng.close()
        _cleanup(path)


def test_structured_attempt_round_trip_and_retrieval() -> None:
    eng, path = _engine()
    try:
        first = _attempt(eng, "checkout")
        second = _attempt(eng, "billing")
        row = (
            eng.db.connect()
            .execute(
                "SELECT initial_goal, final_goal, outcome_summary FROM sessions WHERE id = ?",
                (first,),
            )
            .fetchone()
        )
        assert dict(row) == {
            "initial_goal": "repair checkout",
            "final_goal": "repair checkout deployment",
            "outcome_summary": "Service recovered",
        }
        event = (
            eng.db.connect()
            .execute(
                "SELECT metadata_json FROM raw_events WHERE session_id=? AND type='task_complete'",
                (first,),
            )
            .fetchone()
        )
        assert json.loads(event["metadata_json"])["procedure"]["version"] == 2

        result = ops.activate(
            eng,
            query="repair another kubernetes service",
            scope="project:aiops",
            initial_goal="repair kubernetes service",
            retrieval_context={"aiops": {"platform": "kubernetes", "failure": "crashloop"}},
        )
        assert {item["id"] for item in result["procedures"]} == {
            f"proc_{first}",
            f"proc_{second}",
        }
    finally:
        eng.close()
        _cleanup(path)


def test_context_difference_does_not_gate_precedent_retrieval() -> None:
    eng, path = _engine()
    try:
        _attempt(eng, "checkout")
        _attempt(eng, "billing")
        conflict = ops.activate(
            eng,
            query="repair kubernetes service",
            scope="project:aiops",
            initial_goal="repair kubernetes service",
            retrieval_context={"aiops": {"platform": "kubernetes", "failure": "latency"}},
        )
        assert len(conflict["procedures"]) == 2
        different_service = ops.activate(
            eng,
            query="repair kubernetes service",
            scope="project:aiops",
            initial_goal="repair kubernetes service",
            retrieval_context={
                "aiops": {
                    "platform": "kubernetes",
                    "failure": "crashloop",
                    "service_id": "search",
                }
            },
        )
        assert len(different_service["procedures"]) == 2
    finally:
        eng.close()
        _cleanup(path)


def test_failed_attempts_remain_retrievable_procedures() -> None:
    eng, path = _engine()
    try:
        _attempt(eng, "checkout", outcome="failure")
        _attempt(eng, "billing", outcome="failure")
        result = ops.activate(
            eng,
            query="repair kubernetes service",
            initial_goal="repair service",
            scope="project:aiops",
            retrieval_context={"aiops": {"platform": "kubernetes", "failure": "crashloop"}},
        )
        assert len(result["procedures"]) == 2
        assert {item["outcome"] for item in result["procedures"]} == {"failure"}
    finally:
        eng.close()
        _cleanup(path)


def test_contract_validation_and_legacy_exclusivity() -> None:
    assert flatten_facets({"aiops": {"cluster_id": "Prod-1"}}) == {
        "aiops.cluster_id": frozenset({"prod-1"})
    }
    eng, path = _engine()
    try:
        activated = ops.activate(
            eng, query="task", goal="same", initial_goal="same", scope="project:x"
        )
        with pytest.raises(ValueError, match="either legacy steps"):
            ops.commit(
                eng,
                session_id=activated["session_id"],
                steps=["legacy"],
                procedure=_procedure("x"),
            )
        controlled = _procedure("x")
        controlled["steps"][0]["operation"] = "inspect"
        with pytest.raises(ValueError, match="controlled preconditions"):
            ops.commit(
                eng,
                session_id=activated["session_id"],
                procedure=controlled,
            )
        with pytest.raises(ValueError, match="must match"):
            ops.activate(
                eng,
                query="task",
                goal="one",
                initial_goal="two",
                scope="project:x",
            )
    finally:
        eng.close()
        _cleanup(path)


def test_precedent_contract_and_influence_round_trip() -> None:
    eng, path = _engine()
    try:
        activated = ops.activate(
            eng, query="repair task", initial_goal="repair task", scope="project:x"
        )
        sid = activated["session_id"]
        ops.commit(
            eng,
            session_id=sid,
            outcome="success",
            procedure={
                "summary": "Inspect evidence, make the smallest repair, and verify it",
                "context": {"platform": "kubernetes"},
                "steps": [
                    {"summary": "Inspect the failing workload and its dependencies"},
                    {"summary": "Apply the smallest evidence-backed repair"},
                    {"summary": "Verify the workload with the hidden acceptance check"},
                ],
                "caveats": ["Dependency failures can make a local repair ineffective."],
            },
            procedure_uses=[
                {
                    "procedure_id": "proc_123",
                    "use": "used",
                    "effect": "helped",
                    "contribution": "Its dependency-isolation sequence prompted checking indirect constraints first.",
                },
                {
                    "procedure_id": "proc_456",
                    "use": "not_used",
                    "effect": "unknown",
                },
            ],
        )
        event = (
            eng.db.connect()
            .execute(
                "SELECT metadata_json FROM raw_events WHERE session_id=? AND type='task_complete'",
                (sid,),
            )
            .fetchone()
        )
        metadata = json.loads(event["metadata_json"])
        assert metadata["contract_version"] == 2
        assert metadata["procedure"]["version"] == 2
        assert metadata["procedure"]["context"] == {"platform": "kubernetes"}
        assert metadata["procedure_uses"][0]["effect"] == "helped"

        recalled = ops.activate(
            eng,
            query="inspect evidence before repairing workload",
            initial_goal="repair another workload",
            scope="project:x",
        )
        assert recalled["procedures"][0]["id"] == f"proc_{sid}"
        assert recalled["procedures"][0]["evidence"]["not_used"] == 0
    finally:
        eng.close()
        _cleanup(path)


@pytest.mark.parametrize(
    ("procedure_use", "message"),
    [
        ({"procedure_id": "p", "use": "maybe", "effect": "unknown"}, "use must"),
        ({"procedure_id": "p", "use": "used", "effect": "helped"}, "contribution"),
        (
            {
                "procedure_id": "p",
                "use": "not_used",
                "effect": "helped",
                "contribution": "It helped anyway",
            },
            "not_used",
        ),
    ],
)
def test_procedure_use_cross_field_validation(procedure_use: dict, message: str) -> None:
    eng, path = _engine()
    try:
        activated = ops.activate(eng, query="task", initial_goal="task", scope="project:x")
        with pytest.raises(ValueError, match=message):
            ops.commit(
                eng,
                session_id=activated["session_id"],
                procedure_uses=[procedure_use],
            )
    finally:
        eng.close()
        _cleanup(path)


def test_legacy_goal_is_backfilled_into_initial_goal() -> None:
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    conn = sqlite3.connect(tmp.name)
    conn.execute(
        "CREATE TABLE sessions (id TEXT PRIMARY KEY, agent TEXT NOT NULL, "
        "started_ts INTEGER NOT NULL, goal TEXT, outcome TEXT)"
    )
    conn.execute(
        "INSERT INTO sessions (id, agent, started_ts, goal) VALUES ('legacy', 'test', 1, 'old goal')"
    )
    conn.commit()
    conn.close()
    eng = SlowaveEngine(SlowaveConfig(db_path=tmp.name, dim=8, disable_encoder=True))
    try:
        row = (
            eng.db.connect()
            .execute("SELECT goal, initial_goal FROM sessions WHERE id='legacy'")
            .fetchone()
        )
        assert dict(row) == {"goal": "old goal", "initial_goal": "old goal"}
    finally:
        eng.close()
        _cleanup(tmp.name)


def test_repeated_commit_keeps_one_canonical_completion_event() -> None:
    eng, path = _engine()
    try:
        activated = ops.activate(
            eng,
            query="repair service",
            initial_goal="repair service",
            scope="project:x",
        )
        sid = activated["session_id"]
        ops.commit(eng, session_id=sid, outcome="partial")
        result = ops.commit(
            eng,
            session_id=sid,
            outcome="success",
            final_goal="repair service safely",
            outcome_summary="Service recovered",
            procedure=_procedure("checkout"),
        )
        rows = (
            eng.db.connect()
            .execute(
                "SELECT metadata_json FROM raw_events "
                "WHERE session_id = ? AND type = 'task_complete'",
                (sid,),
            )
            .fetchall()
        )
        assert len(rows) == 1
        assert json.loads(rows[0]["metadata_json"])["outcome"] == "success"
        assert result["episodes_formed"] == 0
        assert result["already_ended"] is True
    finally:
        eng.close()
        _cleanup(path)


def test_reaper_then_late_commit_keeps_one_canonical_completion_event() -> None:
    eng, path = _engine()
    try:
        activated = ops.activate(
            eng,
            query="repair service",
            initial_goal="repair service",
            scope="project:x",
        )
        sid = activated["session_id"]
        conn = eng.db.connect()
        conn.execute("UPDATE sessions SET started_ts = 1 WHERE id = ?", (sid,))
        conn.execute("UPDATE raw_events SET ts = 1 WHERE session_id = ?", (sid,))
        conn.commit()

        def build() -> SlowaveEngine:
            return SlowaveEngine(SlowaveConfig(db_path=path, dim=8, disable_encoder=True))

        assert sid in _reap_once(build, timeout_s=0)
        result = ops.commit(
            eng,
            session_id=sid,
            outcome="success",
            procedure=_procedure("checkout"),
        )
        rows = (
            eng.db.connect()
            .execute(
                "SELECT metadata_json FROM raw_events "
                "WHERE session_id = ? AND type = 'task_complete'",
                (sid,),
            )
            .fetchall()
        )
        assert len(rows) == 1
        assert json.loads(rows[0]["metadata_json"])["procedure"] is not None
        assert result["already_ended"] is True
    finally:
        eng.close()
        _cleanup(path)
