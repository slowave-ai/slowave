from __future__ import annotations

import json
import os
import tempfile
from typing import cast

import numpy as np
import pytest

from slowave import ops
from slowave.core.config import SlowaveConfig
from slowave.core.engine import SlowaveEngine
from slowave.symbolic.encoder import TextEncoder


def _engine() -> tuple[SlowaveEngine, str]:
    f = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    f.close()
    return SlowaveEngine(SlowaveConfig(db_path=f.name, dim=8, disable_encoder=True)), f.name


def _cleanup(path: str) -> None:
    for suffix in ("", "-wal", "-shm"):
        if os.path.exists(path + suffix):
            os.remove(path + suffix)


def _schema(eng: SlowaveEngine, text: str = "memory") -> str:
    vec = np.ones(8, dtype=np.float32)
    vec /= np.linalg.norm(vec)
    sid = eng.schemas.create(content_text=text, facets={}, tags=[], embedding=vec)
    return f"sch_{sid}"


def test_feedback_is_append_only_and_outcome_independent() -> None:
    eng, path = _engine()
    try:
        mid = _schema(eng)
        eng.record_retrieval(
            retrieval_id="rec_v9",
            response={"schemas": [{"id": mid, "content": "memory"}]},
        )
        first = eng.feedback(
            retrieval_id="rec_v9",
            memory_feedback=[
                {
                    "memory_id": mid,
                    "assessment": "stale",
                    "stale_reason": "outdated",
                    "reason": "This claim is no longer current.",
                }
            ],
        )
        second = eng.feedback(
            retrieval_id="rec_v9",
            memory_feedback=[{"memory_id": mid, "assessment": "used"}],
        )
        rows = (
            eng.db.connect()
            .execute("SELECT * FROM feedback_events WHERE target_kind='memory' ORDER BY rowid")
            .fetchall()
        )
        assert len(rows) == 2
        assert rows[1]["refines_event_id"] == rows[0]["event_id"]
        assert "outcome" not in rows[0].keys()
        assert first["applied"]["superseded"] == []
        assert second["applied"]["strengthened"] == []
    finally:
        eng.close()
        _cleanup(path)


@pytest.mark.parametrize("assessment", ["stale"])
def test_v9_stale_or_contradicted_feedback_suppresses_active_schema_from_current_retrieval(
    assessment: str,
) -> None:
    eng, path = _engine()
    try:
        mid = _schema(eng, "The current refund window is 30 days.")
        eng.record_retrieval(retrieval_id=f"rec_{assessment}", response={"schemas": [{"id": mid}]})
        result = eng.feedback(
            retrieval_id=f"rec_{assessment}",
            memory_feedback=[
                {
                    "memory_id": mid,
                    "assessment": assessment,
                    "stale_reason": "outdated",
                    "reason": "Client evidence conflicts with this claim.",
                }
            ],
            coverage="complete",
        )
        schema = eng.schemas.get(int(mid.removeprefix("sch_")))
        expected_reason = "outdated"
        assert result["applied"][expected_reason] == [mid]
        assert schema.is_labile is False
        assert schema.status == "stale"
        assert schema.stale_reason == expected_reason
    finally:
        eng.close()
        _cleanup(path)


def test_unauthorized_target_is_rejected_without_mutation() -> None:
    eng, path = _engine()
    try:
        shown, hidden = _schema(eng, "shown"), _schema(eng, "hidden")
        eng.record_retrieval(retrieval_id="rec_auth", response={"schemas": [{"id": shown}]})
        before = eng.schemas.get(int(hidden[4:])).salience
        result = eng.feedback(
            retrieval_id="rec_auth",
            memory_feedback=[{"memory_id": hidden, "assessment": "used"}],
        )
        assert result["rejected"] == [{"target_id": hidden, "reason": "target_not_exposed"}]
        assert eng.schemas.get(int(hidden[4:])).salience == before
    finally:
        eng.close()
        _cleanup(path)


@pytest.mark.parametrize("assessment", ["wrong", "contradicted"])
def test_legacy_truth_aliases_are_rejected(assessment: str) -> None:
    eng, path = _engine()
    try:
        mid = _schema(eng)
        eng.record_retrieval(
            retrieval_id=f"rec_alias_{assessment}", response={"schemas": [{"id": mid}]}
        )
        result = eng.feedback(
            retrieval_id=f"rec_alias_{assessment}",
            memory_feedback=[{"memory_id": mid, "assessment": assessment}],
        )
        assert result["rejected"] == [{"target_id": mid, "reason": "invalid_memory_assessment"}]
        assert eng.schemas.get(int(mid[4:])).status == "active"
    finally:
        eng.close()
        _cleanup(path)


def test_conflicting_feedback_is_recorded_without_status_oscillation() -> None:
    eng, path = _engine()
    try:
        mid = _schema(eng, "The current refund window is 30 days.")
        eng.record_retrieval(retrieval_id="rec_conflict", response={"schemas": [{"id": mid}]})

        eng.feedback(
            retrieval_id="rec_conflict",
            memory_feedback=[
                {
                    "memory_id": mid,
                    "assessment": "stale",
                    "stale_reason": "outdated",
                    "reason": "No longer current.",
                }
            ],
        )
        result = eng.feedback(
            retrieval_id="rec_conflict",
            memory_feedback=[
                {"memory_id": mid, "assessment": "wrong", "reason": "Contradicts current evidence."}
            ],
        )

        rows = (
            eng.db.connect()
            .execute(
                "SELECT assessment FROM feedback_events "
                "WHERE target_kind='memory' ORDER BY created_at, rowid"
            )
            .fetchall()
        )
        assert [row["assessment"] for row in rows] == ["stale", "wrong"]
        assert eng.schemas.get(int(mid[4:])).status == "stale"
        assert eng.schemas.get(int(mid[4:])).stale_reason == "outdated"
        assert result["applied"]["contradicted"] == []
    finally:
        eng.close()
        _cleanup(path)


def test_irrelevant_feedback_never_changes_semantic_status() -> None:
    eng, path = _engine()
    try:
        mid = _schema(eng, "The refund window is 30 days.")
        eng.record_retrieval(retrieval_id="rec_irrelevant", response={"schemas": [{"id": mid}]})

        result = eng.feedback(
            retrieval_id="rec_irrelevant",
            memory_feedback=[{"memory_id": mid, "assessment": "irrelevant"}],
        )

        assert eng.schemas.get(int(mid[4:])).status == "active"
        assert result["applied"]["access_evidence"] == [mid]
    finally:
        eng.close()
        _cleanup(path)


def test_feedback_records_explicit_replacement_lineage() -> None:
    eng, path = _engine()
    try:
        old = _schema(eng, "The refund window is 30 days.")
        new = _schema(eng, "The refund window is 14 days.")
        eng.record_retrieval(retrieval_id="rec_replace", response={"schemas": [{"id": old}]})

        result = eng.feedback(
            retrieval_id="rec_replace",
            memory_feedback=[
                {
                    "memory_id": old,
                    "assessment": "stale",
                    "stale_reason": "superseded",
                    "reason": "A newer claim replaces this one.",
                    "replacement_memory_id": new,
                }
            ],
        )

        row = (
            eng.db.connect()
            .execute(
                "SELECT replacement_target_id FROM feedback_events "
                "WHERE target_kind='memory' AND target_id=?",
                (old,),
            )
            .fetchone()
        )
        assert row["replacement_target_id"] == new
        assert eng.schemas.get(int(old[4:])).status == "stale"
        assert eng.schemas.get(int(old[4:])).stale_reason == "superseded"
        assert eng.schemas.get(int(new[4:])).status == "active"
        assert result["applied"]["replacements"] == [f"{old}->{new}"]
        current_ids = {
            f"sch_{schema.id}"
            for schema in eng.context_brief(query="refund window", mode="default").schemas
        }
        history_ids = {f"sch_{schema.id}" for schema in eng.schemas.list(status="stale")}
        assert old not in current_ids
        assert old in history_ids
    finally:
        eng.close()
        _cleanup(path)


def test_procedure_unknown_effect_can_be_refined() -> None:
    eng, path = _engine()
    try:
        pid = "proc_example"
        eng.record_retrieval(retrieval_id="rec_proc", response={"procedures": [{"id": pid}]})
        eng.feedback(
            retrieval_id="rec_proc",
            procedure_feedback=[
                {
                    "procedure_id": pid,
                    "use": "used",
                    "effect": "unknown",
                    "contribution": "selected the safe sequence",
                }
            ],
        )
        eng.feedback(
            retrieval_id="rec_proc",
            procedure_feedback=[
                {
                    "procedure_id": pid,
                    "use": "used",
                    "effect": "helped",
                    "contribution": "the safe sequence passed verification",
                }
            ],
        )
        rows = (
            eng.db.connect()
            .execute(
                "SELECT effect, refines_event_id, event_id FROM feedback_events WHERE target_kind='procedure' ORDER BY rowid"
            )
            .fetchall()
        )
        assert [row["effect"] for row in rows] == ["unknown", "helped"]
        assert rows[1]["refines_event_id"] == rows[0]["event_id"]
    finally:
        eng.close()
        _cleanup(path)


def test_used_procedure_requires_contribution_and_unknown_retrieval_fails() -> None:
    eng, path = _engine()
    try:
        eng.record_retrieval(
            retrieval_id="rec_proc_invalid", response={"procedures": [{"id": "proc_x"}]}
        )
        result = eng.feedback(
            retrieval_id="rec_proc_invalid",
            procedure_feedback=[{"procedure_id": "proc_x", "use": "used", "effect": "unknown"}],
        )
        assert result["rejected"][0]["reason"] == "used_procedure_requires_contribution"
        with pytest.raises(ValueError, match="unknown retrieval_id"):
            eng.feedback(retrieval_id="rec_missing")
    finally:
        eng.close()
        _cleanup(path)


def test_partial_silence_creates_no_target_assessment() -> None:
    eng, path = _engine()
    try:
        mid = _schema(eng)
        eng.record_retrieval(retrieval_id="rec_silent", response={"schemas": [{"id": mid}]})
        eng.feedback(retrieval_id="rec_silent", coverage="partial")
        rows = (
            eng.db.connect()
            .execute("SELECT target_kind FROM feedback_events WHERE retrieval_id='rec_silent'")
            .fetchall()
        )
        assert [row["target_kind"] for row in rows] == ["retrieval"]
    finally:
        eng.close()
        _cleanup(path)


def test_complete_coverage_is_mechanically_checked() -> None:
    eng, path = _engine()
    try:
        mid = _schema(eng)
        eng.record_retrieval(retrieval_id="rec_complete", response={"schemas": [{"id": mid}]})
        incomplete = eng.feedback(retrieval_id="rec_complete", coverage="complete")
        assert incomplete["outstanding"]["memory_ids"] == [mid]
        assert incomplete["rejected"][0]["reason"] == "incomplete_coverage"
        complete = eng.feedback(
            retrieval_id="rec_complete",
            memory_feedback=[{"memory_id": mid, "assessment": "used"}],
            coverage="complete",
        )
        assert complete["outstanding"] == {"memory_ids": [], "procedure_ids": []}
    finally:
        eng.close()
        _cleanup(path)


def test_commit_blocks_until_session_exposures_are_complete() -> None:
    eng, path = _engine()
    try:
        sid = eng.session_start(agent="test", scope="project:test", goal="verify gate")
        mid = _schema(eng)
        eng.record_retrieval(
            retrieval_id="ctx_commit_gate",
            session_id=sid,
            response={"schemas": [{"id": mid}]},
        )
        with pytest.raises(ops.IncompleteFeedbackError) as exc:
            ops.commit(
                eng,
                session_id=sid,
                final_goal="verify commit gate",
                outcome="success",
                outcome_summary="gate verified",
                verification={"status": "verified", "summary": "unit test"},
                enforce_feedback=True,
            )
        assert exc.value.outstanding[0]["memory_ids"] == [mid]
        assert not eng.raw_log.is_session_ended(sid)
        eng.feedback(
            retrieval_id="ctx_commit_gate",
            memory_feedback=[{"memory_id": mid, "assessment": "used"}],
            coverage="complete",
        )
        result = ops.commit(
            eng,
            session_id=sid,
            final_goal="verify commit gate",
            outcome="success",
            outcome_summary="gate verified",
            verification={"status": "verified", "summary": "unit test"},
            enforce_feedback=True,
        )
        assert result["operation"] == "closed"
        assert result["feedback_status"] == "complete"
    finally:
        eng.close()
        _cleanup(path)


def test_forced_closure_stays_incomplete_during_delayed_update() -> None:
    eng, path = _engine()
    try:
        sid = eng.session_start(agent="test", scope="project:test", goal="abandoned")
        first = ops.commit(
            eng,
            session_id=sid,
            outcome="unknown",
            verification={"status": "unverified", "summary": "idle closure"},
            enforce_feedback=False,
        )
        assert first["feedback_status"] == "incomplete"
        updated = ops.commit(
            eng,
            session_id=sid,
            final_goal="record delayed result",
            outcome="failure",
            outcome_summary="late failure observed",
            verification={"status": "verified", "summary": "external observation"},
            enforce_feedback=True,
        )
        assert updated["operation"] == "updated"
        assert updated["feedback_status"] == "incomplete"
        row = (
            eng.db.connect()
            .execute(
                "SELECT outcome, verification_json, feedback_status FROM sessions WHERE id = ?",
                (sid,),
            )
            .fetchone()
        )
        assert row["outcome"] == "failure"
        assert row["feedback_status"] == "incomplete"
        assert json.loads(row["verification_json"])["status"] == "verified"
    finally:
        eng.close()
        _cleanup(path)


def test_recall_snapshot_is_bound_to_explicit_matching_session() -> None:
    eng, path = _engine()
    try:

        class StubEncoder:
            def encode(self, _text: str) -> np.ndarray:
                return np.ones(8, dtype=np.float32) / np.sqrt(8)

        eng.encoder = cast(TextEncoder, StubEncoder())
        activated = ops.activate(
            eng,
            query="start",
            initial_goal="test recall binding",
            scope="project:test",
        )
        sid = activated["session_id"]
        recalled = ops.recall(
            eng,
            query="specific lookup",
            session_id=sid,
            scope="project:test",
            task_context={"component": "feedback"},
        )
        row = (
            eng.db.connect()
            .execute(
                "SELECT session_id FROM context_recall_events WHERE context_id = ?",
                (recalled["retrieval_id"],),
            )
            .fetchone()
        )
        assert row["session_id"] == sid
        with pytest.raises(ValueError, match="do not match"):
            ops.recall(
                eng,
                query="wrong scope",
                session_id=sid,
                scope="project:other",
            )
    finally:
        eng.close()
        _cleanup(path)
