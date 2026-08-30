from __future__ import annotations

import json
import os
import tempfile
from types import SimpleNamespace

from slowave import ops
from slowave.core.config import SlowaveConfig
from slowave.core.engine import SlowaveEngine
from slowave.mcp.tools import (
    _canonical_activation_result,
    _canonical_recall_result,
    _normalize_remember_inputs,
    _serialized_chars,
    _validate_scope,
)


def _engine() -> tuple[SlowaveEngine, str]:
    handle = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    handle.close()
    return (
        SlowaveEngine(SlowaveConfig(db_path=handle.name, dim=8, disable_encoder=True)),
        handle.name,
    )


def _cleanup(eng: SlowaveEngine, path: str) -> None:
    eng.close()
    for suffix in ("", "-wal", "-shm"):
        if os.path.exists(path + suffix):
            os.remove(path + suffix)


def test_activation_persists_task_context_and_scope_bound_continuity() -> None:
    eng, path = _engine()
    try:
        result = ops.activate(
            eng,
            query="repair checkout latency",
            task="repair checkout latency",
            initial_goal="restore checkout latency",
            scope="project:shop",
            continuity_id="incident-42",
            task_context={"service": "checkout", "environment": "production"},
            include_peripheral=False,
        )
        row = (
            eng.db.connect()
            .execute(
                "SELECT scope_id, initial_goal, continuity_id, task_context_json "
                "FROM sessions WHERE id = ?",
                (result["session_id"],),
            )
            .fetchone()
        )
        assert row["scope_id"] == "project:shop"
        assert row["initial_goal"] == "restore checkout latency"
        assert row["continuity_id"] == "incident-42"
        assert json.loads(row["task_context_json"]) == {
            "service": "checkout",
            "environment": "production",
        }
    finally:
        _cleanup(eng, path)


def test_each_activation_opens_a_fresh_session() -> None:
    eng, path = _engine()
    try:
        kwargs = {
            "query": "audit contract",
            "task": "audit contract",
            "initial_goal": "audit contract",
            "scope": "project:test",
            "include_peripheral": False,
        }
        first = ops.activate(eng, **kwargs)
        second = ops.activate(eng, **kwargs)
        assert first["session_id"] != second["session_id"]
    finally:
        _cleanup(eng, path)


def test_recall_inherits_and_updates_session_task_context(monkeypatch) -> None:
    eng, path = _engine()
    try:
        activated = ops.activate(
            eng,
            query="investigate service",
            task="investigate service",
            initial_goal="identify service fault",
            scope="project:test",
            task_context={"service": "api"},
            include_peripheral=False,
        )

        def fake_recall(query: str, **_kwargs):
            assert '"runtime": "python"' in query
            assert '"service": "api"' in query
            return SimpleNamespace(
                schemas=[],
                related_schemas=[],
                schema_activations={},
                schema_rank_scores={},
                related_schema_relations={},
                episode_texts=[],
                raw_events=[],
            )

        monkeypatch.setattr(eng, "recall", fake_recall)
        recalled = ops.recall(
            eng,
            query="check runtime",
            session_id=activated["session_id"],
            scope="project:test",
            task_context={"runtime": "python"},
        )
        stored = (
            eng.db.connect()
            .execute(
                "SELECT task_context_json FROM sessions WHERE id = ?",
                (activated["session_id"],),
            )
            .fetchone()["task_context_json"]
        )
        assert json.loads(stored) == {"runtime": "python", "service": "api"}
        telemetry = (
            eng.db.connect()
            .execute(
                "SELECT scope_kind, situation_json FROM context_recall_events "
                "WHERE context_id = ?",
                (recalled["retrieval_id"],),
            )
            .fetchone()
        )
        assert telemetry["scope_kind"] == "project"
        assert json.loads(telemetry["situation_json"]) == {
            "runtime": "python",
            "service": "api",
        }
    finally:
        _cleanup(eng, path)


def test_public_scope_validation_is_shared_by_activate_and_recall() -> None:
    _validate_scope("project:test")
    for invalid in ("", "project", "project:", ":test"):
        try:
            _validate_scope(invalid)
        except ValueError as exc:
            assert str(exc) == "scope must use nonblank kind:id form"
        else:
            raise AssertionError(f"invalid scope was accepted: {invalid!r}")


def test_o2_activation_result_is_canonical_and_hides_diagnostics() -> None:
    result = _canonical_activation_result(
        {
            "retrieval_id": "ctx_1",
            "session_id": "sess_1",
            "cold_start": False,
            "rendered": "duplicate text",
            "cue_terms": ["secret", "diagnostic"],
            "suppressed": {"below_relevance": 2},
            "activation_trace": [{"schema_id": 1}],
            "schemas": [
                {
                    "id": "sch_1",
                    "text": "canonical memory",
                    "activation": 0.91,
                    "reason": "embedding similarity",
                    "pathway": "direct",
                    "source_kind": "explicit_remember",
                }
            ],
            "procedures": [],
            "scope_warning": "Possible scope fragmentation.",
        },
        scope="project:test",
    )
    assert result == {
        "retrieval_id": "ctx_1",
        "session_id": "sess_1",
        "memory_state": "available",
        "memories": [
            {
                "memory_id": "sch_1",
                "content": "canonical memory",
                "pathway": "direct",
                "provenance": {"source_kind": "explicit_remember"},
            }
        ],
        "procedures": [],
        "warnings": [{"code": "scope_fragmentation", "message": "Possible scope fragmentation."}],
    }


def test_activation_reports_origin_scope_for_generalized_memory() -> None:
    result = _canonical_activation_result(
        {
            "retrieval_id": "ctx_generalized",
            "session_id": "sess_generalized",
            "cold_start": False,
            "schemas": [
                {
                    "id": "sch_1",
                    "text": "portable guidance",
                    "pathway": "direct",
                    "scope_id": "project:origin",
                }
            ],
            "procedures": [],
        },
        scope="project:consumer",
    )
    assert result["memories"][0]["provenance"] == {"origin_scope": "project:origin"}


def test_activate_and_recall_share_memory_content_budget() -> None:
    content = "x" * 700
    activation = _canonical_activation_result(
        {
            "retrieval_id": "ctx_budget",
            "session_id": "sess_budget",
            "cold_start": False,
            "schemas": [{"id": "sch_1", "text": content}],
            "procedures": [],
        },
        scope="project:test",
    )
    recall = _canonical_recall_result(
        {
            "retrieval_id": "rec_budget",
            "memories": [{"id": "sch_1", "content_text": content}],
            "related_memories": [],
            "episodes": [],
            "raw_events": [],
            "procedures": [],
        },
        scope="project:test",
        evidence="references",
    )
    assert len(activation["memories"][0]["content"]) == 500
    assert recall["memories"][0]["content"] == activation["memories"][0]["content"]


def test_o2_cold_start_is_structured_without_duplicate_instruction_text() -> None:
    result = _canonical_activation_result(
        {
            "retrieval_id": "ctx_empty",
            "session_id": "sess_empty",
            "cold_start": True,
            "schemas": [],
            "procedures": [],
        },
        scope="project:test",
    )
    assert result["memory_state"] == "cold_start"
    assert result["memories"] == []
    assert "rendered" not in result
    assert "cold_start_hints" not in result


def test_continuity_start_response_has_one_serialized_budget_and_pathways() -> None:
    result = _canonical_activation_result(
        {
            "retrieval_id": "ctx_budget",
            "session_id": "sess_budget",
            "continuity_id": "cont_" + "a" * 43,
            "continuity_state": "started",
            "retrieval_policy_version": "continuity-v1",
            "schemas": [
                {"id": "sch_1", "text": "core", "pathway": "direct"},
                {
                    "id": "sch_2",
                    "text": "x" * 500,
                    "pathway": "context_reinstatement",
                },
                {
                    "id": "sch_3",
                    "text": "y" * 500,
                    "pathway": "context_reinstatement",
                },
                {
                    "id": "sch_4",
                    "text": "z" * 500,
                    "pathway": "context_reinstatement",
                },
            ],
            "procedures": [],
        },
        scope="project:test",
    )
    assert result["continuity_state"] == "started"
    assert result["memories"][0]["pathway"] == "direct"
    assert _serialized_chars(result) <= 1600
    assert result["more_available"] is True


def test_activation_procedures_are_canonical_and_feedback_authorized() -> None:
    eng, path = _engine()
    try:
        scope = "project:test"
        seeded = ops.activate(
            eng,
            query="repair service configuration",
            task="repair service configuration",
            initial_goal="repair service configuration",
            scope=scope,
            include_peripheral=False,
        )
        ops.commit(
            eng,
            session_id=seeded["session_id"],
            outcome="success",
            final_goal="repair service configuration",
            outcome_summary="The service configuration was repaired.",
            verification={"status": "verified", "summary": "Checks passed", "evidence_refs": []},
            procedure={
                "summary": "Inspect and repair service configuration.",
                "context": {},
                "steps": [{"summary": "Inspect configuration."}, {"summary": "Repair it."}],
                "caveats": [],
            },
            enforce_feedback=False,
        )
        activated = ops.activate(
            eng,
            query="repair another service configuration",
            task="repair another service configuration",
            initial_goal="repair another service configuration",
            scope=scope,
            include_peripheral=False,
        )
        procedure_id = activated["procedures"][0]["id"]
        exposed = (
            eng.db.connect()
            .execute(
                "SELECT 1 FROM context_recall_items WHERE context_id=? AND memory_id=? "
                "AND memory_type='procedural_memory' AND admitted=1",
                (activated["retrieval_id"], procedure_id),
            )
            .fetchone()
        )
        assert exposed is not None
        feedback = eng._feedback.feedback_events.record(
            retrieval_id=activated["retrieval_id"],
            procedure_feedback=[
                {"procedure_id": procedure_id, "use": "not_used", "effect": "unknown"}
            ],
            coverage="complete",
            mutation_mode="active",
        )
        assert feedback["rejected"] == []
        assert feedback["outstanding"] == {"memory_ids": [], "procedure_ids": []}
    finally:
        _cleanup(eng, path)


def test_recall_procedures_are_feedback_authorized(monkeypatch) -> None:
    eng, path = _engine()
    try:
        scope = "project:test"
        seeded = ops.activate(
            eng,
            query="repair service configuration",
            task="repair service configuration",
            initial_goal="repair service configuration",
            scope=scope,
            include_peripheral=False,
        )
        ops.commit(
            eng,
            session_id=seeded["session_id"],
            outcome="success",
            final_goal="repair service configuration",
            outcome_summary="The service configuration was repaired.",
            verification={"status": "verified", "summary": "Checks passed", "evidence_refs": []},
            procedure={
                "summary": "Inspect and repair service configuration.",
                "context": {},
                "steps": [{"summary": "Inspect configuration."}, {"summary": "Repair it."}],
                "caveats": [],
            },
            enforce_feedback=False,
        )
        active = ops.activate(
            eng,
            query="investigate another service",
            task="investigate another service",
            initial_goal="investigate another service",
            scope=scope,
            include_peripheral=False,
        )
        monkeypatch.setattr(
            eng,
            "recall",
            lambda *_args, **_kwargs: SimpleNamespace(
                schemas=[],
                related_schemas=[],
                schema_activations={},
                schema_rank_scores={},
                related_schema_relations={},
                episode_texts=[],
                raw_events=[],
            ),
        )
        recalled = ops.recall(
            eng,
            query="repair service configuration",
            session_id=active["session_id"],
            scope=scope,
        )
        procedure_id = recalled["procedures"][0]["id"]
        exposure = (
            eng.db.connect()
            .execute(
                "SELECT memory_type FROM context_recall_items "
                "WHERE context_id=? AND memory_id=? AND admitted=1",
                (recalled["retrieval_id"], procedure_id),
            )
            .fetchone()
        )
        assert exposure["memory_type"] == "procedural_memory"
        feedback = eng._feedback.feedback_events.record(
            retrieval_id=recalled["retrieval_id"],
            procedure_feedback=[
                {"procedure_id": procedure_id, "use": "not_used", "effect": "unknown"}
            ],
            coverage="complete",
            mutation_mode="active",
        )
        assert feedback["rejected"] == []
        assert feedback["outstanding"] == {"memory_ids": [], "procedure_ids": []}
    finally:
        _cleanup(eng, path)


def test_o4_recall_merges_pathways_and_removes_internal_scores() -> None:
    result = _canonical_recall_result(
        {
            "retrieval_id": "rec_1",
            "memories": [
                {
                    "id": "sch_1",
                    "content_text": "direct memory",
                    "activation": 0.9,
                    "rank_score": 1.2,
                    "scope_id": "project:test",
                    "salience": 4.2,
                    "status": "active",
                    "source_kind": "explicit_remember",
                }
            ],
            "related_memories": [
                {
                    "id": "sch_2",
                    "content_text": "associated memory",
                    "activation": 0.7,
                    "rank_score": 0.8,
                    "scope_id": "project:origin",
                    "via": ["internal_relation"],
                }
            ],
            "episodes": [],
            "raw_events": [],
            "procedures": [],
        },
        scope="project:test",
        evidence="references",
    )
    assert result["memories"] == [
        {
            "memory_id": "sch_1",
            "content": "direct memory",
            "pathway": "direct",
            "provenance": {"source_kind": "explicit_remember"},
        },
        {
            "memory_id": "sch_2",
            "content": "associated memory",
            "pathway": "associated",
            "provenance": {"origin_scope": "project:origin"},
        },
    ]
    assert "related_memories" not in result
    assert all("activation" not in memory for memory in result["memories"])


def test_o4_reference_and_full_evidence_are_bounded() -> None:
    payload = {
        "retrieval_id": "rec_evidence",
        "memories": [],
        "related_memories": [],
        "episodes": [{"id": index, "ts": index, "content_text": "x" * 1200} for index in range(10)],
        "raw_events": [],
        "procedures": [],
    }
    references = _canonical_recall_result(payload, scope="project:test", evidence="references")
    assert len(references["evidence"]) == 8
    assert references["evidence_truncated"] is True
    assert "content" not in references["evidence"][0]
    assert references["evidence"][0]["source_ref"] == {"kind": "episode", "id": 0}

    full = _canonical_recall_result(payload, scope="project:test", evidence="full")
    assert len(full["evidence"][0]["content"]) == 1000
    assert full["evidence"][0]["truncated"] is True


def test_o4_procedure_projection_removes_score_and_downstream_ids() -> None:
    result = _canonical_recall_result(
        {
            "retrieval_id": "rec_proc",
            "memories": [],
            "related_memories": [],
            "episodes": [],
            "raw_events": [],
            "procedures": [
                {
                    "id": "proc_1",
                    "scope_id": "project:test",
                    "goal": "repair service",
                    "summary": "Repair safely.",
                    "context": {},
                    "steps": [{"summary": "Inspect first."}],
                    "caveats": [],
                    "outcome": "success",
                    "outcome_summary": "Recovered.",
                    "created_at": 1,
                    "score": 0.99,
                    "evidence": {"retrieved": 10, "used": 2, "helped": 1},
                    "contributions": [
                        {
                            "effect": "helped",
                            "contribution": "Safe order.",
                            "downstream_session_id": "secret-session",
                            "downstream_scope_id": "project:secret",
                            "downstream_outcome": "success",
                            "downstream_outcome_summary": "Passed.",
                            "created_at": 2,
                        }
                    ],
                }
            ],
        },
        scope="project:test",
        evidence="references",
    )
    procedure = result["procedures"][0]
    assert procedure["procedure_id"] == "proc_1"
    assert "score" not in procedure
    assert "downstream_session_id" not in procedure["contributions"][0]
    assert procedure["evidence"] == {"used": 2, "helped": 1}


def test_o5_remember_requires_explicit_strict_type() -> None:
    import pytest

    with pytest.raises(ValueError, match="type must be"):
        _normalize_remember_inputs(content="claim", memory_type=None, memories=None)
    with pytest.raises(ValueError, match="type must be"):
        _normalize_remember_inputs(content="claim", memory_type="procedure", memories=None)
    normalized, is_batch = _normalize_remember_inputs(
        content="Follow this explicit direction.",
        memory_type="instruction",
        memories=None,
    )
    assert normalized == [
        {"content": "Follow this explicit direction.", "type": "instruction", "occurred_at": None}
    ]
    assert is_batch is False


def test_o5_remember_source_time_is_validated_without_overriding_recorded_time() -> None:
    import pytest

    normalized, is_batch = _normalize_remember_inputs(
        content="The incident began before the certificate rotation.",
        memory_type="fact",
        occurred_at="2026-08-19T14:05:00Z",
    )
    assert is_batch is False
    assert normalized[0]["occurred_at"] == 1787148300
    with pytest.raises(ValueError, match="UTC offset"):
        _normalize_remember_inputs(
            content="The incident began before the certificate rotation.",
            memory_type="fact",
            occurred_at="2026-08-19T14:05:00",
        )


def test_remember_preserves_source_time_separately_from_raw_event_timestamp() -> None:
    eng, path = _engine()
    try:
        source_time = 1787148300
        result = ops.remember(
            eng,
            content="The incident began before the certificate rotation.",
            memory_type="fact",
            scope="project:test",
            occurred_at=source_time,
        )
        event_id = int(result["source_event_id"].removeprefix("evt_"))
        event = eng.raw_log.get(event_id)
        assert event.metadata["occurred_at"] == source_time
        assert event.ts != source_time
    finally:
        _cleanup(eng, path)


def test_o5_batch_entries_cannot_override_scope_or_session() -> None:
    import pytest

    with pytest.raises(ValueError, match="content, type, and optional occurred_at"):
        _normalize_remember_inputs(
            content=None,
            memory_type=None,
            memories=[
                {
                    "content": "claim",
                    "type": "fact",
                    "scope": "project:other",
                }
            ],
        )
    with pytest.raises(ValueError, match="content, type, and optional occurred_at"):
        _normalize_remember_inputs(
            content=None,
            memory_type=None,
            memories=[
                {
                    "content": "claim",
                    "type": "fact",
                    "session_id": "sess_other",
                }
            ],
        )


def test_o5_instruction_is_not_execution_backed() -> None:
    eng, path = _engine()
    try:
        result = ops.remember(
            eng,
            content="Always inspect state before applying a change.",
            memory_type="instruction",
            scope="project:test",
        )
        schema = eng.schemas.get(int(result["memory_id"][4:]))
        assert schema.facets["schema_class"] == "instruction"
        assert schema.facets["instruction"] is True
        assert schema.facets["execution_backed"] is False
    finally:
        _cleanup(eng, path)


def test_o6_remember_reports_created_then_matched() -> None:
    eng, path = _engine()
    try:
        kwargs = {
            "content": "Use one canonical memory identifier.",
            "memory_type": "decision",
            "scope": "project:test",
        }
        created = ops.remember(eng, **kwargs)
        matched = ops.remember(eng, **kwargs)
        assert created == {
            "stored": True,
            "memory_id": created["memory_id"],
            "disposition": "created",
            "type": "decision",
            "scope": "project:test",
            "source_event_id": created["source_event_id"],
        }
        assert matched["memory_id"] == created["memory_id"]
        assert matched["disposition"] == "matched"
        assert matched["source_event_id"] != created["source_event_id"]
        assert "schema_id" not in matched
        assert "event_id" not in matched
    finally:
        _cleanup(eng, path)
