from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from slowave import ops
from slowave.core.config import SlowaveConfig
from slowave.core.engine import SlowaveEngine
from slowave.ops import _shadow_task_interpretation

_SCRIPT = Path(__file__).parents[2] / "scripts" / "validate_feedback_v9_replay.py"
_SPEC = importlib.util.spec_from_file_location("validate_feedback_v9_replay", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
report = _MODULE.report


def _without_generated_ids(value):
    if isinstance(value, dict):
        return {
            key: _without_generated_ids(item)
            for key, item in value.items()
            if key not in {"retrieval_id", "session_id", "continuity_id", "continue_from"}
        }
    if isinstance(value, list):
        return [_without_generated_ids(item) for item in value]
    return value


def test_replay_recommends_zero_start_for_outcome_coupled_history(tmp_path) -> None:
    db_path = tmp_path / "replay.db"
    eng = SlowaveEngine(SlowaveConfig(db_path=str(db_path), dim=8, disable_encoder=True))
    try:
        eng.record_retrieval(
            retrieval_id="rec_replay",
            response={"schemas": [{"id": "sch_1"}]},
        )
        conn = eng.db.connect()
        conn.execute(
            "INSERT INTO context_feedback_events (context_id, feedback, outcome, "
            "feedback_signal_json, outcome_reward, used_memory_ids_json, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("rec_replay", "useful", "success", json.dumps({}), 1.0, '["sch_1"]', 1),
        )
        conn.commit()
    finally:
        eng.close()
    result = report(str(db_path))
    assert result["integrity"] == "ok"
    assert result["outcome_coupled_legacy_rows"] == 1
    assert result["safe_for_historical_backfill"] is False
    assert result["recommended_migration"] == "zero_start"


def test_decision_trace_records_candidates_without_changing_feedback_exposure(tmp_path) -> None:
    eng = SlowaveEngine(
        SlowaveConfig(db_path=str(tmp_path / "trace.db"), dim=8, disable_encoder=True)
    )
    eng.schemas.create(
        content_text="alpha deployment policy",
        facets={"schema_class": "fact"},
        tags=[],
        embedding=None,
        scope_id="project:trace",
        dedupe=False,
    )
    result = ops.activate(
        eng, query="alpha deployment policy", initial_goal="inspect alpha", scope="project:trace"
    )
    conn = eng.db.connect()
    parent = conn.execute(
        "SELECT policy_role, trace_origin, trace_complete FROM retrieval_decisions "
        "WHERE retrieval_id = ? AND policy_role = 'current'",
        (result["retrieval_id"],),
    ).fetchone()
    trace_ids = {
        row["candidate_id"]
        for row in conn.execute(
            "SELECT candidate_id FROM retrieval_candidate_decisions WHERE retrieval_id = ?",
            (result["retrieval_id"],),
        )
    }
    exposed_ids = {
        row["memory_id"]
        for row in conn.execute(
            "SELECT memory_id FROM context_recall_items WHERE context_id = ? AND admitted = 1",
            (result["retrieval_id"],),
        )
    }
    assert tuple(parent) == ("current", "prospective", 1)
    assert trace_ids == {"sch_1"}
    assert exposed_ids == {"sch_1"}
    shadow = conn.execute(
        "SELECT action_intent, action_intent_reason FROM retrieval_decisions "
        "WHERE retrieval_id = ? AND policy_role = 'shadow'",
        (result["retrieval_id"],),
    ).fetchone()
    assert tuple(shadow) == ("non_action", "no_imperative_clause")


def test_legacy_trace_backfill_is_metadata_only_and_incomplete(tmp_path) -> None:
    eng = SlowaveEngine(
        SlowaveConfig(db_path=str(tmp_path / "legacy.db"), dim=8, disable_encoder=True)
    )
    eng.record_retrieval(
        retrieval_id="rec_legacy",
        retrieval_policy_version="strict-v9",
        response={"schemas": [{"id": "sch_8", "reason": "direct"}]},
    )
    conn = eng.db.connect()
    conn.execute("DELETE FROM retrieval_decisions WHERE retrieval_id = 'rec_legacy'")
    conn.commit()

    assert eng._feedback.backfill_legacy_decision_traces() == 1
    parent = conn.execute(
        "SELECT policy_role, trace_origin, trace_complete, source_policy_version, source_created_at, reconstruction_reason "
        "FROM retrieval_decisions WHERE retrieval_id = 'rec_legacy'"
    ).fetchone()
    item = conn.execute(
        "SELECT decision, reason_code FROM retrieval_candidate_decisions WHERE retrieval_id = 'rec_legacy'"
    ).fetchone()
    exposure_count = conn.execute(
        "SELECT COUNT(*) FROM context_recall_items WHERE context_id = 'rec_legacy' AND admitted = 1"
    ).fetchone()[0]

    assert tuple(parent[:4]) == ("historical", "legacy_observed", 0, "strict-v9")
    assert parent[4] is not None
    assert parent[5] == "historical_candidate_decisions_not_persisted"
    assert tuple(item) == ("selected", "direct")
    assert exposure_count == 1
    assert eng._feedback.backfill_legacy_decision_traces() == 0


def test_identical_activation_has_canonical_public_response_excluding_generated_ids(
    tmp_path,
) -> None:
    eng = SlowaveEngine(
        SlowaveConfig(db_path=str(tmp_path / "canonical.db"), dim=8, disable_encoder=True)
    )
    eng.schemas.create(
        content_text="alpha deployment policy",
        facets={"schema_class": "fact"},
        tags=[],
        embedding=None,
        scope_id="project:canonical",
        dedupe=False,
    )
    first = ops.activate(
        eng,
        query="alpha deployment policy",
        initial_goal="inspect alpha",
        scope="project:canonical",
    )
    second = ops.activate(
        eng,
        query="alpha deployment policy",
        initial_goal="inspect alpha",
        scope="project:canonical",
    )

    assert _without_generated_ids(first) == _without_generated_ids(second)


def test_shadow_interpretation_is_bounded_deterministic_and_nonoperative() -> None:
    task = "1. Export the evidence bundle\n2. Verify the hash\n3. Archive the result\n4. Notify the owner\n5. Update the ticket"
    first = _shadow_task_interpretation(task, {})
    second = _shadow_task_interpretation(task, {})

    assert first == second
    assert first["action_intent"] == "action"
    assert len(first["task_needs"]) == 4


def test_action_advice_questions_require_procedural_evidence_not_execution() -> None:
    for task in (
        "Should I deploy directly to production and skip health checks?",
        "Could we migrate the billing database?",
    ):
        interpretation = _shadow_task_interpretation(task, {})
        assert interpretation["action_intent"] == "uncertain"
        assert interpretation["action_intent_reason"] == "action_advice_question"
        assert all(need["kind"] == "mixed" for need in interpretation["task_needs"])
    assert (
        _shadow_task_interpretation("Can we see which database stores the ledger?", {})[
            "action_intent"
        ]
        == "non_action"
    )
