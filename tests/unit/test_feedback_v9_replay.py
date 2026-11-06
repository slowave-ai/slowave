from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from slowave import ops
from slowave.core import retrieval_trace
from slowave.core.config import SlowaveConfig
from slowave.core.engine import SlowaveEngine

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


def _trace_engine(tmp_path, name: str) -> SlowaveEngine:
    eng = SlowaveEngine(SlowaveConfig(db_path=str(tmp_path / name), dim=8, disable_encoder=True))
    eng.schemas.create(
        content_text="alpha deployment policy",
        facets={"schema_class": "fact"},
        tags=[],
        embedding=None,
        scope_id="project:trace",
        dedupe=False,
    )
    return eng


def _trace_tables(conn) -> set[str]:
    return {
        row["name"]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'retrieval_%trace%'"
        )
    }


def test_retrieval_trace_is_off_by_default_and_writes_nothing(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("SLOWAVE_RETRIEVAL_TRACE", raising=False)
    eng = _trace_engine(tmp_path, "off.db")
    result = ops.activate(
        eng, query="alpha deployment policy", initial_goal="inspect alpha", scope="project:trace"
    )
    conn = eng.db.connect()
    assert _trace_tables(conn) == set()
    exposed = {
        row["memory_id"]
        for row in conn.execute(
            "SELECT memory_id FROM context_recall_items WHERE context_id = ? AND admitted = 1",
            (result["retrieval_id"],),
        )
    }
    assert exposed == {"sch_1"}


def test_retrieval_trace_records_scores_when_marker_file_present(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("SLOWAVE_RETRIEVAL_TRACE", raising=False)
    eng = _trace_engine(tmp_path, "on.db")
    (tmp_path / ".retrieval_trace").touch()
    result = ops.activate(
        eng, query="alpha deployment policy", initial_goal="inspect alpha", scope="project:trace"
    )
    conn = eng.db.connect()
    run = conn.execute(
        "SELECT selected_total, gate_rejections_json FROM retrieval_trace_runs WHERE retrieval_id = ?",
        (result["retrieval_id"],),
    ).fetchone()
    candidates = conn.execute(
        "SELECT candidate_id, decision, selected_position FROM retrieval_trace_candidates "
        "WHERE retrieval_id = ?",
        (result["retrieval_id"],),
    ).fetchall()
    assert run["selected_total"] == 1
    assert [tuple(row) for row in candidates] == [("sch_1", "selected", 0)]


def test_old_decision_trace_tables_are_dropped_by_migration(tmp_path) -> None:
    db_path = tmp_path / "old.db"
    eng = SlowaveEngine(SlowaveConfig(db_path=str(db_path), dim=8, disable_encoder=True))
    conn = eng.db.connect()
    conn.execute("CREATE TABLE retrieval_decisions (retrieval_id TEXT)")
    conn.execute("CREATE TABLE retrieval_candidate_decisions (retrieval_id TEXT)")
    conn.commit()
    eng.close()

    eng = SlowaveEngine(SlowaveConfig(db_path=str(db_path), dim=8, disable_encoder=True))
    try:
        names = {
            row["name"]
            for row in eng.db.connect().execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
    finally:
        eng.close()
    assert "retrieval_decisions" not in names
    assert "retrieval_candidate_decisions" not in names


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


def test_retrieval_trace_keeps_scores_and_counts_gate_rejections(tmp_path) -> None:
    eng = _trace_engine(tmp_path, "scores.db")
    eng.record_retrieval(retrieval_id="rec_scores", response={"schemas": []})
    conn = eng.db.connect()
    retrieval_trace.record(
        conn,
        retrieval_id="rec_scores",
        policy_version="activation-pool-relative-v1",
        mode="default",
        limit=5,
        catalog_truncated=False,
        selected=[],
        filtered_items=[],
        decision_candidates=[
            {
                "candidate_kind": "memory",
                "candidate_id": "sch_2",
                "decision": "rejected",
                "reason_code": "insufficient_applicability",
                "channel_evidence": [
                    {
                        "dense_cosine": 0.41,
                        "lexical_rank": 7,
                        "rrf_score": 0.3,
                        "applicability_score": -1.5,
                        "threshold": 0.0,
                    },
                    {
                        "dense_cosine": 0.52,
                        "lexical_rank": 3,
                        "rrf_score": 0.2,
                        "applicability_score": -1.5,
                        "threshold": 0.0,
                    },
                ],
            },
            {
                "candidate_kind": "memory",
                "candidate_id": "sch_3",
                "decision": "rejected",
                "reason_code": "wrong_task_facet",
            },
            {
                "candidate_kind": "memory",
                "candidate_id": "sch_4",
                "decision": "rejected",
                "reason_code": "wrong_task_facet",
            },
        ],
    )
    run = conn.execute("SELECT * FROM retrieval_trace_runs").fetchone()
    row = conn.execute("SELECT * FROM retrieval_trace_candidates").fetchone()
    assert json.loads(run["gate_rejections_json"]) == {"wrong_task_facet": 2}
    assert run["scored_total"] == 1
    assert (row["candidate_id"], row["dense_cosine"], row["lexical_rank"]) == ("sch_2", 0.52, 3)
    assert (row["applicability"], row["threshold"]) == (-1.5, 0.0)
