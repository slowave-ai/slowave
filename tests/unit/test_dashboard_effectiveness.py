"""Coverage for the Home memory-effectiveness surface and its data contracts.

These tests pin the cohort-correct, numerator/denominator behaviour that the
beta launch gate depends on: metrics default to the feedback-enforced (v9+)
population, support a selected scope, and never conflate exposure with use.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from slowave.core.config import SlowaveConfig
from slowave.dashboard.app import (
    _effectiveness_payload,
    _retrieval_detail,
    _retrievals_payload,
    _schemas_payload,
    _scopes_payload,
)
from slowave.storage.sqlite_db import SQLiteConfig, SQLiteDB


def _database(path: Path) -> sqlite3.Connection:
    database = SQLiteDB(SQLiteConfig(path=str(path)))
    database.init_schema(SlowaveConfig.default_schema_path())
    database.close()
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    return connection


def _seed(path: Path) -> sqlite3.Connection:
    connection = _database(path)
    connection.execute(
        "INSERT INTO sessions (id, agent, scope_id, started_ts, ended_ts, outcome, "
        "feedback_status, lifecycle_version) "
        "VALUES ('sess_1', 'codex', 'project:demo', 100, 200, 'success', 'complete', 'v10')"
    )
    connection.executemany(
        "INSERT INTO context_recall_events "
        "(context_id, retrieval_type, session_id, scope_id, query, count_n, created_at, "
        "lifecycle_version) VALUES (?, 'context', 'sess_1', 'project:demo', ?, ?, ?, ?)",
        [
            ("ctx_visible", "inspect retrieval", 3, 150, "v10"),
            ("ctx_empty", "no results here", 0, 140, "v10"),
            ("ctx_legacy", "pre-v9 record", 2, 100, None),
        ],
    )
    connection.executemany(
        "INSERT INTO context_recall_items "
        "(context_id, memory_id, memory_type, rank, content_text, admitted, pathway, "
        "created_at) VALUES ('ctx_visible', ?, 'schema', ?, ?, 1, 'direct', 150)",
        [(f"sch_{i}", i, f"memory {i}") for i in range(1, 4)],
    )
    connection.execute(
        "INSERT INTO context_recall_items "
        "(context_id, memory_id, memory_type, rank, content_text, admitted, pathway, "
        "created_at) VALUES ('ctx_visible', 'proc_1', 'procedural_memory', 4, "
        "'a procedure', 1, 'direct', 150)"
    )
    connection.executemany(
        "INSERT INTO feedback_events "
        "(event_id, retrieval_id, target_kind, target_id, assessment, effect, coverage, "
        "source_contract, created_at) VALUES (?, 'ctx_visible', ?, ?, ?, ?, 'complete', "
        "'slowave_feedback:v9', 160)",
        [
            ("fb_used", "memory", "sch_1", "used", None),
            ("fb_irrelevant", "memory", "sch_2", "irrelevant", None),
            ("fb_proc", "procedure", "proc_1", "used", "helped"),
        ],
    )
    for i in range(1, 4):
        connection.execute(
            "INSERT INTO schemas (content_text, scope_id, status, first_formed_ts, "
            "last_updated_ts) VALUES (?, 'project:demo', 'active', 120, 120)",
            (f"memory {i}",),
        )
    connection.commit()
    return connection


def test_effectiveness_defaults_to_v9_cohort_with_numerators_and_denominators(
    tmp_path: Path,
) -> None:
    path = tmp_path / "eff.sqlite3"
    connection = _seed(path)
    connection.close()

    payload = _effectiveness_payload(str(path), {})

    assert payload["cohort"] == "v9"
    assert payload["annotation"] == "Since lifecycle v9 · August 17"
    assert payload["memory_exposed"] == 3
    assert payload["memory_total"] == 3
    assert payload["memory_assessed"] == 2
    assert payload["memory_used"] == 1
    assert payload["memory_irrelevant"] == 1
    assert payload["procedure_exposed"] == 1
    assert payload["procedure_used"] == 1
    assert payload["procedure_helped"] == 1
    assert payload["retrievals_total"] == 2  # legacy excluded
    assert payload["retrievals_no_match"] == 1
    assert payload["retrievals_feedback_complete"] == 1
    assert payload["available_scopes"] == ["project:demo"]


def test_used_never_exceeds_exposed_when_a_memory_is_used_across_retrievals(
    tmp_path: Path,
) -> None:
    """A memory used in several retrievals must still count once per bucket."""
    path = tmp_path / "eff_reuse.sqlite3"
    connection = _database(path)
    connection.execute(
        "INSERT INTO sessions (id, agent, scope_id, started_ts, ended_ts, "
        "lifecycle_version) VALUES ('sess_1', 'codex', 'project:demo', 100, 200, 'v10')"
    )
    connection.executemany(
        "INSERT INTO context_recall_events (context_id, retrieval_type, session_id, "
        "scope_id, query, count_n, created_at, lifecycle_version) "
        "VALUES (?, 'context', 'sess_1', 'project:demo', 'task', 1, ?, 'v10')",
        [("ctx_a", 150), ("ctx_b", 160)],
    )
    connection.executemany(
        "INSERT INTO context_recall_items (context_id, memory_id, memory_type, rank, "
        "content_text, admitted, pathway, created_at) "
        "VALUES (?, 'sch_1', 'schema', 1, 'memory', 1, 'direct', 150)",
        [("ctx_a",), ("ctx_b",)],
    )
    connection.execute(
        "INSERT INTO context_recall_events (context_id, retrieval_type, session_id, "
        "scope_id, query, count_n, created_at, lifecycle_version) "
        "VALUES ('ctx_related', 'context', 'sess_1', 'project:demo', 'task', 1, 170, 'v10')"
    )
    connection.execute(
        "INSERT INTO context_recall_items (context_id, memory_id, memory_type, rank, "
        "content_text, admitted, pathway, created_at) "
        "VALUES ('ctx_related', 'sch_1', 'related', 1, 'memory', 1, 'associated', 170)"
    )
    connection.executemany(
        "INSERT INTO feedback_events (event_id, retrieval_id, target_kind, target_id, "
        "assessment, coverage, source_contract, created_at) "
        "VALUES (?, ?, 'memory', 'sch_1', 'used', 'complete', 'slowave_feedback:v9', 170)",
        [("fb_a", "ctx_a"), ("fb_b", "ctx_b")],
    )
    connection.execute("UPDATE feedback_events SET effect = 'harmed' WHERE event_id = 'fb_b'")
    connection.commit()
    connection.close()

    payload = _effectiveness_payload(str(path), {})
    assert payload["memory_exposed"] == 1
    assert payload["memory_used"] == 1
    assert payload["memory_used"] <= payload["memory_exposed"]
    listing = _retrievals_payload(str(path), {})
    assert [item["signal_counts"]["used"] for item in listing["retrievals"]] == [0, 1, 1]


def test_effectiveness_respects_selected_retrieval_window(tmp_path: Path) -> None:
    path = tmp_path / "eff_window.sqlite3"
    connection = _seed(path)
    connection.close()

    payload = _effectiveness_payload(str(path), {"from": ["145"], "to": ["155"]})

    assert payload["retrievals_total"] == 1
    assert payload["memory_exposed"] == 3
    assert payload["procedure_exposed"] == 1


def test_effectiveness_all_cohort_includes_legacy(tmp_path: Path) -> None:
    path = tmp_path / "eff_all.sqlite3"
    connection = _seed(path)
    connection.close()

    payload = _effectiveness_payload(str(path), {"cohort": ["all"]})
    assert payload["cohort"] == "all"
    assert payload["retrievals_total"] == 3


def test_effectiveness_scope_filter(tmp_path: Path) -> None:
    path = tmp_path / "eff_scope.sqlite3"
    connection = _seed(path)
    connection.close()

    other = _effectiveness_payload(str(path), {"scope": ["project:other"]})
    assert other["retrievals_total"] == 0
    assert other["memory_exposed"] == 0
    assert other["memory_used"] == 0


def test_scopes_payload_lists_distinct_scopes(tmp_path: Path) -> None:
    path = tmp_path / "scopes.sqlite3"
    connection = _seed(path)
    connection.close()
    assert _scopes_payload(str(path))["scopes"] == ["project:demo"]


def test_schemas_payload_reports_exposure_and_usage_columns(tmp_path: Path) -> None:
    path = tmp_path / "schemas_cols.sqlite3"
    connection = _seed(path)
    connection.close()

    memories = _schemas_payload(str(path), {"states": ["active"]})
    assert memories["summary"] == {
        "active": 3,
        "needs_review": 0,
        "stale": 0,
        "retrieved_active": 3,
        "used_active": 1,
    }
    by_content = {m["content"]: m for m in memories["schemas"]}
    used = by_content["memory 1"]
    irrelevant = by_content["memory 2"]
    untouched = by_content["memory 3"]
    assert used["times_exposed"] == 1
    assert used["times_used"] == 1
    assert used["times_irrelevant"] == 0
    assert used["last_used_ts"] == 160
    assert irrelevant["times_irrelevant"] == 1
    assert irrelevant["times_used"] == 0
    assert untouched["times_exposed"] == 1
    assert untouched["times_used"] == 0


def test_retrieval_detail_attaches_assessment_and_effect_per_item(tmp_path: Path) -> None:
    path = tmp_path / "detail.sqlite3"
    connection = _seed(path)
    connection.close()

    detail = _retrieval_detail(str(path), "ctx_visible")
    by_id = {item["memory_id"]: item for item in detail["items"]}
    assert by_id["sch_1"]["assessment"] == "used"
    assert by_id["sch_2"]["assessment"] == "irrelevant"
    assert by_id["sch_3"]["assessment"] is None
    assert by_id["proc_1"]["assessment"] == "used"
    assert by_id["proc_1"]["effect"] == "helped"
    assert detail["session"]["outcome"] == "success"


def test_retrieval_detail_resolves_current_schema_content_and_state(tmp_path: Path) -> None:
    path = tmp_path / "detail_current_schema.sqlite3"
    connection = _seed(path)
    connection.execute(
        "UPDATE schemas SET content_text = ?, status = ? WHERE id = 1",
        ("Current schema text", "needs_review"),
    )
    connection.commit()
    connection.close()

    detail = _retrieval_detail(str(path), "ctx_visible")
    item = next(item for item in detail["items"] if item["memory_id"] == "sch_1")

    assert item["content_text"] == "Current schema text"
    assert item["status"] == "needs_review"


def test_dashboard_feedback_uses_latest_accepted_per_target(tmp_path: Path) -> None:
    path = tmp_path / "latest_feedback.sqlite3"
    connection = _seed(path)

    def feedback(
        event_id: str,
        kind: str,
        target: str,
        assessment: str,
        effect: str | None,
        status: str = "accepted",
    ) -> None:
        # All timestamps tie: insertion order must break the tie consistently.
        connection.execute(
            "INSERT INTO feedback_events (event_id, retrieval_id, target_kind, target_id, "
            "assessment, effect, status, coverage, source_contract, created_at) "
            "VALUES (?, 'ctx_visible', ?, ?, ?, ?, ?, 'complete', 'slowave_feedback:v9', 160)",
            (event_id, kind, target, assessment, effect, status),
        )

    for index in range(5):
        feedback(f"duplicate_memory_{index}", "memory", "sch_1", "used", "helped")
        feedback(f"duplicate_procedure_{index}", "procedure", "proc_1", "used", "harmed")
    connection.commit()
    listing = _retrievals_payload(str(path), {"sort": ["used"], "dir": ["desc"]})
    visible = listing["retrievals"][0]
    assert visible["signal_counts"]["used"] == visible["signal_used"] == 2
    assert visible["effect_rank"] == 3
    assert len(visible["feedback"]) == 13  # The audit history remains intact.

    feedback("refine_memory", "memory", "sch_1", "irrelevant", None)
    feedback("refine_procedure", "procedure", "proc_1", "not_used", "no_effect")
    feedback("rejected_memory", "memory", "sch_1", "used", "helped", "rejected")
    feedback("rejected_procedure", "procedure", "proc_1", "used", "harmed", "rejected")
    connection.commit()
    connection.close()

    listing = _retrievals_payload(str(path), {})
    visible = next(item for item in listing["retrievals"] if item["context_id"] == "ctx_visible")
    assert visible["signal_counts"]["used"] == visible["signal_used"] == 0
    assert visible["signal_counts"]["irrelevant"] == visible["signal_irrelevant"] == 2
    assert visible["signal_counts"]["not_used"] == 1
    assert visible["signal_counts"]["harmed"] == 0
    assert visible["effect_rank"] == 2
    assert listing["summary"]["demonstrated_value"] == 0
    assert listing["summary"]["feedback_complete"] == 1

    effectiveness = _effectiveness_payload(str(path), {})
    assert effectiveness["memory_used"] == effectiveness["procedure_used"] == 0
    assert effectiveness["memory_irrelevant"] == 2
    assert effectiveness["procedure_not_used"] == effectiveness["procedure_no_effect"] == 1
    assert effectiveness["procedure_harmed"] == 0
    memories = _schemas_payload(str(path), {"states": ["active"]})
    memory = next(item for item in memories["schemas"] if item["content"] == "memory 1")
    assert memory["times_used"] == 0
    assert memory["times_irrelevant"] == 1
    assert memory["last_used_ts"] is None
    assert memories["summary"]["used_active"] == 0
    detail = _retrieval_detail(str(path), "ctx_visible")
    by_id = {item["memory_id"]: item for item in detail["items"]}
    assert by_id["sch_1"]["assessment"] == "irrelevant"
    assert by_id["proc_1"]["assessment"] == "not_used"
    assert by_id["proc_1"]["effect"] == "no_effect"
