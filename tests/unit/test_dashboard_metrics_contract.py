"""Fixture-based checks of the October 9 dashboard measurement contract."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from slowave.core.config import SlowaveConfig
from slowave.dashboard import app
from slowave.dashboard.metrics import complete_request_sql, history, occasions, projection
from slowave.storage.sqlite_db import SQLiteConfig, SQLiteDB


@pytest.fixture
def fixture(tmp_path: Path):
    path = tmp_path / "metrics.db"
    db = SQLiteDB(SQLiteConfig(path=str(path)))
    db.init_schema(SlowaveConfig.default_schema_path())
    db.close()
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    for name, start, end, outcome, feedback in [
        ("source", 10, 20, "success", "complete"),
        ("a", 100, 190, "partial", "complete"),
        ("b", 200, 290, None, "complete"),
        ("ongoing", 300, None, None, "pending"),
        ("legacy", 50, 90, "failure", "legacy"),
    ]:
        conn.execute(
            "INSERT INTO sessions(id,agent,scope_id,started_ts,ended_ts,outcome,feedback_status,lifecycle_version) VALUES (?,'test','project:one',?,?,?,?,?)",
            (name, start, end, outcome, feedback, None if name == "legacy" else "v17"),
        )
    procedure = {
        "version": 2,
        "summary": "Check a fixture and inspect the result",
        "context": {},
        "steps": [{"summary": "Create fixture"}, {"summary": "Inspect result"}],
        "caveats": [],
    }
    conn.execute(
        "INSERT INTO raw_events(session_id,ts,type,content,metadata_json) VALUES ('source',20,'task_complete','done',?)",
        [json.dumps({"procedure": procedure})],
    )
    for i, state, stage in [
        (1, "active", 0),
        (2, "active", 1),
        (3, "active", 2),
        (4, "active", 3),
        (5, "needs_review", 0),
        (6, "stale", 1),
        (7, "archived", 2),
        (8, "active", 9),
    ]:
        conn.execute(
            "INSERT INTO schemas(id,content_text,status,scope_id,first_formed_ts,last_updated_ts,generalization_stage) VALUES (?, ?, ?, 'project:one',30,30,?)",
            (i, f"memory {i}", state, stage),
        )
    conn.execute(
        "INSERT INTO schemas(id,content_text,status,scope_id,first_formed_ts,last_updated_ts) VALUES (9,'other','active','project:two',30,30)"
    )
    for rid, task, ts, kind, size in [
        ("r1", "a", 110, "context", 100),
        ("r2", "a", 120, "recall", 200),
        ("r3", "b", 210, "context", 300),
        ("empty", "b", 220, "recall", 0),
        ("r4", "ongoing", 310, "context", None),
        ("source_r", "source", 15, "context", 200),
    ]:
        conn.execute(
            "INSERT INTO context_recall_events(context_id,retrieval_type,session_id,scope_id,created_at,count_n,response_chars,lifecycle_version) VALUES (?,?,?,'project:one',?,99,?,'v17')",
            (rid, kind, task, ts, size),
        )
    for rid, target, kind, admitted, phase in [
        ("r1", "sch_1", "schema", 1, "focus"),
        ("r1", "sch_2", "schema", 1, "continuation"),
        ("r1", "sch_3", "schema", 0, "continuation"),
        ("r1", "sch_5", "schema", 1, "focus"),
        ("r1", "sch_6", "related", 1, "focus"),
        ("r1", "proc_source", "procedure", 1, "focus"),
        ("r2", "sch_1", "related", 1, "focus"),
        ("r2", "proc_source", "procedural_memory", 1, "focus"),
        ("r3", "sch_1", "schema", 1, "focus"),
        ("r3", "sch_4", "schema", 1, "focus"),
        ("r3", "proc_source", "procedure", 1, "focus"),
        ("r4", "sch_2", "schema", 1, "focus"),
        ("source_r", "proc_source", "procedure", 1, "focus"),
    ]:
        conn.execute(
            "INSERT INTO context_recall_items(context_id,memory_id,memory_type,rank,admitted,phase,created_at) VALUES (?,?,?,1,?,?,400)",
            (rid, target, kind, admitted, phase),
        )
    # Retried page writes the same primary key, not a new occasion.
    conn.execute(
        "INSERT OR IGNORE INTO context_recall_items(context_id,memory_id,memory_type,rank,admitted,created_at) VALUES ('r1','sch_2','schema',1,1,401)"
    )
    feedback = [
        ("r1", "memory", "sch_1", "used", None, "accepted"),
        ("r1", "memory", "sch_1", "not_used", None, "accepted"),  # tie, later row wins
        ("r1", "memory", "sch_1", "used", None, "rejected"),
        ("r1", "memory", "sch_2", "irrelevant", None, "accepted"),
        ("r1", "memory", "sch_3", "used", None, "accepted"),  # not delivered
        ("r1", "memory", "sch_5", "used", None, "accepted"),  # outside active inventory
        ("r1", "memory", "sch_6", "used", None, "accepted"),
        ("r1", "procedure", "proc_source", "used", "helped", "accepted"),
        ("r2", "memory", "sch_1", "used", None, "accepted"),
        ("r2", "procedure", "proc_source", "not_used", "helped", "accepted"),  # help isn't use
        ("r3", "memory", "sch_1", "used", None, "accepted"),
        ("r3", "memory", "sch_4", "unassessable", None, "accepted"),
        ("r3", "procedure", "proc_source", "used", None, "accepted"),
        ("source_r", "procedure", "proc_source", "used", "harmed", "accepted"),
    ]
    for index, row in enumerate(feedback):
        conn.execute(
            "INSERT INTO feedback_events(event_id,retrieval_id,target_kind,target_id,assessment,effect,status,coverage,source_contract,created_at) VALUES (?,?,?,?,?,?,?,'complete','slowave_feedback:v17',500)",
            (f"f{index}", *row),
        )
    conn.commit()
    yield path, conn
    conn.close()


def test_inventory_delivery_feedback_and_effect_units(fixture):
    _, conn = fixture
    m = projection(conn, {"scope": ["project:one"], "from": ["100"], "to": ["400"]})
    assert (
        m["memory_total"],
        m["memory_exposed"],
        m["memory_assessed"],
        m["memory_used"],
        m["memory_unknown"],
    ) == (5, 3, 3, 1, 0)
    assert (
        m["procedure_total"],
        m["procedure_exposed"],
        m["procedure_assessed"],
        m["procedure_used"],
    ) == (1, 1, 1, 1)
    assert (m["retrievals_total"], m["retrievals_no_match"], m["retrievals_used"]) == (5, 1, 3)
    assert (m["deliveries"], m["feedback_assessed"], m["feedback_unknown"]) == (11, 10, 1)
    assert (m["irrelevant"], m["assessed_memory_deliveries"], m["missing_memory_feedback"]) == (
        1,
        7,
        1,
    )
    assert (m["effect_known"], m["procedure_helped"], m["effect_unknown"]) == (1, 1, 1)
    assert m["procedure_harmed"] == 0  # outside request period
    assert m["stages"] == [1, 1, 1, 1] and m["unknown_stage"] == 1
    assert m["memory_across_tasks"] == m["procedure_across_tasks"] == 1
    assert m["context_size"]["Activate"] == {"samples": 2, "median_chars": 200, "p95_chars": 300}
    assert m["context_size"]["Recall"]["median_chars"] == 100
    assert m["active_scopes"] == 1


def test_identical_home_libraries_requests_and_list_filters(fixture, monkeypatch):
    path, conn = fixture
    monkeypatch.setattr(app, "_daemon_health", lambda: {"running": False})
    monkeypatch.setattr(app, "_slowave_processes", lambda: [])
    qs = {"scope": ["project:one"], "from": ["100"], "to": ["400"]}
    shared = app._effectiveness_payload(str(path), qs)
    memories = app._schemas_payload(
        str(path), {**qs, "page": ["2"], "per_page": ["1"], "states": ["active"]}
    )
    procedures = app._procedural_memory_payload(
        str(path), {**qs, "cohort": ["all"], "retrieved": ["never"]}
    )
    requests = app._retrievals_payload(str(path), {**qs, "page": ["2"], "per_page": ["1"]})
    assert requests["metrics"] == shared
    assert memories["metrics"]["memory_total"] == shared["memory_total"]
    assert memories["metrics"]["memory_used"] == shared["memory_used"]
    assert procedures["metrics"]["procedure_total"] == shared["procedure_total"]
    assert procedures["metrics"]["procedure_used"] == shared["procedure_used"]
    assert projection(conn, {"scope": ["project:two"]})["memory_total"] == 1


def test_request_time_corrections_history_and_finalization(fixture):
    path, conn = fixture
    m = projection(conn, {"scope": ["project:one"], "from": ["110"], "to": ["110"]})
    assert m["memory_used"] == 0  # latest r1 not_used and inactive used records excluded
    assert m["memory_feedback"] == 2 and m["memory_deliveries"] == 2
    assert (
        sum(b["n"] for b in m["chart"]["channels"]["used_memories"]) == 2
    )  # historical retired deliveries retained
    assert (
        conn.execute(
            f"SELECT COUNT(*) FROM context_recall_events r WHERE r.context_id='r4' AND ({complete_request_sql()})"
        ).fetchone()[0]
        == 0
    )
    conn.execute(
        "INSERT INTO feedback_events(event_id,retrieval_id,target_kind,target_id,coverage,status,source_contract,created_at) VALUES ('declaration','r4','retrieval','r4','complete','accepted','test',600)"
    )
    assert (
        conn.execute(
            f"SELECT COUNT(*) FROM context_recall_events r WHERE r.context_id='r4' AND ({complete_request_sql()})"
        ).fetchone()[0]
        == 0
    )
    totals = history(occasions(conn), "memory", "sch_1")
    assert totals["retrieved"] == 3 and totals["used"] == 2 and totals["last_used"] == 210
    conn.commit()
    assert app._schema_detail(str(path), 1)["schema"]["use_history_totals"] == totals
    summary = app._activity_summary(str(path), "s.scope_id=?", ["project:one"])
    assert summary["closure_eligible"] == 3  # source,a,b; excludes ongoing and legacy
    assert summary["complete"] == 3
    assert summary["unknown_outcome_closed"] == 1
    assert summary["context_denominator"] == 4 and summary["context_use"] == 3


def test_unknown_procedure_use_and_empty_inventory(fixture):
    _, conn = fixture
    conn.execute(
        "UPDATE feedback_events SET assessment='unassessable',effect=NULL WHERE target_kind='procedure'"
    )
    m = projection(conn, {"scope": ["project:one"]})
    assert m["procedure_assessed"] == 0 and m["procedure_unknown"] == 1
    assert m["procedure_feedback"] == 4 and m["effect_known"] == 0
    empty = projection(conn, {"scope": ["project:absent"]})
    assert empty["stages"] == [0, 0, 0, 0] and empty["memory_total"] == empty["deliveries"] == 0
    assert empty["context_size"]["Activate"]["median_chars"] is None


def test_projection_reconciles_against_independent_sql(fixture):
    """Recompute representative card numerators without the projection helper."""
    _, conn = fixture
    metrics = projection(conn, {"scope": ["project:one"], "from": ["100"], "to": ["400"]})
    delivered = """SELECT DISTINCT context_id,memory_id,
        CASE WHEN memory_type IN ('schema','related') THEN 'memory' ELSE 'procedure' END kind
        FROM context_recall_items WHERE admitted=1 AND
        ((memory_type IN ('schema','related') AND memory_id GLOB 'sch_[0-9]*') OR
        memory_type IN ('procedure','procedural_memory'))"""
    inventory = conn.execute(
        "SELECT COUNT(*) FROM schemas WHERE status='active' AND scope_id='project:one'"
    ).fetchone()[0]
    exposed = conn.execute(f"""WITH d AS ({delivered}) SELECT COUNT(DISTINCT d.memory_id) FROM d
        JOIN context_recall_events r ON r.context_id=d.context_id
        JOIN schemas s ON d.kind='memory' AND d.memory_id='sch_'||s.id
        WHERE d.kind='memory' AND s.status='active' AND s.scope_id='project:one'
        AND r.created_at BETWEEN 100 AND 400""").fetchone()[0]
    used = conn.execute(f"""WITH d AS ({delivered}), latest AS (
          SELECT f.*,ROW_NUMBER() OVER(PARTITION BY retrieval_id,target_kind,target_id
          ORDER BY created_at DESC,rowid DESC) n FROM feedback_events f WHERE status='accepted')
        SELECT COUNT(DISTINCT d.memory_id) FROM d JOIN context_recall_events r ON r.context_id=d.context_id
        JOIN schemas s ON d.memory_id='sch_'||s.id AND s.status='active' AND s.scope_id='project:one'
        JOIN latest f ON f.retrieval_id=d.context_id AND f.target_kind=d.kind
          AND f.target_id=d.memory_id AND f.n=1 AND f.assessment='used'
        WHERE d.kind='memory' AND r.created_at BETWEEN 100 AND 400""").fetchone()[0]
    deliveries, assessed_count = conn.execute(f"""WITH d AS ({delivered}), latest AS (
          SELECT f.*,ROW_NUMBER() OVER(PARTITION BY retrieval_id,target_kind,target_id
          ORDER BY created_at DESC,rowid DESC) n FROM feedback_events f WHERE status='accepted')
        SELECT COUNT(*),SUM(CASE WHEN f.assessment IN
          ('used','not_used','irrelevant','already_known','unassessable','stale','wrong')
          OR (d.kind='procedure' AND f.assessment IN ('used','not_used','unassessable'))
          THEN 1 ELSE 0 END) FROM d JOIN context_recall_events r ON r.context_id=d.context_id
        LEFT JOIN latest f ON f.retrieval_id=d.context_id AND f.target_kind=d.kind
          AND f.target_id=d.memory_id AND f.n=1
        WHERE r.scope_id='project:one' AND r.created_at BETWEEN 100 AND 400""").fetchone()
    assert (metrics["memory_total"], metrics["memory_exposed"], metrics["memory_used"]) == (
        inventory,
        exposed,
        used,
    )
    assert (metrics["deliveries"], metrics["feedback_assessed"]) == (deliveries, assessed_count)


def test_memory_sort_uses_displayed_assessed_rate_before_pagination(fixture):
    path, conn = fixture
    # sch_3 has one used occasion and four unassessed occasions: its displayed
    # assessed-use rate is 100%, higher than sch_1's 2/3, despite less exposure.
    for index in range(5):
        conn.execute(
            "INSERT INTO context_recall_events(context_id,retrieval_type,session_id,scope_id,created_at,count_n,lifecycle_version) "
            "VALUES (?,'recall','b','project:one',?,1,'v17')",
            (f"extra_{index}", 230 + index),
        )
        conn.execute(
            "INSERT INTO context_recall_items(context_id,memory_id,memory_type,rank,admitted,created_at) "
            "VALUES (?,'sch_3','schema',1,1,500)",
            (f"extra_{index}",),
        )
    # A retried delivery must not inflate the exposure count.
    conn.execute(
        "INSERT OR IGNORE INTO context_recall_items(context_id,memory_id,memory_type,rank,admitted,created_at) "
        "VALUES ('extra_0','sch_3','related',1,1,501)"
    )
    conn.execute(
        "INSERT INTO feedback_events(event_id,retrieval_id,target_kind,target_id,assessment,status,coverage,source_contract,created_at) "
        "VALUES ('extra_use','extra_0','memory','sch_3','used','accepted','complete','test',900)"
    )
    conn.commit()
    payload = app._schemas_payload(
        str(path),
        {
            "scope": ["project:one"],
            "states": ["active"],
            "sort": ["use_rate"],
            "dir": ["desc"],
            "per_page": ["1"],
        },
    )
    first = payload["schemas"][0]
    assert first["id"] == "sch_3"
    assert (first["times_used"], first["times_assessed"], first["times_exposed"]) == (1, 1, 5)
    assert first["last_used_ts"] == 230  # request time, not feedback time


def test_projection_does_not_mutate_caller_request_bindings(fixture):
    _, conn = fixture
    args = [100, 400]
    metrics = projection(
        conn,
        {"cohort": ["v17"]},
        request_filter=(
            "r.created_at BETWEEN ? AND ?",
            args,
        ),
    )
    assert args == [100, 400]
    assert metrics["retrievals_total"] == 5


def test_numeric_memory_search_matches_inventory_and_table(fixture):
    path, conn = fixture
    conn.execute(
        "INSERT INTO schemas(id,content_text,status,scope_id,first_formed_ts,last_updated_ts) "
        "VALUES (10,'memory 1 text match','active','project:one',30,30)"
    )
    conn.commit()
    for query in ("1", "sch_1", "001"):
        payload = app._schemas_payload(str(path), {"q": [query]})
        assert [row["id"] for row in payload["schemas"]] == ["sch_1"]
        assert payload["metrics"]["memory_total"] == 1
