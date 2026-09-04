from __future__ import annotations

import sqlite3
import time
from pathlib import Path

from slowave.core.config import SlowaveConfig
from slowave.dashboard.app import (
    _activity_payload,
    _home_payload,
    _retrieval_detail,
    _retrievals_payload,
    _schemas_payload,
)
from slowave.storage.sqlite_db import SQLiteConfig, SQLiteDB


def _database(path: Path) -> sqlite3.Connection:
    database = SQLiteDB(SQLiteConfig(path=str(path)))
    database.init_schema(SlowaveConfig.default_schema_path())
    database.close()
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    return connection


def test_retrieval_projection_preserves_exposure_and_feedback_semantics(tmp_path: Path) -> None:
    path = tmp_path / "dashboard.sqlite3"
    connection = _database(path)
    connection.execute(
        "INSERT INTO sessions (id, agent, scope_id, started_ts, ended_ts, feedback_status) "
        "VALUES ('sess_1', 'codex', 'project:test', 100, 200, 'complete')"
    )
    connection.executemany(
        "INSERT INTO context_recall_events "
        "(context_id, retrieval_type, session_id, scope_id, query, count_n, created_at) "
        "VALUES (?, ?, 'sess_1', 'project:test', ?, ?, ?)",
        [
            ("ctx_visible", "recall", "inspect retrieval", 5, 150),
            ("ctx_empty", "context", "new task", 0, 140),
            ("ctx_hook", "context", "SLOWAVE MANDATORY: lifecycle", 0, 130),
        ],
    )
    pathways = ["direct", "graph", "exploration", "context_reinstatement", "legacy_value"]
    connection.executemany(
        "INSERT INTO context_recall_items "
        "(context_id, memory_id, memory_type, rank, content_text, admitted, pathway, created_at) "
        "VALUES ('ctx_visible', ?, 'schema', ?, ?, 1, ?, 150)",
        [
            (f"sch_{index}", index, f"memory {index}", pathway)
            for index, pathway in enumerate(pathways, 1)
        ],
    )
    connection.execute(
        "INSERT INTO feedback_events "
        "(event_id, retrieval_id, session_id, scope_id, target_kind, target_id, assessment, "
        "coverage, source_contract, created_at) "
        "VALUES ('fb_1', 'ctx_visible', 'sess_1', 'project:test', 'memory', 'sch_1', "
        "'used', 'complete', 'v9', 160)"
    )
    connection.commit()
    connection.close()

    listing = _retrievals_payload(str(path), {"include_internal": ["false"]})
    assert [item["context_id"] for item in listing["retrievals"]] == [
        "ctx_visible",
        "ctx_empty",
    ]
    assert listing["summary"]["retrievals"] == 2
    assert listing["summary"]["no_match"] == 1
    assert listing["summary"]["feedback_complete"] == 1
    assert listing["summary"]["demonstrated_value"] == 1
    assert listing["summary"]["unknown"] == 4
    visible = listing["retrievals"][0]
    assert visible["signal_counts"] == {
        "used": 1,
        "not_used": 0,
        "irrelevant": 0,
        "stale": 0,
        "wrong": 0,
        "helped": 0,
        "no_effect": 0,
        "harmed": 0,
        "unknown": 0,
    }
    assert listing["retrievals"][1]["signal_counts"]["unknown"] == 1

    by_exposed = _retrievals_payload(
        str(path), {"include_internal": ["false"], "sort": ["exposed"], "dir": ["asc"]}
    )
    assert [item["context_id"] for item in by_exposed["retrievals"]] == [
        "ctx_empty",
        "ctx_visible",
    ]
    by_used = _retrievals_payload(
        str(path), {"include_internal": ["false"], "sort": ["used"], "dir": ["desc"]}
    )
    assert [item["context_id"] for item in by_used["retrievals"]] == [
        "ctx_visible",
        "ctx_empty",
    ]

    detail = _retrieval_detail(str(path), "ctx_visible")
    assert [item["pathway"] for item in detail["items"]] == pathways
    assert [item["pathway_group"] for item in detail["items"]] == [
        "Direct",
        "Associated",
        "Associated",
        "Associated",
        "Unknown pathway",
    ]
    assert detail["feedback"][0]["assessment"] == "used"


def test_retrieval_effect_sort_matches_the_displayed_effect_priority(tmp_path: Path) -> None:
    path = tmp_path / "effect_sort.sqlite3"
    connection = _database(path)
    connection.executemany(
        "INSERT INTO context_recall_events (context_id, retrieval_type, query, count_n, created_at) "
        "VALUES (?, 'recall', ?, 1, ?)",
        [
            ("ctx_unknown", "unknown effect", 100),
            ("ctx_helped", "helped effect", 101),
            ("ctx_no_effect", "no effect", 102),
            ("ctx_harmed", "harmed effect", 103),
        ],
    )
    connection.executemany(
        "INSERT INTO feedback_events (event_id, retrieval_id, target_kind, target_id, effect, coverage, source_contract, status, created_at) "
        "VALUES (?, ?, 'procedure', 'proc_1', ?, 'complete', 'slowave_feedback:v9', 'accepted', 110)",
        [
            ("fb_helped", "ctx_helped", "helped"),
            ("fb_no_effect", "ctx_no_effect", "no_effect"),
            ("fb_harmed", "ctx_harmed", "harmed"),
        ],
    )
    connection.commit()
    connection.close()

    descending = _retrievals_payload(str(path), {"sort": ["effect"], "dir": ["desc"]})
    assert [item["context_id"] for item in descending["retrievals"]] == [
        "ctx_harmed",
        "ctx_no_effect",
        "ctx_helped",
        "ctx_unknown",
    ]
    ascending = _retrievals_payload(str(path), {"sort": ["effect"], "dir": ["asc"]})
    assert [item["context_id"] for item in ascending["retrievals"]] == [
        "ctx_unknown",
        "ctx_helped",
        "ctx_no_effect",
        "ctx_harmed",
    ]


def test_memory_and_activity_lists_are_server_paginated(tmp_path: Path) -> None:
    path = tmp_path / "dashboard.sqlite3"
    connection = _database(path)
    connection.execute(
        "INSERT INTO sessions (id, agent, scope_id, started_ts, ended_ts, outcome) "
        "VALUES ('sess_1', 'codex', 'project:test', 100, 200, 'success')"
    )
    event = connection.execute(
        "INSERT INTO raw_events (session_id, ts, type, content) "
        "VALUES ('sess_1', 110, 'observation', 'evidence')"
    ).lastrowid
    for index in range(3):
        schema_id = connection.execute(
            "INSERT INTO schemas (content_text, scope_id, status, first_formed_ts, last_updated_ts) "
            "VALUES (?, 'project:test', 'active', ?, ?)",
            (f"memory {index}", 120 + index, 120 + index),
        ).lastrowid
        connection.execute(
            "INSERT INTO schema_evidence (schema_id, raw_event_id) VALUES (?, ?)",
            (schema_id, event),
        )
    connection.commit()
    connection.close()

    memories = _schemas_payload(str(path), {"states": ["active"], "per_page": ["2"]})
    assert len(memories["schemas"]) == 2
    assert memories["pagination"] == {"page": 1, "per_page": 2, "total": 3}
    assert memories["schemas"][0]["evidence_count"] == 1

    activity = _activity_payload(str(path), {"per_page": ["1"]})
    assert activity["pagination"]["total"] == 1
    assert activity["activities"][0]["memory_count"] == 3
    assert activity["summary"] == {
        "eligible": 1,
        "complete": 0,
        "incomplete": 0,
        "pending": 0,
        "closed": 1,
        "successful_closed": 1,
        "partial_failed_closed": 0,
        "unknown_outcome_closed": 0,
        "known_outcome_closed": 1,
        "closure_eligible": 0,
        "closure_unclassified": 1,
        "context_denominator": 0,
        "context_use": 0,
    }
    assert _activity_payload(str(path), {"summary_only": ["true"]})["activities"] == []
    started = time.perf_counter()
    fast_rows = _activity_payload(str(path), {"include_summary": ["false"]})
    assert time.perf_counter() - started < 1.0
    assert fast_rows["summary"] == {}
    lane_activity = _activity_payload(
        str(path), {"lane": ["raw_events"], "from": ["105"], "to": ["115"]}
    )
    assert lane_activity["pagination"]["total"] == 1
    assert (
        _activity_payload(str(path), {"lane": ["raw_events"], "from": ["111"], "to": ["115"]})[
            "pagination"
        ]["total"]
        == 0
    )


def test_home_missing_database_is_a_truthful_first_run_state(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "slowave.dashboard.app._daemon_health",
        lambda: {"running": False, "version": None, "active_sessions": 0, "engines_loaded": []},
    )
    payload = _home_payload(str(tmp_path / "missing.sqlite3"), {})
    assert payload["status"]["db_exists"] is False
    assert payload["database"]["integrity_status"] == "unknown"
    assert payload["recent_changes"] == []
    assert payload["at_a_glance"] == {}


def test_home_payload_supports_all_time_window(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "slowave.dashboard.app._daemon_health",
        lambda: {"running": False, "version": None, "active_sessions": 0, "engines_loaded": []},
    )
    path = tmp_path / "all_time.sqlite3"
    connection = _database(path)
    connection.execute(
        "INSERT INTO sessions (id, agent, started_ts) VALUES ('sess_first', 'test', 200)"
    )
    connection.execute(
        "INSERT INTO raw_events (session_id, ts, type, content) VALUES ('sess_first', 200, 'note', 'first item')"
    )
    connection.commit()
    connection.close()
    payload = _home_payload(str(path), {"hours": ["all"]})
    assert payload["window"]["from"] == 200
    assert payload["window"]["hours"] is None
