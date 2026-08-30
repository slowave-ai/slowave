"""Dashboard exposes client source time separately from recorded time."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from slowave.dashboard.app import _episodes_payload, _schema_detail, _session_timeline


def _db(path: Path) -> None:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.executescript("""
        CREATE TABLE sessions (
          id TEXT PRIMARY KEY, agent TEXT, scope_id TEXT, scope_kind TEXT,
          started_ts INTEGER, ended_ts INTEGER, goal TEXT, initial_goal TEXT,
          final_goal TEXT, outcome TEXT, outcome_summary TEXT,
          verification_json TEXT, feedback_status TEXT,
          retrieval_context_json TEXT, task_context_json TEXT, continuity_id TEXT,
          lifecycle_version TEXT
        );
        CREATE TABLE raw_events (
          id INTEGER PRIMARY KEY, session_id TEXT, ts INTEGER, type TEXT,
          content TEXT, metadata_json TEXT
        );
        CREATE TABLE episodic_memories (
          id INTEGER PRIMARY KEY, event_id INTEGER, ts INTEGER, salience REAL,
          recalled_count INTEGER, metadata_json TEXT
        );
        CREATE TABLE schemas (
          id INTEGER PRIMARY KEY, content_text TEXT, scope_id TEXT, status TEXT,
          confidence REAL, salience REAL, is_labile INTEGER, facets_json TEXT,
          tags_json TEXT, supporting_episode_ids TEXT,
          first_formed_ts INTEGER, last_updated_ts INTEGER, generalization_stage INTEGER
        );
        CREATE TABLE schema_prototype_map (schema_id INTEGER, prototype_id INTEGER);
        CREATE TABLE schema_relations (
          src_schema_id INTEGER, dst_schema_id INTEGER, relation TEXT,
          weight REAL, created_ts INTEGER
        );
        CREATE TABLE schema_evidence (
          id INTEGER PRIMARY KEY, schema_id INTEGER, raw_event_id INTEGER,
          episode_id INTEGER, weight REAL, quote TEXT
        );
        INSERT INTO sessions VALUES
          ('sess-1','agent','project:test','project',100,NULL,'goal','goal',NULL,NULL,NULL,'{}','pending','{}','{}',NULL,'v9');
        INSERT INTO raw_events VALUES
          (1,'sess-1',200,'remember','A dated fact', '{"occurred_at": 50}');
        INSERT INTO episodic_memories VALUES
          (1,1,200,0.8,0,'{"text":"A dated fact","occurred_at":50}');
        INSERT INTO schemas VALUES
              (1,'A schema','project:test','active',0.8,0.8,0,'{}','[]','[]',100,200,0);
        INSERT INTO schema_evidence VALUES (1,1,1,1,0.9,'');
        """)
    conn.commit()
    conn.close()


def test_timeline_and_episode_payload_distinguish_source_and_recorded_time(tmp_path: Path) -> None:
    path = tmp_path / "dashboard.db"
    _db(path)

    timeline = _session_timeline(str(path), "sess-1")
    assert timeline["events"][0]["ts"] == 200
    assert timeline["events"][0]["occurred_at"] == 50
    assert timeline["episodes"][0]["recorded_at"] == 200
    assert timeline["episodes"][0]["occurred_at"] == 50

    episodes = _episodes_payload(str(path), {})["episodes"]
    assert episodes[0]["recorded_at"] == 200
    assert episodes[0]["occurred_at"] == 50


def test_schema_evidence_exposes_source_time(tmp_path: Path) -> None:
    path = tmp_path / "dashboard.db"
    _db(path)
    evidence = _schema_detail(str(path), 1)["evidence"]
    assert evidence[0]["event_ts"] == 200
    assert evidence[0]["event_occurred_at"] == 50
