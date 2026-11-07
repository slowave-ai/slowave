"""Additive dashboard API fields from the UX review (sessions, call log, retrieval, memory, procedure, worker)."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import slowave.dashboard.app as dash
from slowave.core.config import SlowaveConfig
from slowave.storage import session_events
from slowave.storage.sqlite_db import SQLiteConfig, SQLiteDB


def _db(path: Path) -> sqlite3.Connection:
    database = SQLiteDB(SQLiteConfig(path=str(path)))
    database.init_schema(SlowaveConfig.default_schema_path())
    database.close()
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def _seed(tmp_path: Path):
    path = tmp_path / "d.db"
    c = _db(path)
    c.execute(
        "INSERT INTO sessions (id, agent, scope_id, started_ts, ended_ts, outcome, feedback_status,"
        " initial_goal, lifecycle_version) VALUES ('s1','codex','project:t',100,300,'success','complete',"
        " 'Fix the widget','v9')"
    )
    c.execute(
        "INSERT INTO sessions (id, agent, scope_id, started_ts) VALUES ('s2','claude','project:t',400)"
    )
    c.execute(
        "INSERT INTO sessions (id, agent, scope_id, started_ts, ended_ts, feedback_status, lifecycle_version)"
        " VALUES ('s3','codex','project:t',500,600,'pending','v9')"
    )
    for cid, rtype, ts in (("ctx1", "activate", 110), ("rec1", "recall", 120)):
        c.execute(
            "INSERT INTO context_recall_events (context_id, retrieval_type, session_id, scope_id, query,"
            " count_n, created_at) VALUES (?,?, 's1','project:t','widget',2,?)",
            (cid, rtype, ts),
        )
    c.execute(
        "INSERT INTO schemas (content_text, scope_id, first_formed_ts, last_updated_ts, status)"
        " VALUES ('widget cache is an LRU', 'project:t', 150, 150, 'active')"
    )
    c.execute(
        "INSERT INTO schemas (content_text, scope_id, first_formed_ts, last_updated_ts, status)"
        " VALUES ('other memory', 'project:t', 151, 151, 'stale')"
    )
    # the same memory delivered by both retrievals + a procedure
    for cid in ("ctx1", "rec1"):
        c.execute(
            "INSERT INTO context_recall_items (context_id, memory_id, memory_type, rank, content_text,"
            " admitted, created_at) VALUES (?, 'sch_1', 'schema', 1, 'widget cache is an LRU', 1, 110)",
            (cid,),
        )
    c.execute(
        "INSERT INTO context_recall_items (context_id, memory_id, memory_type, rank, admitted, created_at)"
        " VALUES ('ctx1', 'proc_s1', 'procedure', 2, 1, 110)"
    )
    c.execute(
        "INSERT INTO raw_events (session_id, ts, type, content) VALUES ('s1', 130, 'remember:fact', 'widget cache is an LRU')"
    )
    rid = c.execute("SELECT MAX(id) FROM raw_events").fetchone()[0]
    c.execute(
        "INSERT INTO schema_evidence (schema_id, episode_id, raw_event_id, weight, session_id)"
        " VALUES (1, NULL, ?, 1, 's1')",
        (rid,),
    )
    fb = (
        "INSERT INTO feedback_events (event_id, retrieval_id, session_id, scope_id, target_kind, target_id,"
        " assessment, effect, coverage, status, source_contract, created_at)"
        " VALUES (?,?, 's1','project:t',?,?,?,?, 'complete','accepted','v9',?)"
    )
    c.execute(fb, ("f1", "ctx1", "memory", "sch_1", "used", None, 200))
    c.execute(fb, ("f2", "ctx1", "procedure", "proc_s1", "used", "helped", 200))
    c.execute(fb, ("f3", "ctx1", "retrieval", "ctx1", None, None, 200))
    c.execute(fb, ("f4", "rec1", "memory", "sch_1", "used", None, 201))
    # live call log: envelope row, two same-second feedback rows, duplicate commits
    ev = session_events.record
    ev(
        c,
        session_id="s1",
        kind="activate",
        retrieval_id="ctx1",
        ts=110,
        duration_ms=5,
        response_chars=40,
        estimated_tokens=10,
    )
    ev(c, session_id="s1", kind="recall", retrieval_id="rec1", ts=120)
    ev(c, session_id="s1", kind="remember", raw_event_id=rid, ts=130)
    ev(c, session_id="s1", kind="remember", ts=131)  # no memory yet
    c.execute(
        "INSERT INTO context_recall_events (context_id, retrieval_type, session_id, scope_id, created_at)"
        " VALUES ('fbk_x','feedback','s1','project:t',200)"
    )
    ev(c, session_id="s1", kind="feedback", retrieval_id="fbk_x", ts=200)  # envelope
    ev(c, session_id="s1", kind="feedback", retrieval_id="ctx1", ts=200)
    ev(c, session_id="s1", kind="feedback", retrieval_id="rec1", ts=200)
    c.execute(
        "INSERT INTO context_recall_events (context_id, retrieval_type, session_id, scope_id, created_at)"
        " VALUES ('cmt_x','commit','s1','project:t',300)"
    )
    ev(c, session_id="s1", kind="commit", retrieval_id="cmt_x", ts=300)
    ev(c, session_id="s1", kind="commit", ts=300)  # rowless duplicate
    c.commit()
    return str(path), c


def test_worker_runs_sort_decayed_alias(tmp_path):
    path = tmp_path / "w.db"
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE worker_runs (id INTEGER PRIMARY KEY, started_ts INTEGER NOT NULL, ended_ts INTEGER,"
        " duration_ms INTEGER, triggered_by TEXT NOT NULL DEFAULT 'x', prototypes_processed INTEGER NOT NULL DEFAULT 0,"
        " episodes_processed INTEGER NOT NULL DEFAULT 0, schemas_created INTEGER NOT NULL DEFAULT 0,"
        " schemas_reinforced INTEGER NOT NULL DEFAULT 0, schemas_skipped INTEGER NOT NULL DEFAULT 0,"
        " schemas_decayed INTEGER NOT NULL DEFAULT 0, error_text TEXT)"
    )
    for i, d in enumerate((3, 9, 1), start=1):
        conn.execute(
            "INSERT INTO worker_runs (id, started_ts, schemas_decayed) VALUES (?,?,?)", (i, i, d)
        )
    conn.commit()
    conn.close()
    for key in ("decayed", "retired"):
        out = dash._worker_runs_payload(str(path), {"sort": [key], "dir": ["desc"]})
        assert out["sort"] == key
        assert [r["schemas_decayed"] for r in out["runs"]] == [9, 3, 1]


def test_call_log_one_row_per_call_and_real_assessments(tmp_path):
    path, _ = _seed(tmp_path)
    log = dash._session_events_payload(path, "s1", {})
    kinds = [e["kind"] for e in log["events"]]
    assert kinds == ["activate", "recall", "remember", "remember", "feedback", "commit"]
    assert [e["ordinal"] for e in log["events"]] == [1, 2, 3, 4, 5, 6]
    feedback = log["events"][4]
    assert feedback["retrieval_id"] == "ctx1" and len(feedback["merged_event_ids"]) == 2
    assert feedback["assessed_count"] == 3 and feedback["feedback_count"] == 3
    assert feedback["helped_count"] == 1 and feedback["effects"] == {"helped": 1}
    assert [a["target_ordinal"] for a in feedback["assessments"]] == [1, 1, 2]
    assert feedback["target_ordinal"] == 1
    assert all(isinstance(a["target_ordinal"], int) for a in feedback["assessments"])
    assert not any(str(e.get("retrieval_id") or "").startswith("fbk_") for e in log["events"])
    assert log["kind_counts"]["commit"] == 1 and log["total"] == 6
    assert log["approximate"] is False
    first = log["events"][0]
    assert (
        first["duration_ms"] == 5
        and first["estimated_tokens"] == 10
        and first["response_chars"] == 40
    )
    assert (
        first["status"] == "ok" and first["error_code"] is None and first["reconstructed"] is False
    )


def test_call_log_remember_saved_memories_and_pending(tmp_path):
    path, _ = _seed(tmp_path)
    ev = dash._session_events_payload(path, "s1", {"kinds": ["remember"]})["events"]
    assert ev[0]["saved_memories"] == [
        {"schema_id": 1, "preview": "widget cache is an LRU", "status": "active"}
    ]
    assert ev[0]["pending_consolidation"] is False
    assert ev[1]["saved_memories"] == [] and ev[1]["pending_consolidation"] is True


def test_call_log_marks_backfilled_rows_reconstructed(tmp_path):
    path, c = _seed(tmp_path)
    c.execute(
        "INSERT INTO sessions (id, agent, scope_id, started_ts) VALUES ('old','codex','project:t',1)"
    )
    c.execute(
        "INSERT INTO session_events (session_id, ts, kind, origin) VALUES ('old', 2, 'activate', 'backfill')"
    )
    c.commit()
    log = dash._session_events_payload(path, "old", {})
    assert log["approximate"] is True and log["events"][0]["reconstructed"] is True
    assert dash._activity_detail(path, "s1")["call_log_approximate"] is False


def test_sessions_list_fields_status_and_filters(tmp_path):
    path, c = _seed(tmp_path)
    c.execute("UPDATE sessions SET ended_ts=700, outcome=NULL WHERE id='s3'")
    session_events.record(c, session_id="s3", kind="session_reaped", ts=700)
    c.commit()
    out = dash._activity_payload(path, {})
    rows = {r["id"]: r for r in out["activities"]}
    assert rows["s1"]["status"] == "ended" and rows["s2"]["status"] == "running"
    assert rows["s3"]["status"] == "idle_closed"
    assert rows["s1"]["saved_count"] == 1
    # memory delivered twice + procedure once -> DISTINCT
    assert rows["s1"]["returned_count"] == 2
    assert (
        rows["s1"]["retrieved_memory_count"] == 1 and rows["s1"]["retrieved_procedure_count"] == 1
    )
    assert rows["s1"]["used_count"] == 2 and rows["s1"]["assessed_count"] == 2
    assert rows["s1"]["goal"] == "Fix the widget" and rows["s2"]["goal"] == ""
    assert (
        rows["s1"]["feedback_state"] == "complete" and rows["s1"]["feedback_status"] == "complete"
    )
    assert rows["s2"]["feedback_state"] == "pending_open"
    assert rows["s3"]["feedback_state"] == "none"
    # summary tiles share the predicate
    assert out["summary"]["complete"] == sum(
        r["feedback_state"] == "complete" for r in rows.values() if r["lifecycle_version"] == "v9"
    )
    assert out["summary"]["feedback_states"] == {
        "complete": 1,
        "partial": 0,
        "none": 1,
        "pending_open": 1,
    }
    unreported = dash._activity_payload(path, {"outcome": ["unreported"]})
    assert {r["id"] for r in unreported["activities"]} == {"s2", "s3"}
    assert [r["id"] for r in dash._activity_payload(path, {"agent": ["claude"]})["activities"]] == [
        "s2"
    ]
    # aggregate sort path gives identical counts
    sorted_rows = dash._activity_payload(path, {"sort": ["retrieved_memories"]})["activities"]
    assert {r["id"]: r["retrieved_memory_count"] for r in sorted_rows}["s1"] == 1


def test_session_detail_retrieved_and_saved_items(tmp_path):
    path, _ = _seed(tmp_path)
    d = dash._activity_detail(path, "s1")
    items = {i["id"]: i for i in d["retrieved_items"]}
    assert set(items) == {"sch_1", "proc_s1"} and d["retrieved_items_total"] == 2
    assert items["sch_1"]["kind"] == "memory" and items["sch_1"]["title"].startswith("widget cache")
    assert items["sch_1"]["assessment"] == "used" and items["sch_1"]["retrieval_id"] == "rec1"
    assert items["proc_s1"]["title"] == "Fix the widget" and items["proc_s1"]["effect"] == "helped"
    assert d["saved_items"][0]["schema_id"] == 1 and d["saved_items_total"] == 1
    assert d["session"]["status"] == "ended" and d["session"]["saved_count"] == 1
    assert d["session"]["feedback_state"] == "complete" and d["session"]["goal"] == "Fix the widget"


def test_retrieval_list_and_detail_fields(tmp_path):
    path, c = _seed(tmp_path)
    c.execute(
        "INSERT INTO context_recall_events (context_id, retrieval_type, session_id, scope_id, query, count_n, created_at)"
        " VALUES ('empty','context','s1','project:t','nothing',0,105)"
    )
    c.commit()
    rows = {r["context_id"]: r for r in dash._retrievals_payload(path, {})["retrievals"]}
    assert rows["ctx1"]["returned_count"] == 2 and rows["ctx1"]["returned_memory_count"] == 1
    assert (
        rows["ctx1"]["returned_procedure_count"] == 1
        and rows["ctx1"]["session_goal"] == "Fix the widget"
    )
    assert rows["ctx1"]["type_label"] == "Activation" and rows["rec1"]["type_label"] == "Recall"
    ids = lambda qs: {
        r["context_id"] for r in dash._retrievals_payload(path, qs)["retrievals"]
    }  # noqa: E731
    assert ids({"returned": ["none"]}) == {"empty"}
    assert ids({"returned": ["any"]}) == {"ctx1", "rec1"}
    assert ids({"feedback": ["none"]}) == {"empty"}
    assert ids({"feedback": ["complete"]}) == {"ctx1", "rec1"}
    c.execute("DELETE FROM feedback_events WHERE event_id='f2'")  # proc unassessed -> partial
    c.commit()
    assert ids({"feedback": ["partial"]}) == {"ctx1"}
    assert ids({"feedback": ["incomplete"]}) == {"ctx1", "empty"}

    d = dash._retrieval_detail(path, "ctx1")
    assert d["session"]["goal"] == "Fix the widget" and d["session"]["status"] == "ended"
    assert d["retrieval"]["type_label"] == "Activation"
    titles = {i["memory_id"]: i["title"] for i in d["items"]}
    assert titles["proc_s1"] == "Fix the widget" and titles["sch_1"].startswith("widget cache")
    assert [i["rank"] for i in d["items"]] == [1, 2]
    assert all(f["target_kind"] != "retrieval" for f in d["feedback"])
    assert len(d["audit"]["retrieval_observations"]) == 1
    assert d["retrieval"]["assessed_count"] == 1  # procedure feedback was deleted above
    assert {f["target_id"]: f["target_title"] for f in d["feedback"]}["sch_1"].startswith(
        "widget cache"
    )


def test_recall_page_label_and_parent(tmp_path):
    path, c = _seed(tmp_path)
    c.execute(
        "INSERT INTO context_recall_events (context_id, retrieval_type, session_id, scope_id, created_at)"
        " VALUES ('page_1','recall_page','s1','project:t',150)"
    )
    c.execute(
        "INSERT INTO retrieval_continuations (cursor_id, retrieval_id, session_id, scope_id, candidates_json,"
        " offset_n, page_size, created_at) VALUES ('cur','ctx1','s1','project:t','{}',0,5,110)"
    )
    c.commit()
    r = dash._retrieval_detail(path, "page_1")["retrieval"]
    assert r["type_label"] == "Recall page" and r["parent_retrieval_id"] == "ctx1"


def test_memory_detail_and_list_fields(tmp_path):
    path, c = _seed(tmp_path)
    c.execute(
        "INSERT INTO schema_relations (src_schema_id, dst_schema_id, relation, created_ts) VALUES (1,2,'relates_to',1)"
    )
    c.execute(
        "INSERT INTO schema_relations (src_schema_id, dst_schema_id, relation, created_ts) VALUES (2,1,'relates_to',1)"
    )
    c.execute(
        "INSERT INTO schema_evidence (schema_id, episode_id, raw_event_id, weight, session_id) VALUES (1,NULL,NULL,1,'s2')"
    )
    c.execute(
        "INSERT INTO feedback_events (event_id, retrieval_id, session_id, scope_id, target_kind, target_id, assessment,"
        " stale_reason, replacement_target_id, reason, coverage, status, source_contract, created_at)"
        " VALUES ('f9','rec1','s1','project:t','memory','sch_1','stale','superseded','sch_2','replaced','partial','accepted','v9',250)"
    )
    c.commit()
    d = dash._schema_detail(path, 1)
    assert d["evidence_total"] == 2 and len(d["evidence"]) == 2
    by_session = {e["session_id"]: e for e in d["evidence"]}
    assert (
        by_session["s1"]["has_text"] is True
        and by_session["s1"]["session_goal"] == "Fix the widget"
    )
    assert by_session["s2"]["has_text"] is False
    assert len(d["connections"]) == 1
    conn = d["connections"][0]
    assert (
        conn["direction"] == "both"
        and conn["status"] == "stale"
        and conn["preview"] == "other memory"
    )
    assert d["replacement_memory_id"] == "sch_2"
    assert d["stale_reason"] == "superseded"
    assert d["feedback_reasons"][0]["reason"] == "replaced"
    row = dash._schemas_payload(path, {"states": ["active"]})
    assert row["schemas"][0]["created_ts"] == 150
    assert row["schemas"][0]["times_exposed"] == 2
    # status_counts ignore the status filter itself but respect others
    assert row["status_counts"] == {"active": 1, "stale": 1}
    scoped = dash._schemas_payload(path, {"q": ["other"]})
    assert scoped["status_counts"] == {"stale": 1}


def _procedure_db(tmp_path):
    path, c = _seed(tmp_path)
    meta = {
        "procedure": {
            "version": 2,
            "summary": json.dumps({"summary": "Do the thing"}),
            "context": {"language": "python"},
            "steps": [{"summary": "step one"}, {"summary": "step two"}],
        }
    }
    c.execute(
        "INSERT INTO raw_events (session_id, ts, type, content, metadata_json) VALUES ('s1', 290, 'task_complete', 'done', ?)",
        (json.dumps(meta),),
    )
    c.commit()
    return path


def test_procedure_list_and_detail_fields(tmp_path):
    path = _procedure_db(tmp_path)
    out = dash._procedural_memory_payload(path, {"cohort": ["all"]})
    proc = out["procedures"][0]
    assert proc["source_goal"] == "Fix the widget" and proc["step_count"] == 2
    assert proc["helped_count"] == 1 and proc["no_effect_count"] == 0 and proc["harmed_count"] == 0
    assert proc["assessed_count"] == 1 and "outcome" in proc and "verification_status" in proc
    assert proc["summary"] == "Do the thing"  # never a raw JSON string
    assert isinstance(proc["context"], dict) and proc["steps"][0]["summary"] == "step one"
    detail = dash._procedure_detail(path, "proc_s1")["procedure"]
    assert (
        detail["source_session"]["id"] == "s1"
        and detail["source_session"]["goal"] == "Fix the widget"
    )
    assert detail["summary"] == "Do the thing"
