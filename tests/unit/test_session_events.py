"""session_events: live record, idempotent backfill, ordering, reaper, evidence session_id."""

import time

from slowave.core.config import SlowaveConfig
from slowave.core.engine import SlowaveEngine
from slowave.mcp.session_reaper import _reap_once
from slowave.storage import session_events


def _eng(tmp_path):
    return SlowaveEngine(cfg=SlowaveConfig(db_path=str(tmp_path / "t.db"), disable_encoder=True))


def _session(conn, sid, started=1000, ended=None):
    conn.execute(
        "INSERT INTO sessions (id, agent, scope_id, scope_kind, started_ts, ended_ts)"
        " VALUES (?, 't', 'project:t', 'project', ?, ?)",
        (sid, started, ended),
    )


def _ledger(conn, cid, sid, rtype, ts):
    conn.execute(
        "INSERT INTO context_recall_events (context_id, session_id, scope_id, retrieval_type,"
        " continuity_state, response_chars, estimated_tokens, created_at, lifecycle_version)"
        " VALUES (?, ?, 'project:t', ?, 'started', 40, 10, ?, 'v0')",
        (cid, sid, rtype, ts),
    )


def test_record_skips_unknown_session_and_orders_by_id(tmp_path):
    conn = _eng(tmp_path).db.connect()
    _session(conn, "s1")
    for kind in ("activate", "recall", "feedback"):
        session_events.record(conn, session_id="s1", kind=kind, ts=5)  # same second
    session_events.record(conn, session_id="unknown", kind="feedback")
    session_events.record(conn, session_id="missing", kind="feedback")
    conn.commit()
    kinds = [r[0] for r in conn.execute("SELECT kind FROM session_events ORDER BY id")]
    assert kinds == ["activate", "recall", "feedback"]


def test_backfill_is_idempotent_and_ordered(tmp_path):
    conn = _eng(tmp_path).db.connect()
    _session(conn, "s1")
    _ledger(conn, "ctx1", "s1", "context", 100)
    _ledger(conn, "rec1", "s1", "recall", 100)  # same second as activate
    # Legacy thin ledger rows for remember/feedback/commit must not double-count the
    # real calls, which come from raw_events / feedback_events.
    _ledger(conn, "cmt1", "s1", "commit", 200)
    _ledger(conn, "fbk1", "s1", "feedback", 160)
    _ledger(conn, "rem1", "s1", "remember", 150)
    conn.execute(
        "INSERT INTO raw_events (session_id, ts, type, content) VALUES ('s1', 150, 'remember:fact', 'x')"
    )
    conn.execute(
        "INSERT INTO raw_events (session_id, ts, type, content) VALUES ('s1', 200, 'task_complete', 'done')"
    )
    conn.commit()
    first = session_events.run_backfill(conn)
    assert first["sessions"] == 1
    rows = conn.execute(
        "SELECT kind, origin FROM session_events WHERE session_id='s1' ORDER BY id"
    ).fetchall()
    assert [r[0] for r in rows] == ["activate", "recall", "remember", "commit"]
    assert {r[1] for r in rows} == {"backfill"}
    again = session_events.run_backfill(conn)
    assert again == {"sessions": 0, "evidence": 0}
    assert conn.execute("SELECT COUNT(*) FROM session_events").fetchone()[0] == 4


def test_backfill_skips_sessions_with_live_rows(tmp_path):
    conn = _eng(tmp_path).db.connect()
    _session(conn, "s1")
    _ledger(conn, "ctx1", "s1", "context", 100)
    session_events.record(conn, session_id="s1", kind="activate", retrieval_id="ctx1")
    conn.commit()
    session_events.run_backfill(conn)
    assert conn.execute("SELECT COUNT(*) FROM session_events").fetchone()[0] == 1


def test_evidence_session_backfill(tmp_path):
    conn = _eng(tmp_path).db.connect()
    _session(conn, "s1")
    cur = conn.execute(
        "INSERT INTO raw_events (session_id, ts, type, content) VALUES ('s1', 1, 'remember:fact', 'x')"
    )
    conn.execute(
        "INSERT INTO schemas (content_text, scope_id, first_formed_ts, last_updated_ts)"
        " VALUES ('claim', 'project:t', 1, 1)"
    )
    sid = conn.execute("SELECT MAX(id) FROM schemas").fetchone()[0]
    conn.execute(
        "INSERT INTO schema_evidence (schema_id, episode_id, raw_event_id, weight) VALUES (?, NULL, ?, 1)",
        (sid, cur.lastrowid),
    )
    conn.commit()
    session_events.run_backfill(conn)
    assert conn.execute("SELECT session_id FROM schema_evidence").fetchone()[0] == "s1"


def test_reaper_counts_live_session_event_as_activity(tmp_path):
    eng = _eng(tmp_path)
    conn = eng.db.connect()
    _session(conn, "busy", started=int(time.time()) - 7200)
    session_events.record(
        conn, session_id="busy", kind="recall", ts=int(time.time()) - 100
    )  # only a recall: no raw_events
    conn.commit()
    assert _reap_once(lambda: eng, timeout_s=3600) == []
    assert _reap_once(lambda: eng, timeout_s=0) == ["busy"]
    conn = eng.db.connect()
    kinds = [r[0] for r in conn.execute("SELECT kind FROM session_events ORDER BY id")]
    assert "session_reaped" in kinds


def test_dashboard_call_log_and_memory_sessions(tmp_path):
    from slowave.dashboard import app as dash

    eng = _eng(tmp_path)
    conn = eng.db.connect()
    _session(conn, "s1")
    _ledger(conn, "ctx1", "s1", "context", 100)
    conn.execute(
        "INSERT INTO schemas (content_text, scope_id, first_formed_ts, last_updated_ts)"
        " VALUES ('claim', 'project:t', 1, 1)"
    )
    sch = conn.execute("SELECT MAX(id) FROM schemas").fetchone()[0]
    conn.execute(
        "INSERT INTO schema_evidence (schema_id, episode_id, raw_event_id, weight, session_id)"
        " VALUES (?, NULL, NULL, 1, 's1')",
        (sch,),
    )
    for k, ref in (("activate", "ctx1"), ("remember", None), ("commit", None)):
        session_events.record(conn, session_id="s1", kind=k, retrieval_id=ref, summary=k)
    conn.commit()
    path = str(tmp_path / "t.db")

    payload = dash._session_events_payload(path, "s1", {"kinds": ["remember,commit"]})
    assert [e["kind"] for e in payload["events"]] == ["remember", "commit"]
    first = payload["events"][0]["id"]
    after = dash._session_events_payload(path, "s1", {"after_id": [str(first)]})
    assert [e["kind"] for e in after["events"]] == ["commit"]
    assert dash._session_events_payload(path, "nope", {})["error"] == "session not found"

    detail = dash._activity_detail(path, "s1")
    assert [e["kind"] for e in detail["call_log"]] == ["activate", "remember", "commit"]
    assert [m["schema_id"] for m in detail["memories"]] == [sch]
    c = dash._connect(path)
    linked, linked_total = dash._memory_sessions(c, sch)
    assert [(r["session_id"], r["is_origin"]) for r in linked] == [("s1", 1)]
    assert linked_total == 1


def test_session_routes_are_product_routes():
    from slowave.dashboard import app as dash

    assert dash._is_product_route("/sessions")
    assert dash._is_product_route("/sessions/sess_abc")
    assert dash._is_product_route("/activity/sess_abc")


def test_sessions_list_paged_path_matches_full_aggregation(tmp_path):
    """Sorting by a sessions column (paged first) must equal the full-scan aggregate path."""
    from slowave.dashboard import app as dash

    conn = _eng(tmp_path).db.connect()
    for i in range(6):
        _session(conn, f"s{i}", started=1000 + i, ended=1100 + i)
        for j in range(i):
            _ledger(conn, f"c{i}_{j}", f"s{i}", "context", 1000 + i)
        conn.execute(
            "INSERT INTO raw_events (session_id, ts, type, content) VALUES (?, ?, 'task_complete', 'x')",
            (f"s{i}", 1050 + i),
        )
    conn.commit()
    path = str(tmp_path / "t.db")
    keys = ("id", "retrieval_count", "timeline_event_count", "last_event_ts", "duration_s")

    def rows(sort, **kw):
        qs = {"sort": [sort], "dir": ["asc"], "per_page": ["50"], **kw}
        return {
            r["id"]: tuple(r[k] for k in keys)
            for r in dash._activity_payload(path, qs)["activities"]
        }

    assert rows("started") == rows("retrievals")  # paged vs aggregate path
    page2 = dash._activity_payload(
        path, {"sort": ["started"], "dir": ["asc"], "per_page": ["4"], "page": ["2"]}
    )["activities"]
    assert [r["id"] for r in page2] == ["s4", "s5"]
    assert page2[1]["retrieval_count"] == 5


def test_retired_backfill_rows_are_rebuilt(tmp_path):
    conn = _eng(tmp_path).db.connect()
    _session(conn, "s1")
    _ledger(conn, "ctx1", "s1", "context", 100)
    conn.execute(
        "INSERT INTO session_events (session_id, ts, kind, origin) VALUES ('s1', 1, 'feedback', 'backfill')"
    )
    conn.execute(
        "INSERT INTO storage_migrations(version, applied_at) VALUES ('session-events-backfill-v1', 1)"
    )
    conn.commit()
    session_events.run_backfill(conn)
    kinds = [r[0] for r in conn.execute("SELECT kind FROM session_events ORDER BY id")]
    assert kinds == ["activate"]


def test_call_log_carries_returned_items_and_assessments(tmp_path):
    from slowave.dashboard import app as dash

    conn = _eng(tmp_path).db.connect()
    _session(conn, "s1")
    _ledger(conn, "ctx1", "s1", "context", 100)
    conn.execute(
        "INSERT INTO schemas (content_text, scope_id, first_formed_ts, last_updated_ts)"
        " VALUES ('stored claim', 'project:t', 1, 1)"
    )
    sch = conn.execute("SELECT MAX(id) FROM schemas").fetchone()[0]
    conn.execute(
        "INSERT INTO context_recall_items (context_id, memory_id, retrieval_type, memory_type, rank,"
        " created_at) VALUES ('ctx1', ?, 'context', 'schema', 1, 100)",
        (f"sch_{sch}",),  # no content_text snapshot: must fall back to the schema text
    )
    conn.execute(
        "INSERT INTO feedback_events (event_id, retrieval_id, session_id, target_kind, target_id,"
        " assessment, status, coverage, source_contract, created_at)"
        " VALUES ('f1','ctx1','s1','memory',?, 'used','accepted','complete','t',101)",
        (f"sch_{sch}",),
    )
    session_events.record(conn, session_id="s1", kind="activate", retrieval_id="ctx1")
    session_events.record(conn, session_id="s1", kind="feedback", retrieval_id="ctx1")
    conn.commit()
    events = dash._session_events_payload(str(tmp_path / "t.db"), "s1", {})["events"]
    assert events[0]["items"] == [{"id": f"sch_{sch}", "kind": "memory", "text": "stored claim"}]
    assert events[1]["assessments"][0]["assessment"] == "used"


def test_ensure_schema_migrates_pre_session_log_db(tmp_path):
    """A database from before this feature is migrated and backfilled without an engine."""
    from slowave.dashboard import app as dash

    eng = _eng(tmp_path)
    conn = eng.db.connect()
    _session(conn, "s1")
    _ledger(conn, "ctx1", "s1", "context", 100)
    conn.commit()
    conn.execute("DROP TABLE session_events")
    conn.execute("DROP INDEX idx_schema_evidence_session")
    conn.execute("ALTER TABLE schema_evidence DROP COLUMN session_id")
    conn.execute("DELETE FROM storage_migrations WHERE version LIKE 'session-events%'")
    conn.commit()
    eng.db.close()
    path = str(tmp_path / "t.db")

    db = session_events.ensure_schema(path)
    assert session_events.run_backfill(db.connect()) == {"sessions": 1, "evidence": 0}
    db.close()
    payload = dash._activity_payload(path, {})
    assert [a["id"] for a in payload["activities"]] == ["s1"]
    assert dash._session_events_payload(path, "s1", {})["events"][0]["kind"] == "activate"
