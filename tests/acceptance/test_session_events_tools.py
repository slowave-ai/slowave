"""Real MCP tool calls -> ordered session_events, with the legacy ledger intact."""

import asyncio
import sqlite3

from slowave.dashboard import app as dash
from tests.acceptance.mcp_harness import open_harness


def _rows(db, query, args=()):
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in conn.execute(query, args)]
    finally:
        conn.close()


def test_tools_write_ordered_session_events(tmp_path):
    db = tmp_path / "sim.db"

    async def flow():
        async with open_harness(db) as h:
            act, _ = await h.activate(
                "c", "Investigate widget cache", "investigate widget cache", "project:qa"
            )
            sid = act["session_id"]
            await h.remember_batch(
                [
                    {"content": "Widget cache uses an LRU of 128 entries.", "type": "fact"},
                    {"content": "Prefer pytest -x for the widget suite.", "type": "preference"},
                ],
                sid,
                "project:qa",
            )
            await h.remember("The widget cache TTL is 30 seconds.", "fact", sid, "project:qa")
            rec, _ = await h.recall("c", "widget cache ttl", sid, "project:qa")
            await h.feedback_all(act)
            await h.feedback_all(rec)
            bad, _ = await h.raw_call(
                "slowave_feedback", {"retrieval_id": "ctx_nope", "coverage": "complete"}
            )
            await h.commit(sid, "investigate widget cache")
            return sid, act, rec, bad

    sid, act, rec, bad = asyncio.run(flow())
    assert bad["ok"] is False  # unknown retrieval fails cleanly

    events = _rows(str(db), "SELECT * FROM session_events WHERE session_id=? ORDER BY id", (sid,))
    kinds = [e["kind"] for e in events]
    assert kinds[0] == "activate" and kinds[-1] == "commit"
    assert kinds.count("remember") == 3  # one per claim (2 batch + 1 scalar)
    assert kinds.count("recall") == 1 and kinds.count("feedback") >= 1
    assert {e["origin"] for e in events} == {"live"}
    assert events[0]["retrieval_id"] == act["retrieval_id"]
    # exposure/feedback point at real retrievals; remember/commit do not carry a retrieval
    assert all(e["retrieval_id"] for e in events if e["kind"] in ("activate", "recall", "feedback"))
    assert all(e["retrieval_id"] is None for e in events if e["kind"] in ("remember", "commit"))
    assert not _rows(str(db), "SELECT 1 FROM session_events WHERE retrieval_id='ctx_nope'")
    assert not _rows(str(db), "PRAGMA foreign_key_check")

    # each remembered claim references its raw event, which resolves to the created memory
    remembers = [e for e in events if e["kind"] == "remember"]
    assert all(e["raw_event_id"] for e in remembers)
    log = dash._session_events_payload(str(db), sid, {})
    assert [e["kind"] for e in log["events"]] == kinds
    assert [e["ordinal"] for e in log["events"]] == list(range(1, len(kinds) + 1))
    remembered = [e for e in log["events"] if e["kind"] == "remember"]
    assert all(e["memory_id"].startswith("sch_") and e["summary"] for e in remembered)
    assert log["kind_counts"]["remember"] == 3 and log["total"] == len(kinds)

    # legacy ledger rows are still written for the token metrics
    ledger = {
        r["retrieval_type"]
        for r in _rows(
            str(db), "SELECT retrieval_type FROM context_recall_events WHERE session_id=?", (sid,)
        )
    }
    assert {"context", "recall", "remember", "feedback", "commit"} <= ledger
