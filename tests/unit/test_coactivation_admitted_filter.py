"""Regression test for the 2026-07-23 cross-scope co-activation leak.

_write_coactivations() (ConsolidationService) reads context_recall_items to
find schemas "recalled together" in the same session. That table also stores
rank=-1/admitted=0 rows for candidates the working-memory gate evaluated and
REJECTED -- e.g. a cross-scope graph-expansion candidate correctly filtered
out by scope isolation (see feedback.py's "persist filtered items so the
full candidate pool is queryable"). Without an admitted=1 filter, a rejected
candidate reads as co-activated with everything actually admitted in the
same call, silently punching co-activation edges through the scope boundary
the rest of retrieval enforces.
"""

from __future__ import annotations

import time

import pytest

from slowave.core.config import SlowaveConfig
from slowave.core.engine import SlowaveEngine


@pytest.fixture()
def eng(tmp_path):
    engine = SlowaveEngine(
        SlowaveConfig(db_path=str(tmp_path / "test.db"), dim=8, disable_encoder=True)
    )
    yield engine
    engine.close()


def _seed_recall(conn, *, context_id, session_id, ts, admitted_ids, rejected_ids):
    conn.execute(
        "INSERT INTO context_recall_events (context_id, session_id, created_at) "
        "VALUES (?, ?, ?)",
        (context_id, session_id, ts),
    )
    rank = 0
    for schema_id in admitted_ids:
        conn.execute(
            "INSERT INTO context_recall_items "
            "(context_id, memory_id, memory_type, rank, admitted, created_at) "
            "VALUES (?, ?, 'schema', ?, 1, ?)",
            (context_id, f"sch_{schema_id}", rank, ts),
        )
        rank += 1
    for schema_id in rejected_ids:
        conn.execute(
            "INSERT INTO context_recall_items "
            "(context_id, memory_id, memory_type, rank, admitted, created_at) "
            "VALUES (?, ?, 'schema', -1, 0, ?)",
            (context_id, f"sch_{schema_id}", ts),
        )
    conn.commit()


def test_rejected_candidate_does_not_get_coactivated(eng):
    admitted_a = eng.schemas.create(
        content_text="admitted A", facets={}, tags=[], embedding=None, dedupe=False
    )
    admitted_b = eng.schemas.create(
        content_text="admitted B", facets={}, tags=[], embedding=None, dedupe=False
    )
    rejected_c = eng.schemas.create(
        content_text="rejected cross-scope C",
        facets={},
        tags=[],
        embedding=None,
        scope_id="project:other",
        dedupe=False,
    )

    conn = eng.db.connect()
    now_ts = int(time.time())
    _seed_recall(
        conn,
        context_id="ctx_test1",
        session_id="sess_test1",
        ts=now_ts - 10,
        admitted_ids=[admitted_a, admitted_b],
        rejected_ids=[rejected_c],
    )

    eng._consolidation._write_coactivations(conn, now_ts)

    edges_a = eng.schemas.get_coactivations(admitted_a)
    neighbor_ids = {n for n, _rel, _w in edges_a}
    assert admitted_b in neighbor_ids, "both admitted schemas must be co-activated"
    assert rejected_c not in neighbor_ids, "rejected/filtered candidate must NOT be co-activated"

    edges_c = eng.schemas.get_coactivations(rejected_c)
    assert edges_c == [], "rejected candidate must have no co-activation edges at all"
