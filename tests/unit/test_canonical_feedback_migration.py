"""WP-3: legacy-ledger consumers read the canonical view.

Both migrated consumers (consolidation co-use, schema generalization
validation) must count canonical feedback_events rows, keep legacy
context_feedback_events rows contributing (typed legacy, one unit per row),
and let canonical marks dominate conflicts per scope. Uses temporary
databases only.
"""

from __future__ import annotations

import time

import pytest

from slowave.core.config import SlowaveConfig
from slowave.core.engine import SlowaveEngine


@pytest.fixture()
def eng(tmp_path):
    engine = SlowaveEngine(
        SlowaveConfig(db_path=str(tmp_path / "wp3.db"), dim=8, disable_encoder=True)
    )
    yield engine
    engine.close()


def _make(eng, label: str) -> int:
    return eng.schemas.create(
        content_text=label,
        facets={"schema_class": "fact"},
        tags=[],
        embedding=None,
        scope_id="project:alpha",
        salience=1.0,
        dedupe=False,
    )


def _weight_between(eng, a: int, b: int) -> float:
    for neighbor_id, _relation, weight in eng.schemas.get_coactivations(a):
        if neighbor_id == b:
            return weight
    return 0.0


def _expose_pair(eng, context_id: str, p: int, q: int) -> None:
    eng.record_context_recall(
        context_id=context_id,
        session_id="sess_wp3",
        response={
            "schemas": [
                {"id": f"sch_{p}", "activation": 0.8, "pathway": "direct"},
                {"id": f"sch_{q}", "activation": 0.8, "pathway": "direct"},
            ]
        },
    )


def test_canonical_used_rows_drive_explicit_couse(eng):
    """Public v9 `used` feedback (canonical ledger) writes the boosted edge."""
    p = _make(eng, "canonical co-use p")
    q = _make(eng, "canonical co-use q")
    _expose_pair(eng, "ctx_canon_couse", p, q)
    result = eng.feedback(
        retrieval_id="ctx_canon_couse",
        memory_feedback=[
            {"memory_id": f"sch_{p}", "assessment": "used"},
            {"memory_id": f"sch_{q}", "assessment": "used"},
        ],
        coverage="complete",
    )
    assert result["rejected"] == []
    conn = eng.db.connect()
    assert conn.execute("SELECT COUNT(*) FROM context_feedback_events").fetchone()[0] == 0

    now_ts = int(time.time()) + 1
    stats = eng._consolidation._write_coactivations(conn, now_ts)
    assert stats["explicit_pairs_written"] == 1
    assert _weight_between(eng, p, q) > 0.0


def test_retrieval_in_both_ledgers_counts_once(eng):
    """A retrieval covered by the canonical view is not re-counted via legacy."""
    p = _make(eng, "both ledgers p")
    q = _make(eng, "both ledgers q")
    _expose_pair(eng, "ctx_both", p, q)
    result = eng.feedback(
        retrieval_id="ctx_both",
        memory_feedback=[
            {"memory_id": f"sch_{p}", "assessment": "used"},
            {"memory_id": f"sch_{q}", "assessment": "used"},
        ],
        coverage="complete",
    )
    assert result["rejected"] == []
    # Simulate a legacy row for the same retrieval (historical overlap).
    conn = eng.db.connect()
    conn.execute(
        "INSERT INTO context_feedback_events ("
        "context_id, retrieval_type, session_id, scope_id, scope_kind, "
        "situation_json, requirements_json, feedback, outcome, feedback_signal_json, "
        "used_memory_ids_json, created_at) VALUES ("
        "'ctx_both', 'context', NULL, 'project:alpha', 'project', '{}', '[]', "
        "'useful', 'success', '{}', ?, strftime('%s','now'))",
        (f'["sch_{p}", "sch_{q}"]',),
    )
    conn.commit()

    now_ts = int(time.time()) + 1
    stats = eng._consolidation._write_coactivations(conn, now_ts)
    assert stats["explicit_pairs_written"] == 1


def test_legacy_couse_rows_still_contribute(eng):
    """Legacy rows for retrievals absent from the canonical view still count."""
    p = _make(eng, "legacy co-use p")
    q = _make(eng, "legacy co-use q")
    _expose_pair(eng, "ctx_legacy_only", p, q)
    eng.retrieval_feedback(
        retrieval_id="ctx_legacy_only",
        feedback="useful",
        outcome="success",
        used_memory_ids=[f"sch_{p}", f"sch_{q}"],
    )
    conn = eng.db.connect()
    now_ts = int(time.time()) + 1
    stats = eng._consolidation._write_coactivations(conn, now_ts)
    assert stats["explicit_pairs_written"] == 1
    assert _weight_between(eng, p, q) > 0.0


def test_canonical_marks_feed_generalization_validation(tmp_path):
    """A canonical `used` mark in a foreign scope validates that scope."""
    engine = SlowaveEngine(
        SlowaveConfig(db_path=str(tmp_path / "gen.db"), dim=8, disable_encoder=True)
    )
    try:
        sid = engine.schemas.create(
            content_text="Foreign-scope validation target.",
            facets={"schema_class": "fact"},
            tags=[],
            embedding=None,
            scope_id="project:home",
            salience=1.0,
            dedupe=False,
        )
        engine.record_retrieval(
            retrieval_id="r_foreign",
            scope_id="project:foreign",
            query="foreign validation query",
            response={"schemas": [{"id": f"sch_{sid}", "content": "target"}]},
        )
        engine.feedback(
            retrieval_id="r_foreign",
            memory_feedback=[{"memory_id": f"sch_{sid}", "assessment": "used"}],
            coverage="complete",
        )
        engine.schemas._update_utility_scores(sid, recall_hit=True)
        facets = engine.schemas.get(sid).facets
        assert facets["context_used_count"] == 1
        assert "project:foreign" in facets["context_noise_by_scope"]
    finally:
        engine.close()


def test_canonical_negative_dominates_legacy_used_conflict(eng):
    """Canonical-irrelevant wins over legacy-used for the same scope."""
    sid = _make(eng, "conflict target")
    eng.record_context_recall(
        context_id="ctx_conflict",
        session_id="sess_wp3",
        scope_id="project:alpha",
        response={"schemas": [{"id": f"sch_{sid}", "activation": 0.8, "pathway": "direct"}]},
    )
    # Legacy row: used. Canonical row: irrelevant (same retrieval).
    eng.retrieval_feedback(
        retrieval_id="ctx_conflict",
        feedback="useful",
        outcome="success",
        used_memory_ids=[f"sch_{sid}"],
    )
    result = eng.feedback(
        retrieval_id="ctx_conflict",
        memory_feedback=[{"memory_id": f"sch_{sid}", "assessment": "irrelevant"}],
        coverage="complete",
    )
    assert result["rejected"] == []
    eng.schemas._update_utility_scores(sid, recall_hit=True)
    facets = eng.schemas.get(sid).facets
    # Both marks show in the additive totals (real ledger tension).
    assert facets["context_used_count"] == 1
    assert facets["context_irrelevant_count"] == 1
    # Canonical dominance at scope granularity: the scope is negative, so it
    # cannot count as a validated-used (weight 1.0) distinct scope.
    assert facets["context_noise_by_scope"]["project:alpha"] > 0.0
    assert facets["distinct_scope_count"] == 0.0
