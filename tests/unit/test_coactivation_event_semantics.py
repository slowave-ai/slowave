"""WP-6: co-activation writer honesty tests.

Pre-WP-6, `_write_coactivations()` (ConsolidationService) grouped by
`session_id` and treated every admitted schema identically regardless of why
it was shown -- a directly relevant hit, a salience-filled exploration slot,
or a graph-propagated association all earned the same Hebbian edge merely by
being admitted somewhere in the same session. That fabricated two kinds of
edges the plan's Phase 4 flagged as dishonest:

1. Cross-call leakage: two topically unrelated `activate()` calls sharing one
   session_id got pairwise-crossed, and `recall()` (which never set
   `session_id`) was invisible to co-activation entirely.
2. Pathway blindness: an exploration/graph item shown next to a direct hit
   earned that hit an edge exactly like a second genuinely relevant memory
   would have.

These tests lock in the WP-6 rewrite: grouping by `context_id` (one
retrieval call, always present) instead of `session_id`, filtering to
`pathway = 'direct'`, and a separate, stronger boost for explicit
client-confirmed co-use (`used_memory_ids` naming 2+ schemas in one feedback
call). See test_coactivation_admitted_filter.py for the older admitted=1
regression this builds on.
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


def test_cross_call_same_session_does_not_leak(eng):
    """Two topically unrelated activate() calls sharing one session_id must
    not cross-pollinate co-activation -- grouping is per retrieval call
    (context_id), not per session.
    """
    x1 = _make(eng, "topic x one")
    x2 = _make(eng, "topic x two")
    y1 = _make(eng, "topic y one")
    y2 = _make(eng, "topic y two")

    eng.record_context_recall(
        context_id="ctx_topic_x",
        session_id="sess_shared",
        response={
            "schemas": [
                {"id": f"sch_{x1}", "activation": 0.8, "pathway": "direct"},
                {"id": f"sch_{x2}", "activation": 0.8, "pathway": "direct"},
            ]
        },
    )
    eng.record_context_recall(
        context_id="ctx_topic_y",
        session_id="sess_shared",
        response={
            "schemas": [
                {"id": f"sch_{y1}", "activation": 0.8, "pathway": "direct"},
                {"id": f"sch_{y2}", "activation": 0.8, "pathway": "direct"},
            ]
        },
    )

    now_ts = int(time.time()) + 1
    eng._consolidation._write_coactivations(eng.db.connect(), now_ts)

    assert _weight_between(eng, x1, x2) > 0.0, "same-call co-presentation must still earn an edge"
    assert _weight_between(eng, y1, y2) > 0.0
    assert (
        _weight_between(eng, x1, y1) == 0.0
    ), "unrelated calls sharing a session must not cross-pollinate"
    assert _weight_between(eng, x1, y2) == 0.0
    assert _weight_between(eng, x2, y1) == 0.0
    assert _weight_between(eng, x2, y2) == 0.0


def test_recall_without_session_id_now_participates(eng):
    """recall() never sets session_id -- context_id-scoped grouping means it
    no longer needs one to contribute co-activation edges.
    """
    g1 = _make(eng, "recall visible one")
    g2 = _make(eng, "recall visible two")

    eng.record_retrieval(
        retrieval_id="rec_no_session",
        retrieval_type="recall",
        response={
            "schemas": [
                {"id": f"sch_{g1}", "score": 0.8, "pathway": "direct"},
                {"id": f"sch_{g2}", "score": 0.7, "pathway": "direct"},
            ]
        },
    )

    now_ts = int(time.time()) + 1
    eng._consolidation._write_coactivations(eng.db.connect(), now_ts)

    assert _weight_between(eng, g1, g2) > 0.0


def test_exploration_and_graph_pathways_do_not_earn_edges(eng):
    """A direct hit shown next to a salience-filled exploration slot or a
    graph-propagated neighbor must not gain a co-activation edge with either
    -- only pathway='direct' co-presentation counts.
    """
    direct_hit = _make(eng, "pathway direct hit")
    exploration_item = _make(eng, "pathway exploration filler")
    graph_item = _make(eng, "pathway graph neighbor")

    eng.record_context_recall(
        context_id="ctx_pathway_mix",
        session_id="sess_pathway",
        response={
            "schemas": [
                {"id": f"sch_{direct_hit}", "activation": 0.9, "pathway": "direct"},
                {"id": f"sch_{exploration_item}", "activation": 0.3, "pathway": "exploration"},
                {"id": f"sch_{graph_item}", "activation": 0.5, "pathway": "graph"},
            ]
        },
    )

    now_ts = int(time.time()) + 1
    eng._consolidation._write_coactivations(eng.db.connect(), now_ts)

    assert _weight_between(eng, direct_hit, exploration_item) == 0.0
    assert _weight_between(eng, direct_hit, graph_item) == 0.0
    assert _weight_between(eng, exploration_item, graph_item) == 0.0


def test_missing_pathway_defaults_to_direct(eng):
    """Back-compat: a caller that never supplies `pathway` (e.g. an older
    snapshot, or a raw response dict like the one below) must default to
    'direct' rather than being silently dropped from co-activation.
    """
    a = _make(eng, "no pathway key a")
    b = _make(eng, "no pathway key b")

    eng.record_context_recall(
        context_id="ctx_no_pathway",
        session_id="sess_no_pathway",
        response={
            "schemas": [
                {"id": f"sch_{a}", "activation": 0.8},
                {"id": f"sch_{b}", "activation": 0.7},
            ]
        },
    )

    now_ts = int(time.time()) + 1
    eng._consolidation._write_coactivations(eng.db.connect(), now_ts)

    assert _weight_between(eng, a, b) > 0.0


def test_explicit_couse_earns_stronger_edge_than_copresentation(eng):
    """Two schemas named together in a single `used_memory_ids` feedback call
    are grounded in real client-confirmed use, not mere co-presentation --
    the resulting edge must be stronger than an ordinary same-call
    co-presentation edge between two schemas with no such feedback.
    """
    control_a = _make(eng, "control copresentation a")
    control_b = _make(eng, "control copresentation b")
    p = _make(eng, "explicit use p")
    q = _make(eng, "explicit use q")

    eng.record_context_recall(
        context_id="ctx_control",
        session_id="sess_explicit",
        response={
            "schemas": [
                {"id": f"sch_{control_a}", "activation": 0.8, "pathway": "direct"},
                {"id": f"sch_{control_b}", "activation": 0.8, "pathway": "direct"},
            ]
        },
    )
    eng.record_context_recall(
        context_id="ctx_explicit_use",
        session_id="sess_explicit",
        response={
            "schemas": [
                {"id": f"sch_{p}", "activation": 0.8, "pathway": "direct"},
                {"id": f"sch_{q}", "activation": 0.8, "pathway": "direct"},
            ]
        },
    )
    eng.retrieval_feedback(
        retrieval_id="ctx_explicit_use",
        feedback="useful",
        outcome="success",
        used_memory_ids=[f"sch_{p}", f"sch_{q}"],
    )

    now_ts = int(time.time()) + 1
    stats = eng._consolidation._write_coactivations(eng.db.connect(), now_ts)

    control_weight = _weight_between(eng, control_a, control_b)
    explicit_weight = _weight_between(eng, p, q)

    assert control_weight > 0.0
    assert explicit_weight > control_weight
    assert stats["explicit_pairs_written"] == 1


def test_single_used_id_does_not_trigger_explicit_couse(eng):
    """Marking only one schema as used (not a pair) must not write an
    explicit co-use edge -- there is no second schema to pair it with.
    """
    a = _make(eng, "solo used a")
    b = _make(eng, "solo shown b")

    eng.record_context_recall(
        context_id="ctx_solo_used",
        session_id="sess_solo",
        response={
            "schemas": [
                {"id": f"sch_{a}", "activation": 0.8, "pathway": "direct"},
                {"id": f"sch_{b}", "activation": 0.7, "pathway": "direct"},
            ]
        },
    )
    eng.retrieval_feedback(
        retrieval_id="ctx_solo_used",
        feedback="partially_useful",
        outcome="success",
        used_memory_ids=[f"sch_{a}"],
    )

    now_ts = int(time.time()) + 1
    stats = eng._consolidation._write_coactivations(eng.db.connect(), now_ts)

    assert stats["explicit_pairs_written"] == 0
    # Mere co-presentation edge still forms (both are pathway='direct').
    assert _weight_between(eng, a, b) == 1.0
