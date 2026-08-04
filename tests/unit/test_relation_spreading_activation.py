"""Tests for relation-graph spreading activation (2026-07-14): schema_relations
edges from an admitted schema can surface a neighbor that wasn't a direct hit,
in both context_brief() (WorkingMemoryGate.expand_via_relations) and
recall() (RetrievalService, via the same spread_relation_activation core).
"""

from __future__ import annotations

import os
import tempfile
import time

import numpy as np
import pytest

from slowave.core.config import SlowaveConfig
from slowave.core.context import (
    GatePolicy,
    MemoryCue,
    WorkingMemoryGate,
    WorkingMemoryItem,
    WorkingMemoryState,
    spread_relation_activation,
)
from slowave.core.engine import SlowaveEngine


class _StubEncoder:
    """Deterministic hash-based encoder so recall() works without model weights."""

    def __init__(self, dim: int = 8):
        self._dim = dim

    def encode(self, text: str) -> np.ndarray:
        seed = int(abs(hash(text)) % (2**31))
        v = np.random.default_rng(seed).standard_normal(self._dim).astype(np.float32)
        return v / (np.linalg.norm(v) + 1e-12)


# ---------------------------------------------------------------------------
# spread_relation_activation -- pure algorithm, no DB
# ---------------------------------------------------------------------------


def test_single_hop_propagation_above_threshold():
    def fetch_relations(schema_id):
        if schema_id == 1:
            return [(2, "relates_to", 0.9)]
        return []

    winners = spread_relation_activation(
        {1: 0.8}, fetch_relations=fetch_relations, min_activation=0.20
    )
    assert 2 in winners
    activation, via = winners[2]
    assert activation == pytest.approx(0.8 * 0.9 * 0.6, rel=1e-6)
    assert via == {"relates_to"}


def test_below_threshold_neighbor_is_dropped():
    def fetch_relations(schema_id):
        return [(2, "relates_to", 0.3)] if schema_id == 1 else []

    winners = spread_relation_activation(
        {1: 0.3}, fetch_relations=fetch_relations, min_activation=0.20
    )
    assert 2 not in winners


def test_convergent_paths_sum_before_threshold():
    """Neither seed alone clears the bar via a single 0.4-confidence edge, but
    two seeds both linking to the same neighbor sum their contributions."""

    def fetch_relations(schema_id):
        if schema_id in (1, 2):
            return [(99, "relates_to", 0.6)]
        return []

    single = spread_relation_activation(
        {1: 0.5}, fetch_relations=fetch_relations, min_activation=0.35
    )
    assert 99 not in single  # 0.5*0.6*0.6 = 0.18, below 0.35

    converged = spread_relation_activation(
        {1: 0.5, 2: 0.5}, fetch_relations=fetch_relations, min_activation=0.35
    )
    assert 99 in converged
    activation, via = converged[99]
    assert activation == pytest.approx(2 * 0.5 * 0.6 * 0.6, rel=1e-6)
    assert via == {"relates_to"}


def test_admitted_schemas_never_reappear_as_winners():
    def fetch_relations(schema_id):
        return [(1, "relates_to", 1.0)] if schema_id == 2 else [(2, "relates_to", 1.0)]

    winners = spread_relation_activation(
        {1: 1.0, 2: 1.0}, fetch_relations=fetch_relations, min_activation=0.01
    )
    assert 1 not in winners
    assert 2 not in winners


def test_cycle_is_handled_without_infinite_loop():
    """A <-> B <-> C cycle must terminate (visited set) and not double-count
    C's contribution once it's already been reached."""

    def fetch_relations(schema_id):
        edges = {
            1: [(2, "relates_to", 1.0)],
            2: [(1, "relates_to", 1.0), (3, "relates_to", 1.0)],
            3: [(2, "relates_to", 1.0), (1, "relates_to", 1.0)],
        }
        return edges.get(schema_id, [])

    winners = spread_relation_activation(
        {1: 1.0}, fetch_relations=fetch_relations, min_activation=0.01
    )
    # Must terminate and return a finite result; 1 (seed) never a winner.
    assert 1 not in winners
    assert 2 in winners or 3 in winners


def test_max_extra_cap_limits_winner_count():
    def fetch_relations(schema_id):
        if schema_id == 1:
            return [(i, "relates_to", 0.99) for i in range(2, 10)]
        return []

    winners = spread_relation_activation(
        {1: 1.0}, fetch_relations=fetch_relations, min_activation=0.01
    )
    assert len(winners) <= 3  # _GRAPH_MAX_EXTRA


def test_no_seeds_returns_empty():
    assert spread_relation_activation({}, fetch_relations=lambda i: [], min_activation=0.0) == {}


# ---------------------------------------------------------------------------
# Integration: context_brief() and recall() surface graph-linked neighbors
# ---------------------------------------------------------------------------


def _tmp_engine() -> tuple[SlowaveEngine, str]:
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    cfg = SlowaveConfig(db_path=tmp.name, dim=8, disable_encoder=True)
    return SlowaveEngine(cfg), tmp.name


def _cleanup(path: str) -> None:
    for ext in ("", "-wal", "-shm"):
        p = path + ext
        if os.path.exists(p):
            os.remove(p)


def test_context_brief_surfaces_relates_to_neighbor_via_graph_expansion():
    eng, path = _tmp_engine()
    try:
        parent_id = eng.schemas.create(
            content_text="For meal planning, the user prefers vegetarian recipes.",
            facets={
                "schema_class": "preference",
                "topics": ["food", "meal planning"],
                "memory_layer": "profile",
                "stability": "current",
            },
            tags=["food", "meal_planning", "vegetarian"],
            embedding=None,
            salience=5.0,
            dedupe=False,
        )
        child_id = eng.schemas.create(
            content_text="The kitchen restocks olive oil every two weeks.",
            facets={"schema_class": "fact"},
            tags=["unrelated_tag"],
            embedding=None,
            salience=0.01,
            scope_id="proj:other",  # non-None so it doesn't get the scope-less "global" bonus
            dedupe=False,
        )
        eng.schemas.add_relation(
            src_schema_id=min(child_id, parent_id),
            dst_schema_id=max(child_id, parent_id),
            relation="relates_to",
            confidence=0.95,
        )

        # WP-5.1 (2026-07-29) flipped the production graph_channels default
        # to "off" -- this test exercises graph-expansion mechanics
        # specifically, so it opts back in explicitly.
        brief = eng.context_brief(
            query="plan vegetarian meals", topics=["food"], limit=5, graph_channels="combined"
        )
        ids = [item.schema.id for item in brief.items]

        assert parent_id in ids, "direct hit must still be admitted"
        assert child_id in ids, "relates_to neighbor must surface via graph expansion"
        child_item = next(item for item in brief.items if item.schema.id == child_id)
        assert child_item.peripheral is True
        assert child_item.reason.startswith("graph:")
    finally:
        eng.close()
        _cleanup(path)


def test_context_brief_blocks_cross_scope_graph_neighbor_below_stage():
    """schema_relations is not a scope boundary -- relates_to edges can form
    cross-scope (content similarity doesn't check scope at write time), so a
    graph-propagated neighbor from a different scope must still respect
    cross-scope isolation (generalization_stage >= 2) the same way direct
    candidates do."""
    eng, path = _tmp_engine()
    try:
        parent_id = eng.schemas.create(
            content_text="For meal planning, the user prefers vegetarian recipes.",
            facets={
                "schema_class": "preference",
                "topics": ["food", "meal planning"],
                "memory_layer": "profile",
                "stability": "current",
            },
            tags=["food", "meal_planning", "vegetarian"],
            embedding=None,
            salience=5.0,
            scope_id="project:alpha",
            dedupe=False,
        )
        child_id = eng.schemas.create(
            content_text="The kitchen restocks olive oil every two weeks.",
            facets={"schema_class": "fact"},
            tags=["unrelated_tag"],
            embedding=None,
            salience=0.01,
            scope_id="project:beta",  # different scope, stage 0 (default)
            dedupe=False,
        )
        eng.schemas.add_relation(
            src_schema_id=min(child_id, parent_id),
            dst_schema_id=max(child_id, parent_id),
            relation="relates_to",
            confidence=0.95,
        )

        brief = eng.context_brief(
            query="plan vegetarian meals", topics=["food"], scope="project:alpha", limit=5
        )
        ids = [item.schema.id for item in brief.items]
        assert parent_id in ids
        assert child_id not in ids, "stage-0 cross-scope neighbor must not leak via graph expansion"
    finally:
        eng.close()
        _cleanup(path)


def test_recall_surfaces_relates_to_neighbor_via_graph_expansion():
    eng, path = _tmp_engine()
    eng.encoder = _StubEncoder(dim=8)
    try:
        parent_id = eng.schemas.create(
            content_text="For meal planning, the user prefers vegetarian recipes.",
            facets={"schema_class": "preference"},
            tags=["food", "meal_planning", "vegetarian"],
            embedding=None,
            salience=5.0,
            dedupe=False,
        )
        child_id = eng.schemas.create(
            content_text="Specifically, the user avoids mushrooms in vegetarian dishes.",
            facets={"schema_class": "fact"},
            tags=["unrelated_tag"],
            embedding=None,
            salience=0.01,
            dedupe=False,
        )
        eng.schemas.add_relation(
            src_schema_id=min(child_id, parent_id),
            dst_schema_id=max(child_id, parent_id),
            relation="relates_to",
            confidence=0.95,
        )

        # WP-5.1 (2026-07-29) flipped the production graph_channels default
        # to "off" -- this test exercises graph-expansion mechanics
        # specifically, so it opts back in explicitly.
        result = eng.recall("vegetarian meal planning recipes", top_k=5, graph_channels="combined")
        ids = [s.id for s in result.schemas]
        related_ids = [s.id for s in result.related_schemas]

        assert parent_id in ids, "direct hit must still be a top_k result"
        assert child_id not in ids, (
            "graph-propagated neighbors must NOT be merged into schemas -- "
            "every benchmark script assumes len(schemas) <= top_k"
        )
        assert child_id in related_ids, "relates_to neighbor must surface via related_schemas"
        assert result.schema_activations[child_id] > 0
    finally:
        eng.close()
        _cleanup(path)


def test_recall_schemas_never_exceeds_top_k_even_with_graph_winners():
    """Regression guard for the exact bug this fix addresses: benchmark
    scripts (retrieval_metrics.compute_recall_at_k_and_mrr, dmr_original_eval,
    etc.) concatenate result.schemas assuming len() <= top_k."""
    eng, path = _tmp_engine()
    eng.encoder = _StubEncoder(dim=8)
    try:
        parent_id = eng.schemas.create(
            content_text="For meal planning, the user prefers vegetarian recipes.",
            facets={"schema_class": "preference"},
            tags=["food", "meal_planning", "vegetarian"],
            embedding=None,
            salience=5.0,
            dedupe=False,
        )
        child_id = eng.schemas.create(
            content_text="Specifically, the user avoids mushrooms in vegetarian dishes.",
            facets={"schema_class": "fact"},
            tags=["unrelated_tag"],
            embedding=None,
            salience=0.01,
            dedupe=False,
        )
        eng.schemas.add_relation(
            src_schema_id=min(child_id, parent_id),
            dst_schema_id=max(child_id, parent_id),
            relation="relates_to",
            confidence=0.95,
        )

        top_k = 1
        result = eng.recall("vegetarian meal planning recipes", top_k=top_k)
        assert len(result.schemas) <= top_k
    finally:
        eng.close()
        _cleanup(path)


def test_recall_blocks_cross_scope_graph_neighbor_below_stage():
    """Same cross-scope isolation guarantee as context_brief: schema_relations
    is not a scope boundary, so a graph-propagated neighbor from a different,
    stage-0 scope must not leak into related_schemas either."""
    eng, path = _tmp_engine()
    eng.encoder = _StubEncoder(dim=8)
    try:
        parent_id = eng.schemas.create(
            content_text="For meal planning, the user prefers vegetarian recipes.",
            facets={"schema_class": "preference"},
            tags=["food", "meal_planning", "vegetarian"],
            embedding=None,
            salience=5.0,
            scope_id="project:alpha",
            dedupe=False,
        )
        child_id = eng.schemas.create(
            content_text="Specifically, the user avoids mushrooms in vegetarian dishes.",
            facets={"schema_class": "fact"},
            tags=["unrelated_tag"],
            embedding=None,
            salience=0.01,
            scope_id="project:beta",  # different scope, stage 0 (default)
            dedupe=False,
        )
        eng.schemas.add_relation(
            src_schema_id=min(child_id, parent_id),
            dst_schema_id=max(child_id, parent_id),
            relation="relates_to",
            confidence=0.95,
        )

        result = eng.recall("vegetarian meal planning recipes", top_k=5, scope="project:alpha")
        related_ids = [s.id for s in result.related_schemas]
        assert (
            child_id not in related_ids
        ), "stage-0 cross-scope neighbor must not leak into related_schemas"
    finally:
        eng.close()
        _cleanup(path)


# ---------------------------------------------------------------------------
# Co-activation weight normalization (2026-07-23): raw Hebbian weight is an
# unbounded accumulating counter, while content-relation confidence is bounded
# [0, 1]. spread_relation_activation() multiplies this value straight into
# injected activation, so an unnormalized weight would let a well-worn
# co-activation edge dominate the graph-expansion winners regardless of
# topical relevance -- see 20260723_part_of_audit_and_brain_alignment_review.md.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# WP-5: edge-channel filtering (relation_filter) and neighbor-cue relevance
# dual gate (min_neighbor_relevance) -- see plan Phase 3 "Redesign associative
# expansion" and private/experiments/run_graph_channel_sweep.py.
# ---------------------------------------------------------------------------


def _axis(dim: int, index: int) -> np.ndarray:
    v = np.zeros(dim, dtype=np.float32)
    v[index] = 1.0
    return v


def test_relation_filter_excludes_non_matching_edge_type():
    def fetch_relations(schema_id):
        if schema_id == 1:
            return [(2, "relates_to", 0.9), (3, "coactivated_with", 0.9)]
        return []

    winners = spread_relation_activation(
        {1: 0.8},
        fetch_relations=fetch_relations,
        min_activation=0.20,
        relation_filter=frozenset({"relates_to"}),
    )
    assert 2 in winners
    assert 3 not in winners


def test_relation_filter_none_includes_every_edge_type():
    """Default (no filter) is unchanged: every edge type still contributes --
    the pre-WP-5 behavior every other test in this file already exercises."""

    def fetch_relations(schema_id):
        if schema_id == 1:
            return [(2, "relates_to", 0.9), (3, "coactivated_with", 0.9)]
        return []

    winners = spread_relation_activation(
        {1: 0.8}, fetch_relations=fetch_relations, min_activation=0.20
    )
    assert 2 in winners
    assert 3 in winners


def test_expand_via_relations_dual_gate_disabled_by_default():
    """min_neighbor_relevance defaults to 0.0 (disabled) -- an off-topic
    neighbor (cosine 0.0 against the cue) is still admitted purely on graph
    mass, exactly as before WP-5, when the caller doesn't opt into the gate."""
    gate = WorkingMemoryGate()
    cue_embedding = _axis(8, 0)
    seed_schema = _schema(1, embedding=_axis(8, 0))
    neighbor_schema = _schema(2, embedding=_axis(8, 7))  # orthogonal -- cosine 0.0
    state = WorkingMemoryState(
        items=[WorkingMemoryItem(schema=seed_schema, activation=1.0, reason="direct", text="seed")],
        rendered="",
        cue_terms=[],
    )
    expanded = gate.expand_via_relations(
        state,
        fetch_relations=lambda sid: [(2, "relates_to", 0.95)] if sid == 1 else [],
        fetch_schema=lambda sid: neighbor_schema if sid == 2 else None,
        cue=MemoryCue(query="target"),
        policy=GatePolicy(max_items=5, max_chars=10_000),
        cue_embedding=cue_embedding,
        # min_neighbor_relevance omitted -- defaults to 0.0/disabled.
    )
    assert any(i.schema.id == 2 for i in expanded.items)


def test_expand_via_relations_dual_gate_rejects_low_cosine_neighbor():
    gate = WorkingMemoryGate()
    cue_embedding = _axis(8, 0)
    seed_schema = _schema(1, embedding=_axis(8, 0))
    neighbor_schema = _schema(2, embedding=_axis(8, 7))  # orthogonal -- cosine 0.0
    state = WorkingMemoryState(
        items=[WorkingMemoryItem(schema=seed_schema, activation=1.0, reason="direct", text="seed")],
        rendered="",
        cue_terms=[],
    )
    expanded = gate.expand_via_relations(
        state,
        fetch_relations=lambda sid: [(2, "relates_to", 0.95)] if sid == 1 else [],
        fetch_schema=lambda sid: neighbor_schema if sid == 2 else None,
        cue=MemoryCue(query="target"),
        policy=GatePolicy(max_items=5, max_chars=10_000),
        cue_embedding=cue_embedding,
        min_neighbor_relevance=0.25,
    )
    assert not any(i.schema.id == 2 for i in expanded.items)


def test_expand_via_relations_dual_gate_admits_high_cosine_neighbor():
    gate = WorkingMemoryGate()
    cue_embedding = _axis(8, 0)
    seed_schema = _schema(1, embedding=_axis(8, 0))
    neighbor_schema = _schema(2, embedding=_axis(8, 0))  # same axis -- cosine 1.0
    state = WorkingMemoryState(
        items=[WorkingMemoryItem(schema=seed_schema, activation=1.0, reason="direct", text="seed")],
        rendered="",
        cue_terms=[],
    )
    expanded = gate.expand_via_relations(
        state,
        fetch_relations=lambda sid: [(2, "relates_to", 0.95)] if sid == 1 else [],
        fetch_schema=lambda sid: neighbor_schema if sid == 2 else None,
        cue=MemoryCue(query="target"),
        policy=GatePolicy(max_items=5, max_chars=10_000),
        cue_embedding=cue_embedding,
        min_neighbor_relevance=0.25,
    )
    assert any(i.schema.id == 2 for i in expanded.items)


def _schema(schema_id: int, *, embedding: np.ndarray | None):
    from slowave.symbolic.schema_store import Schema

    now = int(time.time())
    return Schema(
        id=schema_id,
        prototype_id=None,
        content_text=f"schema {schema_id}",
        facets={"schema_class": "fact"},
        tags=[],
        scope_id=None,
        status="active",
        confidence=1.0,
        salience=1.0,
        supporting_episode_ids=[],
        contradicting_episode_ids=[],
        is_labile=False,
        first_formed_ts=now,
        last_updated_ts=now,
        embedding=embedding,
    )


def test_context_brief_graph_channels_off_skips_graph_expansion_entirely():
    eng, path = _tmp_engine()
    try:
        parent_id = eng.schemas.create(
            content_text="target deployment policy",
            facets={"schema_class": "fact"},
            tags=["target"],
            embedding=None,
            salience=5.0,
            dedupe=False,
        )
        child_id = eng.schemas.create(
            content_text="unrelated associative detail",
            facets={"schema_class": "fact"},
            tags=[],
            embedding=None,
            salience=0.01,
            scope_id="proj:other",
            dedupe=False,
        )
        eng.schemas.add_relation(
            src_schema_id=min(parent_id, child_id),
            dst_schema_id=max(parent_id, child_id),
            relation="relates_to",
            confidence=0.95,
        )
        brief = eng.context_brief(query="target deployment policy", limit=5, graph_channels="off")
        ids = [item.schema.id for item in brief.items]
        assert parent_id in ids
        assert child_id not in ids
    finally:
        eng.close()
        _cleanup(path)


def test_context_brief_graph_channels_restricts_to_named_relation():
    """graph_channels='relates_to' must not surface a neighbor reachable only
    via a coactivated_with edge."""
    eng, path = _tmp_engine()
    try:
        parent_id = eng.schemas.create(
            content_text="target deployment policy",
            facets={"schema_class": "fact"},
            tags=["target"],
            embedding=None,
            salience=5.0,
            dedupe=False,
        )
        child_id = eng.schemas.create(
            content_text="unrelated associative detail",
            facets={"schema_class": "fact"},
            tags=[],
            embedding=None,
            salience=0.01,
            dedupe=False,
        )
        now = int(time.time())
        conn = eng.db.connect()
        # Raw weight, not confidence: _fetch_all_relations() normalizes via
        # w/(w+1) (see its docstring), so a large raw weight is needed for
        # the normalized value to approach 1.0 the way a direct relates_to
        # confidence of 0.95 would -- a small raw value here would fall below
        # the graph min_activation floor after hop decay for reasons
        # unrelated to this test's channel-filtering assertion.
        conn.execute(
            "INSERT OR REPLACE INTO schema_coactivation "
            "(src_schema_id, dst_schema_id, weight, last_touched_ts) VALUES (?, ?, ?, ?)",
            (parent_id, child_id, 20.0, now),
        )
        conn.commit()

        brief_relates_only = eng.context_brief(
            query="target deployment policy", limit=5, graph_channels="relates_to"
        )
        assert child_id not in [item.schema.id for item in brief_relates_only.items]

        brief_combined = eng.context_brief(
            query="target deployment policy", limit=5, graph_channels="combined"
        )
        assert child_id in [item.schema.id for item in brief_combined.items]
    finally:
        eng.close()
        _cleanup(path)


def test_recall_graph_channels_off_returns_no_related_schemas():
    eng, path = _tmp_engine()
    eng.encoder = _StubEncoder(dim=8)
    try:
        parent_id = eng.schemas.create(
            content_text="For meal planning, the user prefers vegetarian recipes.",
            facets={"schema_class": "preference"},
            tags=["food", "meal_planning", "vegetarian"],
            embedding=None,
            salience=5.0,
            dedupe=False,
        )
        child_id = eng.schemas.create(
            content_text="Specifically, the user avoids mushrooms in vegetarian dishes.",
            facets={"schema_class": "fact"},
            tags=["unrelated_tag"],
            embedding=None,
            salience=0.01,
            dedupe=False,
        )
        eng.schemas.add_relation(
            src_schema_id=min(child_id, parent_id),
            dst_schema_id=max(child_id, parent_id),
            relation="relates_to",
            confidence=0.95,
        )

        result = eng.recall("vegetarian meal planning recipes", top_k=5, graph_channels="off")
        assert result.related_schemas == []
    finally:
        eng.close()
        _cleanup(path)


def test_recall_dual_gate_rejects_offtopic_graph_neighbor():
    """A neighbor reachable via a strong coactivated_with edge but with an
    embedding orthogonal to the query must not surface in related_schemas
    once min_neighbor_relevance is enabled -- the plan's "association is
    mistaken for answer confidence" defect (graph_hub_saturation /
    budget_graph_overflow replay cases)."""
    eng, path = _tmp_engine()
    dim = 8
    q_vec = _axis(dim, 0)
    off_topic_vec = _axis(dim, 7)

    class FixedEncoder:
        def encode(self, text: str) -> np.ndarray:
            return q_vec.copy()

    eng.encoder = FixedEncoder()
    try:
        seed_id = eng.schemas.create(
            content_text="seed on-topic schema",
            facets={"schema_class": "fact"},
            tags=[],
            embedding=q_vec,
            salience=1.0,
            dedupe=False,
        )
        neighbor_id = eng.schemas.create(
            content_text="off-topic neighbor schema",
            facets={"schema_class": "fact"},
            tags=[],
            embedding=off_topic_vec,
            salience=1.0,
            dedupe=False,
        )
        now = int(time.time())
        conn = eng.db.connect()
        conn.execute(
            "INSERT OR REPLACE INTO schema_coactivation "
            "(src_schema_id, dst_schema_id, weight, last_touched_ts) VALUES (?, ?, ?, ?)",
            (seed_id, neighbor_id, 5.0, now),
        )
        conn.commit()

        # WP-5.1 (2026-07-29) flipped the production graph_channels default
        # to "off" -- this test exercises the dual-gate mechanics
        # specifically, so it opts back into the graph channel explicitly.
        without_gate = eng.recall(
            "seed query", top_k=5, graph_channels="combined", min_neighbor_relevance=0.0
        )
        assert neighbor_id in [s.id for s in without_gate.related_schemas]

        with_gate = eng.recall(
            "seed query", top_k=5, graph_channels="combined", min_neighbor_relevance=0.25
        )
        assert neighbor_id not in [s.id for s in with_gate.related_schemas]
    finally:
        eng.close()
        _cleanup(path)


def test_fetch_all_relations_normalizes_coactivation_weight_below_one():
    eng, path = _tmp_engine()
    try:
        id_a = eng.schemas.create(
            content_text="A", facets={}, tags=[], embedding=None, dedupe=False
        )
        id_b = eng.schemas.create(
            content_text="B", facets={}, tags=[], embedding=None, dedupe=False
        )
        now = int(time.time())
        # Repeated same-instant upserts push weight well past 1.0 (each adds
        # +1.0 with zero decay at dt=0), simulating a well-worn pair.
        for _ in range(10):
            eng.schemas.upsert_coactivation(id_a, id_b, now_ts=now)
        raw_weight = eng.schemas.get_coactivations(id_a)[0][2]
        assert raw_weight > 5.0, "sanity check: raw weight really exceeds confidence's ceiling"

        fetched = eng._retrieval._fetch_all_relations(id_a)
        coact_entries = [t for t in fetched if t[1] == "coactivated_with"]
        assert len(coact_entries) == 1
        neighbor_id, _relation, normalized = coact_entries[0]
        assert neighbor_id == id_b
        assert 0.0 < normalized < 1.0, "normalized weight must stay within [0, 1)"
    finally:
        eng.close()
        _cleanup(path)
