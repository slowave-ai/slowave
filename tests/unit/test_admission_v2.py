"""Calibrated similarity-gated admission (admission v2) regression tests."""

import numpy as np
import pytest

from slowave import ops
from slowave.core import admission_v2
from slowave.core.admission_v2 import Candidate
from slowave.core.config import SlowaveConfig
from slowave.core.engine import SlowaveEngine


def test_weak_pool_abstains_and_strong_pool_is_not_capped_at_one():
    weak = np.array([0.30, 0.25, 0.2])
    assert admission_v2.admit(weak, [100, 100, 100]) == []
    strong = np.array([0.9, 0.8, 0.7, 0.1])
    assert admission_v2.admit(strong, [100] * 4) == [0, 1, 2]


def test_item_cap(monkeypatch):
    monkeypatch.setattr(admission_v2, "MAX_ITEMS", 4)
    assert len(admission_v2.admit(np.array([0.9] * 8), [10] * 8)) == 4


def test_features_are_deterministic_and_pool_relative():
    evidence = ({"dense_cosine": 0.7, "dense_rank": 1, "lexical_rank": 2, "rrf_score": 0.9},)
    pool = [Candidate(i, f"alpha beta gamma {i}", (0,), evidence, "lesson") for i in range(4)]
    a = admission_v2.pool_features(
        pool, need_texts=["alpha beta"], user_need_count=1, e5=None, generic=frozenset()
    )
    b = admission_v2.pool_features(
        pool, need_texts=["alpha beta"], user_need_count=1, e5=None, generic=frozenset()
    )
    assert np.array_equal(a, b) and a.shape == (4, len(admission_v2.FEATURES))


def _stub_engine(tmp_path):
    """Engine with a deterministic fake encoder and an overrideable dense search."""

    class Encoder:
        def __init__(self):
            self.last_text = ""

        def encode(self, text):
            self.last_text = text
            return np.zeros(8, dtype=np.float32)

        def encode_many(self, texts):
            return np.stack([self.encode(text) for text in texts])

        def embedding_spec(self):
            return {"model": "stub", "revision": "1", "version": 1}

    engine = SlowaveEngine(
        SlowaveConfig(db_path=str(tmp_path / "admission.db"), dim=8, disable_encoder=True)
    )
    encoder = Encoder()
    engine._retrieval.encoder = encoder
    engine._retrieval._multilingual_index = lambda *, scope, mode: None
    texts: dict[int, str] = {}

    def _dense(vector, limit, scope_id=None):
        return [(sid, 0.5) for sid in texts]

    engine.schemas.search_embedding = _dense
    engine.schemas.search_fts_candidates = lambda *args, **kwargs: []

    def _seed(content, *, facets=None):
        memory = engine.schemas.create(
            content_text=content,
            facets=facets if facets is not None else {},
            embedding=None,
            scope_id="project:test",
        )
        texts[memory] = content
        return memory

    engine._seed = _seed
    engine._texts = texts
    return engine


def _engine_with_cosines(tmp_path, texts_cos):
    engine = _stub_engine(tmp_path)
    ids = [
        engine._seed(text, facets={"source_kind": "explicit_remember", "schema_class": "lesson"})
        for text in texts_cos
    ]
    cosines = dict(zip(ids, texts_cos.values()))
    engine.schemas.search_embedding = lambda vector, limit, scope_id=None: sorted(
        cosines.items(), key=lambda kv: -kv[1]
    )
    return engine, ids


class _CountingScorer:
    def __init__(self, value=-5.0):
        self.scored = 0
        self.calls = 0
        self.value = value

    def score(self, queries, memories):
        self.calls += 1
        self.scored += len(memories)
        return np.full((len(memories), len(queries)), self.value)


def test_unrelated_task_over_weak_pool_returns_at_most_one(tmp_path):
    texts = {f"Memory number {i} about unrelated billing ledger topic {i}": 0.3 for i in range(12)}
    engine, _ = _engine_with_cosines(tmp_path, texts)
    engine._retrieval._applicability_scorer = _CountingScorer(-7.0)
    try:
        result = ops.activate(
            engine,
            query="Write a poem about the ocean at night",
            scope="project:test",
            relevant_set=True,
            complementary_activation=True,
            include_schemas=True,
        )
        assert len(result["schemas"]) <= 1
    finally:
        engine.close()


def test_admission_is_deterministic_and_needs_no_pair_scorer(tmp_path):
    """The calibrated probability is the final score; no second model is consulted."""
    texts = {f"Retry queue worker lesson {i} idempotency key batch": 0.55 for i in range(30)}
    engine, _ = _engine_with_cosines(tmp_path, texts)
    scorer = _CountingScorer(-3.0)
    engine._retrieval._applicability_scorer = scorer
    try:
        query = "Two workers both processed the same batch of retry queue work"
        first = ops.activate(
            engine,
            query=query,
            scope="project:test",
            relevant_set=True,
            complementary_activation=True,
            include_schemas=True,
        )
        assert scorer.scored == 0, "admission must not depend on a second scorer"
        again = ops.activate(
            engine,
            query=query,
            scope="project:test",
            relevant_set=True,
            complementary_activation=True,
            include_schemas=True,
        )
        assert [r["id"] for r in first["schemas"]] == [r["id"] for r in again["schemas"]]
    finally:
        engine.close()


def test_strong_pool_delivers_the_need_matching_memory_without_a_model(tmp_path):
    """Model-free runtimes keep one answer per declared need, unfiltered.

    Multi-item delivery from a strong pool is the calibrated admission contract
    and is pinned by ``test_weak_pool_abstains_and_strong_pool_is_not_capped_at_one``;
    a runtime without an encoder keeps the documented lexical contract instead.
    """
    texts = {
        "Refund endpoint must use the idempotency header for refunds.": 0.85,
        "Refund amounts are integer minor units, never floats.": 0.8,
        "Refund webhook retries use exponential backoff.": 0.75,
        "Marketing banner color is teal.": 0.1,
    }
    engine, _ = _engine_with_cosines(tmp_path, texts)
    engine._retrieval._applicability_scorer = _CountingScorer(1.0)
    try:
        result = ops.activate(
            engine,
            query="Add a refund endpoint with idempotency header and minor units and retries",
            scope="project:test",
            relevant_set=True,
            complementary_activation=True,
            include_schemas=True,
        )
        delivered = [row["id"] for row in result["schemas"]]
        assert delivered, "a matching memory must still be delivered"
        assert all("banner" not in (row.get("text") or "") for row in result["schemas"])
    finally:
        engine.close()


def test_other_scope_memory_never_leaks(tmp_path):
    engine, ids = _engine_with_cosines(tmp_path, {"Refund endpoint idempotency header rule.": 0.9})
    other = engine.schemas.create(
        content_text="Refund endpoint idempotency header rule in another project.",
        facets={"source_kind": "explicit_remember"},
        embedding=None,
        scope_id="project:other",
    )
    try:
        result = ops.activate(
            engine,
            query="Add a refund endpoint with an idempotency header",
            scope="project:test",
            relevant_set=True,
            complementary_activation=True,
            include_schemas=True,
        )
        assert f"sch_{other}" not in [r["id"] for r in result["schemas"]]
    finally:
        engine.close()


def test_terms_are_accent_folded_and_stemmed_across_languages() -> None:
    from slowave.core import admission_v2 as av

    assert av._tokens("aggiungi il campo importo") & av._tokens("Tutti gli importi monetari")
    assert av._tokens("Abhängigkeit") & av._tokens("abhangigkeiten")
    assert "the" not in av._tokens("the cat", frozenset({"the"}))
    assert av._tokens("コミット") == {"コミ", "ミッ", "ット"}
