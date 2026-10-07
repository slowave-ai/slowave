from dataclasses import replace

import pytest

from slowave.core.config import SlowaveConfig
from slowave.core.engine import SlowaveEngine
from slowave.core.hybrid_selection import HybridConfig, admit, discover


def row(sid, dense=None, lexical=None, cosine=0.7, rare=0):
    return dict(
        memory_id=sid,
        dense_rank=dense,
        lexical_rank=lexical,
        cosine=cosine,
        rare_terms=rare,
        eligible=True,
    )


def test_semantic_weight_changes_bounded_pool_without_extra_query_votes():
    query = [row(1, dense=1), row(2, lexical=1)]
    cfg = HybridConfig(depth=1, dense_weight=2, lexical_weight=1)
    pool = discover([query], cfg)
    assert list(pool) == [1]
    assert discover([query, query], cfg) == pool
    assert list(discover([query], replace(cfg, dense_weight=0.5))) == [2]


def test_admission_cannot_be_created_by_high_rrf_rank():
    cfg = HybridConfig(cosine_floor=0.6)
    pool = discover([[row(1, dense=1, cosine=0.2, rare=2), row(2, dense=2, cosine=0.65)]], cfg)
    assert admit(pool, cfg) == [2]
    assert admit(pool, replace(cfg, lexical_rare_terms=2)) == [1, 2]


def test_goal_can_independently_supply_admission_and_rank():
    cfg = HybridConfig()
    task = [row(1, dense=1, cosine=0.2)]
    goal = [row(1, dense=2, cosine=0.7), row(2, dense=1, cosine=0.8)]
    assert admit(discover([task, goal], cfg), cfg) == [1, 2]
    assert admit(discover([task, goal], replace(cfg, query_sources="task")), cfg) == []


def test_native_bm25_weight_preserves_filters_and_default_scores(tmp_path):
    engine = SlowaveEngine(SlowaveConfig(db_path=str(tmp_path / "test.db")))
    try:
        own = engine.schemas.create(
            content_text="quartz quartz guidance", embedding=None, scope_id="project:a"
        )
        engine.schemas.create(
            content_text="quartz other project", embedding=None, scope_id="project:b"
        )
        engine.schemas.create(
            content_text="quartz stale advice", embedding=None, scope_id="project:a", status="stale"
        )
        default = engine.schemas.search_fts_candidates("quartz", scope_id="project:a")
        assert default == engine.schemas.search_fts_candidates(
            "quartz", scope_id="project:a", content_weight=1
        )
        assert [r[0] for r in default] == [own]
        weighted = engine.schemas.search_fts_candidates(
            "quartz", scope_id="project:a", content_weight=4
        )
        assert [r[0] for r in weighted] == [own]
        assert weighted[0][1] != default[0][1]
        for invalid in (0, -1, float("nan"), float("inf")):
            with pytest.raises(ValueError):
                engine.schemas.search_fts_candidates("quartz", content_weight=invalid)
    finally:
        engine.close()


def test_dogfood_adapter_uses_task_only_caps_one_and_bypasses_assessor(tmp_path, monkeypatch):
    import numpy as np

    from slowave.core.activation_selection import ActivationTask

    class Encoder:
        dim = 384
        queries = []

        def encode(self, text):
            self.queries.append(text)
            vector = np.zeros(384, dtype=np.float32)
            vector[0] = 1
            return vector

    encoder = Encoder()
    engine = SlowaveEngine(
        SlowaveConfig(db_path=str(tmp_path / "dogfood.db")), shared_encoder=encoder
    )
    monkeypatch.setenv("SLOWAVE_ACTIVATE_HYBRID", "1")
    try:
        for text in ("quartz guidance alpha", "quartz guidance beta"):
            engine.schemas.create(
                content_text=text, embedding=encoder.encode(text), scope_id="project:a"
            )
        encoder.queries.clear()
        engine._retrieval._shared_scorer = lambda: pytest.fail("dogfood must bypass crossencoder")
        raw_task = "quartz guidance and where should I look?"
        task = ActivationTask.build(raw_task, "A different goal")
        catalog = engine._retrieval.relevant_catalog(
            task.cues, scope="project:a", activation_task=task, hybrid_task=raw_task
        )
        assert len(catalog.items) == 1
        assert catalog.truncated
        assert set(encoder.queries) == {raw_task}
    finally:
        engine.close()
