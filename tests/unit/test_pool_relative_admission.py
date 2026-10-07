"""Pool-relative admission: default behaviors and pinned invariants.

The tests pin the design invariants: pool-relative floors admit a relevant
cluster and abstain on all-noise pools, facet decomposition keeps the
whole-task fallback need, and consolidation-derived episode artifacts without
source provenance stay out of declarative delivery. The floor and margin stay
tunable through their environment overrides.
"""

import numpy as np
import pytest

from slowave import ops
from slowave.core import pool_relative
from slowave.core.activation_selection import ActivationTask
from slowave.core.applicability_ranking import applicability_order, query_needs
from slowave.core.config import SlowaveConfig
from slowave.core.engine import SlowaveEngine
from slowave.core.services.retrieval import RetrievalService

MULTI_CLAUSE_TASK = (
    "Deep clean the espresso machine, backflush the group, then re-dial the"
    " grinder for the new light roast bag, and refresh the daily espresso recipe"
)


def test_floor_and_margin_overrides(monkeypatch):
    assert pool_relative.signal_floor() == pytest.approx(-6.5)
    assert pool_relative.margin() == pytest.approx(1.0)
    monkeypatch.setenv("SLOWAVE_APPLICABILITY_SIGNAL_FLOOR", "-7.25")
    monkeypatch.setenv("SLOWAVE_APPLICABILITY_MARGIN", "0.5")
    assert pool_relative.signal_floor() == pytest.approx(-7.25)
    assert pool_relative.margin() == pytest.approx(0.5)
    monkeypatch.setenv("SLOWAVE_APPLICABILITY_MARGIN", "-1")
    with pytest.raises(ValueError):
        pool_relative.margin()


def test_facet_decomposition_creates_single_topic_needs():
    task = ActivationTask.build(MULTI_CLAUSE_TASK)
    assert task.needs[0].text == MULTI_CLAUSE_TASK
    facet_needs = [need for need in task.needs[1:] if "task_facet" in need.provenance]
    assert facet_needs, "short single-topic facet needs must be created"
    assert all(len(need.text) < len(MULTI_CLAUSE_TASK) for need in facet_needs)
    assert len(task.needs) <= 10


def test_facet_decomposition_respects_budget_and_fragment_floor():
    long_task = "; ".join(f"maintain workstation {number} backup" for number in range(12))
    task = ActivationTask.build(long_task)
    facet_needs = [need for need in task.needs[1:] if "task_facet" in need.provenance]
    assert len(facet_needs) == 8
    assert task.needs[0].text == long_task


def test_query_needs_newline_split():
    joined = "Dial in the espresso grinder\nRefresh the daily espresso recipe"
    assert len(query_needs(joined)) == 2


def test_pool_relative_floors_admit_cluster_and_abstain_all_noise():
    scores = np.array(
        [
            [-3.0, -9.0],
            [-3.4, -9.0],
            [-9.0, -2.5],
            [-9.0, -4.0],
        ]
    )
    order = applicability_order(
        scores, minimum_logit=-6.5, maximum_logit_gap=1.0, retain_weak_support=False
    )
    # Column 0 best -3.0 -> floor -4.0: rows 0 and 1 qualify; column 1 best
    # -2.5 -> floor -3.5: row 2 qualifies, row 3 (-4.0) does not.
    assert set(order) == {0, 1, 2}

    all_noise = np.full((4, 1), -7.5)
    assert (
        applicability_order(
            all_noise, minimum_logit=-6.5, maximum_logit_gap=1.0, retain_weak_support=False
        )
        == []
    ), "an all-noise pool whose best sits below the signal floor must abstain"


def test_policy_version_identities():
    assert (
        ops._retrieval_policy_version(
            relevant_set=True, complementary_activation=True, facet_mode=False
        )
        == "activation-pool-relative-v1"
    )
    assert (
        ops._retrieval_policy_version(
            relevant_set=True, complementary_activation=False, facet_mode=False, recall=True
        )
        == "recall-pool-relative-v1"
    )


def _stub_engine(tmp_path):
    class Encoder:
        def __init__(self):
            self.last_text = ""

        def encode(self, text):
            self.last_text = text
            return np.zeros(8, dtype=np.float32)

    generic = {
        "a",
        "an",
        "the",
        "and",
        "or",
        "to",
        "of",
        "for",
        "in",
        "on",
        "at",
        "by",
        "with",
        "from",
        "is",
        "are",
        "be",
        "this",
        "that",
        "it",
        "my",
        "me",
        "i",
        "how",
        "what",
        "should",
        "help",
        "all",
        "0",
        "1:2",
    }

    def content_words(text):
        return {word for word in text.casefold().split() if word not in generic}

    class Scorer:
        """Cross-encoder stand-in: paraphrases score weakly but inside margin."""

        def score(self, queries, memories):
            matrix = np.full((len(memories), len(queries)), -8.0)
            for column, query in enumerate(queries):
                query_words = content_words(query)
                for row, text in enumerate(memories):
                    overlap = len(query_words & content_words(text))
                    if overlap >= 2:
                        matrix[row, column] = 1.0
                    elif overlap == 1:
                        matrix[row, column] = -1.0
            return matrix

    engine = SlowaveEngine(
        SlowaveConfig(db_path=str(tmp_path / "pool_relative.db"), dim=8, disable_encoder=True)
    )
    encoder = Encoder()
    engine._retrieval.encoder = encoder
    engine._retrieval._applicability_scorer = Scorer()
    engine._retrieval._multilingual_index = lambda *, scope, mode: None
    texts: dict[int, str] = {}

    def _dense(vector, limit, scope_id=None):
        # 0.5: above the 0.20 discovery floor but below the 0.60 semantic
        # contribution floor, so admission must flow through the applicability
        # assessment; 0.05 keeps unrelated candidates out of the pool.
        return [
            (sid, 0.5 if content_words(text) & content_words(encoder.last_text) else 0.05)
            for sid, text in texts.items()
        ]

    engine.schemas.search_embedding = _dense
    engine.schemas.search_fts_candidates = lambda *args, **kwargs: []
    engine._pool_relative_ids = ids = []
    engine._pool_relative_texts = texts

    def _seed(content, *, facets=None, episodes=None):
        memory = engine.schemas.create(
            content_text=content,
            facets=facets if facets is not None else {},
            embedding=None,
            scope_id="project:test",
            supporting_episode_ids=episodes,
        )
        ids.append(memory)
        texts[memory] = content
        return memory

    engine._seed = _seed
    return engine


def test_pool_relative_activate_surfaces_paraphrase(tmp_path):
    """Symptom paraphrase with sub-0.60 cosine admits via relative floors.

    The memory shares one content word with the query, so discovery passes
    (cosine 0.5 > 0.20) but the legacy semantic route (0.60) and lexical
    routes cannot admit it; only the pool-relative assessment floor can.
    """
    engine = _stub_engine(tmp_path)
    try:
        recipe = engine._seed(
            "The household daily espresso recipe is an 18 g dose with a 1:2 ratio.",
            facets={"source_kind": "explicit_remember"},
        )
        result = ops.activate(
            engine,
            query="My morning drink from the home espresso machine tastes sour and thin",
            scope="project:test",
            relevant_set=True,
            complementary_activation=True,
            include_schemas=True,
        )
        assert f"sch_{recipe}" in [row["id"] for row in result["schemas"]], result
    finally:
        engine.close()


def test_pool_relative_activate_excludes_episode_artifact(tmp_path):
    engine = _stub_engine(tmp_path)
    try:
        recipe = engine._seed(
            "The household daily espresso recipe is an 18 g dose with a 1:2 ratio.",
            facets={"source_kind": "explicit_remember"},
        )
        artifact = engine._seed(
            "Service the espresso machine activation probes: all 0 hits recorded.",
            facets={},
            episodes=[7],
        )
        result = ops.activate(
            engine,
            query="Service the espresso machine and check the daily dose",
            scope="project:test",
            relevant_set=True,
            complementary_activation=True,
            include_schemas=True,
        )
        exposed = {row["id"] for row in result["schemas"]}
        assert f"sch_{recipe}" in exposed, result
        assert (
            f"sch_{artifact}" not in exposed
        ), "episode artifact without source provenance must be excluded"
    finally:
        engine.close()


def test_pool_relative_activate_abstains_on_unrelated_task(tmp_path):
    engine = _stub_engine(tmp_path)
    try:
        engine._seed(
            "The household daily espresso recipe is an 18 g dose with a 1:2 ratio.",
            facets={"source_kind": "explicit_remember"},
        )
        result = ops.activate(
            engine,
            query="Write a poem about the ocean at night",
            scope="project:test",
            relevant_set=True,
            complementary_activation=True,
            include_schemas=True,
        )
        assert result["relevant_total"] == 0
        assert result["schemas"] == []
    finally:
        engine.close()


def test_pool_relative_recall_uses_relative_floors(tmp_path):
    engine = _stub_engine(tmp_path)
    try:
        memory = engine._seed(
            "Rest light roasts seven days before brewing; naturals are ready after three days.",
            facets={"source_kind": "explicit_remember"},
        )
        result = ops.recall(
            engine,
            query="How long should lightly roasted coffee rest before brewing?",
            scope="project:test",
            relevant_set=True,
        )
        assert f"sch_{memory}" in [row["id"] for row in result["memories"]]
        assert result["retrieval_policy_version"] == "recall-pool-relative-v1"
    finally:
        engine.close()


def test_delivery_facets_mark_episode_derived_without_source():
    schema = type("SchemaStub", (), {})()
    schema.facets = {}
    schema.supporting_episode_ids = [3, 4]
    assert RetrievalService._delivery_facets(None, schema)["episode_derived"] is True

    schema.facets = {"source_kind": "explicit_remember"}
    assert "episode_derived" not in RetrievalService._delivery_facets(None, schema)

    schema.facets = {}
    schema.supporting_episode_ids = []
    assert "episode_derived" not in RetrievalService._delivery_facets(None, schema)
