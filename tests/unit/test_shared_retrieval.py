import json

import numpy as np
import pytest

from slowave import ops
from slowave.core.config import SlowaveConfig
from slowave.core.engine import SlowaveEngine
from slowave.core.shared_retrieval import SharedRequest, assess, source_windows
from slowave.mcp.activation_catalog import FrozenActivationCatalog


def test_discovery_does_not_split_languages_or_duplicate_query_votes():
    request = SharedRequest("Controlla frontend e grafico", " CONTROLLA frontend E grafico ")
    assert request.queries == ["Controlla frontend e grafico"]
    assert SharedRequest("Task", "Different goal").queries == ["Task", "Different goal"]


def test_one_assessment_preserves_late_source_and_preceding_condition():
    class Scorer:
        calls = []

        def score(self, queries, memories):
            self.calls.append((queries, memories))
            return np.array([[8 if "required-command" in text else -8] for text in memories])

    text = (
        "Earlier unrelated information. " * 60 + "If production is enabled. Use required-command."
    )
    scorer = Scorer()
    scores, spans = assess(
        scorer, SharedRequest("Verify", "Production", context={"env": "prod"}), [text]
    )
    assert scores == [8]
    assert len(scorer.calls) == 1
    excerpt = text[slice(*spans[0])]
    assert "If production is enabled." in excerpt and "required-command" in excerpt
    assert len(excerpt) <= 1024
    assert "Goal: Production" in scorer.calls[0][0][0]


def test_oversized_sentence_is_not_silently_cut():
    text = "Unless " + "x" * 1100 + ", do not deploy."
    assert source_windows(text) == [(0, len(text))]


@pytest.mark.parametrize("value", ["nan", "inf", "-inf"])
def test_invalid_thresholds_are_rejected(monkeypatch, value):
    monkeypatch.setenv("SLOWAVE_ACTIVATE_RELEVANCE_LOGIT", value)
    with pytest.raises(ValueError):
        _ = SharedRequest("task").threshold


def test_shared_activate_and_recall_use_same_assessor_with_absolute_thresholds(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("SLOWAVE_RETRIEVAL_PIPELINE", "shared-v1")
    monkeypatch.setenv("SLOWAVE_ACTIVATE_RELEVANCE_LOGIT", "0")
    monkeypatch.setenv("SLOWAVE_RECALL_RELEVANCE_LOGIT", "-2")

    class Scorer:
        calls = []

        def score(self, queries, memories):
            self.calls.append(queries)
            return np.array(
                [[8 if "strong" in text else 1 if "support" in text else -1] for text in memories]
            )

    engine = SlowaveEngine(
        SlowaveConfig(db_path=str(tmp_path / "shared.db"), dim=8, disable_encoder=True)
    )
    try:
        ids = [
            engine.schemas.create(
                content_text=text, facets={}, embedding=None, scope_id="project:test", dedupe=False
            )
            for text in (
                "Cache strong evidence.",
                "Cache useful support.",
                "Cache weak evidence.",
                "Goal-only artifact.",
            )
        ]
        other = engine.schemas.create(
            content_text="Cache strong evidence.",
            facets={},
            embedding=None,
            scope_id="project:other",
            dedupe=False,
        )
        scorer = Scorer()
        engine._retrieval._applicability_scorer = scorer
        activated = ops.activate(
            engine,
            query="Cache",
            initial_goal="Goal-only",
            scope="project:test",
            relevant_set=True,
            complementary_activation=True,
            include_schemas=True,
        )
        assert {row["id"] for row in activated["schemas"]} == {f"sch_{ids[0]}", f"sch_{ids[1]}"}
        assert f"sch_{other}" not in {row["id"] for row in activated["schemas"]}
        assert len(scorer.calls) == 1  # Empty procedure catalog performs no scoring.
        assert "Goal: Goal-only" in scorer.calls[0][0]
        recalled = ops.recall(
            engine,
            query="Cache",
            session_id=activated["session_id"],
            scope="project:test",
            relevant_set=True,
        )
        assert {row["id"] for row in recalled["memories"]} == {f"sch_{i}" for i in ids}
        assert len(scorer.calls) == 2
        assert scorer.calls[0] == scorer.calls[1]
        assert (
            activated["retrieval_policy_version"]
            == recalled["retrieval_policy_version"]
            == "shared-multilingual-v1"
        )
    finally:
        engine.close()


def test_shared_cursor_policy_is_frozen_independent_of_runtime_switch():
    catalog = FrozenActivationCatalog.prepare(
        retrieval_id="ctx_test",
        session_id="sess_test",
        scope="project:test",
        memories=[{"memory_id": f"sch_{i}", "content": "x" * 1000} for i in range(25)],
        procedures=[],
        contribution_spans={},
        catalog_truncated=False,
        policy_version="shared-multilingual-v1",
    )
    reread = FrozenActivationCatalog.read(catalog.snapshot_json)
    first, cursor = reread.page()
    assert first["retrieval_policy_version"] == "shared-multilingual-v1"
    assert cursor is not None
    tail, _ = reread.page(*cursor)
    assert not {x["memory_id"] for x in first["memories"]} & {
        x["memory_id"] for x in tail["memories"]
    }
    assert (
        json.loads(catalog.snapshot_json)["originating_policy_version"] == "shared-multilingual-v1"
    )


def test_shared_admission_does_not_require_lexical_or_cosine_floor(tmp_path):
    class Encoder:
        def encode(self, text):
            return np.zeros(8, dtype=np.float32)

    class Scorer:
        def score(self, queries, memories):
            return np.full((len(memories), 1), 2.0)

    engine = SlowaveEngine(
        SlowaveConfig(db_path=str(tmp_path / "low-cosine.db"), dim=8, disable_encoder=True)
    )
    try:
        memory = engine.schemas.create(
            content_text="A useful fact in another language.",
            facets={},
            embedding=None,
            scope_id="project:test",
        )
        engine._retrieval.encoder = Encoder()
        engine._retrieval._applicability_scorer = Scorer()
        engine.schemas.search_fts_candidates = lambda *args, **kwargs: []
        engine.schemas.search_embedding = lambda *args, **kwargs: [(memory, 0.01)]
        catalog = engine.relevant_catalog(
            ["Controlla il frontend"],
            scope="project:test",
            shared_request=SharedRequest("Controlla il frontend"),
        )
        assert [item.schema.id for item in catalog.items] == [memory]
    finally:
        engine.close()


def test_multilingual_discovery_channel_recovers_cross_language_candidates(tmp_path, monkeypatch):
    class StubRetrievalEncoder:
        def query(self, text):
            return np.array([1.0, 0, 0, 0], dtype=np.float32)

        def passages(self, texts):
            return np.tile(np.array([1.0, 0, 0, 0], dtype=np.float32), (len(texts), 1))

    class Scorer:
        def score(self, queries, memories):
            return np.full((len(memories), 1), 2.0)

    monkeypatch.setenv("SLOWAVE_RETRIEVAL_PIPELINE", "shared-v1")
    monkeypatch.setattr(
        "slowave.symbolic.retrieval_encoder.get_retrieval_encoder",
        lambda: StubRetrievalEncoder(),
    )
    engine = SlowaveEngine(
        SlowaveConfig(db_path=str(tmp_path / "e5-channel.db"), dim=8, disable_encoder=True)
    )
    try:
        memory = engine.schemas.create(
            content_text="Una memoria utile in un'altra lingua.",
            facets={},
            embedding=None,
            scope_id="project:test",
        )
        engine._retrieval.encoder = None
        engine._retrieval._applicability_scorer = Scorer()
        engine.schemas.search_fts_candidates = lambda *args, **kwargs: []
        engine.schemas.search_embedding = lambda *args, **kwargs: []
        catalog = engine.relevant_catalog(
            ["Una domanda completamente diversa"],
            scope="project:test",
            shared_request=SharedRequest("Una domanda completamente diversa"),
        )
        assert [item.schema.id for item in catalog.items] == [memory]
        evidence = [d for d in catalog.decisions if d.memory_id == memory]
        assert evidence and any(
            entry.get("e5_cosine") is not None for entry in evidence[0].channel_evidence
        )
    finally:
        engine.close()


def test_multilingual_channel_failure_degrades_to_stored_channels(tmp_path, monkeypatch):
    class Encoder:
        def encode(self, text):
            return np.zeros(8, dtype=np.float32)

    class Scorer:
        def score(self, queries, memories):
            return np.full((len(memories), 1), 2.0)

    def unavailable():
        raise RuntimeError("retrieval encoder assets unavailable")

    monkeypatch.setenv("SLOWAVE_RETRIEVAL_PIPELINE", "shared-v1")
    monkeypatch.setattr("slowave.symbolic.retrieval_encoder.get_retrieval_encoder", unavailable)
    engine = SlowaveEngine(
        SlowaveConfig(db_path=str(tmp_path / "e5-fallback.db"), dim=8, disable_encoder=True)
    )
    try:
        memory = engine.schemas.create(
            content_text="A useful stored fact.",
            facets={},
            embedding=None,
            scope_id="project:test",
        )
        engine._retrieval.encoder = Encoder()
        engine._retrieval._applicability_scorer = Scorer()
        engine.schemas.search_fts_candidates = lambda *args, **kwargs: []
        engine.schemas.search_embedding = lambda *args, **kwargs: [(memory, 0.9)]
        catalog = engine.relevant_catalog(
            ["A stored fact"],
            scope="project:test",
            shared_request=SharedRequest("A stored fact"),
        )
        assert [item.schema.id for item in catalog.items] == [memory]
    finally:
        engine.close()


def test_shared_procedures_do_not_require_english_action_words(tmp_path):
    class Scorer:
        def score(self, queries, memories):
            # -8.0 stays below the shared activate default floor (-1.25).
            return np.array([[3.0], [-8.0]])

    engine = SlowaveEngine(
        SlowaveConfig(db_path=str(tmp_path / "procedures.db"), dim=8, disable_encoder=True)
    )
    try:
        engine._retrieval._applicability_scorer = Scorer()
        found = engine._retrieval.shared_procedures(
            SharedRequest("Verifica il frontend"),
            [
                {"id": "proc_a", "goal": "Frontend checks", "summary": "Useful method"},
                {"id": "proc_b", "goal": "Deployment", "summary": "Unrelated method"},
            ],
        )
        assert [procedure["id"] for procedure in found] == ["proc_a"]
    finally:
        engine.close()


def _tied_engine(tmp_path, score: float = 2.0):
    class Scorer:
        def score(self, queries, texts):
            return np.array([[score] for _ in texts])

    engine = SlowaveEngine(
        SlowaveConfig(db_path=str(tmp_path / "tied.db"), dim=8, disable_encoder=True)
    )
    engine._retrieval._applicability_scorer = Scorer()
    return engine


def test_shared_procedures_helped_resolves_relevance_tie(tmp_path):
    """One helpful report lifts an equally applicable procedure above its peer.

    The plain procedure is named so it sorts strictly before the helped one
    alphabetically; with utility disabled the plain peer would win, so this
    test cannot pass by incidental ID ordering.
    """

    engine = _tied_engine(tmp_path)
    try:
        found = engine._retrieval.shared_procedures(
            SharedRequest("Verify the release checks"),
            [
                {"id": "proc_a_plain", "goal": "Release checks", "summary": "Plain method"},
                {
                    "id": "proc_z_helped",
                    "goal": "Release checks",
                    "summary": "Rewarded method",
                    "evidence": {"helped": 1},
                },
            ],
        )
        assert [procedure["id"] for procedure in found] == ["proc_z_helped", "proc_a_plain"]
        # The public payload still carries the pure relevance logit.
        assert found[0]["match"] == {"admission": "shared_relevance", "logit": 2.0}
    finally:
        engine.close()


def test_helped_ordering_depends_on_utility(tmp_path, monkeypatch):
    """The utility wiring is live: disabling it reverts the helped tie order.

    Guards against a positive-feedback test that passes through incidental
    alphabetical ordering rather than the utility term.
    """

    import slowave.symbolic.procedural_memory as procedural_memory

    engine = _tied_engine(tmp_path)
    try:
        procedures = [
            {"id": "proc_a_plain", "goal": "Release checks", "summary": "Plain method"},
            {
                "id": "proc_z_helped",
                "goal": "Release checks",
                "summary": "Rewarded method",
                "evidence": {"helped": 1},
            },
        ]

        def ordered_ids():
            return [
                procedure["id"]
                for procedure in engine._retrieval.shared_procedures(
                    SharedRequest("Verify the release checks"), procedures
                )
            ]

        assert ordered_ids() == ["proc_z_helped", "proc_a_plain"]
        monkeypatch.setattr(procedural_memory, "procedure_feedback_utility", lambda item: 0.0)
        assert ordered_ids() == ["proc_a_plain", "proc_z_helped"]
    finally:
        engine.close()


def test_shared_procedures_harmed_yields_to_equally_applicable_alternative(tmp_path):
    """One harm report reverses a relevance tie safely."""

    engine = _tied_engine(tmp_path)
    try:
        found = engine._retrieval.shared_procedures(
            SharedRequest("Verify the release checks"),
            [
                {
                    "id": "proc_harmed",
                    "goal": "Release checks",
                    "summary": "Damaged method",
                    "evidence": {"harmed": 1},
                },
                {"id": "proc_plain", "goal": "Release checks", "summary": "Plain method"},
            ],
        )
        assert [procedure["id"] for procedure in found] == ["proc_plain", "proc_harmed"]
    finally:
        engine.close()


def test_shared_procedures_reward_cannot_admit_irrelevant_procedure(tmp_path):
    """Heavy reward never rescues a candidate below the relevance threshold."""

    class Scorer:
        def score(self, queries, texts):
            # -8.0 stays below the shared activate default floor (-1.25);
            # reward must never rescue a candidate the floor rejects.
            return np.array([[-8.0] for _ in texts])

    engine = SlowaveEngine(
        SlowaveConfig(db_path=str(tmp_path / "rewarded-irrelevant.db"), dim=8, disable_encoder=True)
    )
    try:
        engine._retrieval._applicability_scorer = Scorer()
        found = engine._retrieval.shared_procedures(
            SharedRequest("Verify the release checks"),
            [
                {
                    "id": "proc_rewarded",
                    "goal": "Release checks",
                    "summary": "Once-useful method",
                    "evidence": {"helped": 3},
                }
            ],
        )
        assert found == []
    finally:
        engine.close()


def test_shared_procedures_share_the_existing_source_outcome_prior(tmp_path):
    """A failed source sorts below a verified peer at a relevance tie."""

    engine = _tied_engine(tmp_path)
    try:
        found = engine._retrieval.shared_procedures(
            SharedRequest("Verify the release checks"),
            [
                {
                    "id": "proc_failed",
                    "goal": "Release checks",
                    "summary": "Failed method",
                    "outcome": "failure",
                },
                {
                    "id": "proc_verified",
                    "goal": "Release checks",
                    "summary": "Verified method",
                    "outcome": "success",
                    "verification_status": "verified",
                },
            ],
        )
        assert [procedure["id"] for procedure in found] == ["proc_verified", "proc_failed"]
    finally:
        engine.close()
