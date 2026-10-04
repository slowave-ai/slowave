import numpy as np
import pytest

from slowave.core.applicability_ranking import applicability_order, query_needs
from slowave.core.config import SlowaveConfig
from slowave.core.engine import SlowaveEngine


def test_direct_answers_lead_but_supported_context_is_retained_without_quota():
    scores = np.array([[1.0], [8.0], [-0.5], [0.4], [-7.0]])
    assert applicability_order(scores, minimum_logit=-4) == [1, 0, 3]
    assert applicability_order(np.ones((37, 1)), minimum_logit=-4) == list(range(37))


def test_weak_query_retains_uncertain_support_but_unrelated_queries_abstain():
    assert applicability_order(np.array([[-1.0], [-2.5], [-3.5]]), minimum_logit=-4) == [0, 1]
    assert applicability_order(np.array([[-6.0], [-8.0]]), minimum_logit=-4) == []


def test_independent_need_answers_precede_repeated_support_for_one_need():
    scores = np.array([[8.0, -8.0], [7.0, -8.0], [-8.0, 3.0], [-8.0, 1.0]])
    assert applicability_order(scores, minimum_logit=-4) == [0, 2, 1, 3]
    assert (
        len(query_needs("What are the pre-commit checks and the correct base for a new branch?"))
        == 2
    )
    assert query_needs("Run the linter and typechecker") == ["Run the linter and typechecker"]


@pytest.mark.parametrize("scores", [np.array([1.0]), np.array([[float("nan")]])])
def test_invalid_model_results_are_rejected(scores):
    with pytest.raises(ValueError):
        applicability_order(scores, minimum_logit=-4)


def test_public_recall_uses_pair_scores_preserving_scope_and_feedback_boundary(tmp_path):
    from slowave import ops

    class Scorer:
        def score(self, queries, memories):
            return np.array(
                [[8.0 if "12 seconds" in text else -8.0 for _ in queries] for text in memories]
            )

    engine = SlowaveEngine(
        SlowaveConfig(db_path=str(tmp_path / "ranking.db"), dim=8, disable_encoder=True)
    )
    try:
        scope = "project:test"
        ids = [
            engine.schemas.create(
                content_text=text, facets={}, embedding=None, scope_id=scope, dedupe=False
            )
            for text in (
                "Migration timeout notes describe a meeting.",
                "Migration timeout must remain 12 seconds.",
            )
        ]
        engine.schemas.create(
            content_text="Migration timeout must remain 12 seconds.",
            facets={},
            embedding=None,
            scope_id="project:other",
            dedupe=False,
        )
        engine._retrieval._applicability_scorer = Scorer()
        result = ops.recall(
            engine, query="What migration timeout applies?", scope=scope, relevant_set=True
        )
        assert [row["id"] for row in result["memories"]] == [f"sch_{ids[1]}"]
        assert result["applicability_status"] == "applied"
        assert result["relevant_total"] == 1

        class MissingScorer:
            def score(self, queries, memories):
                raise OSError("model missing")

        engine._retrieval._applicability_scorer = MissingScorer()
        fallback = ops.recall(
            engine, query="What migration timeout applies?", scope=scope, relevant_set=True
        )
        assert fallback["applicability_status"] == "unavailable"
        assert {row["id"] for row in fallback["memories"]} == {f"sch_{i}" for i in ids}
    finally:
        engine.close()
