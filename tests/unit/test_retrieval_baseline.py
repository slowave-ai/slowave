import numpy as np

from slowave.core.config import SlowaveConfig
from slowave.core.engine import SlowaveEngine
from slowave.core.retrieval_baseline import BaselineConfig, rank_candidates
from slowave.core.shared_retrieval import SharedRequest
from slowave.symbolic import retrieval_encoder


def test_lexical_bonus_cannot_admit_semantically_unrelated_memory():
    rows = [
        dict(id=1, semantic=0.7, lexical_rank=1, explicit=True),
        dict(id=2, semantic=0.85, lexical_rank=None, explicit=False),
    ]
    assert [sid for _, sid in rank_candidates(rows, [])] == [2]


def test_goal_changes_rank_and_recall_can_have_wider_admission():
    task = [
        dict(id=1, semantic=0.84, lexical_rank=None, explicit=False),
        dict(id=2, semantic=0.81, lexical_rank=None, explicit=False),
    ]
    goal = [
        dict(id=1, semantic=0.75, lexical_rank=None, explicit=False),
        dict(id=2, semantic=0.95, lexical_rank=None, explicit=False),
    ]
    assert [sid for _, sid in rank_candidates(task, goal)] == [2]
    assert len(rank_candidates(task, [], BaselineConfig(semantic_floor=0.80, max_memories=3))) == 2


def test_passage_cache_roles_and_content_invalidation(monkeypatch):
    monkeypatch.setattr(retrieval_encoder, "model_root", lambda: None)

    class Backend:
        calls = []

        def __init__(self, root):
            pass

        def encode_many(self, texts):
            self.calls.append(texts)
            return np.ones((len(texts), 384), dtype=np.float32)

        def encode(self, text):
            self.calls.append(text)
            return np.ones(384, dtype=np.float32)

    monkeypatch.setattr(retrieval_encoder, "_RetrievalONNX", Backend)
    encoder = retrieval_encoder.RetrievalEncoder()
    encoder.passages(["original"])
    encoder.passages(["original"])
    encoder.passages(["changed"])
    encoder.query("question")
    assert encoder.backend.calls == [["passage: original"], ["passage: changed"], "query: question"]


def test_catalog_goal_scope_and_procedure_abstention(tmp_path, monkeypatch):
    class Encoder:
        def passages(self, texts):
            return np.array([[1, 0] if "chart" in t else [0, 1] for t in texts], dtype=np.float32)

        def query(self, text):
            if text == "chart goal":
                return np.array([1, 0.4])
            if text == "check goal":
                return np.array([0.4, 1])
            if text == "birthday":
                return np.array([0.1, 0.1])
            return np.array([0.9, 0.9])

    monkeypatch.setattr(retrieval_encoder, "get_retrieval_encoder", lambda: Encoder())
    engine = SlowaveEngine(SlowaveConfig(db_path=str(tmp_path / "db"), disable_encoder=True))
    try:
        chart = engine.schemas.create(
            content_text="chart contract", embedding=None, scope_id="project:a"
        )
        checks = engine.schemas.create(
            content_text="verification commands", embedding=None, scope_id="project:a"
        )
        engine.schemas.create(
            content_text="chart foreign contract", embedding=None, scope_id="project:b"
        )
        for goal, expected in [("chart goal", chart), ("check goal", checks)]:
            catalog = engine._retrieval._baseline_catalog(
                SharedRequest("Review dashboard", goal), "project:a", "strict_scope"
            )
            assert [i.schema.id for i in catalog.items] == [expected]
        assert (
            engine._retrieval.baseline_procedure_filter(
                SharedRequest("birthday", "birthday"),
                [dict(goal="chart goal", summary="chart workflow")],
            )
            == []
        )
    finally:
        engine.close()


def test_feedback_refinement_counts_once_and_is_goal_context_specific():
    import sqlite3

    from slowave.core.retrieval_baseline import feedback_adjustments

    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript("""
      CREATE TABLE sessions(id TEXT, task_context_json TEXT);
      CREATE TABLE context_recall_events(context_id TEXT,session_id TEXT,scope_id TEXT,query TEXT,goal TEXT);
      CREATE TABLE feedback_events(retrieval_id TEXT,target_id TEXT,assessment TEXT,status TEXT,target_kind TEXT,created_at INTEGER);
      INSERT INTO sessions VALUES('s','{"env":"prod"}');
      INSERT INTO context_recall_events VALUES('r','s','project:a','Review dashboard','chart');
      INSERT INTO feedback_events VALUES('r','sch_1','used','accepted','memory',1);
      INSERT INTO feedback_events VALUES('r','sch_1','irrelevant','accepted','memory',2);
      INSERT INTO feedback_events VALUES('r','sch_1','irrelevant','accepted','memory',3);
    """)
    assert feedback_adjustments(
        conn, "Review dashboard", "chart", {"env": "prod"}, "project:a"
    ) == {1: -0.02}
    assert (
        feedback_adjustments(conn, "Review dashboard", "checks", {"env": "prod"}, "project:a") == {}
    )
    assert (
        feedback_adjustments(conn, "Review dashboard", "chart", {"env": "dev"}, "project:a") == {}
    )
    conn.execute("INSERT INTO feedback_events VALUES('r','sch_1','used','accepted','memory',4)")
    assert feedback_adjustments(
        conn, "Review dashboard", "chart", {"env": "prod"}, "project:a"
    ) == {1: 0.005}
    conn.close()


def test_negative_feedback_can_abstain_but_positive_feedback_cannot_create_relevance():
    rows = [
        dict(id=1, semantic=0.83, lexical_rank=1, explicit=True, feedback_adjustment=-0.02),
        dict(id=2, semantic=0.7, lexical_rank=1, explicit=True, feedback_adjustment=0.015),
    ]
    assert rank_candidates(rows, []) == []
    rows[0]["feedback_adjustment"] = 0
    assert [sid for _, sid in rank_candidates(rows, [])] == [1]


def test_recall_preserves_complete_baseline_command_after_legacy_cutoff():
    from slowave.mcp.tools import _canonical_recall_result

    source = (
        "Earlier context. " * 40 + "Run npx tsc --noEmit --noUnusedLocals --noUnusedParameters."
    )
    result = _canonical_recall_result(
        dict(
            retrieval_id="fixture_recall",
            retrieval_policy_version="multilingual-retrieval-baseline-v1",
            memories=[dict(id="sch_1", content_text=source)],
        ),
        scope="project:a",
        evidence="none",
    )
    assert result["memories"][0]["content"] == source
    assert result["memories"][0]["content"].endswith("--noUnusedParameters.")
