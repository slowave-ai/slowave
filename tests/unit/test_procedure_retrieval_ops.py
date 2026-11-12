"""Procedure retrieval through ops.activate / ops.recall: gate wiring and fallback."""

from __future__ import annotations

import os
import tempfile
from typing import Any

import numpy as np
import pytest

from slowave import ops
from slowave.core.cheap_pair_scorer import CheapPairScorer
from slowave.core.config import SlowaveConfig
from slowave.core.engine import SlowaveEngine
from slowave.symbolic.procedural_memory import (
    ProcedureScorerUnavailable,
    retrieve_relevant_procedures,
)


class _KeywordScorer:
    def score(self, queries: list[str], memories: list[str]) -> np.ndarray:
        out = np.full((len(memories), len(queries)), -8.0, dtype=np.float32)
        for j, query in enumerate(queries):
            terms = set(query.casefold().split())
            for i, memory in enumerate(memories):
                out[i, j] = -8.0 + 5.0 * len(terms & set(memory.casefold().split()))
        return out


class _BrokenScorer:
    def score(self, queries: list[str], memories: list[str]) -> np.ndarray:
        raise ProcedureScorerUnavailable("model assets missing")


@pytest.fixture
def engine():
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    eng = SlowaveEngine(SlowaveConfig(db_path=tmp.name, dim=8, disable_encoder=True))
    eng._retrieval._procedure_scorer = _KeywordScorer()
    try:
        yield eng
    finally:
        eng.close()
        for suffix in ("", "-wal", "-shm"):
            if os.path.exists(tmp.name + suffix):
                os.remove(tmp.name + suffix)


def _record(eng: SlowaveEngine, goal: str, summary: str) -> str:
    activated = ops.activate(eng, query=goal, initial_goal=goal, scope="project:ops")
    ops.commit(
        eng,
        session_id=activated["session_id"],
        final_goal=goal,
        outcome="success",
        outcome_summary="done",
        procedure={
            "summary": summary,
            "context": {},
            "steps": [{"summary": "first"}, {"summary": "second"}],
            "caveats": [],
        },
    )
    return activated["session_id"]


def test_activate_returns_only_the_relevant_procedure(engine) -> None:
    dashboard = _record(engine, "restart dashboard server", "restart dashboard server safely")
    _record(engine, "rotate database credentials", "rotate database credentials safely")

    result = ops.activate(
        engine,
        query="restart the dashboard server",
        initial_goal="restart dashboard server",
        scope="project:ops",
    )

    assert [item["id"] for item in result["procedures"]] == [f"proc_{dashboard}"]


def test_activate_returns_nothing_for_an_unrelated_task(engine) -> None:
    _record(engine, "restart dashboard server", "restart dashboard server safely")
    _record(engine, "rotate database credentials", "rotate database credentials safely")

    result = ops.activate(
        engine, query="bake sourdough bread", initial_goal="bake bread", scope="project:ops"
    )

    assert result["procedures"] == []


def test_prompt_without_an_action_verb_is_still_scored(engine) -> None:
    session = _record(
        engine, "dashboard server restart procedure", "dashboard server restart procedure"
    )

    result = ops.activate(
        engine,
        query="dashboard server restart procedure",
        initial_goal="dashboard server restart procedure",
        scope="project:ops",
    )

    assert [item["id"] for item in result["procedures"]] == [f"proc_{session}"]


def test_recall_without_a_goal_uses_the_task_as_the_cue(engine) -> None:
    session = _record(engine, "restart dashboard server", "restart dashboard server safely")
    _record(engine, "rotate database credentials", "rotate database credentials safely")

    # ops.recall calls this helper with goal=None (recall has no goal argument).
    # ops.recall itself needs a real embedding encoder, which offline tests lack.
    hits = ops._retrieve_procedure_hits(
        engine,
        ops.load_procedures(engine.db.connect(), scope="project:ops"),
        task="restart dashboard server",
        goal=None,
        retrieval_context=None,
        suppression_trace=[],
    )

    assert [item["id"] for item in hits] == [f"proc_{session}"]


def test_unavailable_scorer_returns_no_procedures_instead_of_guessing(engine, caplog) -> None:
    _record(engine, "restart dashboard server", "restart dashboard server safely")
    engine._retrieval._procedure_scorer = _BrokenScorer()

    result = ops.activate(
        engine,
        query="restart dashboard server",
        initial_goal="restart dashboard server",
        scope="project:ops",
    )

    assert result["procedures"] == []
    assert "scorer unavailable" in caplog.text


def test_a_real_bug_is_not_swallowed_by_the_fallback(engine, monkeypatch) -> None:
    _record(engine, "restart dashboard server", "restart dashboard server safely")

    def broken(*args: Any, **kwargs: Any):
        raise KeyError("summary")

    monkeypatch.setattr(ops, "retrieve_relevant_procedures", broken)

    with pytest.raises(KeyError):
        ops.activate(
            engine,
            query="restart dashboard server",
            initial_goal="restart dashboard server",
            scope="project:ops",
        )


def test_missing_model_assets_back_off_instead_of_retrying_every_call(monkeypatch) -> None:
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    eng = SlowaveEngine(SlowaveConfig(db_path=tmp.name, dim=8, disable_encoder=True))
    attempts = {"n": 0}

    def _failing_scorer():
        attempts["n"] += 1
        raise OSError("assets not found")

    monkeypatch.setattr(eng._retrieval, "_pair_scorer", _failing_scorer)
    try:
        for _ in range(3):
            with pytest.raises(ProcedureScorerUnavailable):
                eng._retrieval.procedure_scorer()
        assert attempts["n"] == 1
    finally:
        eng.close()
        for suffix in ("", "-wal", "-shm"):
            if os.path.exists(tmp.name + suffix):
                os.remove(tmp.name + suffix)


class _BagEncoder:
    """Deterministic shared encoder: the same text always yields the same vector."""

    dim = 16

    def encode(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dim, dtype=np.float32)
        for token in text.casefold().split():
            vector[sum(token.encode("utf-8")) % self.dim] += 1.0
        norm = float(np.linalg.norm(vector))
        return vector / norm if norm else vector

    def encode_many(self, texts: list[str]) -> np.ndarray:
        return np.stack([self.encode(text) for text in texts])


def _procedure(identifier: str, *, helped: int = 0, harmed: int = 0) -> dict[str, Any]:
    summary = "Repair a Kubernetes checkout service crash loop."
    return {
        "id": identifier,
        "goal": summary,
        "summary": summary,
        "context": {"platform": "kubernetes", "failure": "crashloop"},
        "steps": [{"summary": "Inspect deployment events."}],
        "caveats": [],
        "outcome": "success",
        "verification_status": "verified",
        "evidence": {"used": helped + harmed, "helped": helped, "harmed": harmed},
    }


def _ranked(scorer: CheapPairScorer, encoder: _BagEncoder, procedures: list[dict[str, Any]]):
    summary = "Repair a Kubernetes checkout service crash loop."
    hits = retrieve_relevant_procedures(
        procedures,
        query=summary,
        goal=summary,
        retrieval_context={"platform": "kubernetes", "failure": "crashloop"},
        scorer=scorer,
        encoder=encoder,
    )
    return [hit["id"] for hit in hits]


def test_identical_procedure_text_scores_identically() -> None:
    """List position must not decide relevance: identical text ties exactly."""
    encoder = _BagEncoder()
    scorer = CheapPairScorer(encoder, shared_search=True)
    text = "Repair a Kubernetes checkout service crash loop."

    scores = scorer.score([text], [text, text])

    assert scores[0, 0] == scores[1, 0]


def test_feedback_decides_a_tie_between_identical_procedures() -> None:
    """Reuse evidence orders equal candidates, and one harm report reverses it."""
    encoder = _BagEncoder()
    scorer = CheapPairScorer(encoder, shared_search=True)
    procedures = [_procedure("proc_helped", helped=1), _procedure("proc_plain")]

    assert _ranked(scorer, encoder, procedures) == ["proc_helped", "proc_plain"]

    procedures[0] = _procedure("proc_helped", harmed=1)

    assert _ranked(scorer, encoder, procedures) == ["proc_plain", "proc_helped"]
