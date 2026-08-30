"""Feedback persistence tests for cue-conditioned retrieval access evidence."""

from __future__ import annotations

import os
import tempfile

import numpy as np

from slowave.core.config import SlowaveConfig
from slowave.core.engine import SlowaveEngine
from slowave.core.services.retrieval_access import canonical_cue_text


class _StubEncoder:
    def encode(self, text: str) -> np.ndarray:
        value = float(len(text) or 1)
        vector = np.array([value, 1.0], dtype=np.float32)
        return vector / np.linalg.norm(vector)


def _engine() -> tuple[SlowaveEngine, str]:
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    engine = SlowaveEngine(SlowaveConfig(db_path=tmp.name, dim=2, disable_encoder=True))
    engine.encoder = _StubEncoder()
    return engine, tmp.name


def _cleanup(path: str) -> None:
    for ext in ("", "-wal", "-shm"):
        target = path + ext
        if os.path.exists(target):
            os.remove(target)


def _schema(engine: SlowaveEngine, text: str, seed: int) -> int:
    rng = np.random.default_rng(seed)
    embedding = rng.normal(size=(2,)).astype(np.float32)
    embedding /= np.linalg.norm(embedding)
    return engine.schemas.create(
        content_text=text,
        facets={},
        tags=[],
        embedding=embedding,
        confidence=1.0,
        salience=1.0,
    )


def _snapshot(
    engine: SlowaveEngine, context_id: str, schema_id: int, pathway: str = "direct"
) -> None:
    engine.record_context_recall(
        context_id=context_id,
        scope_id="project:access",
        query="repair access evidence",
        goal="persist access feedback",
        task_type="coding",
        response={
            "schemas": [{"id": f"sch_{schema_id}", "pathway": pathway, "activation": 0.8}],
        },
    )


def test_irrelevant_writes_pathway_evidence_without_semantic_mutation() -> None:
    engine, path = _engine()
    try:
        schema_id = _schema(engine, "valid but irrelevant for this cue", 1)
        before = engine.schemas.get(schema_id)
        _snapshot(engine, "ctx_access_irrelevant", schema_id)

        result = engine.context_feedback(
            context_id="ctx_access_irrelevant",
            feedback="irrelevant",
            irrelevant_memory_ids=[f"sch_{schema_id}"],
        )

        after = engine.schemas.get(schema_id)
        assert after.confidence == before.confidence
        assert after.salience == before.salience
        assert after.status == before.status
        assert after.is_labile == before.is_labile
        assert result["access_evidence"] == {
            "useful": [],
            "irrelevant": [f"sch_{schema_id}"],
            "skipped": [],
        }
        evidence = engine.retrieval_access_evidence(schema_id)
        assert len(evidence) == 1
        assert evidence[0]["pathway"] == "direct"
        assert evidence[0]["useful_count"] == 0
        assert evidence[0]["irrelevant_count"] == 1
        conn = engine.db.connect()
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM context_feedback_events WHERE context_id = ?",
                ("ctx_access_irrelevant",),
            ).fetchone()[0]
            == 1
        )
    finally:
        engine.close()
        _cleanup(path)


def test_useful_and_irrelevant_evidence_are_pathway_isolated() -> None:
    engine, path = _engine()
    try:
        schema_id = _schema(engine, "schema with route-specific feedback", 2)
        _snapshot(engine, "ctx_access_graph", schema_id, pathway="graph")
        _snapshot(engine, "ctx_access_direct", schema_id, pathway="direct")

        engine.context_feedback(
            context_id="ctx_access_graph",
            feedback="irrelevant",
            irrelevant_memory_ids=[f"sch_{schema_id}"],
        )
        engine.context_feedback(
            context_id="ctx_access_direct",
            feedback="useful",
            used_memory_ids=[f"sch_{schema_id}"],
        )

        evidence = engine.retrieval_access_evidence(schema_id)
        assert [
            (row["pathway"], row["useful_count"], row["irrelevant_count"]) for row in evidence
        ] == [
            ("direct", 1, 0),
            ("graph", 0, 1),
        ]
    finally:
        engine.close()
        _cleanup(path)


def test_access_evidence_fails_closed_without_snapshot_cue_or_admitted_item() -> None:
    engine, path = _engine()
    try:
        schema_id = _schema(engine, "schema lacking trustworthy provenance", 3)
        engine.record_context_recall(
            context_id="ctx_access_no_cue",
            response={"schemas": [{"id": f"sch_{schema_id}", "activation": 0.8}]},
        )
        result = engine.context_feedback(
            context_id="ctx_access_no_cue",
            feedback="irrelevant",
            irrelevant_memory_ids=[f"sch_{schema_id}"],
        )
        assert result["access_evidence"]["irrelevant"] == []
        assert result["access_evidence"]["skipped"] == [f"sch_{schema_id}"]
        assert engine.retrieval_access_evidence(schema_id) == []
    finally:
        engine.close()
        _cleanup(path)


def test_canonical_cue_excludes_scope_metadata() -> None:
    kwargs = {
        "query": "repair retrieval admission",
        "goal": "reduce false positives",
        "task_type": "coding",
        "situation": {"component": "retrieval"},
        "requirements": ["preserve snapshots"],
        "topics": ["feedback"],
        "entities": ["schema"],
    }
    assert canonical_cue_text(**kwargs) == canonical_cue_text(**kwargs)
