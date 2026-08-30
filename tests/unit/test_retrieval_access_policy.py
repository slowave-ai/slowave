"""Phase-2 shadow-policy tests: evidence is evaluated but never admitted live."""

from __future__ import annotations

import json
import os
import tempfile

import numpy as np

from slowave import ops
from slowave.core.config import SlowaveConfig
from slowave.core.engine import SlowaveEngine
from slowave.core.services.feedback import _bounded_response_json
from slowave.core.services.retrieval_access import RetrievalAccessPolicy
from slowave.utils.vec import pack_f32


def _engine() -> tuple[SlowaveEngine, str]:
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    return SlowaveEngine(SlowaveConfig(db_path=tmp.name, dim=2, disable_encoder=True)), tmp.name


def _cleanup(path: str) -> None:
    for ext in ("", "-wal", "-shm"):
        target = path + ext
        if os.path.exists(target):
            os.remove(target)


def test_bounded_response_json_remains_parseable_when_truncated() -> None:
    response = {
        "memory_ids": ["sch_1", "sch_2"],
        "shadow_access_traces": [{"id": index, "reason": "x" * 80} for index in range(20)],
    }

    stored = _bounded_response_json(response, 500)
    decoded = json.loads(stored)

    assert len(stored) <= 500
    assert decoded["_truncated"] is True
    assert decoded["memory_ids"] == ["sch_1", "sch_2"]
    assert decoded["omitted_counts"]["shadow_access_traces"] > 0


def _evidence(
    engine: SlowaveEngine, *, schema_id: int, useful: int, irrelevant: int, pathway: str = "direct"
) -> None:
    conn = engine.db.connect()
    cue = np.array([1.0, 0.0], dtype=np.float32)
    cue_id = conn.execute(
        """INSERT INTO retrieval_cue_prototypes
           (embedding, dim, scope_id, scope_kind, task_type, first_seen_ts, last_seen_ts)
           VALUES (?, 2, 'project:access', 'project', 'coding', 1, 1)""",
        (pack_f32(cue),),
    ).lastrowid
    conn.execute(
        """INSERT INTO schema_retrieval_evidence
           (schema_id, cue_prototype_id, pathway, useful_count, irrelevant_count, updated_at)
           VALUES (?, ?, ?, ?, ?, 1)""",
        (schema_id, cue_id, pathway, useful, irrelevant),
    )
    conn.commit()


def test_shadow_policy_inhibits_only_repeated_matching_direct_irrelevance() -> None:
    engine, path = _engine()
    try:
        schema_id = engine.schemas.create(
            content_text="valid claim",
            facets={},
            tags=[],
            embedding=np.array([1.0, 0.0], dtype=np.float32),
        )
        _evidence(engine, schema_id=schema_id, useful=0, irrelevant=2)
        trace = RetrievalAccessPolicy(engine.db).evaluate(
            schema_id=schema_id,
            raw_semantic_relevance=0.80,
            pathway="direct",
            cue_embedding=np.array([1.0, 0.0], dtype=np.float32),
            scope_id="project:access",
            task_type="coding",
        )
        assert trace["hypothetical_admitted"] is False
        assert trace["reason"] == "cue_inhibited"
        assert trace["inhibition_strength"] == 0.35
        assert trace["access_state"] == "inhibited"
    finally:
        engine.close()
        _cleanup(path)


def test_shadow_policy_preserves_pathway_isolation_and_explicit_use_recovery() -> None:
    engine, path = _engine()
    try:
        schema_id = engine.schemas.create(
            content_text="valid claim",
            facets={},
            tags=[],
            embedding=np.array([1.0, 0.0], dtype=np.float32),
        )
        _evidence(engine, schema_id=schema_id, useful=2, irrelevant=2, pathway="graph")
        direct = RetrievalAccessPolicy(engine.db).evaluate(
            schema_id=schema_id,
            raw_semantic_relevance=0.80,
            pathway="direct",
            cue_embedding=np.array([1.0, 0.0], dtype=np.float32),
            scope_id="project:access",
            task_type="coding",
        )
        graph = RetrievalAccessPolicy(engine.db).evaluate(
            schema_id=schema_id,
            raw_semantic_relevance=0.80,
            pathway="graph",
            cue_embedding=np.array([1.0, 0.0], dtype=np.float32),
            scope_id="project:access",
            task_type="coding",
        )
        assert direct["reason"] == "no_matching_access_evidence"
        assert graph["hypothetical_admitted"] is True
        assert graph["inhibition_strength"] == 0.0
    finally:
        engine.close()
        _cleanup(path)


def test_shadow_policy_labels_strong_direct_recovery_override() -> None:
    engine, path = _engine()
    try:
        schema_id = engine.schemas.create(
            content_text="valid claim",
            facets={},
            tags=[],
            embedding=np.array([1.0, 0.0], dtype=np.float32),
        )
        _evidence(engine, schema_id=schema_id, useful=0, irrelevant=3)
        trace = RetrievalAccessPolicy(engine.db).evaluate(
            schema_id=schema_id,
            raw_semantic_relevance=0.96,
            pathway="direct",
            cue_embedding=np.array([1.0, 0.0], dtype=np.float32),
            scope_id="project:access",
            task_type="coding",
        )
        assert trace["hypothetical_admitted"] is True
        assert trace["reason"] == "inhibition_override"
        assert trace["inhibition_strength"] == 0.70
    finally:
        engine.close()
        _cleanup(path)


def test_activate_persists_shadow_trace_without_changing_visible_admission() -> None:
    engine, path = _engine()
    try:

        class Encoder:
            def encode(self, _text: str) -> np.ndarray:
                return np.array([1.0, 0.0], dtype=np.float32)

        engine.encoder = Encoder()
        schema_id = engine.schemas.create(
            content_text="valid claim",
            facets={},
            tags=[],
            embedding=np.array([1.0, 0.0], dtype=np.float32),
            scope_id="project:access",
        )
        _evidence(engine, schema_id=schema_id, useful=0, irrelevant=2)
        response = ops.activate(
            engine,
            query="access policy cue",
            scope="project:access",
            task_type="coding",
            mode="strict_scope",
            min_relevance=0.0,
        )
        assert f"sch_{schema_id}" in [item["id"] for item in response["schemas"]]
        stored = (
            engine.db.connect()
            .execute(
                "SELECT response_json FROM context_recall_events WHERE context_id = ?",
                (response["retrieval_id"],),
            )
            .fetchone()[0]
        )
        trace = json.loads(stored)["shadow_access_traces"][0]
        assert trace["hypothetical_admitted"] is False
        assert trace["reason"] == "cue_inhibited"
    finally:
        engine.close()
        _cleanup(path)
