"""Tests for context feedback system."""

from __future__ import annotations

import os
import tempfile
import uuid

import numpy as np

from slowave.core.config import SlowaveConfig
from slowave.core.engine import SlowaveEngine
from slowave.core.feedback import (
    VALID_FEEDBACK_LABELS,
    FeedbackConfig,
    feedback_signal_for,
    normalize_feedback_label,
    normalize_outcome_label,
)


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


def _create_schema(eng: SlowaveEngine, text: str) -> int:
    rng = np.random.default_rng(42)
    emb = rng.normal(size=(8,)).astype(np.float32)
    emb /= np.linalg.norm(emb) + 1e-12
    return eng.schemas.create(
        content_text=text,
        facets={},
        tags=[],
        embedding=emb,
        confidence=1.0,
        salience=1.0,
    )


def _ctx_id() -> str:
    """Generate a unique context ID."""
    return f"ctx_{uuid.uuid4().hex[:12]}"


class TestFeedbackSignalMapping:
    """Test symbolic label → numeric feedback signal mapping."""

    def test_all_feedback_labels_valid(self) -> None:
        for label in VALID_FEEDBACK_LABELS:
            normalized = normalize_feedback_label(label)
            assert normalized == label

    def test_invalid_feedback_label_raises(self) -> None:
        try:
            normalize_feedback_label("invalid_feedback")
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "Invalid feedback label" in str(e)

    def test_outcome_normalization(self) -> None:
        assert normalize_outcome_label("success") == "success"
        assert normalize_outcome_label("failure") == "failure"
        assert normalize_outcome_label("") == "unknown"
        assert normalize_outcome_label("invalid") == "unknown"

    def test_useful_signal(self) -> None:
        cfg = FeedbackConfig()
        sig = feedback_signal_for("useful", "success", cfg)
        assert sig.valence == 1.0
        assert sig.context_fit == 1.0
        assert sig.salience_delta == cfg.useful_salience_delta
        assert sig.confidence_delta == cfg.useful_confidence_delta
        assert sig.outcome_reward == 1.0

    def test_partially_useful_signal(self) -> None:
        cfg = FeedbackConfig()
        sig = feedback_signal_for("partially_useful", "success", cfg)
        assert sig.valence == 0.4
        assert sig.context_fit == 0.5
        assert sig.salience_delta == cfg.partially_useful_salience_delta
        assert sig.outcome_reward == 1.0

    def test_irrelevant_signal(self) -> None:
        cfg = FeedbackConfig()
        sig = feedback_signal_for("irrelevant", "success", cfg)
        assert sig.valence == -0.4
        assert sig.context_fit == -1.0
        assert sig.salience_delta == cfg.irrelevant_salience_delta
        assert sig.truth_error == 0.0

    def test_stale_signal(self) -> None:
        cfg = FeedbackConfig()
        sig = feedback_signal_for("stale", "failure", cfg)
        assert sig.temporal_error == 1.0
        assert sig.salience_delta == cfg.stale_salience_delta
        assert sig.confidence_delta == cfg.stale_confidence_delta
        # review_pressure is a fixed, informational constant (0.7 for "stale")
        # since 2026-07-10 — stale_review_threshold was removed from
        # FeedbackConfig because nothing ever compared it against anything.
        assert sig.review_pressure == 0.7
        assert sig.outcome_reward == -1.0

    def test_wrong_signal(self) -> None:
        cfg = FeedbackConfig()
        sig = feedback_signal_for("wrong", "failure", cfg)
        assert sig.truth_error == 1.0
        assert sig.salience_delta == cfg.wrong_salience_delta
        assert sig.confidence_delta == cfg.wrong_confidence_delta
        # review_pressure is a fixed, informational constant (1.0 for "wrong")
        # since 2026-07-10 — wrong_review_threshold was removed from
        # FeedbackConfig because nothing ever compared it against anything.
        assert sig.review_pressure == 1.0

    def test_missing_signal(self) -> None:
        cfg = FeedbackConfig()
        sig = feedback_signal_for("missing", "failure", cfg)
        assert sig.missingness == 1.0
        assert sig.salience_delta == 0.0
        assert sig.outcome_reward == -1.0

    def test_too_much_context_signal(self) -> None:
        cfg = FeedbackConfig()
        sig = feedback_signal_for("too_much_context", "success", cfg)
        assert sig.overload == 1.0
        assert sig.salience_delta == 0.0


class TestContextSnapshotPersistence:
    """Test that context calls are recorded."""

    def test_record_context_recall(self) -> None:
        eng, path = _tmp_engine()
        try:
            context_id = _ctx_id()
            response = {
                "memory_ids": ["sch_1", "sch_2"],
                "schemas": [
                    {"id": "sch_1", "content": "memory one", "activation": 0.8},
                    {"id": "sch_2", "content": "memory two", "activation": 0.6},
                ],
            }

            eng.record_context_recall(
                context_id=context_id,
                scope_id="eval:test",
                application="cline-tui",
                query="test query",
                response=response,
            )

            conn = eng.db.connect()
            row = conn.execute(
                "SELECT * FROM context_recall_events WHERE context_id = ?",
                (context_id,),
            ).fetchone()
            assert row is not None
            assert row["scope_id"] == "eval:test"
            assert row["application"] == "cline-tui"

            items = conn.execute(
                "SELECT * FROM context_recall_items WHERE context_id = ?",
                (context_id,),
            ).fetchall()
            assert len(items) == 2
            assert items[0]["memory_id"] == "sch_1"
            assert items[1]["memory_id"] == "sch_2"
        finally:
            eng.close()
            _cleanup(path)


class TestFeedbackEventPersistence:
    """Test that feedback events are stored."""

    def test_context_feedback_persisted(self) -> None:
        eng, path = _tmp_engine()
        try:
            sid = _create_schema(eng, "test memory")
            context_id = _ctx_id()

            # Record context first
            eng.record_context_recall(
                context_id=context_id,
                scope_id="eval:test",
            )

            result = eng.context_feedback(
                context_id=context_id,
                feedback="useful",
                outcome="success",
                scope_id="eval:test",
                used_memory_ids=[f"sch_{sid}"],
            )

            assert result["context_id"] == context_id
            assert result["feedback"] == "useful"
            assert result["outcome"] == "success"

            conn = eng.db.connect()
            row = conn.execute(
                "SELECT * FROM context_feedback_events WHERE context_id = ?",
                (context_id,),
            ).fetchone()
            assert row is not None
            assert row["feedback"] == "useful"
            assert row["outcome"] == "success"
        finally:
            eng.close()
            _cleanup(path)


class TestUsefulFeedbackReinforces:
    """Test that useful feedback reinforces schemas."""

    def test_useful_increases_salience(self) -> None:
        eng, path = _tmp_engine()
        try:
            sid = _create_schema(eng, "useful memory")
            s = eng.schemas.get(sid)
            before_salience = s.salience
            context_id = _ctx_id()

            eng.record_context_recall(context_id=context_id)
            eng.context_feedback(
                context_id=context_id,
                feedback="useful",
                outcome="success",
                used_memory_ids=[f"sch_{sid}"],
            )

            s = eng.schemas.get(sid)
            after_salience = s.salience
            assert after_salience > before_salience
        finally:
            eng.close()
            _cleanup(path)

    def test_partially_useful_weaker_reinforcement(self) -> None:
        cfg = FeedbackConfig()
        useful_sig = feedback_signal_for("useful", "success", cfg)
        partial_sig = feedback_signal_for("partially_useful", "success", cfg)
        assert abs(partial_sig.salience_delta) < abs(useful_sig.salience_delta)


class TestIrrelevantAccessEvidence:
    """Irrelevant feedback is contextual access evidence, not semantic damage."""

    def test_irrelevant_preserves_semantic_state(self) -> None:
        eng, path = _tmp_engine()
        try:
            sid = _create_schema(eng, "irrelevant memory")
            s = eng.schemas.get(sid)
            before_confidence = s.confidence
            before_salience = s.salience
            before_review = s.is_labile
            context_id = _ctx_id()

            eng.record_context_recall(context_id=context_id)
            eng.context_feedback(
                context_id=context_id,
                feedback="irrelevant",
                outcome="success",
                irrelevant_memory_ids=[f"sch_{sid}"],
            )

            s = eng.schemas.get(sid)
            assert s.confidence == before_confidence
            assert s.salience == before_salience
            assert s.is_labile == before_review
        finally:
            eng.close()
            _cleanup(path)


class TestStaleRetirement:

    def test_stale_marks_review(self) -> None:
        eng, path = _tmp_engine()
        try:
            sid = _create_schema(eng, "stale memory")
            s = eng.schemas.get(sid)
            assert s.is_labile is False
            context_id = _ctx_id()

            eng.record_context_recall(context_id=context_id)
            eng.context_feedback(
                context_id=context_id,
                feedback="stale",
                outcome="success",
                stale_memory_ids=[f"sch_{sid}"],
            )

            s = eng.schemas.get(sid)
            assert s.is_labile is False
            assert s.status == "stale"
            assert s.stale_reason == "superseded"
            assert s.salience < 1.0  # also reduced
        finally:
            eng.close()
            _cleanup(path)


class TestWrongRetirement:

    def test_wrong_marks_review(self) -> None:
        eng, path = _tmp_engine()
        try:
            sid = _create_schema(eng, "wrong memory")
            s_before = eng.schemas.get(sid)
            context_id = _ctx_id()

            eng.record_context_recall(context_id=context_id)
            eng.context_feedback(
                context_id=context_id,
                feedback="wrong",
                outcome="failure",
                wrong_memory_ids=[f"sch_{sid}"],
            )

            s = eng.schemas.get(sid)
            assert s.is_labile is False
            assert s.status == "stale"
            assert s.stale_reason == "contradicted"
            assert s.salience < s_before.salience
            assert s.confidence < s_before.confidence
        finally:
            eng.close()
            _cleanup(path)


class TestMissingContextPersists:
    """Test that missing context feedback is stored without creating memory."""

    def test_missing_persists_no_memory_created(self) -> None:
        eng, path = _tmp_engine()
        try:
            initial_count = eng.schemas.count()
            context_id = _ctx_id()

            eng.record_context_recall(context_id=context_id)
            eng.context_feedback(
                context_id=context_id,
                feedback="missing",
                outcome="failure",
                missing_context="Needed decision on feedback MCP naming.",
            )

            final_count = eng.schemas.count()
            assert final_count == initial_count  # no new memory created

            conn = eng.db.connect()
            row = conn.execute(
                "SELECT * FROM context_feedback_events WHERE context_id = ?",
                (context_id,),
            ).fetchone()
            assert row is not None
            assert "naming" in row["missing_context"]
        finally:
            eng.close()
            _cleanup(path)


class TestTooMuchContextDoesNotPenalize:
    """Test that too_much_context doesn't penalize if IDs aren't explicit."""

    def test_too_much_context_no_penalty(self) -> None:
        eng, path = _tmp_engine()
        try:
            sid = _create_schema(eng, "context memory")
            s_before = eng.schemas.get(sid)
            context_id = _ctx_id()

            eng.record_context_recall(context_id=context_id)
            eng.context_feedback(
                context_id=context_id,
                feedback="too_much_context",
                outcome="success",
                used_memory_ids=[],
                irrelevant_memory_ids=[],
            )

            s = eng.schemas.get(sid)
            assert s.salience == s_before.salience  # unchanged
        finally:
            eng.close()
            _cleanup(path)


class TestOutcomeDoesNotAffectSchemaRewardByDefault:
    """Test that outcome is stored but doesn't directly reward schemas."""

    def test_useful_with_failure_still_reinforces(self) -> None:
        eng, path = _tmp_engine()
        try:
            sid = _create_schema(eng, "useful despite failure")
            s_before = eng.schemas.get(sid)
            context_id = _ctx_id()

            eng.record_context_recall(context_id=context_id)
            result = eng.context_feedback(
                context_id=context_id,
                feedback="useful",
                outcome="failure",
                used_memory_ids=[f"sch_{sid}"],
            )

            assert result["signal"]["outcome_reward"] == -1.0

            s = eng.schemas.get(sid)
            # Memory still reinforced because feedback says useful
            assert s.salience > s_before.salience
        finally:
            eng.close()
            _cleanup(path)


class TestMixedFeedbackPayloads:
    """Regression tests for item arrays using label-specific deltas."""

    def test_useful_context_preserves_irrelevant_item_semantics(self) -> None:
        eng, path = _tmp_engine()
        try:
            useful_sid = _create_schema(eng, "actually useful memory")
            irrelevant_sid = _create_schema(eng, "noise memory")
            useful_before = eng.schemas.get(useful_sid)
            irrelevant_before = eng.schemas.get(irrelevant_sid)
            context_id = _ctx_id()

            eng.record_context_recall(context_id=context_id)
            result = eng.context_feedback(
                context_id=context_id,
                feedback="useful",
                outcome="success",
                used_memory_ids=[f"sch_{useful_sid}"],
                irrelevant_memory_ids=[f"sch_{irrelevant_sid}"],
            )

            assert result["applied"]["reinforced"] == [f"sch_{useful_sid}"]
            assert result["applied"]["penalized"] == []
            assert eng.schemas.get(useful_sid).salience > useful_before.salience
            assert eng.schemas.get(irrelevant_sid).salience == irrelevant_before.salience
        finally:
            eng.close()
            _cleanup(path)

    def test_recall_feedback_is_stronger_than_context_feedback(self) -> None:
        eng, path = _tmp_engine()
        try:
            context_sid = _create_schema(eng, "context weighted memory")
            recall_sid = _create_schema(eng, "recall weighted memory")
            context_id = _ctx_id()
            recall_id = "rec_test_weight"

            eng.record_retrieval(retrieval_id=context_id, retrieval_type="context")
            eng.record_retrieval(retrieval_id=recall_id, retrieval_type="recall")

            context_result = eng.context_feedback(
                context_id=context_id,
                feedback="useful",
                outcome="success",
                used_memory_ids=[f"sch_{context_sid}"],
            )
            recall_result = eng.retrieval_feedback(
                retrieval_id=recall_id,
                retrieval_type="recall",
                feedback="useful",
                outcome="success",
                used_memory_ids=[f"sch_{recall_sid}"],
            )

            assert context_result["source_weight"] == 0.5
            assert recall_result["source_weight"] == 1.0
            assert eng.schemas.get(context_sid).salience == 1.05
            assert eng.schemas.get(recall_sid).salience == 1.10
        finally:
            eng.close()
            _cleanup(path)

    def test_recall_retrieval_snapshot_and_feedback_store_type(self) -> None:
        eng, path = _tmp_engine()
        try:
            sid = _create_schema(eng, "recall snapshot memory")
            recall_id = "rec_snapshot_test"
            eng.record_retrieval(
                retrieval_id=recall_id,
                retrieval_type="recall",
                query="specific recall query",
                mode="recall",
                response={
                    "memory_ids": [f"sch_{sid}"],
                    "schemas": [
                        {
                            "id": f"sch_{sid}",
                            "content": "recall snapshot memory",
                            "score": 0.91,
                            "salience": 1.0,
                            "confidence": 1.0,
                        }
                    ],
                },
            )
            result = eng.retrieval_feedback(
                retrieval_id=recall_id,
                retrieval_type="recall",
                feedback="useful",
                outcome="success",
                used_memory_ids=[f"sch_{sid}"],
            )

            conn = eng.db.connect()
            parent = conn.execute(
                "SELECT retrieval_type FROM context_recall_events WHERE context_id = ?",
                (recall_id,),
            ).fetchone()
            item = conn.execute(
                "SELECT retrieval_type, memory_type FROM context_recall_items WHERE context_id = ?",
                (recall_id,),
            ).fetchone()
            feedback = conn.execute(
                "SELECT retrieval_type FROM context_feedback_events WHERE context_id = ?",
                (recall_id,),
            ).fetchone()

            assert result["retrieval_id"] == recall_id
            assert result["recall_id"] == recall_id
            assert result["context_id"] is None
            assert parent["retrieval_type"] == "recall"
            assert item["retrieval_type"] == "recall"
            assert item["memory_type"] == "schema"
            assert feedback["retrieval_type"] == "recall"
        finally:
            eng.close()
            _cleanup(path)

    def test_useful_context_can_still_mark_stale_and_wrong_items(self) -> None:
        eng, path = _tmp_engine()
        try:
            stale_sid = _create_schema(eng, "old but once valid memory")
            wrong_sid = _create_schema(eng, "wrong memory")
            stale_before = eng.schemas.get(stale_sid)
            wrong_before = eng.schemas.get(wrong_sid)
            context_id = _ctx_id()

            eng.record_context_recall(context_id=context_id)
            result = eng.context_feedback(
                context_id=context_id,
                feedback="useful",
                outcome="partial",
                stale_memory_ids=[f"sch_{stale_sid}"],
                wrong_memory_ids=[f"sch_{wrong_sid}"],
            )

            assert result["applied"]["marked_review"] == [f"sch_{stale_sid}", f"sch_{wrong_sid}"]
            stale_after = eng.schemas.get(stale_sid)
            wrong_after = eng.schemas.get(wrong_sid)
            assert stale_after.is_labile is False
            assert wrong_after.is_labile is False
            assert stale_after.status == "stale"
            assert stale_after.stale_reason == "superseded"
            assert wrong_after.status == "stale"
            assert wrong_after.stale_reason == "contradicted"
            assert stale_after.salience < stale_before.salience
            assert wrong_after.salience < wrong_before.salience
            assert stale_after.confidence < stale_before.confidence
            assert wrong_after.confidence < wrong_before.confidence
            assert (wrong_before.confidence - wrong_after.confidence) > (
                stale_before.confidence - stale_after.confidence
            )
        finally:
            eng.close()
            _cleanup(path)

    def test_feedback_without_prior_context_creates_minimal_snapshot(self) -> None:
        eng, path = _tmp_engine()
        try:
            context_id = _ctx_id()
            result = eng.context_feedback(
                context_id=context_id,
                feedback="missing",
                outcome="unknown",
                scope_id="eval:test-project",
                missing_context="Needed additional project context.",
            )
            assert result["feedback"] == "missing"

            conn = eng.db.connect()
            parent = conn.execute(
                "SELECT * FROM context_recall_events WHERE context_id = ?",
                (context_id,),
            ).fetchone()
            child = conn.execute(
                "SELECT * FROM context_feedback_events WHERE context_id = ?",
                (context_id,),
            ).fetchone()
            assert parent is not None
            assert parent["scope_id"] == "eval:test-project"
            assert child is not None
            assert child["feedback"] == "missing"
        finally:
            eng.close()
            _cleanup(path)

    def test_irrelevant_with_success_preserves_semantic_state(self) -> None:
        eng, path = _tmp_engine()
        try:
            sid = _create_schema(eng, "irrelevant despite success")
            s_before = eng.schemas.get(sid)
            context_id = _ctx_id()

            eng.record_context_recall(context_id=context_id)
            result = eng.context_feedback(
                context_id=context_id,
                feedback="irrelevant",
                outcome="success",
                irrelevant_memory_ids=[f"sch_{sid}"],
            )

            assert result["signal"]["outcome_reward"] == 1.0

            s = eng.schemas.get(sid)
            assert s.salience == s_before.salience
        finally:
            eng.close()
            _cleanup(path)
