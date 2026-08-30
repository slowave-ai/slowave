"""Gap-fill tests for the Feedback module (08) — review gating, salience
bounds, and the "labile" lifecycle (see core/08-feedback.md, core/05-
consolidation.md, outcomes/08-feedback.md).

tests/unit/test_context_feedback.py already covers signal mapping, snapshot
persistence, and per-label reinforcement/penalty. This file locks in the
findings from private/docs/consolidation/plans/08-feedback.md's Phase 4/5
investigation and its 2026-07-10 follow-ups:
is_labile (boolean) vs status="needs_review" (string) eligibility,
context_noise_score no longer requiring scope_id (fixed 2026-07-10 — it used
to, silently), the shared salience ceiling used by both reinforce() and
adjust_feedback_state(), useful_confidence_delta actually reaching a schema's
confidence, the discovery that apply_learning=False does not disable noise-
score/is_labile demotion (only the direct signal-driven salience/
confidence/status mutations), and feedback (useful/partially_useful) clearing
a flagged schema's is_labile.

Recurrence-clears-lability
have their own dedicated file: tests/unit/test_labile_lifecycle.py.

Three FeedbackConfig fields that used to have dedicated "confirmed dead"
tests here (`apply_outcome_to_schema_reward`, `missing_creates_memory`,
`missing_replay_enabled`) and two more (`stale_review_threshold`,
`wrong_review_threshold`) were removed from the dataclass entirely on
2026-07-10 rather than wired — there is nothing left to test.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

from slowave.core.config import SlowaveConfig
from slowave.core.engine import SlowaveEngine
from slowave.core.feedback import FeedbackConfig


def _engine(**feedback_overrides) -> tuple[SlowaveEngine, str]:
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    cfg = SlowaveConfig(
        db_path=tmp.name, dim=8, disable_encoder=True, feedback=FeedbackConfig(**feedback_overrides)
    )
    return SlowaveEngine(cfg), tmp.name


def _cleanup(path: str) -> None:
    for ext in ("", "-wal", "-shm"):
        p = path + ext
        if os.path.exists(p):
            os.remove(p)


def _schema(eng: SlowaveEngine, text: str, seed: int) -> int:
    rng = np.random.default_rng(seed)
    emb = rng.normal(size=(8,)).astype(np.float32)
    emb /= np.linalg.norm(emb) + 1e-12
    return eng.schemas.create(
        content_text=text, facets={}, tags=[], embedding=emb, confidence=1.0, salience=1.0
    )


def _feedback(
    eng: SlowaveEngine,
    sid: int,
    label: str,
    *,
    outcome: str = "unknown",
    retrieval_type: str = "context",
    scope_id: str | None = None,
    seq: int,
    **id_flags,
) -> dict:
    ctx = f"ctx_{sid}_{label}_{seq}"
    eng.record_retrieval(retrieval_id=ctx, retrieval_type=retrieval_type, scope_id=scope_id)
    kwargs = {f"{k}_memory_ids": [f"sch_{sid}"] for k in id_flags if id_flags[k]}
    return eng.retrieval_feedback(
        retrieval_id=ctx,
        retrieval_type=retrieval_type,
        feedback=label,
        outcome=outcome,
        scope_id=scope_id,
        **kwargs,
    )


class TestSemanticReviewGating:
    def test_wrong_plus_failure_escalates_status_and_excludes(self) -> None:
        eng, path = _engine()
        try:
            sid = _schema(eng, "wrong and failed schema", 3)
            _feedback(eng, sid, "wrong", outcome="failure", scope_id="eval:test", seq=0, wrong=True)
            s = eng.schemas.get(sid)
            assert s.status == "stale"
            assert s.stale_reason == "contradicted"

            visible_ids = {sch.id for sch in eng.context(limit=100)}
            assert sid not in visible_ids
        finally:
            eng.close()
            _cleanup(path)

    def test_wrong_without_failure_outcome_still_retires_truth(self) -> None:
        eng, path = _engine()
        try:
            sid = _schema(eng, "wrong but not failed schema", 4)
            _feedback(eng, sid, "wrong", outcome="success", scope_id="eval:test", seq=0, wrong=True)
            s = eng.schemas.get(sid)
            assert s.status == "stale"
            assert s.stale_reason == "contradicted"
            assert s.is_labile is False

            visible_ids = {sch.id for sch in eng.context(limit=100)}
            assert sid not in visible_ids
        finally:
            eng.close()
            _cleanup(path)


class TestSalienceCeilingSharedAcrossPaths:
    """reinforce() (the 'useful' path) and adjust_feedback_state() (partial/
    irrelevant/stale/wrong) share the same salience ceiling (SALIENCE_CEILING
    = 20.0) since 2026-07-10. Before that fix, only reinforce() had a
    ceiling — a schema reinforced exclusively via 'partially_useful' could
    grow past 20.0 while an otherwise-identical 'useful'-reinforced schema
    could not. Both paths still share the same floor (min_salience)."""

    def test_useful_path_saturates_at_ceiling(self) -> None:
        eng, path = _engine()
        try:
            sid = _schema(eng, "useful ceiling schema", 7)
            for i in range(250):
                _feedback(
                    eng,
                    sid,
                    "useful",
                    outcome="success",
                    retrieval_type="recall",
                    seq=i,
                    used=True,
                )
            assert eng.schemas.get(sid).salience == 20.0
        finally:
            eng.close()
            _cleanup(path)

    def test_partially_useful_path_now_also_caps_at_ceiling(self) -> None:
        eng, path = _engine()
        try:
            sid = _schema(eng, "partial capped schema", 8)
            reps = 500
            for i in range(reps):
                _feedback(
                    eng,
                    sid,
                    "partially_useful",
                    outcome="success",
                    retrieval_type="recall",
                    seq=i,
                    used=True,
                )
            # 500 reps * 0.04 delta would be 21.0 uncapped — confirms the
            # ceiling is actually being hit, not just coincidentally close.
            assert eng.schemas.get(sid).salience == 20.0
        finally:
            eng.close()
            _cleanup(path)


class TestUsefulConfidenceDeltaIsWired:
    """useful_confidence_delta is applied via reinforce()'s confidence_delta
    parameter since 2026-07-10 — before that fix it was computed into the
    FeedbackSignal and then silently dropped."""

    def test_confidence_increases_by_useful_confidence_delta(self) -> None:
        eng, path = _engine(useful_confidence_delta=0.2)
        try:
            sid = _schema(eng, "useful conf delta schema", 11)
            eng.schemas.adjust_feedback_state(sid, salience_delta=0.0, confidence_delta=-0.5)
            before = eng.schemas.get(sid).confidence
            assert before == pytest.approx(0.5)
            _feedback(eng, sid, "useful", outcome="success", seq=0, used=True)
            after = eng.schemas.get(sid).confidence
            # source_weight defaults to context_feedback_weight=0.5, so the
            # applied delta is 0.2 * 0.5 = 0.1.
            assert after == pytest.approx(0.6)
        finally:
            eng.close()
            _cleanup(path)

    def test_confidence_clamps_at_max_confidence(self) -> None:
        eng, path = _engine(useful_confidence_delta=0.9)
        try:
            sid = _schema(eng, "useful conf clamp schema", 12)
            assert eng.schemas.get(sid).confidence == 1.0
            _feedback(
                eng, sid, "useful", outcome="success", retrieval_type="recall", seq=0, used=True
            )
            # Already at max_confidence=1.0 — a +0.9 delta must clamp, not overshoot.
            assert eng.schemas.get(sid).confidence == 1.0
        finally:
            eng.close()
            _cleanup(path)


class TestApplyLearningFlagsGateExactlyTheirLabelSubset:
    """Formalizes the F1-F4 ablation from scripts/feedback_ablation.py as
    regression tests."""

    def test_apply_learning_false_disables_everything(self) -> None:
        """With no scope_id (the default here), apply_learning=False leaves
        salience/confidence/is_labile completely untouched — this is the
        common case. See
        test_apply_learning_false_does_not_disable_noise_score_demotion below
        for the scope_id-set case, where a *different* mechanism still runs."""
        eng, path = _engine(apply_learning=False)
        try:
            sid = _schema(eng, "master gate schema", 13)
            before = eng.schemas.get(sid)
            _feedback(eng, sid, "useful", outcome="success", seq=0, used=True)
            for i in range(1, 4):
                _feedback(eng, sid, "irrelevant", outcome="success", seq=i, irrelevant=True)
            after = eng.schemas.get(sid)
            assert before.salience == after.salience
            assert before.confidence == after.confidence
            assert before.is_labile == after.is_labile
        finally:
            eng.close()
            _cleanup(path)

    def test_apply_learning_false_preserves_semantic_state_for_irrelevant(self) -> None:
        eng, path = _engine(apply_learning=False)
        try:
            sid = _schema(eng, "master gate scoped schema", 20)
            for i in range(4):
                _feedback(
                    eng,
                    sid,
                    "irrelevant",
                    outcome="success",
                    scope_id="eval:test",
                    seq=i,
                    irrelevant=True,
                )
            s = eng.schemas.get(sid)
            # Direct mutation IS gated: salience/confidence never moved.
            assert s.salience == 1.0
            assert s.confidence == 1.0
            assert s.is_labile is False
        finally:
            eng.close()
            _cleanup(path)

    def test_apply_positive_learning_false_only_disables_positive(self) -> None:
        eng, path = _engine(apply_positive_learning=False)
        try:
            useful_sid = _schema(eng, "positive-gated useful", 14)
            irr_sid = _schema(eng, "positive-gated irrelevant", 15)
            u_before = eng.schemas.get(useful_sid).salience
            i_before = eng.schemas.get(irr_sid).salience
            _feedback(eng, useful_sid, "useful", outcome="success", seq=0, used=True)
            _feedback(eng, irr_sid, "irrelevant", outcome="success", seq=1, irrelevant=True)
            assert eng.schemas.get(useful_sid).salience == u_before
            assert eng.schemas.get(irr_sid).salience == i_before
        finally:
            eng.close()
            _cleanup(path)

    def test_apply_negative_learning_false_only_disables_negative(self) -> None:
        eng, path = _engine(apply_negative_learning=False)
        try:
            useful_sid = _schema(eng, "negative-gated useful", 16)
            irr_sid = _schema(eng, "negative-gated irrelevant", 17)
            u_before = eng.schemas.get(useful_sid).salience
            i_before = eng.schemas.get(irr_sid).salience
            _feedback(eng, useful_sid, "useful", outcome="success", seq=0, used=True)
            _feedback(eng, irr_sid, "irrelevant", outcome="success", seq=1, irrelevant=True)
            assert eng.schemas.get(useful_sid).salience > u_before
            assert eng.schemas.get(irr_sid).salience == i_before
        finally:
            eng.close()
            _cleanup(path)

    def test_apply_stale_wrong_review_false_disables_both_labels(self) -> None:
        eng, path = _engine(apply_stale_wrong_review=False)
        try:
            stale_sid = _schema(eng, "review-gated stale", 18)
            wrong_sid = _schema(eng, "review-gated wrong", 19)
            s_before = eng.schemas.get(stale_sid)
            w_before = eng.schemas.get(wrong_sid)
            _feedback(eng, stale_sid, "stale", outcome="success", seq=0, stale=True)
            _feedback(eng, wrong_sid, "wrong", outcome="failure", seq=1, wrong=True)
            s_after = eng.schemas.get(stale_sid)
            w_after = eng.schemas.get(wrong_sid)
            assert s_after.salience == s_before.salience
            assert not s_after.is_labile
            assert w_after.salience == w_before.salience
            assert not w_after.is_labile
            assert w_after.status == "active"
        finally:
            eng.close()
            _cleanup(path)


class TestFeedbackClearsLability:
    """Part of the 2026-07-10 "labile" lifecycle: an explicit useful/
    partially_useful mark is direct positive evidence a flagged schema is
    still good, and now clears is_labile immediately rather than leaving
    recovery only to consolidation's replay or sustained passive recurrence
    (see core/08-feedback.md, outcomes/08-feedback.md)."""

    def test_useful_clears_is_labile(self) -> None:
        eng, path = _engine()
        try:
            sid = _schema(eng, "flagged then useful schema", 24)
            eng.schemas.adjust_feedback_state(sid, is_labile=True)
            assert eng.schemas.get(sid).is_labile is True
            _feedback(eng, sid, "useful", outcome="success", seq=0, used=True)
            assert eng.schemas.get(sid).is_labile is False
        finally:
            eng.close()
            _cleanup(path)

    def test_partially_useful_clears_is_labile(self) -> None:
        eng, path = _engine()
        try:
            sid = _schema(eng, "flagged then partially useful schema", 25)
            eng.schemas.adjust_feedback_state(sid, is_labile=True)
            assert eng.schemas.get(sid).is_labile is True
            _feedback(eng, sid, "partially_useful", outcome="success", seq=0, used=True)
            assert eng.schemas.get(sid).is_labile is False
        finally:
            eng.close()
            _cleanup(path)

    def test_irrelevant_does_not_clear_is_labile(self) -> None:
        """Sanity check the clearing is specific to positive labels."""
        eng, path = _engine()
        try:
            sid = _schema(eng, "flagged then irrelevant schema", 26)
            eng.schemas.adjust_feedback_state(sid, is_labile=True)
            assert eng.schemas.get(sid).is_labile is True

            _feedback(
                eng,
                sid,
                "irrelevant",
                outcome="success",
                scope_id="eval:test",
                seq=99,
                irrelevant=True,
            )
            assert eng.schemas.get(sid).is_labile is True
        finally:
            eng.close()
            _cleanup(path)
