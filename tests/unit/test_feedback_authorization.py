"""Regression tests for the 2026-07-24 Tier-0 reliability audit fixes to
FeedbackService.retrieval_feedback():

1. Schema-id authorization: reinforce()/adjust_feedback_state()/update_status()
   have no scope filter of their own (bare `WHERE id = ?`), so retrieval_feedback
   must restrict used/irrelevant/stale/wrong ids to schemas the retrieval
   actually surfaced (context_recall_items, admitted=1) before mutating them —
   otherwise a client that hallucinates or reuses an id from a different
   retrieval/scope could silently mutate memory it never saw.

2. Forgotten-schema protection: "forgotten" is a human-only, CLI/dashboard-
   initiated suppression (see schema_store.py's VALID_STATUS comment). An
   ordinary feedback call must never mutate a forgotten schema, and the
   wrong+failure status escalation to "needs_review" must only fire from
   "active" — not resurrect a forgotten/superseded/contradicted/archived
   schema back into needs_review visibility.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np

from slowave.core.config import SlowaveConfig
from slowave.core.engine import SlowaveEngine


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


def _schema(eng: SlowaveEngine, text: str, seed: int) -> int:
    rng = np.random.default_rng(seed)
    emb = rng.normal(size=(8,)).astype(np.float32)
    emb /= np.linalg.norm(emb) + 1e-12
    return eng.schemas.create(
        content_text=text, facets={}, tags=[], embedding=emb, confidence=1.0, salience=1.0
    )


class TestReinforceRejectsUnauthorizedSchemaIds:
    def test_id_outside_the_retrieval_is_rejected_not_mutated(self) -> None:
        eng, path = _tmp_engine()
        try:
            surfaced = _schema(eng, "actually surfaced by this retrieval", 1)
            other_scope = _schema(eng, "belongs to a different retrieval/scope", 2)

            ctx = "ctx_authz_1"
            eng.record_context_recall(
                context_id=ctx,
                scope_id="project:a",
                response={"schemas": [{"id": f"sch_{surfaced}", "activation": 0.9}]},
            )

            before_other = eng.schemas.get(other_scope).salience
            result = eng.retrieval_feedback(
                retrieval_id=ctx,
                retrieval_type="context",
                feedback="useful",
                outcome="success",
                used_memory_ids=[f"sch_{surfaced}", f"sch_{other_scope}"],
            )

            assert f"sch_{surfaced}" in result["applied"]["reinforced"]
            assert f"sch_{other_scope}" in result["applied"]["rejected"]
            assert eng.schemas.get(surfaced).salience > 1.0
            assert eng.schemas.get(other_scope).salience == before_other
        finally:
            eng.close()
            _cleanup(path)

    def test_no_snapshot_recorded_falls_back_to_unrestricted(self) -> None:
        """When record_retrieval/record_context_recall was never called for this
        retrieval_id (e.g. persistence disabled, or an older/unrecorded
        retrieval_id), there's no ground truth to authorize against — the
        restriction must not silently reject everything."""
        eng, path = _tmp_engine()
        try:
            sid = _schema(eng, "no snapshot on record for this retrieval", 3)
            result = eng.retrieval_feedback(
                retrieval_id="ctx_never_recorded",
                retrieval_type="context",
                feedback="useful",
                outcome="success",
                used_memory_ids=[f"sch_{sid}"],
            )
            assert f"sch_{sid}" in result["applied"]["reinforced"]
            assert result["applied"]["rejected"] == []
            assert eng.schemas.get(sid).salience > 1.0
        finally:
            eng.close()
            _cleanup(path)


class TestForgottenSchemaIsProtectedFromFeedback:
    def test_forgotten_schema_is_not_mutated_by_feedback(self) -> None:
        eng, path = _tmp_engine()
        try:
            sid = _schema(eng, "human-forgotten schema", 4)
            eng.schemas.forget(sid)
            assert eng.schemas.get(sid).status == "forgotten"

            ctx = "ctx_forgotten_1"
            eng.record_context_recall(
                context_id=ctx,
                response={"schemas": [{"id": f"sch_{sid}", "activation": 0.5}]},
            )
            eng.retrieval_feedback(
                retrieval_id=ctx,
                retrieval_type="context",
                feedback="wrong",
                outcome="failure",
                wrong_memory_ids=[f"sch_{sid}"],
            )

            after = eng.schemas.get(sid)
            assert after.status == "forgotten"
            assert after.salience == 1.0
        finally:
            eng.close()
            _cleanup(path)

    def test_wrong_failure_does_not_reescalate_superseded_schema(self) -> None:
        eng, path = _tmp_engine()
        try:
            sid = _schema(eng, "already superseded schema", 5)
            eng.schemas.update_status(sid, status="superseded")

            ctx = "ctx_superseded_1"
            eng.record_context_recall(
                context_id=ctx,
                response={"schemas": [{"id": f"sch_{sid}", "activation": 0.5}]},
            )
            eng.retrieval_feedback(
                retrieval_id=ctx,
                retrieval_type="context",
                feedback="wrong",
                outcome="failure",
                wrong_memory_ids=[f"sch_{sid}"],
            )

            assert eng.schemas.get(sid).status == "superseded"
        finally:
            eng.close()
            _cleanup(path)

    def test_wrong_failure_still_escalates_active_schema(self) -> None:
        """Sanity check: the guard is specific to already-resolved statuses,
        not a blanket disabling of the existing escalation behavior."""
        eng, path = _tmp_engine()
        try:
            sid = _schema(eng, "active schema going wrong", 6)
            ctx = "ctx_active_wrong_1"
            eng.record_context_recall(
                context_id=ctx,
                response={"schemas": [{"id": f"sch_{sid}", "activation": 0.5}]},
            )
            eng.retrieval_feedback(
                retrieval_id=ctx,
                retrieval_type="context",
                feedback="wrong",
                outcome="failure",
                wrong_memory_ids=[f"sch_{sid}"],
            )
            assert eng.schemas.get(sid).status == "needs_review"
        finally:
            eng.close()
            _cleanup(path)
