"""Feedback accountability and corrections across derived learning consumers."""

import time

import numpy as np
import pytest

from slowave.core.config import SlowaveConfig
from slowave.core.engine import SlowaveEngine
from slowave.utils.vec import pack_f32


@pytest.fixture
def engine(tmp_path):
    eng = SlowaveEngine(
        SlowaveConfig(db_path=str(tmp_path / "feedback.db"), dim=8, disable_encoder=True)
    )
    yield eng
    eng.close()


def expose(engine, rid="r1", count=1):
    ids = [
        engine.schemas.create(
            content_text=f"Release constraint {i}",
            facets={},
            embedding=None,
            scope_id="project:test",
            dedupe=False,
        )
        for i in range(count)
    ]
    engine.record_retrieval(
        retrieval_id=rid,
        scope_id="project:test",
        response={"schemas": [{"id": f"sch_{sid}", "pathway": "direct"} for sid in ids]},
    )
    conn = engine.db.connect()
    conn.execute(
        "UPDATE context_recall_events SET cue_embedding = ?, cue_dim = 8 WHERE context_id = ?",
        (pack_f32(np.ones(8, dtype=np.float32)), rid),
    )
    conn.commit()
    return ids


def assess(engine, sid, assessment, rid="r1", **fields):
    return engine.feedback(
        retrieval_id=rid,
        memory_feedback=[{"memory_id": f"sch_{sid}", "assessment": assessment, **fields}],
        coverage="complete",
    )


def test_rejected_complete_and_partial_report_actual_outstanding(engine):
    sid = expose(engine)[0]
    rejected = assess(engine, sid, "useful")
    assert rejected["coverage"] == "partial"
    assert rejected["requested_coverage"] == "complete"
    assert rejected["outstanding"]["memory_ids"] == [f"sch_{sid}"]
    assert (
        engine.db.connect()
        .execute(
            "SELECT COUNT(*) FROM feedback_events WHERE target_kind='retrieval' AND status='accepted' AND coverage='complete'"
        )
        .fetchone()[0]
        == 0
    )
    partial = engine.feedback(retrieval_id="r1", coverage="partial")
    assert partial["outstanding"]["memory_ids"] == [f"sch_{sid}"]
    accepted = assess(engine, sid, "not_used")
    assert accepted["coverage"] == "complete"
    assert accepted["outstanding"]["memory_ids"] == []
    # A rejected correction never replaces earlier accepted evidence.
    rejected = assess(engine, sid, "useful")
    assert rejected["coverage"] == "partial"
    assert rejected["outstanding"]["memory_ids"] == []


@pytest.mark.parametrize("assessment", ["not_used", "unassessable"])
def test_neutral_feedback_retracts_use_without_manufacturing_recurrence(engine, assessment):
    sid = expose(engine)[0]
    fields = {"reason": "Usage evidence was lost."} if assessment == "unassessable" else {}
    assess(engine, sid, "used")
    assert engine.schemas.get(sid).facets["recurrence_count"] == 1
    assess(engine, sid, assessment, **fields)
    schema = engine.schemas.get(sid)
    assert schema.salience == pytest.approx(1.0)
    assert schema.facets["recurrence_count"] == 0
    assert schema.facets["context_used_count"] == 0
    assert schema.facets["context_irrelevant_count"] == 0
    assess(engine, sid, assessment, **fields)
    assess(engine, sid, "used")
    assess(engine, sid, "used")
    assert engine.schemas.get(sid).facets["recurrence_count"] == 1


def test_uncertainty_requires_explanation_and_neutral_effect(engine):
    sid = expose(engine)[0]
    assert (
        assess(engine, sid, "unassessable")["rejected"][0]["reason"]
        == "unassessable_requires_reason"
    )
    assert (
        assess(engine, sid, "not_used", effect="helped")["rejected"][0]["reason"]
        == "neutral_requires_unknown_or_absent_effect"
    )


def test_negative_access_retries_corrections_and_legacy_evidence(engine):
    sid = expose(engine)[0]
    engine.retrieval_feedback(
        retrieval_id="r1", feedback="irrelevant", irrelevant_memory_ids=[f"sch_{sid}"]
    )
    for _ in range(2):
        assess(engine, sid, "irrelevant")
        assert engine._feedback.access_evidence.inspect_schema(sid)[0]["irrelevant_count"] == 2
    assess(engine, sid, "used")
    assert engine._feedback.access_evidence.inspect_schema(sid)[0]["irrelevant_count"] == 1
    assess(engine, sid, "not_used")
    assert engine._feedback.access_evidence.inspect_schema(sid)[0]["irrelevant_count"] == 1


def test_retry_after_ledger_persistence_heals_access_projection(engine, monkeypatch):
    sid = expose(engine)[0]
    store = engine._feedback.access_evidence
    reconcile = store.reconcile_feedback

    def fail(*args, **kwargs):
        raise RuntimeError("simulated interruption")

    monkeypatch.setattr(store, "reconcile_feedback", fail)
    with pytest.raises(RuntimeError):
        assess(engine, sid, "irrelevant")
    monkeypatch.setattr(store, "reconcile_feedback", reconcile)
    assess(engine, sid, "irrelevant")
    assert store.inspect_schema(sid)[0]["irrelevant_count"] == 1


def test_effect_response_reports_actual_delta(engine):
    sid = expose(engine)[0]
    assert assess(engine, sid, "used", effect="no_effect")["applied"]["unchanged"] == [f"sch_{sid}"]
    assert assess(engine, sid, "used", effect="harmed")["applied"]["weakened"] == [f"sch_{sid}"]


def test_rejected_duplicate_does_not_hide_accepted_feedback(engine):
    sid = expose(engine)[0]
    result = engine.feedback(
        retrieval_id="r1",
        memory_feedback=[
            {"memory_id": f"sch_{sid}", "assessment": "used"},
            {"memory_id": f"sch_{sid}", "assessment": "useful"},
        ],
        coverage="complete",
    )
    assert result["coverage"] == "partial"
    assert engine.schemas.get(sid).salience == pytest.approx(1.1)


def test_first_use_retry_after_ledger_persistence_earns_one_recurrence(engine, monkeypatch):
    sid = expose(engine)[0]
    projection = engine._feedback.learning_projection
    apply = projection.apply_memory_observation

    def fail(*args, **kwargs):
        raise RuntimeError("interrupted before projection")

    monkeypatch.setattr(projection, "apply_memory_observation", fail)
    with pytest.raises(RuntimeError):
        assess(engine, sid, "used")
    monkeypatch.setattr(projection, "apply_memory_observation", apply)
    assess(engine, sid, "used")
    assert engine.schemas.get(sid).salience == pytest.approx(1.1)
    assert engine.schemas.get(sid).facets["recurrence_count"] == 1


def test_co_use_correction_and_worker_retry_reconcile_edges(engine):
    a, b = expose(engine, count=2)
    engine.feedback(
        retrieval_id="r1",
        memory_feedback=[{"memory_id": f"sch_{sid}", "assessment": "used"} for sid in [a, b]],
        coverage="complete",
    )
    now = int(time.time())

    def weight():
        engine._consolidation._write_coactivations(engine.db.connect(), now)
        return engine.schemas.get_coactivations(a)[0][2]

    positive = weight()
    assert weight() == pytest.approx(positive)
    assess(engine, b, "not_used")
    corrected = weight()
    assert corrected < positive
    assert corrected == pytest.approx(1.0, abs=1e-5)
    assert weight() == pytest.approx(corrected)


def test_procedure_unknown_usage_is_accounted_without_claiming_nonuse(engine):
    engine.record_retrieval(
        retrieval_id="procedure", response={"procedures": [{"id": "proc_test"}]}
    )
    result = engine.feedback(
        retrieval_id="procedure",
        procedure_feedback=[
            {
                "procedure_id": "proc_test",
                "use": "unassessable",
                "reason": "Earlier usage notes were lost.",
            }
        ],
        coverage="complete",
    )
    assert result["rejected"] == []
    assert result["coverage"] == "complete"


def test_legacy_recurrence_is_adopted_without_resetting_genuine_history(engine):
    sid = expose(engine)[0]
    assess(engine, sid, "used")
    conn = engine.db.connect()
    conn.execute("DELETE FROM feedback_recurrence_state WHERE schema_id = ?", (sid,))
    # Simulate pre-upgrade recurrence including genuine recall history.
    conn.execute(
        "UPDATE schemas SET facets_json = json_set(facets_json, '$.recurrence_count', 7) WHERE id = ?",
        (sid,),
    )
    conn.commit()
    assess(engine, sid, "used")
    assert engine.schemas.get(sid).facets["recurrence_count"] == 7
    assess(engine, sid, "not_used")
    assert engine.schemas.get(sid).facets["recurrence_count"] == 6


def test_partial_acceptance_never_advertises_complete_target_events(engine):
    a, b = expose(engine, count=2)
    engine.feedback(
        retrieval_id="r1",
        memory_feedback=[
            {"memory_id": f"sch_{a}", "assessment": "not_used"},
            {"memory_id": f"sch_{b}", "assessment": "useful"},
        ],
        coverage="complete",
    )
    assert (
        engine.db.connect()
        .execute(
            "SELECT COUNT(*) FROM feedback_events WHERE status='accepted' AND coverage='complete'"
        )
        .fetchone()[0]
        == 0
    )
