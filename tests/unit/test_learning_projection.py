"""WP-2 canonical declarative learning projection tests.

Covers the acceptance criteria from the execution spec (rev 2, §2/WP-2):
duplicate submissions are no-ops, refinements correct in one step (including
across a cap), terminal stale semantics block resurrection, adoption does not
double-count historical effects, and replay-from-ledger equals incremental
application. Uses temporary databases only.
"""

import pytest

from slowave.core.config import SlowaveConfig
from slowave.core.engine import SlowaveEngine
from slowave.symbolic.schema_store import SALIENCE_CEILING

SCOPE = "project:projection"


def _engine(tmp_path, name: str) -> SlowaveEngine:
    return SlowaveEngine(
        SlowaveConfig(db_path=str(tmp_path / f"{name}.db"), dim=8, disable_encoder=True)
    )


def _expose(engine: SlowaveEngine, retrieval_id: str, schema_ids: list[int]) -> None:
    """Record an exposure so feedback targets are authorized."""
    engine.record_retrieval(
        retrieval_id=retrieval_id,
        scope_id=SCOPE,
        query="Verify the release checks",
        response={
            "schemas": [
                {"id": f"sch_{sid}", "content": engine.schemas.get(sid).content_text}
                for sid in schema_ids
            ]
        },
    )


def _salience(engine: SlowaveEngine, schema_id: int) -> float:
    return float(engine.schemas.get(schema_id).salience)


def _confidence(engine: SlowaveEngine, schema_id: int) -> float:
    return float(engine.schemas.get(schema_id).confidence)


def _used(engine: SlowaveEngine, retrieval_id: str, schema_id: int) -> None:
    result = engine.feedback(
        retrieval_id=retrieval_id,
        memory_feedback=[{"memory_id": f"sch_{schema_id}", "assessment": "used"}],
        coverage="complete",
    )
    assert result["rejected"] == []


def _refine(engine: SlowaveEngine, retrieval_id: str, schema_id: int, assessment: str) -> None:
    result = engine.feedback(
        retrieval_id=retrieval_id,
        memory_feedback=[{"memory_id": f"sch_{schema_id}", "assessment": assessment}],
        coverage="complete",
    )
    assert result["rejected"] == []


def _state_row(engine: SlowaveEngine, schema_id: int):
    return (
        engine.db.connect()
        .execute("SELECT * FROM learning_projection_state WHERE schema_id = ?", (schema_id,))
        .fetchone()
    )


def _content() -> str:
    return "Release checks run black, isort, and mypy."


def _schema(engine: SlowaveEngine) -> int:
    return engine.schemas.create(
        content_text=_content(),
        facets={},
        embedding=None,
        scope_id=SCOPE,
        dedupe=False,
    )


def test_duplicate_used_feedback_applies_once(tmp_path):
    """F2: identical repeated `used` feedback must not re-apply reinforcement."""
    engine = _engine(tmp_path, "duplicate")
    try:
        sid = _schema(engine)
        _expose(engine, "r1", [sid])
        _used(engine, "r1", sid)
        after_first = _salience(engine, sid)
        assert after_first == pytest.approx(1.10)

        _used(engine, "r1", sid)
        assert _salience(engine, sid) == pytest.approx(after_first)
        _used(engine, "r1", sid)
        assert _salience(engine, sid) == pytest.approx(after_first)
    finally:
        engine.close()


def test_independent_retrievals_apply_once_each(tmp_path):
    """Each distinct retrieval observing a schema is one new observation."""
    engine = _engine(tmp_path, "independent")
    try:
        sid = _schema(engine)
        _expose(engine, "r1", [sid])
        _expose(engine, "r2", [sid])
        _used(engine, "r1", sid)
        assert _salience(engine, sid) == pytest.approx(1.10)
        _used(engine, "r2", sid)
        assert _salience(engine, sid) == pytest.approx(1.20)
    finally:
        engine.close()


def test_refinement_removes_contribution_in_one_step(tmp_path):
    """Refining used -> irrelevant restores the prior value exactly."""
    engine = _engine(tmp_path, "refinement")
    try:
        sid = _schema(engine)
        _expose(engine, "r1", [sid])
        _used(engine, "r1", sid)
        assert _salience(engine, sid) == pytest.approx(1.10)
        _refine(engine, "r1", sid, "irrelevant")
        assert _salience(engine, sid) == pytest.approx(1.0)
    finally:
        engine.close()


def test_refinement_across_salience_cap_restores_exactly(tmp_path):
    """Cap-aware bookkeeping: a clamped increment is undone by its effective size."""
    engine = _engine(tmp_path, "salience_cap")
    try:
        sid = _schema(engine)
        # Push near the ceiling: only part of the next increment can land.
        engine.schemas.reinforce(sid, amount=SALIENCE_CEILING - 1.0 - 0.05)
        pre = _salience(engine, sid)
        assert pre == pytest.approx(SALIENCE_CEILING - 0.05)
        _expose(engine, "r1", [sid])
        _used(engine, "r1", sid)
        assert _salience(engine, sid) == pytest.approx(SALIENCE_CEILING)
        _refine(engine, "r1", sid, "irrelevant")
        assert _salience(engine, sid) == pytest.approx(pre)
    finally:
        engine.close()


def test_refinement_across_confidence_cap_restores_exactly(tmp_path):
    """Confidence at its ceiling must not drift down after a refinement."""
    engine = _engine(tmp_path, "confidence_cap")
    try:
        sid = _schema(engine)
        assert _confidence(engine, sid) == pytest.approx(1.0)
        _expose(engine, "r1", [sid])
        _used(engine, "r1", sid)
        assert _confidence(engine, sid) == pytest.approx(1.0)
        _refine(engine, "r1", sid, "irrelevant")
        assert _confidence(engine, sid) == pytest.approx(1.0)
    finally:
        engine.close()


def test_stale_terminal_blocks_resurrection(tmp_path):
    """A retired schema keeps its terminal state; later use cannot raise it."""
    engine = _engine(tmp_path, "terminal")
    try:
        sid = _schema(engine)
        replacement = engine.schemas.create(
            content_text="Release checks run black, isort, mypy, and the wheel build.",
            facets={},
            embedding=None,
            scope_id=SCOPE,
            dedupe=False,
        )
        _expose(engine, "r1", [sid])
        _used(engine, "r1", sid)
        assert _salience(engine, sid) == pytest.approx(1.10)

        result = engine.feedback(
            retrieval_id="r1",
            memory_feedback=[
                {
                    "memory_id": f"sch_{sid}",
                    "assessment": "stale",
                    "stale_reason": "superseded",
                    "replacement_memory_id": f"sch_{replacement}",
                    "reason": "A newer client-provided memory replaces this claim.",
                }
            ],
            coverage="complete",
        )
        assert result["rejected"] == []
        schema = engine.schemas.get(sid)
        assert schema.status == "stale"
        assert _salience(engine, sid) == pytest.approx(0.05)
        assert _state_row(engine, sid) is None

        # A later positive assessment must not resurrect retired content.
        _expose(engine, "r2", [sid])
        _used(engine, "r2", sid)
        assert engine.schemas.get(sid).status == "stale"
        assert _salience(engine, sid) == pytest.approx(0.05)
    finally:
        engine.close()


def test_legacy_adoption_does_not_double_count_history(tmp_path):
    """Pre-WP-2 ledger rows with baked tallies are adopted, not re-applied."""
    engine = _engine(tmp_path, "legacy")
    try:
        sid = _schema(engine)
        _expose(engine, "r1", [sid])
        _expose(engine, "r2", [sid])
        record = engine._feedback.feedback_events.record
        # Simulate the pre-WP-2 world: accepted ledger rows whose learning was
        # applied by the old in-place tally path.
        for retrieval_id in ("r1", "r2"):
            result = record(
                retrieval_id=retrieval_id,
                memory_feedback=[{"memory_id": f"sch_{sid}", "assessment": "used"}],
                coverage="complete",
                mutation_mode="active",
            )
            assert result["rejected"] == []
        engine.schemas.reinforce(sid, amount=0.10)
        engine.schemas.reinforce(sid, amount=0.10)
        assert _salience(engine, sid) == pytest.approx(1.20)

        # First post-WP-2 observation: exactly one reinforcement, history adopted.
        _expose(engine, "r3", [sid])
        _used(engine, "r3", sid)
        assert _salience(engine, sid) == pytest.approx(1.30)

        # Refining one historical observation self-heals the attribution.
        _refine(engine, "r1", sid, "irrelevant")
        assert _salience(engine, sid) == pytest.approx(1.20)

        # Re-flipping it back restores the attribution again.
        _used(engine, "r1", sid)
        assert _salience(engine, sid) == pytest.approx(1.30)
    finally:
        engine.close()


def test_full_replay_from_ledger_equals_incremental_application(tmp_path):
    """Rebuilding from the canonical ledger matches step-by-step application."""
    incremental = _engine(tmp_path, "incremental")
    rebuilt = _engine(tmp_path, "rebuilt")
    try:
        sid_a = _schema(incremental)
        _expose(incremental, "r1", [sid_a])
        _expose(incremental, "r2", [sid_a])
        _used(incremental, "r1", sid_a)
        _refine(incremental, "r2", sid_a, "irrelevant")
        _used(incremental, "r2", sid_a)
        final_incremental = _salience(incremental, sid_a)
        assert final_incremental == pytest.approx(1.20)

        # Rebuild path: same fixtures, ledger populated without learning,
        # then one replay-mode apply over the full view.
        sid_b = rebuilt.schemas.create(
            content_text=_content(),
            facets={},
            embedding=None,
            scope_id=SCOPE,
            dedupe=False,
        )
        _expose(rebuilt, "r1", [sid_b])
        _expose(rebuilt, "r2", [sid_b])
        record = rebuilt._feedback.feedback_events.record
        record(
            retrieval_id="r1",
            memory_feedback=[{"memory_id": f"sch_{sid_b}", "assessment": "used"}],
            coverage="complete",
            mutation_mode="active",
        )
        record(
            retrieval_id="r2",
            memory_feedback=[{"memory_id": f"sch_{sid_b}", "assessment": "irrelevant"}],
            coverage="complete",
            mutation_mode="active",
        )
        record(
            retrieval_id="r2",
            memory_feedback=[{"memory_id": f"sch_{sid_b}", "assessment": "used"}],
            coverage="complete",
            mutation_mode="active",
        )
        result = rebuilt._feedback.learning_projection.apply_memory_observation(
            sid_b,
            trigger_retrieval_id="r2",
            salience_delta_per_use=0.10,
            confidence_delta_per_use=0.02,
            mode="replay",
        )
        assert result.status == "applied"
        assert _salience(rebuilt, sid_b) == pytest.approx(final_incremental)

        # Replay refuses when bookkeeping already exists (double-apply guard).
        with pytest.raises(ValueError, match="replay requires no existing"):
            rebuilt._feedback.learning_projection.apply_memory_observation(
                sid_b,
                trigger_retrieval_id="r2",
                salience_delta_per_use=0.10,
                confidence_delta_per_use=0.02,
                mode="replay",
            )
    finally:
        incremental.close()
        rebuilt.close()


def test_retry_after_apply_converges(tmp_path):
    """A retried apply is a no-op; the state and schema stay consistent."""
    engine = _engine(tmp_path, "retry")
    try:
        sid = _schema(engine)
        _expose(engine, "r1", [sid])
        _used(engine, "r1", sid)
        salience_after = _salience(engine, sid)
        state_after = _state_row(engine, sid)

        result = engine._feedback.learning_projection.apply_memory_observation(
            sid,
            trigger_retrieval_id="r1",
            salience_delta_per_use=0.10,
            confidence_delta_per_use=0.02,
        )
        assert result.status == "noop"
        assert _salience(engine, sid) == pytest.approx(salience_after)
        new_state = _state_row(engine, sid)
        assert new_state["applied_extra_salience"] == pytest.approx(
            state_after["applied_extra_salience"]
        )
        assert new_state["applied_extra_confidence"] == pytest.approx(
            state_after["applied_extra_confidence"]
        )
    finally:
        engine.close()


def test_used_with_unknown_effect_is_rejected(tmp_path):
    """Section-3: used carries helped/no_effect/harmed; unknown is invalid."""
    engine = _engine(tmp_path, "unknown_effect")
    try:
        sid = _schema(engine)
        _expose(engine, "r1", [sid])
        result = engine.feedback(
            retrieval_id="r1",
            memory_feedback=[
                {"memory_id": f"sch_{sid}", "assessment": "used", "effect": "unknown"}
            ],
            coverage="complete",
        )
        assert result["rejected"][0]["reason"] == "used_requires_helped_no_effect_or_harmed_effect"
        assert _salience(engine, sid) == pytest.approx(1.0)
    finally:
        engine.close()


def test_already_known_is_recorded_without_reinforcement(tmp_path):
    """already_known is a dedup observation: recorded, no reinforcement."""
    engine = _engine(tmp_path, "already_known")
    try:
        sid = _schema(engine)
        _expose(engine, "r1", [sid])
        result = engine.feedback(
            retrieval_id="r1",
            memory_feedback=[{"memory_id": f"sch_{sid}", "assessment": "already_known"}],
            coverage="complete",
        )
        assert result["rejected"] == []
        assert result["applied"]["already_known"] == [f"sch_{sid}"]
        assert _salience(engine, sid) == pytest.approx(1.0)
        # A duplicate is still a no-op.
        result = engine.feedback(
            retrieval_id="r1",
            memory_feedback=[{"memory_id": f"sch_{sid}", "assessment": "already_known"}],
            coverage="complete",
        )
        assert result["applied"]["already_known"] == [f"sch_{sid}"]
        assert _salience(engine, sid) == pytest.approx(1.0)
    finally:
        engine.close()


def test_used_to_already_known_refinement_drops_contribution(tmp_path):
    """Refining used -> already_known removes the contribution in one step."""
    engine = _engine(tmp_path, "known_refine")
    try:
        sid = _schema(engine)
        _expose(engine, "r1", [sid])
        _used(engine, "r1", sid)
        assert _salience(engine, sid) == pytest.approx(1.10)
        result = engine.feedback(
            retrieval_id="r1",
            memory_feedback=[{"memory_id": f"sch_{sid}", "assessment": "already_known"}],
            coverage="complete",
        )
        assert result["rejected"] == []
        assert _salience(engine, sid) == pytest.approx(1.0)
    finally:
        engine.close()


def test_relevance_is_recorded_and_never_gates_usage(tmp_path):
    """Decoupled axes: relevance is stored and never blocks the usage axis."""
    engine = _engine(tmp_path, "relevance")
    try:
        sid = _schema(engine)
        _expose(engine, "r1", [sid])
        # Tension row: irrelevant relevance on a used mark — usage still applies.
        result = engine.feedback(
            retrieval_id="r1",
            memory_feedback=[
                {
                    "memory_id": f"sch_{sid}",
                    "assessment": "used",
                    "effect": "helped",
                    "relevance": "irrelevant",
                }
            ],
            coverage="complete",
        )
        assert result["rejected"] == []
        assert _salience(engine, sid) == pytest.approx(1.10)
        row = _relevance_recorded(engine, "r1", sid)
        assert row["relevance"] == "irrelevant"
        assert row["effect"] == "helped"

        # uncertain relevance: still recorded, usage still applies.
        result = engine.feedback(
            retrieval_id="r1",
            memory_feedback=[
                {
                    "memory_id": f"sch_{sid}",
                    "assessment": "used",
                    "effect": "helped",
                    "relevance": "uncertain",
                }
            ],
            coverage="complete",
        )
        assert result["rejected"] == []
        assert _salience(engine, sid) == pytest.approx(1.10)
    finally:
        engine.close()


def test_invalid_relevance_is_rejected(tmp_path):
    engine = _engine(tmp_path, "bad_relevance")
    try:
        sid = _schema(engine)
        _expose(engine, "r1", [sid])
        result = engine.feedback(
            retrieval_id="r1",
            memory_feedback=[
                {
                    "memory_id": f"sch_{sid}",
                    "assessment": "used",
                    "relevance": "maybe",
                }
            ],
            coverage="complete",
        )
        assert result["rejected"][0]["reason"] == "invalid_relevance"
    finally:
        engine.close()


def _relevance_recorded(engine: SlowaveEngine, retrieval_id: str, schema_id: int):
    conn = engine.db.connect()
    return conn.execute(
        "SELECT relevance, effect FROM feedback_events "
        "WHERE retrieval_id = ? AND target_kind = 'memory' "
        "ORDER BY created_at DESC, rowid DESC LIMIT 1",
        (retrieval_id,),
    ).fetchone()


def test_harmed_effect_reduces_and_refinement_restores(tmp_path):
    """A used+harmed mark subtracts; refining it removes the harm exactly."""
    engine = _engine(tmp_path, "harmed")
    try:
        sid = _schema(engine)
        _expose(engine, "r1", [sid])
        result = engine.feedback(
            retrieval_id="r1",
            memory_feedback=[{"memory_id": f"sch_{sid}", "assessment": "used", "effect": "harmed"}],
            coverage="complete",
        )
        assert result["rejected"] == []
        assert _salience(engine, sid) == pytest.approx(0.90)
        _refine(engine, "r1", sid, "irrelevant")
        assert _salience(engine, sid) == pytest.approx(1.0)
    finally:
        engine.close()


def test_used_with_no_effect_is_recorded_without_reinforcement(tmp_path):
    """A used+no_effect observation is recorded but moves nothing."""
    engine = _engine(tmp_path, "no_effect")
    try:
        sid = _schema(engine)
        _expose(engine, "r1", [sid])
        result = engine.feedback(
            retrieval_id="r1",
            memory_feedback=[
                {"memory_id": f"sch_{sid}", "assessment": "used", "effect": "no_effect"}
            ],
            coverage="complete",
        )
        assert result["rejected"] == []
        assert _salience(engine, sid) == pytest.approx(1.0)
        row = _relevance_recorded(engine, "r1", sid)
        assert row["effect"] == "no_effect"
    finally:
        engine.close()
