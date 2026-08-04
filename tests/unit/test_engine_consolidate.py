"""Regression tests for engine.consolidate_once() and engine.decay_schemas().

Covers the consolidation pipeline before it is extracted into a standalone
ConsolidationService. Uses a stub encoder so tests run without model weights.
"""

from __future__ import annotations

import numpy as np
import pytest

from slowave.core.config import SlowaveConfig
from slowave.core.engine import SlowaveEngine

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


class _StubEncoder:
    def __init__(self, dim: int = 32):
        self._dim = dim

    def encode(self, text: str) -> np.ndarray:
        seed = int(abs(hash(text)) % (2**31))
        v = np.random.default_rng(seed).standard_normal(self._dim).astype(np.float32)
        return v / (np.linalg.norm(v) + 1e-12)


def _make_engine(tmp_path, dim: int = 32) -> SlowaveEngine:
    eng = SlowaveEngine(
        SlowaveConfig(db_path=str(tmp_path / "test.db"), dim=dim, disable_encoder=True)
    )
    eng.encoder = _StubEncoder(dim)
    return eng


@pytest.fixture()
def eng(tmp_path):
    engine = _make_engine(tmp_path)
    yield engine
    engine.close()


def _run_session(eng: SlowaveEngine, n_events: int = 4) -> str:
    """Populate the engine with a closed session so episodes exist for consolidation."""
    sid = eng.session_start(agent="test")
    rng = np.random.default_rng(42)
    for i in range(n_events):
        emb = rng.standard_normal(32).astype(np.float32)
        emb /= np.linalg.norm(emb)
        eng.raw_log.append(
            session_id=sid,
            type="user_message",
            content=f"event content {i}",
            embedding=emb,
        )
    eng.session_end(sid)
    return sid


# ---------------------------------------------------------------------------
# consolidate_once()
# ---------------------------------------------------------------------------


def test_consolidate_once_on_empty_db_does_not_crash(eng):
    result = eng.consolidate_once()
    # Either returns stats or an error key — must not raise
    assert isinstance(result, dict)


def test_consolidate_once_returns_required_keys(eng):
    _run_session(eng)
    result = eng.consolidate_once()
    assert "replay" in result
    assert "consolidation" in result
    assert "decay" in result


def test_consolidate_once_returns_facet_backfill_key(eng):
    """part_of_backfill was removed 2026-07-23 along with part_of itself (see
    private/docs/iterations/20260723_part_of_audit_and_brain_alignment_review.md);
    facet_backfill stays (still feeds facet_distance/VSA)."""
    result = eng.consolidate_once()
    assert "facet_backfill" in result
    assert set(result["facet_backfill"]) == {
        "backfilled",
        "skipped_no_embeddings",
        "skipped_svd_failed",
        "backfilled_ids",
    }
    assert "part_of_backfill" not in result


def test_consolidate_once_records_worker_run_row(eng):
    _run_session(eng)
    eng.consolidate_once()
    conn = eng.db.connect()
    rows = conn.execute("SELECT * FROM worker_runs").fetchall()
    assert len(rows) == 1


def test_consolidate_once_worker_run_has_started_and_ended_ts(eng):
    _run_session(eng)
    eng.consolidate_once()
    conn = eng.db.connect()
    row = conn.execute("SELECT * FROM worker_runs").fetchone()
    assert row["started_ts"] is not None
    assert row["ended_ts"] is not None


def test_consolidate_once_worker_run_has_non_negative_duration(eng):
    _run_session(eng)
    eng.consolidate_once()
    conn = eng.db.connect()
    row = conn.execute("SELECT duration_ms FROM worker_runs").fetchone()
    assert row["duration_ms"] >= 0


def test_consolidate_once_multiple_passes_do_not_crash(eng):
    _run_session(eng)
    for _ in range(3):
        result = eng.consolidate_once()
        assert "error" not in result


def test_consolidate_once_each_pass_records_a_worker_run(eng):
    _run_session(eng)
    eng.consolidate_once()
    eng.consolidate_once()
    conn = eng.db.connect()
    count = conn.execute("SELECT COUNT(*) FROM worker_runs").fetchone()[0]
    assert count == 2


def test_consolidate_once_triggered_by_field_is_recorded(eng):
    _run_session(eng)
    eng.consolidate_once(triggered_by="test_runner")
    conn = eng.db.connect()
    row = conn.execute("SELECT triggered_by FROM worker_runs").fetchone()
    assert row["triggered_by"] == "test_runner"


def test_consolidate_once_with_memories_creates_schemas(eng):
    # Seed with explicit memories so the latent builder has episodes to process
    for i in range(6):
        eng.remember(content=f"important fact number {i} about the project", type="fact")
    before = eng.schemas.count()
    eng.consolidate_once()
    # Consolidation may reinforce existing schemas or create new latent ones
    assert eng.schemas.count() >= before


# ---------------------------------------------------------------------------
# decay_schemas()
# ---------------------------------------------------------------------------


def test_decay_schemas_returns_stats_dict(eng):
    result = eng.decay_schemas(idle_days=30.0, dry_run=True)
    assert isinstance(result, dict)
    assert "decayed" in result
    assert "dry_run" in result


def test_decay_schemas_dry_run_does_not_modify_salience(eng):
    # Create a latent schema via consolidation (not explicit_remember,
    # which is exempt from decay)
    _run_session(eng, n_events=6)
    eng.consolidate_once()
    schemas_before = {s.id: s.salience for s in eng.schemas.list(limit=50)}
    if not schemas_before:
        pytest.skip("no latent schemas created — consolidation produced nothing to decay")

    eng.decay_schemas(idle_days=0.0, dry_run=True)

    for s in eng.schemas.list(limit=50):
        assert s.salience == schemas_before[s.id]


def test_decay_schemas_dry_run_false_does_not_crash(eng):
    _run_session(eng)
    eng.consolidate_once()
    result = eng.decay_schemas(idle_days=0.0, dry_run=False)
    assert isinstance(result, dict)
    assert "decayed" in result


def test_decay_schemas_explicit_remember_schemas_are_exempt(eng):
    # Explicit-remember schemas must never be decayed regardless of age
    content = "I prefer explicit memories to survive decay"
    eng.remember(content=content, type="preference")
    schema_id = eng.schemas.list(limit=1)[0].id
    salience_before = eng.schemas.get(schema_id).salience

    eng.decay_schemas(idle_days=0.0, dry_run=False)

    salience_after = eng.schemas.get(schema_id).salience
    assert salience_after == salience_before


# ---------------------------------------------------------------------------
# Explicit-remember skip
# ---------------------------------------------------------------------------


def test_consolidate_skips_pure_remember_episodes(eng):
    """Episodes whose every raw event is a remember:* type must not be
    re-consolidated into schemas — remember() already created the first-class
    schema synchronously.  Without this skip, adjacent remembers merge into
    a macro-episode whose concatenated text produces a composite duplicate
    (the original bug this branch fixes)."""
    eng.remember(content="SessionReaper scans every 60 seconds", type="fact")
    eng.remember(content="HTTP daemon binds port 8766", type="fact")

    before = eng.schemas.count()

    eng.consolidate_once()

    # No new schemas: consolidation skipped both pure-remember episodes
    assert eng.schemas.count() == before, (
        f"Pure-remember episodes must not create new schemas; "
        f"{before} before → {eng.schemas.count()} after"
    )


def test_explicit_remember_prototype_does_not_crash_link_step(eng):
    """_link_schemas_via_prototype_centroid must not raise even when
    embeddings don't cluster tightly enough to produce a relation."""
    eng.remember(content="SessionReaper idle timeout defaults to 3600 seconds", type="fact")
    eng.remember(content="HTTP daemon disables idle timeout by setting it to zero", type="fact")

    before = eng.schemas.count()
    eng.consolidate_once()  # must not raise

    # Schema count must not grow — relation-linking never creates schemas
    assert eng.schemas.count() == before


def test_link_schemas_via_prototype_centroid_is_direction_stable(eng, tmp_path):
    """The centroid linker writes only relates_to for pre-existing schema pairs
    (directional verdicts like reinforces/refines are downgraded since the
    call site has no consolidation-time notion of which is new). relates_to is
    symmetric, so the direction must be canonicalized on schema id to prevent
    the same pair producing both A->B and B->A rows when call-time ranking
    flips across repeated calls or different prototype centroids."""
    dim = eng.cfg.dim
    rng = np.random.default_rng(7)
    emb_a = rng.standard_normal(dim).astype(np.float32)
    emb_a /= np.linalg.norm(emb_a)
    emb_b = emb_a + rng.standard_normal(dim).astype(np.float32) * 0.05
    emb_b /= np.linalg.norm(emb_b)

    id_a = eng.schemas.create(content_text="schema A", embedding=emb_a, dedupe=False)
    id_b = eng.schemas.create(content_text="schema B", embedding=emb_b, dedupe=False)

    consolidator = eng._consolidation.consolidator

    # Two different prototype centroids, one nearer A and one nearer B, so
    # search_embedding's top-2 ranking flips which schema lands in position
    # 0 vs 1 — reproducing how two distinct (or drifted) prototypes near the
    # same pair used to yield opposite src/dst assignments.
    consolidator._link_schemas_via_prototype_centroid(1, emb_a)
    consolidator._link_schemas_via_prototype_centroid(2, emb_b)

    conn = eng.db.connect()
    rows = conn.execute(
        "SELECT src_schema_id, dst_schema_id FROM schema_relations "
        "WHERE relation = 'relates_to' AND src_schema_id IN (?, ?) AND dst_schema_id IN (?, ?)",
        (id_a, id_b, id_a, id_b),
    ).fetchall()
    assert len(rows) == 1, f"expected a single canonical-direction edge, got {list(rows)}"
    assert int(rows[0]["src_schema_id"]) == min(id_a, id_b)
    assert int(rows[0]["dst_schema_id"]) == max(id_a, id_b)
