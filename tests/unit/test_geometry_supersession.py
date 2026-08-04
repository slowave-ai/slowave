"""Unit tests for remember()'s neighbor-flagging behavior.

remember() no longer classifies relations (supersedes/refines/relates_to)
itself — see private/docs/iterations/20260720_supersession_classification_investigation.md
for why: it used a cruder, single-signal classifier (raw
SupersessionManifold.direction_score, no facet or containment check) than
GeometricContradictionJudge, the one classifier consolidation uses, and on
single-episode schemas (the common case for a fresh remember) it had no
facet signal to fall back on.

remember()'s only side effect on existing schemas now (engine.py remember()):
  same scope AND cosine >= COS_THRESHOLD_EXTENDED_SAME_SCOPE (0.70)
  AND candidate is not profile-layer (preference/constraint/habit/...)
    → flag candidate is_labile=True. No relation written, no status change.
  otherwise (cosine too low, different scope, or profile-layer)
    → no action at all.

is_labile=True is picked up by consolidation's reconsolidate_labile_schemas(),
which runs the real judge with proper facet/chronology data — not tested
here, see tests/unit/test_reconsolidation.py-equivalent consolidation tests.
"""

from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path

import numpy as np
import pytest

from slowave.core.config import SlowaveConfig
from slowave.core.engine import SlowaveEngine
from slowave.core.supersession_manifold import COS_THRESHOLD_EXTENDED_SAME_SCOPE

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DIM = 32


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / (n + 1e-12)


def _make_pair(cos_target: float, dim: int = DIM) -> tuple[np.ndarray, np.ndarray]:
    """Return two unit vectors with cosine ≈ cos_target."""
    base = _unit(np.ones(dim, dtype=np.float32))
    perp = np.zeros(dim, dtype=np.float32)
    perp[0] = 1.0
    perp = _unit(perp - np.dot(perp, base) * base)
    angle = np.arccos(np.clip(cos_target, -1.0, 1.0))
    b = _unit((np.cos(angle) * base + np.sin(angle) * perp).astype(np.float32))
    return base, b


class _ControlledEncoder:
    """Returns pre-defined embeddings by text key, deterministic random otherwise."""

    def __init__(self, mapping: dict[str, np.ndarray], dim: int = DIM):
        self._map = mapping
        self._dim = dim

    def encode(self, text: str) -> np.ndarray:
        if text in self._map:
            return self._map[text].copy()
        seed = int(abs(hash(text)) % (2**31))
        v = np.random.default_rng(seed).standard_normal(self._dim).astype(np.float32)
        return _unit(v)


@pytest.fixture
def tmp_db() -> str:
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    yield tmp.name
    for ext in ("", "-wal", "-shm"):
        p = Path(tmp.name + ext)
        if p.exists():
            p.unlink()


def _eng(tmp_db: str, encoder: _ControlledEncoder) -> SlowaveEngine:
    cfg = SlowaveConfig(db_path=tmp_db, dim=DIM, disable_encoder=True)
    eng = SlowaveEngine(cfg)
    eng.encoder = encoder
    return eng


def _remember(eng: SlowaveEngine, text: str, scope: str | None = None, type: str = "fact") -> int:
    return eng.remember(content=text, type=type, scope=scope).schema_id


def _relation_count(tmp_db: str) -> int:
    conn = sqlite3.connect(tmp_db)
    n = conn.execute("SELECT COUNT(*) FROM schema_relations").fetchone()[0]
    conn.close()
    return int(n)


# ---------------------------------------------------------------------------
# Same-scope: close candidates get flagged labile, nothing else
# ---------------------------------------------------------------------------


class TestSameScopeLabileFlagging:
    def test_close_candidate_flagged_labile(self, tmp_db: str) -> None:
        v_old, v_new = _make_pair(0.93)
        enc = _ControlledEncoder({"old": v_old, "new": v_new})
        eng = _eng(tmp_db, enc)

        old_id = _remember(eng, "old", scope="project:test")
        _remember(eng, "new", scope="project:test")

        old_schema = eng.schemas.get(old_id)
        assert old_schema.is_labile, "Close same-scope candidate must be flagged labile"
        assert old_schema.status == "active", "Flagging labile must not change status"
        eng.close()

    def test_extended_range_candidate_flagged_labile(self, tmp_db: str) -> None:
        """cos in [0.70, 0.85) — the old 'extended range' tier — gets the
        same single treatment as any other close same-scope candidate now."""
        v_old, v_new = _make_pair(0.75)
        enc = _ControlledEncoder({"old": v_old, "new": v_new})
        eng = _eng(tmp_db, enc)

        old_id = _remember(eng, "old", scope="project:test")
        _remember(eng, "new", scope="project:test")

        assert eng.schemas.get(old_id).is_labile
        eng.close()

    def test_cosine_below_threshold_no_action(self, tmp_db: str) -> None:
        v_old, v_new = _make_pair(0.50)
        enc = _ControlledEncoder({"old": v_old, "new": v_new})
        eng = _eng(tmp_db, enc)

        old_id = _remember(eng, "old", scope="project:test")
        _remember(eng, "new", scope="project:test")

        old_schema = eng.schemas.get(old_id)
        assert not old_schema.is_labile
        assert old_schema.status == "active"
        eng.close()

    def test_no_relation_edge_ever_written(self, tmp_db: str) -> None:
        """remember() must never write to schema_relations itself anymore —
        that's consolidation's job now."""
        v_old, v_new = _make_pair(0.93)
        enc = _ControlledEncoder({"old": v_old, "new": v_new})
        eng = _eng(tmp_db, enc)

        _remember(eng, "old", scope="project:test")
        _remember(eng, "new", scope="project:test")

        assert _relation_count(tmp_db) == 0
        eng.close()

    def test_status_never_changed_by_remember(self, tmp_db: str) -> None:
        """remember() must never mark a candidate superseded/contradicted
        itself, regardless of how similar the new content is."""
        v_old, v_new = _make_pair(0.99)
        enc = _ControlledEncoder({"old": v_old, "new": v_new})
        eng = _eng(tmp_db, enc)

        old_id = _remember(eng, "old", scope="project:test")
        result = _remember_result(eng, "new", scope="project:test")

        assert eng.schemas.get(old_id).status == "active"
        assert result.superseded_schema_ids == []
        eng.close()

    def test_superseded_candidate_not_touched(self, tmp_db: str) -> None:
        """A candidate that's already superseded/contradicted must not be
        re-flagged — remember() only considers active/needs_review."""
        v_old, v_new = _make_pair(0.93)
        enc = _ControlledEncoder({"old": v_old, "new": v_new})
        eng = _eng(tmp_db, enc)

        old_id = _remember(eng, "old", scope="project:test")
        eng.schemas.update_status(old_id, status="superseded", salience=0.05)
        _remember(eng, "new", scope="project:test")

        old_schema = eng.schemas.get(old_id)
        assert not old_schema.is_labile
        assert old_schema.status == "superseded"
        eng.close()


def _remember_result(eng: SlowaveEngine, text: str, scope: str | None = None):
    return eng.remember(content=text, type="fact", scope=scope)


# ---------------------------------------------------------------------------
# Profile-layer guard: preferences/constraints/habits are never flagged
# ---------------------------------------------------------------------------


class TestProfileLayerGuard:
    def test_preference_candidate_not_flagged_labile(self, tmp_db: str) -> None:
        v_old, v_new = _make_pair(0.95)
        enc = _ControlledEncoder({"old": v_old, "new": v_new})
        eng = _eng(tmp_db, enc)

        old_id = _remember(eng, "old", scope="project:test", type="preference")
        _remember(eng, "new", scope="project:test", type="preference")

        assert not eng.schemas.get(old_id).is_labile
        eng.close()

    def test_constraint_candidate_not_flagged_labile(self, tmp_db: str) -> None:
        v_old, v_new = _make_pair(0.95)
        enc = _ControlledEncoder({"old": v_old, "new": v_new})
        eng = _eng(tmp_db, enc)

        old_id = _remember(eng, "old", scope="project:test", type="constraint")
        _remember(eng, "new", scope="project:test", type="constraint")

        assert not eng.schemas.get(old_id).is_labile
        eng.close()

    def test_fact_candidate_still_flagged_labile(self, tmp_db: str) -> None:
        """Regression: the profile guard must not block non-profile types."""
        v_old, v_new = _make_pair(0.95)
        enc = _ControlledEncoder({"old": v_old, "new": v_new})
        eng = _eng(tmp_db, enc)

        old_id = _remember(eng, "old", scope="project:test", type="fact")
        _remember(eng, "new", scope="project:test", type="fact")

        assert eng.schemas.get(old_id).is_labile
        eng.close()


# ---------------------------------------------------------------------------
# Cross-scope: unaffected by the relation-classification removal, but no
# longer gated on direction_score either (2026-07-23) — see
# private/docs/iterations/20260723_part_of_audit_and_brain_alignment_review.md.
# Once a candidate clears the cross-scope cosine floor it reinforces
# unconditionally; there's no more "manifold" to mock.
# ---------------------------------------------------------------------------


class TestCrossScopeGeometry:
    def test_cross_scope_candidate_never_flagged_labile(self, tmp_db: str) -> None:
        v_a, v_b = _make_pair(0.95)
        enc = _ControlledEncoder({"content_a": v_a, "content_b": v_b})
        eng = _eng(tmp_db, enc)

        id_a = _remember(eng, "content_a", scope="project:alpha")
        _remember(eng, "content_b", scope="project:beta")

        assert not eng.schemas.get(id_a).is_labile
        eng.close()

    def test_same_concept_cross_scope_reinforces(self, tmp_db: str) -> None:
        """cos >= COS_THRESHOLD_CROSS_SCOPE → reinforce unconditionally
        (feeds the promotion ladder's schema_evidence trail)."""
        v_a, v_b = _make_pair(0.92)
        enc = _ControlledEncoder({"content_a": v_a, "content_b": v_b})
        eng = _eng(tmp_db, enc)

        id_a = _remember(eng, "content_a", scope="project:alpha")
        salience_before = eng.schemas.get(id_a).salience

        _remember(eng, "content_b", scope="project:beta")

        assert eng.schemas.get(id_a).salience > salience_before
        eng.close()

    def test_cross_scope_reinforces_regardless_of_former_divergence_signal(
        self, tmp_db: str
    ) -> None:
        """Before 2026-07-23 this cosine band could still be skipped if
        direction_score read as "value diverged". That gate is gone —
        direction_score was never shown to discriminate this reliably (same
        finding that collapsed supersedes/refines into relates_to), so any
        candidate clearing the cross-scope cosine floor now reinforces."""
        v_a, v_b = _make_pair(0.92)
        enc = _ControlledEncoder({"content_a": v_a, "content_b": v_b})
        eng = _eng(tmp_db, enc)

        id_a = _remember(eng, "content_a", scope="project:alpha")
        salience_before = eng.schemas.get(id_a).salience

        _remember(eng, "content_b", scope="project:beta")

        assert eng.schemas.get(id_a).salience > salience_before
        eng.close()

    def test_cross_scope_below_threshold_no_action(self, tmp_db: str) -> None:
        v_a, v_b = _make_pair(0.50)
        enc = _ControlledEncoder({"content_a": v_a, "content_b": v_b})
        eng = _eng(tmp_db, enc)

        id_a = _remember(eng, "content_a", scope="project:alpha")
        salience_before = eng.schemas.get(id_a).salience

        _remember(eng, "content_b", scope="project:beta")

        assert eng.schemas.get(id_a).salience == salience_before
        eng.close()

    def test_cross_scope_reinforce_records_evidence(self, tmp_db: str) -> None:
        """Cross-scope reinforce records schema_evidence linking the beta raw
        event — required for the promotion ladder's cross-scope-count check."""
        v_a, v_b = _make_pair(0.92)
        enc = _ControlledEncoder({"content_a": v_a, "content_b": v_b})
        eng = _eng(tmp_db, enc)

        id_a = _remember(eng, "content_a", scope="project:alpha")
        _remember(eng, "content_b", scope="project:beta")

        conn = sqlite3.connect(tmp_db)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM schema_evidence WHERE schema_id = ? AND raw_event_id IS NOT NULL",
            (id_a,),
        ).fetchall()
        conn.close()

        assert len(rows) > 0
        eng.close()

    def test_cross_scope_preference_treated_as_same_concept(self, tmp_db: str) -> None:
        """Profile-layer candidates reinforce cross-scope same as everyone
        else now (a mismatched preference across scopes is still fine to
        reinforce; it's never eligible for supersession either way)."""
        v_a, v_b = _make_pair(0.92)
        enc = _ControlledEncoder({"content_a": v_a, "content_b": v_b})
        eng = _eng(tmp_db, enc)

        id_a = _remember(eng, "content_a", scope="project:alpha", type="preference")
        salience_before = eng.schemas.get(id_a).salience

        _remember(eng, "content_b", scope="project:beta", type="preference")

        assert eng.schemas.get(id_a).salience > salience_before
        eng.close()


# ---------------------------------------------------------------------------
# Threshold constant sanity
# ---------------------------------------------------------------------------


def test_extended_same_scope_threshold_in_valid_range() -> None:
    assert 0.0 < COS_THRESHOLD_EXTENDED_SAME_SCOPE < 1.0
