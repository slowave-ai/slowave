"""Tests for the profile-layer geometry guard in remember().

remember() no longer supersedes anything itself (see
test_geometry_supersession.py) — its only side effect on an existing
candidate is flagging it is_labile=True so consolidation's
reconsolidate_labile_schemas() can judge it later with real facet data.
This guard makes sure that flag never lands on profile-layer memories
(preferences, constraints, habits): a preference flipping from "dark mode"
to "light mode" is a divergence, not something worth flagging as a possible
supersession candidate, and GeometricContradictionJudge (the judge that
would eventually see a labile-flagged schema) has no profile-layer
awareness of its own to catch it downstream.
"""

from __future__ import annotations

import numpy as np
import pytest

from slowave.core.config import SlowaveConfig
from slowave.core.engine import SlowaveEngine


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / (n + 1e-12)


def _make_pair(cos_target: float, dim: int = 32) -> tuple[np.ndarray, np.ndarray]:
    """Return two unit vectors with cosine ~= cos_target, so the guard is
    exercised deterministically instead of hoping two distinct strings
    happen to embed close together."""
    base = _unit(np.ones(dim, dtype=np.float32))
    perp = np.zeros(dim, dtype=np.float32)
    perp[0] = 1.0
    perp = _unit(perp - np.dot(perp, base) * base)
    angle = np.arccos(np.clip(cos_target, -1.0, 1.0))
    b = _unit((np.cos(angle) * base + np.sin(angle) * perp).astype(np.float32))
    return base, b


class _ControlledEncoder:
    def __init__(self, mapping: dict[str, np.ndarray], dim: int = 32):
        self._map = mapping
        self._dim = dim

    def encode(self, text: str) -> np.ndarray:
        if text in self._map:
            return self._map[text].copy()
        seed = int(abs(hash(text)) % (2**31))
        v = np.random.default_rng(seed).standard_normal(self._dim).astype(np.float32)
        return _unit(v)


class _StubEncoder:
    def __init__(self, dim: int = 32):
        self._dim = dim

    def encode(self, text: str) -> np.ndarray:
        seed = int(abs(hash(text)) % (2**31))
        v = np.random.default_rng(seed).standard_normal(self._dim).astype(np.float32)
        return v / (np.linalg.norm(v) + 1e-12)


def _make_engine(tmp_path, encoder, dim: int = 32) -> SlowaveEngine:
    eng = SlowaveEngine(
        SlowaveConfig(db_path=str(tmp_path / "test.db"), dim=dim, disable_encoder=True)
    )
    eng.encoder = encoder
    return eng


@pytest.fixture()
def eng(tmp_path):
    engine = _make_engine(tmp_path, _StubEncoder())
    yield engine
    engine.close()


# ---------------------------------------------------------------------------
# profile schema_class guard — close candidates, deterministic cosine
# ---------------------------------------------------------------------------


def test_preference_close_candidate_not_flagged_labile(tmp_path):
    """A close (cos ~ 0.95) same-scope preference candidate must not be
    flagged is_labile — the profile guard blocks it before that point."""
    v_old, v_new = _make_pair(0.95)
    enc = _ControlledEncoder({"old": v_old, "new": v_new})
    eng = _make_engine(tmp_path, enc)

    r1 = eng.remember(content="old", type="preference")
    schema_a = eng.schemas.get(r1.schema_id)
    assert schema_a.facets.get("schema_class") == "preference"
    assert schema_a.facets.get("memory_layer") == "profile"

    eng.remember(content="new", type="preference")

    assert not eng.schemas.get(r1.schema_id).is_labile
    eng.close()


def test_constraint_close_candidate_not_flagged_labile(tmp_path):
    v_old, v_new = _make_pair(0.95)
    enc = _ControlledEncoder({"old": v_old, "new": v_new})
    eng = _make_engine(tmp_path, enc)

    r1 = eng.remember(content="old", type="constraint")
    eng.remember(content="new", type="constraint")

    assert not eng.schemas.get(r1.schema_id).is_labile
    eng.close()


def test_habit_close_candidate_not_flagged_labile(tmp_path):
    v_old, v_new = _make_pair(0.95)
    enc = _ControlledEncoder({"old": v_old, "new": v_new})
    eng = _make_engine(tmp_path, enc)

    r1 = eng.remember(content="old", type="habit")
    eng.remember(content="new", type="habit")

    assert not eng.schemas.get(r1.schema_id).is_labile
    eng.close()


def test_memory_layer_profile_blocks_labile_flag(tmp_path):
    """Direct test: candidates with memory_layer == 'profile' are guarded
    even via a type (interaction_preference) that doesn't literally say
    'preference'."""
    v_old, v_new = _make_pair(0.95)
    enc = _ControlledEncoder({"old": v_old, "new": v_new})
    eng = _make_engine(tmp_path, enc)

    r1 = eng.remember(content="old", type="interaction_preference")
    schema = eng.schemas.get(r1.schema_id)
    assert schema.facets.get("memory_layer") == "profile"

    eng.remember(content="new", type="interaction_preference")

    assert not eng.schemas.get(r1.schema_id).is_labile
    eng.close()


def test_fact_close_candidate_is_flagged_labile(tmp_path):
    """Regression: fact-type schemas must still be eligible for the labile
    flag — the guard is only for profile-layer memories."""
    v_old, v_new = _make_pair(0.95)
    enc = _ControlledEncoder({"old": v_old, "new": v_new})
    eng = _make_engine(tmp_path, enc)

    r1 = eng.remember(content="old", type="fact")
    eng.remember(content="new", type="fact")

    assert eng.schemas.get(r1.schema_id).is_labile
    eng.close()


def test_lesson_and_warning_are_domain_not_profile(tmp_path):
    """Regression: lesson/warning are domain-layer, not profile — they
    should NOT be blocked by the profile guard."""
    v_old, v_new = _make_pair(0.95)
    enc = _ControlledEncoder({"old": v_old, "new": v_new})
    eng = _make_engine(tmp_path, enc)

    r1 = eng.remember(content="old", type="warning")
    schema = eng.schemas.get(r1.schema_id)
    assert schema.facets.get("memory_layer") == "domain"

    eng.remember(content="new", type="warning")

    assert eng.schemas.get(r1.schema_id).is_labile
    eng.close()


# ---------------------------------------------------------------------------
# Sanity: remember() never supersedes, regardless of type (no-crash checks
# with realistic, non-forced embeddings)
# ---------------------------------------------------------------------------


def test_preference_remember_result_has_no_superseded_ids(eng):
    r1 = eng.remember(content="The user prefers dark mode in their editor.", type="preference")
    assert len(r1.superseded_schema_ids) == 0

    r2 = eng.remember(content="The user prefers light mode in their editor.", type="preference")
    assert len(r2.superseded_schema_ids) == 0


def test_fact_remember_result_has_no_superseded_ids(eng):
    """remember() never populates superseded_schema_ids for any type now —
    that's consolidation's job. Just verifying no crash for a non-profile type."""
    eng.remember(content="The project uses SQLite for storage.", type="fact")
    r2 = eng.remember(content="The project uses Postgres for storage.", type="fact")
    assert isinstance(r2.superseded_schema_ids, list)
    assert len(r2.superseded_schema_ids) == 0
