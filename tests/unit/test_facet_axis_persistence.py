"""Regression test for the facet-axis persistence fix (2026-07-09).

Companion to test_contradicts_verdict_unreachable.py, which documents the
bug: Consolidator._write_latent_schema used to reconstruct the *old*
schema's facet axes as an unconditional empty placeholder, because raw
facet axes were never persisted anywhere retrievable. That made the
"contradicts" verdict provably unreachable regardless of threshold tuning.

The fix: `schemas` table gained `facet_axes`/`facet_strengths`/`n_facet_axes`
columns (schema.sql), `SchemaStore.create()` persists them, and
`SchemaStore._row_to_schema()` unpacks them back into `Schema.facet_axes`/
`.facet_strengths`. `_write_latent_schema` now uses `related.facet_axes`
(the real, persisted data) to build `old_view` instead of a hardcoded
zero-row matrix, falling back to the placeholder only when the related
schema genuinely has no persisted facet data.

This test uses a REAL SchemaStore (not mocked) end-to-end: create an "old"
schema with real facet axes, then run a divergent "new" schema through
Consolidator._write_latent_schema and confirm the facet-distance-driven
verdict is now reachable. As of the 2026-07-15 relation-taxonomy rewire
(GeometricContradictionJudge.judge()), high facet_distance WITHOUT a
direction_score signal maps to "refines" (elaboration), not "supersedes" --
"supersedes" now requires direction_score, a distinct signal this test's
no-manifold judge never computes (see test_supersedes_via_direction_score
in test_latent_schema.py for that path). The persistence fix under test
here is unchanged: it's still what makes old.facet_axes real instead of an
empty placeholder, which is what makes a facet-distance-driven verdict
reachable at all.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from slowave.core.consolidation import Consolidator
from slowave.latent.schema import GeometricContradictionJudge, GeometricJudgeConfig, LatentSchema
from slowave.symbolic.schema_store import SchemaStore

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SCHEMA_PATH = str(REPO_ROOT / "slowave" / "storage" / "schema.sql")
DIM = 32


@pytest.fixture()
def real_store():
    db_path = str(Path(tempfile.mkdtemp()) / "test.db")
    from slowave.storage.sqlite_db import SQLiteConfig, SQLiteDB

    db = SQLiteDB(SQLiteConfig(path=db_path))
    db.init_schema(SCHEMA_PATH)
    conn = db.connect()
    conn.execute("PRAGMA foreign_keys = OFF")
    schemas = SchemaStore(db, dim=DIM)

    judge = GeometricContradictionJudge(GeometricJudgeConfig())
    cons = Consolidator(
        db=db,
        semantic=MagicMock(),
        episode_text=MagicMock(),
        schemas=schemas,
        encoder=None,
        latent_builder=MagicMock(),
        geometric_judge=judge,
    )
    yield cons, schemas
    db.close()


def _same_topic_centroids(cos_target: float):
    old_centroid = np.zeros(DIM, dtype=np.float32)
    old_centroid[0] = 1.0
    orth = np.zeros(DIM, dtype=np.float32)
    orth[1] = 1.0
    new_centroid = cos_target * old_centroid + float(np.sqrt(1 - cos_target**2)) * orth
    new_centroid = (new_centroid / np.linalg.norm(new_centroid)).astype(np.float32)
    return old_centroid, new_centroid


def test_persisted_facet_axes_round_trip(real_store):
    """create() -> get() preserves facet_axes/facet_strengths exactly."""
    _, schemas = real_store
    axes = np.eye(4, DIM, dtype=np.float32)
    strengths = np.array([1.0, 0.5, 0.3, 0.1], dtype=np.float32)
    sid = schemas.create(
        content_text="old claim",
        embedding=np.ones(DIM, dtype=np.float32) / np.sqrt(DIM),
        facet_axes=axes,
        facet_strengths=strengths,
        dedupe=False,
    )
    fetched = schemas.get(sid)
    assert fetched.facet_axes.shape == (4, DIM)
    np.testing.assert_allclose(fetched.facet_axes, axes)
    np.testing.assert_allclose(fetched.facet_strengths, strengths)


def test_schema_without_facets_round_trips_to_empty(real_store):
    """No facet_axes passed to create() -> get() returns an empty (0, dim)
    matrix, not None -- same semantics as a legacy pre-migration row."""
    _, schemas = real_store
    sid = schemas.create(
        content_text="singleton claim",
        embedding=np.ones(DIM, dtype=np.float32) / np.sqrt(DIM),
        dedupe=False,
    )
    fetched = schemas.get(sid)
    assert fetched.facet_axes.shape == (0, DIM)
    assert fetched.facet_strengths.shape == (0,)


def test_divergent_facets_now_reachable_with_persisted_facet_axes(real_store):
    """The actual fix, end-to-end: an old schema with real, persisted facet
    axes that are maximally divergent from the new schema's facet axes ->
    the geometric judge, invoked through the real _write_latent_schema
    path, now returns "refines" (previously impossible -- old.facet_axes
    was always an empty placeholder, so has_facet_signal was always False
    and no facet-distance-driven verdict could ever fire)."""
    cons, schemas = real_store
    old_centroid, new_centroid = _same_topic_centroids(cos_target=0.85)

    # Old schema's real facet axes: aligned with dims 2-5.
    old_axes = np.zeros((4, DIM), dtype=np.float32)
    for i in range(4):
        old_axes[i, 2 + i] = 1.0
    schemas.create(
        content_text="the deployment target is docker swarm",
        embedding=old_centroid,
        facet_axes=old_axes,
        facet_strengths=np.array([1.0, 0.8, 0.6, 0.4], dtype=np.float32),
        confidence=1.0,
        salience=1.0,
        dedupe=False,
    )

    # New schema's facet axes: aligned with dims 10-13 -- orthogonal to the
    # old schema's axes, i.e. maximally divergent (pair_cos ~ 0 -> facet_dist ~ 1.0).
    new_axes = np.zeros((4, DIM), dtype=np.float32)
    for i in range(4):
        new_axes[i, 10 + i] = 1.0

    new_schema = LatentSchema(
        centroid=new_centroid,
        facet_axes=new_axes,
        facet_strengths=np.array([1.0, 0.8, 0.6, 0.4], dtype=np.float32),
        member_episode_ids=[1, 2, 3, 4, 5],
        central_episode_id=1,
        central_episode_text="the deployment target is now kubernetes",
        mean_ts=100_000,  # >3600s (min_time_delta_to_supersede_s) after old's implicit ts=0
        ts_span_s=100,
        tags=[],
        confidence=0.9,
        support_count=5,  # >= min_support_to_supersede (2)
        facets={"source_kind": "consolidation"},
    )

    outcome, new_id = cons._write_latent_schema(prototype_id=1, schema=new_schema)

    assert outcome == "reinforced"
    relations = (
        schemas.db.connect()
        .execute("SELECT relation FROM schema_relations WHERE src_schema_id = ?", (new_id,))
        .fetchall()
    )
    assert any(r["relation"] == "relates_to" for r in relations)
