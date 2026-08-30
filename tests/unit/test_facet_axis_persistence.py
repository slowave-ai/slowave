"""Regression tests for persisted facet axes and topical relation diagnostics.

Facet geometry is diagnostic only; contradiction and supersession come from
explicit client feedback.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from slowave.core.consolidation import Consolidator
from slowave.latent.schema import GeometricRelationConfig, GeometricRelationJudge, LatentSchema
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

    judge = GeometricRelationJudge(GeometricRelationConfig())
    cons = Consolidator(
        db=db,
        semantic=MagicMock(),
        episode_text=MagicMock(),
        schemas=schemas,
        encoder=None,
        latent_builder=MagicMock(),
        relation_judge=judge,
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
        mean_ts=100_000,
        ts_span_s=100,
        tags=[],
        confidence=0.9,
        support_count=5,
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
