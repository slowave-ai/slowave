"""Tests for dedup fix #1: one-schema-per-primary-prototype.

Schema identity is its primary prototype. Re-consolidating the SAME prototype
must reactivate/update the single engram in place (ANY status) instead of
minting a fresh duplicate, and must never self-supersede. Only a genuinely
distinct, cross-prototype generalization may retire a schema.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from slowave.core.consolidation import Consolidator
from slowave.latent.schema import GeometricRelationConfig, GeometricRelationJudge, LatentSchema

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SCHEMA_PATH = str(REPO_ROOT / "slowave" / "storage" / "schema.sql")


def make_latent_schema(support_count=3) -> LatentSchema:
    return LatentSchema(
        centroid=np.ones(32, dtype=np.float32) / np.sqrt(32),
        facet_axes=np.zeros((0, 32), dtype=np.float32),
        facet_strengths=np.zeros((0,), dtype=np.float32),
        member_episode_ids=[],
        central_episode_id=0,
        central_episode_text="rust is now the primary language",
        mean_ts=2000,
        ts_span_s=10,
        tags=[],
        confidence=0.8,
        support_count=support_count,
        facets={"source_kind": "consolidation"},
    )


@pytest.fixture()
def consolidator():
    db_path = str(Path(tempfile.mkdtemp()) / "test.db")
    from slowave.storage.sqlite_db import SQLiteConfig, SQLiteDB

    db = SQLiteDB(SQLiteConfig(path=db_path))
    db.init_schema(SCHEMA_PATH)
    conn = db.connect()
    conn.execute("PRAGMA foreign_keys = OFF")

    judge = GeometricRelationJudge(GeometricRelationConfig())
    cons = Consolidator(
        db=db,
        semantic=MagicMock(),
        episode_text=MagicMock(),
        schemas=MagicMock(),
        encoder=None,
        latent_builder=MagicMock(),
        relation_judge=judge,
    )
    cons.schemas.create.return_value = 42
    cons.schemas.last_create_reinforced_existing_id = None
    yield cons
    db.close()


def test_same_prototype_retired_schema_stays_retired_without_duplicate(consolidator):
    """Consolidation may attach evidence but cannot undo client retirement."""
    retired = MagicMock(id=7, status="superseded", scope_id="p:t")
    consolidator.schemas.find_by_primary_prototype.return_value = retired

    outcome, new_id = consolidator._write_latent_schema(prototype_id=7, schema=make_latent_schema())

    consolidator.schemas.find_by_primary_prototype.assert_called_once_with(7)
    consolidator.schemas.reinforce_schema.assert_called_once()
    consolidator.schemas.update_status.assert_not_called()
    # No new row minted and no relation.
    consolidator.schemas.create.assert_not_called()
    consolidator.schemas.add_relation.assert_not_called()
    assert outcome == "reinforced"
    assert new_id == 7


def test_same_prototype_active_keeps_active(consolidator):
    """An already-active same-prototype schema is reinforced without any
    status churn (no needless update_status call)."""
    active = MagicMock(id=7, status="active", scope_id="p:t")
    consolidator.schemas.find_by_primary_prototype.return_value = active

    outcome, new_id = consolidator._write_latent_schema(prototype_id=7, schema=make_latent_schema())

    consolidator.schemas.reinforce_schema.assert_called_once()
    consolidator.schemas.update_status.assert_not_called()
    consolidator.schemas.create.assert_not_called()
    assert outcome == "reinforced"
    assert new_id == 7


def test_same_prototype_forgotten_is_skipped_not_resurrected(consolidator):
    """A forgotten same-prototype schema is skipped: not resurrected, and no
    duplicate is created either."""
    forgotten = MagicMock(id=7, status="forgotten", scope_id="p:t")
    consolidator.schemas.find_by_primary_prototype.return_value = forgotten

    outcome, new_id = consolidator._write_latent_schema(prototype_id=7, schema=make_latent_schema())

    consolidator.schemas.reinforce_schema.assert_not_called()
    consolidator.schemas.create.assert_not_called()
    consolidator.schemas.update_status.assert_not_called()
    assert outcome == "skipped"
    assert new_id == 7


def test_cross_prototype_relation_never_retires_existing_schema(consolidator):
    from slowave.latent.schema import GeometricVerdict

    consolidator.schemas.find_by_primary_prototype.return_value = None
    consolidator.schemas.search_embedding.return_value = []  # no near-dup
    related = MagicMock(id=7, content_text="old claim", confidence=1.0, facets={}, scope_id="p:t")
    old_emb = np.ones(32, dtype=np.float32) / np.sqrt(32)

    with patch.object(consolidator, "_best_related_schema", return_value=related):
        with patch.object(consolidator, "_fetch_schema_embedding", return_value=old_emb):
            with patch.object(consolidator, "_scope_for_episodes", return_value="p:t"):
                with patch.object(
                    consolidator.relation_judge,
                    "judge",
                    return_value=GeometricVerdict(
                        verdict="relates_to",
                        reasoning="test",
                        similarity=0.92,
                    ),
                ):
                    outcome, _ = consolidator._write_latent_schema(
                        prototype_id=8, schema=make_latent_schema(support_count=5)
                    )

    consolidator.schemas.create.assert_called_once()
    consolidator.schemas.update_status.assert_not_called()
    assert outcome == "reinforced"
