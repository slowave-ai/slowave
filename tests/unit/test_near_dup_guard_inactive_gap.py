"""Tests for the near-duplicate guard's active-status gap.

Corrects an initial wrong claim (see private/docs/consolidation/PROGRESS.md,
2026-07-09 CORRECTION) that the topical relation judge's association output
verdicts were provably unreachable because near_dup_guard_cosine (0.92) is
below reinforce_cosine (0.95). Real production data (~/.slowave/backups)
showed association edges, while lifecycle decisions belonged to client feedback,
the judge path (_write_latent_schema), not from a different code path.

The near-duplicate guard's search_embedding(limit=1) returns the single
globally-closest schema *regardless of status*, and only short-circuits
into reinforce_schema() if that closest match is status=="active". When
the closest-by-cosine schema is already inactive (superseded/contradicted
from a prior pass — only possible on a DB with accumulated history), the
guard does not fire, and a different, still-active schema found via
_best_related_schema can genuinely reach the judge.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from slowave.core.consolidation import Consolidator
from slowave.latent.schema import (
    GeometricRelationConfig,
    GeometricRelationJudge,
    GeometricVerdict,
    LatentSchema,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SCHEMA_PATH = str(REPO_ROOT / "slowave" / "storage" / "schema.sql")


def make_latent_schema(support_count=3) -> LatentSchema:
    centroid = np.ones(32, dtype=np.float32) / np.sqrt(32)
    return LatentSchema(
        centroid=centroid,
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
    # Fix #1: no pre-existing schema for this prototype -> the write path
    # reaches the geometric near-dup guard / judge (cross-prototype scenario).
    cons.schemas.find_by_primary_prototype.return_value = None
    yield cons
    db.close()


def test_inactive_near_dup_does_not_block_judge(consolidator):
    """Closest match (cosine 0.97 >= 0.92) is superseded -> guard does not
    intercept -> a different active schema reaches the judge and gets a
    real 'relates_to' verdict."""
    inactive_near_dup = MagicMock(id=99, status="superseded")
    active_related = MagicMock(
        id=7, content_text="rust is the primary language", confidence=1.0, facets={}, scope_id=None
    )
    old_emb = np.ones(32, dtype=np.float32) / np.sqrt(32)

    consolidator.schemas.search_embedding.return_value = [(99, 0.97)]
    consolidator.schemas.get.return_value = inactive_near_dup

    with patch.object(consolidator, "_best_related_schema", return_value=active_related):
        with patch.object(consolidator, "_fetch_schema_embedding", return_value=old_emb):
            with patch.object(consolidator, "_scope_for_episodes", return_value=None):
                with patch.object(
                    consolidator.relation_judge,
                    "judge",
                    return_value=GeometricVerdict(
                        verdict="relates_to",
                        reasoning="test",
                        similarity=0.97,
                    ),
                ) as mock_judge:
                    outcome, new_id = consolidator._write_latent_schema(
                        prototype_id=1, schema=make_latent_schema()
                    )

    mock_judge.assert_called_once()
    consolidator.schemas.reinforce_schema.assert_not_called()
    consolidator.schemas.add_relation.assert_called_once()
    _, kwargs = consolidator.schemas.add_relation.call_args
    assert kwargs["relation"] == "relates_to"
    assert outcome == "reinforced"
    assert new_id == 42


def test_active_near_dup_does_block_judge(consolidator):
    """Contrast case: closest match is active -> guard intercepts, judge
    is never invoked."""
    active_near_dup = MagicMock(id=99, status="active", scope_id=None)
    consolidator.schemas.search_embedding.return_value = [(99, 0.97)]
    consolidator.schemas.get.return_value = active_near_dup

    with patch.object(consolidator.relation_judge, "judge") as mock_judge:
        outcome, new_id = consolidator._write_latent_schema(
            prototype_id=1, schema=make_latent_schema()
        )

    mock_judge.assert_not_called()
    consolidator.schemas.reinforce_schema.assert_called_once()
    assert outcome == "reinforced"
    assert new_id == 99
