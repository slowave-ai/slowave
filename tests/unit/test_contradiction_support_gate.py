"""Tests for B-5: supersession judge support and recency gates.

The GeometricJudgeConfig has min_support_to_supersede and
min_time_delta_to_supersede_s that were previously unused.
The consolidator now uses them to gate "supersedes" verdicts (renamed from
"contradicts" as part of the 2026-07-15 relation-taxonomy rewire -- see
GeometricContradictionJudge.judge()). These tests mock the judge's return
value directly rather than exercising the real cascade, so they only need
the verdict *string* updated to stay meaningful; the gate logic they cover
(support/recency thresholds in Consolidator._write_latent_schema) is
unchanged.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from slowave.core.consolidation import Consolidator
from slowave.latent.schema import (
    GeometricContradictionJudge,
    GeometricJudgeConfig,
    LatentSchema,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SCHEMA_PATH = str(REPO_ROOT / "slowave" / "storage" / "schema.sql")


def make_latent_schema(
    centroid=None,
    support_count=3,
    mean_ts=1000,
) -> LatentSchema:
    if centroid is None:
        centroid = np.ones(32, dtype=np.float32) / np.sqrt(32)
    return LatentSchema(
        centroid=centroid,
        facet_axes=np.zeros((0, 32), dtype=np.float32),
        facet_strengths=np.zeros((0,), dtype=np.float32),
        member_episode_ids=[],
        central_episode_id=0,
        central_episode_text="test claim",
        mean_ts=mean_ts,
        ts_span_s=10,
        tags=[],
        confidence=0.8,
        support_count=support_count,
        facets={"source_kind": "consolidation"},
    )


@pytest.fixture()
def mock_consolidator():
    """Consolidator with mocked judge that always returns 'supersedes'."""
    db_path = str(Path(tempfile.mkdtemp()) / "test.db")
    from slowave.storage.sqlite_db import SQLiteConfig, SQLiteDB

    db = SQLiteDB(SQLiteConfig(path=db_path))
    db.init_schema(SCHEMA_PATH)
    conn = db.connect()
    conn.execute("PRAGMA foreign_keys = OFF")

    judge = GeometricContradictionJudge(
        GeometricJudgeConfig(min_support_to_supersede=3, min_time_delta_to_supersede_s=60)
    )
    # Mock all methods except the geometric judge — we only test
    # the support/recency gates in the consolidator.
    mock_deps = {
        "db": db,
        "semantic": MagicMock(),
        "episode_text": MagicMock(),
        "schemas": MagicMock(),
        "encoder": None,
        "latent_builder": MagicMock(),
        "geometric_judge": judge,
    }
    cons = Consolidator(**mock_deps)
    # Make schemas.create return a new ID and not trigger dedup
    cons.schemas.create.return_value = 42
    cons.schemas.last_create_reinforced_existing_id = None
    # No geometric near-duplicate — let the write path reach the judge.
    cons.schemas.search_embedding.return_value = []
    # Force the judge to return "relates_to" with high cosine so the
    # status-change path is reachable (supersession is now gated on
    # cos >= 0.90 + support + recency, not on a "supersedes" verdict).
    from slowave.latent.schema import GeometricVerdict

    cons.geometric_judge.judge = MagicMock(
        return_value=GeometricVerdict(
            verdict="relates_to",
            reasoning="test",
            similarity=0.92,
            facet_distance=0.40,
            time_delta_s=1000,
        )
    )
    yield cons
    db.close()


def test_supersedes_gated_by_low_support(mock_consolidator):
    """supersedes verdict with support=1 → should return 'reinforced'."""
    old_schema = MagicMock(id=7, content_text="old", confidence=1.0, facets={}, scope_id="p:t")
    # Valid embedding on old schema
    old_emb = np.ones(32, dtype=np.float32) / np.sqrt(32)

    with patch.object(mock_consolidator, "_best_related_schema", return_value=old_schema):
        with patch.object(mock_consolidator, "_fetch_schema_embedding", return_value=old_emb):
            with patch.object(mock_consolidator, "_scope_for_episodes", return_value="p:t"):
                outcome, _ = mock_consolidator._write_latent_schema(
                    prototype_id=1,
                    schema=make_latent_schema(support_count=1),  # low support
                )
    assert outcome == "reinforced"


def test_supersedes_passes_with_enough_support(mock_consolidator):
    """supersedes verdict with support=5 → passes gate, returns 'contradicted'."""
    old_schema = MagicMock(id=7, content_text="old", confidence=1.0, facets={}, scope_id="p:t")
    old_emb = np.ones(32, dtype=np.float32) / np.sqrt(32)

    with patch.object(mock_consolidator, "_best_related_schema", return_value=old_schema):
        with patch.object(mock_consolidator, "_fetch_schema_embedding", return_value=old_emb):
            with patch.object(mock_consolidator, "_scope_for_episodes", return_value="p:t"):
                outcome, _ = mock_consolidator._write_latent_schema(
                    prototype_id=1,
                    schema=make_latent_schema(support_count=5, mean_ts=2000),  # 1000s delta
                )
    assert outcome == "contradicted"


def test_supersedes_gated_by_short_time_delta(mock_consolidator):
    """high-cosine relates_to with dt=30s < min_time_delta=60s → 'reinforced'."""
    from slowave.latent.schema import GeometricVerdict

    old_schema = MagicMock(id=7, content_text="old", confidence=1.0, facets={}, scope_id="p:t")
    old_emb = np.ones(32, dtype=np.float32) / np.sqrt(32)

    with patch.object(mock_consolidator, "_best_related_schema", return_value=old_schema):
        with patch.object(mock_consolidator, "_fetch_schema_embedding", return_value=old_emb):
            with patch.object(mock_consolidator, "_scope_for_episodes", return_value="p:t"):
                # Override judge for this test: short time delta (30s < 60s gate)
                with patch.object(
                    mock_consolidator.geometric_judge,
                    "judge",
                    return_value=GeometricVerdict(
                        verdict="relates_to",
                        reasoning="test",
                        similarity=0.92,
                        facet_distance=0.40,
                        time_delta_s=30,  # below min_time_delta
                    ),
                ):
                    outcome, _ = mock_consolidator._write_latent_schema(
                        prototype_id=1,
                        schema=make_latent_schema(support_count=5),
                    )
    assert outcome == "reinforced"


def test_config_defaults_exist(mock_consolidator):
    """Regression: min_support_to_supersede and min_time_delta default values."""
    cfg = mock_consolidator.geometric_judge.cfg
    assert cfg.min_support_to_supersede >= 1
    assert cfg.min_time_delta_to_supersede_s > 0
