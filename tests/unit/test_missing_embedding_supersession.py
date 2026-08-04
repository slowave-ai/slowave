"""Tests for missing-embedding guard in the consolidation supersession path.

Covers: _write_latent_schema must not supersede when
_fetch_schema_embedding returns None (was zero-vector fallback).

(remember() used to have an equivalent engine-path guard, but remember() no
longer runs any geometric classification itself — see
test_geometry_supersession.py — so that guard and its test were removed
along with the code they protected.)
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from slowave.core.consolidation import Consolidator
from slowave.latent.schema import LatentSchema

# ---- Consolidation-path test fixtures ----


@pytest.fixture()
def mock_deps():
    """Return mocked dependencies for Consolidator."""
    deps = {
        "db": MagicMock(),
        "semantic": MagicMock(),
        "episode_text": MagicMock(),
        "schemas": MagicMock(),
        "encoder": None,
        "latent_builder": MagicMock(),
        "geometric_judge": MagicMock(),
    }
    # No geometric near-duplicate — let the write path reach the judge.
    deps["schemas"].search_embedding.return_value = []
    return deps


def make_latent_schema(centroid=None, claim="test claim") -> LatentSchema:
    if centroid is None:
        centroid = np.ones(32, dtype=np.float32) / np.sqrt(32)
    return LatentSchema(
        centroid=centroid,
        facet_axes=np.zeros((0, 32), dtype=np.float32),
        facet_strengths=np.zeros((0,), dtype=np.float32),
        member_episode_ids=[],
        central_episode_id=0,
        central_episode_text=claim,
        mean_ts=1000,
        ts_span_s=10,
        tags=[],
        confidence=0.8,
        support_count=3,
        facets={"source_kind": "consolidation"},
    )


# ---------------------------------------------------------------------------
# Consolidation path: _write_latent_schema
# ---------------------------------------------------------------------------


def test_write_latent_schema_falls_back_to_created_on_none_embedding(mock_deps):
    """When _fetch_schema_embedding returns None, return 'created' — not a
    geometric verdict based on a missing (formerly zero) vector."""
    mock_deps["schemas"].create.return_value = 42
    mock_deps["schemas"].last_create_reinforced_existing_id = None

    # Simulate: _best_related_schema found a related schema (id=7),
    # but _fetch_schema_embedding returns None for it.
    related = MagicMock(id=7, content_text="old fact", confidence=1.0, facets={}, scope_id="p:test")

    cons = Consolidator(**mock_deps)

    with patch.object(cons, "_best_related_schema", return_value=related):
        with patch.object(cons, "_fetch_schema_embedding", return_value=None):
            with patch.object(cons, "_scope_for_episodes", return_value="p:test"):
                outcome, new_id = cons._write_latent_schema(
                    prototype_id=1,
                    schema=make_latent_schema(),
                )

    assert outcome == "created"
    assert new_id == 42
    # The geometric judge must NOT have been called — a missing embedding
    # should short-circuit before any verdict is formed.
    mock_deps["geometric_judge"].judge.assert_not_called()


def test_write_latent_schema_judge_called_when_embedding_present(mock_deps):
    """When _fetch_schema_embedding returns a valid embedding, the geometric
    judge is still invoked as normal (regression check)."""
    mock_deps["schemas"].create.return_value = 99
    mock_deps["schemas"].last_create_reinforced_existing_id = None
    mock_deps["geometric_judge"].judge.return_value = MagicMock(
        verdict="unrelated",
        time_delta_s=0,
    )

    related = MagicMock(id=7, content_text="old fact", confidence=1.0, facets={}, scope_id="p:test")
    valid_emb = np.ones(32, dtype=np.float32) / np.sqrt(32)

    cons = Consolidator(**mock_deps)

    with patch.object(cons, "_best_related_schema", return_value=related):
        with patch.object(cons, "_fetch_schema_embedding", return_value=valid_emb):
            with patch.object(cons, "_scope_for_episodes", return_value="p:test"):
                outcome, _ = cons._write_latent_schema(
                    prototype_id=1,
                    schema=make_latent_schema(),
                )

    # The geometric judge MUST have been called with valid embeddings.
    mock_deps["geometric_judge"].judge.assert_called_once()
