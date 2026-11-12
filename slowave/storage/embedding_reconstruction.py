"""Pure reconstruction helpers for a future offline embedding migration.

These functions do not read or write a database. They intentionally fail on
missing provenance instead of silently mixing embeddings from different models.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


def _rows(ids: Sequence[int], vectors: Mapping[int, np.ndarray]) -> np.ndarray:
    if not ids:
        raise ValueError("cannot reconstruct from empty provenance")
    try:
        result = np.stack([np.asarray(vectors[key], dtype=np.float32) for key in ids])
    except KeyError as exc:
        raise ValueError(f"missing source vector for id {exc.args[0]}") from exc
    if result.ndim != 2 or result.shape[1] == 0 or not np.isfinite(result).all():
        raise ValueError("invalid source vectors")
    return result


def reconstruct_episode(
    event_ids: Sequence[int], event_vectors: Mapping[int, np.ndarray]
) -> np.ndarray:
    """Match ingest's normalized mean of the episode's original raw events."""
    mean = _rows(event_ids, event_vectors).mean(axis=0).astype(np.float32)
    norm = float(np.linalg.norm(mean))
    return mean / norm if norm > 1e-12 else mean


def reconstruct_centroid(
    episode_ids: Sequence[int], episode_vectors: Mapping[int, np.ndarray]
) -> tuple[np.ndarray, float]:
    """Compute a new centroid from unique members; not the historical replay centroid."""
    members = _rows(episode_ids, episode_vectors)
    centroid = members.mean(axis=0).astype(np.float32)
    variance = float(((members - centroid) ** 2).mean())
    return centroid, variance


def reconstruct_facets(
    episode_ids: Sequence[int], episode_vectors: Mapping[int, np.ndarray], n_axes: int
) -> tuple[np.ndarray, np.ndarray]:
    """Recompute within-schema axes using the schema facet backfill algorithm."""
    if n_axes < 1:
        raise ValueError("facet count must be positive")
    members = _rows(episode_ids, episode_vectors)
    if len(members) < 2 or n_axes > min(members.shape):
        raise ValueError("insufficient source episodes for requested facet count")
    _, strengths, axes = np.linalg.svd(members - members.mean(axis=0), full_matrices=False)
    return axes[:n_axes].astype(np.float32), strengths[:n_axes].astype(np.float32)
