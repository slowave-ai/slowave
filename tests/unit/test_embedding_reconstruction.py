import numpy as np
import pytest

from slowave.storage.embedding_reconstruction import (
    reconstruct_centroid,
    reconstruct_episode,
    reconstruct_facets,
)


def test_reconstruct_episode_matches_ingest_mean_and_rejects_missing():
    vectors = {1: np.array([1, 0], dtype=np.float32), 2: np.array([0, 1], dtype=np.float32)}
    np.testing.assert_allclose(reconstruct_episode([1, 2], vectors), [2**-0.5, 2**-0.5])
    with pytest.raises(ValueError, match="missing source vector"):
        reconstruct_episode([1, 3], vectors)


def test_reconstruct_prototype_and_facets_from_members():
    vectors = {1: np.array([1, 0], dtype=np.float32), 2: np.array([0, 1], dtype=np.float32)}
    centroid, variance = reconstruct_centroid([1, 2], vectors)
    np.testing.assert_allclose(centroid, [0.5, 0.5])
    assert variance == pytest.approx(0.25)
    axes, strengths = reconstruct_facets([1, 2], vectors, 1)
    assert axes.shape == (1, 2)
    np.testing.assert_allclose(abs(axes[0]), [2**-0.5, 2**-0.5])
    np.testing.assert_allclose(strengths, [1.0])
    with pytest.raises(ValueError, match="insufficient source episodes"):
        reconstruct_facets([1], vectors, 2)
