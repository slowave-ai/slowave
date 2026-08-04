"""Linkage algorithms for procedural session clustering.

The single-linkage / greedy connected-components rule used in
analyze_procedural_signal.py and validate_procedural_clustering_backtest.py
merges clusters if ANY pair of members exceeds a threshold. On both the
prehubprune backtest and the synthetic ground-truth benchmark (2026-07-27,
private/docs/iterations/20260727_procedural_memory_phase2_plan.md), this
chains: a single bridging pair drags two otherwise-unrelated clusters
together, and no global threshold gives both good precision and good
recall simultaneously.

This module provides two alternative linkage rules that resist chaining
by requiring more than one weak bridge:

- `average_linkage`: agglomerative clustering that merges the pair of
  clusters with the highest MEAN pairwise similarity, stopping once the
  best remaining merge falls below threshold. A single strong outlier pair
  can't drag two clusters together if the rest of their members are
  dissimilar -- the average dilutes it.
- `mutual_knn`: connects i and j only if each is among the other's top-k
  most similar sessions (a *mutual* nearest-neighbor agreement, not a
  single symmetric threshold), then takes connected components. A session
  that's only weakly, one-sidedly similar to a cluster won't be pulled in.

Both take a plain `sims: dict[tuple[int, int], float]` keyed by
sorted-index pairs (i < j) over `range(n)`, matching what the existing
scripts already compute.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any


def _get_sim(sims: dict[tuple[int, int], float], i: int, j: int) -> float:
    if i == j:
        return 1.0
    return sims[(i, j)] if i < j else sims[(j, i)]


def _connected_components(neighbors: dict[int, set[int]], n: int) -> dict[int, list[int]]:
    visited: set[int] = set()
    clusters: dict[int, list[int]] = {}
    cid = 0
    for i in range(n):
        if i in visited:
            continue
        queue = [i]
        component: list[int] = []
        while queue:
            node = queue.pop(0)
            if node in visited:
                continue
            visited.add(node)
            component.append(node)
            queue.extend(nb for nb in neighbors.get(node, ()) if nb not in visited)
        clusters[cid] = component
        cid += 1
    return clusters


def average_linkage(
    sims: dict[tuple[int, int], float], n: int, threshold: float
) -> dict[int, list[int]]:
    """Bottom-up agglomerative clustering, merging the highest-mean-similarity
    pair of clusters each step, stopping when the best remaining merge is
    below `threshold`. O(n^3) worst case -- fine for the small (tens to a
    few hundred sessions) datasets this is meant for."""
    clusters: dict[int, set[int]] = {i: {i} for i in range(n)}

    def mean_sim(a: set[int], b: set[int]) -> float:
        total = 0.0
        count = 0
        for i in a:
            for j in b:
                total += _get_sim(sims, i, j)
                count += 1
        return total / count if count else 0.0

    while len(clusters) > 1:
        best_pair: tuple[int, int] | None = None
        best_sim = -1.0
        ids = list(clusters.keys())
        for a_idx in range(len(ids)):
            for b_idx in range(a_idx + 1, len(ids)):
                a, b = ids[a_idx], ids[b_idx]
                s = mean_sim(clusters[a], clusters[b])
                if s > best_sim:
                    best_sim = s
                    best_pair = (a, b)
        if best_pair is None or best_sim < threshold:
            break
        a, b = best_pair
        clusters[a] = clusters[a] | clusters[b]
        del clusters[b]

    return {cid: sorted(members) for cid, members in enumerate(clusters.values())}


def mutual_knn(
    sims: dict[tuple[int, int], float], n: int, k: int, min_sim: float = 0.0
) -> dict[int, list[int]]:
    """Connect i-j only if each is in the other's top-k nearest neighbors
    (by similarity), optionally also requiring sim >= min_sim. Then take
    connected components. Resists chaining: a one-sided "I'm close to you
    but you have closer friends" relationship never becomes an edge."""
    top_k: dict[int, set[int]] = {}
    for i in range(n):
        ranked = sorted(
            (j for j in range(n) if j != i),
            key=lambda j: _get_sim(sims, i, j),
            reverse=True,
        )
        top_k[i] = set(ranked[:k])

    neighbors: dict[int, set[int]] = defaultdict(set)
    for i in range(n):
        for j in top_k[i]:
            if i in top_k.get(j, set()) and _get_sim(sims, i, j) >= min_sim:
                neighbors[i].add(j)
                neighbors[j].add(i)

    return _connected_components(neighbors, n)


def single_linkage(
    sims: dict[tuple[int, int], float], n: int, threshold: float
) -> dict[int, list[int]]:
    """The existing baseline rule (greedy connected-components on a single
    symmetric threshold), reimplemented here in index-space so all three
    algorithms share one interface for direct comparison."""
    neighbors: dict[int, set[int]] = defaultdict(set)
    for i in range(n):
        for j in range(i + 1, n):
            if _get_sim(sims, i, j) >= threshold:
                neighbors[i].add(j)
                neighbors[j].add(i)
    return _connected_components(neighbors, n)
