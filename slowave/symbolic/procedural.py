"""Procedural session clustering: groups sessions by similar step content.

Validated method (2026-07-27, see
private/docs/iterations/20260727_procedural_memory_phase2_plan.md secs 7-10):
sentence embeddings (not TF-IDF) for step/goal similarity, greedy sequence
alignment (not positional pairing) for step comparison, and average-linkage
(not single-linkage/greedy-connected-components) for clustering.

Single-linkage was the original approach and it chains: one strong bridging
pair fuses two otherwise-unrelated session groups. Average-linkage requires
the *mean* similarity across every cross-pair between two clusters to clear
threshold, which resists that -- validated against a synthetic ground-truth
benchmark (average-linkage F1=0.82 vs single-linkage's 0.70, with zero
decoy/cross-concept contamination) and re-confirmed on real historical data
(scripts/validate_procedural_clustering_backtest.py).

This is the single implementation of "what counts as a procedure cluster"
in the codebase. Two consumers: scripts/analyze_procedural_signal.py (the
offline gate-check CLI, run periodically against a live DB) and
slowave/dashboard/app.py (the live Procedures tab). Keep it that way --
don't let a third clustering definition creep in.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Embedding similarity
# ---------------------------------------------------------------------------


def build_embedding_caches(
    sessions: list[dict[str, Any]], encoder: Any
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Batch-embed every unique step text and goal text once."""
    all_steps = sorted({s for sess in sessions for s in sess["step_contents"]})
    all_goals = sorted({sess["goal"] for sess in sessions if sess["goal"]})
    step_vecs = encoder.encode_many(all_steps) if all_steps else np.zeros((0, encoder.dim))
    goal_vecs = encoder.encode_many(all_goals) if all_goals else np.zeros((0, encoder.dim))
    return dict(zip(all_steps, step_vecs)), dict(zip(all_goals, goal_vecs))


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a)) or 1e-12
    nb = float(np.linalg.norm(b)) or 1e-12
    return float(np.dot(a, b) / (na * nb))


def aligned_step_similarity(
    steps_a: list[str], steps_b: list[str], step_cache: dict[str, np.ndarray]
) -> float:
    """Greedy best-match alignment: tolerates reordering, insertions, and
    deletions, unlike positional pairing. Score = sum of matched-pair
    similarities / max(len_a, len_b)."""
    if not steps_a or not steps_b:
        return 0.0
    vecs_a = [step_cache[s] for s in steps_a]
    vecs_b = [step_cache[s] for s in steps_b]
    pairs = sorted(
        (
            (cosine(vecs_a[i], vecs_b[j]), i, j)
            for i in range(len(vecs_a))
            for j in range(len(vecs_b))
        ),
        reverse=True,
    )
    used_a: set[int] = set()
    used_b: set[int] = set()
    matched_total = 0.0
    for sim, i, j in pairs:
        if i in used_a or j in used_b:
            continue
        used_a.add(i)
        used_b.add(j)
        matched_total += sim
    return matched_total / max(len(steps_a), len(steps_b))


def goal_coherence(goals: list[str], goal_cache: dict[str, np.ndarray]) -> float:
    uniq = [g for g in dict.fromkeys(goals) if g]
    if len(uniq) < 2:
        return 1.0
    sims = [
        cosine(goal_cache[uniq[i]], goal_cache[uniq[j]])
        for i in range(len(uniq))
        for j in range(i + 1, len(uniq))
    ]
    return float(sum(sims) / len(sims)) if sims else 0.0


# ---------------------------------------------------------------------------
# Linkage algorithms
# ---------------------------------------------------------------------------


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


def single_linkage(
    sims: dict[tuple[int, int], float], n: int, threshold: float
) -> dict[int, list[int]]:
    """Greedy connected-components on one global threshold. Kept only for
    comparison (--legacy) -- chains, see module docstring."""
    neighbors: dict[int, set[int]] = defaultdict(set)
    for i in range(n):
        for j in range(i + 1, n):
            if _get_sim(sims, i, j) >= threshold:
                neighbors[i].add(j)
                neighbors[j].add(i)
    return _connected_components(neighbors, n)


def average_linkage(
    sims: dict[tuple[int, int], float], n: int, threshold: float
) -> dict[int, list[int]]:
    """Bottom-up agglomerative clustering: repeatedly merge the pair of
    clusters with the highest mean cross-pair similarity, stopping once the
    best remaining merge is below threshold. Chosen method -- see module
    docstring."""
    clusters: dict[int, set[int]] = {i: {i} for i in range(n)}

    def mean_sim(a: set[int], b: set[int]) -> float:
        total = sum(_get_sim(sims, i, j) for i in a for j in b)
        count = len(a) * len(b)
        return total / count if count else 0.0

    while len(clusters) > 1:
        best_pair: tuple[int, int] | None = None
        best_sim = -1.0
        ids = list(clusters.keys())
        for ai in range(len(ids)):
            for bi in range(ai + 1, len(ids)):
                a, b = ids[ai], ids[bi]
                s = mean_sim(clusters[a], clusters[b])
                if s > best_sim:
                    best_sim, best_pair = s, (a, b)
        if best_pair is None or best_sim < threshold:
            break
        a, b = best_pair
        clusters[a] = clusters[a] | clusters[b]
        del clusters[b]

    return {cid: sorted(members) for cid, members in enumerate(clusters.values())}


# ---------------------------------------------------------------------------
# Session clustering pipeline
# ---------------------------------------------------------------------------


def cluster_sessions(
    sessions: list[dict[str, Any]],
    step_cache: dict[str, np.ndarray],
    goal_cache: dict[str, np.ndarray],
    threshold: float = 0.4,
) -> dict[int, list[dict[str, Any]]]:
    """sessions: dicts with at least {id, goal, outcome, step_contents,
    has_steps}. Returns {cluster_id: [sessions]}."""
    with_steps = [s for s in sessions if s.get("has_steps")]
    if len(with_steps) < 2:
        return {}
    n = len(with_steps)
    sims: dict[tuple[int, int], float] = {}
    for i in range(n):
        for j in range(i + 1, n):
            step_sim = aligned_step_similarity(
                with_steps[i]["step_contents"], with_steps[j]["step_contents"], step_cache
            )
            gi, gj = with_steps[i]["goal"], with_steps[j]["goal"]
            gsim = cosine(goal_cache[gi], goal_cache[gj]) if gi and gj else 0.0
            sims[(i, j)] = 0.7 * step_sim + 0.3 * gsim
    idx_clusters = average_linkage(sims, n, threshold)
    return {cid: [with_steps[i] for i in idxs] for cid, idxs in idx_clusters.items()}


# ---------------------------------------------------------------------------
# Cluster scoring (anti-pattern / competing-procedure detection)
# ---------------------------------------------------------------------------


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


def score_cluster(
    sessions: list[dict[str, Any]], goal_cache: dict[str, np.ndarray] | None = None
) -> dict[str, Any]:
    n = len(sessions)
    successes = sum(1 for s in sessions if s["outcome"] == "success")
    failures = n - successes
    success_rate = successes / n if n else 0.0
    goals = [s["goal"] for s in sessions]
    coherence = goal_coherence(goals, goal_cache) if goal_cache else 0.0
    # Anti-pattern: repeatedly failing procedure (>=3 observations, <20% success)
    anti_pattern = n >= 3 and success_rate < 0.2
    return {
        "session_count": n,
        "successes": successes,
        "failures": failures,
        "success_rate": round(success_rate, 3),
        "goal_coherence": round(coherence, 3),
        "composite": round(n * success_rate * coherence, 3),
        "anti_pattern": anti_pattern,
        "example_goals": list(dict.fromkeys(goals))[:5],
        "example_session_ids": [s["id"] for s in sessions[:5]],
        "example_steps": sessions[0].get("step_contents", [])[:5] if sessions else [],
    }


def rank_clusters(
    clusters: dict[int, list[dict[str, Any]]],
    min_sessions: int,
    max_clusters: int,
    goal_cache: dict[str, np.ndarray] | None = None,
) -> list[dict[str, Any]]:
    scored = []
    for cid, sessions in clusters.items():
        if len(sessions) < min_sessions:
            continue
        info = score_cluster(sessions, goal_cache=goal_cache)
        info["cluster_id"] = cid
        scored.append(info)
    scored.sort(key=lambda x: x["composite"], reverse=True)

    # Competing procedures: clusters with overlapping goals but divergent outcomes.
    for i in range(len(scored)):
        for j in range(i + 1, len(scored)):
            gi = set(_tokenize(" ".join(scored[i]["example_goals"])))
            gj = set(_tokenize(" ".join(scored[j]["example_goals"])))
            goal_overlap = len(gi & gj) / min(len(gi), len(gj)) if gi and gj else 0.0
            sr_i, sr_j = scored[i]["success_rate"], scored[j]["success_rate"]
            diverged = (sr_i >= 0.7 and sr_j <= 0.2) or (sr_j >= 0.7 and sr_i <= 0.2)
            if goal_overlap >= 0.4 and diverged:
                scored[i].setdefault("competes_with", []).append(scored[j]["cluster_id"])
                scored[j].setdefault("competes_with", []).append(scored[i]["cluster_id"])

    return scored[:max_clusters]
