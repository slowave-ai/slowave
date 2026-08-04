#!/usr/bin/env python3
"""Compare linkage rules on the synthetic ground-truth benchmark.

The synthetic benchmark (validate_procedural_clustering_synthetic.py) showed
single-linkage/greedy-connected-components can't get both good precision and
good recall at any threshold: raising the threshold to stop chaining also
throws away the hardest (tier C/D) true pairs. This script re-runs the exact
same ground truth and evaluation through three linkage rules
(procedural_clustering_algorithms.py): the existing single-linkage baseline,
average-linkage agglomerative clustering, and mutual-top-k nearest-neighbor
agreement -- to see whether either alternative resolves the tradeoff rather
than just shifting it.

Usage:
    python scripts/validate_procedural_clustering_algorithms.py
    python scripts/validate_procedural_clustering_algorithms.py --json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import procedural_clustering_algorithms as algos  # noqa: E402
import validate_procedural_clustering_backtest as vb  # noqa: E402
import validate_procedural_clustering_synthetic as syn  # noqa: E402


def _build_index_sims(sessions: list[dict[str, Any]], step_cache: dict, goal_cache: dict) -> dict:
    n = len(sessions)
    sims = {}
    for i in range(n):
        for j in range(i + 1, n):
            step_sim = vb._aligned_step_similarity(
                sessions[i]["step_contents"], sessions[j]["step_contents"], step_cache
            )
            gi, gj = sessions[i]["goal"], sessions[j]["goal"]
            goal_sim = vb._cosine(goal_cache[gi], goal_cache[gj]) if gi and gj else 0.0
            sims[(i, j)] = 0.7 * step_sim + 0.3 * goal_sim
    return sims


def _index_clusters_to_session_clusters(
    index_clusters: dict[int, list[int]], sessions: list[dict[str, Any]]
) -> dict[int, list[dict[str, Any]]]:
    return {cid: [sessions[i] for i in idxs] for cid, idxs in index_clusters.items()}


def _evaluate(
    label: str,
    param_name: str,
    param_value: Any,
    session_clusters: dict[int, list[dict[str, Any]]],
    sessions: list[dict[str, Any]],
    true_pairs: set,
) -> dict[str, Any]:
    pred_pairs = syn._predicted_pairs(session_clusters)
    metrics = syn._prf(true_pairs, pred_pairs)
    contamination = syn._decoy_contamination(sessions, session_clusters)
    n_clusters = sum(1 for m in session_clusters.values() if len(m) >= 2)
    return {
        "algorithm": label,
        param_name: param_value,
        "n_clusters": n_clusters,
        **metrics,
        "contamination": contamination,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    sessions = syn._to_engine_sessions(syn.SESSIONS)
    true_pairs = syn._true_pairs(sessions)
    n = len(sessions)

    from slowave.symbolic.encoder import TextEncoder

    encoder = TextEncoder()
    step_cache, goal_cache = vb._build_embedding_cache(sessions, encoder)
    sims = _build_index_sims(sessions, step_cache, goal_cache)

    results: list[dict[str, Any]] = []

    for thr in [round(x * 0.05, 2) for x in range(4, 16)]:  # 0.20 .. 0.75
        idx_clusters = algos.single_linkage(sims, n, thr)
        sess_clusters = _index_clusters_to_session_clusters(idx_clusters, sessions)
        results.append(_evaluate("single_linkage", "param", thr, sess_clusters, sessions, true_pairs))

    for thr in [round(x * 0.05, 2) for x in range(2, 14)]:  # 0.10 .. 0.65
        idx_clusters = algos.average_linkage(sims, n, thr)
        sess_clusters = _index_clusters_to_session_clusters(idx_clusters, sessions)
        results.append(_evaluate("average_linkage", "param", thr, sess_clusters, sessions, true_pairs))

    for k in range(1, 9):
        idx_clusters = algos.mutual_knn(sims, n, k)
        sess_clusters = _index_clusters_to_session_clusters(idx_clusters, sessions)
        results.append(_evaluate("mutual_knn", "param", k, sess_clusters, sessions, true_pairs))

    if args.json:
        print(json.dumps({"n_true_pairs": len(true_pairs), "results": results}, indent=2, default=str))
        return

    print(f"Total true (same-concept) pairs: {len(true_pairs)}\n")
    by_algo: dict[str, list[dict[str, Any]]] = {}
    for r in results:
        by_algo.setdefault(r["algorithm"], []).append(r)

    for algo_name, rows in by_algo.items():
        print("=" * 78)
        print(algo_name)
        print("=" * 78)
        print(f"{'param':>7} {'clusters':>9} {'P':>6} {'R':>6} {'F1':>6} {'tp':>4} {'fp':>4} {'fn':>4}  contamination")
        for r in rows:
            contam = ",".join(f"{a}->{b}" for a, b in r["contamination"]) or "-"
            print(f"{r['param']:>7} {r['n_clusters']:>9} {r['precision']:>6.2f} {r['recall']:>6.2f} "
                  f"{r['f1']:>6.2f} {r['tp']:>4} {r['fp']:>4} {r['fn']:>4}  {contam}")
        best = max(rows, key=lambda r: r["f1"])
        print(f"\nBest {algo_name} by F1: param={best['param']} "
              f"P={best['precision']:.2f} R={best['recall']:.2f} F1={best['f1']:.2f} "
              f"contamination={'yes' if best['contamination'] else 'no'}\n")

    print("=" * 78)
    print("Configs with P>=0.9 AND R>=0.9 (the actual target: both at once)")
    print("=" * 78)
    strong = [r for r in results if r["precision"] >= 0.9 and r["recall"] >= 0.9]
    if not strong:
        print("  none found")
    else:
        for r in strong:
            print(f"  {r['algorithm']} param={r['param']} P={r['precision']:.2f} R={r['recall']:.2f} F1={r['f1']:.2f}")


if __name__ == "__main__":
    main()
