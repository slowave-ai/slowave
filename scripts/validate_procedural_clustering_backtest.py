#!/usr/bin/env python3
"""Backtest: does embedding+alignment clustering find signal that Phase 1's
TF-IDF+positional matching missed, using historical remember:* content as an
action-text proxy?

Phase 1 (private/docs/iterations/20260725_procedural_signal_commit_steps.md)
proved event-TYPE sequences carry no procedural signal, using
analyze_procedural_signal.py's TF-IDF + positional-cosine clustering on real
backups. The new `commit(steps=...)` feature has only a handful of real
sessions so far -- too few to validate the upgraded method on real step data.

This script asks a narrower, immediately-answerable question instead: on the
SAME real historical sessions, does swapping TF-IDF for sentence embeddings
and positional pairing for greedy alignment change the verdict? It uses
ordered `remember:*` event content (facts/decisions/lessons/constraints
logged during real work) as a stand-in for step content -- the only rich,
ordered, real text available in bulk today.

This is an imperfect proxy (remember-content is "what was learned", not
strictly "what was done"), so a positive result here is encouraging but not a
substitute for validating on real step data. A negative result is a
trustworthy early signal that the method upgrade alone isn't the missing
piece, without waiting weeks on dogfooding.

Usage:
    python scripts/validate_procedural_clustering_backtest.py \
        --db /tmp/slowave_prehubprune_analysis.db \
        --min-sessions 3 --threshold 0.35 --json
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_procedural_signal as baseline  # noqa: E402  (TF-IDF/positional reference impl)


EXCLUDED_TYPES = {"context_query", "task_complete", "step"}


def _connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


# ---------------------------------------------------------------------------
# Session loading (remember:* content as a step-content proxy)
# ---------------------------------------------------------------------------


def load_sessions_remember_proxy(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    """Load sessions using ordered remember:* event content as a step-content
    proxy. Mirrors analyze_procedural_signal.load_sessions()'s output shape so
    the baseline clustering/scoring functions work unmodified."""
    rows = conn.execute(
        "SELECT id, goal, outcome, scope_id FROM sessions "
        "WHERE goal IS NOT NULL AND outcome IS NOT NULL ORDER BY started_ts"
    ).fetchall()
    sessions = []
    for row in rows:
        events = conn.execute(
            "SELECT type, content FROM raw_events WHERE session_id = ? ORDER BY ts",
            (row["id"],),
        ).fetchall()
        proxy_steps = [
            e["content"] for e in events
            if e["type"] not in EXCLUDED_TYPES and e["content"]
        ]
        if len(proxy_steps) < 2:
            continue
        sessions.append({
            "id": row["id"],
            "goal": row["goal"] or "",
            "outcome": row["outcome"] or "unknown",
            "scope_id": row["scope_id"],
            "step_contents": proxy_steps,
            "has_steps": True,
        })
    return sessions


# ---------------------------------------------------------------------------
# Embedding-based similarity (replaces TF-IDF)
# ---------------------------------------------------------------------------


def _build_embedding_cache(
    sessions: list[dict[str, Any]], encoder: Any
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Batch-embed every unique step text and goal text once."""
    all_steps = sorted({s for sess in sessions for s in sess["step_contents"]})
    all_goals = sorted({sess["goal"] for sess in sessions if sess["goal"]})
    step_vecs = encoder.encode_many(all_steps) if all_steps else np.zeros((0, encoder.dim))
    goal_vecs = encoder.encode_many(all_goals) if all_goals else np.zeros((0, encoder.dim))
    return dict(zip(all_steps, step_vecs)), dict(zip(all_goals, goal_vecs))


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a)) or 1e-12
    nb = float(np.linalg.norm(b)) or 1e-12
    return float(np.dot(a, b) / (na * nb))


def _aligned_step_similarity(
    steps_a: list[str], steps_b: list[str], step_cache: dict[str, np.ndarray]
) -> float:
    """Greedy best-match alignment: unlike positional pairing (step[i] vs
    step[i]), this tolerates reordering, insertions, and deletions between
    the two step sequences. Score = sum of matched-pair similarities /
    max(len_a, len_b) -- a soft penalty for length mismatch, in the spirit
    of normalized LCS."""
    if not steps_a or not steps_b:
        return 0.0
    vecs_a = [step_cache[s] for s in steps_a]
    vecs_b = [step_cache[s] for s in steps_b]

    pairs = sorted(
        (
            (_cosine(vecs_a[i], vecs_b[j]), i, j)
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


def compute_embedding_sims(
    sessions: list[dict[str, Any]],
    step_cache: dict[str, np.ndarray],
    goal_cache: dict[str, np.ndarray],
) -> tuple[list[dict[str, Any]], dict[tuple[int, int], float]]:
    """Pairwise embedding+alignment similarity, exposed separately from
    clustering so different linkage rules (single/average/mutual-kNN, see
    procedural_clustering_algorithms.py) can share the same sims without
    recomputing embeddings."""
    with_steps = [s for s in sessions if s["has_steps"]]
    n = len(with_steps)
    sims: dict[tuple[int, int], float] = {}
    for i in range(n):
        for j in range(i + 1, n):
            step_sim = _aligned_step_similarity(
                with_steps[i]["step_contents"], with_steps[j]["step_contents"], step_cache
            )
            gi, gj = with_steps[i]["goal"], with_steps[j]["goal"]
            goal_sim = _cosine(goal_cache[gi], goal_cache[gj]) if gi and gj else 0.0
            sims[(i, j)] = 0.7 * step_sim + 0.3 * goal_sim
    return with_steps, sims


def cluster_by_embedding_alignment(
    sessions: list[dict[str, Any]],
    step_cache: dict[str, np.ndarray],
    goal_cache: dict[str, np.ndarray],
    threshold: float,
    algorithm: str = "single_linkage",
) -> dict[int, list[dict[str, Any]]]:
    """Embedding+alignment similarity clustered via the chosen linkage rule.
    `algorithm`: "single_linkage" (original greedy connected-components,
    kept for comparison) or "average_linkage" (chaining-resistant, chosen in
    private/docs/iterations/20260727_procedural_memory_phase2_plan.md §9)."""
    with_steps, sims = compute_embedding_sims(sessions, step_cache, goal_cache)
    if len(with_steps) < 2:
        return {}
    n = len(with_steps)

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import procedural_clustering_algorithms as algos

    if algorithm == "average_linkage":
        idx_clusters = algos.average_linkage(sims, n, threshold)
    elif algorithm == "single_linkage":
        idx_clusters = algos.single_linkage(sims, n, threshold)
    else:
        raise ValueError(f"unknown algorithm: {algorithm!r}")

    return {cid: [with_steps[i] for i in idxs] for cid, idxs in idx_clusters.items()}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True, help="Path to a Slowave SQLite DB/backup")
    parser.add_argument("--min-sessions", type=int, default=3)
    parser.add_argument("--max-clusters", type=int, default=20)
    parser.add_argument(
        "--threshold", type=float, default=0.5,
        help="Embedding-cosine similarity runs in a narrower, higher band "
             "than TF-IDF cosine (median ~0.19, p90 ~0.34 on real session "
             "data). Below ~0.45 the greedy connected-component clustering "
             "chains almost everything into one blob via transitive "
             "single-linkage; 0.5-0.55 is where cluster sizes stop being "
             "dominated by a single hub. Sweep before trusting a threshold "
             "on new data.",
    )
    parser.add_argument(
        "--algorithm", choices=["single_linkage", "average_linkage"], default="average_linkage",
        help="Linkage rule for the embedding method. average_linkage is chaining-"
             "resistant (see 20260727_procedural_memory_phase2_plan.md sec 9) and is "
             "the recommended default; single_linkage is kept for comparison.",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    conn = _connect(args.db)
    sessions = load_sessions_remember_proxy(conn)
    conn.close()
    print(
        f"Sessions with >=2 remember-content events (proxy steps): {len(sessions)}",
        file=sys.stderr,
    )

    # --- OLD: TF-IDF + positional (Phase 1 reference, reused unmodified) ---
    t0 = time.time()
    old_clusters = baseline.cluster_by_step_content(sessions, threshold=0.5)
    old_ranked = baseline.rank_clusters(old_clusters, args.min_sessions, args.max_clusters)
    old_time = time.time() - t0

    # --- NEW: embeddings + alignment ---
    from slowave.symbolic.encoder import TextEncoder

    encoder = TextEncoder()
    t0 = time.time()
    step_cache, goal_cache = _build_embedding_cache(sessions, encoder)
    new_clusters = cluster_by_embedding_alignment(
        sessions, step_cache, goal_cache, args.threshold, algorithm=args.algorithm
    )
    new_ranked = baseline.rank_clusters(new_clusters, args.min_sessions, args.max_clusters)
    new_time = time.time() - t0

    def _summarize(ranked: list[dict[str, Any]], seconds: float, name: str) -> dict[str, Any]:
        return {
            "name": name,
            "clusters_found": len(ranked),
            "clusters_coherence_ge_0.3": sum(1 for c in ranked if c["goal_coherence"] >= 0.3),
            "compute_seconds": round(seconds, 2),
            "clusters": ranked,
        }

    report = {
        "sessions_loaded": len(sessions),
        "old_method": _summarize(old_ranked, old_time, "tfidf+positional (Phase 1 reference)"),
        "new_method": _summarize(new_ranked, new_time, f"embedding+alignment ({args.algorithm})"),
    }

    if args.json:
        print(json.dumps(report, indent=2, default=lambda o: float(o)))
        return

    print(f"\n{'=' * 70}")
    print(f"Sessions loaded: {report['sessions_loaded']}")
    print(
        f"OLD (TF-IDF + positional):   {report['old_method']['clusters_found']} clusters, "
        f"{report['old_method']['clusters_coherence_ge_0.3']} with coherence>=0.3 "
        f"({report['old_method']['compute_seconds']}s)"
    )
    print(
        f"NEW (embedding + alignment): {report['new_method']['clusters_found']} clusters, "
        f"{report['new_method']['clusters_coherence_ge_0.3']} with coherence>=0.3 "
        f"({report['new_method']['compute_seconds']}s)"
    )
    print(f"{'=' * 70}")
    for label, ranked in (("OLD", old_ranked), ("NEW", new_ranked)):
        for i, c in enumerate(ranked, 1):
            print(
                f"\n[{label}] Cluster {i}: {c['session_count']} sessions, "
                f"coherence={c['goal_coherence']:.3f}, success_rate={c['success_rate']:.1%}"
                + (" ANTI-PATTERN" if c.get("anti_pattern") else "")
            )
            for g in c["example_goals"][:3]:
                print(f"    - {g[:80]}")


if __name__ == "__main__":
    main()
