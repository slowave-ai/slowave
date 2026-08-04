#!/usr/bin/env python3
"""Procedural signal validation / monitoring — Phase 1 analysis script.

Mines raw_events + sessions for repeated step-content patterns and clusters
sessions by step sequence similarity + goal coherence. Read-only, no
SlowaveEngine/DB-layer imports (the embedding encoder is a standalone,
stateless module -- not the mutable engine).

Default method (as of 2026-07-27, see
private/docs/iterations/20260727_procedural_memory_phase2_plan.md secs 8-10):
sentence embeddings (not TF-IDF) for step/goal similarity, with greedy
sequence alignment (not positional pairing) for step comparison, and
average-linkage (not single-linkage/greedy-connected-components) for
clustering. This combination was validated against a synthetic ground-truth
benchmark and against real historical data before being made the default --
the original TF-IDF + positional + single-linkage method is preserved via
--legacy for comparison; it was proven to find no signal at all (Phase 1,
private/docs/iterations/20260725_procedural_signal_commit_steps.md).

This is the script to run periodically against a live DB to monitor progress
toward the Phase 1 gate (>=5 clusters with goal_coherence >= 0.3 on real
`commit(steps=...)` data) -- see the GATE line in its output.

Usage:
    python scripts/analyze_procedural_signal.py \
        --db ~/.slowave/slowave.db \
        --min-sessions 2 \
        --max-clusters 20
    python scripts/analyze_procedural_signal.py --db ... --legacy  # old TF-IDF method
"""

import argparse
import json
import math
import sqlite3
from collections import Counter, defaultdict
from typing import Any


def _connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


# ---------------------------------------------------------------------------
# Session loading
# ---------------------------------------------------------------------------


def load_sessions(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    """Load sessions with goal + outcome and their event sequences."""
    session_rows = conn.execute(
        "SELECT id, goal, outcome, scope_id "
        "FROM sessions "
        "WHERE goal IS NOT NULL AND outcome IS NOT NULL "
        "ORDER BY started_ts"
    ).fetchall()

    sessions = []
    for srow in session_rows:
        events = conn.execute(
            "SELECT type, content, ts FROM raw_events "
            "WHERE session_id = ? ORDER BY ts",
            (srow["id"],),
        ).fetchall()
        if len(events) < 2:
            continue
        step_contents = [e["content"] for e in events if e["type"] == "step"]
        sessions.append({
            "id": srow["id"],
            "goal": srow["goal"] or "",
            "outcome": srow["outcome"] or "unknown",
            "scope_id": srow["scope_id"],
            "event_types": [e["type"] for e in events],
            "event_contents": [e["content"] for e in events],
            "step_contents": step_contents,
            "has_steps": len(step_contents) > 0,
        })
    return sessions


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


def _tfidf_vectorize(
    docs: list[str],
) -> tuple[list[dict[str, float]], dict[str, float]]:
    N = len(docs)
    tokenized = [_tokenize(d) for d in docs]
    df: dict[str, int] = {}
    for tokens in tokenized:
        for t in set(tokens):
            df[t] = df.get(t, 0) + 1
    idf = {t: math.log((N + 1) / (df[t] + 1)) + 1 for t in df}
    vecs = []
    for tokens in tokenized:
        tf = Counter(tokens)
        total = sum(tf.values()) or 1
        vec = {t: (tf[t] / total) * idf[t] for t in tf}
        vecs.append(vec)
    return vecs, idf


def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
    keys = set(a) | set(b)
    dot = sum(a.get(k, 0) * b.get(k, 0) for k in keys)
    na = math.sqrt(sum(v * v for v in a.values())) or 1e-12
    nb = math.sqrt(sum(v * v for v in b.values())) or 1e-12
    return dot / (na * nb)


def _goal_coherence(goals: list[str]) -> float:
    if len(goals) < 2:
        return 1.0
    vecs, _ = _tfidf_vectorize(goals)
    sims = []
    for i in range(len(vecs)):
        for j in range(i + 1, len(vecs)):
            sims.append(_cosine(vecs[i], vecs[j]))
    return float(sum(sims) / len(sims)) if sims else 0.0


# ---------------------------------------------------------------------------
# Step-content clustering
# ---------------------------------------------------------------------------


def _step_similarity(
    steps_a: list[str], steps_b: list[str]
) -> float:
    """How similar are two step sequences? Mean of per-position cosine,
    skipping extra positions in the longer sequence."""
    if not steps_a or not steps_b:
        return 0.0
    n = min(len(steps_a), len(steps_b))
    if n == 0:
        return 0.0
    # Vectorize the first n steps of each sequence
    docs = steps_a[:n] + steps_b[:n]
    vecs, _ = _tfidf_vectorize(docs)
    sims = []
    for i in range(n):
        sims.append(_cosine(vecs[i], vecs[n + i]))
    return float(sum(sims) / len(sims))


def cluster_by_step_content(
    sessions: list[dict[str, Any]],
    threshold: float = 0.5,
) -> dict[int, list[dict[str, Any]]]:
    """Greedy clustering: sessions with similar step content + similar goals.
    Returns {cluster_id: [sessions]}."""
    if not sessions:
        return {}

    # Only cluster sessions that have steps
    with_steps = [s for s in sessions if s["has_steps"]]
    if len(with_steps) < 2:
        return {}

    # Compute pairwise similarity: α * step_sim + β * goal_sim
    n = len(with_steps)
    sims: dict[tuple[int, int], float] = {}
    for i in range(n):
        for j in range(i + 1, n):
            step_sim = _step_similarity(
                with_steps[i]["step_contents"],
                with_steps[j]["step_contents"],
            )
            # Goal similarity: simple word overlap (fast, no full TF-IDF per pair)
            goals_i = set(_tokenize(with_steps[i]["goal"]))
            goals_j = set(_tokenize(with_steps[j]["goal"]))
            if goals_i and goals_j:
                goal_sim = len(goals_i & goals_j) / min(len(goals_i), len(goals_j))
            else:
                goal_sim = 0.0
            # Weight step similarity higher (0.7) than goal similarity (0.3)
            sims[(i, j)] = 0.7 * step_sim + 0.3 * goal_sim

    # Greedy clustering: connect pairs above threshold, then find components
    neighbors: dict[int, set[int]] = defaultdict(set)
    for (i, j), sim in sims.items():
        if sim >= threshold:
            neighbors[i].add(j)
            neighbors[j].add(i)

    visited: set[int] = set()
    clusters: dict[int, list[dict[str, Any]]] = {}
    cluster_id = 0
    for i in range(n):
        if i in visited:
            continue
        queue = [i]
        component = []
        while queue:
            node = queue.pop(0)
            if node in visited:
                continue
            visited.add(node)
            component.append(with_steps[node])
            for nb in neighbors[node]:
                if nb not in visited:
                    queue.append(nb)
        clusters[cluster_id] = component
        cluster_id += 1

    return clusters


# ---------------------------------------------------------------------------
# Cluster scoring (legacy TF-IDF path only -- the default embedding path
# uses slowave.symbolic.procedural.score_cluster/rank_clusters instead, see
# main() below)
# ---------------------------------------------------------------------------


def _score_cluster(sessions: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(sessions)
    successes = sum(1 for s in sessions if s["outcome"] == "success")
    failures = n - successes
    success_rate = successes / n if n else 0
    goals = [s["goal"] for s in sessions]
    coherence = _goal_coherence(goals)
    # Tag as anti-pattern when repeatedly failing (≥3 observations, <20% success)
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
        "example_steps": sessions[0]["step_contents"][:5] if sessions[0].get("step_contents") else [],
    }


def rank_clusters(
    clusters: dict[int, list[dict[str, Any]]],
    min_sessions: int,
    max_clusters: int,
) -> list[dict[str, Any]]:
    scored = []
    for cid, sessions in clusters.items():
        if len(sessions) < min_sessions:
            continue
        info = _score_cluster(sessions)
        info["cluster_id"] = cid
        scored.append(info)
    scored.sort(key=lambda x: x["composite"], reverse=True)

    # Detect competing procedures: clusters with similar goals but divergent outcomes
    for i in range(len(scored)):
        for j in range(i + 1, len(scored)):
            gi = set(_tokenize(" ".join(scored[i]["example_goals"])))
            gj = set(_tokenize(" ".join(scored[j]["example_goals"])))
            if gi and gj:
                goal_overlap = len(gi & gj) / min(len(gi), len(gj))
            else:
                goal_overlap = 0.0
            # Competing: goal overlap ≥ 0.4 AND one succeeds while the other fails
            sr_i, sr_j = scored[i]["success_rate"], scored[j]["success_rate"]
            diverged = (sr_i >= 0.7 and sr_j <= 0.2) or (sr_j >= 0.7 and sr_i <= 0.2)
            if goal_overlap >= 0.4 and diverged:
                scored[i].setdefault("competes_with", []).append(scored[j]["cluster_id"])
                scored[j].setdefault("competes_with", []).append(scored[i]["cluster_id"])

    return scored[:max_clusters]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Procedural signal validation")
    parser.add_argument("--db", required=True, help="Path to Slowave SQLite DB")
    parser.add_argument(
        "--min-sessions", type=int, default=2,
        help="Minimum sessions per cluster (default: 2)",
    )
    parser.add_argument(
        "--max-clusters", type=int, default=20,
        help="Max clusters to report (default: 20)",
    )
    parser.add_argument(
        "--threshold", type=float, default=None,
        help="Step+goal similarity threshold for clustering. Default: 0.4 "
             "(embedding+average-linkage) or 0.5 (--legacy TF-IDF+positional). "
             "Embedding cosine similarity runs in a narrower, higher band than "
             "TF-IDF -- don't reuse a tuned TF-IDF threshold with --legacy off.",
    )
    parser.add_argument(
        "--legacy", action="store_true",
        help="Use the original TF-IDF + positional-pairing + single-linkage "
             "method (Phase 1, proven to find no signal) instead of the "
             "embedding + alignment + average-linkage default. Kept for "
             "comparison only.",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Output as JSON instead of human-readable",
    )
    args = parser.parse_args()
    threshold = args.threshold if args.threshold is not None else (0.5 if args.legacy else 0.4)

    conn = _connect(args.db)
    sessions = load_sessions(conn)
    conn.close()

    step_sessions = [s for s in sessions if s["has_steps"]]

    if len(step_sessions) < args.min_sessions:
        msg = (
            f"VERDICT: NOT ENOUGH STEP DATA — {len(step_sessions)} sessions "
            f"with steps (need ≥{args.min_sessions})"
        )
        if args.json:
            print(json.dumps({"error": msg, "total_sessions": len(sessions),
                              "step_sessions": len(step_sessions), "clusters": []}))
        else:
            print(f"Sessions loaded: {len(sessions)}")
            print(f"Sessions with steps: {len(step_sessions)}")
            print(msg)
        return

    if args.legacy:
        clusters = cluster_by_step_content(sessions, threshold=threshold)
        ranked = rank_clusters(clusters, args.min_sessions, args.max_clusters)
    else:
        from slowave.symbolic.encoder import TextEncoder
        from slowave.symbolic.procedural import build_embedding_caches, cluster_sessions
        from slowave.symbolic.procedural import rank_clusters as rank_clusters_embedding

        encoder = TextEncoder()
        step_cache, goal_cache = build_embedding_caches(sessions, encoder)
        clusters = cluster_sessions(sessions, step_cache, goal_cache, threshold=threshold)
        ranked = rank_clusters_embedding(clusters, args.min_sessions, args.max_clusters, goal_cache=goal_cache)

    method = "tfidf+positional+single_linkage (--legacy)" if args.legacy else "embedding+alignment+average_linkage"
    n_good = sum(1 for c in ranked if c["goal_coherence"] >= 0.3)
    gate_pass = n_good >= 5

    if args.json:
        print(json.dumps({
            "method": method,
            "threshold": threshold,
            "total_sessions": len(sessions),
            "step_sessions": len(step_sessions),
            "clusters": ranked,
            "gate": {"criterion": ">=5 clusters, goal_coherence>=0.3", "clusters_qualifying": n_good, "pass": gate_pass},
        }, indent=2))
    else:
        print(f"Method: {method} (threshold={threshold})")
        print(f"Sessions loaded: {len(sessions)}")
        print(f"Sessions with steps: {len(step_sessions)}")
        _print_report(ranked, len(step_sessions))


def _print_report(
    clusters: list[dict[str, Any]],
    step_session_count: int,
) -> None:
    if not clusters:
        print("\nVERDICT: NO SIGNAL — no clusters found meeting thresholds")
        return

    n_clusters = len(clusters)
    n_good = sum(1 for c in clusters if c["goal_coherence"] >= 0.3)
    singletons = step_session_count - sum(c["session_count"] for c in clusters)

    print(f"\n{'='*60}")
    print(f"Clusters found: {n_clusters}")
    print(f"Clusters with goal_coherence ≥ 0.3: {n_good}")
    print(f"Unclustered sessions: {singletons}")
    print(f"{'='*60}")

    for i, c in enumerate(clusters, 1):
        print(f"\n  Cluster {i}: {c['session_count']} sessions")
        print(f"    Goal coherence:  {c['goal_coherence']:.3f}")
        print(f"    Success rate:    {c['successes']}/{c['session_count']} "
              f"({c['success_rate']:.1%})")
        print(f"    Composite:       {c['composite']:.3f}")
        if c.get("anti_pattern"):
            print("    ⚠️  ANTI-PATTERN — procedure consistently fails")
        if c.get("competes_with"):
            print(f"    Competes with clusters: {c['competes_with']}")
        if c.get("example_steps"):
            print("    Steps:")
            for s in c["example_steps"]:
                print(f"      - {s[:80]}")
        print("    Goals:")
        for g in c["example_goals"][:3]:
            print(f"      - {g[:80]}")

    print(f"\n{'='*60}")
    print("VERDICT")
    print(f"{'='*60}")
    if n_good >= 2:
        print(f"  → SIGNAL EXISTS — {n_good} clusters with coherent goals")
    elif n_clusters >= 2:
        print(f"  → MARGINAL — {n_clusters} clusters but goal coherence is low")
    else:
        print("  → NO SIGNAL")

    gate_pass = n_good >= 5
    print(f"\nGATE (Phase 2 criterion: >=5 clusters, goal_coherence>=0.3): "
          f"{'PASS' if gate_pass else 'not yet'} — {n_good}/5")


if __name__ == "__main__":
    main()
