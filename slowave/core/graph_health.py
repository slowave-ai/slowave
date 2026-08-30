"""Graph health metrics — shared by CLI, dashboard, and MCP.

All queries are read-only against the SQLite DB. Uses only stdlib sqlite3
for portability; numpy is optional (enables centroid-distance metrics).

Design: one public function ``compute()`` that takes a DB path and returns
a flat dict with every measured metric. Callers format for their surface.
"""

from __future__ import annotations

import os
import sqlite3
import time
from typing import Any

try:
    import numpy as np

    _HAS_NUMPY = True
except ImportError:
    np = None  # type: ignore
    _HAS_NUMPY = False


def _connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{os.path.abspath(db_path)}?mode=ro", uri=True, timeout=10)
    conn.row_factory = sqlite3.Row
    return conn


def compute(db_path: str) -> dict[str, Any]:
    """Return all graph health metrics as a dict.  Read-only, ~10ms."""
    if not os.path.exists(db_path):
        return {"error": "db_not_found", "db_path": db_path}

    conn = _connect(db_path)
    try:
        return {
            "schema_graph": _schema_graph(conn),
            "prototype_graph": _prototype_graph(conn),
            "episode_graph": _episode_graph(conn),
            "cross_cutting": _cross_cutting(conn),
            "supplementary": _supplementary(conn),
        }
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Schema cosine distribution (requires numpy)
# ---------------------------------------------------------------------------


def _schema_cosine_distribution(conn: sqlite3.Connection) -> dict[str, Any] | None:
    """Pairwise cosine similarity distribution across active/labile schemas.

    Buckets pairs into the verdict zones used by GeometricRelationJudge,
    plus fine-grained 0.05-step histogram bins and percentiles.

    Returns None when numpy is unavailable or there are <2 schemas.
    """
    if not _HAS_NUMPY:
        return None

    rows = conn.execute(
        "SELECT id, embedding FROM schemas "
        "WHERE status IN ('active', 'needs_review') AND embedding IS NOT NULL"
    ).fetchall()
    if len(rows) < 2:
        return None

    n = len(rows)
    assert np is not None  # mypy
    embeddings = np.array([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows])
    # Normalise to unit length
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.maximum(norms, 1e-12)

    # Batched upper-triangle pairwise cosine
    batch_size = 200
    chunks: list[np.ndarray] = []
    for i in range(0, n, batch_size):
        end = min(i + batch_size, n)
        batch = embeddings[i:end]
        sims = np.dot(batch, embeddings.T)
        for j in range(sims.shape[0]):
            global_j = i + j
            chunks.append(sims[j, global_j + 1 :])
    all_cos = np.concatenate(chunks)
    total_pairs = int(len(all_cos))

    # Percentiles
    pcts: dict[str, float] = {}
    for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
        pcts[f"p{p}"] = round(float(np.percentile(all_cos, p)), 4)

    # Verdict-zone buckets
    zones = [
        ("unrelated", 0.0, 0.72),
        ("at_risk", 0.72, 0.75),
        ("same_topic", 0.75, 0.92),
        ("near_dup", 0.92, 1.01),
    ]
    zone_counts: dict[str, int] = {}
    zone_pcts: dict[str, float] = {}
    for name, lo, hi in zones:
        cnt = int(np.sum((all_cos >= lo) & (all_cos < hi)))
        zone_counts[name] = cnt
        zone_pcts[name] = round(100.0 * cnt / total_pairs, 2)

    # Fine-grained histogram (0.05-step bins)
    bins: list[dict[str, Any]] = []
    for lo in np.arange(0.0, 1.0, 0.05):
        hi = float(round(lo + 0.05, 4))
        cnt = int(np.sum((all_cos >= lo) & (all_cos < hi)))
        bins.append({"range": [round(float(lo), 2), hi], "count": cnt})

    # Pairs above key thresholds (approximate edge potential)
    above_075 = int(np.sum(all_cos >= 0.75))
    above_092 = int(np.sum(all_cos >= 0.92))

    return {
        "pairs_sampled": total_pairs,
        "schemas_with_embeddings": n,
        "mean": round(float(all_cos.mean()), 4),
        "median": round(float(np.median(all_cos)), 4),
        "std": round(float(all_cos.std()), 4),
        "min": round(float(all_cos.min()), 4),
        "max": round(float(all_cos.max()), 4),
        "percentiles": pcts,
        "zones": zone_counts,
        "zone_pcts": zone_pcts,
        "pairs_above_075": above_075,
        "pairs_above_092": above_092,
        "histogram_005": bins,
    }


# ---------------------------------------------------------------------------
# Schema graph
# ---------------------------------------------------------------------------


def _schema_graph(conn: sqlite3.Connection) -> dict[str, Any]:
    total = conn.execute("SELECT COUNT(*) FROM schemas").fetchone()[0]

    # S1: status distribution
    status_rows = conn.execute(
        "SELECT status, COUNT(*) as cnt FROM schemas GROUP BY status"
    ).fetchall()
    status = {r["status"]: r["cnt"] for r in status_rows}
    stale_pct = round(100.0 * status.get("stale", 0) / max(1, total), 2)

    # S2: relation distribution
    rel_rows = conn.execute(
        "SELECT relation, COUNT(*) as cnt FROM schema_relations GROUP BY relation"
    ).fetchall()
    relations = {r["relation"]: r["cnt"] for r in rel_rows}
    total_relations = sum(relations.values())

    # S3: degree / isolates
    deg_rows = conn.execute("""SELECT s.id, COUNT(sr2.relation) as degree
           FROM schemas s
           LEFT JOIN (
               SELECT src_schema_id as sid, relation FROM schema_relations
               UNION ALL
               SELECT dst_schema_id as sid, relation FROM schema_relations
           ) sr2 ON s.id = sr2.sid
           GROUP BY s.id""").fetchall()
    degrees = [r["degree"] for r in deg_rows]
    isolates = sum(1 for d in degrees if d == 0)
    sorted_deg = sorted(degrees)
    n_deg = len(sorted_deg)

    # S5: salience
    sal_rows = conn.execute("SELECT salience FROM schemas").fetchall()
    salience_vals = sorted(r["salience"] for r in sal_rows)
    n_sal = len(salience_vals)
    ceiling_breaches = sum(1 for v in salience_vals if v > 20.0)

    # S4: connected components (union-find)
    components = _connected_components(conn)

    # S6: pairwise cosine similarity distribution
    cosine_dist = _schema_cosine_distribution(conn)

    return {
        "total": total,
        "status": status,
        "stale_pct": stale_pct,
        "relations": relations,
        "total_relations": total_relations,
        "isolates": isolates,
        "isolate_pct": round(100.0 * isolates / max(1, total), 1),
        "mean_degree": round(sum(degrees) / max(1, n_deg), 1),
        "median_degree": sorted_deg[n_deg // 2] if n_deg else 0,
        "max_degree": max(degrees) if degrees else 0,
        "salience": {
            "min": round(salience_vals[0], 4) if n_sal else 0,
            "max": round(salience_vals[-1], 4) if n_sal else 0,
            "mean": round(sum(salience_vals) / n_sal, 4) if n_sal else 0,
            "median": round(salience_vals[n_sal // 2], 4) if n_sal else 0,
            "ceiling_breaches": ceiling_breaches,
        },
        "components": components,
        "component_coherence": _component_coherence(conn),
        "hubs_authorities": _hubs_authorities(conn),
        "cosine_distribution": cosine_dist,
    }


def _connected_components(conn: sqlite3.Connection) -> dict[str, Any]:
    """Union-find over schema_relations (undirected)."""
    schema_rows = conn.execute("SELECT id FROM schemas").fetchall()
    all_ids = [r["id"] for r in schema_rows]
    id_to_idx = {sid: i for i, sid in enumerate(all_ids)}
    n = len(all_ids)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    edge_rows = conn.execute("SELECT src_schema_id, dst_schema_id FROM schema_relations").fetchall()
    for r in edge_rows:
        a = id_to_idx.get(r["src_schema_id"])
        b = id_to_idx.get(r["dst_schema_id"])
        if a is not None and b is not None:
            union(a, b)

    comp_sizes: dict[int, int] = {}
    for i in range(n):
        root = find(i)
        comp_sizes[root] = comp_sizes.get(root, 0) + 1

    sizes = sorted(comp_sizes.values(), reverse=True)
    from collections import Counter

    hist = Counter(sizes)
    buckets: dict[str, int] = {}
    for s, c in sorted(hist.items()):
        if s == 1:
            buckets["1"] = c
        elif s <= 5:
            buckets["2-5"] = buckets.get("2-5", 0) + c
        elif s <= 20:
            buckets["6-20"] = buckets.get("6-20", 0) + c
        elif s <= 100:
            buckets["21-100"] = buckets.get("21-100", 0) + c
        else:
            buckets["100+"] = buckets.get("100+", 0) + c

    return {
        "total_components": len(sizes),
        "largest_component": sizes[0] if sizes else 0,
        "isolates": sum(1 for s in sizes if s == 1),
        "size_buckets": buckets,
    }


def _component_coherence(conn: sqlite3.Connection) -> dict[str, Any] | None:
    """Scope purity and mean pairwise cosine per connected component.

    Returns aggregate stats plus a per-component breakdown for the top-N
    components. Requires numpy for pairwise cosine computation.
    """
    if not _HAS_NUMPY:
        return None
    assert np is not None

    rows = conn.execute(
        "SELECT id, embedding, scope_id FROM schemas " "WHERE status IN ('active', 'needs_review')"
    ).fetchall()
    if len(rows) < 2:
        return None

    all_ids = [r["id"] for r in rows]
    id_to_idx = {sid: i for i, sid in enumerate(all_ids)}
    n = len(all_ids)

    emb_map: dict[int, np.ndarray] = {}
    scope_map: dict[int, str | None] = {}
    for r in rows:
        scope_map[r["id"]] = r["scope_id"]
        if r["embedding"] is not None:
            vec = np.frombuffer(r["embedding"], dtype=np.float32)
            emb_map[r["id"]] = vec / (np.linalg.norm(vec) + 1e-12)

    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    edge_rows = conn.execute("SELECT src_schema_id, dst_schema_id FROM schema_relations").fetchall()
    for r in edge_rows:
        a = id_to_idx.get(r["src_schema_id"])
        b = id_to_idx.get(r["dst_schema_id"])
        if a is not None and b is not None:
            union(a, b)

    comp_of = [find(i) for i in range(n)]
    comp_to_members: dict[int, list[int]] = {}
    for i, root in enumerate(comp_of):
        comp_to_members.setdefault(root, []).append(i)

    comps = sorted(comp_to_members.values(), key=len, reverse=True)

    from collections import Counter

    purities: list[float] = []
    mean_cosines: list[float] = []
    top_components: list[dict[str, Any]] = []
    low_coherence: list[dict[str, Any]] = []

    for members in comps[:20]:
        mids = [all_ids[i] for i in members]
        sz = len(members)
        scope_counts = Counter(scope_map.get(mid) for mid in mids)
        dominant = scope_counts.most_common(1)[0] if scope_counts else (None, 0)
        purity = round(100.0 * dominant[1] / sz, 1)

        cosines: list[float] = []
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                e1 = emb_map.get(all_ids[members[i]])
                e2 = emb_map.get(all_ids[members[j]])
                if e1 is not None and e2 is not None:
                    cosines.append(float(np.dot(e1, e2)))
        mean_cos = round(float(np.mean(cosines)), 4) if cosines else None

        purities.append(purity)
        if mean_cos is not None:
            mean_cosines.append(mean_cos)

        comp_info: dict[str, Any] = {
            "size": sz,
            "purity_pct": purity,
            "dominant_scope": (dominant[0] or "none") if dominant[0] else "none",
            "mean_pairwise_cosine": mean_cos,
        }
        top_components.append(comp_info)
        if purity < 60 and (mean_cos is not None and mean_cos < 0.45):
            low_coherence.append(comp_info)

    return {
        "mean_purity_pct": round(float(np.mean(purities)), 1) if purities else None,
        "median_purity_pct": round(float(np.median(purities)), 1) if purities else None,
        "mean_within_cosine": round(float(np.mean(mean_cosines)), 4) if mean_cosines else None,
        "median_within_cosine": round(float(np.median(mean_cosines)), 4) if mean_cosines else None,
        "low_coherence_components": len(low_coherence),
        "low_coherence": low_coherence,
        "top_components": top_components,
    }


def _hubs_authorities(conn: sqlite3.Connection) -> dict[str, Any] | None:
    """HITS (Hyperlink-Induced Topic Search) hub/authority scores.

    Iterative computation over the directed schema_relations graph.
    Requires numpy.
    """
    if not _HAS_NUMPY:
        return None
    assert np is not None

    rows = conn.execute(
        "SELECT id FROM schemas WHERE status IN ('active', 'needs_review') ORDER BY id"
    ).fetchall()
    all_ids = [r["id"] for r in rows]
    id_to_idx = {sid: i for i, sid in enumerate(all_ids)}
    n = len(all_ids)
    if n == 0:
        return None

    edges = conn.execute(
        "SELECT src_schema_id, dst_schema_id, confidence FROM schema_relations"
    ).fetchall()

    hub = np.ones(n, dtype=np.float64)
    auth = np.ones(n, dtype=np.float64)
    for _ in range(50):
        new_auth = np.zeros(n, dtype=np.float64)
        new_hub = np.zeros(n, dtype=np.float64)
        for r in edges:
            si = id_to_idx.get(r["src_schema_id"])
            di = id_to_idx.get(r["dst_schema_id"])
            if si is not None and di is not None:
                w = float(r["confidence"] or 1.0)
                new_auth[di] += hub[si] * w
                new_hub[si] += auth[di] * w
        new_auth = new_auth / (np.linalg.norm(new_auth) + 1e-12)
        new_hub = new_hub / (np.linalg.norm(new_hub) + 1e-12)
        if np.max(np.abs(new_auth - auth)) < 1e-8 and np.max(np.abs(new_hub - hub)) < 1e-8:
            break
        auth, hub = new_auth, new_hub

    return {
        "hub_mean": round(float(np.mean(hub)), 4),
        "hub_median": round(float(np.median(hub)), 4),
        "hub_max": round(float(np.max(hub)), 4),
        "hub_p95": round(float(np.percentile(hub, 95)), 4),
        "auth_mean": round(float(np.mean(auth)), 4),
        "auth_median": round(float(np.median(auth)), 4),
        "auth_max": round(float(np.max(auth)), 4),
        "auth_p95": round(float(np.percentile(auth, 95)), 4),
        "schemas_scored": n,
    }


# ---------------------------------------------------------------------------
# Prototype graph
# ---------------------------------------------------------------------------


def _prototype_graph(conn: sqlite3.Connection) -> dict[str, Any]:
    total = conn.execute("SELECT COUNT(*) FROM semantic_prototypes").fetchone()[0]
    scale_rows = conn.execute(
        "SELECT scale, COUNT(*) as cnt FROM semantic_prototypes GROUP BY scale"
    ).fetchall()
    scales = {r["scale"]: r["cnt"] for r in scale_rows}

    # Edge stats
    edge_row = conn.execute(
        "SELECT COUNT(*) as cnt, AVG(weight) as avg_w, "
        "MIN(weight) as min_w, MAX(weight) as max_w FROM prototype_edges"
    ).fetchone()
    edge_count = edge_row["cnt"]
    proto_density = round(100.0 * edge_count / max(1, total * (total - 1)), 2)

    # Component fractions
    comp_rows = conn.execute(
        "SELECT w_similarity, w_transition, w_coactivation FROM prototype_edges"
    ).fetchall()
    sim_dom, trans_dom, coact_dom = 0, 0, 0
    sim_fracs, trans_fracs, coact_fracs = [], [], []
    for r in comp_rows:
        ws, wt, wc = float(r[0]), float(r[1]), float(r[2])
        tw = ws + wt + wc
        if tw > 0:
            sf, tf, cf = ws / tw, wt / tw, wc / tw
            sim_fracs.append(sf)
            trans_fracs.append(tf)
            coact_fracs.append(cf)
            if sf > 0.5:
                sim_dom += 1
            elif tf > 0.5:
                trans_dom += 1
            elif cf > 0.5:
                coact_dom += 1
    n_edges = max(1, edge_count)
    n_fracs = max(1, len(sim_fracs))

    # Support & variance
    sup_rows = conn.execute(
        "SELECT support_count, variance, scale FROM semantic_prototypes"
    ).fetchall()
    fine_s = sorted(r["support_count"] for r in sup_rows if r["scale"] == "fine")
    coarse_s = sorted(r["support_count"] for r in sup_rows if r["scale"] == "coarse")
    fine_v = [r["variance"] for r in sup_rows if r["scale"] == "fine"]
    zero_var_pct = round(100.0 * sum(1 for v in fine_v if v == 0.0) / max(1, len(fine_v)), 1)

    # Age
    now = int(time.time())
    age_rows = conn.execute("SELECT last_updated_ts FROM semantic_prototypes").fetchall()
    fresh_24 = sum(1 for r in age_rows if (now - r[0]) / 86400.0 <= 1)
    fresh_7d = sum(1 for r in age_rows if (now - r[0]) / 86400.0 <= 7)

    # Schema-prototype mapping
    map_n = conn.execute("SELECT COUNT(*) FROM schema_prototype_map").fetchone()[0]
    mapped = conn.execute("SELECT COUNT(DISTINCT schema_id) FROM schema_prototype_map").fetchone()[
        0
    ]
    multi = conn.execute(
        "SELECT COUNT(*) FROM (SELECT schema_id FROM schema_prototype_map "
        "GROUP BY schema_id HAVING COUNT(*) > 1)"
    ).fetchone()[0]

    return {
        "total": total,
        "scales": scales,
        "edge_count": edge_count,
        "density_pct": proto_density,
        "edge_weight": {
            "avg": round(edge_row["avg_w"], 4) if edge_row["avg_w"] else 0,
            "min": round(edge_row["min_w"], 4) if edge_row["min_w"] else 0,
            "max": round(edge_row["max_w"], 4) if edge_row["max_w"] else 0,
        },
        "edge_composition": {
            "similarity_dominated": sim_dom,
            "transition_dominated": trans_dom,
            "coactivation_dominated": coact_dom,
            "similarity_pct": round(100.0 * sim_dom / n_edges, 1),
            "transition_pct": round(100.0 * trans_dom / n_edges, 1),
            "coactivation_pct": round(100.0 * coact_dom / n_edges, 1),
            "mean_similarity_fraction": round(sum(sim_fracs) / n_fracs, 4),
            "mean_transition_fraction": round(sum(trans_fracs) / n_fracs, 4),
            "mean_coactivation_fraction": round(sum(coact_fracs) / n_fracs, 4),
        },
        "support": {
            "fine_min": fine_s[0] if fine_s else 0,
            "fine_max": fine_s[-1] if fine_s else 0,
            "fine_median": fine_s[len(fine_s) // 2] if fine_s else 0,
            "coarse_min": coarse_s[0] if coarse_s else 0,
            "coarse_max": coarse_s[-1] if coarse_s else 0,
            "coarse_median": coarse_s[len(coarse_s) // 2] if coarse_s else 0,
        },
        "zero_variance_fine_pct": zero_var_pct,
        "distances": _centroid_distances(conn),
        "age": {
            "pct_updated_24h": round(100.0 * fresh_24 / max(1, total), 1),
            "pct_updated_7d": round(100.0 * fresh_7d / max(1, total), 1),
        },
        "schema_mapping": {
            "total_mappings": map_n,
            "mapped_schemas": mapped,
            "multi_prototype_schemas": multi,
            "multi_prototype_pct": round(100.0 * multi / max(1, mapped), 1),
        },
    }


def _centroid_distances(conn: sqlite3.Connection) -> dict[str, Any]:
    """Compute intra/inter cluster cosine distances. Requires numpy."""
    try:
        import numpy as np
    except ImportError:
        return {"error": "numpy not available"}
    rows = conn.execute("SELECT centroid, scale FROM semantic_prototypes").fetchall()
    if not rows:
        return {"error": "no prototypes"}
    fine_vecs: list = []
    coarse_vecs: list = []
    for r in rows:
        vec = np.frombuffer(r["centroid"], dtype=np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        (fine_vecs if r["scale"] == "fine" else coarse_vecs).append(vec)

    def _mean_pairwise(vecs):
        if len(vecs) < 2:
            return 0.0
        X = np.stack(vecs)
        sim = X @ X.T
        n = sim.shape[0]
        return float(np.mean(1.0 - sim[np.triu_indices(n, k=1)]))

    def _inter(f, c):
        if not f or not c:
            return 0.0
        return float(np.mean(1.0 - np.stack(f) @ np.stack(c).T))

    return {
        "fine_intra": round(_mean_pairwise(fine_vecs), 4),
        "coarse_intra": round(_mean_pairwise(coarse_vecs), 4),
        "inter": round(_inter(fine_vecs, coarse_vecs), 4),
    }


# ---------------------------------------------------------------------------
# Episode graph
# ---------------------------------------------------------------------------


def _episode_graph(conn: sqlite3.Connection) -> dict[str, Any]:
    row = conn.execute(
        "SELECT COUNT(*) as cnt, AVG(salience) as avg_s, "
        "MIN(salience) as min_s, MAX(salience) as max_s, "
        "AVG(recalled_count) as avg_rc, MAX(recalled_count) as max_rc "
        "FROM episodic_memories"
    ).fetchone()
    total_schemas = conn.execute("SELECT COUNT(*) FROM schemas").fetchone()[0]
    return {
        "total": row["cnt"],
        "salience": {
            "avg": round(row["avg_s"] or 0, 4),
            "min": round(row["min_s"] or 0, 4),
            "max": round(row["max_s"] or 0, 4),
        },
        "recalled": {
            "avg": round(row["avg_rc"] or 0, 2),
            "max": row["max_rc"] or 0,
        },
        "episodes_per_schema": round(row["cnt"] / max(1, total_schemas), 2),
    }


# ---------------------------------------------------------------------------
# Cross-cutting
# ---------------------------------------------------------------------------


def _cross_cutting(conn: sqlite3.Connection) -> dict[str, Any]:
    sess_row = conn.execute(
        "SELECT COUNT(*) as total, "
        "COUNT(CASE WHEN ended_ts IS NULL THEN 1 END) as open_sessions "
        "FROM sessions"
    ).fetchone()
    wr_row = conn.execute(
        "SELECT COUNT(*) as total, AVG(duration_ms) as avg_d, "
        "MAX(duration_ms) as max_d, "
        "SUM(prototypes_processed) as proto, "
        "SUM(episodes_processed) as ep, "
        "SUM(schemas_created) as sc, SUM(schemas_decayed) as sd "
        "FROM worker_runs"
    ).fetchone()
    n_proto = conn.execute("SELECT COUNT(*) FROM semantic_prototypes").fetchone()[0]
    n_schema = conn.execute("SELECT COUNT(*) FROM schemas").fetchone()[0]
    proto_edges = conn.execute("SELECT COUNT(*) FROM prototype_edges").fetchone()[0]
    schema_edges = conn.execute("SELECT COUNT(*) FROM schema_relations").fetchone()[0]
    return {
        "sessions": {
            "total": sess_row["total"],
            "open": sess_row["open_sessions"],
        },
        "worker_runs": {
            "total": wr_row["total"],
            "avg_duration_ms": round(wr_row["avg_d"] or 0, 0),
            "max_duration_ms": wr_row["max_d"] or 0,
            "prototypes_processed": wr_row["proto"] or 0,
            "episodes_processed": wr_row["ep"] or 0,
            "schemas_created": wr_row["sc"] or 0,
            "schemas_decayed": wr_row["sd"] or 0,
        },
        "schema_graph_density_pct": round(
            100.0 * schema_edges / max(1, n_schema * (n_schema - 1)), 3
        ),
        "prototype_graph_density_pct": round(
            100.0 * proto_edges / max(1, n_proto * (n_proto - 1)), 2
        ),
    }


# ---------------------------------------------------------------------------
# Supplementary
# ---------------------------------------------------------------------------


def _supplementary(conn: sqlite3.Connection) -> dict[str, Any]:
    labile = conn.execute("SELECT COUNT(*) FROM schemas WHERE is_labile=1").fetchone()[0]
    gen_rows = conn.execute(
        "SELECT generalization_stage, COUNT(*) FROM schemas "
        "GROUP BY generalization_stage ORDER BY generalization_stage"
    ).fetchall()
    gen_stages = {str(r[0]): r[1] for r in gen_rows}
    proto_scales = dict(
        conn.execute("SELECT scale, COUNT(*) FROM semantic_prototypes GROUP BY scale").fetchall()
    )
    top_scopes = conn.execute(
        "SELECT scope_id, COUNT(*) as cnt FROM schemas "
        "WHERE scope_id IS NOT NULL GROUP BY scope_id ORDER BY cnt DESC LIMIT 5"
    ).fetchall()
    return {
        "labile_schemas": labile,
        "generalization_stages": gen_stages,
        "prototype_scales": proto_scales,
        "top_scopes": [(r["scope_id"], r["cnt"]) for r in top_scopes],
    }


def snapshot(conn: sqlite3.Connection, worker_run_id: int | None) -> int | None:
    """Compute metrics and INSERT a row into graph_health_snapshots.

    Takes a WRITABLE connection (not read-only). Returns the new row id,
    or None if the table doesn't exist yet.
    """
    import time as _time

    gh = {
        "schema_graph": _schema_graph(conn),
        "prototype_graph": _prototype_graph(conn),
        "episode_graph": _episode_graph(conn),
        "cross_cutting": _cross_cutting(conn),
        "supplementary": _supplementary(conn),
    }

    s = gh["schema_graph"]
    p = gh["prototype_graph"]
    e = gh["episode_graph"]
    x = gh["cross_cutting"]
    sup = gh["supplementary"]
    d = p["distances"]

    try:
        cur = conn.execute(
            """INSERT INTO graph_health_snapshots (
               worker_run_id, ts,
               total_schemas, stale_pct,
               schema_isolates, schema_isolate_pct,
               schema_components, schema_largest_component,
               salience_median, salience_ceiling_breaches,
               proto_edge_count, proto_similarity_dom_pct,
               proto_fine_intra, proto_coarse_intra, proto_inter,
               proto_zero_variance_pct,
               episodes_total, episodes_per_schema,
               worker_runs_total, schemas_decayed_total,
               labile_schemas
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                worker_run_id,
                int(_time.time()),
                s["total"],
                s["stale_pct"],
                s["isolates"],
                s["isolate_pct"],
                s["components"]["total_components"],
                s["components"]["largest_component"],
                s["salience"]["median"],
                s["salience"]["ceiling_breaches"],
                p["edge_count"],
                p["edge_composition"]["similarity_pct"],
                d.get("fine_intra"),
                d.get("coarse_intra"),
                d.get("inter"),
                p["zero_variance_fine_pct"],
                e["total"],
                e["episodes_per_schema"],
                x["worker_runs"]["total"],
                x["worker_runs"]["schemas_decayed"],
                sup["labile_schemas"],
            ),
        )
        conn.commit()
        return cur.lastrowid
    except sqlite3.OperationalError:
        # Table doesn't exist yet (schema not migrated)
        return None
