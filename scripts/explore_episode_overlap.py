#!/usr/bin/env python3
"""Episode-overlap containment scan for semantic hierarchy (Gap 1).

Tests the architecture-evaluation recommendation:
    child_is_part_of_parent :=
        child.episodes ⊂ parent.episodes
        AND |child.episodes| >= min_support
        AND |parent.episodes| / |child.episodes| >= asymmetry_ratio

Usage:
    python scripts/explore_episode_overlap.py \
        --db /tmp/slowave_prehubprune_analysis.db \
        --min-support 2 --asymmetry-ratio 3
"""

import argparse
import sqlite3
from collections.abc import Sequence


def load_gold_pairs(db_path: str) -> set[tuple[int, int]]:
    """Load known part_of edges as (parent_id, child_id)."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    rows = conn.execute(
        "SELECT src_schema_id AS child_id, dst_schema_id AS parent_id "
        "FROM schema_relations WHERE relation = 'part_of'"
    ).fetchall()
    conn.close()
    return {(r[1], r[0]) for r in rows}


def load_schema_episodes(db_path: str) -> dict[int, set[int]]:
    """Return {schema_id: {episode_ids}} for active schemas with evidence."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    active = {
        r["id"]
        for r in conn.execute(
            "SELECT id FROM schemas WHERE status = 'active'"
        ).fetchall()
    }
    result: dict[int, set[int]] = {}
    for row in conn.execute(
        "SELECT schema_id, episode_id FROM schema_evidence "
        "WHERE episode_id IS NOT NULL"
    ):
        sid = row["schema_id"]
        eid = row["episode_id"]
        if sid in active:
            result.setdefault(sid, set()).add(eid)
    conn.close()
    return result
def run_scan(
    schema_eps: dict[int, set[int]],
    gold_pairs: set[tuple[int, int]],
    *,
    min_support: int = 2,
    asymmetry_ratio: float = 3.0,
) -> dict:
    """Full-pool episode-overlap scan."""
    sids = sorted(schema_eps)
    n = len(sids)
    candidates: list[dict] = []
    gold_hits: list[tuple[int, int]] = []
    pairs_checked = 0
    for i in range(n):
        pid = sids[i]
        p_eps = schema_eps[pid]
        p_len = len(p_eps)
        if p_len < min_support * asymmetry_ratio:
            continue
        for j in range(n):
            if i == j:
                continue
            cid = sids[j]
            c_eps = schema_eps[cid]
            c_len = len(c_eps)
            if c_len < min_support or c_len >= p_len:
                continue
            pairs_checked += 1
            if not c_eps.issubset(p_eps):
                continue
            if p_len / c_len < asymmetry_ratio:
                continue
            candidates.append({
                "parent_id": pid,
                "child_id": cid,
                "parent_episodes": p_len,
                "child_episodes": c_len,
                "overlap_ratio": c_len / p_len,
            })
            if (pid, cid) in gold_pairs:
                gold_hits.append((pid, cid))
    return {
        "pairs_checked": pairs_checked,
        "candidates_found": len(candidates),
        "gold_hits": len(gold_hits),
        "gold_pairs": len(gold_pairs),
        "candidates": sorted(
            candidates, key=lambda c: (-c["overlap_ratio"], c["parent_id"])
        ),
        "gold_hit_pairs": gold_hits,
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Episode-overlap containment scan"
    )
    parser.add_argument("--db", required=True)
    parser.add_argument("--min-support", type=int, default=2)
    parser.add_argument("--asymmetry-ratio", type=float, default=3.0)
    parser.add_argument("--sample", type=int, default=30)
    parser.add_argument("--show-missed", action="store_true")
    args = parser.parse_args(argv)

    schema_eps = load_schema_episodes(args.db)
    print(f"Schemas with episodes: {len(schema_eps)}")
    ec = sorted(len(v) for v in schema_eps.values())
    nc = len(ec)
    print(f"  Episodes/schema: min={ec[0]} p25={ec[nc//4]} "
          f"median={ec[nc//2]} p75={ec[3*nc//4]} max={ec[-1]}")

    gold_pairs = load_gold_pairs(args.db)
    gpe = {(p, c) for p, c in gold_pairs if p in schema_eps and c in schema_eps}
    missed_ep = len(gold_pairs) - len(gpe)
    print(f"Gold pairs: {len(gold_pairs)} total, {len(gpe)} with episodes"
          f"{f' ({missed_ep} missing data)' if missed_ep else ''}")

    print(f"\nScan: min_support={args.min_support} "
          f"asymmetry_ratio={args.asymmetry_ratio}")
    result = run_scan(
        schema_eps, gpe,
        min_support=args.min_support,
        asymmetry_ratio=args.asymmetry_ratio,
    )
    prec = (
        result["gold_hits"] / result["candidates_found"] * 100
        if result["candidates_found"] else 0.0
    )
    rec = (
        result["gold_hits"] / result["gold_pairs"] * 100
        if result["gold_pairs"] else 0.0
    )
    print(f"  Pairs checked:    {result['pairs_checked']:,}")
    print(f"  Candidates:       {result['candidates_found']}")
    print(f"  Gold hits:        {result['gold_hits']}/{result['gold_pairs']}")
    print(f"  Precision:        {prec:.1f}%")
    print(f"  Recall:           {rec:.1f}%")

    if not result["candidates"]:
        print("\nNo candidates found.")
        return

    conn = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    top = result["candidates"][:args.sample]
    ids = {c["parent_id"] for c in top} | {c["child_id"] for c in top}
    ph = ",".join("?" * len(ids))
    rows = conn.execute(
        f"SELECT id, content_text FROM schemas WHERE id IN ({ph})",
        list(ids),
    ).fetchall()
    texts = {r["id"]: r["content_text"] for r in rows}
    conn.close()

    print(f"\nTop {min(args.sample, len(result['candidates']))} candidates:")
    for i, c in enumerate(top):
        gold = " [GOLD]" if (c["parent_id"], c["child_id"]) in gold_pairs else ""
        print(
            f"\n  #{i+1}{gold} "
            f"parent={c['parent_id']} child={c['child_id']} "
            f"overlap_ratio={c['overlap_ratio']:.3f} "
            f"(p_eps={c['parent_episodes']} c_eps={c['child_episodes']})"
        )
        pt = (texts.get(c["parent_id"]) or "(missing)")[:130]
        ct = (texts.get(c["child_id"]) or "(missing)")[:130]
        print(f"    P: {pt}")
        print(f"    C: {ct}")
    print()


if __name__ == "__main__":
    main()