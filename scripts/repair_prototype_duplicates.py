#!/usr/bin/env python3
"""Repair duplicate schemas per primary prototype (dedup fix #1).

The one-schema-per-primary-prototype invariant was violated by the
active-only near-dup guard: a recurring prototype accumulated many
superseded/contradicted copies plus the live one. This script merges each
prototype's duplicate rows into a single canonical ACTIVE schema and
archives the rest (evidence/prototype links are moved onto the canonical
row). Archiving is reversible (rows are kept, only status flips), so this
is safe to run against a live DB — but a backup is still required first.

Usage:
    python scripts/repair_prototype_duplicates.py [--db PATH] [--dry-run]
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from typing import Any

from slowave.storage.sqlite_db import SQLiteConfig, SQLiteDB
from slowave.symbolic.schema_store import SchemaStore
from slowave.utils.vec import loads_json

DEFAULT_DB = "~/.slowave/slowave.db"


def _load_duplicate_groups(conn) -> list[tuple[int, list[sqlite3.Row]]]:
    """Return (prototype_id, [schema rows]) for prototypes owning >1 schema."""
    rows = conn.execute(
        "SELECT * FROM schemas WHERE prototype_id IS NOT NULL ORDER BY prototype_id, id"
    ).fetchall()
    groups: dict[int, list[sqlite3.Row]] = {}
    for r in rows:
        groups.setdefault(int(r["prototype_id"]), []).append(r)
    return [(pid, items) for pid, items in groups.items() if len(items) > 1]


def _pick_canonical(items):
    """Prefer an active/needs_review copy; otherwise the highest id."""
    active = [r for r in items if r["status"] in ("active", "needs_review")]
    pool = active if active else items
    return max(pool, key=lambda r: int(r["id"]))


def repair(db_path: str, *, dry_run: bool) -> dict[str, Any]:
    db = SQLiteDB(SQLiteConfig(path=db_path))
    schemas = SchemaStore(db, dim=384)  # dim is only used by create(); not called here
    conn = db.connect()

    groups = _load_duplicate_groups(conn)
    result: dict[str, Any] = {
        "prototype_groups": len(groups),
        "duplicate_rows": 0,
        "reactivated": 0,
    }

    for pid, items in groups:
        canonical = _pick_canonical(items)
        canonical_id = int(canonical["id"])
        dupes = [r for r in items if int(r["id"]) != canonical_id]
        result["duplicate_rows"] += len(dupes)

        if dry_run:
            print(
                f"[dry-run] proto {pid}: canonical sch_{canonical_id} "
                f"({canonical['status']}), {len(dupes)} duplicates"
            )
            continue

        # Merge each duplicate's evidence/prototype links onto the canonical,
        # then archive it.
        for dupe in dupes:
            dupe_id = int(dupe["id"])
            evidence = conn.execute(
                "SELECT episode_id, raw_event_id, quote, weight "
                "FROM schema_evidence WHERE schema_id = ?",
                (dupe_id,),
            ).fetchall()
            proto_rows = conn.execute(
                "SELECT prototype_id FROM schema_prototype_map WHERE schema_id = ?",
                (dupe_id,),
            ).fetchall()
            supporting = loads_json(dupe["supporting_episode_ids"]).get("ids", [])
            contradicting = loads_json(dupe["contradicting_episode_ids"]).get("ids", [])

            schemas.reinforce_schema(
                canonical_id,
                prototype_ids=[int(r["prototype_id"]) for r in proto_rows],
                supporting_episode_ids=[int(x) for x in supporting],
                contradicting_episode_ids=[int(x) for x in contradicting],
                evidence=[
                    (r["episode_id"], r["raw_event_id"], r["quote"], float(r["weight"]))
                    for r in evidence
                ],
                confidence=float(dupe["confidence"]),
                facets=loads_json(dupe["facets_json"]),
                tags=[str(t) for t in loads_json(dupe["tags_json"]).get("tags", [])],
            )
            schemas.update_status(dupe_id, status="archived", salience=0.05)

        # Ensure the canonical survives as the single active engram.
        if canonical["status"] != "active":
            schemas.update_status(canonical_id, status="active")
            result["reactivated"] += 1

    db.close()
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=DEFAULT_DB)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    db_path = os.path.expanduser(args.db)
    if args.dry_run:
        print(f"Dry run against {db_path}")
    out = repair(db_path, dry_run=args.dry_run)
    print(out)
    sys.exit(0)


if __name__ == "__main__":
    main()
