#!/usr/bin/env python3
"""Read-only legacy-to-v9 feedback replay and safety report."""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any


def _ids(value: str | None) -> list[str]:
    try:
        parsed = json.loads(value or "[]")
    except (json.JSONDecodeError, TypeError):
        return []
    return [str(item) for item in parsed] if isinstance(parsed, list) else []


def report(db_path: str) -> dict[str, Any]:
    uri = f"file:{Path(db_path).expanduser().resolve()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    integrity = str(conn.execute("PRAGMA integrity_check").fetchone()[0])
    exposures: dict[str, dict[str, set[str]]] = {}
    for row in conn.execute(
        "SELECT context_id, memory_id, memory_type FROM context_recall_items WHERE admitted=1"
    ):
        bucket = exposures.setdefault(row["context_id"], {"memory": set(), "procedure": set()})
        kind = "procedure" if row["memory_type"] == "procedural_memory" else "memory"
        if row["memory_type"] in {"schema", "related", "procedural_memory"}:
            bucket[kind].add(str(row["memory_id"]))

    mappings: Counter[str] = Counter()
    unauthorized: Counter[str] = Counter()
    ambiguous: Counter[str] = Counter()
    outcome_coupled = 0
    rows = conn.execute("SELECT * FROM context_feedback_events ORDER BY id").fetchall()
    memory_columns = {
        "used_memory_ids_json": "used",
        "irrelevant_memory_ids_json": "irrelevant",
        "stale_memory_ids_json": "stale",
        "wrong_memory_ids_json": "wrong",
    }
    procedure_columns = {
        "used_procedure_ids_json": "used_requires_contribution",
        "irrelevant_procedure_ids_json": "not_used",
        "stale_procedure_ids_json": "unsupported_legacy_stale",
        "wrong_procedure_ids_json": "unsupported_legacy_wrong",
    }
    for row in rows:
        retrieval_id = str(row["context_id"])
        exposed = exposures.get(retrieval_id, {"memory": set(), "procedure": set()})
        if float(row["outcome_reward"] or 0.0) != 0.0:
            outcome_coupled += 1
        for column, assessment in memory_columns.items():
            for target_id in _ids(row[column]):
                if target_id in exposed["memory"]:
                    mappings[f"memory:{assessment}"] += 1
                else:
                    unauthorized["memory"] += 1
        for column, assessment in procedure_columns.items():
            for target_id in _ids(row[column]):
                if target_id not in exposed["procedure"]:
                    unauthorized["procedure"] += 1
                elif assessment == "not_used":
                    mappings["procedure:not_used"] += 1
                else:
                    ambiguous[assessment] += 1

    has_v9 = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='feedback_events'"
    ).fetchone()
    v9_rows = (
        int(conn.execute("SELECT COUNT(*) FROM feedback_events").fetchone()[0]) if has_v9 else 0
    )
    conn.close()
    safe_for_historical_backfill = not unauthorized and not ambiguous and outcome_coupled == 0
    return {
        "integrity": integrity,
        "legacy_feedback_rows": len(rows),
        "v9_feedback_rows": v9_rows,
        "authorized_target_mappings": dict(sorted(mappings.items())),
        "unauthorized_or_unprovable_targets": dict(sorted(unauthorized.items())),
        "ambiguous_procedure_mappings": dict(sorted(ambiguous.items())),
        "outcome_coupled_legacy_rows": outcome_coupled,
        "safe_for_historical_backfill": safe_for_historical_backfill,
        "recommended_migration": "backfill" if safe_for_historical_backfill else "zero_start",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True)
    args = parser.parse_args()
    print(json.dumps(report(args.db), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
