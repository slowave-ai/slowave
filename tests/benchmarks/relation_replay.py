"""Offline replay: re-run the geometric relation judge across an existing DB
snapshot with per-pair instrumentation enabled, to build a real dataset of
cosine/direction_score/facet_distance margins.

Motivation: the live production DB cannot answer "how many evaluated pairs
sit close to a decision threshold" -- only `supersedes` verdicts leave a
trail via schema_relations.reason, and even then only for the code paths
that bother to log it (see private/docs/iterations/
20260715_promotion_ladder_and_relation_taxonomy_review.md and the 2026-07-20
razor-thin-margin incident). This script forces a full re-judge over every
prototype in a snapshot with slowave/core/judge_debug.py enabled, so every
evaluated pair -- not just the ones that produced a written edge -- gets
recorded.

SAFETY: always operates on a throwaway copy. Refuses to run directly against
the live ~/.slowave/slowave.db.

Usage:
    python tests/benchmarks/relation_replay.py --from-backup ~/.slowave/backups/slowave-20260716_060649.db.gz
    python tests/benchmarks/relation_replay.py --db /path/to/scratch/copy.db
    python tests/benchmarks/relation_replay.py --db /path/to/scratch/copy.db \\
        --judge-overrides '{"same_topic_cosine": 0.70}'
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import shutil
import sqlite3
import sys
import tempfile
from collections import Counter

from slowave.core.config import SlowaveConfig
from slowave.core.engine import SlowaveEngine
from slowave.core.paths import default_db_path
from slowave.latent.schema import GeometricJudgeConfig


def _prepare_scratch_db(*, from_backup: str | None, db: str | None) -> str:
    """Return a writable scratch DB path, decompressing --from-backup if
    given. Refuses to operate on the live production DB path directly."""
    live_db = os.path.realpath(os.path.expanduser(default_db_path()))

    if from_backup:
        scratch_dir = tempfile.mkdtemp(prefix="slowave_relation_replay_")
        scratch_path = os.path.join(scratch_dir, "snapshot.db")
        with gzip.open(from_backup, "rb") as src, open(scratch_path, "wb") as dst:
            shutil.copyfileobj(src, dst)
        return scratch_path

    if db:
        resolved = os.path.realpath(os.path.expanduser(db))
        if resolved == live_db:
            raise SystemExit(
                f"Refusing to run against the live production DB ({live_db}). "
                "Pass a copy via --db, or use --from-backup on a snapshot."
            )
        return db

    raise SystemExit("Must pass either --from-backup or --db.")


def _all_prototype_ids(db_path: str) -> list[int]:
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute("SELECT id FROM semantic_prototypes").fetchall()
        return [int(r[0]) for r in rows]
    finally:
        conn.close()


def _summarize(log_path: str) -> None:
    if not os.path.exists(log_path):
        print(f"No records written to {log_path} -- nothing to summarize.")
        return

    records = [json.loads(line) for line in open(log_path) if line.strip()]
    print(f"\n{len(records)} judge decisions recorded ({log_path})\n")

    by_verdict = Counter(r["verdict"] for r in records)
    print("Verdict counts:")
    for verdict, count in by_verdict.most_common():
        print(f"  {verdict:20s} {count}")

    # Margin analysis: how close was each supersede-eligible call to
    # DIRECTION_SCORE_THRESHOLD_SUPERSEDE (0.10), the exact signal that
    # produced the 2026-07-20 incident (dir_score=0.102).
    threshold = 0.10
    dir_scores = [r["direction_score"] for r in records if r.get("direction_score") is not None]
    if dir_scores:
        near = [s for s in dir_scores if threshold <= s < threshold + 0.05]
        razor = [s for s in dir_scores if threshold <= s < threshold + 0.01]
        print(f"\ndirection_score populated for {len(dir_scores)}/{len(records)} records")
        print(
            f"  within review-band width of threshold [0.10, 0.15): "
            f"{len(near)} ({100 * len(near) / len(dir_scores):.1f}%)"
        )
        print(f"  razor-thin [0.10, 0.11), same class as the 2026-07-20 incident: " f"{len(razor)}")
    else:
        print("\nNo records had a populated direction_score.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay the geometric judge over a DB snapshot")
    parser.add_argument("--from-backup", default=None, help="path to a .db.gz backup to replay")
    parser.add_argument("--db", default=None, help="path to an already-decompressed scratch copy")
    parser.add_argument(
        "--judge-overrides",
        default="",
        help="JSON dict of GeometricJudgeConfig field overrides, e.g. '{\"same_topic_cosine\": 0.70}'",
    )
    parser.add_argument(
        "--log-path",
        default=os.path.expanduser("~/.slowave/judge_debug_replay.jsonl"),
        help="where to write the JSONL instrumentation output (overwritten each run)",
    )
    args = parser.parse_args()

    scratch_db = _prepare_scratch_db(from_backup=args.from_backup, db=args.db)

    if os.path.exists(args.log_path):
        os.remove(args.log_path)
    os.environ["SLOWAVE_DEBUG_JUDGE_PAIRS"] = "1"
    os.environ["SLOWAVE_DEBUG_JUDGE_LOG_PATH"] = args.log_path

    judge_overrides = json.loads(args.judge_overrides) if args.judge_overrides else {}
    cfg = SlowaveConfig(db_path=scratch_db, judge=GeometricJudgeConfig(**judge_overrides))

    engine = SlowaveEngine(cfg)
    if engine.consolidator is None:
        raise SystemExit("Consolidator unavailable (no encoder?) -- cannot replay.")

    prototype_ids = _all_prototype_ids(scratch_db)
    print(f"Replaying {len(prototype_ids)} prototypes from {scratch_db} ...")
    # Bypass consolidate_once()'s "only prototypes touched by the last replay
    # pass" restriction -- a static snapshot has nothing new to replay, so we
    # explicitly force every existing prototype through the judge.
    stats = engine.consolidator.consolidate(prototype_ids=prototype_ids)
    print(f"consolidate() stats: {stats}")

    engine.close()
    _summarize(args.log_path)


if __name__ == "__main__":
    sys.exit(main())
