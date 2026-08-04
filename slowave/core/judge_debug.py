"""Opt-in per-pair instrumentation for the supersession/relation judge.

No-op unless SLOWAVE_DEBUG_JUDGE_PAIRS is set — zero cost in normal
production operation. Exists to answer a question the production DB can't:
how close (in cosine/direction_score/facet_distance terms) every evaluated
candidate pair was to a decision threshold, not just the ones that ended up
producing a written schema_relations edge. See private/docs/iterations/
20260715_promotion_ladder_and_relation_taxonomy_review.md and the 2026-07-20
razor-thin-margin incident (dir_score=0.102 against a 0.10 threshold) that
motivated this.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any


def _log_path() -> str:
    return os.path.expanduser(
        os.environ.get("SLOWAVE_DEBUG_JUDGE_LOG_PATH", "~/.slowave/judge_debug.jsonl")
    )


def emit_judge_signal(record: dict[str, Any]) -> None:
    """Append one JSONL record describing a single judge decision.

    Call this at every decision point (one call per branch/return), not only
    on verdicts that produce a written relation edge — reinforce/labile/skip
    outcomes are exactly the ones missing from today's schema_relations trail.
    """
    if not os.environ.get("SLOWAVE_DEBUG_JUDGE_PAIRS"):
        return
    payload = {"ts": int(time.time()), **record}
    try:
        path = _log_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a") as f:
            f.write(json.dumps(payload, default=str) + "\n")
    except Exception:
        # Diagnostic-only: never let logging failure affect the judge itself.
        pass
