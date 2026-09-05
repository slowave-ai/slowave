"""Opt-in per-pair instrumentation for the topical relation judge.

No-op unless SLOWAVE_DEBUG_JUDGE_PAIRS is set. It records topical similarity
diagnostics for evaluated pairs; it does not decide lifecycle status.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any


def _log_path() -> str:
    configured = os.environ.get("SLOWAVE_DEBUG_JUDGE_LOG_PATH")
    if configured:
        return os.path.expanduser(configured)
    from slowave.core.paths import runtime_paths

    return str(runtime_paths().judge_debug_log)


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
