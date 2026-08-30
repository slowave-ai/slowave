"""Tests for the read-only human-labelling export of candidate procedures."""

from __future__ import annotations

import os
import tempfile

from slowave import ops
from slowave.core.config import SlowaveConfig
from slowave.core.engine import SlowaveEngine
from slowave.procedural_audit import build_audit_export


def _engine() -> tuple[SlowaveEngine, str]:
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    return SlowaveEngine(SlowaveConfig(db_path=tmp.name, dim=8, disable_encoder=True)), tmp.name


def _cleanup(path: str) -> None:
    for suffix in ("", "-wal", "-shm"):
        candidate = path + suffix
        if os.path.exists(candidate):
            os.remove(candidate)


def _commit(eng: SlowaveEngine, goal: str, steps: list[str]) -> None:
    started = ops.activate(eng, query=goal, goal=goal, scope="project:audit", agent="test")
    ops.commit(eng, session_id=started["session_id"], outcome="success", steps=steps)


def test_audit_export_contains_full_member_traces_and_empty_review_fields() -> None:
    eng, path = _engine()
    try:
        for target in ("auth", "billing", "notifications"):
            _commit(
                eng,
                f"deploy {target} service to staging",
                [
                    "Ran regression tests",
                    "Built container image",
                    "Applied rollout",
                    "Verified health",
                ],
            )

        result = build_audit_export(path, min_sessions=3)

        assert result["procedure_memory_status"] == "inspection_only_not_stored_or_retrieved"
        assert result["method"] == "embedding+alignment+average_linkage"
        assert result["review_instructions"]["labels"] == ["recommend", "warn", "reject"]
        assert result["clusters"]
        candidate = result["clusters"][0]
        assert candidate["audit_id"] == "candidate_01"
        assert len(candidate["members"]) == 3
        assert candidate["members"][0]["steps"]
        assert candidate["review"] == {
            "label": None,
            "action_specific": None,
            "sequence_consistent": None,
            "target_independent": None,
            "outcome_supported": None,
            "future_useful": None,
            "notes": "",
        }
    finally:
        eng.close()
        _cleanup(path)
