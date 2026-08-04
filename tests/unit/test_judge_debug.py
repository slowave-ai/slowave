"""Unit tests for the opt-in judge instrumentation hook (slowave/core/judge_debug.py).

Confirms the no-op default (zero cost, no file touched) and the enabled
behavior (correct JSONL append) — this is diagnostic-only tooling, not
production logic, so these tests focus on the on/off contract rather than
any judge decision semantics.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from slowave.core import judge_debug


def test_emit_judge_signal_is_noop_by_default(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    log_path = tmp_path / "judge_debug.jsonl"
    monkeypatch.delenv("SLOWAVE_DEBUG_JUDGE_PAIRS", raising=False)
    monkeypatch.setenv("SLOWAVE_DEBUG_JUDGE_LOG_PATH", str(log_path))

    judge_debug.emit_judge_signal({"code_path": "test", "verdict": "unrelated"})

    assert not log_path.exists()


def test_emit_judge_signal_appends_jsonl_when_enabled(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    log_path = tmp_path / "nested" / "judge_debug.jsonl"
    monkeypatch.setenv("SLOWAVE_DEBUG_JUDGE_PAIRS", "1")
    monkeypatch.setenv("SLOWAVE_DEBUG_JUDGE_LOG_PATH", str(log_path))

    judge_debug.emit_judge_signal(
        {"code_path": "consolidation_judge", "verdict": "supersedes", "cos": 0.9}
    )
    judge_debug.emit_judge_signal(
        {"code_path": "remember_extended_range", "verdict": "no_action", "cos": 0.8}
    )

    assert log_path.exists()
    lines = log_path.read_text().strip().splitlines()
    assert len(lines) == 2

    first = json.loads(lines[0])
    assert first["code_path"] == "consolidation_judge"
    assert first["verdict"] == "supersedes"
    assert first["cos"] == 0.9
    assert "ts" in first

    second = json.loads(lines[1])
    assert second["code_path"] == "remember_extended_range"
    assert second["verdict"] == "no_action"


def test_emit_judge_signal_never_raises_on_bad_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SLOWAVE_DEBUG_JUDGE_PAIRS", "1")
    # A path that cannot possibly be created (nested under a file, not a dir).
    monkeypatch.setenv("SLOWAVE_DEBUG_JUDGE_LOG_PATH", "/dev/null/impossible/path.jsonl")

    # Must not raise -- this is diagnostic-only and must never break the judge.
    judge_debug.emit_judge_signal({"code_path": "test", "verdict": "unrelated"})
