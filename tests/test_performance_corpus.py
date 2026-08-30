from __future__ import annotations

import json
from pathlib import Path

from tests.benchmarks.performance_corpus import build_corpus


def _artifact(path: Path, rows: list[dict]) -> Path:
    path.write_text(json.dumps({"results": rows}), encoding="utf-8")
    return path


def test_build_corpus_aligns_sources_and_detects_oracle_gain(tmp_path: Path) -> None:
    locomo_context = _artifact(
        tmp_path / "lc.json",
        [
            {
                "conv_id": "conv-1",
                "question": "Q?",
                "hypothesis": "[EPISODE 1 | date=2024-01-01 00:00Z]\nA",
            }
        ],
    )
    locomo_answers = _artifact(
        tmp_path / "la.json",
        [{"conv_id": "conv-1", "question": "Q?", "expected": "A", "is_correct": False}],
    )
    long_context = _artifact(
        tmp_path / "mc.json",
        [
            {
                "question_id": "q1",
                "question_type": "temporal-reasoning",
                "hypothesis": "dated evidence",
            }
        ],
    )
    slow_answers = _artifact(
        tmp_path / "ma.json",
        [
            {
                "question_id": "q1",
                "question": "When?",
                "expected": "Tuesday",
                "generated_answer": "Monday",
                "is_correct": False,
            }
        ],
    )
    oracle_answers = _artifact(
        tmp_path / "mo.json",
        [
            {
                "question_id": "q1",
                "question": "When?",
                "expected": "Tuesday",
                "generated_answer": "Tuesday",
                "is_correct": True,
            }
        ],
    )

    corpus = build_corpus(
        locomo_context=locomo_context,
        locomo_answers=locomo_answers,
        longmemeval_context=long_context,
        longmemeval_answers=slow_answers,
        longmemeval_oracle_answers=oracle_answers,
    )

    assert corpus["summary"]["rows"] == 2
    assert corpus["summary"]["by_outcome"]["oracle_gain"] == 1
    assert corpus["summary"]["baseline_scores"] == {
        "locomo_slowave_answer_accuracy_pct": 0.0,
        "longmemeval_slowave_answer_accuracy_pct": 0.0,
        "longmemeval_oracle_answer_accuracy_pct": 100.0,
        "longmemeval_oracle_gap_pct_points": 100.0,
    }
    assert corpus["rows"][0]["evidence"]["metrics"]["episode_boundaries"] == 1
    assert "hypothesis" not in corpus["rows"][0]["evidence"]
    assert corpus["rows"][1]["diagnosis"]["status"] == "pending"


def test_identical_oracle_gain_is_flagged_as_possible_judge_variance(tmp_path: Path) -> None:
    empty_locomo = _artifact(tmp_path / "empty.json", [])
    context = _artifact(tmp_path / "context.json", [{"question_id": "q1", "hypothesis": "x"}])
    slow = _artifact(
        tmp_path / "slow.json",
        [{"question_id": "q1", "generated_answer": "The Nightingale", "is_correct": False}],
    )
    oracle = _artifact(
        tmp_path / "oracle.json",
        [{"question_id": "q1", "generated_answer": "The Nightingale!", "is_correct": True}],
    )

    corpus = build_corpus(
        locomo_context=empty_locomo,
        locomo_answers=empty_locomo,
        longmemeval_context=context,
        longmemeval_answers=slow,
        longmemeval_oracle_answers=oracle,
        embed_evidence=True,
    )

    row = corpus["rows"][0]
    assert row["answers_normalize_identically"] is True
    assert row["diagnosis"]["label"] == "judge_variance"
    assert row["evidence"]["hypothesis"] == "x"
