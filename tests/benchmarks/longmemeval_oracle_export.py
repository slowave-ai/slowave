#!/usr/bin/env python3
"""Export LongMemEval gold answer sessions for an AML answer-model ceiling test."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]


def format_oracle_context(question: dict[str, Any]) -> str:
    """Render complete annotated answer sessions with source dates."""
    answer_ids = set(question["answer_session_ids"])
    sections: list[str] = ["=== ORACLE ANSWER SESSIONS ==="]
    matched: set[str] = set()
    for session_id, session_date, turns in zip(
        question["haystack_session_ids"],
        question["haystack_dates"],
        question["haystack_sessions"],
        strict=True,
    ):
        if session_id not in answer_ids:
            continue
        matched.add(session_id)
        lines = [f"[SESSION {session_id} | date={session_date}]"]
        lines.extend(
            f"{str(turn.get('role', 'unknown')).upper()}: "
            f"{str(turn.get('content', '')).strip()}"
            for turn in turns
        )
        sections.append("\n".join(lines))
    missing = answer_ids - matched
    if missing:
        raise ValueError(
            f"answer_session_ids missing from haystack for {question['question_id']}: "
            f"{sorted(missing)}"
        )
    return "\n\n".join(sections)


def build_payload(questions: list[dict[str, Any]], dataset: Path, category: str) -> dict[str, Any]:
    selected = [q for q in questions if q["question_type"] == category]
    return {
        "meta": {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "dataset": str(dataset),
            "categories": [category],
            "context_source": "longmemeval_answer_session_ids",
            "oracle_context": True,
            "full_hypotheses": True,
            "evidence_format": "structured_v1",
            "top_k": None,
        },
        "summary": {"n": len(selected)},
        "results": [
            {
                "question_id": q["question_id"],
                "question_type": q["question_type"],
                "question": q["question"],
                "expected_answer": q["answer"],
                "hypothesis": format_oracle_context(q),
                "error": None,
            }
            for q in selected
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export LongMemEval annotated answer sessions for AML-style evaluation"
    )
    parser.add_argument("--dataset", default="data/longmemeval/longmemeval_oracle.json")
    parser.add_argument("--category", default="temporal-reasoning")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    dataset = Path(args.dataset)
    if not dataset.is_absolute():
        dataset = REPO_ROOT / dataset
    output = Path(args.out)
    if not output.is_absolute():
        output = REPO_ROOT / output
    questions = json.loads(dataset.read_text(encoding="utf-8"))
    payload = build_payload(questions, dataset, args.category)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Exported {payload['summary']['n']} oracle questions to: {output}")


if __name__ == "__main__":
    main()
