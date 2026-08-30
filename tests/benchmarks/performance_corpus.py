"""Build a leakage-resistant corpus for measuring Slowave evidence quality.

The corpus stores references and hashes by default instead of copying very large
hypotheses. Pass ``--embed-evidence`` when a self-contained review artifact is
needed. Splits are assigned by conversation/question group, never by row.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "data" / "performance_corpora" / "slowave_evidence_v1.json"

DIAGNOSIS_LABELS = [
    "missing_event",
    "incomplete_multi_event_coverage",
    "wrong_ordering",
    "entity_confusion",
    "distractor_overload",
    "answer_reasoning",
    "judge_variance",
    "out_of_scope",
]


def _load(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
        raise ValueError(f"expected an artifact with a results list: {path}")
    return payload


def _split(group: str) -> str:
    bucket = int.from_bytes(hashlib.sha256(group.encode()).digest()[:4], "big") % 100
    if bucket < 50:
        return "development"
    if bucket < 75:
        return "validation"
    return "holdout"


def _normalized(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.lower()))


def _evidence_metrics(hypothesis: str) -> dict[str, Any]:
    dates = re.findall(r"\b\d{4}-\d{2}-\d{2}\b", hypothesis)
    return {
        "sha256": hashlib.sha256(hypothesis.encode()).hexdigest(),
        "characters": len(hypothesis),
        "schema_boundaries": hypothesis.count("[SCHEMA "),
        "episode_boundaries": hypothesis.count("[EPISODE "),
        "dated_boundaries": len(dates),
        "distinct_dates": len(set(dates)),
    }


def _source_ref(path: Path, index: int) -> dict[str, Any]:
    try:
        display = str(path.resolve().relative_to(ROOT))
    except ValueError:
        display = str(path.resolve())
    return {"artifact": display, "result_index": index}


def _answer_view(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "answer": row.get("generated_answer", ""),
        "correct": bool(row.get("is_correct")),
        "label": row.get("label"),
        "reason": row.get("reason") or row.get("judge_response", ""),
        "parse_ok": row.get("parse_ok", True),
    }


def _locomo_rows(
    context_path: Path, answer_path: Path, *, embed_evidence: bool
) -> list[dict[str, Any]]:
    contexts = _load(context_path)["results"]
    answers = _load(answer_path)["results"]
    context_index = {
        (str(row.get("conv_id")), str(row.get("question"))): (i, row)
        for i, row in enumerate(contexts)
    }
    output = []
    for answer_index, answer in enumerate(answers):
        key = (str(answer.get("conv_id")), str(answer.get("question")))
        if key not in context_index:
            raise ValueError(f"LoCoMo answer row has no context row: {key}")
        context_i, context = context_index[key]
        evidence = str(context.get("hypothesis", ""))
        row = {
            "id": f"locomo:{key[0]}:{answer_index}",
            "dataset": "locomo",
            "group_id": key[0],
            "split": _split(f"locomo:{key[0]}"),
            "category": answer.get("category"),
            "question": answer.get("question"),
            "expected": answer.get("expected"),
            "baseline_outcome": "correct" if answer.get("is_correct") else "wrong",
            "slowave_answer": _answer_view(answer),
            "oracle_answer": None,
            "evidence": {
                "source": _source_ref(context_path, context_i),
                "metrics": _evidence_metrics(evidence),
            },
            "answer_source": _source_ref(answer_path, answer_index),
            "diagnosis": {"status": "pending", "label": None, "notes": ""},
        }
        if embed_evidence:
            row["evidence"]["hypothesis"] = evidence
        output.append(row)
    return output


def _longmemeval_rows(
    context_path: Path,
    slowave_path: Path,
    oracle_path: Path,
    *,
    embed_evidence: bool,
) -> list[dict[str, Any]]:
    contexts = _load(context_path)["results"]
    slowave = _load(slowave_path)["results"]
    oracle = _load(oracle_path)["results"]
    context_index = {str(row["question_id"]): (i, row) for i, row in enumerate(contexts)}
    oracle_index = {str(row["question_id"]): (i, row) for i, row in enumerate(oracle)}
    output = []
    for slowave_i, answer in enumerate(slowave):
        qid = str(answer["question_id"])
        if qid not in context_index or qid not in oracle_index:
            raise ValueError(f"LongMemEval row is not aligned across artifacts: {qid}")
        context_i, context = context_index[qid]
        oracle_i, oracle_answer = oracle_index[qid]
        evidence = str(context.get("hypothesis", ""))
        slow_ok = bool(answer.get("is_correct"))
        oracle_ok = bool(oracle_answer.get("is_correct"))
        if slow_ok and oracle_ok:
            outcome = "both_correct"
        elif not slow_ok and oracle_ok:
            outcome = "oracle_gain"
        elif slow_ok and not oracle_ok:
            outcome = "oracle_regression"
        else:
            outcome = "both_wrong"
        equivalent = _normalized(str(answer.get("generated_answer", ""))) == _normalized(
            str(oracle_answer.get("generated_answer", ""))
        )
        diagnosis = {"status": "pending", "label": None, "notes": ""}
        if outcome == "oracle_gain" and equivalent:
            diagnosis = {
                "status": "suggested",
                "label": "judge_variance",
                "notes": "Slowave and oracle generated answers normalize identically; verify manually.",
            }
        row = {
            "id": f"longmemeval:{qid}",
            "dataset": "longmemeval",
            "group_id": qid,
            "split": _split(f"longmemeval:{qid}"),
            "category": context.get("question_type", "temporal-reasoning"),
            "question": answer.get("question"),
            "expected": answer.get("expected"),
            "baseline_outcome": outcome,
            "slowave_answer": _answer_view(answer),
            "oracle_answer": _answer_view(oracle_answer),
            "answers_normalize_identically": equivalent,
            "evidence": {
                "source": _source_ref(context_path, context_i),
                "metrics": _evidence_metrics(evidence),
            },
            "answer_source": _source_ref(slowave_path, slowave_i),
            "oracle_answer_source": _source_ref(oracle_path, oracle_i),
            "diagnosis": diagnosis,
        }
        if embed_evidence:
            row["evidence"]["hypothesis"] = evidence
        output.append(row)
    return output


def build_corpus(
    *,
    locomo_context: Path,
    locomo_answers: Path,
    longmemeval_context: Path,
    longmemeval_answers: Path,
    longmemeval_oracle_answers: Path,
    embed_evidence: bool = False,
) -> dict[str, Any]:
    rows = _locomo_rows(locomo_context, locomo_answers, embed_evidence=embed_evidence)
    rows += _longmemeval_rows(
        longmemeval_context,
        longmemeval_answers,
        longmemeval_oracle_answers,
        embed_evidence=embed_evidence,
    )
    locomo = [row for row in rows if row["dataset"] == "locomo"]
    longmemeval = [row for row in rows if row["dataset"] == "longmemeval"]
    locomo_correct = sum(row["slowave_answer"]["correct"] for row in locomo)
    longmemeval_correct = sum(row["slowave_answer"]["correct"] for row in longmemeval)
    oracle_correct = sum(row["oracle_answer"]["correct"] for row in longmemeval)
    return {
        "meta": {
            "format": "slowave_performance_corpus_v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "evidence_embedded": embed_evidence,
            "split_policy": "sha256 group split: development 50%, validation 25%, holdout 25%",
            "diagnosis_labels": DIAGNOSIS_LABELS,
        },
        "summary": {
            "rows": len(rows),
            "by_dataset": dict(Counter(row["dataset"] for row in rows)),
            "by_split": dict(Counter(row["split"] for row in rows)),
            "by_outcome": dict(Counter(row["baseline_outcome"] for row in rows)),
            "baseline_scores": {
                "locomo_slowave_answer_accuracy_pct": round(
                    100 * locomo_correct / max(1, len(locomo)), 2
                ),
                "longmemeval_slowave_answer_accuracy_pct": round(
                    100 * longmemeval_correct / max(1, len(longmemeval)), 2
                ),
                "longmemeval_oracle_answer_accuracy_pct": round(
                    100 * oracle_correct / max(1, len(longmemeval)), 2
                ),
                "longmemeval_oracle_gap_pct_points": round(
                    100 * (oracle_correct - longmemeval_correct) / max(1, len(longmemeval)),
                    2,
                ),
            },
        },
        "rows": rows,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--locomo-context",
        type=Path,
        default=ROOT / "data/locomo/runs/locomo_full_context.json",
    )
    parser.add_argument(
        "--locomo-answers",
        type=Path,
        default=ROOT / "data/locomo/runs/locomo_aml_gpt4omini.json",
    )
    parser.add_argument(
        "--longmemeval-context",
        type=Path,
        default=ROOT
        / "data/longmemeval/runs/longmemeval_temporal_corrected_episodes_full_context.json",
    )
    parser.add_argument(
        "--longmemeval-answers",
        type=Path,
        default=ROOT
        / "data/longmemeval/runs/longmemeval_temporal_corrected_episodes_codex_answer_codex_judge.json",
    )
    parser.add_argument(
        "--longmemeval-oracle-answers",
        type=Path,
        default=ROOT
        / "data/longmemeval/runs/longmemeval_temporal_oracle_codex_answer_codex_judge.json",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--embed-evidence", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    corpus = build_corpus(
        locomo_context=args.locomo_context,
        locomo_answers=args.locomo_answers,
        longmemeval_context=args.longmemeval_context,
        longmemeval_answers=args.longmemeval_answers,
        longmemeval_oracle_answers=args.longmemeval_oracle_answers,
        embed_evidence=args.embed_evidence,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(corpus, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(corpus["summary"], indent=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
