"""Compare baseline structured evidence with Evidence Bundle v1 on one split."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tests.benchmarks.evidence_bundle import (
    assemble_evidence_bundle,
    bundle_metrics,
    parse_structured_evidence,
)
from tests.benchmarks.performance_corpus import ROOT


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def compare_corpus(corpus_path: Path, *, split: str, max_items: int = 8) -> dict[str, Any]:
    corpus = _load(corpus_path)
    artifact_cache: dict[str, dict[str, Any]] = {}
    rows = []
    for row in corpus["rows"]:
        if row["split"] != split:
            continue
        source = row["evidence"]["source"]
        artifact_name = source["artifact"]
        artifact = artifact_cache.setdefault(artifact_name, _load(ROOT / artifact_name))
        baseline = str(artifact["results"][source["result_index"]]["hypothesis"])
        eligible = bool(parse_structured_evidence(baseline))
        bundle = assemble_evidence_bundle(str(row["question"]), baseline, max_items=max_items)
        baseline_metrics = bundle_metrics(baseline, expected=str(row["expected"]))
        bundle_values = bundle_metrics(bundle, expected=str(row["expected"]))
        rows.append(
            {
                "id": row["id"],
                "dataset": row["dataset"],
                "baseline_outcome": row["baseline_outcome"],
                "eligible": eligible,
                "baseline": baseline_metrics,
                "bundle": bundle_values,
                "delta": {
                    key: round(float(bundle_values[key]) - float(baseline_metrics[key]), 6)
                    for key in baseline_metrics
                },
            }
        )

    def mean(field: str, arm: str) -> float:
        return round(sum(float(row[arm][field]) for row in rows) / max(1, len(rows)), 6)

    metrics = [
        "characters",
        "items",
        "dated_items",
        "mean_pairwise_jaccard",
        "expected_token_coverage",
    ]
    return {
        "meta": {
            "format": "slowave_evidence_bundle_comparison_v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "corpus": str(corpus_path),
            "split": split,
            "max_items": max_items,
            "answer_generation_run": False,
        },
        "summary": {
            "rows": len(rows),
            "by_dataset": dict(Counter(row["dataset"] for row in rows)),
            "eligible_rows": sum(row["eligible"] for row in rows),
            "baseline_mean": {field: mean(field, "baseline") for field in metrics},
            "bundle_mean": {field: mean(field, "bundle") for field in metrics},
            "expected_coverage_regressions": sum(
                row["delta"]["expected_token_coverage"] < 0 for row in rows
            ),
        },
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus",
        type=Path,
        default=ROOT / "data/performance_corpora/slowave_evidence_v1.json",
    )
    parser.add_argument(
        "--split", choices=("development", "validation", "holdout"), default="development"
    )
    parser.add_argument("--max-items", type=int, default=8)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data/performance_corpora/evidence_bundle_v1_development.json",
    )
    args = parser.parse_args()
    report = compare_corpus(args.corpus, split=args.split, max_items=args.max_items)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
