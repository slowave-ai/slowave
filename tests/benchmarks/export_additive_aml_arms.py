"""Export paired additive-turn arms as AML answer-evaluation inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from tests.benchmarks.performance_corpus import ROOT


def arm_payload(source: dict[str, Any], arm: str) -> dict[str, Any]:
    """Return one AML-compatible artifact while preserving paired row order."""
    if not source.get("meta", {}).get("captured_hypotheses"):
        raise ValueError("source artifact does not contain captured hypotheses")
    results = []
    for row in source["rows"]:
        if arm.startswith("gated-"):
            source_arm = arm.removeprefix("gated-")
            candidate = row["arms"][source_arm]
            evidence = (
                candidate
                if candidate.get("first_candidate_query_cosine", -1.0) >= 0.5
                else row["baseline"]
            )
        else:
            evidence = row["baseline"] if arm == "baseline" else row["arms"][arm]
        result = {
            "id": row["id"],
            "question": row["question"],
            "expected": row["expected"],
            "hypothesis": evidence["hypothesis"],
            "category": row.get("category"),
        }
        if row["dataset"] == "locomo":
            result["conv_id"] = row["id"].split(":", 2)[1]
        else:
            result["question_id"] = row["id"]
        results.append(result)
    return {
        "meta": {
            "format": "additive_turn_aml_arm_v1",
            "source_format": source["meta"]["format"],
            "arm": arm,
            "split": source["meta"]["split"],
            "evidence_format": "structured_v1",
            "full_hypotheses": True,
            "query_method": source["meta"]["query_method"],
            "language_specific_processing": False,
            "admission_rule": (
                "first_candidate_query_cosine >= 0.5" if arm.startswith("gated-") else None
            ),
        },
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "data/performance_corpora/additive_turn_retrieval_paired_development.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data/performance_corpora/additive_turn_aml_arms",
    )
    parser.add_argument("--arms", nargs="+", default=["baseline", "1", "4"])
    args = parser.parse_args()
    source = json.loads(args.input.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for arm in args.arms:
        payload = arm_payload(source, arm)
        output = args.output_dir / f"{arm}.json"
        output.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(f"wrote {output} ({len(payload['results'])} rows)")


if __name__ == "__main__":
    main()
