"""Analyze answer-blind signals for additive turn candidate utility.

This is a development-only proof experiment. It labels add-1 outcomes from the
paired AML run, but computes features solely from the question, baseline
evidence, retained evidence, added turn, and embedding geometry.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from slowave.symbolic.encoder import TextEncoder
from tests.benchmarks.additive_turn_retrieval import split_evidence_units
from tests.benchmarks.performance_corpus import ROOT


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    return float(left @ right)


def candidate_parts(baseline: str, additive: str) -> tuple[str, list[str], list[str]]:
    """Recover the appended candidate and retained/evicted baseline units."""
    baseline_units = split_evidence_units(baseline)
    for kept_count in range(len(baseline_units), -1, -1):
        retained = baseline_units[:kept_count]
        prefix = "\n\n".join(retained)
        boundary = f"{prefix}\n\n" if prefix else ""
        if additive.startswith(boundary):
            candidate = additive[len(boundary) :]
            if candidate:
                return candidate, retained, baseline_units[kept_count:]
    raise ValueError("additive evidence does not preserve a baseline-unit prefix")


def _auc(values: list[float], labels: list[bool]) -> float | None:
    positives = [value for value, label in zip(values, labels) if label]
    negatives = [value for value, label in zip(values, labels) if not label]
    if not positives or not negatives:
        return None
    wins = sum(p > n for p in positives for n in negatives)
    ties = sum(p == n for p in positives for n in negatives)
    return (wins + 0.5 * ties) / (len(positives) * len(negatives))


def run_analysis(
    paired_context: Path,
    baseline_aml: Path,
    additive_aml: Path,
    *,
    encoder: TextEncoder | None = None,
) -> dict[str, Any]:
    contexts = _load(paired_context)["rows"]
    baseline_results = _load(baseline_aml)["results"]
    additive_results = _load(additive_aml)["results"]
    if not (len(contexts) == len(baseline_results) == len(additive_results)):
        raise ValueError("paired artifacts have different row counts")
    encoder = encoder or TextEncoder()
    rows = []
    counts = {"gain": 0, "loss": 0, "neutral": 0}
    for context, baseline_result, additive_result in zip(
        contexts, baseline_results, additive_results
    ):
        if (
            context["question"] != baseline_result["question"]
            or context["question"] != additive_result["question"]
        ):
            raise ValueError("paired artifact row order differs")
        label = (
            "gain"
            if not baseline_result["is_correct"] and additive_result["is_correct"]
            else (
                "loss"
                if baseline_result["is_correct"] and not additive_result["is_correct"]
                else "neutral"
            )
        )
        counts[label] += 1
        if label == "neutral":
            continue
        baseline = context["baseline"]["hypothesis"]
        additive = context["arms"]["1"]["hypothesis"]
        candidate, retained, evicted = candidate_parts(baseline, additive)
        texts = [context["question"], candidate, *retained, *evicted]
        vectors = encoder.encode_many(texts)
        query, candidate_vector = vectors[:2]
        retained_vectors = vectors[2 : 2 + len(retained)]
        evicted_vectors = vectors[2 + len(retained) :]
        candidate_query = _cosine(candidate_vector, query)
        candidate_retained_max = max(
            (_cosine(candidate_vector, vector) for vector in retained_vectors), default=0.0
        )
        evicted_query_max = max((_cosine(query, vector) for vector in evicted_vectors), default=0.0)
        rows.append(
            {
                "id": context["id"],
                "dataset": context["dataset"],
                "label": label,
                "features": {
                    "candidate_query_cosine": round(candidate_query, 6),
                    "candidate_retained_max_cosine": round(candidate_retained_max, 6),
                    "candidate_novelty": round(1.0 - candidate_retained_max, 6),
                    "query_replacement_margin": round(candidate_query - evicted_query_max, 6),
                    "candidate_character_ratio": round(len(candidate) / max(1, len(baseline)), 6),
                    "evicted_units": len(evicted),
                },
            }
        )

    flips = rows
    feature_names = list(rows[0]["features"])
    signals = {}
    for feature in feature_names:
        values = [float(row["features"][feature]) for row in flips]
        labels = [row["label"] == "gain" for row in flips]
        auc = _auc(values, labels)
        signals[feature] = {
            "gain_mean": round(float(np.mean([v for v, label in zip(values, labels) if label])), 6),
            "loss_mean": round(
                float(np.mean([v for v, label in zip(values, labels) if not label])), 6
            ),
            "auc": round(auc, 4) if auc is not None else None,
            "direction_free_auc": round(max(auc, 1.0 - auc), 4) if auc is not None else None,
        }
    return {
        "meta": {
            "format": "additive_candidate_utility_analysis_v1",
            "split": "development",
            "language_specific_processing": False,
            "labels_used_for": "post-hoc signal evaluation only",
        },
        "counts": counts,
        "signals": signals,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--contexts",
        type=Path,
        default=ROOT / "data/performance_corpora/additive_turn_retrieval_paired_development.json",
    )
    parser.add_argument(
        "--baseline-aml",
        type=Path,
        default=ROOT
        / "data/performance_corpora/additive_turn_aml_arms/baseline_aml_gpt4omini.json",
    )
    parser.add_argument(
        "--additive-aml",
        type=Path,
        default=ROOT / "data/performance_corpora/additive_turn_aml_arms/1_aml_gpt4omini.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data/performance_corpora/additive_candidate_utility_development.json",
    )
    args = parser.parse_args()
    result = run_analysis(args.contexts, args.baseline_aml, args.additive_aml)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"counts": result["counts"], "signals": result["signals"]}, indent=2))


if __name__ == "__main__":
    main()
