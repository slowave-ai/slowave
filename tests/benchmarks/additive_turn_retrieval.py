"""Fixed-budget additive experiment for turn-level retrieval.

The existing hypothesis remains the primary path. A small number of absent
full-query turn candidates are appended while the lowest-ranked baseline
evidence units are removed until both the original item and character budgets
are respected. Expected answers and gold turns are scoring-only inputs.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from slowave.symbolic.encoder import TextEncoder
from tests.benchmarks.performance_corpus import ROOT
from tests.benchmarks.turn_level_retrieval import (
    _load,
    _locomo_cases,
    _longmemeval_cases,
    _selected_ids,
    rank_turns,
)

ADDITION_BUDGETS = (1, 2, 4, 8)
_STRUCTURED_BOUNDARY = re.compile(r"^\[(?:SCHEMA|EPISODE)\s+\d+\s*\|[^\]]+\]\n", re.MULTILINE)
_DATED_BOUNDARY = re.compile(r"\[\d{4}-\d{2}-\d{2}\]")


def split_evidence_units(evidence: str) -> list[str]:
    """Split known evidence formats while preserving any unstructured prefix."""
    matches = list(_STRUCTURED_BOUNDARY.finditer(evidence))
    if not matches:
        matches = list(_DATED_BOUNDARY.finditer(evidence))
    if not matches:
        return [evidence] if evidence else []
    units = []
    prefix = evidence[: matches[0].start()].strip()
    if prefix:
        units.append(prefix)
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(evidence)
        units.append(evidence[match.start() : end].strip())
    return units


def fixed_budget_addition(
    baseline: str,
    ranked_turns: list[tuple[str, str]],
    *,
    additions: int,
) -> tuple[str, dict[str, int]]:
    """Add absent, text-distinct turns and evict tail units to preserve budgets."""
    units = split_evidence_units(baseline)
    original_units = len(units)
    candidates = []
    seen_text = set()
    for _, text in ranked_turns:
        normalized = " ".join(text.split())
        if not normalized or normalized in seen_text or text in baseline:
            continue
        seen_text.add(normalized)
        candidates.append(text)
        if len(candidates) == additions:
            break

    kept = list(units)
    while kept and len(kept) + len(candidates) > original_units:
        kept.pop()
    candidate_chars = sum(len(text) for text in candidates)
    removed_chars = len(baseline) - len("\n\n".join(kept))
    while kept and removed_chars < candidate_chars:
        kept.pop()
        removed_chars = len(baseline) - len("\n\n".join(kept))
    combined = "\n\n".join(kept + candidates)
    return combined, {
        "added_turns": len(candidates),
        "evicted_units": original_units - len(kept),
        "baseline_units": original_units,
    }


def _coverage(evidence: str, gold_texts: list[str]) -> dict[str, float | bool]:
    recovered = [text in evidence for text in gold_texts]
    return {
        "any_gold": any(recovered),
        "all_gold": bool(recovered) and all(recovered),
        "gold_recall": sum(recovered) / max(1, len(recovered)),
    }


def _baseline_maps(locomo_context: Path, longmemeval_context: Path):
    locomo = {
        (str(row["conv_id"]), str(row["question"])): row for row in _load(locomo_context)["results"]
    }
    longmemeval = {str(row["question_id"]): row for row in _load(longmemeval_context)["results"]}
    return locomo, longmemeval


def run_experiment(
    *,
    corpus_path: Path,
    locomo_path: Path,
    longmemeval_path: Path,
    locomo_context: Path,
    longmemeval_context: Path,
    budgets: tuple[int, ...] = ADDITION_BUDGETS,
    encoder: TextEncoder | None = None,
    capture_hypotheses: bool = False,
    split: str = "development",
) -> dict[str, Any]:
    selected_locomo, selected_longmemeval = _selected_ids(corpus_path, split)
    cases = list(_locomo_cases(_load(locomo_path), selected_locomo))
    cases += list(_longmemeval_cases(_load(longmemeval_path), selected_longmemeval))
    locomo_baselines, longmemeval_baselines = _baseline_maps(locomo_context, longmemeval_context)
    encoder = encoder or TextEncoder()
    vector_cache = {}
    rows = []
    started = time.perf_counter()

    for case_id, dataset, question, turns, gold_ids in cases:
        cache_key = tuple(turns)
        if cache_key not in vector_cache:
            vector_cache[cache_key] = encoder.encode_many([text for _, text in turns])
        query_vector = encoder.encode(question)
        ranking = rank_turns(query_vector, vector_cache[cache_key])
        ranked_turns = [turns[int(index)] for index in ranking]
        turn_map = dict(turns)
        gold_texts = [turn_map[key] for key in gold_ids]
        if dataset == "locomo":
            _, conv_id, _ = case_id.split(":", 2)
            source_row = locomo_baselines[(conv_id, question)]
        else:
            source_row = longmemeval_baselines[case_id]
        baseline = str(source_row["hypothesis"])
        baseline_coverage = _coverage(baseline, gold_texts)
        arms = {}
        for budget in budgets:
            evidence, assembly = fixed_budget_addition(baseline, ranked_turns, additions=budget)
            arms[str(budget)] = {
                **_coverage(evidence, gold_texts),
                **assembly,
                "characters": len(evidence),
                "character_delta": len(evidence) - len(baseline),
            }
            selected_candidate = next(
                (
                    (int(index), text)
                    for index in ranking
                    for _, text in [turns[int(index)]]
                    if text not in baseline
                ),
                None,
            )
            if selected_candidate is not None:
                index, _ = selected_candidate
                arms[str(budget)]["first_candidate_query_cosine"] = round(
                    float(vector_cache[cache_key][index] @ query_vector), 6
                )
            if capture_hypotheses:
                arms[str(budget)]["hypothesis"] = evidence
        baseline_result = {**baseline_coverage, "characters": len(baseline)}
        if capture_hypotheses:
            baseline_result["hypothesis"] = baseline
        rows.append(
            {
                "id": case_id,
                "dataset": dataset,
                "question": question,
                "expected": source_row.get("expected", source_row.get("expected_answer")),
                "category": source_row.get("category"),
                "baseline": baseline_result,
                "arms": arms,
            }
        )

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    groups["all"] = rows
    for row in rows:
        groups[row["dataset"]].append(row)

    summary = {}
    for name, subset in groups.items():
        baseline_all = sum(row["baseline"]["all_gold"] for row in subset)
        result = {
            "rows": len(subset),
            "baseline": {
                "any_gold_pct": round(
                    100 * sum(row["baseline"]["any_gold"] for row in subset) / max(1, len(subset)),
                    2,
                ),
                "all_gold_pct": round(100 * baseline_all / max(1, len(subset)), 2),
            },
        }
        for budget in budgets:
            key = str(budget)
            result[key] = {
                "any_gold_pct": round(
                    100 * sum(row["arms"][key]["any_gold"] for row in subset) / max(1, len(subset)),
                    2,
                ),
                "all_gold_pct": round(
                    100 * sum(row["arms"][key]["all_gold"] for row in subset) / max(1, len(subset)),
                    2,
                ),
                "wrong_to_complete": sum(
                    not row["baseline"]["all_gold"] and row["arms"][key]["all_gold"]
                    for row in subset
                ),
                "complete_to_wrong": sum(
                    row["baseline"]["all_gold"] and not row["arms"][key]["all_gold"]
                    for row in subset
                ),
                "mean_character_delta": round(
                    sum(row["arms"][key]["character_delta"] for row in subset)
                    / max(1, len(subset)),
                    2,
                ),
                "mean_evicted_units": round(
                    sum(row["arms"][key]["evicted_units"] for row in subset) / max(1, len(subset)),
                    3,
                ),
            }
        summary[name] = result

    return {
        "meta": {
            "format": "additive_turn_retrieval_fixed_budget_v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "split": split,
            "budgets": list(budgets),
            "query_method": "single full-query multilingual embedding",
            "language_specific_processing": False,
            "captured_hypotheses": capture_hypotheses,
            "elapsed_s": round(time.perf_counter() - started, 3),
        },
        "summary": summary,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus", type=Path, default=ROOT / "data/performance_corpora/slowave_evidence_v1.json"
    )
    parser.add_argument(
        "--split", choices=("development", "validation", "holdout"), default="development"
    )
    parser.add_argument("--locomo", type=Path, default=ROOT / "data/locomo/locomo10.json")
    parser.add_argument(
        "--longmemeval", type=Path, default=ROOT / "data/longmemeval/longmemeval_oracle.json"
    )
    parser.add_argument(
        "--locomo-context",
        type=Path,
        default=ROOT / "data/locomo/runs/locomo_full_context_structured_v1.json",
    )
    parser.add_argument(
        "--longmemeval-context",
        type=Path,
        default=ROOT
        / "data/longmemeval/runs/longmemeval_temporal_corrected_episodes_full_context.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data/performance_corpora/additive_turn_retrieval_development.json",
    )
    parser.add_argument(
        "--budgets",
        type=int,
        nargs="+",
        default=list(ADDITION_BUDGETS),
        help="Turn-addition arms to evaluate",
    )
    parser.add_argument(
        "--capture-hypotheses",
        action="store_true",
        help="Persist full baseline and arm evidence for downstream AML evaluation",
    )
    args = parser.parse_args()
    report = run_experiment(
        corpus_path=args.corpus,
        locomo_path=args.locomo,
        longmemeval_path=args.longmemeval,
        locomo_context=args.locomo_context,
        longmemeval_context=args.longmemeval_context,
        budgets=tuple(args.budgets),
        capture_hypotheses=args.capture_hypotheses,
        split=args.split,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
