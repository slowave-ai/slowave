"""Full-query, turn-level retrieval depth experiment.

Uses only the multilingual full-query embedding and source turn boundaries. No
query parsing, cue extraction, answer text, or language-specific rules enter
retrieval. Gold evidence is used only after ranking to compute offline metrics.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from slowave.symbolic.encoder import TextEncoder
from tests.benchmarks.performance_corpus import ROOT

DEPTHS = (20, 50, 100, 200)


def rank_turns(query: np.ndarray, turns: np.ndarray) -> np.ndarray:
    """Return descending cosine rank for already normalized embeddings."""
    if turns.size == 0:
        return np.empty(0, dtype=np.int64)
    return np.argsort(-(turns @ query), kind="stable")


def depth_metrics(ranking: list[str], gold: set[str], depths: tuple[int, ...]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    first = next((i for i, key in enumerate(ranking, start=1) if key in gold), None)
    for depth in depths:
        admitted = ranking[:depth]
        recovered = gold.intersection(admitted)
        output[str(depth)] = {
            "any_gold": bool(recovered),
            "all_gold": bool(gold) and recovered == gold,
            "gold_recall": len(recovered) / max(1, len(gold)),
            "gold_precision": len(recovered) / max(1, len(admitted)),
        }
    output["first_gold_rank"] = first
    return output


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _selected_ids(
    corpus_path: Path, split: str = "development"
) -> tuple[set[tuple[str, str]], set[str]]:
    corpus = _load(corpus_path)
    locomo = {
        (str(row["group_id"]), str(row["question"]))
        for row in corpus["rows"]
        if row["dataset"] == "locomo" and row["split"] == split
    }
    longmemeval = {
        str(row["group_id"])
        for row in corpus["rows"]
        if row["dataset"] == "longmemeval" and row["split"] == split
    }
    return locomo, longmemeval


def _locomo_cases(dataset: list[dict[str, Any]], selected: set[tuple[str, str]]):
    for sample in dataset:
        conv_id = str(sample["sample_id"])
        turns = []
        for key, value in sample["conversation"].items():
            if (
                not key.startswith("session_")
                or key.endswith("_date_time")
                or not isinstance(value, list)
            ):
                continue
            for turn in value:
                turns.append((str(turn["dia_id"]), str(turn["text"])))
        for qa in sample["qa"]:
            question = str(qa["question"])
            if (conv_id, question) not in selected:
                continue
            gold = {
                ref.strip()
                for value in qa.get("evidence", [])
                for ref in str(value).split(";")
                if ref.strip()
            }
            available = {key for key, _ in turns}
            gold &= available
            if gold:
                yield f"locomo:{conv_id}:{question}", "locomo", question, turns, gold


def _longmemeval_cases(dataset: list[dict[str, Any]], selected: set[str]):
    for item in dataset:
        qid = str(item["question_id"])
        if qid not in selected:
            continue
        turns = []
        gold = set()
        for session_id, session in zip(item["haystack_session_ids"], item["haystack_sessions"]):
            for index, turn in enumerate(session):
                key = f"{session_id}:{index}"
                turns.append((key, str(turn["content"])))
                if turn.get("has_answer"):
                    gold.add(key)
        if gold:
            yield qid, "longmemeval", str(item["question"]), turns, gold


def run_experiment(
    *,
    corpus_path: Path,
    locomo_path: Path,
    longmemeval_path: Path,
    depths: tuple[int, ...] = DEPTHS,
    encoder: TextEncoder | None = None,
    split: str = "development",
) -> dict[str, Any]:
    selected_locomo, selected_longmemeval = _selected_ids(corpus_path, split)
    cases = list(_locomo_cases(_load(locomo_path), selected_locomo))
    cases += list(_longmemeval_cases(_load(longmemeval_path), selected_longmemeval))
    encoder = encoder or TextEncoder()
    rows = []
    started = time.perf_counter()

    # LoCoMo shares one turn collection across many questions in a conversation.
    cache: dict[tuple[tuple[str, str], ...], np.ndarray] = {}
    for case_id, dataset, question, turns, gold in cases:
        cache_key = tuple(turns)
        turn_started = time.perf_counter()
        if cache_key not in cache:
            cache[cache_key] = encoder.encode_many([text for _, text in turns])
        turn_vectors = cache[cache_key]
        query = encoder.encode(question)
        ranked_indices = rank_turns(query, turn_vectors)
        ranking = [turns[int(index)][0] for index in ranked_indices]
        metrics = depth_metrics(ranking, gold, depths)
        rows.append(
            {
                "id": case_id,
                "dataset": dataset,
                "turns": len(turns),
                "gold_turns": len(gold),
                "latency_s": round(time.perf_counter() - turn_started, 6),
                "metrics": metrics,
            }
        )

    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_dataset[row["dataset"]].append(row)

    def aggregate(subset: list[dict[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {"rows": len(subset)}
        for depth in depths:
            key = str(depth)
            result[key] = {
                "any_gold_pct": round(
                    100
                    * sum(row["metrics"][key]["any_gold"] for row in subset)
                    / max(1, len(subset)),
                    2,
                ),
                "all_gold_pct": round(
                    100
                    * sum(row["metrics"][key]["all_gold"] for row in subset)
                    / max(1, len(subset)),
                    2,
                ),
                "mean_gold_recall": round(
                    sum(row["metrics"][key]["gold_recall"] for row in subset) / max(1, len(subset)),
                    4,
                ),
                "mean_gold_precision": round(
                    sum(row["metrics"][key]["gold_precision"] for row in subset)
                    / max(1, len(subset)),
                    6,
                ),
            }
        return result

    summary = {"all": aggregate(rows)}
    summary.update({name: aggregate(subset) for name, subset in sorted(by_dataset.items())})
    return {
        "meta": {
            "format": "turn_level_full_query_depth_v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "split": split,
            "depths": list(depths),
            "query_method": "single full-query multilingual embedding",
            "language_specific_processing": False,
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
        "--longmemeval",
        type=Path,
        default=ROOT / "data/longmemeval/longmemeval_oracle.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data/performance_corpora/turn_level_full_query_development.json",
    )
    args = parser.parse_args()
    report = run_experiment(
        corpus_path=args.corpus,
        locomo_path=args.locomo,
        longmemeval_path=args.longmemeval,
        split=args.split,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
