"""Retrieve complete source sessions and assemble them in source chronology.

The experiment ranks raw turns with one multilingual full-query embedding,
scores each source session by its best turn, selects complete sessions under the
existing evidence character budget, then renders selected sessions in source
chronology. Gold annotations are used only after selection for diagnostics.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from slowave.symbolic.encoder import TextEncoder
from tests.benchmarks.performance_corpus import ROOT
from tests.benchmarks.turn_level_retrieval import _load, _selected_ids


@dataclass(frozen=True)
class EventGroup:
    key: str
    date: str
    order: tuple[int, float | int]
    turns: tuple[tuple[str, str], ...]

    def render(self) -> str:
        body = "\n".join(text for _, text in self.turns)
        return f"[EVENT_GROUP | date={self.date}]\n{body}"


def source_date_order(value: str, fallback: int) -> tuple[int, float | int]:
    """Return a language-neutral chronological key for benchmark source dates."""
    for pattern in ("%Y/%m/%d (%a) %H:%M", "%I:%M %p on %d %B, %Y"):
        try:
            return 0, datetime.strptime(value, pattern).timestamp()
        except ValueError:
            continue
    return 1, fallback


def select_groups(
    groups: list[EventGroup],
    turn_vectors: np.ndarray,
    query_vector: np.ndarray,
    character_budget: int,
) -> tuple[str, list[str]]:
    """Select whole groups by best-turn similarity and render chronologically."""
    scores = turn_vectors @ query_vector
    offset = 0
    ranked = []
    for group in groups:
        end = offset + len(group.turns)
        ranked.append((float(np.max(scores[offset:end])), group))
        offset = end
    ranked.sort(key=lambda item: (-item[0], item[1].order))

    selected = []
    used = 0
    for _, group in ranked:
        rendered = group.render()
        separator = 2 if selected else 0
        if used + separator + len(rendered) > character_budget:
            continue
        selected.append(group)
        used += separator + len(rendered)
    selected.sort(key=lambda group: group.order)
    return "\n\n".join(group.render() for group in selected), [group.key for group in selected]


def _locomo_cases(dataset: list[dict[str, Any]], selected: set[tuple[str, str]]):
    for sample in dataset:
        conv_id = str(sample["sample_id"])
        groups = []
        for key, value in sample["conversation"].items():
            if (
                not key.startswith("session_")
                or key.endswith("_date_time")
                or not isinstance(value, list)
            ):
                continue
            fallback_order = int(key.removeprefix("session_"))
            date = str(sample["conversation"].get(f"{key}_date_time", "unknown"))
            groups.append(
                EventGroup(
                    key=key,
                    date=date,
                    order=source_date_order(date, fallback_order),
                    turns=tuple((str(turn["dia_id"]), str(turn["text"])) for turn in value),
                )
            )
        groups.sort(key=lambda group: group.order)
        turn_to_group = {turn_id: group.key for group in groups for turn_id, _ in group.turns}
        for qa in sample["qa"]:
            question = str(qa["question"])
            if (conv_id, question) not in selected:
                continue
            gold_turns = {
                ref.strip()
                for value in qa.get("evidence", [])
                for ref in str(value).split(";")
                if ref.strip() in turn_to_group
            }
            if gold_turns:
                yield (
                    f"locomo:{conv_id}:{question}",
                    "locomo",
                    question,
                    groups,
                    {turn_to_group[turn] for turn in gold_turns},
                )


def _longmemeval_cases(dataset: list[dict[str, Any]], selected: set[str]):
    for item in dataset:
        qid = str(item["question_id"])
        if qid not in selected:
            continue
        groups = []
        gold_groups = set()
        for fallback_order, (session_id, date, session) in enumerate(
            zip(item["haystack_session_ids"], item["haystack_dates"], item["haystack_sessions"])
        ):
            group = EventGroup(
                key=str(session_id),
                date=str(date),
                order=source_date_order(str(date), fallback_order),
                turns=tuple(
                    (f"{session_id}:{index}", str(turn["content"]))
                    for index, turn in enumerate(session)
                ),
            )
            groups.append(group)
            if any(turn.get("has_answer") for turn in session):
                gold_groups.add(group.key)
        if gold_groups:
            yield qid, "longmemeval", str(item["question"]), groups, gold_groups


def run_experiment(
    *,
    corpus_path: Path,
    locomo_path: Path,
    longmemeval_path: Path,
    locomo_context: Path,
    longmemeval_context: Path,
    split: str = "development",
    encoder: TextEncoder | None = None,
) -> dict[str, Any]:
    selected_locomo, selected_longmemeval = _selected_ids(corpus_path, split)
    cases = list(_locomo_cases(_load(locomo_path), selected_locomo))
    cases += list(_longmemeval_cases(_load(longmemeval_path), selected_longmemeval))
    locomo_baseline = {
        (str(row["conv_id"]), str(row["question"])): row for row in _load(locomo_context)["results"]
    }
    longmemeval_baseline = {
        str(row["question_id"]): row for row in _load(longmemeval_context)["results"]
    }
    encoder = encoder or TextEncoder()
    vector_cache: dict[tuple[tuple[str, str], ...], np.ndarray] = {}
    rows = []
    started = time.perf_counter()
    for case_id, dataset, question, groups, gold_groups in cases:
        turns = tuple(turn for group in groups for turn in group.turns)
        if turns not in vector_cache:
            vector_cache[turns] = encoder.encode_many([text for _, text in turns])
        if dataset == "locomo":
            conv_id = case_id.split(":", 2)[1]
            source_row = locomo_baseline[(conv_id, question)]
        else:
            source_row = longmemeval_baseline[case_id]
        baseline = str(source_row["hypothesis"])
        evidence, selected_groups = select_groups(
            groups, vector_cache[turns], encoder.encode(question), len(baseline)
        )
        recovered = gold_groups.intersection(selected_groups)
        rows.append(
            {
                "id": case_id,
                "dataset": dataset,
                "question": question,
                "expected": source_row.get("expected", source_row.get("expected_answer")),
                "category": source_row.get("category"),
                "hypothesis": evidence,
                "baseline_characters": len(baseline),
                "characters": len(evidence),
                "selected_groups": len(selected_groups),
                "total_groups": len(groups),
                "any_gold_group": bool(recovered),
                "all_gold_groups": recovered == gold_groups,
                "gold_group_recall": len(recovered) / len(gold_groups),
            }
        )

    summary = {}
    for name in ("all", "locomo", "longmemeval"):
        subset = rows if name == "all" else [row for row in rows if row["dataset"] == name]
        summary[name] = {
            "rows": len(subset),
            "any_gold_group_pct": round(
                100 * sum(row["any_gold_group"] for row in subset) / len(subset), 2
            ),
            "all_gold_groups_pct": round(
                100 * sum(row["all_gold_groups"] for row in subset) / len(subset), 2
            ),
            "mean_gold_group_recall": round(
                sum(row["gold_group_recall"] for row in subset) / len(subset), 4
            ),
            "mean_characters": round(sum(row["characters"] for row in subset) / len(subset), 1),
            "mean_baseline_characters": round(
                sum(row["baseline_characters"] for row in subset) / len(subset), 1
            ),
            "mean_selected_groups": round(
                sum(row["selected_groups"] for row in subset) / len(subset), 2
            ),
        }
    return {
        "meta": {
            "format": "chronological_event_groups_v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "split": split,
            "query_method": "single full-query multilingual embedding; max-turn group score",
            "language_specific_processing": False,
            "full_hypotheses": True,
            "evidence_format": "structured_v1",
            "elapsed_s": round(time.perf_counter() - started, 3),
        },
        "summary": summary,
        "results": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus", type=Path, default=ROOT / "data/performance_corpora/slowave_evidence_v1.json"
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
        "--split", choices=("development", "validation", "holdout"), default="development"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data/performance_corpora/chronological_event_groups_development.json",
    )
    args = parser.parse_args()
    result = run_experiment(
        corpus_path=args.corpus,
        locomo_path=args.locomo,
        longmemeval_path=args.longmemeval,
        locomo_context=args.locomo_context,
        longmemeval_context=args.longmemeval_context,
        split=args.split,
    )
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
