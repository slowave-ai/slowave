"""Gold-set eval for GeometricContradictionJudge's relation classification.

Nothing like this existed before 2026-07-20 -- tests/benchmarks/ only had
retrieval Recall@K/MRR gold sets, none evaluating relation-classification
correctness (supersedes vs refines vs relates_to vs unrelated) directly.
This is the "before" baseline for any future fix to the decision-tree gaps
found this session (razor-thin margins, no review band outside the
same-scope>=0.85 branch, single-evidence schemas never getting facet_axes).

part_of was removed from the taxonomy on 2026-07-23 (see
private/docs/iterations/20260723_part_of_audit_and_brain_alignment_review.md);
the gold set's former hierarchical_nesting/part_of pairs were relabeled to
relates_to rather than deleted, since they're still valid (old, new) pairs
for exercising the judge.

Usage:
    python tests/benchmarks/relation_gold_eval.py
    python tests/benchmarks/relation_gold_eval.py --judge-overrides '{"same_topic_cosine": 0.70}'
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from slowave.latent.schema import (
    GeometricContradictionJudge,
    GeometricJudgeConfig,
    LatentSchemaBuilder,
)
from slowave.symbolic.encoder import TextEncoder
from slowave.symbolic.episode_text import EpisodeText

GOLD_PATH = Path(__file__).parent / "relation_gold" / "pairs.json"

# Crude synthetic paraphrase variants for multi_evidence pairs -- just enough
# wording variance for LatentSchemaBuilder's SVD to produce non-degenerate
# facet_axes (min_members_for_facets=3). Not meant to be linguistically
# sophisticated; only meant to exercise the facet_axes-available code path.
_VARIANT_TEMPLATES = [
    "{text}",
    "In short: {text}",
    "{text} (as documented).",
]


def _build_schema(
    builder: LatentSchemaBuilder, encoder: TextEncoder, text: str, multi_evidence: bool
):
    texts = [t.format(text=text) for t in _VARIANT_TEMPLATES] if multi_evidence else [text]
    embeddings = encoder.encode_many(texts)
    centroid = embeddings.mean(axis=0)
    episodes = [
        EpisodeText(
            episode_id=i,
            content_text=t,
            source_content=t,
            event_ids=[],
            session_id=None,
        )
        for i, t in enumerate(texts)
    ]
    return builder.build(
        centroid=centroid,
        member_embeddings=embeddings,
        member_episodes=episodes,
        member_episode_ids=list(range(len(texts))),
        member_timestamps=[i for i in range(len(texts))],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Relation-classification gold-set eval")
    parser.add_argument(
        "--judge-overrides",
        default="",
        help="JSON dict of GeometricJudgeConfig field overrides, e.g. '{\"same_topic_cosine\": 0.70}'",
    )
    parser.add_argument("--gold", default=str(GOLD_PATH), help="path to the gold-set JSON file")
    args = parser.parse_args()

    gold = json.loads(Path(args.gold).read_text())["pairs"]

    encoder = TextEncoder()
    builder = LatentSchemaBuilder()
    judge_overrides = json.loads(args.judge_overrides) if args.judge_overrides else {}

    judge = GeometricContradictionJudge(GeometricJudgeConfig(**judge_overrides))

    confusion: dict[str, Counter] = defaultdict(Counter)
    mismatches = []

    for pair in gold:
        multi_evidence = bool(pair.get("multi_evidence", False))
        old_schema = _build_schema(builder, encoder, pair["old"], multi_evidence)
        new_schema = _build_schema(builder, encoder, pair["new"], multi_evidence)
        if old_schema is None or new_schema is None:
            print(f"[{pair['id']}] SKIP -- schema build failed", file=sys.stderr)
            continue

        verdict = judge.judge(old=old_schema, new=new_schema)
        predicted = verdict.verdict
        expected = pair["correct_relation"]
        confusion[expected][predicted] += 1
        if predicted != expected:
            mismatches.append(
                (pair["id"], pair["category"], expected, predicted, verdict.reasoning)
            )

    # Per-relation precision/recall over the labels actually seen.
    labels = sorted(
        {expected for expected in confusion} | {p for c in confusion.values() for p in c}
    )
    print("Confusion matrix (rows=expected, cols=predicted):")
    header = "expected \\ predicted".ljust(22) + "".join(f"{lb:>14}" for lb in labels)
    print(header)
    for expected in labels:
        row = "".join(f"{confusion[expected][lb]:>14}" for lb in labels)
        print(f"{expected:<22}{row}")

    print("\nPer-relation precision/recall:")
    for label in labels:
        tp = confusion[label][label]
        support = sum(confusion[label].values())
        predicted_total = sum(confusion[e][label] for e in labels)
        recall = tp / support if support else float("nan")
        precision = tp / predicted_total if predicted_total else float("nan")
        print(f"  {label:<14} precision={precision:.2f}  recall={recall:.2f}  support={support}")

    total = sum(sum(c.values()) for c in confusion.values())
    correct = sum(confusion[label][label] for label in labels)
    print(
        f"\nOverall accuracy: {correct}/{total} = {correct / total:.2%}"
        if total
        else "No pairs evaluated."
    )

    if mismatches:
        print("\nMismatches:")
        for pid, category, expected, predicted, reasoning in mismatches:
            print(f"  [{pid}] ({category}) expected={expected} got={predicted} -- {reasoning}")


if __name__ == "__main__":
    main()
