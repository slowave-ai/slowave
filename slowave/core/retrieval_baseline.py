"""Shared multilingual task/goal retrieval with bounded relevance admission."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

POLICY_VERSION = "multilingual-retrieval-baseline-v1"


def enabled() -> bool:
    return os.environ.get("SLOWAVE_RETRIEVAL_BASELINE", "0") == "1"


@dataclass(frozen=True)
class BaselineConfig:
    goal_weight: float = 0.25
    semantic_floor: float = 0.82
    lexical_weight: float = 2.0
    explicit_bonus: float = 0.03
    rrf_k: int = 30
    bm25_content_weight: float = 2.0
    max_memories: int = 1


def rank_candidates(
    task: list[dict[str, Any]],
    goal: list[dict[str, Any]],
    config: BaselineConfig = BaselineConfig(),
) -> list[tuple[float, int]]:
    """Semantic relevance admits; lexical ranks and stable source quality order.

    Both role-aware semantic scores use the same retrieval model. Rank fusion
    is a bounded lexical bonus, never an admission gate or probability.
    """
    goals = {row["id"]: row for row in goal}
    results = []
    weight = config.goal_weight if goal else 0.0
    for row in task:
        other = goals.get(row["id"], row)
        semantic = (1 - weight) * row["semantic"] + weight * other["semantic"]
        if semantic + min(0.0, row.get("feedback_adjustment", 0.0)) < config.semantic_floor:
            continue
        lexical = (
            (1 - weight) / (config.rrf_k + row["lexical_rank"]) if row["lexical_rank"] else 0.0
        ) + (weight / (config.rrf_k + other["lexical_rank"]) if other["lexical_rank"] else 0.0)
        score = (
            semantic
            + config.lexical_weight * lexical
            + config.explicit_bonus * row["explicit"]
            + row.get("feedback_adjustment", 0.0)
        )
        results.append((score, row["id"]))
    return sorted(results, key=lambda row: (-row[0], row[1]))[: config.max_memories]


def feedback_adjustments(
    conn: Any, task: str, goal: str | None, context: dict[str, Any], scope: str | None
) -> dict[int, float]:
    """Bounded ranking evidence for an exact task/goal/context, never a blacklist.

    Latest accepted assessment wins per retrieval/target, so refinements and
    duplicate feedback submissions do not multiply the learning signal.
    """
    import json

    normalize = lambda text: " ".join((text or "").casefold().split())
    rows = conn.execute(
        """
        SELECT f.retrieval_id, f.target_id, f.assessment, e.query, e.goal,
               s.task_context_json
        FROM feedback_events f
        JOIN context_recall_events e ON e.context_id=f.retrieval_id
        LEFT JOIN sessions s ON s.id=e.session_id
        WHERE f.status='accepted' AND f.target_kind='memory' AND e.scope_id IS ?
        ORDER BY f.created_at DESC, f.rowid DESC
    """,
        (scope,),
    ).fetchall()
    seen = set()
    counts: dict[int, int] = {}
    for row in rows:
        key = (row["retrieval_id"], row["target_id"])
        if key in seen:
            continue
        seen.add(key)
        if normalize(row["query"]) != normalize(task) or normalize(row["goal"]) != normalize(goal):
            continue
        if json.loads(row["task_context_json"] or "{}") != context:
            continue
        if not str(row["target_id"]).startswith("sch_") or row["assessment"] not in {
            "used",
            "irrelevant",
        }:
            continue
        sid = int(str(row["target_id"])[4:])
        counts[sid] = counts.get(sid, 0) + (1 if row["assessment"] == "used" else -1)
    return {
        sid: min(0.015, 0.005 * net) if net > 0 else max(-0.06, 0.02 * net)
        for sid, net in counts.items()
    }
