"""Pure experimental hybrid selection; no model logits or language heuristics."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class HybridConfig:
    depth: int = 128
    dense_weight: float = 2.0
    lexical_weight: float = 1.0
    rrf_k: int = 60
    cosine_floor: float = 0.60
    # 0 disables lexical-only admission. Otherwise this many independently
    # rare query terms must match; term rarity is corpus-based, not English stopwords.
    lexical_rare_terms: int = 0
    query_sources: str = "both"
    bm25_content_weight: float = 1.0


def discover(
    queries: list[list[dict[str, Any]]], config: HybridConfig
) -> dict[int, dict[str, Any]]:
    """Fuse each independent query, bound its pool, then union by ID.

    Best per-query score prevents repeated task/goal cues gaining extra votes.
    Channel ranks are computed before eligibility, matching the serving adapter.
    """
    selected_queries = queries
    if config.query_sources == "task":
        selected_queries = queries[:1]
    elif config.query_sources == "goal":
        selected_queries = queries[-1:]
    pool: dict[int, dict[str, Any]] = {}
    for query in selected_queries:
        ranked = []
        for candidate in query:
            dense_rank = candidate.get("dense_rank")
            lexical_rank = candidate.get("lexical_ranks", {}).get(
                str(config.bm25_content_weight), candidate.get("lexical_rank")
            )
            dense = (
                config.dense_weight / (config.rrf_k + dense_rank)
                if dense_rank and dense_rank <= config.depth
                else 0.0
            )
            lexical = (
                config.lexical_weight / (config.rrf_k + lexical_rank)
                if lexical_rank and lexical_rank <= config.depth
                else 0.0
            )
            if dense + lexical:
                ranked.append((dense + lexical, candidate))
        ranked.sort(key=lambda item: (-item[0], item[1]["memory_id"]))
        for score, candidate in ranked[: config.depth]:
            sid = candidate["memory_id"]
            if not candidate["eligible"]:
                continue
            previous = pool.get(sid)
            if previous is None:
                pool[sid] = {
                    "rrf": score,
                    "cosine": candidate.get("cosine"),
                    "rare_terms": candidate["rare_terms"],
                }
            else:
                previous["rrf"] = max(previous["rrf"], score)
                if candidate.get("cosine") is not None:
                    previous["cosine"] = max(
                        previous["cosine"] if previous["cosine"] is not None else -1.0,
                        candidate["cosine"],
                    )
                previous["rare_terms"] = max(previous["rare_terms"], candidate["rare_terms"])
    return pool


def admit(pool: dict[int, dict[str, Any]], config: HybridConfig) -> list[int]:
    """One absolute cosine decision, with an explicitly evaluated lexical escape."""
    selected = [
        sid
        for sid, evidence in pool.items()
        if (evidence["cosine"] is not None and evidence["cosine"] >= config.cosine_floor)
        or (config.lexical_rare_terms > 0 and evidence["rare_terms"] >= config.lexical_rare_terms)
    ]
    return sorted(selected, key=lambda sid: (-pool[sid]["rrf"], sid))


POLICY_VERSION = "activate-hybrid-dogfood-v1"
DOGFOOD_CONFIG = HybridConfig(
    depth=32,
    dense_weight=0.5,
    lexical_weight=1,
    rrf_k=30,
    cosine_floor=0.5,
    lexical_rare_terms=0,
    query_sources="task",
    bm25_content_weight=2.0,
)


def dogfood_enabled() -> bool:
    return os.environ.get("SLOWAVE_ACTIVATE_HYBRID", "0") == "1"
