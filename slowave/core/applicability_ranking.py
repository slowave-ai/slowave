"""Query-conditioned ranking, independent of candidate discovery and paging.

Cross-encoder logits are evidence scores, not calibrated probabilities. A low
absolute floor retains plausible support without filling an output quota.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from slowave.core.activation_selection import ActivationTask, specific_terms


@dataclass(frozen=True)
class ApplicabilityConfig:
    enabled: bool = True
    minimum_logit: float = -4.0
    maximum_logit_gap: float = 2.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.minimum_logit):
            raise ValueError("minimum_logit must be finite")
        if not np.isfinite(self.maximum_logit_gap) or self.maximum_logit_gap < 0:
            raise ValueError("maximum_logit_gap must be finite and non-negative")


class PairScorer(Protocol):
    def score(self, queries: list[str], memories: list[str]) -> np.ndarray:
        """Return finite logits with shape (memories, queries)."""


def query_needs(query: str) -> list[str]:
    """Use explicit independent conjunctions when present, otherwise query cues.

    Small noun conjunctions such as 'linter and typechecker' remain together.
    Nothing inferred from an assistant's goal is added to the user request.
    """
    task = ActivationTask.build(query)
    needs = list(task.cues)
    parts = re.split(r"\s+and\s+", query, flags=re.I)
    if len(parts) > 1 and all(len(specific_terms(part)) >= 2 for part in parts):
        needs = parts
    return list(dict.fromkeys(needs))[:16]


def applicability_order(
    scores: np.ndarray, *, minimum_logit: float, maximum_logit_gap: float = 2.0
) -> list[int]:
    """Prefer the best answer to each independent need, then useful support.

    Coverage merely changes ordering: every qualifying memory remains available.
    """
    if scores.ndim != 2 or not np.isfinite(scores).all():
        raise ValueError("applicability scores must be a finite matrix")
    if scores.shape[1] == 0:
        return []
    if scores.shape[0] == 0:
        return []
    # Strong direct answers should not silence independently useful support.
    # For weak queries, retain a permissive neighbourhood of the best evidence;
    # the absolute floor still permits a genuinely empty result.
    floors = np.maximum(minimum_logit, np.minimum(0.0, scores.max(axis=0) - maximum_logit_gap))
    best = scores.max(axis=1)
    qualified = [i for i in range(len(scores)) if (scores[i] >= floors).any()]
    ordered = sorted(qualified, key=lambda i: (-float(best[i]), i))
    if scores.shape[1] <= 1:
        return ordered
    winners: set[int] = set()
    for column in range(scores.shape[1]):
        eligible = [i for i in qualified if scores[i, column] >= floors[column]]
        if eligible:
            winners.add(max(eligible, key=lambda i: (float(scores[i, column]), -i)))
    return [i for i in ordered if i in winners] + [i for i in ordered if i not in winners]
