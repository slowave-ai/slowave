"""Query-need splitting and the pair-scorer protocol.

Scorers return one score per (memory, need) pair; admission and procedures use
them to decide relevance. Cue splitting keeps explicit independent needs apart
instead of diluting them into one concatenated query.
"""

from __future__ import annotations

import re
from typing import Protocol

import numpy as np

from slowave.core.activation_selection import ActivationTask, specific_terms


class PairScorer(Protocol):
    def score(self, queries: list[str], memories: list[str]) -> np.ndarray:
        """Return finite logits with shape (memories, queries)."""


def query_needs(query: str) -> list[str]:
    """Use explicit independent conjunctions when present, otherwise query cues.

    Small noun conjunctions such as 'linter and typechecker' remain together.
    Nothing inferred from an assistant's goal is added to the user request.
    Newline-separated cue lines are assessed as independent need columns
    instead of one concatenated dilution query.
    """
    task = ActivationTask.build(query)
    needs = list(task.cues)
    parts = re.split(r"\s+and\s+", query, flags=re.I)
    if len(parts) > 1 and all(len(specific_terms(part)) >= 2 for part in parts):
        needs = parts
    # Each caller-supplied line is one assessment column; the " and "
    # conjunction rule still applies within a line.
    lines = [line.strip() for line in query.splitlines() if line.strip()]
    if len(lines) > 1:
        expanded: list[str] = []
        for line in lines:
            line_parts = re.split(r"\s+and\s+", line, flags=re.I)
            if len(line_parts) > 1 and all(len(specific_terms(part)) >= 2 for part in line_parts):
                expanded.extend(line_parts)
            else:
                expanded.append(line)
        needs = list(dict.fromkeys(expanded))
    return list(dict.fromkeys(needs))[:16]
