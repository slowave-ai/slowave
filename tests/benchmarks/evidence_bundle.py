"""Deterministic, source-preserving evidence assembly experiment.

This module operates on already-retrieved ``structured_v1`` evidence. It does
not generate answers or recover candidates missing from the retrieval result.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import cached_property

_BOUNDARY = re.compile(r"^\[(SCHEMA|EPISODE)\s+(\d+)\s*\|\s*([^\]]+)\]\n", re.MULTILINE)
_TOKEN = re.compile(r"[a-z0-9]+")
_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
_DATED_BLOCK = re.compile(r"\[(\d{4}-\d{2}-\d{2})\]")
_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "between",
    "did",
    "do",
    "for",
    "from",
    "had",
    "has",
    "have",
    "how",
    "i",
    "in",
    "is",
    "it",
    "me",
    "most",
    "my",
    "of",
    "on",
    "or",
    "the",
    "to",
    "was",
    "were",
    "what",
    "when",
    "which",
    "with",
}


@dataclass(frozen=True)
class EvidenceItem:
    kind: str
    rank: int
    metadata: str
    text: str

    @cached_property
    def tokens(self) -> set[str]:
        return _tokens(self.text)

    @property
    def date(self) -> str | None:
        match = _DATE.search(self.metadata) or _DATE.search(self.text)
        return match.group(0) if match else None


def _tokens(text: str) -> set[str]:
    return {token for token in _TOKEN.findall(text.lower()) if token not in _STOPWORDS}


def parse_structured_evidence(evidence: str) -> list[EvidenceItem]:
    matches = list(_BOUNDARY.finditer(evidence))
    if not matches:
        dated = list(_DATED_BLOCK.finditer(evidence))
        return [
            EvidenceItem(
                kind="episode",
                rank=index + 1,
                metadata=f"date={match.group(1)}",
                text=evidence[
                    match.start() : (
                        dated[index + 1].start() if index + 1 < len(dated) else len(evidence)
                    )
                ].strip(),
            )
            for index, match in enumerate(dated)
        ]
    items = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(evidence)
        items.append(
            EvidenceItem(
                kind=match.group(1).lower(),
                rank=int(match.group(2)),
                metadata=match.group(3),
                text=evidence[match.end() : end].strip(),
            )
        )
    return items


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def assemble_evidence_bundle(
    question: str,
    evidence: str,
    *,
    max_items: int = 8,
    redundancy_weight: float = 0.35,
) -> str:
    """Select a compact diverse bundle without consulting the expected answer."""
    items = parse_structured_evidence(evidence)
    if not items or len(items) <= max_items:
        return evidence

    query_tokens = _tokens(question)
    selected: list[EvidenceItem] = []
    remaining = list(items)
    covered: set[str] = set()
    while remaining and len(selected) < max_items:
        best: EvidenceItem | None = None
        best_score = float("-inf")
        for item in remaining:
            tokens = item.tokens
            relevance = len(tokens & query_tokens) / max(1, len(query_tokens))
            new_coverage = len((tokens & query_tokens) - covered) / max(1, len(query_tokens))
            redundancy = max((_jaccard(tokens, chosen.tokens) for chosen in selected), default=0.0)
            timestamp_bonus = 0.05 if item.kind == "episode" and item.date else 0.0
            rank_tiebreak = 1e-6 / max(1, item.rank)
            score = (
                relevance + 0.20 * new_coverage + timestamp_bonus - redundancy_weight * redundancy
            )
            score += rank_tiebreak
            if score > best_score:
                best, best_score = item, score
        assert best is not None
        selected.append(best)
        covered |= best.tokens & query_tokens
        remaining.remove(best)

    episodes = sorted(
        (item for item in selected if item.kind == "episode"),
        key=lambda item: (item.date is None, item.date or "", item.rank),
    )
    schemas = sorted(
        (item for item in selected if item.kind == "schema"), key=lambda item: item.rank
    )
    sections = []
    if episodes:
        sections.append("=== EPISODES (selected dated evidence, chronological) ===")
        sections.extend(
            f"[EPISODE {item.rank} | {item.metadata}]\n{item.text}" for item in episodes
        )
    if schemas:
        sections.append("=== SCHEMAS (selected stable context) ===")
        sections.extend(f"[SCHEMA {item.rank} | {item.metadata}]\n{item.text}" for item in schemas)
    return "\n\n".join(sections)


def bundle_metrics(evidence: str, *, expected: str = "") -> dict[str, float | int]:
    items = parse_structured_evidence(evidence)
    token_sets = [item.tokens for item in items]
    similarities = [
        _jaccard(token_sets[i], token_sets[j])
        for i in range(len(token_sets))
        for j in range(i + 1, len(token_sets))
    ]
    expected_tokens = _tokens(expected)
    evidence_tokens = _tokens(evidence)
    return {
        "characters": len(evidence),
        "items": len(items),
        "dated_items": sum(item.date is not None for item in items),
        "mean_pairwise_jaccard": round(sum(similarities) / max(1, len(similarities)), 6),
        "expected_token_coverage": round(
            len(expected_tokens & evidence_tokens) / max(1, len(expected_tokens)), 6
        ),
    }
