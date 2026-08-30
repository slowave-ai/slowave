"""Gold labels and measurements for compact retrieval responses.

These contracts intentionally score only the public response. Internal scores
and suppression traces belong in diagnostic sidecars, never in the oracle.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

Surface = Literal["activate", "recall"]


@dataclass(frozen=True)
class RetrievalGold:
    case_id: str
    family: str
    surface: Surface
    scope: str
    required_contents: tuple[str, ...] = ()
    forbidden_contents: tuple[str, ...] = ()
    optional_contents: tuple[str, ...] = ()
    historical_only_contents: tuple[str, ...] = ()
    expected_empty: bool = False
    max_items: int = 2


@dataclass(frozen=True)
class RetrievalObservation:
    case_id: str
    surface: Surface
    retrieval_id: str
    elapsed_ms: float
    serialized_chars: int
    content_chars: int
    estimated_tokens: int
    returned_ids: tuple[str, ...]
    returned_contents: tuple[str, ...]
    pathways: tuple[str, ...]
    raw: dict[str, Any] = field(repr=False)


@dataclass(frozen=True)
class RetrievalEvaluation:
    case_id: str
    family: str
    surface: Surface
    passed: bool
    required_found: tuple[str, ...]
    required_missing: tuple[str, ...]
    forbidden_found: tuple[str, ...]
    historical_found: tuple[str, ...]
    optional_found: tuple[str, ...]
    correct_empty: bool | None
    budget_ok: bool
    precision: float
    strict_useful_precision: float
    intrusion_characters: int
    observation: RetrievalObservation

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _matching_contents(needles: tuple[str, ...], contents: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(needle for needle in needles if any(needle in content for content in contents))


def observe(
    case_id: str, surface: Surface, data: dict[str, Any], elapsed_ms: float
) -> RetrievalObservation:
    memories = data.get("memories", [])
    contents = tuple(str(item.get("content", "")) for item in memories)
    serialized = json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    content_chars = sum(len(content) for content in contents)
    return RetrievalObservation(
        case_id=case_id,
        surface=surface,
        retrieval_id=str(data["retrieval_id"]),
        elapsed_ms=round(elapsed_ms, 3),
        serialized_chars=len(serialized),
        content_chars=content_chars,
        estimated_tokens=math.ceil(len(serialized) / 4),
        returned_ids=tuple(str(item["memory_id"]) for item in memories),
        returned_contents=contents,
        pathways=tuple(str(item.get("pathway", "")) for item in memories),
        raw=data,
    )


def evaluate(gold: RetrievalGold, observation: RetrievalObservation) -> RetrievalEvaluation:
    contents = observation.returned_contents
    required_found = _matching_contents(gold.required_contents, contents)
    required_missing = tuple(item for item in gold.required_contents if item not in required_found)
    forbidden_found = _matching_contents(gold.forbidden_contents, contents)
    historical_found = _matching_contents(gold.historical_only_contents, contents)
    optional_found = _matching_contents(gold.optional_contents, contents)
    correct_empty = (not contents) if gold.expected_empty else None
    budget_ok = len(contents) <= gold.max_items
    relevant_count = len(required_found) + len(optional_found)
    precision = (
        relevant_count / len(contents) if contents else (1.0 if gold.expected_empty else 0.0)
    )
    strict_precision = len(required_found) / len(contents) if contents else 0.0
    intrusion_chars = sum(
        len(content)
        for content in contents
        if any(
            needle in content
            for needle in (*gold.forbidden_contents, *gold.historical_only_contents)
        )
    )
    passed = (
        not required_missing
        and not forbidden_found
        and not historical_found
        and budget_ok
        and (correct_empty is not False)
    )
    return RetrievalEvaluation(
        case_id=gold.case_id,
        family=gold.family,
        surface=gold.surface,
        passed=passed,
        required_found=required_found,
        required_missing=required_missing,
        forbidden_found=forbidden_found,
        historical_found=historical_found,
        optional_found=optional_found,
        correct_empty=correct_empty,
        budget_ok=budget_ok,
        precision=precision,
        strict_useful_precision=strict_precision,
        intrusion_characters=intrusion_chars,
        observation=observation,
    )
