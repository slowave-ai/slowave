"""Compact schema representation for token-efficient MCP responses.

CompactSchema provides a minimal schema serialization (~150-200 tokens vs ~500
for full schemas) while preserving all causal information and memory IDs needed
for downstream slowave_feedback() calls.

This supports the "brain-like memory" design: activation-based selection,
lossy compression, and narrative coherence without metadata bloat.
"""

import re
from dataclasses import asdict, dataclass
from typing import Any

from slowave.symbolic.schema_store import Schema


def _compact(text: str, max_chars: int) -> str:
    """Truncate text to max_chars, removing excess whitespace."""
    one_line = re.sub(r"\s+", " ", str(text).strip())
    if len(one_line) <= max_chars:
        return one_line
    return one_line[: max(0, max_chars - 1)].rstrip() + "…"


def _source_kind(facets: dict[str, Any]) -> str:
    """Extract source_kind from facets (explicit_remember vs inferred vs assistant)."""
    return str(facets.get("source_kind") or facets.get("source") or "").lower().strip()


@dataclass
class CompactSchema:
    """Minimal schema representation for working memory (~150-200 tokens).

    Preserves only the causal information needed for task solving:
    - id: For feedback correlation (slowave_feedback)
    - text: The memory content (truncated to 200 chars for efficiency)
    - activation: Why this memory was selected (0.0-1.0 score)
    - reason: Activation formula breakdown (e.g. "cue_overlap=0.80,salience=0.2")
    - source_kind: Whether this is explicit user knowledge vs inferred

    Intentionally drops:
    - Full facets (confidence, display_label, mean_ts, recurrence_count, etc.)
    - Raw event data (redundant with episodes)
    - Activation traces (debug only)
    - Salience/stability/utility scores (used only internally for ranking)
    """

    id: str  # e.g. "sch_42" — required for feedback
    text: str  # Content, max 200 chars (brain-like lossy compression)
    activation: float  # Activation score 0.0-1.0 (why selected)
    reason: str  # Breakdown: "cue_overlap=0.80"
    source_kind: str  # "explicit_remember" | "inferred" | "assistant_summary"

    @classmethod
    def from_schema(
        cls, schema: Schema, max_chars: int = 500, activation: float | None = None
    ) -> "CompactSchema":
        """Create compact representation from a full schema.

        Args:
            schema: Full Schema object from slowave engine
            max_chars: Max content length (default 500)
            activation: Explicit activation score (0-1, typically cosine similarity).
                       If not provided, falls back to saturating curve over salience.

        Returns:
            CompactSchema with minimal necessary fields
        """
        facets = schema.facets or {}

        # Activation can come from two sources:
        # 1. Explicit cosine score from semantic search (preferred)
        # 2. Saturating fallback over salience for non-semantic retrieval
        if activation is None:
            # Fallback: log-saturating curve over salience
            # Transforms salience [0.01-300+] → activation [0.0-1.0]
            # Using: activation = 2/π * arctan(salience / k)
            # where k ≈ 2.0 is the half-point (salience=2 → activation≈0.5)
            import math

            salience = float(facets.get("salience") or 0)
            k = 2.0  # Half-point: salience=2 gives activation≈0.5
            activation = min(1.0, max(0.0, (2.0 / math.pi) * math.atan(salience / k)))

        return cls(
            id=f"sch_{schema.id}",  # Required for slowave_feedback
            text=_compact(schema.content_text, max_chars),
            activation=round(min(1.0, max(0.0, activation)), 2),  # Clamp to [0,1]
            reason=facets.get("display_label", "general"),
            source_kind=_source_kind(facets),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
