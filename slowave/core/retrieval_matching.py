"""Pure, shared schema matching primitives.

Storage adapters provide candidate IDs, ranks, and raw channel evidence.  This
module deliberately knows nothing about SQLite, encoders, scopes, or MCP
serialization so activate(), recall(), and the white-box harness cannot drift
back to separate scoring formulas.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

SCORING_POLICY_VERSION = "schema-rrf-v1"


@dataclass(frozen=True)
class MatchingConfig:
    """Versioned settings for rank fusion and evidence admission."""

    dense_weight: float = 1.0
    lexical_weight: float = 1.0
    rrf_k: int = 60
    dense_relevance_floor: float = 0.20
    # FTS/BM25 values are corpus-dependent.  A lexical hit is evidence only
    # when the adapter confirms it contains a non-generic query token.
    require_specific_lexical_evidence: bool = True
    salience_factor_min: float = 1.0
    salience_factor_max: float = 1.0

    def __post_init__(self) -> None:
        if self.dense_weight < 0 or self.lexical_weight < 0:
            raise ValueError("RRF weights must be non-negative")
        if self.rrf_k < 0:
            raise ValueError("rrf_k must be non-negative")
        if not -1.0 <= self.dense_relevance_floor <= 1.0:
            raise ValueError("dense_relevance_floor must be in [-1, 1]")
        if self.salience_factor_min <= 0 or self.salience_factor_max < self.salience_factor_min:
            raise ValueError("invalid salience factor bounds")


@dataclass(frozen=True)
class CandidateSignals:
    memory_id: int
    dense_cosine: float | None = None
    dense_rank: int | None = None
    lexical_score: float | None = None
    lexical_rank: int | None = None
    lexical_specific: bool = False
    eligible: bool = True
    eligible_reason: str = "eligible"
    salience: float = 0.0


@dataclass(frozen=True)
class MatchResult:
    memory_id: int
    dense_cosine: float | None
    dense_rank: int | None
    lexical_score: float | None
    lexical_rank: int | None
    fusion_score: float
    eligible: bool
    eligible_reason: str
    relevance_passed: bool
    relevance_reason: str
    memory_adjustment: float
    final_rank_score: float
    normalized_rank_score: float
    selected: bool = False
    selection_reason: str = "not_selected"
    pathway: str = "schema_hybrid"
    scoring_policy_version: str = SCORING_POLICY_VERSION


def _rrf(rank: int | None, weight: float, k: int) -> float:
    return 0.0 if rank is None else weight / (k + rank)


def match_candidates(
    candidates: Iterable[CandidateSignals], *, config: MatchingConfig | None = None
) -> list[MatchResult]:
    """Fuse deduplicated dense and lexical candidates deterministically.

    A rank is one-based.  Missing channels remain ``None`` in diagnostics and
    contribute zero.  RRF orders candidates; independently-calibrated channel
    evidence decides whether they answer the query.
    """
    cfg = config or MatchingConfig()
    by_id: dict[int, CandidateSignals] = {}
    for candidate in candidates:
        previous = by_id.get(candidate.memory_id)
        if previous is None:
            by_id[candidate.memory_id] = candidate
            continue
        # Adapters normally emit one record per ID.  Merging here protects the
        # contract if a caller combines channel lists itself.
        by_id[candidate.memory_id] = CandidateSignals(
            memory_id=candidate.memory_id,
            dense_cosine=(
                candidate.dense_cosine
                if candidate.dense_cosine is not None
                else previous.dense_cosine
            ),
            dense_rank=(
                candidate.dense_rank if candidate.dense_rank is not None else previous.dense_rank
            ),
            lexical_score=(
                candidate.lexical_score
                if candidate.lexical_score is not None
                else previous.lexical_score
            ),
            lexical_rank=(
                candidate.lexical_rank
                if candidate.lexical_rank is not None
                else previous.lexical_rank
            ),
            lexical_specific=candidate.lexical_specific or previous.lexical_specific,
            eligible=candidate.eligible and previous.eligible,
            eligible_reason=(
                candidate.eligible_reason if not candidate.eligible else previous.eligible_reason
            ),
            salience=max(candidate.salience, previous.salience),
        )

    results: list[MatchResult] = []
    max_fusion = _rrf(1, cfg.dense_weight, cfg.rrf_k) + _rrf(1, cfg.lexical_weight, cfg.rrf_k)
    for signal in by_id.values():
        fusion = _rrf(signal.dense_rank, cfg.dense_weight, cfg.rrf_k) + _rrf(
            signal.lexical_rank, cfg.lexical_weight, cfg.rrf_k
        )
        dense_pass = (
            signal.dense_cosine is not None and signal.dense_cosine >= cfg.dense_relevance_floor
        )
        lexical_pass = signal.lexical_rank is not None and (
            signal.lexical_specific or not cfg.require_specific_lexical_evidence
        )
        if dense_pass:
            relevance_reason = "dense_evidence"
        elif lexical_pass:
            relevance_reason = "lexical_evidence"
        elif signal.dense_rank is None and signal.lexical_rank is None:
            relevance_reason = "no_channel_evidence"
        else:
            relevance_reason = "insufficient_channel_evidence"
        # The initial production baseline deliberately makes salience a tie
        # breaker only.  The configurable bounded adjustment is for measured
        # experiments and never changes eligibility or relevance.
        bounded_salience = max(0.0, min(1.0, signal.salience / 20.0))
        adjustment = (
            cfg.salience_factor_min
            + (cfg.salience_factor_max - cfg.salience_factor_min) * bounded_salience
        )
        results.append(
            MatchResult(
                memory_id=signal.memory_id,
                dense_cosine=signal.dense_cosine,
                dense_rank=signal.dense_rank,
                lexical_score=signal.lexical_score,
                lexical_rank=signal.lexical_rank,
                fusion_score=fusion,
                eligible=signal.eligible,
                eligible_reason=signal.eligible_reason,
                relevance_passed=bool(dense_pass or lexical_pass),
                relevance_reason=relevance_reason,
                memory_adjustment=adjustment,
                final_rank_score=fusion * adjustment,
                normalized_rank_score=(fusion / max_fusion if max_fusion > 0.0 else 0.0),
            )
        )
    return sorted(results, key=lambda item: (-item.final_rank_score, item.memory_id))


def select_matches(results: Iterable[MatchResult], limit: int) -> list[MatchResult]:
    """Mark and return eligible, relevant results in stable fused order."""
    selected: list[MatchResult] = []
    for result in results:
        if len(selected) >= limit:
            break
        if not result.eligible or not result.relevance_passed:
            continue
        selected.append(
            MatchResult(**{**result.__dict__, "selected": True, "selection_reason": "selected"})
        )
    return selected
