"""Pair scoring over the pinned encoder, with no second model.

Slowave ships one encoder, so a pair score must come from evidence it already
has. ``CheapPairScorer`` answers with the log-odds of the calibrated cheap
classifier over dense, lexical, RRF and passage evidence, keeping a
well-defined ordering. ``LexicalPairScorer`` does the same from term overlap
alone, for runtimes configured without an encoder.

Both are marked ``is_cheap_only``: their values are calibration log-odds
centred on 0.5, not cross-encoder logits, so procedure admission gates them at
the probability boundary.
"""

from __future__ import annotations

import numpy as np

from slowave.core import admission_v2
from slowave.core.activation_selection import _GENERIC
from slowave.core.retrieval_matching import CandidateSignals, MatchingConfig, match_candidates


def _tied_ranks(values: np.ndarray) -> dict[int, int]:
    """One-based descending ranks where equal values share the best rank.

    An index-order tie-break would let list position decide relevance: two
    candidates with identical text then differ only in where they sit, and no
    amount of helped/harmed evidence could reorder them. Equal values must
    therefore produce equal rank features.
    """
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(-values, kind="stable")
    ranks: dict[int, int] = {}
    rank = 0
    previous: float | None = None
    for position, index in enumerate(order, start=1):
        value = float(values[int(index)])
        if previous is None or value != previous:
            rank = position
            previous = value
        ranks[int(index)] = rank
    return ranks


class CheapPairScorer:
    """Log-odds pair scorer built from the encoder and the cheap classifier."""

    is_cheap_only = True

    def __init__(self, encoder, *, shared_search: bool = False) -> None:
        self.encoder = encoder
        self.shared_search = shared_search

    def _query_vector(self, text: str) -> np.ndarray:
        encode_query = getattr(self.encoder, "encode_query", None)
        return encode_query(text) if encode_query is not None else self.encoder.encode(text)

    def score(self, queries: list[str], memories: list[str]) -> np.ndarray:
        """Log-odds matrix of shape (len(memories), len(queries))."""
        out = np.zeros((len(memories), len(queries)), dtype=np.float32)
        if not memories or not queries:
            return out
        encode_many = getattr(self.encoder, "encode_many", None)
        if encode_many is not None:
            vectors = np.asarray(encode_many(memories), dtype=np.float32)
        else:
            vectors = np.asarray([self.encoder.encode(text) for text in memories], dtype=np.float32)
        for column, query in enumerate(queries):
            cosines = vectors @ self._query_vector(query)
            # Identical candidates must score identically; see ``_tied_ranks``.
            dense_rank = _tied_ranks(cosines)
            need_terms = admission_v2._tokens(query, _GENERIC)
            overlap = [
                len(need_terms & admission_v2._tokens(memory, _GENERIC)) for memory in memories
            ]
            lexical_rank = {
                index: rank
                for index, rank in _tied_ranks(np.asarray(overlap, dtype=np.float64)).items()
                if overlap[index]
            }
            signals = [
                CandidateSignals(
                    memory_id=index,
                    dense_cosine=float(cosines[index]),
                    dense_rank=dense_rank[index],
                    lexical_rank=lexical_rank.get(index),
                    lexical_specific=index in lexical_rank,
                )
                for index in range(len(memories))
            ]
            matched = {
                signal.memory_id: signal
                for signal in match_candidates(
                    signals, config=MatchingConfig(dense_relevance_floor=0.0)
                )
            }
            pool = [
                admission_v2.Candidate(
                    index,
                    text,
                    (0,),
                    (
                        {
                            "dense_cosine": float(cosines[index]),
                            "dense_rank": dense_rank[index],
                            "lexical_rank": lexical_rank.get(index),
                            "rrf_score": matched[index].normalized_rank_score,
                        },
                    ),
                    None,
                )
                for index, text in enumerate(memories)
            ]
            features = admission_v2.pool_features(
                pool,
                need_texts=[query],
                user_need_count=1,
                e5=(
                    {index: (float(cosines[index]), column) for index in range(len(memories))}
                    if self.shared_search
                    else None
                ),
                generic=_GENERIC,
            )
            probability = np.clip(admission_v2.cheap_probability(features), 1e-7, 1 - 1e-7)
            out[:, column] = np.log(probability / (1.0 - probability))
        return out


class LexicalPairScorer:
    """Model-free pair scorer: log-odds from content-word overlap only.

    Used when no encoder is configured (``disable_encoder=True`` runtimes, tests
    and degraded installs). It keeps procedure and applicability scoring
    available without any weights; the calibrated classifier still owns the
    topical admission decision.
    """

    is_cheap_only = True

    def __init__(self, *, per_match: float = 3.0, threshold: float = 2.0) -> None:
        self.per_match = per_match
        self.threshold = threshold

    def score(self, queries: list[str], memories: list[str]) -> np.ndarray:
        """Log-odds matrix of shape (len(memories), len(queries))."""
        out = np.zeros((len(memories), len(queries)), dtype=np.float32)
        if not memories or not queries:
            return out
        memory_terms = [admission_v2._tokens(text, _GENERIC) for text in memories]
        for column, query in enumerate(queries):
            query_terms = admission_v2._tokens(query, _GENERIC)
            for row, terms in enumerate(memory_terms):
                overlap = len(query_terms & terms)
                out[row, column] = self.per_match * overlap - self.threshold
        return out
