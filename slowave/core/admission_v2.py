"""Calibrated, similarity-gated admission for task-specific activation context.

Replaces the pool-relative "best minus margin behind a floor" rule for the
topical tier. Every candidate gets cheap features from channels discovery
already computed (dense cosine, passage cosine, lexical rank and specific-term
overlap, RRF, how many task facets it matches, type, length). A small
logistic model over those features (versioned constants fitted offline,
never trained at runtime) predicts P(candidate is useful for the task).
Admission ranks by that probability and fills an item/byte budget behind an
absolute probability cutoff, so a weak pool yields nothing and a strong pool
with several relevant facts yields several, instead of ``best - 1.0``.

Learning only orders; nothing here reads or writes feedback state.
"""

from __future__ import annotations

import logging
import math
import os
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np

from slowave.core.admission_v2_params import (
    BYTE_BUDGET,
    CHEAP_B,
    CHEAP_W,
)
from slowave.core.admission_v2_params import FEATURES as _FITTED_FEATURES
from slowave.core.admission_v2_params import (
    MAX_ITEMS,
    TAU,
    TAU_BEST,
)

MODEL_VERSION = "admission-v2-1"

# Offline hook (tests / forensics): called with the pool feature payload.
TRACE: Callable[[dict[str, Any]], None] | None = None

FEATURES = (
    "dense_cos",
    "dense_cos_z",
    "dense_rank_inv",
    "e5_cos_z",
    "lex_rank_inv",
    "lex_specific",
    "rrf",
    "rrf_z",
    "idf_overlap",
    "shared_terms_log",
    "need_frac",
    "log_len",
    "is_rule",
)

if tuple(_FITTED_FEATURES) != FEATURES:  # pragma: no cover - guards a stale params file
    raise ImportError("admission_v2_params.py was fitted for a different feature set")

_RULE_CLASSES = frozenset({"constraint", "decision", "preference", "warning"})
_CJK = re.compile(r"[\u2e80-\u9fff\uac00-\ud7af]")
_TOKEN = re.compile(r"[^\W_]+(?:[_./:-][^\W_]+)*", re.UNICODE)


@dataclass(frozen=True)
class Candidate:
    memory_id: int
    text: str
    need_indexes: tuple[int, ...]
    evidence: tuple[dict[str, Any], ...]
    schema_class: str | None = None


STEM_LEN = 6  # language-agnostic truncation stem: importo/importi/importare share "import"


def _fold(text: str) -> str:
    folded = unicodedata.normalize("NFKD", text.casefold())
    return "".join(ch for ch in folded if not unicodedata.combining(ch))


def _tokens(text: str, generic: frozenset[str] | set[str] = frozenset()) -> set[str]:
    """Accent-folded, truncation-stemmed terms; CJK runs become character bigrams.

    Inflected languages (Italian importo/importi, German Abhängigkeit/Abhängigkeiten) otherwise
    share no exact term with an English-fitted lexical feature.  ``generic`` is matched on the
    folded whole token before it is stemmed.
    """
    out: set[str] = set()
    for raw in _TOKEN.findall(_fold(text)):
        for token in [raw, *re.split(r"[_./:-]+", raw)]:
            if len(token) < 2 or token in generic:
                continue
            if _CJK.search(token):
                out.update(token[i : i + 2] for i in range(len(token) - 1))
            else:
                out.add(token[:STEM_LEN])
    return out


def _z(values: np.ndarray, present: np.ndarray) -> np.ndarray:
    """Pool-relative standardisation over candidates that have the channel."""
    out = np.zeros_like(values, dtype=np.float64)
    if present.sum() >= 3:
        mean = float(values[present].mean())
        std = float(values[present].std())
        out[present] = (values[present] - mean) / max(std, 0.02)
    return np.clip(out, -4.0, 6.0)


def pool_features(
    candidates: Sequence[Candidate],
    *,
    need_texts: Sequence[str],
    user_need_count: int,
    e5: dict[int, tuple[float, int]] | None,
    generic: frozenset[str] | set[str],
) -> np.ndarray:
    """Cheap feature matrix, shape (candidates, len(FEATURES)). Pure, no model."""
    n = len(candidates)
    X = np.zeros((n, len(FEATURES)), dtype=np.float64)
    if n == 0:
        return X
    dense = np.zeros(n)
    dense_has = np.zeros(n, dtype=bool)
    e5v = np.zeros(n)
    e5_has = np.zeros(n, dtype=bool)
    rrf = np.zeros(n)
    rrf_has = np.zeros(n, dtype=bool)
    for i, c in enumerate(candidates):
        cos = [e["dense_cosine"] for e in c.evidence if e.get("dense_cosine") is not None]
        if cos:
            dense[i], dense_has[i] = max(cos), True
        ranks = [e["dense_rank"] for e in c.evidence if e.get("dense_rank")]
        X[i, 2] = max((1.0 / r for r in ranks), default=0.0)
        if e5 and c.memory_id in e5:
            e5v[i], e5_has[i] = e5[c.memory_id][0], True
        lranks = [e["lexical_rank"] for e in c.evidence if e.get("lexical_rank")]
        X[i, 4] = max((1.0 / r for r in lranks), default=0.0)
        X[i, 5] = 1.0 if lranks else 0.0
        scores = [e["rrf_score"] for e in c.evidence if e.get("rrf_score") is not None]
        if scores:
            rrf[i], rrf_has[i] = max(scores), True
    X[:, 0] = dense
    X[:, 1] = _z(dense, dense_has)
    X[:, 3] = _z(e5v, e5_has)
    X[:, 6] = rrf
    X[:, 7] = _z(rrf, rrf_has)
    # idf-weighted overlap against every task need; pool document frequency.
    cand_terms = [_tokens(c.text, generic) for c in candidates]
    df: dict[str, int] = {}
    for terms in cand_terms:
        for t in terms:
            df[t] = df.get(t, 0) + 1
    idf = {t: math.log((n + 1.0) / (d + 0.5)) + 0.1 for t, d in df.items()}
    need_terms = [_tokens(t, generic) for t in need_texts]
    for i, terms in enumerate(cand_terms):
        best_overlap, best_shared = 0.0, 0
        for nt in need_terms:
            if not nt:
                continue
            mass = sum(idf.get(t, math.log(n + 1.0) + 0.1) for t in nt)
            hit = nt & terms
            if mass > 0 and hit:
                best_overlap = max(best_overlap, sum(idf[t] for t in hit) / mass)
            best_shared = max(best_shared, len(hit))
        X[i, 8] = best_overlap
        X[i, 9] = math.log1p(best_shared)
        X[i, 10] = len(candidates[i].need_indexes) / max(1, user_need_count)
        X[i, 11] = math.log(max(20, len(candidates[i].text)))
        X[i, 12] = 1.0 if candidates[i].schema_class in _RULE_CLASSES else 0.0
    return X


def sigmoid(z: Any) -> Any:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


def cheap_probability(X: np.ndarray) -> np.ndarray:
    return sigmoid(X @ np.asarray(CHEAP_W) + CHEAP_B)


# Deliberate recall keeps weak support: no best-candidate gate, a lower cutoff.
RECALL_TAU = 0.12


def admit(
    probability: np.ndarray,
    sizes: Sequence[int],
    *,
    tau: float | None = None,
    tau_best: float | None = None,
) -> list[int]:
    """Indexes to deliver, best first.

    The pool must contain at least one convincing candidate (``TAU_BEST``) to
    admit anything, so an unrelated task yields nothing; once open, further
    candidates need ``TAU`` each and must fit the item and byte budget. The
    first item is never dropped for size.
    """
    order = sorted(range(len(probability)), key=lambda i: (-float(probability[i]), i))
    tau = TAU if tau is None else tau
    tau_best = TAU_BEST if tau_best is None else tau_best
    if not order or float(probability[order[0]]) < tau_best:
        return []
    chosen: list[int] = []
    used = 0
    for i in order:
        if float(probability[i]) < tau or len(chosen) >= MAX_ITEMS:
            break
        if chosen and used + sizes[i] > BYTE_BUDGET:
            continue
        chosen.append(i)
        used += sizes[i]
    return chosen
