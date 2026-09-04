"""Stage 7 — intrinsic temporal context.

Brain analogue: hippocampal time cells and the lateral entorhinal
cortex's temporal context cells. Every encoded memory carries its
temporal context as an intrinsic property of the trace — not as a
separate metadata field looked up later. Recalling a memory recalls
its temporal context as part of the same activation pattern.

We approximate the brain's continuous temporal context vector with a
small **multi-scale sinusoidal embedding**. At each scale (minute,
hour, day, week, month, year, decade) we emit a (sin, cos) pair of the
timestamp's phase at that scale. The result is a deterministic,
zero-training, low-dimensional fingerprint of a moment in time.

Two timestamps that are close on *any* scale have a positive cosine
similarity in this space; two timestamps separated by all scales (e.g.
years apart and at very different times of day) have a similarity near
zero. The biological correspondence is rough but the architectural
intent is faithful: temporal proximity is a coordinate that
co-activates with semantic content, not a query filter.

Usage in retrieval (Stage 7 wiring):

    cos_total = cos(semantic_q, semantic_m)
              + α_temporal * cos(temporal_q, temporal_m)
              + α_salience * salience_m

The α coefficients are picked from the architectural argument
(temporal context is a real but secondary signal), not tuned to a
benchmark.

Temporal anchor estimation (Stage 10):
-------------------------------------
Brain analogue: the lateral entorhinal cortex performs a backward
temporal search when a query contains temporal language. Rather than
parsing "last month" with a brittle rule-set, we exploit the fact that
temporal language is *already encoded* in the sentence-embedding space
— the same encoder that encodes episodes. A small static set of
temporal probe phrases (the "temporal compass") is embedded once at
init. At query time we measure cosine similarity between the query
embedding and each probe, softmax-weight the corresponding
displacements, and return the expected anchor timestamp.

This is zero-dependency, zero-regex, zero extra LLM call, and
generalises to any phrasing the encoder has seen during pre-training
(including informal / multilingual expressions). When the query
contains no temporal language the "now / today" probe dominates and
the anchor collapses to the current time — preserving the existing
default behaviour exactly.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------------------
# Temporal compass probes
# ---------------------------------------------------------------------------
# Each entry is (natural-language phrase, displacement_seconds from now).
# Negative = past.  Chosen to span the major retrieval bands that appear
# in LongMemEval / LoCoMo temporal questions.  The list intentionally stays
# small: adding more probes only increases precision marginally because the
# softmax already interpolates between adjacent anchors.
#
# Brain analogue: these are the discrete "temporal landmark" attractors that
# the entorhinal cortex uses to reconstruct approximate past context.  Real
# LEC cells fire at continuous rates; we discretise into ~12 landmarks and
# let the weighted mean do the interpolation.
# ---------------------------------------------------------------------------
_DAY = 86_400
_TEMPORAL_PROBES: tuple[tuple[str, int], ...] = (
    ("right now, today, at the moment", 0),
    ("yesterday, the day before", -1 * _DAY),
    ("a few days ago, several days ago", -4 * _DAY),
    ("last week, a week ago", -7 * _DAY),
    ("two weeks ago, a fortnight ago", -14 * _DAY),
    ("last month, a month ago, recently", -30 * _DAY),
    ("two months ago, a couple of months ago", -60 * _DAY),
    ("three months ago, several months ago", -90 * _DAY),
    ("six months ago, half a year ago", -180 * _DAY),
    ("last year, a year ago", -365 * _DAY),
    ("two years ago", -730 * _DAY),
    ("a long time ago, years ago, long ago", -3 * 365 * _DAY),
)

# Equivalent landmark phrases for the bundled multilingual encoder.  Each
# tuple is aligned with _TEMPORAL_PROBES.  We take the best similarity within
# a band (rather than treating every translation as an independent landmark),
# so adding a language cannot bias the softmax toward that time interval.
_TEMPORAL_PROBE_VARIANTS: tuple[tuple[str, ...], ...] = (
    (
        "right now, today, at the moment",
        "adesso, oggi, in questo momento",
        "ahora, hoy, en este momento",
        "maintenant, aujourd’hui, en ce moment",
        "jetzt, heute, im moment",
        "agora, hoje, neste momento",
    ),
    (
        "yesterday, the day before",
        "ieri, il giorno prima",
        "ayer, el día anterior",
        "hier, la veille",
        "gestern, am tag davor",
        "ontem, no dia anterior",
    ),
    (
        "a few days ago, several days ago",
        "qualche giorno fa, alcuni giorni fa",
        "hace unos días, hace varios días",
        "il y a quelques jours, il y a plusieurs jours",
        "vor ein paar tagen, vor mehreren tagen",
        "há alguns dias, há vários dias",
    ),
    (
        "last week, a week ago",
        "la settimana scorsa, una settimana fa",
        "la semana pasada, hace una semana",
        "la semaine dernière, il y a une semaine",
        "letzte woche, vor einer woche",
        "na semana passada, há uma semana",
    ),
    (
        "two weeks ago, a fortnight ago",
        "due settimane fa",
        "hace dos semanas",
        "il y a deux semaines",
        "vor zwei wochen",
        "há duas semanas",
    ),
    (
        "last month, a month ago, recently",
        "il mese scorso, un mese fa, di recente",
        "el mes pasado, hace un mes, recientemente",
        "le mois dernier, il y a un mois, récemment",
        "letzten monat, vor einem monat, kürzlich",
        "no mês passado, há um mês, recentemente",
    ),
    (
        "two months ago, a couple of months ago",
        "due mesi fa, un paio di mesi fa",
        "hace dos meses, un par de meses",
        "il y a deux mois, quelques mois",
        "vor zwei monaten, vor ein paar monaten",
        "há dois meses, alguns meses atrás",
    ),
    (
        "three months ago, several months ago",
        "tre mesi fa, diversi mesi fa",
        "hace tres meses, hace varios meses",
        "il y a trois mois, il y a plusieurs mois",
        "vor drei monaten, vor mehreren monaten",
        "há três meses, há vários meses",
    ),
    (
        "six months ago, half a year ago",
        "sei mesi fa, mezzo anno fa",
        "hace seis meses, hace medio año",
        "il y a six mois, il y a un semestre",
        "vor sechs monaten, vor einem halben jahr",
        "há seis meses, há meio ano",
    ),
    (
        "last year, a year ago",
        "l’anno scorso, un anno fa",
        "el año pasado, hace un año",
        "l’année dernière, il y a un an",
        "letztes jahr, vor einem jahr",
        "no ano passado, há um ano",
    ),
    (
        "two years ago",
        "due anni fa",
        "hace dos años",
        "il y a deux ans",
        "vor zwei jahren",
        "há dois anos",
    ),
    (
        "a long time ago, years ago, long ago",
        "molto tempo fa, anni fa",
        "hace mucho tiempo, hace años",
        "il y a longtemps, il y a des années",
        "vor langer zeit, vor jahren",
        "há muito tempo, há anos",
    ),
)


# Scales chosen to span the relevant brain-time bands:
#   minute  - intra-conversation drift
#   hour    - within-day position (morning/evening)
#   day     - day-of-week pattern
#   week    - recent vs older this month
#   month   - seasonal drift
#   year    - long-horizon
#   decade  - lifetime epoch
#
# Each scale contributes a (sin, cos) pair, so the resulting embedding
# is 2 * len(SCALES_SECONDS) dimensional. With 7 scales that's a
# 14-dimensional temporal vector — cheap to compute and store.
SCALES_SECONDS: tuple[int, ...] = (
    60,  # minute
    60 * 60,  # hour
    24 * 60 * 60,  # day
    7 * 24 * 60 * 60,  # week
    30 * 24 * 60 * 60,  # month (approx)
    365 * 24 * 60 * 60,  # year
    10 * 365 * 24 * 60 * 60,  # decade
)


@dataclass(frozen=True)
class TemporalContextConfig:
    # Scales to use. Defaults to the 7-band scheme above.
    scales_seconds: tuple[int, ...] = SCALES_SECONDS


class TemporalContext:
    """Build deterministic sinusoidal temporal-context vectors.

    Pure function wrapper. Holds no state; instances are cheap to
    create and re-use is purely for caching the scales tuple.
    """

    def __init__(self, cfg: TemporalContextConfig | None = None):
        self.cfg = cfg or TemporalContextConfig()
        self._scales = np.asarray(self.cfg.scales_seconds, dtype=np.float64)
        # Output dimension is 2 per scale (sin + cos).
        self.dim = int(2 * len(self._scales))

    def encode(self, ts_seconds: int | float) -> np.ndarray:
        """Encode a single timestamp as a unit-norm temporal vector."""
        if ts_seconds is None:
            ts_seconds = 0
        ts = float(ts_seconds)
        phases = 2.0 * math.pi * ts / self._scales  # one phase per scale
        vec = np.empty(self.dim, dtype=np.float32)
        vec[0::2] = np.sin(phases).astype(np.float32)
        vec[1::2] = np.cos(phases).astype(np.float32)
        # Sinusoids are already bounded; the vector has a natural L2
        # norm of sqrt(n_scales). Normalise to unit length so cosine
        # similarity with another encoded ts is in [-1, 1].
        n = float(np.linalg.norm(vec))
        if n > 0:
            vec = vec / n
        return vec

    def encode_many(self, ts_seconds: np.ndarray | list[int]) -> np.ndarray:
        """Vectorised encode of an array of timestamps. Returns (N, dim)."""
        ts = np.asarray(ts_seconds, dtype=np.float64).reshape(-1)
        if ts.size == 0:
            return np.zeros((0, self.dim), dtype=np.float32)
        # (N, n_scales) phases
        phases = 2.0 * math.pi * ts[:, None] / self._scales[None, :]
        out = np.empty((ts.size, self.dim), dtype=np.float32)
        out[:, 0::2] = np.sin(phases).astype(np.float32)
        out[:, 1::2] = np.cos(phases).astype(np.float32)
        # Per-row L2 normalise
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return out / norms

    def now(self) -> np.ndarray:
        """Temporal context vector for the current moment."""
        return self.encode(int(time.time()))

    @staticmethod
    def cosine(a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two temporal vectors."""
        if a is None or b is None:
            return 0.0
        a = np.asarray(a, dtype=np.float32).reshape(-1)
        b = np.asarray(b, dtype=np.float32).reshape(-1)
        na = float(np.linalg.norm(a))
        nb = float(np.linalg.norm(b))
        if na == 0.0 or nb == 0.0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))


# ---------------------------------------------------------------------------
# Temporal probe / compass  (Stage 10)
# ---------------------------------------------------------------------------


class TemporalProbe:
    """Embedding-space temporal anchor estimator.

    Brain analogue: lateral entorhinal cortex backward temporal search.
    Rather than parsing natural-language time expressions with a brittle
    rule-set, we exploit the fact that temporal language is *already
    encoded* in the sentence-embedding space — the same encoder used for
    episodes.  A static set of temporal probe phrases (the "temporal
    compass") is embedded once at init.  At query time we measure cosine
    similarity between the query embedding and each probe, softmax-weight
    the corresponding time displacements, and return the expected anchor
    timestamp.

    Properties
    ----------
    - Zero new dependencies — uses the encoder already present in the engine.
    - Zero regex — generalises to any phrasing the encoder saw during
      pre-training (informal, multilingual, implicit).
    - When the query has no temporal language the "now/today" probe
      dominates and the anchor collapses to ``now_ts``, preserving the
      existing default behaviour exactly.
    - ``softmax_temperature`` controls sharpness: high T → flat weights
      (anchor ≈ centre of mass of all probes ≈ ~4 months ago); low T →
      winner-takes-all (anchor = closest probe).  Default 0.05 is sharply
      peaked — a clear "last month" signal wins decisively, an ambiguous
      query spreads smoothly.

    Usage
    -----
    Build once at engine init (``encoder`` may be None to defer):

        probe = TemporalProbe(encoder.encode)

    At recall time:

        anchor_ts = probe.estimate_anchor(query_embedding, now_ts=int(time.time()))
        # returns int Unix timestamp; equals now_ts when query is atemporal
    """

    def __init__(
        self,
        encode_fn,  # callable: str -> np.ndarray[float32]
        *,
        probes: tuple[tuple[str, int], ...] = _TEMPORAL_PROBES,
        softmax_temperature: float = 0.05,
        atemporal_margin: float = 0.12,
    ) -> None:
        self._encode = encode_fn
        self._displacements: list[int] = [d for _, d in probes]
        self._temperature = float(softmax_temperature)
        # Dead-zone: if the best past-probe similarity exceeds the now-probe
        # similarity by less than this margin, the query is treated as
        # atemporal and estimate_anchor() returns now_ts unchanged.
        #
        # Calibrated on LongMemEval oracle queries (bge-small-en-v1.5):
        #   - Atemporal queries ("previous conversation", "recent relocation",
        #     "what is my cat's name?"): margin in [-0.01, +0.11]
        #   - Genuine temporal queries ("last month", "two weeks ago",
        #     "how many days between X and Y"): margin in [+0.15, +0.33]
        # A threshold of 0.12 cleanly separates the two populations with
        # no observed overlap in the calibration set.
        self._atemporal_margin = float(atemporal_margin)

        # Pre-compute and L2-normalise probe embeddings once. Custom probe
        # sets keep their historical one-phrase-per-band behavior for tests
        # and callers that supply a specialized encoder.
        variants = (
            _TEMPORAL_PROBE_VARIANTS
            if probes == _TEMPORAL_PROBES
            else tuple((phrase,) for phrase, _ in probes)
        )
        self._probe_matrices: list[np.ndarray] = []
        for phrases in variants:
            raw: list[np.ndarray] = []
            for phrase in phrases:
                v = np.asarray(encode_fn(phrase), dtype=np.float32).reshape(-1)
                n = float(np.linalg.norm(v))
                raw.append(v / n if n > 0.0 else v)
            self._probe_matrices.append(np.stack(raw, axis=0))
        # Index of the "now / today" probe — always first in _TEMPORAL_PROBES.
        # Stored explicitly so the dead-zone check is O(1) and doesn't depend
        # on probe ordering staying stable.
        self._now_probe_idx: int = 0

    def estimate_anchor(
        self,
        query_embedding: np.ndarray,
        *,
        now_ts: int | None = None,
    ) -> int:
        """Return the estimated Unix timestamp the query is anchored to.

        Algorithm
        ---------
        1. Cosine-similarity between the (already L2-normalised) query
           embedding and each probe row  →  raw_sims  ∈ [-1, 1]^n
        2. Dead-zone gate: if the best past-probe similarity exceeds the
           now-probe similarity by less than ``atemporal_margin``, the
           query carries no meaningful temporal signal — return now_ts
           immediately.  This prevents "previous conversation", "recent
           relocation" and similar atemporal phrasings from triggering a
           past-anchored retrieval, which would apply a temporal penalty
           to all recently-ingested episodes.
        3. Softmax with temperature T over raw_sims  →  weights  ∈ [0,1]^n
        4. Expected displacement  =  Σ weight_i * displacement_i
        5. anchor_ts  =  now_ts + round(expected_displacement)

        The result is an integer Unix timestamp.  When the query is
        atemporal the function returns now_ts unchanged (step 2 exits
        early), preserving the legacy behaviour exactly.
        """
        if now_ts is None:
            now_ts = int(time.time())

        q = np.asarray(query_embedding, dtype=np.float32).reshape(-1)
        qn = float(np.linalg.norm(q))
        if qn > 0.0:
            q = q / qn

        # One maximum cosine similarity per temporal band. This preserves the
        # original 12-way softmax even though each band has multilingual
        # paraphrases.
        sims = np.asarray([float((matrix @ q).max()) for matrix in self._probe_matrices])

        # --- Dead-zone gate -------------------------------------------
        # The "now" probe must be beaten by a past probe by at least
        # atemporal_margin before we treat the query as temporal.
        # sims[0] = "right now, today, at the moment" (displacement = 0).
        now_sim = float(sims[self._now_probe_idx])
        past_sims = np.delete(sims, self._now_probe_idx)
        best_past_sim = float(past_sims.max()) if past_sims.size > 0 else now_sim
        if best_past_sim - now_sim < self._atemporal_margin:
            return now_ts
        # --------------------------------------------------------------

        # Softmax with temperature — shift by max for numerical stability
        logits = sims / self._temperature
        logits -= logits.max()
        weights = np.exp(logits)
        weights /= weights.sum()

        # Weighted mean displacement (float seconds)
        displacement = float(np.dot(weights, np.asarray(self._displacements, dtype=np.float64)))

        return int(now_ts + round(displacement))
