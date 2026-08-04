"""Latent supersession direction via SVD1 of a multi-domain seed set.

The first right singular vector of the difference-vector matrix computed from
the seed pairs is the single direction in embedding space that best explains
the variance of concrete value-substitution supersession across domains.

Empirical finding (2026-06-19, paraphrase-multilingual-MiniLM-L12-v2, 104 pairs):
  - SVD1 axis gives sep(sup, add) = +0.35 vs +0.09 for mean centroid, +0.32 for cosine
  - Covers: tech, medical, business, financial, hr, legal, science, + multilingual
  - Personal preference domain is anti-aligned (−0.17); excluded from seed set

Usage in engine:
    manifold = SupersessionManifold(encoder)
    score = manifold.direction_score(emb_new, emb_old)
    if cosine(emb_new, emb_old) > 0.35 and score > 0.10:
        flag_needs_review()
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from slowave.symbolic.encoder import TextEncoder

# Seed pairs: (old_fact, new_fact) across 7 domains and 5 languages.
# Covers concrete value-substitution supersession only.
# Personal preference excluded — geometrically anti-aligned with SVD1.
_SEED_PAIRS: list[tuple[str, str]] = [
    # ── tech ─────────────────────────────────────────────────────────────────
    ("The project uses SQLite for storage.", "The project uses DuckDB for storage."),
    ("The backend is written in Python.", "The backend is written in Go."),
    ("CI/CD is handled by Jenkins.", "CI/CD is handled by GitHub Actions."),
    # ── medical ───────────────────────────────────────────────────────────────
    ("The patient takes metformin 500 mg daily.", "The patient takes metformin 1000 mg daily."),
    ("The patient is prescribed lisinopril.", "The patient is prescribed amlodipine."),
    # ── business ──────────────────────────────────────────────────────────────
    ("The account manager for Acme is Alice.", "The account manager for Acme is Bob."),
    ("The deal is in the negotiation stage.", "The deal is in the closing stage."),
    # ── financial ─────────────────────────────────────────────────────────────
    ("The project budget is $50,000.", "The project budget is $75,000."),
    ("The agreed hourly rate is $120.", "The agreed hourly rate is $150."),
    # ── hr ────────────────────────────────────────────────────────────────────
    ("Alice reports to John.", "Alice reports to Sarah."),
    ("The team lead is Marco.", "The team lead is Elena."),
    ("The engineering team has 5 members.", "The engineering team has 8 members."),
    # ── legal ─────────────────────────────────────────────────────────────────
    ("The NDA expires on January 1, 2025.", "The NDA expires on January 1, 2026."),
    ("The contract value is $200,000.", "The contract value is $250,000."),
    # ── science ───────────────────────────────────────────────────────────────
    ("The experiment runs at 37°C.", "The experiment runs at 42°C."),
    (
        "The study uses a sample size of 50 participants.",
        "The study uses a sample size of 100 participants.",
    ),
    # ── multilingual (IT/FR/DE) ───────────────────────────────────────────────
    ("Il progetto usa SQLite per lo storage.", "Il progetto usa DuckDB per lo storage."),
    ("Le projet utilise SQLite pour le stockage.", "Le projet utilise DuckDB pour le stockage."),
    (
        "Il paziente assume metformina 500 mg al giorno.",
        "Il paziente assume metformina 1000 mg al giorno.",
    ),
    ("Il budget del progetto è di 50.000 euro.", "Il budget del progetto è di 75.000 euro."),
    ("Alice riporta a Giovanni.", "Alice riporta a Sara."),
]

# All geometry thresholds for the supersession/reinforce/generalize decision tree.
# Calibrated on 104-pair test set (2026-06-19, paraphrase-multilingual-MiniLM-L12-v2).
#
# Cosine distribution by zone (186-pair eval set):
#   supersession  n=71  median=0.694  p75=0.800  p90=0.904  max=0.981
#   additive      n=17  median=0.295  p90=0.582  max=0.902
#   duplicate     n=6   min=0.822     median=0.952
#   unrelated     n=10  median=0.682  max=0.948   ← noise floor for same-domain facts
#
# The cosine thresholds below are *triage gates*, not supersession detectors.
# They admit candidates for direction_score evaluation. The bulk of supersession
# pairs (79% have cosine < 0.85) are handled by the consolidation path
# (GeometricContradictionJudge), not by remember().

# Minimum direction_score to classify a change as a value substitution (supersede).
# sep(sup, add) = +0.35 vs +0.09; mean(add) = −0.028 at this threshold.
DIRECTION_SCORE_THRESHOLD_SUPERSEDE: float = 0.10

# Lower bound of the ambiguous zone: direction_score in
# [DIRECTION_SCORE_THRESHOLD_REVIEW, DIRECTION_SCORE_THRESHOLD_SUPERSEDE)
# triggers needs_review instead of auto-action.
DIRECTION_SCORE_THRESHOLD_REVIEW: float = 0.05

# Minimum cosine for same-scope action (supersede or reinforce).
# Set at the duplicate-zone floor (min=0.822 rounded down) so only near-identical
# or clearly same-topic schemas trigger immediate action at remember() time.
COS_THRESHOLD_SAME_SCOPE: float = 0.85

# Extended same-scope supersession gate (Gap 3).
# At cos in [COS_THRESHOLD_EXTENDED_SAME_SCOPE, COS_THRESHOLD_SAME_SCOPE) only
# direction_score >= DIRECTION_SCORE_THRESHOLD_SUPERSEDE triggers supersession —
# no reinforce or needs_review, because the cosine signal alone is too weak to act on.
# Covers the 0.70–0.85 range that was previously ignored, catching cases like
# S-1/S-2 wiki scenarios (cos ~0.80) with clear value substitution direction.
COS_THRESHOLD_EXTENDED_SAME_SCOPE: float = 0.70

# Minimum cosine for cross-scope linking (generalization reinforcement).
# Motivated by empirical observation: cos=0.81 for Karpathy guidelines with minor
# framing variation across projects. 3pp buffer below observed gives 0.78.
# Cross-scope never supersedes — only reinforces + records evidence.
COS_THRESHOLD_CROSS_SCOPE: float = 0.78

COS_THRESHOLD_TOPICAL: float = 0.35


class SupersessionManifold:
    """SVD1 direction axis for latent supersession detection.

    Lazy-computed on first use. Consistent with whatever encoder is active —
    call invalidate() or create a new instance if the encoder changes.
    """

    def __init__(self, encoder: TextEncoder) -> None:
        self._encoder = encoder
        self._axis: np.ndarray | None = None

    def _compute(self) -> None:
        flat: list[str] = []
        for old, new in _SEED_PAIRS:
            flat.extend([old, new])
        all_embs = self._encoder.encode_many(flat)
        embs_old = all_embs[0::2]
        embs_new = all_embs[1::2]
        diffs = embs_new - embs_old

        # Normalise each diff vector to unit length before SVD so that all
        # seed pairs contribute equally regardless of raw diff magnitude.
        # Without this, one high-magnitude pair (e.g. SQLite→DuckDB) would
        # dominate SVD1 with a small seed set.
        norms = np.linalg.norm(diffs, axis=1, keepdims=True)
        diffs_n = diffs / (norms + 1e-8)

        _, _, Vt = np.linalg.svd(diffs_n, full_matrices=False)
        axis = Vt[0]

        # SVD sign is arbitrary. Fix: orient axis so the majority of seed
        # pairs have positive alignment (majority-vote convention).
        if float((diffs_n @ axis).mean()) < 0:
            axis = -axis

        self._axis = axis.astype(np.float32)

    @property
    def axis(self) -> np.ndarray:
        if self._axis is None:
            self._compute()
        return self._axis

    def invalidate(self) -> None:
        """Force recomputation on next use (call after encoder change)."""
        self._axis = None

    def direction_score(self, emb_new: np.ndarray, emb_old: np.ndarray) -> float:
        """Alignment of the change vector with the supersession direction.

        Returns a value in [-1, 1]. Positive = aligned with supersession direction.
        Near-zero diff vector (paraphrase) returns 0.0.

        Domains covered by this signal: tech, medical, business, financial,
        hr, legal, science (and their multilingual equivalents).
        Personal preference is anti-aligned — do not use this signal for it.
        """
        d = emb_new - emb_old
        norm = float(np.linalg.norm(d))
        if norm < 1e-8:
            return 0.0
        return float(np.dot(d / norm, self.axis))
