"""Fitted constants for admission v2 (do not edit by hand; see admission_v2.py).

Logistic regression, L2=3.0, fitted offline on labelled dev corpora (task-specific rows; standing-rule
memories excluded), with leave-one-scope-out validation. Shipped as constants; never trained at runtime.
"""

from __future__ import annotations

MODEL_VERSION = "admission-v2-1"
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
CHEAP_W = [
    2.844055,
    -0.22768,
    0.977615,
    0.557595,
    0.882313,
    -0.647871,
    2.043762,
    -0.11299,
    2.537604,
    0.453348,
    0.111952,
    -0.403138,
    0.495649,
]
CHEAP_B = -3.18552
TAU = 0.3
TAU_BEST = 0.5
MAX_ITEMS = 1000000
BYTE_BUDGET = 1000000000
