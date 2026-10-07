"""Pool-relative admission for the current retrieval policy.

Applicability assessment floors are pool-relative per need column
(``logit >= best - margin``) with a low absolute signal-present floor
(``logit >= best only when best itself clears the floor``); ordinary prose
tasks decompose into sentence-boundary facet needs; the E5 multilingual
discovery channel joins the discovery union and reports candidate-depth
truncation truthfully; consolidation-derived episode summaries without any
explicit source are excluded from declarative delivery.

Flooding safety: admission is anchored to the pool's own best score, so the
admitted window moves with the pool; the only flood-shaped failure is a
noise-dominated pool whose best clears the signal-present floor, which admits
at most the bounded ``[floor, best]`` window instead of the whole noise band.
"""

from __future__ import annotations

import os

SIGNAL_FLOOR_ENV = "SLOWAVE_APPLICABILITY_SIGNAL_FLOOR"
MARGIN_ENV = "SLOWAVE_APPLICABILITY_MARGIN"

# The signal-present floor is an abstention check, not a relevance decision.
# It must sit below the relevant-pair logit mass of every supported operating
# regime so a stored-once memory clears it, while an all-noise pool whose best
# is itself noise-level abstains or admits only the bounded tail.
_SIGNAL_FLOOR_DEFAULT = -6.5
# Relative margin above the pool's best per need column. 1.0 logit keeps the
# admitted set to the relevant cluster; a best-anchored window can never reach
# below the pool's own best, so it cannot admit the whole noise band.
_MARGIN_DEFAULT = 1.0


def signal_floor() -> float:
    value = float(os.environ.get(SIGNAL_FLOOR_ENV, str(_SIGNAL_FLOOR_DEFAULT)))
    if value != value or value in (float("inf"), float("-inf")):
        raise ValueError(f"{SIGNAL_FLOOR_ENV} must be finite")
    return value


def margin() -> float:
    value = float(os.environ.get(MARGIN_ENV, str(_MARGIN_DEFAULT)))
    if value != value or value < 0 or value == float("inf"):
        raise ValueError(f"{MARGIN_ENV} must be a finite non-negative number")
    return value
