"""Shared-v1 admission floors pinned to the measured live operating point.

The shipped defaults are the live-validated operating point (activate -1.25,
recall -2), measured on a read-only production-scale snapshot
(private/audits/20261006_memory_recovery/wp4/). The corpus-v1 calibration
(-6.05 both endpoints, commit a38e6c7) is REJECTED by live evidence: at
production scale the ~110-candidate discovery pool sits almost entirely in
the [-6, -1.25] logit band, so -6.05 flooded the client with 43-106 admitted
memories per unrelated request (RESULTS §12). These tests pin the defaults
and the product behaviors they preserve: a genuinely applicable memory whose
production-style logit sits below the legacy floor resurfaces, and a
genuinely unrelated request still returns nothing.
"""

import numpy as np
import pytest

from slowave import ops
from slowave.core.config import SlowaveConfig
from slowave.core.engine import SlowaveEngine
from slowave.core.shared_retrieval import (
    _ACTIVATE_LOGIT_DEFAULT,
    _RECALL_LOGIT_DEFAULT,
    SharedRequest,
)


def test_defaults_pin_the_measured_live_operating_point():
    assert _ACTIVATE_LOGIT_DEFAULT == pytest.approx(-1.25)
    assert _RECALL_LOGIT_DEFAULT == pytest.approx(-2.0)
    request = SharedRequest("Prepare this change for delivery")
    assert request.threshold == pytest.approx(_ACTIVATE_LOGIT_DEFAULT)
    assert SharedRequest("task", endpoint="recall").threshold == pytest.approx(
        _RECALL_LOGIT_DEFAULT
    )
    # Activation must never be looser than recall (enforced invariant).
    assert SharedRequest("task").threshold >= SharedRequest("task", endpoint="recall").threshold


def test_default_floors_reject_the_measured_negative_pair():
    """The ocean-poem negative case stays empty at the live operating point.

    Corpus v1's best negative pair (poem task x release-checks memory)
    measures -6.0816 logits through the real pipeline (probe_pipeline.json);
    every shipped default must sit above it. This holds a fortiori at the
    live operating point (-1.25), which is far above the corpus-v1 plateau
    whose -6.05 midpoint was rejected at production scale.
    """
    assert -6.0816 < _ACTIVATE_LOGIT_DEFAULT


def _stub_engine(tmp_path, logit: float):
    class Encoder:
        def encode(self, text):
            return np.zeros(8, dtype=np.float32)

    class Scorer:
        def score(self, queries, memories):
            return np.full((len(memories), 1), logit)

    engine = SlowaveEngine(
        SlowaveConfig(db_path=str(tmp_path / "calibrated.db"), dim=8, disable_encoder=True)
    )
    engine._retrieval.encoder = Encoder()
    engine._retrieval._applicability_scorer = Scorer()
    engine.schemas.search_fts_candidates = lambda *args, **kwargs: []
    return engine


def _seed_memory(engine) -> int:
    return engine.schemas.create(
        content_text="For this repository the release checks are: run black and isort, then mypy, before any commit is accepted.",
        facets={},
        embedding=None,
        scope_id="project:test",
    )


def test_fresh_memory_below_legacy_floor_resurfaces_on_differently_worded_task(
    tmp_path, monkeypatch
):
    """The core product promise: stored once, resurfaces without re-wording.

    A scorer logit of -1.0 was inadmissible under the legacy default floor
    (0) yet must be admitted at the live operating point (-1.25): it models
    production scoring of genuinely applicable task-vs-rule pairs at live
    scale, where the measured pool sits in the [-6, -1.25] band and the
    -1.25 floor keeps the applicable head of that band admissible.
    """
    monkeypatch.setenv("SLOWAVE_RETRIEVAL_PIPELINE", "shared-v1")
    engine = _stub_engine(tmp_path, logit=-1.0)
    try:
        memory = _seed_memory(engine)
        result = ops.activate(
            engine,
            query="Prepare this change for delivery",
            initial_goal="get the change ready to ship",
            scope="project:test",
            relevant_set=True,
            include_schemas=True,
        )
        assert result["relevant_total"] == 1
        assert [row["id"] for row in result["schemas"]] == [f"sch_{memory}"]
    finally:
        engine.close()


def test_unrelated_request_still_returns_nothing(tmp_path, monkeypatch):
    monkeypatch.setenv("SLOWAVE_RETRIEVAL_PIPELINE", "shared-v1")
    engine = _stub_engine(tmp_path, logit=-7.0)
    try:
        _seed_memory(engine)
        result = ops.activate(
            engine,
            query="Write a poem about the ocean at night",
            initial_goal="draft some poetry",
            scope="project:test",
            relevant_set=True,
            include_schemas=True,
        )
        assert result["relevant_total"] == 0
        assert result["schemas"] == []
    finally:
        engine.close()


def test_env_overrides_remain_authoritative(monkeypatch):
    monkeypatch.setenv("SLOWAVE_ACTIVATE_RELEVANCE_LOGIT", "-1")
    assert SharedRequest("task").threshold == pytest.approx(-1.0)
    monkeypatch.setenv("SLOWAVE_RECALL_RELEVANCE_LOGIT", "-3")
    assert SharedRequest("task", endpoint="recall").threshold == pytest.approx(-3.0)


def test_activate_below_calibrated_recall_default_is_rejected(monkeypatch):
    """The ordering invariant is enforced against the calibrated defaults."""
    monkeypatch.setenv("SLOWAVE_ACTIVATE_RELEVANCE_LOGIT", "-7")
    with pytest.raises(ValueError):
        _ = SharedRequest("task").threshold
