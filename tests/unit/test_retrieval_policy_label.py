"""The recorded retrieval policy must name the rule that actually ranked items.

Behaviour changes used to be invisible: the calibrated admission could rank a
response while the response was still labelled with the pool-relative rule, and a
fallback to that rule produced the same label. These tests pin the identity and
the fallback label.
"""

from __future__ import annotations

from slowave.mcp.activation_catalog import (
    CALIBRATED_POLICY_VERSION,
    CALIBRATED_RECALL_POLICY_VERSION,
    SUPPORTED_POLICIES,
)
from slowave.ops import _retrieval_policy_version


def _defaults(monkeypatch) -> None:
    """Pin the experimental pipeline switches to their shipped defaults."""
    monkeypatch.setenv("SLOWAVE_RETRIEVAL_BASELINE", "0")
    monkeypatch.setenv("SLOWAVE_RETRIEVAL_PIPELINE", "current")
    monkeypatch.setenv("SLOWAVE_ACTIVATE_HYBRID", "0")


def test_calibrated_admission_keeps_its_own_identity(monkeypatch) -> None:
    _defaults(monkeypatch)
    assert (
        _retrieval_policy_version(
            relevant_set=True,
            complementary_activation=True,
            facet_mode=False,
            admission_status="applied",
        )
        == CALIBRATED_POLICY_VERSION
    )
    assert (
        _retrieval_policy_version(
            relevant_set=True,
            complementary_activation=False,
            facet_mode=False,
            recall=True,
            admission_status="applied",
        )
        == CALIBRATED_RECALL_POLICY_VERSION
    )


def test_fallback_keeps_the_historical_identity(monkeypatch) -> None:
    _defaults(monkeypatch)
    for status in ("no_model_encoder", "disabled", "empty", None):
        assert (
            _retrieval_policy_version(
                relevant_set=True,
                complementary_activation=True,
                facet_mode=False,
                admission_status=status,
            )
            == "activation-pool-relative-v1"
        )


def test_both_identities_are_servable() -> None:
    assert {CALIBRATED_POLICY_VERSION, CALIBRATED_RECALL_POLICY_VERSION} <= SUPPORTED_POLICIES
