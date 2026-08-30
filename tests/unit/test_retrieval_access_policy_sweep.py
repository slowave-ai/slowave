"""Regression coverage for the bounded Phase-2 policy sweep."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from private.experiments.run_retrieval_access_policy_sweep import POLICIES, _summary


def test_phase_two_sweep_keeps_the_declared_parameter_set_bounded() -> None:
    assert set(POLICIES) == {
        "conservative_two_marks",
        "lower_cue_match",
        "stronger_inhibition",
        "one_mark_rejected",
    }
    assert POLICIES["one_mark_rejected"].negative_evidence_threshold == 1
    assert POLICIES["conservative_two_marks"].negative_evidence_threshold == 2


def test_sweep_summary_reports_paired_repeat_offender_metrics() -> None:
    summary = _summary(
        [
            {
                "metrics": {
                    "repeat_opportunities": 2,
                    "repeat_offender_hits": 2,
                    "shadow_repeat_offender_hits": 1,
                    "false_inhibition_rate": 0.0,
                    "recovery_success_rate": None,
                }
            },
            {
                "metrics": {
                    "repeat_opportunities": 1,
                    "repeat_offender_hits": 0,
                    "shadow_repeat_offender_hits": 0,
                    "false_inhibition_rate": None,
                    "recovery_success_rate": 1.0,
                }
            },
        ]
    )
    assert summary["paired_repeat_offender_delta"] == -1
    assert summary["shadow_repeat_offender_rate"] == 0.3333
    assert summary["false_inhibition_rate"] == 0.0
    assert summary["recovery_success_rate"] == 1.0


def test_one_mark_policy_is_present_only_as_an_explicitly_rejected_control() -> None:
    assert POLICIES["one_mark_rejected"].negative_evidence_threshold == 1
    assert all(
        policy.negative_evidence_threshold >= 2
        for policy_id, policy in POLICIES.items()
        if policy_id != "one_mark_rejected"
    )
