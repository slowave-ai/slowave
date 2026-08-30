"""Contract tests for retrieval-access lifecycle replay fixtures."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from private.experiments.replay_corpus import ACCESS_LIFECYCLE_SEQUENCES


def test_access_lifecycle_sequences_cover_required_contracts() -> None:
    sequences = {sequence.sequence_id: sequence for sequence in ACCESS_LIFECYCLE_SEQUENCES}
    assert set(sequences) == {
        "high_cosine_repeat_offender",
        "alternate_cue_is_not_poisoned",
        "pathway_isolation",
        "strong_direct_override",
        "explicit_use_recovers_inhibition",
        "passive_exposure_does_not_recover",
        "chronic_zero_positive_retirement_candidate",
    }
    assert {step.pathway for step in sequences["pathway_isolation"].steps} == {
        "direct",
        "graph",
        "exploration",
    }
    assert sequences[
        "chronic_zero_positive_retirement_candidate"
    ].future_retirement_candidate_ids == (2,)
    for sequence in ACCESS_LIFECYCLE_SEQUENCES:
        assert sequence.schemas
        assert sequence.steps
        for step in sequence.steps:
            assert step.query
            assert step.shown_ids == tuple(step.shown_ids)
            assert not step.feedback_ids or step.feedback is not None


def test_access_lifecycle_labels_are_declared_before_runner_execution() -> None:
    for sequence in ACCESS_LIFECYCLE_SEQUENCES:
        for step in sequence.steps:
            if step.feedback == "irrelevant":
                assert step.feedback_ids
            if step.expected_suppressed_ids or step.expected_recovery_ids:
                assert step.note == "" or isinstance(step.note, str)
