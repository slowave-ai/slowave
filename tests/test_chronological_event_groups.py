import numpy as np

from tests.benchmarks.chronological_event_groups import (
    EventGroup,
    select_groups,
    source_date_order,
)


def test_select_groups_ranks_by_best_turn_but_renders_chronologically() -> None:
    groups = [
        EventGroup("early", "2024-01-01", (0, 0), (("a", "early"),)),
        EventGroup("middle", "2024-01-02", (0, 1), (("b", "middle"),)),
        EventGroup("late", "2024-01-03", (0, 2), (("c", "late"),)),
    ]
    vectors = np.array([[0.8, 0.0], [0.1, 0.0], [0.9, 0.0]])

    evidence, selected = select_groups(groups, vectors, np.array([1.0, 0.0]), 200)

    assert selected == ["early", "middle", "late"]
    assert evidence.index("early") < evidence.index("middle") < evidence.index("late")


def test_select_groups_never_splits_group_or_exceeds_budget() -> None:
    groups = [
        EventGroup("large", "2024-01-01", (0, 0), (("a", "x" * 100),)),
        EventGroup("small", "2024-01-02", (0, 1), (("b", "useful"),)),
    ]
    vectors = np.array([[0.9, 0.0], [0.8, 0.0]])

    evidence, selected = select_groups(groups, vectors, np.array([1.0, 0.0]), 60)

    assert selected == ["small"]
    assert "large" not in evidence
    assert len(evidence) <= 60


def test_source_date_order_handles_both_benchmark_formats() -> None:
    early = source_date_order("2023/04/10 (Mon) 14:47", 1)
    late = source_date_order("1:56 pm on 8 May, 2023", 0)

    assert early < late
