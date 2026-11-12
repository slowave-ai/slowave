"""Cue splitting: explicit needs stay independent instead of being diluted."""

from __future__ import annotations

from slowave.core.applicability_ranking import query_needs


def test_explicit_conjunction_becomes_two_needs() -> None:
    needs = query_needs("What are the pre-commit checks and the correct base for a new branch?")

    assert len(needs) == 2


def test_small_noun_conjunction_stays_one_need() -> None:
    assert query_needs("Run the linter and typechecker") == ["Run the linter and typechecker"]


def test_newline_separated_cues_are_independent_needs() -> None:
    joined = "Dial in the espresso grinder\nRefresh the daily espresso recipe"

    assert len(query_needs(joined)) == 2


def test_need_count_is_bounded_and_deduplicated() -> None:
    lines = "\n".join(f"maintain workstation {number} backup" for number in range(40))

    needs = query_needs(lines)

    assert len(needs) <= 16
    assert len(needs) == len(set(needs))
