"""Unit tests for Slowave lifecycle classification.

``slowave/core/lifecycle.py`` decides whether an event is Slowave's own
lifecycle bookkeeping (never declarative memory) versus a memory of the
world/task. It is a narrow, server-owned-vocabulary detector: matching is
exact-after-normalization on canonical lifecycle phrases and server-owned
patterns, so genuine task observation text is never classified as lifecycle.
"""

from __future__ import annotations

from slowave.core.lifecycle import (
    _LIFECYCLE_PHRASES,
    is_slowave_lifecycle,
    normalize,
)
from slowave.ops import filter_lifecycle_trajectory


class TestIsSlowaveLifecycle:
    def test_context_query_type_is_lifecycle(self):
        assert is_slowave_lifecycle("context_query", "any task text")

    def test_task_complete_type_is_lifecycle(self):
        # task_complete is Slowave's own commit marker — never episodic, even
        # if a legacy event carries an embedding and no memory_role.
        assert is_slowave_lifecycle("task_complete", "outcome=success")
        assert is_slowave_lifecycle("task_complete", "outcome=partial")
        assert is_slowave_lifecycle("task_complete", "outcome=failure")

    def test_recall_cue_prefix_is_lifecycle(self):
        assert is_slowave_lifecycle(
            "trajectory:action", "slowave_recall: Explicit user preferences for an AI assistant"
        )

    def test_activate_prefix_is_lifecycle(self):
        assert is_slowave_lifecycle("trajectory:action", "slowave_activate: setup project")

    def test_canonical_lifecycle_trajectory_phrases(self):
        for phrase in _LIFECYCLE_PHRASES:
            # Case/whitespace/punctuation-insensitive and matches as a
            # trajectory entry regardless of exact casing.
            assert is_slowave_lifecycle("trajectory:action", phrase.title() + ".")
            assert is_slowave_lifecycle("trajectory:observation", phrase)

    def test_genuine_task_text_is_not_lifecycle(self):
        # Content that merely *mentions* Slowave's tool/session is a task
        # observation and must remain eligible for episodic formation.
        assert not is_slowave_lifecycle(
            "trajectory:action", "Refactored slowave_recall to return evidence"
        )
        assert not is_slowave_lifecycle(
            "trajectory:action", "Fixed the slowave session resolver timeout"
        )
        assert not is_slowave_lifecycle(
            "trajectory:observation", "Ran the focused test and it passed"
        )
        assert not is_slowave_lifecycle(
            "trajectory:action", "Reviewed commits from the last two days"
        )
        assert not is_slowave_lifecycle("remember:fact", "backend uses retention")

    def test_empty_content_not_lifecycle(self):
        assert not is_slowave_lifecycle("trajectory:action", "")

    def test_remember_and_commit_tool_prefixes(self):
        assert is_slowave_lifecycle("trajectory:action", "slowave_remember: a fact")
        assert is_slowave_lifecycle("trajectory:action", "slowave_feedback: retrieval")
        assert is_slowave_lifecycle("trajectory:action", "slowave_commit: task")


class TestNormalize:
    def test_case_whitespace_punctuation(self):
        assert normalize("  Activated   the Slowave session.  ") == "activated the slowave session"
        assert normalize("Committed the session!") == "committed the session"


class TestFilterLifecycleTrajectory:
    def test_drops_lifecycle_keeps_task(self):
        trajectory = [
            {"kind": "action", "summary": "Activated the Slowave session.", "status": "succeeded"},
            {"kind": "action", "summary": "Added conditional Labs UI", "status": "succeeded"},
            {"kind": "observation", "summary": "Ran the focused test"},
        ]
        kept, dropped = filter_lifecycle_trajectory(trajectory)
        assert dropped == 1
        assert [t["summary"] for t in kept] == ["Added conditional Labs UI", "Ran the focused test"]

    def test_empty_in_empty_out(self):
        assert filter_lifecycle_trajectory([]) == ([], 0)
