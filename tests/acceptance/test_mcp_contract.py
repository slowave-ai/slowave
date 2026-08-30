"""MCP lifecycle contract: invalid requests fail safely and valid work still completes."""

from __future__ import annotations

import asyncio
from pathlib import Path

from tests.acceptance.mcp_harness import (
    assert_acceptance_mutation_is_caught,
    open_harness,
)


def _run(coro) -> None:
    asyncio.run(coro)


def _assert_error(payload: dict, code: str, message_fragment: str) -> None:
    assert payload["ok"] is False
    assert payload["error"]["code"] == code
    assert message_fragment in payload["error"]["message"]


def test_remember_rejects_a_session_from_another_scope_without_closing_it(tmp_path: Path) -> None:
    """A client cannot write into project:beta using a project:alpha session."""

    async def scenario() -> None:
        async with open_harness(tmp_path / "scope-mismatch.db") as harness:
            activation, _ = await harness.activate(
                "scope_mismatch",
                "Record the Alpha deployment policy",
                "store Alpha deployment policy",
                "project:alpha",
            )
            rejected, _ = await harness.raw_call(
                "slowave_remember",
                {
                    "content": "Project Beta deploys from a separate release branch.",
                    "type": "fact",
                    "scope": "project:beta",
                    "session_id": activation["session_id"],
                },
            )
            _assert_error(rejected, "invalid_input", "session_id and scope do not match")

            # The rejected write must not poison the active, correctly scoped session.
            memory_id = await harness.remember(
                "Project Alpha deploys from the protected release branch.",
                "fact",
                activation["session_id"],
                "project:alpha",
            )
            await harness.feedback_all(activation)
            await harness.commit(activation["session_id"], "store Alpha deployment policy")

            later, _ = await harness.activate(
                "scope_mismatch_later",
                "Which branch does Project Alpha deploy from?",
                "retrieve Alpha deployment policy",
                "project:alpha",
            )
            assert [memory["memory_id"] for memory in later["memories"]] == [memory_id]
            await harness.feedback_all(later, used_ids={memory_id})
            await harness.commit(later["session_id"], "retrieve Alpha deployment policy")

    _run(scenario())


def test_remember_rejects_an_ambiguous_source_time_and_accepts_a_valid_one(tmp_path: Path) -> None:
    """Clients must supply an offset when they claim an event happened in the past."""

    async def scenario() -> None:
        async with open_harness(tmp_path / "source-time-validation.db") as harness:
            activation, _ = await harness.activate(
                "source_time_validation",
                "Record a completed incident",
                "store incident history",
                "project:payments",
            )
            rejected, _ = await harness.raw_call(
                "slowave_remember",
                {
                    "content": "The payments incident began at 14:05.",
                    "type": "fact",
                    "occurred_at": "2026-08-19T14:05:00",
                    "scope": "project:payments",
                    "session_id": activation["session_id"],
                },
            )
            _assert_error(rejected, "invalid_input", "UTC offset")

            memory_id = await harness.remember(
                "The payments incident began after the certificate expired.",
                "fact",
                activation["session_id"],
                "project:payments",
                occurred_at="2026-08-19T14:05:00Z",
            )
            await harness.feedback_all(activation)
            await harness.commit(activation["session_id"], "store incident history")

            later, _ = await harness.activate(
                "source_time_validation_later",
                "When did the payments incident begin?",
                "retrieve incident history",
                "project:payments",
            )
            assert [memory["memory_id"] for memory in later["memories"]] == [memory_id]
            await harness.feedback_all(later, used_ids={memory_id})
            await harness.commit(later["session_id"], "retrieve incident history")

    _run(scenario())


def test_commit_requires_complete_feedback_for_every_returned_memory(tmp_path: Path) -> None:
    """A client cannot close a retrieval session while silently ignoring returned memory."""

    async def scenario() -> None:
        fact = "The support refund window is fourteen days."
        async with open_harness(tmp_path / "feedback-completeness.db") as harness:
            seed, _ = await harness.activate(
                "feedback_seed",
                "Store the current refund policy",
                "store current refund policy",
                "project:support",
            )
            memory_id = await harness.remember(
                fact, "decision", seed["session_id"], "project:support"
            )
            await harness.feedback_all(seed)
            await harness.commit(seed["session_id"], "store current refund policy")

            retrieval, _ = await harness.activate(
                "feedback_required",
                "What is the current refund window?",
                "retrieve current refund policy",
                "project:support",
            )
            assert [memory["memory_id"] for memory in retrieval["memories"]] == [memory_id]
            # Partial coverage records what the client knows so far, but must
            # not permit the session to close while any exposed target is
            # still unassessed.
            partial, _ = await harness.raw_call(
                "slowave_feedback",
                {
                    "retrieval_id": retrieval["retrieval_id"],
                    "memory_feedback": [{"memory_id": memory_id, "assessment": "used"}],
                    "coverage": "partial",
                },
            )
            assert partial["ok"] is True
            rejected, _ = await harness.raw_call(
                "slowave_commit",
                {
                    "session_id": retrieval["session_id"],
                    "final_goal": "retrieve current refund policy",
                    "outcome": "success",
                    "outcome_summary": "The policy was retrieved.",
                    "verification": {"status": "verified", "summary": "The policy is visible."},
                },
            )
            _assert_error(rejected, "incomplete_feedback", "feedback is incomplete")
            outstanding = rejected["error"]["outstanding"]
            assert len(outstanding) == 1
            assert outstanding[0]["memory_ids"] == []

            await harness.feedback_all(retrieval, used_ids={memory_id})
            await harness.commit(retrieval["session_id"], "retrieve current refund policy")

    _run(scenario())


def test_feedback_rejects_a_memory_that_the_client_was_not_shown(tmp_path: Path) -> None:
    """Feedback is authorized by the retrieval result, not by a guessed memory identifier."""

    async def scenario() -> None:
        fact = "The support refund window is fourteen days."
        async with open_harness(tmp_path / "feedback-authorization.db") as harness:
            seed, _ = await harness.activate(
                "feedback_authorization_seed",
                "Store the current refund policy",
                "store current refund policy",
                "project:support",
            )
            memory_id = await harness.remember(
                fact, "decision", seed["session_id"], "project:support"
            )
            await harness.feedback_all(seed)
            await harness.commit(seed["session_id"], "store current refund policy")

            retrieval, _ = await harness.activate(
                "feedback_authorization_retrieval",
                "What is the current refund window?",
                "retrieve current refund policy",
                "project:support",
            )
            assert [memory["memory_id"] for memory in retrieval["memories"]] == [memory_id]
            rejected, _ = await harness.raw_call(
                "slowave_feedback",
                {
                    "retrieval_id": retrieval["retrieval_id"],
                    "memory_feedback": [{"memory_id": "sch_not_returned", "assessment": "stale"}],
                    "coverage": "complete",
                },
            )
            assert rejected["ok"] is True
            assert {"target_id": "sch_not_returned", "reason": "target_not_exposed"} in rejected[
                "data"
            ]["rejected"]

            await harness.feedback_all(retrieval, used_ids={memory_id})
            await harness.commit(retrieval["session_id"], "retrieve current refund policy")

    _run(scenario())


def test_feedback_rejects_legacy_truth_aliases(tmp_path: Path) -> None:
    """Canonical MCP feedback accepts only used/irrelevant/stale."""

    async def scenario() -> None:
        async with open_harness(tmp_path / "feedback-aliases.db") as harness:
            retrieval, _ = await harness.activate(
                "feedback_aliases", "Retrieve a fact", "verify feedback contract", "project:test"
            )
            memory_id = await harness.remember(
                "A fact for alias validation", "fact", retrieval["session_id"], "project:test"
            )
            await harness.feedback_all(retrieval)
            shown, _ = await harness.recall(
                "feedback alias validation",
                "A fact for alias validation",
                retrieval["session_id"],
                "project:test",
            )
            rejected, _ = await harness.raw_call(
                "slowave_feedback",
                {
                    "retrieval_id": shown["retrieval_id"],
                    "memory_feedback": [{"memory_id": memory_id, "assessment": "wrong"}],
                },
            )
            assert rejected["ok"] is True
            assert {"target_id": memory_id, "reason": "invalid_memory_assessment"} in rejected[
                "data"
            ]["rejected"]
            await harness.feedback_all(shown, used_ids={memory_id})
            await harness.commit(retrieval["session_id"], "verify feedback contract")

    _run(scenario())


def test_feedback_enforcement_mutation_fails_the_complete_feedback_contract() -> None:
    """Disabling commit feedback enforcement must make this contract fail."""
    assert_acceptance_mutation_is_caught(
        "feedback_enforcement",
        "tests/acceptance/test_mcp_contract.py::test_commit_requires_complete_feedback_for_every_returned_memory",
    )
