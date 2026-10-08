"""MCP lifecycle contract: invalid requests fail safely and valid work still completes."""

from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path

from slowave.dashboard.app import _retrievals_payload
from slowave.mcp.activation_catalog import DEFAULT_MEMORY_PAGE_SIZE
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


def test_commit_rejects_malformed_nested_payload_with_structured_error(tmp_path: Path) -> None:
    """Schema violations retain Slowave's normal structured error envelope."""

    async def scenario() -> None:
        async with open_harness(tmp_path / "commit-schema-validation.db") as harness:
            activation, _ = await harness.activate(
                "commit_schema_validation",
                "Validate the commit payload contract.",
                "validate commit payload contract",
                "project:contract",
            )
            rejected, _ = await harness.raw_call(
                "slowave_commit",
                {
                    "session_id": activation["session_id"],
                    "final_goal": "validate commit payload contract",
                    "outcome": "success",
                    "outcome_summary": "The payload was validated.",
                    "verification": {"status": "verified", "summary": "Schema inspected."},
                    "procedure": {
                        "summary": "Inspect a schema",
                        "steps": ["This must be an object, not a string."],
                    },
                },
            )
            _assert_error(rejected, "invalid_input", "procedure.steps[0]")
            assert rejected["error"]["field_errors"] == [
                {
                    "path": "procedure.steps[0]",
                    "message": "Input should be a valid dictionary or instance of CommitProcedureStep",
                }
            ]

            # Validation must happen before any commit side effect; the caller
            # can correct the payload and close this same active session.
            await harness.feedback_all(activation)
            await harness.commit(activation["session_id"], "validate commit payload contract")

    _run(scenario())


def test_commit_accepts_a_valid_procedure_deduplication_key(tmp_path: Path) -> None:
    async def scenario() -> None:
        async with open_harness(tmp_path / "commit-deduplication-key.db") as harness:
            activation, _ = await harness.activate(
                "commit_deduplication_key",
                "Record the checkout repair procedure.",
                "record checkout repair procedure",
                "project:contract",
            )
            await harness.feedback_all(activation)
            result, _ = await harness.raw_call(
                "slowave_commit",
                {
                    "session_id": activation["session_id"],
                    "final_goal": "record checkout repair procedure",
                    "outcome": "success",
                    "outcome_summary": "The repair procedure was recorded.",
                    "verification": {"status": "verified", "summary": "Procedure tested."},
                    "procedure": {
                        "summary": "Repair checkout and verify recovery.",
                        "deduplication_key": "checkout_repair",
                        "steps": [{"summary": "Inspect checkout."}],
                    },
                },
            )
            assert result["ok"] is True

    _run(scenario())


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
            assert {
                "target_id": "sch_not_returned",
                "reason": "target_not_exposed",
                "field": "target_id",
                "hint": "Use an assessed target ID returned by this retrieval.",
            } in rejected["data"]["rejected"]

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
            assert {
                "target_id": memory_id,
                "reason": "invalid_memory_assessment",
                "field": "assessment",
                "hint": "Use used|not_used|unassessable|irrelevant|already_known|stale.",
            } in rejected["data"]["rejected"]
            await harness.feedback_all(shown, used_ids={memory_id})
            await harness.commit(retrieval["session_id"], "verify feedback contract")

    _run(scenario())


def test_recall_continuation_is_frozen_scoped_and_orientation_only(tmp_path: Path) -> None:
    async def scenario() -> None:
        scope = "project:billing"
        async with open_harness(tmp_path / "continuation.db") as harness:
            activation, _ = await harness.activate(
                "continuation_seed",
                "Store billing policies",
                "store billing policies",
                scope,
            )
            for index in range(40):
                await harness.remember(
                    f"Billing policy section {index} uses ledger code {1000 + index}.",
                    "fact",
                    activation["session_id"],
                    scope,
                )

            first, _ = await harness.recall(
                "continuation_first",
                "Which billing policy sections use ledger codes?",
                activation["session_id"],
                scope,
            )
            assert len(first["memories"]) == 5
            assert first["more_available"] is True
            assert first["continue_from"].startswith("cur_")
            assert set(first["accessible_field"]) == {
                "extra_candidates",
                "kinds",
                "approx_extra_tokens",
            }
            assert "content" not in str(first["accessible_field"]).lower()

            arguments = {
                "continue_from": first["continue_from"],
                "session_id": activation["session_id"],
                "scope": scope,
            }
            repeated_a, _ = await harness.call("slowave_recall", arguments)
            repeated_b, _ = await harness.call("slowave_recall", arguments)
            assert repeated_a == repeated_b

            seen = [item["memory_id"] for item in first["memories"]]
            page = repeated_a
            while True:
                seen.extend(item["memory_id"] for item in page["memories"])
                if not page["more_available"]:
                    break
                page, _ = await harness.call(
                    "slowave_recall",
                    {
                        "continue_from": page["continue_from"],
                        "session_id": activation["session_id"],
                        "scope": scope,
                    },
                )
            assert len(seen) == len(set(seen))
            assert len(seen) == 40
            assert page["accessible_field"] == {}

            wrong_scope, _ = await harness.raw_call(
                "slowave_recall",
                {
                    "continue_from": first["continue_from"],
                    "session_id": activation["session_id"],
                    "scope": "project:other",
                },
            )
            _assert_error(wrong_scope, "invalid_input", "does not match")

    _run(scenario())


def test_activation_pages_one_hundred_explicit_relevant_memories(tmp_path: Path) -> None:
    """The complete catalog is reachable through payload-sized pages."""

    async def scenario() -> None:
        scope = "project:catalog"

        def marker(index: int) -> str:
            # Alphabetic terms stay distinct under SQLite's default tokenizer.
            value = index
            chars = []
            while True:
                chars.append(chr(ord("a") + value % 26))
                value //= 26
                if value == 0:
                    return "marker" + "".join(reversed(chars))

        async with open_harness(tmp_path / "catalog-100.db") as harness:
            seed, _ = await harness.activate(
                "catalog_seed", "Seed catalog fixture", "seed catalog fixture", scope
            )
            memories = [
                {
                    "content": f"Catalog requirement {index}: preserve {marker(index)} with a timeout of {index + 10} seconds.",
                    "type": "fact",
                }
                for index in range(100)
            ]
            stored_ids = await harness.remember_batch(memories, seed["session_id"], scope)
            assert len(set(stored_ids)) == 100
            await harness.feedback_all(seed)
            await harness.commit(seed["session_id"], "seed catalog fixture")

            task = "Review all catalog requirements:\n" + "\n".join(
                f"{index + 1}. preserve {marker(index)}" for index in range(100)
            )
            first, _ = await harness.activate(
                "catalog_100", task, "recover every catalog requirement", scope
            )
            assert first["retrieval_policy_version"] == "activation-pool-relative-v1"
            assert first["relevant_total"] == 100
            # 101 task cues exceed the bounded fan-out cap, so the catalog
            # honestly reports that independent per-need searching may have
            # hidden qualifying candidates; the union still recovered all 100.
            assert first["catalog_truncated"] is True
            assert len(first["memories"]) == 5
            assert first["more_available"] is True

            def displayed_count():
                rows = _retrievals_payload(str(tmp_path / "catalog-100.db"), {})["retrievals"]
                row = next(r for r in rows if r["context_id"] == first["retrieval_id"])
                return row["memory_count"]

            seen = [item["memory_id"] for item in first["memories"]]
            assert displayed_count() == len(seen) < 100
            page = first
            while page["more_available"]:
                input_cursor = page["continue_from"]
                page, _ = await harness.call(
                    "slowave_recall",
                    {
                        "continue_from": page["continue_from"],
                        "session_id": first["session_id"],
                        "scope": scope,
                    },
                )
                seen.extend(item["memory_id"] for item in page["memories"])
                assert len(page["memories"]) == 5
                assert displayed_count() == len(set(seen))
                replay, _ = await harness.call(
                    "slowave_recall",
                    {
                        "continue_from": input_cursor,
                        "session_id": first["session_id"],
                        "scope": scope,
                    },
                )
                assert replay == page
                assert displayed_count() == len(set(seen))
            assert len(seen) == 100
            assert len(seen) == len(set(seen))
            assert page["more_available"] is False
            assert page["accessible_field"] == {}

    _run(scenario())


def test_feedback_enforcement_mutation_fails_the_complete_feedback_contract() -> None:
    """Disabling commit feedback enforcement must make this contract fail."""
    assert_acceptance_mutation_is_caught(
        "feedback_enforcement",
        "tests/acceptance/test_mcp_contract.py::test_commit_requires_complete_feedback_for_every_returned_memory",
    )


def test_complementary_activation_delivers_twelve_memories_over_ten_item_pages(
    tmp_path: Path,
) -> None:
    """A ten-item page never caps the complete relevance-qualified catalog."""

    async def scenario() -> None:
        scope = "project:complementary-public"
        markers = [f"marker{chr(ord('a') + i)}" for i in range(12)]
        task = (
            "Prepare the Atlas deployment and preserve every listed validation marker:\n"
            + "\n".join(
                f"{index + 1}. preserve {marker} for validation"
                for index, marker in enumerate(markers)
            )
        )
        db_path = tmp_path / "complementary-public.db"
        async with open_harness(db_path) as harness:
            seed, _ = await harness.activate(
                "complementary_seed", "Seed activation facts", "seed activation facts", scope
            )
            with sqlite3.connect(db_path) as conn:
                assert conn.execute(
                    "SELECT requested_page_size FROM context_recall_events WHERE context_id=?",
                    (seed["retrieval_id"],),
                ).fetchone() == (DEFAULT_MEMORY_PAGE_SIZE,)
            await harness.remember_batch(
                [
                    {
                        "content": f"Atlas validation must preserve {marker} before deployment.",
                        "type": "fact",
                    }
                    for marker in markers
                ],
                seed["session_id"],
                scope,
            )
            await harness.feedback_all(seed)
            await harness.commit(seed["session_id"], "seed activation facts")

            first, _ = await harness.call(
                "slowave_activate",
                {
                    "task": task,
                    "initial_goal": "Prepare the Atlas deployment",
                    "scope": scope,
                    "page_size": 10,
                },
            )
            assert first["retrieval_policy_version"] == "activation-pool-relative-v1"
            assert first["relevant_total"] == 12
            assert len(first["memories"]) == 10
            assert first["more_available"] is True
            with sqlite3.connect(db_path) as conn:
                assert conn.execute(
                    "SELECT requested_page_size FROM context_recall_events WHERE context_id=?",
                    (first["retrieval_id"],),
                ).fetchone() == (10,)
            tail, _ = await harness.call(
                "slowave_recall",
                {
                    "session_id": first["session_id"],
                    "scope": scope,
                    "continue_from": first["continue_from"],
                },
            )
            assert len(tail["memories"]) == 2
            assert tail["more_available"] is False
            assert "continue_from" not in tail
            memories = first["memories"] + tail["memories"]
            assert len({item["memory_id"] for item in memories}) == 12
            first_content = " ".join(item["content"] for item in memories)
            assert sum(marker in first_content for marker in markers) == 12

            # Fresh recall can choose a smaller bound; unseen continuation
            # targets create no feedback obligation until actually returned.
            small, _ = await harness.call(
                "slowave_recall",
                {
                    "session_id": first["session_id"],
                    "scope": scope,
                    "query": task,
                    "page_size": 3,
                },
            )
            assert len(small["memories"]) == 3
            assert small["more_available"] is True
            invalid, _ = await harness.raw_call(
                "slowave_recall",
                {
                    "session_id": first["session_id"],
                    "scope": scope,
                    "continue_from": small["continue_from"],
                    "page_size": 7,
                },
            )
            _assert_error(invalid, "invalid_input", "page_size")
            await harness.feedback_all(small)

            feedback, _ = await harness.call(
                "slowave_feedback",
                {
                    "retrieval_id": first["retrieval_id"],
                    "memory_feedback": [
                        {"memory_id": item["memory_id"], "assessment": "used"} for item in memories
                    ],
                    "coverage": "complete",
                },
            )
            assert feedback["rejected"] == [], feedback
            await harness.commit(first["session_id"], "complete complementary activation dogfood")

    _run(scenario())


def test_feedback_schema_and_empty_batch_recovery(tmp_path: Path) -> None:
    """Real MCP definitions expose labels; empty/batch feedback can recover and close."""

    async def scenario() -> None:
        async with open_harness(tmp_path / "feedback-guidance.db") as harness:
            tools = await harness.session.list_tools()
            tool = next(t for t in tools.tools if t.name == "slowave_feedback")
            definitions = tool.inputSchema["$defs"]
            assert definitions["MemoryFeedbackEntry"]["properties"]["assessment"]["enum"] == [
                "used",
                "not_used",
                "unassessable",
                "irrelevant",
                "already_known",
                "stale",
            ]
            assert definitions["ProcedureFeedbackEntry"]["properties"]["use"]["enum"] == [
                "used",
                "not_used",
                "unassessable",
            ]
            activation, _ = await harness.activate(
                "feedback_guidance", "Check empty feedback", "verify feedback", "project:empty"
            )
            assert not activation["memories"] and not activation["procedures"]
            rid = activation["retrieval_id"]
            invalid, _ = await harness.raw_call(
                "slowave_feedback", {"items": [{"retrieval_id": rid}], "coverage": "complete"}
            )
            _assert_error(invalid, "invalid_input", "mutually exclusive")
            batch, _ = await harness.call(
                "slowave_feedback",
                {
                    "items": [
                        {"retrieval_id": "ctx_missing", "coverage": "complete"},
                        {"retrieval_id": rid, "coverage": "complete"},
                    ]
                },
            )
            assert batch["results"][0]["ok"] is False
            assert batch["results"][1]["ok"] is True
            assert batch["results"][1]["data"]["rejected"] == []
            await harness.commit(activation["session_id"], "verify feedback")

    _run(scenario())


def test_reported_rating_useful_used_feedback_sequence(tmp_path: Path) -> None:
    """The reported work-agent mistakes produce actionable errors and recover safely."""

    async def scenario() -> None:
        async with open_harness(tmp_path / "reported-feedback-retries.db") as harness:
            activation, _ = await harness.activate(
                "reported_retries", "Check feedback contract", "verify feedback", "project:test"
            )
            await harness.feedback_all(activation)
            mid = await harness.remember(
                "The verification command is pytest.",
                "fact",
                activation["session_id"],
                "project:test",
            )
            shown, _ = await harness.recall(
                "reported_feedback",
                "The verification command is pytest.",
                activation["session_id"],
                "project:test",
            )
            assert mid in {m["memory_id"] for m in shown["memories"]}
            rid = shown["retrieval_id"]
            wrong_field, _ = await harness.raw_call(
                "slowave_feedback",
                {
                    "retrieval_id": rid,
                    "memory_feedback": [{"memory_id": mid, "rating": "useful"}],
                    "coverage": "complete",
                },
            )
            _assert_error(wrong_field, "invalid_input", "assessment")
            paths = {e["path"] for e in wrong_field["error"]["field_errors"]}
            assert paths == {"memory_feedback[0].assessment", "memory_feedback[0].rating"}
            wrong_label, _ = await harness.call(
                "slowave_feedback",
                {
                    "retrieval_id": rid,
                    "memory_feedback": [{"memory_id": mid, "assessment": "useful"}],
                    "coverage": "complete",
                },
            )
            assert wrong_label["coverage"] == "partial"
            assert wrong_label["outstanding"]["memory_ids"] == [mid]
            assert wrong_label["rejected"][0] == {
                "target_id": mid,
                "reason": "invalid_memory_assessment",
                "field": "assessment",
                "hint": "Use used|not_used|unassessable|irrelevant|already_known|stale.",
            }
            assert wrong_label["rejected"][1]["reason"] == "incomplete_coverage"
            corrected, _ = await harness.call(
                "slowave_feedback",
                {
                    "retrieval_id": rid,
                    "memory_feedback": [{"memory_id": mid, "assessment": "used"}],
                    "coverage": "complete",
                },
            )
            assert corrected["rejected"] == []
            assert corrected["outstanding"] == {"memory_ids": [], "procedure_ids": []}
            committed, _ = await harness.call(
                "slowave_commit",
                {
                    "session_id": activation["session_id"],
                    "final_goal": "verify feedback",
                    "outcome": "success",
                    "outcome_summary": "Validated feedback recovery.",
                    "verification": {
                        "status": "verified",
                        "summary": "Tested reported retry sequence.",
                    },
                },
            )
            assert committed["feedback_status"] == "complete"

    _run(scenario())
