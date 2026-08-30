"""End-to-end MCP lifecycle scenarios that span multiple client sessions."""

from __future__ import annotations

import asyncio
from pathlib import Path

from tests.acceptance.mcp_harness import open_harness


def _run(coro) -> None:
    asyncio.run(coro)


def test_activate_accepts_task_context_and_continuity_and_recall_updates_context(
    tmp_path: Path,
) -> None:
    """Context and continuity fields travel through the public lifecycle."""

    async def scenario() -> None:
        scope = "project:checkout"
        async with open_harness(tmp_path / "context-continuity.db") as harness:
            active, _ = await harness.activate(
                "contextual_activation",
                "Investigate checkout cache failures",
                "diagnose checkout cache failures",
                scope,
                task_context={"service": "checkout", "environment": "production"},
            )
            await harness.feedback_all(active)

            recalled, _ = await harness.recall(
                "contextual_recall",
                "Which cache backs the checkout service?",
                active["session_id"],
                scope,
                evidence="references",
                task_context={"component": "response-cache"},
            )
            await harness.feedback_all(recalled)
            await harness.commit(active["session_id"], "diagnose checkout cache failures")

    _run(scenario())


def test_overlapping_sessions_same_scope_can_each_complete(tmp_path: Path) -> None:
    """Two concurrent conversations in one scope remain independently usable."""

    async def scenario() -> None:
        scope = "project:checkout"
        async with open_harness(tmp_path / "overlapping.db") as harness:
            first, _ = await harness.activate(
                "overlap_first",
                "Investigate checkout cache failures",
                "diagnose checkout cache failures",
                scope,
                task_context={"service": "checkout"},
            )
            second, _ = await harness.activate(
                "overlap_second",
                "Review checkout deployment health",
                "review checkout deployment health",
                scope,
                task_context={"service": "checkout", "environment": "staging"},
            )
            assert first["session_id"] != second["session_id"]

            await harness.feedback_all(first)
            await harness.feedback_all(second)
            await harness.commit(first["session_id"], "diagnose checkout cache failures")
            await harness.commit(second["session_id"], "review checkout deployment health")

    _run(scenario())


def test_server_issued_continuity_is_persistent_scoped_and_never_implicit(tmp_path: Path) -> None:
    """One client conversation spans task sessions without scope-session guessing."""

    async def scenario() -> None:
        db_path = tmp_path / "continuity-contract.db"
        scope = "project:continuity"
        async with open_harness(db_path) as harness:
            first, _ = await harness.activate(
                "continuity_first", "What do you know?", "inspect project", scope
            )
            assert first["continuity_state"] == "started"
            assert str(first["continuity_id"]).startswith("cont_")
            await harness.feedback_all(first)
            await harness.commit(first["session_id"], "inspect project")

            continued, _ = await harness.activate(
                "continuity_second",
                "What do you know?",
                "inspect project",
                scope,
                continuity_id=first["continuity_id"],
            )
            assert continued["continuity_state"] == "continued"
            assert continued["continuity_id"] == first["continuity_id"]
            assert continued["session_id"] != first["session_id"]
            await harness.feedback_all(continued)
            await harness.commit(continued["session_id"], "inspect project")

            separate, _ = await harness.activate(
                "continuity_other", "What do you know?", "inspect project", scope
            )
            assert separate["continuity_state"] == "started"
            assert separate["continuity_id"] != first["continuity_id"]
            await harness.feedback_all(separate)
            await harness.commit(separate["session_id"], "inspect project")

            unknown, _ = await harness.raw_call(
                "slowave_activate",
                {
                    "task": "x",
                    "initial_goal": "x",
                    "scope": scope,
                    "continuity_id": "cont_" + "a" * 43,
                },
            )
            assert unknown["ok"] is False and "unknown" in unknown["error"]["message"]
            blank, _ = await harness.raw_call(
                "slowave_activate",
                {"task": "x", "initial_goal": "x", "scope": scope, "continuity_id": " "},
            )
            assert blank["ok"] is False and "nonblank" in blank["error"]["message"]
            mismatch, _ = await harness.raw_call(
                "slowave_activate",
                {
                    "task": "x",
                    "initial_goal": "x",
                    "scope": "project:other",
                    "continuity_id": first["continuity_id"],
                },
            )
            assert mismatch["ok"] is False and "scope" in mismatch["error"]["message"]
            null_start, _ = await harness.raw_call(
                "slowave_activate",
                {"task": "x", "initial_goal": "x", "scope": scope, "continuity_id": None},
            )
            assert null_start["ok"] is True
            assert null_start["data"]["continuity_state"] == "started"
            assert null_start["data"]["continuity_id"] not in {
                first["continuity_id"],
                separate["continuity_id"],
            }
            await harness.feedback_all(null_start["data"])
            await harness.commit(null_start["data"]["session_id"], "inspect project")

        # The registry is durable across a fresh server process.
        async with open_harness(db_path) as restarted:
            resumed, _ = await restarted.activate(
                "continuity_restart",
                "What do you know?",
                "inspect project",
                scope,
                continuity_id=first["continuity_id"],
            )
            assert resumed["continuity_state"] == "continued"
            await restarted.feedback_all(resumed)
            await restarted.commit(resumed["session_id"], "inspect project")

    _run(scenario())


def test_broad_continuity_start_returns_diverse_labelled_context(tmp_path: Path) -> None:
    """A broad opening gets reinstatement context without widening direct answers."""

    async def scenario() -> None:
        scope = "project:reinstatement"
        async with open_harness(tmp_path / "reinstatement.db") as harness:
            seed, _ = await harness.activate(
                "reinstatement_seed",
                "Seed realistic project context",
                "seed project context",
                scope,
            )
            await harness.remember(
                "The project deploys through a protected production release branch.",
                "decision",
                seed["session_id"],
                scope,
            )
            await harness.remember(
                "The project stores ledger data in PostgreSQL 16.",
                "fact",
                seed["session_id"],
                scope,
            )
            await harness.remember(
                "The project requires two reviewers for production changes.",
                "constraint",
                seed["session_id"],
                scope,
            )
            await harness.remember(
                "The project uses Redis for the checkout response cache.",
                "fact",
                seed["session_id"],
                scope,
            )
            await harness.feedback_all(seed)
            await harness.commit(seed["session_id"], "seed project context")

            broad, _ = await harness.activate(
                "reinstatement_broad",
                "What do you know about this project?",
                "recover project context",
                scope,
                task_context={"project": "ledger cache deployment"},
            )
            assert broad["continuity_state"] == "started"
            assert any(
                item["pathway"] == "context_reinstatement" for item in broad["memories"]
            ), broad
            assert len({item["memory_id"] for item in broad["memories"]}) == len(broad["memories"])
            await harness.feedback_all(broad)
            await harness.commit(broad["session_id"], "recover project context")

    _run(scenario())


def test_lifecycle_survives_a_separate_mcp_client_connection(tmp_path: Path) -> None:
    """A later MCP connection can retrieve a committed memory from the first."""

    async def scenario() -> None:
        db_path = tmp_path / "separate-client.db"
        scope = "project:billing"
        fact = "The billing ledger is stored in PostgreSQL 16."
        async with open_harness(db_path) as first_client:
            active, _ = await first_client.activate(
                "first_client_store",
                "Store the billing ledger database",
                "preserve billing storage guidance",
                scope,
            )
            memory_id = await first_client.remember(fact, "fact", active["session_id"], scope)
            await first_client.feedback_all(active)
            await first_client.commit(active["session_id"], "preserve billing storage guidance")

        async with open_harness(db_path) as second_client:
            later, _ = await second_client.activate(
                "second_client_retrieve",
                "Which database stores the billing ledger?",
                "retrieve billing storage guidance",
                scope,
            )
            assert [item["memory_id"] for item in later["memories"]] == [memory_id]
            await second_client.feedback_all(later, used_ids={memory_id})
            await second_client.commit(later["session_id"], "retrieve billing storage guidance")

    _run(scenario())


def test_abandoned_session_is_closed_by_the_public_server_reaper(tmp_path: Path) -> None:
    """An uncommitted session is eventually rejected after idle reaping."""

    async def scenario() -> None:
        db_path = tmp_path / "abandoned.db"
        reaper_env = {
            "SLOWAVE_SESSION_IDLE_TIMEOUT": "1",
            "SLOWAVE_ACCEPTANCE_REAPER_POLL_INTERVAL": "0.05",
        }
        async with open_harness(db_path, extra_env=reaper_env) as first_client:
            abandoned, _ = await first_client.activate(
                "abandoned_session",
                "Investigate an abandoned incident",
                "preserve abandoned incident context",
                "project:incidents",
            )
            abandoned_session_id = abandoned["session_id"]

        # The first stdio client has disconnected without committing.  The
        # second server process runs the same public reaper against that DB.
        async with open_harness(db_path, extra_env=reaper_env) as second_client:
            await asyncio.sleep(1.5)
            rejected, _ = await second_client.raw_call(
                "slowave_recall",
                {
                    "query": "recover abandoned incident context",
                    "session_id": abandoned_session_id,
                    "scope": "project:incidents",
                },
            )
            assert rejected["ok"] is False
            assert "session is already ended" in rejected["error"]["message"]

    _run(scenario())


def test_fresh_database_supports_a_complete_first_lifecycle(tmp_path: Path) -> None:
    """A clean database starts cold and supports store, retrieve, and assess."""

    async def scenario() -> None:
        db_path = tmp_path / "fresh-install.db"
        scope = "project:fresh"
        fact = "The fresh installation uses the default local database."
        async with open_harness(db_path) as harness:
            first, _ = await harness.activate(
                "fresh_install_start",
                "Start from a completely fresh installation",
                "verify fresh installation lifecycle",
                scope,
                task_context={"installation": "fresh"},
            )
            assert first["memory_state"] == "cold_start"
            memory_id = await harness.remember(fact, "fact", first["session_id"], scope)
            await harness.feedback_all(first)
            await harness.commit(first["session_id"], "verify fresh installation lifecycle")

            later, _ = await harness.activate(
                "fresh_install_retrieval",
                "Which database does the fresh installation use?",
                "retrieve fresh installation guidance",
                scope,
            )
            assert [item["memory_id"] for item in later["memories"]] == [memory_id]
            await harness.feedback_all(later, used_ids={memory_id})
            await harness.commit(later["session_id"], "retrieve fresh installation guidance")

    _run(scenario())
