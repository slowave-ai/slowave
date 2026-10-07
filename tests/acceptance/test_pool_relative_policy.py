"""Pool-relative admission through the real MCP endpoint.

Both retrieval endpoints report the distinct pool-relative policy identity, and
a stored memory is admitted through the pool-relative floors on a deliberate
recall.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from tests.acceptance.mcp_harness import open_harness


def _run(coro) -> None:
    asyncio.run(coro)


def test_pool_relative_policy_identity_flows_through_mcp(tmp_path: Path) -> None:
    async def scenario() -> None:
        async with open_harness(tmp_path / "pool-relative-identity.db") as harness:
            activation, _ = await harness.activate(
                "pool_relative_identity",
                "Record the household espresso routine: dial in the grinder,"
                " then refresh the daily recipe.",
                "record the household espresso routine",
                "project:pool-relative",
            )
            assert activation["retrieval_policy_version"] == "activation-pool-relative-v1"

            memory_id = await harness.remember(
                "The household espresso dose is 18 g with a 1:2 ratio.",
                "fact",
                activation["session_id"],
                "project:pool-relative",
            )
            await harness.feedback_all(activation)

            lookup, _ = await harness.recall(
                "pool_relative_recall",
                "What is the household espresso dose and ratio?",
                activation["session_id"],
                "project:pool-relative",
            )
            assert lookup["retrieval_policy_version"] == "recall-pool-relative-v1"
            assert memory_id in [row["memory_id"] for row in lookup["memories"]]
            await harness.feedback_all(lookup, used_ids={memory_id})

            await harness.commit(activation["session_id"], "record the household espresso routine")

    _run(scenario())
