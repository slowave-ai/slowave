"""Real stdio MCP client harness with boundary-level cost measurements."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

from tests.retrieval_quality.contracts import (
    RetrievalEvaluation,
    RetrievalObservation,
    observe,
)

ROOT = Path(__file__).resolve().parents[2]
SERVER = ROOT / "tests" / "acceptance" / "server.py"


def assert_acceptance_mutation_is_caught(mutation: str, target: str) -> None:
    """Require a deliberately broken public MCP path to fail its target test.

    The child pytest process starts a fresh acceptance server with one named
    mutation enabled.  Keeping this assertion beside the real MCP harness
    makes the mutation checks use the same client/server boundary as every
    other acceptance scenario.
    """
    env = os.environ.copy()
    env["SLOWAVE_ACCEPTANCE_ENCODER"] = "deterministic"
    env["SLOWAVE_ACCEPTANCE_MUTATION"] = mutation
    env["SLOWAVE_ACCEPTANCE_QUIET"] = "1"
    result = subprocess.run(
        [sys.executable, "-m", "pytest", target, "-q", "-p", "no:randomly"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode != 0, (
        f"acceptance mutation {mutation!r} survived target {target}:\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )


class MCPHarness:
    def __init__(self, session: ClientSession):
        self.session = session

    def record_evaluation(self, evaluation: RetrievalEvaluation) -> None:
        report_path = os.environ.get("SLOWAVE_RETRIEVAL_REPORT")
        if not report_path:
            return
        with Path(report_path).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(evaluation.as_dict(), sort_keys=True) + "\n")

    async def raw_call(self, name: str, arguments: dict[str, Any]) -> tuple[dict[str, Any], float]:
        """Call one public MCP tool without assuming whether it succeeds."""
        started = time.perf_counter()
        result = await self.session.call_tool(name, arguments)
        elapsed_ms = (time.perf_counter() - started) * 1000
        payload = result.structuredContent
        if not isinstance(payload, dict):
            raise AssertionError(f"{name} returned no structured payload: {result}")
        return payload, elapsed_ms

    async def call(self, name: str, arguments: dict[str, Any]) -> tuple[dict[str, Any], float]:
        """Call one public MCP tool and require its documented success envelope."""
        payload, elapsed_ms = await self.raw_call(name, arguments)
        if not payload.get("ok"):
            raise AssertionError(f"{name} failed: {payload}")
        data = payload.get("data")
        if not isinstance(data, dict):
            raise AssertionError(f"{name} returned malformed data: {payload}")
        return data, elapsed_ms

    async def activate(
        self,
        case_id: str,
        task: str,
        goal: str,
        scope: str,
        *,
        continuity_id: str | None = None,
        task_context: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], RetrievalObservation]:
        arguments: dict[str, Any] = {
            "task": task,
            "initial_goal": goal,
            "scope": scope,
        }
        if continuity_id is not None:
            arguments["continuity_id"] = continuity_id
        if task_context is not None:
            arguments["task_context"] = task_context
        data, elapsed = await self.call("slowave_activate", arguments)
        return data, observe(case_id, "activate", data, elapsed)

    async def recall(
        self,
        case_id: str,
        query: str,
        session_id: str,
        scope: str,
        *,
        evidence: str = "references",
        task_context: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], RetrievalObservation]:
        arguments: dict[str, Any] = {
            "query": query,
            "session_id": session_id,
            "scope": scope,
            "evidence": evidence,
        }
        if task_context is not None:
            arguments["task_context"] = task_context
        data, elapsed = await self.call("slowave_recall", arguments)
        return data, observe(case_id, "recall", data, elapsed)

    async def remember(
        self,
        content: str,
        memory_type: str,
        session_id: str,
        scope: str,
        *,
        occurred_at: str | None = None,
    ) -> str:
        arguments: dict[str, Any] = {
            "content": content,
            "type": memory_type,
            "session_id": session_id,
            "scope": scope,
        }
        if occurred_at is not None:
            arguments["occurred_at"] = occurred_at
        data, _ = await self.call(
            "slowave_remember",
            arguments,
        )
        return str(data["memory_id"])

    async def remember_batch(
        self,
        memories: list[dict[str, str]],
        session_id: str,
        scope: str,
    ) -> list[str]:
        data, _ = await self.call(
            "slowave_remember",
            {"memories": memories, "session_id": session_id, "scope": scope},
        )
        results = data["results"]
        failures = [item for item in results if not item.get("ok")]
        if failures:
            raise AssertionError(f"batch remember failed: {failures}")
        return [str(item["data"]["memory_id"]) for item in results]

    async def feedback_all(
        self,
        retrieval: dict[str, Any],
        *,
        used_ids: set[str] | None = None,
        stale_ids: set[str] | None = None,
        replacements: dict[str, str] | None = None,
        used_procedure_ids: set[str] | None = None,
    ) -> None:
        used_ids = used_ids or set()
        stale_ids = stale_ids or set()
        replacements = replacements or {}
        used_procedure_ids = used_procedure_ids or set()
        memory_feedback = []
        for memory in retrieval.get("memories", []):
            memory_id = str(memory["memory_id"])
            assessment = (
                "stale"
                if memory_id in stale_ids
                else "used" if memory_id in used_ids else "irrelevant"
            )
            item = {"memory_id": memory_id, "assessment": assessment}
            if assessment == "stale":
                item["stale_reason"] = "superseded" if memory_id in replacements else "outdated"
                item["reason"] = (
                    "A newer client-provided memory replaces this claim."
                    if memory_id in replacements
                    else "The client marked this claim as no longer current."
                )
            if memory_id in replacements:
                item["replacement_memory_id"] = replacements[memory_id]
            memory_feedback.append(item)
        procedure_feedback = []
        for procedure in retrieval.get("procedures", []):
            procedure_id = str(procedure["procedure_id"])
            if procedure_id in used_procedure_ids:
                procedure_feedback.append(
                    {
                        "procedure_id": procedure_id,
                        "use": "used",
                        "effect": "helped",
                        "contribution": "The retrieved procedure supplied useful verified guidance.",
                    }
                )
            else:
                procedure_feedback.append(
                    {"procedure_id": procedure_id, "use": "not_used", "effect": "unknown"}
                )
        data, _ = await self.call(
            "slowave_feedback",
            {
                "retrieval_id": retrieval["retrieval_id"],
                "memory_feedback": memory_feedback,
                "procedure_feedback": procedure_feedback,
                "coverage": "complete",
            },
        )
        assert data["rejected"] == []
        assert data["outstanding"] == {"memory_ids": [], "procedure_ids": []}

    async def commit(
        self,
        session_id: str,
        goal: str,
        *,
        outcome: str = "success",
        procedure: dict[str, Any] | None = None,
    ) -> None:
        payload: dict[str, Any] = {
            "session_id": session_id,
            "final_goal": goal,
            "outcome": outcome,
            "outcome_summary": f"Completed retrieval acceptance setup for {goal}.",
            "verification": {"status": "verified", "summary": "Acceptance fixture completed."},
        }
        if procedure is not None:
            payload["procedure"] = procedure
        await self.call(
            "slowave_commit",
            payload,
        )


@asynccontextmanager
async def open_harness(
    db_path: Path, *, extra_env: dict[str, str] | None = None
) -> AsyncIterator[MCPHarness]:
    env = os.environ.copy()
    env["SLOWAVE_DB"] = str(db_path)
    env["PYTHONPATH"] = str(ROOT)
    # Prevent dependency advisory messages from contaminating pytest's
    # per-scenario progress output (the child does not need PyTorch).
    env.setdefault("TRANSFORMERS_VERBOSITY", "error")
    env.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    # Keep normal acceptance output focused on scenario names/results.
    # Set SLOWAVE_ACCEPTANCE_VERBOSE=1 when diagnosing the child server.
    if env.get("SLOWAVE_ACCEPTANCE_VERBOSE") != "1":
        env.setdefault("SLOWAVE_ACCEPTANCE_QUIET", "1")
    if extra_env:
        env.update(extra_env)
    params = StdioServerParameters(
        command=sys.executable,
        args=[str(SERVER)],
        cwd=ROOT,
        env=env,
    )
    async with stdio_client(params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            yield MCPHarness(session)
