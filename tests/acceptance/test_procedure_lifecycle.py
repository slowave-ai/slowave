"""Black-box procedure lifecycle stories told entirely through the MCP contract."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest

from tests.acceptance.mcp_harness import open_harness


def _run(coro) -> None:
    asyncio.run(coro)


@pytest.mark.xfail(
    condition=os.environ.get("SLOWAVE_ACCEPTANCE_ENCODER", "deterministic") == "deterministic",
    strict=True,
    reason="Procedure retrieval needs production semantic similarity; the deterministic encoder only validates the public transport contract.",
)
def test_successful_procedure_is_retrieved_and_marked_helpful(tmp_path: Path) -> None:
    """Verified guidance is reusable for a later, related task in the same scope."""

    async def scenario() -> None:
        scope = "project:operations"
        successful_procedure = {
            "summary": "Repair a Kubernetes service that is crash-looping.",
            "context": {"platform": "kubernetes", "failure": "crashloop"},
            "steps": [
                {"summary": "Inspect pod events and deployment configuration."},
                {"summary": "Repair the deployment and verify health checks."},
            ],
            "caveats": ["Dependency failures may require an indirect repair."],
        }
        async with open_harness(tmp_path / "successful-procedure.db") as harness:
            completed, _ = await harness.activate(
                "successful_procedure_created",
                "Repair the checkout service crash loop on Kubernetes.",
                "restore the checkout service",
                scope,
            )
            await harness.feedback_all(completed)
            await harness.commit(
                completed["session_id"],
                "repair checkout service crash loop",
                outcome="success",
                procedure=successful_procedure,
            )

            later, _ = await harness.activate(
                "successful_procedure_retrieved",
                "Repair another Kubernetes service crash loop.",
                "restore a crash-looping Kubernetes service",
                scope,
            )
            assert len(later["procedures"]) == 1
            retrieved = later["procedures"][0]
            assert retrieved["summary"] == successful_procedure["summary"]
            assert retrieved["outcome"] == "success"
            await harness.feedback_all(later, used_procedure_ids={str(retrieved["procedure_id"])})
            await harness.commit(later["session_id"], "repair another crash-looping service")

    _run(scenario())


def test_failed_procedure_is_automatically_warned_and_not_followed(tmp_path: Path) -> None:
    """A failed procedure receives a product warning even without client prose."""

    async def scenario() -> None:
        scope = "project:operations"
        failed_procedure = {
            "summary": "Deploy directly to production without a health check.",
            "context": {"platform": "kubernetes", "risk": "unverified-production-deploy"},
            "steps": [
                {"summary": "Apply the production manifests immediately."},
                {"summary": "Skip verification and assume the rollout is healthy."},
            ],
            "caveats": [],
        }
        async with open_harness(tmp_path / "failed-procedure.db") as harness:
            failed, _ = await harness.activate(
                "failed_procedure_created",
                "Deploy a service directly to production without health checks.",
                "record the failed deployment approach",
                scope,
            )
            await harness.feedback_all(failed)
            await harness.commit(
                failed["session_id"],
                "record failed production deployment approach",
                outcome="failure",
                procedure=failed_procedure,
            )

            later, _ = await harness.activate(
                "failed_procedure_retrieved",
                "Should I deploy directly to production and skip health checks?",
                "avoid known deployment failures",
                scope,
            )
            assert len(later["procedures"]) == 1
            retrieved = later["procedures"][0]
            assert retrieved["summary"] == failed_procedure["summary"]
            assert retrieved["outcome"] == "failure"
            assert (
                "This procedure previously failed; treat it as cautionary evidence, "
                "not recommended guidance."
            ) in retrieved["caveats"]

            # The client saw the failed procedure but did not follow it.
            await harness.feedback_all(later)
            await harness.commit(later["session_id"], "avoid failed production deployment approach")

    _run(scenario())


def test_deliberate_recall_procedure_is_exposed_for_feedback(tmp_path: Path) -> None:
    """A procedure returned by recall is assessable, just like one from activate."""

    async def scenario() -> None:
        scope = "project:operations"
        query = "Verify production deployment health checks."
        procedure = {
            "summary": "Verify production deployment health checks.",
            "context": {"environment": "production"},
            "steps": [
                {"summary": "Inspect deployment health-check configuration."},
                {"summary": "Run the production health verification."},
            ],
            "caveats": [],
        }
        async with open_harness(tmp_path / "recall-procedure-feedback.db") as harness:
            seed, _ = await harness.activate(
                "recall_procedure_created",
                query,
                "verify production deployment health checks",
                scope,
            )
            await harness.feedback_all(seed)
            await harness.commit(
                seed["session_id"],
                "verify production deployment health checks",
                procedure=procedure,
            )

            active, _ = await harness.activate(
                "recall_procedure_session",
                "Review the quarterly invoice ledger.",
                "review invoice ledger",
                scope,
            )
            recalled, _ = await harness.recall(
                "recall_procedure_retrieved",
                query,
                active["session_id"],
                scope,
            )
            assert len(recalled["procedures"]) == 1
            procedure_id = str(recalled["procedures"][0]["procedure_id"])
            await harness.feedback_all(recalled, used_procedure_ids={procedure_id})
            await harness.feedback_all(active)
            await harness.commit(active["session_id"], "review deployment verification procedure")

    _run(scenario())


def test_helped_and_harmed_feedback_reorder_equally_relevant_procedures(tmp_path: Path) -> None:
    """One helpful report resolves a tie; one harm report reverses it safely."""

    async def scenario() -> None:
        scope = "project:ranking"
        procedure = {
            "summary": "Repair a Kubernetes checkout service crash loop.",
            "context": {"platform": "kubernetes", "failure": "crashloop"},
            "steps": [
                {"summary": "Inspect deployment events."},
                {"summary": "Repair configuration and verify health."},
            ],
            "caveats": [],
        }
        query = "Repair a Kubernetes checkout service crash loop."
        async with open_harness(tmp_path / "procedure-feedback-order.db") as harness:
            for label in ("first", "second"):
                created, _ = await harness.activate(f"ranking_{label}_created", query, query, scope)
                await harness.feedback_all(created)
                await harness.commit(created["session_id"], query, procedure=procedure)

            baseline, _ = await harness.activate("ranking_baseline", query, query, scope)
            assert len(baseline["procedures"]) == 2
            favored_id = str(baseline["procedures"][0]["procedure_id"])
            other_id = str(baseline["procedures"][1]["procedure_id"])
            await harness.feedback_all(baseline, used_procedure_ids={favored_id})
            await harness.commit(baseline["session_id"], query)

            helped, _ = await harness.activate("ranking_helped", query, query, scope)
            assert str(helped["procedures"][0]["procedure_id"]) == favored_id
            await harness.feedback_all(helped)
            await harness.commit(helped["session_id"], query)

            data, _ = await harness.call(
                "slowave_feedback",
                {
                    "retrieval_id": baseline["retrieval_id"],
                    "procedure_feedback": [
                        {
                            "procedure_id": favored_id,
                            "use": "used",
                            "effect": "harmed",
                            "contribution": "The previously favored repair made the incident worse.",
                        },
                        {"procedure_id": other_id, "use": "not_used", "effect": "unknown"},
                    ],
                    "coverage": "complete",
                },
            )
            assert data["rejected"] == []

            harmed, _ = await harness.activate("ranking_harmed", query, query, scope)
            assert str(harmed["procedures"][0]["procedure_id"]) == other_id
            await harness.feedback_all(harmed)
            await harness.commit(harmed["session_id"], query)

    _run(scenario())


def test_procedure_hard_negative_and_scope_isolation(tmp_path: Path) -> None:
    """Only the relevant same-scope procedure is returned; other scopes stay isolated."""

    async def scenario() -> None:
        scope = "project:checkout"
        other_scope = "project:billing"
        repair_query = "Repair a Kubernetes checkout service crash loop."
        repair_procedure = {
            "summary": repair_query,
            "context": {"platform": "kubernetes", "failure": "crashloop"},
            "steps": [
                {"summary": "Inspect checkout pod events."},
                {"summary": "Repair deployment configuration and verify health."},
            ],
            "caveats": [],
        }
        unrelated_procedure = {
            "summary": "Rotate an expired database credential.",
            "context": {"system": "database", "failure": "expired-credential"},
            "steps": [
                {"summary": "Issue a replacement credential."},
                {"summary": "Update the database client secret."},
            ],
            "caveats": [],
        }
        async with open_harness(tmp_path / "procedure-scope-negative.db") as harness:
            for label, task, procedure in (
                ("relevant", repair_query, repair_procedure),
                ("irrelevant", "Rotate an expired database credential.", unrelated_procedure),
            ):
                created, _ = await harness.activate(f"negative_{label}_created", task, task, scope)
                await harness.feedback_all(created)
                await harness.commit(created["session_id"], task, procedure=procedure)

            other, _ = await harness.activate(
                "scope_twin_created", repair_query, repair_query, other_scope
            )
            await harness.feedback_all(other)
            await harness.commit(other["session_id"], repair_query, procedure=repair_procedure)

            related, _ = await harness.activate(
                "hard_negative_query", repair_query, repair_query, scope
            )
            assert [item["summary"] for item in related["procedures"]] == [repair_query]
            await harness.feedback_all(related)
            await harness.commit(related["session_id"], repair_query)

            isolated, _ = await harness.activate(
                "cross_scope_query", repair_query, repair_query, "project:empty"
            )
            assert isolated["procedures"] == []
            await harness.feedback_all(isolated)
            await harness.commit(isolated["session_id"], repair_query)

    _run(scenario())


def test_procedure_not_used_and_unexposed_feedback_are_accounted_for(tmp_path: Path) -> None:
    """Not-used is neutral; IDs absent from the response are rejected without mutation."""

    async def scenario() -> None:
        scope = "project:feedback"
        query = "Verify a production deployment health check."
        procedure = {
            "summary": query,
            "context": {"environment": "production"},
            "steps": [
                {"summary": "Inspect deployment health-check configuration."},
                {"summary": "Run the production verification."},
            ],
            "caveats": [],
        }
        async with open_harness(tmp_path / "procedure-feedback-accounting.db") as harness:
            created, _ = await harness.activate("feedback_created", query, query, scope)
            await harness.feedback_all(created)
            await harness.commit(created["session_id"], query, procedure=procedure)

            retrieved, _ = await harness.activate("feedback_retrieved", query, query, scope)
            procedure_id = str(retrieved["procedures"][0]["procedure_id"])
            accepted, _ = await harness.call(
                "slowave_feedback",
                {
                    "retrieval_id": retrieved["retrieval_id"],
                    "procedure_feedback": [
                        {"procedure_id": procedure_id, "use": "not_used", "effect": "unknown"}
                    ],
                    "coverage": "complete",
                },
            )
            assert accepted["rejected"] == []
            assert accepted["outstanding"] == {"memory_ids": [], "procedure_ids": []}

            rejected, _ = await harness.call(
                "slowave_feedback",
                {
                    "retrieval_id": retrieved["retrieval_id"],
                    "procedure_feedback": [
                        {
                            "procedure_id": "proc_not_exposed",
                            "use": "not_used",
                            "effect": "unknown",
                        }
                    ],
                    "coverage": "partial",
                },
            )
            # The partial submission can record retrieval-level evidence, but
            # it must never accept the unexposed procedure target itself.
            assert rejected["rejected"][0]["reason"] == "target_not_exposed"
            await harness.commit(retrieved["session_id"], query)

    _run(scenario())
