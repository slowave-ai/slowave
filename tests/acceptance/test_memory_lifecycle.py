"""Black-box memory lifecycle stories told entirely through the MCP contract."""

from __future__ import annotations

import asyncio
import os
from dataclasses import replace
from pathlib import Path

import pytest

from tests.acceptance.mcp_harness import (
    MCPHarness,
    assert_acceptance_mutation_is_caught,
    open_harness,
)
from tests.retrieval_quality.contracts import (
    RetrievalEvaluation,
    RetrievalGold,
)
from tests.retrieval_quality.contracts import evaluate as _evaluate


def evaluate(gold: RetrievalGold, observation) -> RetrievalEvaluation:
    """Apply the benchmark's selected activation budget to every public oracle."""
    if gold.surface == "activate":
        gold = replace(
            gold,
            max_items=int(os.environ.get("SLOWAVE_ACCEPTANCE_ACTIVATE_LIMIT", "2")),
        )
    return _evaluate(gold, observation)


def _run(coro) -> None:
    asyncio.run(coro)


def _assert_passed(harness: MCPHarness, result: RetrievalEvaluation) -> None:
    harness.record_evaluation(result)
    assert result.passed, result.as_dict()


async def _seed(
    harness: MCPHarness,
    scope: str,
    memories: tuple[tuple[str, str], ...],
) -> dict[str, str]:
    activation, _ = await harness.activate(
        "fixture_seed",
        f"Seed retrieval acceptance fixture for {scope}",
        "seed retrieval acceptance fixture",
        scope,
    )
    await harness.feedback_all(activation)
    ids = {}
    for content, memory_type in memories:
        ids[content] = await harness.remember(
            content=content,
            memory_type=memory_type,
            session_id=activation["session_id"],
            scope=scope,
        )
    await harness.commit(activation["session_id"], "seed retrieval acceptance fixture")
    return ids


async def _finish(
    harness: MCPHarness,
    retrieval: dict,
    *,
    used_ids: set[str] | None = None,
) -> None:
    await harness.feedback_all(retrieval, used_ids=used_ids)
    await harness.commit(retrieval["session_id"], "evaluate compact retrieval")


@pytest.mark.parametrize(
    ("case_id", "task", "required", "forbidden"),
    (
        (
            "direct_fact",
            "Which database stores the billing ledger?",
            "The billing ledger database is PostgreSQL 16.",
            "The billing deployment runs from the production release branch.",
        ),
        (
            "direct_decision",
            "What cache did checkout decide to use?",
            "The checkout team decided to use Redis for its response cache.",
            "The billing ledger database is PostgreSQL 16.",
        ),
    ),
    ids=("direct_fact", "direct_decision"),
)
def test_direct_fact_and_decision_via_mcp(
    tmp_path: Path, case_id: str, task: str, required: str, forbidden: str
) -> None:
    async def scenario() -> None:
        db_path = tmp_path / "direct.db"
        fact = "The billing ledger database is PostgreSQL 16."
        decision = "The checkout team decided to use Redis for its response cache."
        distractor = "The billing deployment runs from the production release branch."
        async with open_harness(db_path) as harness:
            ids = await _seed(
                harness,
                "project:shop",
                ((fact, "fact"), (decision, "decision"), (distractor, "fact")),
            )
            retrieval, observation = await harness.activate(
                case_id, task, "retrieve current architecture", "project:shop"
            )
            result = evaluate(
                RetrievalGold(
                    case_id=case_id,
                    family="direct_fact_decision",
                    surface="activate",
                    scope="project:shop",
                    required_contents=(required,),
                    forbidden_contents=(forbidden,),
                ),
                observation,
            )
            await _finish(harness, retrieval, used_ids={ids[required]})
            _assert_passed(harness, result)

    _run(scenario())


def test_semantic_paraphrase_via_mcp(tmp_path: Path) -> None:
    async def assessed_scenario() -> None:
        db_path = tmp_path / "paraphrase-assessed.db"
        target = "Authentication credentials expire after forty-five minutes."
        distractor = "Production deployments require two reviewers."
        async with open_harness(db_path) as harness:
            ids = await _seed(
                harness,
                "project:identity",
                ((target, "fact"), (distractor, "constraint")),
            )
            activation, _ = await harness.activate(
                "paraphrase_session",
                "Investigate authentication timeouts",
                "understand credential lifetime",
                "project:identity",
            )
            retrieval, observation = await harness.recall(
                "semantic_paraphrase",
                "How long do login tokens remain valid?",
                activation["session_id"],
                "project:identity",
            )
            result = evaluate(
                RetrievalGold(
                    case_id="semantic_paraphrase",
                    family="paraphrase",
                    surface="recall",
                    scope="project:identity",
                    required_contents=(target,),
                    forbidden_contents=(distractor,),
                ),
                observation,
            )
            await harness.feedback_all(activation, used_ids={ids[target]})
            await harness.feedback_all(retrieval, used_ids={ids[target]})
            await harness.commit(activation["session_id"], "evaluate semantic paraphrase")
            _assert_passed(harness, result)

    _run(assessed_scenario())


def test_deliberate_recall_adds_useful_context_beyond_initial_activation(tmp_path: Path) -> None:
    """A focused recall recovers useful context omitted by broad activation."""

    async def scenario() -> None:
        target = "The Orion payments-api rollback command is kubectl rollout undo deployment/payments-api."
        distractor = "The weekly deployment review covers release ownership and incident follow-up."
        scope = "project:release"
        async with open_harness(tmp_path / "deliberate-recall.db") as harness:
            seed, _ = await harness.activate(
                "deliberate_recall_seed",
                "Seed a deployment memory for incremental recall testing",
                "preserve deployment guidance",
                scope,
            )
            target_id = await harness.remember(target, "fact", seed["session_id"], scope)
            await harness.remember(distractor, "fact", seed["session_id"], scope)
            await harness.feedback_all(seed)
            await harness.commit(seed["session_id"], "seed deliberate recall fixture")

            baseline, _ = await harness.activate(
                "deliberate_recall_baseline",
                "Prepare the weekly deployment review",
                "prepare the deployment review",
                scope,
            )
            baseline_contents = [item.get("content", "") for item in baseline["memories"]]
            assert target not in baseline_contents, (
                "the broad activation baseline already returned the target; "
                "the recall ablation is not incremental"
            )
            await harness.feedback_all(baseline)

            recalled, _ = await harness.recall(
                "deliberate_recall_probe",
                "What is the Orion payments-api rollback command?",
                baseline["session_id"],
                scope,
            )
            recalled_contents = [item.get("content", "") for item in recalled["memories"]]
            assert any(
                target in content for content in recalled_contents
            ), "deliberate recall did not recover the useful deployment guidance"
            assert not any(
                distractor in content for content in recalled_contents
            ), "deliberate recall added an unrelated deployment-review item"
            assert any(item["memory_id"] == target_id for item in recalled["memories"])
            await harness.feedback_all(recalled, used_ids={target_id})
            await harness.commit(baseline["session_id"], "use recalled rollback guidance")

    _run(scenario())


def test_same_scope_hard_negative_returns_empty(tmp_path: Path) -> None:
    async def scenario() -> None:
        db_path = tmp_path / "hard-negative.db"
        memories = (
            ("The billing ledger database is PostgreSQL 16.", "fact"),
            ("Authentication credentials expire after forty-five minutes.", "fact"),
        )
        async with open_harness(db_path) as harness:
            await _seed(harness, "project:shop", memories)
            retrieval, observation = await harness.activate(
                "hard_negative",
                "Which paint colour was selected for the lobby mural?",
                "identify lobby paint palette",
                "project:shop",
            )
            result = evaluate(
                RetrievalGold(
                    case_id="hard_negative",
                    family="hard_semantic_negative",
                    surface="activate",
                    scope="project:shop",
                    forbidden_contents=tuple(content for content, _ in memories),
                    expected_empty=True,
                ),
                observation,
            )
            await _finish(harness, retrieval)
            _assert_passed(harness, result)

    _run(scenario())


def test_scope_twins_do_not_leak(tmp_path: Path) -> None:
    async def assessed_scenario() -> None:
        db_path = tmp_path / "scope-twins-assessed.db"
        alpha = "Project Alpha rotates authentication tokens every 30 minutes."
        beta = "Project Beta rotates authentication tokens every 90 minutes."
        async with open_harness(db_path) as harness:
            alpha_ids = await _seed(harness, "project:alpha", ((alpha, "decision"),))
            await _seed(harness, "project:beta", ((beta, "decision"),))
            activation, _ = await harness.activate(
                "scope_session",
                "Audit Project Alpha authentication token rotation",
                "verify Alpha token policy",
                "project:alpha",
            )
            retrieval, observation = await harness.recall(
                "scope_twins",
                "How often does Project Alpha rotate authentication tokens?",
                activation["session_id"],
                "project:alpha",
            )
            result = evaluate(
                RetrievalGold(
                    case_id="scope_twins",
                    family="scope_twins",
                    surface="recall",
                    scope="project:alpha",
                    required_contents=(alpha,),
                    forbidden_contents=(beta,),
                ),
                observation,
            )
            await harness.feedback_all(activation, used_ids={alpha_ids[alpha]})
            await harness.feedback_all(retrieval, used_ids={alpha_ids[alpha]})
            await harness.commit(activation["session_id"], "verify scope isolation")
            _assert_passed(harness, result)

    _run(assessed_scenario())


def test_stale_memory_is_not_returned_as_current_guidance_after_replacement(tmp_path: Path) -> None:
    async def scenario() -> None:
        db_path = tmp_path / "current-old.db"
        scope = "project:support"
        old = "The current refund window is 30 days."
        current = "The current refund window is 14 days."
        async with open_harness(db_path) as harness:
            old_ids = await _seed(harness, scope, ((old, "decision"),))
            correction, _ = await harness.activate(
                "mark_stale",
                "Update the current refund window from 30 days",
                "replace stale refund policy",
                scope,
            )
            current_id = await harness.remember(
                current, "decision", correction["session_id"], scope
            )
            await harness.feedback_all(
                correction,
                stale_ids={old_ids[old]},
                replacements={old_ids[old]: current_id},
            )
            await harness.commit(correction["session_id"], "replace stale refund policy")

            retrieval, observation = await harness.activate(
                "current_value",
                "What is the current refund window?",
                "answer with current refund policy",
                scope,
            )
            result = evaluate(
                RetrievalGold(
                    case_id="current_value",
                    family="current_old",
                    surface="activate",
                    scope=scope,
                    required_contents=(current,),
                    historical_only_contents=(old,),
                ),
                observation,
            )
            await _finish(harness, retrieval, used_ids={current_id})
            _assert_passed(harness, result)

    _run(scenario())


def test_known_present_and_known_absent_are_distinguished(tmp_path: Path) -> None:
    async def scenario() -> None:
        db_path = tmp_path / "present-absent.db"
        target = "The background job queue uses Redis."
        async with open_harness(db_path) as harness:
            ids = await _seed(harness, "project:jobs", ((target, "fact"),))
            present, present_observation = await harness.activate(
                "known_present",
                "Which system stores the background job queue?",
                "identify job queue storage",
                "project:jobs",
            )
            present_result = evaluate(
                RetrievalGold(
                    case_id="known_present",
                    family="present_absent",
                    surface="activate",
                    scope="project:jobs",
                    required_contents=(target,),
                ),
                present_observation,
            )
            await _finish(harness, present, used_ids={ids[target]})
            _assert_passed(harness, present_result)

            absent, absent_observation = await harness.activate(
                "known_absent",
                "Which paint colour is used in the executive office?",
                "identify office paint colour",
                "project:jobs",
            )
            absent_result = evaluate(
                RetrievalGold(
                    case_id="known_absent",
                    family="present_absent",
                    surface="activate",
                    scope="project:jobs",
                    forbidden_contents=(target,),
                    expected_empty=True,
                ),
                absent_observation,
            )
            await _finish(harness, absent)
            _assert_passed(harness, absent_result)

    _run(scenario())


def test_repeated_retrieval_without_feedback_is_stable(tmp_path: Path) -> None:
    async def scenario() -> None:
        db_path = tmp_path / "repeat.db"
        target = "The billing ledger database is PostgreSQL 16."
        async with open_harness(db_path) as harness:
            ids = await _seed(harness, "project:billing", ((target, "fact"),))
            activation, _ = await harness.activate(
                "repeat_session",
                "Audit the billing ledger database",
                "verify billing storage",
                "project:billing",
            )
            recalls = []
            for index in range(3):
                retrieval, observation = await harness.recall(
                    f"repeat_{index}",
                    "Which database stores the billing ledger?",
                    activation["session_id"],
                    "project:billing",
                )
                recalls.append((retrieval, observation))
            assert len({observation.returned_ids for _retrieval, observation in recalls}) == 1
            assert recalls[0][1].returned_contents == (target,)
            for index, (_retrieval, observation) in enumerate(recalls):
                result = evaluate(
                    RetrievalGold(
                        case_id=f"repeat_{index}",
                        family="repeated_retrieval",
                        surface="recall",
                        scope="project:billing",
                        required_contents=(target,),
                    ),
                    observation,
                )
                harness.record_evaluation(result)
            await harness.feedback_all(activation, used_ids={ids[target]})
            for retrieval, _observation in recalls:
                await harness.feedback_all(retrieval, used_ids={ids[target]})
            await harness.commit(activation["session_id"], "verify repeated retrieval stability")

    _run(scenario())


def test_irrelevant_feedback_does_not_delete_a_memory_needed_by_a_later_client(
    tmp_path: Path,
) -> None:
    """Contextual irrelevance changes access evidence, not the underlying remembered fact."""

    async def scenario() -> None:
        target = "The support team retrospective happens every second Friday."
        scope = "project:support"
        async with open_harness(tmp_path / "irrelevant-feedback.db") as harness:
            ids = await _seed(harness, scope, ((target, "fact"),))
            first, _ = await harness.activate(
                "irrelevant_feedback_first_retrieval",
                "When is the support team retrospective?",
                "retrieve the retrospective schedule",
                scope,
            )
            assert [memory["memory_id"] for memory in first["memories"]] == [ids[target]]

            # A client may decide this result was not useful for its immediate task.
            await harness.feedback_all(first)
            await harness.commit(first["session_id"], "decline the retrospective schedule")

            later, _ = await harness.activate(
                "irrelevant_feedback_later_retrieval",
                "When is the support team retrospective?",
                "retrieve the retrospective schedule for a later task",
                scope,
            )
            assert [memory["memory_id"] for memory in later["memories"]] == [ids[target]]
            await harness.feedback_all(later, used_ids={ids[target]})
            await harness.commit(later["session_id"], "use the retrospective schedule")

    _run(scenario())


def test_remember_preserves_client_source_time_in_public_recall_evidence(tmp_path: Path) -> None:
    async def scenario() -> None:
        target = "The payments incident started before the certificate rotation."
        source_time = "2026-08-19T14:05:00Z"
        scope = "project:payments"
        async with open_harness(tmp_path / "occurred-at.db") as harness:
            seed, _ = await harness.activate(
                "occurred_at_seed",
                "Record the payments incident",
                "preserve incident history",
                scope,
            )
            memory_id = await harness.remember(
                target,
                "fact",
                seed["session_id"],
                scope,
                occurred_at=source_time,
            )
            await harness.feedback_all(seed)
            await harness.commit(seed["session_id"], "record payments incident")

            later, _ = await harness.activate(
                "occurred_at_later",
                "Investigate the payments incident",
                "recover incident timing",
                scope,
            )
            retrieval, observation = await harness.recall(
                "occurred_at_recall",
                "When did the payments incident start?",
                later["session_id"],
                scope,
                evidence="full",
            )
            result = evaluate(
                RetrievalGold(
                    case_id="occurred_at_recall",
                    family="episodic_temporal",
                    surface="recall",
                    scope=scope,
                    required_contents=(target,),
                ),
                observation,
            )
            evidence = retrieval["evidence"]
            source_event = next(
                item
                for item in evidence
                if item["source_kind"] == "event" and item.get("content") == target
            )
            assert source_event["occurred_at"] == 1787148300
            assert source_event["recorded_at"] != source_event["occurred_at"]
            await harness.feedback_all(later, used_ids={memory_id})
            await harness.feedback_all(retrieval, used_ids={memory_id})
            await harness.commit(later["session_id"], "recover payments incident timing")
            _assert_passed(harness, result)

    _run(scenario())


def test_temporal_recall_prefers_the_episode_matching_client_source_time(tmp_path: Path) -> None:
    """Past-tense retrieval is time-matched; semantic retrieval returns the newer episode."""

    async def scenario() -> None:
        older = "The payments incident was caused by an expired certificate."
        newer = "The payments incident was caused by a gateway timeout."
        scope = "project:payments"
        async with open_harness(tmp_path / "temporal-ranking.db") as harness:
            seed, _ = await harness.activate(
                "temporal_ranking_seed",
                "Record two payments incidents from different weeks",
                "preserve dated incident history",
                scope,
            )
            older_id = await harness.remember(
                older,
                "fact",
                seed["session_id"],
                scope,
                occurred_at="2026-08-19T14:05:00Z",
            )
            await harness.feedback_all(seed)
            await harness.commit(seed["session_id"], "record the earlier payments incident")

            newer_seed, _ = await harness.activate(
                "temporal_ranking_newer_seed",
                "Record this week's payments incident",
                "preserve the newer dated incident",
                scope,
            )
            await harness.remember(
                newer,
                "fact",
                newer_seed["session_id"],
                scope,
                occurred_at="2026-08-25T14:05:00Z",
            )
            await harness.feedback_all(newer_seed)
            await harness.commit(newer_seed["session_id"], "record the newer payments incident")

            later, _ = await harness.activate(
                "temporal_ranking_later",
                "Investigate last week's payments incident",
                "retrieve the incident from last week",
                scope,
            )
            retrieval, _ = await harness.recall(
                "temporal_ranking_recall",
                "What caused the payments incident last week?",
                later["session_id"],
                scope,
                evidence="full",
            )
            episodes = [item for item in retrieval["evidence"] if item["source_kind"] == "episode"]
            assert episodes, "public recall did not expose any ranked episodic evidence"
            assert older in episodes[0].get("content", "")
            assert episodes[0]["occurred_at"] == 1787148300
            await harness.feedback_all(later, used_ids={older_id})
            await harness.feedback_all(retrieval, used_ids={older_id})
            await harness.commit(later["session_id"], "retrieve last week's payments incident")

            current, _ = await harness.activate(
                "temporal_ranking_current",
                "Investigate the gateway-timeout payments incident",
                "retrieve the gateway-timeout incident without a time anchor",
                scope,
            )
            current_retrieval, _ = await harness.recall(
                "temporal_ranking_current_recall",
                "Which payments incident involved a gateway timeout?",
                current["session_id"],
                scope,
                evidence="full",
            )
            current_episodes = [
                item for item in current_retrieval["evidence"] if item["source_kind"] == "episode"
            ]
            assert current_episodes, "public recall did not expose current episodic evidence"
            # Evidence is a bounded collection, not a documented score-ordered
            # list. Assert the semantic target is present with its source time
            # rather than relying on incidental insertion order between CI runs.
            newer_episode = next(
                (item for item in current_episodes if newer in item.get("content", "")),
                None,
            )
            assert newer_episode is not None, (
                "semantic retrieval did not expose the gateway-timeout episode: "
                f"{current_episodes}"
            )
            assert newer_episode["occurred_at"] == 1787666700
            await harness.feedback_all(current)
            await harness.feedback_all(current_retrieval)
            await harness.commit(
                current["session_id"], "retrieve the gateway-timeout payments incident"
            )

    _run(scenario())


@pytest.mark.xfail(
    condition=os.environ.get("SLOWAVE_ACCEPTANCE_ENCODER", "deterministic") == "deterministic",
    strict=True,
    reason="The deterministic transport encoder cannot rank the multi-facet interference pair reliably; production encoder coverage is authoritative for this quality case.",
)
def test_required_fact_survives_same_scope_interference(tmp_path: Path) -> None:
    async def scenario() -> None:
        db_path = tmp_path / "interference.db"
        scope = "project:billing-growth"
        target = "The billing ledger database is PostgreSQL 16."
        adjacent = (
            "The billing ledger export format is CSV.",
            "The billing ledger reconciliation job runs nightly.",
            "The billing database backup retention is seven days.",
            "The billing deployment uses blue-green releases.",
        )
        unrelated = tuple(
            f"Background worker group {index} processes image conversion jobs from queue {index}."
            for index in range(46)
        )
        forbidden = (*adjacent, *unrelated)
        async with open_harness(db_path) as harness:
            target_ids = await _seed(harness, scope, ((target, "fact"),))
            clean, clean_observation = await harness.activate(
                "interference_clean",
                "Which database stores the billing ledger?",
                "identify billing ledger storage",
                scope,
            )
            clean_result = evaluate(
                RetrievalGold(
                    case_id="interference_clean",
                    family="interference",
                    surface="activate",
                    scope=scope,
                    required_contents=(target,),
                    forbidden_contents=forbidden,
                ),
                clean_observation,
            )
            await _finish(harness, clean, used_ids={target_ids[target]})
            _assert_passed(harness, clean_result)

            seed, _ = await harness.activate(
                "interference_seed",
                "Store the background worker and billing operations fixture",
                "grow the retrieval acceptance store",
                scope,
            )
            await harness.feedback_all(seed, used_ids={target_ids[target]})
            forbidden_ids = await harness.remember_batch(
                [{"content": content, "type": "fact"} for content in forbidden],
                seed["session_id"],
                scope,
            )
            await harness.commit(seed["session_id"], "grow the retrieval acceptance store")

            multifacet, multifacet_observation = await harness.activate(
                "interference_multifacet_positive",
                "Which database stores the billing ledger and which export format does it use?",
                "identify billing ledger storage and export format",
                scope,
            )
            multifacet_result = evaluate(
                RetrievalGold(
                    case_id="interference_multifacet_positive",
                    family="interference",
                    surface="activate",
                    scope=scope,
                    required_contents=(target, adjacent[0]),
                    forbidden_contents=(*adjacent[1:], *unrelated),
                ),
                multifacet_observation,
            )
            await _finish(
                harness,
                multifacet,
                used_ids={target_ids[target], forbidden_ids[0]},
            )
            _assert_passed(harness, multifacet_result)

            heavy, heavy_observation = await harness.activate(
                "interference_heavy",
                "Which database stores the billing ledger?",
                "identify billing ledger storage",
                scope,
            )
            heavy_result = evaluate(
                RetrievalGold(
                    case_id="interference_heavy",
                    family="interference",
                    surface="activate",
                    scope=scope,
                    required_contents=(target,),
                    forbidden_contents=forbidden,
                ),
                heavy_observation,
            )
            await _finish(harness, heavy, used_ids={target_ids[target]})
            _assert_passed(harness, heavy_result)

    _run(scenario())


def test_singular_activation_excludes_adjacent_facets_across_domains(tmp_path: Path) -> None:
    """A compact answer must not pad a singular question with a sibling facet."""

    async def scenario() -> None:
        cases = (
            (
                "project:search",
                "Which platform stores the candidate search index?",
                "The candidate search index is stored in Elasticsearch.",
                "The candidate search export format is NDJSON.",
            ),
            (
                "project:identity",
                "Which provider signs the customer session tokens?",
                "Customer session tokens are signed by Auth0.",
                "Customer session token audit logs are retained for 90 days.",
            ),
            (
                "project:shipping",
                "Which carrier handles priority shipments?",
                "Priority shipments are handled by DHL Express.",
                "Priority shipment labels use the A6 thermal format.",
            ),
        )
        async with open_harness(tmp_path / "cross-domain-facets.db") as harness:
            for index, (scope, task, target, adjacent) in enumerate(cases):
                ids = await _seed(harness, scope, ((target, "fact"), (adjacent, "fact")))
                retrieval, observation = await harness.activate(
                    f"singular_facet_{index}", task, "answer the requested fact", scope
                )
                result = evaluate(
                    RetrievalGold(
                        case_id=f"singular_facet_{index}",
                        family="hard_semantic_negative",
                        surface="activate",
                        scope=scope,
                        required_contents=(target,),
                        forbidden_contents=(adjacent,),
                    ),
                    observation,
                )
                await _finish(harness, retrieval, used_ids={ids[target]})
                _assert_passed(harness, result)

    _run(scenario())


@pytest.mark.parametrize(
    ("mutation", "target"),
    (
        (
            "scope_filtering",
            "tests/acceptance/test_memory_lifecycle.py::test_scope_twins_do_not_leak",
        ),
        (
            "stale_suppression",
            "tests/acceptance/test_memory_lifecycle.py::test_stale_memory_is_not_returned_as_current_guidance_after_replacement",
        ),
        (
            "activation_budget",
            "tests/acceptance/test_memory_lifecycle.py::test_singular_activation_excludes_adjacent_facets_across_domains",
        ),
        (
            "relevance_admission",
            "tests/acceptance/test_memory_lifecycle.py::test_same_scope_hard_negative_returns_empty",
        ),
    ),
    ids=("scope_filtering", "stale_suppression", "activation_budget", "relevance_admission"),
)
def test_launch_critical_retrieval_mutation_fails_its_behavioral_contract(
    mutation: str, target: str
) -> None:
    """Each launch-critical retrieval guard must be caught when disabled."""
    assert_acceptance_mutation_is_caught(mutation, target)
