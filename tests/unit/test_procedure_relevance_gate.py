"""Graded relevance gate: procedures are returned only when relevant, up to the cap."""

from __future__ import annotations

from typing import Any

import numpy as np

from slowave.symbolic.procedural_memory import (
    PROCEDURE_RESULT_LIMIT,
    retrieve_relevant_procedures,
)


class _KeywordScorer:
    """Deterministic stand-in for the cross-encoder: logit = overlap of shared words."""

    def __init__(self) -> None:
        self.calls: list[tuple[list[str], int]] = []

    def score(self, queries: list[str], memories: list[str]) -> np.ndarray:
        self.calls.append((list(queries), len(memories)))
        out = np.full((len(memories), len(queries)), -8.0, dtype=np.float32)
        for j, query in enumerate(queries):
            terms = set(query.casefold().split())
            for i, memory in enumerate(memories):
                shared = terms & set(memory.casefold().split())
                out[i, j] = -8.0 + 5.0 * len(shared)
        return out


def _proc(index: int, goal: str, summary: str, **extra: Any) -> dict[str, Any]:
    item = {
        "id": f"proc_{index}",
        "goal": goal,
        "summary": summary,
        "steps": [{"summary": f"step {index}"}],
        "context": {},
        "evidence": {},
    }
    item.update(extra)
    return item


def _run(procs, query, goal=None, *, scorer=None, context=None, limit=PROCEDURE_RESULT_LIMIT):
    return retrieve_relevant_procedures(
        procs,
        query=query,
        goal=goal,
        retrieval_context=context,
        scorer=scorer or _KeywordScorer(),
        encoder=None,
        limit=limit,
    )


def test_unrelated_task_returns_nothing_instead_of_filling_the_cap() -> None:
    procs = [_proc(i, f"deploy service {i}", "deploy kubernetes service") for i in range(5)]
    assert _run(procs, "bake sourdough bread", "bake bread") == []


def test_returns_only_the_relevant_procedures_not_always_three() -> None:
    procs = [
        _proc(1, "restart dashboard server", "restart dashboard server after asset rebuild"),
        _proc(2, "rotate database credentials", "rotate database credentials safely"),
        _proc(3, "tune chart palette", "tune chart palette colors"),
        _proc(4, "profile retrieval latency", "profile retrieval latency on a database copy"),
    ]
    one = _run(procs, "restart the dashboard server", "restart dashboard server")
    assert [item["id"] for item in one] == ["proc_1"]
    two = _run(
        procs,
        "rotate credentials then profile latency",
        "rotate database credentials profile retrieval latency",
    )
    assert {item["id"] for item in two} == {"proc_2", "proc_4"}


def test_result_never_exceeds_the_cap() -> None:
    procs = [
        _proc(i, "restart dashboard server", f"restart dashboard server variant {i}")
        for i in range(8)
    ]
    assert (
        len(_run(procs, "restart dashboard server", "restart dashboard server"))
        == PROCEDURE_RESULT_LIMIT
    )


def test_incompatible_context_is_never_returned() -> None:
    procs = [
        _proc(1, "restart dashboard server", "restart dashboard server", context={"env": ["prod"]}),
    ]
    assert (
        _run(
            procs, "restart dashboard server", "restart dashboard server", context={"env": ["dev"]}
        )
        == []
    )


def test_utility_cannot_admit_an_irrelevant_procedure() -> None:
    helped = _proc(1, "tune chart palette", "tune chart palette", evidence={"helped": 9})
    assert _run([helped], "bake sourdough bread", "bake bread") == []


def test_utility_only_breaks_ties_between_equally_relevant_procedures() -> None:
    plain = _proc(1, "restart dashboard server", "restart dashboard server")
    helped = _proc(
        2, "restart dashboard server", "restart dashboard server", evidence={"helped": 3}
    )
    hits = _run([plain, helped], "restart dashboard server", "restart dashboard server")
    assert [item["id"] for item in hits] == ["proc_2", "proc_1"]


def test_long_verbatim_task_is_not_used_as_a_cue() -> None:
    scorer = _KeywordScorer()
    procs = [_proc(1, "restart dashboard server", "restart dashboard server")]
    long_task = "please " + "restart dashboard server " * 40
    _run(procs, long_task, "restart dashboard server", scorer=scorer)
    queries, _ = scorer.calls[0]
    assert queries == ["restart dashboard server"]


def test_short_task_is_a_second_cue_when_the_goal_is_vague() -> None:
    scorer = _KeywordScorer()
    procs = [_proc(1, "repair task", "inspect evidence then repair the workload")]
    hits = _run(procs, "inspect evidence then repair the workload", "do the thing", scorer=scorer)
    assert len(scorer.calls[0][0]) == 2
    assert [item["id"] for item in hits] == ["proc_1"]


def test_without_a_goal_the_task_head_is_the_cue() -> None:
    scorer = _KeywordScorer()
    procs = [_proc(1, "restart dashboard server", "restart dashboard server")]
    hits = _run(procs, "restart dashboard server", None, scorer=scorer)
    assert scorer.calls[0][0] == ["restart dashboard server"]
    assert [item["id"] for item in hits] == ["proc_1"]


def test_non_english_task_is_scored_without_an_action_verb_gate() -> None:
    procs = [_proc(1, "riavviare il server della dashboard", "riavviare il server della dashboard")]
    hits = _run(procs, "riavviare il server della dashboard", "riavviare il server della dashboard")
    assert [item["id"] for item in hits] == ["proc_1"]


def test_candidate_stage_bounds_cross_encoder_work_on_large_scopes() -> None:
    scorer = _KeywordScorer()
    procs = [_proc(i, f"unrelated topic {i}", f"unrelated topic words {i}") for i in range(60)]
    procs.append(_proc(99, "restart dashboard server", "restart dashboard server"))
    hits = _run(procs, "restart dashboard server", "restart dashboard server", scorer=scorer)
    assert scorer.calls[0][1] <= 15
    assert [item["id"] for item in hits] == ["proc_99"]


def test_scorer_failure_propagates_so_the_caller_can_fall_back() -> None:
    class _Broken:
        def score(self, queries, memories):
            raise RuntimeError("model assets missing")

    procs = [_proc(1, "restart dashboard server", "restart dashboard server")]
    try:
        _run(procs, "restart dashboard server", "restart dashboard server", scorer=_Broken())
    except RuntimeError:
        return
    raise AssertionError("expected the scorer error to propagate")


class _ConstantScorer:
    """Every candidate is exactly as relevant as every other."""

    def score(self, queries: list[str], memories: list[str]) -> np.ndarray:
        return np.full((len(memories), len(queries)), 3.0, dtype=np.float32)


_COMMON: dict[str, Any] = {
    "goal": "repair checkout crash loop",
    "summary": "inspect events then repair and validate",
    "context": {"platform": "kubernetes"},
    "steps": [{"summary": "inspect events"}],
    "caveats": [],
    "outcome": "success",
    "verification_status": "verified",
    "evidence": {"helped": 0, "harmed": 0},
}


def _equal(procs, trace=None):
    return retrieve_relevant_procedures(
        procs,
        query="repair checkout crash loop",
        goal="repair checkout crash loop",
        retrieval_context={},
        scorer=_ConstantScorer(),
        encoder=None,
        suppression_trace=trace,
    )


def test_verified_success_ranks_above_equally_relevant_failed_precedent() -> None:
    failed = {
        **_COMMON,
        "id": "proc_failed",
        "outcome": "failure",
        "verification_status": "unverified",
    }
    verified = {**_COMMON, "id": "proc_verified"}
    assert [item["id"] for item in _equal([failed, verified])] == ["proc_verified", "proc_failed"]


def test_equal_procedures_use_content_not_random_source_id_as_tie_breaker() -> None:
    first = {**_COMMON, "id": "proc_random_z", "goal": "repair checkout api"}
    second = {**_COMMON, "id": "proc_random_a", "goal": "repair checkout worker"}
    assert [item["id"] for item in _equal([second, first])] == ["proc_random_z", "proc_random_a"]


def test_duplicate_attempts_do_not_fill_the_capped_result() -> None:
    distinct = {
        **_COMMON,
        "id": "proc_distinct",
        "summary": "restore a known-good revision then validate",
    }
    duplicates = [{**_COMMON, "id": f"proc_duplicate_{index}"} for index in range(3)]
    selected = _equal([*duplicates, distinct])
    assert len(selected) == 2
    assert "proc_distinct" in {item["id"] for item in selected}


def test_declared_duplicate_group_is_suppressed_with_a_representative() -> None:
    trace: list[dict[str, str]] = []
    selected = _equal(
        [
            {**_COMMON, "id": "proc_primary", "deduplication_key": "checkout_repair"},
            {**_COMMON, "id": "proc_alternative", "deduplication_key": "checkout_repair"},
            {
                **_COMMON,
                "id": "proc_distinct",
                "goal": "restore checkout revision",
                "summary": "restore known-good revision",
                "deduplication_key": "checkout_rollback",
            },
            {
                **_COMMON,
                "id": "proc_other",
                "goal": "validate checkout health",
                "summary": "validate health after repair",
                "deduplication_key": "checkout_validate",
            },
        ],
        trace,
    )
    assert len(selected) == 3
    assert len(trace) == 1
    assert trace[0]["candidate_id"] in {"proc_primary", "proc_alternative"}
    assert trace[0]["redundant_with"] in {"proc_primary", "proc_alternative"}


def test_declared_duplicate_group_is_suppressed_below_the_cap() -> None:
    trace: list[dict[str, str]] = []
    group = {**_COMMON, "deduplication_key": "checkout_repair"}
    selected = _equal([{**group, "id": "proc_primary"}, {**group, "id": "proc_alternative"}], trace)
    assert len(selected) == 1
    assert trace[0]["candidate_id"] != trace[0]["redundant_with"]


def test_recorded_attempts_stay_comparable_when_the_cap_is_not_exceeded() -> None:
    # Identical independently recorded attempts both survive under the cap so
    # helped/harmed feedback can still compare them.
    selected = _equal([{**_COMMON, "id": "proc_one"}, {**_COMMON, "id": "proc_two"}])
    assert len(selected) == 2


def test_cjk_goal_with_many_candidates_reaches_the_cross_encoder() -> None:
    # CJK has no whitespace, so word-overlap evidence is ~0. Stage one must not
    # fail and the cross-encoder must still be consulted on a bounded pool.
    scorer = _KeywordScorer()
    procs = [_proc(i, f"无关主题{i}", f"无关主题 内容{i}") for i in range(30)]
    _run(procs, "重启 仪表盘 服务器", "重启 仪表盘 服务器", scorer=scorer)
    assert scorer.calls and scorer.calls[0][1] <= 15
