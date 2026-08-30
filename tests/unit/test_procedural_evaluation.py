from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from slowave.core.config import SlowaveConfig
from slowave.procedural_evaluation import (
    ChronologicalReplay,
    DiscoveryOutput,
    ExperienceTrace,
    ReplayExpectation,
    ReplayQueryResult,
    SourcedValue,
    ThresholdDiscoveryStrategy,
    build_blind_label_packet,
    freeze_corpus,
    scope_authorized,
    summarize_replay,
    transform_database,
)
from slowave.storage.sqlite_db import SQLiteConfig, SQLiteDB


def _database(tmp_path: Path) -> Path:
    path = tmp_path / "trace.db"
    database = SQLiteDB(SQLiteConfig(str(path)))
    database.init_schema(SlowaveConfig.default_schema_path())
    database.close()
    conn = sqlite3.connect(path)
    conn.execute(
        "INSERT INTO sessions "
        "(id, agent, scope_id, scope_kind, started_ts, ended_ts, goal, outcome) "
        "VALUES ('late', 'agent-b', 'project:a', 'project', 20, 30, 'deploy beta', 'success'), "
        "('early', 'agent-a', 'project:a', 'project', 10, 15, 'deploy alpha', 'failure'), "
        "('tie', 'agent-a', 'project:a', 'project', 20, 31, 'same time', 'partial')"
    )
    conn.execute(
        "INSERT INTO raw_events (id, session_id, ts, type, content, metadata_json) VALUES "
        "(4, 'early', 12, 'step', 'verify', '{}'), "
        "(3, 'early', 12, 'tool', 'run', '{\"actor_id\":\"worker\",\"evidence_refs\":[\"log:1\"]}'), "
        "(5, 'early', 13, 'remember:feedback', 'reviewer correction', '{}'), "
        "(8, 'late', 22, 'step', 'deploy', '{}')"
    )
    conn.commit()
    conn.close()
    return path


def test_transform_is_deterministic_ordered_and_preserves_provenance(tmp_path: Path) -> None:
    path = _database(tmp_path)
    first = transform_database(path)
    second = transform_database(path)

    assert first == second
    assert [trace.trace_id for trace in first.traces] == ["early", "late", "tie"]
    early = first.traces[0]
    assert early.source_event_ids == (3, 4, 5)
    assert [event.sequence for event in early.events] == [0, 1, 2]
    assert early.events[0].actor_id == "worker"
    assert early.events[0].evidence_refs == ("log:1",)
    assert early.events[0].attribute_origins["actor_id"] == "observed"
    assert early.feedback == (SourcedValue("reviewer correction", "observed"),)


def test_sparse_trace_does_not_fabricate_unobserved_structure(tmp_path: Path) -> None:
    corpus = transform_database(_database(tmp_path))
    tie = next(trace for trace in corpus.traces if trace.trace_id == "tie")

    assert tie.events == ()
    assert tie.context == {}
    assert tie.feedback == ()
    assert tie.source_event_ids == ()


def test_literal_scope_gate_requires_same_nonempty_scope() -> None:
    def trace(identifier: str, scope: str | None) -> ExperienceTrace:
        return ExperienceTrace(
            identifier,
            scope,
            "project",
            "agent",
            1,
            2,
            SourcedValue("goal", "observed"),
            {},
            (),
            None,
            (),
            (),
        )

    assert scope_authorized(trace("a", "project:x"), trace("b", "project:x"))
    assert not scope_authorized(trace("a", "project:x"), trace("b", "project:y"))
    assert not scope_authorized(trace("a", None), trace("b", None))


def test_chronological_replay_has_strict_past_visibility(tmp_path: Path) -> None:
    corpus = transform_database(_database(tmp_path))

    class RecordingStrategy:
        name = "recording"

        def discover(self, past):
            return DiscoveryOutput(
                abstentions=({"seen": [trace.trace_id for trace in past.traces]},)
            )

    seen_cues = []

    def query(cue, output):
        seen_cues.append(cue)
        return ReplayQueryResult(abstained=True)

    replay = ChronologicalReplay(RecordingStrategy(), query)
    rows = {row.target_trace_id: row for row in replay.run(corpus)}

    assert rows["early"].visible_trace_ids == ()
    assert rows["late"].visible_trace_ids == ("early",)
    assert rows["tie"].visible_trace_ids == ("early",)
    assert "late" not in rows["late"].visible_trace_ids
    assert not hasattr(seen_cues[0], "outcome")
    assert not hasattr(seen_cues[0], "events")


def test_threshold_strategy_enforces_scope_and_complete_linkage() -> None:
    def trace(identifier: str, scope: str) -> ExperienceTrace:
        return ExperienceTrace(
            identifier, scope, "project", "agent", 1, 2, None, {}, (), None, (), ()
        )

    corpus = freeze_corpus(
        [
            trace("a", "project:x"),
            trace("b", "project:x"),
            trace("c", "project:x"),
            trace("z", "project:y"),
        ]
    )
    values = {("a", "b"): 0.9, ("a", "c"): 0.8, ("b", "c"): 0.2}

    def scorer(left, right):
        return values.get(tuple(sorted((left.trace_id, right.trace_id))), 0.95), {"sequence": 1.0}

    output = ThresholdDiscoveryStrategy("test", scorer, threshold=0.5).discover(corpus)

    assert output.candidate_trace_groups == (("a", "b"),)
    cross_scope = next(
        score
        for score in output.pair_scores
        if {score.left_trace_id, score.right_trace_id} == {"a", "z"}
    )
    assert not cross_scope.scope_authorized
    assert cross_scope.abstention_reason == "literal_scope_mismatch"


def test_replay_metrics_cover_false_empty_harm_and_abstention(tmp_path: Path) -> None:
    corpus = transform_database(_database(tmp_path))

    class EmptyStrategy:
        name = "empty"

        def discover(self, past):
            return DiscoveryOutput()

    answers = {
        "early": ReplayQueryResult(abstained=True),
        "late": ReplayQueryResult(recommendation_trace_ids=("early",), abstained=False),
        "tie": ReplayQueryResult(warning_trace_ids=("early",), abstained=False),
    }
    rows = ChronologicalReplay(EmptyStrategy(), lambda cue, output: answers[cue.trace_id]).run(
        corpus
    )
    metrics = summarize_replay(
        rows,
        {
            "early": ReplayExpectation(),
            "late": ReplayExpectation(
                applicable_trace_ids=("other",), harmful_trace_ids=("early",)
            ),
            "tie": ReplayExpectation(warning_expected=True),
        },
    )

    assert metrics.known_present_false_empty_rate == 1.0
    assert metrics.harmful_guidance_rate == 1.0
    assert metrics.correct_abstention_rate == 1.0
    assert metrics.warning_precision == 1.0


def test_chronological_replay_rejects_query_ids_outside_visible_history(tmp_path: Path) -> None:
    corpus = transform_database(_database(tmp_path))

    class EmptyStrategy:
        name = "empty"

        def discover(self, past):
            return DiscoveryOutput()

    replay = ChronologicalReplay(
        EmptyStrategy(),
        lambda cue, output: ReplayQueryResult(recommendation_trace_ids=("late",)),
    )

    with pytest.raises(ValueError, match="outside strict past visibility"):
        replay.run(corpus)


def test_blind_packet_hides_model_scores_and_exposes_independent_labels(tmp_path: Path) -> None:
    corpus = transform_database(_database(tmp_path))
    packet = build_blind_label_packet(
        corpus,
        {("early", "late"): 0.41},
        {("early", "late"): 0.46},
        baseline_threshold=0.40,
        challenger_threshold=0.475,
    )

    assert len(packet["pairs"]) == 1
    assert "baseline" not in str(packet["pairs"][0])
    assert "same_reusable_pattern" in packet["labels"]
    assert packet["pairs"][0]["review"]["label"] is None
