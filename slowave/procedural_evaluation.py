"""Offline foundations for evaluating procedural-memory discovery.

The types in this module describe immutable experience traces and experiment
outputs.  They deliberately do not store or retrieve production procedures.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, Protocol, Sequence

TRACE_FORMAT_VERSION = 1
TRANSFORM_VERSION = "raw-session-v1"
LABEL_PACKET_VERSION = 1

Origin = Literal["observed", "adapter_supplied", "inferred"]
PairLabel = Literal[
    "same_reusable_pattern",
    "context_specific_variant",
    "related_different_procedure",
    "workstream_continuation",
    "generic_no_signal",
    "warning_failed_pattern",
]


@dataclass(frozen=True)
class SourcedValue:
    value: Any
    origin: Origin


@dataclass(frozen=True)
class TraceEvent:
    event_id: int
    sequence: int
    timestamp: int
    event_type: str
    content: str
    actor_id: str | None
    attributes: Mapping[str, Any] = field(default_factory=dict)
    attribute_origins: Mapping[str, Origin] = field(default_factory=dict)
    relation_refs: tuple[str, ...] = ()
    evidence_refs: tuple[str, ...] = ()


@dataclass(frozen=True)
class ExperienceTrace:
    trace_id: str
    scope_id: str | None
    scope_kind: str | None
    actor_id: str
    started_ts: int
    ended_ts: int | None
    intent: SourcedValue | None
    context: Mapping[str, SourcedValue]
    events: tuple[TraceEvent, ...]
    outcome: SourcedValue | None
    feedback: tuple[SourcedValue, ...]
    source_event_ids: tuple[int, ...]
    transform_version: str = TRANSFORM_VERSION
    format_version: int = TRACE_FORMAT_VERSION


@dataclass(frozen=True)
class FrozenTraceCorpus:
    traces: tuple[ExperienceTrace, ...]
    corpus_sha256: str
    format_version: int = TRACE_FORMAT_VERSION
    transform_version: str = TRANSFORM_VERSION


@dataclass(frozen=True)
class PairScore:
    left_trace_id: str
    right_trace_id: str
    score: float
    channels: Mapping[str, float] = field(default_factory=dict)
    scope_authorized: bool = True
    abstention_reason: str | None = None


@dataclass(frozen=True)
class DiscoveryOutput:
    pair_scores: tuple[PairScore, ...] = ()
    candidate_trace_groups: tuple[tuple[str, ...], ...] = ()
    canonical_pattern_proposals: tuple[Mapping[str, Any], ...] = ()
    warnings: tuple[Mapping[str, Any], ...] = ()
    abstentions: tuple[Mapping[str, Any], ...] = ()


class DiscoveryStrategy(Protocol):
    """One pluggable discovery experiment over a frozen, past-only corpus."""

    name: str

    def discover(self, corpus: FrozenTraceCorpus) -> DiscoveryOutput: ...


PairScorer = Callable[[ExperienceTrace, ExperienceTrace], tuple[float, Mapping[str, float]]]


def _pair_ids(left: str, right: str) -> tuple[str, str]:
    return (left, right) if left < right else (right, left)


class ThresholdDiscoveryStrategy:
    """Conservative complete-linkage adapter for pluggable pair scorers."""

    def __init__(
        self,
        name: str,
        scorer: PairScorer,
        *,
        threshold: float,
        min_support: int = 2,
        enforce_literal_scope: bool = True,
    ) -> None:
        self.name = name
        self.scorer = scorer
        self.threshold = float(threshold)
        self.min_support = max(2, int(min_support))
        self.enforce_literal_scope = enforce_literal_scope

    def discover(self, corpus: FrozenTraceCorpus) -> DiscoveryOutput:
        scores: list[PairScore] = []
        compatible: dict[tuple[str, str], bool] = {}
        for index, left in enumerate(corpus.traces):
            for right in corpus.traces[index + 1 :]:
                pair = _pair_ids(left.trace_id, right.trace_id)
                authorized = not self.enforce_literal_scope or scope_authorized(left, right)
                score, channels = self.scorer(left, right) if authorized else (0.0, {})
                compatible[pair] = authorized and score >= self.threshold
                scores.append(
                    PairScore(
                        left_trace_id=pair[0],
                        right_trace_id=pair[1],
                        score=float(score),
                        channels=dict(channels),
                        scope_authorized=authorized,
                        abstention_reason=None if authorized else "literal_scope_mismatch",
                    )
                )

        # Greedy complete linkage: a trace joins a group only when it is
        # compatible with every existing member, preventing chaining.
        groups: list[list[str]] = []
        for trace in corpus.traces:
            destination = next(
                (
                    group
                    for group in groups
                    if all(
                        compatible.get(_pair_ids(trace.trace_id, member), False) for member in group
                    )
                ),
                None,
            )
            if destination is None:
                groups.append([trace.trace_id])
            else:
                destination.append(trace.trace_id)
        candidates = tuple(tuple(group) for group in groups if len(group) >= self.min_support)
        assigned = {identifier for group in candidates for identifier in group}
        abstentions = tuple(
            {"trace_id": trace.trace_id, "reason": "insufficient_compatible_support"}
            for trace in corpus.traces
            if trace.trace_id not in assigned
        )
        proposals = tuple(
            {
                "proposal_id": hashlib.sha256("\0".join(group).encode()).hexdigest()[:16],
                "trace_ids": list(group),
                "state": "candidate",
                "strategy": self.name,
            }
            for group in candidates
        )
        return DiscoveryOutput(
            pair_scores=tuple(scores),
            candidate_trace_groups=candidates,
            canonical_pattern_proposals=proposals,
            abstentions=abstentions,
        )


def _jsonable(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return {key: _jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in sorted(value.items())}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _canonical_json(value: Any) -> str:
    return json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def freeze_corpus(traces: Sequence[ExperienceTrace]) -> FrozenTraceCorpus:
    ordered = tuple(sorted(traces, key=lambda trace: (trace.started_ts, trace.trace_id)))
    digest = hashlib.sha256(_canonical_json(ordered).encode()).hexdigest()
    return FrozenTraceCorpus(traces=ordered, corpus_sha256=digest)


def write_frozen_corpus(corpus: FrozenTraceCorpus, output: str | Path) -> None:
    Path(output).write_text(json.dumps(_jsonable(corpus), indent=2, ensure_ascii=False) + "\n")


def _strings(metadata: Mapping[str, Any], key: str) -> tuple[str, ...]:
    value = metadata.get(key, ())
    if isinstance(value, str):
        return (value,)
    if isinstance(value, list):
        return tuple(str(item) for item in value)
    return ()


def transform_database(
    db_path: str | Path,
    *,
    scope: str | None = None,
    adapter: Callable[[sqlite3.Row, list[sqlite3.Row]], Mapping[str, Any]] | None = None,
) -> FrozenTraceCorpus:
    """Deterministically transform immutable session/event rows into traces.

    Existing free text remains content.  Optional structure is copied only
    from explicit metadata or from an adapter and is tagged with its origin.
    """
    conn = sqlite3.connect(f"file:{Path(db_path)}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    where = " WHERE scope_id = ?" if scope is not None else ""
    params: tuple[str, ...] = (scope,) if scope is not None else ()
    sessions = conn.execute(
        "SELECT id, agent, scope_id, scope_kind, started_ts, ended_ts, goal, outcome "
        f"FROM sessions{where} ORDER BY started_ts, id",
        params,
    ).fetchall()
    traces: list[ExperienceTrace] = []
    try:
        for session in sessions:
            rows = conn.execute(
                "SELECT id, ts, type, content, metadata_json FROM raw_events "
                "WHERE session_id = ? ORDER BY ts, id",
                (session["id"],),
            ).fetchall()
            supplied = dict(adapter(session, list(rows))) if adapter else {}
            context: dict[str, SourcedValue] = {}
            for key, value in dict(supplied.get("context", {})).items():
                context[str(key)] = SourcedValue(value, "adapter_supplied")
            feedback = tuple(
                SourcedValue(value, "adapter_supplied") for value in supplied.get("feedback", ())
            )
            events: list[TraceEvent] = []
            observed_feedback: list[SourcedValue] = []
            for sequence, row in enumerate(rows):
                try:
                    metadata = json.loads(row["metadata_json"] or "{}")
                except (TypeError, json.JSONDecodeError):
                    metadata = {}
                if not isinstance(metadata, dict):
                    metadata = {}
                origins = {str(key): "observed" for key in metadata}
                events.append(
                    TraceEvent(
                        event_id=int(row["id"]),
                        sequence=sequence,
                        timestamp=int(row["ts"]),
                        event_type=str(row["type"]),
                        content=str(row["content"]),
                        actor_id=(str(metadata["actor_id"]) if metadata.get("actor_id") else None),
                        attributes=metadata,
                        attribute_origins=origins,  # type: ignore[arg-type]
                        relation_refs=_strings(metadata, "relation_refs"),
                        evidence_refs=_strings(metadata, "evidence_refs"),
                    )
                )
                if str(row["type"]) in {"feedback", "remember:feedback"}:
                    observed_feedback.append(SourcedValue(str(row["content"]), "observed"))
            traces.append(
                ExperienceTrace(
                    trace_id=str(session["id"]),
                    scope_id=session["scope_id"],
                    scope_kind=session["scope_kind"],
                    actor_id=str(session["agent"]),
                    started_ts=int(session["started_ts"]),
                    ended_ts=(
                        int(session["ended_ts"]) if session["ended_ts"] is not None else None
                    ),
                    intent=(
                        SourcedValue(str(session["goal"]), "observed") if session["goal"] else None
                    ),
                    context=context,
                    events=tuple(events),
                    outcome=(
                        SourcedValue(str(session["outcome"]), "observed")
                        if session["outcome"] is not None
                        else None
                    ),
                    feedback=tuple(observed_feedback) + feedback,
                    source_event_ids=tuple(event.event_id for event in events),
                )
            )
    finally:
        conn.close()
    return freeze_corpus(traces)


def scope_authorized(left: ExperienceTrace, right: ExperienceTrace) -> bool:
    """Literal patterns may only cross an exact, non-empty scope boundary."""
    return bool(left.scope_id and left.scope_id == right.scope_id)


@dataclass(frozen=True)
class ReplayObservation:
    target_trace_id: str
    target_started_ts: int
    visible_trace_ids: tuple[str, ...]
    discovery: DiscoveryOutput
    query_result: "ReplayQueryResult"


@dataclass(frozen=True)
class TraceCue:
    """Only information available before the target session's outcome."""

    trace_id: str
    scope_id: str | None
    scope_kind: str | None
    actor_id: str
    started_ts: int
    intent: SourcedValue | None
    context: Mapping[str, SourcedValue]


@dataclass(frozen=True)
class ReplayQueryResult:
    recommendation_trace_ids: tuple[str, ...] = ()
    warning_trace_ids: tuple[str, ...] = ()
    abstained: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ReplayExpectation:
    applicable_trace_ids: tuple[str, ...] = ()
    harmful_trace_ids: tuple[str, ...] = ()
    warning_expected: bool = False


@dataclass(frozen=True)
class ReplayMetrics:
    evaluated_targets: int
    known_present_targets: int
    known_present_false_empty_rate: float
    harmful_guidance_rate: float
    no_applicable_targets: int
    correct_abstention_rate: float
    warning_targets: int
    warning_precision: float


class ChronologicalReplay:
    """Replay sessions with a strict timestamp boundary: visible.started_ts < t."""

    def __init__(
        self,
        strategy: DiscoveryStrategy,
        query: Callable[[TraceCue, DiscoveryOutput], ReplayQueryResult],
    ) -> None:
        self.strategy = strategy
        self.query = query

    def run(self, corpus: FrozenTraceCorpus) -> tuple[ReplayObservation, ...]:
        observations: list[ReplayObservation] = []
        for target in corpus.traces:
            visible = tuple(
                trace for trace in corpus.traces if trace.started_ts < target.started_ts
            )
            past = freeze_corpus(visible)
            output = self.strategy.discover(past)
            cue = TraceCue(
                trace_id=target.trace_id,
                scope_id=target.scope_id,
                scope_kind=target.scope_kind,
                actor_id=target.actor_id,
                started_ts=target.started_ts,
                intent=target.intent,
                context=target.context,
            )
            query_result = self.query(cue, output)
            visible_ids = {trace.trace_id for trace in visible}
            surfaced_ids = set(query_result.recommendation_trace_ids) | set(
                query_result.warning_trace_ids
            )
            leaked = surfaced_ids - visible_ids
            if leaked:
                raise ValueError(
                    "replay query surfaced traces outside strict past visibility: "
                    + ", ".join(sorted(leaked))
                )
            observations.append(
                ReplayObservation(
                    target_trace_id=target.trace_id,
                    target_started_ts=target.started_ts,
                    visible_trace_ids=tuple(trace.trace_id for trace in visible),
                    discovery=output,
                    query_result=query_result,
                )
            )
        return tuple(observations)


def summarize_replay(
    observations: Sequence[ReplayObservation],
    expectations: Mapping[str, ReplayExpectation],
) -> ReplayMetrics:
    """Compute safety-oriented longitudinal metrics from independent labels."""
    evaluated = [row for row in observations if row.target_trace_id in expectations]
    known = [row for row in evaluated if expectations[row.target_trace_id].applicable_trace_ids]
    false_empty = sum(
        not (
            set(row.query_result.recommendation_trace_ids)
            & set(expectations[row.target_trace_id].applicable_trace_ids)
        )
        for row in known
    )
    surfaced_count = sum(len(row.query_result.recommendation_trace_ids) for row in evaluated)
    harmful_count = sum(
        len(
            set(row.query_result.recommendation_trace_ids)
            & set(expectations[row.target_trace_id].harmful_trace_ids)
        )
        for row in evaluated
    )
    no_applicable = [
        row
        for row in evaluated
        if not expectations[row.target_trace_id].applicable_trace_ids
        and not expectations[row.target_trace_id].warning_expected
    ]
    correct_abstentions = sum(row.query_result.abstained for row in no_applicable)
    warning_outputs = sum(len(row.query_result.warning_trace_ids) for row in evaluated)
    correct_warnings = sum(
        len(row.query_result.warning_trace_ids)
        for row in evaluated
        if expectations[row.target_trace_id].warning_expected
    )
    warning_targets = sum(expectations[row.target_trace_id].warning_expected for row in evaluated)
    return ReplayMetrics(
        evaluated_targets=len(evaluated),
        known_present_targets=len(known),
        known_present_false_empty_rate=false_empty / len(known) if known else 0.0,
        harmful_guidance_rate=harmful_count / surfaced_count if surfaced_count else 0.0,
        no_applicable_targets=len(no_applicable),
        correct_abstention_rate=(
            correct_abstentions / len(no_applicable) if no_applicable else 0.0
        ),
        warning_targets=warning_targets,
        warning_precision=correct_warnings / warning_outputs if warning_outputs else 0.0,
    )


def build_blind_label_packet(
    corpus: FrozenTraceCorpus,
    baseline_scores: Mapping[tuple[str, str], float],
    challenger_scores: Mapping[tuple[str, str], float],
    *,
    baseline_threshold: float,
    challenger_threshold: float,
    borderline_margin: float = 0.05,
    max_pairs: int = 250,
) -> dict[str, Any]:
    """Create a deterministic blind queue emphasizing disagreements/borderlines."""
    traces = {trace.trace_id: trace for trace in corpus.traces}
    pairs = sorted(set(baseline_scores) | set(challenger_scores))
    selected: list[tuple[tuple[int, float, str, str], dict[str, Any]]] = []
    for raw_left, raw_right in pairs:
        left, right = sorted((raw_left, raw_right))
        if left not in traces or right not in traces:
            continue
        baseline = float(baseline_scores.get((raw_left, raw_right), 0.0))
        challenger = float(challenger_scores.get((raw_left, raw_right), 0.0))
        disagreement = (baseline >= baseline_threshold) != (challenger >= challenger_threshold)
        borderline = (
            abs(baseline - baseline_threshold) <= borderline_margin
            or abs(challenger - challenger_threshold) <= borderline_margin
        )
        if not disagreement and not borderline:
            continue
        a, b = traces[left], traces[right]
        item = {
            "pair_id": hashlib.sha256(f"{left}\0{right}".encode()).hexdigest()[:16],
            "traces": [
                {
                    "trace_id": trace.trace_id,
                    "scope_id": trace.scope_id,
                    "intent": trace.intent.value if trace.intent else None,
                    "events": [event.content for event in trace.events],
                }
                for trace in (a, b)
            ],
            "review": {
                "label": None,
                "confidence": None,
                "high_risk": None,
                "reordered_events": None,
                "same_topic_different_behavior": None,
                "same_behavior_different_target": None,
                "context_or_scope_mismatch": None,
                "notes": "",
            },
        }
        distance = min(abs(baseline - baseline_threshold), abs(challenger - challenger_threshold))
        selected.append(((0 if disagreement else 1, distance, left, right), item))
    selected.sort(key=lambda entry: entry[0])
    items = [item for _, item in selected[: max(1, int(max_pairs))]]
    return {
        "label_packet_version": LABEL_PACKET_VERSION,
        "blinded": True,
        "selection": "configuration_disagreements_and_borderlines",
        "labels": [
            "same_reusable_pattern",
            "context_specific_variant",
            "related_different_procedure",
            "workstream_continuation",
            "generic_no_signal",
            "warning_failed_pattern",
        ],
        "confidence_values": ["low", "medium", "high"],
        "corpus_sha256": corpus.corpus_sha256,
        "eligible_pair_count": len(selected),
        "packet_pair_limit": max(1, int(max_pairs)),
        "pairs": items,
    }
