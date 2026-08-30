"""Procedure contracts and legacy family compatibility."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Iterable

_NAME = re.compile(r"^[a-z][a-z0-9_.:-]{0,63}$")
_SCALAR = (str, int, float, bool)
_FAILED_PROCEDURE_WARNING = (
    "This procedure previously failed; treat it as cautionary evidence, not recommended guidance."
)
_HELPED_BONUS = 0.03
_HELPED_BONUS_CAP = 0.09
_HARMED_PENALTY = 0.06
_HARMED_PENALTY_CAP = 0.18


def _clean_name(value: Any, field: str) -> str:
    text = str(value or "").strip().lower()
    if not _NAME.fullmatch(text):
        raise ValueError(f"{field} must be a lowercase identifier")
    return text


def normalize_facets(value: dict[str, Any] | None, field: str) -> dict[str, Any]:
    """Validate an opaque JSON object without assigning domain semantics."""

    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be a JSON object")

    def visit(item: Any, path: str, depth: int) -> Any:
        if depth > 8:
            raise ValueError(f"{field} is nested too deeply")
        if isinstance(item, dict):
            result: dict[str, Any] = {}
            for key, child in item.items():
                clean_key = _clean_name(key, f"{field} key")
                result[clean_key] = visit(
                    child, f"{path}.{clean_key}" if path else clean_key, depth + 1
                )
            return result
        if isinstance(item, list):
            if any(not isinstance(child, _SCALAR) or child is None for child in item):
                raise ValueError(f"{field}.{path} arrays may contain scalar values only")
            return list(dict.fromkeys(item))
        if isinstance(item, _SCALAR) and item is not None:
            return item
        raise ValueError(f"{field}.{path} must be a scalar, scalar array, or object")

    normalized = visit(value, "", 0)
    if len(json.dumps(normalized, ensure_ascii=False)) > 16_384:
        raise ValueError(f"{field} exceeds 16 KiB")
    return normalized


def flatten_facets(value: dict[str, Any] | None) -> dict[str, frozenset[str]]:
    """Flatten nested JSON into canonical exact-match path/value facets."""

    normalized = normalize_facets(value, "facets")
    out: dict[str, frozenset[str]] = {}

    def visit(item: Any, path: str) -> None:
        if isinstance(item, dict):
            for key, child in item.items():
                visit(child, f"{path}.{key}" if path else key)
        elif isinstance(item, list):
            out[path] = frozenset(_facet_value(child) for child in item)
        else:
            out[path] = frozenset({_facet_value(item)})

    visit(normalized, "")
    return out


def _facet_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return format(value, ".12g")
    return str(value).strip().lower()


@dataclass(frozen=True)
class ProcedureStep:
    summary: str
    operation: str
    target: str

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ProcedureStep":
        if not isinstance(value, dict):
            raise ValueError("procedure.steps entries must be objects")
        summary = str(value.get("summary") or "").strip()
        if not summary:
            raise ValueError("procedure step summary must not be empty")
        return cls(
            summary=summary,
            operation=_clean_name(value.get("operation"), "procedure step operation"),
            target=_clean_name(value.get("target"), "procedure step target"),
        )

    def as_dict(self) -> dict[str, str]:
        return {
            "summary": self.summary,
            "operation": self.operation,
            "target": self.target,
        }

    @property
    def key(self) -> tuple[str, str]:
        return self.operation, self.target


@dataclass(frozen=True)
class ProcedureAttempt:
    session_id: str
    scope_id: str | None
    initial_goal: str
    final_goal: str
    outcome: str
    outcome_summary: str
    summary: str
    preconditions: dict[str, Any]
    retrieval_context: dict[str, Any]
    steps: tuple[ProcedureStep, ...]


@dataclass(frozen=True)
class ProcedureFamily:
    family_id: str
    scope_id: str | None
    member_ids: tuple[str, ...]
    summary: str
    steps: tuple[ProcedureStep, ...]
    preconditions: dict[str, frozenset[str]]
    successes: int
    partials: int
    failures: int
    min_pairwise_alignment: float
    context_facets: dict[str, dict[str, int]]
    warnings: tuple[str, ...]
    source_goals: tuple[str, ...]

    @property
    def status(self) -> str:
        if self.successes >= 2:
            return "supported"
        if self.successes == 0:
            return "warning"
        return "insufficient_success_evidence"

    def as_dict(self) -> dict[str, Any]:
        return {
            "family_id": self.family_id,
            "scope_id": self.scope_id,
            "member_ids": list(self.member_ids),
            "summary": self.summary,
            "steps": [step.as_dict() for step in self.steps],
            "preconditions": {key: sorted(values) for key, values in self.preconditions.items()},
            "successes": self.successes,
            "partials": self.partials,
            "failures": self.failures,
            "min_pairwise_alignment": round(self.min_pairwise_alignment, 4),
            "context_facets": self.context_facets,
            "warnings": list(self.warnings),
            "source_goals": list(self.source_goals),
            "status": self.status,
        }


def validate_procedure(value: dict[str, Any] | None) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("procedure must be a JSON object")
    summary = str(value.get("summary") or "").strip()
    if not summary:
        raise ValueError("procedure.summary must not be empty")
    raw_steps = value.get("steps")
    if not isinstance(raw_steps, list) or not raw_steps:
        raise ValueError("procedure.steps must be a non-empty list")
    for step in raw_steps:
        if not isinstance(step, dict):
            raise ValueError("procedure.steps entries must be objects")
        if not str(step.get("summary") or "").strip():
            raise ValueError("procedure step summary must not be empty")

    rejected = {"preconditions", "retrieval_context"} & value.keys()
    if rejected or any("operation" in step or "target" in step for step in raw_steps):
        raise ValueError(
            "controlled preconditions, retrieval_context, operation, and target "
            "are not part of the procedure contract"
        )
    if value.get("version", 2) != 2:
        raise ValueError("procedure.version must be 2")

    raw_caveats = value.get("caveats", [])
    if not isinstance(raw_caveats, list):
        raise ValueError("procedure.caveats must be a list")
    caveats: list[str] = []
    for caveat in raw_caveats:
        text = str(caveat or "").strip()
        if not text:
            raise ValueError("procedure.caveats entries must not be empty")
        caveats.append(text)
    return {
        "version": 2,
        "summary": summary,
        "context": normalize_facets(value.get("context"), "procedure.context"),
        "steps": [{"summary": str(step["summary"]).strip()} for step in raw_steps],
        "caveats": caveats,
    }


def validate_procedure_uses(value: list[dict[str, Any]] | None) -> list[dict[str, str]]:
    """Validate observed influence without inferring procedure identity."""

    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError("procedure_uses must be a list")
    allowed_use = {"used", "not_used"}
    allowed_effect = {"helped", "no_effect", "harmed", "unknown"}
    normalized: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, dict):
            raise ValueError("procedure_uses entries must be objects")
        procedure_id = str(item.get("procedure_id") or "").strip()
        if not procedure_id:
            raise ValueError("procedure_uses procedure_id must not be empty")
        if procedure_id in seen:
            raise ValueError("procedure_uses procedure_id must be unique")
        seen.add(procedure_id)
        use = str(item.get("use") or "").strip()
        effect = str(item.get("effect") or "").strip()
        contribution = str(item.get("contribution") or "").strip()
        if use not in allowed_use:
            raise ValueError("procedure_uses use must be used or not_used")
        if effect not in allowed_effect:
            raise ValueError("procedure_uses effect must be helped, no_effect, harmed, or unknown")
        if use == "not_used":
            if effect != "unknown" or contribution:
                raise ValueError(
                    "not_used procedure_uses require effect=unknown and no contribution"
                )
        elif not contribution:
            raise ValueError("used procedure_uses require a contribution")
        normalized_item = {
            "procedure_id": procedure_id,
            "use": use,
            "effect": effect,
        }
        if contribution:
            normalized_item["contribution"] = contribution
        normalized.append(normalized_item)
    return normalized


def load_procedures(conn: Any, *, scope: str | None = None) -> list[dict[str, Any]]:
    """Load standalone procedures and aggregate their observed influence."""

    params: list[Any] = []
    where = "WHERE e.type='task_complete'"
    if scope:
        where += " AND s.scope_id = ?"
        params.append(scope)
    rows = conn.execute(
        "SELECT s.id, s.scope_id, COALESCE(s.final_goal, s.initial_goal, s.goal, '') AS goal, "
        "COALESCE(s.outcome, 'unknown') AS outcome, COALESCE(s.outcome_summary, '') AS outcome_summary, "
        "s.started_ts, e.ts AS completed_ts, e.metadata_json "
        "FROM sessions s JOIN raw_events e ON e.session_id=s.id "
        f"{where} ORDER BY s.started_ts, e.id",
        params,
    ).fetchall()
    procedures: list[dict[str, Any]] = []
    by_id: dict[str, dict[str, Any]] = {}
    influence_rows: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for row in rows:
        metadata = json.loads(row["metadata_json"] or "{}")
        for use in metadata.get("procedure_uses") or []:
            influence_rows.append((use, dict(row)))
        raw = metadata.get("procedure")
        if not isinstance(raw, dict) or raw.get("version") != 2:
            continue
        procedure = validate_procedure(raw)
        if procedure is None:
            continue
        item = {
            "id": f"proc_{row['id']}",
            "scope_id": row["scope_id"],
            "goal": str(row["goal"]),
            "summary": procedure["summary"],
            "context": procedure["context"],
            "steps": procedure["steps"],
            "caveats": [
                *procedure["caveats"],
                *([_FAILED_PROCEDURE_WARNING] if row["outcome"] == "failure" else []),
            ],
            "outcome": str(row["outcome"]),
            "outcome_summary": str(row["outcome_summary"]),
            "created_at": row["completed_ts"] or row["started_ts"],
            "evidence": {
                "retrieved": 0,
                "used": 0,
                "not_used": 0,
                "helped": 0,
                "no_effect": 0,
                "harmed": 0,
                "unknown": 0,
            },
            "contributions": [],
        }
        procedures.append(item)
        by_id[item["id"]] = item
    for use, downstream in influence_rows:
        influenced = by_id.get(str(use.get("procedure_id") or ""))
        if influenced is None:
            continue
        evidence = influenced["evidence"]
        evidence["retrieved"] += 1
        if use.get("use") == "used":
            evidence["used"] += 1
            effect = str(use.get("effect") or "unknown")
            evidence[effect] += 1
            influenced["contributions"].append(
                {
                    "effect": effect,
                    "contribution": str(use.get("contribution") or ""),
                    "downstream_session_id": str(downstream["id"]),
                    "downstream_scope_id": downstream["scope_id"],
                    "downstream_goal": str(downstream["goal"]),
                    "downstream_outcome": str(downstream["outcome"]),
                    "downstream_outcome_summary": str(downstream["outcome_summary"]),
                    "created_at": downstream["completed_ts"] or downstream["started_ts"],
                }
            )
        else:
            evidence["not_used"] += 1
    feedback_where = "WHERE f.target_kind = 'procedure' AND f.status = 'accepted'"
    feedback_params: list[Any] = []
    if scope:
        feedback_where += " AND f.scope_id = ?"
        feedback_params.append(scope)
    feedback_rows = conn.execute(
        "SELECT f.retrieval_id, f.target_id, f.assessment, f.effect, f.contribution, "
        "f.created_at, f.rowid FROM feedback_events f "
        f"{feedback_where} ORDER BY f.created_at, f.rowid",
        feedback_params,
    ).fetchall()
    latest_feedback: dict[tuple[str, str], Any] = {}
    for row in feedback_rows:
        latest_feedback[(str(row["retrieval_id"]), str(row["target_id"]))] = row
    for row in latest_feedback.values():
        procedure = by_id.get(str(row["target_id"]))
        if procedure is None:
            continue
        evidence = procedure["evidence"]
        assessment = str(row["assessment"] or "")
        effect = str(row["effect"] or "unknown")
        if assessment in {"used", "not_used"}:
            evidence[assessment] += 1
        if effect in {"helped", "no_effect", "harmed", "unknown"}:
            evidence[effect] += 1
        if assessment == "used":
            procedure["contributions"].append(
                {
                    "effect": effect,
                    "contribution": str(row["contribution"] or ""),
                    "downstream_session_id": "",
                    "downstream_scope_id": "",
                    "downstream_goal": "",
                    "downstream_outcome": "",
                    "downstream_outcome_summary": "",
                    "created_at": row["created_at"],
                }
            )
    return procedures


def retrieve_procedures(
    procedures: list[dict[str, Any]],
    *,
    query: str,
    retrieval_context: dict[str, Any] | None = None,
    encoder: Any = None,
    limit: int = 3,
    min_similarity: float = 0.5,
) -> list[dict[str, Any]]:
    """Admit procedures by raw semantics, then rank with bounded use evidence."""

    cue = " ".join((query, json.dumps(retrieval_context or {}, ensure_ascii=False)))
    cue_terms = set(re.findall(r"\w+", cue.casefold()))
    query_vector = encoder.encode(cue) if encoder is not None else None
    ranked: list[tuple[float, dict[str, Any]]] = []
    for item in procedures:
        text = " ".join(
            (
                item["goal"],
                item["summary"],
                *(step["summary"] for step in item["steps"]),
                *item["caveats"],
                item["outcome_summary"],
                *(entry["contribution"] for entry in item["contributions"]),
                json.dumps(item["context"], ensure_ascii=False),
            )
        )
        if query_vector is not None:
            candidate_vector = encoder.encode(text)
            semantic = max(0.0, float(query_vector.dot(candidate_vector)))
        else:
            terms = set(re.findall(r"\w+", text.casefold()))
            semantic = len(cue_terms & terms) / len(cue_terms) if cue_terms else 0.0
        evidence = item["evidence"]
        # One helpful report resolves a near tie (0.03), while one harmful
        # report outweighs two helpful reports (0.06): safety evidence must
        # dominate endorsement. Three reports saturate each side so repeated
        # feedback cannot overpower semantic relevance or admission.
        utility = min(_HELPED_BONUS_CAP, evidence["helped"] * _HELPED_BONUS) - min(
            _HARMED_PENALTY_CAP, evidence["harmed"] * _HARMED_PENALTY
        )
        score = semantic + utility
        if semantic >= min_similarity:
            result = dict(item)
            result["score"] = round(score, 4)
            ranked.append((score, result))
    ranked.sort(key=lambda pair: (-pair[0], pair[1]["id"]))
    return [item for _, item in ranked[: max(0, limit)]]


def step_alignment(left: Iterable[ProcedureStep], right: Iterable[ProcedureStep]) -> float:
    a, b = tuple(left), tuple(right)
    if not a or not b:
        return 0.0
    table = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i, first in enumerate(a, 1):
        for j, second in enumerate(b, 1):
            table[i][j] = (
                table[i - 1][j - 1] + 1
                if first.key == second.key
                else max(table[i - 1][j], table[i][j - 1])
            )
    return 2 * table[-1][-1] / (len(a) + len(b))


def common_steps(
    left: Iterable[ProcedureStep], right: Iterable[ProcedureStep]
) -> tuple[ProcedureStep, ...]:
    a, b = tuple(left), tuple(right)
    table = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i, first in enumerate(a, 1):
        for j, second in enumerate(b, 1):
            table[i][j] = (
                table[i - 1][j - 1] + 1
                if first.key == second.key
                else max(table[i - 1][j], table[i][j - 1])
            )
    result: list[ProcedureStep] = []
    i, j = len(a), len(b)
    while i and j:
        if a[i - 1].key == b[j - 1].key:
            result.append(a[i - 1])
            i -= 1
            j -= 1
        elif table[i - 1][j] >= table[i][j - 1]:
            i -= 1
        else:
            j -= 1
    return tuple(reversed(result))


def _preconditions_compatible(left: ProcedureAttempt, right: ProcedureAttempt) -> bool:
    a, b = flatten_facets(left.preconditions), flatten_facets(right.preconditions)
    return all(not (a[key].isdisjoint(b[key])) for key in a.keys() & b.keys())


def attempts_compatible(
    left: ProcedureAttempt, right: ProcedureAttempt, *, min_alignment: float = 0.6
) -> bool:
    return (
        left.scope_id == right.scope_id
        and step_alignment(left.steps, right.steps) >= min_alignment
        and _preconditions_compatible(left, right)
    )


def form_families(
    attempts: list[ProcedureAttempt],
    *,
    min_support: int = 2,
    min_alignment: float = 0.6,
) -> list[ProcedureFamily]:
    """Complete-link grouping prevents one bridging execution from chaining families."""

    groups: list[list[ProcedureAttempt]] = []
    for attempt in sorted(attempts, key=lambda item: (item.scope_id or "", item.session_id)):
        eligible = [
            group
            for group in groups
            if all(
                attempts_compatible(attempt, member, min_alignment=min_alignment)
                for member in group
            )
        ]
        if not eligible:
            groups.append([attempt])
            continue
        best = max(
            eligible,
            key=lambda group: (
                sum(step_alignment(attempt.steps, member.steps) for member in group) / len(group)
            ),
        )
        best.append(attempt)
    return [
        family
        for group in groups
        if len(group) >= min_support
        for family in [_consolidate(group)]
        if len(family.steps) >= 2
    ]


def _consolidate(members: list[ProcedureAttempt]) -> ProcedureFamily:
    core = members[0].steps
    for member in members[1:]:
        core = common_steps(core, member.steps)
    pair_scores = [
        step_alignment(members[i].steps, members[j].steps)
        for i in range(len(members))
        for j in range(i + 1, len(members))
    ]
    precondition_maps = [flatten_facets(member.preconditions) for member in members]
    common_preconditions: dict[str, frozenset[str]] = {}
    if precondition_maps:
        for key in set.intersection(*(set(item) for item in precondition_maps)):
            values = set.intersection(*(set(item[key]) for item in precondition_maps))
            if values:
                common_preconditions[key] = frozenset(values)
    context_counts: dict[str, dict[str, int]] = {}
    for member in members:
        for key, context_values in flatten_facets(member.retrieval_context).items():
            bucket = context_counts.setdefault(key, {})
            for value in context_values:
                bucket[value] = bucket.get(value, 0) + 1
    successes = sum(member.outcome == "success" for member in members)
    partials = sum(member.outcome == "partial" for member in members)
    failures = sum(member.outcome == "failure" for member in members)
    warnings = tuple(
        member.outcome_summary or f"{member.outcome}: {member.final_goal}"
        for member in members
        if member.outcome != "success"
    )
    member_ids = tuple(member.session_id for member in members)
    identity = {
        "scope": members[0].scope_id,
        "steps": [step.key for step in core],
        "preconditions": {key: sorted(values) for key, values in common_preconditions.items()},
    }
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:12]
    representative = max(members, key=lambda item: (len(item.summary), item.session_id))
    return ProcedureFamily(
        family_id=f"proc_{digest}",
        scope_id=members[0].scope_id,
        member_ids=member_ids,
        summary=representative.summary,
        steps=core,
        preconditions=common_preconditions,
        successes=successes,
        partials=partials,
        failures=failures,
        min_pairwise_alignment=min(pair_scores) if pair_scores else 1.0,
        context_facets=context_counts,
        warnings=warnings,
        source_goals=tuple(dict.fromkeys(member.final_goal for member in members)),
    )


def _query_score(query: str, family: ProcedureFamily) -> float:
    query_terms = set(re.findall(r"[a-z0-9]+", query.lower()))
    family_terms = set(
        re.findall(r"[a-z0-9]+", " ".join((family.summary, *family.source_goals)).lower())
    )
    return len(query_terms & family_terms) / len(query_terms) if query_terms else 0.0


def _preconditions_match(
    required: dict[str, frozenset[str]], current: dict[str, frozenset[str]]
) -> bool:
    return all(
        key in current and not values.isdisjoint(current[key]) for key, values in required.items()
    )


def retrieve_families(
    families: list[ProcedureFamily],
    *,
    query: str,
    retrieval_context: dict[str, Any] | None = None,
    limit: int = 3,
    min_score: float = 0.15,
    statuses: tuple[str, ...] = ("supported",),
) -> list[dict[str, Any]]:
    current = flatten_facets(retrieval_context)
    hits: list[dict[str, Any]] = []
    for family in families:
        if family.status not in statuses or not _preconditions_match(family.preconditions, current):
            continue
        semantic = _query_score(query, family)
        comparable = 0
        matched = 0
        for key, values in current.items():
            if key not in family.context_facets:
                continue
            comparable += 1
            if any(value in family.context_facets[key] for value in values):
                matched += 1
        context_score = matched / comparable if comparable else 0.0
        score = 0.75 * semantic + 0.25 * context_score
        if score < min_score:
            continue
        item = family.as_dict()
        item.update(
            {
                "score": round(score, 4),
                "query_score": round(semantic, 4),
                "context_score": round(context_score, 4),
            }
        )
        hits.append(item)
    hits.sort(key=lambda item: (item["score"], item["successes"]), reverse=True)
    return hits[: max(0, limit)]


def load_attempts(conn: Any, *, scope: str | None = None) -> tuple[list[ProcedureAttempt], int]:
    """Load canonical structured attempts and count legacy string-step sessions."""

    columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(sessions)").fetchall()}
    initial_expr = "COALESCE(initial_goal, goal)" if "initial_goal" in columns else "goal"
    final_expr = "COALESCE(final_goal, goal)" if "final_goal" in columns else "goal"
    outcome_summary_expr = "COALESCE(outcome_summary, '')" if "outcome_summary" in columns else "''"
    params: list[Any] = []
    where = "WHERE s.outcome IS NOT NULL"
    if scope:
        where += " AND s.scope_id = ?"
        params.append(scope)
    rows = conn.execute(
        f"SELECT s.id, s.scope_id, s.outcome, {initial_expr} AS initial_goal, "
        f"{final_expr} AS final_goal, {outcome_summary_expr} AS outcome_summary, "
        "e.content, e.metadata_json FROM sessions s "
        "JOIN raw_events e ON e.session_id=s.id AND e.type='task_complete' "
        f"{where} ORDER BY s.started_ts, e.id",
        params,
    ).fetchall()
    attempts: list[ProcedureAttempt] = []
    for row in rows:
        metadata = json.loads(row["metadata_json"] or "{}")
        procedure = validate_procedure(metadata.get("procedure"))
        # The family builder is a v1 compatibility experiment. Natural-language
        # Procedures must not be forced back through controlled identity fields.
        if procedure is None or procedure["version"] != 1:
            continue
        attempts.append(
            ProcedureAttempt(
                session_id=str(row["id"]),
                scope_id=row["scope_id"],
                initial_goal=str(row["initial_goal"] or ""),
                final_goal=str(row["final_goal"] or row["initial_goal"] or ""),
                outcome=str(row["outcome"] or "unknown"),
                outcome_summary=str(row["outcome_summary"] or ""),
                summary=procedure["summary"],
                preconditions=procedure["preconditions"],
                retrieval_context=procedure["retrieval_context"],
                steps=tuple(ProcedureStep.from_dict(item) for item in procedure["steps"]),
            )
        )
    legacy_row = conn.execute(
        "SELECT COUNT(DISTINCT s.id) FROM sessions s JOIN raw_events e ON e.session_id=s.id "
        "AND e.type='step' WHERE s.outcome IS NOT NULL" + (" AND s.scope_id = ?" if scope else ""),
        ([scope] if scope else []),
    ).fetchone()
    return attempts, int(legacy_row[0]) if legacy_row else 0
