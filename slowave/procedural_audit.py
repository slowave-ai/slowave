"""Read-only export of human-reviewable procedural-memory candidates.

This module deliberately exports *candidate clusters*, not stored procedures.
It shares the validated embedding/alignment/average-linkage definition from
``slowave.symbolic.procedural`` and never imports or constructs SlowaveEngine.
"""

from __future__ import annotations

import sqlite3
import time
from typing import Any

AUDIT_FORMAT_VERSION = 1
DEFAULT_THRESHOLD = 0.4


def _connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _load_sessions(conn: sqlite3.Connection, scope: str | None) -> list[dict[str, Any]]:
    params: tuple[str, ...] = (scope,) if scope else ()
    scope_clause = "AND scope_id = ?" if scope else ""
    rows = conn.execute(
        "SELECT id, goal, outcome, scope_id FROM sessions "
        "WHERE goal IS NOT NULL AND outcome IS NOT NULL "
        + scope_clause
        + " ORDER BY started_ts, id",
        params,
    ).fetchall()
    sessions: list[dict[str, Any]] = []
    for row in rows:
        step_rows = conn.execute(
            "SELECT content FROM raw_events WHERE session_id = ? AND type = 'step' ORDER BY ts, id",
            (row["id"],),
        ).fetchall()
        steps = [str(step["content"]) for step in step_rows if step["content"]]
        sessions.append(
            {
                "id": str(row["id"]),
                "goal": str(row["goal"] or ""),
                "outcome": str(row["outcome"] or "unknown"),
                "scope_id": row["scope_id"],
                "step_contents": steps,
                "has_steps": bool(steps),
            }
        )
    return sessions


def build_audit_export(
    db_path: str,
    *,
    min_sessions: int = 3,
    max_clusters: int = 20,
    scope: str | None = None,
) -> dict[str, Any]:
    """Build a JSON-serializable, read-only review document for candidate clusters."""
    min_sessions = max(2, min(20, int(min_sessions)))
    max_clusters = max(1, min(50, int(max_clusters)))
    scope = scope.strip() if scope else None

    conn = _connect(db_path)
    try:
        sessions = _load_sessions(conn, scope)
        from slowave.symbolic.procedural_memory import form_families, load_attempts

        structured_attempts, legacy_trace_sessions = load_attempts(conn, scope=scope)
    finally:
        conn.close()

    if structured_attempts:
        families = form_families(structured_attempts, min_support=min_sessions)[:max_clusters]
        by_id = {attempt.session_id: attempt for attempt in structured_attempts}
        return {
            "audit_format_version": AUDIT_FORMAT_VERSION + 1,
            "generated_at": int(time.time()),
            "procedure_memory_status": "structured_exploration_advisory",
            "method": "controlled_steps+complete_link",
            "scope": scope,
            "min_sessions": min_sessions,
            "structured_attempts": len(structured_attempts),
            "legacy_trace_sessions_excluded": legacy_trace_sessions,
            "gate": {
                "status": "families_found" if families else "insufficient_structured_data",
                "criterion": f"at least {min_sessions} compatible structured attempts",
                "pass": bool(families),
            },
            "clusters": [
                {
                    "audit_id": family.family_id,
                    "metrics": {
                        "session_count": len(family.member_ids),
                        "successes": family.successes,
                        "partials": family.partials,
                        "failures": family.failures,
                        "min_pairwise_alignment": family.min_pairwise_alignment,
                        "status": family.status,
                    },
                    "summary": family.summary,
                    "common_steps": [step.as_dict() for step in family.steps],
                    "preconditions": {
                        key: sorted(values) for key, values in family.preconditions.items()
                    },
                    "context_facets": family.context_facets,
                    "warnings": list(family.warnings),
                    "members": [
                        {
                            "session_id": member_id,
                            "goal": by_id[member_id].final_goal,
                            "outcome": by_id[member_id].outcome,
                            "outcome_summary": by_id[member_id].outcome_summary,
                            "steps": [step.as_dict() for step in by_id[member_id].steps],
                        }
                        for member_id in family.member_ids
                    ],
                    "review": {"label": None, "notes": ""},
                }
                for family in families
            ],
        }

    step_sessions = [session for session in sessions if session["has_steps"]]
    document: dict[str, Any] = {
        "audit_format_version": AUDIT_FORMAT_VERSION,
        "generated_at": int(time.time()),
        "procedure_memory_status": "inspection_only_not_stored_or_retrieved",
        "method": "embedding+alignment+average_linkage",
        "threshold": DEFAULT_THRESHOLD,
        "scope": scope,
        "min_sessions": min_sessions,
        "eligible_sessions": len(sessions),
        "step_sessions": len(step_sessions),
        "review_instructions": {
            "labels": [
                "recommend",
                "warn",
                "reject",
            ],
            "requirement": (
                "Judge full member traces and whether a future matching session would benefit "
                "from action guidance, avoidance guidance, or neither."
            ),
            "promotion_hint": (
                "recommend = future action guidance; warn = contextual avoidance guidance; "
                "reject = noise, project history, generic workflow, or declarative knowledge."
            ),
        },
        "clusters": [],
    }
    if len(step_sessions) < min_sessions:
        document["gate"] = {
            "status": "insufficient_data",
            "criterion": f"at least {min_sessions} sessions with recorded steps",
            "pass": False,
        }
        return document

    from slowave.symbolic.encoder import TextEncoder
    from slowave.symbolic.procedural import (
        build_embedding_caches,
        cluster_sessions,
        rank_clusters,
    )

    encoder = TextEncoder()
    step_cache, goal_cache = build_embedding_caches(sessions, encoder)
    clusters = cluster_sessions(sessions, step_cache, goal_cache, threshold=DEFAULT_THRESHOLD)
    ranked = rank_clusters(clusters, min_sessions, max_clusters, goal_cache=goal_cache)
    qualifying = sum(1 for cluster in ranked if cluster["goal_coherence"] >= 0.3)
    document["gate"] = {
        "status": "target_met" if qualifying >= 5 else "not_met",
        "criterion": ">=5 clusters with goal_coherence >=0.3",
        "clusters_qualifying": qualifying,
        "pass": qualifying >= 5,
    }

    for rank, cluster in enumerate(ranked, start=1):
        members = clusters[cluster["cluster_id"]]
        document["clusters"].append(
            {
                "audit_id": f"candidate_{rank:02d}",
                "cluster_id": cluster["cluster_id"],
                "metrics": {
                    key: cluster[key]
                    for key in (
                        "session_count",
                        "successes",
                        "failures",
                        "success_rate",
                        "goal_coherence",
                        "composite",
                        "anti_pattern",
                    )
                },
                "competes_with": cluster.get("competes_with", []),
                "members": [
                    {
                        "session_id": member["id"],
                        "scope_id": member["scope_id"],
                        "goal": member["goal"],
                        "outcome": member["outcome"],
                        "steps": member["step_contents"],
                    }
                    for member in members
                ],
                "review": {
                    "label": None,
                    "action_specific": None,
                    "sequence_consistent": None,
                    "target_independent": None,
                    "outcome_supported": None,
                    "future_useful": None,
                    "notes": "",
                },
            }
        )
    return document
