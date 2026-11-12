"""Read-only checks before migrating an existing database to a new embedding space.

Only aggregate counts are returned; memory texts and vectors are never emitted.
This audit does not perform migration and cannot establish its safety by itself.
"""

from __future__ import annotations

import sqlite3
from typing import Any


def audit_embedding_migration(
    conn: sqlite3.Connection, *, migrated_cue_dim: int | None = None
) -> dict[str, Any]:
    """Inventory embedding-derived state and missing reconstruction sources.

    The caller must supply an offline backup connection. No tables are changed.
    Cue prototypes refer to historical snapshots by the original vector, scope
    and task type; the relationship is implicit rather than a foreign key. Those
    rows can only be matched in the *old* space, so ``migrated_cue_dim`` exempts
    the ones a run has already migrated: their source was resolved when they
    moved, and an interrupted migration must stay resumable.

    Only pass ``migrated_cue_dim`` when the migration state *records* that the
    cue step finished in that space. The dimension itself identifies nothing:
    the legacy MiniLM space and the Bekko space are both 384 wide, so a legacy
    prototype would be mistaken for a migrated one and the migration would
    report success while keeping a vector that no snapshot can explain.
    """

    def count(sql: str, params: tuple[Any, ...] = ()) -> int:
        return int(conn.execute(sql, params).fetchone()[0])

    counts = {
        "raw_event_vectors": count("SELECT count(*) FROM raw_events WHERE embedding IS NOT NULL"),
        "episode_vectors": count("SELECT count(*) FROM episodic_memories"),
        "schema_vectors": count("SELECT count(*) FROM schemas WHERE embedding IS NOT NULL"),
        "semantic_prototypes": count("SELECT count(*) FROM semantic_prototypes"),
        "prototype_edges": count("SELECT count(*) FROM prototype_edges"),
        "prototype_support_differs_from_members": count(
            "SELECT count(*) FROM semantic_prototypes p WHERE p.support_count != "
            "(SELECT count(*) FROM episode_prototype_map m WHERE m.prototype_id=p.id)"
        ),
        "schema_facets": count("SELECT count(*) FROM schemas WHERE n_facet_axes > 0"),
        "cue_snapshots": count(
            "SELECT count(*) FROM context_recall_events WHERE cue_embedding IS NOT NULL"
        ),
        "cue_prototypes": count("SELECT count(*) FROM retrieval_cue_prototypes"),
        "feedback_evidence": count("SELECT count(*) FROM schema_retrieval_evidence"),
        "passage_vectors": count("SELECT count(*) FROM schema_passage_embeddings"),
        "procedure_vectors": count("SELECT count(*) FROM procedure_embeddings"),
    }
    missing_sources = {
        "episodes_without_provenance": count(
            "SELECT count(*) FROM episodic_memories e LEFT JOIN episode_text t "
            "ON t.episode_id=e.id WHERE t.episode_id IS NULL OR t.event_ids IS NULL"
        ),
        "episodes_with_missing_event_vectors": count(
            "SELECT count(*) FROM episode_text t WHERE json_valid(t.event_ids) "
            "AND (NOT EXISTS (SELECT 1 FROM json_each(t.event_ids, '$.ids')) OR EXISTS "
            "(SELECT 1 FROM json_each(t.event_ids, '$.ids') j LEFT JOIN raw_events r "
            "ON r.id=j.value WHERE r.id IS NULL OR r.embedding IS NULL))"
        ),
        "episodes_with_invalid_provenance": count(
            "SELECT count(*) FROM episode_text WHERE NOT json_valid(event_ids)"
        ),
        "facets_with_missing_episode_vectors": count(
            "SELECT count(*) FROM schemas s WHERE s.n_facet_axes > 0 AND CASE "
            "WHEN NOT json_valid(s.supporting_episode_ids) THEN 1 "
            "WHEN COALESCE(json_array_length(s.supporting_episode_ids, '$.ids'), 0) < 2 THEN 1 "
            "ELSE EXISTS (SELECT 1 FROM json_each(s.supporting_episode_ids, '$.ids') j "
            "LEFT JOIN episodic_memories e ON e.id=j.value WHERE e.id IS NULL) END"
        ),
        "semantic_prototypes_without_members": count(
            "SELECT count(*) FROM semantic_prototypes p WHERE NOT EXISTS "
            "(SELECT 1 FROM episode_prototype_map m WHERE m.prototype_id=p.id)"
        ),
        "cue_prototypes_without_snapshot": count(
            "SELECT count(*) FROM retrieval_cue_prototypes p WHERE "
            "(? IS NULL OR p.dim IS NULL OR p.dim <> ?) AND NOT EXISTS "
            "(SELECT 1 FROM context_recall_events c WHERE c.cue_embedding=p.embedding "
            "AND c.cue_dim=p.dim AND c.scope_id IS p.scope_id AND c.task_type IS p.task_type)",
            (migrated_cue_dim, migrated_cue_dim),
        ),
    }
    # This is conservative: different stored fields may still render the same
    # canonical cue. Callers must compare canonical strings before deciding.
    ambiguous_snapshot_fields = count(
        "SELECT count(*) FROM retrieval_cue_prototypes p WHERE "
        "(SELECT count(DISTINCT json_array(c.query,c.goal,c.task_type,c.situation_json,"
        "c.requirements_json,c.topics_json,c.entities_json)) FROM context_recall_events c "
        "WHERE c.cue_embedding=p.embedding AND c.cue_dim=p.dim "
        "AND c.scope_id IS p.scope_id AND c.task_type IS p.task_type) > 1"
    )
    return {
        "counts": counts,
        "missing_sources": missing_sources,
        "cue_prototypes_with_distinct_snapshot_fields": ambiguous_snapshot_fields,
    }


class UnreconstructableEmbeddingState(ValueError):
    """Migration must stop rather than discard or misattribute stored evidence."""


def require_reconstructable_sources(
    conn: sqlite3.Connection, *, migrated_cue_dim: int | None = None
) -> None:
    """Fail closed when a stored vector has no valid reconstruction source.

    Ambiguous cue prototypes are not fatal: their snapshots share one old vector,
    so the migration embeds the centroid of every candidate cue instead of
    choosing one arbitrarily (see ``cue_text_alternatives``). Missing or invalid
    provenance is fatal, because no faithful reconstruction exists. Cue
    prototypes already recorded as migrated to ``migrated_cue_dim`` are exempt:
    their source was resolved when they moved, and without the exemption a
    resumed run could never continue.
    """
    audit = audit_embedding_migration(conn, migrated_cue_dim=migrated_cue_dim)
    if any(audit["missing_sources"].values()):
        raise UnreconstructableEmbeddingState("embedding sources are missing or invalid")


def cue_text_alternatives(conn: sqlite3.Connection) -> dict[int, tuple[str, ...]]:
    """Canonical cue text produced by each cue prototype's source snapshots.

    A prototype with several alternatives is one cluster that the old encoder
    collapsed into a single vector; the migration must represent the whole
    cluster rather than pick one member. Text is never returned to callers
    outside the migration.
    """
    import json

    from slowave.core.services.retrieval_access import canonical_cue_text

    rows = conn.execute(
        "SELECT p.id, p.scope_id, p.task_type, c.scope_id, c.task_type, c.query, c.goal, "
        "c.situation_json, c.requirements_json, c.topics_json, c.entities_json "
        "FROM retrieval_cue_prototypes p JOIN context_recall_events c "
        "ON c.cue_embedding=p.embedding AND c.cue_dim=p.dim "
        "AND c.scope_id IS p.scope_id AND c.task_type IS p.task_type "
        "ORDER BY p.id"
    )
    alternatives: dict[int, set[str]] = {}
    for row in rows:
        prototype_id = int(row[0])
        if row[1] != row[3] or row[2] != row[4]:
            raise UnreconstructableEmbeddingState("cue prototype snapshots disagree on scope")
        try:
            cue = canonical_cue_text(
                query=row[5],
                goal=row[6],
                task_type=row[2],
                situation=json.loads(row[7]),
                requirements=json.loads(row[8]),
                topics=json.loads(row[9]),
                entities=json.loads(row[10]),
            )
        except (TypeError, ValueError, AttributeError) as exc:
            raise UnreconstructableEmbeddingState("cue snapshot cannot be reconstructed") from exc
        if not cue:
            raise UnreconstructableEmbeddingState("cue snapshot has no reconstructable cue")
        alternatives.setdefault(prototype_id, set()).add(cue)
    return {key: tuple(sorted(values)) for key, values in alternatives.items()}
