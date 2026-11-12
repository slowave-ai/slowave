"""Offline migration of every stored vector into one embedding space.

Run only against an exclusive copy of a database: the source file is the
rollback. The migration re-embeds source text, rebuilds geometric state from
migrated members instead of reusing old vectors, and refuses to finish when a
source link is missing or a cue prototype's origin is ambiguous.

Never migrate one column: ``schemas.embedding``, episode and prototype
vectors, cue embeddings and the versioned passage/procedure vectors must all
describe the same space, or search silently compares incomparable vectors.

Command line (on a copy, with no daemon running; there is a single pinned
encoder, so no profile switch is involved):

    python -m slowave.storage.embedding_migration --db /path/to/copy.db --verify-only
    python -m slowave.storage.embedding_migration --db /path/to/copy.db --prune --vacuum
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from slowave.storage.embedding_migration_audit import (
    UnreconstructableEmbeddingState,
    audit_embedding_migration,
    cue_text_alternatives,
    require_reconstructable_sources,
)
from slowave.storage.embedding_reconstruction import (
    reconstruct_centroid,
    reconstruct_episode,
)

VERSION = 1
JOB = "embedding-space-v1"
STATE_TABLE = """
CREATE TABLE IF NOT EXISTS embedding_migration_state (
    job TEXT PRIMARY KEY,
    target_fingerprint TEXT NOT NULL,
    step TEXT NOT NULL,
    started_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL,
    completed_at INTEGER,
    error TEXT,
    counts_json TEXT,
    cue_fingerprint TEXT
)
"""
# Columns added after the table first shipped; ``_ensure_state_table`` adds them
# to a database that an interrupted run already created.
STATE_COLUMNS = {"cue_fingerprint": "TEXT"}


@dataclass(frozen=True)
class MigrationTarget:
    """Identity of the space every stored vector must end up in."""

    formation: dict[str, Any]
    search: dict[str, Any] | None
    fingerprint: str
    dim: int


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def build_target(formation_encoder, search_encoder=None) -> MigrationTarget:
    """Target space from the encoders that will serve the migrated database."""
    formation = dict(formation_encoder.embedding_spec())
    search = dict(search_encoder.embedding_spec()) if search_encoder is not None else None
    payload = {"version": VERSION, "formation": formation, "search": search}
    fingerprint = hashlib.sha256(_canonical(payload).encode()).hexdigest()
    return MigrationTarget(
        formation=formation,
        search=search,
        fingerprint=fingerprint,
        dim=int(formation_encoder.dim),
    )


def _chunks(values: Sequence[Any], size: int) -> Iterator[Sequence[Any]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def _encode(encoder, texts: Sequence[str], batch: int) -> np.ndarray:
    vectors = []
    for chunk in _chunks(list(texts), batch):
        rows = np.asarray(encoder.encode_many(list(chunk)), dtype=np.float32)
        if rows.shape != (len(chunk), int(encoder.dim)):
            raise ValueError("encoder returned an unexpected vector shape")
        if not np.isfinite(rows).all():
            raise ValueError("encoder returned non-finite vectors")
        vectors.append(rows)
    if not vectors:
        return np.zeros((0, int(encoder.dim)), dtype=np.float32)
    return np.concatenate(vectors)


# --------------------------------------------------------------------------- steps


def _migrate_raw_events(conn: sqlite3.Connection, encoder, batch: int) -> int:
    """Re-embed every stored event text, one bounded window at a time.

    Reading the whole table and encoding it in one call kept every row and every
    vector in memory; a window keeps peak memory independent of table size.
    """
    updated = 0
    cursor = -1
    while True:
        rows = conn.execute(
            "SELECT id, content FROM raw_events WHERE embedding IS NOT NULL AND id > ? "
            "ORDER BY id LIMIT ?",
            (cursor, batch),
        ).fetchall()
        if not rows:
            return updated
        vectors = _encode(encoder, [row[1] or "" for row in rows], batch)
        conn.executemany(
            "UPDATE raw_events SET embedding=?, dim=? WHERE id=?",
            [
                (vector.astype("<f4").tobytes(), int(vector.size), int(row[0]))
                for row, vector in zip(rows, vectors)
            ],
        )
        conn.commit()
        updated += len(rows)
        cursor = int(rows[-1][0])


def _rebuild_episodes(conn: sqlite3.Connection) -> int:
    """Episode vectors are the normalized mean of their own event vectors."""
    members: dict[int, np.ndarray] = {
        int(row[0]): np.frombuffer(row[1], dtype="<f4").astype(np.float32)
        for row in conn.execute("SELECT id, embedding FROM raw_events WHERE embedding IS NOT NULL")
    }
    updated = 0
    for episode_id, event_ids in conn.execute(
        "SELECT e.id, t.event_ids FROM episodic_memories e JOIN episode_text t "
        "ON t.episode_id=e.id ORDER BY e.id"
    ).fetchall():
        try:
            ids = [int(value) for value in json.loads(event_ids or "{}").get("ids", [])]
        except (TypeError, ValueError, AttributeError):
            continue
        try:
            vector = reconstruct_episode(ids, members)
        except ValueError:
            continue
        conn.execute(
            "UPDATE episodic_memories SET embedding=?, dim=? WHERE id=?",
            (vector.astype("<f4").tobytes(), int(vector.size), int(episode_id)),
        )
        updated += 1
    conn.commit()
    return updated


def _migrate_schemas(conn: sqlite3.Connection, encoder, batch: int) -> int:
    """Re-embed stored memory text in bounded windows (see ``_migrate_raw_events``)."""
    updated = 0
    cursor = -1
    while True:
        rows = conn.execute(
            "SELECT id, content_text FROM schemas WHERE embedding IS NOT NULL AND id > ? "
            "ORDER BY id LIMIT ?",
            (cursor, batch),
        ).fetchall()
        if not rows:
            return updated
        vectors = _encode(encoder, [row[1] or "" for row in rows], batch)
        conn.executemany(
            "UPDATE schemas SET embedding=?, dim=? WHERE id=?",
            [
                (vector.astype("<f4").tobytes(), int(vector.size), int(row[0]))
                for row, vector in zip(rows, vectors)
            ],
        )
        conn.commit()
        updated += len(rows)
        cursor = int(rows[-1][0])


def _rebuild_prototypes(conn: sqlite3.Connection) -> int:
    """Centroids are rebuilt from migrated episode vectors, not rescaled."""
    episode_vectors: dict[int, np.ndarray] = {
        int(row[0]): np.frombuffer(row[1], dtype="<f4").astype(np.float32)
        for row in conn.execute(
            "SELECT id, embedding FROM episodic_memories WHERE embedding IS NOT NULL"
        )
    }
    members: dict[int, list[int]] = {}
    for prototype_id, episode_id in conn.execute(
        "SELECT prototype_id, episode_id FROM episode_prototype_map "
        "ORDER BY prototype_id, episode_id"
    ).fetchall():
        members.setdefault(int(prototype_id), []).append(int(episode_id))
    updated = 0
    for prototype_id, episode_ids in members.items():
        unique = sorted(set(episode_ids))
        try:
            centroid, variance = reconstruct_centroid(unique, episode_vectors)
        except ValueError:
            continue
        conn.execute(
            "UPDATE semantic_prototypes SET centroid=?, dim=?, variance=?, support_count=? "
            "WHERE id=?",
            (
                centroid.astype("<f4").tobytes(),
                int(centroid.size),
                float(variance),
                len(unique),
                prototype_id,
            ),
        )
        updated += 1
    conn.commit()
    return updated


def _state_columns(conn: sqlite3.Connection) -> set[str]:
    return {str(row[1]) for row in conn.execute("PRAGMA table_info(embedding_migration_state)")}


def _ensure_state_table(conn: sqlite3.Connection) -> None:
    """Create the state table and add columns an interrupted run predates."""
    conn.execute(STATE_TABLE)
    present = _state_columns(conn)
    for column, declaration in STATE_COLUMNS.items():
        if column not in present:
            conn.execute(f"ALTER TABLE embedding_migration_state ADD COLUMN {column} {declaration}")


def read_state(conn: sqlite3.Connection) -> dict[str, Any] | None:
    """Recorded migration state, or None; never writes (plans use read-only connections)."""
    exists = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='embedding_migration_state'"
    ).fetchone()
    if exists is None:
        return None
    recorded = "cue_fingerprint" in _state_columns(conn)
    columns = (
        "job, target_fingerprint, step, started_at, updated_at, completed_at, error, counts_json"
    )
    if recorded:
        columns += ", cue_fingerprint"
    row = conn.execute(
        f"SELECT {columns} FROM embedding_migration_state WHERE job=?",
        (JOB,),
    ).fetchone()
    if row is None:
        return None
    return {
        "job": row[0],
        "target_fingerprint": row[1],
        "step": row[2],
        "started_at": row[3],
        "updated_at": row[4],
        "completed_at": row[5],
        "error": row[6],
        "counts": json.loads(row[7]) if row[7] else {},
        "cue_fingerprint": (row[8] if recorded else None),
    }


def _write_state(
    conn: sqlite3.Connection,
    target: MigrationTarget,
    *,
    step: str,
    counts: dict[str, int] | None = None,
    completed: bool = False,
    error: str | None = None,
    cue_fingerprint: str | None = None,
    commit: bool = True,
) -> None:
    """Record progress; ``cue_fingerprint`` is only ever written with the cue step's commit.

    The cue mark is merged rather than overwritten so a later step's write cannot
    erase which space the cue rows were migrated into.
    """
    now = int(time.time())
    _ensure_state_table(conn)
    existing = read_state(conn)
    started = existing["started_at"] if existing else now
    conn.execute(
        "INSERT INTO embedding_migration_state"
        "(job,target_fingerprint,step,started_at,updated_at,completed_at,error,counts_json,"
        "cue_fingerprint) VALUES(?,?,?,?,?,?,?,?,?) ON CONFLICT(job) DO UPDATE SET "
        "target_fingerprint=excluded.target_fingerprint, step=excluded.step, "
        "updated_at=excluded.updated_at, completed_at=excluded.completed_at, "
        "error=excluded.error, counts_json=excluded.counts_json, "
        "cue_fingerprint=COALESCE(excluded.cue_fingerprint, embedding_migration_state.cue_fingerprint)",
        (
            JOB,
            target.fingerprint,
            step,
            started,
            now,
            now if completed else None,
            error,
            _canonical(counts or (existing or {}).get("counts") or {}),
            cue_fingerprint,
        ),
    )
    if commit:
        conn.commit()


def cue_space_recorded(state: dict[str, Any] | None, target: MigrationTarget) -> bool:
    """Whether the cue step provably finished in *target*'s space on this database.

    This is the only trustworthy evidence that the cue rows are migrated: the
    dimension cannot tell the spaces apart (MiniLM and Bekko are both 384 wide),
    and a resumed run needs a record that the step committed before it was
    interrupted.
    """
    return bool(state and state.get("cue_fingerprint") == target.fingerprint)


def migration_plan(
    conn: sqlite3.Connection, target: MigrationTarget | None = None
) -> dict[str, Any]:
    """Read-only blockers and volumes for a migration of *conn*."""
    state = read_state(conn)
    # Only the recorded cue step -- never the dimension alone -- exempts cue
    # prototypes from the provenance match (see ``audit_embedding_migration``).
    resumed_cues = target is not None and cue_space_recorded(state, target)
    audit = audit_embedding_migration(
        conn, migrated_cue_dim=target.dim if resumed_cues and target else None
    )
    blockers: list[str] = []
    if any(audit["missing_sources"].values()):
        blockers.append("embedding sources are missing or invalid")
    if state is not None and target is not None:
        if state["completed_at"] and state["target_fingerprint"] == target.fingerprint:
            blockers.append("this database is already migrated to the target space")
        if not state["completed_at"] and state["target_fingerprint"] != target.fingerprint:
            blockers.append("a different migration target was started on this database")
    return {
        "audit": audit,
        "cue_prototypes_with_multiple_cues": int(
            audit["cue_prototypes_with_distinct_snapshot_fields"]
        ),
        "state": state,
        "cue_space_recorded": resumed_cues,
        "target_fingerprint": target.fingerprint if target else None,
        "blockers": blockers,
    }


def _cue_texts(conn: sqlite3.Connection) -> dict[int, tuple[str, ...]]:
    """Canonical cue text alternatives per cue prototype.

    Prototypes reach their snapshots through the old vector, so this must run
    before any cue embedding is replaced. A prototype with several alternatives
    represents the whole cluster: the migration embeds their centroid instead of
    attributing one member's text to all of them.
    """
    return cue_text_alternatives(conn)


def _migrate_cue_prototypes(
    conn: sqlite3.Connection,
    encoder,
    batch: int,
    texts: dict[int, tuple[str, ...]],
    *,
    commit: bool = True,
) -> dict[str, int]:
    if not texts:
        return {"cue_prototypes": 0, "cue_prototypes_merged": 0}
    ids = sorted(texts)
    flat: list[str] = []
    spans: list[tuple[int, int]] = []
    for prototype_id in ids:
        alternatives = texts[prototype_id]
        spans.append((len(flat), len(flat) + len(alternatives)))
        flat.extend(alternatives)
    encoded = _encode(encoder, flat, batch)
    rows = []
    merged = 0
    for prototype_id, (start, stop) in zip(ids, spans):
        vector = encoded[start:stop].mean(axis=0).astype(np.float32)
        norm = float(np.linalg.norm(vector))
        vector = vector / norm if norm > 1e-12 else vector
        merged += int(stop - start > 1)
        rows.append((vector.astype("<f4").tobytes(), int(vector.size), prototype_id))
    conn.executemany("UPDATE retrieval_cue_prototypes SET embedding=?, dim=? WHERE id=?", rows)
    if commit:
        conn.commit()
    return {"cue_prototypes": len(ids), "cue_prototypes_merged": merged}


def _migrate_cue_snapshots(
    conn: sqlite3.Connection, encoder, batch: int, *, commit: bool = True
) -> int:
    from slowave.core.services.retrieval_access import canonical_cue_text

    rows = conn.execute(
        "SELECT context_id, query, goal, task_type, situation_json, requirements_json, "
        "topics_json, entities_json FROM context_recall_events "
        "WHERE cue_embedding IS NOT NULL ORDER BY context_id"
    ).fetchall()
    texts: list[str] = []
    targets: list[str] = []
    for row in rows:
        try:
            cue = canonical_cue_text(
                query=row[1],
                goal=row[2],
                task_type=row[3],
                situation=json.loads(row[4]),
                requirements=json.loads(row[5]),
                topics=json.loads(row[6]),
                entities=json.loads(row[7]),
            )
        except (TypeError, ValueError, AttributeError):
            continue
        if not cue:
            continue
        texts.append(cue)
        targets.append(str(row[0]))
    if not texts:
        return 0
    vectors = _encode(encoder, texts, batch)
    conn.executemany(
        "UPDATE context_recall_events SET cue_embedding=?, cue_dim=? WHERE context_id=?",
        [
            (vector.astype("<f4").tobytes(), int(vector.size), context_id)
            for context_id, vector in zip(targets, vectors)
        ],
    )
    if commit:
        conn.commit()
    return len(texts)


def _migrate_cue_geometry(
    conn: sqlite3.Connection,
    encoder,
    batch: int,
    texts: dict[int, tuple[str, ...]],
    target: MigrationTarget,
) -> dict[str, int]:
    """Cue prototypes and their snapshots move together, or not at all.

    The prototype vectors are the only handle on snapshot text (see
    ``_cue_texts``), so a run interrupted between the two steps could never
    resolve those texts again and the database would be unresumable. One
    transaction removes that window: an interruption rolls both back.

    The transaction also records *which space* the cue rows now belong to. A
    merged prototype keeps a pooled vector that matches no snapshot afterwards,
    so the next run cannot verify it by matching; and the dimension proves
    nothing, because MiniLM and Bekko are both 384 wide.
    """
    result = _migrate_cue_prototypes(conn, encoder, batch, texts, commit=False)
    result["cue_snapshots"] = _migrate_cue_snapshots(conn, encoder, batch, commit=False)
    _write_state(
        conn,
        target,
        step="cue_geometry",
        cue_fingerprint=target.fingerprint,
        commit=False,
    )
    conn.commit()
    return result


def _invalidate_derived_geometry(conn: sqlite3.Connection) -> dict[str, int]:
    """Drop geometry that must be recomputed in the new space.

    Facet axes are a PCA of supporting episode vectors; keeping the old axes
    would compare a new-space vector against an old-space subspace. The
    runtime facet backfill recomputes them from the migrated episodes.
    """
    facets = conn.execute(
        "UPDATE schemas SET facet_axes=NULL, facet_strengths=NULL, n_facet_axes=0 "
        "WHERE n_facet_axes>0"
    ).rowcount
    conn.commit()
    return {"facet_axes_invalidated": int(facets)}


def _recompute_similarity_edges(db) -> dict[str, int]:
    """Recompute geometric prototype edges from migrated centroids."""
    from slowave.latent.graph_manager import GraphConfig, GraphManager

    conn = db.connect()
    rows = conn.execute(
        "SELECT id, centroid FROM semantic_prototypes WHERE centroid IS NOT NULL ORDER BY id"
    ).fetchall()
    if not rows:
        return {"similarity_edges": 0}
    ids = [int(row[0]) for row in rows]
    centroids = np.stack([np.frombuffer(row[1], dtype="<f4").astype(np.float32) for row in rows])
    GraphManager(db, GraphConfig()).set_similarity_edges(prototype_ids=ids, centroids=centroids)
    return {"similarity_edges": len(ids)}


def schema_encoder_agreement(conn: sqlite3.Connection, encoder, samples: int = 8) -> float | None:
    """Mean cosine between stored schema vectors and *encoder*'s fresh vectors of their text.

    About 1.0 when the stored vectors came from *encoder*, near 0 when they came from another
    model, even one with the same width. None when no schema has a vector.
    """
    rows = conn.execute(
        "SELECT content_text, embedding FROM schemas WHERE embedding IS NOT NULL "
        "ORDER BY id LIMIT ?",
        (max(int(samples), 1),),
    ).fetchall()
    if not rows:
        return None
    fresh = np.asarray(encoder.encode_many([row[0] or "" for row in rows]), dtype=np.float32)
    cosines = []
    for vector, row in zip(fresh, rows):
        stored = np.frombuffer(row[1], dtype="<f4").astype(np.float32)
        if stored.size != vector.size:
            cosines.append(0.0)
            continue
        norm = max(float(np.linalg.norm(vector) * np.linalg.norm(stored)), 1e-12)
        cosines.append(float(vector @ stored) / norm)
    return float(np.mean(cosines))


# Stored vectors of the target model agree at ~1.0; another model's sit near 0.
_FOREIGN_AGREEMENT = 0.9


def needs_migration(conn: sqlite3.Connection, formation_encoder, search_encoder=None) -> bool:
    """Whether *conn*'s vectors must be migrated into the encoders' space; never writes."""
    state = read_state(conn)
    target = build_target(formation_encoder, search_encoder)
    if state is not None and not state["completed_at"]:
        return True  # started and never finished: resume
    if state is not None and state["target_fingerprint"] == target.fingerprint:
        return False
    agreement = schema_encoder_agreement(conn, formation_encoder)
    return agreement is not None and agreement < _FOREIGN_AGREEMENT


def verify_migration(
    db,
    target: MigrationTarget,
    *,
    formation_encoder=None,
    search_encoder=None,
    samples: int = 8,
    expected: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Re-read the migrated database and prove the space is consistent."""
    conn = db.connect()
    problems: list[str] = []

    def check_dims(table: str, column: str, dim_column: str) -> int:
        bad = conn.execute(
            f"SELECT count(*) FROM {table} WHERE {column} IS NOT NULL AND "
            f"(length({column}) != {target.dim} * 4 OR {dim_column} != ?)",
            (target.dim,),
        ).fetchone()[0]
        if bad:
            problems.append(f"{table}.{column} has {bad} rows outside the target space")
        return int(bad)

    counts = {
        "raw_events": check_dims("raw_events", "embedding", "dim"),
        "episodic_memories": check_dims("episodic_memories", "embedding", "dim"),
        "schemas": check_dims("schemas", "embedding", "dim"),
        "semantic_prototypes": check_dims("semantic_prototypes", "centroid", "dim"),
        "cue_prototypes": check_dims("retrieval_cue_prototypes", "embedding", "dim"),
        "cue_snapshots": check_dims("context_recall_events", "cue_embedding", "cue_dim"),
    }
    # Dimensions cannot prove the cue space (MiniLM and Bekko are both 384 wide),
    # so cue prototypes are only verified through the recorded cue step.
    cue_prototypes = int(
        conn.execute(
            "SELECT count(*) FROM retrieval_cue_prototypes WHERE embedding IS NOT NULL"
        ).fetchone()[0]
    )
    cue_recorded = cue_space_recorded(read_state(conn), target)
    if cue_prototypes and not cue_recorded:
        problems.append(
            "cue prototypes have no recorded cue step for the target space: "
            "their vectors cannot be verified"
        )
    if expected is not None:
        for table, column, key in (
            ("raw_events", "embedding", "raw_event_vectors"),
            ("schemas", "embedding", "schema_vectors"),
            ("semantic_prototypes", "centroid", "semantic_prototypes"),
            ("retrieval_cue_prototypes", "embedding", "cue_prototypes"),
            ("context_recall_events", "cue_embedding", "cue_snapshots"),
        ):
            before = int(expected.get(key) or 0)
            now = int(
                conn.execute(f"SELECT count(*) FROM {table} WHERE {column} IS NOT NULL").fetchone()[
                    0
                ]
            )
            if now < before:
                problems.append(f"{table}.{column} lost {before - now} vectors")
    orphan_evidence = conn.execute(
        "SELECT count(*) FROM schema_retrieval_evidence e LEFT JOIN retrieval_cue_prototypes p "
        "ON p.id=e.cue_prototype_id WHERE p.id IS NULL"
    ).fetchone()[0]
    if orphan_evidence:
        problems.append(f"{orphan_evidence} feedback evidence rows lost their cue prototype")
    non_finite = 0
    for table, column in (
        ("schemas", "embedding"),
        ("semantic_prototypes", "centroid"),
        ("retrieval_cue_prototypes", "embedding"),
    ):
        for (blob,) in conn.execute(
            f"SELECT {column} FROM {table} WHERE {column} IS NOT NULL LIMIT 5000"
        ):
            vector = np.frombuffer(blob, dtype="<f4")
            if vector.size != target.dim or not np.isfinite(vector).all():
                non_finite += 1
    if non_finite:
        problems.append(f"{non_finite} sampled vectors are invalid")
    agreement = None
    if formation_encoder is not None:
        agreement = schema_encoder_agreement(conn, formation_encoder, samples)
        if agreement is not None and agreement < 0.999:
            problems.append(
                f"stored schema vectors disagree with the target encoder ({agreement:.4f})"
            )
    expected_search = None
    if search_encoder is not None:
        encoded = json.dumps(
            dict(search_encoder.embedding_spec()), sort_keys=True, separators=(",", ":")
        )
        encoder_id = hashlib.sha256(encoded.encode()).hexdigest()
        expected_search = int(
            conn.execute(
                "SELECT count(*) FROM schema_passage_embeddings WHERE encoder_id=?",
                (encoder_id,),
            ).fetchone()[0]
        )
        if not expected_search:
            problems.append("no passage vectors exist for the target search encoder")
    return {
        "ok": not problems,
        "problems": problems,
        "bad_rows": counts,
        "cue_prototypes": cue_prototypes,
        "cue_space_recorded": cue_recorded,
        "schema_encoder_agreement": agreement,
        "passage_vectors_for_target": expected_search,
    }


def _backfill_search_vectors(db, search_encoder, batch: int) -> dict[str, int]:
    """Versioned passage/procedure vectors for the new space; old rows stay."""
    from slowave.storage.embedding_store import EmbeddingStore

    conn = db.connect()
    store = EmbeddingStore(db)
    schemas = conn.execute("SELECT id, content_text FROM schemas ORDER BY id").fetchall()
    passages = 0
    for chunk in _chunks(list(schemas), max(batch * 8, 64)):
        store.vectors(
            "schema", [(row[0], "passage", row[1] or "") for row in chunk], search_encoder
        )
        passages += len(chunk)
    procedures = conn.execute(
        "SELECT procedure_id, applicability_text, strategy_text FROM procedure_search_documents "
        "ORDER BY procedure_id"
    ).fetchall()
    entries = [
        (row[0], role, row[column])
        for row in procedures
        for role, column in (("applicability", 1), ("strategy", 2))
        if row[column]
    ]
    counted = 0
    for chunk in _chunks(entries, max(batch * 8, 64)):
        store.vectors("procedure", list(chunk), search_encoder)
        counted += len(chunk)
    return {"passage_vectors": passages, "procedure_vectors": counted}


def migrate_embeddings(
    db,
    *,
    formation_encoder,
    search_encoder=None,
    batch: int = 32,
    progress: Callable[[str, dict[str, int]], None] | None = None,
    skip_search_vectors: bool = False,
    prune_retired: bool = False,
    vacuum: bool = False,
) -> dict[str, Any]:
    """Migrate every stored vector of *db* into the encoders' space.

    The caller must pass a disposable copy. Progress is recorded per step so an
    interrupted run never looks finished; ``completed_at`` is written only after
    verification passes, and pruning runs *after* that mark so a prune problem
    cannot undo a verified migration.
    """
    conn = db.connect()
    target = build_target(formation_encoder, search_encoder)
    resumed_cues = cue_space_recorded(read_state(conn), target)
    require_reconstructable_sources(conn, migrated_cue_dim=target.dim if resumed_cues else None)
    plan = migration_plan(conn, target)
    if plan["blockers"]:
        raise UnreconstructableEmbeddingState("; ".join(plan["blockers"]))
    counts: dict[str, int] = {}
    report = progress or (lambda step, partial: None)

    def step(name: str, work: Callable[[], int | dict[str, int]]) -> None:
        _write_state(conn, target, step=name, counts=counts)
        result = work()
        counts.update(result if isinstance(result, dict) else {name: int(result)})
        _write_state(conn, target, step=name, counts=counts)
        report(name, dict(counts))

    try:
        # Resolve cue text while the old cue vectors still identify snapshots.
        cue_texts = _cue_texts(conn)
        counts["cue_prototypes_resolved"] = len(cue_texts)
        step("raw_events", lambda: _migrate_raw_events(conn, formation_encoder, batch))
        step("episodes", lambda: _rebuild_episodes(conn))
        step("schemas", lambda: _migrate_schemas(conn, formation_encoder, batch))
        step("prototypes", lambda: _rebuild_prototypes(conn))
        step(
            "cue_geometry",
            lambda: _migrate_cue_geometry(conn, formation_encoder, batch, cue_texts, target),
        )
        if search_encoder is not None and not skip_search_vectors:
            step("search_vectors", lambda: _backfill_search_vectors(db, search_encoder, batch))
        step("derived_geometry", lambda: _invalidate_derived_geometry(conn))
        step("similarity_edges", lambda: _recompute_similarity_edges(db))
        verification = verify_migration(
            db,
            target,
            formation_encoder=formation_encoder,
            search_encoder=search_encoder,
            expected=plan["audit"]["counts"],
        )
        if not verification["ok"]:
            raise UnreconstructableEmbeddingState("; ".join(verification["problems"]))
    except Exception as exc:  # record the failure, never mark complete
        # Discard anything the failed step had written but not committed, so a
        # half-applied step cannot make the database unresumable.
        conn.rollback()
        _write_state(conn, target, step="failed", counts=counts, error=repr(exc))
        raise
    _write_state(conn, target, step="complete", counts=counts, completed=True)
    # Pruning deletes the rollback material, so it only runs once the migration
    # is verified AND recorded as complete. A prune problem is reported, never
    # turned into a failed migration the database has already earned.
    pruned = None
    prune_error = None
    if prune_retired and search_encoder is not None:
        try:
            pruned = prune_retired_vectors(
                db,
                search_encoder=search_encoder,
                formation_encoder=formation_encoder,
                vacuum=vacuum,
            )
            report("pruned", pruned["deleted"])
        except Exception as exc:
            prune_error = repr(exc)
            report("prune_failed", {})
    return {
        "target_fingerprint": target.fingerprint,
        "dim": target.dim,
        "counts": counts,
        "resumed_cue_step": resumed_cues,
        "verification": verification,
        "pruned": pruned,
        "prune_error": prune_error,
    }


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point; always operates on the file given with ``--db``."""
    from slowave.storage.sqlite_db import SQLiteConfig, SQLiteDB

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True, help="path to an exclusive database copy")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument(
        "--prune-only",
        action="store_true",
        help="delete vectors belonging to retired encoders (requires a completed migration)",
    )
    parser.add_argument("--prune", action="store_true", help="prune after a successful migration")
    parser.add_argument("--vacuum", action="store_true", help="VACUUM after pruning")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args(argv)
    path = Path(args.db)
    if not path.exists():
        parser.error(f"{path} does not exist")
    db = SQLiteDB(SQLiteConfig(path=str(path)))
    from slowave.symbolic.encoder import TextEncoder

    formation = TextEncoder()
    formation.dim  # load now so a broken model fails before any write
    search = formation
    target = build_target(formation, search)
    if args.prune_only:
        print(
            json.dumps(
                prune_retired_vectors(
                    db,
                    search_encoder=search,
                    formation_encoder=formation,
                    vacuum=args.vacuum,
                ),
                indent=2,
            )
        )
        return 0
    if args.verify_only:
        result = verify_migration(
            db,
            target,
            formation_encoder=formation,
            search_encoder=search,
            expected=migration_plan(db.connect(), target)["audit"]["counts"],
        )
        print(json.dumps(result, indent=2))
        return 0 if result["ok"] else 1
    result = migrate_embeddings(
        db,
        formation_encoder=formation,
        search_encoder=search,
        batch=args.batch,
        prune_retired=args.prune,
        vacuum=args.vacuum,
        progress=(
            (lambda step, counts: print(step, counts, flush=True)) if args.progress else None
        ),
    )
    print(json.dumps(result, indent=2))
    if result.get("prune_error"):
        print(f"prune failed: {result['prune_error']}", file=sys.stderr)
        return 1
    return 0


def active_encoder_id(spec: dict) -> str:
    """Registry id `EmbeddingStore` assigns to an embedding spec."""
    encoded = json.dumps(dict(spec), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


def prune_retired_vectors(
    db,
    *,
    search_encoder,
    formation_encoder=None,
    vacuum: bool = False,
) -> dict[str, Any]:
    """Delete versioned vectors that belong to a retired encoder.

    Only runs on a database this migration completed: the rollback point is the
    untouched source copy, not a second set of vectors. Refuses when the active
    encoder holds no vectors, so a mis-specified encoder can never empty the
    store.
    """
    conn = db.connect()
    keep = active_encoder_id(search_encoder.embedding_spec())
    state = read_state(conn)
    if state is None or not state["completed_at"]:
        raise UnreconstructableEmbeddingState(
            "refusing to prune before a completed embedding migration"
        )
    if formation_encoder is not None:
        target = build_target(formation_encoder, search_encoder)
        if state["target_fingerprint"] != target.fingerprint:
            raise UnreconstructableEmbeddingState(
                "refusing to prune: the completed migration targets a different space"
            )
    tables = ("schema_passage_embeddings", "procedure_embeddings")
    kept = sum(
        int(conn.execute(f"SELECT count(*) FROM {table} WHERE encoder_id=?", (keep,)).fetchone()[0])
        for table in tables
    )
    if kept == 0:
        raise UnreconstructableEmbeddingState(
            "refusing to prune: the active encoder holds no vectors"
        )
    retired = [
        str(row[0])
        for row in conn.execute(
            "SELECT encoder_id FROM embedding_encoders WHERE encoder_id<>?", (keep,)
        )
    ]
    deleted: dict[str, int] = {}
    with conn:
        for table in tables:
            removed = 0
            for encoder_id in retired:
                removed += conn.execute(
                    f"DELETE FROM {table} WHERE encoder_id=?", (encoder_id,)
                ).rowcount
            deleted[table] = int(removed)
        deleted["embedding_encoders"] = int(
            conn.execute("DELETE FROM embedding_encoders WHERE encoder_id<>?", (keep,)).rowcount
        )
    if vacuum:
        conn.execute("VACUUM")
    return {
        "active_encoder_id": keep,
        "retired_encoder_ids": retired,
        "deleted": deleted,
        "kept_vectors": kept,
    }


if __name__ == "__main__":  # pragma: no cover - manual entry point
    raise SystemExit(main())
