"""Offline embedding migration: plan, steps, fail-closed state and verification."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile

import numpy as np
import pytest

from slowave.core.config import SlowaveConfig
from slowave.core.engine import SlowaveEngine
from slowave.storage import embedding_migration as migration
from slowave.storage.embedding_migration_audit import (
    UnreconstructableEmbeddingState,
    cue_text_alternatives,
)
from slowave.storage.sqlite_db import SQLiteConfig, SQLiteDB


class FakeEncoder:
    """Deterministic encoder whose space differs from anything already stored."""

    def __init__(self, dim: int = 6, salt: str = "a") -> None:
        self.dim = dim
        self.salt = salt

    def encode_many(self, texts: list[str]) -> np.ndarray:
        rows = []
        for text in texts:
            vector = np.zeros(self.dim, dtype=np.float32)
            for index, char in enumerate((self.salt + text).casefold()):
                vector[(ord(char) + index) % self.dim] += 1.0
            vector /= max(float(np.linalg.norm(vector)), 1e-9)
            rows.append(vector)
        return np.asarray(rows, dtype=np.float32)

    def embedding_spec(self) -> dict:
        return {"model": f"fake-{self.dim}", "revision": self.salt, "version": 1}


def _database() -> tuple[SQLiteDB, str]:
    path = tempfile.NamedTemporaryFile(suffix=".db", delete=False).name
    engine = SlowaveEngine(SlowaveConfig(db_path=path, dim=4, disable_encoder=True))
    engine.remember(
        content="Use tabs, never spaces, in Go files.",
        type="constraint",
        scope="project:migration",
    )
    engine.remember(
        content="Staging resets every Monday at 03:00 UTC.",
        type="fact",
        scope="project:migration",
    )
    # A model-free runtime stores no vectors; write the old space explicitly.
    db = SQLiteDB(SQLiteConfig(path=path))
    conn = db.connect()
    for index, row in enumerate(conn.execute("SELECT id FROM schemas ORDER BY id").fetchall()):
        vector = np.full(4, 0.5 + index, dtype="<f4")
        conn.execute(
            "UPDATE schemas SET embedding=?, dim=4 WHERE id=?", (vector.tobytes(), row["id"])
        )
    for index, row in enumerate(conn.execute("SELECT id FROM raw_events ORDER BY id").fetchall()):
        vector = np.full(4, 0.25 + index, dtype="<f4")
        conn.execute(
            "UPDATE raw_events SET embedding=?, dim=4 WHERE id=?", (vector.tobytes(), row["id"])
        )
    conn.commit()
    return db, path


def _cleanup(path: str) -> None:
    for suffix in ("", "-wal", "-shm"):
        if os.path.exists(path + suffix):
            os.remove(path + suffix)


def test_target_fingerprint_tracks_the_encoder_space() -> None:
    first = migration.build_target(FakeEncoder(salt="a"))
    second = migration.build_target(FakeEncoder(salt="b"))

    assert first.dim == 6
    assert first.fingerprint != second.fingerprint
    assert first.fingerprint == migration.build_target(FakeEncoder(salt="a")).fingerprint


def test_migration_moves_every_vector_into_the_target_space() -> None:
    db, path = _database()
    try:
        encoder = FakeEncoder()
        result = migration.migrate_embeddings(
            db, formation_encoder=encoder, search_encoder=encoder, batch=8
        )

        assert result["verification"]["ok"] is True, result["verification"]["problems"]
        assert result["counts"]["schemas"] >= 2
        assert result["verification"]["schema_encoder_agreement"] > 0.999
        state = migration.read_state(db.connect())
        assert state is not None and state["completed_at"] is not None
        assert state["target_fingerprint"] == result["target_fingerprint"]
        row = db.connect().execute("SELECT embedding, dim FROM schemas LIMIT 1").fetchone()
        assert row["dim"] == 6
        assert len(row["embedding"]) == 6 * 4
    finally:
        _cleanup(path)


def test_verification_detects_a_vector_outside_the_target_space() -> None:
    db, path = _database()
    try:
        target = migration.build_target(FakeEncoder())
        db.connect().execute(
            "UPDATE schemas SET embedding=?, dim=? WHERE id=(SELECT min(id) FROM schemas)",
            (np.ones(4, dtype="<f4").tobytes(), 4),
        )
        db.connect().commit()

        report = migration.verify_migration(db, target)

        assert report["ok"] is False
        assert any("schemas.embedding" in problem for problem in report["problems"])
    finally:
        _cleanup(path)


def test_a_failed_step_never_looks_complete(monkeypatch) -> None:
    db, path = _database()
    try:

        def explode(conn, encoder, batch):
            raise RuntimeError("encoder crashed")

        monkeypatch.setattr(migration, "_migrate_schemas", explode)
        with pytest.raises(RuntimeError):
            migration.migrate_embeddings(
                db, formation_encoder=FakeEncoder(), search_encoder=FakeEncoder()
            )

        state = migration.read_state(db.connect())
        assert state is not None
        assert state["completed_at"] is None
        assert state["step"] == "failed"
        assert "encoder crashed" in (state["error"] or "")
    finally:
        _cleanup(path)


def test_an_already_migrated_database_is_refused() -> None:
    db, path = _database()
    try:
        encoder = FakeEncoder()
        migration.migrate_embeddings(db, formation_encoder=encoder, search_encoder=encoder)

        with pytest.raises(UnreconstructableEmbeddingState):
            migration.migrate_embeddings(db, formation_encoder=encoder, search_encoder=encoder)
    finally:
        _cleanup(path)


def test_derived_facet_geometry_is_invalidated_for_recomputation() -> None:
    """Facet axes are a subspace of episode vectors; they cannot cross spaces."""
    db, path = _database()
    try:
        conn = db.connect()
        conn.execute(
            "UPDATE schemas SET facet_axes=?, facet_strengths=?, n_facet_axes=2",
            (np.zeros((2, 4), dtype="<f4").tobytes(), np.ones(2, dtype="<f4").tobytes()),
        )
        conn.commit()

        result = migration._invalidate_derived_geometry(conn)

        assert result["facet_axes_invalidated"] == 2
        row = conn.execute(
            "SELECT count(*) FROM schemas WHERE n_facet_axes>0 OR facet_axes IS NOT NULL"
        ).fetchone()[0]
        assert row == 0
    finally:
        _cleanup(path)


def test_ambiguous_cue_prototype_is_embedded_as_its_cluster_centroid() -> None:
    """Snapshots sharing one old vector are a cluster, not one chosen member."""
    db, path = _database()
    try:
        conn = db.connect()
        vector = np.linspace(0.1, 0.4, 4, dtype="<f4").tobytes()
        conn.execute(
            "INSERT INTO context_recall_events(context_id, retrieval_type, session_id, scope_id, "
            "query, goal, task_type, situation_json, requirements_json, topics_json, "
            "entities_json, cue_embedding, cue_dim, created_at) VALUES"
            "('ctx_1','context','s1','project:migration','alpha','one',NULL,'[]','{}','{}','{}',?,4,1),"
            "('ctx_2','context','s1','project:migration','beta','two',NULL,'[]','{}','{}','{}',?,4,1)",
            (vector, vector),
        )
        conn.execute(
            "INSERT INTO retrieval_cue_prototypes(embedding, dim, scope_id, scope_kind, task_type, "
            "support_count, first_seen_ts, last_seen_ts) "
            "VALUES(?,4,'project:migration','project',NULL,2,1,1)",
            (vector,),
        )
        conn.commit()

        encoder = FakeEncoder()
        # The fixture really is ambiguous; that is what makes the centroid rule matter.
        assert len(cue_text_alternatives(conn)[1]) == 2
        result = migration.migrate_embeddings(db, formation_encoder=encoder)

        assert result["counts"]["cue_prototypes"] == 1
        assert result["counts"]["cue_prototypes_merged"] == 1
        stored = (
            db.connect().execute("SELECT embedding, dim FROM retrieval_cue_prototypes").fetchone()
        )
        assert stored["dim"] == encoder.dim
        # Independent expectation: the cue texts are written out here and pooled by
        # hand, so a wrong cue-text mapping cannot pass this test.
        first, second = encoder.encode_many(["alpha one", "beta two"])
        expected = (first + second) / 2.0
        expected = expected / np.linalg.norm(expected)
        assert np.allclose(np.frombuffer(stored["embedding"], dtype="<f4"), expected, atol=1e-6)
    finally:
        _cleanup(path)


def _seed_foreign_encoder(db, spec: dict, *, tags: list[str]) -> str:
    """Register another encoder and give it vectors, as a legacy install would."""
    from slowave.storage.embedding_store import text_hash

    encoder_id = text_hash(json.dumps(spec, sort_keys=True, separators=(",", ":")))
    conn = db.connect()
    conn.execute(
        "INSERT OR REPLACE INTO embedding_encoders(encoder_id,spec_json,dim,created_at) "
        "VALUES(?,?,?,?)",
        (encoder_id, json.dumps(spec), 4, 1),
    )
    schema_ids = [row[0] for row in conn.execute("SELECT id FROM schemas ORDER BY id")]
    for tag, schema_id in zip(tags, schema_ids * len(tags)):
        conn.execute(
            "INSERT INTO schema_passage_embeddings(schema_id,role,encoder_id,source_hash,dim,"
            "vector,created_at) VALUES(?,'passage',?,?,4,?,1)",
            (
                schema_id,
                encoder_id,
                hashlib.sha256(tag.encode()).hexdigest(),
                np.zeros(4, dtype="<f4").tobytes(),
            ),
        )
    conn.commit()
    return encoder_id


def test_pruning_requires_a_completed_migration() -> None:
    db, path = _database()
    try:
        encoder = FakeEncoder()
        _seed_foreign_encoder(db, {"model": "legacy", "revision": "0"}, tags=["old"])
        with pytest.raises(UnreconstructableEmbeddingState):
            migration.prune_retired_vectors(db, search_encoder=encoder)
        assert db.connect().execute("SELECT count(*) FROM embedding_encoders").fetchone()[0] == 1
    finally:
        _cleanup(path)


def test_pruning_removes_only_superseded_encoders() -> None:
    db, path = _database()
    try:
        encoder = FakeEncoder()
        legacy = _seed_foreign_encoder(
            db, {"model": "legacy-minilm", "revision": "0"}, tags=["a", "b"]
        )
        migration.migrate_embeddings(db, formation_encoder=encoder, search_encoder=encoder)
        active = migration.active_encoder_id(encoder.embedding_spec())

        result = migration.prune_retired_vectors(
            db, search_encoder=encoder, formation_encoder=encoder
        )

        assert legacy in result["retired_encoder_ids"]
        assert result["deleted"]["schema_passage_embeddings"] == 2
        assert result["kept_vectors"] >= 1
        conn = db.connect()
        remaining = {
            row[0]
            for row in conn.execute("SELECT DISTINCT encoder_id FROM schema_passage_embeddings")
        }
        assert remaining == {active}
        assert [row[0] for row in conn.execute("SELECT encoder_id FROM embedding_encoders")] == [
            active
        ]
    finally:
        _cleanup(path)


def test_pruning_refuses_when_the_active_encoder_has_no_vectors() -> None:
    db, path = _database()
    try:
        encoder = FakeEncoder()
        _seed_foreign_encoder(db, {"model": "legacy", "revision": "0"}, tags=["a"])
        migration.migrate_embeddings(db, formation_encoder=encoder, search_encoder=encoder)
        # Remove the active vectors, leaving only the legacy rows.
        db.connect().execute("DELETE FROM schema_passage_embeddings")
        db.connect().commit()

        with pytest.raises(UnreconstructableEmbeddingState):
            migration.prune_retired_vectors(db, search_encoder=encoder)
    finally:
        _cleanup(path)


def test_documented_prune_and_vacuum_command_completes_in_one_run(monkeypatch) -> None:
    """`--prune --vacuum` prunes after the migration records completion."""
    db, path = _database()
    try:
        foreign = _seed_foreign_encoder(
            db, {"model": "legacy-minilm", "revision": "0"}, tags=["a", "b"]
        )
        seen: dict = {}
        real_prune = migration.prune_retired_vectors

        def spy(inner_db, **kwargs):
            seen.update(kwargs)
            return real_prune(inner_db, **kwargs)

        monkeypatch.setattr(migration, "prune_retired_vectors", spy)
        monkeypatch.setattr("slowave.symbolic.encoder.TextEncoder", FakeEncoder)

        assert migration.main(["--db", path, "--prune", "--vacuum"]) == 0

        conn = db.connect()
        state = migration.read_state(conn)
        assert state["step"] == "complete" and state["completed_at"] is not None
        assert seen["vacuum"] is True
        assert conn.execute("SELECT count(*) FROM embedding_encoders").fetchone()[0] == 1
        removed = conn.execute(
            "SELECT count(*) FROM schema_passage_embeddings WHERE encoder_id=?", (foreign,)
        ).fetchone()[0]
        assert removed == 0
    finally:
        _cleanup(path)


def test_prune_failure_keeps_the_migration_complete(monkeypatch) -> None:
    """A prune problem is reported; it must not undo a verified migration."""
    db, path = _database()
    try:
        _seed_foreign_encoder(db, {"model": "legacy-minilm", "revision": "0"}, tags=["a"])
        encoder = FakeEncoder()

        def boom(*args, **kwargs):
            raise RuntimeError("prune exploded")

        monkeypatch.setattr(migration, "prune_retired_vectors", boom)
        result = migration.migrate_embeddings(
            db, formation_encoder=encoder, search_encoder=encoder, prune_retired=True
        )
        assert result["pruned"] is None
        assert "prune exploded" in (result["prune_error"] or "")
        state = migration.read_state(db.connect())
        assert state["step"] == "complete" and state["completed_at"] is not None
        assert not state["error"]
    finally:
        _cleanup(path)


def test_interrupted_cue_migration_leaves_a_resumable_database(monkeypatch) -> None:
    """Cue prototypes and snapshots commit together, so an interruption rolls back."""
    db, path = _database()
    try:
        conn = db.connect()
        vector = np.linspace(0.1, 0.4, 4, dtype="<f4").tobytes()
        conn.execute(
            "INSERT INTO context_recall_events(context_id, retrieval_type, session_id, scope_id, "
            "query, goal, task_type, situation_json, requirements_json, topics_json, "
            "entities_json, cue_embedding, cue_dim, created_at) VALUES"
            "('ctx_1','context','s1','project:migration','alpha','one',NULL,'[]','{}','{}','{}',?,4,1),"
            "('ctx_2','context','s1','project:migration','beta','two',NULL,'[]','{}','{}','{}',?,4,1)",
            (vector, vector),
        )
        conn.execute(
            "INSERT INTO retrieval_cue_prototypes(embedding, dim, scope_id, scope_kind, task_type, "
            "support_count, first_seen_ts, last_seen_ts) "
            "VALUES(?,4,'project:migration','project',NULL,2,1,1)",
            (vector,),
        )
        conn.commit()

        encoder = FakeEncoder()

        def crash(*args, **kwargs):
            raise RuntimeError("interrupted")

        with monkeypatch.context() as patch:
            patch.setattr(migration, "_migrate_cue_snapshots", crash)
            with pytest.raises(RuntimeError):
                migration.migrate_embeddings(db, formation_encoder=encoder)

        rolled_back = db.connect().execute("SELECT dim FROM retrieval_cue_prototypes").fetchone()
        assert rolled_back["dim"] == 4

        retried = migration.migrate_embeddings(db, formation_encoder=encoder)
        assert retried["counts"]["cue_prototypes"] == 1
        migrated = db.connect().execute("SELECT dim FROM retrieval_cue_prototypes").fetchone()
        assert migrated["dim"] == encoder.dim
    finally:
        _cleanup(path)


def test_interruption_after_the_cue_step_still_resumes(monkeypatch) -> None:
    """A kill after the cue step commits must not block the next run."""
    db, path = _database()
    try:
        conn = db.connect()
        vector = np.linspace(0.1, 0.4, 4, dtype="<f4").tobytes()
        conn.execute(
            "INSERT INTO context_recall_events(context_id, retrieval_type, session_id, scope_id, "
            "query, goal, task_type, situation_json, requirements_json, topics_json, "
            "entities_json, cue_embedding, cue_dim, created_at) VALUES"
            "('ctx_1','context','s1','project:migration','alpha','one',NULL,'[]','{}','{}','{}',?,4,1),"
            "('ctx_2','context','s1','project:migration','beta','two',NULL,'[]','{}','{}','{}',?,4,1)",
            (vector, vector),
        )
        conn.execute(
            "INSERT INTO retrieval_cue_prototypes(embedding, dim, scope_id, scope_kind, task_type, "
            "support_count, first_seen_ts, last_seen_ts) "
            "VALUES(?,4,'project:migration','project',NULL,2,1,1)",
            (vector,),
        )
        conn.commit()

        encoder = FakeEncoder()

        def crash_on_later_step(step_name, counts):
            if step_name == "derived_geometry":
                raise RuntimeError("interrupted after the cue step committed")

        with pytest.raises(RuntimeError):
            migration.migrate_embeddings(
                db, formation_encoder=encoder, progress=crash_on_later_step
            )
        committed = db.connect().execute("SELECT dim FROM retrieval_cue_prototypes").fetchone()
        assert committed["dim"] == encoder.dim

        resumed = migration.migrate_embeddings(db, formation_encoder=encoder)
        assert resumed["counts"]["cue_prototypes"] == 0  # nothing left to resolve
        state = migration.read_state(db.connect())
        assert state["completed_at"] is not None and state["step"] == "complete"
    finally:
        _cleanup(path)


def _seed_legacy_cue(db, *, dim: int, snapshots: int) -> bytes:
    """One legacy cue prototype and the snapshots it was built from.

    ``snapshots=0`` models a prototype whose evidence was pruned away: nothing
    can reconstruct the text it was embedded from.
    """
    conn = db.connect()
    vector = np.linspace(0.1, 0.4, dim, dtype="<f4").tobytes()
    for index in range(snapshots):
        conn.execute(
            "INSERT INTO context_recall_events(context_id, retrieval_type, session_id, scope_id, "
            "query, goal, task_type, situation_json, requirements_json, topics_json, "
            "entities_json, cue_embedding, cue_dim, created_at) VALUES"
            "(?,'context','s1','project:migration',?,'goal',NULL,'[]','{}','{}','{}',?,?,1)",
            (f"ctx_{index}", f"query {index}", vector, dim),
        )
    conn.execute(
        "INSERT INTO retrieval_cue_prototypes(embedding, dim, scope_id, scope_kind, task_type, "
        "support_count, first_seen_ts, last_seen_ts) "
        "VALUES(?,?,'project:migration','project',NULL,?,1,1)",
        (vector, dim, snapshots),
    )
    conn.commit()
    return vector


def test_same_dimension_cue_prototype_without_snapshot_blocks_the_migration() -> None:
    """Legacy MiniLM and Bekko are both 384 wide: a dimension proves no space."""
    db, path = _database()
    try:
        _seed_legacy_cue(db, dim=4, snapshots=0)
        # Same width as the legacy space, so nothing distinguishes the vector.
        encoder = FakeEncoder(dim=4, salt="b")

        with pytest.raises(UnreconstructableEmbeddingState, match="missing or invalid"):
            migration.migrate_embeddings(db, formation_encoder=encoder)

        state = migration.read_state(db.connect())
        assert state is None or state["completed_at"] is None
        assert (state or {}).get("cue_fingerprint") is None
        stored = db.connect().execute("SELECT dim FROM retrieval_cue_prototypes").fetchone()
        assert stored["dim"] == 4, "an unattributable prototype must not be migrated"

        report = migration.verify_migration(db, migration.build_target(encoder))
        assert report["cue_space_recorded"] is False
        assert report["ok"] is False, "verification must not accept unrecorded cue vectors"
    finally:
        _cleanup(path)


def test_same_dimension_interruption_after_the_cue_step_still_resumes() -> None:
    """With equal widths only the recorded cue step can prove what was migrated."""
    db, path = _database()
    try:
        _seed_legacy_cue(db, dim=4, snapshots=2)
        encoder = FakeEncoder(dim=4, salt="b")
        target = migration.build_target(encoder)

        def crash_on_later_step(step_name, counts):
            if step_name == "derived_geometry":
                raise RuntimeError("interrupted after the cue step committed")

        with pytest.raises(RuntimeError):
            migration.migrate_embeddings(
                db, formation_encoder=encoder, progress=crash_on_later_step
            )

        conn = db.connect()
        state = migration.read_state(conn)
        assert state["cue_fingerprint"] == target.fingerprint
        prototype = conn.execute("SELECT embedding, dim FROM retrieval_cue_prototypes").fetchone()
        assert prototype["dim"] == encoder.dim
        # The pooled prototype matches no snapshot any more, so matching cannot
        # prove it and the run has to trust the record instead.
        assert (
            conn.execute(
                "SELECT count(*) FROM context_recall_events WHERE cue_embedding=?",
                (prototype["embedding"],),
            ).fetchone()[0]
            == 0
        )

        resumed = migration.migrate_embeddings(db, formation_encoder=encoder)

        assert resumed["resumed_cue_step"] is True
        assert resumed["verification"]["ok"] is True, resumed["verification"]["problems"]
        assert resumed["verification"]["cue_space_recorded"] is True
        assert migration.read_state(db.connect())["completed_at"] is not None
    finally:
        _cleanup(path)


def test_a_state_table_from_an_earlier_run_gains_the_cue_mark() -> None:
    """An interrupted run left the old table behind; it must still be resumable."""
    db, path = _database()
    try:
        encoder = FakeEncoder(dim=4, salt="b")
        target = migration.build_target(encoder)
        conn = db.connect()
        conn.execute(
            "CREATE TABLE embedding_migration_state (job TEXT PRIMARY KEY, target_fingerprint "
            "TEXT NOT NULL, step TEXT NOT NULL, started_at INTEGER NOT NULL, updated_at INTEGER "
            "NOT NULL, completed_at INTEGER, error TEXT, counts_json TEXT)"
        )
        conn.execute(
            "INSERT INTO embedding_migration_state VALUES(?,?,?,?,?,?,?,?)",
            (migration.JOB, target.fingerprint, "failed", 1, 1, None, "interrupted", "{}"),
        )
        conn.commit()
        assert migration.read_state(conn)["cue_fingerprint"] is None

        result = migration.migrate_embeddings(db, formation_encoder=encoder)

        assert result["verification"]["ok"] is True, result["verification"]["problems"]
        assert migration.read_state(db.connect())["cue_fingerprint"] == target.fingerprint
    finally:
        _cleanup(path)


def test_a_database_written_by_another_model_needs_migration() -> None:
    db, path = _database()
    try:
        assert migration.needs_migration(db.connect(), FakeEncoder()) is True
    finally:
        _cleanup(path)


def test_a_database_already_in_the_target_space_needs_no_migration() -> None:
    db, path = _database()
    try:
        encoder = FakeEncoder()
        conn = db.connect()
        for row in conn.execute("SELECT id, content_text FROM schemas").fetchall():
            vector = encoder.encode_many([row["content_text"]])[0]
            conn.execute(
                "UPDATE schemas SET embedding=?, dim=? WHERE id=?",
                (vector.astype("<f4").tobytes(), encoder.dim, row["id"]),
            )
        conn.commit()

        assert migration.needs_migration(conn, encoder) is False
    finally:
        _cleanup(path)


def test_a_migrated_database_needs_no_migration() -> None:
    db, path = _database()
    try:
        encoder = FakeEncoder()
        migration.migrate_embeddings(db, formation_encoder=encoder, search_encoder=encoder)

        assert migration.needs_migration(db.connect(), encoder) is False
    finally:
        _cleanup(path)


def test_an_interrupted_migration_needs_to_be_resumed(monkeypatch) -> None:
    db, path = _database()
    try:
        encoder = FakeEncoder()

        def explode(conn, encoder, batch):
            raise RuntimeError("encoder crashed")

        monkeypatch.setattr(migration, "_migrate_schemas", explode)
        with pytest.raises(RuntimeError):
            migration.migrate_embeddings(db, formation_encoder=encoder, search_encoder=encoder)

        assert migration.needs_migration(db.connect(), encoder) is True
    finally:
        _cleanup(path)


def test_a_database_without_memories_needs_no_migration(tmp_path) -> None:
    path = str(tmp_path / "empty.db")
    engine = SlowaveEngine(SlowaveConfig(db_path=path, dim=4, disable_encoder=True))
    try:
        assert migration.needs_migration(engine.db.connect(), FakeEncoder()) is False
    finally:
        engine.close()
