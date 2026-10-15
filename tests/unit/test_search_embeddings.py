"""Persistence, upgrade and invalidation guarantees for search projections."""

import sqlite3

import numpy as np
import pytest

from slowave import ops
from slowave.core.config import SlowaveConfig
from slowave.core.engine import SlowaveEngine
from slowave.storage.embedding_store import EmbeddingStore, text_hash


class Encoder:
    dim = 4

    def __init__(self, revision="one"):
        self.revision = revision
        self.calls = []
        self.hook = None

    def embedding_spec(self):
        return {"revision": self.revision, "format": "float32-le"}

    def encode_many(self, texts):
        self.calls.extend(texts)
        if self.hook:
            self.hook()
        return np.array([[len(text), 1, 2, 3] for text in texts], dtype=np.float32)


@pytest.fixture
def engine(tmp_path):
    eng = SlowaveEngine(
        SlowaveConfig(db_path=str(tmp_path / "memory.db"), dim=4, disable_encoder=True)
    )
    yield eng
    eng.close()


def schema(engine, text="original"):
    return engine.schemas.create(
        content_text=text,
        embedding=np.array([1, 0, 0, 0], dtype=np.float32),
        scope_id="project:test",
    )


def test_vectors_survive_restart_and_only_changed_text_is_reencoded(engine):
    sid = schema(engine)
    encoder = Encoder()
    store = EmbeddingStore(engine.db)
    expected = store.vectors("schema", [(sid, "passage", "original")], encoder)
    engine.db.close()
    np.testing.assert_array_equal(
        store.vectors("schema", [(sid, "passage", "original")], encoder), expected
    )
    assert encoder.calls == ["original"]
    conn = engine.db.connect()
    with conn:
        conn.execute("UPDATE schemas SET content_text='edited' WHERE id=?", (sid,))
    store.vectors("schema", [(sid, "passage", "edited")], encoder)
    assert encoder.calls == ["original", "edited"]
    store.vectors("schema", [(sid, "passage", "edited")], Encoder("two"))
    assert conn.execute("SELECT COUNT(*) FROM schema_passage_embeddings").fetchone()[0] == 2


def test_concurrent_source_edit_cannot_publish_obsolete_vector(engine):
    sid = schema(engine)
    encoder = Encoder()

    def edit():
        conn = engine.db.connect()
        with conn:
            conn.execute("UPDATE schemas SET content_text='new' WHERE id=?", (sid,))

    encoder.hook = edit
    EmbeddingStore(engine.db).vectors("schema", [(sid, "passage", "original")], encoder)
    assert (
        engine.db.connect().execute("SELECT COUNT(*) FROM schema_passage_embeddings").fetchone()[0]
        == 0
    )


def test_interrupted_batches_resume_from_missing_rows(engine):
    entries = [(schema(engine, str(i)), "passage", str(i)) for i in range(20)]
    encoder = Encoder()
    count = 0

    def fail_second_batch():
        nonlocal count
        count += 1
        if count == 2:
            raise RuntimeError("interrupted")

    with pytest.raises(RuntimeError, match="interrupted"):
        EmbeddingStore(engine.db).vectors("schema", entries, encoder, heartbeat=fail_second_batch)
    assert (
        engine.db.connect().execute("SELECT COUNT(*) FROM schema_passage_embeddings").fetchone()[0]
        == 16
    )
    encoder.calls.clear()
    EmbeddingStore(engine.db).vectors("schema", entries, encoder)
    assert len(encoder.calls) == 4


def test_constraints_cascade_and_no_open_transaction_inference(engine):
    sid = schema(engine)
    encoder = Encoder()
    store = EmbeddingStore(engine.db)
    store.vectors("schema", [(sid, "passage", "original")], encoder)
    conn = engine.db.connect()
    row = conn.execute("SELECT encoder_id FROM embedding_encoders").fetchone()
    with pytest.raises(sqlite3.IntegrityError):
        with conn:
            conn.execute("UPDATE embedding_encoders SET dim=5")
    with pytest.raises(sqlite3.IntegrityError):
        with conn:
            conn.execute("UPDATE schema_passage_embeddings SET vector=?", (b"bad",))
    with pytest.raises(sqlite3.IntegrityError):
        with conn:
            conn.execute(
                "INSERT INTO schema_passage_embeddings VALUES(999,'passage',?,?,4,?,0)",
                (row[0], text_hash("x"), np.ones(4, dtype="<f4").tobytes()),
            )
    conn.execute("UPDATE schemas SET content_text='pending' WHERE id=?", (sid,))
    with pytest.raises(RuntimeError, match="open write transaction"):
        store.vectors("schema", [(sid, "passage", "pending")], encoder)
    conn.rollback()
    with conn:
        conn.execute("DELETE FROM schemas WHERE id=?", (sid,))
    assert conn.execute("SELECT COUNT(*) FROM schema_passage_embeddings").fetchone()[0] == 0
    assert not conn.execute("PRAGMA foreign_key_check").fetchall()


def test_nonfinite_inference_is_not_persisted(engine):
    sid = schema(engine)
    encoder = Encoder()
    encoder.encode_many = lambda texts: np.full((len(texts), 4), np.nan)
    with pytest.raises(ValueError, match="invalid search embeddings"):
        EmbeddingStore(engine.db).vectors("schema", [(sid, "passage", "original")], encoder)
    assert not engine.db.connect().execute("SELECT * FROM schema_passage_embeddings").fetchall()


def test_procedure_vectors_reference_search_document_and_delete_with_session(engine):
    session = ops.activate(
        engine,
        query="Deploy safely",
        task="Deploy safely",
        initial_goal="Deploy safely",
        scope="project:test",
    )["session_id"]
    ops.commit(
        engine,
        session_id=session,
        outcome="success",
        final_goal="Deploy safely",
        procedure={
            "version": 2,
            "summary": "Test method",
            "steps": [{"summary": "Back up"}, {"summary": "Deploy"}],
        },
    )
    conn = engine.db.connect()
    document = conn.execute("SELECT * FROM procedure_search_documents").fetchone()
    entries = [
        (document["procedure_id"], role, document[column])
        for role, column in [("applicability", "applicability_text"), ("strategy", "strategy_text")]
    ]
    EmbeddingStore(engine.db).vectors("procedure", entries, Encoder())
    with conn:
        conn.execute("DELETE FROM sessions WHERE id=?", (session,))
    assert conn.execute("SELECT COUNT(*) FROM procedure_embeddings").fetchone()[0] == 0


def test_dense_index_observes_other_connection_edits_and_deletions(engine):
    sid = schema(engine)
    assert (
        engine.schemas.search_embedding(
            np.array([1, 0, 0, 0], dtype=np.float32), scope_id="project:test"
        )[0][0]
        == sid
    )
    other = sqlite3.connect(engine.cfg.db_path)
    with other:
        other.execute("UPDATE schemas SET status='stale' WHERE id=?", (sid,))
    assert (
        engine.schemas.search_embedding(
            np.array([1, 0, 0, 0], dtype=np.float32), scope_id="project:test"
        )
        == []
    )
    other.close()


def test_migration_is_repeatable_and_does_not_replay_canonical_events(engine):
    sid = schema(engine)
    store = EmbeddingStore(engine.db)
    store.vectors("schema", [(sid, "passage", "original")], Encoder())
    engine.db.init_schema(SlowaveConfig.default_schema_path())
    assert (
        engine.db.connect()
        .execute("SELECT COUNT(*) FROM storage_migrations WHERE version='search-vectors-v1'")
        .fetchone()[0]
        == 1
    )
    assert (
        engine.db.connect().execute("SELECT COUNT(*) FROM schema_passage_embeddings").fetchone()[0]
        == 1
    )


def test_partial_projection_migration_recreates_missing_objects_without_touching_sources(engine):
    sid = schema(engine)
    conn = engine.db.connect()
    with conn:
        conn.execute("DROP TABLE schema_passage_embeddings")
        conn.execute("DROP TABLE storage_migrations")
    engine.db.init_schema(SlowaveConfig.default_schema_path())
    assert (
        conn.execute("SELECT content_text FROM schemas WHERE id=?", (sid,)).fetchone()[0]
        == "original"
    )
    EmbeddingStore(engine.db).vectors("schema", [(sid, "passage", "original")], Encoder())
    assert not conn.execute("PRAGMA foreign_key_check").fetchall()


def test_failed_schema_transaction_rolls_back_and_releases_writer(engine, tmp_path):
    sid = schema(engine)
    script = tmp_path / "failed.sql"
    script.write_text(
        "BEGIN IMMEDIATE; CREATE TABLE failed_projection(id INTEGER); INSERT INTO nonexistent_table VALUES(1); COMMIT;"
    )
    with pytest.raises(sqlite3.OperationalError):
        engine.db.init_schema(str(script))
    conn = engine.db.connect()
    assert not conn.in_transaction
    assert (
        conn.execute("SELECT name FROM sqlite_master WHERE name='failed_projection'").fetchone()
        is None
    )
    assert (
        conn.execute("SELECT content_text FROM schemas WHERE id=?", (sid,)).fetchone()[0]
        == "original"
    )
    other = sqlite3.connect(engine.cfg.db_path, timeout=0.1)
    with other:
        other.execute("UPDATE schemas SET content_text='after recovery' WHERE id=?", (sid,))
    other.close()


def test_conflicting_source_snapshots_keep_their_own_vectors(engine):
    sid = schema(engine)
    result = EmbeddingStore(engine.db).vectors(
        "schema", [(sid, "passage", "original"), (sid, "passage", "other")], Encoder()
    )
    assert result[:, 0].tolist() == [8, 5]
    assert engine.db.connect().execute(
        "SELECT source_hash FROM schema_passage_embeddings"
    ).fetchone()[0] == text_hash("original")


def test_unchanged_procedure_backfill_does_not_rewrite_search_projection(engine):
    from slowave.symbolic.procedure_search import backfill_procedure_search

    session = engine.session_start(agent="test", scope="project:test", initial_goal="Deploy safely")
    ops.commit(
        engine,
        session_id=session,
        outcome="success",
        procedure={
            "version": 2,
            "summary": "Deploy",
            "steps": [{"summary": "Back up"}, {"summary": "Restart"}],
        },
    )
    conn = engine.db.connect()
    backfill_procedure_search(conn)
    changes = conn.total_changes
    backfill_procedure_search(conn)
    assert conn.total_changes == changes


def test_preparation_lease_recovers_and_reuses_completed_vectors(engine, monkeypatch):
    import slowave.symbolic.applicability_encoder as scorer
    import slowave.symbolic.retrieval_encoder as retrieval
    from slowave.core.services.search_preparation import prepare_search

    schema(engine)
    encoder = Encoder()
    engine.encoder = encoder
    monkeypatch.setattr(retrieval, "get_retrieval_encoder", lambda: encoder)
    monkeypatch.setattr(scorer, "_backend", lambda: None)
    conn = engine.db.connect()
    with conn:
        conn.execute(
            "INSERT INTO search_backfill_state(job,owner,lease_until) VALUES('search-vectors-v1','other',9999999999)"
        )
    assert prepare_search(engine) == {"ready": False, "reason": "preparation_in_progress"}
    assert encoder.calls == []
    with conn:
        conn.execute("UPDATE search_backfill_state SET lease_until=0")
    assert prepare_search(engine)["ready"]
    assert encoder.calls == ["original"]
    assert prepare_search(engine)["ready"]
    assert encoder.calls == ["original"]
    assert (
        conn.execute("SELECT completed_at,error FROM search_backfill_state").fetchone()["error"]
        is None
    )


def test_preparation_failure_releases_lease_and_retries(engine, monkeypatch):
    import slowave.symbolic.applicability_encoder as scorer
    import slowave.symbolic.retrieval_encoder as retrieval
    from slowave.core.services.search_preparation import prepare_search

    schema(engine)
    encoder = Encoder()
    engine.encoder = encoder
    monkeypatch.setattr(retrieval, "get_retrieval_encoder", lambda: encoder)
    monkeypatch.setattr(scorer, "_backend", lambda: None)
    encoder.hook = lambda: (_ for _ in ()).throw(RuntimeError("model failed"))
    with pytest.raises(RuntimeError, match="model failed"):
        prepare_search(engine)
    row = (
        engine.db.connect()
        .execute("SELECT lease_until,error FROM search_backfill_state")
        .fetchone()
    )
    assert row["lease_until"] == 0
    assert row["error"] == "model failed"
    encoder.hook = None
    assert prepare_search(engine)["ready"]


def test_exact_text_cache_is_bounded_and_returned_arrays_are_independent():
    from slowave.symbolic.encoder import TextEncoder

    encoder = TextEncoder()
    encoder._backend = Encoder()
    encoder._dim = 4
    texts = [str(i) for i in range(4097)]
    output = encoder.encode_many(texts)
    assert len(encoder._vectors) == 4096
    output[-1] = 0
    assert encoder.encode(texts[-1])[1] == 1
    assert len(encoder._backend.calls) == 4097


def test_passage_cache_keeps_recent_entries_above_capacity():
    from collections import OrderedDict
    from threading import RLock

    from slowave.symbolic.retrieval_encoder import RetrievalEncoder

    encoder = RetrievalEncoder.__new__(RetrievalEncoder)
    encoder._passages = OrderedDict()
    encoder._lock = RLock()

    class Backend:
        calls = 0

        def encode_many(self, texts):
            self.calls += len(texts)
            return np.ones((len(texts), 384), dtype=np.float32)

    encoder.backend = Backend()
    encoder.passages([str(i) for i in range(4097)])
    assert len(encoder._passages) == 4096
    encoder.passages(["4096"])
    assert encoder.backend.calls == 4097


def test_engine_preparation_waits_for_a_recoverable_conflict(engine, monkeypatch):
    import time

    import slowave.core.services.search_preparation as preparation

    outcomes = iter([{"ready": False, "reason": "preparation_in_progress"}, {"ready": True}])
    monkeypatch.setattr(preparation, "prepare_search", lambda *args: next(outcomes))
    clock = [0.0]
    monkeypatch.setattr(time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds))
    assert engine.prepare_retrieval(wait=True, timeout=2)["ready"]
    assert clock[0] == 0.25
    assert engine.retrieval_ready
    assert engine.retrieval_preparation_error is None


def test_vector_publication_locks_before_rechecking_source(engine):
    sid = schema(engine)
    statements = []
    conn = engine.db.connect()
    conn.set_trace_callback(statements.append)
    try:
        EmbeddingStore(engine.db).vectors("schema", [(sid, "passage", "original")], Encoder())
    finally:
        conn.set_trace_callback(None)
    lock = statements.index("BEGIN IMMEDIATE")
    check = next(i for i, sql in enumerate(statements) if sql.startswith("SELECT content_text"))
    publication = next(
        i
        for i, sql in enumerate(statements)
        if sql.startswith("INSERT INTO schema_passage_embeddings")
    )
    assert lock < check < publication
    assert "COMMIT" in statements[publication:]
