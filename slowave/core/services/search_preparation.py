"""Resumable search-vector backfill shared by setup and daemon warm-up."""

from __future__ import annotations

import secrets
import time
from typing import Any, Callable

from slowave.storage.embedding_store import EmbeddingStore


def prepare_search(engine: Any, progress: Callable[[str], None] | None = None) -> dict:
    from slowave.symbolic.applicability_encoder import _backend
    from slowave.symbolic.retrieval_encoder import get_retrieval_encoder

    if engine.encoder is None:
        return {"ready": False, "reason": "encoder_disabled"}
    report = progress or (lambda message: None)
    report("Loading search models")
    retrieval = get_retrieval_encoder()
    retrieval.embedding_spec()
    engine.encoder.embedding_spec()
    _backend()
    conn = engine.db.connect()
    owner = secrets.token_hex(16)
    now = int(time.time())
    with conn:
        claimed = conn.execute(
            "INSERT INTO search_backfill_state(job,owner,lease_until) VALUES('search-vectors-v1',?,?) "
            "ON CONFLICT(job) DO UPDATE SET owner=excluded.owner,lease_until=excluded.lease_until,completed_at=NULL,error=NULL "
            "WHERE search_backfill_state.lease_until<?",
            (owner, now + 120, now),
        ).rowcount
    if not claimed:
        return {"ready": False, "reason": "preparation_in_progress"}

    def heartbeat() -> None:
        with conn:
            changed = conn.execute(
                "UPDATE search_backfill_state SET lease_until=? WHERE job='search-vectors-v1' AND owner=?",
                (int(time.time()) + 120, owner),
            ).rowcount
        if not changed:
            raise RuntimeError("search preparation lease was lost")

    store = EmbeddingStore(engine.db)
    try:
        schemas = conn.execute("SELECT id,content_text FROM schemas ORDER BY id").fetchall()
        report(f"Preparing passage embeddings for {len(schemas)} memories")
        store.vectors(
            "schema",
            [(r["id"], "passage", r["content_text"] or "") for r in schemas],
            retrieval,
            heartbeat=heartbeat,
            require_current=True,
        )
        procedures = conn.execute(
            "SELECT procedure_id,applicability_text,strategy_text FROM procedure_search_documents ORDER BY procedure_id"
        ).fetchall()
        report(f"Preparing embeddings for {len(procedures)} procedures")
        store.vectors(
            "procedure",
            [
                (r["procedure_id"], role, r[column])
                for r in procedures
                for role, column in [
                    ("applicability", "applicability_text"),
                    ("strategy", "strategy_text"),
                ]
            ],
            engine.encoder,
            heartbeat=heartbeat,
            require_current=True,
        )
        with conn:
            conn.execute(
                "UPDATE search_backfill_state SET lease_until=0,completed_at=?,error=NULL WHERE job='search-vectors-v1' AND owner=?",
                (int(time.time()), owner),
            )
        return {"ready": True, "memories": len(schemas), "procedures": len(procedures)}
    except Exception as exc:
        with conn:
            conn.execute(
                "UPDATE search_backfill_state SET lease_until=0,error=? WHERE job='search-vectors-v1' AND owner=?",
                (str(exc)[:1000], owner),
            )
        raise
