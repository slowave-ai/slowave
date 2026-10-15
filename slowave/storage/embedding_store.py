"""Rebuildable, versioned search vectors. Never owns canonical memory text."""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any

import numpy as np

from slowave.storage.sqlite_db import SQLiteDB


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class SearchSourceChanged(RuntimeError):
    """A backfill source changed during inference; completed batches are safe."""


class EmbeddingStore:
    """Compute outside write locks; publish only against unchanged source text.

    Every batch commits independently. Missing/hash-mismatched rows are the
    resume checkpoint, so a killed process cannot incorrectly mark work done.
    """

    def __init__(self, db: SQLiteDB):
        self.db = db

    def vectors(
        self,
        kind: str,
        entries: list[tuple[Any, str, str]],
        encoder: Any,
        *,
        heartbeat: Any = None,
        require_current: bool = False,
    ) -> np.ndarray:
        if kind not in {"procedure", "schema"}:
            raise ValueError("unknown embedding owner")
        valid_roles = {"applicability", "strategy"} if kind == "procedure" else {"passage"}
        if any(role not in valid_roles or not isinstance(text, str) for _, role, text in entries):
            raise ValueError("invalid search embedding entry")
        spec = encoder.embedding_spec()
        encoded_spec = json.dumps(spec, sort_keys=True, separators=(",", ":"))
        encoder_id = text_hash(encoded_spec)
        dimension = int(encoder.dim)
        conn = self.db.connect()
        if conn.in_transaction:
            raise RuntimeError("search vector preparation requires no open write transaction")
        if not conn.execute(
            "SELECT 1 FROM embedding_encoders WHERE encoder_id=?", (encoder_id,)
        ).fetchone():
            with conn:
                conn.execute(
                    "INSERT INTO embedding_encoders(encoder_id,spec_json,dim,created_at) "
                    "VALUES(?,?,?,?) ON CONFLICT(encoder_id) DO NOTHING",
                    (encoder_id, encoded_spec, dimension, int(time.time())),
                )
        table = "procedure_embeddings" if kind == "procedure" else "schema_passage_embeddings"
        owner = "procedure_id" if kind == "procedure" else "schema_id"
        stored = {}
        identities = list(dict.fromkeys(identity for identity, _, _ in entries))
        for offset in range(0, len(identities), 500):
            chunk = identities[offset : offset + 500]
            placeholders = ",".join("?" for _ in chunk)
            for row in conn.execute(
                f"SELECT {owner},role,source_hash,vector,dim FROM {table} WHERE encoder_id=? AND {owner} IN ({placeholders})",
                (encoder_id, *chunk),
            ):
                stored[(row[owner], row["role"])] = row
        result: dict[tuple[Any, str, str], np.ndarray] = {}
        missing = []
        source_changed = False
        for identity, role, text in entries:
            row = stored.get((identity, role))
            if row is not None and row["source_hash"] == text_hash(text):
                vector = np.frombuffer(row["vector"], dtype="<f4").copy()
                if vector.size == dimension and np.isfinite(vector).all():
                    result[(identity, role, text_hash(text))] = vector
                    continue
            missing.append((identity, role, text))
        for offset in range(0, len(missing), 16):
            if heartbeat is not None:
                heartbeat()
            batch = missing[offset : offset + 16]
            vectors = np.asarray(encoder.encode_many([text for _, _, text in batch]), dtype="<f4")
            if vectors.shape != (len(batch), dimension) or not np.isfinite(vectors).all():
                raise ValueError("invalid search embeddings")
            # No downloads or inference inside this short transaction.
            with conn:
                conn.execute("BEGIN IMMEDIATE")
                for (identity, role, text), vector in zip(batch, vectors):
                    if kind == "procedure":
                        column = {
                            "applicability": "applicability_text",
                            "strategy": "strategy_text",
                        }[role]
                        source = conn.execute(
                            f"SELECT {column} FROM procedure_search_documents WHERE procedure_id=?",
                            (identity,),
                        ).fetchone()
                    else:
                        if role != "passage":
                            raise ValueError("invalid schema embedding role")
                        source = conn.execute(
                            "SELECT content_text FROM schemas WHERE id=?", (identity,)
                        ).fetchone()
                    # A concurrent edit/delete must never publish an obsolete vector.
                    if source is not None and (source[0] or "") == text:
                        conn.execute(
                            f"INSERT INTO {table}({owner},role,encoder_id,source_hash,dim,vector,created_at) "
                            "VALUES(?,?,?,?,?,?,?) ON CONFLICT DO UPDATE SET "
                            "source_hash=excluded.source_hash,dim=excluded.dim,vector=excluded.vector,created_at=excluded.created_at",
                            (
                                identity,
                                role,
                                encoder_id,
                                text_hash(text),
                                dimension,
                                vector.tobytes(),
                                int(time.time()),
                            ),
                        )
                    else:
                        source_changed = True
                    # This caller's source snapshot still has a matching vector.
                    result[(identity, role, text_hash(text))] = vector.copy()
        if source_changed and require_current:
            raise SearchSourceChanged("search sources changed during backfill; retry required")
        return np.asarray(
            [result[(identity, role, text_hash(text))] for identity, role, text in entries],
            dtype=np.float32,
        ).reshape(len(entries), dimension)
