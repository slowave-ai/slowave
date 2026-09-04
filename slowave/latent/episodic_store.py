from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Iterable

import faiss
import numpy as np

from slowave.latent.types import EpisodicMemory
from slowave.storage.sqlite_db import SQLiteDB
from slowave.utils.vec import dumps_json, loads_json, pack_f32, to_f32, unpack_f32


@dataclass(frozen=True)
class EpisodicStoreConfig:
    dim: int
    db_path: str = "slowwave.db"


class EpisodicStore:
    """Append-only episodic store.

    Storage:
      - relational fields in SQLite
      - vector similarity via an in-memory FAISS index (inner product on
        normalized vectors)

    SQLite is the durable source of truth for embeddings.  The FAISS index is
    rebuilt from it when the engine starts; it must never be persisted relative
    to the caller's current working directory.
    """

    def __init__(self, db: SQLiteDB, cfg: EpisodicStoreConfig):
        self.db = db
        self.cfg = cfg

        self._index = self._create_index(dim=cfg.dim)
        # FAISS doesn't store external IDs by default; we use an IDMap.
        # Keys are SQLite episode IDs.

    def _create_index(self, dim: int) -> faiss.Index:
        # Use cosine similarity as inner product on L2-normalized vectors.
        base = faiss.IndexFlatIP(dim)
        return faiss.IndexIDMap2(base)

    def reset_faiss_from_db(self) -> None:
        """Rebuild FAISS index by scanning SQLite."""
        conn = self.db.connect()
        cur = conn.execute("SELECT id, embedding, dim FROM episodic_memories ORDER BY id")
        ids: list[int] = []
        vecs: list[np.ndarray] = []
        for row in cur:
            ids.append(int(row["id"]))
            vecs.append(unpack_f32(row["embedding"], int(row["dim"])))
        if not ids:
            return
        X = to_f32(np.stack(vecs, axis=0))
        faiss.normalize_L2(X)
        self._index.reset()
        self._index.add_with_ids(X, np.asarray(ids, dtype=np.int64))

    def add(
        self,
        *,
        event_id: str,
        ts: int | None,
        embedding: np.ndarray,
        salience: float,
        metadata: dict[str, Any],
    ) -> int:
        """Insert episode into SQLite + FAISS.

        Returns SQLite row id.
        """
        if ts is None:
            ts = int(time.time())
        emb = to_f32(embedding).reshape(-1)
        if emb.size != self.cfg.dim:
            raise ValueError(f"dim mismatch: expected {self.cfg.dim}, got {emb.size}")

        conn = self.db.connect()
        cur = conn.execute(
            """
            INSERT INTO episodic_memories (event_id, ts, embedding, dim, salience, last_salience_ts, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event_id,
                ts,
                pack_f32(emb),
                self.cfg.dim,
                float(salience),
                int(ts),
                dumps_json(metadata),
            ),
        )
        episode_id = int(cur.lastrowid or 0)
        conn.commit()

        x = emb.reshape(1, -1).copy()
        faiss.normalize_L2(x)
        self._index.add_with_ids(x, np.asarray([episode_id], dtype=np.int64))
        return episode_id

    def get(self, episode_id: int) -> EpisodicMemory:
        conn = self.db.connect()
        row = conn.execute("SELECT * FROM episodic_memories WHERE id = ?", (episode_id,)).fetchone()
        if row is None:
            raise KeyError(f"No episode id={episode_id}")
        return EpisodicMemory(
            id=int(row["id"]),
            event_id=str(row["event_id"]),
            ts=int(row["ts"]),
            embedding=unpack_f32(row["embedding"], int(row["dim"])),
            salience=float(row["salience"]),
            metadata=loads_json(row["metadata_json"]),
            recalled_count=int(row["recalled_count"]),
        )

    def get_many(self, episode_ids: Iterable[int]) -> list[EpisodicMemory]:
        ids = list(dict.fromkeys(int(i) for i in episode_ids))
        if not ids:
            return []
        placeholders = ",".join(["?"] * len(ids))
        conn = self.db.connect()
        rows = conn.execute(
            f"SELECT * FROM episodic_memories WHERE id IN ({placeholders})", tuple(ids)
        ).fetchall()
        by_id = {int(r["id"]): r for r in rows}
        out: list[EpisodicMemory] = []
        for i in ids:
            r = by_id.get(i)
            if r is None:
                continue
            out.append(
                EpisodicMemory(
                    id=int(r["id"]),
                    event_id=str(r["event_id"]),
                    ts=int(r["ts"]),
                    embedding=unpack_f32(r["embedding"], int(r["dim"])),
                    salience=float(r["salience"]),
                    metadata=loads_json(r["metadata_json"]),
                    recalled_count=int(r["recalled_count"]),
                )
            )
        return out

    def count(self) -> int:
        conn = self.db.connect()
        row = conn.execute("SELECT COUNT(*) AS n FROM episodic_memories").fetchone()
        return int(row["n"])

    def update_salience(self, episode_id: int, new_salience: float) -> None:
        conn = self.db.connect()
        conn.execute(
            "UPDATE episodic_memories SET salience = ?, last_salience_ts = ? WHERE id = ?",
            (float(new_salience), int(time.time()), int(episode_id)),
        )
        conn.commit()

    def increment_recall(self, episode_ids: list[int], reinforcement: float) -> None:
        if not episode_ids:
            return
        conn = self.db.connect()
        # Reinforce salience + recall count.
        # SQLite has limited IN handling; we do executemany for simplicity.
        for eid in episode_ids:
            conn.execute(
                """
                UPDATE episodic_memories
                SET recalled_count = recalled_count + 1,
                    salience = salience + ?
                    , last_salience_ts = ?
                WHERE id = ?
                """,
                (float(reinforcement), int(time.time()), int(eid)),
            )
        conn.commit()

    def search(self, query: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (scores, episode_ids)."""
        q = to_f32(query).reshape(1, -1).copy()
        if q.shape[1] != self.cfg.dim:
            raise ValueError(f"dim mismatch: expected {self.cfg.dim}, got {q.shape[1]}")
        faiss.normalize_L2(q)
        scores, ids = self._index.search(q, top_k)
        return scores.reshape(-1), ids.reshape(-1)

    def list_saliences(self) -> list[tuple[int, float]]:
        conn = self.db.connect()
        rows = conn.execute("SELECT id, salience FROM episodic_memories").fetchall()
        return [(int(r["id"]), float(r["salience"])) for r in rows]

    def load_embeddings(self, episode_ids: list[int]) -> np.ndarray:
        mems = self.get_many(episode_ids)
        if not mems:
            return np.zeros((0, self.cfg.dim), dtype=np.float32)
        X = np.stack([m.embedding for m in mems], axis=0).astype(np.float32)
        return X
