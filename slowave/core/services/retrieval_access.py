"""Persistence for cue-conditioned retrieval-access evidence."""

from __future__ import annotations

import sqlite3
import time
from dataclasses import asdict, dataclass

import numpy as np

from slowave.storage.sqlite_db import SQLiteDB
from slowave.utils.vec import pack_f32, unpack_f32


@dataclass(frozen=True)
class RetrievalAccessPolicyConfig:
    """Declared Phase-2 shadow-policy parameters; none affects live admission."""

    cue_match_threshold: float = 0.90
    negative_evidence_threshold: int = 2
    inhibition_per_net_negative: float = 0.35
    inhibition_cap: float = 0.70
    direct_override_threshold: float = 0.95


class RetrievalAccessPolicy:
    """Read-only, cue- and pathway-specific hypothetical access evaluator."""

    def __init__(self, db: SQLiteDB, config: RetrievalAccessPolicyConfig | None = None):
        self.db = db
        self.config = config or RetrievalAccessPolicyConfig()

    def evaluate(
        self,
        *,
        schema_id: int,
        raw_semantic_relevance: float,
        pathway: str,
        cue_embedding: np.ndarray | None,
        scope_id: str | None,
        task_type: str | None,
    ) -> dict[str, object]:
        """Return a trace only; policy callers must never mutate admission from it."""
        trace: dict[str, object] = {
            "policy_mode": "shadow",
            "parameters": asdict(self.config),
            "schema_id": schema_id,
            "pathway": pathway,
            "raw_semantic_relevance": round(raw_semantic_relevance, 6),
            "cue_prototype_id": None,
            "cue_prototype_similarity": None,
            "useful_count": 0,
            "irrelevant_count": 0,
            "inhibition_strength": 0.0,
            "access_state": "eligible",
            "conscious_score": round(raw_semantic_relevance, 6),
            "hypothetical_admitted": True,
            "reason": "no_matching_access_evidence",
        }
        if pathway not in RetrievalAccessEvidenceStore._PATHWAYS or cue_embedding is None:
            trace["reason"] = "unsupported_pathway_or_missing_cue"
            return trace

        cue = np.asarray(cue_embedding, dtype=np.float32)
        cue_norm = float(np.linalg.norm(cue))
        if cue.ndim != 1 or cue.size == 0 or cue_norm < 1e-12:
            trace["reason"] = "invalid_cue_embedding"
            return trace
        rows = (
            self.db.connect()
            .execute(
                """
            SELECT e.cue_prototype_id, e.useful_count, e.irrelevant_count,
                   c.embedding, c.dim
            FROM schema_retrieval_evidence e
            JOIN retrieval_cue_prototypes c ON c.id = e.cue_prototype_id
            WHERE e.schema_id = ? AND e.pathway = ?
              AND c.dim = ? AND c.scope_id IS ? AND c.task_type IS ?
            ORDER BY e.cue_prototype_id
            """,
                (schema_id, pathway, int(cue.size), scope_id, task_type),
            )
            .fetchall()
        )
        best: tuple[float, sqlite3.Row] | None = None
        for row in rows:
            try:
                prototype = unpack_f32(row["embedding"], int(row["dim"]))
            except (TypeError, ValueError):
                continue
            similarity = float(
                cue.dot(prototype) / (cue_norm * float(np.linalg.norm(prototype)) + 1e-12)
            )
            if best is None or similarity > best[0]:
                best = (similarity, row)
        if best is None or best[0] < self.config.cue_match_threshold:
            trace["reason"] = "no_matching_access_evidence"
            return trace

        similarity, row = best
        useful = int(row["useful_count"])
        irrelevant = int(row["irrelevant_count"])
        net_negative = max(0, irrelevant - useful)
        inhibition = min(
            self.config.inhibition_cap,
            max(0, net_negative - self.config.negative_evidence_threshold + 1)
            * self.config.inhibition_per_net_negative,
        )
        override = (
            pathway == "direct" and raw_semantic_relevance >= self.config.direct_override_threshold
        )
        inhibited = inhibition > 0.0 and not override
        trace.update(
            cue_prototype_id=int(row["cue_prototype_id"]),
            cue_prototype_similarity=round(similarity, 6),
            useful_count=useful,
            irrelevant_count=irrelevant,
            inhibition_strength=round(inhibition, 6),
            access_state="inhibited" if inhibition > 0.0 else "eligible",
            conscious_score=round(raw_semantic_relevance - inhibition, 6),
            hypothetical_admitted=not inhibited,
            reason=(
                "inhibition_override" if override else "cue_inhibited" if inhibited else "eligible"
            ),
        )
        return trace


class RetrievalAccessEvidenceStore:
    """Write and inspect evidence without deciding retrieval admission."""

    _PATHWAYS = frozenset({"direct", "graph", "exploration"})

    def __init__(self, db: SQLiteDB):
        self.db = db

    def record_feedback(
        self,
        conn: sqlite3.Connection,
        *,
        retrieval_id: str,
        useful_ids: list[int],
        irrelevant_ids: list[int],
    ) -> dict[str, list[str]]:
        """Persist authorized access evidence from a recorded retrieval snapshot.

        Missing cue provenance, unadmitted items, and unknown pathways produce no
        derived evidence. The caller owns the surrounding feedback transaction.
        """
        labels_by_schema: dict[int, str] = {}
        for schema_id in useful_ids:
            labels_by_schema[schema_id] = "useful"
        for schema_id in irrelevant_ids:
            labels_by_schema[schema_id] = "irrelevant"
        applied: dict[str, list[str]] = {"useful": [], "irrelevant": [], "skipped": []}
        if not labels_by_schema:
            return applied

        snapshot = conn.execute(
            """
            SELECT cue_embedding, cue_dim, scope_id, scope_kind, task_type
            FROM context_recall_events WHERE context_id = ?
            """,
            (retrieval_id,),
        ).fetchone()
        if snapshot is None or snapshot["cue_embedding"] is None or snapshot["cue_dim"] is None:
            applied["skipped"] = [f"sch_{schema_id}" for schema_id in labels_by_schema]
            return applied

        cue_id = self._cue_prototype_id(conn, snapshot)
        rows = conn.execute(
            """
            SELECT memory_id, pathway FROM context_recall_items
            WHERE context_id = ? AND admitted = 1 AND memory_type IN ('schema', 'related')
            """,
            (retrieval_id,),
        ).fetchall()
        pathways = {
            int(row["memory_id"][4:]): str(row["pathway"])
            for row in rows
            if isinstance(row["memory_id"], str)
            and row["memory_id"].startswith("sch_")
            and row["memory_id"][4:].isdigit()
            and row["pathway"] in self._PATHWAYS
        }
        now = int(time.time())
        for schema_id, label in labels_by_schema.items():
            pathway = pathways.get(schema_id)
            if pathway is None:
                applied["skipped"].append(f"sch_{schema_id}")
                continue
            if label == "useful":
                conn.execute(
                    """
                    INSERT INTO schema_retrieval_evidence
                      (schema_id, cue_prototype_id, pathway, useful_count, last_useful_ts, updated_at)
                    VALUES (?, ?, ?, 1, ?, ?)
                    ON CONFLICT(schema_id, cue_prototype_id, pathway) DO UPDATE SET
                      useful_count = useful_count + 1,
                      last_useful_ts = excluded.last_useful_ts,
                      updated_at = excluded.updated_at
                    """,
                    (schema_id, cue_id, pathway, now, now),
                )
            else:
                conn.execute(
                    """
                    INSERT INTO schema_retrieval_evidence
                      (schema_id, cue_prototype_id, pathway, irrelevant_count, last_irrelevant_ts, updated_at)
                    VALUES (?, ?, ?, 1, ?, ?)
                    ON CONFLICT(schema_id, cue_prototype_id, pathway) DO UPDATE SET
                      irrelevant_count = irrelevant_count + 1,
                      last_irrelevant_ts = excluded.last_irrelevant_ts,
                      updated_at = excluded.updated_at
                    """,
                    (schema_id, cue_id, pathway, now, now),
                )
            applied[label].append(f"sch_{schema_id}")
        return applied

    def inspect_schema(self, schema_id: int) -> list[dict[str, object]]:
        """Return read-only access evidence for one semantic schema."""
        rows = (
            self.db.connect()
            .execute(
                """
            SELECT e.cue_prototype_id, e.pathway, e.useful_count, e.irrelevant_count,
                   e.last_useful_ts, e.last_irrelevant_ts, e.inhibition_strength,
                   e.access_state, e.updated_at, c.scope_id, c.task_type
            FROM schema_retrieval_evidence e
            JOIN retrieval_cue_prototypes c ON c.id = e.cue_prototype_id
            WHERE e.schema_id = ?
            ORDER BY e.updated_at DESC, e.cue_prototype_id, e.pathway
            """,
                (schema_id,),
            )
            .fetchall()
        )
        return [dict(row) for row in rows]

    def shadow_policy(self) -> RetrievalAccessPolicy:
        """Build the read-only Phase-2 policy over this evidence store's database."""
        return RetrievalAccessPolicy(self.db)

    @staticmethod
    def _cue_prototype_id(conn: sqlite3.Connection, snapshot: sqlite3.Row) -> int:
        row = conn.execute(
            """
            SELECT id FROM retrieval_cue_prototypes
            WHERE embedding = ? AND dim = ? AND scope_id IS ? AND task_type IS ?
            ORDER BY id LIMIT 1
            """,
            (
                snapshot["cue_embedding"],
                snapshot["cue_dim"],
                snapshot["scope_id"],
                snapshot["task_type"],
            ),
        ).fetchone()
        now = int(time.time())
        if row is not None:
            cue_id = int(row["id"])
            conn.execute(
                "UPDATE retrieval_cue_prototypes "
                "SET support_count = support_count + 1, last_seen_ts = ? WHERE id = ?",
                (now, cue_id),
            )
            return cue_id
        cur = conn.execute(
            """
            INSERT INTO retrieval_cue_prototypes
              (embedding, dim, scope_id, scope_kind, task_type, first_seen_ts, last_seen_ts)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                snapshot["cue_embedding"],
                snapshot["cue_dim"],
                snapshot["scope_id"],
                snapshot["scope_kind"],
                snapshot["task_type"],
                now,
                now,
            ),
        )
        if cur.lastrowid is None:
            raise RuntimeError("cue prototype insert did not return an ID")
        return int(cur.lastrowid)


def canonical_cue_text(
    *,
    query: str | None,
    goal: str | None,
    task_type: str | None,
    situation: dict[str, object] | None,
    requirements: list[str] | None,
    topics: list[str] | None,
    entities: list[str] | None,
) -> str:
    """Compose the semantic retrieval cue; scope remains metadata."""
    return " ".join(
        part
        for part in (
            query or "",
            goal or "",
            task_type or "",
            " ".join(f"{key} {value}" for key, value in sorted((situation or {}).items())),
            " ".join(requirements or []),
            " ".join(topics or []),
            " ".join(entities or []),
        )
        if part
    )


def packed_cue_embedding(encoder, cue_text: str) -> tuple[bytes, int] | None:
    """Encode a non-empty canonical cue once for snapshot persistence."""
    if encoder is None or not cue_text:
        return None
    vector = np.asarray(encoder.encode(cue_text), dtype=np.float32)
    if vector.ndim != 1 or vector.size == 0:
        return None
    return pack_f32(vector), int(vector.size)
