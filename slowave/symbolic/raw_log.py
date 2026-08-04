"""Raw event log: the canonical source of truth.

Every observation passes through here first. Episodes and schemas cite back
to raw_events.id for provenance.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from slowave.storage.sqlite_db import SQLiteDB
from slowave.utils.vec import dumps_json, loads_json, pack_f32, to_f32, unpack_f32


@dataclass(frozen=True)
class RawEvent:
    id: int
    session_id: str
    ts: int
    type: str
    content: str
    metadata: dict[str, Any]
    embedding: np.ndarray | None
    dim: int | None
    logic_version: str = "0"


class RawLog:
    """Append-only event log. No deletions."""

    def __init__(self, db: SQLiteDB):
        self.db = db

    def append(
        self,
        *,
        session_id: str,
        type: str,
        content: str,
        ts: int | None = None,
        metadata: dict[str, Any] | None = None,
        embedding: np.ndarray | None = None,
        logic_version: str = "0",
    ) -> int:
        if ts is None:
            ts = int(time.time())
        meta = metadata or {}
        conn = self.db.connect()
        if embedding is not None:
            emb = to_f32(embedding).reshape(-1)
            dim = int(emb.size)
            cur = conn.execute(
                "INSERT INTO raw_events "
                "(session_id, ts, type, content, metadata_json, embedding, dim, logic_version) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    str(session_id),
                    int(ts),
                    str(type),
                    str(content),
                    dumps_json(meta),
                    pack_f32(emb),
                    dim,
                    str(logic_version),
                ),
            )
        else:
            cur = conn.execute(
                "INSERT INTO raw_events "
                "(session_id, ts, type, content, metadata_json, embedding, dim, logic_version) "
                "VALUES (?, ?, ?, ?, ?, NULL, NULL, ?)",
                (
                    str(session_id),
                    int(ts),
                    str(type),
                    str(content),
                    dumps_json(meta),
                    str(logic_version),
                ),
            )
        event_id = int(cur.lastrowid or 0)
        conn.execute(
            "INSERT INTO raw_events_fts (rowid, content) VALUES (?, ?)",
            (event_id, content),
        )
        conn.commit()
        return event_id

    def get(self, event_id: int) -> RawEvent:
        conn = self.db.connect()
        row = conn.execute("SELECT * FROM raw_events WHERE id = ?", (int(event_id),)).fetchone()
        if row is None:
            raise KeyError(f"No raw event id={event_id}")
        return self._row_to_event(row)

    def get_many(self, ids: Iterable[int]) -> list[RawEvent]:
        ids = list(dict.fromkeys(int(i) for i in ids))
        if not ids:
            return []
        ph = ",".join(["?"] * len(ids))
        conn = self.db.connect()
        rows = conn.execute(f"SELECT * FROM raw_events WHERE id IN ({ph})", tuple(ids)).fetchall()
        by_id = {int(r["id"]): r for r in rows}
        out: list[RawEvent] = []
        for i in ids:
            r = by_id.get(i)
            if r is None:
                continue
            out.append(self._row_to_event(r))
        return out

    def list_session(self, session_id: str) -> list[RawEvent]:
        conn = self.db.connect()
        rows = conn.execute(
            "SELECT * FROM raw_events WHERE session_id = ? ORDER BY ts, id",
            (str(session_id),),
        ).fetchall()
        return [self._row_to_event(r) for r in rows]

    def search_fts(self, query: str, limit: int = 20) -> list[int]:
        conn = self.db.connect()
        rows = conn.execute(
            "SELECT rowid FROM raw_events_fts WHERE raw_events_fts MATCH ? "
            "ORDER BY rank LIMIT ?",
            (query, int(limit)),
        ).fetchall()
        return [int(r["rowid"]) for r in rows]

    def _row_to_event(self, row: Any) -> RawEvent:
        emb = None
        dim = None
        if row["embedding"] is not None and row["dim"] is not None:
            dim = int(row["dim"])
            emb = unpack_f32(row["embedding"], dim)
        return RawEvent(
            id=int(row["id"]),
            session_id=str(row["session_id"]),
            ts=int(row["ts"]),
            type=str(row["type"]),
            content=str(row["content"]),
            metadata=loads_json(row["metadata_json"]),
            embedding=emb,
            dim=dim,
            logic_version=str(row["logic_version"]),
        )

    # session lifecycle helpers
    def session_exists(self, session_id: str) -> bool:
        """Return True if *session_id* is registered in the sessions table."""
        conn = self.db.connect()
        row = conn.execute(
            "SELECT 1 FROM sessions WHERE id = ? LIMIT 1", (str(session_id),)
        ).fetchone()
        return row is not None

    def is_session_ended(self, session_id: str) -> bool:
        """Return True if this session already has an ended_ts recorded.

        Used by SlowaveEngine.session_end() to guard against re-forming
        episodes for a session that was already closed (e.g. the idle-session
        reaper ends it, and the client later calls commit() on the same
        session_id) -- without this check, form_episodes() would reprocess
        every raw event from scratch and insert duplicate episode rows.
        """
        conn = self.db.connect()
        row = conn.execute(
            "SELECT 1 FROM sessions WHERE id = ? AND ended_ts IS NOT NULL LIMIT 1",
            (str(session_id),),
        ).fetchone()
        return row is not None

    def start_session(
        self,
        *,
        session_id: str,
        agent: str,
        scope_id: str | None = None,
        scope_kind: str | None = None,
        ts: int | None = None,
        goal: str | None = None,
        lifecycle_version: str | None = None,
    ) -> None:
        conn = self.db.connect()
        conn.execute(
            "INSERT INTO sessions (id, agent, scope_id, scope_kind, started_ts, goal, lifecycle_version) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                str(session_id),
                str(agent),
                scope_id,
                scope_kind,
                int(ts) if ts is not None else int(time.time()),
                goal,
                lifecycle_version,
            ),
        )
        conn.commit()

    def end_session(
        self, session_id: str, ts: int | None = None, outcome: str | None = None
    ) -> None:
        conn = self.db.connect()
        conn.execute(
            "UPDATE sessions SET ended_ts = ?, outcome = ? WHERE id = ?",
            (int(ts) if ts is not None else int(time.time()), outcome, str(session_id)),
        )
        conn.commit()

    def list_session_ids(
        self, *, scope_id: str | None = None, since: int | None = None
    ) -> list[str]:
        conn = self.db.connect()
        sql = "SELECT id FROM sessions WHERE 1=1"
        args: list[Any] = []
        if scope_id is not None:
            sql += " AND scope_id = ?"
            args.append(scope_id)
        if since is not None:
            sql += " AND started_ts >= ?"
            args.append(int(since))
        sql += " ORDER BY started_ts DESC"
        rows = conn.execute(sql, tuple(args)).fetchall()
        return [str(r["id"]) for r in rows]
