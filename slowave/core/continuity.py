"""Persistent, server-issued MCP conversation continuity identifiers."""

from __future__ import annotations

import re
import secrets
import time
from dataclasses import dataclass

from slowave.storage.sqlite_db import SQLiteDB

_CONTINUITY_RE = re.compile(r"^cont_[A-Za-z0-9_-]{32,96}$")


class ContinuityError(ValueError):
    """A client supplied a continuity token that cannot be resumed."""


@dataclass(frozen=True)
class ContinuityResolution:
    continuity_id: str
    state: str  # started | continued


def resolve_continuity(
    db: SQLiteDB,
    *,
    scope_id: str,
    supplied_id: str | None,
    integration: str | None = None,
    client_identity: str | None = None,
) -> ContinuityResolution:
    """Create or validate a continuity before its new task session is inserted.

    Continuities deliberately have their own durable table.  Sessions are
    task-scoped and always new; a scope-keyed recent-session resolver must not
    become accidental conversation identity.
    """
    conn = db.connect()
    if supplied_id is None:
        # Retry only for the astronomically unlikely token collision.
        for _ in range(3):
            continuity_id = f"cont_{secrets.token_urlsafe(32)}"
            try:
                conn.execute(
                    "INSERT INTO continuities "
                    "(continuity_id, scope_id, integration, client_identity, created_at, last_seen_at) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        continuity_id,
                        scope_id,
                        integration,
                        client_identity,
                        int(time.time()),
                        int(time.time()),
                    ),
                )
                conn.commit()
                return ContinuityResolution(continuity_id, "started")
            except Exception as exc:
                if "unique" not in str(exc).lower():
                    raise
        raise RuntimeError("could not allocate continuity_id")

    if not isinstance(supplied_id, str) or not supplied_id.strip():
        raise ContinuityError("continuity_id must be nonblank when supplied")
    continuity_id = supplied_id.strip()
    if not _CONTINUITY_RE.fullmatch(continuity_id):
        raise ContinuityError("continuity_id is malformed")
    row = conn.execute(
        "SELECT scope_id, client_identity FROM continuities WHERE continuity_id = ?",
        (continuity_id,),
    ).fetchone()
    if row is None:
        raise ContinuityError("continuity_id is unknown")
    if row["scope_id"] != scope_id:
        raise ContinuityError("continuity_id and scope do not match")
    # Only bind when a durable client identity is explicitly available.  Most
    # MCP transports expose a per-connection id, which is intentionally not a
    # continuity binding because it would break a restarted client.
    if row["client_identity"] and client_identity and row["client_identity"] != client_identity:
        raise ContinuityError("continuity_id belongs to a different client")
    conn.execute(
        "UPDATE continuities SET last_seen_at = ? WHERE continuity_id = ?",
        (int(time.time()), continuity_id),
    )
    conn.commit()
    return ContinuityResolution(continuity_id, "continued")
