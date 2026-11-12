"""Project-local suppression of user-global rules (RT-H4 mitigation).

User-global memories (scope ``global:user``) reach every project.  Letting any one project
retire such a rule with ``stale`` feedback would let a single (possibly poisoned or simply
mistaken) session change behaviour everywhere.  The server cannot tell human-originated from
agent-originated writes within the five tools, so instead:

* ``stale`` feedback on a user-global rule, sent from a *project* retrieval, records a row in
  ``standing_suppressions(scope_id, memory_id)`` and leaves the memory ``active``.  The standing
  tier of THAT project no longer delivers it; other projects are unaffected.
* ``used`` feedback on the same rule in the same project lifts the suppression (reversible).
* Feedback sent from a retrieval in the global scope itself retires the rule as before.

Residual risk (documented, not closed): any project session can still PLANT a user-global rule
with ``remember(visibility="user")``.  Mitigations are structural - the originating project is
recorded in provenance and shown on the tier line, user-global lines are capped per activation,
and the injection filter still applies - but a poisoned write that looks like ordinary guidance
reaches every project until a human audits and retires it from a global-scope session.

Suppression applies to the standing tier only; the memory can still surface in topical
retrieval in that project.  The table is derived/auxiliary data and has a reversible migration.
"""

from __future__ import annotations

import json
import sqlite3
import time
from typing import Any

from slowave.core.scope import is_global_scope

SUPPRESS_DDL = (
    """CREATE TABLE IF NOT EXISTS standing_suppressions (
      scope_id   TEXT NOT NULL,
      memory_id  INTEGER NOT NULL,
      reason     TEXT NOT NULL DEFAULT '',
      created_ts INTEGER NOT NULL,
      PRIMARY KEY (scope_id, memory_id)
    )""",
)
SUPPRESS_DOWN = ("DROP TABLE IF EXISTS standing_suppressions",)

# Exclusion fragment for queries over ``schemas s`` (one ``?`` placeholder: the project scope).
NOT_SUPPRESSED_SQL = (
    "NOT EXISTS (SELECT 1 FROM standing_suppressions x "
    "WHERE x.scope_id = ? AND x.memory_id = s.id)"
)


def retrieval_scope(conn: sqlite3.Connection, retrieval_id: str) -> str | None:
    row = conn.execute(
        "SELECT scope_id FROM context_recall_events WHERE context_id = ?", (str(retrieval_id),)
    ).fetchone()
    return None if row is None else row["scope_id"]


def is_cross_scope_global(rule_scope: str | None, retrieval_scope_id: str | None) -> bool:
    """True when a user-global rule is being assessed from a non-global (project) retrieval."""
    return (
        (rule_scope in (None, "") or is_global_scope(rule_scope))
        and bool(retrieval_scope_id)
        and not is_global_scope(retrieval_scope_id)
    )


def suppress(conn: sqlite3.Connection, scope_id: str, memory_id: int, reason: str) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO standing_suppressions (scope_id, memory_id, reason, created_ts) "
        "VALUES (?, ?, ?, ?)",
        (scope_id, int(memory_id), str(reason)[:200], int(time.time())),
    )


def lift(conn: sqlite3.Connection, scope_id: str, memory_id: int) -> None:
    try:
        conn.execute(
            "DELETE FROM standing_suppressions WHERE scope_id = ? AND memory_id = ?",
            (scope_id, int(memory_id)),
        )
    except sqlite3.OperationalError:
        pass


def origin_scopes(conn: sqlite3.Connection, memory_ids: list[int]) -> dict[int, str]:
    """Originating project of user-global rules written from a project session (provenance)."""
    out: dict[int, str] = {}
    if not memory_ids:
        return out
    marks = ",".join("?" for _ in memory_ids)
    for row in conn.execute(
        f"SELECT id, facets_json FROM schemas WHERE id IN ({marks})", list(memory_ids)
    ):
        try:
            facets: dict[str, Any] = json.loads(row["facets_json"] or "{}")
        except Exception:
            continue
        origin = (facets.get("source_provenance") or {}).get("origin_scope")
        if origin:
            out[int(row["id"])] = str(origin)[:60]
    return out
