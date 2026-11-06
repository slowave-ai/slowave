"""Identify which supported coding-agent client produced a session.

Sessions record ``agent`` as ``mcp:<MCP clientInfo.name>`` (or ``cli`` for the
Slowave CLI).  The dashboard maps that transport-derived name onto the closed
set of supported clients so every memory, procedure, retrieval and activity can
show a logo and name.  Anything unrecognised maps to ``None`` rather than a
guess.
"""

from __future__ import annotations

import logging
import re
import sqlite3
from collections.abc import Iterable
from typing import Any

log = logging.getLogger(__name__)

# key -> display name.  Keys match the logo files in ui/public/img/clients.
SUPPORTED_CLIENTS: dict[str, str] = {
    "claude-code": "Claude Code",
    "claude-desktop": "Claude Desktop",
    "codex": "Codex",
    "copilot": "GitHub Copilot",
    "cursor": "Cursor",
    "opencode": "OpenCode",
    "windsurf": "Windsurf",
    "antigravity": "Antigravity",
    "cline": "Cline",
}

# Ordered (substring, key) rules applied to the lower-cased MCP client name.
_RULES: tuple[tuple[str, str], ...] = (
    ("claude-code", "claude-code"),
    ("claude code", "claude-code"),
    ("claude-ai", "claude-desktop"),
    ("claude-desktop", "claude-desktop"),
    ("claude desktop", "claude-desktop"),
    ("codex", "codex"),
    ("copilot", "copilot"),
    ("cursor", "cursor"),
    ("opencode", "opencode"),
    ("visual studio code", "copilot"),
    ("vscode", "copilot"),
    ("windsurf", "windsurf"),
    ("codeium", "windsurf"),
    ("antigravity", "antigravity"),
    ("cline", "cline"),
)


def client_key(agent: Any) -> str | None:
    """Return the supported client key for a stored session ``agent`` value."""
    if not agent:
        return None
    name = str(agent).strip().lower()
    if name.startswith("mcp:"):
        name = name[4:]
    # Forks that merely embed another client's name are different products.
    if "roo-cline" in name or "roo-code" in name or "roocode" in name:
        return None
    for needle, key in _RULES:
        if re.search(rf"(?<![a-z0-9]){re.escape(needle)}(?![a-z0-9])", name):
            return key
    return None


def client_info(agent: Any) -> dict[str, str] | None:
    key = client_key(agent)
    return {"key": key, "name": SUPPORTED_CLIENTS[key]} if key else None


def session_clients(conn: sqlite3.Connection, session_ids: Iterable[Any]) -> dict[str, dict | None]:
    """Map session id -> client info for the given ids."""
    ids = sorted({str(i) for i in session_ids if i})
    out: dict[str, dict | None] = {}
    for start in range(0, len(ids), 500):
        chunk = ids[start : start + 500]
        marks = ",".join("?" * len(chunk))
        for row in conn.execute(f"SELECT id, agent FROM sessions WHERE id IN ({marks})", chunk):
            out[str(row[0])] = client_info(row[1])
    return out


def schema_clients(conn: sqlite3.Connection, schema_ids: Iterable[int]) -> dict[int, list[dict]]:
    """Map schema (memory) id -> distinct clients that produced its evidence."""
    ids = sorted({int(i) for i in schema_ids})
    found: dict[int, dict[str, dict]] = {i: {} for i in ids}
    for start in range(0, len(ids), 500):
        chunk = ids[start : start + 500]
        marks = ",".join("?" * len(chunk))
        try:
            rows = conn.execute(
                "SELECT DISTINCT se.schema_id AS sid, s.agent AS agent FROM schema_evidence se "
                "LEFT JOIN raw_events re ON re.id = se.raw_event_id "
                "LEFT JOIN episode_text et ON et.episode_id = se.episode_id "
                "JOIN sessions s ON s.id = COALESCE(re.session_id, et.session_id) "
                f"WHERE se.schema_id IN ({marks})",
                chunk,
            ).fetchall()
        except sqlite3.OperationalError as exc:
            # Older databases may lack episode_text; clients are then unknown.
            log.warning("schema_clients unavailable: %s", exc)
            rows = []
        for row in rows:
            info = client_info(row["agent"])
            if info:
                found[int(row["sid"])][info["key"]] = info
    return {sid: sorted(v.values(), key=lambda c: c["name"]) for sid, v in found.items()}
