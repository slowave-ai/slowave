from __future__ import annotations

import pytest

from slowave.dashboard.clients import SUPPORTED_CLIENTS, client_info, client_key


@pytest.mark.parametrize(
    ("agent", "expected"),
    [
        ("mcp:claude-code", "claude-code"),
        ("mcp:claude-ai", "claude-desktop"),
        ("mcp:codex-mcp-client", "codex"),
        ("mcp:codex", "codex"),
        ("mcp:@cline/core", "cline"),
        ("mcp:Cline", "cline"),
        ("mcp:opencode", "opencode"),
        ("mcp:cursor-vscode", "cursor"),
        ("mcp:windsurf", "windsurf"),
        ("mcp:github-copilot-cli", "copilot"),
        ("mcp:antigravity", "antigravity"),
        ("mcp:Visual Studio Code", "copilot"),
        ("mcp:vscode", "copilot"),
        ("mcp:roo-cline", None),
        ("mcp:incline", None),
        ("cli", None),
        ("mcp", None),
        ("mcp:rmcp", None),
        ("mcp:slowave-agent", None),
        (None, None),
        ("", None),
    ],
)
def test_client_key(agent: str | None, expected: str | None) -> None:
    assert client_key(agent) == expected


def test_client_info_uses_display_names() -> None:
    assert client_info("mcp:claude-code") == {"key": "claude-code", "name": "Claude Code"}
    assert client_info("cli") is None
    assert set(SUPPORTED_CLIENTS) >= {"codex", "cline", "opencode"}


def _db():
    import sqlite3

    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript("""
        CREATE TABLE sessions(id TEXT PRIMARY KEY, agent TEXT);
        CREATE TABLE raw_events(id INTEGER PRIMARY KEY, session_id TEXT);
        CREATE TABLE episode_text(episode_id INTEGER, session_id TEXT);
        CREATE TABLE schema_evidence(schema_id INTEGER, raw_event_id INTEGER, episode_id INTEGER);
        INSERT INTO sessions VALUES ('a','mcp:claude-code'),('b','mcp:codex-mcp-client'),('c','cli');
        INSERT INTO raw_events VALUES (1,'a');
        INSERT INTO episode_text VALUES (10,'b'),(11,'c');
        INSERT INTO schema_evidence VALUES (1,1,NULL),(1,NULL,10),(2,NULL,10),(3,NULL,11);
        """)
    return conn


def test_session_and_schema_clients_resolve_sources() -> None:
    from slowave.dashboard.clients import schema_clients, session_clients

    conn = _db()
    sessions = session_clients(conn, ["a", "c", "missing", None])
    assert sessions["a"]["key"] == "claude-code"
    assert sessions["c"] is None and "missing" not in sessions
    by_schema = schema_clients(conn, [1, 2, 3, 4])
    assert [c["key"] for c in by_schema[1]] == ["claude-code", "codex"]
    assert [c["key"] for c in by_schema[2]] == ["codex"]
    assert by_schema[3] == [] and by_schema[4] == []
