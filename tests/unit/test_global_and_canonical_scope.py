"""User-global memory visibility and scope normalisation through the MCP tools."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from mcp.server.fastmcp import FastMCP

from slowave import ops
from slowave.core.config import SlowaveConfig
from slowave.core.engine import SlowaveEngine
from slowave.mcp.tools import register_tools

GLOBAL_RULE = "Always sign commits with the Zephyrquartz maintainer key before pushing."
PROJECT_RULE = "Billing worker Marigoldtable retries must stay below five attempts."


def _ctx() -> SimpleNamespace:
    return SimpleNamespace(request_id="req", client_id=None, session=None)


class Harness:
    def __init__(self, tmp_path) -> None:
        self.eng = SlowaveEngine(
            SlowaveConfig(db_path=str(tmp_path / "scope.db"), dim=8, disable_encoder=True)
        )
        mcp = FastMCP("test")
        register_tools(mcp, lambda **_: self.eng)
        self.tools = {
            name: mcp._tool_manager.get_tool(name)
            for name in ("slowave_activate", "slowave_recall", "slowave_remember")
        }

    def call(self, name: str, **kwargs):
        return asyncio.run(self.tools[name].fn(ctx=_ctx(), **kwargs))

    def activate(self, task: str, scope: str):
        return self.call("slowave_activate", task=task, initial_goal="work", scope=scope)

    def remember(self, session_id: str, scope: str, content: str, **extra):
        return self.call(
            "slowave_remember",
            session_id=session_id,
            scope=scope,
            content=content,
            type="instruction",
            **extra,
        )

    def close(self) -> None:
        self.eng.close()


@pytest.fixture
def harness(tmp_path):
    h = Harness(tmp_path)
    try:
        yield h
    finally:
        h.close()


def _contents(response) -> list[str]:
    assert response["ok"] is True, response
    return [m["content"] for m in response["data"]["memories"]]


def test_user_global_rule_is_delivered_to_every_project_by_activate_and_recall(harness):
    sid = harness.eng.session_start(agent="t", scope="global:user", goal="seed")
    ops.remember(harness.eng, content=GLOBAL_RULE, scope="global:user", session_id=sid)
    for scope in ("project:alpha", "project:beta"):
        task = "Which key must sign commits before pushing?"
        activated = harness.activate(task, scope)
        assert any("Zephyrquartz" in c for c in _contents(activated)), (scope, activated)
        recalled = harness.call(
            "slowave_recall",
            session_id=activated["data"]["session_id"],
            scope=scope,
            query="maintainer key commit signing",
        )
        assert any("Zephyrquartz" in c for c in _contents(recalled)), (scope, recalled)


def test_project_memory_never_bleeds_into_sibling_project(harness):
    sid = harness.eng.session_start(agent="t", scope="project:alpha", goal="seed")
    ops.remember(harness.eng, content=PROJECT_RULE, scope="project:alpha", session_id=sid)
    sibling = harness.activate(
        "How many retries may the billing worker Marigoldtable use?", "project:beta"
    )
    assert not any("Marigoldtable" in c for c in _contents(sibling))
    own = harness.activate(
        "How many retries may the billing worker Marigoldtable use?", "project:alpha"
    )
    assert any("Marigoldtable" in c for c in _contents(own))


def test_remember_can_write_a_user_global_rule_from_a_project_session(harness):
    first = harness.activate("start", "project:alpha")
    sid = first["data"]["session_id"]
    stored = harness.remember(sid, "project:alpha", GLOBAL_RULE, visibility="user")
    assert stored["ok"] is True, stored
    assert stored["data"]["scope"] == "global:user"
    row = (
        harness.eng.db.connect()
        .execute("SELECT scope_id FROM schemas WHERE id=?", (int(stored["data"]["memory_id"][4:]),))
        .fetchone()
    )
    assert row["scope_id"] == "global:user"
    other = harness.activate("Which key must sign commits before pushing?", "project:beta")
    assert any("Zephyrquartz" in c for c in _contents(other))
    # Default writes stay project-scoped.
    local = harness.remember(sid, "project:alpha", PROJECT_RULE)
    assert local["data"]["scope"] == "project:alpha"
    # Only the documented value is accepted.
    bad = harness.remember(sid, "project:alpha", "x y z w", visibility="everyone")
    assert bad["ok"] is False


def test_canonical_scope_rules():
    from slowave.core.scope import canonical_scope

    assert canonical_scope("PROJECT:billing-api") == "project:billing-api"
    assert canonical_scope("  project:billing-api/ \n") == "project:billing-api"
    assert canonical_scope("project:Billing-API//") == "project:Billing-API"
    assert canonical_scope("Global:User") == "global:user"
    assert canonical_scope("household") == "household"
    assert canonical_scope("   ") is None


@pytest.mark.parametrize(
    "variant",
    [
        "PROJECT:billing-api",
        "project:Billing-API",
        "project:billing-api/",
        " Project:BILLING-API/ ",
    ],
)
def test_scope_spelling_variants_resolve_to_the_first_registered_scope(harness, variant):
    first = harness.activate("start", "project:billing-api")
    assert first["data"]["memory_state"] == "cold_start"
    sid = first["data"]["session_id"]
    harness.remember(sid, "project:billing-api", "Billing API uses Quillbright tokens.")
    second = harness.activate("Which tokens does the billing API use? Quillbright", variant)
    assert second["ok"] is True, second
    assert second["data"]["memory_state"] == "available"
    assert any("Quillbright" in c for c in _contents(second))
    # session_id/scope agreement tolerates the same spelling variants.
    stored = harness.remember(second["data"]["session_id"], variant, "Another billing fact.")
    assert stored["ok"] is True, stored
    assert stored["data"]["scope"] == "project:billing-api"
    scopes = {r[0] for r in harness.eng.db.connect().execute("SELECT scope_id FROM scope_registry")}
    assert scopes == {"project:billing-api"}


def test_id_case_is_preserved_when_first_registered_with_capitals(harness):
    harness.activate("start", "project:Billing-API")
    again = harness.activate("start", "project:billing-api")
    scopes = {r[0] for r in harness.eng.db.connect().execute("SELECT scope_id FROM scope_registry")}
    assert scopes == {"project:Billing-API"}
    assert again["data"]["memory_state"] in {"cold_start", "available"}


@pytest.mark.parametrize(
    "scope", ["рroject:billing-api", "project:billing-арі", "project:..", "project:a\x00b"]
)
def test_lookalike_and_hostile_scopes_stay_isolated_or_rejected(harness, scope):
    harness.activate("start", "project:billing-api")
    response = harness.activate("start", scope)
    if response["ok"]:
        assert response["data"]["memory_state"] == "cold_start"
        scopes = {
            r[0] for r in harness.eng.db.connect().execute("SELECT scope_id FROM scope_registry")
        }
        assert "project:billing-api" in scopes and scope in scopes


def test_overlong_scope_is_rejected(harness):
    response = harness.activate("start", "project:" + "a" * 5000)
    assert response["ok"] is False


def test_unknown_activate_argument_is_rejected_at_the_tool_layer(tmp_path):
    from slowave.mcp.tools import register_tools as _register

    eng = SlowaveEngine(SlowaveConfig(db_path=str(tmp_path / "x.db"), dim=8, disable_encoder=True))
    try:
        mcp = FastMCP("test")
        _register(mcp, lambda **_: eng)
        out = asyncio.run(
            mcp._tool_manager.call_tool(
                "slowave_activate",
                {"task": "t", "initial_goal": "g", "scope": "project:x", "foo": 1},
            )
        )
        assert out["ok"] is False and out["error"]["code"] == "invalid_input"
        assert out["error"]["field_errors"][0]["path"] == "foo"
    finally:
        eng.close()


def test_legacy_slash_scope_remains_accessible(harness):
    sid = harness.eng.session_start(agent="t", scope="project:legacy", goal="seed")
    ops.remember(
        harness.eng,
        content="Legacy Quillbright scope retains its tokens.",
        scope="project:legacy",
        session_id=sid,
    )
    conn = harness.eng.db.connect()
    conn.execute(
        "UPDATE scope_registry SET scope_id='project:legacy/' WHERE scope_id='project:legacy'"
    )
    conn.execute("UPDATE schemas SET scope_id='project:legacy/' WHERE scope_id='project:legacy'")
    conn.commit()
    response = harness.activate(
        "Which tokens does the legacy scope retain? Quillbright", "project:legacy"
    )
    assert response["data"]["memory_state"] == "available"
    assert any("Quillbright" in c for c in _contents(response))
    recalled = harness.call(
        "slowave_recall",
        session_id=response["data"]["session_id"],
        scope="project:legacy",
        query="Legacy Quillbright tokens",
    )
    assert any("Quillbright" in c for c in _contents(recalled))
    stored = harness.remember(
        response["data"]["session_id"], "project:legacy", "Legacy scope uses silver keys."
    )
    assert stored["ok"] and stored["data"]["scope"] == "project:legacy/"


def test_colliding_legacy_scope_spellings_stay_separate(harness):
    from slowave.mcp.tools import _resolve_scope

    harness.eng.session_start(agent="t", scope="project:legacy", goal="seed")
    conn = harness.eng.db.connect()
    conn.execute(
        "UPDATE scope_registry SET scope_id='project:legacy/' WHERE scope_id='project:legacy'"
    )
    conn.commit()
    harness.eng.session_start(agent="t", scope="project:legacy", goal="seed")
    assert _resolve_scope(harness.eng, "project:legacy/") == "project:legacy/"
    assert _resolve_scope(harness.eng, "project:legacy") == "project:legacy"
