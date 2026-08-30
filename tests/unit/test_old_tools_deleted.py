"""Integration test: assert old MCP tool names are absent from the FastMCP registry.

Validates Step 6.6 (hard break — delete old tools).
Old / bare names forbidden: context, session_start, session_end, event, retrieval_feedback,
  context_feedback, activate, remember, recall, reinforce, commit, stats
  (bare names without the slowave_ prefix are not presented correctly to Cline TUI)
New tools present: slowave_activate, slowave_remember, slowave_recall, slowave_feedback,
                   slowave_commit
"""

from __future__ import annotations


class TestOldToolsDeleted:
    _tool_names: set[str] | None = None

    @classmethod
    def _get_tool_names(cls) -> set[str]:
        """Return the set of registered tool names from the FastMCP instance.

        Cached at class level so we only construct the event loop once.
        """
        if cls._tool_names is not None:
            return cls._tool_names
        import asyncio

        import slowave.mcp.server as srv

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            tools = loop.run_until_complete(srv.mcp.list_tools())
        finally:
            loop.close()
        cls._tool_names = {t.name for t in tools}
        return cls._tool_names

    def test_old_tools_absent(self) -> None:
        tool_names = self._get_tool_names()
        deleted = {
            "context",
            "session_start",
            "session_end",
            "event",
            "retrieval_feedback",
            "context_feedback",
            "slowave_reinforce",
        }
        present_old = deleted & tool_names
        assert not present_old, f"Old tools still registered: {present_old}"

    def test_bare_names_absent(self) -> None:
        """Bare names (without slowave_ prefix) must not be registered.

        Cline TUI presents tool names to the LLM exactly as registered on the
        server.  If a tool is registered as 'activate', the LLM sees 'activate'
        in the system prompt and is forced to call it as 'activate'.  The
        lifecycle block in .clinerules instructs the LLM to call 'slowave_activate',
        so the names MUST match — bare names guarantee a mismatch and broken tools.
        """
        tool_names = self._get_tool_names()
        bare = {"activate", "remember", "recall", "reinforce", "feedback", "commit", "stats"}
        present_bare = bare & tool_names
        assert (
            not present_bare
        ), f"Bare tool names registered (must use slowave_ prefix): {present_bare}"

    def test_new_tools_present(self) -> None:
        tool_names = self._get_tool_names()
        expected = {
            "slowave_activate",
            "slowave_remember",
            "slowave_recall",
            "slowave_feedback",
            "slowave_commit",
        }
        missing = expected - tool_names
        assert not missing, f"New tools missing from registry: {missing}"

    def test_administrative_stats_absent(self) -> None:
        """Administrative health data must not appear in the cognitive registry."""
        assert "slowave_stats" not in self._get_tool_names()

    def test_current_feedback_commit_and_write_schemas_are_breaking(self) -> None:
        import asyncio

        import slowave.mcp.server as srv

        loop = asyncio.new_event_loop()
        try:
            tools = {tool.name: tool for tool in loop.run_until_complete(srv.mcp.list_tools())}
        finally:
            loop.close()
        feedback_fields = set(tools["slowave_feedback"].inputSchema["properties"])
        assert {"memory_feedback", "procedure_feedback", "coverage"} <= feedback_fields
        assert (
            not {
                "feedback",
                "outcome",
                "used_memory_ids",
                "used_procedure_ids",
            }
            & feedback_fields
        )
        commit_fields = set(tools["slowave_commit"].inputSchema["properties"])
        assert commit_fields == {
            "session_id",
            "final_goal",
            "outcome",
            "outcome_summary",
            "verification",
            "procedure",
            "trajectory",
        }
        recall_fields = set(tools["slowave_recall"].inputSchema["properties"])
        assert recall_fields == {"query", "session_id", "scope", "task_context", "evidence"}
        activate = tools["slowave_activate"].inputSchema
        assert set(activate["properties"]) == {
            "task",
            "initial_goal",
            "scope",
            "continuity_id",
            "task_context",
        }
        assert set(activate["required"]) == {"task", "initial_goal", "scope"}
        remember = tools["slowave_remember"].inputSchema
        assert set(remember["properties"]) == {
            "scope",
            "session_id",
            "content",
            "type",
            "memories",
            "occurred_at",
        }
        assert set(remember["required"]) == {"scope", "session_id"}
        assert "items" not in remember["properties"]
