"""Tests for slowave setup/cleanup core logic.

Uses a fake home directory (tmp_path) so no real config files are touched.
All tests are offline — no binaries, no subprocesses, no network.

Coverage:
  - _patch_mcp_servers          idempotence, HTTP format, legacy stdio migration
  - _remove_mcp_servers_from_settings
  - _patch_claude_code_hooks    idempotence, new
  - _patch_codex_mcp / _patch_codex_hooks / _remove_codex_hooks  (Codex, TOML)
  - _read_toml / _write_toml    round-trips comments, backup creation
  - _inject_block               new file, idempotent update, legacy strip
  - _write_json / _backup_file  backup creation
  - malformed JSON              (SystemExit)
  - _read_json                  missing file returns {}
  - cleanup helpers             _remove_lifecycle_blocks, _remove_mcp_entry
"""

from __future__ import annotations

import json

import pytest

from slowave.cli.setup import (
    _MARKER_START,
    _backup_file,
    _detect_lifecycle_version,
    _inject_block,
    _lifecycle_block,
    _patch_claude_code_hooks,
    _patch_codex_hooks,
    _patch_codex_mcp,
    _patch_mcp_servers,
    _patch_opencode_instructions,
    _patch_opencode_mcp,
    _read_json,
    _read_toml,
    _remove_codex_hooks,
    _remove_mcp_servers_from_settings,
    _write_json,
    _write_toml,
)
from slowave.lifecycle import LIFECYCLE_VERSION

HTTP_URL = "http://127.0.0.1:8766/mcp"


# ===========================================================================
# _patch_mcp_servers
# ===========================================================================


class TestPatchMcpServers:
    def test_adds_server_to_empty_config(self):
        cfg, changed = _patch_mcp_servers({})
        assert changed is True
        assert cfg["mcpServers"]["slowave"] == {"url": HTTP_URL}

    def test_idempotent_http_format(self):
        cfg = {"mcpServers": {"slowave": {"url": HTTP_URL}}}
        _, changed = _patch_mcp_servers(cfg)
        assert changed is False

    def test_migrates_legacy_stdio_format(self):
        # Old stdio entry should be replaced with HTTP
        cfg = {
            "mcpServers": {"slowave": {"type": "stdio", "command": "/usr/local/bin/slowave-mcp"}}
        }
        cfg2, changed = _patch_mcp_servers(cfg)
        assert changed is True
        assert cfg2["mcpServers"]["slowave"] == {"url": HTTP_URL}

    def test_migrates_legacy_command_only_format(self):
        cfg = {"mcpServers": {"slowave": {"command": "/usr/local/bin/slowave-mcp"}}}
        cfg2, changed = _patch_mcp_servers(cfg)
        assert changed is True
        assert cfg2["mcpServers"]["slowave"] == {"url": HTTP_URL}

    def test_preserves_other_mcp_servers(self):
        cfg = {"mcpServers": {"othertool": {"command": "/usr/bin/other"}}}
        cfg2, _ = _patch_mcp_servers(cfg)
        assert "othertool" in cfg2["mcpServers"]

    def test_include_type_writes_type_field(self):
        """Claude Code requires type:http."""
        cfg, changed = _patch_mcp_servers({}, include_type=True)
        assert changed is True
        assert cfg["mcpServers"]["slowave"] == {"type": "http", "url": HTTP_URL}

    def test_no_type_by_default(self):
        """Cline / Cursor / Windsurf use url-only."""
        cfg, changed = _patch_mcp_servers({})
        assert changed is True
        assert cfg["mcpServers"]["slowave"] == {"url": HTTP_URL}
        assert "type" not in cfg["mcpServers"]["slowave"]


# ===========================================================================
# _remove_mcp_servers_from_settings
# ===========================================================================


class TestRemoveMcpServersFromSettings:
    def test_removes_slowave_entry(self):
        cfg = {"mcpServers": {"slowave": {"command": "/usr/local/bin/slowave-mcp"}}}
        cfg2, changed = _remove_mcp_servers_from_settings(cfg)
        assert changed is True
        assert "slowave" not in cfg2.get("mcpServers", {})

    def test_removes_empty_mcpServers_key(self):
        cfg = {"mcpServers": {"slowave": {"command": "/usr/local/bin/slowave-mcp"}}}
        cfg2, _ = _remove_mcp_servers_from_settings(cfg)
        assert "mcpServers" not in cfg2

    def test_no_change_when_absent(self):
        _, changed = _remove_mcp_servers_from_settings({"otherKey": "value"})
        assert changed is False

    def test_no_change_when_slowave_not_present(self):
        _, changed = _remove_mcp_servers_from_settings({"mcpServers": {"othertool": {}}})
        assert changed is False

    def test_preserves_other_servers(self):
        cfg = {"mcpServers": {"slowave": {}, "other": {"command": "/x"}}}
        cfg2, changed = _remove_mcp_servers_from_settings(cfg)
        assert changed is True
        assert "other" in cfg2["mcpServers"]


# ===========================================================================
# _patch_claude_code_hooks
# ===========================================================================


class TestPatchClaudeCodeHooks:
    def test_adds_hooks_to_empty_config(self):
        cfg, changed = _patch_claude_code_hooks({})
        assert changed is True
        assert "UserPromptSubmit" in cfg["hooks"]
        assert "Stop" in cfg["hooks"]

    def test_idempotent_when_hooks_present(self):
        cfg, _ = _patch_claude_code_hooks({})
        _, changed2 = _patch_claude_code_hooks(cfg)
        assert changed2 is False

    def test_preserves_unrelated_hooks(self):
        existing = {
            "hooks": {
                "PreToolUse": [
                    {"matcher": "", "hooks": [{"type": "command", "command": "echo hi"}]}
                ]
            }
        }
        cfg2, _ = _patch_claude_code_hooks(existing)
        assert "PreToolUse" in cfg2["hooks"]

    def test_replaces_stale_hook_command(self):
        """If hook is present but command text differs (version upgrade), it is replaced."""
        stale_cmd = "echo 'SLOWAVE MANDATORY: old instructions'"
        cfg = {
            "hooks": {
                "UserPromptSubmit": [
                    {"matcher": "", "hooks": [{"type": "command", "command": stale_cmd}]}
                ]
            }
        }
        cfg2, changed = _patch_claude_code_hooks(cfg)
        assert changed is True
        # Stale command should be gone
        cmds = [h["command"] for g in cfg2["hooks"]["UserPromptSubmit"] for h in g.get("hooks", [])]
        assert stale_cmd not in cmds
        # Current command should be present
        from slowave.cli.setup import _USER_PROMPT_CMD

        assert any(_USER_PROMPT_CMD in c for c in cmds)

    def test_idempotent_with_current_hook_command(self):
        """If hook already has the exact current command, no change."""
        from slowave.cli.setup import _STOP_CMD, _USER_PROMPT_CMD

        cfg = {
            "hooks": {
                "UserPromptSubmit": [
                    {"matcher": "", "hooks": [{"type": "command", "command": _USER_PROMPT_CMD}]}
                ],
                "Stop": [{"matcher": "", "hooks": [{"type": "command", "command": _STOP_CMD}]}],
            }
        }
        _, changed = _patch_claude_code_hooks(cfg)
        assert changed is False


# ===========================================================================
# Codex — _patch_codex_mcp / _patch_codex_hooks / _remove_codex_hooks
# ===========================================================================


class TestPatchCodexMcp:
    def test_adds_server_to_empty_config(self):
        cfg, changed = _patch_codex_mcp({})
        assert changed is True
        assert cfg["mcp_servers"]["slowave"] == {"url": HTTP_URL}

    def test_idempotent_when_present(self):
        cfg, _ = _patch_codex_mcp({})
        _, changed2 = _patch_codex_mcp(cfg)
        assert changed2 is False

    def test_no_auth_or_type_field(self):
        """Codex needs only `url` for an unauthenticated local Streamable HTTP server."""
        cfg, _ = _patch_codex_mcp({})
        assert set(cfg["mcp_servers"]["slowave"].keys()) == {"url"}

    def test_preserves_other_mcp_servers(self):
        cfg = {"mcp_servers": {"othertool": {"command": "npx"}}}
        cfg2, _ = _patch_codex_mcp(cfg)
        assert "othertool" in cfg2["mcp_servers"]

    def test_updates_stale_url(self):
        cfg = {"mcp_servers": {"slowave": {"url": "http://old-host:1234/mcp"}}}
        cfg2, changed = _patch_codex_mcp(cfg)
        assert changed is True
        assert cfg2["mcp_servers"]["slowave"] == {"url": HTTP_URL}

    def test_round_trips_through_tomlkit(self, tmp_path):
        """Patched config must serialize to valid, re-parseable TOML."""
        target = tmp_path / "config.toml"
        target.write_text('# user comment\nmodel = "gpt-5.5"\n', encoding="utf-8")
        cfg = _read_toml(target)
        cfg, changed = _patch_codex_mcp(cfg)
        assert changed is True
        _write_toml(target, cfg)
        content = target.read_text()
        assert "# user comment" in content
        reparsed = _read_toml(target)
        assert reparsed["mcp_servers"]["slowave"]["url"] == HTTP_URL


class TestPatchOpencodeMcp:
    def test_adds_server_to_empty_config(self):
        cfg, changed = _patch_opencode_mcp({})
        assert changed is True
        assert cfg["mcp"]["slowave"] == {"type": "remote", "url": HTTP_URL, "enabled": True}

    def test_idempotent_when_present(self):
        cfg, _ = _patch_opencode_mcp({})
        _, changed2 = _patch_opencode_mcp(cfg)
        assert changed2 is False

    def test_preserves_other_mcp_entries(self):
        cfg = {"mcp": {"othertool": {"type": "local", "command": ["npx"]}}}
        cfg2, _ = _patch_opencode_mcp(cfg)
        assert "othertool" in cfg2["mcp"]


class TestPatchOpencodeInstructions:
    def test_adds_path_to_empty_config(self):
        cfg, changed = _patch_opencode_instructions(
            {}, "/home/user/.config/opencode/slowave-instructions.md"
        )
        assert changed is True
        assert cfg["instructions"] == ["/home/user/.config/opencode/slowave-instructions.md"]

    def test_idempotent_when_present(self):
        cfg, _ = _patch_opencode_instructions({}, "/path/to/instructions.md")
        _, changed2 = _patch_opencode_instructions(cfg, "/path/to/instructions.md")
        assert changed2 is False

    def test_preserves_other_instructions_entries(self):
        cfg = {"instructions": [".claude/CLAUDE.md"]}
        cfg2, changed = _patch_opencode_instructions(cfg, "/path/to/slowave-instructions.md")
        assert changed is True
        assert cfg2["instructions"] == [".claude/CLAUDE.md", "/path/to/slowave-instructions.md"]

    def test_changed_independent_of_mcp_patch(self):
        """Regression: MCP entry already present but instructions not yet registered —
        the instructions patch must still report changed=True so the caller persists it."""
        cfg, _ = _patch_opencode_mcp({})
        _, mcp_changed_again = _patch_opencode_mcp(cfg)
        cfg, instructions_changed = _patch_opencode_instructions(
            cfg, "/path/to/slowave-instructions.md"
        )
        assert mcp_changed_again is False
        assert instructions_changed is True


class TestPatchCodexHooks:
    def test_adds_hooks_to_empty_config(self):
        cfg, changed = _patch_codex_hooks({})
        assert changed is True
        assert "UserPromptSubmit" in cfg["hooks"]
        assert "Stop" in cfg["hooks"]

    def test_idempotent_when_hooks_present(self):
        cfg, _ = _patch_codex_hooks({})
        _, changed2 = _patch_codex_hooks(cfg)
        assert changed2 is False

    def test_replaces_stale_hook_command(self):
        stale_cmd = "echo 'SLOWAVE MANDATORY: old instructions'"
        cfg = {
            "hooks": {"UserPromptSubmit": [{"hooks": [{"type": "command", "command": stale_cmd}]}]}
        }
        cfg2, changed = _patch_codex_hooks(cfg)
        assert changed is True
        cmds = [h["command"] for g in cfg2["hooks"]["UserPromptSubmit"] for h in g.get("hooks", [])]
        assert stale_cmd not in cmds

    def test_preserves_unrelated_hook_events(self):
        existing = {
            "hooks": {"PreToolUse": [{"hooks": [{"type": "command", "command": "echo hi"}]}]}
        }
        cfg2, _ = _patch_codex_hooks(existing)
        assert "PreToolUse" in cfg2["hooks"]

    def test_round_trips_as_array_of_tables(self, tmp_path):
        """Written TOML must use [[hooks.Event]] array-of-tables syntax."""
        target = tmp_path / "config.toml"
        cfg = _read_toml(target)
        cfg, _ = _patch_codex_hooks(cfg)
        _write_toml(target, cfg)
        content = target.read_text()
        assert "[[hooks.UserPromptSubmit]]" in content
        assert "[[hooks.Stop]]" in content
        reparsed = _read_toml(target)
        assert len(reparsed["hooks"]["UserPromptSubmit"]) == 1


class TestRemoveCodexHooks:
    def test_removes_slowave_hooks(self):
        cfg, _ = _patch_codex_hooks({})
        cfg2, changed = _remove_codex_hooks(cfg)
        assert changed is True
        assert cfg2["hooks"]["UserPromptSubmit"] == []
        assert cfg2["hooks"]["Stop"] == []

    def test_no_change_when_absent(self):
        _, changed = _remove_codex_hooks({})
        assert changed is False

    def test_preserves_unrelated_hooks(self):
        cfg, _ = _patch_codex_hooks(
            {"hooks": {"PreToolUse": [{"hooks": [{"type": "command", "command": "echo hi"}]}]}}
        )
        cfg2, _ = _remove_codex_hooks(cfg)
        assert "PreToolUse" in cfg2["hooks"]
        assert cfg2["hooks"]["PreToolUse"][0]["hooks"][0]["command"] == "echo hi"


# ===========================================================================
# _read_toml / _write_toml
# ===========================================================================


class TestReadWriteToml:
    def test_returns_empty_doc_for_missing_file(self, tmp_path):
        cfg = _read_toml(tmp_path / "nonexistent.toml")
        assert dict(cfg) == {}

    def test_reads_valid_toml(self, tmp_path):
        f = tmp_path / "config.toml"
        f.write_text('key = "value"\n', encoding="utf-8")
        assert _read_toml(f)["key"] == "value"

    def test_exits_on_malformed_toml(self, tmp_path):
        f = tmp_path / "bad.toml"
        f.write_text("this is not [valid toml", encoding="utf-8")
        with pytest.raises(SystemExit):
            _read_toml(f)

    def test_backup_created_before_overwrite(self, tmp_path):
        target = tmp_path / "config.toml"
        target.write_text("original = true\n", encoding="utf-8")
        cfg = _read_toml(target)
        cfg["updated"] = True
        _write_toml(target, cfg)
        backups = list(tmp_path.glob("config.toml.bak.*"))
        assert len(backups) == 1
        assert "original = true" in backups[0].read_text()

    def test_preserves_comments_on_write(self, tmp_path):
        target = tmp_path / "config.toml"
        target.write_text('# important comment\nmodel = "gpt-5.5"\n', encoding="utf-8")
        cfg = _read_toml(target)
        cfg["extra"] = "value"
        _write_toml(target, cfg)
        assert "# important comment" in target.read_text()


# ===========================================================================
# _inject_block
# ===========================================================================


class TestInjectBlock:
    def test_creates_new_file(self, tmp_path):
        target = tmp_path / "CLAUDE.md"
        changed = _inject_block(target, _lifecycle_block("claude-code"))
        assert changed is True
        assert target.exists()
        assert _MARKER_START in target.read_text()

    def test_idempotent_on_second_call(self, tmp_path):
        target = tmp_path / "CLAUDE.md"
        block = _lifecycle_block("claude-code")
        _inject_block(target, block)
        changed = _inject_block(target, block)
        assert changed is False

    def test_updates_stale_v1_block(self, tmp_path):
        target = tmp_path / "CLAUDE.md"
        old = (
            "<!-- slowave-lifecycle-start v1 -->\nold content\n<!-- slowave-lifecycle-end v1 -->\n"
        )
        target.write_text(old, encoding="utf-8")
        changed = _inject_block(target, _lifecycle_block("claude-code"))
        assert changed is True
        content = target.read_text()
        assert "old content" not in content
        assert _MARKER_START in content

    def test_prepends_before_existing_user_content(self, tmp_path):
        target = tmp_path / ".clinerules"
        target.write_text("# My existing rules\n", encoding="utf-8")
        _inject_block(target, _lifecycle_block("cline-tui"))
        content = target.read_text()
        assert content.index(_MARKER_START) < content.index("# My existing rules")

    def test_creates_parent_dirs(self, tmp_path):
        target = tmp_path / "deep" / "nested" / "CLAUDE.md"
        _inject_block(target, _lifecycle_block("claude-code"))
        assert target.exists()

    def test_strips_legacy_unmarked_section(self, tmp_path):
        # Legacy section ends when the next same-level (##) heading is found.
        legacy = "## Slowave memory\nsome old content\n\n## My Notes\nuser content\n"
        target = tmp_path / "CLAUDE.md"
        target.write_text(legacy, encoding="utf-8")
        _inject_block(target, _lifecycle_block("claude-code"))
        content = target.read_text()
        assert "some old content" not in content
        assert "## My Notes" in content
        assert "user content" in content


# ===========================================================================
# _detect_lifecycle_version (WP-8)
# ===========================================================================


class TestDetectLifecycleVersion:
    def test_detects_current_version_in_generated_block(self):
        assert _detect_lifecycle_version(_lifecycle_block("claude-code")) == LIFECYCLE_VERSION

    def test_detects_stale_v1(self):
        text = "<!-- slowave-lifecycle-start v1 -->\nold\n<!-- slowave-lifecycle-end v1 -->\n"
        assert _detect_lifecycle_version(text) == "v1"

    def test_detects_stale_v2_among_other_content(self):
        text = (
            "# My rules\n\n<!-- slowave-lifecycle-start v2 -->\nold\n"
            "<!-- slowave-lifecycle-end v2 -->\n\n## More rules\n"
        )
        assert _detect_lifecycle_version(text) == "v2"

    def test_returns_none_when_absent(self):
        assert _detect_lifecycle_version("# just some notes, no slowave block\n") is None

    def test_generated_template_markers_match_the_constant_not_a_hardcoded_literal(self):
        """Regression guard for the "verify every integration receives the
        current lifecycle version, not only the template in source" gap
        (WP-8): the start/end markers must both derive from LIFECYCLE_VERSION,
        so a future version bump can't silently drift between the two.
        """
        block = _lifecycle_block("claude-code")
        assert block.count(f"-start {LIFECYCLE_VERSION} -->") == 1
        assert block.count(f"-end {LIFECYCLE_VERSION} -->") == 1


# ===========================================================================
# _write_json + _backup_file
# ===========================================================================


class TestWriteJsonBackup:
    def test_backup_created_before_overwrite(self, tmp_path):
        target = tmp_path / "config.json"
        target.write_text('{"original": true}\n', encoding="utf-8")
        _write_json(target, {"updated": True})
        backups = list(tmp_path.glob("config.json.bak.*"))
        assert len(backups) == 1
        assert json.loads(backups[0].read_text()) == {"original": True}

    def test_no_backup_when_file_missing(self, tmp_path):
        _write_json(tmp_path / "new.json", {"key": "val"})
        assert list(tmp_path.glob("new.json.bak.*")) == []

    def test_write_creates_parent_dirs(self, tmp_path):
        target = tmp_path / "a" / "b" / "cfg.json"
        _write_json(target, {"x": 1})
        assert target.exists()
        assert json.loads(target.read_text()) == {"x": 1}

    def test_backup_file_direct(self, tmp_path):
        f = tmp_path / "myfile.txt"
        f.write_text("hello", encoding="utf-8")
        bak = _backup_file(f)
        assert bak is not None and bak.exists()
        assert bak.read_text() == "hello"
        assert ".bak." in bak.name

    def test_backup_file_returns_none_when_missing(self, tmp_path):
        assert _backup_file(tmp_path / "nonexistent.txt") is None

    def test_only_one_backup_kept_on_multiple_writes(self, tmp_path):
        """Re-running setup must not accumulate backup copies."""
        target = tmp_path / "config.json"
        target.write_text('{"v": 1}\n', encoding="utf-8")
        _write_json(target, {"v": 2})
        _write_json(target, {"v": 3})
        backups = list(tmp_path.glob("config.json.bak.*"))
        assert len(backups) == 1
        # The surviving backup is from the second write (before v3 was written)
        assert json.loads(backups[0].read_text()) == {"v": 2}


class TestInjectBlockBackup:
    def test_backup_on_update(self, tmp_path):
        target = tmp_path / "CLAUDE.md"
        original = "<!-- slowave-lifecycle-start v1 -->\nold\n<!-- slowave-lifecycle-end v1 -->\n"
        target.write_text(original, encoding="utf-8")
        _inject_block(target, _lifecycle_block("claude-code"))
        backups = list(tmp_path.glob("CLAUDE.md.bak.*"))
        assert len(backups) == 1
        assert backups[0].read_text() == original

    def test_backup_when_prepending_to_existing(self, tmp_path):
        target = tmp_path / ".clinerules"
        target.write_text("# existing\n", encoding="utf-8")
        _inject_block(target, _lifecycle_block("cline-tui"))
        assert len(list(tmp_path.glob(".clinerules.bak.*"))) == 1

    def test_no_backup_for_brand_new_file(self, tmp_path):
        target = tmp_path / "CLAUDE.md"
        _inject_block(target, _lifecycle_block("claude-code"))
        assert list(tmp_path.glob("CLAUDE.md.bak.*")) == []


# ===========================================================================
# _read_json
# ===========================================================================


class TestReadJson:
    def test_returns_empty_dict_for_missing_file(self, tmp_path):
        assert _read_json(tmp_path / "nonexistent.json") == {}

    def test_reads_valid_json(self, tmp_path):
        f = tmp_path / "config.json"
        f.write_text('{"key": "value"}', encoding="utf-8")
        assert _read_json(f) == {"key": "value"}

    def test_exits_on_malformed_json(self, tmp_path):
        f = tmp_path / "bad.json"
        f.write_text("{not valid json", encoding="utf-8")
        with pytest.raises(SystemExit):
            _read_json(f)


# ===========================================================================
# Cleanup helpers — _remove_lifecycle_blocks, _remove_mcp_configs
# Monkey-patches _home() in both modules to redirect to tmp_path.
# ===========================================================================

import slowave.cli.cleanup as _cleanup_mod
import slowave.cli.setup as _setup_mod


@pytest.fixture()
def fake_home(tmp_path, monkeypatch):
    """Redirect _home() to tmp_path in both setup and cleanup modules."""
    monkeypatch.setattr(_setup_mod, "_home", lambda: tmp_path)
    monkeypatch.setattr(_cleanup_mod, "_home", lambda: tmp_path)
    return tmp_path


class TestCleanupRemoveLifecycleBlocks:
    def test_removes_block_from_clinerules(self, fake_home):
        target = fake_home / ".cline" / "rules" / "slowave.md"
        target.parent.mkdir(parents=True, exist_ok=True)
        block = _lifecycle_block("cline-tui")
        target.write_text(block + "\n# My Notes\n", encoding="utf-8")

        count = _cleanup_mod._remove_lifecycle_blocks(dry_run=False)

        assert count >= 1
        remaining = target.read_text()
        assert _MARKER_START not in remaining
        assert "# My Notes" in remaining

    def test_removes_block_from_claude_md(self, fake_home):
        claude_dir = fake_home / ".claude"
        claude_dir.mkdir(parents=True)
        target = claude_dir / "CLAUDE.md"
        block = _lifecycle_block("claude-code")
        target.write_text(block, encoding="utf-8")

        count = _cleanup_mod._remove_lifecycle_blocks(dry_run=False)

        assert count >= 1
        # File with only the block becomes empty → unlinked
        assert not target.exists() or _MARKER_START not in target.read_text()

    def test_dry_run_does_not_modify_files(self, fake_home):
        target = fake_home / ".cline" / "rules" / "slowave.md"
        target.parent.mkdir(parents=True, exist_ok=True)
        block = _lifecycle_block("cline-tui")
        original = block + "\n# Notes\n"
        target.write_text(original, encoding="utf-8")

        _cleanup_mod._remove_lifecycle_blocks(dry_run=True)

        assert target.read_text() == original

    def test_no_op_on_file_without_slowave_content(self, fake_home):
        target = fake_home / ".cline" / "rules" / "slowave.md"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("# Regular rules\n", encoding="utf-8")

        count = _cleanup_mod._remove_lifecycle_blocks(dry_run=False)

        assert count == 0
        assert target.read_text() == "# Regular rules\n"


class TestCleanupRemoveMcpConfigs:
    def test_removes_slowave_from_cursor_mcp(self, fake_home):
        cursor_dir = fake_home / ".cursor"
        cursor_dir.mkdir()
        cfg_path = cursor_dir / "mcp.json"
        cfg_path.write_text(
            json.dumps(
                {"mcpServers": {"slowave": {"command": "/usr/local/bin/slowave-mcp"}, "other": {}}}
            ),
            encoding="utf-8",
        )

        count = _cleanup_mod._remove_mcp_configs(dry_run=False)

        assert count >= 1
        remaining = json.loads(cfg_path.read_text())
        assert "slowave" not in remaining.get("mcpServers", {})
        assert "other" in remaining["mcpServers"]

    def test_dry_run_does_not_write_mcp_configs(self, fake_home):
        cursor_dir = fake_home / ".cursor"
        cursor_dir.mkdir()
        cfg_path = cursor_dir / "mcp.json"
        original = json.dumps(
            {"mcpServers": {"slowave": {"command": "/usr/local/bin/slowave-mcp"}}}
        )
        cfg_path.write_text(original, encoding="utf-8")

        _cleanup_mod._remove_mcp_configs(dry_run=True)

        assert cfg_path.read_text() == original

    def test_no_op_when_no_mcp_files_exist(self, fake_home):
        count = _cleanup_mod._remove_mcp_configs(dry_run=False)
        assert count == 0

    def test_removes_slowave_mcp_and_hooks_from_codex_config(self, fake_home):
        """Codex keeps MCP entry + hooks in one TOML file — both must be removed in one write."""
        codex_dir = fake_home / ".codex"
        codex_dir.mkdir()
        cfg_path = codex_dir / "config.toml"
        cfg_path.write_text(
            'model = "gpt-5.5"\n\n'
            "[mcp_servers.slowave]\n"
            'url = "http://127.0.0.1:8766/mcp"\n\n'
            "[mcp_servers.other]\n"
            'command = "npx"\n\n'
            "[[hooks.UserPromptSubmit]]\n"
            "[[hooks.UserPromptSubmit.hooks]]\n"
            'type = "command"\n'
            f'command = "{_setup_mod._USER_PROMPT_CMD}"\n',
            encoding="utf-8",
        )

        count = _cleanup_mod._remove_mcp_configs(dry_run=False)

        assert count >= 1
        remaining = _read_toml(cfg_path)
        assert "slowave" not in remaining.get("mcp_servers", {})
        assert "other" in remaining["mcp_servers"]
        assert remaining["hooks"]["UserPromptSubmit"] == []
        assert remaining["model"] == "gpt-5.5"

    def test_dry_run_does_not_write_codex_config(self, fake_home):
        codex_dir = fake_home / ".codex"
        codex_dir.mkdir()
        cfg_path = codex_dir / "config.toml"
        original = '[mcp_servers.slowave]\nurl = "http://127.0.0.1:8766/mcp"\n'
        cfg_path.write_text(original, encoding="utf-8")

        _cleanup_mod._remove_mcp_configs(dry_run=True)

        assert cfg_path.read_text() == original


class TestCleanupRemoveSetupBackups:
    def test_removes_bak_files_from_home(self, fake_home):
        bak = fake_home / ".clinerules.bak.20260611_120000"
        bak.write_text("old content", encoding="utf-8")

        count = _cleanup_mod._remove_setup_backups(dry_run=False)

        assert count == 1
        assert not bak.exists()

    def test_removes_bak_files_from_claude_dir(self, fake_home):
        (fake_home / ".claude").mkdir()
        bak = fake_home / ".claude" / "settings.json.bak.20260611_120000"
        bak.write_text("{}", encoding="utf-8")

        count = _cleanup_mod._remove_setup_backups(dry_run=False)

        assert count == 1
        assert not bak.exists()

    def test_dry_run_does_not_delete_backups(self, fake_home):
        bak = fake_home / ".clinerules.bak.20260611_120000"
        bak.write_text("old content", encoding="utf-8")

        _cleanup_mod._remove_setup_backups(dry_run=True)

        assert bak.exists()

    def test_no_op_when_no_backups_exist(self, fake_home):
        count = _cleanup_mod._remove_setup_backups(dry_run=False)
        assert count == 0
