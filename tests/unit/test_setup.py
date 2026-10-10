"""Tests for slowave setup/cleanup core logic.

Uses a fake home directory (tmp_path) so no real config files are touched.
All tests are offline — no binaries, no subprocesses, no network.

Coverage:
  - _patch_mcp_servers          idempotence, HTTP format, legacy stdio migration
  - _remove_mcp_servers_from_settings
  - _patch_codex_mcp            TOML configuration
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
from click.testing import CliRunner

from slowave.cli.main import cli
from slowave.cli.setup import (
    _MARKER_START,
    _backup_file,
    _build_summary,
    _detect_lifecycle_version,
    _detected_clients,
    _inject_block,
    _lifecycle_block,
    _patch_codex_mcp,
    _patch_mcp_servers,
    _patch_opencode_instructions,
    _patch_opencode_mcp,
    _read_json,
    _read_toml,
    _remove_mcp_servers_from_settings,
    _write_json,
    _write_toml,
    setup_cmd,
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
# Codex — _patch_codex_mcp
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
        content = target.read_text(encoding="utf-8")
        assert "# user comment" in content
        reparsed = _read_toml(target)
        assert reparsed["mcp_servers"]["slowave"]["url"] == HTTP_URL


class TestPatchOpencodeMcp:
    def test_adds_server_to_empty_config(self):
        cfg, changed = _patch_opencode_mcp({})
        assert changed is True
        assert cfg["mcp"]["slowave"] == {
            "type": "remote",
            "url": HTTP_URL,
            "enabled": True,
        }

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
        assert cfg2["instructions"] == [
            ".claude/CLAUDE.md",
            "/path/to/slowave-instructions.md",
        ]

    def test_changed_independent_of_mcp_patch(self):
        """Regression: MCP entry already present but instructions not yet registered —
        the instructions patch must still report changed=True so the caller persists it.
        """
        cfg, _ = _patch_opencode_mcp({})
        _, mcp_changed_again = _patch_opencode_mcp(cfg)
        cfg, instructions_changed = _patch_opencode_instructions(
            cfg, "/path/to/slowave-instructions.md"
        )
        assert mcp_changed_again is False
        assert instructions_changed is True


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
        assert "original = true" in backups[0].read_text(encoding="utf-8")

    def test_preserves_comments_on_write(self, tmp_path):
        target = tmp_path / "config.toml"
        target.write_text('# important comment\nmodel = "gpt-5.5"\n', encoding="utf-8")
        cfg = _read_toml(target)
        cfg["extra"] = "value"
        _write_toml(target, cfg)
        assert "# important comment" in target.read_text(encoding="utf-8")


# ===========================================================================
# _inject_block
# ===========================================================================


class TestInjectBlock:
    def test_creates_new_file(self, tmp_path):
        target = tmp_path / "CLAUDE.md"
        changed = _inject_block(target, _lifecycle_block("claude-code"))
        assert changed is True
        assert target.exists()
        assert _MARKER_START in target.read_text(encoding="utf-8")

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
        content = target.read_text(encoding="utf-8")
        assert "old content" not in content
        assert _MARKER_START in content

    def test_same_version_with_stale_content_is_not_up_to_date(self):
        from slowave.cli.setup import _lifecycle_block_up_to_date

        stale = (
            "<!-- slowave-lifecycle-start v9 -->\nold content\n" "<!-- slowave-lifecycle-end v9 -->"
        )
        assert not _lifecycle_block_up_to_date(stale, _lifecycle_block("claude-code"))

    def test_exact_current_block_is_up_to_date(self):
        from slowave.cli.setup import _lifecycle_block_up_to_date

        block = _lifecycle_block("claude-code")
        assert _lifecycle_block_up_to_date(block, block)

    def test_prepends_before_existing_user_content(self, tmp_path):
        target = tmp_path / ".clinerules"
        target.write_text("# My existing rules\n", encoding="utf-8")
        _inject_block(target, _lifecycle_block("cline-tui"))
        content = target.read_text(encoding="utf-8")
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
        content = target.read_text(encoding="utf-8")
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

    def test_generated_template_markers_match_the_constant_not_a_hardcoded_literal(
        self,
    ):
        """Regression guard for the "verify every integration receives the
        current lifecycle version, not only the template in source" gap
        (WP-8): the start/end markers must both derive from LIFECYCLE_VERSION,
        so a future version bump can't silently drift between the two.
        """
        block = _lifecycle_block("claude-code")
        assert block.count(f"-start {LIFECYCLE_VERSION} -->") == 1
        assert block.count(f"-end {LIFECYCLE_VERSION} -->") == 1

    def test_generated_block_mandates_clear_procedures_and_memory_quality(self):
        block = _lifecycle_block("claude-code")
        assert "reusable multi-step method" in block
        assert "two ordered task actions" in block
        assert "standalone `outcome_summary`" in block
        assert "`verification`" in block
        assert "connected MCP tools' schemas" in block
        assert "conditional requirements" in block

    def test_generated_block_hardens_client_memory_responsibilities(self):
        block = _lifecycle_block("claude-code")
        assert "Use this loop once per user task" in block
        assert "Activate before your first response" in block
        assert "Assess every retrieval" in block
        assert "Commit before the final response" in block
        assert '`{"ok":false,...}`' in block
        assert "project:<repository-root-name>" in block
        assert "project:<basename(cwd)>" in block
        assert "Never activate because of a hook, stop event" in block
        assert "Do not invent IDs, scope, continuity, cursors, or success" in block
        assert "including continuations and empty results" in block
        assert "active `session_id` and matching scope" in block

    def test_generated_block_explains_continuation_and_feedback_recovery(self):
        block = _lifecycle_block("claude-code")
        assert "Account for every warning" in block
        assert "A continuation sends only\n`session_id`, `scope`, and `continue_from`" in block
        assert "incomplete_feedback" in block
        assert "feedback_status" in block
        assert "verification_status" in block
        assert "rejected/outstanding feedback before committing" in block
        assert "entries task-only" in block
        assert "coverage inside each item" in block
        assert "each batch item's `ok`/data" in block


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
        assert json.loads(backups[0].read_text(encoding="utf-8")) == {"original": True}

    def test_no_backup_when_file_missing(self, tmp_path):
        _write_json(tmp_path / "new.json", {"key": "val"})
        assert list(tmp_path.glob("new.json.bak.*")) == []

    def test_write_creates_parent_dirs(self, tmp_path):
        target = tmp_path / "a" / "b" / "cfg.json"
        _write_json(target, {"x": 1})
        assert target.exists()
        assert json.loads(target.read_text(encoding="utf-8")) == {"x": 1}

    def test_backup_file_direct(self, tmp_path):
        f = tmp_path / "myfile.txt"
        f.write_text("hello", encoding="utf-8")
        bak = _backup_file(f)
        assert bak is not None and bak.exists()
        assert bak.read_text(encoding="utf-8") == "hello"
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
        assert json.loads(backups[0].read_text(encoding="utf-8")) == {"v": 2}


class TestInjectBlockBackup:
    def test_backup_on_update(self, tmp_path):
        target = tmp_path / "CLAUDE.md"
        original = "<!-- slowave-lifecycle-start v1 -->\nold\n<!-- slowave-lifecycle-end v1 -->\n"
        target.write_text(original, encoding="utf-8")
        _inject_block(target, _lifecycle_block("claude-code"))
        backups = list(tmp_path.glob("CLAUDE.md.bak.*"))
        assert len(backups) == 1
        assert backups[0].read_text(encoding="utf-8") == original

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


class TestRemovalCommandSurface:
    def test_uninstall_dry_run_preserves_local_data(self, fake_home):
        result = CliRunner().invoke(cli, ["uninstall", "--dry-run"])

        assert result.exit_code == 0, result.output
        assert "Slowave uninstall" in result.output
        assert "Local data and database" not in result.output
        assert "No files were changed" in result.output

    def test_purge_dry_run_does_not_prompt(self, fake_home):
        result = CliRunner().invoke(cli, ["purge", "--dry-run"])

        assert result.exit_code == 0, result.output
        assert "Slowave purge" in result.output
        assert "Local data and database" in result.output
        assert "Continue?" not in result.output

    def test_cleanup_is_a_purge_alias(self, fake_home):
        result = CliRunner().invoke(cli, ["cleanup", "--dry-run"])

        assert result.exit_code == 0, result.output
        assert "Slowave purge" in result.output


class TestDetectedClients:
    def test_only_detected_clients_are_selected(self, fake_home):
        (fake_home / ".claude").mkdir()
        (fake_home / ".cline").mkdir()

        assert [spec.key for spec in _detected_clients("all")] == [
            "claude-code",
            "cline",
        ]

    def test_summary_contains_no_undetected_clients(self, fake_home):
        (fake_home / ".claude").mkdir()
        (fake_home / ".cline").mkdir()

        summary = _build_summary("all", worker=False, slowave_bin="slowave")

        assert {change.client for change in summary.changes} == {"Claude Code", "Cline"}

    def test_macos_worker_summary_formats_runtime_placeholders(self, fake_home, monkeypatch):
        monkeypatch.setattr(_setup_mod, "SYSTEM", "Darwin")

        summary = _build_summary("all", worker=True, slowave_bin="slowave")

        assert any(
            change.change_type.value == "worker_service" and change.client == "macOS"
            for change in summary.changes
        )

    def test_setup_output_omits_undetected_clients_and_telemetry(self, fake_home):
        (fake_home / ".claude").mkdir()

        result = CliRunner().invoke(setup_cmd, ["--dry-run", "--no-worker"])

        assert result.exit_code == 0, result.output
        assert "Claude Code" in result.output
        assert "Not installed" not in result.output
        for client in ("Claude Desktop", "Cline", "Cursor", "Windsurf", "OpenCode", "Codex"):
            assert client not in result.output
        assert "Lifecycle Version Telemetry" not in result.output


class TestCleanupRemoveLifecycleBlocks:
    def test_removes_block_from_clinerules(self, fake_home):
        target = fake_home / ".cline" / "rules" / "slowave.md"
        target.parent.mkdir(parents=True, exist_ok=True)
        block = _lifecycle_block("cline-tui")
        target.write_text(block + "\n# My Notes\n", encoding="utf-8")

        count = _cleanup_mod._remove_lifecycle_blocks(dry_run=False)

        assert count >= 1
        remaining = target.read_text(encoding="utf-8")
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
        assert not target.exists() or _MARKER_START not in target.read_text(encoding="utf-8")

    def test_dry_run_does_not_modify_files(self, fake_home):
        target = fake_home / ".cline" / "rules" / "slowave.md"
        target.parent.mkdir(parents=True, exist_ok=True)
        block = _lifecycle_block("cline-tui")
        original = block + "\n# Notes\n"
        target.write_text(original, encoding="utf-8")

        _cleanup_mod._remove_lifecycle_blocks(dry_run=True)

        assert target.read_text(encoding="utf-8") == original

    def test_no_op_on_file_without_slowave_content(self, fake_home):
        target = fake_home / ".cline" / "rules" / "slowave.md"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("# Regular rules\n", encoding="utf-8")

        count = _cleanup_mod._remove_lifecycle_blocks(dry_run=False)

        assert count == 0
        assert target.read_text(encoding="utf-8") == "# Regular rules\n"


class TestCleanupRemoveMcpConfigs:
    def test_removes_slowave_from_cursor_mcp(self, fake_home):
        cursor_dir = fake_home / ".cursor"
        cursor_dir.mkdir()
        cfg_path = cursor_dir / "mcp.json"
        cfg_path.write_text(
            json.dumps(
                {
                    "mcpServers": {
                        "slowave": {"command": "/usr/local/bin/slowave-mcp"},
                        "other": {},
                    }
                }
            ),
            encoding="utf-8",
        )

        count = _cleanup_mod._remove_mcp_configs(dry_run=False)

        assert count >= 1
        remaining = json.loads(cfg_path.read_text(encoding="utf-8"))
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

        assert cfg_path.read_text(encoding="utf-8") == original

    def test_no_op_when_no_mcp_files_exist(self, fake_home):
        count = _cleanup_mod._remove_mcp_configs(dry_run=False)
        assert count == 0

    def test_removes_slowave_mcp_from_codex_config(self, fake_home):
        codex_dir = fake_home / ".codex"
        codex_dir.mkdir()
        cfg_path = codex_dir / "config.toml"
        cfg_path.write_text(
            'model = "gpt-5.5"\n\n'
            "[mcp_servers.slowave]\n"
            'url = "http://127.0.0.1:8766/mcp"\n\n'
            "[mcp_servers.other]\n"
            'command = "npx"\n',
            encoding="utf-8",
        )

        count = _cleanup_mod._remove_mcp_configs(dry_run=False)

        assert count >= 1
        remaining = _read_toml(cfg_path)
        assert "slowave" not in remaining.get("mcp_servers", {})
        assert "other" in remaining["mcp_servers"]
        assert remaining["model"] == "gpt-5.5"

    def test_removes_legacy_hooks_without_an_mcp_entry(self, fake_home):
        codex_dir = fake_home / ".codex"
        codex_dir.mkdir()
        cfg_path = codex_dir / "config.toml"
        cfg_path.write_text(
            'model = "gpt-5.5"\n\n'
            "[[hooks.UserPromptSubmit]]\n"
            "[[hooks.UserPromptSubmit.hooks]]\n"
            'type = "command"\n'
            "command = \"echo 'SLOWAVE MANDATORY: activate'\"\n\n"
            "[[hooks.Stop]]\n"
            "[[hooks.Stop.hooks]]\n"
            'type = "command"\n'
            'command = "slowave hook codex-stop # SLOWAVE MANDATORY"\n\n'
            "[[hooks.Stop]]\n"
            "[[hooks.Stop.hooks]]\n"
            'type = "command"\n'
            'command = "echo keep-me"\n',
            encoding="utf-8",
        )

        count = _cleanup_mod._remove_mcp_configs(dry_run=False)

        assert count == 1
        remaining = _read_toml(cfg_path)
        assert remaining["hooks"]["UserPromptSubmit"] == []
        assert len(remaining["hooks"]["Stop"]) == 1
        assert remaining["hooks"]["Stop"][0]["hooks"][0]["command"] == "echo keep-me"

    def test_removes_legacy_claude_hooks_and_preserves_unrelated_hooks(self, fake_home):
        claude_dir = fake_home / ".claude"
        claude_dir.mkdir()
        cfg_path = claude_dir / "settings.json"
        cfg_path.write_text(
            json.dumps(
                {
                    "hooks": {
                        "UserPromptSubmit": [
                            {
                                "matcher": "",
                                "hooks": [
                                    {
                                        "type": "command",
                                        "command": "echo 'SLOWAVE MANDATORY: activate'",
                                    }
                                ],
                            }
                        ],
                        "Stop": [{"hooks": [{"type": "command", "command": "echo keep-me"}]}],
                    }
                }
            ),
            encoding="utf-8",
        )

        count = _cleanup_mod._remove_mcp_configs(dry_run=False)

        assert count == 1
        remaining = json.loads(cfg_path.read_text(encoding="utf-8"))
        assert remaining["hooks"]["UserPromptSubmit"] == []
        assert remaining["hooks"]["Stop"][0]["hooks"][0]["command"] == "echo keep-me"

    def test_dry_run_does_not_write_codex_config(self, fake_home):
        codex_dir = fake_home / ".codex"
        codex_dir.mkdir()
        cfg_path = codex_dir / "config.toml"
        original = '[mcp_servers.slowave]\nurl = "http://127.0.0.1:8766/mcp"\n'
        cfg_path.write_text(original, encoding="utf-8")

        _cleanup_mod._remove_mcp_configs(dry_run=True)

        assert cfg_path.read_text(encoding="utf-8") == original


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


# ===========================================================================
# _cline_mcp_settings_path — Cline MCP config file resolution
# ===========================================================================


class TestClineMcpSettingsPath:
    """Current Cline (>= 3.0.x / platform v24) reads MCP config only from
    ~/.cline/data/settings/cline_mcp_settings.json (resolveMcpSettingsPath()).
    A stale legacy ~/.cline/mcp.json (written by old slowave setup) must never
    shadow it, or Cline silently ignores the config while doctor reports
    \"configured\"."""

    def test_prefers_current_path_over_legacy_mcp_json(self, fake_home):
        # Simulate a machine where an older setup left ~/.cline/mcp.json behind.
        legacy = fake_home / ".cline" / "mcp.json"
        legacy.parent.mkdir(parents=True, exist_ok=True)
        legacy.write_text('{"mcpServers": {}}', encoding="utf-8")

        # Even though the legacy file exists, current Cline reads data/settings.
        got = _setup_mod._cline_mcp_settings_path()
        assert got == (fake_home / ".cline" / "data" / "settings" / "cline_mcp_settings.json")

    def test_defaults_to_current_path_when_no_cline_dir(self, fake_home):
        got = _setup_mod._cline_mcp_settings_path()
        assert got == (fake_home / ".cline" / "data" / "settings" / "cline_mcp_settings.json")

    def test_targets_current_path_when_cli_dir_exists(self, fake_home):
        (fake_home / ".cline" / "data").mkdir(parents=True, exist_ok=True)
        got = _setup_mod._cline_mcp_settings_path()
        assert got == (fake_home / ".cline" / "data" / "settings" / "cline_mcp_settings.json")


# ===========================================================================
# launchd EnvironmentVariables preservation across setup regeneration
# ===========================================================================


def _write_plist(path, environment: dict) -> None:
    """Write a minimal launchd plist with the given EnvironmentVariables."""
    import plistlib

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        plistlib.dumps(
            {"Label": "com.slowave.daemon", "EnvironmentVariables": environment},
            fmt=plistlib.FMT_XML,
        )
    )


class TestPreservedServiceEnvironment:
    """setup regenerates the launchd plists from a template whose environment
    holds only the managed runtime keys. The merge carries unmanaged keys from
    the existing plist file over the regeneration boundary, so manually pinned
    environment settings survive every setup run. Managed keys must always
    take the fresh template value; only unmanaged keys carry over."""

    def test_extra_keys_survive_and_managed_keys_take_template_values(self, fake_home, monkeypatch):
        monkeypatch.delenv("SLOWAVE_DB", raising=False)
        monkeypatch.setenv("SLOWAVE_MCP_HTTP_PORT", "8766")
        plist_path = fake_home / "Library" / "LaunchAgents" / "com.slowave.daemon.plist"
        _write_plist(
            plist_path,
            {
                "SLOWAVE_HOME": "/stale/old-root",
                "SLOWAVE_MCP_HTTP_PORT": "9999",
                "SLOWAVE_EXPERIMENT_PIN": "1",
            },
        )

        merged = _setup_mod._preserved_service_environment(plist_path)

        assert merged["SLOWAVE_EXPERIMENT_PIN"] == "1"
        # Managed keys are recomputed, never carried over from the old plist.
        assert merged["SLOWAVE_HOME"] != "/stale/old-root"
        assert merged["SLOWAVE_MCP_HTTP_PORT"] == "8766"

    def test_missing_or_malformed_plist_yields_template_environment(self, fake_home, monkeypatch):
        monkeypatch.delenv("SLOWAVE_DB", raising=False)
        template = _setup_mod._launchd_runtime_environment()
        missing = fake_home / "Library" / "LaunchAgents" / "absent.plist"
        malformed = fake_home / "Library" / "LaunchAgents" / "broken.plist"
        malformed.parent.mkdir(parents=True, exist_ok=True)
        malformed.write_bytes(b'<plist version="1.0"><dict>truncated')

        assert _setup_mod._preserved_service_environment(missing) == dict(
            _setup_mod._runtime_service_env()
        )
        assert _setup_mod._preserved_service_environment(malformed) == dict(
            _setup_mod._runtime_service_env()
        )
        assert _setup_mod._launchd_runtime_environment(preserve_from=malformed) == template

    def test_non_scalar_and_bool_values_are_skipped(self, fake_home, monkeypatch):
        import plistlib

        monkeypatch.delenv("SLOWAVE_DB", raising=False)
        monkeypatch.setenv("SLOWAVE_MCP_HTTP_PORT", "8766")
        plist_path = fake_home / "Library" / "LaunchAgents" / "com.slowave.daemon.plist"
        raw = plistlib.dumps(
            {
                "Label": "com.slowave.daemon",
                "EnvironmentVariables": {
                    "SLOWAVE_GOOD_PIN": "1",
                    "SLOWAVE_BOOL_PIN": True,
                    "SLOWAVE_LIST_PIN": ["a"],
                    "SLOWAVE_DICT_PIN": {"k": "v"},
                },
            },
            fmt=plistlib.FMT_XML,
        )
        plist_path.parent.mkdir(parents=True, exist_ok=True)
        plist_path.write_bytes(raw)

        merged = _setup_mod._preserved_service_environment(plist_path)

        assert merged["SLOWAVE_GOOD_PIN"] == "1"
        for bad in ("SLOWAVE_BOOL_PIN", "SLOWAVE_LIST_PIN", "SLOWAVE_DICT_PIN"):
            assert bad not in merged


class TestInstallDaemonMacosPreservesEnvPins:
    """Installer-level regression: regenerate the daemon/worker plists on a
    fake home while an experiment pin lives in the existing file; the pin
    must survive, and a second identical run must be a no-op."""

    def test_regenerated_daemon_plist_keeps_unmanaged_env_keys(self, fake_home, monkeypatch):
        import plistlib

        monkeypatch.delenv("SLOWAVE_DB", raising=False)
        monkeypatch.setenv("SLOWAVE_HOME", str(fake_home / "runtime"))
        monkeypatch.setenv("SLOWAVE_MCP_HTTP_PORT", "8766")
        monkeypatch.setattr(_setup_mod.subprocess, "run", lambda *args, **kwargs: None)
        plist_path = fake_home / "Library" / "LaunchAgents" / "com.slowave.daemon.plist"
        _write_plist(
            plist_path,
            {
                "SLOWAVE_HOME": "/stale/old-root",
                "SLOWAVE_MCP_HTTP_PORT": "9999",
                "SLOWAVE_EXPERIMENT_PIN": "1",
            },
        )

        written, changed = _setup_mod._install_daemon_macos("slowave")

        assert changed is True
        assert written == str(plist_path)
        regen = plistlib.loads(plist_path.read_bytes())
        env = regen["EnvironmentVariables"]
        assert env["SLOWAVE_EXPERIMENT_PIN"] == "1"
        assert env["SLOWAVE_HOME"] == str((fake_home / "runtime").resolve())
        assert env["SLOWAVE_MCP_HTTP_PORT"] == "8766"

    def test_regenerated_worker_plist_keeps_unmanaged_env_keys(self, fake_home, monkeypatch):
        import plistlib

        monkeypatch.delenv("SLOWAVE_DB", raising=False)
        monkeypatch.setenv("SLOWAVE_HOME", str(fake_home / "runtime"))
        monkeypatch.setenv("SLOWAVE_MCP_HTTP_PORT", "8766")
        monkeypatch.setattr(_setup_mod.subprocess, "run", lambda *args, **kwargs: None)
        plist_path = fake_home / "Library" / "LaunchAgents" / "com.slowave.worker.plist"
        _write_plist(plist_path, {"SLOWAVE_EXPERIMENT_PIN": "1"})

        written, changed = _setup_mod._install_worker_macos("slowave")

        assert changed is True
        assert written == str(plist_path)
        regen = plistlib.loads(plist_path.read_bytes())
        assert regen["EnvironmentVariables"]["SLOWAVE_EXPERIMENT_PIN"] == "1"

    def test_second_run_is_idempotent_with_preserved_keys(self, fake_home, monkeypatch):
        monkeypatch.delenv("SLOWAVE_DB", raising=False)
        monkeypatch.setenv("SLOWAVE_HOME", str(fake_home / "runtime"))
        monkeypatch.setenv("SLOWAVE_MCP_HTTP_PORT", "8766")
        monkeypatch.setattr(_setup_mod.subprocess, "run", lambda *args, **kwargs: None)
        plist_path = fake_home / "Library" / "LaunchAgents" / "com.slowave.daemon.plist"
        _write_plist(plist_path, {"SLOWAVE_EXPERIMENT_PIN": "1"})

        _setup_mod._install_daemon_macos("slowave")
        _, changed_again = _setup_mod._install_daemon_macos("slowave")

        assert changed_again is False


@pytest.mark.parametrize("client", ["cursor", "claude-desktop"])
def test_manual_clients_receive_current_pasteable_block_even_when_configured(
    fake_home, monkeypatch, client
):
    monkeypatch.setattr(_setup_mod, "SYSTEM", "Darwin")
    monkeypatch.delenv("SLOWAVE_DB", raising=False)
    monkeypatch.setenv("SLOWAVE_HOME", str(fake_home / "runtime"))
    monkeypatch.setattr(_setup_mod, "_find_slowave_binary", lambda: "/fake/slowave")
    spec = next(spec for spec in _setup_mod._clients() if spec.key == client)
    spec.mcp_path().parent.mkdir(parents=True, exist_ok=True)
    # Isolate the already-configured path: this must still print the current block.
    summary = _setup_mod.Summary()
    summary.add_manual_step(spec.manual_note)
    monkeypatch.setattr(_setup_mod, "_build_summary", lambda *args, **kwargs: summary)
    result = CliRunner().invoke(setup_cmd, ["--client", client, "--no-worker"])
    assert result.exit_code == 0, result.output
    assert _lifecycle_block(spec.lifecycle_agent) in result.output
    assert "Everything already configured" in result.output
    assert not spec.mcp_path().exists()


@pytest.mark.parametrize("client", ["claude-code", "cline", "windsurf", "opencode", "codex"])
def test_current_rules_replace_stale_installed_blocks_without_changing_user_rules(
    fake_home, monkeypatch, client
):
    from slowave.cli.setup import _lifecycle_block_up_to_date

    monkeypatch.setattr(_setup_mod, "SYSTEM", "Darwin")
    monkeypatch.setenv("CODEX_HOME", str(fake_home / ".codex"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(fake_home / ".config"))
    spec = next(spec for spec in _setup_mod._clients() if spec.key == client)
    spec.mcp_path().parent.mkdir(parents=True, exist_ok=True)
    target = spec.lifecycle_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        "# User rules before\n\n<!-- slowave-lifecycle-start v13 -->\nold rules\n"
        "<!-- slowave-lifecycle-end v13 -->\n\n# User rules after\n",
        encoding="utf-8",
    )
    summary = _build_summary(client, worker=False, slowave_bin="/fake/slowave")
    update = next(c for c in summary.changes if c.change_type.value == "lifecycle_block")
    assert update.status.value == "update"
    block = _lifecycle_block(spec.lifecycle_agent)
    assert _inject_block(target, block)
    installed = target.read_text(encoding="utf-8")
    assert _lifecycle_block_up_to_date(installed, block)
    assert installed.startswith("# User rules before\n\n")
    assert installed.endswith("\n# User rules after\n")
    assert "old rules" not in installed
    assert installed.count("<!-- slowave-lifecycle-start") == 1
    assert not _inject_block(target, block)
    summary = _build_summary(client, worker=False, slowave_bin="/fake/slowave")
    update = next(c for c in summary.changes if c.change_type.value == "lifecycle_block")
    assert update.status.value == "skip"


def test_upgrading_preserves_user_heading_immediately_after_end_marker(tmp_path):
    target = tmp_path / "AGENTS.md"
    target.write_text(
        "<!-- slowave-lifecycle-start v13 -->\nold\n"
        "<!-- slowave-lifecycle-end v13 -->\n# User rules\nKeep this.\n",
        encoding="utf-8",
    )
    block = _lifecycle_block("codex")
    assert _inject_block(target, block)
    expected = block + "\n# User rules\nKeep this.\n"
    assert target.read_text(encoding="utf-8") == expected
    assert not _inject_block(target, block)
    assert target.read_text(encoding="utf-8") == expected


def test_force_reapplies_matching_client_and_runs_verification(fake_home, monkeypatch):
    monkeypatch.setenv("SLOWAVE_HOME", str(fake_home / "runtime"))
    monkeypatch.delenv("SLOWAVE_DB", raising=False)
    monkeypatch.setattr(_setup_mod, "_find_slowave_binary", lambda: "/fake/slowave")
    spec = next(s for s in _setup_mod._clients() if s.key == "codex")
    monkeypatch.setattr(_setup_mod, "_detected_clients", lambda client: [spec])
    monkeypatch.setattr(_setup_mod, "_build_summary", lambda *a, **k: _setup_mod.Summary())
    cfg = spec.mcp_path()
    cfg.parent.mkdir(parents=True, exist_ok=True)
    cfg.write_text(
        '[unrelated]\nkeep = "yes"\n[mcp_servers.slowave]\nurl = "http://127.0.0.1:8766/mcp"\n',
        encoding="utf-8",
    )
    instructions = spec.lifecycle_path()
    instructions.write_text(_lifecycle_block("codex") + "\n\nUser rules\n", encoding="utf-8")
    before = instructions.read_text(encoding="utf-8")
    calls = []
    monkeypatch.setattr(_setup_mod.subprocess, "run", lambda args, **kw: calls.append(args))
    result = CliRunner().invoke(
        setup_cmd, ["--client", "codex", "--no-worker", "--force"], input="y\n"
    )
    assert result.exit_code == 0, result.output
    assert "Everything already configured" not in result.output
    assert instructions.read_text(encoding="utf-8") == before
    assert _read_toml(cfg)["unrelated"]["keep"] == "yes"
    assert list(instructions.parent.glob(instructions.name + ".bak.*"))
    assert any(args[-1] == "doctor" for args in calls)
    calls.clear()
    result = CliRunner().invoke(
        setup_cmd, ["--client", "codex", "--no-worker", "--force", "--dry-run"]
    )
    assert result.exit_code == 0, result.output
    assert "Would inject" in result.output
    assert instructions.read_text(encoding="utf-8") == before
    assert not calls


@pytest.mark.parametrize("kind", ["daemon", "worker", "backup"])
def test_force_reloads_matching_macos_service_preserving_environment(fake_home, monkeypatch, kind):
    import plistlib

    monkeypatch.setattr(_setup_mod.os, "getuid", lambda: 501, raising=False)
    monkeypatch.setenv("SLOWAVE_HOME", str(fake_home / "runtime"))
    monkeypatch.delenv("SLOWAVE_DB", raising=False)
    monkeypatch.setenv("SLOWAVE_MCP_HTTP_PORT", "8766")
    calls = []
    monkeypatch.setattr(_setup_mod.subprocess, "run", lambda args, **kw: calls.append(args))
    install = getattr(_setup_mod, f"_install_{kind}_macos")
    path, _ = install("/fake/slowave")
    p = _setup_mod.Path(path)
    d = plistlib.loads(p.read_bytes())
    d["EnvironmentVariables"]["SLOWAVE_CUSTOM_PIN"] = "preserve"
    p.write_bytes(plistlib.dumps(d))
    install("/fake/slowave")
    assert install("/fake/slowave")[1] is False
    calls.clear()
    assert install("/fake/slowave", force=True)[1] is True
    assert [c[1] for c in calls] == ["print", "bootout", "bootstrap"]
    assert (
        plistlib.loads(p.read_bytes())["EnvironmentVariables"]["SLOWAVE_CUSTOM_PIN"] == "preserve"
    )


@pytest.mark.parametrize("kind", ["daemon", "worker", "backup"])
def test_force_restarts_matching_linux_service(fake_home, monkeypatch, kind):
    monkeypatch.setenv("SLOWAVE_HOME", str(fake_home / "runtime"))
    monkeypatch.delenv("SLOWAVE_DB", raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(fake_home / ".config"))
    calls = []
    monkeypatch.setattr(_setup_mod.subprocess, "run", lambda args, **kw: calls.append(args))
    install = getattr(_setup_mod, f"_install_{kind}_linux")
    path, _ = install("/fake/slowave")
    assert install("/fake/slowave")[1] is False
    calls.clear()
    assert install("/fake/slowave", force=True)[1] is True
    target = f"slowave-{kind}" + (".timer" if kind == "backup" else "")
    assert ["systemctl", "--user", "restart", target] in calls
    if kind == "backup":
        service = _setup_mod.Path(path).with_suffix(".service").read_text(encoding="utf-8")
        assert f'SLOWAVE_HOME={fake_home / "runtime"}' in service


def test_force_reregisters_matching_windows_task(monkeypatch):
    from subprocess import CompletedProcess

    calls = []

    def run(args, **kwargs):
        calls.append(args)
        return CompletedProcess(
            args, 0, stdout=f"{_setup_mod._WINDOWS_TASK_MARKER}|slowave|worker", stderr=""
        )

    monkeypatch.setattr(_setup_mod.subprocess, "run", run)
    assert (
        _setup_mod._register_windows_task("SlowaveWorker", "slowave", "worker")[1]
        == "already up-to-date"
    )
    calls.clear()
    assert (
        _setup_mod._register_windows_task("SlowaveWorker", "slowave", "worker", force=True)[1]
        == "registered and started"
    )
    assert any("Register-ScheduledTask" in args[-1] for args in calls)


def test_windows_replacement_stops_before_registration(monkeypatch):
    from subprocess import CompletedProcess

    calls = []

    def run(args, **kwargs):
        calls.append(args)
        return CompletedProcess(args, 0, stdout="", stderr="")

    monkeypatch.setattr(_setup_mod.subprocess, "run", run)
    assert _setup_mod._register_windows_task("SlowaveWorker", "slowave", "worker", force=True)[0]
    script = calls[-1][-1]
    assert script.index("Disable-ScheduledTask") < script.index("Stop-ScheduledTask")
    assert script.index("Stop-ScheduledTask") < script.index("Register-ScheduledTask")
    assert script.index("Register-ScheduledTask") < script.index("Start-ScheduledTask")
    assert "Task did not stop" in script


def test_setup_reapplies_services_without_force(fake_home, monkeypatch):
    from click.testing import CliRunner

    calls = []
    monkeypatch.setattr(_setup_mod, "SYSTEM", "Linux")
    monkeypatch.setattr(_setup_mod, "_verify_daemon_health", lambda *a, **kw: True)
    for kind in ("daemon", "worker", "backup"):

        def install(binary, *, force=False, kind=kind):
            calls.append((kind, force))
            return "/fake/service", True

        monkeypatch.setattr(_setup_mod, f"_install_{kind}_linux", install)
    monkeypatch.setattr(_setup_mod.subprocess, "run", lambda *a, **kw: None)
    result = CliRunner().invoke(setup_cmd, ["--client", "codex"], input="y\n")
    assert result.exit_code == 0, result.output
    assert calls == [("daemon", True), ("worker", True), ("backup", True)]


def test_setup_reenables_windows_backup_without_running_it(monkeypatch):
    from subprocess import CompletedProcess

    calls = []

    def run(args, **kwargs):
        calls.append(args)
        return CompletedProcess(args, 0, stdout="", stderr="")

    monkeypatch.setattr(_setup_mod.subprocess, "run", run)
    _setup_mod._install_backup_windows("slowave", force=True)
    script = calls[-1][-1]
    assert "Enable-ScheduledTask" in script
    assert "Start-ScheduledTask" not in script


def test_setup_binary_prefers_current_environment_over_path(tmp_path, monkeypatch):
    import sysconfig

    directory = tmp_path / "current environment" / "Scripts"
    directory.mkdir(parents=True)
    monkeypatch.setattr(_setup_mod, "SYSTEM", "Windows")
    binary = directory / "slowave.exe"
    binary.touch()
    monkeypatch.setattr(sysconfig, "get_path", lambda *a, **kw: str(directory))
    monkeypatch.setattr(_setup_mod.shutil, "which", lambda *a: "/wrong/environment/slowave")
    assert _setup_mod._find_slowave_binary() == str(binary.resolve())


def test_linux_service_quotes_executable_path_with_spaces(fake_home, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(fake_home / "config"))
    monkeypatch.setattr(_setup_mod.subprocess, "run", lambda *a, **kw: None)
    path, _ = _setup_mod._install_daemon_linux("/path with spaces/bin/slowave", force=True)
    assert 'ExecStart="/path with spaces/bin/slowave" serve start' in _setup_mod.Path(
        path
    ).read_text(encoding="utf-8")


def test_setup_does_not_claim_success_if_doctor_fails(fake_home, monkeypatch):
    import subprocess

    monkeypatch.setattr(_setup_mod, "_find_slowave_binary", lambda: "/fake/slowave")

    def run(args, **kwargs):
        raise subprocess.CalledProcessError(1, args)

    monkeypatch.setattr(_setup_mod.subprocess, "run", run)
    result = CliRunner().invoke(
        setup_cmd, ["--client", "codex", "--no-worker", "--force"], input="y\n"
    )
    assert result.exit_code != 0
    assert "verification failed" in result.output
    assert "Setup complete" not in result.output


def test_launchd_reapply_waits_before_bootstrap(monkeypatch, tmp_path):
    import plistlib
    from subprocess import CompletedProcess

    from slowave.cli import services

    calls = []
    monkeypatch.setattr(_setup_mod.os, "getuid", lambda: 501, raising=False)

    def run(args, **kwargs):
        calls.append(args[1])
        return CompletedProcess(args, 0, "pid = 12345\n", "")

    monkeypatch.setattr(_setup_mod.subprocess, "run", run)
    monkeypatch.setattr(services, "wait_for_process_exit", lambda pid: calls.append(pid))
    content = plistlib.dumps({"Label": "com.slowave.daemon"}).decode()
    _setup_mod._apply_launchd_service(tmp_path / "daemon.plist", content, force=True)
    assert calls == ["print", "bootout", 12345, "bootstrap"]
