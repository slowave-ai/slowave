"""slowave setup — one-command post-install wiring.

Automates:
  1. Patching MCP client configs (Claude Code, Claude Desktop, Cline, Cursor, Windsurf)
     to point to the Slowave HTTP MCP daemon at http://127.0.0.1:8766/mcp.
  2. Injecting lifecycle instructions (CLAUDE.md, .clinerules, etc.) into client
     rule files, and UserPromptSubmit/Stop hooks into ~/.claude/settings.json.
  3. Installing the background worker as a user service
     (launchd on macOS, systemd on Linux, Task Scheduler on Windows).
  4. Running `slowave doctor` to verify the result.

All steps are idempotent — re-running is always safe.
The HTTP MCP daemon and background worker are installed as system services and started automatically.
"""

from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import click

from slowave.cli.output import safe_emoji
from slowave.lifecycle import LIFECYCLE_VERSION

# ---------------------------------------------------------------------------
# Change tracking for summary
# ---------------------------------------------------------------------------


class ChangeType(str, Enum):
    """Types of changes that can be tracked."""

    MCP_CONFIG = "mcp_config"
    LIFECYCLE_BLOCK = "lifecycle_block"
    HOOKS = "hooks"
    WORKER_SERVICE = "worker_service"
    MANUAL_STEP = "manual_step"


class ChangeStatus(str, Enum):
    """Status of a change."""

    NEW = "new"
    UPDATE = "update"
    SKIP = "skip"  # Already present, no change needed


@dataclass
class Change:
    """Represents a single change to be applied."""

    change_type: ChangeType
    client: str  # e.g., "claude-code", "claude-desktop", etc.
    status: ChangeStatus
    path: str  # File path or service name
    description: str  # Human-readable description
    details: dict[str, Any] | None = None  # Extra context


class Summary:
    """Collects and formats changes for display."""

    def __init__(self):
        self.changes: list[Change] = []
        self.binaries: dict[str, str] = {}  # name -> path
        self.manual_steps: list[str] = []

    def add_binary(self, name: str, path: str) -> None:
        """Record a binary location."""
        self.binaries[name] = path

    def add_change(self, change: Change) -> None:
        """Add a change to the summary."""
        self.changes.append(change)

    def add_manual_step(self, step: str) -> None:
        """Add a manual step required after setup."""
        self.manual_steps.append(step)

    def has_changes(self) -> bool:
        """Return True if any change has status other than SKIP."""
        return any(c.status != ChangeStatus.SKIP for c in self.changes)

    def _group_changes(self) -> dict[ChangeType, list[Change]]:
        """Group changes by type."""
        grouped: dict[ChangeType, list[Change]] = {}
        for change_type in ChangeType:
            grouped[change_type] = [c for c in self.changes if c.change_type == change_type]
        return grouped

    def format(self) -> str:
        """Format summary as a human-readable string."""
        lines: list[str] = []

        # Header
        lines.append(click.style("\n" + "━" * 70, fg="cyan"))
        lines.append(click.style("SUMMARY: Changes to be applied", bold=True, fg="cyan"))
        lines.append(click.style("━" * 70, fg="cyan"))
        lines.append("")

        # Binaries
        if self.binaries:
            lines.append(click.style(f"{safe_emoji('📦', '[package]')} Binaries", bold=True))
            for name, path in self.binaries.items():
                lines.append(f"  ✓ {name}: {path}")
            lines.append("")

        # Group by change type
        grouped = self._group_changes()

        # MCP Configs
        if grouped[ChangeType.MCP_CONFIG]:
            configs = grouped[ChangeType.MCP_CONFIG]
            lines.append(
                click.style(
                    f"{safe_emoji('🔌', '[plug]')} MCP Configurations ({len(configs)} file{'s' if len(configs) != 1 else ''})",
                    bold=True,
                )
            )
            for change in configs:
                status_label = f"({change.status.value.upper()})"
                lines.append(
                    f"  ✓ {change.client} → {change.path} {click.style(status_label, fg='bright_black')}"
                )
            lines.append("")

        # Lifecycle Blocks
        if grouped[ChangeType.LIFECYCLE_BLOCK]:
            blocks = grouped[ChangeType.LIFECYCLE_BLOCK]
            lines.append(
                click.style(
                    f"{safe_emoji('📝', '[doc]')} Lifecycle Blocks ({len(blocks)} file{'s' if len(blocks) != 1 else ''})",
                    bold=True,
                )
            )
            for change in blocks:
                status_label = f"({change.status.value.upper()})"
                lines.append(
                    f"  ✓ {change.client} → {change.path} {click.style(status_label, fg='bright_black')}"
                )
            lines.append("")

        # Hooks
        if grouped[ChangeType.HOOKS]:
            hooks = grouped[ChangeType.HOOKS]
            lines.append(click.style(f"{safe_emoji('🔐', '[lock]')} Lifecycle Hooks", bold=True))
            for change in hooks:
                status_label = f"({change.status.value.upper()})"
                lines.append(
                    f"  ✓ {change.description} {click.style(status_label, fg='bright_black')}"
                )
            lines.append("")

        # Worker Service
        if grouped[ChangeType.WORKER_SERVICE]:
            services = grouped[ChangeType.WORKER_SERVICE]
            lines.append(
                click.style(f"{safe_emoji('⚙️', '[gear]')}  Background Worker Service", bold=True)
            )
            for change in services:
                status_label = f"({change.status.value.upper()})"
                lines.append(
                    f"  ✓ {change.description} → {change.path} {click.style(status_label, fg='bright_black')}"
                )
            lines.append("")

        # Manual Steps
        if self.manual_steps:
            lines.append(
                click.style(
                    f"{safe_emoji('⚠️', '!!')}  Manual Steps Required ({len(self.manual_steps)})",
                    bold=True,
                )
            )
            for step in self.manual_steps:
                lines.append(f"  {safe_emoji('⚠', '! ')} {step}")
            lines.append("")

        lines.append(click.style("━" * 70, fg="cyan"))
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Platform
# ---------------------------------------------------------------------------

SYSTEM = platform.system()  # "Darwin", "Linux", "Windows"


def _home() -> Path:
    return Path.home()


def _setup_sentinel_path() -> Path:
    return _home() / ".slowave" / ".setup_done"


def is_setup_done() -> bool:
    return _setup_sentinel_path().exists()


def mark_setup_done() -> None:
    sentinel = _setup_sentinel_path()
    sentinel.parent.mkdir(parents=True, exist_ok=True)
    sentinel.touch()


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def _claude_settings_path() -> Path:
    """~/.claude/settings.json — hooks, permissions, env only (NOT mcpServers)."""
    return _home() / ".claude" / "settings.json"


def _claude_json_path() -> Path:
    """~/.claude.json — where Claude Code stores MCP server configs (user scope)."""
    return _home() / ".claude.json"


def _claude_md_path() -> Path:
    return _home() / ".claude" / "CLAUDE.md"


def _clinerules_path() -> Path:
    """Global lifecycle rules file for Cline TUI.

    Cline TUI reads rules from:
      1. <cwd>/.clinerules           (project-local)
      2. ~/.cline/rules/             (global rules directory — .md files)
      3. ~/Documents/Cline/Rules/    (global rules directory — .md files)

    ~/.clinerules is only read when it happens to be inside the cwd
    (i.e. when the user runs cline from their home directory).  It is NOT
    a globally-read path for Cline TUI.

    We write to ~/.cline/rules/slowave.md so the lifecycle block is picked
    up regardless of which project directory the user is in.
    """
    return _home() / ".cline" / "rules" / "slowave.md"


def _claude_desktop_config_path() -> Path:
    if SYSTEM == "Darwin":
        return _home() / "Library" / "Application Support" / "Claude" / "claude_desktop_config.json"
    elif SYSTEM == "Windows":
        appdata = os.environ.get("APPDATA", str(_home() / "AppData" / "Roaming"))
        return Path(appdata) / "Claude" / "claude_desktop_config.json"
    else:
        xdg = os.environ.get("XDG_CONFIG_HOME", str(_home() / ".config"))
        return Path(xdg) / "Claude" / "claude_desktop_config.json"


def _cursor_mcp_config_path() -> Path:
    """Cursor native MCP config — ~/.cursor/mcp.json (all platforms)."""
    return _home() / ".cursor" / "mcp.json"


def _cursor_rules_path() -> Path:
    """Cursor global rules file — ~/.cursor/rules (all platforms)."""
    return _home() / ".cursor" / "rules"


def _windsurf_mcp_config_path() -> Path:
    """Windsurf/Devin Desktop MCP config — ~/.codeium/windsurf/mcp_config.json (all platforms)."""
    return _home() / ".codeium" / "windsurf" / "mcp_config.json"


def _windsurf_global_rules_path() -> Path:
    """Windsurf global rules — ~/.codeium/windsurf/memories/global_rules.md (all platforms).

    This file is always-on: injected into every Cascade conversation.
    It is fully injectable programmatically (no manual step required).
    """
    return _home() / ".codeium" / "windsurf" / "memories" / "global_rules.md"


def _cline_mcp_settings_path() -> Path:
    """Return the active Cline MCP settings path (best-effort detection).

    Priority:
      1. ~/.cline/mcp.json             — Cline CLI v3 (preferred)
      2. ~/.cline/data/settings/cline_mcp_settings.json  — older CLI / TUI
      3. VS Code globalStorage path    — VS Code extension
      4. Cursor globalStorage path     — Cursor extension
    """
    if SYSTEM == "Darwin":
        candidates = [
            _home() / ".cline" / "mcp.json",  # Cline CLI v3
            _home() / ".cline/data/settings/cline_mcp_settings.json",  # older TUI
            _home() / "Library/Application Support/Code/User/globalStorage"
            "/saoudrizwan.claude-dev/settings/cline_mcp_settings.json",
            _home() / "Library/Application Support/Cursor/User/globalStorage"
            "/saoudrizwan.claude-dev/settings/cline_mcp_settings.json",
        ]
    elif SYSTEM == "Windows":
        appdata = os.environ.get("APPDATA", str(_home() / "AppData" / "Roaming"))
        candidates = [
            _home() / ".cline" / "mcp.json",  # Cline CLI v3
            _home() / ".cline/data/settings/cline_mcp_settings.json",  # older TUI
            Path(appdata) / "Code/User/globalStorage"
            "/saoudrizwan.claude-dev/settings/cline_mcp_settings.json",
            Path(appdata) / "Cursor/User/globalStorage"
            "/saoudrizwan.claude-dev/settings/cline_mcp_settings.json",
        ]
    else:
        xdg = os.environ.get("XDG_CONFIG_HOME", str(_home() / ".config"))
        candidates = [
            _home() / ".cline" / "mcp.json",  # Cline CLI v3
            _home() / ".cline/data/settings/cline_mcp_settings.json",  # older TUI
            Path(xdg) / "Code/User/globalStorage"
            "/saoudrizwan.claude-dev/settings/cline_mcp_settings.json",
            Path(xdg) / "Cursor/User/globalStorage"
            "/saoudrizwan.claude-dev/settings/cline_mcp_settings.json",
        ]
    for p in candidates:
        if p.exists():
            return p
    # Default to CLI v3 path (creates it if needed)
    return _home() / ".cline" / "mcp.json"


def _opencode_config_dir() -> Path:
    """OpenCode global config directory — respects XDG_CONFIG_HOME."""
    xdg = os.environ.get("XDG_CONFIG_HOME", str(_home() / ".config"))
    return Path(xdg) / "opencode"


def _opencode_config_path() -> Path:
    """OpenCode global config file — ~/.config/opencode/opencode.json (or .jsonc).

    OpenCode supports both JSON and JSONC formats.  If a .jsonc already
    exists, prefer it so existing user config is preserved.  Otherwise
    default to .json.
    """
    jsonc = _opencode_config_dir() / "opencode.jsonc"
    if jsonc.exists():
        return jsonc
    return _opencode_config_dir() / "opencode.json"


def _opencode_instructions_path() -> Path:
    """Slowave-owned lifecycle instruction file for OpenCode."""
    return _opencode_config_dir() / "slowave-instructions.md"


def _codex_home() -> Path:
    """Codex home directory — respects CODEX_HOME."""
    codex_home = os.environ.get("CODEX_HOME")
    return Path(codex_home) if codex_home else _home() / ".codex"


def _codex_config_path() -> Path:
    """Codex global config — ~/.codex/config.toml (or $CODEX_HOME/config.toml).

    Both the MCP server registry (``mcp_servers``) and lifecycle enforcement
    hooks (``hooks``) live in this single TOML file — unlike Claude Code,
    which splits them across ``~/.claude.json`` and ``~/.claude/settings.json``.
    """
    return _codex_home() / "config.toml"


def _codex_agents_md_path() -> Path:
    """Codex global instructions file — ~/.codex/AGENTS.md.

    Directly analogous to Claude Code's CLAUDE.md: Codex reads this file (or
    AGENTS.override.md, if present, instead) once per session and folds its
    content into the first turn. Uses the same marker-based _inject_block()
    mechanism as every other auto-injected client.
    """
    return _codex_home() / "AGENTS.md"


# ---------------------------------------------------------------------------
# ClientSpec — single source of truth for every supported client
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ClientSpec:
    """Describes one AI client that Slowave can configure.

    All setup, cleanup, summary-preview, and backup-cleanup code iterates
    ``_clients()`` so that adding a new client is a one-line change here.
    Adding a new enforcement mechanism to an existing client is also a
    one-field change: set ``hooks_config_path`` and ``hooks_patch_fn``.

    Fields
    ------
    key         CLI ``--client`` value, e.g. ``"claude-code"``
    label       Human-readable display name

    mcp_path    Callable[[], Path] — the JSON file where mcpServers lives.

    lifecycle_path
                Callable[[], Path] | None — the markdown/rules file to
                auto-inject the lifecycle block into.
                None means injection is manual (see manual_note).
    lifecycle_agent
                Agent-name token passed to ``_lifecycle_block()``.
    manual_lifecycle
                True when lifecycle instructions must be added manually
                (e.g. Claude Desktop custom-instructions, Cursor rules).
    manual_note Human-readable guidance printed after setup for manual steps.

    hooks_config_path
                Callable[[], Path] | None — config file that receives
                enforcement hooks (e.g. ``~/.claude/settings.json``).
                None = this client has no scriptable enforcement mechanism.
    hooks_patch_fn
                Callable[(dict, str), tuple[dict, bool]] | None —
                function that applies/updates enforcement hooks in a config
                dict.  Signature: ``fn(config) -> (new_config, changed)``.
                Must be None when hooks_config_path is None.
    hooks_cleanup_fn
                Callable[(dict), tuple[dict, bool]] | None —
                function that removes Slowave enforcement hooks from a
                config dict.  Signature: ``fn(config) -> (new_config, changed)``.

    require_dir_exists
                When True, skip MCP patching silently if the config file's
                parent directory doesn't exist (client probably not installed).
    restart_note
                Short string shown at the end of setup reminding the user
                what to restart, e.g. ``"Restart Claude Code"``.
    """

    key: str
    label: str
    mcp_path: Any  # Callable[[], Path]
    lifecycle_path: Any  # Callable[[], Path] | None
    lifecycle_agent: str
    manual_lifecycle: bool = False
    manual_note: str = ""
    hooks_config_path: Any = None  # Callable[[], Path] | None
    hooks_patch_fn: Any = None  # Callable[(dict), (dict, bool)] | None
    hooks_cleanup_fn: Any = None  # Callable[(dict), (dict, bool)] | None
    require_dir_exists: bool = False
    restart_note: str = ""


def _clients() -> list[ClientSpec]:
    """Return the list of all supported client specs for the current platform.

    This is the **single definition** of what clients exist.  Setup, cleanup,
    dry-run preview, backup-dir derivation, and docs all derive from here.
    """
    return [
        ClientSpec(
            key="claude-code",
            label="Claude Code",
            mcp_path=_claude_json_path,
            lifecycle_path=_claude_md_path,
            lifecycle_agent="claude-code",
            # Enforcement: UserPromptSubmit + Stop hooks in settings.json
            hooks_config_path=_claude_settings_path,
            hooks_patch_fn=_patch_claude_code_hooks,
            hooks_cleanup_fn=_remove_claude_code_hooks,
            require_dir_exists=True,
            restart_note="Restart Claude Code to apply changes.",
        ),
        ClientSpec(
            key="claude-desktop",
            label="Claude Desktop",
            mcp_path=_claude_desktop_config_path,
            lifecycle_path=None,
            lifecycle_agent="claude-desktop",
            manual_lifecycle=True,
            manual_note=(
                "Claude Desktop: add Slowave lifecycle block to Custom Instructions.\n"
                "     Settings → General → Instructions for Claude\n"
                "     https://github.com/mrsalty/slowave/blob/main/integrations/claude-desktop/README.md"
            ),
            require_dir_exists=True,
            # No scriptable enforcement: Custom Instructions field is server-side.
            restart_note="Restart Claude Desktop to apply changes.",
        ),
        ClientSpec(
            key="cline",
            label="Cline",
            mcp_path=_cline_mcp_settings_path,
            lifecycle_path=_clinerules_path,
            lifecycle_agent="cline-tui",
            require_dir_exists=True,
            # No enforcement hooks yet — lifecycle relies on .clinerules instructions.
            # When Cline adds a hook/trigger surface, add hooks_config_path + hooks_patch_fn here.
            restart_note="Reload Cline (or restart VS Code / Cursor) to apply changes.",
        ),
        ClientSpec(
            key="cursor",
            label="Cursor",
            mcp_path=_cursor_mcp_config_path,
            lifecycle_path=None,
            lifecycle_agent="cursor",
            manual_lifecycle=True,
            manual_note=(
                "Cursor: add Slowave lifecycle block to Rules for AI.\n"
                "     Settings → Rules for AI  (or add a .cursorrules file at repo root)\n"
                "     https://github.com/mrsalty/slowave/blob/main/integrations/cursor/README.md"
            ),
            require_dir_exists=True,
            # No scriptable enforcement: Rules for AI is manual.
            restart_note="Restart Cursor to apply changes.",
        ),
        ClientSpec(
            key="windsurf",
            label="Windsurf",
            mcp_path=_windsurf_mcp_config_path,
            lifecycle_path=_windsurf_global_rules_path,
            lifecycle_agent="windsurf",
            require_dir_exists=True,
            # No enforcement hooks yet — lifecycle relies on global_rules.md instructions.
            # When Windsurf adds a hook surface, add hooks_config_path + hooks_patch_fn here.
            restart_note="Restart Windsurf to apply changes.",
        ),
        ClientSpec(
            key="opencode",
            label="OpenCode",
            mcp_path=_opencode_config_path,
            lifecycle_path=_opencode_instructions_path,
            lifecycle_agent="opencode",
            require_dir_exists=True,
            restart_note="Restart OpenCode to apply changes.",
        ),
        ClientSpec(
            # "Codex", not "Codex CLI": ~/.codex/config.toml is shared by the CLI, the
            # Codex Desktop app (in ChatGPT), and the IDE extension — one integration
            # covers all three, so the label doesn't imply CLI-only.
            key="codex",
            label="Codex",
            mcp_path=_codex_config_path,
            lifecycle_path=_codex_agents_md_path,
            lifecycle_agent="codex",
            # Enforcement: UserPromptSubmit + Stop hooks, same TOML file as MCP config.
            hooks_config_path=_codex_config_path,
            hooks_patch_fn=_patch_codex_hooks,
            hooks_cleanup_fn=_remove_codex_hooks,
            require_dir_exists=True,
            restart_note="Restart Codex (CLI, Desktop, or IDE extension) to apply changes.",
        ),
    ]


def _clients_for(client_arg: str) -> list[ClientSpec]:
    """Return the subset of clients selected by the --client flag."""
    if client_arg == "all":
        return _clients()
    return [c for c in _clients() if c.key == client_arg]


def _all_mcp_paths() -> list[Path]:
    """All MCP config file paths across all clients (for backup cleanup)."""
    return [c.mcp_path() for c in _clients()]


def _all_lifecycle_paths() -> list[Path]:
    """All lifecycle-block file paths across all clients that support auto-injection."""
    return [c.lifecycle_path() for c in _clients() if c.lifecycle_path is not None]


# ---------------------------------------------------------------------------
# MCP binary detection
# ---------------------------------------------------------------------------


def _add_exe_if_windows(binary_name: str) -> str:
    """Add .exe extension to binary name on Windows if not already present.

    Args:
        binary_name: The base binary name (e.g., 'slowave', 'slowave-mcp')

    Returns:
        The binary name with .exe extension on Windows, unchanged on other platforms.
    """
    if SYSTEM == "Windows" and not binary_name.lower().endswith(".exe"):
        return f"{binary_name}.exe"
    return binary_name


def _find_mcp_binary(slowave_bin: str) -> str:
    """Derive the slowave-mcp binary path from the slowave binary path.

    Handles Windows .exe extensions correctly.
    """
    return str(Path(slowave_bin).parent / _add_exe_if_windows("slowave-mcp"))


def _find_slowave_binary() -> str:
    """Return the absolute path to the `slowave` CLI binary."""
    found = shutil.which("slowave")
    if found:
        return str(Path(found).resolve())
    # Common install locations
    candidates: list[Path] = [
        _home() / ".local" / "bin" / _add_exe_if_windows("slowave"),
        _home() / ".local" / "pipx" / "venvs" / "slowave" / "bin" / _add_exe_if_windows("slowave"),
        _home()
        / ".local"
        / "pipx"
        / "venvs"
        / "slowave"
        / "Scripts"
        / _add_exe_if_windows("slowave"),  # Windows pipx
        Path("/opt/homebrew/bin/slowave"),  # macOS Homebrew
        Path("/usr/local/bin/slowave"),  # Linux/Unix
    ]
    # On Windows, add common AppData paths
    if SYSTEM == "Windows":
        appdata_local = os.environ.get("LOCALAPPDATA", str(_home() / "AppData" / "Local"))
        candidates.extend(
            [
                Path(appdata_local) / "Programs" / "Python" / "Scripts" / "slowave.exe",
            ]
        )
    for c in candidates:
        if c.exists():
            return str(c)
    return "slowave"  # fallback – rely on PATH


def _backup_file(path: Path) -> Path | None:
    """Create a timestamped backup of *path* before overwriting it.

    Only one backup is kept per file: any older ``<name>.bak.*`` siblings
    are removed before the new backup is written, so re-running setup never
    accumulates stale copies.

    Returns the backup path if a backup was created, or None if the file
    did not exist (nothing to back up).  The backup sits next to the
    original:  ``<name>.bak.<YYYYMMDD_HHMMSS>``
    """
    if not path.exists():
        return None
    # Remove any previous backups for this file so we keep exactly one.
    for old in path.parent.glob(f"{path.name}.bak.*"):
        try:
            old.unlink()
        except OSError:
            pass
    import datetime

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = path.with_name(f"{path.name}.bak.{ts}")
    shutil.copy2(path, backup)
    return backup


def _read_json(path: Path) -> dict[str, Any]:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            click.echo(
                click.style(
                    f"  ✗  {path} contains invalid JSON and cannot be patched safely.\n"
                    f"     Fix the file first, then re-run slowave setup.\n"
                    f"     Error: {exc}",
                    fg="red",
                )
            )
            sys.exit(1)
    return {}


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    bak = _backup_file(path)
    if bak:
        click.echo(click.style(f"  ↩  backup → {bak}", fg="cyan"))
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _read_toml(path: Path) -> Any:
    """Read a TOML file (Codex's config.toml) as a tomlkit document.

    Uses tomlkit rather than stdlib tomllib because tomlkit round-trips
    comments and formatting on write — a plain parse-then-dump would silently
    strip every comment in the user's config.toml. Returns an empty tomlkit
    document if the file does not exist.
    """
    import tomlkit

    if path.exists():
        try:
            return tomlkit.parse(path.read_text(encoding="utf-8"))
        except tomlkit.exceptions.TOMLKitError as exc:
            click.echo(
                click.style(
                    f"  ✗  {path} contains invalid TOML and cannot be patched safely.\n"
                    f"     Fix the file first, then re-run slowave setup.\n"
                    f"     Error: {exc}",
                    fg="red",
                )
            )
            sys.exit(1)
    return tomlkit.document()


def _write_toml(path: Path, data: Any) -> None:
    import tomlkit

    path.parent.mkdir(parents=True, exist_ok=True)
    bak = _backup_file(path)
    if bak:
        click.echo(click.style(f"  ↩  backup → {bak}", fg="cyan"))
    path.write_text(tomlkit.dumps(data), encoding="utf-8")


def _patch_mcp_servers(
    config: dict[str, Any],
    url: str = "http://127.0.0.1:8766/mcp",
    *,
    include_type: bool = False,
    use_sse: bool = False,
) -> tuple[dict[str, Any], bool]:
    """Patch mcpServers in a config dict to use the Slowave HTTP daemon.

    Most clients (Claude Desktop, Cline, Cursor, Windsurf) use url-only: {url: <url>}.
    Claude Code requires {type: http, url: <url>} (MCP Streamable HTTP transport).
    Claude Desktop does NOT support type: http — url-only only.
    Pass include_type=True only for Claude Code.
    Migrates legacy stdio entries (command/type:stdio) transparently.
    Returns (updated_config, changed).
    """
    servers = config.setdefault("mcpServers", {})
    existing = servers.get("slowave", {})
    effective_url = url.replace("/mcp", "/sse") if use_sse else url
    want = {"type": "http", "url": effective_url} if include_type else {"url": effective_url}
    # Already correct (exact match)
    if existing == want:
        return config, False
    # url-only entry exists and we do not need type — already fine
    if (
        not include_type
        and isinstance(existing, dict)
        and existing.get("url") == effective_url
        and "command" not in existing
    ):
        # Strip any stale type field written by a previous version
        if list(existing.keys()) != ["url"] or existing.get("type"):
            servers["slowave"] = want
            return config, True
        return config, False
    # Anything else (stdio, missing, different url) — overwrite
    servers["slowave"] = want
    return config, True


def _patch_mcp_servers_stdio(
    config: dict[str, Any],
    command: str,
) -> tuple[dict[str, Any], bool]:
    """Patch mcpServers in a config dict to use the Slowave stdio server.

    Claude Desktop uses the stdio (command-based) MCP transport — it does NOT
    support the url-only or type:http formats used by Claude Code / Cline.
    The correct entry is: {"command": "<absolute-path-to-slowave-mcp>"}
    Returns (updated_config, changed).
    """
    servers = config.setdefault("mcpServers", {})
    existing = servers.get("slowave", {})
    want = {"command": command}
    if existing == want:
        return config, False
    servers["slowave"] = want
    return config, True


def _patch_opencode_mcp(
    config: dict[str, Any],
    url: str = "http://127.0.0.1:8766/mcp",
) -> tuple[dict[str, Any], bool]:
    """Patch MCP config for OpenCode (uses ``mcp`` key, ``"type": "remote"``).

    OpenCode MCP entry shape:
        {"type": "remote", "url": "http://127.0.0.1:8766/mcp", "enabled": true}

    Returns (updated_config, changed).
    """
    mcp = config.setdefault("mcp", {})
    existing = mcp.get("slowave", {})
    want = {"type": "remote", "url": url, "enabled": True}
    if existing == want:
        return config, False
    mcp["slowave"] = want
    return config, True


def _patch_opencode_instructions(
    config: dict[str, Any],
    instructions_path: str,
) -> tuple[dict[str, Any], bool]:
    """Register a Slowave instruction file in OpenCode's ``instructions`` array.

    Appends the absolute path to the instructions array if not already present.
    Returns (updated_config, changed).
    """
    instructions: list[str] = config.setdefault("instructions", [])
    if instructions_path in instructions:
        return config, False
    # Idempotent: only add if the path is not already present
    instructions.append(instructions_path)
    return config, True


def _patch_codex_mcp(
    config: Any,
    url: str = "http://127.0.0.1:8766/mcp",
) -> tuple[Any, bool]:
    """Patch MCP config for Codex (uses ``mcp_servers`` key, Streamable HTTP).

    Codex natively supports Streamable HTTP MCP servers — no stdio wrapper,
    no auth fields needed for an unauthenticated local server:

        [mcp_servers.slowave]
        url = "http://127.0.0.1:8766/mcp"

    Returns (updated_config, changed).
    """
    servers = config.setdefault("mcp_servers", {})
    existing = servers.get("slowave")
    if isinstance(existing, dict) and dict(existing) == {"url": url}:
        return config, False
    servers["slowave"] = {"url": url}
    return config, True


def _remove_mcp_servers_from_settings(config: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    """Remove any slowave mcpServers entry from settings.json.

    Handles both old stdio entries and any stale entries left in the wrong file.
    MCP config belongs in the client-specific JSON (e.g. ~/.claude.json), not settings.json.
    """
    if "mcpServers" not in config:
        return config, False
    mcp = config.get("mcpServers", {})
    if "slowave" not in mcp:
        return config, False
    del mcp["slowave"]
    if not mcp:
        del config["mcpServers"]
    return config, True


# ---------------------------------------------------------------------------
# CLAUDE.md / .clinerules injection
# ---------------------------------------------------------------------------

_MARKER_START = "<!-- slowave-lifecycle-start"  # prefix match — covers v1 and v2
_MARKER_END = "<!-- slowave-lifecycle-end"  # prefix match — covers v1 and v2

# Extracts the version token from an installed marker line, e.g. "v3" from
# "<!-- slowave-lifecycle-start v3 -->". Used by clients.py to detect a
# stale (un-upgraded) block instead of only checking presence/absence.
_VERSION_MARKER_RE = re.compile(r"<!-- slowave-lifecycle-start (v\d+) -->")


def _detect_lifecycle_version(text: str) -> str | None:
    """Return the lifecycle block version marker found in `text`, or None."""
    m = _VERSION_MARKER_RE.search(text)
    return m.group(1) if m else None


_LIFECYCLE_BLOCK_TEMPLATE = f"""\
<!-- slowave-lifecycle-start {LIFECYCLE_VERSION} -->
## MANDATORY — Slowave memory (5-verb cognitive cycle)

You are the reasoning module; Slowave is the memory module. Give it honest signals — what you encoded, what helped, what was noise, the outcome — and trust consolidation to do the rest. Do not respond until step 1 completes. Do not end the task without step 5.

**1 — `slowave_activate` (before your first response)**
`slowave_activate(query="<verbatim task>", goal="<short goal>", scope="project:<basename(cwd)>")` → store `retrieval_id`.
- `query`: the task verbatim — do not summarize (raw text drives retrieval).
- `goal`: 3–6 word verb-noun phrase (e.g. `"fix auth null pointer"`). Phrase it naturally; it is folded into the retrieval cue, so roughly consistent wording for the same kind of task gives a small overlap boost. Exact matching is NOT needed.
- `scope`: `project:<name>` (or `user:<id>` / `domain:<topic>`). Never omit.
- Call ONCE.

   **Cold start gate — if the response contains `cold_start: true`:**
   - Find the most stable context document available (project README/overview, system instructions, or user profile).
   - For each fact, ask: is it durable AND not already observable from the current context? If yes to both, call `slowave_remember(content, type, scope)` — one call per fact, never grouped.
   - Exhaust that document before responding. Do NOT scan the full codebase.

**2 — `slowave_remember` (encode durable knowledge)**
`slowave_remember(content, type, scope="project:<basename(cwd)>")` — call per durable fact.
- Novelty gate — skip if it already surfaced in activate/recall, is reconstructible from current context, or is transient/session-only state.
- ONE fact per call (never bundle — it blurs the embedding).
- Blank-slate phrasing: write so a reader with zero session context understands it. WRONG: `"fixed it by adding the field"`. RIGHT: `"SessionReaper idle timeout defaults to 3600s; the HTTP daemon disables it (0)"`.
- `type` (pick the most specific; default `decision`): `fact` · `preference` (how the user wants things) · `decision` (choice + reason) · `constraint` (invariant) · `procedure` (repeatable steps) · `lesson` (from failure/surprise) · `warning` (hazard) · `open_question` · `task` (durable to-do) · `artifact` (produced/external ref).
- If a remembered fact changed: remember the corrected version AND flag the old one via `stale_memory_ids`/`wrong_memory_ids` in step 4.
- Never encode: what is observable right now, transient state, vague impressions, or what you did this session (step 5 captures that).

**3 — `slowave_recall` (mid-task lookups — call whenever you pivot to a new sub-question, not only on failure)**
`slowave_recall(query, scope="project:<basename(cwd)>")` — specific, semantic query. WRONG: `"what about auth"`. RIGHT: `"decision on daemon single-instance enforcement"`. Always pass `scope` (omitting returns ALL projects). Store the returned `retrieval_id`. Not a substitute for activate — a deliberate lookup you reach for as often as the task's sub-questions change, not a fallback for when activate came up empty.

**4 — `slowave_reinforce` (after ANY retrieval — reward hits, suppress noise)**
Call whenever activate/recall returned memories — not only when you used some. Penalizing noise is how the store stays clean.
`slowave_reinforce(retrieval_id=<id>, feedback="useful|partially_useful|irrelevant|stale|wrong|missing|too_much_context", outcome="success|partial|failure|unknown", used_memory_ids=[...], irrelevant_memory_ids=[...], stale_memory_ids=[...], wrong_memory_ids=[...])`
- `used_memory_ids`: IDs you actually relied on (strengthens them).
- `irrelevant`/`stale`/`wrong_memory_ids`: IDs that were noise, outdated, or incorrect (this is how the store self-cleans). Use real IDs only — never invent.
- `feedback` and `outcome`: honest, not optimistic. Use `missing` to flag a needed-but-absent memory.

**5 — `slowave_commit` (session close — always)**
`slowave_commit(scope="project:<basename(cwd)>", outcome="success|partial|failure", steps=[...])`. Non-negotiable. Scope must match activate; outcome honest (`partial` if anything was incomplete). Skipping = no episodes form; the session lingers until the idle reaper closes it with no outcome.
- `steps`: ordered list of what you actually did this session, one sentence per meaningful phase. Example: `steps=["Ran full test suite (142 passed)", "Built and pushed Docker image", "Deployed to staging and health-checked"]`. This is how Slowave learns procedural patterns from repeated experience — be honest about what happened, even if it didn't all go perfectly. Do NOT include memory operations (activate/remember/recall/reinforce) as steps — those are the scaffolding, not the work.

Anti-patterns: skip activate · `remember` without `scope` · bundle facts in one call · context-dependent phrasing · re-encode facts already surfaced · leave a superseded fact unflagged · reinforce only hits and never penalize noise · default feedback to `useful` · invent memory IDs · report `success` when partial/failed · skip reinforce or commit · use deleted tools (`slowave_context`, `slowave_session_start/end`, `slowave_event`, `slowave_retrieval_feedback`, `slowave_context_feedback`).
<!-- slowave-lifecycle-end {LIFECYCLE_VERSION} -->"""


def _lifecycle_block(agent: str) -> str:
    # NOTE: `agent` is intentionally unused. The v3 lifecycle block is
    # client-agnostic — every AI client receives identical instructions, by
    # design, so Slowave stays consistent across models. The parameter is
    # kept for ClientSpec API stability; the .format() call is a no-op (the
    # template contains no {agent} placeholder). Do NOT re-add {agent} to
    # the template without a brain-system justification.
    return _LIFECYCLE_BLOCK_TEMPLATE.format(agent=agent)


# Heading used by legacy (pre-marker) slowave setup to write lifecycle instructions.
# Present as un-markered content in files written by slowave ≤0.4.x.
_LEGACY_SECTION_HEADING = "## Slowave memory"


def _strip_legacy_slowave_section(content: str) -> str:
    """Remove the legacy un-markered '## Slowave memory' block written by old setup versions.

    Old setup (pre-marker) wrote a plain markdown section starting with
    '## Slowave memory' that is not wrapped in HTML comment markers.  When
    users upgrade and run 'slowave setup' the new marker-based block is
    prepended, but the old section survives below it causing Claude to read
    two contradictory sets of instructions.  This helper strips that tail
    section so the file stays clean after cleanup + setup.

    Only removes content that is recognisably slowave-generated (contains the
    heading '## Slowave memory').  Unrelated user content is left untouched.
    Returns the cleaned string (may equal the input if nothing was found).
    """
    if _LEGACY_SECTION_HEADING not in content:
        return content

    lines = content.splitlines(keepends=True)
    result: list[str] = []
    in_legacy = False
    for line in lines:
        stripped = line.strip()
        if not in_legacy and stripped == _LEGACY_SECTION_HEADING:
            in_legacy = True
            # Drop any blank lines immediately before this heading that we
            # already buffered.
            while result and result[-1].strip() == "":
                result.pop()
            continue
        if in_legacy:
            # End of section = next heading of the same or higher level, or EOF.
            if stripped.startswith("## ") and stripped != _LEGACY_SECTION_HEADING:
                in_legacy = False
                result.append(line)
            # otherwise: keep skipping legacy lines
        else:
            result.append(line)
    return "".join(result)


def _inject_block(path: Path, block: str) -> bool:
    """Inject block between markers. Idempotent — replaces existing block. Returns True if changed.

    Also strips any legacy un-markered '## Slowave memory' section written by
    old setup versions (≤0.4.x) so that Claude never reads two conflicting
    sets of lifecycle instructions after an upgrade.

    A timestamped backup is created before any write (see _backup_file).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = path.read_text(encoding="utf-8") if path.exists() else ""

    if _MARKER_START in existing and _MARKER_END in existing:
        start = existing.index(_MARKER_START)
        end_marker_pos = existing.index(_MARKER_END)
        # Advance past the full marker line (handles versioned suffixes like " v2 -->")
        end_of_line = existing.find("\n", end_marker_pos)
        end = end_of_line + 1 if end_of_line != -1 else len(existing)
        before = existing[:start]
        after = _strip_legacy_slowave_section(existing[end:])
        new_content = before + block + after
        if new_content == existing:
            return False
        bak = _backup_file(path)
        if bak:
            click.echo(click.style(f"  ↩  backup → {bak}", fg="cyan"))
        path.write_text(new_content, encoding="utf-8")
        return True

    # No markers yet — strip any legacy section before prepending.
    rest = _strip_legacy_slowave_section(existing)
    sep = "\n\n" if rest.strip() else ""
    new_content = block + sep + rest
    if new_content == existing:
        return False
    bak = _backup_file(path)
    if bak:
        click.echo(click.style(f"  ↩  backup → {bak}", fg="cyan"))
    path.write_text(new_content, encoding="utf-8")
    return True


# ---------------------------------------------------------------------------
# Claude Code hooks
# ---------------------------------------------------------------------------

_HOOKS_MARKER = "SLOWAVE MANDATORY"

_USER_PROMPT_CMD = (
    "echo 'SLOWAVE MANDATORY: Before writing your first word call: "
    "(1) derive goal=3-6 word verb-noun phrase e.g. implement-oauth-login fix-auth-bug refactor-db-layer, "
    "(2) slowave_activate(query=<verbatim task>,goal=<derived goal>,scope=project:<basename(cwd)>) "
    "store retrieval_id and session_id. "
    "Produce NO output until this call succeeds.'"
)

_STOP_CMD = (
    "echo 'SLOWAVE MANDATORY: Before finishing this turn call: "
    "(1) if you used memories: slowave_reinforce(retrieval_id=<id>,feedback=useful|irrelevant|stale|wrong,outcome=success|partial|failure|unknown), "
    "(2) slowave_commit(scope=project:<basename(cwd)>,outcome=success|partial|failure|unknown). "
    "Do NOT end the turn without step 2.'"
)


def _hooks_up_to_date(config: dict[str, Any], event: str, cmd: str) -> bool:
    """Return True iff a Slowave hook for *event* already has exactly *cmd* as its command.

    Checks for exact command text rather than mere marker presence so that
    updated hook commands are applied on re-run.
    """
    for group in config.get("hooks", {}).get(event, []):
        for h in group.get("hooks", []):
            if _HOOKS_MARKER in h.get("command", ""):
                return h.get("command", "") == cmd
    return False


def _patch_claude_code_hooks(config: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    """Inject or update UserPromptSubmit + Stop hooks.

    Always writes the current command text.  If a stale Slowave hook already
    exists (marker present but command differs), it is replaced in-place.
    If no Slowave hook exists yet, a new group is appended.
    """
    changed = False
    hooks = config.setdefault("hooks", {})
    for event, cmd in [("UserPromptSubmit", _USER_PROMPT_CMD), ("Stop", _STOP_CMD)]:
        if _hooks_up_to_date(config, event, cmd):
            continue  # already correct — skip
        # Remove any stale Slowave hook group for this event, then re-add.
        if event in hooks:
            hooks[event] = [
                g
                for g in hooks[event]
                if not any(_HOOKS_MARKER in h.get("command", "") for h in g.get("hooks", []))
            ]
        hooks.setdefault(event, []).append(
            {"matcher": "", "hooks": [{"type": "command", "command": cmd}]}
        )
        changed = True
    return config, changed


def _remove_claude_code_hooks(config: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    """Remove all Slowave enforcement hooks from a Claude Code settings dict.

    Used by cleanup.  Removes every hook group whose command contains
    ``_HOOKS_MARKER`` from the ``UserPromptSubmit`` and ``Stop`` events.
    """
    changed = False
    for event in ["UserPromptSubmit", "Stop"]:
        before = config.get("hooks", {}).get(event, [])
        after = [
            g
            for g in before
            if not any(_HOOKS_MARKER in h.get("command", "") for h in g.get("hooks", []))
        ]
        if after != before:
            config.setdefault("hooks", {})[event] = after
            changed = True
    return config, changed


# ---------------------------------------------------------------------------
# Codex hooks
# ---------------------------------------------------------------------------
#
# Codex's [[hooks.<Event>]] TOML array-of-tables shape parses down to the
# same nested dict/list-of-dicts data model as Claude Code's JSON hooks, minus
# the "matcher" field (only relevant for tool-scoped events like PreToolUse,
# not UserPromptSubmit/Stop). _hooks_up_to_date() is structure-agnostic (only
# calls .get() on whatever it's handed) so it is reused as-is against the
# tomlkit document returned by _read_toml().


def _patch_codex_hooks(config: Any) -> tuple[Any, bool]:
    """Inject or update Codex UserPromptSubmit + Stop hooks.

    Mirrors _patch_claude_code_hooks(): always writes the current command
    text, replacing any stale Slowave hook group in-place.
    """
    changed = False
    hooks = config.setdefault("hooks", {})
    for event, cmd in [("UserPromptSubmit", _USER_PROMPT_CMD), ("Stop", _STOP_CMD)]:
        if _hooks_up_to_date(config, event, cmd):
            continue
        existing = [
            g
            for g in hooks.get(event, [])
            if not any(_HOOKS_MARKER in h.get("command", "") for h in g.get("hooks", []))
        ]
        existing.append({"hooks": [{"type": "command", "command": cmd}]})
        hooks[event] = existing
        changed = True
    return config, changed


def _remove_codex_hooks(config: Any) -> tuple[Any, bool]:
    """Remove all Slowave enforcement hooks from a Codex config.toml document.

    Used by cleanup. Same logic as _remove_claude_code_hooks(), just without
    the "matcher" field Codex doesn't use for these events.
    """
    changed = False
    for event in ["UserPromptSubmit", "Stop"]:
        before = config.get("hooks", {}).get(event, [])
        after = [
            g
            for g in before
            if not any(_HOOKS_MARKER in h.get("command", "") for h in g.get("hooks", []))
        ]
        if after != before:
            config.setdefault("hooks", {})[event] = after
            changed = True
    return config, changed


# ---------------------------------------------------------------------------
# Worker service templates
# ---------------------------------------------------------------------------

_LAUNCHD_PLIST = """\
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
  <dict>
    <key>Label</key><string>com.slowave.worker</string>
    <key>ProgramArguments</key>
    <array>
      <string>{bin}</string>
      <string>worker</string>
      <string>--interval</string>
      <string>300</string>
    </array>
    <key>RunAtLoad</key><true/>
    <key>KeepAlive</key><true/>
    <key>StandardOutPath</key><string>/tmp/slowave-worker.log</string>
    <key>StandardErrorPath</key><string>/tmp/slowave-worker.err</string>
  </dict>
</plist>
"""

_LAUNCHD_DAEMON_PLIST = """\
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
  <dict>
    <key>Label</key><string>com.slowave.daemon</string>
    <key>ProgramArguments</key>
    <array>
      <string>{bin}</string>
      <string>serve</string>
      <string>start</string>
    </array>
    <key>RunAtLoad</key><true/>
    <key>KeepAlive</key><true/>
    <key>StandardOutPath</key><string>/tmp/slowave-daemon.log</string>
    <key>StandardErrorPath</key><string>/tmp/slowave-daemon.err</string>
  </dict>
</plist>
"""

_SYSTEMD_SERVICE = """\
[Unit]
Description=Slowave background consolidation worker
After=network.target

[Service]
ExecStart={bin} worker --interval 300
Restart=always
RestartSec=10

[Install]
WantedBy=default.target
"""

_SYSTEMD_DAEMON_SERVICE = """\
[Unit]
Description=Slowave HTTP MCP daemon
After=network.target

[Service]
ExecStart={bin} serve start
Restart=always
RestartSec=5

[Install]
WantedBy=default.target
"""


def _install_worker_macos(slowave_bin: str) -> tuple[str, bool]:
    plist_dir = _home() / "Library" / "LaunchAgents"
    plist_path = plist_dir / "com.slowave.worker.plist"
    content = _LAUNCHD_PLIST.format(bin=slowave_bin)
    if plist_path.exists() and plist_path.read_text(encoding="utf-8") == content:
        return str(plist_path), False
    plist_dir.mkdir(parents=True, exist_ok=True)
    plist_path.write_text(content, encoding="utf-8")
    try:
        subprocess.run(["launchctl", "unload", str(plist_path)], capture_output=True, check=False)
        subprocess.run(["launchctl", "load", str(plist_path)], capture_output=True, check=False)
    except FileNotFoundError:
        pass
    return str(plist_path), True


def _install_worker_linux(slowave_bin: str) -> tuple[str, bool]:
    xdg = os.environ.get("XDG_CONFIG_HOME", str(_home() / ".config"))
    svc_dir = Path(xdg) / "systemd" / "user"
    svc_path = svc_dir / "slowave-worker.service"
    content = _SYSTEMD_SERVICE.format(bin=slowave_bin)
    if svc_path.exists() and svc_path.read_text(encoding="utf-8") == content:
        return str(svc_path), False
    svc_dir.mkdir(parents=True, exist_ok=True)
    svc_path.write_text(content, encoding="utf-8")
    try:
        subprocess.run(["systemctl", "--user", "daemon-reload"], capture_output=True, check=False)
        subprocess.run(
            ["systemctl", "--user", "enable", "--now", "slowave-worker"],
            capture_output=True,
            check=False,
        )
    except FileNotFoundError:
        pass
    return str(svc_path), True


def _find_pythonw() -> str | None:
    """Return the path to pythonw.exe (no-console Python launcher) on Windows.

    pythonw.exe lives in the same directory as python.exe and is shipped with
    every Python Windows installer.  Using it instead of slowave.EXE for the
    Task Scheduler action prevents a visible console window from opening.
    """
    import os as _os

    py_dir = _os.path.dirname(sys.executable)
    candidate = _os.path.join(py_dir, "pythonw.exe")
    if Path(candidate).exists():
        return candidate
    return None


# Description marker: bump the version to force re-registration of tasks
# created by older slowave versions (they lack battery/keep-alive settings).
_WINDOWS_TASK_MARKER = "Managed by slowave setup (v2)"


def _ps_squote(s: str) -> str:
    """Escape a string for a single-quoted PowerShell literal."""
    return s.replace("'", "''")


def _register_windows_task(task_name: str, execute: str, argument: str) -> tuple[bool, str]:
    """Register (or upgrade) a keep-alive Slowave task in Task Scheduler.

    Reliability settings (all absent from pre-v2 registrations):
    - -AllowStartIfOnBatteries / -DontStopIfGoingOnBatteries: the Task
      Scheduler DEFAULT is to skip the logon trigger on battery and kill the
      task when unplugging — the main reason services "never start" on laptops.
    - A second trigger repeating every 5 minutes indefinitely, with
      -MultipleInstances IgnoreNew: while the process runs, ticks are ignored;
      if it dies, the next tick relaunches it. This is the Windows equivalent
      of launchd KeepAlive / systemd Restart=always.

    Returns (ok, detail). Registration failures are surfaced with the
    PowerShell error text instead of being swallowed.
    """
    exe_q, arg_q, name_q = _ps_squote(execute), _ps_squote(argument), _ps_squote(task_name)

    # Idempotency: skip only when the existing task matches this exact form
    # (marker + executable + arguments). Older registrations get upgraded.
    try:
        check = subprocess.run(
            [
                "powershell",
                "-NonInteractive",
                "-Command",
                f"$t = Get-ScheduledTask -TaskName '{name_q}' -ErrorAction SilentlyContinue; "
                f"if ($t) {{ $t.Description + '|' + $t.Actions[0].Execute + '|' + $t.Actions[0].Arguments }}",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
        )
        parts = check.stdout.strip().split("|")
        if (
            len(parts) == 3
            and parts[0].strip() == _WINDOWS_TASK_MARKER
            and parts[1].strip().lower() == execute.lower()
            and parts[2].strip().lower() == argument.lower()
        ):
            return True, "already up-to-date"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False, "powershell not available"

    ps = (
        f"$ErrorActionPreference='Stop';"
        f"$a=New-ScheduledTaskAction -Execute '{exe_q}' -Argument '{arg_q}';"
        f"$logon=New-ScheduledTaskTrigger -AtLogOn;"
        f"$tick=New-ScheduledTaskTrigger -Once -At (Get-Date) "
        f"-RepetitionInterval (New-TimeSpan -Minutes 5) "
        f"-RepetitionDuration (New-TimeSpan -Days 3650);"
        f"$s=New-ScheduledTaskSettingsSet -ExecutionTimeLimit 0 -RestartCount 3 "
        f"-RestartInterval (New-TimeSpan -Minutes 1) "
        f"-AllowStartIfOnBatteries -DontStopIfGoingOnBatteries "
        f"-MultipleInstances IgnoreNew;"
        f"$p=New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Limited;"
        f"Register-ScheduledTask -TaskName '{name_q}' -Description '{_WINDOWS_TASK_MARKER}' "
        f"-Action $a -Trigger $logon,$tick -Settings $s -Principal $p -Force | Out-Null;"
        f"Start-ScheduledTask -TaskName '{name_q}'"
    )
    try:
        result = subprocess.run(
            ["powershell", "-NonInteractive", "-Command", ps],
            capture_output=True,
            text=True,
            check=False,
            timeout=120,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False, "powershell not available"
    if result.returncode != 0:
        err = (result.stderr or result.stdout).strip().replace("\r\n", " ")
        return False, f"registration failed: {err[:400]}"
    return True, "registered and started"


def _install_worker_windows(slowave_bin: str) -> tuple[bool, str]:
    """Register SlowaveWorker in Task Scheduler.

    Uses ``pythonw.exe -m slowave worker`` so the worker runs without opening a
    visible console window.  Falls back to ``slowave_bin`` if pythonw.exe is not
    found (rare custom installs without the no-console launcher).
    """
    pythonw = _find_pythonw()
    if pythonw:
        execute, argument = pythonw, "-m slowave worker --interval 300"
    else:
        execute, argument = slowave_bin, "worker --interval 300"
    return _register_windows_task("SlowaveWorker", execute, argument)


def _install_daemon_macos(slowave_bin: str) -> tuple[str, bool]:
    """Install the HTTP MCP daemon as a launchd user agent (macOS)."""
    plist_dir = _home() / "Library" / "LaunchAgents"
    plist_path = plist_dir / "com.slowave.daemon.plist"
    content = _LAUNCHD_DAEMON_PLIST.format(bin=slowave_bin)
    if plist_path.exists() and plist_path.read_text(encoding="utf-8") == content:
        return str(plist_path), False
    plist_dir.mkdir(parents=True, exist_ok=True)
    plist_path.write_text(content, encoding="utf-8")
    try:
        subprocess.run(["launchctl", "unload", str(plist_path)], capture_output=True, check=False)
        subprocess.run(["launchctl", "load", str(plist_path)], capture_output=True, check=False)
    except FileNotFoundError:
        pass
    return str(plist_path), True


def _install_daemon_linux(slowave_bin: str) -> tuple[str, bool]:
    """Install the HTTP MCP daemon as a systemd user service (Linux)."""
    xdg = os.environ.get("XDG_CONFIG_HOME", str(_home() / ".config"))
    svc_dir = Path(xdg) / "systemd" / "user"
    svc_path = svc_dir / "slowave-daemon.service"
    content = _SYSTEMD_DAEMON_SERVICE.format(bin=slowave_bin)
    if svc_path.exists() and svc_path.read_text(encoding="utf-8") == content:
        return str(svc_path), False
    svc_dir.mkdir(parents=True, exist_ok=True)
    svc_path.write_text(content, encoding="utf-8")
    try:
        subprocess.run(["systemctl", "--user", "daemon-reload"], capture_output=True, check=False)
        subprocess.run(
            ["systemctl", "--user", "enable", "--now", "slowave-daemon"],
            capture_output=True,
            check=False,
        )
    except FileNotFoundError:
        pass
    return str(svc_path), True


def _install_daemon_windows(slowave_bin: str) -> tuple[bool, str]:
    """Register the HTTP MCP daemon as a Windows Scheduled Task.

    Uses ``pythonw.exe -m slowave serve start`` so the daemon runs without a
    visible console window (closing that window would kill the daemon).
    """
    pythonw = _find_pythonw()
    if pythonw:
        execute, argument = pythonw, "-m slowave serve start"
    else:
        execute, argument = slowave_bin, "serve start"
    return _register_windows_task("SlowaveDaemon", execute, argument)


# Windows Task Scheduler dispatch is asynchronous (Start-ScheduledTask returns
# before the process is actually spawned), and the daemon's first import of
# faiss/numpy is commonly slowed further by antivirus/Defender scanning the
# freshly-installed DLLs. 15s is routinely too short there even though the
# daemon comes up fine a little later, so give Windows more room.
_DAEMON_HEALTH_TIMEOUT = 45.0 if SYSTEM == "Windows" else 15.0


def _verify_daemon_health(timeout: float = _DAEMON_HEALTH_TIMEOUT) -> bool:
    """Poll the daemon /health endpoint until it responds or *timeout* elapses.

    Setup previously claimed success as soon as the service files were written;
    this turns "setup said OK but nothing is listening" into an immediate error.
    """
    import time
    import urllib.request

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen("http://127.0.0.1:8766/health", timeout=1) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


def _build_summary(client: str, worker: bool, install_hooks: bool, slowave_bin: str) -> Summary:
    """Build a summary of changes without modifying any files."""
    summary = Summary()
    summary.add_binary("slowave", slowave_bin)

    for spec in _clients_for(client):
        mcp_file = spec.mcp_path()
        if spec.require_dir_exists and not mcp_file.parent.exists():
            continue

        cfg = _read_toml(mcp_file) if spec.key == "codex" else _read_json(mcp_file)
        if spec.key == "claude-desktop":
            slowave_mcp_bin = _find_mcp_binary(slowave_bin)
            _, changed_mcp = _patch_mcp_servers_stdio(cfg, command=slowave_mcp_bin)
        elif spec.key == "opencode":
            cfg, changed_mcp_only = _patch_opencode_mcp(cfg)
            # Also check instructions registration
            _, changed_instructions = _patch_opencode_instructions(
                cfg, str(_opencode_instructions_path().resolve())
            )
            changed_mcp = changed_mcp_only or changed_instructions
        elif spec.key == "codex":
            _, changed_mcp = _patch_codex_mcp(cfg)
        else:
            _, changed_mcp = _patch_mcp_servers(
                cfg, include_type=spec.key == "claude-code", use_sse=spec.key == "cline"
            )
        summary.add_change(
            Change(
                change_type=ChangeType.MCP_CONFIG,
                client=spec.label,
                status=ChangeStatus.UPDATE if changed_mcp else ChangeStatus.SKIP,
                path=str(mcp_file),
                description="MCP server configuration",
            )
        )
        if spec.hooks_config_path is not None and spec.hooks_patch_fn is not None and install_hooks:
            hooks_file = spec.hooks_config_path()
            # Codex keeps hooks in the same file as MCP config (cfg, already patched above) —
            # re-patch that in-memory doc rather than re-reading, so the preview reflects both
            # changes as they'd actually land in one combined write.
            hooks_source = cfg if spec.key == "codex" else _read_json(hooks_file)
            _, changed_hooks = spec.hooks_patch_fn(hooks_source)
            summary.add_change(
                Change(
                    change_type=ChangeType.HOOKS,
                    client=spec.label,
                    status=ChangeStatus.UPDATE if changed_hooks else ChangeStatus.SKIP,
                    path=str(hooks_file),
                    description="Enforcement hooks",
                )
            )
        if spec.lifecycle_path is not None:
            lc_file = spec.lifecycle_path()
            existing = lc_file.read_text(encoding="utf-8") if lc_file.exists() else ""
            needs_update = _MARKER_START not in existing or _MARKER_END not in existing
            summary.add_change(
                Change(
                    change_type=ChangeType.LIFECYCLE_BLOCK,
                    client=spec.label,
                    status=ChangeStatus.UPDATE if needs_update else ChangeStatus.SKIP,
                    path=str(lc_file),
                    description="Lifecycle instruction block",
                )
            )
        elif spec.manual_lifecycle and spec.manual_note:
            summary.add_manual_step(spec.manual_note)

    # Worker service
    if worker:
        if SYSTEM == "Darwin":
            plist_dir = _home() / "Library" / "LaunchAgents"
            plist_path = plist_dir / "com.slowave.worker.plist"
            plist_content = _LAUNCHD_PLIST.format(bin=slowave_bin)
            changed = not (
                plist_path.exists() and plist_path.read_text(encoding="utf-8") == plist_content
            )
            status = ChangeStatus.UPDATE if changed else ChangeStatus.SKIP
            summary.add_change(
                Change(
                    change_type=ChangeType.WORKER_SERVICE,
                    client="macOS",
                    status=status,
                    path=str(plist_path),
                    description="launchd service",
                )
            )
        elif SYSTEM == "Linux":
            xdg = os.environ.get("XDG_CONFIG_HOME", str(_home() / ".config"))
            svc_path = Path(xdg) / "systemd" / "user" / "slowave-worker.service"
            svc_content = _SYSTEMD_SERVICE.format(bin=slowave_bin)
            changed = not (
                svc_path.exists() and svc_path.read_text(encoding="utf-8") == svc_content
            )
            status = ChangeStatus.UPDATE if changed else ChangeStatus.SKIP
            summary.add_change(
                Change(
                    change_type=ChangeType.WORKER_SERVICE,
                    client="Linux",
                    status=status,
                    path=str(svc_path),
                    description="systemd service",
                )
            )
        elif SYSTEM == "Windows":
            # Check if the task already exists with the current binary
            task_exists = False
            try:
                check = subprocess.run(
                    [
                        "powershell",
                        "-NonInteractive",
                        "-Command",
                        "$t = Get-ScheduledTask -TaskName 'SlowaveWorker' -ErrorAction SilentlyContinue; "
                        "if ($t) { $t.Actions[0].Execute }",
                    ],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                existing_exe = check.stdout.strip()
                task_exists = bool(existing_exe and existing_exe.lower() == slowave_bin.lower())
            except FileNotFoundError:
                pass
            status = ChangeStatus.SKIP if task_exists else ChangeStatus.UPDATE
            summary.add_change(
                Change(
                    change_type=ChangeType.WORKER_SERVICE,
                    client="Windows",
                    status=status,
                    path="Task Scheduler",
                    description="SlowaveWorker task",
                )
            )

    return summary


# Output helpers
# ---------------------------------------------------------------------------


def _ok(msg: str) -> None:
    click.echo(click.style(f"  ✓  {msg}", fg="green"))


def _skip(msg: str) -> None:
    click.echo(click.style(f"  –  {msg}", fg="bright_black"))


def _warn(msg: str) -> None:
    click.echo(click.style(f"  ⚠  {msg}", fg="yellow"))


def _err(msg: str) -> None:
    click.echo(click.style(f"  ✗  {msg}", fg="red"))


def _section(title: str) -> None:
    click.echo(f"\n{click.style(title, bold=True)}")


# ---------------------------------------------------------------------------
# `slowave setup` Click command
# ---------------------------------------------------------------------------


@click.command("setup")
@click.option(
    "--client",
    type=click.Choice(
        [
            "claude-code",
            "claude-desktop",
            "cline",
            "cursor",
            "windsurf",
            "opencode",
            "codex",
            "all",
        ],
        case_sensitive=False,
    ),
    default="all",
    show_default=True,
    help="Which client(s) to configure.",
)
@click.option(
    "--worker/--no-worker",
    default=True,
    show_default=True,
    help="Install the background worker as a system service.",
)
@click.option(
    "--hooks/--no-hooks",
    "install_hooks",
    default=True,
    show_default=True,
    help="Inject UserPromptSubmit + Stop hooks (Claude Code and Codex only).",
)
@click.option("--dry-run", is_flag=True, help="Preview changes without writing any files.")
@click.option("--json", "as_json", is_flag=True, help="Machine-readable JSON output.")
def setup_cmd(
    client: str, worker: bool, install_hooks: bool, dry_run: bool, as_json: bool = False
) -> None:
    """One-command post-install wiring for Claude Code, Claude Desktop, Cline, Cursor, Windsurf, OpenCode, and Codex.

    Configures every detected client to connect to the Slowave HTTP MCP daemon
    at http://127.0.0.1:8766/mcp.  The daemon and background worker start
    automatically as system services — no manual steps needed.

    Automates MCP config, lifecycle instruction injection, enforcement hooks,
    and the daemon + worker services. All steps are idempotent.

    \b
    Examples:
      slowave setup                       # wire everything
      slowave setup --client cline        # Cline only
      slowave setup --no-worker           # skip service install
      slowave setup --dry-run             # preview without writing
    """
    click.echo(click.style("\nSlowave setup", bold=True))
    if dry_run:
        click.echo(click.style("  [DRY RUN — no files will be changed]\n", fg="yellow"))

    # 1. Binaries
    _section("1. Locating binaries")
    slowave_bin = _find_slowave_binary()
    _ok(f"slowave: {slowave_bin}")
    _ok("MCP endpoint: http://127.0.0.1:8766/mcp  (daemon auto-starts via system service)")

    # Quick detection pass — show which clients were found
    _section("2. Detecting clients")
    detected: list[str] = []
    skipped: list[str] = []
    for spec in _clients_for(client):
        mcp_file = spec.mcp_path()
        if spec.require_dir_exists and not mcp_file.parent.exists():
            skipped.append(spec.label)
        else:
            detected.append(spec.label)
    if detected:
        _ok(f"Found: {', '.join(detected)}")
    if skipped:
        _warn(f"Not installed — skipping: {', '.join(skipped)}")

    # Build and display summary
    summary = _build_summary(client, worker, install_hooks, slowave_bin)
    click.echo(summary.format())

    # Confirm unless dry-run — skip if nothing to do
    if not dry_run:
        if not summary.has_changes():
            click.echo(click.style("\nEverything already configured. Nothing to do.", fg="green"))
            if summary.manual_steps:
                click.echo(click.style("\nReminders:", bold=True))
                for step in summary.manual_steps:
                    _warn(step)
            sys.exit(0)
        if not click.confirm("\nConfigure detected clients?", default=True):
            click.echo(click.style("\nSetup cancelled.", fg="yellow"))
            sys.exit(0)

    # 3-N. Clients (data-driven — add new clients in _clients() only)
    for i, spec in enumerate(_clients_for(client), start=3):
        _section(f"{i}. {spec.label}")
        mcp_file = spec.mcp_path()

        # Skip if the config directory doesn't exist and client marks require_dir_exists
        if spec.require_dir_exists and not mcp_file.parent.exists():
            _warn(
                f"{spec.label} config dir not found: {mcp_file.parent}  ({spec.label} installed?)"
            )
            continue

        if spec.key == "codex":
            # Codex keeps MCP config and hooks in the same TOML file (unlike Claude
            # Code's split ~/.claude.json / ~/.claude/settings.json). Read once, apply
            # both patches to the same in-memory doc, write once — reading it twice and
            # writing it twice (as the generic path below does for two-file clients)
            # would leave the retained *.bak.* as the post-MCP-patch state instead of
            # the true pre-Slowave original.
            cfg = _read_toml(mcp_file)
            cfg, mcp_changed = _patch_codex_mcp(cfg)
            if mcp_changed:
                (
                    _ok(f"Would set MCP server (HTTP) → {mcp_file}")
                    if dry_run
                    else _ok(f"MCP server set (HTTP) → {mcp_file}")
                )
            else:
                _skip(f"MCP server already configured (HTTP) in {mcp_file}")

            hooks_changed = False
            if spec.hooks_config_path is not None and spec.hooks_patch_fn is not None:
                if install_hooks:
                    cfg, hooks_changed = spec.hooks_patch_fn(cfg)
                    if hooks_changed:
                        (
                            _ok(f"Would update enforcement hooks → {mcp_file}")
                            if dry_run
                            else _ok(f"Enforcement hooks updated → {mcp_file}")
                        )
                    else:
                        _skip(f"Enforcement hooks already up-to-date in {mcp_file}")
                else:
                    _skip(f"Enforcement hooks skipped (--no-hooks) for {spec.label}")

            if not dry_run and (mcp_changed or hooks_changed):
                _write_toml(mcp_file, cfg)
        else:
            cfg = _read_json(mcp_file)
            if spec.key == "claude-desktop":
                slowave_mcp_bin = _find_mcp_binary(slowave_bin)
                cfg, changed = _patch_mcp_servers_stdio(cfg, command=slowave_mcp_bin)
                transport_label = "stdio"
            elif spec.key == "opencode":
                cfg, mcp_changed = _patch_opencode_mcp(cfg)
                transport_label = "remote"
                # Also register the instructions file in the config
                instructions_path = str(_opencode_instructions_path().resolve())
                cfg, instructions_changed = _patch_opencode_instructions(cfg, instructions_path)
                changed = mcp_changed or instructions_changed
            else:
                cfg, changed = _patch_mcp_servers(
                    cfg, include_type=spec.key == "claude-code", use_sse=spec.key == "cline"
                )
                transport_label = "HTTP"
            if changed:
                if dry_run:
                    _ok(f"Would set MCP server ({transport_label}) → {mcp_file}")
                else:
                    _write_json(mcp_file, cfg)
                    _ok(f"MCP server set ({transport_label}) → {mcp_file}")
            else:
                _skip(f"MCP server already configured ({transport_label}) in {mcp_file}")

            # Enforcement hooks — data-driven via spec.hooks_patch_fn
            if spec.hooks_config_path is not None and spec.hooks_patch_fn is not None:
                hooks_file = spec.hooks_config_path()
                cfg_hooks = _read_json(hooks_file)
                if install_hooks:
                    cfg_hooks, changed = spec.hooks_patch_fn(cfg_hooks)
                    if changed:
                        if dry_run:
                            _ok(f"Would update enforcement hooks → {hooks_file}")
                        else:
                            _write_json(hooks_file, cfg_hooks)
                            _ok(f"Enforcement hooks updated → {hooks_file}")
                    else:
                        _skip(f"Enforcement hooks already up-to-date in {hooks_file}")
                else:
                    _skip(f"Enforcement hooks skipped (--no-hooks) for {spec.label}")

        # Lifecycle block — auto-inject or print manual instruction
        if spec.lifecycle_path is not None:
            lc_file = spec.lifecycle_path()
            if dry_run:
                _ok(f"Would inject lifecycle block → {lc_file}")
            else:
                changed = _inject_block(lc_file, _lifecycle_block(spec.lifecycle_agent))
                if changed:
                    _ok(f"Lifecycle block injected → {lc_file}")
                else:
                    _skip("Lifecycle block already present")
        elif spec.manual_lifecycle and spec.manual_note:
            _warn(f"REQUIRED — {spec.manual_note}")

    # HTTP MCP daemon service
    _section("7. HTTP MCP daemon service")
    if worker:
        if dry_run:
            if SYSTEM == "Darwin":
                _ok(
                    "Would install launchd service → ~/Library/LaunchAgents/com.slowave.daemon.plist"
                )
            elif SYSTEM == "Linux":
                _ok("Would install systemd service → ~/.config/systemd/user/slowave-daemon.service")
            elif SYSTEM == "Windows":
                _ok("Would register Task Scheduler task: SlowaveDaemon")
            else:
                _warn(f"Unknown platform '{SYSTEM}' — run manually: slowave serve start")
        else:
            if SYSTEM == "Darwin":
                path, changed = _install_daemon_macos(slowave_bin)
                if changed:
                    _ok(f"launchd daemon service installed → {path}")
                else:
                    _skip("launchd daemon service already up-to-date")
            elif SYSTEM == "Linux":
                path, changed = _install_daemon_linux(slowave_bin)
                if changed:
                    _ok(f"systemd daemon service installed → {path}")
                    _ok("Verify:  systemctl --user status slowave-daemon")
                else:
                    _skip("systemd daemon service already up-to-date")
            elif SYSTEM == "Windows":
                ok, detail = _install_daemon_windows(slowave_bin)
                if ok:
                    _ok(f"Task Scheduler task SlowaveDaemon: {detail}")
                else:
                    _err(f"Task Scheduler task SlowaveDaemon: {detail}")
                    _warn("Start manually: slowave serve start")
            else:
                _warn(f"Unknown platform '{SYSTEM}'. Run manually: slowave serve start")
            if _verify_daemon_health():
                _ok("Daemon is live: http://127.0.0.1:8766/health")
            else:
                _err(
                    f"Daemon did not respond on http://127.0.0.1:8766/health "
                    f"within {_DAEMON_HEALTH_TIMEOUT:g}s."
                )
                if SYSTEM == "Windows":
                    _warn(
                        "It may still come up — Task Scheduler retries this task every "
                        "5 minutes if it isn't already running. Check: "
                        "Get-ScheduledTask -TaskName SlowaveDaemon"
                    )
                    _warn("Check logs: ~/.slowave/logs/pythonw-serve.log")
                elif SYSTEM == "Linux":
                    _warn("Check logs: journalctl --user -u slowave-daemon")
                else:
                    _warn("Check logs: /tmp/slowave-daemon.err")
    else:
        _skip("Skipped (--no-worker). Run manually: slowave serve start")

    # Background worker service
    _section("8. Background worker service")
    if worker:
        if dry_run:
            if SYSTEM == "Darwin":
                _ok(
                    "Would install launchd service → ~/Library/LaunchAgents/com.slowave.worker.plist"
                )
            elif SYSTEM == "Linux":
                _ok("Would install systemd service → ~/.config/systemd/user/slowave-worker.service")
            elif SYSTEM == "Windows":
                _ok("Would register Task Scheduler task: SlowaveWorker")
            else:
                _warn(f"Unknown platform '{SYSTEM}' — run manually: slowave worker --interval 300")
        else:
            if SYSTEM == "Darwin":
                path, changed = _install_worker_macos(slowave_bin)
                if changed:
                    _ok(f"launchd worker service installed → {path}")
                else:
                    _skip("launchd worker service already up-to-date")
            elif SYSTEM == "Linux":
                path, changed = _install_worker_linux(slowave_bin)
                if changed:
                    _ok(f"systemd worker service installed → {path}")
                    _ok("Verify:  systemctl --user status slowave-worker")
                else:
                    _skip("systemd worker service already up-to-date")
            elif SYSTEM == "Windows":
                ok, detail = _install_worker_windows(slowave_bin)
                if ok:
                    _ok(f"Task Scheduler task SlowaveWorker: {detail}")
                    _ok("Verify:  Get-ScheduledTask -TaskName SlowaveWorker")
                else:
                    _err(f"Task Scheduler task SlowaveWorker: {detail}")
                    _warn("Start manually: slowave worker --interval 300")
            else:
                _warn(f"Unknown platform '{SYSTEM}'. Run manually: slowave worker --interval 300")
    else:
        _skip("Skipped (--no-worker). Run manually: slowave worker --interval 300")

    # Daily database backup service
    _section("9. Daily database backup")
    if worker:
        if dry_run:
            if SYSTEM == "Darwin":
                _ok(
                    "Would install launchd service → ~/Library/LaunchAgents/com.slowave.backup.plist"
                )
            elif SYSTEM == "Linux":
                _ok("Would install systemd timer → ~/.config/systemd/user/slowave-backup.timer")
            elif SYSTEM == "Windows":
                _ok("Would register Task Scheduler task: SlowaveBackup")
            else:
                _warn(f"Unknown platform '{SYSTEM}' — run manually: slowave backup")
        else:
            if SYSTEM == "Darwin":
                path, changed = _install_backup_macos(slowave_bin)
                if changed:
                    _ok(f"launchd backup service installed → {path}")
                else:
                    _skip("launchd backup service already up-to-date")
            elif SYSTEM == "Linux":
                path, changed = _install_backup_linux(slowave_bin)
                if changed:
                    _ok(f"systemd backup timer installed → {path}")
                    _ok("Verify:  systemctl --user status slowave-backup.timer")
                else:
                    _skip("systemd backup timer already up-to-date")
            elif SYSTEM == "Windows":
                task, _ = _install_backup_windows(slowave_bin)
                _ok(f"Task Scheduler task registered: {task}")
                _ok("Verify:  Get-ScheduledTask -TaskName SlowaveBackup")
            else:
                _warn(f"Unknown platform '{SYSTEM}'. Run manually: slowave backup")
    else:
        _skip("Skipped (--no-worker). Run manually: slowave backup")

    # 10. Doctor
    _section("10. Verification")
    if dry_run:
        _skip("Dry-run — skipping doctor check.")
    else:
        click.echo()
        try:
            subprocess.run([sys.executable, "-m", "slowave", "doctor"], check=False)
        except Exception as exc:
            _warn(f"Could not run slowave doctor: {exc}")

    click.echo()
    click.echo(click.style("Setup complete.", bold=True))
    if not dry_run:
        mark_setup_done()


# ---------------------------------------------------------------------------
# Backup service templates
# ---------------------------------------------------------------------------

_LAUNCHD_BACKUP_PLIST = """\
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
  <dict>
    <key>Label</key><string>com.slowave.backup</string>
    <key>ProgramArguments</key>
    <array>
      <string>{bin}</string>
      <string>backup</string>
    </array>
    <key>StartCalendarInterval</key>
    <dict>
      <key>Hour</key><integer>3</integer>
      <key>Minute</key><integer>0</integer>
    </dict>
    <key>StandardOutPath</key><string>/tmp/slowave-backup.log</string>
    <key>StandardErrorPath</key><string>/tmp/slowave-backup.err</string>
  </dict>
</plist>
"""

_SYSTEMD_BACKUP_SERVICE = """\
[Unit]
Description=Slowave daily database backup

[Service]
Type=oneshot
ExecStart={bin} backup
"""

_SYSTEMD_BACKUP_TIMER = """\
[Unit]
Description=Daily Slowave database backup timer

[Timer]
OnCalendar=daily
Persistent=true

[Install]
WantedBy=timers.target
"""


def _install_backup_macos(slowave_bin: str) -> tuple[str, bool]:
    """Install the daily database backup as a launchd user agent (macOS).

    Uses StartCalendarInterval to run once per day at 03:00.
    """
    plist_dir = _home() / "Library" / "LaunchAgents"
    plist_path = plist_dir / "com.slowave.backup.plist"
    content = _LAUNCHD_BACKUP_PLIST.format(bin=slowave_bin)
    if plist_path.exists() and plist_path.read_text(encoding="utf-8") == content:
        return str(plist_path), False
    plist_dir.mkdir(parents=True, exist_ok=True)
    plist_path.write_text(content, encoding="utf-8")
    try:
        subprocess.run(["launchctl", "unload", str(plist_path)], capture_output=True, check=False)
        subprocess.run(["launchctl", "load", str(plist_path)], capture_output=True, check=False)
    except FileNotFoundError:
        pass
    return str(plist_path), True


def _install_backup_linux(slowave_bin: str) -> tuple[str, bool]:
    """Install the daily database backup as a systemd timer + oneshot service (Linux)."""
    xdg = os.environ.get("XDG_CONFIG_HOME", str(_home() / ".config"))
    svc_dir = Path(xdg) / "systemd" / "user"
    svc_path = svc_dir / "slowave-backup.service"
    timer_path = svc_dir / "slowave-backup.timer"
    svc_content = _SYSTEMD_BACKUP_SERVICE.format(bin=slowave_bin)
    timer_content = _SYSTEMD_BACKUP_TIMER
    svc_changed = not svc_path.exists() or svc_path.read_text(encoding="utf-8") != svc_content
    timer_changed = (
        not timer_path.exists() or timer_path.read_text(encoding="utf-8") != timer_content
    )
    if not svc_changed and not timer_changed:
        return str(timer_path), False
    svc_dir.mkdir(parents=True, exist_ok=True)
    svc_path.write_text(svc_content, encoding="utf-8")
    timer_path.write_text(timer_content, encoding="utf-8")
    try:
        subprocess.run(["systemctl", "--user", "daemon-reload"], capture_output=True, check=False)
        subprocess.run(
            ["systemctl", "--user", "enable", "--now", "slowave-backup.timer"],
            capture_output=True,
            check=False,
        )
    except FileNotFoundError:
        pass
    return str(timer_path), True


def _install_backup_windows(slowave_bin: str) -> tuple[str, bool]:
    """Register a daily database backup as a Windows Scheduled Task."""
    task_name = "SlowaveBackup"
    already_registered = False
    try:
        check = subprocess.run(
            [
                "powershell",
                "-NonInteractive",
                "-Command",
                f"$t = Get-ScheduledTask -TaskName '{task_name}' -ErrorAction SilentlyContinue; "
                f"if ($t) {{ $t.Actions[0].Execute + '|' + $t.Actions[0].Arguments }}",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        existing = check.stdout.strip()
        if existing:
            existing_exe, _, existing_args = existing.partition("|")
            if (
                existing_exe.strip().lower() == slowave_bin.lower()
                and "backup" in existing_args.lower()
            ):
                already_registered = True
    except FileNotFoundError:
        pass

    if already_registered:
        return task_name, False

    # Daily trigger at 03:00
    ps = (
        f"$a=New-ScheduledTaskAction -Execute '{slowave_bin}' -Argument 'backup';"
        f"$t=New-ScheduledTaskTrigger -Daily -At 03:00;"
        f"$s=New-ScheduledTaskSettingsSet -ExecutionTimeLimit 0;"
        f"$p=New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Limited;"
        f"Register-ScheduledTask -TaskName '{task_name}' -Action $a -Trigger $t -Settings $s "
        f"-Principal $p -Force"
    )
    try:
        subprocess.run(
            ["powershell", "-NonInteractive", "-Command", ps], capture_output=True, check=False
        )
    except FileNotFoundError:
        pass
    return task_name, True
