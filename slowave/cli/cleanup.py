"""Removal commands for Slowave-managed configuration, services, and data."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import click

# Import helpers from setup
from slowave.cli.setup import (
    _MARKER_END,
    _MARKER_START,
    _clients,
    _home,
    _ok,
    _opencode_instructions_path,
    _read_json,
    _read_toml,
    _section,
    _skip,
    _strip_legacy_slowave_section,
    _warn,
    _write_json,
    _write_toml,
)
from slowave.core.paths import runtime_paths

SYSTEM = platform.system()

# These signatures identify hooks written by released versions before hooks
# were removed from setup.  Cleanup retains this migration path so uninstall
# cannot leave a broken `slowave hook codex-stop` command behind.
_LEGACY_HOOK_MARKER = "SLOWAVE MANDATORY"
_LEGACY_CODEX_STOP_COMMAND = "slowave hook codex-stop"


def _runtime_cleanup_targets() -> tuple[Path, list[Path], bool]:
    """Return root, removable targets, and whether the root is dedicated.

    A legacy ``SLOWAVE_DB`` may live in an arbitrary directory (including a
    project or ``/tmp``), so purge must never sweep that parent wholesale.
    """
    paths = runtime_paths()
    dedicated_root = "SLOWAVE_DB" not in os.environ
    if dedicated_root:
        targets = [item for item in paths.root.iterdir()] if paths.root.is_dir() else []
    else:
        targets = [
            paths.database,
            *(Path(f"{paths.database}{suffix}") for suffix in ("-wal", "-shm", "-journal", ".bak")),
            paths.pid_file,
            paths.logs_dir,
            paths.setup_sentinel,
            paths.judge_debug_log,
            paths.root / "config.toml",
        ]
    return paths.root, targets, dedicated_root


def _remove_service(kind: str, dry_run: bool) -> int:
    from slowave.cli.services import remove_service

    return remove_service(kind, dry_run=dry_run, system=SYSTEM, home=_home())


def _remove_daemon_service(dry_run: bool) -> int:
    return _remove_service("daemon", dry_run)


def _remove_worker_service(dry_run: bool) -> int:
    return _remove_service("worker", dry_run)


def _remove_backup_service(dry_run: bool) -> int:
    return _remove_service("backup", dry_run)


def _remove_lifecycle_blocks(dry_run: bool) -> int:
    """Remove lifecycle instruction blocks from all clients that support auto-injection.

    Iterates ``_clients()`` and processes every client whose ``lifecycle_path``
    is not None.  Removes both the marker-bounded block (all versions) and any
    legacy un-markered '## Slowave memory' section.  User content outside those
    sections is never touched.  Returns the count of files changed.
    """
    count = 0

    def _strip_file(path: Path) -> int:
        if not path.exists():
            return 0
        content = path.read_text(encoding="utf-8")
        new_content = content
        if _MARKER_START in new_content and _MARKER_END in new_content:
            start = new_content.index(_MARKER_START)
            # Advance past the full end-marker line (e.g. "<!-- slowave-lifecycle-end v2 -->")
            # Using only len(_MARKER_END) would leave the " v2 -->" suffix on the next line.
            end_marker_pos = new_content.index(_MARKER_END)
            end_of_line = new_content.find("\n", end_marker_pos)
            end = end_of_line + 1 if end_of_line != -1 else len(new_content)
            new_content = new_content[:start] + new_content[end:]
        new_content = _strip_legacy_slowave_section(new_content).lstrip("\n")
        if new_content == content:
            return 0
        if not new_content.strip():
            path.unlink()
            _ok(f"Removed (now empty): {path}")
        else:
            path.write_text(new_content, encoding="utf-8")
            _ok(f"Removed slowave block from: {path}")
        return 1

    for spec in _clients():
        if spec.lifecycle_path is None:
            continue
        lc_file = spec.lifecycle_path()
        if not lc_file.exists():
            _skip(f"{spec.label}: {lc_file} not found")
            continue
        content = lc_file.read_text(encoding="utf-8")
        has_marker = _MARKER_START in content and _MARKER_END in content
        has_legacy = "## Slowave memory" in content
        if has_marker or has_legacy:
            if dry_run:
                _ok(f"Would remove slowave block from: {lc_file}")
            else:
                count += _strip_file(lc_file)
        else:
            _skip(f"{spec.label}: no slowave content in {lc_file}")

    return count


def _remove_legacy_slowave_hooks(config: Any) -> tuple[Any, bool]:
    """Remove only legacy Slowave hooks from a JSON or TOML client config.

    Older releases installed hooks under ``UserPromptSubmit`` and ``Stop``.
    New setup no longer writes hooks, but uninstall and purge must remove these
    historic entries.  Preserve unrelated hooks, including hooks in the same
    event group.
    """
    if not isinstance(config, dict):
        return config, False

    hooks = config.get("hooks")
    if not isinstance(hooks, dict):
        return config, False

    changed = False
    for event in ("UserPromptSubmit", "Stop"):
        groups = hooks.get(event)
        if not isinstance(groups, list):
            continue
        retained_groups = []
        for group in groups:
            if not isinstance(group, dict):
                retained_groups.append(group)
                continue
            entries = group.get("hooks")
            if not isinstance(entries, list):
                retained_groups.append(group)
                continue
            retained_entries = [
                entry
                for entry in entries
                if not (
                    isinstance(entry, dict)
                    and (
                        _LEGACY_HOOK_MARKER in str(entry.get("command", ""))
                        or _LEGACY_CODEX_STOP_COMMAND in str(entry.get("command", ""))
                    )
                )
            ]
            if retained_entries == entries:
                retained_groups.append(group)
                continue
            changed = True
            if retained_entries:
                group["hooks"] = retained_entries
                retained_groups.append(group)
        if retained_groups != groups:
            hooks[event] = retained_groups

    return config, changed


def _remove_legacy_hook_file(path: Path, *, toml: bool, dry_run: bool) -> int:
    """Remove hooks written by prior releases from one known configuration file."""
    if not path.exists():
        return 0
    config = _read_toml(path) if toml else _read_json(path)
    config, changed = _remove_legacy_slowave_hooks(config)
    if not changed:
        return 0
    if dry_run:
        _ok(f"Would remove legacy Slowave hooks from: {path}")
        return 0
    if toml:
        _write_toml(path, config)
    else:
        _write_json(path, config)
    _ok(f"Removed legacy Slowave hooks from: {path}")
    return 1


def _remove_mcp_configs(dry_run: bool) -> int:
    """Remove MCP server entries and legacy Slowave hooks from client configs.

    Iterates ``_clients()`` — adding a new client in setup.py automatically
    includes it here. Returns the count of config files modified.
    """
    count = 0

    for spec in _clients():
        if spec.key == "codex":
            # Codex stores its MCP entry in a TOML configuration file.
            mcp_file = spec.mcp_path()
            if not mcp_file.exists():
                _skip(f"{spec.label}: {mcp_file} not found")
                continue
            cfg = _read_toml(mcp_file)
            changed = False
            if "mcp_servers" not in cfg or "slowave" not in cfg["mcp_servers"]:
                _skip(f"{spec.label}: no slowave entry in {mcp_file}")
            else:
                if dry_run:
                    _ok(f"Would remove slowave MCP entry from: {mcp_file}")
                else:
                    del cfg["mcp_servers"]["slowave"]
                    changed = True
                    _ok(f"Removed slowave MCP entry from: {mcp_file}")
            if changed and not dry_run:
                _write_toml(mcp_file, cfg)
                count += 1
            continue

        # MCP entry
        mcp_file = spec.mcp_path()
        if not mcp_file.exists():
            _skip(f"{spec.label}: {mcp_file} not found")
        else:
            cfg = _read_json(mcp_file)
            # OpenCode uses `mcp` key; other clients use `mcpServers`
            if spec.key == "opencode":
                if "mcp" not in cfg or "slowave" not in cfg["mcp"]:
                    _skip(f"{spec.label}: no slowave entry in {mcp_file}")
                else:
                    if dry_run:
                        _ok(f"Would remove slowave MCP entry from: {mcp_file}")
                    else:
                        del cfg["mcp"]["slowave"]
                        _write_json(mcp_file, cfg)
                        _ok(f"Removed slowave MCP entry from: {mcp_file}")
                        count += 1
                    # Also remove instructions entry
                    if "instructions" in cfg:
                        inst_path = str(_opencode_instructions_path().resolve())
                        instructions = cfg.get("instructions", [])
                        if inst_path in instructions:
                            instructions.remove(inst_path)
                            if not instructions:
                                del cfg["instructions"]
                            if not dry_run:
                                _write_json(mcp_file, cfg)
                                _ok(f"Removed slowave instructions entry from: {mcp_file}")
                            else:
                                _ok(f"Would remove slowave instructions entry from: {mcp_file}")
            elif "mcpServers" not in cfg or "slowave" not in cfg["mcpServers"]:
                _skip(f"{spec.label}: no slowave entry in {mcp_file}")
            else:
                if dry_run:
                    _ok(f"Would remove slowave MCP entry from: {mcp_file}")
                else:
                    del cfg["mcpServers"]["slowave"]
                    _write_json(mcp_file, cfg)
                    _ok(f"Removed slowave MCP entry from: {mcp_file}")
                    count += 1

    # Hooks are no longer installed, but prior releases wrote them to these
    # files.  Keep this migration cleanup separate from ClientSpec so the
    # current setup surface has no hook-installation knowledge.
    count += _remove_legacy_hook_file(
        _home() / ".claude" / "settings.json", toml=False, dry_run=dry_run
    )
    count += _remove_legacy_hook_file(
        _home() / ".codex" / "config.toml", toml=True, dry_run=dry_run
    )
    return count


def _remove_setup_backups(dry_run: bool) -> int:
    """Remove ``*.bak.*`` files left by _backup_file() next to config files.

    The directory list is derived directly from the same path-helper functions
    used during setup, so it is always complete regardless of platform.

    Returns the number of backup files removed.
    """
    count = 0
    # Build the set of directories that may contain .bak.* files directly
    # from the ClientSpec fields — no manual list to maintain.
    dirs: set[Path] = set()
    for spec in _clients():
        dirs.add(spec.mcp_path().parent)
        if spec.lifecycle_path is not None:
            dirs.add(spec.lifecycle_path().parent)
    candidates: list[Path] = sorted(dirs)
    for directory in candidates:
        if not directory.is_dir():
            continue
        for bak in sorted(directory.glob("*.bak.*")):
            if dry_run:
                _ok(f"Would remove backup: {bak}")
            else:
                try:
                    bak.unlink()
                    _ok(f"Removed backup: {bak}")
                    count += 1
                except OSError as exc:
                    _warn(f"Could not remove {bak}: {exc}")
    if count == 0 and not dry_run:
        _skip("No setup backup files found")
    return count


@click.command("purge")
@click.option(
    "--dry-run", is_flag=True, help="Preview what would be removed without changing files."
)
@click.option("--json", "as_json", is_flag=True, help="Machine-readable JSON output.")
@click.option("--yes", is_flag=True, help="Confirm permanent removal without prompting.")
def cleanup_cmd(dry_run: bool, as_json: bool = False, yes: bool = False) -> None:
    """Permanently remove all Slowave configuration and local data.

    This command removes everything that 'slowave setup' installed:
    - MCP server configs and lifecycle blocks for every supported client
    - HTTP daemon, background worker, and daily backup services
    - Local database and data in the effective runtime root (database archives are retained)
    - Setup-created *.bak.* configuration backups

    Use 'slowave uninstall' instead to remove integrations while keeping memories.

    \\b
    Example:
      slowave purge              # interactive confirmation
      slowave purge --dry-run    # preview without removing
    """
    if not dry_run and not yes:
        click.confirm(
            "This will permanently remove Slowave configuration and local data. Continue?",
            abort=True,
        )

    click.echo(click.style("\nSlowave purge", bold=True))
    if dry_run:
        click.echo(click.style("  [DRY RUN — no files will be removed]\n", fg="yellow"))

    removed_count = 0

    # 1. Stop and remove HTTP MCP daemon service
    _section("1. HTTP MCP daemon service")
    removed_count += _remove_daemon_service(dry_run)

    # 2. Stop and remove background worker service
    _section("2. Background worker service")
    removed_count += _remove_worker_service(dry_run)

    # 3. Stop and remove daily backup service
    _section("3. Daily database backup service")
    removed_count += _remove_backup_service(dry_run)

    if not dry_run:
        from slowave.cli.services import require_stopped_runtime

        require_stopped_runtime("Purge")

    # 4. Remove lifecycle blocks
    _section("4. Lifecycle instruction blocks")
    removed_count += _remove_lifecycle_blocks(dry_run)

    # 5. Remove MCP server configs
    _section("5. MCP server configurations")
    removed_count += _remove_mcp_configs(dry_run)

    # 7. Remove data directory
    _section("6. Local data and database")
    slowave_dir, cleanup_targets, dedicated_root = _runtime_cleanup_targets()
    if slowave_dir.exists():
        if dry_run:
            if dedicated_root:
                _ok(f"Would remove runtime data in: {slowave_dir}")
            else:
                _ok(f"Would remove only known Slowave artifacts in: {slowave_dir}")
        else:
            # On Windows the DB may still be held open by a running worker or MCP
            # process even after the scheduler task was deleted. require_stopped_runtime()
            # above is the primary guard; this only force-stops lingering processes
            # from this exact Python environment (e.g. a pythonw.exe the supervisor
            # reported stopped but that is still shutting down), never arbitrary
            # processes that merely have "slowave" somewhere in their path.
            if SYSTEM == "Windows":
                try:
                    from pathlib import Path as _Path

                    targets = {_Path(sys.executable).resolve()}
                    pythonw = _Path(sys.executable).with_name("pythonw.exe")
                    if pythonw.exists():
                        targets.add(pythonw.resolve())
                    target_list = ", ".join(
                        "'" + str(path).replace("'", "''") + "'" for path in sorted(targets)
                    )
                    subprocess.run(
                        [
                            "powershell",
                            "-NonInteractive",
                            "-Command",
                            "$targets=@(" + target_list + ");"
                            "Get-Process | Where-Object { $_.Path -and ($targets -icontains $_.Path) } "
                            "| Stop-Process -Force -ErrorAction SilentlyContinue",
                        ],
                        capture_output=True,
                        check=False,
                        timeout=5,
                    )
                    import time as _time

                    _time.sleep(0.6)
                except Exception:
                    pass

            # Preserve the backups/ subdirectory if it exists and has content.
            backups_dir = slowave_dir / "backups"
            backups_exist = backups_dir.is_dir() and any(backups_dir.iterdir())
            if backups_exist:
                backup_files = sorted(backups_dir.glob("slowave-????????_??????.db.gz"))
                backup_list = "\n    ".join(p.name for p in backup_files[-5:])
                if len(backup_files) > 5:
                    backup_list = f"... and {len(backup_files) - 5} more\n    " + backup_list

            try:
                # Remove a dedicated SLOWAVE_HOME/default tree wholesale, but
                # only known Slowave artifacts when legacy SLOWAVE_DB makes an
                # arbitrary parent directory the coherence root.
                for item in sorted(cleanup_targets):
                    if not item.exists():
                        continue
                    if item.name == "backups" and backups_exist:
                        continue
                    if item.is_dir():
                        shutil.rmtree(item)
                    else:
                        item.unlink()
            except OSError as exc:
                _warn(
                    f"Could not clean {slowave_dir}: {exc.strerror}.\n"
                    "  The database may still be in use by a running worker or MCP process.\n"
                    "  Stop those processes, then re-run 'slowave purge'."
                )
            else:
                if backups_exist:
                    _warn(
                        f"Preserved {len(backup_files)} database backup(s) in {backups_dir}:\n"
                        f"    {backup_list}\n"
                        f"  To remove them, delete the directory manually:\n"
                        f"    rm -rf {backups_dir}"
                    )
                if dedicated_root:
                    try:
                        slowave_dir.rmdir()
                    except OSError:
                        pass
                _ok(f"Cleaned Slowave runtime data in: {slowave_dir}")
                removed_count += 1
    else:
        _skip(f"No runtime data directory found at {slowave_dir}")

    # 7. Remove setup backup files
    _section("7. Setup backup files")
    removed_count += _remove_setup_backups(dry_run)

    # Summary
    click.echo()
    if dry_run:
        click.echo(click.style("Dry run complete. No files were removed.", bold=True))
    else:
        click.echo(click.style(f"Purge complete. {removed_count} items removed.", bold=True))
        click.echo("\nManual removal still needed:")
        click.echo("  • Claude Desktop → Settings → General → Instructions for Claude")
        click.echo("    (Remove any slowave lifecycle instructions)")
        click.echo("\nYou can now safely run: pipx uninstall slowave")
