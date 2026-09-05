"""Removal commands for Slowave-managed configuration, services, and data."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
from pathlib import Path

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


def _remove_daemon_service(dry_run: bool) -> int:
    """Remove HTTP MCP daemon service. Returns 1 if removed, 0 otherwise."""
    if SYSTEM == "Darwin":
        plist_path = _home() / "Library" / "LaunchAgents" / "com.slowave.daemon.plist"
        if plist_path.exists():
            if dry_run:
                _ok(f"Would stop and remove: {plist_path}")
                return 0
            try:
                subprocess.run(
                    ["launchctl", "unload", str(plist_path)], check=False, capture_output=True
                )
                plist_path.unlink()
                _ok(f"Removed launchd daemon service: {plist_path}")
                return 1
            except Exception as e:
                _warn(f"Could not remove launchd daemon service: {e}")
        else:
            _skip("No launchd daemon service found")

    elif SYSTEM == "Linux":
        import os

        xdg = os.environ.get("XDG_CONFIG_HOME", str(_home() / ".config"))
        service_path = Path(xdg) / "systemd" / "user" / "slowave-daemon.service"
        if service_path.exists():
            if dry_run:
                _ok(f"Would stop and remove: {service_path}")
                return 0
            try:
                subprocess.run(
                    ["systemctl", "--user", "stop", "slowave-daemon"],
                    check=False,
                    capture_output=True,
                )
                subprocess.run(
                    ["systemctl", "--user", "disable", "slowave-daemon"],
                    check=False,
                    capture_output=True,
                )
                service_path.unlink()
                subprocess.run(
                    ["systemctl", "--user", "daemon-reload"], check=False, capture_output=True
                )
                _ok(f"Removed systemd daemon service: {service_path}")
                return 1
            except Exception as e:
                _warn(f"Could not remove systemd daemon service: {e}")
        else:
            _skip("No systemd daemon service found")

    elif SYSTEM == "Windows":
        if dry_run:
            _ok("Would remove Task Scheduler task: SlowaveDaemon")
            return 0
        try:
            subprocess.run(
                ["schtasks", "/Delete", "/TN", "SlowaveDaemon", "/F"],
                check=False,
                capture_output=True,
            )
            _ok("Removed Task Scheduler task: SlowaveDaemon")
            return 1
        except Exception as e:
            _warn(f"Could not remove scheduled task SlowaveDaemon: {e}")
    else:
        _skip(f"Unknown platform: {SYSTEM}")
    return 0


def _remove_worker_service(dry_run: bool) -> int:
    """Remove background worker service. Returns 1 if removed, 0 otherwise."""
    if SYSTEM == "Darwin":
        plist_path = _home() / "Library" / "LaunchAgents" / "com.slowave.worker.plist"
        if plist_path.exists():
            if dry_run:
                _ok(f"Would stop and remove: {plist_path}")
                return 0
            try:
                subprocess.run(
                    ["launchctl", "unload", str(plist_path)], check=False, capture_output=True
                )
                plist_path.unlink()
                _ok(f"Removed launchd service: {plist_path}")
                return 1
            except Exception as e:
                _warn(f"Could not remove launchd service: {e}")
        else:
            _skip("No launchd service found")

    elif SYSTEM == "Linux":
        service_path = _home() / ".config" / "systemd" / "user" / "slowave-worker.service"
        if service_path.exists():
            if dry_run:
                _ok(f"Would stop and remove: {service_path}")
                return 0
            try:
                subprocess.run(
                    ["systemctl", "--user", "stop", "slowave-worker"],
                    check=False,
                    capture_output=True,
                )
                subprocess.run(
                    ["systemctl", "--user", "disable", "slowave-worker"],
                    check=False,
                    capture_output=True,
                )
                service_path.unlink()
                subprocess.run(
                    ["systemctl", "--user", "daemon-reload"], check=False, capture_output=True
                )
                _ok(f"Removed systemd service: {service_path}")
                return 1
            except Exception as e:
                _warn(f"Could not remove systemd service: {e}")
        else:
            _skip("No systemd service found")

    elif SYSTEM == "Windows":
        if dry_run:
            _ok("Would remove Task Scheduler task: SlowaveWorker")
            return 0
        try:
            subprocess.run(
                ["schtasks", "/Delete", "/TN", "SlowaveWorker", "/F"],
                check=False,
                capture_output=True,
            )
            _ok("Removed Task Scheduler task: SlowaveWorker")
            return 1
        except Exception as e:
            _warn(f"Could not remove scheduled task: {e}")
    else:
        _skip(f"Unknown platform: {SYSTEM}")
    return 0


def _remove_backup_service(dry_run: bool) -> int:
    """Remove daily database backup service. Returns 1 if removed, 0 otherwise."""
    if SYSTEM == "Darwin":
        plist_path = _home() / "Library" / "LaunchAgents" / "com.slowave.backup.plist"
        if plist_path.exists():
            if dry_run:
                _ok(f"Would stop and remove: {plist_path}")
                return 0
            try:
                subprocess.run(
                    ["launchctl", "unload", str(plist_path)], check=False, capture_output=True
                )
                plist_path.unlink()
                _ok(f"Removed launchd backup service: {plist_path}")
                return 1
            except Exception as e:
                _warn(f"Could not remove launchd backup service: {e}")
        else:
            _skip("No launchd backup service found")

    elif SYSTEM == "Linux":
        svc_dir = _home() / ".config" / "systemd" / "user"
        timer_path = svc_dir / "slowave-backup.timer"
        svc_path = svc_dir / "slowave-backup.service"
        removed = 0
        for p, name in [(timer_path, "timer"), (svc_path, "service")]:
            if p.exists():
                if dry_run:
                    _ok(f"Would stop and remove: {p}")
                    continue
                try:
                    subprocess.run(
                        ["systemctl", "--user", "stop", f"slowave-backup.{name}"],
                        check=False,
                        capture_output=True,
                    )
                    subprocess.run(
                        ["systemctl", "--user", "disable", f"slowave-backup.{name}"],
                        check=False,
                        capture_output=True,
                    )
                    p.unlink()
                    _ok(f"Removed systemd backup {name}: {p}")
                    removed = 1
                except Exception as e:
                    _warn(f"Could not remove systemd backup {name}: {e}")
        if removed:
            try:
                subprocess.run(
                    ["systemctl", "--user", "daemon-reload"], check=False, capture_output=True
                )
            except Exception:
                pass
        if not timer_path.exists() and not svc_path.exists() and removed == 0:
            _skip("No systemd backup service found")
        return removed

    elif SYSTEM == "Windows":
        if dry_run:
            _ok("Would remove Task Scheduler task: SlowaveBackup")
            return 0
        try:
            subprocess.run(
                ["schtasks", "/Delete", "/TN", "SlowaveBackup", "/F"],
                check=False,
                capture_output=True,
            )
            _ok("Removed Task Scheduler task: SlowaveBackup")
            return 1
        except Exception as e:
            _warn(f"Could not remove scheduled task: {e}")
    else:
        _skip(f"Unknown platform: {SYSTEM}")
    return 0


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


def _remove_mcp_configs(dry_run: bool) -> int:
    """Remove MCP server entries and enforcement hooks from all client configs.

    Iterates ``_clients()`` — adding a new client in setup.py automatically
    includes it here.  Enforcement hook removal is also data-driven via
    ``spec.hooks_cleanup_fn``: no per-client special-cases needed.
    Returns the count of config files modified.
    """
    count = 0

    for spec in _clients():
        if spec.key == "codex":
            # Codex keeps the MCP entry and enforcement hooks in the same TOML
            # file — patch both against one loaded doc and write once, same
            # reasoning as the combined read/write in setup.py's setup loop.
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
            if spec.hooks_cleanup_fn is not None:
                cfg, hooks_changed = spec.hooks_cleanup_fn(cfg)
                if hooks_changed:
                    if dry_run:
                        _ok(f"Would remove slowave enforcement hooks from: {mcp_file}")
                    else:
                        changed = True
                        _ok(f"Removed slowave enforcement hooks from: {mcp_file}")
                else:
                    _skip(f"{spec.label}: no slowave enforcement hooks in {mcp_file}")
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

        # Enforcement hooks — data-driven via spec.hooks_cleanup_fn
        if spec.hooks_config_path is not None and spec.hooks_cleanup_fn is not None:
            hooks_file = spec.hooks_config_path()
            if not hooks_file.exists():
                _skip(f"{spec.label}: hooks file not found ({hooks_file})")
            else:
                hcfg = _read_json(hooks_file)
                hcfg, hooks_changed = spec.hooks_cleanup_fn(hcfg)
                if hooks_changed:
                    if dry_run:
                        _ok(f"Would remove slowave enforcement hooks from: {hooks_file}")
                    else:
                        _write_json(hooks_file, hcfg)
                        _ok(f"Removed slowave enforcement hooks from: {hooks_file}")
                else:
                    _skip(f"{spec.label}: no slowave enforcement hooks in {hooks_file}")

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
        if spec.hooks_config_path is not None:
            dirs.add(spec.hooks_config_path().parent)
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
    - MCP server configs, lifecycle blocks, and enforcement hooks for every supported client
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
            # process even after the scheduler task was deleted.  Attempt to kill
            # any lingering slowave processes before removing the directory.
            if SYSTEM == "Windows":
                try:
                    subprocess.run(
                        [
                            "powershell",
                            "-NonInteractive",
                            "-Command",
                            "Get-Process | Where-Object { $_.Path -like '*slowave*' } "
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
