"""Explicit migration from the legacy ``~/.slowave`` runtime tree."""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import click

from slowave.core.paths import default_runtime_root


@dataclass(frozen=True)
class MigrationPlan:
    source_root: Path
    destination_root: Path
    source_database: Path
    destination_database: Path
    needed: bool
    reason: str


def plan_migration(*, home: Path | None = None, destination: Path | None = None) -> MigrationPlan:
    """Plan legacy migration without considering Slowave path overrides."""
    selected_home = Path.home() if home is None else Path(home)
    source = (selected_home / ".slowave").resolve(strict=False)
    target = (
        default_runtime_root() if destination is None else Path(destination).resolve(strict=False)
    )
    source_db = source / "slowave.db"
    target_db = target / "slowave.db"

    if source == target:
        return MigrationPlan(source, target, source_db, target_db, False, "roots are equivalent")
    if not source_db.is_file():
        return MigrationPlan(
            source, target, source_db, target_db, False, "legacy database not found"
        )
    return MigrationPlan(source, target, source_db, target_db, True, "legacy database is available")


def _destination_is_empty(path: Path) -> bool:
    return not path.exists() or (path.is_dir() and next(path.iterdir(), None) is None)


def _sqlite_backup(source: Path, destination: Path) -> None:
    source_uri = f"{source.resolve(strict=True).as_uri()}?mode=ro"
    source_conn = sqlite3.connect(source_uri, uri=True, timeout=30.0)
    try:
        destination_conn = sqlite3.connect(destination, timeout=30.0)
        try:
            source_conn.backup(destination_conn)
        finally:
            destination_conn.close()
    finally:
        source_conn.close()


def _validate_database(path: Path) -> None:
    uri = f"{path.resolve(strict=True).as_uri()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=30.0)
    try:
        result = conn.execute("PRAGMA integrity_check").fetchone()
    finally:
        conn.close()
    if result != ("ok",):
        detail = result[0] if result else "no result"
        raise click.ClickException(f"Staged database failed SQLite integrity_check: {detail}")


def _stop_legacy_daemon(pid_file: Path, *, timeout_s: float = 5.0) -> bool:
    from slowave.mcp.daemon import is_running, stop_daemon

    if not is_running(pid_file):
        return False
    if not stop_daemon(pid_file):
        raise click.ClickException("Could not stop the daemon using the legacy runtime root")
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if not is_running(pid_file):
            return True
        time.sleep(0.1)
    raise click.ClickException(
        f"Legacy daemon is still running; stop it before migrating ({pid_file})"
    )


def execute_migration(
    plan: MigrationPlan,
    *,
    validate: Callable[[Path], None] = _validate_database,
) -> dict[str, object]:
    """Execute a staged, validated migration and preserve the legacy source."""
    if not plan.needed:
        return {"migrated": False, "reason": plan.reason}
    if not _destination_is_empty(plan.destination_root):
        raise click.ClickException(
            f"Destination is not empty: {plan.destination_root}. "
            "Move it aside or choose an explicit rollback; automatic merging is not supported."
        )

    daemon_stopped = _stop_legacy_daemon(plan.source_root / "daemon.pid")
    plan.destination_root.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{plan.destination_root.name}-migrate-",
            dir=str(plan.destination_root.parent),
        )
    )
    promoted = False
    try:
        if os.name != "nt":
            staging.chmod(0o700)
        staged_db = staging / "slowave.db"
        _sqlite_backup(plan.source_database, staged_db)
        validate(staged_db)

        copied: list[str] = []
        for name in ("backups", "logs"):
            source_item = plan.source_root / name
            if source_item.is_dir():
                shutil.copytree(source_item, staging / name)
                copied.append(name)
        for name in (".setup_done", "judge_debug.jsonl"):
            source_item = plan.source_root / name
            if source_item.is_file():
                shutil.copy2(source_item, staging / name)
                copied.append(name)

        if plan.destination_root.exists():
            plan.destination_root.rmdir()  # proven empty above
        os.replace(staging, plan.destination_root)
        promoted = True
        return {
            "migrated": True,
            "source_root": str(plan.source_root),
            "destination_root": str(plan.destination_root),
            "database": str(plan.destination_database),
            "copied_artifacts": copied,
            "legacy_preserved": True,
            "daemon_stopped": daemon_stopped,
            "rollback": f"SLOWAVE_HOME={plan.source_root}",
        }
    finally:
        if not promoted and staging.exists():
            shutil.rmtree(staging, ignore_errors=True)


@click.command("migrate-data")
@click.option("--yes", is_flag=True, help="Run without an interactive confirmation.")
@click.option("--dry-run", is_flag=True, help="Show the migration plan without changing files.")
@click.option("--json", "as_json", is_flag=True, help="Machine-readable JSON output.")
def migrate_data_cmd(yes: bool, dry_run: bool, as_json: bool) -> None:
    """Migrate legacy ~/.slowave data to the native per-user data directory."""
    plan = plan_migration()
    plan_payload = {
        key: str(value) if isinstance(value, Path) else value for key, value in asdict(plan).items()
    }

    if dry_run or not plan.needed:
        payload = {"status": "planned" if plan.needed else "not_needed", **plan_payload}
        click.echo(json.dumps(payload, indent=2) if as_json else _migration_text(payload))
        return

    if not _destination_is_empty(plan.destination_root):
        raise click.ClickException(
            f"Destination is not empty: {plan.destination_root}. Automatic merging is not supported."
        )

    if not yes:
        click.echo("Slowave data migration")
        click.echo(f"  source      : {plan.source_root}")
        click.echo(f"  destination : {plan.destination_root}")
        click.echo("  The legacy source will be preserved; the legacy daemon may be stopped.")
        click.confirm("Continue?", abort=True, default=False)

    result = execute_migration(plan)
    if as_json:
        click.echo(json.dumps(result, indent=2))
    else:
        click.echo("Migration complete and SQLite integrity verified.")
        click.echo(f"  new root : {result['destination_root']}")
        click.echo(f"  legacy   : {result['source_root']} (preserved)")
        click.echo(f"  rollback : {result['rollback']} slowave doctor")


def _migration_text(payload: dict[str, object]) -> str:
    if payload["status"] == "not_needed":
        return f"No migration needed: {payload['reason']}."
    return (
        "Migration plan (no changes made):\n"
        f"  source      : {payload['source_root']}\n"
        f"  destination : {payload['destination_root']}\n"
        "  policy      : destination must be empty; legacy source is preserved"
    )
