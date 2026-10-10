"""Database backup command.

Uses SQLite's built-in .backup() API for consistent online snapshots — no WAL
checkpoint coordination needed, no blocking of concurrent readers. Backups are
gzip-compressed (SQLite DBs typically compress 5–10x).
"""

from __future__ import annotations

import gzip
import logging
import os
import shutil
import sqlite3
import tempfile
import time
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import click

from slowave.cli.output import safe_emoji
from slowave.core.paths import runtime_paths

log = logging.getLogger(__name__)

ENV_BACKUP_DIR = "SLOWAVE_BACKUP_DIR"
ENV_BACKUP_KEEP = "SLOWAVE_BACKUP_KEEP"
DEFAULT_KEEP = 7


def _default_backup_dir() -> Path:
    if ENV_BACKUP_DIR in os.environ:
        return Path(os.environ[ENV_BACKUP_DIR]).expanduser()
    return runtime_paths().backups_dir


def _resolve_keep(keep: int | None) -> int:
    if keep is not None and keep > 0:
        return keep
    env_val = os.environ.get(ENV_BACKUP_KEEP, "").strip()
    if env_val:
        try:
            parsed = int(env_val)
            if parsed > 0:
                return parsed
        except ValueError:
            pass
    return DEFAULT_KEEP


def _backup_name(ts: float) -> str:
    """Generate a backup filename for a given timestamp."""
    dt = datetime.fromtimestamp(ts, tz=timezone.utc)
    return f"slowave-{dt.strftime('%Y%m%d_%H%M%S')}.db.gz"


def _list_existing_backups(backup_dir: Path) -> list[Path]:
    """Return existing compressed backup files sorted oldest-first."""
    if not backup_dir.exists():
        return []
    files = sorted(
        backup_dir.glob("slowave-????????_??????.db.gz"),
        key=lambda p: p.stat().st_mtime,
    )
    return files


def _rotate_backups(backup_dir: Path, keep: int) -> list[str]:
    """Remove oldest backups beyond *keep* count, plus any legacy uncompressed
    .db backups from before compression was added. Returns paths of removed files."""
    removed: list[str] = []

    # Clean up legacy uncompressed .db backups (pre-compression format).
    for p in sorted(backup_dir.glob("slowave-????????_??????.db")):
        try:
            p.unlink()
            removed.append(str(p))
        except OSError as exc:
            log.warning("could not remove legacy backup %s: %s", p, exc)

    # Rotate compressed backups.
    existing = _list_existing_backups(backup_dir)
    if len(existing) > keep:
        to_remove = existing[: len(existing) - keep]
        for p in to_remove:
            try:
                p.unlink()
                removed.append(str(p))
            except OSError as exc:
                log.warning("could not remove old backup %s: %s", p, exc)
    return removed


def run_backup(
    *,
    db_path: str,
    backup_dir: Path,
    keep: int,
) -> dict[str, Any]:
    """Execute one backup pass.

    Returns a dict suitable for JSON or human-readable output.
    """
    src = os.path.abspath(os.path.expanduser(db_path))
    if not os.path.isfile(src):
        raise click.ClickException(f"Database not found: {src}")

    backup_dir.mkdir(parents=True, exist_ok=True)

    now = time.time()
    name = _backup_name(now)
    dst = str(backup_dir / name)

    src_size_before = os.path.getsize(src)
    started = time.monotonic()

    # Phase 1: create an uncompressed SQLite backup via .backup() API.
    # Use a temp file so we never leave a partial .db.gz on disk.
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False, dir=str(backup_dir))
    tmp_path = tmp.name
    tmp.close()  # Windows must release the handle before SQLite opens/removes it.
    try:
        src_conn = sqlite3.connect(f"file:{src}?mode=ro", uri=True, timeout=30.0)
        try:
            dst_conn = sqlite3.connect(tmp_path, timeout=30.0)
            try:
                src_conn.backup(dst_conn)
            finally:
                dst_conn.close()
        finally:
            src_conn.close()

        uncompressed_size = os.path.getsize(tmp_path)

        # Phase 2: gzip-compress the backup in place.
        with open(tmp_path, "rb") as f_in:
            with gzip.open(dst, "wb", compresslevel=6) as f_out:
                shutil.copyfileobj(f_in, f_out, length=1024 * 1024)
    finally:
        # Always remove the temp file.
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    elapsed_ms = int((time.monotonic() - started) * 1000)
    dst_size = os.path.getsize(dst)
    compression_ratio = uncompressed_size / dst_size if dst_size > 0 else 0.0

    # Rotate old backups (only after a successful write).
    removed = _rotate_backups(backup_dir, keep)

    return {
        "source": src,
        "source_size_bytes": src_size_before,
        "backup_path": dst,
        "backup_size_bytes": dst_size,
        "uncompressed_size_bytes": uncompressed_size,
        "compression_ratio": round(compression_ratio, 1),
        "elapsed_ms": elapsed_ms,
        "rotation_removed": removed,
        "keep_policy": keep,
    }


@click.command("backup")
@click.option(
    "--dir",
    "backup_dir",
    default=None,
    help=f"Backup directory (default: runtime root/backups; env: {ENV_BACKUP_DIR}).",
)
@click.option(
    "--keep",
    type=int,
    default=None,
    help=f"Number of backups to retain (default: {DEFAULT_KEEP}; env: {ENV_BACKUP_KEEP}).",
)
@click.option("--json", "as_json", is_flag=True, help="JSON output.")
@click.pass_context
def backup_cmd(
    ctx: click.Context,
    backup_dir: str | None,
    keep: int | None,
    as_json: bool,
) -> None:
    """Create a consistent, gzip-compressed backup of the Slowave database.

    Uses SQLite's online backup API — safe while the worker or MCP server
    are running. Old backups are rotated automatically; the default policy
    keeps the 7 most recent backups (one week of daily snapshots).

    SQLite databases compress extremely well with gzip (typically 5–10x),
    so a 500 MB database produces a ~50 MB backup.

    \b
    Examples:
      slowave backup                         # backup with defaults (keep 7)
      slowave backup --dir ~/Dropbox/slowave  # custom backup location
      slowave backup --keep 14               # retain last 14 backups
      slowave backup --json                  # machine-readable output
    """
    keep_val = _resolve_keep(keep)

    if backup_dir:
        bd = Path(backup_dir).expanduser()
    else:
        bd = _default_backup_dir()

    db_path = ctx.obj["db"]
    result = run_backup(db_path=db_path, backup_dir=bd, keep=keep_val)

    if as_json:
        import json

        click.echo(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    else:
        click.echo(
            click.style(f"  {safe_emoji('💾', '[ok]')} backup complete", fg="green", bold=True)
        )
        click.echo(f"     source : {result['source']} ({_fmt_bytes(result['source_size_bytes'])})")
        click.echo(
            f"     target : {result['backup_path']} ({_fmt_bytes(result['backup_size_bytes'])}"
            f" ← {_fmt_bytes(result['uncompressed_size_bytes'])} uncompressed,"
            f" {result['compression_ratio']}× smaller)"
        )
        click.echo(f"     time   : {result['elapsed_ms']} ms")
        if result["rotation_removed"]:
            for r in result["rotation_removed"]:
                click.echo(f"     removed: {r}")
        click.echo(f"     policy : keep last {result['keep_policy']}")


@click.command("restore")
@click.argument("backup_path", type=click.Path(exists=True, dir_okay=False))
@click.option("--yes", is_flag=True, help="Skip confirmation prompt.")
@click.option("--json", "as_json", is_flag=True, help="JSON output.")
@click.pass_context
def restore_cmd(
    ctx: click.Context,
    backup_path: str,
    yes: bool,
    as_json: bool,
) -> None:
    """Restore a Slowave database from a compressed backup.

    \\b
    BACKUP_PATH must be a slowave-*.db.gz file created by 'slowave backup'.

    Stop services with `slowave stop` and close foreground processes first.
    Restore refuses active writers, validates the backup before replacing the
    database, and preserves the previous database as slowave.db.bak.
    Run `slowave start` afterward to resume installed services.

    \\b
    Examples:
      slowave restore /path/to/runtime/backups/slowave-20260615_120000.db.gz
      slowave restore /path/to/runtime/backups/slowave-20260615_120000.db.gz --yes
    """
    db_path = ctx.obj["db"]
    dest = Path(db_path).expanduser().resolve()
    src = Path(backup_path).expanduser().resolve()

    if not src.suffixes == [".db", ".gz"]:
        raise click.BadParameter(f"Expected a .db.gz backup file, got: {src.name}")

    if not yes:
        click.echo(click.style("  ⚠  Restore database backup", fg="yellow", bold=True))
        click.echo(f"     from : {src}")
        click.echo(f"     to   : {dest}")
        click.echo(
            click.style(
                f"\n  This will REPLACE your current database. Any data not in the\n"
                f"  backup will be lost. A backup of the current database will be\n"
                f"  saved to {dest}.bak before the swap.\n",
                fg="yellow",
            )
        )
        click.confirm("  Continue?", abort=True, default=False)

    started = time.monotonic()

    # Never signal workers by name: they may belong to another runtime, and
    # KeepAlive/Restart policies can immediately respawn them during the swap.
    from slowave.cli.main import _slowave_processes
    from slowave.mcp.daemon import is_running

    writers = [
        p
        for p in _slowave_processes()
        if "slowave worker" in p.get("command", "") or "slowave dashboard" in p.get("command", "")
    ]
    if is_running() or writers:
        raise click.ClickException(
            "Restore requires stopped services and foreground processes. Run slowave stop, "
            "close foreground workers/dashboards, then retry. No database files were changed."
        )

    dest.parent.mkdir(parents=True, exist_ok=True)
    bak_path = Path(str(dest) + ".bak")
    tmp_dest: str | None = None
    try:
        # Stage and validate the entire replacement before touching the current
        # database or its WAL. A truncated or non-SQLite backup is harmless.
        fd, tmp_dest = tempfile.mkstemp(
            prefix=".slowave-restore-", suffix=".db", dir=str(dest.parent)
        )
        os.close(fd)
        with gzip.open(src, "rb") as f_in, open(tmp_dest, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out, length=1024 * 1024)
        with closing(sqlite3.connect(Path(tmp_dest).as_uri() + "?mode=ro", uri=True)) as conn:
            if (
                conn.execute("PRAGMA quick_check").fetchall() != [("ok",)]
                or not conn.execute("SELECT count(*) FROM sqlite_master").fetchone()[0]
            ):
                raise click.ClickException("Restored file failed SQLite integrity validation")

        if dest.exists():
            # SQLite's backup API includes committed WAL data. Copying only the
            # main file after deleting its WAL silently loses recent writes.
            with closing(sqlite3.connect(dest.as_uri() + "?mode=ro", uri=True)) as current:
                with closing(sqlite3.connect(str(bak_path))) as previous:
                    current.backup(previous)
            # Writers must be stopped; refuse a reader/writer that prevents a
            # complete checkpoint rather than deleting an in-use WAL.
            with closing(sqlite3.connect(str(dest), timeout=1)) as current:
                busy, _, _ = current.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
                if busy:
                    raise click.ClickException(
                        "Database is still in use; close all readers/writers"
                    )

        os.replace(tmp_dest, dest)
        tmp_dest = None
        for suffix in ("-wal", "-shm", "-journal"):
            Path(str(dest) + suffix).unlink(missing_ok=True)
    except (OSError, EOFError, sqlite3.Error) as exc:
        raise click.ClickException(f"Restore failed before completion: {exc}") from exc
    finally:
        if tmp_dest is not None:
            Path(tmp_dest).unlink(missing_ok=True)

    elapsed_ms = int((time.monotonic() - started) * 1000)
    src_size = src.stat().st_size
    dest_size = dest.stat().st_size

    # Recreate the backups directory in case it was missing.
    _default_backup_dir().mkdir(parents=True, exist_ok=True)

    result = {
        "restored_from": str(src),
        "restored_to": str(dest),
        "source_size_bytes": src_size,
        "restored_size_bytes": dest_size,
        "elapsed_ms": elapsed_ms,
        "daemon_was_stopped": False,
        "worker_pids_stopped": [],
        "previous_database": str(bak_path) if bak_path.exists() else None,
    }

    if as_json:
        import json as _json

        click.echo(_json.dumps(result, ensure_ascii=False, indent=2, default=str))
    else:
        click.echo(click.style("  ✓ restore complete", fg="green", bold=True))
        click.echo(f"     from : {src} ({_fmt_bytes(src_size)} compressed)")
        click.echo(f"     to   : {dest} ({_fmt_bytes(dest_size)} restored)")
        click.echo(f"     time : {elapsed_ms} ms")
        click.echo(
            "     note : previous database preserved as .bak; run slowave start to resume services"
        )


def _fmt_bytes(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f} MB"
    if n >= 1_000:
        return f"{n / 1_000:.1f} kB"
    return f"{n} B"
