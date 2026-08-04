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
import signal
import sqlite3
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import click

from slowave.cli.output import safe_emoji

log = logging.getLogger(__name__)

ENV_BACKUP_DIR = "SLOWAVE_BACKUP_DIR"
ENV_BACKUP_KEEP = "SLOWAVE_BACKUP_KEEP"
DEFAULT_KEEP = 7


def _default_backup_dir() -> Path:
    if ENV_BACKUP_DIR in os.environ:
        return Path(os.environ[ENV_BACKUP_DIR]).expanduser()
    return Path.home() / ".slowave" / "backups"


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
    help=f"Backup directory (default: ~/.slowave/backups; env: {ENV_BACKUP_DIR}).",
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

    This stops the running worker, replaces the current database with the
    backup, then restarts the worker. The previous database is backed up
    in-place as slowave.db.bak before the swap.

    \\b
    Examples:
      slowave restore ~/.slowave/backups/slowave-20260615_120000.db.gz
      slowave restore ~/.slowave/backups/slowave-20260615_120000.db.gz --yes
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

    # Stop the daemon so the DB isn't held open.
    daemon_stopped = False
    try:
        from slowave.mcp.daemon import is_running as _daemon_running
        from slowave.mcp.daemon import stop_daemon as _stop_daemon

        if _daemon_running():
            _stop_daemon()
            import time as _time

            _time.sleep(0.5)
            daemon_stopped = True
    except Exception:
        pass

    # Stop the background consolidation worker too, if one is detected. It
    # isn't managed via daemon.py's PID-file convention -- it typically runs
    # under an external supervisor (e.g. launchd's com.slowave.worker) -- so
    # process detection is all restore has to go on. Leaving it running
    # during the file swap below is the real risk: it can hold an open
    # WAL-mode connection to the pre-restore file, see the rewritten bytes
    # mid-copy, and then have its in-memory WAL state invalidated the moment
    # the sidecar-clearing below deletes -wal/-shm for what is now a
    # different database on disk.
    worker_stopped_pids: list[int] = []
    try:
        from slowave.cli.main import _slowave_processes

        for p in _slowave_processes():
            if "slowave worker" in p.get("command", ""):
                try:
                    os.kill(int(p["pid"]), signal.SIGTERM)
                    worker_stopped_pids.append(int(p["pid"]))
                except OSError:
                    pass
        if worker_stopped_pids:
            time.sleep(0.5)
    except Exception:
        pass

    # WAL/SHM sidecars are keyed by the main db file's path, not its content.
    # Leftovers from the pre-restore DB would otherwise get replayed against
    # the just-restored file — a structural mismatch SQLite reports as
    # "database disk image is malformed". Clear them before touching dest.
    sidecars = [Path(str(dest) + suf) for suf in ("-wal", "-shm", "-journal")]

    def _clear_sidecars() -> None:
        for p in sidecars:
            try:
                p.unlink()
            except FileNotFoundError:
                pass

    _clear_sidecars()

    # Backup current DB before overwriting.
    bak_path = Path(str(dest) + ".bak")
    if dest.exists():
        try:
            shutil.copy2(dest, bak_path)
        except OSError as exc:
            raise click.ClickException(f"Could not backup current database: {exc}")

    # Decompress into a temp file in the same directory as dest (so the final
    # os.replace() is an atomic rename on the same filesystem), then swap it
    # in as one step. Writing straight into `dest` via open(dest, "wb") would
    # let any process still holding the old file open (a worker that didn't
    # stop in time, a lingering reader) observe a partially-rewritten file
    # mid-copy instead of either the old or the new content.
    tmp_dest: str | None = None
    try:
        fd, tmp_dest = tempfile.mkstemp(
            prefix=".slowave-restore-", suffix=".db", dir=str(dest.parent)
        )
        os.close(fd)
        with gzip.open(src, "rb") as f_in:
            with open(tmp_dest, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out, length=1024 * 1024)
        os.replace(tmp_dest, dest)
        tmp_dest = None
    except OSError as exc:
        if tmp_dest is not None:
            try:
                os.unlink(tmp_dest)
            except OSError:
                pass
        # Try to restore the .bak if the write failed part-way.
        if bak_path.exists() and not dest.exists():
            shutil.move(str(bak_path), str(dest))
        elif bak_path.exists() and dest.exists():
            shutil.move(str(bak_path), str(dest))
        _clear_sidecars()
        raise click.ClickException(f"Restore failed: {exc}")

    # The restored file starts fresh — any sidecars written mid-copy (or
    # still lingering from the pre-restore DB) must not carry over.
    _clear_sidecars()

    # Verify the restored file is a valid SQLite database.
    try:
        conn = sqlite3.connect(f"file:{dest}?mode=ro", uri=True)
        conn.execute("SELECT count(*) FROM sqlite_master").fetchone()
        conn.close()
    except sqlite3.Error as exc:
        if bak_path.exists():
            shutil.move(str(bak_path), str(dest))
        _clear_sidecars()
        raise click.ClickException(f"Restored file is not a valid SQLite database: {exc}")

    elapsed_ms = int((time.monotonic() - started) * 1000)
    src_size = src.stat().st_size
    dest_size = dest.stat().st_size

    # Remove the .bak since the restore succeeded.
    try:
        if bak_path.exists():
            bak_path.unlink()
    except OSError:
        pass

    # Recreate the backups directory in case it was missing.
    _default_backup_dir().mkdir(parents=True, exist_ok=True)

    result = {
        "restored_from": str(src),
        "restored_to": str(dest),
        "source_size_bytes": src_size,
        "restored_size_bytes": dest_size,
        "elapsed_ms": elapsed_ms,
        "daemon_was_stopped": daemon_stopped,
        "worker_pids_stopped": worker_stopped_pids,
    }

    if as_json:
        import json as _json

        click.echo(_json.dumps(result, ensure_ascii=False, indent=2, default=str))
    else:
        click.echo(click.style("  ✓ restore complete", fg="green", bold=True))
        click.echo(f"     from : {src} ({_fmt_bytes(src_size)} compressed)")
        click.echo(f"     to   : {dest} ({_fmt_bytes(dest_size)} restored)")
        click.echo(f"     time : {elapsed_ms} ms")
        if daemon_stopped:
            click.echo(
                click.style(
                    "     note : daemon was stopped; run 'slowave serve start' to restart",
                    fg="yellow",
                )
            )
        if worker_stopped_pids:
            click.echo(
                click.style(
                    f"     note : worker process(es) {worker_stopped_pids} stopped; "
                    "restart 'slowave worker' if you run it manually",
                    fg="yellow",
                )
            )


def _fmt_bytes(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f} MB"
    if n >= 1_000:
        return f"{n / 1_000:.1f} kB"
    return f"{n} B"
