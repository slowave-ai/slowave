"""Resolve per-user Slowave runtime paths from one isolation boundary.

The default root is the current operating-system user's application-data
directory. ``SLOWAVE_HOME`` relocates the complete runtime tree, while the
legacy ``SLOWAVE_DB`` override keeps its exact database path and uses that
file's parent as the runtime root. Resolution never creates directories.
"""

from __future__ import annotations

import os
import platform as platform_module
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from platformdirs import user_data_dir


class RuntimePathError(ValueError):
    """Raised when runtime path configuration is invalid."""


@dataclass(frozen=True)
class RuntimePaths:
    """All runtime artifacts belonging to one Slowave data boundary."""

    root: Path
    database: Path
    pid_file: Path
    logs_dir: Path
    backups_dir: Path
    setup_sentinel: Path
    judge_debug_log: Path


def _normalized_path(value: str, *, home: Path | None = None) -> Path:
    """Expand a configured path without requiring it to exist."""
    if value.startswith("~") and home is not None:
        if value == "~":
            path = home
        elif value.startswith("~/") or value.startswith("~\\"):
            path = home / value[2:]
        else:
            raise RuntimePathError(f"Unsupported user expansion in path: {value!r}")
    else:
        path = Path(value).expanduser()
    return path.resolve(strict=False)


def _controlled_default_root(env: Mapping[str, str], platform: str, home: Path) -> Path:
    """Mirror platformdirs defaults for deterministic cross-platform tests."""
    name = platform.lower()
    xdg = env.get("XDG_DATA_HOME", "").strip()
    if name in {"darwin", "macos"}:
        base = Path(xdg) if xdg else home / "Library" / "Application Support"
    elif name in {"windows", "win32"}:
        local = env.get("LOCALAPPDATA", "").strip()
        base = Path(local) if local else home / "AppData" / "Local"
    else:
        base = Path(xdg) if xdg else home / ".local" / "share"
    return (base / "slowave").resolve(strict=False)


def default_runtime_root(
    *,
    env: Mapping[str, str] | None = None,
    platform: str | None = None,
    home: str | Path | None = None,
) -> Path:
    """Return the platform user-data root, ignoring Slowave overrides.

    Supplying environment/platform/home inputs selects a deterministic test
    path. Production calls delegate the native rendering to ``platformdirs``.
    """
    if env is None and platform is None and home is None:
        return Path(user_data_dir("slowave", appauthor=False)).resolve(strict=False)
    source_env = os.environ if env is None else env
    system = platform_module.system() if platform is None else platform
    selected_home = Path.home() if home is None else Path(home)
    return _controlled_default_root(source_env, system, selected_home)


def resolve_runtime_paths(
    env: Mapping[str, str] | None = None,
    platform: str | None = None,
    home: str | Path | None = None,
) -> RuntimePaths:
    """Purely resolve Slowave runtime paths for an environment.

    No filesystem entries are created. Existing paths are inspected only to
    reject a runtime root that is already a regular file.
    """
    source_env = os.environ if env is None else env
    selected_home = Path.home() if home is None else Path(home)

    home_set = "SLOWAVE_HOME" in source_env
    db_set = "SLOWAVE_DB" in source_env
    home_value = source_env.get("SLOWAVE_HOME", "")
    db_value = source_env.get("SLOWAVE_DB", "")

    if home_set and not home_value.strip():
        raise RuntimePathError("SLOWAVE_HOME is set but empty; unset it or provide a directory")
    if db_set and not db_value.strip():
        raise RuntimePathError("SLOWAVE_DB is set but empty; unset it or provide a database file")
    if home_set and db_set:
        raise RuntimePathError(
            "SLOWAVE_HOME and SLOWAVE_DB cannot both be set; use SLOWAVE_HOME "
            "to relocate all runtime data or SLOWAVE_DB for the legacy DB-only override"
        )

    if home_set:
        root = _normalized_path(home_value, home=selected_home)
        database = root / "slowave.db"
    elif db_set:
        database = _normalized_path(db_value, home=selected_home)
        if database.exists() and database.is_dir():
            raise RuntimePathError(f"SLOWAVE_DB must name a file, not a directory: {database}")
        root = database.parent
    else:
        root = default_runtime_root(env=env, platform=platform, home=home)
        database = root / "slowave.db"

    if root.exists() and not root.is_dir():
        raise RuntimePathError(f"Slowave runtime root is not a directory: {root}")

    return RuntimePaths(
        root=root,
        database=database,
        pid_file=root / "daemon.pid",
        logs_dir=root / "logs",
        backups_dir=root / "backups",
        setup_sentinel=root / ".setup_done",
        judge_debug_log=root / "judge_debug.jsonl",
    )


def runtime_paths() -> RuntimePaths:
    """Resolve paths from the current process environment."""
    return resolve_runtime_paths()


def ensure_runtime_dirs(paths: RuntimePaths) -> None:
    """Create runtime directories with user-only permissions where supported."""
    try:
        paths.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        paths.database.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        paths.logs_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        paths.backups_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    except OSError as exc:
        raise RuntimePathError(f"Cannot create Slowave runtime directories: {exc}") from exc


def default_db_path() -> str:
    """Compatibility wrapper returning the resolved database path."""
    return str(runtime_paths().database)
