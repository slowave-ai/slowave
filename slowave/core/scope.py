"""Generic scope helpers.

Slowave's memory model is generic. Scope strings encode context such as
``project:slowave``, ``domain:cooking``, ``relationship:alex`` or ``household``.
The format is ``<kind>:<value>`` or just ``<value>`` (kind resolves to ``generic``).
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CodingScopeResolution:
    """Resolved coding-workspace scope and its discovery provenance."""

    scope: str
    root: Path
    source: str
    warning: str | None = None


def resolve_coding_scope(
    *,
    cwd: str | Path,
    workspace_root: str | Path | None = None,
) -> CodingScopeResolution:
    """Resolve a stable coding scope from the repository/workspace root.

    A client-supplied workspace root is authoritative. Otherwise ask Git for
    the nearest working-tree root, which handles calls made from nested
    directories, submodules, and linked worktrees. Non-Git directories fall
    back to the current directory and return an explicit warning.

    The emitted ``project:<root-name>`` format intentionally preserves the
    existing coding-agent scope identity for ordinary root-launched sessions.
    """

    cwd_path = Path(cwd).expanduser().resolve()
    if workspace_root is not None:
        root = Path(workspace_root).expanduser().resolve()
        return CodingScopeResolution(
            scope=f"project:{root.name}",
            root=root,
            source="workspace_root",
        )

    try:
        completed = subprocess.run(
            ["git", "-C", str(cwd_path), "rev-parse", "--show-toplevel"],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
        root = Path(completed.stdout.strip()).expanduser().resolve()
        if root.name:
            return CodingScopeResolution(
                scope=f"project:{root.name}",
                root=root,
                source="git_root",
            )
    except (FileNotFoundError, subprocess.SubprocessError):
        pass

    return CodingScopeResolution(
        scope=f"project:{cwd_path.name}",
        root=cwd_path,
        source="cwd_fallback",
        warning=(
            "No client workspace root or Git repository root was available; "
            "scope was derived from the current directory."
        ),
    )


def normalize_scope(*, scope: str | None = None) -> str | None:
    """Return a canonical scope id, or None if no scope is given."""
    if scope is not None and str(scope).strip():
        return str(scope).strip()
    return None


def scope_kind(scope: str | None) -> str | None:
    """Return the scope prefix/kind, or ``generic`` for un-prefixed scopes."""
    if not scope:
        return None
    text = str(scope).strip()
    if not text:
        return None
    if ":" in text:
        return text.split(":", 1)[0] or "generic"
    return "generic"


def scope_value(scope: str | None) -> str | None:
    """Return the value part of a scope id."""
    if not scope:
        return None
    text = str(scope).strip()
    if not text:
        return None
    if ":" in text:
        return text.split(":", 1)[1]
    return text
