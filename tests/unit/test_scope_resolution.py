"""Coding-workspace scope resolution tests."""

from __future__ import annotations

import subprocess
from pathlib import Path

from slowave.core.scope import resolve_coding_scope


def _git(*args: str, cwd: Path) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)


def test_client_workspace_root_is_authoritative(tmp_path: Path) -> None:
    workspace = tmp_path / "repo-name"
    nested = workspace / "src" / "package"
    nested.mkdir(parents=True)

    result = resolve_coding_scope(cwd=nested, workspace_root=workspace)

    assert result.scope == "project:repo-name"
    assert result.root == workspace.resolve()
    assert result.source == "workspace_root"
    assert result.warning is None


def test_git_root_is_used_from_nested_directory(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    nested = repository / "tests" / "unit"
    nested.mkdir(parents=True)
    _git("init", "-q", cwd=repository)

    result = resolve_coding_scope(cwd=nested)

    assert result.scope == "project:repository"
    assert result.root == repository.resolve()
    assert result.source == "git_root"
    assert result.warning is None


def test_nested_repository_uses_nearest_git_root(tmp_path: Path) -> None:
    outer = tmp_path / "outer"
    inner = outer / "vendor" / "inner"
    nested = inner / "src"
    nested.mkdir(parents=True)
    _git("init", "-q", cwd=outer)
    _git("init", "-q", cwd=inner)

    result = resolve_coding_scope(cwd=nested)

    assert result.scope == "project:inner"
    assert result.root == inner.resolve()


def test_non_git_directory_falls_back_with_warning(tmp_path: Path) -> None:
    directory = tmp_path / "plain-workspace"
    directory.mkdir()

    result = resolve_coding_scope(cwd=directory)

    assert result.scope == "project:plain-workspace"
    assert result.root == directory.resolve()
    assert result.source == "cwd_fallback"
    assert result.warning is not None
