from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "push-public.sh"

pytestmark = pytest.mark.skipif(
    shutil.which("rsync") is None or shutil.which("bash") is None,
    reason="push-public.sh needs bash and rsync",
)

PRIVATE_PYPROJECT = """[project]
name = "slowave"
version = "0.14.2"
dependencies = [
  "tokenizers>=0.20",
  "mcp[cli]>=1.27.1,<1.28.0",
]
"""
PUBLIC_PYPROJECT = """[project]
name = "slowave"
version = "0.23.4"
dependencies = [
  "transformers>=4.30",
  "mcp[cli]>=1.0,<2",
]
"""
RELEASE_DIRS = ["slowave", "tests", "docs", "integrations", "img"]
RELEASE_FILES = [
    "README.md",
    "README.zh-CN.md",
    "LICENSE",
    "CONTRIBUTING.md",
    "CLA.md",
    "COMMERCIAL.md",
    "SECURITY.md",
    "GNUmakefile",
    "Makefile",
    "package.json",
    "uv.lock",
    ".gitignore",
]


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@example.com", *args],
        cwd=repo,
        check=True,
        capture_output=True,
    )


def _repos(tmp_path: Path) -> tuple[Path, Path]:
    private, public = tmp_path / "private", tmp_path / "public"
    for directory in RELEASE_DIRS:
        (private / directory).mkdir(parents=True)
        (private / directory / "keep.txt").write_text("x")
    for name in RELEASE_FILES:
        (private / name).write_text("x")
    (private / "pyproject.toml").write_text(PRIVATE_PYPROJECT)
    _git(tmp_path, "init", "-q", str(private))
    _git(private, "add", "-A")
    _git(private, "commit", "-q", "-m", "release source")

    public.mkdir()
    (public / "pyproject.toml").write_text(PUBLIC_PYPROJECT)
    for name in [
        "CHANGELOG.md",
        "release-please-config.json",
        ".release-please-manifest.json",
        "server.json",
    ]:
        (public / name).write_text("{}")
    for directory in ["Formula", ".github"]:
        (public / directory).mkdir()
    return private, public


def _publish(private: Path, public: Path, *args: str) -> subprocess.CompletedProcess[str]:
    env = {
        **os.environ,
        "SLOWAVE_PRIVATE": str(private),
        "SLOWAVE_PUBLIC": str(public),
        "SLOWAVE_REF": "HEAD",
    }
    return subprocess.run(
        ["bash", str(SCRIPT), *args], env=env, capture_output=True, text=True, check=True
    )


def test_publishing_copies_pyproject_but_keeps_the_public_version(tmp_path):
    private, public = _repos(tmp_path)

    _publish(private, public)

    assert (public / "pyproject.toml").read_text() == PRIVATE_PYPROJECT.replace("0.14.2", "0.23.4")


def test_a_dry_run_shows_the_pyproject_change_without_writing_it(tmp_path):
    private, public = _repos(tmp_path)

    result = _publish(private, public, "--dry-run")

    assert (public / "pyproject.toml").read_text() == PUBLIC_PYPROJECT
    assert '+  "tokenizers>=0.20",' in result.stdout


def test_publishing_twice_leaves_pyproject_unchanged(tmp_path):
    private, public = _repos(tmp_path)
    _publish(private, public)

    result = _publish(private, public)

    assert "already matches the private file" in result.stdout
