from __future__ import annotations

from pathlib import Path

import pytest

from slowave.cli import setup
from slowave.cli.cleanup import _runtime_cleanup_targets
from slowave.core.paths import RuntimePathError, ensure_runtime_dirs, resolve_runtime_paths
from slowave.mcp import daemon


@pytest.mark.parametrize(
    ("platform", "env", "suffix"),
    [
        ("Darwin", {}, Path("Library/Application Support/slowave")),
        ("Linux", {}, Path(".local/share/slowave")),
        ("Linux", {"XDG_DATA_HOME": "/xdg data"}, Path("xdg data/slowave")),
        ("Windows", {"LOCALAPPDATA": "/local data"}, Path("local data/slowave")),
    ],
)
def test_platform_defaults_are_per_user(platform, env, suffix, tmp_path):
    paths = resolve_runtime_paths(env=env, platform=platform, home=tmp_path / "user")
    assert str(paths.root).endswith(str(suffix))
    assert paths.database == paths.root / "slowave.db"
    for artifact in (
        paths.database,
        paths.pid_file,
        paths.logs_dir,
        paths.backups_dir,
        paths.setup_sentinel,
        paths.judge_debug_log,
    ):
        assert artifact.is_relative_to(paths.root)


def test_slowave_home_relocates_all_artifacts(tmp_path):
    root = tmp_path / "runtime with spaces"
    paths = resolve_runtime_paths(env={"SLOWAVE_HOME": str(root)})
    assert paths.root == root
    assert paths.database == root / "slowave.db"
    assert paths.pid_file == root / "daemon.pid"
    assert paths.logs_dir == root / "logs"
    assert paths.backups_dir == root / "backups"


def test_legacy_db_uses_exact_file_and_parent_root(tmp_path):
    database = tmp_path / "legacy" / "custom.sqlite"
    paths = resolve_runtime_paths(env={"SLOWAVE_DB": str(database)})
    assert paths.database == database
    assert paths.root == database.parent


@pytest.mark.parametrize(
    "env",
    [
        {"SLOWAVE_HOME": "", "SLOWAVE_DB": "/tmp/x.db"},
        {"SLOWAVE_HOME": "/tmp/x", "SLOWAVE_DB": "/tmp/x.db"},
        {"SLOWAVE_DB": ""},
    ],
)
def test_invalid_overrides_are_rejected(env):
    with pytest.raises(RuntimePathError):
        resolve_runtime_paths(env=env)


def test_resolution_does_not_create_and_creation_is_explicit(tmp_path):
    root = tmp_path / "runtime"
    paths = resolve_runtime_paths(env={"SLOWAVE_HOME": str(root)})
    assert not root.exists()
    ensure_runtime_dirs(paths)
    assert paths.root.is_dir()
    assert paths.logs_dir.is_dir()
    assert paths.backups_dir.is_dir()


def test_existing_file_cannot_be_runtime_root(tmp_path):
    root = tmp_path / "file"
    root.write_text("not a directory")
    with pytest.raises(RuntimePathError, match="not a directory"):
        resolve_runtime_paths(env={"SLOWAVE_HOME": str(root)})


def test_daemon_pid_state_is_independent_per_injected_root(monkeypatch, tmp_path):
    one = tmp_path / "one" / "daemon.pid"
    two = tmp_path / "two" / "daemon.pid"
    monkeypatch.setattr(daemon, "_pid_exists", lambda pid: True)
    monkeypatch.setattr(daemon, "_is_slowave_process", lambda pid: True)

    daemon.write_pid(one)
    assert daemon.is_running(one)
    assert not daemon.is_running(two)
    daemon.write_pid(two)
    assert daemon.is_running(one)
    assert daemon.is_running(two)
    daemon.remove_pid(one)
    daemon.remove_pid(two)


def test_generated_launchd_service_pins_and_escapes_runtime_root(monkeypatch, tmp_path):
    root = tmp_path / "data & state"
    monkeypatch.delenv("SLOWAVE_DB", raising=False)
    monkeypatch.setenv("SLOWAVE_HOME", str(root))
    monkeypatch.setattr(setup, "_home", lambda: tmp_path / "home")
    monkeypatch.setattr(setup.subprocess, "run", lambda *args, **kwargs: None)

    plist_path, changed = setup._install_daemon_macos("/bin/slowave")

    content = Path(plist_path).read_text()
    assert changed is True
    assert "<key>SLOWAVE_HOME</key>" in content
    assert "data &amp; state" in content
    assert str(root / "logs" / "daemon.log").replace("&", "&amp;") in content


def test_generated_service_preserves_legacy_exact_db_override(monkeypatch, tmp_path):
    database = tmp_path / "custom name.sqlite"
    monkeypatch.delenv("SLOWAVE_HOME", raising=False)
    monkeypatch.setenv("SLOWAVE_DB", str(database))
    key, value = setup._runtime_service_env()
    assert (key, value) == ("SLOWAVE_DB", str(database))


def test_windows_task_action_sets_runtime_environment(monkeypatch, tmp_path):
    root = tmp_path / "windows data"
    monkeypatch.delenv("SLOWAVE_DB", raising=False)
    monkeypatch.setenv("SLOWAVE_HOME", str(root))
    monkeypatch.setattr(setup, "_find_pythonw", lambda: "C:/Python/pythonw.exe")
    execute, argument = setup._windows_runtime_action("slowave.exe", "serve start")
    assert execute == "powershell.exe"
    assert f"$env:SLOWAVE_HOME='{root}'" in argument
    assert "pythonw.exe' -m slowave serve start" in argument


def test_legacy_db_cleanup_never_sweeps_arbitrary_parent(monkeypatch, tmp_path):
    database = tmp_path / "custom.sqlite"
    unrelated = tmp_path / "unrelated.txt"
    unrelated.write_text("keep")
    monkeypatch.delenv("SLOWAVE_HOME", raising=False)
    monkeypatch.setenv("SLOWAVE_DB", str(database))

    root, targets, dedicated = _runtime_cleanup_targets()

    assert root == tmp_path
    assert dedicated is False
    assert database in targets
    assert unrelated not in targets
