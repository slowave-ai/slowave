"""Supervisor lifecycle regression tests; no live services are modified."""

from subprocess import CompletedProcess

import pytest
from click.testing import CliRunner

from slowave.cli import services


@pytest.mark.parametrize("system", ["Darwin", "Linux", "Windows"])
def test_restart_stops_all_services_before_starting(system, monkeypatch, tmp_path):
    monkeypatch.setattr(services.platform, "system", lambda: system)
    monkeypatch.setattr(services.Path, "home", lambda: tmp_path)
    monkeypatch.setattr(services.os, "getuid", lambda: 501, raising=False)
    folder = tmp_path / "Library" / "LaunchAgents"
    folder.mkdir(parents=True)
    for kind in services.KINDS:
        (folder / f"com.slowave.{kind}.plist").touch()
    calls = []
    loaded = set(services.KINDS)

    def run(args, **kwargs):
        calls.append(args)
        if system == "Darwin":
            kind = args[-1].split(".")[-1]
            if args[1] == "print":
                return CompletedProcess(args, 0 if kind in loaded else 1, "", "")
            if args[1] == "bootout":
                loaded.discard(kind)
        return CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(services.subprocess, "run", run)
    services.control("restart")
    if system == "Darwin":
        operations = [c[1] for c in calls if c[1] != "print"]
        assert operations == ["bootout"] * 3 + ["bootstrap"] * 3
    elif system == "Linux":
        operations = [c[2] for c in calls]
        assert operations == ["stop"] * 4 + ["start"] * 3
        assert calls[0][-1] == "slowave-backup.timer"
    else:
        scripts = [c[-1] for c in calls]
        assert all("Disable-ScheduledTask" in s and "Stop-ScheduledTask" in s for s in scripts[:3])
        assert all("Enable-ScheduledTask" in s for s in scripts[3:])
        assert "Start-ScheduledTask" not in scripts[-1]  # Don't run a backup now.


def test_failed_service_command_exits_nonzero(monkeypatch):
    monkeypatch.setattr(services.platform, "system", lambda: "Linux")
    monkeypatch.setattr(
        services.subprocess, "run", lambda *a, **kw: CompletedProcess(a, 1, "", "no user bus")
    )
    result = CliRunner().invoke(services.start_cmd)
    assert result.exit_code != 0
    assert "no user bus" in result.output


def test_launchd_waits_for_asynchronous_removal(monkeypatch):
    import time

    results = iter([0, 0, 1])
    calls = []
    monkeypatch.setattr(
        services, "_run", lambda args, **kw: CompletedProcess(args, next(results), "", "")
    )
    monkeypatch.setattr(time, "sleep", lambda seconds: calls.append(seconds))
    services.wait_launchd_stopped("gui/501/com.slowave.daemon")
    assert calls == [0.1, 0.1]


def test_missing_macos_registration_does_not_launch_foreground_server(monkeypatch, tmp_path):
    monkeypatch.setattr(services.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(services.Path, "home", lambda: tmp_path)
    result = CliRunner().invoke(services.start_cmd)
    assert result.exit_code != 0
    assert "setup first" in result.output


def test_verification_rejects_old_daemon_version(monkeypatch):
    import json
    import time
    import urllib.request

    import click

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self):
            return json.dumps({"version": "old-version"}).encode()

    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **kw: Response())
    monkeypatch.setattr(time, "sleep", lambda *a: None)
    times = iter([0, 0, 46])
    monkeypatch.setattr(time, "monotonic", lambda: next(times))
    with pytest.raises(click.ClickException, match="running version old-version"):
        services.verify_daemon()


def test_daemon_health_requires_the_selected_database(monkeypatch, tmp_path):
    from slowave import __version__

    monkeypatch.delenv("SLOWAVE_DB", raising=False)
    monkeypatch.setenv("SLOWAVE_HOME", str(tmp_path / "selected"))
    assert services.daemon_health_matches(
        {"version": __version__, "db": str(tmp_path / "selected" / "slowave.db")}
    )
    for payload in (
        {"version": __version__, "db": str(tmp_path / "other" / "slowave.db")},
        {"version": __version__},
        {"version": "old", "db": str(tmp_path / "selected" / "slowave.db")},
        [],
    ):
        assert not services.daemon_health_matches(payload)


def test_service_status_does_not_open_database(monkeypatch):
    import urllib.request

    monkeypatch.setattr(services.platform, "system", lambda: "Linux")
    monkeypatch.setattr(
        services.subprocess,
        "run",
        lambda *a, **kw: CompletedProcess(a, 0, "ActiveState=inactive", ""),
    )

    def offline(*args, **kwargs):
        raise OSError("offline")

    monkeypatch.setattr(urllib.request, "urlopen", offline)
    result = services.service_status()
    assert result["daemon_version_matches"] is False
    assert set(result["services"]) == set(services.KINDS)


def test_docs_command_is_discoverable_without_browser(monkeypatch):
    import webbrowser

    from slowave.cli.main import cli

    def unexpected(*args, **kwargs):
        raise AssertionError("Browser should not open")

    monkeypatch.setattr(webbrowser, "open", unexpected)
    result = CliRunner().invoke(cli, ["docs", "troubleshooting", "--no-open"])
    assert result.exit_code == 0
    assert result.output.strip().endswith("/docs/troubleshooting.md")
