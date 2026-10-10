"""Supervisor lifecycle regression tests; no live services are modified."""

from subprocess import CompletedProcess

import pytest
from click.testing import CliRunner

from slowave.cli import services


@pytest.mark.parametrize("system", ["Darwin", "Linux", "Windows"])
@pytest.mark.parametrize("healthy_at", [0.0, 60.0, 119.5, None])
def test_daemon_readiness_allows_slow_startup_with_bounded_deadline(
    system, healthy_at, monkeypatch, tmp_path
):
    import io
    import json
    import time
    import urllib.request

    from slowave import __version__
    from slowave.core.paths import runtime_paths

    monkeypatch.setattr(services.platform, "system", lambda: system)
    monkeypatch.delenv("SLOWAVE_DB", raising=False)
    monkeypatch.setenv("SLOWAVE_HOME", str(tmp_path))
    clock = [0.0]
    requests = []
    monkeypatch.setattr(time, "monotonic", lambda: clock[0])

    def sleep(seconds):
        clock[0] += seconds

    def request(url, *, timeout):
        requests.append((clock[0], timeout))
        if healthy_at is not None and clock[0] + timeout >= healthy_at:
            clock[0] = max(clock[0], healthy_at)
            return io.BytesIO(
                json.dumps({"version": __version__, "db": str(runtime_paths().database)}).encode()
            )
        # Account for time spent in the HTTP request, not just poll sleeps.
        clock[0] += timeout
        raise TimeoutError("still starting")

    monkeypatch.setattr(time, "sleep", sleep)
    monkeypatch.setattr(urllib.request, "urlopen", request)
    result = services.wait_for_daemon_health(8766)
    if healthy_at is None:
        assert result == "still starting"
        assert clock[0] == 120.0
    else:
        assert result is None
        assert healthy_at <= clock[0] < 120.0
        if healthy_at == 0:
            assert len(requests) == 1
    assert all(start + timeout <= 120.0 for start, timeout in requests)


@pytest.mark.parametrize(
    "body, message",
    [
        (b"not json", "Expecting value"),
        (b"[]", "expected a JSON object"),
        (b'{"version": "old"}', "running version old"),
    ],
)
def test_daemon_readiness_retries_invalid_health(body, message, monkeypatch):
    import io
    import time
    import urllib.request

    clock = [0.0]
    monkeypatch.setattr(time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds))
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **kw: io.BytesIO(body))
    assert message in services.wait_for_daemon_health(8766, timeout=1.25)
    assert clock[0] == 1.25


@pytest.mark.parametrize("db", [None, 123, "", "/wrong/database.db"])
def test_daemon_readiness_rejects_invalid_database_identity(db, monkeypatch, tmp_path):
    import io
    import json
    import time
    import urllib.request

    from slowave import __version__

    monkeypatch.delenv("SLOWAVE_DB", raising=False)
    monkeypatch.setenv("SLOWAVE_HOME", str(tmp_path))
    clock = iter([0.0, 0.0, 120.0, 120.0])
    monkeypatch.setattr(time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda *a, **kw: io.BytesIO(json.dumps({"version": __version__, "db": db}).encode()),
    )
    assert "database" in services.wait_for_daemon_health(8766)


def test_daemon_readiness_retries_malformed_http(monkeypatch):
    import http.client
    import time
    import urllib.request

    clock = iter([0.0, 0.0, 120.0, 120.0])
    monkeypatch.setattr(time, "monotonic", lambda: next(clock))

    def request(*args, **kwargs):
        raise http.client.BadStatusLine("invalid HTTP status")

    monkeypatch.setattr(urllib.request, "urlopen", request)
    assert "invalid HTTP status" in services.wait_for_daemon_health(8766)


def test_daemon_readiness_recovers_from_http_error_on_real_loopback(monkeypatch, tmp_path):
    import json
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    from slowave import __version__
    from slowave.core.paths import runtime_paths

    monkeypatch.delenv("SLOWAVE_DB", raising=False)
    monkeypatch.setenv("SLOWAVE_HOME", str(tmp_path))
    requests = []
    payload = json.dumps({"version": __version__, "db": str(runtime_paths().database)}).encode()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            requests.append(self.path)
            self.send_response(503 if len(requests) == 1 else 200)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        assert services.wait_for_daemon_health(server.server_port, timeout=5.0) is None
        assert requests == ["/health", "/health"]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5.0)


@pytest.mark.parametrize("system", ["Darwin", "Linux", "Windows"])
def test_restart_stops_all_services_before_starting(system, monkeypatch, tmp_path):
    monkeypatch.setattr(services.platform, "system", lambda: system)
    monkeypatch.setattr(services.Path, "home", lambda: tmp_path)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / ".config"))
    monkeypatch.setattr(services.os, "getuid", lambda: 501, raising=False)
    folder = tmp_path / "Library" / "LaunchAgents"
    folder.mkdir(parents=True)
    for kind in services.KINDS:
        (folder / f"com.slowave.{kind}.plist").touch()
    if system == "Linux":
        units = tmp_path / ".config" / "systemd" / "user"
        units.mkdir(parents=True)
        for kind in services.KINDS:
            (units / f"slowave-{kind}.service").touch()
            if kind == "backup":
                (units / "slowave-backup.timer").touch()
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


def test_failed_service_command_exits_nonzero(tmp_path, monkeypatch):
    monkeypatch.setattr(services.platform, "system", lambda: "Linux")
    monkeypatch.setattr(services.Path, "home", lambda: tmp_path)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / ".config"))
    unit_dir = tmp_path / ".config" / "systemd" / "user"
    unit_dir.mkdir(parents=True)
    (unit_dir / "slowave-daemon.service").touch()
    monkeypatch.setattr(
        services.subprocess, "run", lambda *a, **kw: CompletedProcess(a, 1, "", "no user bus")
    )
    result = CliRunner().invoke(services.start_cmd)
    assert result.exit_code != 0
    assert "no user bus" in result.output


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
    clock = iter([0.0, 1.0, 121.0, 121.0])
    monkeypatch.setattr(time, "monotonic", lambda: next(clock))
    with pytest.raises(click.ClickException, match="running version old-version"):
        services.verify_daemon()


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


@pytest.mark.parametrize("system", ["Darwin", "Linux", "Windows"])
def test_remove_service_failure_preserves_registration(system, tmp_path, monkeypatch):
    import click

    monkeypatch.setattr(services.os, "getuid", lambda: 501, raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    files = services.service_files("worker", system=system, home=tmp_path)
    for path in files:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("original registration")

    def run(args, **kwargs):
        # A macOS print succeeds, but the actual stop fails.
        if args[:2] == ["launchctl", "print"]:
            return CompletedProcess(args, 0, "running", "")
        return CompletedProcess(args, 1, "", "stop denied")

    monkeypatch.setattr(services.subprocess, "run", run)
    with pytest.raises(click.ClickException, match="stop denied"):
        services.remove_service("worker", system=system, home=tmp_path)
    assert all(path.read_text() == "original registration" for path in files)


def test_linux_backup_removal_honors_xdg_and_stops_before_unlink(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "custom-config"))
    paths = services.service_files("backup", system="Linux", home=tmp_path)
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("registration")
    calls = []

    def run(args, **kwargs):
        calls.append(args)
        if args[2] in ("stop", "disable"):
            assert all(path.exists() for path in paths)
        return CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(services.subprocess, "run", run)
    assert services.remove_service("backup", system="Linux", home=tmp_path) == 1
    assert not any(path.exists() for path in paths)
    assert [c[2] for c in calls] == ["stop", "stop", "disable", "daemon-reload"]
    assert all("custom-config" in str(path) for path in paths)


def test_purge_stops_before_data_deletion_if_service_removal_fails(tmp_path, monkeypatch):
    import click

    from slowave.cli.cleanup import cleanup_cmd

    monkeypatch.delenv("SLOWAVE_DB", raising=False)
    monkeypatch.setenv("SLOWAVE_HOME", str(tmp_path))
    database = tmp_path / "slowave.db"
    database.write_text("preserved data")

    def fail(*args, **kwargs):
        raise click.ClickException("service still running")

    monkeypatch.setattr("slowave.cli.cleanup._remove_daemon_service", fail)
    result = CliRunner().invoke(cleanup_cmd, ["--yes"])
    assert result.exit_code != 0
    assert database.read_text() == "preserved data"
    assert "complete" not in result.output.lower()


def test_uninstall_does_not_claim_success_when_service_removal_fails(monkeypatch):
    import click

    from slowave.cli.main import cli

    def fail(*args, **kwargs):
        raise click.ClickException("service still running")

    monkeypatch.setattr("slowave.cli.cleanup._remove_daemon_service", fail)
    result = CliRunner().invoke(cli, ["uninstall"])
    assert result.exit_code != 0
    assert "Uninstall complete" not in result.output


def test_macos_stop_boots_out_orphaned_job_without_plist(monkeypatch, tmp_path):
    monkeypatch.setattr(services.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(services.Path, "home", lambda: tmp_path)
    monkeypatch.setattr(services.os, "getuid", lambda: 501, raising=False)
    calls = []

    def run(args, **kwargs):
        calls.append(args)
        return CompletedProcess(args, 0, "running", "")

    monkeypatch.setattr(services.subprocess, "run", run)
    services.control("stop", ("worker",))
    assert calls[-1] == ["launchctl", "bootout", "gui/501/com.slowave.worker"]


def test_verification_rejects_wrong_runtime_database(monkeypatch):
    import json
    import time
    import urllib.request

    import click

    from slowave import __version__

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self):
            return json.dumps({"version": __version__, "db": "/wrong/database.db"}).encode()

    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **kw: Response())
    monkeypatch.setattr(time, "sleep", lambda *a: None)
    clock = iter([0.0, 1.0, 121.0, 121.0])
    monkeypatch.setattr(time, "monotonic", lambda: next(clock))
    with pytest.raises(click.ClickException, match="running database"):
        services.verify_daemon()


@pytest.mark.parametrize("action", ["start", "stop", "remove"])
def test_windows_lifecycle_targets_sid_task_and_owned_legacy_task(action, monkeypatch):
    monkeypatch.setattr(services.platform, "system", lambda: "Windows")
    calls = []

    def run(args, **kwargs):
        calls.append(args)
        return CompletedProcess(args, 0, "removed", "")

    monkeypatch.setattr(services.subprocess, "run", run)
    if action == "remove":
        services.remove_service("worker", system="Windows")
    else:
        services.control(action, ("worker",))
    script = calls[-1][-1]
    assert "$identity.User.Value" in script
    assert "SlowaveWorker-'" in script
    if action == "start":
        assert "Enable-ScheduledTask -TaskName $taskName" in script
        assert "Stop-ScheduledTask" not in script
    else:
        assert "Get-SlowaveTaskSid $_.Principal.UserId" in script
        assert "TaskPath -eq '\\'" in script
        assert "Disable-ScheduledTask -InputObject $t" in script
        if action == "remove":
            assert "Unregister-ScheduledTask -InputObject $t" in script


def test_windows_task_name_is_account_scoped_and_legacy_matching_is_opt_in():
    current = services.windows_task_context("SlowaveWorker")
    migrating = services.windows_task_context("SlowaveWorker", include_legacy=True)
    assert "$taskName='SlowaveWorker-'+$identity.User.Value" in current
    assert "$legacyTask" not in current
    assert "Get-SlowaveTaskSid $_.Principal.UserId" in migrating


def test_linux_start_without_registration_requests_setup(tmp_path, monkeypatch):
    import click

    monkeypatch.setattr(services.platform, "system", lambda: "Linux")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "empty-config"))

    def unexpected(*args, **kwargs):
        raise AssertionError("systemctl must not run without a registration")

    monkeypatch.setattr(services.subprocess, "run", unexpected)
    with pytest.raises(click.ClickException, match="Run slowave setup first"):
        services.control("start", ("daemon",))


def test_linux_stop_without_registration_is_a_no_op(tmp_path, monkeypatch):
    monkeypatch.setattr(services.platform, "system", lambda: "Linux")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "empty-config"))

    def unexpected(*args, **kwargs):
        raise AssertionError("systemctl must not run without a registration")

    monkeypatch.setattr(services.subprocess, "run", unexpected)
    services.control("stop", ("daemon",))


def test_windows_start_without_task_registration_requests_setup(monkeypatch):
    import click

    monkeypatch.setattr(services.platform, "system", lambda: "Windows")
    calls = []

    def run(args, **kwargs):
        calls.append(args)
        return CompletedProcess(args, 1, "missing-registration", "")

    monkeypatch.setattr(services.subprocess, "run", run)
    with pytest.raises(click.ClickException, match="Run slowave setup first"):
        services.control("start", ("daemon",))
    assert "if (-not $currentTask)" in calls[0][-1]


def test_windows_start_failure_keeps_generic_error(monkeypatch):
    import click

    monkeypatch.setattr(services.platform, "system", lambda: "Windows")
    monkeypatch.setattr(
        services.subprocess,
        "run",
        lambda *a, **kw: CompletedProcess(a, 1, "", "Enable denied"),
    )
    with pytest.raises(click.ClickException, match="Enable denied"):
        services.control("start", ("daemon",))


def test_stop_registered_services_waits_for_daemon_exit(tmp_path, monkeypatch):
    import time

    import slowave.mcp.daemon as daemon_mod

    monkeypatch.setattr(services.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(services.Path, "home", lambda: tmp_path)
    monkeypatch.setattr(services.os, "getuid", lambda: 501, raising=False)
    agents = tmp_path / "Library" / "LaunchAgents"
    agents.mkdir(parents=True)
    for kind in services.KINDS:
        (agents / f"com.slowave.{kind}.plist").touch()
    monkeypatch.setattr(
        services.subprocess,
        "run",
        lambda args, **kwargs: CompletedProcess(args, 1, "", ""),
    )
    states = iter([True, True, False])
    monkeypatch.setattr(daemon_mod, "is_running", lambda: next(states))
    sleeps = []
    monkeypatch.setattr(time, "sleep", lambda s: sleeps.append(s))
    stopped = services.stop_registered_services()
    assert stopped == ["backup", "worker", "daemon"]
    assert sleeps == [0.2, 0.2]


def test_stop_registered_services_gives_up_waiting_after_timeout(tmp_path, monkeypatch):
    import time

    import slowave.mcp.daemon as daemon_mod

    monkeypatch.setattr(services.platform, "system", lambda: "Linux")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    unit = tmp_path / "xdg" / "systemd" / "user" / "slowave-daemon.service"
    unit.parent.mkdir(parents=True)
    unit.write_text("unit")
    monkeypatch.setattr(
        services.subprocess, "run", lambda args, **kw: CompletedProcess(args, 0, "", "")
    )
    monkeypatch.setattr(daemon_mod, "is_running", lambda: True)
    ticks = iter([0.0, 20.0])
    monkeypatch.setattr(time, "monotonic", lambda: next(ticks))
    monkeypatch.setattr(time, "sleep", lambda s: None)
    assert services.stop_registered_services() == ["daemon"]


def test_require_stopped_runtime_ignores_other_users_processes(tmp_path, monkeypatch):
    import click

    import slowave.mcp.daemon as daemon_mod
    from slowave.cli.services import require_stopped_runtime

    monkeypatch.setattr(daemon_mod, "is_running", lambda: False)
    monkeypatch.setattr(
        "slowave.cli.main._slowave_processes",
        lambda: [
            {
                "uid": 9999,
                "pid": 111,
                "ppid": 1,
                "stat": "S",
                "rss_kb": 1,
                "command": "slowave serve start",
            },
            {
                "uid": 501,
                "pid": 222,
                "ppid": 1,
                "stat": "S",
                "rss_kb": 1,
                "command": "slowave dashboard",
            },
        ],
    )
    monkeypatch.setattr(services.os, "getuid", lambda: 501, raising=False)
    with pytest.raises(click.ClickException, match="Restore aborted"):
        require_stopped_runtime("Restore", timeout=0)

    monkeypatch.setattr(
        "slowave.cli.main._slowave_processes",
        lambda: [
            {
                "uid": 9999,
                "pid": 111,
                "ppid": 1,
                "stat": "S",
                "rss_kb": 1,
                "command": "slowave serve start",
            },
            {
                "uid": 9999,
                "pid": 112,
                "ppid": 1,
                "stat": "S",
                "rss_kb": 1,
                "command": "slowave worker",
            },
        ],
    )
    require_stopped_runtime(
        "Restore", timeout=0
    )  # other accounts' runtimes must not block this one


def test_require_stopped_runtime_waits_for_exiting_processes(tmp_path, monkeypatch):
    import time

    import slowave.mcp.daemon as daemon_mod
    from slowave.cli.services import require_stopped_runtime

    monkeypatch.setattr(daemon_mod, "is_running", lambda: False)
    monkeypatch.setattr(services.os, "getuid", lambda: 501, raising=False)
    monkeypatch.setattr("slowave.cli.main._slowave_processes", lambda: [])
    sleeps = []
    monkeypatch.setattr(time, "sleep", lambda s: sleeps.append(s))
    ticks = iter([0.0, 0.1, 0.2, 0.3, 0.4])
    monkeypatch.setattr(time, "monotonic", lambda: next(ticks))

    # First scan finds a worker that is still exiting; the second scan is clear.
    scans = iter(
        [
            [
                {
                    "uid": 501,
                    "pid": 7,
                    "ppid": 1,
                    "stat": "S",
                    "rss_kb": 1,
                    "command": "slowave worker",
                }
            ],
            [],
        ]
    )
    monkeypatch.setattr(services, "_current_user_runtime_processes", lambda: next(scans))
    require_stopped_runtime("Restore")  # the grace period absorbed the exiting process
    assert sleeps == [0.25]


def test_windows_purge_kill_targets_only_registered_interpreters(tmp_path, monkeypatch):
    import sys

    from slowave.cli.cleanup import cleanup_cmd

    monkeypatch.delenv("SLOWAVE_DB", raising=False)
    monkeypatch.setenv("SLOWAVE_HOME", str(tmp_path))
    monkeypatch.setattr("slowave.cli.cleanup.SYSTEM", "Windows")
    monkeypatch.setattr("slowave.cli.services.require_stopped_runtime", lambda *_: None)
    for name in ("_remove_daemon_service", "_remove_worker_service", "_remove_backup_service"):
        monkeypatch.setattr(f"slowave.cli.cleanup.{name}", lambda *a, **kw: 0)
    monkeypatch.setattr("slowave.cli.cleanup._remove_lifecycle_blocks", lambda *a, **kw: 0)
    monkeypatch.setattr("slowave.cli.cleanup._remove_mcp_configs", lambda *a, **kw: 0)
    (tmp_path / "slowave.db").write_text("data")
    scripts = []

    def run(args, **kwargs):
        scripts.append(args[-1])
        return CompletedProcess(args, 0, "", "")

    monkeypatch.setattr("slowave.cli.cleanup.subprocess.run", run)
    from pathlib import Path

    CliRunner().invoke(cleanup_cmd, ["--yes"])
    kill_scripts = [s for s in scripts if "Stop-Process" in s]
    assert len(kill_scripts) == 1
    assert str(Path(sys.executable).resolve()) in kill_scripts[0]
    assert "-icontains" in kill_scripts[0]
    assert "-like '*slowave*'" not in kill_scripts[0]


def test_macos_stop_waits_for_old_process(monkeypatch, tmp_path):
    monkeypatch.setattr(services.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(services.Path, "home", lambda: tmp_path)
    monkeypatch.setattr(services.os, "getuid", lambda: 501, raising=False)
    calls = []

    def run(args, **kwargs):
        calls.append(args[1])
        return CompletedProcess(args, 0, "pid = 12345\n", "")

    monkeypatch.setattr(services.subprocess, "run", run)
    monkeypatch.setattr(services, "wait_for_process_exit", lambda pid: calls.append(pid))
    services.control("stop", ("daemon",))
    assert calls == ["print", "bootout", 12345]


def test_process_exit_wait_has_a_deadline(monkeypatch):
    import time

    import click

    monkeypatch.setattr(services.os, "kill", lambda *args: None)
    with pytest.raises(click.ClickException, match="did not stop"):
        services.wait_for_process_exit(12345, timeout=0)
    attempts = []

    def kill(*args):
        attempts.append(args)
        if len(attempts) == 2:
            raise ProcessLookupError

    monkeypatch.setattr(services.os, "kill", kill)
    monkeypatch.setattr(time, "sleep", lambda _: None)
    services.wait_for_process_exit(12345)
    assert len(attempts) == 2


def test_windows_process_detection_sees_direct_pythonw_launcher(monkeypatch):
    import importlib
    import json

    main = importlib.import_module("slowave.cli.main")
    command = (
        'pythonw.exe -c "import os,runpy,sys;'
        "sys.argv=['slowave', 'worker', '--interval', '300'];"
        "runpy.run_module('slowave',run_name='__main__',alter_sys=True)\""
    )
    calls = []

    def run(args, **kwargs):
        calls.append(args[-1])
        return CompletedProcess(
            args, 0, json.dumps({"ProcessId": 12345, "CommandLine": command}), ""
        )

    monkeypatch.setattr(main.subprocess, "run", run)
    processes = main._slowave_processes_windows()
    assert len(processes) == 1
    assert "slowave worker --interval 300" in processes[0]["command"]
    assert "GetOwnerSid().Sid -eq $sid" in calls[0]
