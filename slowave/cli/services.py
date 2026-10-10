"""Control installed user services through their owning OS supervisor."""

from __future__ import annotations

import os
import platform
import subprocess
from pathlib import Path
from typing import Any

import click

KINDS = ("daemon", "worker", "backup")
DAEMON_HEALTH_TIMEOUT = 120.0


def _run(
    args: list[str],
    *,
    check: bool = True,
    missing_output: str | None = None,
    missing_message: str | None = None,
) -> subprocess.CompletedProcess:
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise click.ClickException(
            f"Service manager unavailable: {exc}. See docs/troubleshooting.md"
        ) from exc
    if check and result.returncode:
        if missing_output and result.stdout.strip() == missing_output:
            raise click.ClickException(
                missing_message or "Missing service. Run slowave setup first."
            )
        raise click.ClickException(
            f"Service command failed: {' '.join(args)}\n"
            f"{(result.stderr or result.stdout).strip()}\nRun slowave doctor; see docs/troubleshooting.md"
        )
    return result


def wait_for_process_exit(pid: int, timeout: float = 10.0) -> None:
    """Wait for a supervisor-stopped process before accepting a replacement."""
    import time

    deadline = time.monotonic() + timeout
    while True:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        if time.monotonic() >= deadline:
            raise click.ClickException(f"Service process {pid} did not stop within {timeout:g}s.")
        time.sleep(0.1)


def launchd_pid(detail: str) -> int | None:
    import re

    match = re.search(r"\bpid = (\d+)", detail)
    return int(match.group(1)) if match else None


def windows_task_context(base_name: str, *, include_legacy: bool = False) -> str:
    """Resolve this account's unique task name and its same-owner legacy task."""
    base = base_name.replace("'", "''")
    script = (
        "$identity=[System.Security.Principal.WindowsIdentity]::GetCurrent();"
        f"$taskName='{base}-'+$identity.User.Value;"
        "$currentTask=Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue;"
    )
    if include_legacy:
        script += (
            "function Get-SlowaveTaskSid($name) { try { if ($name -like 'S-1-*') { return $name };"
            "return ([System.Security.Principal.NTAccount]::new($name)).Translate("
            "[System.Security.Principal.SecurityIdentifier]).Value } catch { return '' } };"
            f"$legacyTask=Get-ScheduledTask -TaskName '{base}' -ErrorAction SilentlyContinue | "
            "Where-Object { $_.TaskPath -eq '\\' -and $_.Principal.UserId -and "
            "(Get-SlowaveTaskSid $_.Principal.UserId) -eq $identity.User.Value } | "
            "Select-Object -First 1;"
        )
    return script


def windows_task_candidates(base_name: str) -> str:
    """Set $tasks to the current account's task plus its root-name legacy task."""
    return (
        windows_task_context(base_name, include_legacy=True)
        + "$tasks=@(); if ($currentTask) { $tasks+=@($currentTask) };"
        + "if ($legacyTask) { $tasks+=@($legacyTask) };"
    )


def control(action: str, kinds: tuple[str, ...] = KINDS) -> None:
    """Start/stop registered jobs without bypassing automatic recovery policies."""
    system = platform.system()
    if action == "restart":
        control("stop", kinds)
        control("start", kinds)
        return
    for kind in (tuple(reversed(kinds)) if action == "stop" else kinds):
        if system == "Darwin":
            path = Path.home() / "Library" / "LaunchAgents" / f"com.slowave.{kind}.plist"
            if action == "start" and not path.exists():
                raise click.ClickException(f"Missing {kind} service. Run slowave setup first.")
            domain = f"gui/{os.getuid()}"
            target = f"{domain}/com.slowave.{kind}"
            state = _run(["launchctl", "print", target], check=False)
            loaded = state.returncode == 0
            if action == "stop" and loaded:
                _run(["launchctl", "bootout", target])
                pid = launchd_pid(state.stdout)
                if pid is not None:
                    wait_for_process_exit(pid)
            elif action == "start" and not loaded:
                _run(["launchctl", "bootstrap", domain, str(path)])
        elif system == "Linux":
            registered = any(path.exists() for path in service_files(kind))
            if action == "start" and not registered:
                raise click.ClickException(f"Missing {kind} service. Run slowave setup first.")
            if action == "stop" and not registered:
                continue
            target = f"slowave-{kind}" + (".timer" if kind == "backup" else ".service")
            _run(["systemctl", "--user", action, target])
            if action == "stop" and kind == "backup":
                # A timer stop does not stop an already dispatched backup.
                _run(["systemctl", "--user", "stop", "slowave-backup.service"])
        elif system == "Windows":
            base_name = f"Slowave{kind.title()}"
            if action == "stop":
                script = (
                    "$ErrorActionPreference='Stop';"
                    + windows_task_candidates(base_name)
                    + "foreach ($t in $tasks) {"
                    "Disable-ScheduledTask -InputObject $t | Out-Null;"
                    "Stop-ScheduledTask -InputObject $t;"
                    "$deadline=(Get-Date).AddSeconds(30);"
                    "while ((Get-ScheduledTask -TaskPath $t.TaskPath -TaskName $t.TaskName).State -eq 'Running') {"
                    "if ((Get-Date) -gt $deadline) { throw 'Task did not stop' };"
                    "Start-Sleep -Milliseconds 200 }}"
                )
                missing_output = missing_message = None
            else:
                script = (
                    "$ErrorActionPreference='Stop';"
                    + windows_task_context(base_name)
                    + "if (-not $currentTask) { Write-Output 'missing-registration'; exit 1 };"
                    "Enable-ScheduledTask -TaskName $taskName | Out-Null;"
                )
                if kind != "backup":
                    script += "Start-ScheduledTask -TaskName $taskName"
                missing_output = "missing-registration"
                missing_message = f"Missing {kind} service. Run slowave setup first."
            _run(
                ["powershell", "-NonInteractive", "-Command", script],
                missing_output=missing_output,
                missing_message=missing_message,
            )
        else:
            raise click.ClickException(f"Unsupported service platform: {system}")
        click.echo(f"{kind}: {action}")


def wait_for_daemon_health(port: int, timeout: float = DAEMON_HEALTH_TIMEOUT) -> str | None:
    """Return None when the expected daemon is healthy, or the last failure.

    Use one wall-clock budget for setup/start/restart on every OS. Connection
    failures, malformed responses and an old daemon are retried while the
    supervisor starts the replacement. No database or model is opened here.
    """
    import http.client
    import json
    import time
    import urllib.request

    from slowave import __version__
    from slowave.core.paths import runtime_paths

    expected_db = runtime_paths().database.resolve()
    url = f"http://127.0.0.1:{port}/health"
    last = "not responding"
    deadline = time.monotonic() + timeout
    while (remaining := deadline - time.monotonic()) > 0:
        try:
            with urllib.request.urlopen(url, timeout=min(1.0, remaining)) as response:
                data = json.load(response)
            if not isinstance(data, dict):
                last = "invalid health response: expected a JSON object"
            elif data.get("version") != __version__:
                last = f"running version {data.get('version')}, installed {__version__}"
            elif not isinstance(data.get("db"), str) or not data["db"]:
                last = "health response has no database path"
            elif Path(data["db"]).resolve() != expected_db:
                last = f"running database {data.get('db')}, expected {expected_db}"
            else:
                return None
        except (OSError, ValueError, http.client.HTTPException) as exc:
            last = str(exc)
        remaining = deadline - time.monotonic()
        if remaining > 0:
            time.sleep(min(0.5, remaining))
    return last


def verify_daemon() -> None:
    from slowave import __version__
    from slowave.core.paths import daemon_port

    failure = wait_for_daemon_health(daemon_port())
    if failure is not None:
        raise click.ClickException(
            f"Daemon verification failed after {DAEMON_HEALTH_TIMEOUT:g}s: {failure}. "
            "Run slowave status --services and slowave doctor."
        )
    click.echo(f"daemon: healthy, version {__version__}")


@click.command("start")
def start_cmd() -> None:
    """Start installed daemon, worker, and backup schedule. Run setup first."""
    control("start")
    verify_daemon()


@click.command("stop")
def stop_cmd() -> None:
    """Stop daemon, worker, and backup schedule (until start/setup or next login)."""
    control("stop")
    click.echo("Stop any foreground dashboard with Ctrl+C before upgrading.")


@click.command("restart")
def restart_cmd() -> None:
    """Restart installed services through the OS supervisor; verify daemon version."""
    control("restart")
    verify_daemon()
    click.echo("Restart foreground dashboards separately with Ctrl+C, then slowave dashboard.")


def service_status() -> dict:
    """Read supervisor state without initializing models or touching the DB."""
    import json
    import urllib.request

    from slowave import __version__
    from slowave.core.paths import daemon_port, runtime_paths

    system = platform.system()
    states = {}
    for kind in KINDS:
        if system == "Darwin":
            command = ["launchctl", "print", f"gui/{os.getuid()}/com.slowave.{kind}"]
        elif system == "Linux":
            target = f"slowave-{kind}" + (".timer" if kind == "backup" else ".service")
            command = [
                "systemctl",
                "--user",
                "show",
                target,
                "--property=LoadState,ActiveState,SubState,MainPID",
            ]
        elif system == "Windows":
            base_name = f"Slowave{kind.title()}"
            command = [
                "powershell",
                "-NonInteractive",
                "-Command",
                "$ErrorActionPreference='Stop';"
                + windows_task_candidates(base_name)
                + "$tasks | Select-Object TaskName,TaskPath,State | ConvertTo-Json -Compress",
            ]
        else:
            raise click.ClickException(f"Unsupported service platform: {system}")
        result = _run(command, check=False)
        states[kind] = {
            "returncode": result.returncode,
            "detail": (result.stdout or result.stderr).strip(),
        }
    health = {}
    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{daemon_port()}/health", timeout=2
        ) as response:
            health = json.load(response)
    except (OSError, ValueError):
        pass
    return {
        "installed_version": __version__,
        "daemon_health": health,
        "daemon_version_matches": health.get("version") == __version__,
        "daemon_database_matches": bool(health.get("db"))
        and Path(health["db"]).resolve() == runtime_paths().database.resolve(),
        "services": states,
        "runtime_root": str(runtime_paths().root),
        "logs_dir": str(runtime_paths().logs_dir),
        "dashboard": "Foreground process; stop with Ctrl+C and run slowave dashboard again.",
    }


def service_files(
    kind: str, *, system: str | None = None, home: Path | None = None
) -> tuple[Path, ...]:
    """Resolve registrations using the same XDG paths as setup."""
    system = system or platform.system()
    home = home or Path.home()
    if system == "Darwin":
        return (home / "Library" / "LaunchAgents" / f"com.slowave.{kind}.plist",)
    if system == "Linux":
        directory = (
            Path(os.environ.get("XDG_CONFIG_HOME", str(home / ".config"))) / "systemd" / "user"
        )
        suffixes = ("timer", "service") if kind == "backup" else ("service",)
        return tuple(directory / f"slowave-{kind}.{suffix}" for suffix in suffixes)
    return ()


def remove_service(
    kind: str, *, dry_run: bool = False, system: str | None = None, home: Path | None = None
) -> int:
    """Remove one registration; failures abort uninstall/purge before data deletion."""
    system = system or platform.system()
    files = service_files(kind, system=system, home=home)
    existing = [path for path in files if path.exists()]
    if system == "Linux" and not existing:
        return 0
    if dry_run and system == "Darwin" and not existing:
        return 0
    if dry_run:
        click.echo(
            f"Would stop and remove {kind}: {', '.join(map(str, existing)) or 'Slowave' + kind.title()}"
        )
        return 0
    if system == "Darwin":
        target = f"gui/{os.getuid()}/com.slowave.{kind}"
        state = _run(["launchctl", "print", target], check=False)
        loaded = state.returncode == 0
        if loaded:
            _run(["launchctl", "bootout", target])
            pid = launchd_pid(state.stdout)
            if pid is not None:
                wait_for_process_exit(pid)
        elif not existing:
            return 0
        for path in existing:
            path.unlink()
    elif system == "Linux":
        # Stop everything before deleting any registration, especially a backup
        # timer that can dispatch work while uninstall is in progress.
        for path in existing:
            _run(["systemctl", "--user", "stop", path.name])
        for path in existing:
            if path.suffix == ".timer" or kind != "backup":
                _run(["systemctl", "--user", "disable", path.name])
        for path in existing:
            path.unlink()
        _run(["systemctl", "--user", "daemon-reload"])
    elif system == "Windows":
        script = (
            "$ErrorActionPreference='Stop';"
            + windows_task_candidates(f"Slowave{kind.title()}")
            + "foreach ($t in $tasks) {"
            "Disable-ScheduledTask -InputObject $t | Out-Null;"
            "Stop-ScheduledTask -InputObject $t;"
            "$deadline=(Get-Date).AddSeconds(30);"
            "while ((Get-ScheduledTask -TaskPath $t.TaskPath -TaskName $t.TaskName).State -eq 'Running') {"
            "if ((Get-Date) -gt $deadline) { throw 'Task did not stop' };"
            "Start-Sleep -Milliseconds 200 };"
            "Unregister-ScheduledTask -InputObject $t -Confirm:$false };"
            "if ($tasks.Count -gt 0) { Write-Output 'removed' }"
        )
        if _run(["powershell", "-NonInteractive", "-Command", script]).stdout.strip() != "removed":
            return 0
    else:
        raise click.ClickException(f"Unsupported service platform: {system}")
    click.echo(f"Removed {kind} service registration.")
    return 1


def _current_user_runtime_processes() -> list[dict[str, Any]]:
    """Token-matched slowave processes owned by this OS account."""
    from slowave.cli.main import _slowave_processes

    tokens = (
        "slowave worker",
        "slowave dashboard",
        "slowave serve",
        "slowave-mcp",
        "slowave.mcp.server",
        "slowave.mcp.http_server",
    )
    return [
        process
        for process in _slowave_processes()
        if int(process.get("pid", 0)) != os.getpid()
        and (process.get("uid") is None or process.get("uid") == os.getuid())
        and any(token in process.get("command", "") for token in tokens)
    ]


def _wait_for_local_runtimes_exit(timeout: float = 10.0) -> None:
    """Wait until stopped daemon/worker processes have fully exited.

    Supervisor stop operations (launchctl bootout in particular) can return
    before the stopped processes have terminated. Without this wait, restore's
    require_stopped_runtime() check would abort on runtimes that are already
    on their way out.
    """
    import time

    from slowave.mcp.daemon import is_running

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not is_running() and not _current_user_runtime_processes():
            return
        time.sleep(0.2)


def stop_registered_services() -> list[str]:
    """Quiesce registered services before database restore, including recovery triggers."""
    import contextlib
    import io

    stopped = []
    system = platform.system()
    for kind in reversed(KINDS):
        if system == "Windows":
            result = _run(
                [
                    "powershell",
                    "-NonInteractive",
                    "-Command",
                    "$ErrorActionPreference='Stop';"
                    + windows_task_candidates(f"Slowave{kind.title()}")
                    + "if ($tasks.Count -gt 0) { Write-Output 'registered' }",
                ]
            )
            registered = result.stdout.strip() == "registered"
        else:
            registered = any(path.exists() for path in service_files(kind))
        if registered:
            # Restore's JSON output must remain machine-readable.
            with contextlib.redirect_stdout(io.StringIO()):
                control("stop", (kind,))
            stopped.append(kind)
    if stopped:
        _wait_for_local_runtimes_exit()
    return stopped


def require_stopped_runtime(operation: str, timeout: float = 10.0) -> None:
    """Refuse destructive file work while a foreground runtime remains visible.

    Detected processes get a short grace period to exit before the refusal, so
    a supervisor stop that returned early does not produce a spurious abort.
    """
    import time

    from slowave.mcp.daemon import is_running

    deadline = time.monotonic() + timeout
    while True:
        remaining = _current_user_runtime_processes()
        if not remaining and not is_running():
            return
        if time.monotonic() >= deadline:
            break
        time.sleep(0.25)
    raise click.ClickException(
        f"{operation} aborted: Slowave processes are still running. Close foreground "
        "dashboard/worker/server terminals and MCP clients, then retry. "
        "Registered services remain stopped."
    )
