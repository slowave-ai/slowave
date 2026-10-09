"""Control installed user services through their owning OS supervisor."""

from __future__ import annotations

import os
import platform
import subprocess
from pathlib import Path

import click

KINDS = ("daemon", "worker", "backup")


def _run(args: list[str], *, check: bool = True) -> subprocess.CompletedProcess:
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise click.ClickException(
            f"Service manager unavailable: {exc}. See docs/troubleshooting.md"
        ) from exc
    if check and result.returncode:
        raise click.ClickException(
            f"Service command failed: {' '.join(args)}\n"
            f"{(result.stderr or result.stdout).strip()}\nRun slowave doctor; see docs/troubleshooting.md"
        )
    return result


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
            if not path.exists():
                if action == "stop":
                    continue
                raise click.ClickException(f"Missing {kind} service. Run slowave setup first.")
            domain = f"gui/{os.getuid()}"
            target = f"{domain}/com.slowave.{kind}"
            loaded = _run(["launchctl", "print", target], check=False).returncode == 0
            if action == "stop" and loaded:
                _run(["launchctl", "bootout", target])
            elif action == "start" and not loaded:
                _run(["launchctl", "bootstrap", domain, str(path)])
        elif system == "Linux":
            target = f"slowave-{kind}" + (".timer" if kind == "backup" else ".service")
            _run(["systemctl", "--user", action, target])
            if action == "stop" and kind == "backup":
                # A timer stop does not stop an already dispatched backup.
                _run(["systemctl", "--user", "stop", "slowave-backup.service"])
        elif system == "Windows":
            name = f"Slowave{kind.title()}"
            if action == "stop":
                script = (
                    "$ErrorActionPreference='Stop';"
                    f"$t=Get-ScheduledTask -TaskName '{name}' -ErrorAction SilentlyContinue;"
                    "if ($t) {"
                    f"Disable-ScheduledTask -TaskName '{name}' | Out-Null;"
                    f"Stop-ScheduledTask -TaskName '{name}';"
                    "$deadline=(Get-Date).AddSeconds(30);"
                    f"while ((Get-ScheduledTask -TaskName '{name}').State -eq 'Running') {{"
                    "if ((Get-Date) -gt $deadline) { throw 'Task did not stop' };"
                    "Start-Sleep -Milliseconds 200 }}"
                )
            else:
                script = (
                    "$ErrorActionPreference='Stop';"
                    f"Enable-ScheduledTask -TaskName '{name}' | Out-Null;"
                )
                if kind != "backup":
                    script += f"Start-ScheduledTask -TaskName '{name}'"
            _run(["powershell", "-NonInteractive", "-Command", script])
        else:
            raise click.ClickException(f"Unsupported service platform: {system}")
        click.echo(f"{kind}: {action}")


def verify_daemon() -> None:
    import json
    import time
    import urllib.request

    from slowave import __version__
    from slowave.core.paths import daemon_port

    url = f"http://127.0.0.1:{daemon_port()}/health"
    last = "not responding"
    for _ in range(90):
        try:
            with urllib.request.urlopen(url, timeout=1) as response:
                data = json.load(response)
            if data.get("version") == __version__:
                click.echo(f"daemon: healthy, version {__version__}")
                return
            last = f"running version {data.get('version')}, installed {__version__}"
        except (OSError, ValueError) as exc:
            last = str(exc)
        time.sleep(0.5)
    raise click.ClickException(f"Daemon verification failed: {last}. Run slowave doctor.")


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
            command = [
                "powershell",
                "-NonInteractive",
                "-Command",
                f"$ErrorActionPreference='Stop'; Get-ScheduledTask -TaskName 'Slowave{kind.title()}' | Select-Object TaskName,State | ConvertTo-Json -Compress",
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
        "services": states,
        "runtime_root": str(runtime_paths().root),
        "dashboard": "Foreground process; stop with Ctrl+C and run slowave dashboard again.",
    }
