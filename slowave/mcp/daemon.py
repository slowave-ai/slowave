"""Daemon lifecycle management for the Slowave HTTP MCP daemon.

Handles PID file creation/cleanup and single-instance enforcement so that
``slowave serve start`` can guarantee only ONE backend process is running.

PID file location comes from ``RuntimePaths`` (or the legacy exact-path
``SLOWAVE_DAEMON_PID`` override).
"""

from __future__ import annotations

import logging
import os
import signal
import sys
from pathlib import Path

from slowave.core.paths import runtime_paths

log = logging.getLogger(__name__)


def _pid_file_path(pid_file: Path | None = None) -> Path:
    if pid_file is not None:
        return Path(pid_file)
    env = os.environ.get("SLOWAVE_DAEMON_PID")
    return Path(env).expanduser() if env else runtime_paths().pid_file


def write_pid(pid_file: Path | None = None) -> Path:
    """Write current process PID to the PID file.

    Creates its parent if it does not exist. Returns the PID file path.
    """
    pid_path = _pid_file_path(pid_file)
    pid_path.parent.mkdir(parents=True, exist_ok=True)
    pid_path.write_text(str(os.getpid()))
    log.info("PID file written: %s (pid=%d)", pid_path, os.getpid())
    return pid_path


def remove_pid(pid_file: Path | None = None) -> None:
    """Remove the PID file if it belongs to this process."""
    pid_path = _pid_file_path(pid_file)
    try:
        if pid_path.exists():
            stored = int(pid_path.read_text().strip())
            if stored == os.getpid():
                pid_path.unlink()
                log.info("PID file removed: %s", pid_path)
    except Exception as e:
        log.warning("Could not remove PID file: %s", e)


def read_pid(pid_file: Path | None = None) -> int | None:
    """Return the PID stored in the PID file, or None if not found / unreadable."""
    pid_path = _pid_file_path(pid_file)
    try:
        if pid_path.exists():
            return int(pid_path.read_text().strip())
    except Exception:
        pass
    return None


def _cleanup_stale_pid_file(pid_file: Path | None = None) -> None:
    """Remove the PID file when the stored PID no longer exists or isn't a slowave process."""
    pid_path = _pid_file_path(pid_file)
    try:
        if pid_path.exists():
            pid_path.unlink()
            log.info("Removed stale PID file: %s", pid_path)
    except Exception as e:
        log.warning("Could not remove stale PID file %s: %s", pid_path, e)


def _pid_exists(pid: int) -> bool:
    """Return True if a process with *pid* is alive, cross-platform.

    On Unix this uses ``os.kill(pid, 0)``.  On Windows signal 0 is not
    supported, so we use ``ctypes.windll.kernel32.OpenProcess``.
    """
    if sys.platform != "win32":
        # Unix: signal 0 does an existence check without sending a signal.
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True  # exists, but owned by another user
        return True

    # ---------- Windows ----------
    import ctypes

    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    kernel32 = ctypes.windll.kernel32
    handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if handle == 0:
        # 87 = invalid parameter (pid doesn't exist), 5 = access denied
        # (exists but we can't open it — treat as running).
        err = ctypes.get_last_error()
        if err == 87:  # ERROR_INVALID_PARAMETER
            return False
        # err 5 = ERROR_ACCESS_DENIED — process exists, treat as alive.
        return err == 5
    kernel32.CloseHandle(handle)
    return True


def _is_slowave_process(pid: int) -> bool:
    """Check whether *pid* is actually a slowave process (not a PID-reuse collision).

    On Unix uses ``ps``, on Windows uses PowerShell CIM to read the full
    command line (``wmic`` is removed from Windows 11 24H2+, and ``tasklist
    /V`` only exposes the window title, which a headless daemon launched as
    ``python -m slowave...`` would not contain — that would cause a false
    negative, falsely rejecting a live daemon and allowing a second one to
    start). Returns True when verification succeeds and ``slowave`` appears
    in the command line; returns True on any error so a live daemon is never
    falsely rejected.
    """
    import subprocess

    try:
        if sys.platform == "win32":
            result = subprocess.run(
                [
                    "powershell",
                    "-NonInteractive",
                    "-Command",
                    f"(Get-CimInstance Win32_Process -Filter 'ProcessId={int(pid)}').CommandLine",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode != 0:
                return True  # can't verify — don't reject a live daemon
            return "slowave" in result.stdout.lower()
        else:
            result = subprocess.run(
                ["ps", "-p", str(pid), "-o", "args="],
                capture_output=True,
                text=True,
                timeout=3,
            )
            return "slowave" in result.stdout
    except Exception:
        # Can't verify — err on the side of not breaking a live daemon.
        return True


def is_running(pid_file: Path | None = None) -> bool:
    """Return True if a daemon process with the stored PID is alive *and* is a slowave process."""
    pid = read_pid(pid_file)
    if pid is None:
        return False

    if not _pid_exists(pid):
        _cleanup_stale_pid_file(pid_file)
        return False

    # PID exists, but verify it's actually a slowave process (not a PID-reuse
    # collision from a SIGKILL'd daemon whose PID was reassigned).
    if not _is_slowave_process(pid):
        _cleanup_stale_pid_file(pid_file)
        return False
    return True


def stop_daemon(pid_file: Path | None = None) -> bool:
    """Send SIGTERM (or terminate on Windows) to the running daemon.

    Returns True if a signal was sent, False if no daemon was found.
    Cleans up the stale PID file when the stored process is already gone.
    """
    pid = read_pid(pid_file)
    if pid is None:
        return False
    try:
        if sys.platform == "win32":
            # /F force-terminates (WM_CLOSE won't stop a headless daemon);
            # /T takes down any child processes it spawned.
            import subprocess

            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(pid)],
                capture_output=True,
                timeout=10,
            )
            log.info("Terminated daemon pid=%d via taskkill", pid)
        else:
            os.kill(pid, signal.SIGTERM)
            log.info("Sent SIGTERM to daemon pid=%d", pid)
        return True
    except ProcessLookupError:
        log.warning("Daemon pid=%d not found — removing stale PID file.", pid)
        _cleanup_stale_pid_file(pid_file)
        return False
    except PermissionError as e:
        log.error("Cannot stop daemon pid=%d: %s", pid, e)
        return False
    except Exception as e:
        log.warning("Failed to stop daemon pid=%d: %s", pid, e)
        return False


def daemon_status(pid_file: Path | None = None) -> dict:
    """Return a dict describing daemon state for `slowave serve status`."""
    pid = read_pid(pid_file)
    running = is_running(pid_file)
    return {
        "running": running,
        "pid": pid if running else None,
        "pid_file": str(_pid_file_path(pid_file)),
    }
