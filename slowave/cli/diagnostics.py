"""Diagnostic and health check utilities."""

from __future__ import annotations

import os
import sqlite3
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from slowave.cli.output import CheckResult, Status
from slowave.core.paths import runtime_paths


@dataclass
class RuntimeInfo:
    python_version: str
    python_executable: str
    db_path: str
    config_path: str
    slowave_dir: str
    version: str
    warnings: tuple[str, ...]


def get_runtime_info(version: str, db_path: str | None = None) -> RuntimeInfo:
    paths = runtime_paths()
    effective_db = str(paths.database) if db_path is None else db_path
    slowave_dir = str(paths.root)
    warnings: list[str] = []
    try:
        paths.root.relative_to(Path.home())
    except ValueError:
        warnings.append(f"Runtime root is outside the current user's home/profile: {paths.root}")
    if os.name != "nt" and paths.root.exists():
        try:
            mode = paths.root.stat().st_mode & 0o777
            if mode & 0o022:
                warnings.append(
                    f"Runtime root is group/world writable (mode {mode:04o}): {paths.root}"
                )
        except OSError:
            pass

    return RuntimeInfo(
        python_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        python_executable=sys.executable,
        db_path=effective_db,
        config_path=str(Path(slowave_dir) / "config.toml"),
        slowave_dir=slowave_dir,
        version=version,
        warnings=tuple(warnings),
    )


def check_python() -> CheckResult:
    vi = sys.version_info
    py_ok = (vi.major == 3) and (vi.minor >= 10)
    return CheckResult(
        label=f"Python {vi.major}.{vi.minor}.{vi.micro}",
        status=Status.OK if py_ok else Status.FAIL,
        detail="" if py_ok else "Slowave requires Python 3.10+",
    )


def check_faiss() -> CheckResult:
    try:
        import faiss

        return CheckResult(label=f"FAISS {faiss.__version__}", status=Status.OK)
    except Exception as e:
        return CheckResult(
            label="FAISS",
            status=Status.FAIL,
            detail=str(e)[:100],
            remediation="pip install faiss-cpu",
        )


def check_onnxruntime() -> CheckResult:
    try:
        import onnxruntime as _ort

        return CheckResult(label=f"ONNX Runtime {_ort.__version__}", status=Status.OK)
    except Exception as e:
        return CheckResult(
            label="ONNX Runtime",
            status=Status.FAIL,
            detail=str(e)[:100],
            remediation="pip install onnxruntime",
        )


def check_embedding_backend() -> CheckResult:
    try:
        old_warn = os.environ.get("TRANSFORMERS_NO_ADVISORY_WARNINGS")
        os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
        try:
            from slowave.symbolic.encoder import TextEncoder

            enc = TextEncoder()
            v = enc.encode("test")
            dim = v.shape[0]
            return CheckResult(label=f"Embedding backend (dim={dim})", status=Status.OK)
        finally:
            if old_warn is not None:
                os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = old_warn
            else:
                os.environ.pop("TRANSFORMERS_NO_ADVISORY_WARNINGS", None)
    except Exception as e:
        return CheckResult(
            label="Embedding backend",
            status=Status.FAIL,
            detail=str(e)[:100],
            remediation="check FAISS and ONNX Runtime installation",
        )


def check_sqlite_write() -> CheckResult:
    try:
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            con = sqlite3.connect(tmp_path)
            con.execute("CREATE TABLE t (x INTEGER)")
            con.execute("INSERT INTO t VALUES (1)")
            con.commit()
            con.close()
            return CheckResult(label="SQLite write access", status=Status.OK)
        finally:
            os.unlink(tmp_path)
    except Exception as e:
        return CheckResult(
            label="SQLite write access",
            status=Status.FAIL,
            detail=str(e)[:100],
        )


def check_mcp_server() -> CheckResult:
    """Check that the slowave CLI entry point exists (proxy for correct install)."""
    import shutil

    try:
        bin_path = shutil.which("slowave")
        if bin_path:
            return CheckResult(label="slowave binary", status=Status.OK, detail=bin_path)
        return CheckResult(
            label="slowave binary",
            status=Status.FAIL,
            remediation="run: pip install slowave (or pipx install slowave)",
        )
    except Exception as e:
        return CheckResult(label="slowave binary", status=Status.FAIL, detail=str(e)[:100])


def check_http_daemon(host: str = "127.0.0.1", port: int = 8766) -> CheckResult:
    """Check whether the Slowave HTTP MCP daemon is reachable."""
    import json as _json
    import urllib.error
    import urllib.request

    url = f"http://{host}:{port}/health"
    try:
        with urllib.request.urlopen(url, timeout=2) as resp:
            data = _json.loads(resp.read())
        version = data.get("version", "?")
        sessions = data.get("active_sessions", 0)
        return CheckResult(
            label=f"HTTP MCP daemon (:{port})",
            status=Status.OK,
            detail=f"v{version}, {sessions} active session(s)",
        )
    except urllib.error.URLError:
        return CheckResult(
            label=f"HTTP MCP daemon (:{port})",
            status=Status.SKIP,
            detail="not running",
            remediation="Run: slowave serve start",
        )
    except Exception as e:
        return CheckResult(
            label=f"HTTP MCP daemon (:{port})",
            status=Status.WARN,
            detail=str(e)[:80],
        )
