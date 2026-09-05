"""Slowave MCP server (stdio transport).

Exposes Slowave as an MCP server so any MCP-aware agent (Cline CLI,
Claude Code, Cursor, ...) can use it as a tool via stdio subprocess.

For a shared HTTP daemon that multiple clients can connect to concurrently,
use ``slowave serve --http`` (or the ``slowave-mcp-http`` entry point).

Tools exposed (5-verb cognitive cycle):
  - slowave_activate    : prime working memory; opens implicit session
  - slowave_remember    : explicitly encode a durable typed claim
  - slowave_recall      : semantic retrieval mid-task
  - slowave_feedback    : append target-specific memory/procedure evidence
  - slowave_commit      : close the task; form episodes

Administrative counts and health remain available through ``slowave stats``;
they are intentionally absent from the cognitive MCP registry.

Deleted (hard break from old surface):
  slowave_context, slowave_session_start, slowave_session_end,
  slowave_event, slowave_retrieval_feedback, slowave_context_feedback

Run directly:
  python -m slowave.mcp.server

Or install and let MCP clients launch it. See README for registration.
"""

from __future__ import annotations

import os

# macOS: avoid OpenMP-duplication crashes when faiss + ONNX Runtime coexist.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")

import logging as _logging

_logging.getLogger("huggingface_hub").setLevel(_logging.ERROR)
_logging.getLogger("onnxruntime").setLevel(_logging.ERROR)

import atexit
import logging
import signal
import sys
import time
from typing import Any

from mcp.server.fastmcp import FastMCP

from slowave.core.config import SlowaveConfig
from slowave.core.engine import SlowaveEngine
from slowave.core.paths import RuntimePaths, ensure_runtime_dirs, runtime_paths
from slowave.mcp import session_reaper
from slowave.mcp.tools import register_tools
from slowave.symbolic.encoder import EncoderConfig

log = logging.getLogger(__name__)

_SERVER_PATHS: RuntimePaths | None = None


def _server_paths() -> RuntimePaths:
    """Resolve once on first server use, after callers have configured env."""
    global _SERVER_PATHS
    if _SERVER_PATHS is None:
        _SERVER_PATHS = runtime_paths()
    return _SERVER_PATHS


# ---------------------------------------------------------------------------
# Engine singleton cache
# ---------------------------------------------------------------------------
_ENGINES: dict[tuple[bool], SlowaveEngine] = {}


def _build_engine(disable_encoder: bool = False) -> SlowaveEngine:
    """Return a cached engine for this configuration.

    Engines are expensive to construct (sentence-transformers model load,
    FAISS index rebuild from SQLite). Caching across MCP calls is essential
    for tolerable latency, since FastMCP keeps the server process alive
    across many tool invocations.

    The cache is keyed by the engine mode because we sometimes want a cheap
    encoder-free engine (e.g. for stats) and sometimes a full latent engine.
    All engines share the same SQLite DB so writes are visible across them.
    """
    key = (disable_encoder,)
    eng = _ENGINES.get(key)
    if eng is not None:
        return eng
    database = str(_server_paths().database)
    db_dir = os.path.dirname(os.path.abspath(database))
    if db_dir and not os.path.exists(db_dir):
        os.makedirs(db_dir, exist_ok=True)
    cfg = SlowaveConfig(
        db_path=database,
        dim=384,
        encoder=EncoderConfig(),
        disable_encoder=disable_encoder,
    )
    eng = SlowaveEngine(cfg)
    _ENGINES[key] = eng
    return eng


# ---------------------------------------------------------------------------
# FastMCP instance (stdio)
# ---------------------------------------------------------------------------
mcp = FastMCP("slowave")

# Register all 6 cognitive-cycle tools from the shared module
register_tools(mcp, _build_engine)


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------
def main() -> None:
    """Entry point: run the MCP server on stdio.

    Registers signal handlers for graceful shutdown and an idle-timeout
    watchdog.  The watchdog exits the process when no MCP message has been
    received for ``SLOWAVE_MCP_IDLE_TIMEOUT`` seconds (default: 1800 = 30 min).
    This is the primary defence against zombie processes: when Cline / Claude
    Code abandons a connection without closing stdin (because the hub-daemon
    keeps the socket alive), the idle timer fires and the process self-exits.

    Set ``SLOWAVE_MCP_IDLE_TIMEOUT=0`` to disable the watchdog entirely.

    Logging note: stdio MCP transport requires that stdout/stderr carry only
    JSON-RPC messages.  Any stray text (including Python log output) corrupts
    the protocol and prevents clients (Claude Desktop, etc.) from detecting the
    server. All log output is therefore redirected below the effective runtime
    root so it is never mixed into the MCP stream.
    """
    # ---------------------------------------------------------------------------
    # Redirect ALL logging to a file — stdout/stderr must stay clean for JSON-RPC
    # ---------------------------------------------------------------------------
    paths = _server_paths()
    ensure_runtime_dirs(paths)
    _log_dir = paths.logs_dir
    _log_file = _log_dir / "mcp-stdio.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [slowave-mcp] %(levelname)s %(name)s: %(message)s",
        handlers=[logging.FileHandler(_log_file, encoding="utf-8")],
        force=True,
    )

    # -- idle-timeout watchdog -----------------------------------------------
    _IDLE_TIMEOUT_S = int(os.environ.get("SLOWAVE_MCP_IDLE_TIMEOUT", "1800"))
    _last_activity: list[float] = [float(time.time())]

    def _touch_activity() -> None:
        _last_activity[0] = float(time.time())

    _orig_build_engine = _build_engine

    def _build_engine_with_touch(disable_encoder: bool = False) -> SlowaveEngine:
        _touch_activity()
        return _orig_build_engine(disable_encoder=disable_encoder)

    import slowave.mcp.server as _self_module

    _self_module._build_engine = _build_engine_with_touch  # type: ignore[attr-defined]

    if _IDLE_TIMEOUT_S > 0:
        import threading

        def _watchdog() -> None:
            while True:
                time.sleep(60)
                idle = time.time() - _last_activity[0]
                if idle >= _IDLE_TIMEOUT_S:
                    log.info(
                        "slowave-mcp: idle for %.0f s (limit %d s), exiting.",
                        idle,
                        _IDLE_TIMEOUT_S,
                    )
                    _cleanup()
                    os._exit(0)

        t = threading.Thread(target=_watchdog, daemon=True, name="slowave-mcp-watchdog")
        t.start()
        log.info(
            "slowave-mcp: idle watchdog active (timeout=%ds, env SLOWAVE_MCP_IDLE_TIMEOUT)",
            _IDLE_TIMEOUT_S,
        )

    # -- cleanup helper ------------------------------------------------------
    def _cleanup() -> None:
        if _ENGINES:
            log.info("Cleaning up cached engines...")
            for key, engine in list(_ENGINES.items()):
                try:
                    engine.close()
                except Exception as e:
                    log.warning("Error closing engine %s: %s", key, e)
            _ENGINES.clear()

    # -- signal handlers -----------------------------------------------------
    def _signal_handler(signum: int, frame: Any) -> None:
        sig_name = signal.Signals(signum).name if hasattr(signal, "Signals") else str(signum)
        log.info("Received signal %s, shutting down gracefully...", sig_name)
        _cleanup()
        sys.exit(0)

    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)
    if hasattr(signal, "SIGHUP"):
        signal.signal(signal.SIGHUP, _signal_handler)

    atexit.register(_cleanup)

    # -- session-idle reaper -------------------------------------------------
    session_reaper.start(build_engine=_build_engine_with_touch, poll_interval_s=120)

    # -- run -----------------------------------------------------------------
    try:
        mcp.run()
    except KeyboardInterrupt:
        log.info("Interrupted by user")
        _cleanup()
        sys.exit(0)
    except Exception as e:
        log.error("MCP server error: %s", e, exc_info=True)
        _cleanup()
        sys.exit(1)


if __name__ == "__main__":
    main()
