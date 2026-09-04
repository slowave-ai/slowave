"""Local Slowave dashboard.

Dependency-free at runtime: stdlib HTTP server + SQLite read APIs + packaged
React UI assets.
The dashboard is local-only by default, with its mutating schema actions
(forget/unforget) available in the normal dashboard command.
"""

from __future__ import annotations

import json
import mimetypes
import os
import re
import sqlite3
import subprocess
import threading
import time
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

from slowave import __version__
from slowave.lifecycle import LIFECYCLE_VERSION

VALID_SCHEMA_STATUSES = (
    "active",
    "needs_review",
    "stale",
    "archived",
    "forgotten",
)
# Contradiction and supersession are client-feedback lifecycle decisions. The
# dashboard displays their resulting statuses; geometry only reports topical
# association (``relates_to``), the sole content-relation type.
# VALID_RELATIONS comment. Kept in sync here since the dashboard is a standalone
# stdlib server with its own copy of this list.
VALID_SCHEMA_RELATIONS = ("relates_to",)

# Dashboard wordmark, copied into the package-owned static output so installed
# wheels do not depend on the repository checkout or current working directory.
_WORDMARK_RELATIVE = Path("img") / "slowave-logo-text-small.jpeg"

_ACTIVITY_SUMMARY_CACHE: dict[tuple[str, str, tuple[Any, ...]], tuple[float, dict[str, int]]] = {}
_ACTIVITY_SUMMARY_CACHE_LOCK = threading.Lock()
_ACTIVITY_SUMMARY_CACHE_TTL = 5.0
_STATIC_DIR = Path(__file__).with_name("static")
_WORDMARK_PATH: Path = _STATIC_DIR / _WORDMARK_RELATIVE
_ICON_PATH: Path = _STATIC_DIR / "img" / "slowave-logo-small.jpeg"

_PRODUCT_LIST_ROUTES = {
    "/",
    "/memory",
    "/retrieval",
    "/procedures",
    "/activity",
    "/diagnostics",
    "/graph",
    "/docs",
}
_PRODUCT_DETAIL_ROUTE = re.compile(r"^/(memory|retrieval|procedures|activity)/[^/]+$")


def _is_product_route(path: str, *, experimental: bool) -> bool:
    """Return whether *path* is an explicitly supported React product route."""

    if path in _PRODUCT_LIST_ROUTES or _PRODUCT_DETAIL_ROUTE.fullmatch(path):
        return True
    return experimental and path == "/diagnostics/labs"


def run_dashboard(
    *,
    db_path: str,
    host: str = "127.0.0.1",
    port: int = 8765,
    refresh_ms: int = 2000,
    allow_actions: bool = True,
    experimental_dashboard: bool = False,
    open_browser: bool = True,
) -> None:
    """Run the local dashboard HTTP server."""
    import sys as _sys

    db_path = os.path.abspath(os.path.expanduser(db_path))
    if host not in ("127.0.0.1", "localhost", "::1"):
        print(
            "WARNING: slowave dashboard is intended for localhost use; "
            f"binding to {host!r} may expose private memories on your network.",
            flush=True,
        )

    handler = _make_handler(
        db_path=db_path,
        refresh_ms=int(refresh_ms),
        allow_actions=bool(allow_actions),
        experimental_dashboard=bool(experimental_dashboard),
    )
    try:
        server = ThreadingHTTPServer((host, int(port)), handler)
    except OSError as exc:
        if exc.errno == 48 or exc.errno == 98:  # EADDRINUSE (macOS=48, Linux=98)
            print(
                f"\n✗  Port {port} is already in use.\n"
                f"   Another slowave dashboard may already be running.\n"
                f"   Open http://{host}:{port} in your browser, or stop it first:\n"
                f"     pkill -f 'slowave dashboard'\n"
                f"   Then re-run: slowave dashboard",
                flush=True,
            )
            _sys.exit(1)
        raise
    url = f"http://{host}:{int(port)}"
    print(f"slowave dashboard: {url}", flush=True)
    print(f"db: {db_path}", flush=True)
    print("Press Ctrl-C to stop.", flush=True)
    if open_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nslowave dashboard: stopping", flush=True)
    finally:
        server.server_close()


def _make_handler(
    *, db_path: str, refresh_ms: int, allow_actions: bool, experimental_dashboard: bool = False
):
    class DashboardHandler(BaseHTTPRequestHandler):
        server_version = "slowave-dashboard/0.2"

        def log_message(self, fmt: str, *args: Any) -> None:
            return

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            path = parsed.path.rstrip("/") or "/"
            qs = parse_qs(parsed.query)
            try:
                if _is_product_route(path, experimental=experimental_dashboard):
                    if (_STATIC_DIR / "index.html").is_file():
                        self._send_static("/index.html", experimental=experimental_dashboard)
                    else:
                        self._send_html(render_index_html(experimental=experimental_dashboard))
                elif path == "/api/home":
                    self._send_json(_home_payload(db_path, qs))
                elif path == "/api/scopes":
                    self._send_json(_scopes_payload(db_path))
                elif path == "/api/status":
                    self._send_json(_status_payload(db_path))
                elif path == "/api/daemon":
                    self._send_json(_daemon_health())
                elif path == "/api/pulse":
                    self._send_json(_pulse_payload(db_path, qs))
                elif path == "/api/histogram":
                    self._send_json(_histogram_payload(db_path, qs))
                elif path == "/api/db/health":
                    self._send_json(_db_health(db_path))
                elif path == "/api/schemas":
                    self._send_json(_schemas_payload(db_path, qs))
                elif path == "/api/retrievals":
                    self._send_json(_retrievals_payload(db_path, qs))
                elif path.startswith("/api/retrievals/"):
                    self._send_json(_retrieval_detail(db_path, unquote(path.split("/")[-1])))
                elif path == "/api/activity":
                    self._send_json(_activity_payload(db_path, qs))
                elif path.startswith("/api/activity/"):
                    self._send_json(_activity_detail(db_path, unquote(path.split("/")[-1])))
                elif path.startswith("/api/procedures/"):
                    self._send_json(_procedure_detail(db_path, unquote(path.split("/")[-1])))
                elif path == "/api/graph/schemas":
                    self._send_json(_schema_graph_payload(db_path, qs))
                elif path.startswith("/api/schemas/"):
                    schema_id = int(path.split("/")[-1].replace("sch_", ""))
                    self._send_json(_schema_detail(db_path, schema_id))
                elif path == "/api/worker/runs":
                    self._send_json(_worker_runs_payload(db_path, qs))
                elif path == "/api/generalization":
                    self._send_json(_generalization_payload(db_path))
                elif path == "/api/episodes":
                    self._send_json(_episodes_payload(db_path, qs))
                elif path == "/api/procedures":
                    self._send_json(_procedures_payload(db_path, qs))
                elif path == "/api/procedural-memory":
                    self._send_json(_procedural_memory_payload(db_path, qs))
                elif path == "/api/labs/rollout" and experimental_dashboard:
                    self._send_json(_labs_rollout_payload(db_path))
                elif path == "/api/prototypes":
                    self._send_json(_prototypes_payload(db_path, qs))
                elif path.startswith("/api/prototypes/") and path.endswith("/members"):
                    proto_id = int(path.split("/")[-2])
                    self._send_json(_prototype_members(db_path, proto_id))
                elif path.startswith("/api/sessions/") and path.endswith("/timeline"):
                    session_id = path.split("/")[-2]
                    self._send_json(_session_timeline(db_path, session_id))
                elif path.startswith("/api/events/"):
                    event_id = int(path.split("/")[-1])
                    self._send_json(_event_detail(db_path, event_id))
                elif path == "/api/debug/graph":
                    self._send_json(_graph_health_payload(db_path))
                elif path == "/img/slowave-logo-text-small.jpeg":
                    self._send_file(_WORDMARK_PATH, "image/jpeg")
                elif path == "/img/slowave-logo-small.jpeg":
                    self._send_file(_ICON_PATH, "image/jpeg")
                elif path.startswith("/assets/") or path in {"/favicon.svg"}:
                    self._send_static(path, experimental=experimental_dashboard)
                else:
                    self._send_json(
                        {"error": "not found", "path": path},
                        status=HTTPStatus.NOT_FOUND,
                    )
            except ValueError as e:
                # Malformed input (a non-numeric id/limit/offset in the path or
                # query string) is a client error, not a server bug -- and
                # str(ValueError) is already a clean message ("invalid literal
                # for int() with base 10: 'abc'"), safe to return as-is (no
                # traceback, unlike the bare `except Exception` below).
                self._send_json({"error": f"invalid request: {e}"}, status=HTTPStatus.BAD_REQUEST)
            except Exception as e:
                self._send_json({"error": str(e)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)

        def do_POST(self) -> None:  # noqa: N802
            if not allow_actions:
                self._send_json(
                    {"error": "mutating actions disabled for this dashboard instance"},
                    status=HTTPStatus.FORBIDDEN,
                )
                return
            parsed = urlparse(self.path)
            path = parsed.path.rstrip("/") or "/"
            try:
                if path.startswith("/api/schemas/") and path.endswith("/forget"):
                    schema_id = int(path.split("/")[-2])
                    length = int(self.headers.get("Content-Length", 0) or 0)
                    body = self.rfile.read(length) if length else b""
                    reason = None
                    if body:
                        try:
                            reason = json.loads(body).get("reason")
                        except Exception:
                            reason = None
                    self._send_json(_forget_schema_action(db_path, schema_id, reason))
                elif path.startswith("/api/schemas/") and path.endswith("/unforget"):
                    schema_id = int(path.split("/")[-2])
                    self._send_json(_unforget_schema_action(db_path, schema_id))
                else:
                    self._send_json(
                        {"error": "not found", "path": path},
                        status=HTTPStatus.NOT_FOUND,
                    )
            except ValueError as e:
                self._send_json({"error": f"invalid request: {e}"}, status=HTTPStatus.BAD_REQUEST)
            except Exception as e:
                self._send_json({"error": str(e)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)

        def _send_html(self, html: str) -> None:
            html = html.replace("__REFRESH_MS__", str(refresh_ms)).replace(
                "__ALLOW_ACTIONS__", "true" if allow_actions else "false"
            )
            html = html.replace("__SLOWAVE_VERSION__", __version__)
            html = html.replace("__LIFECYCLE_VERSION__", LIFECYCLE_VERSION)
            data = html.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _send_static(self, request_path: str, *, experimental: bool) -> None:
            """Serve only files below the package-owned Vite output directory."""
            relative = unquote(request_path).lstrip("/")
            if ".." in Path(relative).parts:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            root = _STATIC_DIR.resolve()
            candidate = (root / relative).resolve()
            try:
                candidate.relative_to(root)
            except ValueError:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            if not candidate.is_file():
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            content_type = mimetypes.guess_type(candidate.name)[0] or "application/octet-stream"
            if candidate.name == "index.html":
                html = candidate.read_text(encoding="utf-8")
                html = html.replace("__EXPERIMENTAL__", "true" if experimental else "false")
                html = html.replace("__REFRESH_MS__", str(refresh_ms)).replace(
                    "__ALLOW_ACTIONS__", "true" if allow_actions else "false"
                )
                html = html.replace("__SLOWAVE_VERSION__", __version__)
                html = html.replace("__LIFECYCLE_VERSION__", LIFECYCLE_VERSION)
                data = html.encode("utf-8")
                content_type = "text/html; charset=utf-8"
            else:
                data = candidate.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header(
                "Cache-Control",
                (
                    "no-store"
                    if candidate.name == "index.html"
                    else "public, max-age=31536000, immutable"
                ),
            )
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _send_json(self, obj: Any, *, status: HTTPStatus = HTTPStatus.OK) -> None:
            data = json.dumps(obj, ensure_ascii=False, default=str).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _send_file(self, file_path: Path, content_type: str) -> None:
            try:
                data = file_path.read_bytes()
            except OSError:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    return DashboardHandler


def _connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    # SQLite performance pragmas: WAL mode allows concurrent readers while a writer is active
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA cache_size=-65536")  # 64MB page cache
    conn.execute("PRAGMA temp_store=MEMORY")
    return conn


def _table_count(conn: sqlite3.Connection, table: str) -> int:
    try:
        row = conn.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()
        return int(row["n"] if row else 0)
    except sqlite3.Error:
        return 0


def _json_loads(value: Any, default: Any) -> Any:
    if value is None:
        return default
    try:
        return json.loads(str(value))
    except Exception:
        return default


def _ids_from_json(value: Any) -> list[int]:
    payload = _json_loads(value, {})
    if isinstance(payload, dict):
        raw = payload.get("ids", [])
    elif isinstance(payload, list):
        raw = payload
    else:
        raw = []
    out: list[int] = []
    for item in raw:
        try:
            out.append(int(item))
        except Exception:
            continue
    return out


def _tags_from_json(value: Any) -> list[str]:
    payload = _json_loads(value, {})
    raw = payload.get("tags", []) if isinstance(payload, dict) else []
    return [str(x) for x in raw]


def _schema_class(facets: dict[str, Any]) -> str | None:
    value = facets.get("schema_class") or facets.get("class") or facets.get("type")
    return None if value in (None, "") else str(value)


# Keys stored in schema facets that are internal to the retrieval engine.
# Keep in sync with slowave/mcp/tools.py::_INTERNAL_FACET_KEYS.
_INTERNAL_FACET_KEYS: frozenset[str] = frozenset({"vsa_vec"})


def _public_facets(facets: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of *facets* with internal/bulky keys removed."""
    return {k: v for k, v in facets.items() if k not in _INTERNAL_FACET_KEYS}


def _schema_row_to_node(row: sqlite3.Row, prototype_ids: list[int] | None = None) -> dict[str, Any]:
    facets = _json_loads(row["facets_json"], {})
    if not isinstance(facets, dict):
        facets = {}
    facets = _public_facets(facets)
    tags = _tags_from_json(row["tags_json"])
    supporting = _ids_from_json(row["supporting_episode_ids"])
    content = str(row["content_text"])
    # Generalization stage (Stage 11) — default 0 for legacy rows without the column
    try:
        gen_stage = int(row["generalization_stage"])
    except (KeyError, TypeError, IndexError):
        gen_stage = 0
    return {
        "id": f"sch_{int(row['id'])}",
        "schema_id": int(row["id"]),
        "label": content if len(content) <= 80 else content[:77] + "...",
        "content": content,
        "scope": row["scope_id"],
        "status": str(row["status"]),
        "stale_reason": row["stale_reason"] if "stale_reason" in row.keys() else None,
        "confidence": float(row["confidence"]),
        "salience": float(row["salience"]),
        "is_labile": bool(row["is_labile"]),
        "facets": facets,
        "schema_class": _schema_class(facets),
        "tags": tags,
        "support_count": len(supporting),
        "prototype_ids": prototype_ids or [],
        "first_formed_ts": int(row["first_formed_ts"]),
        "last_updated_ts": int(row["last_updated_ts"]),
        # Generalization fields (Stage 11)
        "generalization_stage": gen_stage,
        "distinct_scope_count": int(facets.get("distinct_scope_count", 0)),
        "distinct_scope_kind_count": int(facets.get("distinct_scope_kind_count", 0)),
        "scope_breadth_pct": float(facets.get("scope_breadth_pct", 0.0)),
        "scope_kind_breadth_pct": float(facets.get("scope_kind_breadth_pct", 0.0)),
        "cross_scope_recall_count": int(facets.get("cross_scope_recall_count", 0)),
    }


def _count_by_gen_stage(conn: sqlite3.Connection, *, min_stage: int) -> int:
    """Count active schemas with generalization_stage >= min_stage."""
    try:
        row = conn.execute(
            "SELECT COUNT(*) AS n FROM schemas WHERE status = 'active' AND generalization_stage >= ?",
            (min_stage,),
        ).fetchone()
        return int(row["n"]) if row else 0
    except sqlite3.Error:
        return 0


def _status_payload(db_path: str) -> dict[str, Any]:
    exists = os.path.exists(db_path)
    db_file = Path(db_path)
    conn = _connect(db_path) if exists else None
    try:
        stats: dict[str, Any] = {}
        schema_health: dict[str, Any] = {}
        scopes: list[dict[str, Any]] = []
        recent_sessions: list[dict[str, Any]] = []
        last_consolidation_ts: int | None = None
        if conn is not None:
            stats = {
                "sessions": _table_count(conn, "sessions"),
                "raw_events": _table_count(conn, "raw_events"),
                "episodes": _table_count(conn, "episodic_memories"),
                "episode_texts": _table_count(conn, "episode_text"),
                "prototypes": _table_count(conn, "semantic_prototypes"),
                "schemas": _table_count(conn, "schemas"),
                "edges": _table_count(conn, "prototype_edges"),
                "schema_relations": _table_count(conn, "schema_relations"),
                "schema_evidence": _table_count(conn, "schema_evidence"),
                "legacy_feedback_events": _table_count(conn, "context_feedback_events"),
                "v9_feedback_events": _table_count(conn, "feedback_events"),
                "feedback_events": _table_count(conn, "context_feedback_events")
                + _table_count(conn, "feedback_events"),
                # Generalization stage counts (Stage 11)
                "promoted_schemas": _count_by_gen_stage(conn, min_stage=1),
                "global_schemas": _count_by_gen_stage(conn, min_stage=3),
                "known_scopes": _table_count(conn, "scope_registry"),
            }
            lifecycle_health = _lifecycle_health(conn)
            schema_health = _schema_health(conn)
            scopes = [
                {"scope": r["scope"], "sessions": int(r["n"])}
                for r in conn.execute(
                    "SELECT COALESCE(scope_id, '(none)') AS scope, COUNT(*) AS n "
                    "FROM sessions GROUP BY COALESCE(scope_id, '(none)') ORDER BY n DESC"
                ).fetchall()
            ]
            raw_sessions = conn.execute("""
                SELECT s.id, s.agent, s.scope_id, s.started_ts, s.ended_ts,
                       COUNT(re.id) AS events,
                       COUNT(DISTINCT et.episode_id) AS episodes
                FROM sessions s
                LEFT JOIN raw_events re ON re.session_id = s.id
                LEFT JOIN episode_text et ON et.session_id = s.id
                GROUP BY s.id
                ORDER BY s.started_ts DESC
                LIMIT 10
                """).fetchall()
            recent_sessions = []
            for r in raw_sessions:
                d = dict(r)
                started = d.get("started_ts") or 0
                ended = d.get("ended_ts")
                d["duration_seconds"] = int(ended) - int(started) if ended else None
                recent_sessions.append(d)
            # last consolidation: most recently ended session
            lc = conn.execute(
                "SELECT MAX(ended_ts) AS ts FROM sessions WHERE ended_ts IS NOT NULL"
            ).fetchone()
            if lc and lc["ts"]:
                last_consolidation_ts = int(lc["ts"])
            try:
                worker_lc = conn.execute(
                    "SELECT MAX(ended_ts) AS ts FROM worker_runs WHERE ended_ts IS NOT NULL"
                ).fetchone()
                if worker_lc and worker_lc["ts"]:
                    last_consolidation_ts = int(worker_lc["ts"])
            except sqlite3.Error:
                pass
        daemon = _daemon_health()
        processes = _slowave_processes()
        return {
            "slowave_version": __version__,
            "db_path": db_path,
            "db_exists": exists,
            "db_size_bytes": db_file.stat().st_size if exists else 0,
            "wal_size_bytes": (
                Path(db_path + "-wal").stat().st_size if Path(db_path + "-wal").exists() else 0
            ),
            "shm_size_bytes": (
                Path(db_path + "-shm").stat().st_size if Path(db_path + "-shm").exists() else 0
            ),
            "stats": stats,
            "schema_health": schema_health,
            "lifecycle_health": lifecycle_health if conn is not None else {},
            "scopes": scopes,
            "recent_sessions": recent_sessions,
            "daemon": daemon,
            "processes": processes,
            "warnings": _warnings(daemon),
            "last_consolidation_ts": last_consolidation_ts,
            "now_ts": int(time.time()),
        }
    finally:
        if conn is not None:
            conn.close()


def _pulse_payload(db_path: str, qs: dict[str, list[str]]) -> dict[str, Any]:
    """Return three zero-filled bucket series for the EEG multi-channel view.

    Channels:
      - raw_events   : incoming observations (raw_events.ts)
      - episodes     : consolidation pulses  (episodic_memories.ts)
      - schemas      : durable memory writes (schemas.first_formed_ts)

    All three share the same bucket grid so they can be overlaid on one canvas.

    Query params:
        - hours:    look-back window in hours  (default 2, max 8760)
        - bucket_m: bucket size in minutes     (default 5, max 10080)
    """
    requested_hours = (qs.get("hours") or ["2"])[0].strip().lower()
    bucket_m = min(max(_qs_int(qs, "bucket_m", 5), 1), 10080)
    bucket_s = bucket_m * 60
    now = int(time.time())
    if requested_hours == "all":
        window_start = _earliest_memory_history_ts(db_path, fallback=now)
        hours = max(1, (now - window_start + 3599) // 3600)
    else:
        hours = min(max(_qs_int(qs, "hours", 2), 1), 8760)
        window_start = now - hours * 3600
    first_bucket = (window_start // bucket_s) * bucket_s
    last_bucket = (now // bucket_s) * bucket_s

    all_ts: list[int] = []
    t = first_bucket
    while t <= last_bucket:
        all_ts.append(t)
        t += bucket_s

    def _bucketize(rows: list) -> list[dict[str, int]]:
        counts = {int(r["bucket_ts"]): int(r["n"]) for r in rows}
        return [{"ts": ts, "n": counts.get(ts, 0)} for ts in all_ts]

    def _bucket_rows(conn: sqlite3.Connection, table: str, ts_col: str) -> list:
        # Tables may not exist yet on a brand-new database -- treat that the
        # same as "no rows in range" so the pulse still renders (flatlined)
        # instead of the endpoint 500ing.
        try:
            return conn.execute(
                f"""SELECT ({ts_col} / ?) * ? AS bucket_ts, COUNT(*) AS n
                   FROM {table} WHERE {ts_col} >= ? AND {ts_col} <= ?
                   GROUP BY bucket_ts ORDER BY bucket_ts""",
                (bucket_s, bucket_s, window_start, now),
            ).fetchall()
        except sqlite3.Error:
            return []

    conn = _connect(db_path)
    try:
        raw_rows = _bucket_rows(conn, "raw_events", "ts")
        epi_rows = _bucket_rows(conn, "episodic_memories", "ts")
        sch_rows = _bucket_rows(conn, "schemas", "first_formed_ts")

        channels = {
            "raw_events": _bucketize(raw_rows),
            "episodes": _bucketize(epi_rows),
            "schemas": _bucketize(sch_rows),
        }
        global_max = max(
            (b["n"] for ch in channels.values() for b in ch),
            default=0,
        )
        return {
            "channels": channels,
            "global_max": global_max,
            "window_hours": hours,
            "bucket_minutes": bucket_m,
            "window_start": window_start,
            "now_ts": now,
            # legacy single-channel keys so old code doesn't break
            "buckets": channels["raw_events"],
            "total_events": sum(b["n"] for b in channels["raw_events"]),
            "max_n": max((b["n"] for b in channels["raw_events"]), default=0),
        }
    finally:
        conn.close()


def _earliest_memory_history_ts(db_path: str, *, fallback: int) -> int:
    """Return the first retained user-memory record timestamp, if available."""
    if not os.path.exists(db_path):
        return fallback
    conn = _connect(db_path)
    try:
        rows = conn.execute(
            "SELECT MIN(ts) AS ts FROM raw_events "
            "UNION ALL SELECT MIN(ts) AS ts FROM episodic_memories "
            "UNION ALL SELECT MIN(first_formed_ts) AS ts FROM schemas"
        ).fetchall()
        timestamps = [
            int(row["ts"]) for row in rows if row["ts"] is not None and int(row["ts"]) > 0
        ]
        return min(timestamps, default=fallback)
    except sqlite3.Error:
        return fallback
    finally:
        conn.close()


def _qs_int(qs: dict[str, list[str]], key: str, default: int) -> int:
    """Extract a single int query-string param."""
    try:
        return int(qs.get(key, [str(default)])[0])
    except (ValueError, IndexError):
        return default


_HISTOGRAM_RANGES = {
    "1w": (7 * 86400, 86400),
    "1m": (30 * 86400, 86400),
    "1y": (365 * 86400, 7 * 86400),
}

# Worker chart time-range windows (seconds), mirroring the Overview histogram
# toolbar. "all" (no key here) leaves the run list unbounded in time.
_WORKER_RANGES = {
    "1w": 7 * 86400,
    "1m": 30 * 86400,
    "1y": 365 * 86400,
}

# Worker chart bucket sizes (seconds) per range, so longer ranges stay legible
# instead of plotting one bar per pass. Mirrors _HISTOGRAM_RANGES.
_WORKER_BUCKET_RANGES = {
    "1w": (7 * 86400, 86400),
    "1m": (30 * 86400, 86400),
    "1y": (365 * 86400, 7 * 86400),
}


def _earliest_activity_ts(conn: sqlite3.Connection) -> int | None:
    """Earliest timestamp across the three channels backing the histogram."""
    earliest: int | None = None
    for table, col in (
        ("raw_events", "ts"),
        ("episodic_memories", "ts"),
        ("schemas", "first_formed_ts"),
    ):
        try:
            row = conn.execute(f"SELECT MIN({col}) AS m FROM {table}").fetchone()
        except sqlite3.Error:
            continue
        if row and row["m"] is not None:
            m = int(row["m"])
            earliest = m if earliest is None else min(earliest, m)
    return earliest


def _histogram_payload(db_path: str, qs: dict[str, list[str]]) -> dict[str, Any]:
    """Return stacked, zero-filled bucket series for the Overview histogram.

    Same three channels as `/api/pulse` (raw_events, episodes, schemas) but
    bucketed over a much longer, user-selectable window so it can show the
    creation timeline rather than the live short-term pulse.

    Query params:
        - range: one of "1w", "1m", "1y", "all" (default "1w")
    """
    range_key = qs.get("range", ["1w"])[0]
    if range_key not in _HISTOGRAM_RANGES and range_key != "all":
        range_key = "1w"
    now = int(time.time())

    conn = _connect(db_path)
    try:
        if range_key == "all":
            earliest = _earliest_activity_ts(conn)
            window_s = max(now - earliest, 86400) if earliest is not None else 86400
            if window_s <= 60 * 86400:
                bucket_s = 86400
            elif window_s <= 2 * 365 * 86400:
                bucket_s = 7 * 86400
            else:
                bucket_s = 30 * 86400
        else:
            window_s, bucket_s = _HISTOGRAM_RANGES[range_key]

        window_start = now - window_s
        first_bucket = (window_start // bucket_s) * bucket_s

        all_ts: list[int] = []
        t = first_bucket
        while t <= now:
            all_ts.append(t)
            t += bucket_s

        def _bucketize(rows: list) -> list[dict[str, int]]:
            counts = {int(r["bucket_ts"]): int(r["n"]) for r in rows}
            return [{"ts": ts, "n": counts.get(ts, 0)} for ts in all_ts]

        raw_rows = conn.execute(
            """SELECT (ts / ?) * ? AS bucket_ts, COUNT(*) AS n
               FROM raw_events WHERE ts >= ? AND ts <= ?
               GROUP BY bucket_ts ORDER BY bucket_ts""",
            (bucket_s, bucket_s, first_bucket, now),
        ).fetchall()
        epi_rows = conn.execute(
            """SELECT (ts / ?) * ? AS bucket_ts, COUNT(*) AS n
               FROM episodic_memories WHERE ts >= ? AND ts <= ?
               GROUP BY bucket_ts ORDER BY bucket_ts""",
            (bucket_s, bucket_s, first_bucket, now),
        ).fetchall()
        sch_rows = conn.execute(
            """SELECT (first_formed_ts / ?) * ? AS bucket_ts, COUNT(*) AS n
               FROM schemas WHERE first_formed_ts >= ? AND first_formed_ts <= ?
               GROUP BY bucket_ts ORDER BY bucket_ts""",
            (bucket_s, bucket_s, first_bucket, now),
        ).fetchall()

        channels = {
            "raw_events": _bucketize(raw_rows),
            "episodes": _bucketize(epi_rows),
            "schemas": _bucketize(sch_rows),
        }
        stacked_max = max(
            (sum(channels[k][i]["n"] for k in channels) for i in range(len(all_ts))),
            default=0,
        )
        return {
            "channels": channels,
            "stacked_max": stacked_max,
            "range": range_key,
            "bucket_seconds": bucket_s,
            "window_start": first_bucket,
            "now_ts": now,
        }
    finally:
        conn.close()


def _schema_health(conn: sqlite3.Connection) -> dict[str, Any]:
    total = _table_count(conn, "schemas")
    by_status = {
        str(r["status"]): int(r["n"])
        for r in conn.execute(
            "SELECT status, COUNT(*) AS n FROM schemas GROUP BY status"
        ).fetchall()
    }
    active = int(by_status.get("active", 0))
    needs_review = int(by_status.get("needs_review", 0))
    sal = conn.execute(
        "SELECT MIN(salience) AS min_salience, AVG(salience) AS avg_salience, "
        "MAX(salience) AS max_salience FROM schemas WHERE status IN ('active', 'needs_review')"
    ).fetchone()
    dup_rows = 0
    try:
        rows = conn.execute("""
            SELECT scope_id, lower(trim(content_text)) AS norm, COUNT(*) AS n
            FROM schemas
            WHERE status IN ('active', 'needs_review')
            GROUP BY scope_id, lower(trim(content_text))
            HAVING COUNT(*) > 1
            """).fetchall()
        dup_rows = sum(int(r["n"]) - 1 for r in rows)
    except sqlite3.Error:
        dup_rows = 0
    denom = max(1, active + needs_review)
    return {
        "schemas_total": total,
        "schemas_by_status": by_status,
        "active_schemas": active,
        "needs_review_schemas": needs_review,
        "active_exact_duplicate_rows": dup_rows,
        "active_exact_duplicate_ratio": dup_rows / denom,
        "active_salience": {
            "min": (
                0.0 if sal is None or sal["min_salience"] is None else float(sal["min_salience"])
            ),
            "avg": (
                0.0 if sal is None or sal["avg_salience"] is None else float(sal["avg_salience"])
            ),
            "max": (
                0.0 if sal is None or sal["max_salience"] is None else float(sal["max_salience"])
            ),
        },
    }


def _warnings(daemon: dict[str, Any]) -> list[str]:
    out: list[str] = []
    if not daemon.get("running"):
        out.append("HTTP MCP daemon is not running. Run: slowave serve start")
    return out


def _daemon_health() -> dict[str, Any]:
    """Fetch live status from the HTTP MCP daemon health endpoint."""
    try:
        import json as _json
        import urllib.error
        import urllib.request

        with urllib.request.urlopen("http://127.0.0.1:8766/health", timeout=2) as resp:
            data = _json.loads(resp.read())
        return {
            "running": True,
            "version": data.get("version", "?"),
            "active_sessions": data.get("active_sessions", 0),
            "engines_loaded": data.get("engines_loaded", []),
            "url": "http://127.0.0.1:8766/mcp",
            "health_url": "http://127.0.0.1:8766/health",
        }
    except Exception:
        return {
            "running": False,
            "version": None,
            "active_sessions": 0,
            "engines_loaded": [],
            "url": "http://127.0.0.1:8766/mcp",
            "health_url": "http://127.0.0.1:8766/health",
        }


def _slowave_processes() -> list[dict[str, Any]]:
    """List running Slowave worker and dashboard processes (not daemon — managed separately)."""
    try:
        out = subprocess.check_output(
            ["ps", "-axo", "pid,ppid,stat,etime,rss,command"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return []

    # Build a pid→command map for ALL processes in one pass so we can look up
    # parent commands without spawning one ps(1) subprocess per slowave process.
    all_commands: dict[int, str] = {}
    lines = out.splitlines()
    for line in lines[1:]:
        parts = line.strip().split(None, 5)
        if len(parts) >= 6:
            try:
                all_commands[int(parts[0])] = parts[5]
            except (ValueError, IndexError):
                pass

    rows: list[dict[str, Any]] = []
    for line in lines[1:]:
        parts = line.strip().split(None, 5)
        if len(parts) < 6:
            continue
        pid, ppid, stat, etime, rss, command = parts
        is_worker = "slowave worker" in command or (
            "slowave.cli.main" in command and " worker" in command
        )
        is_dashboard = "slowave dashboard" in command or (
            "slowave.cli.main" in command and " dashboard" in command
        )
        if not (is_worker or is_dashboard):
            continue
        all_commands.get(int(ppid)) or None
        rows.append(
            {
                "pid": int(pid),
                "ppid": int(ppid),
                "stat": stat,
                "age_seconds": _parse_etime_seconds(etime),
                "rss_kb": int(rss),
                "command": command,
                "kind": "worker" if is_worker else "dashboard",
            }
        )
    return rows


def _parse_etime_seconds(value: str) -> int:
    """Parse ps(1) elapsed time [[dd-]hh:]mm:ss into seconds."""
    value = str(value).strip()
    days = 0
    if "-" in value:
        day_s, value = value.split("-", 1)
        try:
            days = int(day_s)
        except ValueError:
            days = 0
    parts = value.split(":")
    try:
        if len(parts) == 3:
            hours, minutes, seconds = (int(x) for x in parts)
        elif len(parts) == 2:
            hours = 0
            minutes, seconds = (int(x) for x in parts)
        elif len(parts) == 1:
            hours = 0
            minutes = 0
            seconds = int(parts[0])
        else:
            return 0
    except ValueError:
        return 0
    return days * 86400 + hours * 3600 + minutes * 60 + seconds


def _db_health(db_path: str) -> dict[str, Any]:
    if not os.path.exists(db_path):
        return {
            "db_path": db_path,
            "db_exists": False,
            "checked_at": int(time.time()),
            "integrity_status": "unknown",
        }
    conn = _connect(db_path)
    try:
        pragmas: dict[str, Any] = {}
        for name in (
            "journal_mode",
            "foreign_keys",
            "page_count",
            "page_size",
            "freelist_count",
            "auto_vacuum",
            "synchronous",
            "cache_size",
            "temp_store",
            "busy_timeout",
            "wal_autocheckpoint",
            "user_version",
            "schema_version",
        ):
            try:
                row = conn.execute(f"PRAGMA {name}").fetchone()
                pragmas[name] = row[0] if row is not None else None
            except sqlite3.Error as e:
                pragmas[name] = f"error: {e}"
        try:
            integrity = [r[0] for r in conn.execute("PRAGMA integrity_check").fetchall()]
        except sqlite3.Error as e:
            integrity = [f"error: {e}"]
        try:
            fk = [dict(r) for r in conn.execute("PRAGMA foreign_key_check").fetchall()]
        except sqlite3.Error as e:
            fk = [{"error": str(e)}]
        tables = [
            {
                "name": r["name"],
                "type": r["type"],
                "count": (
                    _table_count(conn, r["name"])
                    if r["type"] == "table" and not str(r["name"]).startswith("sqlite_")
                    else None
                ),
            }
            for r in conn.execute(
                "SELECT name, type FROM sqlite_master "
                "WHERE type IN ('table', 'view') AND name NOT LIKE 'sqlite_%' "
                "ORDER BY type, name"
            ).fetchall()
        ]
        page_count = int(pragmas.get("page_count") or 0)
        page_size = int(pragmas.get("page_size") or 0)
        free_pages = int(pragmas.get("freelist_count") or 0)
        allocated_bytes = page_count * page_size
        free_bytes = free_pages * page_size
        object_counts = {
            str(r["type"]): int(r["count"])
            for r in conn.execute(
                "SELECT type, COUNT(*) AS count FROM sqlite_master "
                "WHERE name NOT LIKE 'sqlite_%' GROUP BY type"
            ).fetchall()
        }
        return {
            "db_path": db_path,
            "db_exists": True,
            "checked_at": int(time.time()),
            "file_size_bytes": os.path.getsize(db_path),
            "wal_size_bytes": _file_size_or_zero(f"{db_path}-wal"),
            "shm_size_bytes": _file_size_or_zero(f"{db_path}-shm"),
            "modified_at": os.path.getmtime(db_path),
            "sqlite_version": sqlite3.sqlite_version,
            "pragmas": pragmas,
            "storage": {
                "allocated_bytes": allocated_bytes,
                "used_bytes": max(0, allocated_bytes - free_bytes),
                "free_bytes": free_bytes,
                "utilization_percent": (
                    round((page_count - free_pages) / page_count * 100, 1) if page_count else 0.0
                ),
            },
            "object_counts": object_counts,
            "integrity_check": integrity,
            "integrity_status": "ok" if integrity == ["ok"] else "needs_attention",
            "foreign_key_check": fk,
            "tables": tables,
        }
    finally:
        conn.close()


def _file_size_or_zero(path: str) -> int:
    try:
        return os.path.getsize(path)
    except OSError:
        return 0


# Server-side sort columns for the Schemas table. Keys are the column ids the
# frontend header sorters send; values are fixed, safe SQL expressions. Sorting
# is applied here (not client-side) so the returned page is ranked across the
# full filtered result set in the DB, not just the previously fetched rows.
_SCHEMA_SORT_COLS: dict[str, str] = {
    "id": "id",
    "status": "status",
    "salience": "salience",
    "confidence": "confidence",
    "stage": "generalization_stage",
    "class": (
        "COALESCE("
        "NULLIF(json_extract(facets_json, '$.schema_class'), ''), "
        "NULLIF(json_extract(facets_json, '$.class'), ''), "
        "NULLIF(json_extract(facets_json, '$.type'), ''), ''"
        ")"
    ),
    "scope": "scope_id",
    # supporting_episode_ids is stored as a JSON *object* {"ids": [...]}; a
    # bare json_array_length() returns 0 for objects (not NULL), so branch on
    # the JSON type to mirror _ids_from_json (handles both object and array).
    "support": (
        "CASE WHEN json_type(supporting_episode_ids) = 'array' "
        "THEN json_array_length(supporting_episode_ids) "
        "ELSE COALESCE(json_array_length(json_extract(supporting_episode_ids, '$.ids')), 0) END"
    ),
    "content": "lower(content_text)",
    "changed": "last_updated_ts",
    "formed": "first_formed_ts",
    "evidence": "evidence_count",
    "exposed": "times_exposed",
    "used": "times_used",
    "use_rate": "CASE WHEN times_exposed > 0 THEN CAST(times_used AS REAL) / times_exposed ELSE -1 END",
    "irrelevant": "times_irrelevant",
    "stale": "times_stale",
    "wrong": "times_wrong",
    "related": "related_count",
    "source_activity": "source_activity_count",
    "last_retrieved": "last_retrieved_ts",
    "last_used": "last_used_ts",
}


def _schemas_payload(db_path: str, qs: dict[str, list[str]]) -> dict[str, Any]:
    limit = max(1, min(100, _qs_int(qs, "limit", _qs_int(qs, "per_page", 50))))
    page = max(1, _qs_int(qs, "page", 1))
    offset = (page - 1) * limit
    status = (qs.get("status") or [""])[0]
    states = [item for item in (qs.get("states") or [status])[0].split(",") if item]
    scope = (qs.get("scope") or [""])[0]
    q = (qs.get("q") or [""])[0].strip().lower()
    if any(item not in VALID_SCHEMA_STATUSES for item in states):
        raise ValueError(f"unknown status filter: {','.join(states)!r}")
    args: list[Any] = []
    where = " WHERE 1=1"
    if states:
        where += f" AND status IN ({','.join(['?'] * len(states))})"
        args.extend(states)
    if scope:
        where += " AND scope_id = ?"
        args.append(scope)
    if q:
        schema_ref = q.removeprefix("sch_")
        if schema_ref.isdigit():
            where += " AND id = ?"
            args.append(int(schema_ref))
        else:
            where += " AND lower(content_text) LIKE ?"
            args.append(f"%{q}%")
    changed_from = _qs_int(qs, "from", 0)
    changed_to = _qs_int(qs, "to", 0)
    if changed_from:
        where += " AND last_updated_ts >= ?"
        args.append(changed_from)
    if changed_to:
        where += " AND last_updated_ts <= ?"
        args.append(changed_to)
    sql = (
        "SELECT schemas.*, (SELECT COUNT(*) FROM schema_evidence se "
        "WHERE se.schema_id = schemas.id) AS evidence_count, "
        "(SELECT COUNT(*) FROM context_recall_items cri "
        "  WHERE cri.memory_id = 'sch_' || schemas.id AND cri.admitted = 1 "
        "  AND cri.memory_type IN ('schema','related')) AS times_exposed, "
        "(SELECT COUNT(DISTINCT fe.retrieval_id) FROM feedback_events fe "
        "  WHERE fe.target_kind = 'memory' AND fe.assessment = 'used' "
        "  AND fe.status = 'accepted' AND fe.target_id = 'sch_' || schemas.id) AS times_used, "
        "(SELECT COUNT(DISTINCT fe.retrieval_id) FROM feedback_events fe "
        "  WHERE fe.target_kind = 'memory' AND fe.assessment = 'irrelevant' "
        "  AND fe.status = 'accepted' AND fe.target_id = 'sch_' || schemas.id) AS times_irrelevant, "
        "(SELECT COUNT(DISTINCT fe.retrieval_id) FROM feedback_events fe "
        "  WHERE fe.target_kind = 'memory' AND fe.assessment = 'stale' "
        "  AND fe.status = 'accepted' AND fe.target_id = 'sch_' || schemas.id) AS times_stale, "
        "(SELECT COUNT(DISTINCT fe.retrieval_id) FROM feedback_events fe "
        "  WHERE fe.target_kind = 'memory' AND fe.assessment = 'wrong' "
        "  AND fe.status = 'accepted' AND fe.target_id = 'sch_' || schemas.id) AS times_wrong, "
        "(SELECT COUNT(*) FROM schema_relations sr WHERE sr.src_schema_id = schemas.id OR sr.dst_schema_id = schemas.id) AS related_count, "
        "(SELECT COUNT(DISTINCT COALESCE(re.session_id, et.session_id)) FROM schema_evidence se "
        "  LEFT JOIN raw_events re ON re.id = se.raw_event_id LEFT JOIN episode_text et ON et.episode_id = se.episode_id "
        "  WHERE se.schema_id = schemas.id AND COALESCE(re.session_id, et.session_id) IS NOT NULL) AS source_activity_count, "
        "(SELECT MAX(cre.created_at) FROM context_recall_items cri JOIN context_recall_events cre ON cre.context_id = cri.context_id "
        "  WHERE cri.memory_id = 'sch_' || schemas.id AND cri.admitted = 1 AND cri.memory_type IN ('schema','related')) AS last_retrieved_ts, "
        "(SELECT MAX(fe.created_at) FROM feedback_events fe "
        "  WHERE fe.target_kind = 'memory' AND fe.assessment = 'used' "
        "  AND fe.status = 'accepted' AND fe.target_id = 'sch_' || schemas.id) AS last_used_ts "
        "FROM schemas" + where
    )
    # Server-side ordering so the displayed page ranks across the full result set.
    sort_col = (qs.get("sort") or [""])[0]
    sort_dir = (qs.get("dir") or ["asc"])[0].lower()
    order_expr = _SCHEMA_SORT_COLS.get(sort_col)
    if order_expr is not None:
        order_dir = "ASC" if sort_dir == "asc" else "DESC"
        # Tie-break on id in the same direction so that toggling a column visibly
        # reverses the whole page even when the primary value is heavily tied
        # (e.g. status/stage/support where most rows share the same value).
        sql += f" ORDER BY {order_expr} {order_dir}, id {order_dir} LIMIT ? OFFSET ?"
    else:
        sql += " ORDER BY last_updated_ts DESC, id DESC LIMIT ? OFFSET ?"
    args.extend((limit, offset))
    conn = _connect(db_path)
    try:
        rows = conn.execute(sql, tuple(args)).fetchall()
        proto_map = _prototype_map(conn, [int(r["id"]) for r in rows])
        items = [_schema_row_to_node(r, proto_map.get(int(r["id"]), [])) for r in rows]
        for item, row in zip(items, rows):
            item["evidence_count"] = int(row["evidence_count"] or 0)
            item["times_exposed"] = int(row["times_exposed"] or 0)
            item["times_used"] = int(row["times_used"] or 0)
            item["times_irrelevant"] = int(row["times_irrelevant"] or 0)
            item["times_stale"] = int(row["times_stale"] or 0)
            item["times_wrong"] = int(row["times_wrong"] or 0)
            item["related_count"] = int(row["related_count"] or 0)
            item["source_activity_count"] = int(row["source_activity_count"] or 0)
            item["last_retrieved_ts"] = (
                int(row["last_retrieved_ts"] or 0) if row["last_retrieved_ts"] else None
            )
            item["last_used_ts"] = int(row["last_used_ts"] or 0) if row["last_used_ts"] else None
        count_args = args[:-2]
        total_row = conn.execute("SELECT COUNT(*) AS n FROM schemas" + where, count_args).fetchone()
        status_rows = conn.execute(
            "SELECT status, COUNT(*) AS n FROM schemas"
            + (" WHERE scope_id = ?" if scope else "")
            + " GROUP BY status",
            [scope] if scope else [],
        ).fetchall()
        summary_scope = " AND scope_id = ?" if scope else ""
        summary_args: list[Any] = [scope] if scope else []
        active_row = conn.execute(
            "SELECT COUNT(*) AS n FROM schemas WHERE status = 'active'" + summary_scope,
            summary_args,
        ).fetchone()
        review_row = conn.execute(
            "SELECT COUNT(*) AS n FROM schemas WHERE status = 'needs_review'" + summary_scope,
            summary_args,
        ).fetchone()
        stale_row = conn.execute(
            "SELECT COUNT(*) AS n FROM schemas WHERE status = 'stale'" + summary_scope,
            summary_args,
        ).fetchone()
        retrieval_conditions = [
            "cri.admitted = 1",
            "cri.memory_id LIKE 'sch_%'",
            "s.status = 'active'",
        ]
        retrieval_args: list[Any] = []
        if scope:
            retrieval_conditions.append("s.scope_id = ?")
            retrieval_args.append(scope)
        if changed_from:
            retrieval_conditions.append("cre.created_at >= ?")
            retrieval_args.append(changed_from)
        if changed_to:
            retrieval_conditions.append("cre.created_at <= ?")
            retrieval_args.append(changed_to)
        retrieved_sql = (
            "FROM context_recall_items cri JOIN context_recall_events cre "
            "ON cre.context_id = cri.context_id JOIN schemas s "
            "ON cri.memory_id = 'sch_' || s.id WHERE " + " AND ".join(retrieval_conditions)
        )
        retrieved_row = conn.execute(
            "SELECT COUNT(DISTINCT cri.memory_id) AS n " + retrieved_sql,
            retrieval_args,
        ).fetchone()
        used_conditions = [
            "f.status = 'accepted'",
            "f.target_kind = 'memory'",
            "f.assessment = 'used'",
            "f.target_id = cri.memory_id",
            "s.status = 'active'",
        ]
        used_args: list[Any] = []
        if scope:
            used_conditions.append("s.scope_id = ?")
            used_args.append(scope)
        if changed_from:
            used_conditions.append("r.created_at >= ?")
            used_args.append(changed_from)
        if changed_to:
            used_conditions.append("r.created_at <= ?")
            used_args.append(changed_to)
        used_row = conn.execute(
            "SELECT COUNT(DISTINCT f.target_id) AS n FROM feedback_events f "
            "JOIN context_recall_events r ON r.context_id = f.retrieval_id "
            "JOIN context_recall_items cri ON cri.context_id = r.context_id "
            "JOIN schemas s ON f.target_id = 'sch_' || s.id WHERE " + " AND ".join(used_conditions),
            used_args,
        ).fetchone()
        return {
            "schemas": items,
            "pagination": {
                "page": page,
                "per_page": limit,
                "total": int(total_row["n"] if total_row else 0),
            },
            "status_counts": {str(r["status"]): int(r["n"]) for r in status_rows},
            "summary": {
                "active": int(active_row["n"] if active_row else 0),
                "needs_review": int(review_row["n"] if review_row else 0),
                "stale": int(stale_row["n"] if stale_row else 0),
                "retrieved_active": int(retrieved_row["n"] if retrieved_row else 0),
                "used_active": int(used_row["n"] if used_row else 0),
            },
        }
    finally:
        conn.close()


def _prototype_map(conn: sqlite3.Connection, schema_ids: list[int]) -> dict[int, list[int]]:
    if not schema_ids:
        return {}
    ph = ",".join(["?"] * len(schema_ids))
    out: dict[int, list[int]] = {sid: [] for sid in schema_ids}
    for r in conn.execute(
        f"SELECT schema_id, prototype_id FROM schema_prototype_map WHERE schema_id IN ({ph})",
        tuple(schema_ids),
    ).fetchall():
        out.setdefault(int(r["schema_id"]), []).append(int(r["prototype_id"]))
    return out


_GRAPH_HARD_CAP = 2000


def _schema_graph_payload(db_path: str, qs: dict[str, list[str]]) -> dict[str, Any]:
    limit_raw = str((qs.get("limit") or [120])[0]).strip().lower()
    limit_all = limit_raw in ("all", "*")
    try:
        limit = (
            _GRAPH_HARD_CAP if limit_all else max(1, min(_GRAPH_HARD_CAP, int(limit_raw or "120")))
        )
    except ValueError:
        limit = 120
    scope = (qs.get("scope") or [""])[0]
    statuses_raw = (qs.get("statuses") or ["active,needs_review,stale"])[0]
    relations_raw = (qs.get("relations") or ["relates_to"])[0]
    statuses = [s for s in statuses_raw.split(",") if s in VALID_SCHEMA_STATUSES]
    relations = [
        r
        for r in relations_raw.split(",")
        if r in VALID_SCHEMA_RELATIONS or r == "coactivated_with"
    ]
    if not statuses:
        statuses = ["active", "needs_review"]
    min_salience = _optional_float((qs.get("min_salience") or [""])[0])
    max_salience = _optional_float((qs.get("max_salience") or [""])[0])
    args: list[Any] = []
    ph_status = ",".join(["?"] * len(statuses))
    sql = f"SELECT * FROM schemas WHERE status IN ({ph_status})"
    args.extend(statuses)
    if scope == "(none)":
        sql += " AND scope_id IS NULL"
    elif scope:
        sql += " AND scope_id = ?"
        args.append(scope)
    if min_salience is not None:
        sql += " AND salience >= ?"
        args.append(float(min_salience))
    if max_salience is not None:
        sql += " AND salience <= ?"
        args.append(float(max_salience))
    sql += " ORDER BY salience DESC, last_updated_ts DESC LIMIT ?"
    args.append(limit)
    conn = _connect(db_path)
    try:
        rows = conn.execute(sql, tuple(args)).fetchall()
        schema_ids = [int(r["id"]) for r in rows]
        proto_map = _prototype_map(conn, schema_ids)
        nodes = [_schema_row_to_node(r, proto_map.get(int(r["id"]), [])) for r in rows]
        edges: list[dict[str, Any]] = []
        schema_id_set = set(schema_ids)
        ph_ids = ",".join(["?"] * len(schema_ids))
        if schema_ids and relations:
            ph_rel = ",".join(["?"] * len(relations))
            edge_rows = [
                r
                for r in conn.execute(
                    f"""
                    SELECT * FROM schema_relations
                    WHERE src_schema_id IN ({ph_ids})
                      AND relation IN ({ph_rel})
                    ORDER BY created_ts DESC
                    """,
                    tuple(schema_ids + relations),
                ).fetchall()
                if int(r["dst_schema_id"]) in schema_id_set
            ]
            for r in edge_rows:
                src = int(r["src_schema_id"])
                dst = int(r["dst_schema_id"])
                rel = str(r["relation"])
                edges.append(
                    {
                        "id": f"rel_{src}_{dst}_{rel}",
                        "source": f"sch_{src}",
                        "target": f"sch_{dst}",
                        "src_schema_id": src,
                        "dst_schema_id": dst,
                        "relation": rel,
                        "confidence": float(r["confidence"]),
                        "reason": r["reason"],
                        "created_ts": int(r["created_ts"]),
                    }
                )
        # Co-activation edges (Phase 2) — usage-based, queried separately from
        # schema_relations because they live in their own table with a different
        # lifecycle (Hebbian strengthen-and-decay, no confidence/reason fields).
        if schema_ids and "coactivated_with" in relations:
            try:
                coact_rows = conn.execute(
                    f"""
                    SELECT src_schema_id, dst_schema_id, weight
                    FROM schema_coactivation
                    WHERE src_schema_id IN ({ph_ids})
                       OR dst_schema_id IN ({ph_ids})
                    """,
                    tuple(schema_ids + schema_ids),
                ).fetchall()
                for r in coact_rows:
                    src = int(r["src_schema_id"])
                    dst = int(r["dst_schema_id"])
                    # Both ends must be visible -- an edge to a node that
                    # didn't make the cut (excluded by scope/status/salience
                    # filters) has no node to attach to. cytoscape() throws
                    # on a dangling source/target reference with no
                    # try/catch around it (_js.py drawGraph()), which kills
                    # the ENTIRE graph render, not just that one edge.
                    # schema_relations' query above already requires this
                    # (dst_schema_id IN schema_id_set); coactivation needs
                    # the same requirement on both ends, not just one.
                    if src not in schema_id_set or dst not in schema_id_set:
                        continue
                    edges.append(
                        {
                            "id": f"coact_{src}_{dst}",
                            "source": f"sch_{src}",
                            "target": f"sch_{dst}",
                            "src_schema_id": src,
                            "dst_schema_id": dst,
                            "relation": "coactivated_with",
                            "confidence": float(r["weight"]),
                            "reason": None,
                            "created_ts": None,
                        }
                    )
            except Exception:
                pass
        return {
            "nodes": nodes,
            "edges": edges,
            "limit": limit,
            "statuses": statuses,
            "relations": list(set(relations) | {"coactivated_with"}),
            "salience_filter": {"min": min_salience, "max": max_salience},
        }
    finally:
        conn.close()


def _optional_float(value: Any) -> float | None:
    try:
        text = str(value).strip()
        if text == "":
            return None
        return float(text)
    except Exception:
        return None


def _occurred_at(metadata: Any) -> int | None:
    """Return the client-supplied source time from decoded event metadata."""
    value = metadata.get("occurred_at") if isinstance(metadata, dict) else None
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _schema_detail(db_path: str, schema_id: int) -> dict[str, Any]:
    conn = _connect(db_path)
    try:
        row = conn.execute("SELECT * FROM schemas WHERE id = ?", (schema_id,)).fetchone()
        if row is None:
            return {"error": "schema not found", "schema_id": schema_id}
        proto_map = _prototype_map(conn, [schema_id])
        schema = _schema_row_to_node(row, proto_map.get(schema_id, []))
        evidence = [
            dict(r)
            for r in conn.execute(
                "SELECT se.*, re.content AS event_content, re.type AS event_type, "
                "re.ts AS event_ts, re.session_id AS event_session, "
                "re.metadata_json AS event_metadata_json "
                "FROM schema_evidence se "
                "LEFT JOIN raw_events re ON re.id = se.raw_event_id "
                "WHERE se.schema_id = ? ORDER BY se.weight DESC LIMIT 50",
                (schema_id,),
            ).fetchall()
        ]
        # Fill missing quote with event_content or episode metadata
        ep_ids = [e["episode_id"] for e in evidence if e.get("episode_id")]
        ep_meta: dict[int, dict[str, Any]] = {}
        if ep_ids:
            placeholders = ",".join(["?"] * len(ep_ids))
            ep_rows = conn.execute(
                f"SELECT id, metadata_json FROM episodic_memories WHERE id IN ({placeholders})",
                ep_ids,
            ).fetchall()
            for r in ep_rows:
                ep_meta[r["id"]] = _json_loads(r["metadata_json"], {})
        for ev in evidence:
            event_metadata = _json_dict(ev.pop("event_metadata_json", "{}"))
            ev["event_occurred_at"] = _occurred_at(event_metadata)
            eid = ev.get("episode_id")
            if eid and eid in ep_meta:
                meta = ep_meta[eid]
                ev["episode_kind"] = str(meta.get("kind", ""))
                ev["episode_session"] = str(meta.get("session_id", ""))
                if not ev.get("quote"):
                    ev["quote"] = str(meta.get("text", meta.get("content", "")))[:300]
            # Fallback: when episode_id is NULL (e.g. remember() with a live
            # session), use the raw_event's session_id from the LEFT JOIN.
            if not ev.get("episode_session"):
                ev["episode_session"] = str(ev.get("event_session") or "")
        # Collect scopes this schema was actually recalled in
        schema["recalled_scopes"] = []
        try:
            scope_rows = conn.execute(
                "SELECT DISTINCT cre.scope_id FROM context_recall_items cri "
                "JOIN context_recall_events cre ON cre.id = cri.context_recall_id "
                "WHERE cri.schema_id = ? AND cri.admitted = 1 AND cre.scope_id IS NOT NULL "
                "ORDER BY cre.scope_id LIMIT 30",
                (schema_id,),
            ).fetchall()
            schema["recalled_scopes"] = [str(r["scope_id"]) for r in scope_rows]
        except Exception:
            pass
        outgoing = [
            dict(r)
            for r in conn.execute(
                "SELECT * FROM schema_relations WHERE src_schema_id = ? ORDER BY created_ts DESC",
                (schema_id,),
            ).fetchall()
        ]
        incoming = [
            dict(r)
            for r in conn.execute(
                "SELECT * FROM schema_relations WHERE dst_schema_id = ? ORDER BY created_ts DESC",
                (schema_id,),
            ).fetchall()
        ]
        # Co-activation edges (Phase 2)
        coact_outgoing: list[dict[str, Any]] = []
        coact_incoming: list[dict[str, Any]] = []
        try:
            coact_outgoing = [
                {
                    "src_schema_id": int(r["src_schema_id"]),
                    "dst_schema_id": int(r["dst_schema_id"]),
                    "relation": "coactivated_with",
                    "weight": float(r["weight"]),
                }
                for r in conn.execute(
                    "SELECT src_schema_id, dst_schema_id, weight FROM schema_coactivation WHERE src_schema_id = ?",
                    (schema_id,),
                ).fetchall()
            ]
            coact_incoming = [
                {
                    "src_schema_id": int(r["src_schema_id"]),
                    "dst_schema_id": int(r["dst_schema_id"]),
                    "relation": "coactivated_with",
                    "weight": float(r["weight"]),
                }
                for r in conn.execute(
                    "SELECT src_schema_id, dst_schema_id, weight FROM schema_coactivation WHERE dst_schema_id = ?",
                    (schema_id,),
                ).fetchall()
            ]
        except Exception:
            pass
        retrievals: list[dict[str, Any]] = []
        feedback: list[dict[str, Any]] = []
        audit: list[dict[str, Any]] = []
        try:
            retrievals = [
                dict(r)
                for r in conn.execute(
                    "SELECT r.context_id AS retrieval_id, r.retrieval_type, r.session_id, "
                    "r.scope_id, r.created_at, i.pathway, i.reason "
                    "FROM context_recall_items i JOIN context_recall_events r "
                    "ON r.context_id = i.context_id "
                    "WHERE i.memory_id IN (?, ?) AND i.admitted = 1 "
                    "ORDER BY r.created_at DESC LIMIT 50",
                    (f"sch_{schema_id}", str(schema_id)),
                ).fetchall()
            ]
        except sqlite3.Error:
            pass
        try:
            feedback = [
                dict(r)
                for r in conn.execute(
                    "SELECT event_id, retrieval_id, assessment, stale_reason, "
                    "replacement_target_id, reason, status, created_at "
                    "FROM feedback_events WHERE target_kind = 'memory' "
                    "AND target_id IN (?, ?) ORDER BY created_at DESC LIMIT 50",
                    (f"sch_{schema_id}", str(schema_id)),
                ).fetchall()
            ]
        except sqlite3.Error:
            pass
        try:
            audit = [
                dict(r)
                for r in conn.execute(
                    "SELECT action, prior_status, reason, created_ts FROM schema_forget_log "
                    "WHERE schema_id = ? ORDER BY created_ts DESC LIMIT 50",
                    (schema_id,),
                ).fetchall()
            ]
        except sqlite3.Error:
            pass
        return {
            "schema": schema,
            "evidence": evidence,
            "outgoing": outgoing,
            "incoming": incoming,
            "coact_outgoing": coact_outgoing,
            "coact_incoming": coact_incoming,
            "retrievals": retrievals,
            "feedback": feedback,
            "audit": audit,
        }
    finally:
        conn.close()


# Mutating actions below are only reachable when the server instance allows
# actions; the standard dashboard command enables them. Logic
# is intentionally a standalone copy of SchemaStore.forget/unforget rather than
# importing schema_store.py, matching this module's "dependency-free: stdlib
# HTTP server + SQLite" design (see module docstring, and VALID_SCHEMA_STATUSES
# above which already duplicates schema_store.py's copy for the same reason).
def _ensure_schema_forget_log(conn: sqlite3.Connection) -> None:
    """Create schema_forget_log if missing.

    The dashboard talks to the DB via raw sqlite3 (_connect), never through
    SlowaveEngine/SQLiteDB.init_schema() -- so a DB whose schema was last
    initialized before this table existed (i.e. any DB only ever opened by
    the dashboard, never by a CLI command that constructs an engine) would
    otherwise 500 with "no such table: schema_forget_log" the first time
    Forget is clicked. Mirrors the CREATE TABLE in storage/schema.sql.
    """
    conn.execute("""
        CREATE TABLE IF NOT EXISTS schema_forget_log (
          id            INTEGER PRIMARY KEY AUTOINCREMENT,
          schema_id     INTEGER NOT NULL,
          action        TEXT NOT NULL,
          prior_status  TEXT NOT NULL,
          reason        TEXT,
          created_ts    INTEGER NOT NULL,
          FOREIGN KEY (schema_id) REFERENCES schemas(id) ON DELETE CASCADE
        )
        """)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_schema_forget_log_schema " "ON schema_forget_log(schema_id)"
    )


def _forget_schema_action(db_path: str, schema_id: int, reason: str | None) -> dict[str, Any]:
    conn = _connect(db_path)
    try:
        _ensure_schema_forget_log(conn)
        row = conn.execute(
            "SELECT status, generalization_stage FROM schemas WHERE id = ?",
            (schema_id,),
        ).fetchone()
        if row is None:
            return {"error": "schema not found", "schema_id": schema_id}
        prior_status = str(row["status"])
        if prior_status == "forgotten":
            return {
                "schema_id": f"sch_{schema_id}",
                "status": "forgotten",
                "prior_status": "forgotten",
            }
        now = int(time.time())
        conn.execute(
            "INSERT INTO schema_forget_log (schema_id, action, prior_status, reason, created_ts) "
            "VALUES (?, 'forget', ?, ?, ?)",
            (schema_id, prior_status, reason, now),
        )
        conn.execute(
            "UPDATE schemas SET status = 'forgotten', last_updated_ts = ? WHERE id = ?",
            (now, schema_id),
        )
        conn.commit()
        result: dict[str, Any] = {
            "schema_id": f"sch_{schema_id}",
            "status": "forgotten",
            "prior_status": prior_status,
        }
        gen_stage = int(row["generalization_stage"] or 0)
        if gen_stage >= 1:
            result["warning"] = (
                f"sch_{schema_id} is generalized (generalization_stage={gen_stage}); "
                "forgetting it removes it from every scope that reuses it, not just this one."
            )
        return result
    finally:
        conn.close()


def _unforget_schema_action(db_path: str, schema_id: int) -> dict[str, Any]:
    conn = _connect(db_path)
    try:
        _ensure_schema_forget_log(conn)
        row = conn.execute("SELECT status FROM schemas WHERE id = ?", (schema_id,)).fetchone()
        if row is None:
            return {"error": "schema not found", "schema_id": schema_id}
        if str(row["status"]) != "forgotten":
            return {
                "error": "schema is not currently forgotten",
                "schema_id": schema_id,
            }
        log_row = conn.execute(
            "SELECT prior_status FROM schema_forget_log "
            "WHERE schema_id = ? AND action = 'forget' ORDER BY id DESC LIMIT 1",
            (schema_id,),
        ).fetchone()
        prior_status = str(log_row["prior_status"]) if log_row is not None else "active"
        now = int(time.time())
        conn.execute(
            "INSERT INTO schema_forget_log (schema_id, action, prior_status, reason, created_ts) "
            "VALUES (?, 'unforget', ?, NULL, ?)",
            (schema_id, prior_status, now),
        )
        conn.execute(
            "UPDATE schemas SET status = ?, last_updated_ts = ? WHERE id = ?",
            (prior_status, now, schema_id),
        )
        conn.commit()
        return {"schema_id": f"sch_{schema_id}", "status": prior_status}
    finally:
        conn.close()


def _generalization_payload(db_path: str) -> dict[str, Any]:
    """Return cross-scope generalization stats: stage distribution + scope registry."""
    if not os.path.exists(db_path):
        return {"stage_distribution": {}, "scope_registry": [], "top_promoted": []}
    conn = _connect(db_path)
    try:
        # Stage distribution across active schemas
        stage_rows = conn.execute(
            "SELECT generalization_stage AS stage, COUNT(*) AS n "
            "FROM schemas WHERE status = 'active' "
            "GROUP BY generalization_stage ORDER BY generalization_stage"
        ).fetchall()
        stage_dist = {int(r["stage"]): int(r["n"]) for r in stage_rows}

        # Scope registry (may not exist on older DBs)
        reg_rows: list[dict[str, Any]] = []
        try:
            reg_rows = [
                {
                    "scope_id": str(r["scope_id"]),
                    "scope_kind": r["scope_kind"],
                    "session_count": int(r["session_count"]),
                    "recall_count": int(r["recall_count"]),
                    "last_active_ts": int(r["last_active_ts"]),
                    "first_seen_ts": int(r["first_seen_ts"]),
                }
                for r in conn.execute(
                    "SELECT * FROM scope_registry ORDER BY last_active_ts DESC"
                ).fetchall()
            ]
        except Exception:
            pass

        # Top promoted schemas (stage >= 1), sorted by breadth
        promoted_rows = conn.execute("""
            SELECT id, content_text, scope_id, generalization_stage,
                   salience, facets_json
            FROM schemas
            WHERE generalization_stage >= 1 AND status = 'active'
            ORDER BY generalization_stage DESC, salience DESC
            LIMIT 50
            """).fetchall()
        top_promoted = []
        for r in promoted_rows:
            facets = _json_loads(r["facets_json"], {})
            top_promoted.append(
                {
                    "id": f"sch_{int(r['id'])}",
                    "schema_id": int(r["id"]),
                    "content": str(r["content_text"])[:200],
                    "scope": r["scope_id"],
                    "stage": int(r["generalization_stage"]),
                    "salience": float(r["salience"]),
                    "distinct_scope_count": int(facets.get("distinct_scope_count", 0)),
                    "distinct_scope_kind_count": int(facets.get("distinct_scope_kind_count", 0)),
                    "scope_breadth_pct": float(facets.get("scope_breadth_pct", 0.0)),
                    "scope_kind_breadth_pct": float(facets.get("scope_kind_breadth_pct", 0.0)),
                    "cross_scope_recall_count": int(facets.get("cross_scope_recall_count", 0)),
                }
            )

        total_active = sum(stage_dist.values())
        promoted_count = sum(v for k, v in stage_dist.items() if k >= 1)
        global_count = stage_dist.get(3, 0)

        return {
            "stage_distribution": stage_dist,
            "scope_registry": reg_rows,
            "top_promoted": top_promoted,
            "summary": {
                "total_active_schemas": total_active,
                "promoted_schemas": promoted_count,
                "global_schemas": global_count,
                "total_known_scopes": len(reg_rows),
                "total_scope_kinds": len({r["scope_kind"] for r in reg_rows if r["scope_kind"]}),
            },
        }
    finally:
        conn.close()


def _worker_chart_buckets(conn: sqlite3.Connection, range_key: str, now: int) -> dict[str, Any]:
    """Bucket worker outcomes over a time window, mirroring the Overview histogram.

    Returns zero-filled per-bucket aggregates (created / reinforced / skipped
    plus decayed, pass and error counts). Bucketing is
    required to make a selected range meaningful: the raw runs table is so dense
    (often 100-260 passes per day) that the most recent 50 passes always sit in
    the last ~1-2 days, so plotting raw runs would keep every range pinned to a
    short window.
    """
    if range_key in _WORKER_BUCKET_RANGES:
        window_s, bucket_s = _WORKER_BUCKET_RANGES[range_key]
        window_start = now - window_s
    else:
        earliest = conn.execute("SELECT MIN(started_ts) AS m FROM worker_runs").fetchone()
        earliest_ts = int(earliest["m"]) if earliest and earliest["m"] else now
        window_start = min(earliest_ts, now)
        window_s = now - earliest_ts
        if window_s <= 60 * 86400:
            bucket_s = 86400
        elif window_s <= 2 * 365 * 86400:
            bucket_s = 7 * 86400
        else:
            bucket_s = 30 * 86400

    first_bucket = (window_start // bucket_s) * bucket_s
    all_ts: list[int] = []
    t = first_bucket
    while t <= now:
        all_ts.append(t)
        t += bucket_s

    rows = conn.execute(
        """SELECT (started_ts / ?) * ? AS bucket_ts,
                  SUM(schemas_created) AS created,
                  SUM(schemas_reinforced) AS reinforced,
                  SUM(schemas_skipped) AS skipped,
                  SUM(schemas_decayed) AS decayed,
                  COUNT(*) AS pass_count,
                  SUM(CASE WHEN error_text IS NOT NULL THEN 1 ELSE 0 END) AS errors
           FROM worker_runs
           WHERE started_ts >= ? AND started_ts <= ?
           GROUP BY bucket_ts ORDER BY bucket_ts""",
        (bucket_s, bucket_s, first_bucket, now),
    ).fetchall()
    counts = {int(r["bucket_ts"]): r for r in rows}

    buckets = []
    for ts in all_ts:
        r = counts.get(ts)
        buckets.append(
            {
                "ts": ts,
                "created": int(r["created"] or 0) if r else 0,
                "reinforced": int(r["reinforced"] or 0) if r else 0,
                "skipped": int(r["skipped"] or 0) if r else 0,
                "decayed": int(r["decayed"] or 0) if r else 0,
                "pass_count": int(r["pass_count"] or 0) if r else 0,
                "errors": int(r["errors"] or 0) if r else 0,
            }
        )
    stacked_max = max(
        (b["created"] + b["reinforced"] for b in buckets),
        default=0,
    )
    return {
        "buckets": buckets,
        "bucket_seconds": bucket_s,
        "stacked_max": stacked_max,
        "window_start": first_bucket,
        "now_ts": now,
        "range": range_key,
        "pass_count_total": sum(b["pass_count"] for b in buckets),
    }


def _worker_runs_payload(db_path: str, qs: dict[str, list[str]]) -> dict[str, Any]:
    """Return worker consolidation run history and summary statistics.

    Query params:
        - limit: maximum number of runs to return (1..200, default 50)
        - range: restrict the run list and the time-bucketed chart to "1w",
          "1m", or "1y". Any other value (including "all") leaves them
          unbounded in time. Summary/trigger statistics are always computed
          over the whole table, matching the Overview stat cards.

    Returns a ``chart`` object with zero-filled time buckets so the Worker chart
    can actually honor the selected range even when thousands of runs fit inside it.
    """
    if not os.path.exists(db_path):
        return {"runs": [], "summary": {}}
    limit = max(1, min(200, _qs_int(qs, "limit", 50)))
    sort = (qs.get("sort") or ["started"])[0].strip()
    direction = (qs.get("dir") or ["desc"])[0].strip().lower()
    run_sorts = {
        "started": "started_ts",
        "duration": "duration_ms",
        "result": "error_text",
        "episodes": "episodes_processed",
        "formed": "schemas_created",
        "reinforced": "schemas_reinforced",
        "retired": "schemas_decayed",
        "errors": "error_text",
        "skipped": "schemas_skipped",
        "pass": "id",
        "error_category": "error_text",
    }
    if sort not in run_sorts:
        sort = "started"
    if direction not in {"asc", "desc"}:
        direction = "desc"
    range_key = qs.get("range", ["all"])[0]
    window_s = _WORKER_RANGES.get(range_key)
    where = ""
    params: list[int] = []
    if window_s is not None:
        where = " WHERE started_ts >= ?"
        params = [int(time.time()) - window_s]
    now = int(time.time())
    conn = _connect(db_path)
    try:
        rows = conn.execute(
            f"SELECT * FROM worker_runs{where} ORDER BY {run_sorts[sort]} {direction.upper()}, started_ts DESC, id DESC LIMIT ?",
            params + [limit],
        ).fetchall()
        runs = [dict(r) for r in rows]
        status_row = conn.execute(
            "SELECT COUNT(*) AS total_passes, MAX(started_ts) AS last_ts,"
            " SUM(CASE WHEN ended_ts IS NOT NULL AND error_text IS NULL THEN 1 ELSE 0 END)"
            " AS successful_passes,"
            " SUM(CASE WHEN error_text IS NOT NULL THEN 1 ELSE 0 END) AS failed_passes,"
            " SUM(CASE WHEN ended_ts IS NULL THEN 1 ELSE 0 END) AS incomplete_passes"
            " FROM worker_runs"
        ).fetchone()
        totals_row = conn.execute(
            "SELECT SUM(prototypes_processed) AS prototypes_processed,"
            " SUM(episodes_processed) AS episodes_processed,"
            " SUM(schemas_created) AS schemas_created,"
            " SUM(schemas_reinforced) AS schemas_reinforced,"
            " SUM(schemas_skipped) AS schemas_skipped,"
            " SUM(schemas_decayed) AS schemas_decayed,"
            " AVG(duration_ms) AS avg_ms"
            " FROM worker_runs WHERE ended_ts IS NOT NULL AND error_text IS NULL"
        ).fetchone()
        recent_cutoff = now - 7 * 86400
        recent_rows = conn.execute(
            "SELECT ended_ts, error_text, duration_ms FROM worker_runs WHERE started_ts >= ?",
            (recent_cutoff,),
        ).fetchall()
        recent_durations = sorted(
            float(row["duration_ms"])
            for row in recent_rows
            if row["duration_ms"] is not None
            and row["ended_ts"] is not None
            and row["error_text"] is None
        )
        if recent_durations:
            p95_index = min(
                len(recent_durations) - 1, max(0, int(len(recent_durations) * 0.95) - 1)
            )
            recent_duration_ms = (
                recent_durations[p95_index]
                if len(recent_durations) >= 4
                else recent_durations[len(recent_durations) // 2]
            )
            recent_duration_stat = "p95" if len(recent_durations) >= 4 else "median"
        else:
            recent_duration_ms = None
            recent_duration_stat = "median"
        trigger_counts = {
            str(r["triggered_by"]): int(r["count"])
            for r in conn.execute(
                "SELECT triggered_by, COUNT(*) AS count FROM worker_runs "
                "GROUP BY triggered_by ORDER BY count DESC"
            ).fetchall()
        }
        worker_processes = [p for p in _slowave_processes() if p["kind"] == "worker"]

        def _total(name: str) -> int:
            return int(totals_row[name] or 0) if totals_row else 0

        return {
            "runs": runs,
            "sort": sort,
            "sort_direction": direction,
            "chart": _worker_chart_buckets(conn, range_key, now),
            "worker": {
                "running": bool(worker_processes),
                "process_count": len(worker_processes),
                "processes": worker_processes,
            },
            "trigger_counts": trigger_counts,
            "summary": {
                "total_passes": int(status_row["total_passes"] or 0) if status_row else 0,
                "successful_passes": (
                    int(status_row["successful_passes"] or 0) if status_row else 0
                ),
                "failed_passes": int(status_row["failed_passes"] or 0) if status_row else 0,
                "incomplete_passes": (
                    int(status_row["incomplete_passes"] or 0) if status_row else 0
                ),
                "last_run_ts": status_row["last_ts"] if status_row else None,
                "total_prototypes_processed": _total("prototypes_processed"),
                "total_episodes_processed": _total("episodes_processed"),
                "total_schemas_created": _total("schemas_created"),
                "total_schemas_reinforced": _total("schemas_reinforced"),
                "total_schemas_skipped": _total("schemas_skipped"),
                "total_schemas_decayed": _total("schemas_decayed"),
                "avg_duration_ms": (
                    round(float(totals_row["avg_ms"] or 0), 1) if totals_row else 0
                ),
                "recent_7d": {
                    "runs": len(recent_rows),
                    "successful": sum(
                        1
                        for row in recent_rows
                        if row["ended_ts"] is not None and row["error_text"] is None
                    ),
                    "failed": sum(1 for row in recent_rows if row["error_text"] is not None),
                    "incomplete": sum(
                        1
                        for row in recent_rows
                        if row["ended_ts"] is None and row["error_text"] is None
                    ),
                    "duration_ms": (
                        round(recent_duration_ms, 1) if recent_duration_ms is not None else None
                    ),
                    "duration_stat": recent_duration_stat,
                },
            },
        }
    except sqlite3.Error:
        return {"runs": [], "summary": {}}
    finally:
        conn.close()


def _episodes_payload(db_path: str, qs: dict[str, list[str]]) -> dict[str, Any]:
    """Return paginated episode list."""
    if not os.path.exists(db_path):
        return {"episodes": [], "total": 0}
    limit = max(1, min(200, _qs_int(qs, "limit", 50)))
    offset = max(0, _qs_int(qs, "offset", 0))
    search = (qs.get("q") or [""])[0].strip()
    conn = _connect(db_path)
    try:
        base_sql = "FROM episodic_memories e LEFT JOIN raw_events r ON r.id = e.event_id"
        base_params: list[Any] = []
        if search:
            base_sql += " WHERE e.metadata_json LIKE ?"
            base_params.append(f"%{search}%")
        total_row = conn.execute(f"SELECT COUNT(*) AS n {base_sql}", base_params).fetchone()
        rows = conn.execute(
            f"SELECT e.id, e.event_id, e.ts, e.ts AS recorded_at, e.salience, e.recalled_count, "
            f"e.metadata_json, r.metadata_json AS event_metadata_json {base_sql} "
            f"ORDER BY e.ts DESC LIMIT ? OFFSET ?",
            base_params + [limit, offset],
        ).fetchall()
        episodes = []
        for r in rows:
            rec = dict(r)
            meta = _json_loads(rec.pop("metadata_json", None), {})
            event_meta = _json_dict(rec.pop("event_metadata_json", "{}"))
            rec["occurred_at"] = _occurred_at(meta) or _occurred_at(event_meta)
            rec["content_preview"] = str(
                meta.get(
                    "text",
                    meta.get(
                        "content",
                        f"{meta.get('kind', '')} session={meta.get('session_id', '')}",
                    ),
                )
            )[:200]
            rec["type"] = str(meta.get("type", meta.get("event_type", "")))
            rec["session_id"] = str(meta.get("session_id", rec.get("event_id", "")))
            episodes.append(rec)
        return {
            "episodes": episodes,
            "total": int(total_row["n"]) if total_row else 0,
        }
    finally:
        conn.close()


def _activity_bucket_minutes(window_seconds: int) -> int:
    """Choose a legible Home activity bucket size for the selected window.

    Keep roughly 24–60 bars on screen: this preserves detail for short
    histories while avoiding an unreadable wall of narrow bars for long ones.
    """

    if window_seconds <= 24 * 3600:
        return 60  # 24 hourly bars
    if window_seconds <= 7 * 24 * 3600:
        return 6 * 60  # 28 six-hour bars
    if window_seconds <= 31 * 24 * 3600:
        return 24 * 60  # 31 daily bars
    if window_seconds <= 366 * 24 * 3600:
        return 7 * 24 * 60  # 53 weekly bars
    if window_seconds <= 2 * 366 * 24 * 3600:
        return 14 * 24 * 60  # about 52 two-week bars
    if window_seconds <= 5 * 366 * 24 * 3600:
        return 30 * 24 * 60  # at most about 61 monthly bars
    return 90 * 24 * 60  # quarterly bars for longer histories


def _home_payload(db_path: str, qs: dict[str, list[str]]) -> dict[str, Any]:
    """Shape conservative Home observations without inventing lifecycle transitions."""

    requested_hours = (qs.get("hours") or ["all"])[0].strip().lower()
    all_time = requested_hours == "all"
    hours = None if all_time else max(1, min(24 * 365, _qs_int(qs, "hours", 3)))
    scope = (qs.get("scope") or [""])[0].strip()
    now = int(time.time())
    since = (
        _earliest_memory_history_ts(db_path, fallback=now) if all_time else now - int(hours) * 3600
    )
    activity_bucket_minutes = _activity_bucket_minutes(max(1, now - since))
    exists = os.path.exists(db_path)
    status = _status_payload(db_path)
    database = _db_health(db_path)
    workers = _worker_runs_payload(db_path, {"limit": ["10"], "range": ["1m"]})
    base = {
        "observed_at": now,
        "window": {"from": since, "to": now, "hours": hours},
        "status": status,
        "database": database,
        "workers": workers,
        "attention": [],
        "recent_changes": [],
        "at_a_glance": {},
        "activity": (
            _pulse_payload(
                db_path,
                {
                    "hours": ["all" if all_time else str(hours)],
                    "bucket_m": [str(activity_bucket_minutes)],
                },
            )
            if exists
            else {"channels": {}, "window_start": since, "now_ts": now}
        ),
    }
    if not exists:
        return base
    conn = _connect(db_path)
    try:
        scope_sql = " AND scope_id = ?" if scope else ""
        scope_args: list[Any] = [scope] if scope else []
        attention: list[dict[str, Any]] = []
        if database.get("integrity_status") == "needs_attention":
            attention.append(
                {
                    "kind": "database",
                    "title": "Database integrity needs attention",
                    "observed_at": database.get("checked_at") or now,
                    "rule": "Shown when SQLite integrity_check does not return ok.",
                    "href": "/diagnostics#database",
                }
            )
        runs = workers.get("runs") or []
        failed = next((run for run in runs if run.get("error_text")), None)
        if failed:
            attention.append(
                {
                    "kind": "maintenance",
                    "title": "A recent maintenance run failed",
                    "observed_at": failed.get("ended_ts") or failed.get("started_ts"),
                    "rule": "Shown for a recorded worker run with an error; idle installations are not flagged.",
                    "href": "/diagnostics#maintenance",
                }
            )
        try:
            incomplete = conn.execute(
                "SELECT COUNT(*) AS n, MAX(ended_ts) AS ts FROM sessions "
                "WHERE ended_ts IS NOT NULL AND feedback_status = 'incomplete'" + scope_sql,
                scope_args,
            ).fetchone()
            if incomplete and int(incomplete["n"] or 0):
                attention.append(
                    {
                        "kind": "feedback",
                        "title": f"Feedback incomplete in {int(incomplete['n'])} ended sessions",
                        "observed_at": incomplete["ts"],
                        "rule": "Counts ended sessions explicitly recorded with incomplete feedback; pending active sessions are excluded.",
                        "href": "/activity?feedback=incomplete",
                    }
                )
        except sqlite3.Error:
            pass
        try:
            review = conn.execute(
                "SELECT COUNT(*) AS n, MAX(last_updated_ts) AS ts FROM schemas "
                "WHERE status IN ('needs_review','stale') AND last_updated_ts >= ?" + scope_sql,
                [since, *scope_args],
            ).fetchone()
            if review and int(review["n"] or 0):
                attention.append(
                    {
                        "kind": "memory",
                        "title": f"{int(review['n'])} memories need review or are out of date",
                        "observed_at": review["ts"],
                        "rule": "Counts current memory rows explicitly marked needs_review or stale during this period.",
                        "href": "/memory?states=needs_review,stale",
                    }
                )
        except sqlite3.Error:
            pass
        changes: list[dict[str, Any]] = []
        try:
            rows = conn.execute(
                "SELECT id, content_text, scope_id, status, first_formed_ts, last_updated_ts "
                "FROM schemas WHERE last_updated_ts >= ?"
                + scope_sql
                + " ORDER BY last_updated_ts DESC LIMIT 30",
                [since, *scope_args],
            ).fetchall()
            changes.extend(
                {
                    "kind": "memory",
                    "id": f"sch_{int(r['id'])}",
                    "title": (
                        "Memory formed"
                        if r["first_formed_ts"] == r["last_updated_ts"]
                        else "Memory state observed"
                    ),
                    "preview": r["content_text"],
                    "scope": r["scope_id"],
                    "state": r["status"],
                    "observed_at": r["last_updated_ts"],
                    "href": f"/memory/sch_{int(r['id'])}",
                    "source": "current memory timestamps",
                }
                for r in rows
            )
        except sqlite3.Error:
            pass
        try:
            rows = conn.execute(
                "SELECT context_id, retrieval_type, query, goal, scope_id, count_n, created_at "
                "FROM context_recall_events WHERE created_at >= ?"
                + scope_sql
                + " ORDER BY created_at DESC LIMIT 30",
                [since, *scope_args],
            ).fetchall()
            changes.extend(
                {
                    "kind": "retrieval",
                    "id": str(r["context_id"]),
                    "title": "Retrieval recorded",
                    "preview": r["query"] or r["goal"] or "Context exposure",
                    "scope": r["scope_id"],
                    "state": "No match" if int(r["count_n"] or 0) == 0 else "Recorded",
                    "observed_at": r["created_at"],
                    "href": f"/retrieval/{r['context_id']}",
                    "source": "retrieval snapshot",
                }
                for r in rows
            )
        except sqlite3.Error:
            pass
        try:
            rows = conn.execute(
                "SELECT id, COALESCE(final_goal, initial_goal, goal, '') AS goal, scope_id, "
                "outcome, feedback_status, ended_ts FROM sessions WHERE ended_ts >= ?"
                + scope_sql
                + " ORDER BY ended_ts DESC LIMIT 30",
                [since, *scope_args],
            ).fetchall()
            changes.extend(
                {
                    "kind": "activity",
                    "id": str(r["id"]),
                    "title": "Activity completed",
                    "preview": r["goal"] or "Completed session",
                    "scope": r["scope_id"],
                    "state": r["feedback_status"] or r["outcome"] or "Complete",
                    "observed_at": r["ended_ts"],
                    "href": f"/activity/{r['id']}",
                    "source": "session close",
                }
                for r in rows
            )
        except sqlite3.Error:
            pass
        try:
            from slowave.symbolic.procedural_memory import load_procedures

            procedures = load_procedures(conn, scope=scope or None)
            changes.extend(
                {
                    "kind": "procedure",
                    "id": str(item["id"]),
                    "title": "Procedure captured",
                    "preview": item.get("summary") or item.get("goal") or "Captured procedure",
                    "scope": item.get("scope_id"),
                    "state": "Captured",
                    "observed_at": item.get("created_at"),
                    "href": f"/procedures/{item['id']}",
                    "source": "session close procedure record",
                }
                for item in procedures
                if int(item.get("created_at") or 0) >= since
            )
        except (sqlite3.Error, ValueError):
            pass
        try:
            current = conn.execute(
                "SELECT COUNT(*) AS n FROM schemas WHERE status IN ('active','needs_review')"
                + scope_sql,
                scope_args,
            ).fetchone()
            changed = conn.execute(
                "SELECT COUNT(*) AS n FROM schemas WHERE last_updated_ts >= ?" + scope_sql,
                [since, *scope_args],
            ).fetchone()
            scopes = conn.execute(
                "SELECT COUNT(DISTINCT scope_id) AS n FROM sessions WHERE scope_id IS NOT NULL"
            ).fetchone()
            base["at_a_glance"] = {
                "current_memories": int(current["n"] if current else 0),
                "changed_memories": int(changed["n"] if changed else 0),
                "active_scopes": int(scopes["n"] if scopes else 0),
            }
        except sqlite3.Error:
            pass
        base["attention"] = attention
        base["recent_changes"] = sorted(
            changes, key=lambda item: int(item.get("observed_at") or 0), reverse=True
        )[:30]
        effectiveness_qs = dict(qs)
        effectiveness_qs["hours"] = ["all" if all_time else str(hours)]
        effectiveness_qs["from"] = [str(since)]
        effectiveness_qs["to"] = [str(now)]
        base["effectiveness"] = _effectiveness_payload(db_path, effectiveness_qs)
        return base
    finally:
        conn.close()


def _scopes_payload(db_path: str) -> dict[str, Any]:
    """List every distinct scope across retrievals and sessions for selection."""
    if not os.path.exists(db_path):
        return {"scopes": []}
    conn = _connect(db_path)
    try:
        rows = conn.execute(
            "SELECT scope_id FROM context_recall_events "
            "WHERE scope_id IS NOT NULL AND scope_id <> '' "
            "UNION SELECT scope_id FROM sessions "
            "WHERE scope_id IS NOT NULL AND scope_id <> '' "
            "UNION SELECT scope_id FROM schemas "
            "WHERE scope_id IS NOT NULL AND scope_id <> '' "
            "ORDER BY scope_id"
        ).fetchall()
        return {"scopes": [str(r["scope_id"]) for r in rows]}
    except sqlite3.Error:
        return {"scopes": []}
    finally:
        conn.close()


def _effectiveness_payload(db_path: str, qs: dict[str, list[str]]) -> dict[str, Any]:
    """Compute cohort-correct memory-effectiveness metrics for the Home surface.

    Defaults to the feedback-enforced population: retrievals recorded under the
    lifecycle v9 contract (introduced 2026-08-17) and later, which carries the
    target-specific, client-authoritative feedback model. Pre-v9 and NULL
    lifecycle records are excluded by default so mixed-history counts cannot
    mislead; ``cohort=all`` opts back in to every readable record.

    Every rate is expressed as (numerator, denominator) so the UI can render
    honest ``X of Y`` statements rather than a bare percentage. Numerators and
    denominators share the same unit: both count distinct targets (a memory or
    procedure), so a memory exposed once but used across several retrievals
    still counts once in each bucket and ``used`` can never exceed ``exposed``.
    Memory exposure is restricted to admitted schema targets (``sch_*``) and,
    when Home supplies a window, to retrievals recorded in that window.
    """
    cohort = (qs.get("cohort") or ["v9"])[0].strip()
    if cohort not in {"v9", "all"}:
        cohort = "v9"
    scope = (qs.get("scope") or [""])[0].strip()
    window_from = _qs_int(qs, "from", 0)
    window_to = _qs_int(qs, "to", 0)
    window_hours = _qs_int(qs, "hours", 0)
    window_sql = ""
    if window_from:
        window_sql += " AND r.created_at >= ?"
    if window_to:
        window_sql += " AND r.created_at <= ?"
    retrieval_args: list[Any] = [scope] if scope else []
    if window_from:
        retrieval_args.append(window_from)
    if window_to:
        retrieval_args.append(window_to)

    base: dict[str, Any] = {
        "cohort": cohort,
        "annotation": (
            "Since lifecycle v9 · August 17"
            if cohort == "v9"
            else "All readable records, including legacy"
        ),
        "scope": scope or None,
        "available_scopes": [],
        "window_hours": window_hours or None,
        "memory_exposed": 0,
        "memory_total": 0,
        "memory_used": 0,
        "memory_assessed": 0,
        "memory_irrelevant": 0,
        "memory_stale": 0,
        "memory_wrong": 0,
        "procedure_exposed": 0,
        "procedure_used": 0,
        "procedure_not_used": 0,
        "procedure_helped": 0,
        "procedure_no_effect": 0,
        "procedure_harmed": 0,
        "retrievals_total": 0,
        "retrievals_no_match": 0,
        "retrievals_feedback_complete": 0,
    }
    if not os.path.exists(db_path):
        return base

    # Numeric comparison: 'v10' would sort before 'v9' under a naive string
    # comparison, so strip the leading 'v' and compare as integers. This keeps
    # v9, v10, and any future v11+ in the feedback-enforced cohort while
    # excluding pre-v9 (v8 and earlier) and NULL legacy records.
    cohort_sql = "CAST(substr(r.lifecycle_version, 2) AS INTEGER) >= 9" if cohort == "v9" else "1=1"
    scope_sql = " AND r.scope_id = ?" if scope else ""

    conn = _connect(db_path)
    try:
        try:
            base["available_scopes"] = [
                str(r["scope_id"])
                for r in conn.execute(
                    "SELECT scope_id FROM context_recall_events "
                    "WHERE scope_id IS NOT NULL AND scope_id <> '' "
                    "UNION SELECT scope_id FROM sessions "
                    "WHERE scope_id IS NOT NULL AND scope_id <> '' "
                    "ORDER BY scope_id"
                ).fetchall()
            ]
        except sqlite3.Error:
            pass

        try:
            schema_scope_sql = " AND s.scope_id = ?" if scope else ""
            schema_row = conn.execute(
                "SELECT COUNT(*) AS n FROM schemas s "
                "WHERE s.status = 'active'" + schema_scope_sql,
                [scope] if scope else [],
            ).fetchone()
            base["memory_total"] = int(schema_row["n"] if schema_row else 0)
        except sqlite3.Error:
            pass

        try:
            total_row = conn.execute(
                f"SELECT COUNT(*) AS total, "
                f"SUM(CASE WHEN r.count_n = 0 THEN 1 ELSE 0 END) AS no_match, "
                f"SUM(CASE WHEN EXISTS (SELECT 1 FROM feedback_events f "
                f"  WHERE f.retrieval_id = r.context_id AND f.status = 'accepted' "
                f"  AND f.coverage = 'complete') THEN 1 ELSE 0 END) AS feedback_complete "
                f"FROM context_recall_events r WHERE {cohort_sql}{scope_sql}{window_sql}",
                retrieval_args,
            ).fetchone()
            if total_row:
                base["retrievals_total"] = int(total_row["total"] or 0)
                base["retrievals_no_match"] = int(total_row["no_match"] or 0)
                base["retrievals_feedback_complete"] = int(total_row["feedback_complete"] or 0)
        except sqlite3.Error:
            pass

        try:
            exposure_row = conn.execute(
                f"SELECT COUNT(DISTINCT i.memory_id) AS n "
                f"FROM context_recall_items i "
                f"JOIN context_recall_events r ON r.context_id = i.context_id "
                f"WHERE i.admitted = 1 AND i.memory_type IN ('schema', 'related') "
                f"AND i.memory_id LIKE 'sch_%' AND {cohort_sql}{scope_sql}{window_sql}",
                retrieval_args,
            ).fetchone()
            if exposure_row:
                base["memory_exposed"] = int(exposure_row["n"] or 0)

            procedure_row = conn.execute(
                f"SELECT COUNT(DISTINCT i.memory_id) AS n "
                f"FROM context_recall_items i "
                f"JOIN context_recall_events r ON r.context_id = i.context_id "
                f"WHERE i.admitted = 1 AND i.memory_type IN ('procedural_memory', 'procedure') "
                f"AND {cohort_sql}{scope_sql}{window_sql}",
                retrieval_args,
            ).fetchone()
            if procedure_row:
                base["procedure_exposed"] = int(procedure_row["n"] or 0)
        except sqlite3.Error:
            pass

        try:
            assessed_row = conn.execute(
                f"SELECT COUNT(DISTINCT f.target_id) AS n FROM feedback_events f "
                f"JOIN context_recall_events r ON r.context_id = f.retrieval_id "
                f"JOIN context_recall_items i ON i.context_id = r.context_id AND i.memory_id = f.target_id AND i.admitted = 1 "
                f"JOIN schemas s ON f.target_id = 'sch_' || s.id "
                f"WHERE f.status = 'accepted' AND f.target_kind = 'memory' AND s.status = 'active' "
                f"AND {cohort_sql}{scope_sql}{window_sql}",
                retrieval_args,
            ).fetchone()
            base["memory_assessed"] = int(assessed_row["n"] or 0) if assessed_row else 0
        except sqlite3.Error:
            base["memory_assessed"] = 0

        try:
            for row in conn.execute(
                f"SELECT f.target_kind AS kind, f.assessment AS assessment, "
                f"f.effect AS effect, COUNT(DISTINCT f.target_id) AS n "
                f"FROM feedback_events f "
                f"JOIN context_recall_events r ON r.context_id = f.retrieval_id "
                f"WHERE f.status = 'accepted' "
                f"AND (f.target_kind = 'procedure' OR f.target_id LIKE 'sch_%') "
                f"AND {cohort_sql}{scope_sql}{window_sql} "
                f"GROUP BY f.target_kind, f.assessment, f.effect",
                retrieval_args,
            ).fetchall():
                kind = str(row["kind"])
                assessment = str(row["assessment"] or "")
                effect = str(row["effect"] or "")
                n = int(row["n"] or 0)
                if kind == "memory":
                    if assessment == "used":
                        base["memory_used"] += n
                    elif assessment == "irrelevant":
                        base["memory_irrelevant"] += n
                    elif assessment == "stale":
                        base["memory_stale"] += n
                    elif assessment == "wrong":
                        base["memory_wrong"] += n
                elif kind == "procedure":
                    if assessment == "used":
                        base["procedure_used"] += n
                    elif assessment == "not_used":
                        base["procedure_not_used"] += n
                    if effect == "helped":
                        base["procedure_helped"] += n
                    elif effect == "no_effect":
                        base["procedure_no_effect"] += n
                    elif effect == "harmed":
                        base["procedure_harmed"] += n
        except sqlite3.Error:
            pass
        return base
    finally:
        conn.close()


_RETRIEVAL_SIGNAL_KEYS = (
    "used",
    "not_used",
    "irrelevant",
    "stale",
    "wrong",
    "helped",
    "no_effect",
    "harmed",
    "unknown",
)


def _retrieval_signal_expression(key: str) -> str:
    observed = (
        "(SELECT COUNT(*) FROM feedback_events f "
        "WHERE f.retrieval_id = r.context_id AND f.status = 'accepted' "
        f"AND (f.assessment = '{key}' OR f.effect = '{key}')"
        ")"
    )
    if key != "unknown":
        return observed
    return (
        f"({observed} + CASE WHEN NOT EXISTS ("
        "SELECT 1 FROM feedback_events f WHERE f.retrieval_id = r.context_id "
        "AND f.status = 'accepted' AND (f.assessment IS NOT NULL OR f.effect IS NOT NULL)"
        ") THEN 1 ELSE 0 END)"
    )


def _retrieval_effect_expression() -> str:
    """Rank the effect badge shown in the retrieval list from unknown to harmful."""
    return (
        "CASE "
        "WHEN EXISTS (SELECT 1 FROM feedback_events f WHERE f.retrieval_id=r.context_id "
        "AND f.status='accepted' AND f.effect='harmed') THEN 3 "
        "WHEN EXISTS (SELECT 1 FROM feedback_events f WHERE f.retrieval_id=r.context_id "
        "AND f.status='accepted' AND f.effect='no_effect') THEN 2 "
        "WHEN EXISTS (SELECT 1 FROM feedback_events f WHERE f.retrieval_id=r.context_id "
        "AND f.status='accepted' AND f.effect='helped') THEN 1 "
        "ELSE 0 END"
    )


def _retrieval_sort(qs: dict[str, list[str]]) -> tuple[str, str]:
    sort = (qs.get("sort") or ["when"])[0].strip()
    allowed = {
        "when",
        "task",
        "type",
        "scope",
        "activity",
        "result",
        "exposed",
        "memories_retrieved",
        "procedures_retrieved",
        "effect",
        "feedback",
    }
    allowed.update(_RETRIEVAL_SIGNAL_KEYS)
    if sort not in allowed:
        sort = "when"
    direction = (qs.get("dir") or ["desc"])[0].strip().lower()
    if direction not in {"asc", "desc"}:
        direction = "desc"
    return sort, direction


def _retrieval_filters(qs: dict[str, list[str]]) -> tuple[str, list[Any]]:
    clauses = ["1=1"]
    args: list[Any] = []
    scope = (qs.get("scope") or [""])[0].strip()
    retrieval_type = (qs.get("type") or [""])[0].strip()
    feedback = (qs.get("feedback") or [""])[0].strip()
    no_match = (qs.get("no_match") or [""])[0].strip()
    contains = (qs.get("contains") or [""])[0].strip()
    if scope:
        clauses.append("r.scope_id = ?")
        args.append(scope)
    if retrieval_type:
        clauses.append("r.retrieval_type = ?")
        args.append(retrieval_type)
    if feedback == "complete":
        clauses.append(
            "EXISTS (SELECT 1 FROM feedback_events f WHERE f.retrieval_id=r.context_id AND f.status='accepted' AND f.coverage='complete')"
        )
    elif feedback == "incomplete":
        clauses.append(
            "NOT EXISTS (SELECT 1 FROM feedback_events f WHERE f.retrieval_id=r.context_id AND f.status='accepted' AND f.coverage='complete')"
        )
    if no_match == "true":
        clauses.append("r.count_n = 0")
    elif no_match == "false":
        clauses.append("r.count_n > 0")
    if contains in {"memory", "procedure"}:
        kind = "procedure" if contains == "procedure" else "schema"
        clauses.append(
            "EXISTS (SELECT 1 FROM context_recall_items i WHERE i.context_id=r.context_id AND i.admitted=1 AND i.memory_type=?)"
        )
        args.append(kind)
    if (qs.get("include_internal") or ["false"])[0] != "true":
        clauses.append(
            "lower(COALESCE(r.query,'')) NOT LIKE '%<hook_prompt%' AND lower(COALESCE(r.query,'')) NOT LIKE '%slowave mandatory:%'"
        )
    from_ts = _qs_int(qs, "from", 0)
    to_ts = _qs_int(qs, "to", 0)
    if from_ts:
        clauses.append("r.created_at >= ?")
        args.append(from_ts)
    if to_ts:
        clauses.append("r.created_at <= ?")
        args.append(to_ts)
    search = (qs.get("q") or [""])[0].strip().casefold()
    if search:
        clauses.append("(lower(COALESCE(r.query,'')) LIKE ? OR lower(COALESCE(r.goal,'')) LIKE ?)")
        args.extend((f"%{search}%", f"%{search}%"))
    return " AND ".join(clauses), args


def _retrievals_payload(db_path: str, qs: dict[str, list[str]]) -> dict[str, Any]:
    if not os.path.exists(db_path):
        return {
            "retrievals": [],
            "summary": {},
            "pagination": {"page": 1, "per_page": 50, "total": 0},
        }
    page = max(1, _qs_int(qs, "page", 1))
    per_page = max(1, min(100, _qs_int(qs, "per_page", 50)))
    where, args = _retrieval_filters(qs)
    sort, direction = _retrieval_sort(qs)
    sort_columns = {
        "when": "r.created_at",
        "task": "lower(COALESCE(NULLIF(r.query, ''), NULLIF(r.goal, ''), ''))",
        "type": "r.retrieval_type",
        "scope": "lower(COALESCE(r.scope_id, ''))",
        "activity": "lower(COALESCE(r.session_id, ''))",
        "result": "CASE WHEN r.count_n = 0 THEN 0 ELSE 1 END",
        "exposed": "exposed_count",
        "memories_retrieved": "memory_count",
        "procedures_retrieved": "procedure_count",
        "effect": "effect_rank",
        "feedback": "feedback_complete",
        **{key: f"signal_{key}" for key in _RETRIEVAL_SIGNAL_KEYS},
    }
    signal_select = ", ".join(
        f"{_retrieval_signal_expression(key)} AS signal_{key}" for key in _RETRIEVAL_SIGNAL_KEYS
    )
    conn = _connect(db_path)
    try:
        total = conn.execute(
            f"SELECT COUNT(*) AS n FROM context_recall_events r WHERE {where}", args
        ).fetchone()
        rows = conn.execute(
            "SELECT r.*, "
            "(SELECT COUNT(*) FROM context_recall_items i WHERE i.context_id=r.context_id AND i.admitted=1) AS exposed_count, "
            "(SELECT COUNT(*) FROM context_recall_items i WHERE i.context_id=r.context_id AND i.admitted=1 AND i.memory_type IN ('schema','related')) AS memory_count, "
            "(SELECT COUNT(*) FROM context_recall_items i WHERE i.context_id=r.context_id AND i.admitted=1 AND i.memory_type IN ('procedure','procedural_memory')) AS procedure_count, "
            "EXISTS(SELECT 1 FROM feedback_events f WHERE f.retrieval_id=r.context_id AND f.status='accepted' AND f.coverage='complete') AS feedback_complete, "
            "EXISTS(SELECT 1 FROM feedback_events f WHERE f.retrieval_id=r.context_id AND f.status='accepted') AS feedback_observed, "
            f"{_retrieval_effect_expression()} AS effect_rank, "
            f"{signal_select} "
            f"FROM context_recall_events r WHERE {where} "
            f"ORDER BY {sort_columns[sort]} {direction.upper()}, r.created_at DESC, r.context_id DESC LIMIT ? OFFSET ?",
            [*args, per_page, (page - 1) * per_page],
        ).fetchall()
        ids = [str(r["context_id"]) for r in rows]
        feedback_by_id: dict[str, list[dict[str, Any]]] = {item: [] for item in ids}
        if ids:
            placeholders = ",".join(["?"] * len(ids))
            for row in conn.execute(
                f"SELECT retrieval_id, target_kind, assessment, effect, status FROM feedback_events WHERE retrieval_id IN ({placeholders}) ORDER BY created_at",
                ids,
            ).fetchall():
                feedback_by_id[str(row["retrieval_id"])].append(dict(row))
        retrievals = []
        for row in rows:
            item = dict(row)
            item.pop("cue_embedding", None)
            item["task_preview"] = item.get("query") or item.get("goal") or "Context exposure"
            item["is_internal"] = _is_lifecycle_hook_query(item.get("query"))
            item["feedback"] = feedback_by_id.get(str(item["context_id"]), [])
            signal_counts = {key: 0 for key in _RETRIEVAL_SIGNAL_KEYS}
            observed = False
            for feedback_item in item["feedback"]:
                if feedback_item.get("status") != "accepted":
                    continue
                for field in ("assessment", "effect"):
                    value = str(feedback_item.get(field) or "")
                    if not value:
                        continue
                    observed = True
                    if value in signal_counts:
                        signal_counts[value] += 1
            if not observed:
                signal_counts["unknown"] = 1
            item["signal_counts"] = signal_counts
            item["result"] = "no_match" if int(item.get("exposed_count") or 0) == 0 else "retrieved"
            item["feedback_state"] = (
                "complete"
                if item.get("feedback_complete")
                else "incomplete" if item.get("feedback_observed") else "unknown"
            )
            retrievals.append(item)
        summary = conn.execute(
            "SELECT COUNT(*) AS retrievals, SUM(CASE WHEN r.count_n=0 THEN 1 ELSE 0 END) AS no_match, "
            "SUM(CASE WHEN EXISTS(SELECT 1 FROM feedback_events f WHERE f.retrieval_id=r.context_id AND f.status='accepted' AND f.coverage='complete') THEN 1 ELSE 0 END) AS feedback_complete, "
            "SUM((SELECT COUNT(*) FROM context_recall_items i WHERE i.context_id=r.context_id AND i.admitted=1)) AS exposed "
            f"FROM context_recall_events r WHERE {where}",
            args,
        ).fetchone()
        assessed = conn.execute(
            "SELECT COUNT(DISTINCT f.retrieval_id || ':' || f.target_kind || ':' || f.target_id) AS n "
            "FROM feedback_events f JOIN context_recall_events r ON r.context_id=f.retrieval_id "
            f"WHERE f.status='accepted' AND f.target_kind IN ('memory','procedure') AND {where}",
            args,
        ).fetchone()
        summary_dict = dict(summary) if summary else {}
        summary_dict["assessed"] = int(assessed["n"] if assessed else 0)
        demonstrated = conn.execute(
            "SELECT COUNT(*) AS n FROM context_recall_events r "
            "WHERE " + where + " AND EXISTS ("
            "SELECT 1 FROM feedback_events fc WHERE fc.retrieval_id = r.context_id "
            "AND fc.status = 'accepted' AND fc.coverage = 'complete'"
            ") AND EXISTS ("
            "SELECT 1 FROM feedback_events f JOIN context_recall_items i "
            "ON i.context_id = r.context_id AND i.memory_id = f.target_id AND i.admitted = 1 "
            "WHERE f.retrieval_id = r.context_id AND f.status = 'accepted' "
            "AND (f.assessment = 'used' OR f.effect = 'helped')"
            ")",
            args,
        ).fetchone()
        summary_dict["demonstrated_value"] = int(demonstrated["n"] if demonstrated else 0)
        summary_dict["unknown"] = max(
            0, int(summary_dict.get("exposed") or 0) - summary_dict["assessed"]
        )
        return {
            "retrievals": retrievals,
            "summary": summary_dict,
            "pagination": {
                "page": page,
                "per_page": per_page,
                "total": int(total["n"] if total else 0),
            },
        }
    except sqlite3.Error:
        return {
            "retrievals": [],
            "summary": {},
            "pagination": {"page": page, "per_page": per_page, "total": 0},
        }
    finally:
        conn.close()


def _retrieval_detail(db_path: str, retrieval_id: str) -> dict[str, Any]:
    if not os.path.exists(db_path):
        return {"error": "db not found"}
    conn = _connect(db_path)
    try:
        row = conn.execute(
            "SELECT * FROM context_recall_events WHERE context_id = ?", (retrieval_id,)
        ).fetchone()
        if row is None:
            return {"error": "retrieval not found", "retrieval_id": retrieval_id}
        retrieval = dict(row)
        retrieval.pop("cue_embedding", None)
        retrieval["situation"] = _json_dict(retrieval.pop("situation_json", "{}"))
        retrieval["requirements"] = _json_list(retrieval.pop("requirements_json", "[]"))
        retrieval["topics"] = _json_list(retrieval.pop("topics_json", "[]"))
        retrieval["entities"] = _json_list(retrieval.pop("entities_json", "[]"))
        retrieval["is_internal"] = _is_lifecycle_hook_query(retrieval.get("query"))
        items = [
            dict(item)
            for item in conn.execute(
                "SELECT i.memory_id, i.memory_type, i.rank, i.reason, "
                "COALESCE(s.content_text, i.content_text) AS content_text, "
                "COALESCE(s.status, i.status) AS status, "
                "i.pathway, i.admitted, i.created_at "
                "FROM context_recall_items i "
                "LEFT JOIN schemas s ON i.memory_type IN ('schema', 'related') "
                "AND i.memory_id = 'sch_' || s.id "
                "WHERE i.context_id = ? AND i.admitted = 1 ORDER BY i.rank",
                (retrieval_id,),
            ).fetchall()
        ]
        for item in items:
            pathway = str(item.get("pathway") or "")
            mapping = {
                "direct": ("Direct", "Matched the active task/query."),
                "graph": ("Associated", "Related memory; the stored pathway is graph."),
                "exploration": ("Associated", "Exploratory; filled a bounded exploratory slot."),
                "context_reinstatement": (
                    "Associated",
                    "Prior context from the active continuity.",
                ),
            }
            item["pathway_group"], item["pathway_explanation"] = mapping.get(
                pathway,
                ("Unknown pathway", "The persisted pathway is not recognized and was not guessed."),
            )
        feedback = [
            dict(item)
            for item in conn.execute(
                "SELECT event_id, target_kind, target_id, replacement_target_id, assessment, stale_reason, effect, contribution, reason, coverage, retrieval_quality, status, rejection_reason, created_at "
                "FROM feedback_events WHERE retrieval_id = ? ORDER BY created_at, rowid",
                (retrieval_id,),
            ).fetchall()
        ]
        try:
            legacy_rows = conn.execute(
                "SELECT * FROM context_feedback_events WHERE context_id = ? "
                "ORDER BY created_at, id",
                (retrieval_id,),
            ).fetchall()
            for legacy in legacy_rows:
                targets = (
                    ("memory", "used", _json_list(legacy["used_memory_ids_json"])),
                    (
                        "memory",
                        "irrelevant",
                        _json_list(legacy["irrelevant_memory_ids_json"]),
                    ),
                    ("memory", "stale", _json_list(legacy["stale_memory_ids_json"])),
                    ("memory", "wrong", _json_list(legacy["wrong_memory_ids_json"])),
                    (
                        "procedure",
                        "used",
                        _json_list(legacy["used_procedure_ids_json"]),
                    ),
                    (
                        "procedure",
                        "irrelevant",
                        _json_list(legacy["irrelevant_procedure_ids_json"]),
                    ),
                    (
                        "procedure",
                        "stale",
                        _json_list(legacy["stale_procedure_ids_json"]),
                    ),
                    (
                        "procedure",
                        "wrong",
                        _json_list(legacy["wrong_procedure_ids_json"]),
                    ),
                )
                expanded = 0
                for target_kind, assessment, target_ids in targets:
                    for target_id in target_ids:
                        expanded += 1
                        feedback.append(
                            {
                                "event_id": f"legacy_{legacy['id']}_{expanded}",
                                "target_kind": target_kind,
                                "target_id": str(target_id),
                                "assessment": assessment,
                                "effect": None,
                                "coverage": "legacy / not recorded",
                                "status": "legacy",
                                "reason": legacy["notes"],
                                "created_at": legacy["created_at"],
                                "source": "legacy feedback record",
                            }
                        )
        except sqlite3.Error:
            pass
        # Attach each exposed item's most recent target-specific assessment and
        # effect so the retrieval trace can show "why retrieved -> assessed ->
        # effect / task outcome" without conflating exposure with use. Feedback
        # is append-only, so the last matching record wins per target.
        kind_map = {
            "schema": "memory",
            "related": "memory",
            "episode": "memory",
            "raw_event": "memory",
            "procedural_memory": "procedure",
            "procedure": "procedure",
        }
        latest: dict[tuple[str, str], dict[str, Any]] = {}
        for fb in feedback:
            if str(fb.get("target_kind")) in {"memory", "procedure"}:
                latest[(str(fb["target_kind"]), str(fb.get("target_id")))] = fb
        for item in items:
            kind = kind_map.get(str(item.get("memory_type")), "memory")
            fb = latest.get((kind, str(item.get("memory_id"))))
            item["assessment"] = fb.get("assessment") if fb else None
            item["effect"] = fb.get("effect") if fb else None
            item["stale_reason"] = fb.get("stale_reason") if fb else None
            item["feedback_reason"] = fb.get("reason") if fb else None
        session = None
        if retrieval.get("session_id"):
            session_row = conn.execute(
                "SELECT id, agent, scope_id, started_ts, ended_ts, feedback_status, outcome, "
                "COALESCE(final_goal, initial_goal, goal, '') AS goal FROM sessions WHERE id = ?",
                (retrieval["session_id"],),
            ).fetchone()
            session = dict(session_row) if session_row else None
        return {"retrieval": retrieval, "items": items, "feedback": feedback, "session": session}
    except sqlite3.Error as exc:
        return {"error": str(exc), "retrieval_id": retrieval_id}
    finally:
        conn.close()


def _activity_summary(db_path: str, where: str, args: list[Any]) -> dict[str, int]:
    key = (db_path, where, tuple(args))
    now = time.monotonic()
    with _ACTIVITY_SUMMARY_CACHE_LOCK:
        cached = _ACTIVITY_SUMMARY_CACHE.get(key)
        if cached and now - cached[0] < _ACTIVITY_SUMMARY_CACHE_TTL:
            return cached[1]
    conn = _connect(db_path)
    try:
        summary_row = conn.execute(
            "SELECT COUNT(*) AS eligible, "
            "SUM(CASE WHEN s.ended_ts IS NOT NULL AND s.feedback_status = 'complete' THEN 1 ELSE 0 END) AS complete, "
            "SUM(CASE WHEN s.ended_ts IS NOT NULL AND COALESCE(s.feedback_status, '') = 'incomplete' THEN 1 ELSE 0 END) AS incomplete, "
            "SUM(CASE WHEN s.ended_ts IS NULL THEN 1 ELSE 0 END) AS pending, "
            "SUM(CASE WHEN s.ended_ts IS NOT NULL THEN 1 ELSE 0 END) AS closed, "
            "SUM(CASE WHEN s.ended_ts IS NOT NULL AND s.outcome = 'success' THEN 1 ELSE 0 END) AS successful_closed, "
            "SUM(CASE WHEN s.ended_ts IS NOT NULL AND s.outcome IN ('partial', 'failure') THEN 1 ELSE 0 END) AS partial_failed_closed, "
            "SUM(CASE WHEN s.ended_ts IS NOT NULL AND (s.outcome IS NULL OR s.outcome NOT IN ('success', 'partial', 'failure')) THEN 1 ELSE 0 END) AS unknown_outcome_closed, "
            "SUM(CASE WHEN s.ended_ts IS NOT NULL AND s.outcome IN ('success', 'partial', 'failure') THEN 1 ELSE 0 END) AS known_outcome_closed, "
            "SUM(CASE WHEN s.feedback_status IN ('complete', 'incomplete') OR s.ended_ts IS NULL THEN 1 ELSE 0 END) AS closure_eligible, "
            "SUM(CASE WHEN s.ended_ts IS NOT NULL AND (s.feedback_status IS NULL OR s.feedback_status NOT IN ('complete', 'incomplete')) THEN 1 ELSE 0 END) AS closure_unclassified, "
            "SUM(CASE WHEN s.feedback_status = 'complete' AND EXISTS(SELECT 1 FROM context_recall_events r WHERE r.session_id = s.id) THEN 1 ELSE 0 END) AS context_denominator, "
            "SUM(CASE WHEN s.feedback_status = 'complete' AND EXISTS(SELECT 1 FROM context_recall_events r WHERE r.session_id = s.id) "
            "AND EXISTS(SELECT 1 FROM feedback_events f WHERE f.session_id = s.id AND f.status = 'accepted' AND (f.assessment = 'used' OR f.effect = 'helped')) THEN 1 ELSE 0 END) AS context_use "
            f"FROM sessions s WHERE {where}",
            args,
        ).fetchone()
        summary = {
            key: int(summary_row[key] or 0) if summary_row else 0
            for key in (
                "eligible",
                "complete",
                "incomplete",
                "pending",
                "closed",
                "successful_closed",
                "partial_failed_closed",
                "unknown_outcome_closed",
                "known_outcome_closed",
                "closure_eligible",
                "closure_unclassified",
                "context_denominator",
                "context_use",
            )
        }
        with _ACTIVITY_SUMMARY_CACHE_LOCK:
            _ACTIVITY_SUMMARY_CACHE[key] = (now, summary)
            if len(_ACTIVITY_SUMMARY_CACHE) > 128:
                oldest = min(
                    _ACTIVITY_SUMMARY_CACHE, key=lambda item: _ACTIVITY_SUMMARY_CACHE[item][0]
                )
                _ACTIVITY_SUMMARY_CACHE.pop(oldest, None)
        return summary
    finally:
        conn.close()


def _activity_payload(db_path: str, qs: dict[str, list[str]]) -> dict[str, Any]:
    if not os.path.exists(db_path):
        return {"activities": [], "pagination": {"page": 1, "per_page": 50, "total": 0}}
    page = max(1, _qs_int(qs, "page", 1))
    per_page = max(1, min(100, _qs_int(qs, "per_page", 50)))
    sort = (qs.get("sort") or ["started"])[0].strip()
    direction = (qs.get("dir") or ["desc"])[0].strip().lower()
    activity_sorts = {
        "started": "s.started_ts",
        "task": "goal_preview",
        "scope": "s.scope_id",
        "outcome": "s.outcome",
        "closure": "s.feedback_status",
        "duration": "duration_s",
        "retrievals": "retrieval_count",
        "memories_touched": "memory_count",
        "procedure": "procedure_state",
        "retrieved_memories": "retrieved_memory_count",
        "retrieved_procedures": "retrieved_procedure_count",
        "feedback_coverage": "s.feedback_status",
        "episodes": "episode_count",
        "timeline_events": "timeline_event_count",
        "last_event": "last_event_ts",
        "verification": "verification_status",
    }
    if sort not in activity_sorts:
        sort = "started"
    if direction not in {"asc", "desc"}:
        direction = "desc"
    clauses = ["1=1"]
    args: list[Any] = []
    for key, column in (
        ("scope", "s.scope_id"),
        ("outcome", "s.outcome"),
        ("feedback", "s.feedback_status"),
        ("agent", "s.agent"),
        ("continuity", "s.continuity_id"),
    ):
        value = (qs.get(key) or [""])[0].strip()
        if value:
            clauses.append(f"{column} = ?")
            args.append(value)
    from_ts = _qs_int(qs, "from", 0)
    to_ts = _qs_int(qs, "to", 0)
    lane = (qs.get("lane") or [""])[0].strip()
    if from_ts and not lane:
        clauses.append("s.started_ts >= ?")
        args.append(from_ts)
    if to_ts and not lane:
        clauses.append("s.started_ts <= ?")
        args.append(to_ts)
    search = (qs.get("q") or [""])[0].strip().casefold()
    if search:
        clauses.append("lower(COALESCE(s.final_goal,s.initial_goal,s.goal,'')) LIKE ?")
        args.append(f"%{search}%")
    if lane == "raw_events":
        lane_clause = "EXISTS (SELECT 1 FROM raw_events re WHERE re.session_id=s.id"
        if from_ts:
            lane_clause += " AND re.ts >= ?"
            args.append(from_ts)
        if to_ts:
            lane_clause += " AND re.ts <= ?"
            args.append(to_ts)
        clauses.append(lane_clause + ")")
    elif lane == "episodes":
        lane_clause = (
            "EXISTS (SELECT 1 FROM episode_text et JOIN episodic_memories em "
            "ON em.id=et.episode_id WHERE et.session_id=s.id"
        )
        if from_ts:
            lane_clause += " AND em.ts >= ?"
            args.append(from_ts)
        if to_ts:
            lane_clause += " AND em.ts <= ?"
            args.append(to_ts)
        clauses.append(lane_clause + ")")
    elif lane == "schemas":
        lane_clause = (
            "EXISTS (SELECT 1 FROM schema_evidence se "
            "JOIN schemas sm ON sm.id=se.schema_id "
            "LEFT JOIN raw_events re ON re.id=se.raw_event_id "
            "LEFT JOIN episode_text et ON et.episode_id=se.episode_id "
            "WHERE (re.session_id=s.id OR et.session_id=s.id)"
        )
        if from_ts:
            lane_clause += " AND sm.first_formed_ts >= ?"
            args.append(from_ts)
        if to_ts:
            lane_clause += " AND sm.first_formed_ts <= ?"
            args.append(to_ts)
        clauses.append(lane_clause + ")")
    where = " AND ".join(clauses)
    summary_only = (qs.get("summary_only") or ["false"])[0] == "true"
    include_summary = (qs.get("include_summary") or ["true"])[0] != "false"
    conn = _connect(db_path)
    try:
        total = conn.execute(f"SELECT COUNT(*) AS n FROM sessions s WHERE {where}", args).fetchone()
        if summary_only:
            return {
                "activities": [],
                "summary": _activity_summary(db_path, where, args),
                "pagination": {
                    "page": page,
                    "per_page": per_page,
                    "total": int(total["n"] if total else 0),
                },
            }
        rows = conn.execute(
            "WITH retrieval_counts AS ("
            " SELECT session_id, COUNT(*) AS retrieval_count"
            " FROM context_recall_events GROUP BY session_id"
            "), retrieved_counts AS ("
            " SELECT r.session_id,"
            " COUNT(CASE WHEN i.admitted = 1 AND i.memory_type IN ('schema','related') THEN 1 END) AS retrieved_memory_count,"
            " COUNT(CASE WHEN i.admitted = 1 AND i.memory_type IN ('procedure','procedural_memory') THEN 1 END) AS retrieved_procedure_count"
            " FROM context_recall_events r LEFT JOIN context_recall_items i ON i.context_id = r.context_id"
            " GROUP BY r.session_id"
            "), memory_counts AS ("
            " SELECT session_id, COUNT(DISTINCT schema_id) AS memory_count FROM ("
            "  SELECT re.session_id, se.schema_id FROM schema_evidence se JOIN raw_events re ON re.id = se.raw_event_id"
            "  UNION ALL"
            "  SELECT et.session_id, se.schema_id FROM schema_evidence se JOIN episode_text et ON et.episode_id = se.episode_id"
            " ) WHERE session_id IS NOT NULL GROUP BY session_id"
            "), episode_counts AS ("
            " SELECT re.session_id, COUNT(*) AS episode_count"
            " FROM episodic_memories em JOIN raw_events re ON re.id = em.event_id GROUP BY re.session_id"
            "), event_counts AS ("
            " SELECT session_id, COUNT(*) AS timeline_event_count, MAX(ts) AS last_event_ts,"
            " MAX(CASE WHEN type = 'task_complete' AND metadata_json LIKE '%\"procedure\"%' THEN 1 ELSE 0 END) AS has_procedure"
            " FROM raw_events GROUP BY session_id"
            ")"
            " SELECT s.*, COALESCE(s.final_goal,s.initial_goal,s.goal,'') AS goal_preview,"
            " COALESCE(rc.retrieval_count, 0) AS retrieval_count,"
            " COALESCE(xc.retrieved_memory_count, 0) AS retrieved_memory_count,"
            " COALESCE(xc.retrieved_procedure_count, 0) AS retrieved_procedure_count,"
            " COALESCE(mc.memory_count, 0) AS memory_count,"
            " COALESCE(ec.episode_count, 0) AS episode_count,"
            " COALESCE(ev.timeline_event_count, 0) AS timeline_event_count,"
            " ev.last_event_ts,"
            " CASE WHEN s.ended_ts IS NULL THEN 'pending' WHEN COALESCE(ev.has_procedure, 0) = 1 THEN 'captured' ELSE 'none' END AS procedure_state,"
            " CASE WHEN s.ended_ts IS NULL THEN NULL ELSE MAX(0, s.ended_ts-s.started_ts) END AS duration_s,"
            " COALESCE(json_extract(s.verification_json, '$.status'), '') AS verification_status"
            " FROM sessions s"
            " LEFT JOIN retrieval_counts rc ON rc.session_id = s.id"
            " LEFT JOIN retrieved_counts xc ON xc.session_id = s.id"
            " LEFT JOIN memory_counts mc ON mc.session_id = s.id"
            " LEFT JOIN episode_counts ec ON ec.session_id = s.id"
            " LEFT JOIN event_counts ev ON ev.session_id = s.id"
            f" WHERE {where} ORDER BY {activity_sorts[sort]} {direction.upper()}, s.started_ts DESC, s.id DESC LIMIT ? OFFSET ?",
            [*args, per_page, (page - 1) * per_page],
        ).fetchall()
        activities = []
        for row in rows:
            item = dict(row)
            for key in ("verification_json", "retrieval_context_json", "task_context_json"):
                item.pop(key, None)
            activities.append(item)
        activity_summary = _activity_summary(db_path, where, args) if include_summary else {}
        return {
            "activities": activities,
            "sort": sort,
            "sort_direction": direction,
            "pagination": {
                "page": page,
                "per_page": per_page,
                "total": int(total["n"] if total else 0),
            },
            "summary": activity_summary,
        }
    except sqlite3.Error:
        return {"activities": [], "pagination": {"page": page, "per_page": per_page, "total": 0}}
    finally:
        conn.close()


def _activity_detail(db_path: str, session_id: str) -> dict[str, Any]:
    payload = _session_timeline(db_path, session_id)
    if payload.get("error"):
        return payload
    conn = _connect(db_path)
    try:
        retrievals = [
            dict(r)
            for r in conn.execute(
                "SELECT context_id, retrieval_type, query, goal, count_n, created_at FROM context_recall_events WHERE session_id = ? ORDER BY created_at",
                (session_id,),
            ).fetchall()
        ]
        feedback = [
            dict(r)
            for r in conn.execute(
                "SELECT event_id, retrieval_id, target_kind, target_id, assessment, effect, coverage, status, created_at FROM feedback_events WHERE session_id = ? ORDER BY created_at, rowid",
                (session_id,),
            ).fetchall()
        ]
        memories = [
            dict(r)
            for r in conn.execute(
                "SELECT DISTINCT s.id AS schema_id, s.content_text, s.status, s.first_formed_ts "
                "FROM schemas s JOIN schema_evidence se ON se.schema_id=s.id "
                "LEFT JOIN raw_events re ON re.id=se.raw_event_id "
                "LEFT JOIN episode_text et ON et.episode_id=se.episode_id "
                "WHERE re.session_id=? OR et.session_id=? ORDER BY s.first_formed_ts",
                (session_id, session_id),
            ).fetchall()
        ]
        related: list[dict[str, Any]] = []
        continuity = payload.get("session", {}).get("continuity_id")
        if continuity:
            related = [
                dict(r)
                for r in conn.execute(
                    "SELECT id, started_ts, ended_ts, outcome, feedback_status, COALESCE(final_goal,initial_goal,goal,'') AS goal FROM sessions WHERE continuity_id=? ORDER BY started_ts",
                    (continuity,),
                ).fetchall()
            ]
        procedure = _procedure_detail(db_path, f"proc_{session_id}", conn=conn, missing_ok=True)
        payload.update(
            {
                "retrievals": retrievals,
                "feedback": feedback,
                "memories": memories,
                "related_sessions": related,
                "procedure": None if procedure.get("error") else procedure.get("procedure"),
            }
        )
        return payload
    except sqlite3.Error:
        payload.update(
            {
                "retrievals": [],
                "feedback": [],
                "memories": [],
                "related_sessions": [],
                "procedure": None,
            }
        )
        return payload
    finally:
        conn.close()


def _procedure_detail(
    db_path: str,
    procedure_id: str,
    *,
    conn: sqlite3.Connection | None = None,
    missing_ok: bool = False,
) -> dict[str, Any]:
    if not os.path.exists(db_path):
        return {"error": "db not found"}
    owns_connection = conn is None
    connection = conn or _connect(db_path)
    try:
        from slowave.symbolic.procedural_memory import load_procedures

        procedure = next(
            (item for item in load_procedures(connection) if item["id"] == procedure_id), None
        )
        if procedure is None:
            return {"error": "procedure not found", "procedure_id": procedure_id}
        session_id = procedure_id.removeprefix("proc_")
        session_row = connection.execute(
            "SELECT id, agent, scope_id, started_ts, ended_ts, verification_json, feedback_status, lifecycle_version FROM sessions WHERE id=?",
            (session_id,),
        ).fetchone()
        source_session = dict(session_row) if session_row else None
        if source_session:
            source_session["verification"] = _json_dict(
                source_session.pop("verification_json", "{}")
            )
        retrievals: list[dict[str, Any]] = []
        try:
            retrievals = [
                dict(r)
                for r in connection.execute(
                    "SELECT DISTINCT r.context_id AS retrieval_id, r.session_id, r.scope_id, r.created_at, r.retrieval_type "
                    "FROM context_recall_events r LEFT JOIN context_recall_items i ON i.context_id=r.context_id "
                    "WHERE (i.memory_id=? AND i.admitted=1) OR r.response_json LIKE ? ORDER BY r.created_at DESC LIMIT 100",
                    (procedure_id, f'%"{procedure_id}"%'),
                ).fetchall()
            ]
        except sqlite3.Error:
            pass
        feedback = [
            dict(r)
            for r in connection.execute(
                "SELECT event_id, retrieval_id, assessment, effect, contribution, reason, status, created_at FROM feedback_events WHERE target_kind='procedure' AND target_id=? ORDER BY created_at DESC",
                (procedure_id,),
            ).fetchall()
        ]
        procedure["retrievals"] = retrievals
        procedure["feedback"] = feedback
        procedure["source_session"] = source_session
        return {"procedure": procedure}
    except sqlite3.Error as exc:
        return {"error": str(exc), "procedure_id": procedure_id}
    finally:
        if owns_connection:
            connection.close()


def _prototypes_payload(db_path: str, qs: dict[str, list[str]]) -> dict[str, Any]:
    """Return prototype list with member counts."""
    if not os.path.exists(db_path):
        return {"prototypes": [], "total": 0}
    limit = max(1, min(100, _qs_int(qs, "limit", 50)))
    conn = _connect(db_path)
    try:
        total_row = conn.execute("SELECT COUNT(*) AS n FROM semantic_prototypes").fetchone()
        rows = conn.execute(
            "SELECT p.id, p.support_count, p.variance, p.scale, p.last_updated_ts, "
            "COUNT(epm.episode_id) AS member_count "
            "FROM semantic_prototypes p "
            "LEFT JOIN episode_prototype_map epm ON epm.prototype_id = p.id "
            "GROUP BY p.id "
            "ORDER BY p.support_count DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return {
            "prototypes": [dict(r) for r in rows],
            "total": int(total_row["n"]) if total_row else 0,
        }
    finally:
        conn.close()


def _prototype_members(db_path: str, proto_id: int) -> dict[str, Any]:
    """Return episodes belonging to a prototype."""
    if not os.path.exists(db_path):
        return {"error": "db not found"}
    conn = _connect(db_path)
    try:
        proto_row = conn.execute(
            "SELECT * FROM semantic_prototypes WHERE id = ?", (proto_id,)
        ).fetchone()
        if not proto_row:
            return {"error": "prototype not found"}
        eps = conn.execute(
            "SELECT e.id, e.event_id, e.ts, e.salience, r.content, r.type "
            "FROM episodic_memories e "
            "JOIN episode_prototype_map epm ON epm.episode_id = e.id "
            "JOIN raw_events r ON r.id = e.event_id "
            "WHERE epm.prototype_id = ? "
            "ORDER BY e.ts DESC",
            (proto_id,),
        ).fetchall()
        return {
            "prototype": dict(proto_row),
            "episodes": [dict(r) for r in eps],
        }
    finally:
        conn.close()


def _event_detail(db_path: str, event_id: int) -> dict[str, Any]:
    """Return a single raw event with its content."""
    if not os.path.exists(db_path):
        return {"error": "db not found"}
    conn = _connect(db_path)
    try:
        row = conn.execute(
            "SELECT id, ts, type, content, session_id, metadata_json "
            "FROM raw_events WHERE id = ?",
            (event_id,),
        ).fetchone()
        if not row:
            return {"error": "event not found"}
        return {"event": dict(row)}
    finally:
        conn.close()


def _session_timeline(db_path: str, session_id: str) -> dict[str, Any]:
    """Return chronological timeline of a session with raw events and episodes."""
    if not os.path.exists(db_path):
        return {"error": "db not found"}
    conn = _connect(db_path)
    try:
        sess = conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
        if not sess:
            return {"error": "session not found"}
        events = conn.execute(
            "SELECT id, ts, type, content, metadata_json "
            "FROM raw_events WHERE session_id = ? ORDER BY ts ASC",
            (session_id,),
        ).fetchall()
        episodes = conn.execute(
            "SELECT e.id, e.event_id, e.ts, e.ts AS recorded_at, e.salience, e.recalled_count, "
            "r.content, r.metadata_json AS event_metadata_json "
            "FROM episodic_memories e "
            "JOIN raw_events r ON r.id = e.event_id "
            "WHERE r.session_id = ? "
            "ORDER BY e.ts ASC",
            (session_id,),
        ).fetchall()
        session = dict(sess)
        for key in ("task_context_json", "verification_json"):
            session[key.removesuffix("_json")] = _json_dict(session.pop(key, "{}"))
        event_items = []
        for row in events:
            item = dict(row)
            metadata = _json_dict(item.pop("metadata_json", "{}"))
            item["occurred_at"] = _occurred_at(metadata)
            provenance = _json_dict(metadata.get("provenance"))
            provenance.pop("request_id", None)
            provenance.pop("client_id", None)
            item["status"] = metadata.get("status")
            item["provenance"] = provenance
            event_items.append(item)
        episode_items = []
        for row in episodes:
            item = dict(row)
            event_metadata = _json_dict(item.pop("event_metadata_json", "{}"))
            item["occurred_at"] = _occurred_at(event_metadata)
            episode_items.append(item)
        return {
            "session": session,
            "events": event_items,
            "episodes": episode_items,
        }
    finally:
        conn.close()


def _pick_diverse_goals(goals: list[str], max_count: int = 3) -> list[str]:
    """Pick diverse example goals, avoiding near-duplicate phrases."""
    if not goals:
        return []
    deduped: list[str] = []
    seen_words: set[str] = set()
    for g in sorted(goals, key=len):
        if len(deduped) >= max_count:
            break
        words = set(g.lower().split())
        overlap = len(words & seen_words) / max(len(words), 1)
        if overlap < 0.6 or not seen_words:
            deduped.append(g)
            seen_words |= words
    return deduped


_PROC_ENCODER: Any = None  # module-level lazy singleton -- avoid reloading the
# ONNX embedding model on every dashboard request; the tab is fetched
# on-demand (tab click), not on the 2s auto-refresh loop, so a per-process
# singleton is enough (no cross-request cache/TTL needed).


def _get_proc_encoder() -> Any:
    global _PROC_ENCODER
    if _PROC_ENCODER is None:
        from slowave.symbolic.encoder import TextEncoder

        _PROC_ENCODER = TextEncoder()
    return _PROC_ENCODER


def _json_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    try:
        parsed = json.loads(value or "{}")
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _json_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    try:
        parsed = json.loads(value or "[]")
    except (TypeError, ValueError):
        return []
    return parsed if isinstance(parsed, list) else []


def _is_lifecycle_hook_query(value: Any) -> bool:
    """Identify injected lifecycle reminders so they do not masquerade as use cases."""

    query = str(value or "").casefold()
    return "<hook_prompt" in query or "slowave mandatory:" in query


def _lifecycle_health(conn) -> dict[str, Any]:
    """Return current lifecycle adoption/closure indicators without legacy mixing."""
    session_rows = conn.execute(
        "SELECT lifecycle_version, feedback_status, COUNT(*) AS n FROM sessions "
        "GROUP BY lifecycle_version, feedback_status"
    ).fetchall()
    versions: dict[str, int] = {}
    closure = {"complete": 0, "incomplete": 0, "pending": 0}
    for row in session_rows:
        version = str(row["lifecycle_version"] or "legacy")
        versions[version] = versions.get(version, 0) + int(row["n"])
        if version == LIFECYCLE_VERSION:
            status = str(row["feedback_status"] or "pending")
            closure[status if status in closure else "pending"] += int(row["n"])
    feedback = {
        str(row["status"]): int(row["n"])
        for row in conn.execute(
            "SELECT status, COUNT(*) AS n FROM feedback_events GROUP BY status"
        ).fetchall()
    }
    provenance = conn.execute(
        "SELECT COUNT(*) AS total, SUM(CASE WHEN metadata_json LIKE '%\"provenance\"%' "
        "THEN 1 ELSE 0 END) AS available FROM raw_events"
    ).fetchone()
    trajectories = conn.execute(
        "SELECT COUNT(*) AS events, COUNT(DISTINCT session_id) AS sessions FROM raw_events "
        "WHERE type LIKE 'trajectory:%'"
    ).fetchone()
    trajectory_episodes = conn.execute(
        "SELECT COUNT(*) AS n FROM episodic_memories e JOIN raw_events r ON r.id=e.event_id "
        "WHERE r.type LIKE 'trajectory:%'"
    ).fetchone()
    return {
        "session_versions": versions,
        "feedback_closure": closure,
        "feedback_events": feedback,
        "provenance": {
            "available": int(provenance["available"] or 0),
            "total": int(provenance["total"] or 0),
        },
        "trajectory": {
            "events": int(trajectories["events"] or 0),
            "sessions": int(trajectories["sessions"] or 0),
            "episodes_formed": int(trajectory_episodes["n"] or 0),
        },
    }


def _procedural_memory_payload(db_path: str, qs: dict[str, list[str]]) -> dict[str, Any]:
    """Return only canonical structured procedural-memory dogfood data.

    This view reports standalone procedures, retrievals, observed influence,
    downstream outcomes, and truth-maintenance feedback. It does not infer
    procedure families or hard applicability.
    """
    if not os.path.exists(db_path):
        return {"status": "db_not_found", "procedures": []}

    scope_filter = (qs.get("scope") or [""])[0].strip()
    cohort = (qs.get("cohort") or [LIFECYCLE_VERSION])[0].strip()
    # Preserve the prior v9 cohort for historical inspection while making the
    # default follow the installed lifecycle contract.
    if cohort not in {LIFECYCLE_VERSION, "v9", "all"}:
        cohort = LIFECYCLE_VERSION
    sort = (qs.get("sort") or ["recent"])[0].strip()
    if sort not in {
        "recent",
        "id",
        "summary",
        "scope",
        "outcome",
        "verification",
        "retrieved",
        "used",
        "use_rate",
        "helped",
        "no_effect",
        "harmed",
        "unknown",
        "feedback_coverage",
        "last_retrieved",
        "last_used",
        "source_activity",
    }:
        sort = "recent"
    direction = (qs.get("dir") or ["desc"])[0].strip()
    if direction not in {"asc", "desc"}:
        direction = "desc"
    per_page = max(5, min(100, _qs_int(qs, "per_page", _qs_int(qs, "limit", 25))))
    page = max(1, _qs_int(qs, "page", 1))
    conn = _connect(db_path)
    try:
        from slowave.symbolic.procedural_memory import load_procedures

        precedents = load_procedures(conn, scope=scope_filter or None)
        if cohort != "all":
            cohort_session_ids = {
                str(row["id"])
                for row in conn.execute(
                    "SELECT id FROM sessions WHERE lifecycle_version = ?", (cohort,)
                )
            }
            precedents = [
                item
                for item in precedents
                if str(item["id"]).removeprefix("proc_") in cohort_session_ids
            ]
        outcome_counts = {"success": 0, "partial": 0, "failure": 0, "unknown": 0}
        for precedent in precedents:
            key = precedent["outcome"] if precedent["outcome"] in outcome_counts else "unknown"
            outcome_counts[key] += 1

        where = "WHERE outcome IS NOT NULL"
        params: list[Any] = []
        if cohort != "all":
            where += " AND lifecycle_version = ?"
            params.append(cohort)
        if scope_filter:
            where += " AND scope_id = ?"
            params.append(scope_filter)
        completed_row = conn.execute(
            f"SELECT COUNT(*) AS n FROM sessions {where}", params
        ).fetchone()
        completed_sessions = int(completed_row["n"] if completed_row else 0)

        retrieval_where = ""
        retrieval_params: list[Any] = []
        retrieval_conditions: list[str] = []
        if cohort != "all":
            retrieval_conditions.append("s.lifecycle_version = ?")
            retrieval_params.append(cohort)
        if scope_filter:
            retrieval_conditions.append("r.scope_id = ?")
            retrieval_params.append(scope_filter)
        if retrieval_conditions:
            retrieval_where = "WHERE " + " AND ".join(retrieval_conditions)
        retrieval_rows = conn.execute(
            "SELECT r.context_id, r.retrieval_type, r.session_id, r.scope_id, r.query, r.goal, "
            "r.response_json, r.created_at FROM context_recall_events r "
            "JOIN sessions s ON s.id = r.session_id "
            f"{retrieval_where} ORDER BY created_at DESC",
            retrieval_params,
        ).fetchall()
        procedure_retrievals: list[dict[str, Any]] = []
        for row in retrieval_rows:
            procedure_ids = _json_list(_json_dict(row["response_json"]).get("procedure_ids"))
            if procedure_ids:
                procedure_retrievals.append(
                    {
                        "retrieval_id": row["context_id"],
                        "retrieval_type": row["retrieval_type"],
                        "session_id": row["session_id"],
                        "scope_id": row["scope_id"],
                        "query": row["query"] or row["goal"] or "",
                        "procedure_ids": procedure_ids,
                        "is_lifecycle_hook": _is_lifecycle_hook_query(row["query"] or ""),
                        "created_at": row["created_at"],
                    }
                )

        use_where = "WHERE e.type = 'task_complete'"
        use_params: list[Any] = []
        if scope_filter:
            use_where += " AND s.scope_id = ?"
            use_params.append(scope_filter)
        use_rows = conn.execute(
            "SELECT s.id AS session_id, e.metadata_json FROM sessions s "
            "JOIN raw_events e ON e.session_id = s.id "
            f"{use_where} ORDER BY e.id",
            use_params,
        ).fetchall()
        procedure_uses_by_session: dict[str, dict[str, dict[str, Any]]] = {}
        for row in use_rows:
            uses = _json_list(_json_dict(row["metadata_json"]).get("procedure_uses"))
            procedure_uses_by_session[str(row["session_id"])] = {
                str(use.get("procedure_id")): use
                for use in uses
                if isinstance(use, dict) and use.get("procedure_id")
            }

        v9_where = "WHERE f.target_kind = 'procedure'"
        v9_params: list[Any] = []
        if cohort != "all":
            v9_where += " AND s.lifecycle_version = ?"
            v9_params.append(cohort)
        if scope_filter:
            v9_where += " AND f.scope_id = ?"
            v9_params.append(scope_filter)
        v9_rows = conn.execute(
            "SELECT f.event_id, f.retrieval_id, f.target_id, f.assessment, f.effect, f.coverage, "
            "f.contribution, f.reason, f.refines_event_id, f.mutation_mode, f.status, "
            "f.rejection_reason, f.created_at "
            "FROM feedback_events f JOIN sessions s ON s.id = f.session_id "
            f"{v9_where} ORDER BY f.created_at, f.rowid",
            v9_params,
        ).fetchall()
        v9_feedback_by_retrieval: dict[str, list[dict[str, Any]]] = {}
        v9_latest_assessment: dict[tuple[str, str], dict[str, Any]] = {}
        v9_feedback_counts = {
            "used": 0,
            "not_used": 0,
            "helped": 0,
            "no_effect": 0,
            "harmed": 0,
            "unknown": 0,
            "accepted": 0,
            "rejected": 0,
        }
        for row in v9_rows:
            item = dict(row)
            item["source"] = "v9"
            retrieval_id = str(row["retrieval_id"])
            v9_feedback_by_retrieval.setdefault(retrieval_id, []).append(item)
            status = str(row["status"])
            v9_feedback_counts[status if status in {"accepted", "rejected"} else "rejected"] += 1
            if status == "accepted":
                assessment = str(row["assessment"] or "")
                effect = str(row["effect"] or "unknown")
                if assessment in {"used", "not_used"}:
                    v9_feedback_counts[assessment] += 1
                if effect in {"helped", "no_effect", "harmed", "unknown"}:
                    v9_feedback_counts[effect] += 1
                v9_latest_assessment[(retrieval_id, str(row["target_id"]))] = {
                    "procedure_id": str(row["target_id"]),
                    "use": assessment,
                    "effect": effect,
                    "contribution": row["contribution"],
                    "reason": row["reason"],
                    "refines_event_id": row["refines_event_id"],
                    "mutation_mode": row["mutation_mode"],
                    "created_at": row["created_at"],
                    "source": "v9",
                }

        feedback_where = ""
        feedback_params: list[Any] = []
        feedback_conditions: list[str] = []
        if cohort != "all":
            feedback_conditions.append("s.lifecycle_version = ?")
            feedback_params.append(cohort)
        if scope_filter:
            feedback_conditions.append("f.scope_id = ?")
            feedback_params.append(scope_filter)
        if feedback_conditions:
            feedback_where = "WHERE " + " AND ".join(feedback_conditions)
        feedback_rows = conn.execute(
            "SELECT f.context_id, f.feedback, f.outcome, f.used_procedure_ids_json, "
            "f.irrelevant_procedure_ids_json, f.stale_procedure_ids_json, "
            "f.wrong_procedure_ids_json, f.created_at FROM context_feedback_events f "
            "JOIN sessions s ON s.id = f.session_id "
            f"{feedback_where} ORDER BY created_at DESC",
            feedback_params,
        ).fetchall()
        legacy_feedback_counts = {"used": 0, "irrelevant": 0, "stale": 0, "wrong": 0}
        feedback_by_retrieval: dict[str, list[dict[str, Any]]] = {}
        for row in feedback_rows:
            item = {
                "feedback": row["feedback"],
                "outcome": row["outcome"],
                "used": _json_list(row["used_procedure_ids_json"]),
                "irrelevant": _json_list(row["irrelevant_procedure_ids_json"]),
                "stale": _json_list(row["stale_procedure_ids_json"]),
                "wrong": _json_list(row["wrong_procedure_ids_json"]),
                "created_at": row["created_at"],
            }
            if any(item[key] for key in legacy_feedback_counts):
                item["source"] = "legacy"
                feedback_by_retrieval.setdefault(str(row["context_id"]), []).append(item)
                for key in legacy_feedback_counts:
                    legacy_feedback_counts[key] += len(item[key])

        reverse = direction == "desc"

        def sort_key(item: dict[str, Any]) -> Any:
            if sort == "retrieved":
                return (
                    item["evidence"]["retrieved"],
                    item["evidence"]["used"],
                    item["created_at"],
                )
            if sort == "used":
                return (
                    item["evidence"]["used"],
                    item["evidence"]["retrieved"],
                    item["created_at"],
                )
            if sort in {"helped", "no_effect", "harmed", "unknown"}:
                return (
                    item["evidence"].get(sort, 0),
                    item["evidence"]["retrieved"],
                    item["created_at"],
                )
            if sort == "use_rate":
                retrieved = item["evidence"]["retrieved"]
                return (
                    item["evidence"]["used"] / retrieved if retrieved else -1,
                    retrieved,
                    item["created_at"],
                )
            if sort == "feedback_coverage":
                return (item.get("feedback_complete", 0), item["created_at"])
            if sort == "last_retrieved":
                return (item.get("last_retrieved_ts") or 0, item["created_at"])
            if sort == "last_used":
                return (item.get("last_used_ts") or 0, item["created_at"])
            if sort == "source_activity":
                return str(item.get("source_activity_id") or "").casefold()
            if sort == "id":
                return str(item["id"]).casefold()
            if sort == "summary":
                return str(item.get("summary") or item.get("goal") or "").casefold()
            if sort == "scope":
                return str(item.get("scope_id") or "").casefold()
            if sort == "outcome":
                return str(item.get("outcome") or "").casefold()
            if sort == "verification":
                return str(item.get("verification", {}).get("status") or "").casefold()
            return item["created_at"]

        # Count every exposure here, including unassessed ones.  Canonical
        # v9 feedback already contributes use/effect evidence in
        # load_procedures(), which also feeds retrieval ranking.
        v9_exposures = {
            (str(item["retrieval_id"]), str(procedure_id))
            for item in procedure_retrievals
            for procedure_id in item["procedure_ids"]
        }
        for precedent in precedents:
            evidence = precedent["evidence"]
            procedure_id = str(precedent["id"])
            evidence["retrieved"] = sum(
                exposed_id == procedure_id for _retrieval_id, exposed_id in v9_exposures
            )

        for precedent in precedents:
            session_id = str(precedent["id"]).removeprefix("proc_")
            session_row = conn.execute(
                "SELECT verification_json, lifecycle_version FROM sessions WHERE id = ?",
                (session_id,),
            ).fetchone()
            precedent["verification"] = (
                _json_dict(session_row["verification_json"]) if session_row else {}
            )
            precedent["lifecycle_version"] = (
                session_row["lifecycle_version"] if session_row else None
            )
            precedent["source_activity_id"] = session_id
            related_retrievals = [
                item for item in procedure_retrievals if precedent["id"] in item["procedure_ids"]
            ]
            precedent["last_retrieved_ts"] = (
                max((int(item.get("created_at") or 0) for item in related_retrievals), default=0)
                or None
            )
            precedent["last_used_ts"] = (
                max(
                    (
                        int(row["created_at"] or 0)
                        for row in v9_rows
                        if str(row["target_id"]) == str(precedent["id"])
                        and str(row["status"]) == "accepted"
                        and str(row["assessment"]) == "used"
                    ),
                    default=0,
                )
                or None
            )
            precedent["feedback_complete"] = sum(
                1
                for row in v9_rows
                if str(row["target_id"]) == str(precedent["id"])
                and str(row["status"]) == "accepted"
                and str(row["coverage"] or "") == "complete"
            )

        outcome_filter = (qs.get("outcome") or [""])[0].strip()
        verification_filter = (qs.get("verification") or [""])[0].strip()
        retrieved_filter = (qs.get("retrieved") or [""])[0].strip()
        from_ts = _qs_int(qs, "from", 0)
        if outcome_filter:
            precedents = [item for item in precedents if item.get("outcome") == outcome_filter]
        if verification_filter:
            precedents = [
                item
                for item in precedents
                if item.get("verification", {}).get("status") == verification_filter
            ]
        if retrieved_filter == "yes":
            precedents = [item for item in precedents if item["evidence"]["retrieved"] > 0]
        elif retrieved_filter == "never":
            precedents = [item for item in precedents if item["evidence"]["retrieved"] == 0]
        if from_ts:
            precedents = [
                item for item in precedents if int(item.get("created_at") or 0) >= from_ts
            ]

        ordered_precedents = sorted(precedents, key=sort_key, reverse=reverse)
        filtered_total = len(ordered_precedents)
        recent_precedents = ordered_precedents[(page - 1) * per_page : page * per_page]
        for precedent in recent_precedents:
            precedent_retrievals = []
            for item in procedure_retrievals:
                if precedent["id"] not in item["procedure_ids"]:
                    continue
                item = dict(item)
                item["procedure_assessment"] = v9_latest_assessment.get(
                    (str(item["retrieval_id"]), precedent["id"])
                ) or procedure_uses_by_session.get(str(item["session_id"]), {}).get(precedent["id"])
                precedent_retrievals.append(item)
            precedent["retrievals"] = precedent_retrievals[:20]
        recent_retrievals = []
        for item in procedure_retrievals[:per_page]:
            item = dict(item)
            item["feedback"] = [
                *v9_feedback_by_retrieval.get(item["retrieval_id"], []),
                *feedback_by_retrieval.get(item["retrieval_id"], []),
            ]
            recent_retrievals.append(item)

        retrieved_procedures = sum(
            1 for item in precedents if int(item["evidence"].get("retrieved", 0)) > 0
        )
        assessed_procedures = sum(
            1
            for item in precedents
            if int(item["evidence"].get("retrieved", 0)) > 0
            and int(item["evidence"].get("used", 0)) + int(item["evidence"].get("not_used", 0)) > 0
        )
        effect_assessed = sum(
            int(item["evidence"].get(key, 0))
            for item in precedents
            for key in ("helped", "no_effect", "harmed")
        )
        helpful_assessments = sum(int(item["evidence"].get("helped", 0)) for item in precedents)
        harmful_assessments = sum(int(item["evidence"].get("harmed", 0)) for item in precedents)

        return {
            "status": "dogfooding" if precedents else "awaiting_structured_attempts",
            "scope": scope_filter or None,
            "cohort": cohort,
            "sort": sort,
            "sort_direction": direction,
            "completed_sessions": completed_sessions,
            "structured_attempts": len(precedents),
            "capture_rate": (len(precedents) / completed_sessions if completed_sessions else 0.0),
            "outcomes": outcome_counts,
            "procedures": recent_precedents,
            "pagination": {"page": page, "per_page": per_page, "total": filtered_total},
            "influence_counts": {
                key: sum(item["evidence"][key] for item in precedents)
                for key in (
                    "retrieved",
                    "used",
                    "not_used",
                    "helped",
                    "no_effect",
                    "harmed",
                    "unknown",
                )
            },
            "procedure_retrievals": len(procedure_retrievals),
            "hook_retrievals": sum(item["is_lifecycle_hook"] for item in procedure_retrievals),
            "hook_procedure_exposures": sum(
                len(item["procedure_ids"])
                for item in procedure_retrievals
                if item["is_lifecycle_hook"]
            ),
            "feedback_counts": {
                **legacy_feedback_counts,
                "v9": v9_feedback_counts,
                "legacy": legacy_feedback_counts,
            },
            "recent_retrievals": recent_retrievals,
            "summary": {
                "current_procedures": len(precedents),
                "retrieved_procedures": retrieved_procedures,
                "used_procedures": sum(
                    int(item["evidence"].get("used", 0)) > 0 for item in precedents
                ),
                "assessed_retrieved_procedures": assessed_procedures,
                "helpful_assessments": helpful_assessments,
                "effect_assessed": effect_assessed,
                "harmful_assessments": harmful_assessments,
            },
        }
    finally:
        conn.close()


def _labs_rollout_payload(db_path: str) -> dict[str, Any]:
    """Return exploratory post-v9 measurements for the opt-in Labs surface."""
    if not os.path.exists(db_path):
        return {"status": "db_not_found"}
    conn = _connect(db_path)
    try:
        lifecycle_version = LIFECYCLE_VERSION
        session_rows = conn.execute(
            "SELECT id, started_ts, ended_ts, outcome, feedback_status FROM sessions "
            "WHERE lifecycle_version = ? ORDER BY started_ts",
            (lifecycle_version,),
        ).fetchall()
        session_ids = {str(row["id"]) for row in session_rows}
        completed = [row for row in session_rows if row["ended_ts"] is not None]
        incomplete = [row for row in completed if row["feedback_status"] == "incomplete"]
        active_pending = [row for row in session_rows if row["ended_ts"] is None]
        outcomes = {"success": 0, "partial": 0, "failure": 0, "unknown": 0}
        for row in completed:
            outcome = str(row["outcome"] or "unknown")
            outcomes[outcome if outcome in outcomes else "unknown"] += 1

        provenance_epoch_row = conn.execute(
            "SELECT MIN(e.ts) AS started_at FROM raw_events e "
            "JOIN sessions s ON s.id=e.session_id WHERE s.lifecycle_version = ? "
            "AND json_extract(e.metadata_json, '$.provenance') IS NOT NULL",
            (lifecycle_version,),
        ).fetchone()
        provenance_started_at = provenance_epoch_row["started_at"] if provenance_epoch_row else None
        raw_rows = conn.execute(
            "SELECT e.metadata_json FROM raw_events e JOIN sessions s ON s.id=e.session_id "
            "WHERE s.lifecycle_version = ? AND (? IS NULL OR e.ts >= ?)",
            (lifecycle_version, provenance_started_at, provenance_started_at),
        ).fetchall()
        provenance_available = sum(
            isinstance(_json_dict(row["metadata_json"]).get("provenance"), dict) for row in raw_rows
        )

        retrieval_rows = conn.execute(
            "SELECT r.context_id, r.response_json, r.query, r.goal, r.count_n, "
            "r.response_chars, r.estimated_tokens FROM context_recall_events r "
            "JOIN sessions s ON s.id=r.session_id WHERE s.lifecycle_version = ?",
            (lifecycle_version,),
        ).fetchall()
        memory_exposures = 0
        no_match_retrievals = 0
        response_chars: list[int] = []
        estimated_tokens: list[int] = []
        procedure_exposures = 0
        hook_procedure_exposures = 0
        retrievals_with_procedures = 0
        hook_retrievals_with_procedures = 0
        hook_retrieval_ids: set[str] = set()
        for row in retrieval_rows:
            response = _json_dict(row["response_json"])
            memory_exposures += len(_json_list(response.get("memory_ids")))
            no_match_retrievals += int(row["count_n"] or 0) == 0
            if row["response_chars"] is not None:
                response_chars.append(int(row["response_chars"]))
            if row["estimated_tokens"] is not None:
                estimated_tokens.append(int(row["estimated_tokens"]))
            procedure_ids = _json_list(response.get("procedure_ids"))
            procedure_exposures += len(procedure_ids)
            retrievals_with_procedures += bool(procedure_ids)
            if procedure_ids and _is_lifecycle_hook_query(row["query"] or row["goal"] or ""):
                hook_retrievals_with_procedures += 1
                hook_procedure_exposures += len(procedure_ids)
                hook_retrieval_ids.add(str(row["context_id"]))

        feedback_rows = conn.execute(
            "SELECT f.retrieval_id, f.target_kind, f.assessment, f.effect, f.status FROM feedback_events f "
            "JOIN sessions s ON s.id=f.session_id WHERE s.lifecycle_version = ?",
            (lifecycle_version,),
        ).fetchall()
        procedure_feedback = {
            "accepted": 0,
            "rejected": 0,
            "used": 0,
            "not_used": 0,
            "helped": 0,
            "no_effect": 0,
            "harmed": 0,
            "unknown": 0,
            "non_hook_used": 0,
            "non_hook_not_used": 0,
            "non_hook_helped": 0,
            "non_hook_no_effect": 0,
            "non_hook_harmed": 0,
            "non_hook_unknown": 0,
        }
        memory_feedback = {"irrelevant": 0, "stale": 0, "contradicted": 0, "accepted": 0}
        for row in feedback_rows:
            if row["target_kind"] == "memory" and row["status"] == "accepted":
                memory_feedback["accepted"] += 1
                assessment = str(row["assessment"] or "")
                if assessment in memory_feedback:
                    memory_feedback[assessment] += 1
            if row["target_kind"] != "procedure":
                continue
            status = str(row["status"] or "rejected")
            procedure_feedback[status if status in {"accepted", "rejected"} else "rejected"] += 1
            if status != "accepted":
                continue
            assessment = str(row["assessment"] or "")
            effect = str(row["effect"] or "unknown")
            if assessment in {"used", "not_used"}:
                procedure_feedback[assessment] += 1
                if str(row["retrieval_id"]) not in hook_retrieval_ids:
                    procedure_feedback[f"non_hook_{assessment}"] += 1
            if effect in {"helped", "no_effect", "harmed", "unknown"}:
                procedure_feedback[effect] += 1
                if str(row["retrieval_id"]) not in hook_retrieval_ids:
                    procedure_feedback[f"non_hook_{effect}"] += 1

        # Truth-maintenance is measured from client feedback, not from the
        # removed geometric/consolidation counter. Keep current status counts
        # plus a small, inspectable sample of v9 retirement decisions.
        status_rows = conn.execute(
            "SELECT status, COUNT(*) AS n FROM schemas GROUP BY status"
        ).fetchall()
        truth_status_counts = {str(row["status"]): int(row["n"]) for row in status_rows}
        truth_rows = conn.execute(
            "SELECT f.target_id, f.assessment, f.stale_reason, f.replacement_target_id, f.reason, "
            "f.created_at, s.content_text AS retired_content, "
            "r.content_text AS replacement_content "
            "FROM feedback_events f "
            "JOIN sessions se ON se.id=f.session_id AND se.lifecycle_version=? "
            "LEFT JOIN schemas s ON s.id=CAST(REPLACE(f.target_id, 'sch_', '') AS INTEGER) "
            "LEFT JOIN schemas r ON r.id=CAST(REPLACE(f.replacement_target_id, 'sch_', '') AS INTEGER) "
            "WHERE f.target_kind='memory' AND f.status='accepted' "
            "AND f.assessment IN ('stale','wrong') "
            "ORDER BY f.created_at DESC, f.rowid DESC LIMIT 20",
            (lifecycle_version,),
        ).fetchall()
        truth_feedback_counts = {
            "stale": 0,
            "contradicted": 0,
            "superseded": 0,
            "outdated": 0,
            "unsupported": 0,
            "withdrawn": 0,
            "with_replacement": 0,
        }
        truth_sample = []
        for row in truth_rows:
            assessment = str(row["assessment"])
            truth_feedback_counts["stale"] += 1
            reason = str(row["stale_reason"] or "outdated")
            if reason in truth_feedback_counts:
                truth_feedback_counts[reason] += 1
            if row["replacement_target_id"]:
                truth_feedback_counts["with_replacement"] += 1
            truth_sample.append(
                {
                    "memory_id": str(row["target_id"]),
                    "assessment": assessment,
                    "stale_reason": row["stale_reason"],
                    "replacement_memory_id": row["replacement_target_id"],
                    "retired_content": row["retired_content"],
                    "replacement_content": row["replacement_content"],
                    "reason": row["reason"],
                    "created_at": row["created_at"],
                }
            )

        from slowave.symbolic.procedural_memory import load_procedures

        precedents = [
            item
            for item in load_procedures(conn)
            if str(item["id"]).removeprefix("proc_") in session_ids
        ]
        return {
            "status": "experimental",
            "cohort": {
                "lifecycle_version": lifecycle_version,
                "started_at": session_rows[0]["started_ts"] if session_rows else None,
                "sessions": len(session_rows),
                "completed_sessions": len(completed),
                "feedback_complete": sum(row["feedback_status"] == "complete" for row in completed),
                "feedback_incomplete": len(incomplete),
                "active_pending": len(active_pending),
                "outcomes": outcomes,
            },
            "provenance": {
                "epoch_started_at": provenance_started_at,
                "available": provenance_available,
                "eligible_events": len(raw_rows),
            },
            "retrieval": {
                "retrievals": len(retrieval_rows),
                "memory_exposures": memory_exposures,
                "no_match_retrievals": no_match_retrievals,
                "response_chars": {
                    "observed": len(response_chars),
                    "total": sum(response_chars),
                    "average": (
                        round(sum(response_chars) / len(response_chars)) if response_chars else None
                    ),
                },
                "estimated_tokens": {
                    "observed": len(estimated_tokens),
                    "total": sum(estimated_tokens),
                    "average": (
                        round(sum(estimated_tokens) / len(estimated_tokens))
                        if estimated_tokens
                        else None
                    ),
                },
                "latency": "not_persisted",
                "memory_feedback": memory_feedback,
                "retrievals_with_procedures": retrievals_with_procedures,
                "procedure_exposures": procedure_exposures,
                "hook_retrievals_with_procedures": hook_retrievals_with_procedures,
                "hook_procedure_exposures": hook_procedure_exposures,
                "non_hook_retrievals_with_procedures": (
                    retrievals_with_procedures - hook_retrievals_with_procedures
                ),
                "non_hook_procedure_exposures": (procedure_exposures - hook_procedure_exposures),
                "procedure_feedback": procedure_feedback,
            },
            "truth_maintenance": {
                "schemas_by_status": truth_status_counts,
                "v9_feedback": truth_feedback_counts,
                "sample": truth_sample,
            },
            "procedures": {
                "captured": len(precedents),
                "capture_rate": len(precedents) / len(completed) if completed else 0.0,
            },
        }
    finally:
        conn.close()


def _procedures_payload(db_path: str, qs: dict[str, list[str]]) -> dict[str, Any]:
    """Return procedural memory clusters via the validated embedding +
    alignment + average-linkage method -- the same method
    scripts/analyze_procedural_signal.py uses to check the Phase 2 gate.
    See slowave/symbolic/procedural.py and
    private/docs/iterations/20260727_procedural_memory_phase2_plan.md.

    Previously this endpoint clustered by raw event-TYPE signature, which
    Phase 1 proved carries zero procedural signal -- replaced entirely
    rather than kept alongside, so there's exactly one definition of
    "procedure cluster" in the codebase.
    """
    from slowave.symbolic.procedural import (
        build_embedding_caches,
        cluster_sessions,
        rank_clusters,
    )

    if not os.path.exists(db_path):
        return {
            "clusters": [],
            "gate": "db_not_found",
            "step_sessions": 0,
            "qualified_sessions": 0,
        }

    min_sessions = max(2, min(20, _qs_int(qs, "min_sessions", 2)))
    max_clusters = max(5, min(50, _qs_int(qs, "limit", 20)))
    threshold = 0.4
    scope_filter = (qs.get("scope") or [""])[0].strip()

    conn = _connect(db_path)
    try:
        from slowave.symbolic.procedural_memory import form_families, load_attempts

        structured_attempts, legacy_trace_sessions = load_attempts(conn, scope=scope_filter or None)
        if structured_attempts:
            families = [
                family
                for family in form_families(structured_attempts, min_support=min_sessions)
                if len(family.member_ids) >= min_sessions
            ][:max_clusters]
            member_ids = {member for family in families for member in family.member_ids}
            clusters_out = []
            for family in families:
                data = family.as_dict()
                total = len(family.member_ids)
                clusters_out.append(
                    {
                        "cluster_id": family.family_id,
                        "family_id": family.family_id,
                        "session_count": total,
                        "successes": family.successes,
                        "partials": family.partials,
                        "failures": family.failures,
                        "success_rate": family.successes / total if total else 0.0,
                        "goal_coherence": 0.0,
                        "anti_pattern": family.status == "warning",
                        "competes_with": [],
                        "example_goals": list(family.source_goals)[:3],
                        "example_steps": [step.summary for step in family.steps],
                        "structured_steps": data["steps"],
                        "preconditions": data["preconditions"],
                        "context_facets": data["context_facets"],
                        "warnings": data["warnings"],
                        "status": family.status,
                        "min_pairwise_alignment": data["min_pairwise_alignment"],
                        "scope_id": family.scope_id or "",
                        "session_ids": list(family.member_ids)[:20],
                        "total_session_ids": total,
                    }
                )
            supported = sum(family.status == "supported" for family in families)
            return {
                "mode": "structured",
                "clusters": clusters_out,
                "total_clusters_found": len(clusters_out),
                "total_sessions_in_clusters": len(member_ids),
                "structured_attempts": len(structured_attempts),
                "legacy_trace_sessions": legacy_trace_sessions,
                "unassigned_attempts": len(structured_attempts) - len(member_ids),
                "eligible_families": supported,
                "warning_families": sum(family.status == "warning" for family in families),
                "min_sessions": min_sessions,
                "gate": (
                    "structured_families_found" if families else "insufficient_structured_data"
                ),
                "method": "controlled_steps+complete_link",
            }

        qualified_row = conn.execute(
            "SELECT COUNT(*) AS n FROM sessions WHERE goal IS NOT NULL AND outcome IS NOT NULL"
        ).fetchone()
        qualified_sessions = int(qualified_row["n"]) if qualified_row else 0

        scope_clause = ""
        scope_params: list[Any] = []
        if scope_filter:
            scope_clause = "AND scope_id = ?"
            scope_params.append(scope_filter)

        session_rows = conn.execute(
            f"SELECT id, goal, outcome, scope_id FROM sessions "
            f"WHERE goal IS NOT NULL AND outcome IS NOT NULL {scope_clause} "
            f"ORDER BY started_ts",
            scope_params,
        ).fetchall()

        sessions: list[dict[str, Any]] = []
        for row in session_rows:
            step_rows = conn.execute(
                "SELECT content FROM raw_events WHERE session_id = ? AND type = 'step' ORDER BY ts",
                (row["id"],),
            ).fetchall()
            step_contents = [r["content"] for r in step_rows if r["content"]]
            sessions.append(
                {
                    "id": row["id"],
                    "goal": row["goal"] or "",
                    "outcome": row["outcome"] or "unknown",
                    "scope_id": row["scope_id"],
                    "step_contents": step_contents,
                    "has_steps": len(step_contents) > 0,
                }
            )

        step_sessions = [s for s in sessions if s["has_steps"]]

        if len(step_sessions) < min_sessions:
            return {
                "clusters": [],
                "total_clusters_found": 0,
                "total_sessions_in_clusters": 0,
                "step_sessions": len(step_sessions),
                "qualified_sessions": qualified_sessions,
                "min_sessions": min_sessions,
                "min_for_detection": min_sessions,
                "gate": "insufficient_data",
            }

        encoder = _get_proc_encoder()
        step_cache, goal_cache = build_embedding_caches(sessions, encoder)
        clusters = cluster_sessions(sessions, step_cache, goal_cache, threshold=threshold)
        ranked = rank_clusters(clusters, min_sessions, max_clusters, goal_cache=goal_cache)

        n_good = sum(1 for c in ranked if c["goal_coherence"] >= 0.3)
        gate_pass = n_good >= 5
        total_sessions_in_clusters = sum(c["session_count"] for c in ranked)

        clusters_out: list[dict[str, Any]] = []
        for c in ranked:
            full_members = clusters.get(c["cluster_id"], [])
            session_ids_full = [m["id"] for m in full_members]
            scope_id = scope_filter or (full_members[0]["scope_id"] if full_members else "") or ""
            clusters_out.append(
                {
                    "cluster_id": c["cluster_id"],
                    "session_count": c["session_count"],
                    "successes": c["successes"],
                    "failures": c["failures"],
                    "success_rate": c["success_rate"],
                    "goal_coherence": c["goal_coherence"],
                    "anti_pattern": c["anti_pattern"],
                    "competes_with": c.get("competes_with", []),
                    "example_goals": _pick_diverse_goals(c["example_goals"], max_count=3),
                    "example_steps": c["example_steps"],
                    "scope_id": str(scope_id),
                    "session_ids": session_ids_full[:20],
                    "total_session_ids": len(session_ids_full),
                }
            )

        return {
            "mode": "legacy",
            "clusters": clusters_out,
            "total_clusters_found": len(clusters_out),
            "total_sessions_in_clusters": total_sessions_in_clusters,
            "step_sessions": len(step_sessions),
            "qualified_sessions": qualified_sessions,
            "min_sessions": min_sessions,
            "n_good_clusters": n_good,
            "gate_target": 5,
            "gate_pass": gate_pass,
            "gate": (
                "target_met"
                if gate_pass
                else ("clusters_found" if clusters_out else "no_clusters_found")
            ),
            "threshold": threshold,
            "method": "embedding+alignment+average_linkage",
            "legacy_trace_sessions": len(step_sessions),
            "structured_attempts": 0,
        }
    finally:
        conn.close()


from slowave.dashboard._html import render_index_html  # noqa: E402


def _graph_health_payload(db_path: str) -> dict[str, Any]:
    """Return full graph health metrics for the /api/debug/graph endpoint."""
    from slowave.core.graph_health import compute

    return compute(db_path)
