"""Local Slowave dashboard.

Dependency-free: stdlib HTTP server + SQLite read APIs + embedded UI.
The dashboard is local-only by default. The one mutating action (schema
forget/unforget) is enabled by default too -- pass --no-allow-actions for a
strictly read-only dashboard, e.g. before sharing a screen or a port.
"""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import time
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

VALID_SCHEMA_STATUSES = (
    "active",
    "needs_review",
    "superseded",
    "contradicted",
    "archived",
    "forgotten",
)
# "contradicts" and "related_to" were removed from schema_store.py's VALID_RELATIONS
# (2026-07-14): contradicts required an unreachable time_delta_s<=0 tie (every call
# site now writes "supersedes" for that case too), related_to was only add_relation()'s
# own silent fallback for invalid input, never a real caller. "relates_to" (2026-07-15,
# distinct spelling from the dead "related_to") was reintroduced as the taxonomy's
# honest catch-all for "same topic, no stronger geometric signal applies". "part_of"
# (subspace containment) was removed 2026-07-23 after a hand-labeled audit of
# production edges found ~11% precision -- see
# private/docs/iterations/20260723_part_of_audit_and_brain_alignment_review.md.
# relates_to is now the only content-relation type -- see schema_store.py's
# VALID_RELATIONS comment. Kept in sync here since the dashboard is a standalone
# stdlib server with its own copy of this list.
VALID_SCHEMA_RELATIONS = ("relates_to",)


def run_dashboard(
    *,
    db_path: str,
    host: str = "127.0.0.1",
    port: int = 8765,
    refresh_ms: int = 2000,
    allow_actions: bool = True,
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


def _make_handler(*, db_path: str, refresh_ms: int, allow_actions: bool):
    class DashboardHandler(BaseHTTPRequestHandler):
        server_version = "slowave-dashboard/0.2"

        def log_message(self, fmt: str, *args: Any) -> None:
            return

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            path = parsed.path.rstrip("/") or "/"
            qs = parse_qs(parsed.query)
            try:
                if path == "/":
                    self._send_html(_INDEX_HTML)
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
                else:
                    self._send_json(
                        {"error": "not found", "path": path}, status=HTTPStatus.NOT_FOUND
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
                    {"error": "mutating actions disabled; restart with --allow-actions"},
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
                        {"error": "not found", "path": path}, status=HTTPStatus.NOT_FOUND
                    )
            except ValueError as e:
                self._send_json({"error": f"invalid request: {e}"}, status=HTTPStatus.BAD_REQUEST)
            except Exception as e:
                self._send_json({"error": str(e)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)

        def _send_html(self, html: str) -> None:
            html = html.replace("__REFRESH_MS__", str(refresh_ms)).replace(
                "__ALLOW_ACTIONS__", "true" if allow_actions else "false"
            )
            data = html.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
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
    contradicting = _ids_from_json(row["contradicting_episode_ids"])
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
        "confidence": float(row["confidence"]),
        "salience": float(row["salience"]),
        "is_labile": bool(row["is_labile"]),
        "facets": facets,
        "schema_class": _schema_class(facets),
        "tags": tags,
        "support_count": len(supporting),
        "contradict_count": len(contradicting),
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
                "feedback_events": _table_count(conn, "context_feedback_events"),
                # Generalization stage counts (Stage 11)
                "promoted_schemas": _count_by_gen_stage(conn, min_stage=1),
                "global_schemas": _count_by_gen_stage(conn, min_stage=3),
                "known_scopes": _table_count(conn, "scope_registry"),
            }
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
        daemon = _daemon_health()
        processes = _slowave_processes()
        return {
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
        - hours:    look-back window in hours  (default 2, max 24)
        - bucket_m: bucket size in minutes     (default 5, max 60)
    """
    hours = min(max(_qs_int(qs, "hours", 2), 1), 24)
    bucket_m = min(max(_qs_int(qs, "bucket_m", 5), 1), 60)
    bucket_s = bucket_m * 60
    now = int(time.time())
    window_start = now - hours * 3600
    first_bucket = (window_start // bucket_s) * bucket_s

    all_ts: list[int] = []
    t = first_bucket
    while t <= now:
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
                (bucket_s, bucket_s, first_bucket, now),
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
            "now_ts": now,
            # legacy single-channel keys so old code doesn't break
            "buckets": channels["raw_events"],
            "total_events": sum(b["n"] for b in channels["raw_events"]),
            "max_n": max((b["n"] for b in channels["raw_events"]), default=0),
        }
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
        return {"db_path": db_path, "db_exists": False}
    conn = _connect(db_path)
    try:
        pragmas: dict[str, Any] = {}
        for name in ("journal_mode", "foreign_keys", "page_count", "page_size"):
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
                "SELECT name, type FROM sqlite_master WHERE type IN ('table', 'view') ORDER BY type, name"
            ).fetchall()
        ]
        return {
            "db_path": db_path,
            "db_exists": True,
            "pragmas": pragmas,
            "integrity_check": integrity,
            "foreign_key_check": fk,
            "tables": tables,
        }
    finally:
        conn.close()


def _schemas_payload(db_path: str, qs: dict[str, list[str]]) -> dict[str, Any]:
    limit = max(1, min(500, _qs_int(qs, "limit", 100)))
    status = (qs.get("status") or [""])[0]
    scope = (qs.get("scope") or [""])[0]
    q = (qs.get("q") or [""])[0].strip().lower()
    if status and status not in VALID_SCHEMA_STATUSES:
        raise ValueError(f"unknown status filter: {status!r}")
    args: list[Any] = []
    sql = "SELECT * FROM schemas WHERE 1=1"
    if status:
        sql += " AND status = ?"
        args.append(status)
    if scope:
        sql += " AND scope_id = ?"
        args.append(scope)
    if q:
        sql += " AND lower(content_text) LIKE ?"
        args.append(f"%{q}%")
    sql += " ORDER BY salience DESC, last_updated_ts DESC LIMIT ?"
    args.append(limit)
    conn = _connect(db_path)
    try:
        rows = conn.execute(sql, tuple(args)).fetchall()
        proto_map = _prototype_map(conn, [int(r["id"]) for r in rows])
        return {"schemas": [_schema_row_to_node(r, proto_map.get(int(r["id"]), [])) for r in rows]}
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
    statuses_raw = (qs.get("statuses") or ["active,needs_review,contradicted,superseded"])[0]
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
        if schema_ids:
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
                "re.ts AS event_ts, re.session_id AS event_session "
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
        return {
            "schema": schema,
            "evidence": evidence,
            "outgoing": outgoing,
            "incoming": incoming,
            "coact_outgoing": coact_outgoing,
            "coact_incoming": coact_incoming,
        }
    finally:
        conn.close()


# Mutating actions below are only reachable when the server was started with
# --allow-actions (see do_POST); the dashboard is read-only by default. Logic
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
            "SELECT status, generalization_stage FROM schemas WHERE id = ?", (schema_id,)
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
            return {"error": "schema is not currently forgotten", "schema_id": schema_id}
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


def _worker_runs_payload(db_path: str, qs: dict[str, list[str]]) -> dict[str, Any]:
    """Return worker consolidation run history and summary statistics."""
    if not os.path.exists(db_path):
        return {"runs": [], "summary": {}}
    limit = max(1, min(200, _qs_int(qs, "limit", 50)))
    conn = _connect(db_path)
    try:
        rows = conn.execute(
            "SELECT * FROM worker_runs ORDER BY started_ts DESC LIMIT ?",
            (limit,),
        ).fetchall()
        runs = [dict(r) for r in rows]
        # summary stats
        total_row = conn.execute(
            "SELECT COUNT(*) AS n, MAX(started_ts) AS last_ts,"
            " SUM(schemas_created) AS schemas_created,"
            " SUM(schemas_reinforced) AS schemas_reinforced,"
            " SUM(schemas_decayed) AS schemas_decayed,"
            " AVG(duration_ms) AS avg_ms"
            " FROM worker_runs WHERE error_text IS NULL"
        ).fetchone()
        return {
            "runs": runs,
            "summary": {
                "total_passes": int(total_row["n"]) if total_row else 0,
                "last_run_ts": total_row["last_ts"] if total_row else None,
                "total_schemas_created": int(total_row["schemas_created"] or 0) if total_row else 0,
                "total_schemas_reinforced": (
                    int(total_row["schemas_reinforced"] or 0) if total_row else 0
                ),
                "total_schemas_decayed": int(total_row["schemas_decayed"] or 0) if total_row else 0,
                "avg_duration_ms": round(float(total_row["avg_ms"] or 0), 1) if total_row else 0,
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
        base_sql = "FROM episodic_memories e"
        base_params: list[Any] = []
        if search:
            base_sql += " WHERE e.metadata_json LIKE ?"
            base_params.append(f"%{search}%")
        total_row = conn.execute(f"SELECT COUNT(*) AS n {base_sql}", base_params).fetchone()
        rows = conn.execute(
            f"SELECT e.id, e.event_id, e.ts, e.salience, e.recalled_count, "
            f"e.metadata_json {base_sql} "
            f"ORDER BY e.ts DESC LIMIT ? OFFSET ?",
            base_params + [limit, offset],
        ).fetchall()
        episodes = []
        for r in rows:
            rec = dict(r)
            meta = _json_loads(rec.pop("metadata_json", None), {})
            rec["content_preview"] = str(
                meta.get(
                    "text",
                    meta.get(
                        "content", f'{meta.get("kind","")} session={meta.get("session_id","")}'
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
            "SELECT e.id, e.event_id, e.ts, e.salience, e.recalled_count, r.content "
            "FROM episodic_memories e "
            "JOIN raw_events r ON r.id = e.event_id "
            "WHERE r.session_id = ? "
            "ORDER BY e.ts ASC",
            (session_id,),
        ).fetchall()
        return {
            "session": dict(sess),
            "events": [dict(r) for r in events],
            "episodes": [dict(r) for r in episodes],
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
    from slowave.symbolic.procedural import build_embedding_caches, cluster_sessions, rank_clusters

    if not os.path.exists(db_path):
        return {"clusters": [], "gate": "db_not_found", "step_sessions": 0, "qualified_sessions": 0}

    min_sessions = max(2, min(20, _qs_int(qs, "min_sessions", 2)))
    max_clusters = max(5, min(50, _qs_int(qs, "limit", 20)))
    threshold = 0.4
    scope_filter = (qs.get("scope") or [""])[0].strip()

    conn = _connect(db_path)
    try:
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
        }
    finally:
        conn.close()


from slowave.dashboard._html import _INDEX_HTML  # noqa: E402


def _graph_health_payload(db_path: str) -> dict[str, Any]:
    """Return full graph health metrics for the /api/debug/graph endpoint."""
    from slowave.core.graph_health import compute

    return compute(db_path)
