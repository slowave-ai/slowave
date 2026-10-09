"""Dashboard measurement projection: canonical delivery occasions, never page rows."""

from __future__ import annotations

import math
import sqlite3
import statistics
import time
from collections import Counter, defaultdict
from typing import Any

MEMORY_ASSESSMENTS = frozenset(
    {"used", "not_used", "irrelevant", "already_known", "unassessable", "stale", "wrong"}
)
PROCEDURE_ASSESSMENTS = frozenset({"used", "not_used", "unassessable"})


def value(qs: dict[str, list[str]], key: str, default: str = "") -> str:
    """Read the first query parameter, falling back only when it is absent."""
    return (qs.get(key) or [default])[0]


def bounds(qs: dict[str, list[str]]) -> tuple[int, int]:
    """Resolve an inclusive request-time interval; zero means unbounded start."""
    end = int(value(qs, "to", "0") or 0) or int(time.time())
    start = int(value(qs, "from", "0") or 0)
    hours = value(qs, "hours", "all")
    if not start and hours != "all":
        start = end - max(1, min(8760, int(hours or 3))) * 3600
    return start, end


def delivered_sql() -> str:
    """Select unique supported request/target deliveries, collapsing kind aliases."""
    # Kind aliases collapse to the same occasion, including historical duplicate rows.
    return """SELECT DISTINCT context_id, memory_id AS target_id,
        CASE WHEN memory_type IN ('schema','related') THEN 'memory'
             ELSE 'procedure' END AS target_kind
        FROM context_recall_items WHERE admitted=1 AND
        ((memory_type IN ('schema','related') AND memory_id GLOB 'sch_[0-9]*')
         OR memory_type IN ('procedure','procedural_memory'))"""


def occasions(
    conn: sqlite3.Connection, where: str = "1=1", args: list[Any] | None = None
) -> list[dict[str, Any]]:
    """Join actual deliveries to their latest accepted target assessment."""
    return [
        dict(row)
        for row in conn.execute(
            f"""WITH delivered AS ({delivered_sql()}), latest AS (
        SELECT f.*, ROW_NUMBER() OVER (
            PARTITION BY retrieval_id,target_kind,target_id
            ORDER BY created_at DESC,rowid DESC) AS position
        FROM feedback_events f WHERE status='accepted')
        SELECT d.*, r.created_at AS request_ts, r.session_id, r.scope_id,
               f.assessment, f.effect, f.created_at AS feedback_ts
        FROM delivered d JOIN context_recall_events r ON r.context_id=d.context_id
        LEFT JOIN latest f ON f.retrieval_id=d.context_id AND f.target_kind=d.target_kind
            AND f.target_id=d.target_id AND f.position=1
        WHERE {where}""",
            args or [],
        )
    ]


def assessed(item: dict[str, Any]) -> bool:
    """Whether an occasion has feedback supported by its target contract."""
    return item["assessment"] in (
        MEMORY_ASSESSMENTS if item["target_kind"] == "memory" else PROCEDURE_ASSESSMENTS
    )


def _memory_matches(target_id: str, record: dict[str, Any], query: str) -> bool:
    """Mirror the library's exact numeric-ID and case-insensitive text search."""
    schema_id = query.removeprefix("sch_")
    if schema_id.isdigit():
        return int(target_id.removeprefix("sch_")) == int(schema_id)
    return query in str(record["content_text"]).lower()


def projection(
    conn: sqlite3.Connection,
    qs: dict[str, list[str]],
    *,
    request_filter: tuple[str, list[Any]] | None = None,
) -> dict[str, Any]:
    """Compute cards and charts from a shared request and inventory population."""
    from slowave.symbolic.procedural_memory import load_procedures

    scope = value(qs, "scope")
    start, end = bounds(qs)
    where, filter_args = request_filter or ("r.created_at >= ? AND r.created_at <= ?", [start, end])
    args = list(filter_args)  # Caller-owned bindings must not be extended in place.
    cohort = value(qs, "cohort", "all")
    if cohort == "v9":
        where += " AND CAST(substr(r.lifecycle_version,2) AS INTEGER)>=9"
    elif cohort != "all":
        where += " AND r.lifecycle_version=?"
        args.append(cohort)
    if scope and request_filter is None:
        where += " AND r.scope_id=?"
        args.append(scope)
    requests = [
        dict(r)
        for r in conn.execute(f"SELECT r.* FROM context_recall_events r WHERE {where}", args)
    ]
    items = occasions(conn, where, args)
    session_starts = {
        row["id"]: int(row["started_ts"])
        for row in conn.execute(
            f"SELECT DISTINCT s.id, s.started_ts FROM sessions s "
            f"JOIN context_recall_events r ON r.session_id=s.id WHERE {where}",
            args,
        )
    }
    memories = {
        f"sch_{r['id']}": dict(r)
        for r in conn.execute(
            "SELECT * FROM schemas WHERE status='active'" + (" AND scope_id=?" if scope else ""),
            [scope] if scope else [],
        )
    }
    # Text and Saved since define a library, exposure/use controls do not.
    query = value(qs, "q").strip().lower() if value(qs, "library") == "memory" else ""
    saved_since = int(value(qs, "saved_from", "0") or 0)
    memories = {
        k: r
        for k, r in memories.items()
        if (not query or _memory_matches(k, r, query))
        and int(r["first_formed_ts"] or 0) >= saved_since
    }
    procedures = {str(p["id"]): p for p in load_procedures(conn, scope=scope or None)}
    if value(qs, "library") == "procedure":
        procedures = {
            k: p
            for k, p in procedures.items()
            if (not value(qs, "outcome") or p["outcome"] == value(qs, "outcome"))
            and (
                not value(qs, "verification")
                or p["verification_status"] == value(qs, "verification")
            )
            and int(p["created_at"] or 0) >= saved_since
        }
    result: dict[str, Any] = {
        "available": True,
        "cohort": cohort,
        "scope": scope,
        "window": {"from": start, "to": end},
    }
    for kind, inventory in [("memory", memories), ("procedure", procedures)]:
        selected = [i for i in items if i["target_kind"] == kind and i["target_id"] in inventory]
        exposed = {i["target_id"] for i in selected}
        known = {
            i["target_id"]
            for i in selected
            if assessed(i) and (kind == "memory" or i["assessment"] != "unassessable")
        }
        used = {i["target_id"] for i in selected if i["assessment"] == "used"}
        tasks: dict[str, set[str]] = defaultdict(set)
        for i in selected:
            if i["assessment"] == "used" and i["session_id"]:
                if kind == "procedure":
                    source = i["target_id"].removeprefix("proc_")
                    task_started = session_starts.get(i["session_id"])
                    if (
                        i["session_id"] == source
                        or task_started is None
                        or task_started <= int(inventory[i["target_id"]]["created_at"] or 0)
                    ):
                        continue
                tasks[i["target_id"]].add(i["session_id"])
        result.update(
            {
                f"{kind}_total": len(inventory),
                f"{kind}_exposed": len(exposed),
                f"{kind}_assessed": len(known),
                f"{kind}_used": len(used),
                f"{kind}_unknown": len(exposed - known),
                f"{kind}_deliveries": len(selected),
                f"{kind}_feedback": sum(assessed(i) for i in selected),
                f"{kind}_across_tasks": sum(len(t) >= 2 for t in tasks.values()),
                f"{kind}_across_task_ids": sorted(k for k, t in tasks.items() if len(t) >= 2),
            }
        )
    stages = Counter(r["generalization_stage"] for r in memories.values())
    result["stages"] = [stages[k] for k in range(4)]
    result["unknown_stage"] = sum(n for k, n in stages.items() if k not in range(4))
    result["active_scopes"] = len({r["scope_id"] for r in memories.values() if r["scope_id"]})
    matched = {i["context_id"] for i in items}
    result["retrievals_feedback_complete"] = int(
        conn.execute(
            f"SELECT COUNT(*) FROM context_recall_events r WHERE ({where}) AND ({complete_request_sql()})",
            args,
        ).fetchone()[0]
    )
    result.update(
        retrievals_total=len(requests),
        retrievals_no_match=len(requests) - len(matched),
        retrievals_used=len({i["context_id"] for i in items if i["assessment"] == "used"}),
        deliveries=len(items),
        feedback_assessed=sum(assessed(i) for i in items),
        feedback_unknown=sum(not assessed(i) for i in items),
        task_starts=sum(r["retrieval_type"] in {"context", "activate"} for r in requests),
        additional_requests=sum(
            r["retrieval_type"] not in {"context", "activate"} for r in requests
        ),
    )
    memory_items = [i for i in items if i["target_kind"] == "memory"]
    result.update(
        irrelevant=sum(i["assessment"] == "irrelevant" for i in memory_items),
        assessed_memory_deliveries=sum(assessed(i) for i in memory_items),
        missing_memory_feedback=sum(not assessed(i) for i in memory_items),
    )
    used_procedures = [
        i
        for i in items
        if i["target_kind"] == "procedure"
        and i["target_id"] in procedures
        and i["assessment"] == "used"
    ]
    effects = Counter(i["effect"] for i in used_procedures)
    result.update(
        effect_known=sum(effects[k] for k in ("helped", "no_effect", "harmed")),
        effect_unknown=sum(
            n for k, n in effects.items() if k not in {"helped", "no_effect", "harmed"}
        ),
        procedure_helped=effects["helped"],
        procedure_no_effect=effects["no_effect"],
        procedure_harmed=effects["harmed"],
    )
    result["context_size"] = {}
    for name, kinds in [("Activate", {"context", "activate"}), ("Recall", {"recall"})]:
        samples = sorted(
            int(r["response_chars"])
            for r in requests
            if r["retrieval_type"] in kinds and r.get("response_chars") is not None
        )
        result["context_size"][name] = {
            "samples": len(samples),
            "median_chars": statistics.median(samples) if samples else None,
            "p95_chars": samples[math.ceil(len(samples) * 0.95) - 1] if samples else None,
        }
    result["unsupported_deliveries"] = int(
        conn.execute(
            f"SELECT COUNT(*) FROM context_recall_items i JOIN context_recall_events r ON r.context_id=i.context_id WHERE i.admitted=1 AND NOT ((i.memory_type IN ('schema','related') AND i.memory_id GLOB 'sch_[0-9]*') OR i.memory_type IN ('procedure','procedural_memory')) AND {where}",
            args,
        ).fetchone()[0]
    )
    result["chart"] = chart(conn, qs, items, memories, procedures, start, end)
    return result


def chart(
    conn: sqlite3.Connection,
    qs: dict[str, list[str]],
    items: list[dict[str, Any]],
    memories: dict[str, Any],
    procedures: dict[str, Any],
    start: int,
    end: int,
) -> dict[str, Any]:
    """Bucket saved records and delivered occasions by their original timestamps."""
    scope = value(qs, "scope")
    # Historical records remain in activity even after retirement.
    saved = [
        (int(r["first_formed_ts"]), "schemas")
        for r in conn.execute(
            "SELECT first_formed_ts FROM schemas" + (" WHERE scope_id=?" if scope else ""),
            [scope] if scope else [],
        )
    ]
    if value(qs, "library") == "memory" and (value(qs, "q") or value(qs, "saved_from")):
        historical = {
            f"sch_{r['id']}": dict(r)
            for r in conn.execute(
                "SELECT * FROM schemas" + (" WHERE scope_id=?" if scope else ""),
                [scope] if scope else [],
            )
        }
        query = value(qs, "q").strip().lower()
        memories = {
            k: r
            for k, r in historical.items()
            if (not query or _memory_matches(k, r, query))
            and int(r["first_formed_ts"] or 0) >= int(value(qs, "saved_from", "0") or 0)
        }
        saved = [(int(r["first_formed_ts"]), "schemas") for r in memories.values()]
        items = [i for i in items if i["target_kind"] != "memory" or i["target_id"] in memories]
    saved += [(int(p["created_at"] or 0), "procedures") for p in procedures.values()]
    if not start:
        start = min([end, *[ts for ts, _ in saved], *[i["request_ts"] for i in items]])
    window = max(1, end - start)
    bucket = (
        300
        if window <= 86400
        else 3600 if window <= 7 * 86400 else 86400 if window <= 365 * 86400 else 604800
    )
    grid = range(start // bucket * bucket, end // bucket * bucket + 1, bucket)
    channels: dict[str, Counter[int]] = {
        k: Counter()
        for k in [
            "schemas",
            "procedures",
            "memory_retrievals",
            "procedure_retrievals",
            "used_memories",
            "used_procedures",
        ]
    }
    for ts, channel in saved:
        if start <= ts <= end:
            channels[channel][ts // bucket * bucket] += 1
    for i in items:
        ts = i["request_ts"] // bucket * bucket
        channels[f"{i['target_kind']}_retrievals"][ts] += 1
        if i["assessment"] == "used":
            channels["used_memories" if i["target_kind"] == "memory" else "used_procedures"][
                ts
            ] += 1
    return {
        "channels": {k: [{"ts": ts, "n": c[ts]} for ts in grid] for k, c in channels.items()},
        "bucket_minutes": bucket // 60,
        "window_start": start,
        "now_ts": end,
        "global_max": max((max(c.values(), default=0) for c in channels.values()), default=0),
        "window_hours": max(1, (end - start) // 3600),
    }


def history(items: list[dict[str, Any]], kind: str, target_id: str) -> dict[str, Any]:
    """Summarize a target independently of displayed history pagination."""
    selected = [i for i in items if i["target_kind"] == kind and i["target_id"] == target_id]
    known = [
        i
        for i in selected
        if assessed(i) and (kind == "memory" or i["assessment"] != "unassessable")
    ]
    result: dict[str, Any] = {
        "retrieved": len(selected),
        "assessed": len(known),
        "unknown": len(selected) - len(known),
        "last_retrieved": max((i["request_ts"] for i in selected), default=None),
        "last_used": max(
            (i["request_ts"] for i in selected if i["assessment"] == "used"), default=None
        ),
    }
    for key in MEMORY_ASSESSMENTS | PROCEDURE_ASSESSMENTS:
        result[key] = sum(i["assessment"] == key for i in selected)
    for key in ("helped", "harmed", "no_effect"):
        result[key] = sum(i["assessment"] == "used" and i["effect"] == key for i in selected)
    result["unknown_effect"] = sum(
        i["assessment"] == "used" and i["effect"] not in {"helped", "harmed", "no_effect"}
        for i in selected
    )
    return result


def complete_request_sql() -> str:
    """Require a complete declaration and supported feedback for every delivery."""
    # A declaration alone cannot discharge delivered item accountability.
    return f"""EXISTS(SELECT 1 FROM feedback_events declaration WHERE
        declaration.retrieval_id=r.context_id AND declaration.status='accepted'
        AND declaration.coverage='complete') AND NOT EXISTS(
        SELECT 1 FROM ({delivered_sql()}) d WHERE d.context_id=r.context_id AND NOT EXISTS(
        SELECT 1 FROM feedback_events f WHERE f.retrieval_id=d.context_id
        AND f.target_kind=d.target_kind AND f.target_id=d.target_id AND f.status='accepted'
        AND NOT EXISTS(SELECT 1 FROM feedback_events newer WHERE newer.status='accepted'
          AND newer.retrieval_id=f.retrieval_id AND newer.target_kind=f.target_kind
          AND newer.target_id=f.target_id AND (newer.created_at>f.created_at OR
          (newer.created_at=f.created_at AND newer.rowid>f.rowid)))
        AND ((d.target_kind='memory' AND f.assessment IN ('used','not_used','irrelevant','already_known','unassessable','stale','wrong'))
          OR (d.target_kind='procedure' AND f.assessment IN ('used','not_used','unassessable')))))"""
