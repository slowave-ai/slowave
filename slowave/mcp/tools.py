"""Shared MCP tool registration for Slowave.

Provides a single ``register_tools(mcp, build_engine)`` function that attaches
all 5 cognitive lifecycle tools to any FastMCP instance.  Both the stdio server (server.py) and
the HTTP daemon (http_server.py) call this function so there is no
duplication of tool logic.

Tools registered (5 cognitive-cycle verbs):
  activate, remember, recall, feedback, commit

remember and feedback each also accept an optional `items` list for
batching several calls into one round trip (see their docstrings) — this
does not add new tool names, just an alternate parameter shape on the
existing two.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import secrets
import time
from datetime import datetime, timezone
from typing import Any, Callable

from mcp.server.fastmcp import Context, FastMCP

import slowave.ops as ops
from slowave.mcp import session_resolver

log = logging.getLogger(__name__)

# Keys stored in schema facets that are internal to the retrieval engine.
_INTERNAL_FACET_KEYS: frozenset[str] = frozenset({"vsa_vec"})

# Phase-1 trustworthy-memory dogfood defaults (2026-08-08). These are scoped
# to the MCP product surface; library/benchmark callers retain their existing
# defaults, including Recall@20. Calibrated on the frozen 60-call live replay.
_MCP_ACTIVATE_LIMIT_DEFAULT = 2
_MCP_ACTIVATE_MIN_RELEVANCE_DEFAULT = 0.20
_MCP_RECALL_TOP_K_DEFAULT = 2
_MCP_RECALL_MIN_RELEVANCE_DEFAULT = 0.40
_MCP_FROZEN_CANDIDATE_LIMIT = _MCP_RECALL_TOP_K_DEFAULT + 8
_MCP_CONTINUATION_PAGE_SIZE = 2
_MCP_FIELD_RESPONSE_CHARS = 160
_MCP_MEMORY_CONTENT_LIMIT = 500
_MCP_CONTINUITY_START_RESPONSE_CHARS = 1600
_MCP_CONTINUATION_RESPONSE_CHARS = 900
_MCP_EVIDENCE_LIMIT = 8
_MCP_EVIDENCE_CONTENT_LIMIT = 1000
_REMEMBER_TYPES = {
    "fact",
    "preference",
    "decision",
    "constraint",
    "instruction",
    "lesson",
    "warning",
    "open_question",
    "task",
    "artifact",
}


def _validate_scope(scope: str) -> None:
    """Apply the same public scope contract to every scoped lifecycle verb."""
    if (
        not scope.strip()
        or ":" not in scope
        or not all(part.strip() for part in scope.split(":", 1))
    ):
        raise ValueError("scope must use nonblank kind:id form")


def _compact_source_provenance(item: dict[str, Any]) -> dict[str, Any]:
    provenance: dict[str, Any] = {}
    if item.get("source_kind"):
        provenance["source_kind"] = item["source_kind"]
    source = item.get("source_provenance") or {}
    for key in ("integration", "integration_version", "observed"):
        if key in source:
            provenance[key] = source[key]
    return provenance


def _parse_occurred_at(value: str | None) -> int | None:
    """Parse client source time without changing internal event ordering."""
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError("occurred_at must be a nonblank RFC 3339 UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(
            "occurred_at must be an RFC 3339 timestamp, for example 2026-08-26T09:30:00Z"
        ) from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("occurred_at must include a UTC offset, for example 2026-08-26T09:30:00Z")
    return int(parsed.astimezone(timezone.utc).timestamp())


def _normalize_remember_inputs(
    *,
    content: str | None,
    memory_type: str | None,
    occurred_at: str | None = None,
    memories: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], bool]:
    if memories is not None and (content is not None or memory_type is not None):
        raise ValueError("memories is mutually exclusive with content and type")
    if memories is None:
        if not content or not content.strip():
            raise ValueError("content must be nonblank")
        if memory_type not in _REMEMBER_TYPES:
            raise ValueError(f"type must be one of {sorted(_REMEMBER_TYPES)}")
        return [
            {
                "content": content,
                "type": memory_type,
                "occurred_at": _parse_occurred_at(occurred_at),
            }
        ], False
    if not memories:
        raise ValueError("memories must be a non-empty list")
    normalized: list[dict[str, Any]] = []
    for item in memories:
        if (
            not isinstance(item, dict)
            or not {"content", "type"} <= set(item)
            or set(item) - {"content", "type", "occurred_at"}
        ):
            raise ValueError(
                "each memories entry must contain content, type, and optional occurred_at"
            )
        if not isinstance(item["content"], str) or not item["content"].strip():
            raise ValueError("each memories content must be nonblank")
        if item["type"] not in _REMEMBER_TYPES:
            raise ValueError(f"type must be one of {sorted(_REMEMBER_TYPES)}")
        normalized.append(
            {
                "content": item["content"],
                "type": item["type"],
                "occurred_at": _parse_occurred_at(item.get("occurred_at")),
            }
        )
    return normalized, True


def _serialized_chars(value: dict[str, Any]) -> int:
    return len(json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True))


def _candidate_kind(item: dict[str, Any]) -> str:
    if item["kind"] == "procedure":
        return "procedure"
    return str(item["value"].get("pathway") or "memory")


def _accessible_field(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    """Return a bounded orientation signal containing no candidate content."""
    if not candidates:
        return {}
    field = {
        "extra_candidates": len(candidates),
        "kinds": sorted({_candidate_kind(item) for item in candidates}),
        "approx_extra_tokens": math.ceil(
            sum(_serialized_chars(item["value"]) for item in candidates) / 4
        ),
    }
    # The field has its own hard ceiling.  Kinds are lower priority than the
    # count, so discard them if future labels make the envelope too large.
    if _serialized_chars(field) > _MCP_FIELD_RESPONSE_CHARS:
        field = {
            "extra_candidates": len(candidates),
            "approx_extra_tokens": field["approx_extra_tokens"],
        }
    return field


def _freeze_continuation(
    eng: Any,
    *,
    retrieval_id: str,
    session_id: str,
    scope: str,
    candidates: list[dict[str, Any]],
    offset: int,
    page_size: int = _MCP_CONTINUATION_PAGE_SIZE,
    cursor_id: str | None = None,
) -> str | None:
    if offset >= len(candidates):
        return None
    cursor = cursor_id or "cur_" + secrets.token_urlsafe(24)
    eng.db.connect().execute(
        "INSERT OR IGNORE INTO retrieval_continuations "
        "(cursor_id, retrieval_id, session_id, scope_id, candidates_json, offset_n, page_size, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            cursor,
            retrieval_id,
            session_id,
            scope,
            json.dumps(candidates, ensure_ascii=False, separators=(",", ":"), sort_keys=True),
            offset,
            page_size,
            int(time.time()),
        ),
    )
    eng.db.connect().commit()
    return cursor


def _read_continuation(
    eng: Any, *, cursor: str, session_id: str, scope: str
) -> tuple[str, list[dict[str, Any]], int, int]:
    row = (
        eng.db.connect()
        .execute(
            "SELECT retrieval_id, session_id, scope_id, candidates_json, offset_n, page_size "
            "FROM retrieval_continuations WHERE cursor_id = ?",
            (cursor,),
        )
        .fetchone()
    )
    if row is None:
        raise ValueError("unknown continue_from cursor")
    if row["session_id"] != session_id or row["scope_id"] != scope:
        raise ValueError("continue_from cursor does not match session_id and scope")
    session = (
        eng.db.connect()
        .execute(
            "SELECT ended_ts FROM sessions WHERE id = ? AND scope_id = ?",
            (session_id, scope),
        )
        .fetchone()
    )
    if session is None:
        raise ValueError("continue_from cursor session no longer exists")
    if session["ended_ts"] is not None:
        raise ValueError("continue_from cursor session is already ended")
    return (
        str(row["retrieval_id"]),
        json.loads(row["candidates_json"]),
        int(row["offset_n"]),
        int(row["page_size"]),
    )


def _continuation_page(eng: Any, *, cursor: str, session_id: str, scope: str) -> dict[str, Any]:
    retrieval_id, candidates, offset, page_size = _read_continuation(
        eng, cursor=cursor, session_id=session_id, scope=scope
    )
    data: dict[str, Any] = {
        "retrieval_id": retrieval_id,
        "memories": [],
        "procedures": [],
        "evidence": [],
        "evidence_mode": "references",
        "evidence_truncated": False,
        # Reserve cursor metadata before admitting content so the focus page
        # itself never overflows when a real tail is attached below.  The
        # orientation field has its own independent ceiling.
        "more_available": True,
        "continue_from": "cur_" + "x" * 32,
    }
    next_offset = offset
    admitted = 0
    while next_offset < len(candidates) and admitted < page_size:
        item = candidates[next_offset]
        key = "memories" if item["kind"] == "memory" else "procedures"
        data[key].append(item["value"])
        if _serialized_chars(data) > _MCP_CONTINUATION_RESPONSE_CHARS:
            data[key].pop()
            break
        next_offset += 1
        admitted += 1
    remaining = candidates[next_offset:]
    next_cursor = _freeze_continuation(
        eng,
        retrieval_id=retrieval_id,
        session_id=session_id,
        scope=scope,
        candidates=candidates,
        offset=next_offset,
        page_size=page_size,
        cursor_id=(
            "cur_"
            + hashlib.sha256(f"{cursor}:{retrieval_id}:{next_offset}".encode("utf-8")).hexdigest()[
                :32
            ]
        ),
    )
    data["more_available"] = next_cursor is not None
    if next_cursor is not None:
        data["continue_from"] = next_cursor
    else:
        data.pop("continue_from", None)
    data["accessible_field"] = _accessible_field(remaining)
    _authorize_continuation_exposure(eng, retrieval_id=retrieval_id, data=data)
    return data


def _canonical_candidates(data: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = [{"kind": "memory", "value": item} for item in data.get("memories", [])]
    candidates.extend({"kind": "procedure", "value": item} for item in data.get("procedures", []))
    return candidates


def _candidate_fits_continuation(item: dict[str, Any]) -> bool:
    probe = {
        "retrieval_id": "rec_x",
        "memories": [item["value"]] if item["kind"] == "memory" else [],
        "procedures": [item["value"]] if item["kind"] == "procedure" else [],
        "evidence": [],
        "evidence_mode": "references",
        "evidence_truncated": False,
        "more_available": True,
        "continue_from": "cur_" + "x" * 32,
        "accessible_field": {},
    }
    return _serialized_chars(probe) <= _MCP_CONTINUATION_RESPONSE_CHARS


def _attach_frozen_tail(
    eng: Any,
    *,
    data: dict[str, Any],
    candidates: list[dict[str, Any]],
    session_id: str,
    scope: str,
    exposed_ids: set[str],
) -> None:
    tail = [
        item
        for item in candidates
        if (item["value"].get("memory_id") or item["value"].get("procedure_id")) not in exposed_ids
        and _candidate_fits_continuation(item)
    ]
    cursor = _freeze_continuation(
        eng,
        retrieval_id=data["retrieval_id"],
        session_id=session_id,
        scope=scope,
        candidates=tail,
        offset=0,
    )
    data["more_available"] = cursor is not None
    if cursor is not None:
        data["continue_from"] = cursor
    else:
        data.pop("continue_from", None)
    data["accessible_field"] = _accessible_field(tail)


def _authorize_continuation_exposure(eng: Any, *, retrieval_id: str, data: dict[str, Any]) -> None:
    """Make only actually rendered continuation items eligible for feedback."""
    conn = eng.db.connect()
    retrieval_type = "recall"
    parent = conn.execute(
        "SELECT retrieval_type FROM context_recall_events WHERE context_id = ?",
        (retrieval_id,),
    ).fetchone()
    if parent is not None and parent["retrieval_type"]:
        retrieval_type = str(parent["retrieval_type"])
    rank = int(
        conn.execute(
            "SELECT COUNT(*) FROM context_recall_items WHERE context_id = ? AND admitted = 1",
            (retrieval_id,),
        ).fetchone()[0]
    )
    exposed: list[str] = []
    for memory_type, items in (
        ("schema", data.get("memories", [])),
        ("procedural_memory", data.get("procedures", [])),
    ):
        for item in items:
            memory_id = item.get("memory_id") or item.get("procedure_id")
            if not memory_id:
                continue
            conn.execute(
                "INSERT OR IGNORE INTO context_recall_items "
                "(context_id, memory_id, retrieval_type, memory_type, rank, content_text, "
                "admitted, pathway, phase, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, 1, ?, 'continuation', ?)",
                (
                    retrieval_id,
                    memory_id,
                    retrieval_type,
                    memory_type,
                    rank,
                    str(item.get("content") or item.get("summary") or "")[
                        :_MCP_MEMORY_CONTENT_LIMIT
                    ],
                    item.get("pathway", "direct"),
                    int(time.time()),
                ),
            )
            exposed.append(str(memory_id))
            rank += 1
    if exposed:
        current = conn.execute(
            "SELECT memory_ids_json FROM context_recall_events WHERE context_id = ?",
            (retrieval_id,),
        ).fetchone()
        memory_ids = set(json.loads(current[0] or "[]")) if current else set()
        memory_ids.update(exposed)
        conn.execute(
            "UPDATE context_recall_events SET memory_ids_json = ?, count_n = ? WHERE context_id = ?",
            (json.dumps(sorted(memory_ids)), len(memory_ids), retrieval_id),
        )
    conn.commit()


def _compact_activation_procedure(item: dict[str, Any], scope: str) -> dict[str, Any]:
    """Return the minimum safe activation preview for a procedure."""
    full = _canonical_procedure(item, scope)
    preview = {
        "procedure_id": full["procedure_id"],
        "goal": full.get("goal", ""),
        "summary": full.get("summary", ""),
        "outcome": full.get("outcome", "unknown"),
        "outcome_summary": full.get("outcome_summary", ""),
    }
    # Safety caveats are never silently shortened.  A procedure that cannot
    # fit as an honest preview is left for deliberate recall.
    if full.get("caveats"):
        preview["caveats"] = full["caveats"]
    if full.get("origin_scope"):
        preview["origin_scope"] = full["origin_scope"]
    return preview


def _activation_candidates(result: dict[str, Any], scope: str) -> list[dict[str, Any]]:
    memories = []
    for item in result.get("schemas", []):
        memory: dict[str, Any] = {
            "memory_id": item["id"],
            "content": str(item.get("text") or "")[:_MCP_MEMORY_CONTENT_LIMIT],
            "pathway": item.get("pathway", "direct"),
        }
        provenance = _compact_source_provenance(item)
        if item.get("scope_id") and item["scope_id"] != scope:
            provenance["origin_scope"] = item["scope_id"]
        if provenance:
            memory["provenance"] = provenance
        memories.append(memory)
    ordered = [item for item in memories if item["pathway"] != "context_reinstatement"]
    candidates = [{"kind": "memory", "value": item} for item in ordered]
    candidates.extend(
        {"kind": "procedure", "value": _compact_activation_procedure(item, scope)}
        for item in result.get("procedures", [])
    )
    candidates.extend(
        {"kind": "memory", "value": item}
        for item in memories
        if item["pathway"] == "context_reinstatement"
    )
    return candidates


def _canonical_activation_result(result: dict[str, Any], *, scope: str) -> dict[str, Any]:
    """Project the internal activation result onto the stable O2 payload."""
    candidates = _activation_candidates(result, scope)
    warnings = []
    if result.get("scope_warning"):
        warnings.append({"code": "scope_fragmentation", "message": result["scope_warning"]})
    data: dict[str, Any] = {
        "retrieval_id": result["retrieval_id"],
        "session_id": result["session_id"],
        "memory_state": "cold_start" if result.get("cold_start") else "available",
        "memories": [],
        "procedures": [],
        "warnings": warnings,
    }
    if "continuity_id" in result:
        data["continuity_id"] = result["continuity_id"]
        data["continuity_state"] = result["continuity_state"]
        data["more_available"] = False
    if result.get("retrieval_policy_version"):
        data["retrieval_policy_version"] = result["retrieval_policy_version"]

    budget = (
        _MCP_CONTINUITY_START_RESPONSE_CHARS
        if result.get("continuity_state") == "started"
        else _MCP_CONTINUATION_RESPONSE_CHARS
    ) - 50  # reserve an opaque continue_from cursor on the final envelope
    # Core memory, procedure outcome/safety, then reinstatement context.
    omitted = False
    for envelope in candidates:
        kind, candidate = envelope["kind"], envelope["value"]
        field = "memories" if kind == "memory" else "procedures"
        data[field].append(candidate)
        if _serialized_chars(data) > budget:
            data[field].pop()
            omitted = True
    if omitted and "more_available" in data:
        data["more_available"] = True
        # Metadata must fit too; if it does not, remove lowest-priority context
        # first, then procedures, without ever truncating safety content.
        while _serialized_chars(data) > budget:
            contexts = [
                i
                for i, item in enumerate(data["memories"])
                if item["pathway"] == "context_reinstatement"
            ]
            if contexts:
                data["memories"].pop(contexts[-1])
            elif data["procedures"]:
                data["procedures"].pop()
            else:
                break
    return data


def _canonical_procedure(item: dict[str, Any], scope: str) -> dict[str, Any]:
    evidence = item.get("evidence") or {}
    compact_evidence = {
        key: int(evidence.get(key, 0))
        for key in ("used", "not_used", "helped", "no_effect", "harmed", "unknown")
        if int(evidence.get(key, 0))
    }
    contributions = [
        {
            "effect": entry.get("effect", "unknown"),
            "contribution": entry.get("contribution", ""),
            "outcome": entry.get("downstream_outcome", "unknown"),
            "outcome_summary": entry.get("downstream_outcome_summary", ""),
            "occurred_at": entry.get("created_at"),
        }
        for entry in (item.get("contributions") or [])[:3]
    ]
    procedure = {
        "procedure_id": item["id"],
        "goal": item.get("goal", ""),
        "summary": item.get("summary", ""),
        "context": item.get("context", {}),
        "steps": item.get("steps", []),
        "caveats": item.get("caveats", []),
        "outcome": item.get("outcome", "unknown"),
        "outcome_summary": item.get("outcome_summary", ""),
        "created_at": item.get("created_at"),
        "evidence": compact_evidence,
        "contributions": contributions,
    }
    if item.get("scope_id") and item["scope_id"] != scope:
        procedure["origin_scope"] = item["scope_id"]
    return procedure


def _canonical_recall_result(
    result: dict[str, Any], *, scope: str, evidence: str
) -> dict[str, Any]:
    memories: list[dict[str, Any]] = []
    seen: set[str] = set()
    for pathway, items in (
        ("direct", result.get("memories", [])),
        ("associated", result.get("related_memories", [])),
    ):
        for item in items:
            memory_id = item["id"]
            if memory_id in seen:
                continue
            seen.add(memory_id)
            provenance = _compact_source_provenance(item)
            if item.get("scope_id") and item["scope_id"] != scope:
                provenance["origin_scope"] = item["scope_id"]
            memory: dict[str, Any] = {
                "memory_id": memory_id,
                "content": str(item.get("content_text") or "")[:_MCP_MEMORY_CONTENT_LIMIT],
                "pathway": pathway,
            }
            if provenance:
                memory["provenance"] = provenance
            memories.append(memory)

    evidence_records: list[dict[str, Any]] = []
    raw_records = [("episode", item, "content_text") for item in result.get("episodes", [])] + [
        ("event", item, "content") for item in result.get("raw_events", [])
    ]
    for source_kind, item, content_key in raw_records[:_MCP_EVIDENCE_LIMIT]:
        raw_id = item.get("id")
        prefix = "ep" if source_kind == "episode" else "evt"
        record: dict[str, Any] = {
            "evidence_id": f"{prefix}_{raw_id}",
            "source_kind": source_kind,
            "recorded_at": item.get("recorded_at", item.get("ts")),
            "occurred_at": item.get("occurred_at", item.get("ts")),
            "source_ref": {"kind": source_kind, "id": raw_id},
        }
        if evidence == "full":
            content = str(item.get(content_key) or "")
            record["content"] = content[:_MCP_EVIDENCE_CONTENT_LIMIT]
            record["truncated"] = len(content) > _MCP_EVIDENCE_CONTENT_LIMIT
        evidence_records.append(record)

    return {
        "retrieval_id": result["retrieval_id"],
        "memories": memories,
        "procedures": [_canonical_procedure(item, scope) for item in result.get("procedures", [])],
        "evidence": evidence_records,
        "evidence_mode": evidence,
        "evidence_truncated": len(raw_records) > _MCP_EVIDENCE_LIMIT,
    }


def _focus_recall_data(data: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Apply the unchanged top-2 focus and the continuation char ceiling."""
    candidates = _canonical_candidates(data)
    focused = dict(data)
    focused["memories"] = []
    focused["procedures"] = []
    if len(candidates) > _MCP_CONTINUATION_PAGE_SIZE:
        focused["more_available"] = True
        focused["continue_from"] = "cur_" + "x" * 32
    admitted = 0
    for item in candidates:
        if admitted >= _MCP_CONTINUATION_PAGE_SIZE:
            break
        key = "memories" if item["kind"] == "memory" else "procedures"
        focused[key].append(item["value"])
        if _serialized_chars(focused) > _MCP_CONTINUATION_RESPONSE_CHARS:
            focused[key].pop()
            continue
        admitted += 1
    return focused, candidates


def _restrict_activation_exposure(eng: Any, *, retrieval_id: str, data: dict[str, Any]) -> None:
    """Keep feedback authorization exactly aligned with the serialized reply."""
    exposed = {item["memory_id"] for item in data.get("memories", [])}
    exposed.update(item["procedure_id"] for item in data.get("procedures", []))
    conn = eng.db.connect()
    rows = conn.execute(
        "SELECT memory_id FROM context_recall_items WHERE context_id = ? AND admitted = 1",
        (retrieval_id,),
    ).fetchall()
    for row in rows:
        if row["memory_id"] not in exposed:
            conn.execute(
                "DELETE FROM context_recall_items WHERE context_id = ? AND memory_id = ?",
                (retrieval_id, row["memory_id"]),
            )
    conn.execute(
        "UPDATE context_recall_events SET memory_ids_json = ?, count_n = ?, "
        "response_chars = ?, estimated_tokens = ? WHERE context_id = ?",
        (
            json.dumps(sorted(exposed)),
            len(exposed),
            _serialized_chars(data),
            math.ceil(_serialized_chars(data) / 4),
            retrieval_id,
        ),
    )
    conn.commit()


def _public_facets(facets: dict) -> dict:
    """Return a copy of *facets* with internal/bulky keys removed."""
    return {k: v for k, v in facets.items() if k not in _INTERNAL_FACET_KEYS}


def _dedup_episodes(episodes: list[dict]) -> list[dict]:
    """Return *episodes* with exact-content duplicates removed (first wins)."""
    seen: set[str] = set()
    out: list[dict] = []
    for ep in episodes:
        key = ep.get("content_text") or ep.get("content", "")
        if key not in seen:
            seen.add(key)
            out.append(ep)
    return out


async def _bg_record_context_recall(eng, **kwargs):
    """Fire-and-forget background task to record context recall."""
    try:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, lambda: eng.record_context_recall(**kwargs))
    except Exception as e:
        log.warning("_bg_record_context_recall failed: %s", e)


async def _bg_record_retrieval(eng, **kwargs):
    """Fire-and-forget background task to record retrieval."""
    try:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, lambda: eng.record_retrieval(**kwargs))
    except Exception as e:
        log.warning("_bg_record_retrieval failed: %s", e)


def _integration_provenance(ctx: Context) -> dict[str, Any]:
    """Derive source identity from the MCP transport, never model-authored input."""
    client_name = "unknown"
    client_version = None
    try:
        params = getattr(ctx.session, "client_params", None)
        info = getattr(params, "clientInfo", None) or getattr(params, "client_info", None)
        if info is not None:
            client_name = str(getattr(info, "name", None) or "unknown")
            client_version = getattr(info, "version", None)
    except Exception:
        pass
    provenance: dict[str, Any] = {
        "source_kind": "integration",
        "observed": True,
        "integration": client_name,
        "request_id": ctx.request_id,
    }
    if client_version:
        provenance["integration_version"] = str(client_version)
    if ctx.client_id:
        provenance["client_id"] = ctx.client_id
    return provenance


async def _bg_log_event(
    eng,
    session_id: str,
    event_type: str,
    content: str,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Fire-and-forget: log a synthetic session event.

    Always written as ``memory_role="control"``: these events are Slowave's
    own lifecycle operations (activate cue, recall-cue log), which must remain
    auditable raw history but never enter episodic/declarative consolidation.
    """
    try:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: eng.event_append(
                session_id=session_id,
                type=event_type,
                content=content or "[empty]",
                metadata=metadata,
                memory_role="control",
            ),
        )
    except Exception as e:
        log.warning("_bg_log_event failed: %s", e)


def register_tools(mcp: FastMCP, build_engine: Callable) -> None:
    """Register all 5 Slowave cognitive-cycle tools onto *mcp*.

    Args:
        mcp: A FastMCP instance (stdio or HTTP).
        build_engine: Callable(disable_encoder=False) -> SlowaveEngine.
                      Must be the process-local cached version.
    """

    @mcp.tool(name="slowave_activate")
    async def slowave_activate(
        task: str,
        initial_goal: str,
        scope: str,
        ctx: Context,
        continuity_id: str | None = None,
        task_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Prime working memory with relevant context. Opens an implicit session.

        Call this once at the beginning of every task. Spreading activation surfaces
        relevant memories and procedures, and opens a server-side session so you
        never need to call session_start manually.

        The cognitive cycle:
            1. slowave_activate(task, initial_goal, scope)      <- start here
            2. slowave_remember(content, type, scope)           <- for durable facts
            3. slowave_recall(query)                            <- mid-task lookup
            4. slowave_feedback(retrieval_id, feedback, ...)    <- after using memories
            5. slowave_commit(session_id, outcome, ...)         <- close the task

        Args:
            task: verbatim task description (required, nonblank).
            initial_goal: concise action-led provisional objective (required, nonblank).
            scope: required retrieval boundary in ``kind:id`` form.
            continuity_id: omit on the first client-conversation activation;
                retain and resend the returned opaque token unchanged on later
                activations in that conversation. Never invent or reuse it.
            task_context: optional structured facts that condition retrieval.

        Returns:
            retrieval_id: pass to slowave_feedback.
            session_id: required by recall and commit.
            memory_state: cold_start or available for the resolved scope.
            memories: canonical [{memory_id, content, pathway, provenance?}].
            procedures: execution-backed procedures, pending O4 canonicalization.
            warnings: stable structured safety warnings.
            continuity_id: server-issued opaque client-conversation token.
            continuity_state: started on omission, continued on valid reuse.
            retrieval_policy_version: server-selected retrieval-policy identifier.
            more_available: whether a frozen continuation page is available.
            continue_from: opaque continuation cursor, present only when more_available.
            accessible_field: bounded orientation for unreturned candidates;
                contains extra_candidates, kinds, and approx_extra_tokens, never content.
        """
        try:
            if not task.strip():
                raise ValueError("task must be nonblank")
            if not initial_goal.strip():
                raise ValueError("initial_goal must be nonblank")
            _validate_scope(scope)
            provenance = _integration_provenance(ctx)
            eng = build_engine(disable_encoder=False)
            result = ops.activate(
                eng,
                query=task,
                task=task,
                scope=scope,
                initial_goal=initial_goal,
                task_context=task_context,
                continuity_id=continuity_id,
                mode="strict_scope",
                limit=_MCP_ACTIVATE_LIMIT_DEFAULT,
                agent=f"mcp:{provenance['integration']}",
                include_peripheral=False,
                include_schemas=True,
                include_diagnostics=False,
                min_relevance=_MCP_ACTIVATE_MIN_RELEVANCE_DEFAULT,
                manage_continuity=True,
                continuity_integration=str(provenance["integration"]),
            )
            session_resolver.bind(scope, result["session_id"])
            asyncio.create_task(
                _bg_log_event(
                    eng,
                    result["session_id"],
                    "context_query",
                    task,
                    {"provenance": provenance},
                )
            )
            data = _canonical_activation_result(result, scope=scope)
            all_candidates = _activation_candidates(result, scope)
            exposed_ids = {item["memory_id"] for item in data.get("memories", [])}
            exposed_ids.update(item["procedure_id"] for item in data.get("procedures", []))
            _attach_frozen_tail(
                eng,
                data=data,
                candidates=all_candidates,
                session_id=result["session_id"],
                scope=scope,
                exposed_ids=exposed_ids,
            )
            _restrict_activation_exposure(eng, retrieval_id=result["retrieval_id"], data=data)
            return {"ok": True, "data": data}
        except Exception as e:
            log.error("slowave_activate failed: %s", e, exc_info=True)
            return {
                "ok": False,
                "error": {"code": "invalid_input", "message": str(e), "retryable": False},
            }

    @mcp.tool(name="slowave_recall")
    async def slowave_recall(
        session_id: str,
        scope: str,
        ctx: Context,
        query: str | None = None,
        task_context: dict[str, Any] | None = None,
        evidence: str = "references",
        continue_from: str | None = None,
    ) -> dict[str, Any]:
        """Semantic retrieval: bring relevant memories into working memory.
        Use for deliberate mid-task lookups when you need specific historical
        context beyond what activate surfaced.
        Recall is explicitly bound to the active session and matching scope.
        Args:
            query: natural-language query; omit when continuing a frozen result.
            session_id: active session returned by slowave_activate.
            scope: required retrieval boundary; must match the session.
            task_context: optional context update for this sub-question.
            evidence: references (default) or full; budget and policy are server-owned.
            continue_from: opaque cursor returned by activate or recall. The
                cursor is bound to this session and scope and replays a frozen tail.
        Returns:
            retrieval_id: pass to slowave_feedback after using memories.
            memories: canonical direct/associated memories with stable pathways.
            procedures: canonical procedures, including their ID, goal, summary,
                context, steps, caveats, outcome, outcome_summary, created_at,
                aggregate evidence, and recent contributions; no ranking scores.
            evidence: bounded records with evidence_id, source_kind, recorded_at,
                occurred_at, and source_ref; full mode also returns content and
                per-record truncated.
            evidence_mode: the applied references or full mode.
            evidence_truncated: whether evidence records exceeded the response budget.
            more_available: whether a frozen continuation page is available.
            continue_from: opaque continuation cursor, present only when more_available.
            accessible_field: bounded orientation for unreturned candidates;
                contains extra_candidates, kinds, and approx_extra_tokens, never content.
        """
        try:
            _validate_scope(scope)
            if evidence not in {"references", "full"}:
                raise ValueError("evidence must be references or full")
            eng = build_engine()
            if continue_from is not None:
                if query is not None or task_context is not None or evidence != "references":
                    raise ValueError(
                        "continue_from is mutually exclusive with query, task_context, and full evidence"
                    )
                return {
                    "ok": True,
                    "data": _continuation_page(
                        eng,
                        cursor=continue_from,
                        session_id=session_id,
                        scope=scope,
                    ),
                }
            if query is None or not query.strip():
                raise ValueError("query must be nonblank when continue_from is omitted")
            result = ops.recall(
                eng,
                query=query,
                session_id=session_id,
                top_k=_MCP_FROZEN_CANDIDATE_LIMIT,
                evidence=evidence == "full",
                scope=scope,
                mode="strict_scope",
                min_relevance=_MCP_RECALL_MIN_RELEVANCE_DEFAULT,
                task_context=task_context,
            )
            full_data = _canonical_recall_result(result, scope=scope, evidence=evidence)
            data, candidates = _focus_recall_data(full_data)
            exposed_ids = {item["memory_id"] for item in data.get("memories", [])}
            exposed_ids.update(item["procedure_id"] for item in data.get("procedures", []))
            _attach_frozen_tail(
                eng,
                data=data,
                candidates=candidates,
                session_id=session_id,
                scope=scope,
                exposed_ids=exposed_ids,
            )
            _restrict_activation_exposure(eng, retrieval_id=result["retrieval_id"], data=data)
            response = {"ok": True, "data": data}
            asyncio.create_task(
                _bg_log_event(
                    eng,
                    session_id,
                    "trajectory:action",
                    f"slowave_recall: {query}"[:1000],
                    {
                        "status": "succeeded",
                        "provenance": {
                            **_integration_provenance(ctx),
                            "source_kind": "tool",
                        },
                    },
                )
            )
            return response
        except Exception as e:
            log.error("slowave_recall failed: %s", e, exc_info=True)
            return {
                "ok": False,
                "error": {"code": "invalid_input", "message": str(e), "retryable": False},
            }

    @mcp.tool(name="slowave_remember")
    async def slowave_remember(
        scope: str,
        session_id: str,
        ctx: Context,
        content: str | None = None,
        type: str | None = None,
        occurred_at: str | None = None,
        memories: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Explicitly encode a durable typed claim into long-term memory.
        Scalar and batch forms inherit one explicitly verified session and scope.
        Args:
            scope: required scope matching the active session.
            session_id: required active session returned by slowave_activate.
            content: one standalone durable claim; mutually exclusive with memories.
            type: required scalar type. Reusable directions use instruction;
                  only verified commit procedures are execution-backed procedures.
            occurred_at: optional RFC 3339 source-event time, such as
                         `2026-08-26T09:30:00Z`. Use only when the claim records
                         an event that happened at a different time from this MCP
                         call. Slowave always sets internal raw-event `ts` itself
                         to the write time; occurred_at never changes event order.
            memories: strict batch of {content, type, occurred_at?} objects inheriting the
                  outer scope and session.
        IMPORTANT: Use ONLY for durable knowledge that should persist across sessions.
        Do NOT store ephemeral task state — that belongs in session events.

        Returns:
            stored: true when the scalar claim was accepted.
            memory_id: canonical identifier for a scalar stored or matched memory.
            disposition: created or matched for a scalar claim.
            type: confirmed scalar memory type.
            scope: confirmed scalar memory scope.
            source_event_id: source provenance event for a scalar claim.
            results: for batch input, ordered item envelopes with index and
                independent ok/data or ok/error results.
        """
        try:
            eng = build_engine()
            session = (
                eng.db.connect()
                .execute("SELECT scope_id, ended_ts FROM sessions WHERE id = ?", (session_id,))
                .fetchone()
            )
            if session is None:
                raise ValueError(f"unknown session_id: {session_id}")
            if session["ended_ts"] is not None:
                raise ValueError(f"session is already ended: {session_id}")
            if session["scope_id"] != scope:
                raise ValueError("session_id and scope do not match")
            provenance = _integration_provenance(ctx)
            normalized, is_batch = _normalize_remember_inputs(
                content=content, memory_type=type, occurred_at=occurred_at, memories=memories
            )
            if not is_batch:
                item = normalized[0]
                return {
                    "ok": True,
                    "data": ops.remember(
                        eng,
                        content=item["content"],
                        memory_type=item["type"],
                        scope=scope,
                        session_id=session_id,
                        provenance=provenance,
                        occurred_at=item["occurred_at"],
                    ),
                }
            results: list[dict[str, Any]] = []
            for index, item in enumerate(normalized):
                try:
                    results.append(
                        {
                            "index": index,
                            "ok": True,
                            "data": ops.remember(
                                eng,
                                content=item["content"],
                                memory_type=item["type"],
                                scope=scope,
                                session_id=session_id,
                                provenance=provenance,
                                occurred_at=item["occurred_at"],
                            ),
                        }
                    )
                except Exception as e:
                    log.error("slowave_remember (batch item) failed: %s", e, exc_info=True)
                    results.append(
                        {
                            "index": index,
                            "ok": False,
                            "error": {
                                "code": "storage_error",
                                "message": str(e),
                                "retryable": False,
                            },
                        }
                    )
            return {"ok": True, "data": {"results": results}}
        except Exception as e:
            log.error("slowave_remember failed: %s", e, exc_info=True)
            return {
                "ok": False,
                "error": {"code": "invalid_input", "message": str(e), "retryable": False},
            }

    @mcp.tool(name="slowave_feedback")
    async def slowave_feedback(
        ctx: Context,
        retrieval_id: str | None = None,
        memory_feedback: list[dict[str, Any]] | None = None,
        procedure_feedback: list[dict[str, Any]] | None = None,
        retrieval_quality: str | None = None,
        missing: list[str] | None = None,
        coverage: str = "partial",
        items: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Record append-only evidence about retrieved memories and procedures.

        Task outcome does not belong here; slowave_commit owns it. Declarative
        assessments are used|irrelevant|stale. A stale assessment must include
        ``stale_reason`` (contradicted|superseded|outdated|unsupported|withdrawn)
        and a concise ``reason``; superseded additionally requires
        ``replacement_memory_id``. Procedure feedback keeps
        use (used|not_used) separate from effect
        (helped|no_effect|harmed|unknown), with contribution required when used.

        Args:
            retrieval_id: opaque ID returned by activate/recall.
            memory_feedback: [{memory_id, assessment, stale_reason?, replacement_memory_id?, reason?}].
            procedure_feedback: [{procedure_id, use, effect, contribution?, reason?}].
            retrieval_quality: optional whole-result quality assessment.
            missing: optional descriptions of expected but absent knowledge.
            coverage: partial or complete; silence under partial is not negative.
            items: batch of records with the same fields. Scalar feedback fields
                   and items are mutually exclusive.
        Returns:
            retrieval_id: the assessed scalar retrieval.
            coverage: applied partial or complete coverage.
            outstanding: memory_ids and procedure_ids still requiring assessment.
            accepted_event_ids: append-only feedback events accepted by the server.
            rejected: feedback targets or shapes the server did not apply, with reasons.
            applied: IDs grouped by the feedback effect recorded by the server.
            results: for batch input, ordered item envelopes with independent
                ok/data or ok/error results.
        """
        # Feedback follows activate/recall in the public lifecycle, so the
        # encoder-enabled engine is already warm.  Reusing it avoids building
        # a second SQLite/FAISS engine just for this short write operation.
        eng = build_engine(disable_encoder=False)
        if items is not None:
            if (
                any(
                    value is not None
                    for value in (
                        retrieval_id,
                        memory_feedback,
                        procedure_feedback,
                        retrieval_quality,
                        missing,
                    )
                )
                or coverage != "partial"
            ):
                return {
                    "ok": False,
                    "error": {
                        "code": "invalid_input",
                        "message": "items is mutually exclusive with scalar feedback fields",
                        "retryable": False,
                    },
                }
            results: list[dict[str, Any]] = []
            for item in items:
                item_rid = item.get("retrieval_id", "")
                try:
                    results.append(
                        {
                            "ok": True,
                            "data": ops.feedback(
                                eng,
                                retrieval_id=item_rid,
                                memory_feedback=item.get("memory_feedback"),
                                procedure_feedback=item.get("procedure_feedback"),
                                retrieval_quality=item.get("retrieval_quality"),
                                missing=item.get("missing"),
                                coverage=item.get("coverage", "partial"),
                            ),
                        }
                    )
                except Exception as e:
                    log.error("slowave_feedback (batch item) failed: %s", e, exc_info=True)
                    results.append(
                        {
                            "ok": False,
                            "error": {
                                "code": "invalid_input",
                                "message": str(e),
                                "retryable": False,
                            },
                        }
                    )
            return {"ok": True, "data": {"results": results}}
        try:
            if not retrieval_id:
                raise ValueError("retrieval_id is required")
            data = ops.feedback(
                eng,
                retrieval_id=retrieval_id,
                memory_feedback=memory_feedback,
                procedure_feedback=procedure_feedback,
                retrieval_quality=retrieval_quality,
                missing=missing,
                coverage=coverage,
            )
            return {"ok": True, "data": data}
        except Exception as e:
            log.error("slowave_feedback failed: %s", e, exc_info=True)
            code = "not_found" if "unknown retrieval_id" in str(e) else "invalid_input"
            return {"ok": False, "error": {"code": code, "message": str(e), "retryable": False}}

    @mcp.tool(name="slowave_commit")
    async def slowave_commit(
        session_id: str,
        final_goal: str,
        outcome: str,
        outcome_summary: str,
        verification: dict[str, Any],
        ctx: Context,
        procedure: dict[str, Any] | None = None,
        trajectory: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Close the current task and trigger offline memory consolidation.

        Call at the end of every task. If skipped, the idle-session reaper closes
        the session after SLOWAVE_SESSION_IDLE_TIMEOUT seconds (default 3600).
        Args:
            session_id: required active session from activate.
            final_goal: required confirmed goal.
            outcome: required success|partial|failure.
            outcome_summary: required standalone actual result.
            verification: required status, summary, and optional evidence refs.
            procedure: executed reusable method when one was attempted.
            trajectory: optional executed-attempt trace of at most 32 action/observation
                entries. Must contain TASK actions/observations only. Do NOT include
                Slowave lifecycle bookkeeping (activate/recall/feedback/commit calls,
                "Activated the Slowave session." etc.) -- the server filters those out
                and reports the count as trajectory_lifecycle_filtered. If you have no
                task-level actions, omit the trajectory.
        Returns:
            session_id: the session that was closed.
            episodes_formed: number of episodic memories created.
            feedback_status: complete for normal closure.
            committed: true when the session outcome was recorded.
            outcome: confirmed success, partial, or failure outcome.
            verification_status: confirmed verification status.
            operation: closed for a new close or updated for an already-ended session.
            trajectory_lifecycle_filtered: number of lifecycle entries removed,
                when any were filtered.
            already_ended: true when the session had already been closed.
        """
        try:
            # O10 trajectories must be embedded so session_end can form episodic
            # memories from the attempted path; an encoder-free commit would
            # preserve rows for audit but silently exclude them from consolidation.
            eng = build_engine(disable_encoder=False)
            result = ops.commit(
                eng,
                session_id=session_id,
                outcome=outcome,
                final_goal=final_goal,
                outcome_summary=outcome_summary,
                procedure=procedure,
                verification=verification,
                trajectory=trajectory,
                provenance=_integration_provenance(ctx),
                enforce_feedback=True,
            )
            return {"ok": True, "data": result}
        except ops.IncompleteFeedbackError as e:
            return {
                "ok": False,
                "error": {
                    "code": "incomplete_feedback",
                    "message": str(e),
                    "retryable": True,
                    "outstanding": e.outstanding,
                },
            }
        except Exception as e:
            log.error("slowave_commit failed: %s", e, exc_info=True)
            return {
                "ok": False,
                "error": {"code": "invalid_input", "message": str(e), "retryable": False},
            }
