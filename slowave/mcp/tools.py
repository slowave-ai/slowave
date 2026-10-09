"""Shared MCP tool registration for Slowave.

Provides a single ``register_tools(mcp, build_engine)`` function that attaches
all 5 cognitive lifecycle tools to any FastMCP instance.  Both the stdio server (server.py) and
the HTTP daemon (http_server.py) call this function so there is no
duplication of tool logic.

Tools registered (5 cognitive-cycle verbs):
  activate, remember, recall, feedback, commit

remember accepts `memories` and feedback accepts `items` for
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
from typing import Annotated, Any, Callable, Literal

from mcp.server.fastmcp import Context, FastMCP
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    ValidationError,
    field_validator,
    model_validator,
)

import slowave.ops as ops
from slowave.mcp import session_resolver
from slowave.mcp.activation_catalog import (
    DEFAULT_MEMORY_PAGE_SIZE,
    MEMORY_PAGE_SIZE,
    SUPPORTED_POLICIES,
    CatalogCompatibilityError,
    FrozenActivationCatalog,
)

log = logging.getLogger(__name__)

# Keep references to fire-and-forget background logging tasks so the event
# loop cannot garbage-collect them mid-flight.
_bg_tasks: set[asyncio.Task] = set()


def _spawn_bg_task(coro: Any) -> None:
    task = asyncio.create_task(coro)
    _bg_tasks.add(task)
    task.add_done_callback(_bg_tasks.discard)


# Keys stored in schema facets that are internal to the retrieval engine.
_INTERNAL_FACET_KEYS: frozenset[str] = frozenset({"vsa_vec"})

# Phase-1 trustworthy-memory dogfood defaults (2026-08-08). These are scoped
# to the MCP product surface; library/benchmark callers retain their existing
# defaults, including Recall@20. Calibrated on the frozen 60-call live replay.
_MCP_ACTIVATE_LIMIT_DEFAULT = 2
_MCP_ACTIVATE_LIMIT_MAX = 3
_MCP_RECALL_TOP_K_DEFAULT = 2
_MCP_FROZEN_CANDIDATE_LIMIT = _MCP_RECALL_TOP_K_DEFAULT + 8
# `relevant-set-v2` builds the complete discovered catalog before paging it.
# This is a page target, not a relevance or catalog limit.
_MCP_CONTINUATION_PAGE_SIZE = DEFAULT_MEMORY_PAGE_SIZE
_MCP_FIELD_RESPONSE_CHARS = 160
_MCP_MEMORY_CONTENT_LIMIT = 500
_MCP_CONTINUITY_START_RESPONSE_CHARS = 3000
_MCP_CONTINUATION_RESPONSE_CHARS = 3000
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


# ---------------------------------------------------------------------------
# Public MCP commit contract
# ---------------------------------------------------------------------------
#
# These models deliberately live at the transport boundary rather than leaving
# FastMCP to infer ``dict[str, Any]``.  The latter renders as an unconstrained
# JSON object in tools/list even though ops.commit() has always enforced a
# strict nested contract.  Keep these models aligned with the normalizers in
# ``slowave.ops`` and ``slowave.symbolic.procedural_memory``; the contract test
# asserts that the MCP schema remains explicit.


class _StrictCommitModel(BaseModel):
    """Strict JSON-object base for the public commit payload."""

    model_config = ConfigDict(extra="forbid")


class _StrictMCPModel(BaseModel):
    """Strict transport object base for every public MCP request."""

    model_config = ConfigDict(extra="forbid")


RememberType = Literal[
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
]


class CommitVerification(_StrictCommitModel):
    """Evidence that supports the reported task outcome."""

    status: Literal["verified", "partially_verified", "unverified"]
    summary: Annotated[str, Field(min_length=1)]
    evidence_refs: list[str] = Field(default_factory=list)

    @field_validator("summary")
    @classmethod
    def _nonblank_summary(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("summary must be nonblank")
        return value


class CommitProcedureStep(_StrictCommitModel):
    """One ordered, human-readable action in a reusable procedure."""

    summary: Annotated[str, Field(min_length=1)]

    @field_validator("summary")
    @classmethod
    def _nonblank_summary(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("summary must be nonblank")
        return value


class CommitProcedure(_StrictCommitModel):
    """A reusable multi-step method that was actually attempted."""

    version: Literal[2] = 2
    summary: Annotated[str, Field(min_length=1)]
    context: dict[str, JsonValue] = Field(default_factory=dict)
    steps: Annotated[list[CommitProcedureStep], Field(min_length=1)]
    caveats: list[str] = Field(default_factory=list)
    # Optional producer assertion that multiple procedures are alternate
    # renderings of one method. It is intentionally not inferred from text.
    deduplication_key: Annotated[
        str | None,
        Field(pattern=r"^[a-z][a-z0-9_.:-]{0,63}$"),
    ] = None

    @field_validator("summary")
    @classmethod
    def _nonblank_summary(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("summary must be nonblank")
        return value

    @field_validator("caveats")
    @classmethod
    def _nonblank_caveats(cls, values: list[str]) -> list[str]:
        normalized = [value.strip() for value in values]
        if any(not value for value in normalized):
            raise ValueError("caveat entries must be nonblank")
        return normalized


class CommitTrajectoryEntry(_StrictCommitModel):
    """One task-level action or observation; never lifecycle bookkeeping."""

    kind: Literal["action", "observation"]
    summary: Annotated[str, Field(min_length=1, max_length=1000)]
    status: Literal["started", "succeeded", "failed", "unknown"] = "unknown"

    @field_validator("summary")
    @classmethod
    def _nonblank_summary(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("summary must be nonblank")
        return value


class CommitArguments(_StrictCommitModel):
    """Canonical public request contract for ``slowave_commit``."""

    session_id: Annotated[str, Field(min_length=1)]
    final_goal: Annotated[str, Field(min_length=1)]
    outcome: Literal["success", "partial", "failure"]
    outcome_summary: Annotated[str, Field(min_length=1)]
    verification: CommitVerification
    procedure: CommitProcedure | None = None
    trajectory: Annotated[list[CommitTrajectoryEntry] | None, Field(max_length=32)] = None

    @field_validator("session_id", "final_goal", "outcome_summary")
    @classmethod
    def _nonblank_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must be nonblank")
        return value


PageSize = Annotated[
    int,
    Field(
        strict=True,
        ge=1,
        le=10,
        description=(
            f"Maximum memories per page (default {DEFAULT_MEMORY_PAGE_SIZE}); "
            "continuations inherit this bound. Procedures have a separate limit."
        ),
    ),
]


class ActivateArguments(_StrictMCPModel):
    page_size: PageSize = DEFAULT_MEMORY_PAGE_SIZE
    task: Annotated[str, Field(min_length=1)]
    initial_goal: Annotated[str, Field(min_length=1)]
    scope: Annotated[str, Field(min_length=3)]
    continuity_id: str | None = None
    task_context: dict[str, JsonValue] | None = None
    semantic_context: str | None = None


class RecallArguments(_StrictMCPModel):
    page_size: PageSize | None = None
    session_id: Annotated[str, Field(min_length=1)]
    scope: Annotated[str, Field(min_length=3)]
    query: str | None = None
    task_context: dict[str, JsonValue] | None = None
    semantic_context: str | None = None
    evidence: Literal["references", "full"] = "references"
    continue_from: str | None = None

    @model_validator(mode="after")
    def _validate_mode(self) -> "RecallArguments":
        if self.continue_from is not None:
            if (
                self.query is not None
                or self.page_size is not None
                or self.task_context is not None
                or self.semantic_context is not None
                or self.evidence != "references"
            ):
                raise ValueError(
                    "continue_from is mutually exclusive with query, task_context, page_size, and full evidence"
                )
        elif not self.query or not self.query.strip():
            raise ValueError("query must be nonblank when continue_from is omitted")
        return self


class RememberEntry(_StrictMCPModel):
    content: Annotated[str, Field(min_length=1)]
    type: RememberType
    occurred_at: str | None = None


class RememberArguments(_StrictMCPModel):
    scope: Annotated[str, Field(min_length=3)]
    session_id: Annotated[str, Field(min_length=1)]
    content: str | None = None
    type: RememberType | None = None
    occurred_at: str | None = None
    memories: Annotated[list[RememberEntry] | None, Field(min_length=1)] = None

    @model_validator(mode="after")
    def _validate_shape(self) -> "RememberArguments":
        if self.memories is not None:
            if self.content is not None or self.type is not None:
                raise ValueError("memories is mutually exclusive with content and type")
        elif not self.content or not self.content.strip() or self.type is None:
            raise ValueError("content and type are required when memories is omitted")
        return self


# Advertise enum choices while retaining string validation in the handler.
# Invalid semantic labels remain per-target rejections, preserving valid siblings.
def _feedback_label(values: list[str], description: str, *, nullable: bool = False) -> Any:
    choices: list[JsonValue] = list(values)
    if nullable:
        choices.append(None)
    return Field(description=description, json_schema_extra={"enum": choices})


class MemoryFeedbackEntry(_StrictMCPModel):
    memory_id: Annotated[
        str, Field(min_length=1, description="Assessed memory ID exposed by this retrieval.")
    ]
    assessment: Annotated[
        str,
        Field(min_length=1),
        _feedback_label(
            ["used", "not_used", "unassessable", "irrelevant", "already_known", "stale"],
            "used requires observed influence on reasoning/action/check/constraint, not reading or topical similarity. not_used means no influence; irrelevant means task mismatch; unassessable means lost/uncertain usage evidence and requires reason. stale requires stale_reason and reason.",
        ),
    ]
    relevance: Annotated[
        str | None,
        _feedback_label(
            ["relevant", "irrelevant", "uncertain"],
            "Optional relevance, independent of usage.",
            nullable=True,
        ),
    ] = None
    effect: Annotated[
        str | None,
        _feedback_label(
            ["helped", "no_effect", "harmed", "unknown"],
            "used: helped/no_effect/harmed; omitted means helped. Other assessments: omit or unknown.",
            nullable=True,
        ),
    ] = None
    stale_reason: Annotated[
        str | None,
        _feedback_label(
            ["contradicted", "superseded", "outdated", "unsupported", "withdrawn"],
            "Only for stale. superseded also requires replacement_memory_id.",
            nullable=True,
        ),
    ] = None
    replacement_memory_id: str | None = Field(
        default=None,
        description="For stale: different active memory in the retrieval scope; need not be exposed.",
    )
    reason: str | None = Field(
        default=None, description="Nonblank explanation required for stale or unassessable."
    )


class ProcedureFeedbackEntry(_StrictMCPModel):
    procedure_id: Annotated[
        str, Field(min_length=1, description="Procedure ID exposed by this retrieval.")
    ]
    use: Annotated[
        str,
        Field(min_length=1),
        _feedback_label(
            ["used", "not_used", "unassessable"],
            "used requires nonblank contribution; not_used/unassessable forbid contribution; unassessable requires reason.",
        ),
    ]
    effect: Annotated[
        str | None,
        _feedback_label(
            ["helped", "no_effect", "harmed", "unknown"],
            "Omitted means unknown. not_used permits only omitted or unknown.",
            nullable=True,
        ),
    ] = None
    contribution: str | None = Field(
        default=None,
        description="Nonblank description of how the procedure contributed; required when used, omit when not_used.",
    )
    reason: str | None = None


class FeedbackItem(_StrictMCPModel):
    retrieval_id: Annotated[str, Field(min_length=1)]
    memory_feedback: list[MemoryFeedbackEntry] | None = None
    procedure_feedback: list[ProcedureFeedbackEntry] | None = None
    retrieval_quality: str | None = None
    missing: list[str] | None = None
    coverage: Literal["partial", "complete"] = "partial"


class FeedbackArguments(_StrictMCPModel):
    retrieval_id: str | None = None
    memory_feedback: list[MemoryFeedbackEntry] | None = None
    procedure_feedback: list[ProcedureFeedbackEntry] | None = None
    retrieval_quality: str | None = None
    missing: list[str] | None = None
    coverage: Literal["partial", "complete"] = "partial"
    items: Annotated[list[FeedbackItem] | None, Field(min_length=1)] = None

    @model_validator(mode="after")
    def _validate_shape(self) -> "FeedbackArguments":
        if self.items is not None:
            if (
                any(
                    value is not None
                    for value in (
                        self.retrieval_id,
                        self.memory_feedback,
                        self.procedure_feedback,
                        self.retrieval_quality,
                        self.missing,
                    )
                )
                or self.coverage != "partial"
            ):
                raise ValueError("items is mutually exclusive with scalar feedback fields")
        elif not self.retrieval_id or not self.retrieval_id.strip():
            raise ValueError("retrieval_id is required")
        return self


def _format_validation_error(exc: ValidationError) -> tuple[str, list[dict[str, str]]]:
    """Return stable, actionable client errors from the canonical model."""

    field_errors: list[dict[str, str]] = []
    for error in exc.errors(include_url=False):
        path = "".join(
            f"[{part}]" if isinstance(part, int) else ("." if index else "") + str(part)
            for index, part in enumerate(error["loc"])
        )
        field_errors.append({"path": path, "message": error["msg"]})
    message = "; ".join(f"{item['path']}: {item['message']}" for item in field_errors)
    return message, field_errors


def _publish_schema(mcp: FastMCP, name: str, model: type[BaseModel]) -> None:
    """Publish a canonical schema while retaining handler-owned errors.

    FastMCP 1.x derives an input schema from a callback's annotations and then
    validates those annotations before invoking the callback.  For this tool,
    that would turn malformed input into an unstructured framework error and
    bypass Slowave's ``{ok: false, error: ...}`` contract.  The callback is
    therefore deliberately permissive and validates its canonical model
    itself; this small, version-pinned adapter replaces only the advertised
    schema.  Keep the private FastMCP access here, with transport tests.
    """

    tool = mcp._tool_manager.get_tool(name)
    if tool is None:  # pragma: no cover - registration immediately precedes this call.
        raise RuntimeError(f"{name} was not registered")
    tool.parameters = model.model_json_schema()


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
    activation_cursor = _read_bound_activation_cursor(
        eng, cursor=cursor, session_id=session_id, scope=scope
    )
    if activation_cursor is not None:
        row, payload = activation_cursor
        return _complementary_continuation_page(eng, cursor=cursor, row=row, payload=payload)
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
        page_size=max(1, len(tail)),
    )
    data["more_available"] = cursor is not None
    if cursor is not None:
        data["continue_from"] = cursor
    else:
        data.pop("continue_from", None)
    data["accessible_field"] = _accessible_field(tail)


def _authorize_continuation_exposure(
    eng: Any, *, retrieval_id: str, data: dict[str, Any], commit: bool = True
) -> None:
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
    if commit:
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


def _activation_candidates(
    result: dict[str, Any], scope: str, *, preserve_full_source: bool = False
) -> list[dict[str, Any]]:
    memories = []
    for item in result.get("schemas", []):
        memory: dict[str, Any] = {
            "memory_id": item["id"],
            "content": (
                str(item.get("text") or "")
                if preserve_full_source
                else str(item.get("text") or "")[:_MCP_MEMORY_CONTENT_LIMIT]
            ),
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


def _complementary_cursor_payload(catalog: FrozenActivationCatalog, offset: tuple[int, int]) -> str:
    payload = json.loads(catalog.snapshot_json)
    payload["page_offset"] = {"memories": int(offset[0]), "procedures": int(offset[1])}
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _store_complementary_cursor(
    conn: Any,
    *,
    catalog: FrozenActivationCatalog,
    offset: tuple[int, int],
    cursor_id: str | None = None,
) -> str:
    snapshot = json.loads(catalog.snapshot_json)
    retrieval_id, session_id, scope = (
        snapshot["retrieval_id"],
        snapshot["session_id"],
        snapshot["scope"],
    )
    cursor = cursor_id or "cur_" + secrets.token_urlsafe(24)
    conn.execute(
        "INSERT OR IGNORE INTO retrieval_continuations "
        "(cursor_id,retrieval_id,session_id,scope_id,candidates_json,offset_n,page_size,created_at) "
        "VALUES (?,?,?,?,?,?,?,?)",
        (
            cursor,
            retrieval_id,
            session_id,
            scope,
            _complementary_cursor_payload(catalog, offset),
            offset[0],
            MEMORY_PAGE_SIZE,
            int(time.time()),
        ),
    )
    return cursor


def _read_bound_activation_cursor(
    eng: Any, *, cursor: str, session_id: str, scope: str
) -> tuple[Any, dict[str, Any]] | None:
    conn = eng.db.connect()
    row = conn.execute(
        "SELECT retrieval_id,session_id,scope_id,candidates_json FROM retrieval_continuations "
        "WHERE cursor_id=?",
        (cursor,),
    ).fetchone()
    if row is None:
        return None
    if row["session_id"] != session_id or row["scope_id"] != scope:
        raise ValueError("continue_from cursor does not match session_id and scope")
    active = conn.execute(
        "SELECT ended_ts FROM sessions WHERE id=? AND scope_id=?", (session_id, scope)
    ).fetchone()
    if active is None:
        raise ValueError("continue_from cursor session no longer exists")
    if active["ended_ts"] is not None:
        raise ValueError("continue_from cursor session is already ended")
    payload = json.loads(row["candidates_json"])
    if (
        isinstance(payload, dict)
        and payload.get("originating_policy_version") in SUPPORTED_POLICIES
    ):
        return row, payload
    return None


def _complementary_continuation_page(
    eng: Any, *, cursor: str, row: Any, payload: dict[str, Any]
) -> dict[str, Any]:
    offset_data = payload.get("page_offset") or {}
    offset = (int(offset_data.get("memories", 0)), int(offset_data.get("procedures", 0)))
    frozen = dict(payload)
    frozen.pop("page_offset", None)
    catalog = FrozenActivationCatalog.read(
        json.dumps(frozen, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    )
    data, successor = catalog.page(*offset)
    next_cursor = None
    if successor is not None:
        next_cursor = (
            "cur_"
            + hashlib.sha256(
                f"{cursor}:{row['retrieval_id']}:{successor[0]}:{successor[1]}".encode("utf-8")
            ).hexdigest()[:32]
        )
        data["continue_from"] = next_cursor
    _bounded_activation_response(data, has_cursor=next_cursor is not None)
    conn = eng.db.connect()
    try:
        with conn:
            if next_cursor:
                if successor is None:  # pragma: no cover - guarded by next_cursor
                    raise RuntimeError("cursor successor missing")
                _store_complementary_cursor(
                    conn, catalog=catalog, offset=successor, cursor_id=next_cursor
                )
            _authorize_continuation_exposure(
                eng, retrieval_id=str(row["retrieval_id"]), data=data, commit=False
            )
    except Exception:
        conn.rollback()
        raise
    return data


def _bounded_activation_response(data: dict[str, Any], *, has_cursor: bool) -> None:
    # Explicit activation catalog envelope, independent of the recall budget.
    from slowave.mcp.activation_catalog import MAX_PAGE_CHARS

    value = {"ok": True, "data": data}
    if has_cursor and "continue_from" not in data:
        data["continue_from"] = "cur_" + "x" * 64
    if _serialized_chars(value) > MAX_PAGE_CHARS:
        raise ValueError("unsupported oversized activation page")


def _compensate_failed_activation(
    eng: Any,
    *,
    result: dict[str, Any],
    scope: str,
    supplied_continuity: str | None,
    prior_continuity: Any,
    prior_scope_registry: Any,
) -> None:
    """Remove prepared activation rows after response/publication failure."""
    conn = eng.db.connect()
    with conn:
        conn.execute(
            "DELETE FROM context_recall_events WHERE context_id=?", (result["retrieval_id"],)
        )
        conn.execute("DELETE FROM sessions WHERE id=?", (result["session_id"],))
        continuity_id = result.get("continuity_id")
        if continuity_id:
            if prior_continuity is None:
                conn.execute("DELETE FROM continuities WHERE continuity_id=?", (continuity_id,))
            else:
                conn.execute(
                    "UPDATE continuities SET last_seen_at=? WHERE continuity_id=?",
                    (prior_continuity["last_seen_at"], continuity_id),
                )
        if prior_scope_registry is None:
            conn.execute("DELETE FROM scope_registry WHERE scope_id=?", (scope,))
        else:
            conn.execute(
                "UPDATE scope_registry SET scope_kind=?,first_seen_ts=?,last_active_ts=?,"
                "session_count=?,recall_count=? WHERE scope_id=?",
                (
                    prior_scope_registry["scope_kind"],
                    prior_scope_registry["first_seen_ts"],
                    prior_scope_registry["last_active_ts"],
                    prior_scope_registry["session_count"],
                    prior_scope_registry["recall_count"],
                    scope,
                ),
            )


def _prepare_complementary_activation(
    eng: Any,
    *,
    result: dict[str, Any],
    scope: str,
    page_size: int = DEFAULT_MEMORY_PAGE_SIZE,
) -> tuple[dict[str, Any], FrozenActivationCatalog, tuple[int, int] | None]:
    candidates = _activation_candidates(result, scope, preserve_full_source=True)
    memories = [item["value"] for item in candidates if item["kind"] == "memory"]
    procedures = [item["value"] for item in candidates if item["kind"] == "procedure"]
    warnings = []
    if result.get("scope_warning"):
        warnings.append({"code": "scope_fragmentation", "message": result["scope_warning"]})
    metadata: dict[str, Any] = {
        "memory_state": "cold_start" if result.get("cold_start") else "available",
        "warnings": warnings,
        "continuity_id": result["continuity_id"],
        "continuity_state": result["continuity_state"],
    }
    if result.get("applicability_status") == "unavailable":
        warnings.append(
            {
                "code": "applicability_unavailable",
                "message": "Local applicability model unavailable; lexical and semantic fallback used.",
            }
        )
    spans = {
        memory_id: tuple(tuple(span) for span in values)
        for memory_id, values in (result.get("contribution_spans") or {}).items()
    }
    catalog = FrozenActivationCatalog.prepare(
        retrieval_id=result["retrieval_id"],
        session_id=result["session_id"],
        scope=scope,
        memories=memories,
        procedures=procedures,
        contribution_spans=spans,
        catalog_truncated=bool(result.get("catalog_truncated")),
        first_page_metadata=metadata,
        page_size=page_size,
        policy_version=result.get("retrieval_policy_version", "activation-complementary-v1"),
    )
    data, successor = catalog.page()
    if result.get("retrieval_policy_version") != data["retrieval_policy_version"]:
        raise CatalogCompatibilityError("prepared activation policy does not match serving policy")
    return data, catalog, successor


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
    if "relevant_total" in result:
        data["relevant_total"] = result["relevant_total"]
        data["catalog_truncated"] = bool(result.get("catalog_truncated"))

    budget = (
        _MCP_CONTINUITY_START_RESPONSE_CHARS
        if result.get("continuity_state") == "started"
        else _MCP_CONTINUATION_RESPONSE_CHARS
    ) - 50  # reserve an opaque continue_from cursor on the final envelope
    # Core memory, procedure outcome/safety, then reinstatement context.
    omitted = False
    admitted_memories = 0
    for envelope in candidates:
        kind, candidate = envelope["kind"], envelope["value"]
        if kind == "memory" and admitted_memories >= _MCP_CONTINUATION_PAGE_SIZE:
            omitted = True
            continue
        field = "memories" if kind == "memory" else "procedures"
        data[field].append(candidate)
        if _serialized_chars(data) > budget:
            data[field].pop()
            omitted = True
        elif kind == "memory":
            admitted_memories += 1
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
        for key in ("used", "not_used", "unassessable", "helped", "no_effect", "harmed", "unknown")
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
                "content": (
                    str(item.get("content_text") or "")
                    if item.get("preview_prepared")
                    or result.get("retrieval_policy_version")
                    == "multilingual-retrieval-baseline-v1"
                    else str(item.get("content_text") or "")[:_MCP_MEMORY_CONTENT_LIMIT]
                ),
                "pathway": pathway,
            }
            if item.get("excerpt"):
                provenance["excerpt"] = item["excerpt"]
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

    data = {
        "retrieval_id": result["retrieval_id"],
        "memories": memories,
        "procedures": [_canonical_procedure(item, scope) for item in result.get("procedures", [])],
        "evidence": evidence_records,
        "evidence_mode": evidence,
        "evidence_truncated": len(raw_records) > _MCP_EVIDENCE_LIMIT,
    }
    for key in ("retrieval_policy_version", "relevant_total", "catalog_truncated"):
        if key in result:
            data[key] = result[key]
    if result.get("applicability_status") == "unavailable":
        data["warnings"] = [
            {
                "code": "applicability_unavailable",
                "message": "Local applicability model unavailable; lexical and semantic fallback used.",
            }
        ]
    return data


def _focus_recall_data(data: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Pack actual admitted context under the serialized response ceiling."""
    candidates = _canonical_candidates(data)
    focused = dict(data)
    focused["memories"] = []
    focused["procedures"] = []
    # Reserve continuation metadata before packing by serialized payload.
    focused["more_available"] = True
    focused["continue_from"] = "cur_" + "x" * 32
    for item in candidates:
        key = "memories" if item["kind"] == "memory" else "procedures"
        focused[key].append(item["value"])
        if _serialized_chars(focused) > _MCP_CONTINUATION_RESPONSE_CHARS:
            focused[key].pop()
            continue
    return focused, candidates


def _restrict_activation_exposure(
    eng: Any, *, retrieval_id: str, data: dict[str, Any], commit: bool = True
) -> None:
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
    if commit:
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
        ctx: Context,
        page_size: Any = DEFAULT_MEMORY_PAGE_SIZE,
        task: Any = None,
        initial_goal: Any = None,
        scope: Any = None,
        continuity_id: Any = None,
        task_context: Any = None,
        semantic_context: Any = None,
    ) -> dict[str, Any]:
        """Prime working memory with relevant context. Opens an implicit session.

            Call this once at the beginning of every task. Spreading activation surfaces
            relevant memories and procedures, and opens a server-side session so you
            never need to call session_start manually.
            Retrieval considers meaning and specific terms, then selects context
            applicable to the task. The bounded result is not a complete inventory
            or a verification of the returned claims; assess their actual use.

            The cognitive cycle:
                1. slowave_activate(task, initial_goal, scope)      <- start here
                2. slowave_remember(session_id, scope, content, type)
                   <- when you discover new knowledge worth retaining as durable memory
                      because it could help you with future tasks
                3. slowave_recall(session_id, scope, query)                            <- mid-task lookup
                4. slowave_feedback(retrieval_id, memory_feedback, procedure_feedback, coverage)    <- after using memories
                5. slowave_commit(session_id, final_goal, outcome, outcome_summary, verification)         <- close the task

            Args:
                task: verbatim task description (required, nonblank).
                initial_goal: concise action-led provisional objective (required, nonblank).
                scope: required retrieval boundary in ``kind:id`` form.
                continuity_id: omit on the first client-conversation activation;
                    retain and resend the returned opaque token unchanged on later
                    activations in that conversation. Never invent it or reuse it across different client conversations.
                page_size: optional strict integer 1–10; omitted values use the server default.
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
        relevant_total: number of discovered relevance-qualified declarative
            memories in a relevant-set response.
        catalog_truncated: true when the bounded candidate search could have
            omitted qualifying memories; false does not include procedure results.
                more_available: whether a frozen continuation page is available.
                continue_from: opaque continuation cursor, present only when more_available.
        accessible_field: bounded orientation for unreturned candidates;
                    contains extra_candidates, kinds, and approx_extra_tokens, never content.
        """
        try:
            request = ActivateArguments.model_validate(
                {
                    "page_size": page_size,
                    "task": task,
                    "initial_goal": initial_goal,
                    "scope": scope,
                    "continuity_id": continuity_id,
                    "task_context": task_context,
                    "semantic_context": semantic_context,
                }
            )
        except ValidationError as exc:
            message, field_errors = _format_validation_error(exc)
            return {
                "ok": False,
                "error": {
                    "code": "invalid_input",
                    "message": message,
                    "retryable": False,
                    "field_errors": field_errors,
                },
            }
        published_result: dict[str, Any] | None = None
        prior_resolver_binding: str | None = None
        prior_continuity: Any = None
        prior_scope_registry: Any = None
        prepared_eng: Any = None
        try:
            _validate_scope(request.scope)
            provenance = _integration_provenance(ctx)
            eng = build_engine(disable_encoder=False)
            prepared_eng = eng
            prior_resolver_binding = (
                session_resolver.snapshot().get(request.scope, {}).get("session_id")
            )
            conn = eng.db.connect()
            prior_continuity = (
                conn.execute(
                    "SELECT * FROM continuities WHERE continuity_id=?",
                    (request.continuity_id,),
                ).fetchone()
                if request.continuity_id
                else None
            )
            prior_scope_registry = conn.execute(
                "SELECT * FROM scope_registry WHERE scope_id=?", (request.scope,)
            ).fetchone()
            result = ops.activate(
                eng,
                query=request.task,
                task=request.task,
                scope=request.scope,
                initial_goal=request.initial_goal,
                task_context=request.task_context,
                semantic_context=request.semantic_context,
                continuity_id=request.continuity_id,
                mode="strict_scope",
                limit=ops.task_facet_limit(
                    request.task,
                    base_limit=_MCP_ACTIVATE_LIMIT_DEFAULT,
                    max_limit=_MCP_ACTIVATE_LIMIT_MAX,
                ),
                agent=f"mcp:{provenance['integration']}",
                include_peripheral=False,
                include_schemas=True,
                include_diagnostics=False,
                manage_continuity=True,
                continuity_integration=str(provenance["integration"]),
                relevant_set=True,
                complementary_activation=True,
            )
            published_result = result
            data, frozen_catalog, successor = _prepare_complementary_activation(
                eng, result=result, scope=request.scope, page_size=request.page_size
            )
            next_cursor = None
            if successor is not None:
                next_cursor = "cur_" + secrets.token_urlsafe(24)
                data["continue_from"] = next_cursor
            _bounded_activation_response(data, has_cursor=next_cursor is not None)
            # Cursor creation, first-page exposure and the implicit resolver
            # binding publish in one transaction; on failure the prior
            # binding is restored below before yielding.
            conn = eng.db.connect()
            try:
                with conn:
                    if next_cursor:
                        if successor is None:  # pragma: no cover - guarded above
                            raise RuntimeError("cursor successor missing")
                        _store_complementary_cursor(
                            conn,
                            catalog=frozen_catalog,
                            offset=successor,
                            cursor_id=next_cursor,
                        )
                    conn.execute(
                        "UPDATE context_recall_events SET requested_page_size=? "
                        "WHERE context_id=?",
                        (request.page_size, result["retrieval_id"]),
                    )
                    session_resolver.bind(request.scope, result["session_id"])
                    _restrict_activation_exposure(
                        eng, retrieval_id=result["retrieval_id"], data=data, commit=False
                    )
            except Exception:
                session_resolver.clear(request.scope)
                if prior_resolver_binding:
                    session_resolver.bind(request.scope, prior_resolver_binding)
                conn.rollback()
                raise
            _spawn_bg_task(
                _bg_log_event(
                    eng,
                    result["session_id"],
                    "context_query",
                    request.task,
                    {"provenance": provenance},
                )
            )
            return {"ok": True, "data": data}
        except Exception as e:
            if prepared_eng is not None and published_result is not None:
                try:
                    _compensate_failed_activation(
                        prepared_eng,
                        result=published_result,
                        scope=request.scope,
                        supplied_continuity=request.continuity_id,
                        prior_continuity=prior_continuity,
                        prior_scope_registry=prior_scope_registry,
                    )
                except Exception:
                    log.exception("failed to compensate unpublished activation state")
                session_resolver.clear(request.scope)
                if prior_resolver_binding:
                    session_resolver.bind(request.scope, prior_resolver_binding)
            log.error("slowave_activate failed: %s", e, exc_info=True)
            return {
                "ok": False,
                "error": {"code": "invalid_input", "message": str(e), "retryable": False},
            }

    @mcp.tool(name="slowave_recall")
    async def slowave_recall(
        ctx: Context,
        session_id: Any = None,
        scope: Any = None,
        page_size: Any = None,
        query: Any = None,
        task_context: Any = None,
        semantic_context: Any = None,
        evidence: Any = "references",
        continue_from: Any = None,
    ) -> dict[str, Any]:
        """Semantic retrieval: bring relevant memories into working memory.
        Use for deliberate mid-task lookups when you need specific historical
        context beyond what activate surfaced.
        Recall is explicitly bound to the active session and matching scope.
        Retrieval considers semantic meaning and specific lexical matches, then
        selects context relevant to the current need. Ask a focused natural-language
        question naming the subject, goal, and relevant conditions; preserve exact
        project/service names, identifiers, and error text when known. Paraphrases
        can match; you do not need to reproduce the stored wording. Avoid keyword
        stuffing or assuming the answer in the query. For example, a stored claim
        "Atlas authentication credentials expire after 45 minutes" can be queried
        with "How long do Atlas login tokens remain valid?"
        State independent needs as separate sentences when asking several questions.
        Results are bounded suggestions, not proof of truth or a complete inventory.
        If needed history is missing, clarify the question or use a returned
        continuation when more context would help; an empty result does not prove
        absence.
        Inspect provenance/evidence and procedure outcomes/caveats before applying
        guidance; use evidence="full" when source content is needed, noting truncation.
        Args:
            page_size: optional strict integer 1–10; omitted values use the server default for a new query. Omit with continue_from.
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
            retrieval_policy_version: server-selected retrieval-policy identifier.
            relevant_total: number of discovered relevance-qualified declarative
                memories in a relevant-set response.
            catalog_truncated: true when the bounded candidate search could have
                omitted qualifying memories.
            more_available: whether a frozen continuation page is available.
            continue_from: opaque continuation cursor, present only when more_available.
            accessible_field: bounded orientation for unreturned candidates;
                contains extra_candidates, kinds, and approx_extra_tokens, never content.
        """
        try:
            request = RecallArguments.model_validate(
                {
                    "session_id": session_id,
                    "scope": scope,
                    "page_size": page_size,
                    "query": query,
                    "task_context": task_context,
                    "semantic_context": semantic_context,
                    "evidence": evidence,
                    "continue_from": continue_from,
                }
            )
        except ValidationError as exc:
            message, field_errors = _format_validation_error(exc)
            return {
                "ok": False,
                "error": {
                    "code": "invalid_input",
                    "message": message,
                    "retryable": False,
                    "field_errors": field_errors,
                },
            }
        prepared_retrieval_id: str | None = None
        try:
            _validate_scope(request.scope)
            eng = build_engine()
            if request.continue_from is not None:
                return {
                    "ok": True,
                    "data": _continuation_page(
                        eng,
                        cursor=request.continue_from,
                        session_id=request.session_id,
                        scope=request.scope,
                    ),
                }
            assert request.query is not None
            result = ops.recall(
                eng,
                query=request.query,
                session_id=request.session_id,
                top_k=_MCP_FROZEN_CANDIDATE_LIMIT,
                evidence=request.evidence == "full",
                scope=request.scope,
                mode="strict_scope",
                task_context=request.task_context,
                semantic_context=request.semantic_context,
                relevant_set=True,
            )
            prepared_retrieval_id = result["retrieval_id"]
            full_data = _canonical_recall_result(
                result, scope=request.scope, evidence=request.evidence
            )
            catalog = FrozenActivationCatalog.prepare(
                retrieval_id=result["retrieval_id"],
                session_id=request.session_id,
                scope=request.scope,
                memories=full_data["memories"],
                procedures=full_data["procedures"],
                page_size=(
                    request.page_size if request.page_size is not None else DEFAULT_MEMORY_PAGE_SIZE
                ),
                contribution_spans={},
                catalog_truncated=bool(full_data.get("catalog_truncated")),
                first_page_metadata=(
                    {"warnings": full_data["warnings"]} if full_data.get("warnings") else None
                ),
                policy_version=full_data["retrieval_policy_version"],
                recall_evidence={
                    key: full_data[key]
                    for key in ("evidence", "evidence_mode", "evidence_truncated")
                },
            )
            data, successor = catalog.page()
            conn = eng.db.connect()
            with conn:
                if successor is not None:
                    data["continue_from"] = _store_complementary_cursor(
                        conn, catalog=catalog, offset=successor
                    )
                _restrict_activation_exposure(
                    eng, retrieval_id=result["retrieval_id"], data=data, commit=False
                )
            response = {"ok": True, "data": data}
            _spawn_bg_task(
                _bg_log_event(
                    eng,
                    request.session_id,
                    "trajectory:action",
                    f"slowave_recall: {request.query}"[:1000],
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
            if prepared_retrieval_id is not None:
                conn = eng.db.connect()
                with conn:
                    conn.execute(
                        "DELETE FROM context_recall_events WHERE context_id = ?",
                        (prepared_retrieval_id,),
                    )
            log.error("slowave_recall failed: %s", e, exc_info=True)
            return {
                "ok": False,
                "error": {"code": "invalid_input", "message": str(e), "retryable": False},
            }

    @mcp.tool(name="slowave_remember")
    async def slowave_remember(
        ctx: Context,
        scope: Any = None,
        session_id: Any = None,
        content: Any = None,
        type: Any = None,
        occurred_at: Any = None,
        memories: Any = None,
    ) -> dict[str, Any]:
        """Record newly discovered knowledge worth retaining for future tasks in this scope.
        Scalar and batch forms inherit one explicitly verified session and scope.
        Proactively call this when you discover new knowledge worth retaining
        as durable memory because it could help you with future tasks:
        pursue a goal, avoid rework or a wrong assumption, or avoid asking
        the user again. Do not wait for the user to say “remember”.
        Good candidates are confirmed facts, user preferences, decisions,
        constraints, lessons, warnings, and open questions that affect future
        goals. Save the claim when it becomes clear; review for missed claims
        before commit. Use concise, standalone wording with enough context for
        a later session, and batch independent claims. Check activated memories
        and task context for duplicates; a matched claim is acceptable. Skip
        speculation, progress notes, pending steps, temporary task state, and
        details unlikely to matter again.
        A fact recorded elsewhere can still be worth saving when rediscovery
        would cost meaningful work.

        Write for later retrieval: use one independent claim per entry and name
        its subject explicitly instead of "this" or "that approach". Preserve
        conditions, exceptions, and negation; include the reason for a decision
        when it explains when to apply it. Keep meaningful exact names, identifiers,
        commands, and error text alongside natural-language context. Retrieval
        considers meaning and specific terms, so avoid keyword lists, repeated
        synonyms, and unrelated claims in one entry. For example, prefer
        "Atlas authentication credentials expire after 45 minutes" to
        "Tokens: timeout, expiry, login, auth; same as discussed above."
        Include source/time context when it affects applicability, and use
        occurred_at only for the source-event time described below.

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
            request = RememberArguments.model_validate(
                {
                    "scope": scope,
                    "session_id": session_id,
                    "content": content,
                    "type": type,
                    "occurred_at": occurred_at,
                    "memories": memories,
                }
            )
        except ValidationError as exc:
            message, field_errors = _format_validation_error(exc)
            return {
                "ok": False,
                "error": {
                    "code": "invalid_input",
                    "message": message,
                    "retryable": False,
                    "field_errors": field_errors,
                },
            }
        try:
            eng = build_engine()
            session = (
                eng.db.connect()
                .execute(
                    "SELECT scope_id, ended_ts FROM sessions WHERE id = ?", (request.session_id,)
                )
                .fetchone()
            )
            if session is None:
                raise ValueError(f"unknown session_id: {request.session_id}")
            if session["ended_ts"] is not None:
                raise ValueError(f"session is already ended: {request.session_id}")
            if session["scope_id"] != request.scope:
                raise ValueError("session_id and scope do not match")
            provenance = _integration_provenance(ctx)
            normalized, is_batch = _normalize_remember_inputs(
                content=request.content,
                memory_type=request.type,
                occurred_at=request.occurred_at,
                memories=(
                    [item.model_dump() for item in request.memories]
                    if request.memories is not None
                    else None
                ),
            )
            if not is_batch:
                item = normalized[0]
                return {
                    "ok": True,
                    "data": ops.remember(
                        eng,
                        content=item["content"],
                        memory_type=item["type"],
                        scope=request.scope,
                        session_id=request.session_id,
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
                                scope=request.scope,
                                session_id=request.session_id,
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
        retrieval_id: Any = None,
        memory_feedback: Any = None,
        procedure_feedback: Any = None,
        retrieval_quality: Any = None,
        missing: Any = None,
        coverage: Any = "partial",
        items: Any = None,
    ) -> dict[str, Any]:
        """Record append-only evidence about retrieved memories and procedures.

        Task outcome does not belong here; slowave_commit owns it. Memory
        feedback informs future retrieval; procedure use/effect informs future
        ordering. Task mismatch does not establish that a claim is false.
        When a retrieved claim is replaced, remember the corrected claim first,
        then mark the old one stale/superseded with its replacement_memory_id.
        Declarative assessments are used|not_used|unassessable|irrelevant|already_known|stale.
        Report used only after observed influence on reasoning, an action, a check,
        or an applied constraint; reading or topical similarity is insufficient.
        not_used means no influence; irrelevant means task mismatch.
        unassessable requires reason and records unknown usage, never guessed non-use.
        Both not_used and unassessable are neutral and complete target accountability.
        Complete all exposed targets after the relevant work, before commit.
        Partial checkpoints may report observed use or stale evidence earlier.
        Only returned targets need feedback; do not exhaust continuation pages.
        You may batch multiple retrievals in items. A minimal memory entry is
        {"memory_id":"<returned>","assessment":"not_used"}. A stale assessment must include
        ``stale_reason`` (contradicted|superseded|outdated|unsupported|withdrawn)
        and a concise ``reason``; superseded additionally requires
        ``replacement_memory_id``. Procedure feedback keeps
        use (used|not_used|unassessable) separate from effect
        (helped|no_effect|harmed|unknown), with contribution required when used.
        Memory feedback additionally accepts: ``relevance``
        (relevant|irrelevant|uncertain), ``effect`` for used marks
        (helped|no_effect|harmed; absent means helped, unknown is invalid),
        and the ``already_known`` assessment (dedup observation; no
        reinforcement, no suppression). Relevance never gates the usage axis
        and vice versa (decoupled axes).
        For assessments other than used, omit effect or set unknown.
        not_used/unassessable procedures require omitted/unknown effect and no contribution;
        unassessable additionally requires reason.
        Replacement memories must be active, different, and in the retrieval scope;
        replacement IDs need not have been exposed by this retrieval.
        Even an empty retrieval requires {"retrieval_id": "<returned>", "coverage": "complete"}.
        Default coverage is partial. For batches put coverage inside each item,
        never at top level. Outer ok=true can contain rejected targets or failed
        batch items: inspect every results[i].ok, data.rejected and data.outstanding.
        Correct rejected entries and outstanding targets, then declare complete.

        Args:
            retrieval_id: opaque ID returned by activate/recall.
            memory_feedback: [{memory_id, assessment, relevance?, effect?, stale_reason?, replacement_memory_id?, reason?}].
            procedure_feedback: [{procedure_id, use, effect, contribution?, reason?}].
            retrieval_quality: optional whole-result quality assessment.
            missing: optional descriptions of expected but absent knowledge.
            coverage: partial or complete; silence under partial is not negative.
            items: batch of records with the same fields. Scalar feedback fields
                   and items are mutually exclusive.
        Returns:
            retrieval_id: the assessed scalar retrieval.
            coverage: achieved partial or complete coverage. A rejected complete request reports partial.
            requested_coverage: the client's requested coverage.
            outstanding: memory_ids and procedure_ids still requiring assessment.
            accepted_event_ids: append-only feedback events accepted by the server.
            rejected: feedback targets or shapes the server did not apply, with reasons.
            applied: actual strengthened/weakened/unchanged numeric changes,
                plus neutral not_used/unassessable observations and stale transitions.
            results: for batch input, ordered item envelopes with independent
                ok/data or ok/error results.
        """
        try:
            request = FeedbackArguments.model_validate(
                {
                    "retrieval_id": retrieval_id,
                    "memory_feedback": memory_feedback,
                    "procedure_feedback": procedure_feedback,
                    "retrieval_quality": retrieval_quality,
                    "missing": missing,
                    "coverage": coverage,
                    "items": items,
                }
            )
        except ValidationError as exc:
            message, field_errors = _format_validation_error(exc)
            return {
                "ok": False,
                "error": {
                    "code": "invalid_input",
                    "message": message,
                    "retryable": False,
                    "field_errors": field_errors,
                },
            }
        # Feedback follows activate/recall in the public lifecycle, so the
        # encoder-enabled engine is already warm.  Reusing it avoids building
        # a second SQLite/FAISS engine just for this short write operation.
        eng = build_engine(disable_encoder=False)
        if request.items is not None:
            if (
                any(
                    value is not None
                    for value in (
                        request.retrieval_id,
                        request.memory_feedback,
                        request.procedure_feedback,
                        request.retrieval_quality,
                        request.missing,
                    )
                )
                or request.coverage != "partial"
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
            for item in request.items:
                item_rid = item.retrieval_id
                try:
                    results.append(
                        {
                            "ok": True,
                            "data": ops.feedback(
                                eng,
                                retrieval_id=item_rid,
                                memory_feedback=(
                                    [x.model_dump() for x in item.memory_feedback]
                                    if item.memory_feedback is not None
                                    else None
                                ),
                                procedure_feedback=(
                                    [x.model_dump() for x in item.procedure_feedback]
                                    if item.procedure_feedback is not None
                                    else None
                                ),
                                retrieval_quality=item.retrieval_quality,
                                missing=item.missing,
                                coverage=item.coverage,
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
            if not request.retrieval_id:
                raise ValueError("retrieval_id is required")
            data = ops.feedback(
                eng,
                retrieval_id=request.retrieval_id,
                memory_feedback=(
                    [x.model_dump() for x in request.memory_feedback]
                    if request.memory_feedback is not None
                    else None
                ),
                procedure_feedback=(
                    [x.model_dump() for x in request.procedure_feedback]
                    if request.procedure_feedback is not None
                    else None
                ),
                retrieval_quality=request.retrieval_quality,
                missing=request.missing,
                coverage=request.coverage,
            )
            return {"ok": True, "data": data}
        except Exception as e:
            log.error("slowave_feedback failed: %s", e, exc_info=True)
            code = "not_found" if "unknown retrieval_id" in str(e) else "invalid_input"
            return {"ok": False, "error": {"code": code, "message": str(e), "retryable": False}}

    @mcp.tool(name="slowave_commit")
    async def slowave_commit(
        ctx: Context,
        session_id: Any = None,
        final_goal: Any = None,
        outcome: Any = None,
        outcome_summary: Any = None,
        verification: Any = None,
        procedure: Any = None,
        trajectory: Any = None,
    ) -> dict[str, Any]:
        """Close the current task and trigger offline memory consolidation.

        Before calling, review whether the task revealed useful knowledge that
        a future task should retrieve directly. Store any such claims with
        slowave_remember; if none qualifies, commit normally. Commit records the
        task outcome and optional procedure, not a substitute for explicit claims.
        Call at the end of every task. If skipped, the idle-session reaper closes
        the session after SLOWAVE_SESSION_IDLE_TIMEOUT seconds (default 3600).
        Args:
            session_id: required active session from activate.
            final_goal: required confirmed goal.
            outcome: required success|partial|failure.
            outcome_summary: required standalone actual result.
            verification: required {status, summary, evidence_refs?}; status is
                verified|partially_verified|unverified.
            procedure: optional {version: 2, summary, context: {}, steps:
                [{summary}], caveats: []}; include only for a reusable method
                that was actually attempted.
            trajectory: optional executed-attempt trace of at most 32 action/observation
                entries, each {kind, summary, status?}. Must contain TASK
                actions/observations only. Do NOT include
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
            request = CommitArguments.model_validate(
                {
                    "session_id": session_id,
                    "final_goal": final_goal,
                    "outcome": outcome,
                    "outcome_summary": outcome_summary,
                    "verification": verification,
                    "procedure": procedure,
                    "trajectory": trajectory,
                }
            )
        except ValidationError as exc:
            message, field_errors = _format_validation_error(exc)
            return {
                "ok": False,
                "error": {
                    "code": "invalid_input",
                    "message": message,
                    "retryable": False,
                    "field_errors": field_errors,
                },
            }
        try:
            # O10 trajectories must be embedded so session_end can form episodic
            # memories from the attempted path; an encoder-free commit would
            # preserve rows for audit but silently exclude them from consolidation.
            eng = build_engine(disable_encoder=False)
            result = ops.commit(
                eng,
                session_id=request.session_id,
                outcome=request.outcome,
                final_goal=request.final_goal,
                outcome_summary=request.outcome_summary,
                procedure=request.procedure.model_dump() if request.procedure is not None else None,
                verification=request.verification.model_dump(),
                trajectory=(
                    [item.model_dump() for item in request.trajectory]
                    if request.trajectory
                    else None
                ),
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

    _publish_schema(mcp, "slowave_activate", ActivateArguments)
    _publish_schema(mcp, "slowave_recall", RecallArguments)
    _publish_schema(mcp, "slowave_remember", RememberArguments)
    _publish_schema(mcp, "slowave_feedback", FeedbackArguments)
    _publish_schema(mcp, "slowave_commit", CommitArguments)
