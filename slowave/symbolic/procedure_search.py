"""Derived search documents for advisory procedural memory.

Procedure payloads are immutable task-completion evidence.  This module builds
an independently rebuildable search representation from that evidence and the
activation that led to it.  The backfill is deliberately best-effort: a bad
historical row is recorded and skipped rather than preventing Slowave from
opening its database.
"""

from __future__ import annotations

import json
import logging
import re
import time
from collections import Counter
from dataclasses import dataclass
from typing import Any

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProcedureSearchDocument:
    procedure_id: str
    scope_id: str | None
    applicability_text: str
    strategy_text: str


@dataclass(frozen=True)
class ProcedureLexicalCandidate:
    procedure_id: str
    rank: int
    specific: bool


def procedure_cue(
    *,
    query: str | None,
    goal: str | None = None,
    semantic_context: str | None = None,
    task_type: str | None = None,
    situation: dict[str, Any] | None = None,
    requirements: list[str] | None = None,
    topics: list[str] | None = None,
    entities: list[str] | None = None,
) -> str:
    """Build the semantic applicability cue without embedding opaque context JSON."""
    parts: list[str] = [
        str(query or "").strip(),
        str(goal or "").strip(),
        str(semantic_context or "").strip(),
        str(task_type or "").strip(),
    ]
    parts.extend(f"{key} {value}" for key, value in sorted((situation or {}).items()))
    parts.extend(str(item).strip() for item in requirements or [])
    parts.extend(str(item).strip() for item in topics or [])
    parts.extend(str(item).strip() for item in entities or [])
    return " ".join(part for part in parts if part)


def backfill_procedure_search(conn: Any, *, session_id: str | None = None) -> int:
    """Materialize all backfillable procedure search documents.

    This function is safe to call during startup and after a commit.  It never
    propagates a historical-data error: procedure search is advisory and must
    not block the canonical event log or normal memory retrieval.
    """
    try:
        return _backfill(conn, session_id=session_id)
    except Exception:
        log.exception("procedure search backfill skipped after an unexpected error")
        return 0


def _backfill(conn: Any, *, session_id: str | None) -> int:
    where = "WHERE e.type = 'task_complete'"
    params: list[Any] = []
    if session_id is not None:
        where += " AND e.session_id = ?"
        params.append(session_id)
    rows = conn.execute(
        "SELECT e.id AS event_id, e.session_id, e.ts AS completed_at, e.metadata_json, "
        "s.scope_id, s.initial_goal, s.goal, s.task_context_json, s.outcome, "
        "s.verification_json FROM raw_events e JOIN sessions s ON s.id = e.session_id "
        f"{where} ORDER BY e.id",
        params,
    ).fetchall()
    fts_available = _fts_available(conn)
    stored = 0
    for row in rows:
        # Procedure identity is historically session-based.  Keep this stable
        # so feedback references and dashboard links continue to resolve.
        procedure_id = f"proc_{row['session_id']}"
        try:
            document = _document_from_row(conn, row, procedure_id)
            if document is None:
                continue
            _upsert(conn, row, document, update_fts=fts_available)
            stored += 1
        except Exception as exc:
            log.warning("skipped procedure search backfill for %s: %s", procedure_id, exc)
    try:
        conn.commit()
    except Exception:
        log.exception("could not commit procedure search backfill")
    return stored


def _document_from_row(
    conn: Any, row: Any, procedure_id: str
) -> tuple[ProcedureSearchDocument, str | None] | None:
    metadata = _json_object(row["metadata_json"])
    procedure = metadata.get("procedure")
    if not isinstance(procedure, dict) or procedure.get("version") != 2:
        return None
    summary = _text(procedure.get("summary"))
    steps = procedure.get("steps")
    if not summary or not isinstance(steps, list):
        raise ValueError("procedure payload is missing a summary or steps")
    source = conn.execute(
        "SELECT context_id, query, goal, application, task_type, situation_json, "
        "requirements_json, topics_json, entities_json FROM context_recall_events "
        "WHERE session_id = ? AND retrieval_type = 'context' "
        "ORDER BY created_at, rowid LIMIT 1",
        (row["session_id"],),
    ).fetchone()
    if source is None:
        applicability = procedure_cue(
            query=None,
            goal=_text(row["initial_goal"]) or _text(row["goal"]),
        )
        source_context_id = None
    else:
        applicability = procedure_cue(
            query=_text(source["query"]),
            goal=_text(source["goal"]) or _text(row["initial_goal"]) or _text(row["goal"]),
            semantic_context=_text(source["application"]),
            task_type=_text(source["task_type"]),
            situation=_json_object(source["situation_json"]),
            requirements=_json_list(source["requirements_json"]),
            topics=_json_list(source["topics_json"]),
            entities=_json_list(source["entities_json"]),
        )
        source_context_id = str(source["context_id"])
    if not applicability:
        raise ValueError("source activation contains no applicability cue")
    step_text = [_text(step.get("summary")) for step in steps if isinstance(step, dict)]
    strategy = " ".join(part for part in [summary, *step_text] if part)
    if not strategy:
        raise ValueError("procedure contains no searchable strategy text")
    return (
        ProcedureSearchDocument(
            procedure_id=procedure_id,
            scope_id=row["scope_id"],
            applicability_text=applicability,
            strategy_text=strategy,
        ),
        source_context_id,
    )


def _upsert(
    conn: Any,
    row: Any,
    payload: tuple[ProcedureSearchDocument, str | None],
    *,
    update_fts: bool,
) -> None:
    document, source_context_id = payload
    context = _json_object(_json_object(row["metadata_json"]).get("procedure", {}).get("context"))
    verification = _json_object(row["verification_json"])
    conn.execute(
        "INSERT INTO procedure_search_documents "
        "(procedure_id, source_session_id, source_context_id, scope_id, completed_at, "
        "applicability_text, strategy_text, context_json, outcome, verification_status, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
        "ON CONFLICT(procedure_id) DO UPDATE SET "
        "source_session_id=excluded.source_session_id, source_context_id=excluded.source_context_id, "
        "scope_id=excluded.scope_id, completed_at=excluded.completed_at, "
        "applicability_text=excluded.applicability_text, strategy_text=excluded.strategy_text, "
        "context_json=excluded.context_json, outcome=excluded.outcome, "
        "verification_status=excluded.verification_status, updated_at=excluded.updated_at",
        (
            document.procedure_id,
            row["session_id"],
            source_context_id,
            document.scope_id,
            row["completed_at"],
            document.applicability_text,
            document.strategy_text,
            json.dumps(context, ensure_ascii=False, sort_keys=True),
            _text(row["outcome"]) or "unknown",
            _text(verification.get("status")) or "unverified",
            int(time.time()),
        ),
    )
    if update_fts:
        # FTS5 is a performance accelerator, not migration-critical state.
        # The searchable document above remains available when an embedded
        # SQLite was built without FTS5 or a damaged index needs a later retry.
        try:
            conn.execute(
                "DELETE FROM procedure_search_fts WHERE procedure_id = ?",
                (document.procedure_id,),
            )
            conn.execute(
                "INSERT INTO procedure_search_fts (procedure_id, applicability_text, strategy_text) "
                "VALUES (?, ?, ?)",
                (document.procedure_id, document.applicability_text, document.strategy_text),
            )
        except Exception:
            log.warning("procedure FTS update skipped for %s", document.procedure_id, exc_info=True)


def load_procedure_search_documents(
    conn: Any, *, scope: str | None
) -> dict[str, ProcedureSearchDocument]:
    try:
        where, params = "", []
        if scope is not None:
            where, params = " WHERE scope_id = ?", [scope]
        rows = conn.execute(
            "SELECT procedure_id, scope_id, applicability_text, strategy_text "
            "FROM procedure_search_documents" + where,
            params,
        ).fetchall()
    except Exception:
        return {}
    return {
        str(row["procedure_id"]): ProcedureSearchDocument(
            procedure_id=str(row["procedure_id"]),
            scope_id=row["scope_id"],
            applicability_text=_text(row["applicability_text"]),
            strategy_text=_text(row["strategy_text"]),
        )
        for row in rows
    }


def search_procedure_fts(
    conn: Any, *, query: str, scope: str | None, limit: int = 20
) -> dict[str, ProcedureLexicalCandidate]:
    tokens = _tokens(query)
    if not tokens:
        return {}
    expression = " OR ".join('"' + token.replace('"', '""') + '"' for token in tokens)
    try:
        sql = (
            "SELECT f.procedure_id, f.rank, d.applicability_text, d.strategy_text "
            "FROM procedure_search_fts f JOIN procedure_search_documents d "
            "ON d.procedure_id = f.procedure_id WHERE procedure_search_fts MATCH ?"
        )
        params: list[Any] = [expression]
        if scope is not None:
            sql += " AND d.scope_id = ?"
            params.append(scope)
        sql += " ORDER BY f.rank ASC, f.procedure_id ASC LIMIT ?"
        params.append(limit)
        rows = conn.execute(sql, params).fetchall()
    except Exception:
        return {}
    query_terms = {token.casefold() for token in tokens}
    document_count, document_frequency = _document_frequency(conn, scope=scope)
    results: dict[str, ProcedureLexicalCandidate] = {}
    for rank, row in enumerate(rows, 1):
        text_terms = {
            token.casefold()
            for token in _tokens(
                _text(row["applicability_text"]) + " " + _text(row["strategy_text"])
            )
        }
        overlap = query_terms & text_terms
        results[str(row["procedure_id"])] = ProcedureLexicalCandidate(
            procedure_id=str(row["procedure_id"]),
            rank=rank,
            specific=_has_discriminative_term(overlap, document_count, document_frequency),
        )
    return results


def _document_frequency(conn: Any, *, scope: str | None) -> tuple[int, Counter[str]]:
    try:
        where, params = "", []
        if scope is not None:
            where, params = " WHERE scope_id = ?", [scope]
        rows = conn.execute(
            "SELECT applicability_text, strategy_text FROM procedure_search_documents" + where,
            params,
        ).fetchall()
    except Exception:
        return 0, Counter()
    frequencies: Counter[str] = Counter()
    for row in rows:
        frequencies.update(
            set(_tokens(_text(row["applicability_text"]) + " " + _text(row["strategy_text"])))
        )
    return len(rows), frequencies


def _has_discriminative_term(
    overlap: set[str], document_count: int, document_frequency: Counter[str]
) -> bool:
    """Require a term that is uncommon in this procedure corpus.

    BM25 ranks lexical matches; document frequency only decides whether
    lexical evidence is strong enough to bypass the dense admission floor.
    This is language-neutral and adapts to each scope's vocabulary.
    """
    if not overlap:
        return False
    # Multiple independent overlapping terms are useful lexical evidence even
    # when a very small corpus makes every term look common.
    if len(overlap) >= 2:
        return True
    if document_count == 0:
        return False
    cutoff = max(1, document_count // 2)
    return any(document_frequency[token] <= cutoff for token in overlap)


def _fts_available(conn: Any) -> bool:
    try:
        return (
            conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'procedure_search_fts'"
            ).fetchone()
            is not None
        )
    except Exception:
        return False


def _json_object(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    try:
        decoded = json.loads(value or "{}")
    except (TypeError, ValueError):
        return {}
    return decoded if isinstance(decoded, dict) else {}


def _json_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    try:
        decoded = json.loads(value or "[]")
    except (TypeError, ValueError):
        return []
    return [str(item) for item in decoded] if isinstance(decoded, list) else []


def _text(value: Any) -> str:
    return str(value or "").strip()


def _tokens(value: str) -> list[str]:
    tokens: list[str] = []
    for raw in re.findall(r"[^\W_]+(?:[_./:-][^\W_]+)*", value, flags=re.UNICODE):
        for token in (raw, *re.split(r"[_./:-]+", raw)):
            token = token.strip()
            if len(token) >= 2 and token not in tokens:
                tokens.append(token)
    return tokens
