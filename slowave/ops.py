"""Shared operation contracts for the Slowave 5-verb cognitive cycle.

Both the MCP tools (slowave/mcp/tools.py) and the CLI (slowave/cli/main.py)
delegate to these functions so the input/output contract is defined once.
If a field is added, renamed, or removed here, both interfaces update together.

Functions are synchronous; the MCP layer may wrap side-effects in background
tasks for performance, but the contract shape is identical.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import asdict
from typing import Any

from slowave.core.config import DEFAULT_RECALL_TOP_K
from slowave.core.engine import SlowaveEngine


def _fragmentation_key(scope_id: str) -> str:
    """Normalize a scope id for fuzzy fragmentation comparison.

    Lowercases and treats '-'/'_' as equivalent separators, so
    "project:my-repo" and "project:my_repo" collapse to the same key. Two
    scope ids sharing a key are almost certainly meant to be the same scope.
    """
    return re.sub(r"[-_]+", "-", scope_id.strip().lower())


def _scope_fragmentation_warning(eng: SlowaveEngine, scope_id: str) -> str | None:
    """On cold start, warn if an existing scope looks like a typo/separator
    variant of this "new" one (e.g. "project:my-repo" vs "project:my_repo").

    Scope strings are never validated or canonicalized (slowave/core/scope.py
    only strips whitespace) -- a typo silently creates a second, fully
    isolated memory store with no other signal. This check is advisory only:
    it never rejects or rewrites the scope, just surfaces the collision so
    the AI client can ask the user before re-ingesting into what might be a
    fragment rather than a genuinely new scope.
    """
    key = _fragmentation_key(scope_id)
    for existing in eng.schemas.scope_registry.list_scope_ids():
        if existing == scope_id:
            continue
        if _fragmentation_key(existing) == key:
            return (
                f"scope {scope_id!r} looks like a variant of existing scope "
                f"{existing!r} (case/separator difference) -- if this wasn't "
                "intentional, memories will silently split across the two."
            )
    return None


def _is_scope_rejection(reason: str) -> bool:
    """True if an ActivationTrace rejection reason is scope-related.

    activate()'s filtered_items list intentionally excludes these -- a
    candidate rejected purely for belonging to a different scope isn't an
    interesting "close but filtered" signal worth persisting for trace
    analysis. This used to check `reason != "scope_mismatch"`, an exact-match
    test that never actually matched: WorkingMemoryGate's real reason strings
    are either a compound diagnostic (e.g.
    "cosine=0.22,cue_overlap=0.04,...,scope_mismatch", from _activation()) or
    the unrelated literal "strict_scope_excluded" (from _eligible()'s hard
    scope wall) -- never the bare string "scope_mismatch" itself. The broken
    check silently let every scope-rejected candidate through, which is what
    fed the cross-scope co-activation leak (see
    private/docs/iterations/20260723_part_of_audit_and_brain_alignment_review.md).
    """
    return (
        reason == "strict_scope_excluded"
        or "scope_mismatch" in reason
        or reason == "cross_scope_below_floor"
        or reason.startswith("cross_scope_low_cosine")
    )


def _pathway_for(item: Any) -> str:
    """Classify a WorkingMemoryItem's admission channel (WP-6).

    Distinguishes 'direct' (query-relevant), 'exploration' (salience-filled
    slot, item.peripheral with no graph reason), and 'graph' (association via
    expand_via_relations, reason prefixed "graph:") -- the three structurally
    different reasons an item can end up admitted, previously conflated by
    the single `peripheral` flag and not persisted at all. See
    private/docs/iterations/20260728_retrieval_quality_execution_progress.md
    (WP-6) for why this matters to the co-activation writer.
    """
    if item.reason.startswith("graph:"):
        return "graph"
    if item.peripheral:
        return "exploration"
    return "direct"


def activate(
    eng: SlowaveEngine,
    *,
    query: str,
    scope: str | None = None,
    goal: str | None = None,
    task_type: str | None = None,
    situation: dict[str, Any] | None = None,
    requirements: list[str] | None = None,
    topics: list[str] | None = None,
    entities: list[str] | None = None,
    mode: str = "strict_scope",
    limit: int = 8,
    session_id: str | None = None,
    agent: str = "cli",
    include_peripheral: bool = True,
    include_schemas: bool = True,
    min_relevance: float | None = None,
    graph_channels: str | None = None,
    min_neighbor_relevance: float | None = None,
) -> dict[str, Any]:
    """Prime working memory.  Opens a session when session_id is None.

    Args:
        include_peripheral: when False, drops the trailing salience-filled
            "exploration slots" (labelled "(peripheral)") from both `rendered`
            and `schemas` — a pure relevance-ranked brief.
        include_schemas: when False, omits the structured `schemas` array.
            `rendered` and `schemas` currently encode the identical set of
            memories in two formats; a caller that only consumes one of them
            can skip paying for the other.
        min_relevance: optional floor override on topical evidence alone
            (cosine + lexical cue_overlap, pre-prior) — see
            GatePolicy.min_relevance. Omit (None) to use
            RetrievalService.context_brief's calibrated default
            (_ACTIVATE_MIN_RELEVANCE_DEFAULT); pass 0.0 to disable the floor
            entirely. NOTE: prior to WP-5 this parameter defaulted to `0.0`
            and was always forwarded, which silently disabled the WP-4
            relevance floor for every production caller (MCP tool, CLI) since
            none of them passed a value — fixed to the same optional-override
            pattern already used by recall()'s min_relevance.
        graph_channels / min_neighbor_relevance: optional WP-5 associative-
            retrieval overrides — see RetrievalService.context_brief's
            docstring. Omit to use the calibrated production defaults.

    Returns:
        retrieval_id      – pass to reinforce()
        session_id        – pass to commit()
        rendered          – human-readable brief
        schemas           – [{id, text, activation, reason, source_kind}, ...]
                             (omitted when include_schemas=False)
        cold_start        – True when the scope has no memories yet
        cue_terms         – extracted query terms
        suppressed        – gate rejection counts by reason
        activation_trace  – full trace (only when mode="debug")
    """
    situation = situation or {}
    requirements = requirements or []
    topics = topics or []
    entities = entities or []

    if session_id is None:
        session_id = eng.session_start(agent=agent, scope=scope, goal=goal)

    _brief_kwargs: dict[str, Any] = {}
    if min_relevance is not None:
        _brief_kwargs["min_relevance"] = min_relevance
    if graph_channels is not None:
        _brief_kwargs["graph_channels"] = graph_channels
    if min_neighbor_relevance is not None:
        _brief_kwargs["min_neighbor_relevance"] = min_neighbor_relevance

    brief = eng.context_brief(
        query=query,
        scope=scope,
        goal=goal,
        task_type=task_type,
        situation=situation,
        requirements=requirements,
        topics=topics,
        entities=entities,
        mode=mode,
        limit=limit,
        include_peripheral=include_peripheral,
        **_brief_kwargs,
    )

    context_id = f"ctx_{uuid.uuid4().hex[:12]}"
    scope_id = scope.strip() if scope else None
    scope_kind = scope.split(":", 1)[0] if scope and ":" in scope else None
    # With a scope, cold start means "this scope has no memories yet". Without
    # one, it must fall back to "the whole DB has no memories yet" rather than
    # hardcoding False -- a scopeless activate() on a genuinely empty DB is
    # still a first-ever encounter, not a normal retrieval.
    cold_start = eng.schemas.count_by_scope(scope_id) == 0 if scope_id else eng.schemas.count() == 0
    scope_warning = _scope_fragmentation_warning(eng, scope_id) if cold_start and scope_id else None

    _internal = {
        "memory_ids": [f"sch_{item.schema.id}" for item in brief.items],
        "schemas": [
            {
                "id": f"sch_{item.schema.id}",
                "activation": item.activation,
                "reason": item.reason,
                "pathway": _pathway_for(item),
            }
            for item in brief.items
        ],
    }
    _filtered = [
        {"memory_id": f"sch_{t.schema_id}", "activation": t.activation, "reason": t.reason}
        for t in brief.activation_trace
        if not t.admitted and not _is_scope_rejection(t.reason)
    ]
    eng.record_context_recall(
        context_id=context_id,
        session_id=session_id,
        scope_id=scope_id,
        scope_kind=scope_kind,
        query=query,
        goal=goal,
        task_type=task_type,
        situation=situation,
        requirements=requirements,
        mode=mode,
        limit=limit,
        topics=topics,
        entities=entities,
        cue_terms=brief.cue_terms,
        suppressed=brief.suppressed,
        response=_internal,
        filtered_items=_filtered,
    )

    result: dict[str, Any] = {
        "retrieval_id": context_id,
        "session_id": session_id,
        "rendered": brief.rendered,
        "cold_start": cold_start,
        "cue_terms": brief.cue_terms,
        "suppressed": brief.suppressed,
    }
    if scope_warning:
        result["scope_warning"] = scope_warning
    if include_schemas:
        result["schemas"] = [
            {
                "id": f"sch_{item.schema.id}",
                "text": str(item.schema.content_text or "")[:500],
                "activation": round(min(1.0, max(0.0, item.activation)), 4),
                "reason": item.reason,
                "source_kind": str((item.schema.facets or {}).get("source_kind", "")),
            }
            for item in brief.items
        ]
    if mode == "debug":
        result["activation_trace"] = [asdict(t) for t in brief.activation_trace]
    return result


def remember(
    eng: SlowaveEngine,
    *,
    content: str,
    memory_type: str = "decision",
    scope: str | None = None,
    session_id: str | None = None,
) -> dict[str, Any]:
    """Encode a durable typed claim.

    Returns:
        stored      – True on success
        event_id    – evt_N
        schema_id   – sch_N (or None)
        memory_type – echoed back
        scope       – echoed back
    """
    if not content or not str(content).strip():
        return {"stored": False, "skipped": True, "reason": "content is empty", "scope": scope}
    rid = eng.remember(content=content, type=memory_type, scope=scope, session_id=session_id)
    schema_id = rid.schema_id if hasattr(rid, "schema_id") and rid.schema_id else None
    return {
        "stored": True,
        "event_id": f"evt_{rid}",
        "schema_id": f"sch_{schema_id}" if schema_id else None,
        "memory_type": memory_type,
        "scope": scope,
    }


def recall(
    eng: SlowaveEngine,
    *,
    query: str,
    scope: str | None = None,
    mode: str = "default",
    top_k: int = DEFAULT_RECALL_TOP_K,
    evidence: bool = False,
    min_relevance: float | None = None,
    graph_channels: str | None = None,
    min_neighbor_relevance: float | None = None,
) -> dict[str, Any]:
    """Semantic retrieval.

    Args:
        min_relevance: optional relevance floor override — see
            SlowaveEngine.recall's docstring for the default and its scale
            caveats. Omit to use the engine's default floor; pass 0.0 to
            disable it entirely for this call.
        graph_channels / min_neighbor_relevance: optional WP-5 associative-
            retrieval overrides — see RetrievalService.recall's docstring.
            Omit to use the calibrated production defaults.

    Returns:
        retrieval_id     – pass to reinforce()
        memories         – [{id, content_text, activation, rank_score, scope_id, ...}, ...]
                           `activation` is raw topical/channel relevance evidence
                           (what min_relevance gates on); `rank_score` is
                           activation + salience_weight*normalized salience
                           (what determines sort order, WP-4). They differ
                           by design — see SlowaveEngine.recall's docstring.
        related_memories – schema_relations-linked schemas surfaced via spreading
                           activation from `memories`, NOT counted toward top_k
                           (same shape as `memories`, plus a `via` field naming
                           the relation type(s) it arrived through)
        episodes         – raw episode text records (when evidence=True)
        raw_events       – raw event records (when evidence=True)
    """
    result = eng.recall(
        query,
        top_k=top_k,
        evidence=evidence,
        scope=scope,
        mode=mode,
        min_relevance=min_relevance,
        graph_channels=graph_channels,
        min_neighbor_relevance=min_neighbor_relevance,
    )
    recall_id = f"rec_{uuid.uuid4().hex[:12]}"
    _internal = {
        "memory_ids": [f"sch_{s.id}" for s in result.schemas],
        "schemas": [
            {
                "id": f"sch_{s.id}",
                "score": result.schema_activations.get(s.id),
                "pathway": "direct",
            }
            for s in result.schemas
        ],
        "related_schemas": [
            {
                "id": f"sch_{s.id}",
                "score": result.schema_activations.get(s.id),
                "pathway": "graph",
            }
            for s in result.related_schemas
        ],
    }
    eng.record_retrieval(
        retrieval_id=recall_id,
        retrieval_type="recall",
        query=query,
        scope_id=scope,
        mode=mode,
        limit=top_k,
        response=_internal,
    )
    memories = [
        {
            "id": f"sch_{s.id}",
            "content_text": str(s.content_text or "")[:500],
            "activation": round(result.schema_activations.get(s.id, 0.0), 4),
            "rank_score": round(result.schema_rank_scores.get(s.id, 0.0), 4),
            "scope_id": s.scope_id,
            "status": s.status,
            "salience": s.salience,
            "needs_review": s.is_labile,
            "generalization_stage": s.generalization_stage,
        }
        for s in result.schemas
    ]
    related_memories = [
        {
            "id": f"sch_{s.id}",
            "content_text": str(s.content_text or "")[:500],
            "activation": round(result.schema_activations.get(s.id, 0.0), 4),
            "rank_score": round(result.schema_rank_scores.get(s.id, 0.0), 4),
            "scope_id": s.scope_id,
            "status": s.status,
            "generalization_stage": s.generalization_stage,
            "via": result.related_schema_relations.get(s.id, []),
        }
        for s in result.related_schemas
    ]
    return {
        "retrieval_id": recall_id,
        "memories": memories,
        "related_memories": related_memories,
        "episodes": result.episode_texts,
        "raw_events": result.raw_events,
    }


def reinforce(
    eng: SlowaveEngine,
    *,
    retrieval_id: str,
    feedback: str = "useful",
    outcome: str = "unknown",
    used_memory_ids: list[str] | None = None,
    irrelevant_memory_ids: list[str] | None = None,
    stale_memory_ids: list[str] | None = None,
    wrong_memory_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Apply feedback to retrieved memories.

    Returns the raw retrieval_feedback dict from the engine.
    """
    return eng.retrieval_feedback(
        retrieval_id=retrieval_id,
        feedback=feedback,
        outcome=outcome,
        used_memory_ids=used_memory_ids,
        irrelevant_memory_ids=irrelevant_memory_ids,
        stale_memory_ids=stale_memory_ids,
        wrong_memory_ids=wrong_memory_ids,
    )


def commit(
    eng: SlowaveEngine,
    *,
    session_id: str,
    outcome: str = "unknown",
    steps: list[str] | None = None,
) -> dict[str, Any]:
    """Close a session and encode events into episodic memories.

    Args:
        session_id: from activate response.
        outcome: task result — success|partial|failure|unknown.
        steps: optional ordered list of what the agent did (e.g. "Ran full test suite",
               "Built Docker image"). Each becomes a raw_event with type="step".

    Returns:
        session_id      – echoed back
        episodes_formed – number of episodic memories created
        session_note    – informational only; present when the session had
                           several tool-use events but never called recall().
    """
    if steps:
        for step_text in steps:
            eng.event_append(
                session_id=session_id,
                type="step",
                content=step_text.strip(),
            )
    result = eng.session_end(session_id, consolidate=False, outcome=outcome)
    out: dict[str, Any] = {
        "session_id": session_id,
        "episodes_formed": result.get("episodes_formed", 0),
    }

    tool_event_count = len(eng.raw_log.list_session(session_id))
    if tool_event_count >= 3:
        row = (
            eng.db.connect()
            .execute(
                "SELECT COUNT(*) AS cnt FROM context_recall_events "
                "WHERE session_id = ? AND retrieval_type = 'recall'",
                (session_id,),
            )
            .fetchone()
        )
        recall_count = int(row["cnt"]) if row else 0
        if recall_count == 0:
            out["session_note"] = (
                f"{tool_event_count} tool-use events, 0 recall() calls this session"
            )
    return out


def stats(eng: SlowaveEngine) -> dict[str, Any]:
    """Return system counts: episodes, prototypes, schemas, procedures, edges."""
    return eng.stats()
