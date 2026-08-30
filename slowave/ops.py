"""Shared operation contracts for the Slowave 5-verb cognitive cycle.

Both the MCP tools (slowave/mcp/tools.py) and the CLI (slowave/cli/main.py)
delegate to these functions so the input/output contract is defined once.
If a field is added, renamed, or removed here, both interfaces update together.

Functions are synchronous; the MCP layer may wrap side-effects in background
tasks for performance, but the contract shape is identical.
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import asdict, replace
from typing import Any

import numpy as np

from slowave.core.config import DEFAULT_RECALL_TOP_K
from slowave.core.continuity import resolve_continuity
from slowave.core.engine import SlowaveEngine
from slowave.core.lifecycle import is_slowave_lifecycle
from slowave.core.services.retrieval_access import canonical_cue_text
from slowave.symbolic.procedural_memory import (
    load_procedures,
    normalize_facets,
    retrieve_procedures,
    validate_procedure,
    validate_procedure_uses,
)

# Start-only reinstatement is for a broad, under-specified client opening, not
# a way to pad a clearly answered question with neighbouring facets.  This is
# on the existing normalized gate-activation scale and deliberately remains
# server-owned and language-neutral.
_CONTEXT_REINSTATEMENT_MAX_CORE_ACTIVATION = 0.60
_CONTEXT_REINSTATEMENT_CONTEXTUAL_MAX_CORE_ACTIVATION = 0.75


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
    if item.reason.startswith("context_reinstatement"):
        return "context_reinstatement"
    if item.reason.startswith("graph:"):
        return "graph"
    if item.peripheral:
        return "exploration"
    return "direct"


def _item_similarity(left: Any, right: Any) -> float:
    """Language-neutral similarity used only to diversify already-admitted items."""
    left_embedding = getattr(left.schema, "embedding", None)
    right_embedding = getattr(right.schema, "embedding", None)
    if left_embedding is None or right_embedding is None:
        return 1.0 if left.schema.content_text == right.schema.content_text else 0.0
    a = np.asarray(left_embedding, dtype=np.float32)
    b = np.asarray(right_embedding, dtype=np.float32)
    return float(a.dot(b) / ((float(np.linalg.norm(a)) * float(np.linalg.norm(b))) + 1e-12))


def _select_context_reinstatement(
    *, candidates: list[Any], core: list[Any], scope_id: str | None, max_items: int
) -> list[Any]:
    """MMR-select same-scope context without weakening core admission.

    Candidate admission comes entirely from ``context_brief``'s current
    status, scope, class, relevance, salience, utility and recency-aware gate.
    This final pass only trades a little relevance for non-duplicate context.
    """
    selected: list[Any] = []
    pool = [
        item
        for item in candidates
        if item.schema.id not in {core_item.schema.id for core_item in core}
        and item.schema.scope_id == scope_id
        and item.schema.status == "active"
    ]
    while pool and len(selected) < max_items:

        def mmr(item: Any) -> float:
            neighbours = list(core) + selected
            duplicate = max((_item_similarity(item, other) for other in neighbours), default=0.0)
            # Activation already includes semantic evidence plus the existing
            # class/utility/salience/recency signals; MMR never admits a raw
            # unscored candidate.
            return 0.72 * float(item.activation) - 0.28 * max(0.0, duplicate)

        best = max(pool, key=mmr)
        duplicate = max(
            (_item_similarity(best, other) for other in list(core) + selected), default=0.0
        )
        pool.remove(best)
        if duplicate >= 0.92:
            continue
        selected.append(replace(best, peripheral=False, reason="context_reinstatement"))
    return selected


def _retrieval_exposure_snapshot(
    *,
    schemas: list[dict[str, Any]],
    procedures: list[dict[str, Any]],
    related_schemas: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build the shared activate/recall exposure representation.

    The feedback ledger authorizes only targets persisted from these structured
    arrays.  Keeping the representation in one helper prevents a public result
    from being returned without also becoming assessable through
    ``slowave_feedback``.
    """
    related = related_schemas or []
    return {
        "memory_ids": [item["id"] for item in schemas] + [item["id"] for item in related],
        "procedure_ids": [item["id"] for item in procedures],
        "schemas": schemas,
        "related_schemas": related,
        "procedures": procedures,
    }


def _shadow_access_traces(
    eng: SlowaveEngine,
    *,
    candidates: list[tuple[int, float, str]],
    query: str | None,
    goal: str | None,
    task_type: str | None,
    situation: dict[str, Any] | None,
    requirements: list[str] | None,
    topics: list[str] | None,
    entities: list[str] | None,
    scope_id: str | None,
) -> list[dict[str, object]]:
    """Evaluate all recorded candidates under Phase-2 shadow policy only."""
    if eng.encoder is None:
        return []
    cue_text = canonical_cue_text(
        query=query,
        goal=goal,
        task_type=task_type,
        situation=situation,
        requirements=requirements,
        topics=topics,
        entities=entities,
    )
    if not cue_text:
        return []
    try:
        cue_embedding = eng.encoder.encode(cue_text)
    except Exception:
        return []
    return [
        eng.shadow_retrieval_access(
            schema_id=schema_id,
            raw_semantic_relevance=activation,
            pathway=pathway,
            cue_embedding=cue_embedding,
            scope_id=scope_id,
            task_type=task_type,
        )
        for schema_id, activation, pathway in candidates
    ]


def activate(
    eng: SlowaveEngine,
    *,
    query: str,
    task: str | None = None,
    scope: str | None = None,
    goal: str | None = None,
    initial_goal: str | None = None,
    retrieval_context: dict[str, Any] | None = None,
    task_context: dict[str, Any] | None = None,
    continuity_id: str | None = None,
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
    include_diagnostics: bool = True,
    min_relevance: float | None = None,
    graph_channels: str | None = None,
    min_neighbor_relevance: float | None = None,
    manage_continuity: bool = False,
    continuity_integration: str | None = None,
    continuity_client_identity: str | None = None,
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
        include_diagnostics: when False, omits client-facing `cue_terms` and
            `suppressed` telemetry. Retrieval snapshots still retain both for
            dashboard and feedback diagnostics.
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
        cue_terms         – extracted query terms (when diagnostics enabled)
        suppressed        – gate rejection counts (when diagnostics enabled)
        activation_trace  – full trace (only when mode="debug")
    """
    if task is not None:
        if query and query.strip() != task.strip():
            raise ValueError("query and task must match when both are supplied")
        query = task
    if not query or not query.strip():
        raise ValueError("task must be nonblank")
    query = query.strip()
    if goal and initial_goal and goal.strip() != initial_goal.strip():
        raise ValueError("goal and initial_goal must match when both are supplied")
    resolved_goal = (initial_goal or goal or "").strip() or None
    resolved_context = normalize_facets(
        task_context if task_context is not None else retrieval_context,
        "task_context",
    )
    cue_situation = (
        resolved_context
        if task_context is not None or retrieval_context is not None
        else situation or {}
    )
    continuity_id = continuity_id.strip() if continuity_id else None
    situation = situation or {}
    requirements = requirements or []
    topics = topics or []
    entities = entities or []

    continuity_state: str | None = None
    if manage_continuity:
        if scope is None:
            raise ValueError("scope is required for managed continuity")
        continuity = resolve_continuity(
            eng.db,
            scope_id=scope.strip(),
            supplied_id=continuity_id,
            integration=continuity_integration,
            client_identity=continuity_client_identity,
        )
        continuity_id, continuity_state = continuity.continuity_id, continuity.state

    if session_id is None:
        session_id = eng.session_start(
            agent=agent,
            scope=scope,
            goal=resolved_goal,
            initial_goal=resolved_goal,
            retrieval_context=resolved_context,
            task_context=resolved_context,
            continuity_id=continuity_id,
        )

    _brief_kwargs: dict[str, Any] = {}
    if min_relevance is not None:
        _brief_kwargs["min_relevance"] = min_relevance
    if graph_channels is not None:
        _brief_kwargs["graph_channels"] = graph_channels
    if min_neighbor_relevance is not None:
        _brief_kwargs["min_neighbor_relevance"] = min_neighbor_relevance

    # A strict core stays intentionally tiny on every activation.  The start
    # policy may add separately-selected context below; it never widens this
    # base top-K or its relevance floor.
    brief = eng.context_brief(
        query=query,
        scope=scope,
        goal=resolved_goal,
        task_type=task_type,
        situation=cue_situation,
        requirements=requirements,
        topics=topics,
        entities=entities,
        mode=mode,
        limit=limit,
        include_peripheral=include_peripheral,
        **_brief_kwargs,
    )

    scope_id = scope.strip() if scope else None
    core_activation = max((float(item.activation) for item in brief.items), default=0.0)
    reinstatement_ceiling = (
        _CONTEXT_REINSTATEMENT_CONTEXTUAL_MAX_CORE_ACTIVATION
        if resolved_context
        else _CONTEXT_REINSTATEMENT_MAX_CORE_ACTIVATION
    )
    if continuity_state == "started" and core_activation < reinstatement_ceiling:
        # Fetch a larger *eligible* pool using the same strict-scope, active
        # and relevance gates.  Selection is MMR-diversified against the core,
        # rather than a flat [:5] slice.
        candidate_brief = eng.context_brief(
            query=query,
            scope=scope,
            goal=resolved_goal,
            task_type=task_type,
            situation=cue_situation,
            requirements=requirements,
            topics=topics,
            entities=entities,
            mode=mode,
            limit=5,
            include_peripheral=False,
            **_brief_kwargs,
        )
        selected = _select_context_reinstatement(
            candidates=candidate_brief.items,
            core=brief.items,
            scope_id=scope_id,
            max_items=3,
        )
        if selected:
            brief = type(brief)(
                items=list(brief.items) + selected,
                rendered=brief.rendered,
                cue_terms=brief.cue_terms,
                suppressed=brief.suppressed,
                activation_trace=brief.activation_trace,
            )

    context_id = f"ctx_{uuid.uuid4().hex[:12]}"
    scope_kind = scope.split(":", 1)[0] if scope and ":" in scope else None
    # With a scope, cold start means "this scope has no memories yet". Without
    # one, it must fall back to "the whole DB has no memories yet" rather than
    # hardcoding False -- a scopeless activate() on a genuinely empty DB is
    # still a first-ever encounter, not a normal retrieval.
    cold_start = eng.schemas.count_by_scope(scope_id) == 0 if scope_id else eng.schemas.count() == 0
    scope_warning = _scope_fragmentation_warning(eng, scope_id) if cold_start and scope_id else None

    procedure_hits = retrieve_procedures(
        load_procedures(eng.db.connect(), scope=scope_id),
        query=query,
        retrieval_context=resolved_context,
        encoder=getattr(eng, "encoder", None),
    )
    _schema_items = []
    for index, item in enumerate(brief.items):
        next_score = brief.items[index + 1].activation if index + 1 < len(brief.items) else None
        _schema_items.append(
            {
                "id": f"sch_{item.schema.id}",
                "score": item.activation,
                "topical_relevance": item.activation,
                "final_rank_score": item.activation,
                "score_margin": item.activation - next_score if next_score is not None else None,
                "reason": item.reason,
                "pathway": _pathway_for(item),
            }
        )
    _internal = _retrieval_exposure_snapshot(
        schemas=_schema_items,
        procedures=procedure_hits,
    )
    _filtered = [
        {
            "memory_id": f"sch_{t.schema_id}",
            "activation": t.activation,
            "reason": t.reason,
        }
        for t in brief.activation_trace
        if not t.admitted and not _is_scope_rejection(t.reason)
    ]
    _shadow_candidates = [
        (item.schema.id, item.activation, _pathway_for(item)) for item in brief.items
    ] + [
        (trace.schema_id, trace.activation, "direct")
        for trace in brief.activation_trace
        if not trace.admitted and not _is_scope_rejection(trace.reason)
    ]
    _shadow = _shadow_access_traces(
        eng,
        candidates=_shadow_candidates,
        query=query,
        goal=resolved_goal,
        task_type=task_type,
        situation=cue_situation,
        requirements=requirements,
        topics=topics,
        entities=entities,
        scope_id=scope_id,
    )
    _internal["shadow_access_traces"] = _shadow
    eng.record_context_recall(
        context_id=context_id,
        session_id=session_id,
        scope_id=scope_id,
        scope_kind=scope_kind,
        query=query,
        goal=resolved_goal,
        task_type=task_type,
        situation=cue_situation,
        requirements=requirements,
        mode=mode,
        limit=limit,
        topics=topics,
        entities=entities,
        cue_terms=brief.cue_terms,
        suppressed=brief.suppressed,
        response=_internal,
        filtered_items=_filtered,
        retrieval_policy_version="continuity-v1" if continuity_state else "strict-v9",
        continuity_state=continuity_state,
    )

    result: dict[str, Any] = {
        "retrieval_id": context_id,
        "session_id": session_id,
        "rendered": brief.rendered,
        "cold_start": cold_start,
    }
    if continuity_state:
        result["continuity_id"] = continuity_id
        result["continuity_state"] = continuity_state
        result["retrieval_policy_version"] = "continuity-v1"
    if include_diagnostics or mode == "debug":
        result["cue_terms"] = brief.cue_terms
        result["suppressed"] = brief.suppressed
    if scope_warning:
        result["scope_warning"] = scope_warning
    if include_schemas or mode == "debug":
        result["schemas"] = [
            {
                "id": f"sch_{item.schema.id}",
                "text": str(item.schema.content_text or ""),
                "activation": round(min(1.0, max(0.0, item.activation)), 4),
                "reason": item.reason,
                "pathway": _pathway_for(item),
                "source_kind": str((item.schema.facets or {}).get("source_kind", "")),
                "source_provenance": (item.schema.facets or {}).get("source_provenance", {}),
                "scope_id": item.schema.scope_id,
            }
            for item in brief.items
        ]
    if mode == "debug":
        result["activation_trace"] = [asdict(t) for t in brief.activation_trace]
        result["shadow_access_traces"] = _shadow
    result["procedures"] = procedure_hits
    return result


def remember(
    eng: SlowaveEngine,
    *,
    content: str,
    memory_type: str = "decision",
    scope: str | None = None,
    session_id: str | None = None,
    provenance: dict[str, Any] | None = None,
    occurred_at: int | None = None,
) -> dict[str, Any]:
    """Encode a durable typed claim.

    Returns:
        stored       – True on success
        memory_id    – stable sch_N identifier
        disposition  – created or matched (reconsolidated is reserved for a
                       future synchronous signal; current reconsolidation is async)
        type          – canonical stored type
        scope         – confirmed isolation boundary
        source_event_id – source provenance event
    """
    if not content or not str(content).strip():
        return {
            "stored": False,
            "skipped": True,
            "reason": "content is empty",
            "scope": scope,
        }
    rid = eng.remember(
        content=content,
        type=memory_type,
        scope=scope,
        session_id=session_id,
        provenance=provenance,
        occurred_at=occurred_at,
    )
    schema_id = rid.schema_id if hasattr(rid, "schema_id") and rid.schema_id else None
    return {
        "stored": True,
        "memory_id": f"sch_{schema_id}" if schema_id else None,
        "disposition": getattr(rid, "disposition", "created"),
        "type": memory_type,
        "scope": scope,
        "source_event_id": f"evt_{rid}",
    }


def recall(
    eng: SlowaveEngine,
    *,
    query: str,
    session_id: str | None = None,
    scope: str | None = None,
    mode: str = "default",
    top_k: int = DEFAULT_RECALL_TOP_K,
    evidence: bool = False,
    min_relevance: float | None = None,
    graph_channels: str | None = None,
    min_neighbor_relevance: float | None = None,
    task_context: dict[str, Any] | None = None,
    retrieval_context: dict[str, Any] | None = None,
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
    session_context: dict[str, Any] = {}
    if session_id is not None:
        session = (
            eng.db.connect()
            .execute(
                "SELECT scope_id, ended_ts, task_context_json FROM sessions WHERE id = ?",
                (session_id,),
            )
            .fetchone()
        )
        if session is None:
            raise ValueError(f"unknown session_id: {session_id}")
        if session["ended_ts"] is not None:
            raise ValueError(f"session is already ended: {session_id}")
        if session["scope_id"] != scope:
            raise ValueError("session_id and scope do not match")
        session_context = json.loads(session["task_context_json"] or "{}")
    context_delta = normalize_facets(
        task_context if task_context is not None else retrieval_context,
        "task_context",
    )
    session_context.update(context_delta)
    retrieval_context = session_context
    effective_query = " ".join(
        part
        for part in (query, json.dumps(retrieval_context, ensure_ascii=False, sort_keys=True))
        if part and part != "{}"
    )
    result = eng.recall(
        effective_query,
        top_k=top_k,
        evidence=evidence,
        scope=scope,
        mode=mode,
        min_relevance=min_relevance,
        graph_channels=graph_channels,
        min_neighbor_relevance=min_neighbor_relevance,
    )
    if session_id is not None and task_context is not None:
        eng.db.connect().execute(
            "UPDATE sessions SET task_context_json = ? WHERE id = ?",
            (json.dumps(session_context, ensure_ascii=False, sort_keys=True), session_id),
        )
        eng.db.connect().commit()
    recall_id = f"rec_{uuid.uuid4().hex[:12]}"
    procedure_hits = retrieve_procedures(
        load_procedures(eng.db.connect(), scope=scope),
        query=query,
        retrieval_context=retrieval_context,
        encoder=getattr(eng, "encoder", None),
    )
    _internal = _retrieval_exposure_snapshot(
        schemas=[
            {
                "id": f"sch_{s.id}",
                "score": result.schema_activations.get(s.id),
                "pathway": "direct",
            }
            for s in result.schemas
        ],
        related_schemas=[
            {
                "id": f"sch_{s.id}",
                "score": result.schema_activations.get(s.id),
                "pathway": "graph",
            }
            for s in result.related_schemas
        ],
        procedures=procedure_hits,
    )
    _internal["shadow_access_traces"] = _shadow_access_traces(
        eng,
        candidates=[
            (s.id, float(result.schema_activations.get(s.id, 0.0)), "direct")
            for s in result.schemas
        ]
        + [
            (s.id, float(result.schema_activations.get(s.id, 0.0)), "graph")
            for s in result.related_schemas
        ],
        query=query,
        goal=None,
        task_type=None,
        situation=retrieval_context,
        requirements=None,
        topics=None,
        entities=None,
        scope_id=scope,
    )
    eng.record_retrieval(
        retrieval_id=recall_id,
        retrieval_type="recall",
        session_id=session_id,
        query=query,
        scope_id=scope,
        scope_kind=scope.split(":", 1)[0] if scope and ":" in scope else None,
        situation=retrieval_context,
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
            "source_kind": str((s.facets or {}).get("source_kind", "")),
            "source_provenance": (s.facets or {}).get("source_provenance", {}),
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
            "source_kind": str((s.facets or {}).get("source_kind", "")),
            "source_provenance": (s.facets or {}).get("source_provenance", {}),
            "status": s.status,
            "generalization_stage": s.generalization_stage,
            "via": result.related_schema_relations.get(s.id, []),
        }
        for s in result.related_schemas
    ]
    out = {
        "retrieval_id": recall_id,
        "memories": memories,
        "related_memories": related_memories,
        "episodes": result.episode_texts,
        "raw_events": result.raw_events,
    }
    out["procedures"] = procedure_hits
    return out


def feedback(
    eng: SlowaveEngine,
    *,
    retrieval_id: str,
    memory_feedback: list[dict[str, Any]] | None = None,
    procedure_feedback: list[dict[str, Any]] | None = None,
    retrieval_quality: str | None = None,
    missing: list[str] | None = None,
    coverage: str = "partial",
) -> dict[str, Any]:
    """Record append-only v9 feedback without task-outcome coupling."""
    return eng.feedback(
        retrieval_id=retrieval_id,
        memory_feedback=memory_feedback,
        procedure_feedback=procedure_feedback,
        retrieval_quality=retrieval_quality,
        missing=missing,
        coverage=coverage,
    )


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
    used_procedure_ids: list[str] | None = None,
    irrelevant_procedure_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Internal CLI compatibility; v9 does not register this as an MCP tool."""
    return eng.retrieval_feedback(
        retrieval_id=retrieval_id,
        feedback=feedback,
        outcome=outcome,
        used_memory_ids=used_memory_ids,
        irrelevant_memory_ids=irrelevant_memory_ids,
        stale_memory_ids=stale_memory_ids,
        wrong_memory_ids=wrong_memory_ids,
        used_procedure_ids=used_procedure_ids,
        irrelevant_procedure_ids=irrelevant_procedure_ids,
    )


def commit(
    eng: SlowaveEngine,
    *,
    session_id: str,
    outcome: str = "unknown",
    steps: list[str] | None = None,
    final_goal: str | None = None,
    outcome_summary: str | None = None,
    procedure: dict[str, Any] | None = None,
    procedure_uses: list[dict[str, Any]] | None = None,
    verification: dict[str, Any] | None = None,
    trajectory: list[dict[str, Any]] | None = None,
    provenance: dict[str, Any] | None = None,
    enforce_feedback: bool = False,
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
    if not eng.raw_log.session_exists(session_id):
        raise ValueError(f"unknown session_id: {session_id}")
    if outcome not in {"success", "partial", "failure", "unknown"}:
        raise ValueError("outcome must be success, partial, failure, or unknown")
    if enforce_feedback and outcome == "unknown":
        raise ValueError("normal commit outcome must be success, partial, or failure")
    if enforce_feedback and not (final_goal or "").strip():
        raise ValueError("final_goal is required")
    if enforce_feedback and not (outcome_summary or "").strip():
        raise ValueError("outcome_summary is required")
    verification = verification or {"status": "unverified", "summary": "No verification supplied"}
    if verification.get("status") not in {"verified", "partially_verified", "unverified"}:
        raise ValueError("verification.status must be verified, partially_verified, or unverified")
    if not str(verification.get("summary", "")).strip():
        raise ValueError("verification.summary is required")
    if steps and procedure:
        raise ValueError("use either legacy steps or procedure.steps, not both")
    already_ended = eng.raw_log.is_session_ended(session_id)
    session_row = (
        eng.db.connect()
        .execute("SELECT feedback_status FROM sessions WHERE id = ?", (session_id,))
        .fetchone()
    )
    prior_feedback_status = str(session_row["feedback_status"] or "pending")
    if enforce_feedback and not already_ended:
        outstanding = eng._feedback.feedback_events.incomplete_for_session(session_id)
        if outstanding:
            raise IncompleteFeedbackError(outstanding)
    normalized_procedure = validate_procedure(procedure)
    normalized_procedure_uses = validate_procedure_uses(procedure_uses)
    normalized_trajectory = validate_trajectory(trajectory)
    # The trajectory is a narration of the agent's *task*; drop any entries
    # that instead describe Slowave's own lifecycle operations, so they never
    # enter raw_events as episodic ``experience``. Surface the drop so a
    # non-conforming client gets feedback to stop tracking lifecycle steps.
    filtered_trajectory, filtered_lifecycle_count = filter_lifecycle_trajectory(
        normalized_trajectory
    )
    normalized_trajectory = filtered_trajectory
    if steps and not already_ended:
        for step_text in steps:
            eng.event_append(
                session_id=session_id,
                type="step",
                content=step_text.strip(),
            )
    if normalized_trajectory and not already_ended:
        for index, item in enumerate(normalized_trajectory):
            eng.event_append(
                session_id=session_id,
                type=f"trajectory:{item['kind']}",
                content=item["summary"],
                metadata={
                    "sequence": index,
                    "status": item.get("status", "unknown"),
                    "provenance": {
                        **(provenance or {}),
                        "source_kind": "agent_inference",
                        "observed": False,
                    },
                },
            )
    metadata = {
        "contract_version": 2,
        "final_goal": (final_goal or "").strip() or None,
        "outcome": outcome,
        "outcome_summary": (outcome_summary or "").strip() or None,
        "procedure": normalized_procedure,
        "procedure_uses": normalized_procedure_uses,
        "verification": verification,
        "trajectory_event_count": len(normalized_trajectory),
        "provenance": provenance or {},
    }
    existing_completion = (
        eng.db.connect()
        .execute(
            "SELECT id FROM raw_events "
            "WHERE session_id = ? AND type = 'task_complete' "
            "ORDER BY id LIMIT 1",
            (session_id,),
        )
        .fetchone()
    )
    if existing_completion is None:
        eng.event_append(
            session_id=session_id,
            type="task_complete",
            content=f"outcome={outcome}",
            metadata=metadata,
            memory_role="procedural_evidence",
        )
    else:
        conn = eng.db.connect()
        conn.execute(
            "UPDATE raw_events SET content = ?, metadata_json = ?, embedding = NULL, dim = NULL "
            "WHERE id = ?",
            (
                f"outcome={outcome}",
                json.dumps(
                    {**metadata, "memory_role": "procedural_evidence"},
                    separators=(",", ":"),
                    sort_keys=True,
                ),
                int(existing_completion["id"]),
            ),
        )
        conn.commit()
    final_feedback_status = (
        prior_feedback_status
        if already_ended and prior_feedback_status == "incomplete"
        else ("complete" if enforce_feedback else "incomplete")
    )
    result = eng.session_end(
        session_id,
        consolidate=False,
        outcome=outcome,
        final_goal=(final_goal or "").strip() or None,
        outcome_summary=(outcome_summary or "").strip() or None,
        verification=verification,
        feedback_status=final_feedback_status,
    )
    out: dict[str, Any] = {
        "session_id": session_id,
        "episodes_formed": result.get("episodes_formed", 0),
        "committed": True,
        "outcome": outcome,
        "feedback_status": final_feedback_status,
        "verification_status": verification["status"],
        "operation": "updated" if already_ended else "closed",
    }
    if filtered_lifecycle_count:
        out["trajectory_lifecycle_filtered"] = filtered_lifecycle_count
    if result.get("already_ended"):
        out["already_ended"] = True

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


def validate_trajectory(trajectory: list[dict[str, Any]] | None) -> list[dict[str, str]]:
    """Validate a bounded model-supplied executed-attempt trace.

    Integrations own provenance; callers can describe actions and observations
    but cannot claim a source identity or observed status for those descriptions.
    """
    if trajectory is None:
        return []
    if not isinstance(trajectory, list) or len(trajectory) > 32:
        raise ValueError("trajectory must be a list of at most 32 entries")
    normalized: list[dict[str, str]] = []
    for item in trajectory:
        if not isinstance(item, dict) or not set(item) <= {"kind", "summary", "status"}:
            raise ValueError("trajectory entries may contain only kind, summary, and status")
        if item.get("kind") not in {"action", "observation"}:
            raise ValueError("trajectory kind must be action or observation")
        summary = item.get("summary")
        if not isinstance(summary, str) or not summary.strip() or len(summary) > 1000:
            raise ValueError("trajectory summary must be nonblank and at most 1000 characters")
        status = item.get("status", "unknown")
        if status not in {"started", "succeeded", "failed", "unknown"}:
            raise ValueError("trajectory status must be started, succeeded, failed, or unknown")
        normalized.append({"kind": item["kind"], "summary": summary.strip(), "status": status})
    return normalized


def filter_lifecycle_trajectory(
    trajectory: list[dict[str, str]],
) -> tuple[list[dict[str, str]], int]:
    """Drop trajectory entries that describe Slowave's own lifecycle operations.

    The trajectory contract is a narration of the agent's *task* behavior.
    Lifecycle bookkeeping ("Activated the Slowave session.", "Committed the
    session.") describes the agent operating Slowave, not the task, and has no
    semantic utility — it must never be stored as episodic ``experience``.

    Returns ``(kept, dropped_count)``. Matching is the same narrow,
    server-owned-vocabulary detector used by ``IngestService.form_episodes``,
    so a genuine task observation is never dropped.
    """
    kept: list[dict[str, str]] = []
    dropped = 0
    for item in trajectory:
        if is_slowave_lifecycle(f"trajectory:{item['kind']}", item["summary"]):
            dropped += 1
        else:
            kept.append(item)
    return kept, dropped


class IncompleteFeedbackError(ValueError):
    def __init__(self, outstanding: list[dict[str, Any]]):
        super().__init__("feedback is incomplete")
        self.outstanding = outstanding


def stats(eng: SlowaveEngine) -> dict[str, Any]:
    """Return system counts: episodes, prototypes, schemas, procedures, edges."""
    return eng.stats()
