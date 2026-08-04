"""Shared MCP tool registration for Slowave.

Provides a single ``register_tools(mcp, build_engine)`` function that attaches
all 6 tools to any FastMCP instance.  Both the stdio server (server.py) and
the HTTP daemon (http_server.py) call this function so there is no
duplication of tool logic.

Tools registered (5 cognitive-cycle verbs + 1 ops tool):
  activate, remember, recall, reinforce, commit, stats

remember and reinforce each also accept an optional `items` list for
batching several calls into one round trip (see their docstrings) — this
does not add new tool names, just an alternate parameter shape on the
existing two.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable

from mcp.server.fastmcp import FastMCP

import slowave.ops as ops
from slowave.core.config import DEFAULT_RECALL_TOP_K
from slowave.mcp import session_resolver

log = logging.getLogger(__name__)

# Keys stored in schema facets that are internal to the retrieval engine.
_INTERNAL_FACET_KEYS: frozenset[str] = frozenset({"vsa_vec"})


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


async def _bg_log_event(eng, session_id: str, event_type: str, content: str) -> None:
    """Fire-and-forget: log a synthetic session event."""
    try:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: eng.event_append(
                session_id=session_id, type=event_type, content=content or "[empty]"
            ),
        )
    except Exception as e:
        log.warning("_bg_log_event failed: %s", e)


def register_tools(mcp: FastMCP, build_engine: Callable) -> None:
    """Register all 6 Slowave cognitive-cycle tools onto *mcp*.

    Args:
        mcp: A FastMCP instance (stdio or HTTP).
        build_engine: Callable(disable_encoder=False) -> SlowaveEngine.
                      Must be the process-local cached version.
    """

    @mcp.tool(name="slowave_activate")
    async def slowave_activate(
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
        include_peripheral: bool = True,
        include_schemas: bool = True,
    ) -> dict[str, Any]:
        """Prime working memory with relevant context. Opens an implicit session.

        Call this once at the beginning of every task. Spreading activation surfaces
        relevant memories and procedures, and opens a server-side session so you
        never need to call session_start manually.

        The cognitive cycle:
            1. slowave_activate(query, scope, goal, task_type)  <- start here
            2. slowave_remember(content, type, scope)           <- for durable facts
            3. slowave_recall(query)                            <- mid-task lookup
            4. slowave_reinforce(retrieval_id, feedback, ...)   <- after using memories
            5. slowave_commit(scope, outcome)                   <- close the task

        Args:
            query: current task description (required).
            scope: STRONGLY RECOMMENDED for any project work (e.g. \"project:my-repo\").
                   Omitting scope causes memories from all scopes to bleed into retrieval.
                   Use \"project:<name>\" for project-scoped work, \"user:<id>\" for user profiles.
            goal: 3-6 word verb-noun phrase, e.g. \"implement oauth login\".
            task_type: category, e.g. \"coding\", \"debugging\".
            mode: strict_scope (default), broad, or debug.
                  When scope is set, strict_scope hard-isolates to that scope (+ global/profile).
                  When scope is None, strict_scope is equivalent to default — no behaviour change.
            limit: max schemas returned (default 8).
            include_peripheral: set False to drop the trailing salience-filled
                   "exploration slots" (tagged "(peripheral)") from the brief.
            include_schemas: set False to omit the structured `schemas` array
                   and return only `rendered` — cheaper if you only read the prose.

        Returns:
            retrieval_id: pass to slowave_reinforce.
            session_id: informational; used automatically by remember/commit.
            rendered: human-readable memory brief.
            schemas: [{id, text, activation, reason, source_kind}, ...] (omitted
                     when include_schemas=False).
        """
        try:
            eng = build_engine(disable_encoder=not bool(query or topics or entities))
            # Resolve or open a session via the MCP implicit-session resolver.
            sid = session_resolver.resolve(scope)
            result = ops.activate(
                eng,
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
                session_id=sid,
                agent="mcp",
                include_peripheral=include_peripheral,
                include_schemas=include_schemas,
            )
            session_resolver.bind(scope, result["session_id"])
            # MCP-specific: add cold-start hint text and fire a synthetic event.
            if result["cold_start"]:
                scope_id = scope.strip() if scope else None
                scope_label = (
                    scope_id or "global (no scope set — consider passing scope='project:<name>')"
                )
                hint = (
                    f"[cold start] No memories found for scope '{scope_label}'.\n"
                    "Recommended on cold start:\n"
                    "  1. Check for these files (in order, stop at first found): "
                    "CLAUDE.md, README.md, AGENTS.override.md, AGENTS.md. Read the first one that exists.\n"
                    "  2. For each fact in that document, ask: would this be useful in any future "
                    "interaction within this scope? Can it be inferred as durable and critical — "
                    "something a future session could not assume without it? "
                    "If yes to both, call slowave_remember() — one call per fact, not one call per group. "
                    "Exhaust the document before moving on.\n"
                    "  3. If neither file exists, apply the same questions to what is visible "
                    "from the current request.\n"
                    "  4. Then respond to the user."
                )
                result["rendered"] = f"{result['rendered']}\n\n{hint}".lstrip("\n")
                result["suggested_actions"] = ["remember_project_facts"]
                result["cold_start_hints"] = (
                    "Memory is empty for this scope. "
                    "Consider reading CLAUDE.md, README.md, AGENTS.override.md, or AGENTS.md (whichever exists first). "
                    "For each fact ask: would this be useful in any future interaction within this scope? "
                    "Can it be inferred as durable and critical — something a future session could not assume without it? "
                    "If yes to both, call slowave_remember() — one call per fact, not one call per group. "
                    "Exhaust the document before moving on. "
                    "If neither file exists, apply the same questions to what is visible from the current request. "
                    "Then respond to the user."
                )
            asyncio.create_task(_bg_log_event(eng, result["session_id"], "context_query", query))
            return result
        except Exception as e:
            log.error("slowave_activate failed: %s", e, exc_info=True)
            return {
                "retrieval_id": None,
                "session_id": None,
                "rendered": "",
                "schemas": [],
                "error": str(e),
            }

    @mcp.tool(name="slowave_recall")
    async def slowave_recall(
        query: str,
        top_k: int = DEFAULT_RECALL_TOP_K,
        evidence: bool = False,
        scope: str | None = None,
        mode: str = "default",
        min_relevance: float | None = None,
    ) -> dict[str, Any]:
        """Semantic retrieval: bring relevant memories into working memory.
        Use for deliberate mid-task lookups when you need specific historical
        context beyond what activate surfaced.
        WARNING: omitting scope returns memories from ALL projects. Always pass
        scope=\"project:<name>\" when working within a specific project.
        Args:
            query: natural-language query.
            top_k: max memories returned (default 20).
            evidence: if true, include source raw events for provenance.
            scope: optional scope filter (e.g. \"project:myrepo\"). Recommended.
            mode: \"default\" (active only), \"strict_scope\" (scope-hard-filtered),
                  \"broad\" (active + needs_review), \"debug\" (all statuses).
            min_relevance: optional relevance floor override. Omit to use the
                built-in default (rejects near-noise matches instead of always
                returning top_k); pass 0.0 to disable the floor for this call.
        Returns:
            retrieval_id: pass to slowave_reinforce after using memories.
            memories: list of {id, content_text, activation, scope_id, ...}.
            related_memories: schema_relations-linked memories (relates_to a
                memory in `memories`) surfaced via spreading activation, NOT
                counted toward top_k -- worth a glance but not guaranteed to
                directly match the query the way `memories` does.
        """
        try:
            eng = build_engine()
            return ops.recall(
                eng,
                query=query,
                top_k=top_k,
                evidence=evidence,
                scope=scope,
                mode=mode,
                min_relevance=min_relevance,
            )
        except Exception as e:
            log.error("slowave_recall failed: %s", e, exc_info=True)
            return {"retrieval_id": None, "memories": [], "error": str(e)}

    @mcp.tool(name="slowave_remember")
    async def slowave_remember(
        content: str | None = None,
        type: str = "decision",
        scope: str | None = None,
        session_id: str | None = None,
        items: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Explicitly encode a durable typed claim into long-term memory.
        Use for decisions, preferences, constraints, lessons, or any fact that
        should persist across sessions. If called after slowave_activate without
        a session_id, the implicit session is used automatically.
        Args:
            content: the claim, as a complete sentence. Omit when passing `items`.
            type: fact|preference|decision|constraint|procedure|task|
                  open_question|warning|lesson|artifact.
            scope: optional scope, e.g. \"project:my-repo\".
            session_id: optional; inferred from activate's implicit session if omitted.
            items: optional list of {content, type, scope, session_id} for encoding
                   several already-decided facts in one round trip (session-end/
                   retroactive cleanup only — each item is still embedded
                   independently, the "one fact per call" rule still applies per
                   item, this is a transport optimization not a way to bundle
                   facts into one embedding). When given, `content`/`type`/`scope`/
                   `session_id` above are ignored and the return shape is
                   {"results": [...]} instead of the single-item shape below.
        IMPORTANT: Use ONLY for durable knowledge that should persist across sessions.
        Do NOT store ephemeral task state — that belongs in session events.
        """
        eng = build_engine()
        if items is not None:
            results: list[dict[str, Any]] = []
            for item in items:
                item_type = item.get("type", "decision")
                item_scope = item.get("scope")
                try:
                    resolved_sid = item.get("session_id") or session_resolver.resolve(item_scope)
                    results.append(
                        ops.remember(
                            eng,
                            content=item.get("content", ""),
                            memory_type=item_type,
                            scope=item_scope,
                            session_id=resolved_sid,
                        )
                    )
                except Exception as e:
                    log.error("slowave_remember (batch item) failed: %s", e, exc_info=True)
                    results.append(
                        {"stored": False, "error": str(e), "type": item_type, "scope": item_scope}
                    )
            return {"results": results}
        try:
            resolved_sid = session_id or session_resolver.resolve(scope)
            return ops.remember(
                eng, content=content or "", memory_type=type, scope=scope, session_id=resolved_sid
            )
        except Exception as e:
            log.error("slowave_remember failed: %s", e, exc_info=True)
            return {"stored": False, "error": str(e), "type": type, "scope": scope}

    @mcp.tool(name="slowave_reinforce")
    async def slowave_reinforce(
        retrieval_id: str | None = None,
        feedback: str = "useful",
        outcome: str = "unknown",
        used_memory_ids: list[str] | None = None,
        irrelevant_memory_ids: list[str] | None = None,
        stale_memory_ids: list[str] | None = None,
        wrong_memory_ids: list[str] | None = None,
        items: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Strengthen or suppress memories based on how useful they were.
        Call after using memories from slowave_activate or slowave_recall.
        goal, task_type, scope, session, situation, requirements are auto-derived
        from the original retrieval snapshot — you only supply the signal.
        Feedback: useful|partially_useful|irrelevant|stale|wrong|missing|too_much_context
        Outcome:  success|partial|failure|unknown
        Args:
            retrieval_id: from slowave_activate or slowave_recall response.
                          Omit when passing `items`.
            feedback: quality label for the retrieved memories.
            outcome: result of the downstream task.
            used_memory_ids: schema IDs that were actually relied on.
            irrelevant/stale/wrong_memory_ids: IDs needing penalty or review.
            items: optional list of {retrieval_id, feedback, outcome,
                   used_memory_ids, irrelevant_memory_ids, stale_memory_ids,
                   wrong_memory_ids} for reinforcing several retrievals in one
                   round trip (session-end/retroactive cleanup — e.g. patching
                   several retrievals you deferred reinforcing during a long
                   session). When given, the flat args above are ignored and
                   the return shape is {"results": [...]} instead of the
                   single-item shape below.
        """
        eng = build_engine(disable_encoder=True)
        if items is not None:
            results: list[dict[str, Any]] = []
            for item in items:
                item_rid = item.get("retrieval_id", "")
                try:
                    results.append(
                        ops.reinforce(
                            eng,
                            retrieval_id=item_rid,
                            feedback=item.get("feedback", "useful"),
                            outcome=item.get("outcome", "unknown"),
                            used_memory_ids=item.get("used_memory_ids"),
                            irrelevant_memory_ids=item.get("irrelevant_memory_ids"),
                            stale_memory_ids=item.get("stale_memory_ids"),
                            wrong_memory_ids=item.get("wrong_memory_ids"),
                        )
                    )
                except Exception as e:
                    log.error("slowave_reinforce (batch item) failed: %s", e, exc_info=True)
                    results.append({"error": str(e), "retrieval_id": item_rid})
            return {"results": results}
        try:
            return ops.reinforce(
                eng,
                retrieval_id=retrieval_id or "",
                feedback=feedback,
                outcome=outcome,
                used_memory_ids=used_memory_ids,
                irrelevant_memory_ids=irrelevant_memory_ids,
                stale_memory_ids=stale_memory_ids,
                wrong_memory_ids=wrong_memory_ids,
            )
        except Exception as e:
            log.error("slowave_reinforce failed: %s", e, exc_info=True)
            return {"error": str(e), "retrieval_id": retrieval_id}

    @mcp.tool(name="slowave_commit")
    async def slowave_commit(
        outcome: str | None = None,
        scope: str | None = None,
        session_id: str | None = None,
        steps: list[str] | None = None,
    ) -> dict[str, Any]:
        """Close the current task and trigger offline memory consolidation.

        Call at the end of every task. If skipped, the idle-session reaper closes
        the session after SLOWAVE_SESSION_IDLE_TIMEOUT seconds (default 3600).
        Args:
            outcome: task result — success|partial|failure|unknown.
            scope: scope used in activate (clears the implicit session binding).
            session_id: from activate response; inferred from scope if omitted.
            steps: ordered list of what you did this session. Each step is one
                   sentence describing a meaningful phase of work, e.g.
                   ["Ran full test suite (142 passed)",
                    "Built and pushed Docker image",
                    "Deployed to staging and health-checked"].
                   This is how Slowave learns procedural patterns — be honest
                   and complete, but keep each step to one sentence.
        Returns:
            session_id: the session that was closed.
            episodes_formed: number of episodic memories created.
            session_note: present when the session had several tool-use events
                but never called slowave_recall — informational, not a gate.
        """
        try:
            eng = build_engine(disable_encoder=True)
            resolved_sid = session_id or session_resolver.resolve(scope)
            if resolved_sid is None:
                return {
                    "session_id": None,
                    "episodes_formed": 0,
                    "warning": "no active session found",
                }
            outcome_str = outcome or "unknown"
            asyncio.create_task(
                _bg_log_event(eng, resolved_sid, "task_complete", f"outcome={outcome_str}")
            )
            await asyncio.sleep(0.05)
            result = ops.commit(eng, session_id=resolved_sid, outcome=outcome_str, steps=steps)
            session_resolver.clear(scope)
            return result
        except Exception as e:
            log.error("slowave_commit failed: %s", e, exc_info=True)
            return {"session_id": session_id, "episodes_formed": 0, "error": str(e)}

    @mcp.tool(name="slowave_stats")
    async def slowave_stats() -> dict[str, Any]:
        """Return system counts: episodes, prototypes, schemas, edges."""
        try:
            eng = build_engine(disable_encoder=True)
            return ops.stats(eng)
        except Exception as e:
            log.error("slowave_stats failed: %s", e, exc_info=True)
            return {"error": str(e), "episodes": 0, "schemas": 0, "procedures": 0}
