"""Slowave CLI entry point.

Provides the agent-facing surface: session/event/remember/recall/context/show.

Design goals:
- Every command prints either JSON or a compact human-readable form.
- JSON mode is selected with --json (recommended for agent integrations).
- The CLI never calls an LLM: ingest, consolidation, and recall are local.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import asdict
from typing import Any

# macOS: avoid OpenMP runtime crashes when FAISS, torch, and tokenizers coexist.
# `python -m slowave` sets these in __main__, but the installed console script
# enters here directly.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

import logging as _logging

_logging.getLogger("huggingface_hub").setLevel(_logging.ERROR)
_logging.getLogger("onnxruntime").setLevel(_logging.ERROR)
_logging.getLogger("transformers").setLevel(_logging.ERROR)

import click

import slowave.ops as ops
from slowave.cli.output import safe_emoji
from slowave.cli.setup import setup_cmd
from slowave.core.config import DEFAULT_RECALL_TOP_K, SlowaveConfig
from slowave.core.engine import SlowaveEngine
from slowave.core.paths import default_db_path
from slowave.lifecycle import LIFECYCLE_VERSION
from slowave.symbolic.encoder import EncoderConfig

DEFAULT_DB = "__DEFAULT_DB__"


def _ensure_db_dir(path: str) -> None:
    d = os.path.dirname(os.path.abspath(path))
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def _resolve_db_path(db: str) -> str:
    if db == DEFAULT_DB:
        return default_db_path()
    return os.path.expanduser(db)


def _build_engine(db: str, *, disable_encoder: bool = False) -> SlowaveEngine:
    db = _resolve_db_path(db)
    _ensure_db_dir(db)
    cfg = SlowaveConfig(
        db_path=db,
        dim=384,
        encoder=EncoderConfig(),
        disable_encoder=disable_encoder,
    )
    return SlowaveEngine(cfg)


def _print(obj: Any, as_json: bool) -> None:
    if as_json:
        click.echo(json.dumps(obj, ensure_ascii=False, indent=2, default=str))
    else:
        click.echo(
            obj
            if isinstance(obj, str)
            else json.dumps(obj, ensure_ascii=False, indent=2, default=str)
        )


@click.group()
@click.version_option(package_name="slowave")
@click.option(
    "--db",
    default=DEFAULT_DB,
    show_default="SLOWAVE_DB or ~/.slowave/slowave.db",
    help="SQLite db path override.",
)
@click.option("--json", "as_json", is_flag=True, help="JSON output.")
@click.pass_context
def cli(ctx: click.Context, db: str, as_json: bool) -> None:
    """Slowave: brain-inspired memory for AI agents."""
    ctx.ensure_object(dict)
    ctx.obj["db"] = _resolve_db_path(db)
    ctx.obj["json"] = as_json


@cli.group()
def session() -> None:
    """Session lifecycle."""


@session.command("start")
@click.option("--agent", default="cline-tui")
@click.option("--scope", default=None, help="Scope id, e.g. 'project:my-repo' or 'domain:cooking'.")
@click.pass_context
def session_start(ctx: click.Context, agent: str, scope: str | None) -> None:
    eng = _build_engine(ctx.obj["db"])  # no LLM needed at start
    sid = eng.session_start(agent=agent, scope=scope)
    _print({"session_id": sid}, ctx.obj["json"])
    eng.close()


@session.command("end")
@click.argument("session_id")
@click.option(
    "--consolidate",
    is_flag=True,
    help="Also run replay + latent consolidation synchronously. "
    "Default: encode only; run 'slowave consolidate' separately.",
)
@click.pass_context
def session_end(ctx: click.Context, session_id: str, consolidate: bool) -> None:
    """End a session and encode events into episodic memories.

    Fast by default: no blocking. Use --consolidate only in scripts
    or tests. In production let the background worker handle consolidation.
    """
    eng = _build_engine(ctx.obj["db"])
    stats = eng.session_end(session_id, consolidate=consolidate)
    _print(stats, ctx.obj["json"])
    eng.close()


@cli.command("commit")
@click.argument("session_id")
@click.option(
    "--outcome",
    default="unknown",
    show_default=True,
    type=click.Choice(["success", "partial", "failure", "unknown"]),
    help="Task result.",
)
@click.option(
    "--step",
    "steps",
    multiple=True,
    help="What you did this session (repeatable, ordered).",
)
@click.pass_context
def commit_cmd(ctx: click.Context, session_id: str, outcome: str, steps: tuple[str, ...]) -> None:
    """Close a task session and encode events into episodic memories.

    MCP equivalent of slowave_commit.  Pass the session_id from 'slowave activate'.

    Use --step repeatedly to record what you did, e.g.
        slowave commit sess_abc123 --outcome success \
            --step "Ran full test suite" \
            --step "Built and pushed Docker image"
    """
    eng = _build_engine(ctx.obj["db"], disable_encoder=True)
    step_list = list(steps) if steps else None
    result = ops.commit(eng, session_id=session_id, outcome=outcome, steps=step_list)
    _print(result, ctx.obj["json"])
    eng.close()


@cli.command("event")
@click.option("--session", "session_id", required=True)
@click.option("--type", "type_", required=True)
@click.option("--content", required=True)
@click.pass_context
def event_append(ctx: click.Context, session_id: str, type_: str, content: str) -> None:
    """Append an event to a session."""
    eng = _build_engine(ctx.obj["db"])
    rid = eng.event_append(session_id=session_id, type=type_, content=content)
    _print({"event_id": rid}, ctx.obj["json"])
    eng.close()


@cli.command("remember")
@click.argument("content")
@click.option(
    "--type",
    "memory_type",
    default="decision",
    help="Memory type: fact|decision|lesson|warning|...",
)
@click.option("--scope", default=None, help="Scope id, e.g. 'project:my-repo'.")
@click.option("--session", "session_id", default=None, help="Session id to attach this memory to.")
@click.pass_context
def remember_cmd(
    ctx: click.Context, content: str, memory_type: str, scope: str | None, session_id: str | None
) -> None:
    """Explicitly remember a typed claim.

    MCP equivalent of slowave_remember.
    """
    eng = _build_engine(ctx.obj["db"])
    result = ops.remember(
        eng, content=content, memory_type=memory_type, scope=scope, session_id=session_id
    )
    _print(result, ctx.obj["json"])
    eng.close()


@cli.command("recall")
@click.argument("query")
@click.option("--scope", default=None, help="Scope filter, e.g. 'project:my-repo'. Recommended.")
@click.option(
    "--mode",
    default="default",
    show_default=True,
    type=click.Choice(["default", "strict_scope", "broad", "debug"]),
)
@click.option("--top-k", default=DEFAULT_RECALL_TOP_K, show_default=True)
@click.option("--evidence", is_flag=True, help="Include raw event citations.")
@click.pass_context
def recall_cmd(
    ctx: click.Context, query: str, scope: str | None, mode: str, top_k: int, evidence: bool
) -> None:
    """Recall memories relevant to a query.

    MCP equivalent of slowave_recall.
    """
    eng = _build_engine(ctx.obj["db"])
    payload = ops.recall(eng, query=query, scope=scope, mode=mode, top_k=top_k, evidence=evidence)
    if ctx.obj["json"]:
        _print(payload, True)
    else:
        click.echo(f"  retrieval_id: {payload['retrieval_id']}")
        # _format_recall_human expects schema dicts with int `id` and all Schema fields.
        # The ops layer returns compact dicts; fall back to a simple listing here.
        for m in payload["memories"]:
            click.echo(f"  [{m['id']}] {m['content_text'][:100]}  act={m['activation']:.3f}")
    eng.close()


def _sal_bar(sal: float, width: int = 8, scale: float = 80.0) -> str:
    """Render a compact salience bar.  sal is unbounded; normalise to [0,1] with a soft cap."""
    import math

    norm = 1.0 - math.exp(-sal / scale)
    filled = round(norm * width)
    return "█" * filled + "░" * (width - filled)


def _status_icon(status: str) -> str:
    return {"active": "🟢", "needs_review": "🟡", "superseded": "⚫", "contradicted": "🔴"}.get(
        status, "⚪"
    )


def _format_recall_human(payload: dict[str, Any]) -> None:
    schemas = payload.get("schemas", [])
    episodes = payload.get("episodes", [])
    raw_events = payload.get("raw_events", [])
    divider = "  " + "─" * 54

    # ── Schemas ────────────────────────────────────────────
    click.echo()
    click.echo("  📖  schemas")
    if not schemas:
        click.echo("  (none)")
    for s in schemas:
        sal = float(s.get("salience", 0.0))
        status = s.get("status", "active")
        n_ep = len(s.get("supporting_episode_ids", []))
        tags = s.get("tags") or []
        tag_str = "  #" + " #".join(tags) if tags else ""
        text = (s.get("content_text") or "").replace("\n", " ").strip()
        if len(text) > 120:
            text = text[:120] + "…"
        nr = "  🔔 needs review" if s.get("needs_review") else ""
        click.echo(divider)
        click.echo(
            f"  {_status_icon(status)}  sch_{s['id']}"
            f"  {_sal_bar(sal)}  sal={sal:.0f}"
            f"  {n_ep} ep{'' if n_ep == 1 else 's'}"
            f"{tag_str}{nr}"
        )
        click.echo(f"  {text}")

    # ── Episodes ───────────────────────────────────────────
    if episodes:
        click.echo()
        click.echo("  💬  episodes")
        click.echo(divider)
        for ep in episodes:
            sal = float(ep.get("salience", 0.0))
            ts = ep.get("ts") or ep.get("created_at") or 0
            # ts is a unix timestamp (seconds); convert to YYYY-MM-DD
            try:
                import datetime

                date = datetime.datetime.fromtimestamp(int(ts)).strftime("%Y-%m-%d")
            except Exception:
                date = str(ts)[:10] if ts else "—"
            text = (ep.get("content_text") or "").replace("\n", " ").strip()
            # Strip leading [YYYY-MM-DD] date prefix already shown in the date column
            import re as _re

            text = _re.sub(r"^\[\d{4}-\d{2}-\d{2}\]\s*", "", text)
            if len(text) > 100:
                text = text[:100] + "…"
            # Episodes have salience ~0–1 typically; use a tighter scale
            click.echo(f"  epi_{ep['id']}  {date}  {_sal_bar(sal, 6, scale=0.5)}  {text}")

    # ── Raw events ─────────────────────────────────────────
    if raw_events:
        click.echo()
        click.echo("  🗒   evidence")
        click.echo(divider)
        for r in raw_events:
            text = (r.get("content") or "").replace("\n", " ").strip()
            if len(text) > 100:
                text = text[:100] + "…"
            click.echo(f"  evt_{r['id']}  {r.get('type', '')}  {text}")

    click.echo()


@cli.command("activate")
@click.option("--query", required=True, help="Current task description.")
@click.option("--scope", default=None, help="Scope id, e.g. 'project:my-repo'.")
@click.option(
    "--goal", default=None, help="3-6 word verb-noun phrase, e.g. 'fix session reaper race'."
)
@click.option("--task-type", default=None, help="Category, e.g. 'coding' or 'debugging'.")
@click.option("--situation", default=None, help="JSON object with situational metadata.")
@click.option("--requirement", "requirements", multiple=True, help="Requirement cue; repeatable.")
@click.option("--topic", "topics", multiple=True, help="Topic cue; repeatable.")
@click.option("--entity", "entities", multiple=True, help="Entity cue; repeatable.")
@click.option(
    "--mode",
    default="strict_scope",
    show_default=True,
    type=click.Choice(["default", "strict_scope", "broad", "debug"]),
)
@click.option("--limit", default=8, show_default=True)
@click.pass_context
def activate_cmd(
    ctx: click.Context,
    query: str,
    scope: str | None,
    goal: str | None,
    task_type: str | None,
    situation: str | None,
    requirements: tuple[str, ...],
    topics: tuple[str, ...],
    entities: tuple[str, ...],
    mode: str,
    limit: int,
) -> None:
    """Prime working memory and open a task session.

    MCP equivalent of slowave_activate.  Returns retrieval_id + session_id;
    pass them to 'slowave reinforce' and 'slowave commit'.
    """
    eng = _build_engine(ctx.obj["db"])
    situation_obj = json.loads(situation) if situation else {}
    result = ops.activate(
        eng,
        query=query,
        scope=scope,
        goal=goal,
        task_type=task_type,
        situation=situation_obj,
        requirements=list(requirements),
        topics=list(topics),
        entities=list(entities),
        mode=mode,
        limit=limit,
    )
    if ctx.obj["json"]:
        _print(result, True)
    else:
        click.echo(f"  retrieval_id: {result['retrieval_id']}  session_id: {result['session_id']}")
        click.echo(result["rendered"] or "  (no memories yet)")
    eng.close()


@cli.command("context")
@click.option(
    "--scope", default=None, help="Generic scope id, e.g. project:slowave or domain:cooking."
)
@click.option(
    "--session",
    "session_id",
    default=None,
    help="Bind to existing session (for recording). Auto-starts one if --scope is set.",
)
@click.option("--query", default=None, help="Current task/chat cue for relevance gating.")
@click.option("--goal", default=None, help="Goal-oriented cue for context/procedure recall.")
@click.option(
    "--task-type", default=None, help="Broad activity type, e.g. writing/planning/debugging."
)
@click.option("--situation", default=None, help="JSON object with situational metadata.")
@click.option(
    "--requirement",
    "requirements",
    multiple=True,
    help="Requirement/condition cue; can be repeated.",
)
@click.option(
    "--application",
    default=None,
    help="Calling app/channel cue, e.g. chatbot or cline-tui.",
)
@click.option("--topic", "topics", multiple=True, help="High-level topic cue; can be repeated.")
@click.option("--entity", "entities", multiple=True, help="Salient entity cue; can be repeated.")
@click.option(
    "--mode",
    default="strict_scope",
    show_default=True,
    type=click.Choice(["default", "strict_scope", "broad", "debug"]),
)
@click.option("--limit", default=10, show_default=True)
@click.pass_context
def context_cmd(
    ctx: click.Context,
    scope: str | None,
    session_id: str | None,
    query: str | None,
    goal: str | None,
    task_type: str | None,
    situation: str | None,
    requirements: tuple[str, ...],
    application: str | None,
    topics: tuple[str, ...],
    entities: tuple[str, ...],
    mode: str,
    limit: int,
) -> None:
    """Return a gated working-memory brief for an agent/chatbot prompt.

    Mirrors slowave_activate: opens an implicit session when --scope is set,
    records the context retrieval for reinforce correlation, and returns a
    retrieval_id to pass to 'slowave reinforce'.
    """
    import uuid

    eng = _build_engine(ctx.obj["db"])
    situation_obj = json.loads(situation) if situation else {}

    # Open a session if scope is set but no session was supplied (mirrors MCP activate).
    active_session_id = session_id
    if active_session_id is None and scope is not None:
        active_session_id = eng.session_start(agent="cli", scope=scope, goal=goal)

    brief = eng.context_brief(
        query=query,
        scope=scope,
        goal=goal,
        task_type=task_type,
        situation=situation_obj,
        requirements=list(requirements),
        application=application,
        topics=list(topics),
        entities=list(entities),
        mode=mode,
        limit=limit,
    )

    context_id = f"ctx_{uuid.uuid4().hex[:12]}"
    scope_id = scope.strip() if scope else None
    scope_kind = scope.split(":", 1)[0] if scope and ":" in scope else None
    _internal = {
        "memory_ids": [f"sch_{item.schema.id}" for item in brief.items],
        "schemas": [
            {"id": f"sch_{item.schema.id}", "activation": item.activation} for item in brief.items
        ],
    }
    eng.record_context_recall(
        context_id=context_id,
        session_id=active_session_id,
        scope_id=scope_id,
        scope_kind=scope_kind,
        query=query,
        goal=goal,
        task_type=task_type,
        situation=situation_obj,
        requirements=list(requirements),
        mode=mode,
        limit=limit,
        cue_terms=brief.cue_terms,
        suppressed=brief.suppressed,
        response=_internal,
    )

    cold_start = eng.schemas.count_by_scope(scope_id) == 0 if scope_id else False

    if ctx.obj["json"]:
        _print(
            {
                "retrieval_id": context_id,
                "session_id": active_session_id,
                "cold_start": cold_start,
                "scope": scope,
                "query": query,
                "goal": goal,
                "task_type": task_type,
                "situation": situation_obj,
                "requirements": list(requirements),
                "application": application,
                "topics": list(topics),
                "entities": list(entities),
                "mode": mode,
                "rendered": brief.rendered,
                "cue_terms": brief.cue_terms,
                "suppressed": brief.suppressed,
                "schemas": [
                    {
                        "id": f"sch_{item.schema.id}",
                        "content_text": item.text,
                        "activation": round(min(1.0, max(0.0, item.activation)), 4),
                        "reason": item.reason,
                        "source_kind": str((item.schema.facets or {}).get("source_kind", "")),
                    }
                    for item in brief.items
                ],
                "procedures": [],
                "activation_trace": (
                    [asdict(t) for t in brief.activation_trace] if mode == "debug" else []
                ),
            },
            True,
        )
    else:
        click.echo(
            f"  retrieval_id: {context_id}"
            + (f"  session: {active_session_id}" if active_session_id else "")
        )
        click.echo("=== Working Memory Context ===")
        if not brief.items:
            click.echo("  (no memories yet)")
        for item in brief.items:
            s = item.schema
            label = s.facets.get("display_label", "") if isinstance(s.facets, dict) else ""
            label_str = f" [{label}]" if label else ""
            click.echo(
                f"  [sch_{s.id}]{label_str} {item.text}"
                f"  act={item.activation:.3f} status={s.status} sal={s.salience:.3f}"
                f" supports={len(s.supporting_episode_ids)}"
                f" tags={','.join(s.tags)}"
                f" reason={item.reason}" + ("  needs_review" if s.is_labile else "")
            )


@cli.command("show")
@click.argument("ref")
@click.pass_context
def show(ctx: click.Context, ref: str) -> None:
    """Show a schema/episode/event by ref (sch_NN, epi_NN, evt_NN)."""
    eng = _build_engine(ctx.obj["db"])
    if ref.startswith("sch_"):
        sid = int(ref[4:])
        try:
            s = eng.get_schema(sid)
            _print(asdict(s), ctx.obj["json"])
        except KeyError:
            _print({"error": "not found"}, ctx.obj["json"])
    elif ref.startswith("epi_"):
        eid = int(ref[4:])
        et = eng.episode_text.get(eid)
        _print(asdict(et) if et else {"error": "not found"}, ctx.obj["json"])
    elif ref.startswith("evt_"):
        eid = int(ref[4:])
        try:
            e = eng.raw_log.get(eid)
            _print(
                {
                    "id": e.id,
                    "session_id": e.session_id,
                    "ts": e.ts,
                    "type": e.type,
                    "content": e.content,
                    "metadata": e.metadata,
                },
                ctx.obj["json"],
            )
        except KeyError:
            _print({"error": "not found"}, ctx.obj["json"])
    else:
        _print({"error": f"unknown ref prefix: {ref}"}, ctx.obj["json"])
    eng.close()


@cli.command("reinforce")
@click.argument("retrieval_id")
@click.option(
    "--feedback",
    default="useful",
    show_default=True,
    type=click.Choice(
        [
            "useful",
            "partially_useful",
            "irrelevant",
            "stale",
            "wrong",
            "missing",
            "too_much_context",
        ]
    ),
    help="Quality label for the retrieved memories.",
)
@click.option(
    "--outcome",
    default="unknown",
    show_default=True,
    type=click.Choice(["success", "partial", "failure", "unknown"]),
    help="Result of the downstream task.",
)
@click.option(
    "--used",
    "used_memory_ids",
    multiple=True,
    metavar="SCH_ID",
    help="Schema IDs that were relied on (e.g. sch_5). Repeatable.",
)
@click.option(
    "--irrelevant",
    "irrelevant_memory_ids",
    multiple=True,
    metavar="SCH_ID",
    help="Schema IDs that were not relevant. Repeatable.",
)
@click.option(
    "--stale",
    "stale_memory_ids",
    multiple=True,
    metavar="SCH_ID",
    help="Schema IDs that are outdated. Repeatable.",
)
@click.option(
    "--wrong",
    "wrong_memory_ids",
    multiple=True,
    metavar="SCH_ID",
    help="Schema IDs that are factually wrong. Repeatable.",
)
@click.pass_context
def reinforce_cmd(
    ctx: click.Context,
    retrieval_id: str,
    feedback: str,
    outcome: str,
    used_memory_ids: tuple[str, ...],
    irrelevant_memory_ids: tuple[str, ...],
    stale_memory_ids: tuple[str, ...],
    wrong_memory_ids: tuple[str, ...],
) -> None:
    """Strengthen or suppress memories based on how useful they were.

    MCP equivalent of slowave_reinforce.
    Pass the retrieval_id from 'slowave activate' or 'slowave recall'.
    Schema IDs use the sch_N format returned by those commands.
    """
    eng = _build_engine(ctx.obj["db"], disable_encoder=True)
    result = ops.reinforce(
        eng,
        retrieval_id=retrieval_id,
        feedback=feedback,
        outcome=outcome,
        used_memory_ids=list(used_memory_ids) or None,
        irrelevant_memory_ids=list(irrelevant_memory_ids) or None,
        stale_memory_ids=list(stale_memory_ids) or None,
        wrong_memory_ids=list(wrong_memory_ids) or None,
    )
    _print(result, ctx.obj["json"])
    eng.close()


@cli.command("schema")
@click.option("--needs-review", is_flag=True)
@click.option("--limit", default=50, show_default=True)
@click.pass_context
def schema_list(ctx: click.Context, needs_review: bool, limit: int) -> None:
    """List schemas (optionally filtered)."""
    eng = _build_engine(ctx.obj["db"])
    kwargs: dict[str, Any] = {"limit": limit}
    if needs_review:
        kwargs["is_labile"] = True
    items = eng.list_schemas(**kwargs)
    if ctx.obj["json"]:
        _print([asdict(s) for s in items], True)
    else:
        for s in items:
            label = s.facets.get("display_label", "") if isinstance(s.facets, dict) else ""
            label_str = f" [{label}]" if label else ""
            click.echo(
                f"  [sch_{s.id}]{label_str} {s.content_text}"
                f"  status={s.status} sal={s.salience:.3f} supports={len(s.supporting_episode_ids)}"
                f" tags={','.join(s.tags)}" + ("  needs_review" if s.is_labile else "")
            )
    eng.close()


def _parse_schema_ref(ref: str) -> int:
    if not ref.startswith("sch_"):
        raise click.BadParameter(f"expected sch_N format, got: {ref}")
    return int(ref[4:])


@cli.command("forget")
@click.argument("ref")
@click.option("--reason", default=None, help="Optional note on why this is being forgotten.")
@click.pass_context
def forget_cmd(ctx: click.Context, ref: str, reason: str | None) -> None:
    """Suppress a schema from all future retrieval.

    CLI/dashboard only -- there is deliberately no MCP equivalent (forgetting
    requires a human looking at a specific memory, not an agent inferring
    intent). Reversible via `slowave unforget`.
    """
    sid = _parse_schema_ref(ref)
    eng = _build_engine(ctx.obj["db"], disable_encoder=True)
    try:
        schema = eng.get_schema(sid)
    except KeyError:
        _print({"error": "not found"}, ctx.obj["json"])
        eng.close()
        return
    warning = None
    if schema.generalization_stage >= 1:
        warning = (
            f"sch_{sid} is generalized (generalization_stage={schema.generalization_stage}); "
            "forgetting it removes it from every scope that reuses it, not just this one."
        )
    prior_status = schema.status
    eng.forget_schema(sid, reason=reason)
    eng.close()
    result: dict[str, Any] = {
        "schema_id": f"sch_{sid}",
        "status": "forgotten",
        "prior_status": prior_status,
    }
    if warning:
        result["warning"] = warning
    _print(result, ctx.obj["json"])
    if not ctx.obj["json"] and warning:
        click.echo(f"  warning: {warning}")


@cli.command("unforget")
@click.argument("ref")
@click.pass_context
def unforget_cmd(ctx: click.Context, ref: str) -> None:
    """Undo `slowave forget`, returning a schema to its prior status.

    Named "unforget" rather than "restore" to avoid colliding with the
    unrelated `slowave restore` DB-backup command.
    """
    sid = _parse_schema_ref(ref)
    eng = _build_engine(ctx.obj["db"], disable_encoder=True)
    try:
        restored_status = eng.unforget_schema(sid)
        _print({"schema_id": f"sch_{sid}", "status": restored_status}, ctx.obj["json"])
    except KeyError as e:
        _print({"error": str(e)}, ctx.obj["json"])
    eng.close()


@cli.command("stats")
@click.option("--scope", default=None, help="Filter by scope (e.g., project:myrepo).")
@click.option("--verbose", is_flag=True, help="Detailed breakdown.")
@click.option("--graph", "show_graph", is_flag=True, help="Full graph health report.")
@click.pass_context
def stats_cmd(ctx: click.Context, scope: str | None, verbose: bool, show_graph: bool) -> None:
    """Print memory and storage statistics."""
    from pathlib import Path

    from slowave.cli.output import get_renderer

    eng = _build_engine(ctx.obj["db"], disable_encoder=True)  # Don't load embeddings
    data = eng.stats()

    # Get file size
    db_path = Path(ctx.obj["db"])
    db_size_bytes = db_path.stat().st_size if db_path.exists() else 0
    db_size_mb = db_size_bytes / (1024 * 1024)

    # Get health
    health = eng.schema_health()
    eng.close()

    as_json = ctx.obj["json"]

    if as_json:
        result = {
            "version": "1.0",
            "memory": {
                "episodes": data.get("episodes", 0),
                "schemas": data.get("schemas", 0),
                "procedures": data.get("procedures", 0),
                "prototypes": data.get("prototypes", 0),
                "edges": data.get("edges", 0),
            },
            "storage": {
                "db_path": str(db_path),
                "db_size_bytes": db_size_bytes,
                "db_size_mb": round(db_size_mb, 2),
            },
            "health": health,
        }
        _print(result, True)
    else:
        renderer = get_renderer(use_emoji=False)
        renderer.title("Slowave Stats")

        renderer.section("Memory (Stored Schemas)")
        renderer.item("Episodes", f"{data.get('episodes', 0):,}")
        renderer.item("Schemas", f"{data.get('schemas', 0):,}")
        renderer.item("Prototypes", f"{data.get('prototypes', 0):,}")
        renderer.item("Edges", f"{data.get('edges', 0):,}")

        renderer.section("Storage")
        renderer.item("Database", str(db_path), dim=True)
        renderer.item("Size", f"{db_size_mb:.1f} MB")

        renderer.section("Health")
        active = health.get("active_schemas", 0)
        unique = health.get("active_unique_exact_by_scope", 0)
        dup_ratio = health.get("active_exact_duplicate_ratio", 0.0)
        renderer.item("Active schemas", f"{active:,}")
        renderer.item("Unique (exact)", f"{unique:,}")
        renderer.item("Duplicate ratio", f"{dup_ratio:.1%}")

        # Hints
        if data.get("episodes", 0) == 0:
            renderer.hint("No memories yet. Episodes will appear after sessions.")

        if show_graph:
            _print_graph_health(str(db_path), renderer, as_json)

        click.echo()


def _print_graph_health(db_path: str, renderer: Any, as_json: bool) -> None:
    """Print full graph health report from the shared metrics module."""
    from slowave.core.graph_health import compute

    gh = compute(db_path)
    if "error" in gh:
        renderer.item("Graph Health", f"Error: {gh['error']}")
        return

    if as_json:
        _print(gh, True)
        return

    s = gh["schema_graph"]
    p = gh["prototype_graph"]
    e = gh["episode_graph"]
    x = gh["cross_cutting"]
    sup = gh["supplementary"]

    renderer.section("Schema Graph")
    renderer.item("Total", f"{s['total']:,}")
    renderer.item("Status", ", ".join(f"{k}={v}" for k, v in s["status"].items()))
    renderer.item("Superseded", f"{s['superseded_pct']}%")
    renderer.item(
        "Relations",
        f"{s['total_relations']} edges: "
        + ", ".join(f"{k}={v}" for k, v in s["relations"].items()),
    )
    renderer.item("Isolates", f"{s['isolates']} ({s['isolate_pct']}%)")
    renderer.item(
        "Components",
        f"{s['components']['total_components']} "
        f"(largest={s['components']['largest_component']})",
    )
    renderer.item(
        "Salience",
        f"median={s['salience']['median']}, "
        f"ceiling breaches={s['salience']['ceiling_breaches']}",
    )
    cd = s.get("cosine_distribution")
    if cd:
        renderer.item("Cosine distribution", f"N/A (numpy not available)" if cd is None else "")
        renderer.item(
            "  Summary",
            f"mean={cd['mean']}, median={cd['median']}, std={cd['std']}, "
            f"({cd['schemas_with_embeddings']} schemas, {cd['pairs_sampled']:,} pairs)",
        )
        renderer.item(
            "  Verdict zones",
            ", ".join(
                f"{k}={cd['zone_pcts'][k]}%"
                for k in ["unrelated", "at_risk", "same_topic", "near_dup"]
            ),
        )
        renderer.item("  Pairs >= same_topic (0.75)", f"{cd['pairs_above_075']:,}")
    cc = s.get("component_coherence")
    if cc:
        renderer.item(
            "Component coherence",
            f"mean purity={cc['mean_purity_pct']}%, "
            f"mean within-cos={cc['mean_within_cosine']}, "
            f"low-coherence={cc['low_coherence_components']}",
        )
    ha = s.get("hubs_authorities")
    if ha:
        renderer.item(
            "Hubs/Authorities",
            f"hub p95={ha['hub_p95']}, auth p95={ha['auth_p95']}, "
            f"({ha['schemas_scored']} scored)",
        )

    renderer.section("Prototype Graph")
    renderer.item(
        "Total", f"{p['total']:,} " + ", ".join(f"{k}={v}" for k, v in p["scales"].items())
    )
    renderer.item("Edges", f"{p['edge_count']} (density={p['density_pct']}%)")
    ec = p["edge_composition"]
    renderer.item(
        "Edge dominance",
        f"similarity={ec['similarity_pct']}%, "
        f"transition={ec['transition_pct']}%, "
        f"coactivation={ec['coactivation_pct']}%",
    )
    d = p["distances"]
    if "error" not in d:
        renderer.item(
            "Intra/Inter distances",
            f"fine={d['fine_intra']}, coarse={d['coarse_intra']}, " f"inter={d['inter']}",
        )
    renderer.item(
        "Updated 24h / 7d", f"{p['age']['pct_updated_24h']}% / {p['age']['pct_updated_7d']}%"
    )
    renderer.item("Zero-variance fine", f"{p['zero_variance_fine_pct']}%")
    sm = p["schema_mapping"]
    renderer.item(
        "Multi-prototype schemas", f"{sm['multi_prototype_schemas']} ({sm['multi_prototype_pct']}%)"
    )

    renderer.section("Episode Graph")
    renderer.item("Total", f"{e['total']:,}")
    renderer.item(
        "Salience",
        f"avg={e['salience']['avg']}, " f"range=[{e['salience']['min']}, {e['salience']['max']}]",
    )
    renderer.item("Episodes per schema", f"{e['episodes_per_schema']}")

    renderer.section("Cross-Cutting")
    sess = x["sessions"]
    wr = x["worker_runs"]
    renderer.item("Sessions", f"{sess['total']} ({sess['open']} open)")
    renderer.item(
        "Worker runs",
        f"{wr['total']}, avg={wr['avg_duration_ms']}ms, " f"max={wr['max_duration_ms']}ms",
    )
    renderer.item("Schemas decayed", f"{wr['schemas_decayed']}")
    renderer.item("Schema density", f"{x['schema_graph_density_pct']}%")
    renderer.item("Prototype density", f"{x['prototype_graph_density_pct']}%")

    renderer.section("Supplementary")
    renderer.item("Labile schemas", f"{sup['labile_schemas']}")
    renderer.item(
        "Generalization stages",
        ", ".join(f"stage {k}={v}" for k, v in sorted(sup["generalization_stages"].items())),
    )
    renderer.item("Top scopes", ", ".join(f"{s}={c}" for s, c in sup["top_scopes"]))


def _slowave_processes() -> list[dict[str, Any]]:
    """Best-effort local process snapshot for operational hygiene.

    On Windows ``ps`` is not available; delegates to a PowerShell WMI query.
    On macOS / Linux uses ``ps -axo``.
    """
    import platform as _platform

    if _platform.system() == "Windows":
        return _slowave_processes_windows()

    try:
        out = subprocess.check_output(
            ["ps", "-axo", "pid,ppid,stat,rss,command"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return []
    rows: list[dict[str, Any]] = []
    for line in out.splitlines()[1:]:
        parts = line.strip().split(None, 4)
        if len(parts) < 5:
            continue
        pid, ppid, stat, rss, command = parts
        if not any(
            token in command
            for token in (
                "slowave.mcp.http_server",
                "slowave-mcp-http",
                "slowave worker",
                "slowave.cli.main",
                "slowave serve",
            )
        ):
            continue
        rows.append(
            {
                "pid": int(pid),
                "ppid": int(ppid),
                "stat": stat,
                "rss_kb": int(rss),
                "command": command,
            }
        )
    return rows


def _slowave_processes_windows() -> list[dict[str, Any]]:
    """Windows-specific process detection using PowerShell WMI.

    Queries Win32_Process for Python processes whose CommandLine contains
    slowave-related keywords.  Returns an empty list on any error — health
    checks are best-effort only.
    """
    ps_cmd = (
        "Get-WmiObject Win32_Process -Filter \"Name LIKE '%python%'\" | "
        "Select-Object ProcessId,ParentProcessId,WorkingSetSize,CommandLine | "
        "ConvertTo-Json -Compress"
    )
    try:
        result = subprocess.run(
            ["powershell", "-NonInteractive", "-Command", ps_cmd],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except Exception:
        return []

    if result.returncode != 0 or not result.stdout.strip():
        return []

    try:
        import json as _json

        procs = _json.loads(result.stdout.strip())
        if isinstance(procs, dict):  # single result → normalise to list
            procs = [procs]
    except Exception:
        return []

    rows: list[dict[str, Any]] = []
    for proc in procs:
        cmd = str(proc.get("CommandLine") or "")
        if (
            "slowave-mcp" not in cmd
            and "slowave worker" not in cmd
            and "slowave.mcp.server" not in cmd
            and "slowave.cli.main" not in cmd
        ):
            continue
        try:
            pid = int(proc.get("ProcessId", 0))
            ppid = int(proc.get("ParentProcessId", 0))
            rss_kb = int(proc.get("WorkingSetSize", 0)) // 1024
        except (TypeError, ValueError):
            pid = ppid = rss_kb = 0
        rows.append({"pid": pid, "ppid": ppid, "stat": "running", "rss_kb": rss_kb, "command": cmd})
    return rows


def _worker_health() -> dict[str, Any]:
    """Best-effort health for the background consolidation worker.

    Worker health is independent from feedback/reinforcement health. A running
    worker means Slowave can perform background consolidation. It does not imply
    that MCP clients are sending post-recall feedback.
    """
    processes = _slowave_processes()
    worker_processes = [p for p in processes if "slowave worker" in p.get("command", "")]
    return {
        "process_detected": bool(worker_processes),
        "process_count": len(worker_processes),
        "processes": worker_processes,
        "warnings": (
            []
            if worker_processes
            else [
                "no background worker process detected; run 'slowave worker' or configure the worker service if you want automatic consolidation"
            ]
        ),
    }


def _feedback_health(db_path: str) -> dict[str, Any]:
    """Best-effort feedback/reinforcement health from the local SQLite DB.

    This answers a different question from worker health: whether clients appear
    to send post-recall feedback/reinforcement events after retrieving context.
    """
    import sqlite3
    import time

    path = os.path.abspath(os.path.expanduser(db_path))
    result: dict[str, Any] = {
        "db_path": path,
        "available": False,
        "recall_or_context_calls": 0,
        "remember_calls": 0,
        "feedback_or_reinforcement_calls": 0,
        "last_feedback_ts": None,
        "last_feedback_age_days": None,
        "warnings": [],
    }

    if not os.path.exists(path):
        result["warnings"].append("database does not exist yet")
        return result

    def _has_table(cur: Any, name: str) -> bool:
        cur.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,))
        return cur.fetchone() is not None

    def _columns(cur: Any, table: str) -> set[str]:
        try:
            cur.execute(f"PRAGMA table_info({table})")
            return {str(row[1]) for row in cur.fetchall()}
        except Exception:
            return set()

    def _count(cur: Any, sql: str) -> int:
        try:
            cur.execute(sql)
            row = cur.fetchone()
            return int(row[0] or 0) if row else 0
        except Exception:
            return 0

    def _max_ts(cur: Any, table: str, candidates: tuple[str, ...]) -> float | None:
        cols = _columns(cur, table)
        for col in candidates:
            if col not in cols:
                continue
            try:
                cur.execute(f"SELECT MAX({col}) FROM {table}")
                value = cur.fetchone()[0]
                if value is None:
                    continue
                if isinstance(value, (int, float)):
                    return float(value)
                return float(value)
            except Exception:
                continue
        return None

    try:
        conn = sqlite3.connect(path)
        cur = conn.cursor()
        # Best-effort introspection only. Some legacy/broken SQLite schemas can
        # raise "malformed database schema" while reading sqlite_master. This
        # pragma lets diagnostics continue instead of surfacing unreadable schema
        # text in `slowave doctor`.
        try:
            cur.execute("PRAGMA writable_schema=ON")
        except Exception:
            pass
        result["available"] = True

        event_tables = [
            table
            for table in (
                "raw_events",
                "events",
                "event_log",
                "turn_events",
                "memory_events",
            )
            if _has_table(cur, table)
        ]

        for table in event_tables:
            cols = _columns(cur, table)
            type_col = next((c for c in ("type", "event_type", "kind", "name") if c in cols), None)
            content_col = next((c for c in ("content", "payload", "text") if c in cols), None)
            searchable_col = type_col or content_col
            if not searchable_col:
                continue

            result["recall_or_context_calls"] += _count(
                cur,
                f"SELECT COUNT(*) FROM {table} WHERE lower({searchable_col}) LIKE '%recall%' OR lower({searchable_col}) LIKE '%context%'",
            )
            result["remember_calls"] += _count(
                cur,
                f"SELECT COUNT(*) FROM {table} WHERE lower({searchable_col}) LIKE '%remember%'",
            )
            result["feedback_or_reinforcement_calls"] += _count(
                cur,
                f"SELECT COUNT(*) FROM {table} WHERE lower({searchable_col}) LIKE '%feedback%' OR lower({searchable_col}) LIKE '%reinforce%' OR lower({searchable_col}) LIKE '%suppress%'",
            )

        for table in (
            "feedback",
            "memory_feedback",
            "retrieval_feedback",
            "turn_feedbacks",
        ):
            if not _has_table(cur, table):
                continue
            count = _count(cur, f"SELECT COUNT(*) FROM {table}")
            result["feedback_or_reinforcement_calls"] += count
            ts = _max_ts(cur, table, ("ts", "created_at", "timestamp", "updated_at"))
            if ts is not None:
                current = result["last_feedback_ts"]
                result["last_feedback_ts"] = ts if current is None else max(float(current), ts)

        conn.close()
    except Exception as exc:
        msg = str(exc)
        if "malformed database schema" in msg.lower():
            result["warnings"].append(
                "feedback health could not inspect legacy/malformed SQLite schema; counters unavailable"
            )
        else:
            result["warnings"].append(f"could not inspect feedback health: {msg}")
        return result

    last_ts = result.get("last_feedback_ts")
    if isinstance(last_ts, (int, float)) and last_ts > 0:
        ts = float(last_ts) / 1000.0 if float(last_ts) > 10_000_000_000 else float(last_ts)
        result["last_feedback_age_days"] = round((time.time() - ts) / 86400.0, 2)

    if result["recall_or_context_calls"] > 0 and result["feedback_or_reinforcement_calls"] == 0:
        result["warnings"].append(
            "recall/context activity detected, but no post-recall feedback or reinforcement events were found; this is a client integration issue, not a worker issue"
        )

    return result


def _session_lifecycle_health(db_path: str) -> dict[str, Any]:
    """Best-effort session start/end health from the local SQLite DB."""
    import sqlite3

    path = os.path.abspath(os.path.expanduser(db_path))
    result: dict[str, Any] = {
        "db_path": path,
        "available": False,
        "sessions_started": 0,
        "sessions_committed": 0,
        "warnings": [],
    }

    if not os.path.exists(path):
        result["warnings"].append("database does not exist yet")
        return result

    def _has_table(cur: Any, name: str) -> bool:
        cur.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,))
        return cur.fetchone() is not None

    def _columns(cur: Any, table: str) -> set[str]:
        try:
            cur.execute(f"PRAGMA table_info({table})")
            return {str(row[1]) for row in cur.fetchall()}
        except Exception:
            return set()

    def _count(cur: Any, sql: str) -> int:
        try:
            cur.execute(sql)
            row = cur.fetchone()
            return int(row[0] or 0) if row else 0
        except Exception:
            return 0

    try:
        conn = sqlite3.connect(path)
        cur = conn.cursor()
        # Best-effort introspection only. Some legacy/broken SQLite schemas can
        # raise "malformed database schema" while reading sqlite_master. This
        # pragma lets diagnostics continue instead of surfacing unreadable schema
        # text in `slowave doctor`.
        try:
            cur.execute("PRAGMA writable_schema=ON")
        except Exception:
            pass
        result["available"] = True

        for table in ("sessions", "agent_sessions", "memory_sessions"):
            if not _has_table(cur, table):
                continue
            cols = _columns(cur, table)
            result["sessions_started"] = max(
                result["sessions_started"],
                _count(cur, f"SELECT COUNT(*) FROM {table}"),
            )
            if "ended_at" in cols:
                result["sessions_committed"] = max(
                    result["sessions_committed"],
                    _count(cur, f"SELECT COUNT(*) FROM {table} WHERE ended_at IS NOT NULL"),
                )
            elif "closed_at" in cols:
                result["sessions_committed"] = max(
                    result["sessions_committed"],
                    _count(cur, f"SELECT COUNT(*) FROM {table} WHERE closed_at IS NOT NULL"),
                )
            elif "status" in cols:
                result["sessions_committed"] = max(
                    result["sessions_committed"],
                    _count(
                        cur,
                        f"SELECT COUNT(*) FROM {table} WHERE status IN ('ended', 'closed', 'committed')",
                    ),
                )

        conn.close()
    except Exception as exc:
        msg = str(exc)
        if "malformed database schema" in msg.lower():
            result["warnings"].append(
                "session lifecycle health could not inspect legacy/malformed SQLite schema; counters unavailable"
            )
        else:
            result["warnings"].append(f"could not inspect session lifecycle health: {msg}")
        return result

    if result["sessions_started"] > 0 and result["sessions_committed"] == 0:
        result["warnings"].append("sessions are started but never ended/committed")
    elif (
        result["sessions_started"] >= 3
        and result["sessions_committed"] / max(result["sessions_started"], 1) < 0.5
    ):
        result["warnings"].append(
            "many sessions are started but fewer than half are ended/committed"
        )

    return result


def _lifecycle_version_health(db_path: str) -> dict[str, Any]:
    """Per-lifecycle-version activate/recall/feedback telemetry (WP-8).

    Groups context_recall_events (activate = retrieval_type 'context', recall
    = retrieval_type 'recall') by the lifecycle_version stamped on the row
    itself at call time (see slowave.lifecycle.LIFECYCLE_VERSION), and
    context_feedback_events by joining back to its context_recall_events row
    via context_id (a mandatory FK). Per-call stamping, not a session join,
    is the reliable attribution point here: recall() has no session concept
    at all (ops.recall() never sets session_id), so a session-keyed join
    would silently bucket every recall() call as "unknown". This reflects
    which cognitive-cycle *contract* the server enforced for that call -- not
    necessarily what's physically installed in the caller's client
    instruction file; `slowave doctor`'s per-client lifecycle_version reports
    that separately. Rows predating this column are bucketed as "unknown".

    The recall/activate ratio is a behavioral diagnostic only (per the
    2026-07-28 retrieval-quality plan's Phase 6): it is reported for
    visibility, not as a target to optimize toward.
    """
    import sqlite3

    path = os.path.abspath(os.path.expanduser(db_path))
    result: dict[str, Any] = {
        "db_path": path,
        "available": False,
        "current_version": LIFECYCLE_VERSION,
        "by_version": {},
        "warnings": [],
    }

    if not os.path.exists(path):
        result["warnings"].append("database does not exist yet")
        return result

    def _has_table(cur: Any, name: str) -> bool:
        cur.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,))
        return cur.fetchone() is not None

    def _has_column(cur: Any, table: str, column: str) -> bool:
        try:
            cur.execute(f"PRAGMA table_info({table})")
            return any(str(row[1]) == column for row in cur.fetchall())
        except Exception:
            return False

    by_version: dict[str, dict[str, Any]] = {}

    def _bucket(version: Any) -> dict[str, Any]:
        key = str(version) if version else "unknown"
        return by_version.setdefault(
            key, {"activate_calls": 0, "recall_calls": 0, "feedback_calls": 0}
        )

    try:
        conn = sqlite3.connect(path)
        cur = conn.cursor()
        try:
            cur.execute("PRAGMA writable_schema=ON")
        except Exception:
            pass
        result["available"] = True

        if not _has_table(cur, "context_recall_events") or not _has_column(
            cur, "context_recall_events", "lifecycle_version"
        ):
            result["warnings"].append(
                "context_recall_events.lifecycle_version column not present yet; "
                "run any slowave command once to apply migrations"
            )
            conn.close()
            return result

        cur.execute(
            "SELECT COALESCE(lifecycle_version, 'unknown') AS version, "
            "retrieval_type, COUNT(*) "
            "FROM context_recall_events "
            "GROUP BY version, retrieval_type"
        )
        for version, retrieval_type, count in cur.fetchall():
            bucket = _bucket(version)
            if retrieval_type == "recall":
                bucket["recall_calls"] += int(count)
            else:
                bucket["activate_calls"] += int(count)

        if _has_table(cur, "context_feedback_events"):
            cur.execute(
                "SELECT COALESCE(cre.lifecycle_version, 'unknown') AS version, COUNT(*) "
                "FROM context_feedback_events cfe "
                "LEFT JOIN context_recall_events cre ON cre.context_id = cfe.context_id "
                "GROUP BY version"
            )
            for version, count in cur.fetchall():
                _bucket(version)["feedback_calls"] += int(count)

        conn.close()
    except Exception as exc:
        msg = str(exc)
        if "malformed database schema" in msg.lower():
            result["warnings"].append(
                "lifecycle version health could not inspect legacy/malformed SQLite schema; counters unavailable"
            )
        else:
            result["warnings"].append(f"could not inspect lifecycle version health: {msg}")
        return result

    for counts in by_version.values():
        total = counts["activate_calls"] + counts["recall_calls"]
        counts["recall_activate_ratio"] = (
            round(counts["recall_calls"] / total, 3) if total else None
        )

    result["by_version"] = by_version
    return result


@cli.command("status")
@click.pass_context
def status_cmd(ctx: click.Context) -> None:
    """Print DB, memory-health, and local process status."""
    db = ctx.obj["db"]
    eng = _build_engine(db)
    payload = {
        "db_path": os.path.abspath(os.path.expanduser(db)),
        "db_exists": os.path.exists(os.path.expanduser(db)),
        "stats": eng.stats(),
        "schema_health": eng.schema_health(),
        "processes": _slowave_processes(),
        "worker_health": _worker_health(),
        "session_lifecycle_health": _session_lifecycle_health(db),
        "feedback_health": _feedback_health(db),
        "lifecycle_version_health": _lifecycle_version_health(db),
    }
    eng.close()
    if ctx.obj["json"]:
        _print(payload, True)
        return
    click.echo(f"DB: {payload['db_path']} ({'exists' if payload['db_exists'] else 'missing'})")
    click.echo(f"Stats: {payload['stats']}")
    h = payload["schema_health"]
    click.echo(
        "Schema health: "
        f"active={h['active_schemas']} unique_exact={h['active_unique_exact_by_scope']} "
        f"dup_rows={h['active_exact_duplicate_rows']} "
        f"dup_ratio={h['active_exact_duplicate_ratio']:.1%} "
        f"status={h['schemas_by_status']}"
    )
    click.echo("Processes:")
    for p in payload["processes"]:
        click.echo(
            f"  pid={p['pid']} ppid={p['ppid']} rss={p['rss_kb']}KB "
            f"stat={p['stat']} {p['command']}"
        )

    wh = payload["worker_health"]
    click.echo("Worker health: " f"detected={wh['process_detected']} count={wh['process_count']}")
    sh = payload["session_lifecycle_health"]
    click.echo(
        "Session lifecycle health: "
        f"started={sh['sessions_started']} committed={sh['sessions_committed']}"
    )
    fh = payload["feedback_health"]
    click.echo(
        "Feedback health: "
        f"recall_or_context={fh['recall_or_context_calls']} "
        f"remember={fh['remember_calls']} "
        f"feedback_or_reinforcement={fh['feedback_or_reinforcement_calls']}"
    )
    lvh = payload["lifecycle_version_health"]
    click.echo(f"Lifecycle version telemetry (current={lvh['current_version']}):")
    for version, counts in lvh.get("by_version", {}).items():
        click.echo(
            f"  {version}: activate={counts['activate_calls']} recall={counts['recall_calls']} "
            f"feedback={counts['feedback_calls']} "
            f"recall/activate={counts['recall_activate_ratio']}"
        )


@cli.command("dashboard")
@click.option("--host", default="127.0.0.1", show_default=True, help="HTTP bind host.")
@click.option("--port", default=8765, show_default=True, help="HTTP bind port.")
@click.option("--refresh-ms", default=2000, show_default=True, help="Overview refresh interval.")
@click.option(
    "--allow-actions/--no-allow-actions",
    default=True,
    show_default=True,
    help="Enable the Forget/Unforget buttons. Disable for a strictly read-only dashboard.",
)
@click.option("--no-open", is_flag=True, help="Do not open the browser automatically.")
@click.pass_context
def dashboard_cmd(
    ctx: click.Context,
    host: str,
    port: int,
    refresh_ms: int,
    allow_actions: bool,
    no_open: bool,
) -> None:
    """Run the local Slowave web dashboard."""
    from slowave.dashboard.app import run_dashboard

    run_dashboard(
        db_path=ctx.obj["db"],
        host=host,
        port=port,
        refresh_ms=refresh_ms,
        allow_actions=allow_actions,
        open_browser=not no_open,
    )


@cli.command("dedup-schemas")
@click.option("--apply", "apply_changes", is_flag=True, help="Apply cleanup. Default is dry-run.")
@click.pass_context
def dedup_schemas_cmd(ctx: click.Context, apply_changes: bool) -> None:
    """Merge exact duplicate active schemas within each scope."""
    eng = _build_engine(ctx.obj["db"])
    before = eng.schema_health()
    result = eng.dedup_schemas_exact(dry_run=not apply_changes)
    after = eng.schema_health()
    eng.close()
    payload = {"before": before, "dedup": result, "after": after}
    if ctx.obj["json"]:
        _print(payload, True)
        return
    click.echo("Schema deduplication " + ("APPLIED" if apply_changes else "DRY RUN"))
    click.echo(
        f"Before: active={before['active_schemas']} unique_exact={before['active_unique_exact_by_scope']} "
        f"dup_rows={before['active_exact_duplicate_rows']} dup_ratio={before['active_exact_duplicate_ratio']:.1%}"
    )
    click.echo(f"Dedup: {result}")
    click.echo(
        f"After : active={after['active_schemas']} unique_exact={after['active_unique_exact_by_scope']} "
        f"dup_rows={after['active_exact_duplicate_rows']} dup_ratio={after['active_exact_duplicate_ratio']:.1%}"
    )


@cli.command("consolidate")
@click.option(
    "--decay-idle-days",
    default=30.0,
    show_default=True,
    help="Override the idle-days threshold for the decay step. "
    "Useful in tests: --decay-idle-days 0 decays all eligible schemas immediately.",
)
@click.pass_context
def consolidate_cmd(ctx: click.Context, decay_idle_days: float) -> None:
    """Manually trigger a replay + latent consolidation pass."""
    eng = _build_engine(ctx.obj["db"])
    result = eng.consolidate_once(decay_idle_days=decay_idle_days)
    _print(result, ctx.obj["json"])
    eng.close()


@cli.command("rebuild")
@click.option(
    "--force",
    "force",
    is_flag=True,
    help="Wipe and rebuild derived memory state from raw_events unconditionally, "
    "bypassing the usual logic_version check. For support/debugging a stuck "
    "instance — normal auto-migration on startup does not need this.",
)
@click.pass_context
def rebuild_cmd(ctx: click.Context, force: bool) -> None:
    """Rebuild derived memory state (episodes/prototypes/schemas) from raw_events.

    Normally this happens automatically, once, the next time a Slowave
    process starts after current_logic_version is bumped — nothing to run
    manually. --force is for support/debugging only.
    """
    if not force:
        raise click.UsageError("Pass --force to run a rebuild manually (see --help).")

    import dataclasses as _dataclasses

    from slowave.core.services.rebuild import RebuildService

    eng = _build_engine(ctx.obj["db"])
    stats = RebuildService.run(eng.db, eng.cfg, encoder=eng.encoder)
    eng.close()
    _print(_dataclasses.asdict(stats), ctx.obj["json"])


@cli.command("worker")
@click.option(
    "--interval",
    default=300,
    show_default=True,
    help="Seconds between consolidation passes (simulates sleep cycles).",
)
@click.option(
    "--once",
    is_flag=True,
    help="Run a single consolidation pass then exit (useful for cron/tests).",
)
@click.pass_context
def worker_cmd(ctx: click.Context, interval: int, once: bool) -> None:
    """Background consolidation worker — the sleep simulator.

    Runs replay + latent schema construction on a schedule, decoupled from session
    ingest. Mimics slow-wave sleep: episodes accumulate during waking sessions,
    then are consolidated offline.

    In production: run as a background process or cron job.
    In tests/scripts: use --once to trigger a single pass.

    Examples:
      slowave worker --once                  # one pass, then exit
      slowave worker --interval 600          # consolidate every 10 min
      slowave worker --interval 3600 &       # background hourly consolidation
    """
    import signal
    import time as _time

    eng = _build_engine(ctx.obj["db"])
    stop = False

    def _handle_signal(sig: int, frame: Any) -> None:
        nonlocal stop
        click.echo("\nworker: received signal, stopping after current pass.")
        stop = True

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    def _run_pass() -> dict[str, Any]:
        return eng.consolidate_once()

    if once:
        result = _run_pass()
        _print(result, ctx.obj["json"])
        eng.close()
        return

    from slowave.utils.spinner import SleepSpinner

    def _fmt_interval(s: int) -> str:
        return f"{s // 60}m" if s >= 60 else f"{s}s"

    click.echo(f"  🧠 worker starting  (interval={_fmt_interval(interval)})  Ctrl-C to stop.")
    while not stop:
        result = _run_pass()
        if ctx.obj["json"]:
            _print(result, True)
        else:
            cs = result.get("consolidation", {})
            ts = __import__("datetime").datetime.now().strftime("%H:%M:%S")
            click.echo(
                f"  🧠 [{ts}]  "
                f"schemas +{cs.get('schemas_created', 0)} "
                f"~{cs.get('schemas_reinforced', 0)} "
                f"skip={cs.get('schemas_skipped', 0)}"
            )
        # rebuild indices so next pass sees fresh state
        eng.refresh_indices()
        spinner = SleepSpinner(interval_s=interval)
        spinner.start()
        for _ in range(interval):
            if stop:
                break
            _time.sleep(1)
        spinner.stop()

    eng.close()
    click.echo("  💤 worker stopped.")


@cli.command("doctor")
@click.option("--json", "as_json", is_flag=True, help="Machine-readable JSON output.")
@click.option("--verbose", is_flag=True, help="Verbose diagnostics.")
@click.pass_context
def doctor_cmd(ctx: click.Context, as_json: bool, verbose: bool) -> None:
    """Check the local Slowave environment and report any issues.

    Verifies Python version, core dependencies, the embedding backend,
    SQLite write access, and MCP server availability.
    Exits with code 1 if any check fails (FAIL status).
    """
    from slowave import __version__
    from slowave.cli.clients import get_client_statuses, summarize_client_status
    from slowave.cli.diagnostics import (
        check_embedding_backend,
        check_faiss,
        check_http_daemon,
        check_mcp_server,
        check_onnxruntime,
        check_python,
        check_sqlite_write,
        get_runtime_info,
    )
    from slowave.cli.output import Status, get_renderer

    as_json = as_json or ctx.obj["json"]
    renderer = get_renderer(use_emoji=False)

    if as_json:
        # JSON mode
        result = {"status": "ok", "version": __version__, "runtime": {}}

        runtime = get_runtime_info(__version__)
        result["runtime_info"] = {
            "python_version": runtime.python_version,
            "executable": runtime.python_executable,
            "db_path": runtime.db_path,
            "config_path": runtime.config_path,
            "slowave_dir": runtime.slowave_dir,
        }

        # Run checks
        checks = {
            "python": check_python(),
            "faiss": check_faiss(),
            "onnxruntime": check_onnxruntime(),
            "embedding_backend": check_embedding_backend(),
            "sqlite_write": check_sqlite_write(),
            "mcp_server": check_mcp_server(),
            "http_daemon": check_http_daemon(),
        }

        result["runtime"] = {
            name: {
                "status": check.status.value,
                "label": check.label,
                "detail": check.detail,
            }
            for name, check in checks.items()
        }

        # Operational checks
        worker = _worker_health()
        session_lifecycle = _session_lifecycle_health(ctx.obj["db"])
        feedback = _feedback_health(ctx.obj["db"])
        lifecycle_version = _lifecycle_version_health(ctx.obj["db"])
        result["worker_health"] = worker
        result["session_lifecycle_health"] = session_lifecycle
        result["feedback_health"] = feedback
        result["lifecycle_version_health"] = lifecycle_version

        # Client checks
        clients = get_client_statuses()
        result["clients"] = {}
        warnings = []
        for source, health in (
            ("WORKER_HEALTH", worker),
            ("SESSION_LIFECYCLE_HEALTH", session_lifecycle),
            ("FEEDBACK_HEALTH", feedback),
            ("LIFECYCLE_VERSION_HEALTH", lifecycle_version),
        ):
            for warning in health.get("warnings", []):
                warnings.append(
                    {
                        "code": source,
                        "message": str(warning),
                    }
                )

        for key, client in clients.items():
            status, detail = summarize_client_status(client)
            result["clients"][key] = {
                "name": client.name,
                "status": status.value,
                "detail": detail,
                "lifecycle_version": client.lifecycle_version,
            }
            if status == Status.WARN:
                warnings.append(
                    {
                        "code": f"{key.upper()}_INCOMPLETE",
                        "message": f"{client.name}: {detail}",
                    }
                )

        # Determine overall status
        has_fail = any(c.status == Status.FAIL for c in checks.values())
        has_warn = any(c.status == Status.WARN for c in checks.values()) or bool(warnings)

        if has_fail:
            result["status"] = "fail"
        elif has_warn:
            result["status"] = "warn"

        result["warnings"] = warnings
        renderer.json(result)

        if has_fail:
            sys.exit(1)
    else:
        # Human-readable mode
        runtime = get_runtime_info(__version__)
        renderer.title("Slowave Doctor", f"v{__version__}")

        renderer.section("Environment")
        renderer.item("Python", runtime.python_version)
        renderer.item("Data dir", runtime.slowave_dir, dim=True)
        renderer.item("Config", runtime.config_path, dim=True)

        # Runtime checks
        renderer.section("Runtime")
        checks = [
            check_python(),
            check_faiss(),
            check_onnxruntime(),
            check_embedding_backend(),
            check_sqlite_write(),
            check_mcp_server(),
            check_http_daemon(),
        ]

        for check in checks:
            renderer.check(check.label, check.status, check.detail, check.remediation)

        # Operational health
        worker = _worker_health()
        session_lifecycle = _session_lifecycle_health(ctx.obj["db"])
        feedback = _feedback_health(ctx.obj["db"])
        lifecycle_version = _lifecycle_version_health(ctx.obj["db"])

        renderer.section("Worker Health")
        renderer.item("Worker process detected", str(worker.get("process_detected", False)))
        renderer.item("Worker process count", f"{worker.get('process_count', 0):,}")

        renderer.section("Session Lifecycle Health")
        renderer.item("Sessions started", f"{session_lifecycle.get('sessions_started', 0):,}")
        renderer.item(
            "Sessions ended/committed", f"{session_lifecycle.get('sessions_committed', 0):,}"
        )

        renderer.section("Feedback Health")
        renderer.item("Recall/context calls", f"{feedback.get('recall_or_context_calls', 0):,}")
        renderer.item("Remember calls", f"{feedback.get('remember_calls', 0):,}")
        renderer.item(
            "Feedback/reinforcement calls",
            f"{feedback.get('feedback_or_reinforcement_calls', 0):,}",
        )
        age = feedback.get("last_feedback_age_days")
        renderer.item(
            "Last feedback",
            "never" if age is None else f"{age:g} day(s) ago",
        )

        renderer.section("Lifecycle Version Telemetry")
        renderer.item("Current contract version", str(lifecycle_version.get("current_version")))
        by_version = lifecycle_version.get("by_version", {})
        if not by_version:
            renderer.item("Activity by version", "none recorded yet")
        for version, counts in by_version.items():
            renderer.item(
                version,
                f"activate={counts['activate_calls']} recall={counts['recall_calls']} "
                f"feedback={counts['feedback_calls']} "
                f"recall/activate={counts['recall_activate_ratio']} (diagnostic, not a target)",
            )

        # Client checks
        renderer.section("Clients")
        clients = get_client_statuses()
        warnings_list = []

        for key, client in clients.items():
            status, detail = summarize_client_status(client)
            renderer.check(client.name, status, detail)
            if status == Status.WARN:
                warnings_list.append((client.name, detail))

        # Warnings
        operational_warnings = []
        for label, health in (
            ("Worker", worker),
            ("Session lifecycle", session_lifecycle),
            ("Feedback", feedback),
            ("Lifecycle version", lifecycle_version),
        ):
            for warning in health.get("warnings", []):
                operational_warnings.append((label, str(warning)))

        if warnings_list or operational_warnings:
            renderer.section("Warnings")
            for label, warning in operational_warnings:
                remediation = None
                if label == "Worker":
                    remediation = "Run: slowave worker --once, or configure the worker service if you want automatic consolidation."
                elif label == "Feedback":
                    if "counters unavailable" in warning:
                        remediation = "Run `slowave status --json` or inspect the SQLite DB schema; this diagnostic is best-effort and does not mean feedback is broken."
                    else:
                        remediation = "Check client lifecycle instructions and post-recall feedback hooks. This is independent from the background worker."
                elif label == "Session lifecycle":
                    if "counters unavailable" in warning:
                        remediation = "Run `slowave status --json` or inspect the SQLite DB schema; this diagnostic is best-effort and does not mean lifecycle hooks are broken."
                    else:
                        remediation = "Check that the client calls session end/commit hooks when a conversation or task finishes."
                renderer.warning(f"{label}: {warning}", remediation)
            for name, detail in warnings_list:
                if "custom instructions" in detail.lower():
                    renderer.warning(
                        f"{name}: {detail}", "Run: slowave setup --dry-run, then slowave setup"
                    )
                else:
                    renderer.warning(f"{name}: {detail}")

        # Summary
        has_fail = any(c.status == Status.FAIL for c in checks)
        has_warn = bool(warnings_list) or bool(operational_warnings)

        if has_fail:
            msg = f"Setup required. {len([c for c in checks if c.status == Status.FAIL])} check(s) failed."
            renderer.summary(False, msg)
            sys.exit(1)
        elif has_warn:
            msg = f"Usable with {len(warnings_list) + len(operational_warnings)} warning(s)."
            renderer.summary(True, msg)
        else:
            renderer.summary(True, "All systems ready.")


@cli.command("uninstall")
@click.option("--dry-run", is_flag=True, help="Preview what would be removed.")
def uninstall_cmd(dry_run: bool) -> None:
    """Remove all Slowave configuration (keeps database by default).

    Removes ONLY Slowave-specific entries: MCP servers, lifecycle blocks,
    hooks, and worker service. Never deletes entire files or breaks configs.
    Database at ~/.slowave/ is preserved — remove manually if desired.
    """
    import platform
    from pathlib import Path

    from slowave.cli.setup import (
        _HOOKS_MARKER,
        _MARKER_END,
        _MARKER_START,
        _claude_desktop_config_path,
        _claude_md_path,
        _claude_settings_path,
        _cline_mcp_settings_path,
        _clinerules_path,
        _home,
        _read_json,
        _write_json,
    )

    click.echo(click.style("\nSlowave uninstall", bold=True))
    if dry_run:
        click.echo(click.style("  [DRY RUN]\n", fg="yellow"))

    changes = []
    errors = []

    def safe_remove_from_json(path: Path, remove_fn, desc: str):
        """Safely remove Slowave config from JSON without breaking structure."""
        if not path.exists():
            return
        try:
            cfg = _read_json(path)
            original = json.dumps(cfg, sort_keys=True)
            modified = remove_fn(cfg)
            if modified and json.dumps(cfg, sort_keys=True) != original:
                if not dry_run:
                    _write_json(path, cfg)
                changes.append(desc)
        except Exception as e:
            errors.append(f"{desc}: {str(e)[:100]}")

    def safe_remove_block(path: Path, desc: str):
        """Safely remove lifecycle block between markers."""
        if not path.exists():
            return
        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
            if _MARKER_START not in content or _MARKER_END not in content:
                return
            start_idx = content.index(_MARKER_START)
            try:
                end_idx = content.index(_MARKER_END, start_idx)
            except ValueError:
                errors.append(f"{desc}: mismatched markers")
                return
            new_content = content[:start_idx] + content[end_idx + len(_MARKER_END) :]
            new_content = new_content.lstrip("\n")
            if new_content != content and not dry_run:
                path.write_text(new_content, encoding="utf-8")
            if new_content != content:
                changes.append(desc)
        except Exception as e:
            errors.append(f"{desc}: {str(e)[:100]}")

    # Claude Code - MCP and hooks
    def remove_cc_mcp_hooks(cfg):
        modified = False
        if "slowave" in cfg.get("mcpServers", {}):
            del cfg["mcpServers"]["slowave"]
            modified = True
        # Remove only Slowave hooks, preserve other hooks
        if "hooks" in cfg:
            for event in ["UserPromptSubmit", "Stop"]:
                if event in cfg["hooks"]:
                    orig_len = len(cfg["hooks"][event])
                    cfg["hooks"][event] = [
                        g
                        for g in cfg["hooks"][event]
                        if not any(
                            _HOOKS_MARKER in h.get("command", "") for h in g.get("hooks", [])
                        )
                    ]
                    if len(cfg["hooks"][event]) < orig_len:
                        modified = True
                    # Remove empty event arrays
                    if not cfg["hooks"][event]:
                        del cfg["hooks"][event]
            # Remove hooks key only if empty
            if not cfg["hooks"]:
                del cfg["hooks"]
        return modified

    safe_remove_from_json(_claude_settings_path(), remove_cc_mcp_hooks, "Claude Code MCP + hooks")
    safe_remove_block(_claude_md_path(), "Claude Code lifecycle")

    # Claude Desktop - use safe helper
    def remove_cd_mcp(cfg):
        if "slowave" in cfg.get("mcpServers", {}):
            del cfg["mcpServers"]["slowave"]
            return True
        return False

    safe_remove_from_json(_claude_desktop_config_path(), remove_cd_mcp, "Claude Desktop MCP")

    # Cline
    def remove_cline_mcp(cfg):
        if "slowave" in cfg.get("mcpServers", {}):
            del cfg["mcpServers"]["slowave"]
            return True
        return False

    safe_remove_from_json(_cline_mcp_settings_path(), remove_cline_mcp, "Cline MCP")
    safe_remove_block(_clinerules_path(), "Cline lifecycle")

    # Worker
    system = platform.system()
    if not dry_run:
        if system == "Darwin":
            plist = _home() / "Library/LaunchAgents/com.slowave.worker.plist"
            if plist.exists():
                subprocess.run(
                    ["launchctl", "unload", str(plist)], capture_output=True, check=False
                )
                plist.unlink()
                changes.append("launchd worker")
        elif system == "Linux":
            xdg = Path(os.environ.get("XDG_CONFIG_HOME", str(_home() / ".config")))
            svc = xdg / "systemd/user/slowave-worker.service"
            if svc.exists():
                subprocess.run(
                    ["systemctl", "--user", "disable", "--now", "slowave-worker"],
                    capture_output=True,
                    check=False,
                )
                svc.unlink()
                changes.append("systemd worker")
        elif system == "Windows":
            try:
                subprocess.run(
                    [
                        "powershell",
                        "-NonInteractive",
                        "-Command",
                        "Unregister-ScheduledTask -TaskName SlowaveDaemon -Confirm:$false -ErrorAction SilentlyContinue",
                    ],
                    capture_output=True,
                    check=False,
                )
                changes.append("Task Scheduler daemon (SlowaveDaemon)")
            except Exception:
                pass
            try:
                subprocess.run(
                    [
                        "powershell",
                        "-NonInteractive",
                        "-Command",
                        "Unregister-ScheduledTask -TaskName SlowaveWorker -Confirm:$false -ErrorAction SilentlyContinue",
                    ],
                    capture_output=True,
                    check=False,
                )
                changes.append("Task Scheduler worker (SlowaveWorker)")
            except Exception:
                pass
    else:
        if system == "Windows":
            changes.append("Task Scheduler daemon (SlowaveDaemon)")
            changes.append("Task Scheduler worker (SlowaveWorker)")
        else:
            changes.append(f"worker ({system})")

    if changes:
        click.echo("  Removed:" if not dry_run else "  Would remove:")
        for c in changes:
            click.echo(f"    - {c}")
    else:
        click.echo("  No configuration found")

    if not dry_run:
        click.echo(f"\n  {safe_emoji('⚠️', '!!')}  Manual steps:")
        click.echo(
            "    - Claude Desktop Custom Instructions (delete the Slowave block if you added it manually)"
        )
        click.echo("    - Package: pipx uninstall slowave")
        click.echo("    - Database (optional): rm -rf ~/.slowave")
    click.echo()


from slowave.cli.backup import backup_cmd, restore_cmd
from slowave.cli.cleanup import cleanup_cmd

cli.add_command(setup_cmd)
cli.add_command(cleanup_cmd)
cli.add_command(backup_cmd)
cli.add_command(restore_cmd)


def main() -> None:
    cli(obj={})


if __name__ == "__main__":
    main()

# ---------------------------------------------------------------------------
# serve command group
# ---------------------------------------------------------------------------


@cli.group("serve")
def serve_cmd() -> None:
    """Manage the Slowave HTTP MCP daemon."""


@serve_cmd.command("start")
@click.option(
    "--host",
    default="127.0.0.1",
    show_default=True,
    help="Bind host (default 127.0.0.1, localhost only).",
)
@click.option(
    "--port",
    default=8766,
    show_default=True,
    help="Bind port for the HTTP MCP daemon.",
)
@click.option(
    "--log-level",
    default="INFO",
    show_default=True,
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False),
)
@click.option(
    "--foreground",
    "-f",
    is_flag=True,
    help="Run in foreground (block). Default: run in foreground (background not yet supported).",
)
def serve_start(
    host: str,
    port: int,
    log_level: str,
    foreground: bool,
) -> None:
    """Start the Slowave HTTP MCP daemon.

    Starts a persistent HTTP MCP server on http://<HOST>:<PORT>/mcp
    (default http://127.0.0.1:8766/mcp).  Only ONE daemon process should
    run at a time; the daemon enforces this via a PID file.

    Configure AI clients to use HTTP mode:

    \b
      {"mcpServers": {"slowave": {"type": "http", "url": "http://127.0.0.1:8766/mcp"}}}
    """
    from slowave.mcp.daemon import is_running, read_pid
    from slowave.mcp.http_server import main as http_main

    if is_running():
        click.echo(
            click.style("  ERROR", fg="red", bold=True)
            + f"  Daemon already running (pid={read_pid()}). "
            "Run 'slowave serve stop' to stop it."
        )
        sys.exit(1)

    click.echo(
        click.style("  Starting", fg="green", bold=True)
        + f"  Slowave HTTP MCP daemon on http://{host}:{port}/mcp"
    )
    click.echo("  Press Ctrl+C to stop.")
    http_main(host=host, port=port, log_level=log_level)


@serve_cmd.command("stop")
def serve_stop() -> None:
    """Stop the running Slowave HTTP MCP daemon."""
    from slowave.mcp.daemon import is_running, read_pid, stop_daemon

    if not is_running():
        click.echo("  No daemon is currently running.")
        return

    pid = read_pid()
    if stop_daemon():
        click.echo(click.style("  Stopped", fg="green") + f"  daemon (pid={pid}).")
    else:
        click.echo(click.style("  ERROR", fg="red") + "  Could not stop daemon.")
        sys.exit(1)


@serve_cmd.command("status")
@click.option("--json", "as_json", is_flag=True, help="JSON output.")
def serve_status(as_json: bool) -> None:
    """Show the status of the Slowave HTTP MCP daemon."""
    import json as _json
    import urllib.error
    import urllib.request

    from slowave.mcp.daemon import daemon_status

    status = daemon_status()
    mcp_url = "http://127.0.0.1:8766/mcp"
    health_url = "http://127.0.0.1:8766/health"

    # Try to fetch live health from running daemon
    health: dict = {}
    if status["running"]:
        try:
            with urllib.request.urlopen(health_url, timeout=2) as resp:
                health = _json.loads(resp.read())
        except Exception:
            pass

    payload = {
        **status,
        "mcp_url": mcp_url,
        "health": health,
    }

    if as_json:
        click.echo(_json.dumps(payload, indent=2))
        return

    from slowave.cli.output import Status, get_renderer

    renderer = get_renderer(use_emoji=False)
    renderer.title("Slowave HTTP MCP Daemon")

    renderer.section("Daemon")
    daemon_status_label = Status.OK if status["running"] else Status.SKIP
    renderer.check(
        "HTTP MCP Daemon",
        daemon_status_label,
        f"pid={status['pid']}" if status["running"] else "not running",
    )
    renderer.item("PID file", status["pid_file"], dim=True)
    renderer.item("MCP URL", mcp_url)

    if health:
        renderer.section("Live Status")
        renderer.item("Version", health.get("version", "?"))
        renderer.item("Active sessions", str(health.get("active_sessions", 0)))
        renderer.item("Database", health.get("db", "?"), dim=True)

    if not status["running"]:
        renderer.hint("Start with: slowave serve start")


@serve_cmd.command("restart")
@click.option("--host", default="127.0.0.1", show_default=True)
@click.option("--port", default=8766, show_default=True)
@click.option("--log-level", default="INFO", show_default=True)
def serve_restart(host: str, port: int, log_level: str) -> None:
    """Restart the Slowave HTTP MCP daemon."""
    import time as _time

    from slowave.mcp.daemon import is_running, read_pid, stop_daemon
    from slowave.mcp.http_server import main as http_main

    if is_running():
        pid = read_pid()
        click.echo(f"  Stopping daemon (pid={pid})...")
        stop_daemon()
        # Wait briefly for old process to exit
        for _ in range(10):
            _time.sleep(0.3)
            if not is_running():
                break
        if is_running():
            click.echo(click.style("  ERROR", fg="red") + "  Old daemon did not exit in time.")
            sys.exit(1)
        click.echo("  Old daemon stopped.")

    click.echo(
        click.style("  Starting", fg="green", bold=True)
        + f"  Slowave HTTP MCP daemon on http://{host}:{port}/mcp"
    )
    http_main(host=host, port=port, log_level=log_level)


# Expose serve under the main CLI
cli.add_command(serve_cmd)
