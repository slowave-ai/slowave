"""Slowave engine: top-level facade.

Wires SlowWave's latent CLS substrate (episodic+semantic+graph+transition+replay)
to Slowave's symbolic layer (raw events + episode text + typed schemas).
Public API for CLI and MCP integrations.
"""

from __future__ import annotations

import dataclasses
import logging
import sys
import uuid
from typing import Any

from slowave.core.config import DEFAULT_RECALL_TOP_K, SlowaveConfig
from slowave.core.consolidation import Consolidator
from slowave.core.context import WorkingMemoryGate, WorkingMemoryState
from slowave.core.scope import normalize_scope, scope_kind
from slowave.core.services.consolidation import ConsolidationService
from slowave.core.services.feedback import FeedbackService
from slowave.core.services.ingest import IngestService
from slowave.core.services.pattern_completion import PatternCompletionService
from slowave.core.services.retrieval import RecallResult, RetrievalService
from slowave.latent.episodic_store import EpisodicStore, EpisodicStoreConfig
from slowave.latent.graph_manager import GraphManager
from slowave.latent.replay_engine import ReplayEngine
from slowave.latent.retrieval import RetrievalPipeline
from slowave.latent.salience import SalienceEngine
from slowave.latent.semantic_store import SemanticStore, SemanticStoreConfig
from slowave.latent.temporal import TemporalProbe
from slowave.latent.transition_model import TransitionModel, TransitionModelConfig
from slowave.lifecycle import LIFECYCLE_VERSION
from slowave.storage.sqlite_db import SQLiteConfig, SQLiteDB
from slowave.symbolic.encoder import TextEncoder
from slowave.symbolic.episode_text import EpisodeTextStore
from slowave.symbolic.raw_log import RawLog
from slowave.symbolic.schema_store import Schema, SchemaStore

log = logging.getLogger(__name__)


def _prefix_date(text: str, ts: int) -> str:
    """Prepend an ISO date tag to an episode's text representation.

    Format: "[YYYY-MM-DD] <text>"

    Brain analogue: episodic memories are always bound to their temporal
    context — recalling an event recalls *when* it happened as part of
    the same trace.  Surfacing the date in the text lets a downstream
    answer layer (and keyword scorers) answer "when" and "how long ago" questions
    without needing a separate lookup.

    Falls back silently on any conversion error so a bad timestamp never
    breaks recall.
    """
    from datetime import datetime, timezone

    try:
        date_str = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")
        return f"[{date_str}] {text}" if text else f"[{date_str}]"
    except Exception:
        return text


def _default_memory_layer(schema_type: str) -> str:
    """Best-effort generic layer for explicit user memories."""
    t = str(schema_type or "").strip().lower()
    if t in {"preference", "interaction_preference", "constraint", "habit", "relationship"}:
        return "profile"
    if t in {"fact", "lesson", "warning"}:
        return "domain"
    return "workspace"


# --- Backward-compatible remember() result object ---
class RememberResult(int):
    """Backward-compatible result returned by ``SlowaveEngine.remember``.

    ``RememberResult`` is an ``int`` subclass whose integer value is the
    ``event_id``. Existing callers that compare, serialize, or store the return
    value as an integer continue to work, while Python API users can access the
    created memory/schema metadata through attributes.

    ``superseded_schema_ids`` is always ``[]``: supersession is decided
    asynchronously by consolidation's ``reconsolidate_labile_schemas()``, never
    at ``remember()`` time (see ``remember()``'s docstring and
    ``private/docs/iterations/20260720_supersession_classification_investigation.md``).
    The field is kept only for return-shape stability — do not poll it
    expecting same-call supersession signal.
    """

    event_id: int
    schema_id: int
    created_schema: "Schema | None"
    superseded_schema_ids: list[int]  # always [] — see class docstring

    def __new__(
        cls,
        event_id: int,
        *,
        schema_id: int,
        created_schema: "Schema | None" = None,
        superseded_schema_ids: list[int] | None = None,
    ) -> "RememberResult":
        obj = int.__new__(cls, event_id)
        obj.event_id = event_id
        obj.schema_id = schema_id
        obj.created_schema = created_schema
        obj.superseded_schema_ids = list(superseded_schema_ids or [])
        return obj

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation of the remember result."""
        return {
            "event_id": self.event_id,
            "schema_id": self.schema_id,
            "superseded_schema_ids": list(self.superseded_schema_ids),
        }


class SlowaveEngine:
    """Top-level facade: wires the latent substrate to the symbolic layer.

    Two halves, per docs/architecture.md:
    - Latent substrate ("SlowWave"): episodic/semantic vector stores, the
      association graph, and the replay/consolidation worker — the
      brain-inspired mechanism that decides what's salient and how memories
      cluster and connect, entirely via geometry (no LLM calls).
    - Symbolic layer ("Slowave"): the append-only raw event log, human-
      readable episode text, and typed schemas — what a caller actually
      reads and writes through the public API (remember/recall/etc).

    All public methods are thin wrappers delegating to one of the private
    `_ingest`/`_consolidation`/`_retrieval`/`_feedback` service objects built
    at the end of `__init__`; `remember()` is the one exception with
    non-trivial logic of its own (see its docstring).
    """

    def __init__(
        self,
        cfg: SlowaveConfig | None = None,
        *,
        shared_encoder: "TextEncoder | None" = None,
    ):
        """Build every store/service the engine needs and rebuild its FAISS
        indices from the DB, in four ordered phases (each a private method
        below, called in this exact order — the order is load-bearing, see
        each phase method's docstring for why): stores + migration, latent
        substrate, symbolic layer, services.
        """
        self.cfg = cfg or SlowaveConfig()
        self._init_stores_and_migration(shared_encoder)
        self._init_latent_substrate()
        self._init_symbolic_layer()
        self._init_services()

        # rebuild FAISS indices from DB
        self.episodic.reset_faiss_from_db()
        self.semantic.reset_faiss_from_db()

    def _init_stores_and_migration(self, shared_encoder: "TextEncoder | None") -> None:
        """Phase 1: DB, encoder, and the logic_version auto-migration.

        Must run before `_init_latent_substrate()`: a migration replays
        raw_events into derived tables, and the latent stores built next
        must see post-migration state before their `reset_faiss_from_db()`
        calls (end of `__init__`) run. The encoder is built before the
        migration check within this same phase because a migration needs a
        real encoder for schema embeddings.
        """
        schema_path = self.cfg.schema_path or SlowaveConfig.default_schema_path()

        self.db = SQLiteDB(SQLiteConfig(path=self.cfg.db_path))
        self.db.init_schema(schema_path)

        # encoder (lazy) — accept a pre-built shared encoder to avoid
        # reloading weights across multiple engines (e.g. in benchmarking).
        # Constructed here, before the migration check below, so a
        # logic_version rebuild (which needs a real encoder for schema
        # embeddings) can reuse it instead of loading a second copy.
        if shared_encoder is not None:
            self._encoder: TextEncoder | None = shared_encoder
        elif self.cfg.disable_encoder:
            self._encoder = None
        else:
            self._encoder = TextEncoder(self.cfg.encoder)

        # Auto-migration: if a release bumped current_logic_version, rebuild
        # all derived memory state from raw_events before anything below
        # reads it. Must run before EpisodicStore/SemanticStore/etc so their
        # reset_faiss_from_db() calls near the end of __init__ see
        # post-migration state. See slowave/core/services/rebuild.py and
        # private/docs/iterations/20260716_event-store-replay.md.
        from slowave.core.services.rebuild import RebuildService

        if RebuildService.needs_rebuild(self.db, self.cfg):
            try:
                if RebuildService.try_claim(self.db, self.cfg):
                    RebuildService.run(
                        self.db,
                        self.cfg,
                        encoder=self._encoder,
                        on_start=lambda: print(
                            f"Slowave: rebuilding memory for logic v{self.cfg.current_logic_version}"
                            " — one-time, may take a moment",
                            file=sys.stderr,
                        ),
                    )
                else:
                    # Another process is migrating (or just did). Wait
                    # briefly for its checkpoint rather than building our
                    # own stores against a mid-rebuild derived-table state;
                    # give up and proceed on current state if it's taking a
                    # while — self-heals on a later restart via the
                    # claim/reclaim logic in try_claim().
                    RebuildService.wait_for_completion(self.db, self.cfg)
            except Exception:
                log.exception("logic_version rebuild failed; continuing on current derived state")

    def _init_latent_substrate(self) -> None:
        """Phase 2: SlowWave latent substrate — geometry-only, no LLM calls
        anywhere below.

        Must run after `_init_stores_and_migration()`: every store built
        here reads `self.db`, which must already reflect post-migration
        state.
        """
        # self.salience: novelty/surprise scoring for new episodes (what's worth
        #   keeping); read by ingest when scoring episodes and by retrieval
        #   ranking. Brain analogue: what gets encoded strongly vs. fades.
        self.salience = SalienceEngine(self.cfg.salience)
        # self.episodic: the episode vector store (hippocampus analogue) —
        #   one row per formed episode, written during ingest and read/updated
        #   by the replay engine during consolidation.
        self.episodic = EpisodicStore(
            self.db, EpisodicStoreConfig(dim=self.cfg.dim, db_path=self.cfg.db_path)
        )
        # self.semantic: prototype centroid store — the consolidated,
        #   abstracted-away-from-any-single-episode representations replay
        #   clusters episodes into (neocortex analogue).
        self.semantic = SemanticStore(self.db, SemanticStoreConfig(dim=self.cfg.dim))
        # self.graph: association edges between prototypes (similarity,
        #   co-occurrence, predicted transition) that spreading-activation
        #   retrieval walks; maintained/pruned by the replay engine.
        self.graph = GraphManager(self.db, self.cfg.graph)
        # TransitionModel is always instantiated so predictive completion
        # fires in every benchmark run. The trained_steps == 0 guard in predict()
        # keeps it inert until at least one consolidation pass has run, so there
        # is no cost during the first session before any graph edges exist.
        # An explicit cfg.transition lets callers override dim or other params;
        # the default auto-derives dim from cfg.dim.
        _transition_cfg = (
            self.cfg.transition
            if self.cfg.transition is not None
            else TransitionModelConfig(dim=self.cfg.dim)
        )
        self.transition_model = TransitionModel(_transition_cfg)
        # Attach graph and semantic stores for graph-based prediction
        self.transition_model.attach_stores(self.graph, self.semantic)
        # Apply assignment_threshold shorthand if set in SlowaveConfig.
        # This overrides whatever is in cfg.replay so callers don't have to
        # construct a full ReplayConfig just to tune this one parameter.
        replay_cfg = self.cfg.replay
        if self.cfg.assignment_threshold is not None:
            replay_cfg = dataclasses.replace(
                replay_cfg,
                assignment_threshold=self.cfg.assignment_threshold,
                coarse_assignment_threshold=self.cfg.assignment_threshold,
            )
        replay_cfg = dataclasses.replace(
            replay_cfg, current_logic_version=self.cfg.current_logic_version
        )
        self.replay_engine = ReplayEngine(
            db=self.db,
            episodic=self.episodic,
            semantic=self.semantic,
            graph=self.graph,
            salience=self.salience,
            transition_model=self.transition_model,
            cfg=replay_cfg,
        )
        self.retrieval = RetrievalPipeline(
            episodic=self.episodic,
            semantic=self.semantic,
            graph=self.graph,
            cfg=self.cfg.retrieval,
            # Pass the trained transition model so retrieval can
            # use predicted next-state embeddings as a second cosine seed.
            transition_model=self.transition_model,
        )
        # let the replay engine rehearse retrieval against
        # prototype membership during the worker pass.
        self.replay_engine.attach_retrieval(self.retrieval)

    def _init_symbolic_layer(self) -> None:
        """Phase 3: Slowave symbolic layer — what callers actually
        read/write via the public API.

        Must run after `_init_latent_substrate()`: the consolidator built
        here needs `self.episodic`/`self.semantic` to already exist.
        """
        # self.raw_log: append-only log of every event_append() call; the
        #   source of truth episodes/schemas are derived from, and what a
        #   logic_version rebuild replays from scratch.
        self.raw_log = RawLog(self.db)
        # self.episode_text: human-readable text for each formed episode
        #   (episodic store above holds only the embedding + metadata).
        self.episode_text = EpisodeTextStore(self.db)
        # self.schemas: typed, consolidated facts/decisions/etc — the layer an
        #   LLM caller actually reads back via recall()/context_brief().
        self.schemas = SchemaStore(self.db, dim=self.cfg.dim)
        # self.working_memory_gate: assembles the salience/recency-gated
        #   "working memory" brief returned by context_brief().
        self.working_memory_gate = WorkingMemoryGate()

        # Latent consolidator: schemas are prototype geometry + lexical signatures.
        # Zero LLM calls in ingest, consolidation, and retrieval.
        from slowave.latent.schema import (
            GeometricContradictionJudge,
            LatentSchemaBuilder,
        )

        self.consolidator: Consolidator | None = Consolidator(
            db=self.db,
            semantic=self.semantic,
            episode_text=self.episode_text,
            schemas=self.schemas,
            encoder=self.encoder,
            latent_builder=LatentSchemaBuilder(),
            geometric_judge=GeometricContradictionJudge(self.cfg.judge),
            # The latent consolidator needs episode embeddings + ts.
            episodic_store=self.episodic,
            logic_version=self.cfg.current_logic_version,
        )

        # Temporal probe (embedding-space temporal compass).
        # Built once if an encoder is available; None otherwise (no-op at
        # recall time).  The probe pre-embeds 12 temporal-landmark phrases
        # so estimate_anchor() is just 12 dot products at query time.
        self._temporal_probe: TemporalProbe | None = None
        if self.encoder is not None:
            try:
                self._temporal_probe = TemporalProbe(self.encoder.encode)
            except Exception as e:
                log.warning("temporal probe init failed (will use now() fallback): %s", e)

    def _init_services(self) -> None:
        """Phase 4: public-facing service objects.

        Must run after `_init_symbolic_layer()`: every service below reads
        stores built there. Within this phase, order matters too —
        `_ingest` first, since `_consolidation`/`_retrieval` both depend on
        it.
        """
        self._ingest = IngestService(
            raw_log=self.raw_log,
            episodic=self.episodic,
            episode_text=self.episode_text,
            salience=self.salience,
            transition_model=self.transition_model,
            db=self.db,
        )
        self._consolidation = ConsolidationService(
            db=self.db,
            replay_engine=self.replay_engine,
            consolidator=self.consolidator,
            schemas=self.schemas,
            ingest=self._ingest,
            encoder=self.encoder,
        )
        self._retrieval = RetrievalService(
            episodic=self.episodic,
            semantic=self.semantic,
            graph=self.graph,
            schemas=self.schemas,
            encoder=self.encoder,
            episode_text=self.episode_text,
            raw_log=self.raw_log,
            retrieval=self.retrieval,
            transition_model=self.transition_model,
            temporal_probe=self._temporal_probe,
            working_memory_gate=self.working_memory_gate,
            db=self.db,
            retrieval_cfg=self.cfg.retrieval,
        )
        self._feedback = FeedbackService(
            db=self.db,
            schemas=self.schemas,
            cfg=self.cfg.feedback,
        )
        self._pattern_completion = PatternCompletionService(
            schemas=self.schemas,
            db=self.db,
        )

    @property
    def encoder(self) -> "TextEncoder | None":
        return self._encoder

    @encoder.setter
    def encoder(self, value: "TextEncoder | None") -> None:
        self._encoder = value
        # Propagate to services that hold their own encoder reference so that
        # post-construction assignment (e.g. test monkey-patching) stays in sync.
        if hasattr(self, "_retrieval"):
            self._retrieval.encoder = value

    @classmethod
    def from_config(
        cls,
        cfg: "SlowaveConfig | None" = None,
        *,
        shared_encoder: "TextEncoder | None" = None,
    ) -> "SlowaveEngine":
        """Canonical construction entry point. Prefer this over calling the
        constructor directly — the name makes intent explicit and call sites
        become easy to grep for engine construction."""
        return cls(cfg, shared_encoder=shared_encoder)

    # ---- sessions ----------------------------------------------------------
    def session_start(
        self,
        *,
        agent: str,
        scope: str | None = None,
        ts: int | None = None,
        goal: str | None = None,
        lifecycle_version: str | None = None,
    ) -> str:
        """Open a new session: mint an id every subsequent event_append() call
        in this task will be tagged with, before any perception/encoding
        happens. A session is the pre-episode grouping unit that session_end()
        later converts into episodic memories.

        Always mints a brand-new id — this method never reuses an existing
        session for a given scope/agent. Where session reuse exists at all,
        it's an MCP-layer concern (slowave/mcp/session_resolver.py, scope-keyed
        with a 1-hour TTL); the CLI path has no reuse and opens a fresh session
        on every `activate` call. A reader of this method in isolation would
        otherwise assume no reuse exists anywhere in the system.

        lifecycle_version: stamped onto the session row for later per-version
            activate/recall/feedback telemetry (WP-8). Defaults to the
            server's current contract version (slowave.lifecycle.LIFECYCLE_VERSION)
            -- this is the version of the cognitive-cycle contract the server
            enforces, not necessarily what's physically written into the
            caller's client instruction file (`slowave doctor` reports that).
            Callers (e.g. offline benchmark harnesses) may pass an explicit
            value or "" to leave it unset.
        """
        session_id = f"sess_{uuid.uuid4().hex[:12]}"
        scope_id = normalize_scope(scope=scope)
        self.raw_log.start_session(
            session_id=session_id,
            agent=agent,
            scope_id=scope_id,
            scope_kind=scope_kind(scope_id),
            ts=ts,
            goal=goal,
            lifecycle_version=(
                LIFECYCLE_VERSION if lifecycle_version is None else (lifecycle_version or None)
            ),
        )
        # Record the scope in the registry so the generalization denominator
        # (total_active_scopes) stays current without expensive table scans.
        # Note: this registers the scope even if the session never logs an
        # event or is never closed — a side effect of calling this method,
        # not of anything actually happening in the session.
        if scope_id:
            self.schemas.scope_registry.record(scope_id, scope_kind(scope_id), is_recall=False)
        return session_id

    def session_end(
        self,
        session_id: str,
        *,
        consolidate: bool = False,
        ts: int | None = None,
        outcome: str | None = None,
    ) -> dict[str, Any]:
        """End a session: form episodes from raw events.

        consolidate=False (default): fast path — only encodes the session into
        episodic memories. No LLM call, no replay, no blocking. The agent is
        never made to wait for consolidation.

        consolidate=True: additionally runs replay + latent schema consolidation
        synchronously. Use only for tests, scripts, or explicit one-shot
        invocations. In production, leave consolidate=False and run the
        background worker (slowave worker start) or call
        `slowave worker` or `slowave consolidate` on a schedule.

        Args:
            outcome: "success", "failure", "partial", or None — a session-level
                verdict, distinct from the per-memory reinforcement feedback
                recorded via retrieval_feedback()/context_feedback().
        """
        # No existence check: ending a session_id that was never started (or
        # already ended) is a silent no-op, not an error.
        #
        # Episode formation, however, must not silently re-run: if this
        # session was already ended (e.g. the idle-session reaper closed it,
        # and the client's later slowave_commit call reaches the same
        # session_id), form_episodes() would reprocess every raw event from
        # scratch and insert duplicate episode rows -- episodic_memories has
        # no UNIQUE constraint on event_id to catch this. Check before
        # end_session() overwrites ended_ts.
        already_ended = self.raw_log.is_session_ended(session_id)
        self.raw_log.end_session(session_id, ts=ts, outcome=outcome)

        if already_ended:
            episode_ids: list[int] = []
            stats: dict[str, Any] = {
                "session_id": session_id,
                "episodes_formed": 0,
                "already_ended": True,
            }
        else:
            episode_ids = self._ingest.form_episodes(session_id)
            stats = {"session_id": session_id, "episodes_formed": len(episode_ids)}

            # Back-link newly-formed episodes to schemas that were created via
            # remember() during this live session. During a live session, remember()
            # creates schemas with empty supporting_episode_ids and stores
            # schema_evidence rows with episode_id=NULL. The episodes are only
            # formed now, at session end, so we must update the links retroactively.
            # Without this, support_count stays 0 forever for agent-remembered facts,
            # depressing stability_score and schema_utility.
            if episode_ids:
                self._ingest.link_session_episodes(session_id=session_id, episode_ids=episode_ids)

        if consolidate:
            replay_stats = self.replay_engine.replay_once()
            stats["replay"] = replay_stats
            if self.consolidator is not None:
                # Consolidate the prototypes touched by this replay's mapped episodes.
                # Touched prototypes are those that have at least one of our new
                # episodes mapped to them, but we conservatively grab all current
                # prototypes that have a mapped episode in this session.
                touched = self._ingest.prototypes_for_episodes(episode_ids)
                cstats = self.consolidator.consolidate(prototype_ids=touched)
                stats["consolidation"] = {
                    "prototypes_processed": cstats.prototypes_processed,
                    "schemas_created": cstats.schemas_created,
                    "schemas_reinforced": cstats.schemas_reinforced,
                    "schemas_contradicted": cstats.schemas_contradicted,
                    "schemas_skipped": cstats.schemas_skipped,
                    "verdict_counts": dict(cstats.verdict_counts),
                    "near_dup_intercepts": cstats.near_dup_intercepts,
                    "gate_downgrades": dict(cstats.gate_downgrades),
                    "confidence_histogram": list(cstats.confidence_histogram),
                }
        return stats

    # ---- ingest -----------------------------------------------------------
    def event_append(
        self,
        *,
        session_id: str,
        type: str,
        content: str,
        metadata: dict[str, Any] | None = None,
        ts: int | None = None,
    ) -> int:
        # Sanitize content: strip whitespace and handle empty strings.
        # This prevents the error "messages: text content blocks must be non-empty"
        # from downstream Claude API calls. Empty content is logged with a placeholder.
        content_stripped = str(content).strip() if content else ""
        if not content_stripped:
            content_stripped = "[empty content]"
            log.warning(
                "event_append called with empty content for session %s, using placeholder",
                session_id,
            )

        # Graceful degradation: if the session_id doesn't exist in the sessions
        # table (e.g. the caller used "placeholder" or forgot to call
        # session_start), auto-register it as an ad-hoc session rather than
        # crashing with a FOREIGN KEY constraint failed error.
        # This is the most common mistake made by AI agents (including Claude Code)
        # when they skip the session_start → event → session_end lifecycle.
        if not self.raw_log.session_exists(session_id):
            log.warning(
                "event_append: session_id %r not found in sessions table — "
                "auto-registering as ad-hoc session. Call slowave_session_start "
                "first to associate events with a proper session.",
                session_id,
            )
            self.raw_log.start_session(
                session_id=str(session_id),
                agent="adhoc",
                ts=ts,
            )

        emb = None
        if self.encoder is not None:
            try:
                emb = self.encoder.encode(content_stripped)
            except Exception as e:
                log.warning("encoder failed: %s", e)
        return self.raw_log.append(
            session_id=session_id,
            type=type,
            content=content_stripped,
            metadata=metadata,
            embedding=emb,
            ts=ts,
            logic_version=self.cfg.current_logic_version,
        )

    def remember(
        self,
        *,
        content: str,
        type: str = "decision",
        session_id: str | None = None,
        agent: str = "cli",
        scope: str | None = None,
    ) -> RememberResult:
        """Explicit user-driven memory. Logged as a high-salience event.

        Two paths depending on whether a live session_id is provided:

        - No session_id: create an ad-hoc session, append the event, end
          the session immediately, form episodes, then create the schema
          backed by those episodes.  Fully self-contained; nothing leaks
          into any other session.

        - session_id provided: append the remember event to the caller's
          live session (it will be encoded into episodes when the caller
          eventually calls session_end), then create the schema immediately
          with an empty episode list.  The session is NOT ended here — that
          is the caller's responsibility.  This avoids double episode
          formation: once here and again when session_end runs.

        Returns a ``RememberResult``. It behaves like the old integer event id
        for backward compatibility, and also exposes ``event_id``, ``schema_id``,
        and ``created_schema`` for Python API users. It also exposes
        ``superseded_schema_ids``, which is always ``[]`` — see
        ``RememberResult``'s docstring for why.
        """
        caller_owns_session = session_id is not None

        if not caller_owns_session:
            session_id = self.session_start(agent=agent, scope=scope)

        event_id = self.event_append(
            session_id=session_id,
            type=f"remember:{type}",
            content=content,
            metadata={"explicit": True, "declared_type": type},
        )

        emb = self.encoder.encode(content) if self.encoder is not None else None

        if caller_owns_session:
            # The caller's session is still live — do not end or re-encode it.
            # Create the schema immediately so it is available for recall, but
            # leave supporting_episode_ids empty; the episodes will be formed
            # and linked during the caller's session_end.
            episode_ids: list[int] = []
        else:
            # Ad-hoc session: close it and form episodes right now so the
            # schema is immediately backed by episodic evidence.
            self.raw_log.end_session(session_id)
            episode_ids = self._ingest.form_episodes(session_id)

        scope_id = normalize_scope(scope=scope)
        new_schema_id = self.schemas.create(
            content_text=content,
            facets={
                "schema_class": type,
                "source": "explicit_remember",
                "source_kind": "explicit_remember",
                "memory_layer": _default_memory_layer(type),
                "injectable": True,
            },
            tags=[type, "explicit"],
            embedding=emb,
            scope_id=scope_id,
            confidence=1.0,
            salience=1.4,
            supporting_episode_ids=episode_ids,
            evidence=[(episode_ids[0] if episode_ids else None, event_id, content, 1.0)],
            logic_version=self.cfg.current_logic_version,
        )

        # Pattern-completion check (hippocampal familiarity check at encoding
        # time, not classification): flags a close same-scope neighbor as
        # labile for consolidation to reconsolidate later, and
        # reinforces/skips cross-scope neighbors for the promotion ladder.
        # remember() itself never decides supersedes/refines/relates_to —
        # see PatternCompletionService's class docstring for the full
        # rationale (moved there from this method in the 2026-07-21
        # simplification pass; originally documented in
        # private/docs/iterations/20260720_supersession_classification_investigation.md).
        if emb is not None:
            self._pattern_completion.process_candidates(
                new_schema_id=new_schema_id,
                emb=emb,
                event_id=event_id,
                content=content,
                scope_id=scope_id,
            )

        try:
            created_schema = self.schemas.get(new_schema_id)
        except KeyError:
            created_schema = None

        return RememberResult(
            event_id,
            schema_id=new_schema_id,
            created_schema=created_schema,
            superseded_schema_ids=[],
        )

    # ---- consolidation ----------------------------------------------------
    def consolidate_once(
        self, *, triggered_by: str = "worker", decay_idle_days: float = 30.0
    ) -> dict[str, Any]:
        return self._consolidation.consolidate_once(
            triggered_by=triggered_by, decay_idle_days=decay_idle_days
        )

    def refresh_indices(self) -> None:
        self._retrieval.refresh_indices()

    def recall(
        self,
        query: str,
        *,
        top_k: int = DEFAULT_RECALL_TOP_K,
        evidence: bool = False,
        scope: str | None = None,
        mode: str = "default",
        diagnose: bool = False,
        refresh: bool = True,
        min_relevance: float | None = None,
        graph_channels: str | None = None,
        min_neighbor_relevance: float | None = None,
    ) -> RecallResult:
        kwargs: dict[str, Any] = {}
        if min_relevance is not None:
            kwargs["min_relevance"] = min_relevance
        if graph_channels is not None:
            kwargs["graph_channels"] = graph_channels
        if min_neighbor_relevance is not None:
            kwargs["min_neighbor_relevance"] = min_neighbor_relevance
        return self._retrieval.recall(
            query,
            top_k=top_k,
            evidence=evidence,
            scope=scope,
            mode=mode,
            diagnose=diagnose,
            refresh=refresh,
            **kwargs,
        )

    def context(self, *, scope: str | None = None, limit: int = 10) -> list[Schema]:
        return self._retrieval.context(scope=scope, limit=limit)

    def context_brief(self, **kwargs: Any) -> WorkingMemoryState:
        return self._retrieval.context_brief(**kwargs)

    # ---- inspection -------------------------------------------------------
    def get_schema(self, schema_id: int) -> Schema:
        return self.schemas.get(schema_id)

    def list_schemas(self, **kwargs: Any) -> list[Schema]:
        return self.schemas.list(**kwargs)

    def stats(self) -> dict[str, Any]:
        return {
            "episodes": self.episodic.count(),
            "prototypes": self.semantic.count(),
            "schemas": self.schemas.count(),
            # Always 0: Slowave has no separate procedural-memory (skills/
            # habits) store by design — see docs/design.md "Behavioral
            # Patterns". Behavioral patterns are captured implicitly via
            # graph association strength instead of an explicit store.
            "procedures": 0,
            "edges": self.graph.edge_count(),
        }

    def schema_health(self) -> dict[str, Any]:
        return self.schemas.health()

    def dedup_schemas_exact(self, *, dry_run: bool = True) -> dict[str, Any]:
        return self.schemas.dedup_exact(dry_run=dry_run)

    def forget_schema(self, schema_id: int, *, reason: str | None = None) -> None:
        """Suppress a schema from retrieval. CLI/dashboard-initiated only --
        deliberately not exposed as an MCP tool (see schema_store.VALID_STATUS
        comment for the trust-boundary rationale)."""
        self.schemas.forget(schema_id, reason=reason)

    def unforget_schema(self, schema_id: int) -> str:
        """Undo forget_schema(), returning the schema to its prior status."""
        return self.schemas.unforget(schema_id)

    def decay_schemas(self, *, idle_days: float = 30.0, dry_run: bool = False) -> dict[str, Any]:
        """Decay salience of active schemas that have never been recalled.

        Wraps ``SchemaStore.decay_unused``. Exposed here so the CLI and MCP
        can trigger decay independently of a full consolidation pass.
        """
        return self.schemas.decay_unused(idle_days=idle_days, dry_run=dry_run)

    def close(self) -> None:
        self.db.close()

    def record_retrieval(self, **kwargs: Any) -> None:
        self._feedback.record_retrieval(**kwargs)

    def record_context_recall(self, *, context_id: str, **kwargs: Any) -> None:
        self._feedback.record_context_recall(context_id=context_id, **kwargs)

    def retrieval_feedback(self, **kwargs: Any) -> dict[str, Any]:
        return self._feedback.retrieval_feedback(**kwargs)

    def context_feedback(self, *, context_id: str, **kwargs: Any) -> dict[str, Any]:
        return self._feedback.context_feedback(context_id=context_id, **kwargs)
