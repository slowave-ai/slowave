"""RebuildService: auto-migrates derived memory state on a logic_version bump.

When Sibill ships a change to ingest/replay/consolidation logic that would
produce different output for already-ingested raw_events, it bumps
SlowaveConfig.current_logic_version. Every customer's local Slowave instance
notices this on its own next startup (SlowaveEngine.__init__ calls into this
module before constructing any of the "live" stores) and rebuilds its entire
derived memory state from raw_events — zero manual action from the customer.

See private/docs/iterations/20260716_event-store-replay.md for the full
design rationale, in particular why this rebuilds in place against the live
DB (no file copy/swap, no ATTACH-based staging promotion) rather than the
file-swap pattern slowave/cli/backup.py uses for `slowave restore`: this
module runs before the engine is handed to any caller, so there is no
concurrent reader of derived state to protect with a snapshot, and safety
instead comes from idempotency — raw_events (the source of truth) is never
touched, so an interrupted rebuild is always safe to wipe and retry.
"""

from __future__ import annotations

import dataclasses
import logging
import sqlite3
import time
from dataclasses import dataclass
from typing import Any, Callable

from slowave.core.config import SlowaveConfig
from slowave.core.consolidation import ConsolidationStats, Consolidator
from slowave.core.services.ingest import IngestService
from slowave.latent.episodic_store import EpisodicStore, EpisodicStoreConfig
from slowave.latent.graph_manager import GraphManager
from slowave.latent.replay_engine import ReplayEngine
from slowave.latent.salience import SalienceEngine
from slowave.latent.semantic_store import SemanticStore, SemanticStoreConfig
from slowave.latent.transition_model import TransitionModel, TransitionModelConfig
from slowave.storage.sqlite_db import SQLiteDB
from slowave.symbolic.episode_text import EpisodeTextStore
from slowave.symbolic.raw_log import RawLog
from slowave.symbolic.schema_store import SchemaStore

log = logging.getLogger(__name__)

# Stale-claim lease window: if a claimed migration hasn't completed within
# this many seconds, a later startup may steal it and retry. Short on
# purpose — a rebuild on a single local customer DB should be fast, and
# "zero manual action" means recovery can't depend on a human noticing a
# stuck instance and intervening.
_CLAIM_STALE_SECONDS = 180
# After this many attempts on the same version, stop retrying automatically
# (a deterministically-crashing migration would otherwise retry forever, once
# per startup) — the instance keeps running on its last-good derived state.
_MAX_CLAIM_ATTEMPTS = 5

# Tables holding *derived* memory state — wiped and rebuilt from raw_events
# on migration. Deliberately excludes: sessions/raw_events/raw_events_fts
# (source of truth, never touched) and logic_versions/replay_checkpoints/
# worker_runs/scope_registry/graph_health_snapshots/context_recall_*/
# context_feedback_events (audit/telemetry, not "latent state").
#
# Ordered children-before-parents so plain DELETE never trips a foreign-key
# violation regardless of each table's ON DELETE clause (see schema.sql).
# Adding a new derived table there? Add it here too, in the right position,
# or a rebuild will silently leave it stale — same discipline as the
# missing_columns catalogue in sqlite_db.py.
_DERIVED_TABLES = (
    "schema_evidence",
    "schema_prototype_map",
    "schema_relations",
    "episode_prototype_map",
    "prototype_edges",
    "consolidation_debug",
    "episode_text",
    "episodes_fts",
    "schemas_fts",
    "schemas",
    "semantic_prototypes",
    "episodic_memories",
)


@dataclass(frozen=True)
class RebuildStats:
    logic_version: str
    sessions_processed: int
    episodes_formed: int
    episode_count: int
    prototype_count: int
    schema_count: int
    duration_ms: int
    replay_stats: dict[str, Any]
    consolidation_stats: ConsolidationStats


class RebuildService:
    """Stateless — every method takes the DB/config it needs explicitly."""

    @staticmethod
    def needs_rebuild(db: SQLiteDB, cfg: SlowaveConfig) -> bool:
        """True iff raw_events contains anything tagged with a different
        logic_version than the one currently running, and this version
        hasn't been rebuilt yet.

        Deliberately NOT just "no replay_checkpoints row for this version" —
        a DB that has always been ingested under the current version (the
        overwhelmingly common case: no version bump has ever happened) would
        never have a checkpoint row either, since only a rebuild writes one.
        Without the logic_version comparison, every second engine
        construction against any populated DB would look like it "needs"
        a rebuild and wipe perfectly current derived state for no reason.
        """
        conn = db.connect()
        stale_row = conn.execute(
            "SELECT 1 FROM raw_events WHERE logic_version != ? LIMIT 1",
            (cfg.current_logic_version,),
        ).fetchone()
        if stale_row is None:
            return False
        checkpoint_row = conn.execute(
            "SELECT 1 FROM replay_checkpoints WHERE logic_version = ? LIMIT 1",
            (cfg.current_logic_version,),
        ).fetchone()
        return checkpoint_row is None

    @staticmethod
    def try_claim(db: SQLiteDB, cfg: SlowaveConfig, *, now: int | None = None) -> bool:
        """Attempt to become the process that performs this rebuild.

        Returns True iff this call now owns the claim (either by inserting
        the first row for this version, or by stealing a stale/abandoned
        one). Returns False if another process already holds a live claim —
        callers should defer, not retry immediately.
        """
        if now is None:
            now = int(time.time())
        conn = db.connect()
        try:
            conn.execute(
                "INSERT INTO logic_versions "
                "(version, applied_ts, description, replayed_from_scratch, "
                " claimed_ts, claim_attempts) "
                "VALUES (?, ?, ?, 0, ?, 1)",
                (
                    cfg.current_logic_version,
                    now,
                    cfg.current_logic_version_description or None,
                    now,
                ),
            )
            conn.commit()
            return True
        except sqlite3.IntegrityError:
            conn.rollback()
            cur = conn.execute(
                "UPDATE logic_versions SET claimed_ts = ?, claim_attempts = claim_attempts + 1 "
                "WHERE version = ? AND replayed_from_scratch = 0 "
                "AND claimed_ts < ? AND claim_attempts < ?",
                (
                    now,
                    cfg.current_logic_version,
                    now - _CLAIM_STALE_SECONDS,
                    _MAX_CLAIM_ATTEMPTS,
                ),
            )
            conn.commit()
            return cur.rowcount > 0

    @staticmethod
    def wait_for_completion(
        db: SQLiteDB,
        cfg: SlowaveConfig,
        *,
        timeout_s: float = 5.0,
        poll_interval_s: float = 0.2,
    ) -> bool:
        """Poll for another process's in-flight rebuild to finish.

        Called by a process that lost the claim race in try_claim(): rather
        than immediately constructing its own stores against a mid-rebuild
        derived-table state, wait briefly for the winner's checkpoint to
        appear. Returns True if the rebuild completed within the window,
        False on timeout (caller proceeds anyway — safe, just means this
        one startup may see stale-but-consistent old state; it self-heals
        on a later restart).
        """
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if not RebuildService.needs_rebuild(db, cfg):
                return True
            time.sleep(poll_interval_s)
        return not RebuildService.needs_rebuild(db, cfg)

    @staticmethod
    def run(
        db: SQLiteDB,
        cfg: SlowaveConfig,
        *,
        encoder: Any = None,
        on_start: Callable[[], None] | None = None,
    ) -> RebuildStats:
        """Wipe and rebuild all derived memory state from raw_events.

        Constructs only the components needed to drive form_episodes() +
        replay_all() + consolidate_all() directly against `db` — not a
        nested SlowaveEngine, which would re-run this same startup check
        and either recurse or need a special internal flag to avoid it.

        `encoder`, if given, should be the caller's already-loaded
        TextEncoder (reused, never constructed here) so rebuilt schemas get
        real embeddings instead of degrading to embedding-less claims.

        Raises on failure rather than swallowing errors — the caller
        (SlowaveEngine.__init__) is responsible for catching broadly so a
        migration bug never prevents the engine from starting.
        """
        started = time.monotonic()
        if on_start is not None:
            on_start()

        conn = db.connect()
        for table in _DERIVED_TABLES:
            conn.execute(f"DELETE FROM {table}")
        conn.commit()

        salience = SalienceEngine(cfg.salience)
        episodic = EpisodicStore(db, EpisodicStoreConfig(dim=cfg.dim, db_path=cfg.db_path))
        semantic = SemanticStore(db, SemanticStoreConfig(dim=cfg.dim))
        graph = GraphManager(db, cfg.graph)
        transition_cfg = (
            cfg.transition if cfg.transition is not None else TransitionModelConfig(dim=cfg.dim)
        )
        transition_model = TransitionModel(transition_cfg)
        transition_model.attach_stores(graph, semantic)

        replay_cfg = cfg.replay
        if cfg.assignment_threshold is not None:
            replay_cfg = dataclasses.replace(
                replay_cfg,
                assignment_threshold=cfg.assignment_threshold,
                coarse_assignment_threshold=cfg.assignment_threshold,
            )
        replay_cfg = dataclasses.replace(
            replay_cfg, current_logic_version=cfg.current_logic_version
        )
        replay_engine = ReplayEngine(
            db=db,
            episodic=episodic,
            semantic=semantic,
            graph=graph,
            salience=salience,
            transition_model=transition_model,
            cfg=replay_cfg,
        )

        raw_log = RawLog(db)
        episode_text = EpisodeTextStore(db)
        schemas = SchemaStore(db, dim=cfg.dim)
        ingest = IngestService(
            raw_log=raw_log,
            episodic=episodic,
            episode_text=episode_text,
            salience=salience,
            transition_model=transition_model,
            db=db,
        )

        from slowave.latent.schema import GeometricContradictionJudge, LatentSchemaBuilder

        consolidator = Consolidator(
            db=db,
            semantic=semantic,
            episode_text=episode_text,
            schemas=schemas,
            encoder=encoder,
            latent_builder=LatentSchemaBuilder(),
            geometric_judge=GeometricContradictionJudge(cfg.judge),
            episodic_store=episodic,
            logic_version=cfg.current_logic_version,
        )

        session_rows = conn.execute(
            "SELECT id FROM sessions ORDER BY started_ts ASC, id ASC"
        ).fetchall()
        session_ids = [str(r["id"]) for r in session_rows]

        episodes_formed = 0
        for session_id in session_ids:
            episodes_formed += len(ingest.form_episodes(session_id))

        replay_stats = replay_engine.replay_all()
        consolidation_stats = consolidator.consolidate_all()

        now = int(time.time())
        duration_ms = int((time.monotonic() - started) * 1000)

        last_event_id = conn.execute("SELECT COALESCE(MAX(id), 0) AS m FROM raw_events").fetchone()[
            "m"
        ]
        last_episode_id = conn.execute(
            "SELECT COALESCE(MAX(id), 0) AS m FROM episodic_memories"
        ).fetchone()["m"]
        episode_count = conn.execute("SELECT COUNT(*) AS n FROM episodic_memories").fetchone()["n"]
        prototype_count = conn.execute("SELECT COUNT(*) AS n FROM semantic_prototypes").fetchone()[
            "n"
        ]
        schema_count = conn.execute("SELECT COUNT(*) AS n FROM schemas").fetchone()["n"]

        conn.execute(
            "INSERT INTO replay_checkpoints "
            "(created_ts, logic_version, last_event_id, last_episode_id, "
            " episode_count, prototype_count, schema_count, duration_ms) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                now,
                cfg.current_logic_version,
                last_event_id,
                last_episode_id,
                episode_count,
                prototype_count,
                schema_count,
                duration_ms,
            ),
        )
        conn.execute(
            "UPDATE logic_versions SET replayed_from_scratch = 1 WHERE version = ?",
            (cfg.current_logic_version,),
        )
        conn.commit()

        return RebuildStats(
            logic_version=cfg.current_logic_version,
            sessions_processed=len(session_ids),
            episodes_formed=episodes_formed,
            episode_count=episode_count,
            prototype_count=prototype_count,
            schema_count=schema_count,
            duration_ms=duration_ms,
            replay_stats=replay_stats,
            consolidation_stats=consolidation_stats,
        )
