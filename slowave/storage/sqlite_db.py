from __future__ import annotations

import sqlite3
import threading
from dataclasses import dataclass, field
from pathlib import Path

from slowave.core.paths import default_db_path


@dataclass(frozen=True)
class SQLiteConfig:
    path: str = field(default_factory=default_db_path)


class SQLiteDB:
    """Very small SQLite wrapper.

    We keep this intentionally minimal to avoid overengineering.

    Thread safety: each thread gets its own sqlite3.Connection via
    threading.local().  SQLite in WAL mode is safe for concurrent reads
    from multiple connections; writes serialize at the SQLite level.
    This avoids the "SQLite objects created in a thread can only be used
    in that same thread" error that surfaces when the MCP server mixes
    asyncio event-loop calls with run_in_executor threadpool calls.
    """

    def __init__(self, cfg: SQLiteConfig):
        self.cfg = cfg
        self._local = threading.local()  # per-thread connection storage

    def connect(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.cfg.path, timeout=30.0)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA busy_timeout = 30000")  # 30 second timeout for concurrent access
            conn.execute("PRAGMA foreign_keys = ON")
            # SQLite performance pragmas: WAL mode allows concurrent readers while a writer is active
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA cache_size=-65536")  # 64MB page cache
            conn.execute("PRAGMA temp_store=MEMORY")
            self._local.conn = conn
        return conn

    def close(self) -> None:
        """Close the current thread's connection, if open."""
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None

    def init_schema(self, schema_path: str) -> None:
        conn = self.connect()
        sql = Path(schema_path).read_text(encoding="utf-8")
        # Pre-migrations: bring legacy tables up to the column shape the
        # schema script expects. The script creates indexes on columns
        # (e.g. scale) that did not exist before Stage 9; without this
        # pre-pass, executing the script against an old DB would fail
        # on the CREATE INDEX before the migrations had a chance to run.
        self._apply_pre_migrations()
        conn.executescript(sql)
        conn.commit()
        # Post-migrations: anything that needs the new schema script to
        # have run first (table rebuilds for PK changes, index refresh).
        self._apply_post_migrations()

    def _apply_pre_migrations(self) -> None:
        """Bring legacy tables up to the column shape the schema script
        expects. Runs BEFORE the schema script's CREATE INDEX statements.

        Adds columns that didn't exist in earlier versions of the DB but
        that newer indexes / queries depend on. Must be idempotent and
        must not fail when the table doesn't exist yet (fresh install).

        The catalogue below lists every (table, column, sqlite_type)
        added across the codebase's lifetime. Adding a new column to
        schema.sql? Add a row here too, or legacy DBs will break on the
        next open.
        """
        import sqlite3 as _sqlite3

        conn = self.connect()

        # Legacy DBs may still have the schemas table's boolean "labile" flag
        # under its old column name, needs_review — distinct from the
        # unrelated status='needs_review' string value on the same table
        # (see core/08-feedback.md's "Labile State & Reconsolidation"
        # section). Rename it to is_labile before schema.sql's CREATE INDEX
        # on is_labile runs, or that statement fails against a DB that still
        # has the old name. RENAME COLUMN requires SQLite >= 3.25 (2018);
        # every supported Python's stdlib sqlite3 bundles a newer version.
        # The ("schemas", "is_labile", ...) row in missing_columns below is
        # a safety net for a DB old enough to have neither column at all.
        t = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='schemas'"
        ).fetchone()
        if t is not None:
            cols = {r["name"] for r in conn.execute("PRAGMA table_info(schemas)").fetchall()}
            if "needs_review" in cols and "is_labile" not in cols:
                conn.execute("ALTER TABLE schemas RENAME COLUMN needs_review TO is_labile")
                conn.commit()

        missing_columns = [
            # Stage 9 (commit 777ea1d)
            ("semantic_prototypes", "scale", "TEXT NOT NULL DEFAULT 'fine'"),
            # `schemas` table evolved heavily between v1 (May) and now.
            # Every column the current schema.sql declares but legacy
            # DBs may lack:
            ("schemas", "prototype_id", "INTEGER"),
            ("schemas", "facets_json", "TEXT NOT NULL DEFAULT '{}'"),
            ("schemas", "tags_json", "TEXT NOT NULL DEFAULT '{\"tags\":[]}'"),
            ("schemas", "scope_id", "TEXT"),
            ("schemas", "scope_kind", "TEXT"),
            ("schemas", "status", "TEXT NOT NULL DEFAULT 'active'"),
            ("schemas", "confidence", "REAL NOT NULL DEFAULT 1.0"),
            ("schemas", "salience", "REAL NOT NULL DEFAULT 1.0"),
            ("schemas", "embedding", "BLOB"),
            ("schemas", "dim", "INTEGER"),
            ("schemas", "facet_axes", "BLOB"),
            ("schemas", "facet_strengths", "BLOB"),
            ("schemas", "n_facet_axes", "INTEGER NOT NULL DEFAULT 0"),
            # Safety net for the needs_review -> is_labile rename above: a DB
            # old enough to have neither column gets this added fresh.
            ("schemas", "is_labile", "INTEGER NOT NULL DEFAULT 0"),
            ("sessions", "scope_id", "TEXT"),
            ("sessions", "scope_kind", "TEXT"),
            ("context_recall_events", "retrieval_type", "TEXT NOT NULL DEFAULT 'context'"),
            ("context_recall_events", "scope_id", "TEXT"),
            ("context_recall_events", "scope_kind", "TEXT"),
            ("context_recall_events", "goal", "TEXT"),
            ("context_recall_events", "task_type", "TEXT"),
            ("context_recall_events", "situation_json", "TEXT NOT NULL DEFAULT '{}'"),
            ("context_recall_events", "requirements_json", "TEXT NOT NULL DEFAULT '[]'"),
            ("context_recall_items", "retrieval_type", "TEXT NOT NULL DEFAULT 'context'"),
            ("context_recall_items", "admitted", "INTEGER NOT NULL DEFAULT 1"),
            ("context_feedback_events", "retrieval_type", "TEXT NOT NULL DEFAULT 'context'"),
            ("context_feedback_events", "scope_id", "TEXT"),
            ("context_feedback_events", "scope_kind", "TEXT"),
            ("context_feedback_events", "situation_json", "TEXT NOT NULL DEFAULT '{}'"),
            ("context_feedback_events", "requirements_json", "TEXT NOT NULL DEFAULT '[]'"),
            ("context_feedback_events", "used_procedure_ids_json", "TEXT NOT NULL DEFAULT '[]'"),
            (
                "context_feedback_events",
                "irrelevant_procedure_ids_json",
                "TEXT NOT NULL DEFAULT '[]'",
            ),
            ("context_feedback_events", "stale_procedure_ids_json", "TEXT NOT NULL DEFAULT '[]'"),
            ("context_feedback_events", "wrong_procedure_ids_json", "TEXT NOT NULL DEFAULT '[]'"),
            # source_content: raw event content joined without role prefix; used as schema claim
            ("episode_text", "source_content", "TEXT"),
            # generalization_stage: cross-scope generalization level (Stage 11)
            # Procedural memory Tier 1 (v4 §7: schema migrations for enforcement)
            ("sessions", "goal", "TEXT"),
            ("sessions", "outcome", "TEXT"),
            ("procedural_memories", "source", "TEXT NOT NULL DEFAULT 'implicit'"),
            ("procedural_memories", "superseded_by_id", "INTEGER"),
            ("procedural_memories", "generalization_stage", "INTEGER NOT NULL DEFAULT 0"),
            ("schemas", "generalization_stage", "INTEGER NOT NULL DEFAULT 0"),
            # Worker run log: additional tracking columns
            ("worker_runs", "procedures_promoted", "INTEGER NOT NULL DEFAULT 0"),
            ("worker_runs", "procedures_generalized", "INTEGER NOT NULL DEFAULT 0"),
            ("worker_runs", "schemas_decayed", "INTEGER NOT NULL DEFAULT 0"),
            # Event-store replay point 2 (2026-07-16): logic version tagging.
            ("raw_events", "logic_version", "TEXT NOT NULL DEFAULT '0'"),
            ("schemas", "logic_version", "TEXT NOT NULL DEFAULT '0'"),
            ("semantic_prototypes", "logic_version", "TEXT NOT NULL DEFAULT '0'"),
            # Auto-migration lock fields (2026-07-16): RebuildService.try_claim.
            ("logic_versions", "claimed_ts", "INTEGER"),
            ("logic_versions", "claim_attempts", "INTEGER NOT NULL DEFAULT 0"),
            # WP-6 (2026-07-28): distinguishes why an admitted row was shown --
            # 'direct' (query-relevant), 'exploration' (salience-filled slot),
            # or 'graph' (association, not direct relevance) -- so the
            # co-activation writer can stop treating serendipitous
            # co-presentation the same as genuinely query-driven co-retrieval.
            # Existing rows default to 'direct', preserving their prior
            # (undifferentiated) treatment.
            ("context_recall_items", "pathway", "TEXT NOT NULL DEFAULT 'direct'"),
            # WP-8 (2026-07-28): lifecycle-instructions contract version
            # stamped at session_start time -- see slowave/lifecycle.py.
            ("sessions", "lifecycle_version", "TEXT"),
            # WP-8 (2026-07-28): per-call lifecycle-instructions contract
            # version -- the reliable attribution point for activate/recall
            # telemetry, since recall() never sets session_id.
            ("context_recall_events", "lifecycle_version", "TEXT"),
        ]

        for table, column, type_spec in missing_columns:
            # Skip silently when the table itself doesn't exist (fresh
            # install — the schema script will create it with the right
            # shape in a moment).
            t = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                (table,),
            ).fetchone()
            if t is None:
                continue
            try:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {type_spec}")
            except _sqlite3.OperationalError as e:
                if "duplicate column" not in str(e).lower():
                    raise
        conn.commit()

    def _apply_post_migrations(self) -> None:
        """Idempotent forward migrations that need the schema script to
        have already run (e.g. table rebuilds that change a PK).

        See slowave/storage/schema.sql for the authoritative shape.
        """
        conn = self.connect()

        # ---- Phase 1 P1: drop procedural_memories tables (2026-06-25) ---------
        # Procedural behavior is now implicit via schemas + prototypes +
        # TransitionModel + spreading activation.
        for tbl in ("procedural_memory_evidence", "procedural_memories"):
            conn.execute(f"DROP TABLE IF EXISTS {tbl}")
        # Legacy schema declared PRIMARY KEY (episode_id) which constrained
        # an episode to one prototype. Stage 9 needs one mapping per
        # scale, so the PK is now (episode_id, prototype_id). SQLite
        # cannot ALTER a PRIMARY KEY in place; detect the old shape and
        # rebuild the table preserving every row.
        row = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' " "AND name='episode_prototype_map'"
        ).fetchone()
        if row is not None:
            old_sql = str(row["sql"])
            needs_rebuild = (
                "PRIMARY KEY (episode_id)" in old_sql
                and "PRIMARY KEY (episode_id, prototype_id)" not in old_sql
            )
            if needs_rebuild:
                # Disable FK enforcement for the rebuild. Some legacy
                # rows may reference deleted episodes/prototypes (the
                # old PK constraint didn't have FKs); the rebuild
                # preserves what's there and the engine treats orphans
                # as no-ops on read. Re-enable FKs after.
                conn.execute("PRAGMA foreign_keys = OFF")
                conn.executescript("""
                    CREATE TABLE episode_prototype_map_new (
                      episode_id INTEGER NOT NULL,
                      prototype_id INTEGER NOT NULL,
                      PRIMARY KEY (episode_id, prototype_id),
                      FOREIGN KEY (episode_id) REFERENCES episodic_memories(id) ON DELETE CASCADE,
                      FOREIGN KEY (prototype_id) REFERENCES semantic_prototypes(id) ON DELETE CASCADE
                    );
                    INSERT OR IGNORE INTO episode_prototype_map_new (episode_id, prototype_id)
                      SELECT episode_id, prototype_id FROM episode_prototype_map;
                    DROP TABLE episode_prototype_map;
                    ALTER TABLE episode_prototype_map_new RENAME TO episode_prototype_map;
                    """)
                conn.execute("PRAGMA foreign_keys = ON")
        # Always (re-)create the indexes; IF NOT EXISTS makes it safe.
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_map_prototype_id "
            "ON episode_prototype_map (prototype_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_map_episode_id " "ON episode_prototype_map (episode_id)"
        )

        # Fix Bug-2: schema_evidence duplicate NULL-key rows.
        # The table's PRIMARY KEY is (schema_id, episode_id, raw_event_id).
        # In SQLite NULL != NULL in PK constraints, so INSERT OR REPLACE
        # never deduplicates rows where raw_event_id IS NULL — it just
        # inserts a new row every time. A partial UNIQUE index covering
        # the NULL case makes INSERT OR REPLACE behave correctly for free,
        # with no application-layer changes needed.
        conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_schema_evidence_episode_null "
            "ON schema_evidence(schema_id, episode_id) "
            "WHERE raw_event_id IS NULL"
        )

        # B-14: drop LLM-era columns from consolidation_debug (2026-07-04).
        # Consolidation is zero-LLM; these columns were always empty.
        for col in ("prompt_text", "response_json", "extracted_claims_json"):
            try:
                conn.execute(f"ALTER TABLE consolidation_debug DROP COLUMN {col}")
            except Exception:
                pass  # column already gone or table doesn't exist

        # part_of removed from the relation taxonomy (2026-07-23, see
        # private/docs/iterations/20260723_part_of_audit_and_brain_alignment_review.md).
        # Existing DBs may still carry part_of edges written before the
        # removal; VALID_RELATIONS no longer accepts the value, so drop
        # them rather than leave orphaned rows no code path can ever read.
        conn.execute("DELETE FROM schema_relations WHERE relation = 'part_of'")

        conn.commit()
