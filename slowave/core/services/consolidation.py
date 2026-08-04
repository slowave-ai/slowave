"""ConsolidationService: replay + latent schema consolidation + decay.

Previously implemented as engine.consolidate_once(). Extracted so it can be
tested and reasoned about independently of the full engine.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import time
from typing import Any

from slowave.core.consolidation import Consolidator
from slowave.core.services.ingest import IngestService
from slowave.latent.replay_engine import ReplayEngine
from slowave.storage.sqlite_db import SQLiteDB
from slowave.symbolic.schema_store import SchemaStore

log = logging.getLogger(__name__)

# WP-6: explicit client-confirmed co-use (both schemas named in the same
# `used_memory_ids` feedback call) is grounded in real behavior, not mere
# co-presentation -- it earns a bigger Hebbian bump than the ordinary
# same-call co-presentation signal (see _write_coactivations).
_EXPLICIT_COUSE_BOOST = 2.0


class ConsolidationService:
    """Runs one replay + latent consolidation + decay pass."""

    def __init__(
        self,
        *,
        db: SQLiteDB,
        replay_engine: ReplayEngine,
        consolidator: Consolidator | None,
        schemas: SchemaStore,
        ingest: IngestService,
        encoder: Any = None,
        full_sweep_interval_s: float = 86400.0,
    ):
        self.db = db
        self.replay_engine = replay_engine
        self.consolidator = consolidator
        self.schemas = schemas
        self._ingest = ingest
        # How often full_generalization_sweep() runs (default: daily).
        # generalization_stage is slow-moving; the safety net doesn't need
        # every-5-minutes freshness, just enough to catch denominator drift
        # eventually. Tests override this (e.g. to 0) to force a sweep.
        self._full_sweep_interval_s = full_sweep_interval_s

    def consolidate_once(
        self, *, triggered_by: str = "worker", decay_idle_days: float = 30.0
    ) -> dict[str, Any]:
        """Run one replay + latent consolidation pass, reconsolidate labile
        schemas, then decay unused schemas.

        Returns a stats dict with keys ``replay``, ``consolidation``,
        ``reconsolidation``, and ``decay``.
        """
        conn = self.db.connect()
        started_ts = int(time.time())
        run_id: int | None = None
        try:
            cur = conn.execute(
                "INSERT INTO worker_runs (started_ts, triggered_by) VALUES (?, ?)",
                (started_ts, triggered_by),
            )
            conn.commit()
            run_id = cur.lastrowid
        except Exception as e:
            log.warning("worker_runs insert failed: %s", e)

        error_text: str | None = None
        result: dict[str, Any] = {}
        try:
            replay_stats = self.replay_engine.replay_once()
            consolidation: dict[str, Any] = {}
            reconsolidation: dict[str, Any] = {}
            if self.consolidator is not None:
                # Consolidate only the prototypes this replay pass actually
                # touched (new/updated episode assignments), not every
                # prototype in the store — reprocessing untouched prototypes
                # every tick re-triggers their near-duplicate "reinforces"
                # verdict against an unchanged schema, inflating salience
                # indefinitely with no new evidence behind it.
                protos = replay_stats.get("touched_prototype_ids", [])
                cs = self.consolidator.consolidate(prototype_ids=protos)
                consolidation = dataclasses.asdict(cs)
                # Reconsolidation (2026-07-10): re-examine labile schemas
                # (needs_review=True) by replaying them against their
                # nearest active neighbor via the same judge, instead of
                # leaving them flagged indefinitely. "Labile" is the state,
                # "reconsolidation" is the process that resolves it — see
                # core/08-feedback.md's "Labile State & Reconsolidation"
                # section and outcomes/08-feedback.md.
                reconsolidation = self.consolidator.reconsolidate_labile_schemas()
            # Backfill facet axes for schemas that have accumulated enough
            # supporting episodes but lack them (engine.remember() creates
            # schemas from single episodes so axes can only be computed
            # retroactively once 3+ supporting IDs exist). part_of (the only
            # consumer that gated on this) was removed 2026-07-23 -- kept for
            # facet_distance (relates_to's reason string) and the dormant VSA
            # hypervector encoding, both of which still read facet_axes. See
            # private/docs/iterations/20260723_part_of_audit_and_brain_alignment_review.md.
            facet_backfill = self.schemas.backfill_facet_axes(limit=200)
            # Phase 2 — co-activation: strengthen associative edges from
            # real recall patterns (Hebbian: schemas recalled together in
            # the same session strengthen their directional edges).
            coactivation = self._write_coactivations(conn, started_ts)
            # Generalization-stage refresh (2026-07-23): recompute stage for
            # schemas with new cross-scope evidence since last run, plus a
            # rare full sweep for staleness the incremental pass can't catch
            # (denominator drift on schemas nothing else touches). Previously
            # this only ran reactively from feedback.py, one schema at a time.
            generalization = self._refresh_generalization(conn, started_ts)
            decay = self.schemas.decay_unused(idle_days=decay_idle_days, dry_run=False)

            result = {
                "replay": replay_stats,
                "consolidation": consolidation,
                "reconsolidation": reconsolidation,
                "facet_backfill": facet_backfill,
                "coactivation": coactivation,
                "generalization": generalization,
                "decay": decay,
                "procedures": {},  # removed Phase 1 P1
            }
        except Exception as e:
            error_text = str(e)
            log.error("consolidate_once failed: %s", e, exc_info=True)
            result = {"error": error_text}
        finally:
            if run_id is not None:
                ended_ts = int(time.time())
                cs = result.get("consolidation", {})
                replay_stats = result.get("replay", {})
                decay = result.get("decay", {})

                try:
                    conn.execute(
                        """
                        UPDATE worker_runs SET
                          ended_ts=?, duration_ms=?, prototypes_processed=?,
                          episodes_processed=?,
                          schemas_created=?, schemas_reinforced=?,
                          schemas_contradicted=?, schemas_skipped=?,

                          schemas_decayed=?, error_text=?
                        WHERE id=?
                        """,
                        (
                            ended_ts,
                            (ended_ts - started_ts) * 1000,
                            cs.get("prototypes_processed", 0),
                            replay_stats.get("replay_sampled", 0),
                            cs.get("schemas_created", 0),
                            cs.get("schemas_reinforced", 0),
                            cs.get("schemas_contradicted", 0),
                            cs.get("schemas_skipped", 0),
                            decay.get("decayed", 0),
                            error_text,
                            run_id,
                        ),
                    )
                    conn.commit()
                except Exception as e2:
                    log.warning("worker_runs update failed: %s", e2)

                # Graph health snapshot — best-effort, non-fatal
                try:
                    from slowave.core.graph_health import snapshot

                    snapshot(conn, run_id)
                except Exception as e3:
                    log.debug("graph_health snapshot skipped: %s", e3)
        return result

    def _write_coactivations(self, conn: Any, now_ts: int) -> dict[str, Any]:
        """WP-6: write Hebbian co-activation edges from two honestly-distinct
        signals, replacing the pre-WP-6 "any schema admitted anywhere in the
        same session" rule the plan's Phase 4 flagged as dishonest:

        1. Same-call, directly-relevant co-presentation -- schemas the client
           literally saw together in one activate()/recall() response, and
           only the ones admitted for actual query relevance (`pathway =
           'direct'`), not a salience-filled exploration slot or a
           graph-propagated association (`pathway` in ('exploration',
           'graph') -- see WorkingMemoryItem.peripheral / expand_via_relations
           and ops._pathway_for). Grouped by `context_id` (one retrieval
           call), not `session_id` (the whole session): the old session-wide
           grouping pairwise-crossed schemas across unrelated activate()
           calls within one long session purely because they shared a
           session_id, and it silently excluded every recall() call, which
           never sets session_id at all. context_id is set on every
           activate() and recall() call, so this also makes recall()
           schemas visible to co-activation for the first time.
        2. Explicit client-confirmed co-use -- when a single
           `retrieval_feedback()` call names 2+ schemas in `used_memory_ids`,
           that is a real behavioral signal grounded in what the client
           actually relied on, not incidental co-presentation. Each such pair
           gets an additional, stronger boost (_EXPLICIT_COUSE_BOOST) on top
           of whatever ordinary co-presentation already wrote for the same
           call.

        See private/experiments/validate_retrieval_quality_plan.py's
        experiment_wp6_coactivation_event_semantics for the deterministic
        evidence, and the WP-6 section of
        private/docs/iterations/20260728_retrieval_quality_execution_progress.md
        for the comparison against the session-wide co-presentation rule this
        replaced. Then applies pure exponential decay (half-life ~7 days) to
        all untouched rows.
        """
        import re

        half_life_s = 604800.0
        # Find the cutoff: process events newer than the last completed
        # worker run (or all events on first run). Use ended_ts, not
        # started_ts -- events created while the previous run was still
        # executing were already processed by it, so anchoring on started_ts
        # would reprocess and double-strengthen that overlap window on every
        # subsequent pass.
        cutoff_row = conn.execute(
            "SELECT ended_ts FROM worker_runs "
            "WHERE ended_ts IS NOT NULL AND started_ts < ? "
            "ORDER BY started_ts DESC LIMIT 1",
            (int(now_ts),),
        ).fetchone()
        cutoff_ts = int(cutoff_row["ended_ts"]) if cutoff_row else 0

        _sch_id_pat = re.compile(r"sch_(\d+)")

        def _ordered_pairs(ids: list[int]) -> list[tuple[int, int]]:
            ordered = list(dict.fromkeys(ids))  # de-dup, preserve first-seen order
            return [
                (ordered[i], ordered[j])
                for i in range(len(ordered))
                for j in range(i + 1, len(ordered))
            ]

        # --- Signal 1: same-call, directly-relevant co-presentation ---
        # admitted=1 only -- context_recall_items also stores rank=-1/
        # admitted=0 rows for candidates the working-memory gate evaluated
        # and REJECTED (e.g. cross-scope graph-expansion candidates correctly
        # filtered out by scope isolation). Without this filter, a rejected
        # candidate reads as "recalled together" with everything actually
        # admitted in the same call, silently punching co-activation edges
        # through the same scope boundary the rest of retrieval enforces.
        # pathway='direct' only -- excludes exploration-slot and
        # graph-propagated rows, which are shown alongside the direct items
        # but were never themselves evidence the query needed them.
        rows = conn.execute(
            "SELECT cri.context_id, cri.memory_id "
            "FROM context_recall_items cri "
            "WHERE cri.memory_type = 'schema' "
            "AND cri.admitted = 1 "
            "AND cri.pathway = 'direct' "
            "AND cri.created_at > ? "
            "ORDER BY cri.context_id, cri.rank ASC",
            (cutoff_ts,),
        ).fetchall()

        calls: dict[str, list[int]] = {}  # context_id -> [schema_id, ...]
        for row in rows:
            cid = str(row["context_id"])
            m = _sch_id_pat.match(str(row["memory_id"]))
            if not m:
                continue
            calls.setdefault(cid, []).append(int(m.group(1)))

        pairs_written = 0
        calls_processed = len(calls)
        for ids in calls.values():
            if len(ids) < 2:
                continue
            for src, dst in _ordered_pairs(ids):
                self.schemas.upsert_coactivation(
                    src,
                    dst,
                    now_ts=now_ts,
                    half_life_s=half_life_s,
                )
                pairs_written += 1

        # --- Signal 2: explicit client-confirmed co-use ---
        feedback_rows = conn.execute(
            "SELECT used_memory_ids_json FROM context_feedback_events " "WHERE created_at > ?",
            (cutoff_ts,),
        ).fetchall()
        explicit_pairs_written = 0
        for row in feedback_rows:
            try:
                used = json.loads(row["used_memory_ids_json"] or "[]")
            except (TypeError, ValueError):
                continue
            ids = []
            for mid in used:
                m = _sch_id_pat.match(str(mid))
                if m:
                    ids.append(int(m.group(1)))
            if len(ids) < 2:
                continue
            for src, dst in _ordered_pairs(ids):
                self.schemas.upsert_coactivation(
                    src,
                    dst,
                    now_ts=now_ts,
                    half_life_s=half_life_s,
                    boost=_EXPLICIT_COUSE_BOOST,
                )
                explicit_pairs_written += 1

        # Pure decay for all rows
        decayed = self.schemas.decay_all_coactivations(
            now_ts=now_ts,
            half_life_s=half_life_s,
        )

        return {
            "calls_processed": calls_processed,
            "pairs_written": pairs_written,
            "explicit_pairs_written": explicit_pairs_written,
            "decayed": decayed,
        }

    def _refresh_generalization(self, conn: Any, now_ts: int) -> dict[str, Any]:
        """Recompute generalization_stage for schemas with new evidence since
        the last run, plus a rare full sweep for staleness that can't catch.

        Two tiers, same reasoning as _write_coactivations' admitted filter and
        cutoff pattern:

        1. Incremental (every cycle, cheap): find schemas with a new admitted
           recall (context_recall_items) or a new cross-scope evidence
           attestation (schema_evidence, joined through whichever source row
           it points at -- explicit remember() sets raw_event_id, the
           consolidation/prototype path sets episode_id instead) since the
           last worker run, and refresh just those. Fixes "stale-low": a
           schema whose evidence now qualifies for promotion but hasn't been
           touched since.

        2. Full sweep (rare, gated by _full_sweep_interval_s): recompute every
           active/needs_review schema regardless of activity. Fixes
           "stale-high": total_active_scopes (the scope_breadth_pct
           denominator) can grow or shrink without ever touching a dormant
           schema directly, so its cached stage silently drifts from what its
           own evidence would justify -- nothing schema-local ever triggers a
           recompute for that case. Not run every cycle: generalization_stage
           is slow-moving, and this is O(active schema count) in
           _update_utility_scores queries.
        """
        cutoff_row = conn.execute(
            "SELECT ended_ts FROM worker_runs "
            "WHERE ended_ts IS NOT NULL AND started_ts < ? "
            "ORDER BY started_ts DESC LIMIT 1",
            (int(now_ts),),
        ).fetchone()
        cutoff_ts = int(cutoff_row["ended_ts"]) if cutoff_row else 0

        import re

        _sch_id_pat = re.compile(r"sch_(\d+)")
        schema_ids: set[int] = set()

        recall_rows = conn.execute(
            "SELECT DISTINCT memory_id FROM context_recall_items "
            "WHERE memory_type = 'schema' AND admitted = 1 AND created_at > ?",
            (cutoff_ts,),
        ).fetchall()
        for r in recall_rows:
            m = _sch_id_pat.match(str(r["memory_id"]))
            if m:
                schema_ids.add(int(m.group(1)))

        evidence_rows = conn.execute(
            """
            SELECT se.schema_id FROM schema_evidence se
            JOIN raw_events re ON re.id = se.raw_event_id
            WHERE re.ts > ?
            UNION
            SELECT se.schema_id FROM schema_evidence se
            JOIN episodic_memories em ON em.id = se.episode_id
            WHERE em.ts > ?
            """,
            (cutoff_ts, cutoff_ts),
        ).fetchall()
        for r in evidence_rows:
            schema_ids.add(int(r["schema_id"]))

        for sid in schema_ids:
            self.schemas.refresh_utility(sid)

        full_swept = 0
        sweep_row = conn.execute(
            "SELECT last_full_sweep_ts FROM generalization_sweep_state WHERE id = 1"
        ).fetchone()
        last_full_sweep_ts = int(sweep_row["last_full_sweep_ts"]) if sweep_row else 0
        if now_ts - last_full_sweep_ts >= self._full_sweep_interval_s:
            full_result = self.schemas.full_generalization_sweep()
            full_swept = full_result.get("swept", 0)
            conn.execute(
                "INSERT INTO generalization_sweep_state (id, last_full_sweep_ts) VALUES (1, ?) "
                "ON CONFLICT(id) DO UPDATE SET last_full_sweep_ts = excluded.last_full_sweep_ts",
                (now_ts,),
            )
            conn.commit()

        return {
            "incremental_refreshed": len(schema_ids),
            "full_swept": full_swept,
        }
