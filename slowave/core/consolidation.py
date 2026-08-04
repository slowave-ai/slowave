"""Replay-time consolidation into first-class symbolic schemas."""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field

import numpy as np

from slowave.latent.semantic_store import SemanticStore
from slowave.storage.sqlite_db import SQLiteDB
from slowave.symbolic.encoder import TextEncoder
from slowave.symbolic.episode_text import EpisodeTextStore
from slowave.symbolic.schema_store import Schema, SchemaStore
from slowave.utils.vec import dumps_json

log = logging.getLogger(__name__)


def _classify_consolidated_schema(text: str, source_kind: str | None) -> str | None:
    """Classify consolidated schema by provenance and structure.

    Args:
        text: Schema content text
        source_kind: Source kind from facets (e.g., "explicit_remember", "consolidation")

    Returns:
        "episodic_summary" if consolidated multi-sentence summary, None for explicit or short,
        "fact" for other consolidated schemas.
    """
    # Explicit memories are never reclassified
    if source_kind == "explicit_remember":
        return None

    # Count sentences: more heuristic to catch multi-claim summaries
    sentence_count = len(re.findall(r"[.!?]", text))
    text_length = len(text)

    # Multi-sentence summaries (>= 3 sentences OR > 300 chars) are tagged as episodic_summary
    if sentence_count >= 3 or text_length > 300:
        return "episodic_summary"

    # Short consolidated schemas are tagged as fact
    return "fact"


@dataclass(frozen=True)
class ConsolidationStats:
    prototypes_processed: int
    schemas_created: int
    schemas_reinforced: int
    schemas_contradicted: int
    schemas_skipped: int
    # Diagnostics. verdict_counts keys:
    # "no_candidate" (nothing to compare against), "missing_embedding"
    # (related schema found but its stored embedding is unreadable), and the
    # GeometricContradictionJudge verdicts ("unrelated", "relates_to" --
    # "relates_to" counts every judge call that cleared the same-topic floor).
    verdict_counts: dict[str, int] = field(default_factory=dict)
    # Prototypes absorbed by the >=0.92 near-duplicate guard.
    near_dup_intercepts: int = 0
    # Of judge verdicts "relates_to", how many triggered metadata-gated
    # supersession vs stayed as plain reinforcement.
    gate_downgrades: dict[str, int] = field(default_factory=dict)
    # Per-prototype LatentSchemaBuilder confidence values, for calibration checks
    # against variance_floor (Q5).
    confidence_histogram: list[float] = field(default_factory=list)


class Consolidator:
    """Lift replayed latent prototypes into symbolic semantic memory."""

    def __init__(
        self,
        *,
        db: SQLiteDB,
        semantic: SemanticStore,
        episode_text: EpisodeTextStore,
        schemas: SchemaStore,
        encoder: TextEncoder | None,
        max_episodes_per_prototype: int = 8,
        # Brain-only path. Schemas are built from prototype geometry and
        # lexical signatures. Zero LLM calls.
        latent_builder=None,
        geometric_judge=None,
        episodic_store=None,
        logic_version: str = "0",
    ):
        self.db = db
        self.semantic = semantic
        self.episode_text = episode_text
        self.schemas = schemas
        self.encoder = encoder
        self.max_episodes_per_prototype = max_episodes_per_prototype
        self.latent_builder = latent_builder
        self.geometric_judge = geometric_judge
        self._episodic_store_ref = episodic_store
        self._latent_mode = latent_builder is not None
        if self._latent_mode and geometric_judge is None:
            raise ValueError("Consolidator: latent_builder given but no geometric_judge")
        # Stamped onto every schema this consolidator creates (see schema.sql
        # comment on schemas.logic_version and
        # private/docs/iterations/20260716_event-store-replay.md point 2).
        self.logic_version = logic_version

    def consolidate(self, *, prototype_ids: list[int]) -> ConsolidationStats:
        """Consolidate prototypes into latent schemas. Zero LLM calls."""
        return self._consolidate_latent(prototype_ids=prototype_ids)

    def consolidate_all(self) -> ConsolidationStats:
        """Consolidate every prototype currently in the store, in ID order.

        Companion to ReplayEngine.replay_all(): a regular consolidate() call
        only processes the prototypes a replay pass just touched, so after a
        full deterministic replay this instead walks the complete prototype
        set — same deterministic ID ordering consolidate() already relies on
        for _episodes_for_prototype().
        """
        conn = self.db.connect()
        rows = conn.execute("SELECT id FROM semantic_prototypes ORDER BY id ASC").fetchall()
        prototype_ids = [int(r["id"]) for r in rows]
        return self.consolidate(prototype_ids=prototype_ids)

    # ------------------------------------------------------------------
    # Stage 6: brain-only consolidation path
    # ------------------------------------------------------------------

    def _consolidate_latent(self, *, prototype_ids: list[int]) -> ConsolidationStats:
        """Latent-schema consolidation. Zero LLM calls.

        Changing this method's schema-writing output for existing data?
        Bump ``SlowaveConfig.current_logic_version`` (slowave/core/config.py)
        so customer DBs auto-rebuild via RebuildService.
        """
        created = 0
        reinforced = 0
        contradicted = 0
        skipped = 0
        diag: dict = {
            "verdict_counts": {
                "no_candidate": 0,
                "missing_embedding": 0,
                "unrelated": 0,
                "relates_to": 0,
            },
            "near_dup_intercepts": 0,
            "gate_downgrades": {"support_gate": 0, "recency_gate": 0},
            "confidence_histogram": [],
        }

        # Build global background corpus for contrastive TF-IDF.
        # Using all existing schema content texts as the background
        # ensures IDF reflects global rarity, not intra-cluster
        # commonality. Terms appearing across many schemas get low
        # IDF (generic/stopword-like); terms unique to this cluster
        # get high IDF (truly distinctive).
        _global_corpus: list[str] = []
        try:
            _global_corpus = [
                s.content_text for s in self.schemas.list(limit=500) if s.content_text
            ]
        except Exception:
            pass

        for pid in prototype_ids:
            ep_ids = self._episodes_for_prototype(pid)
            if not ep_ids:
                skipped += 1
                continue
            sample_ids = ep_ids[: self.max_episodes_per_prototype]
            ep_texts = self.episode_text.get_many(sample_ids)
            if not ep_texts:
                skipped += 1
                continue

            # Episodes backed exclusively by explicit remember events never
            # re-consolidate into schemas: remember() already created the
            # first-class schema synchronously, so lifting these episodes only
            # produces composite near-duplicates (adjacent remembers merged
            # into one macro-episode concatenate two unrelated claims, which
            # defeats both text dedupe and the geometric near-dup guard).
            # Still link related schemas via the prototype centroid so the
            # schema graph is populated even in pure-remember sessions.
            if self._episodes_all_explicit_remember(sample_ids):
                try:
                    proto = self.semantic.get(pid)
                    self._link_schemas_via_prototype_centroid(pid, proto.centroid)
                except Exception:
                    pass
                skipped += 1
                continue

            # Fetch the prototype's centroid (already maintained by replay).
            try:
                proto = self.semantic.get(pid)
            except KeyError:
                skipped += 1
                continue
            centroid = proto.centroid

            # Fetch member-episode embeddings + timestamps.
            ep_records = []
            store = getattr(self, "_episodic_store_ref", None)
            if store is not None:
                ep_records = store.get_many([e.episode_id for e in ep_texts])
            by_eid = {r.id: r for r in ep_records}
            embs, ts_list, kept_eps, kept_ids = [], [], [], []
            for ep in ep_texts:
                rec = by_eid.get(ep.episode_id)
                if rec is None or rec.embedding is None:
                    continue
                embs.append(rec.embedding)
                ts_list.append(int(rec.ts))
                kept_eps.append(ep)
                kept_ids.append(int(ep.episode_id))
            if not embs:
                skipped += 1
                continue

            member_embeddings = np.asarray(embs, dtype=np.float32)
            schema = self.latent_builder.build(
                centroid=centroid,
                member_embeddings=member_embeddings,
                member_episodes=kept_eps,
                member_episode_ids=kept_ids,
                member_timestamps=ts_list,
                background_corpus_texts=_global_corpus,
            )
            if schema is None:
                skipped += 1
                continue
            diag["confidence_histogram"].append(float(schema.confidence))

            outcome, new_id = self._write_latent_schema(
                prototype_id=pid,
                schema=schema,
                diag=diag,
            )
            if outcome == "created":
                created += 1
            elif outcome == "reinforced":
                reinforced += 1
            elif outcome == "contradicted":
                contradicted += 1
            else:
                skipped += 1

            self._record_debug(
                prototype_id=pid,
                episode_ids=sample_ids,
                created_schema_ids=[new_id] if new_id is not None else [],
            )

        return ConsolidationStats(
            prototypes_processed=len(prototype_ids),
            schemas_created=created,
            schemas_reinforced=reinforced,
            schemas_contradicted=contradicted,
            schemas_skipped=skipped,
            verdict_counts=diag["verdict_counts"],
            near_dup_intercepts=diag["near_dup_intercepts"],
            gate_downgrades=diag["gate_downgrades"],
            confidence_histogram=diag["confidence_histogram"],
        )

    def _write_latent_schema(
        self,
        *,
        prototype_id: int,
        schema,
        diag: dict | None = None,
    ) -> tuple[str, int | None]:
        """Persist a LatentSchema. Geometric verdict against the closest
        existing schema. Mirrors `_create_and_relate_schema`."""
        from slowave.latent.schema import LatentSchema as _LS

        claim_embedding = schema.centroid
        claim_text = schema.claim

        evidence_rows: list[tuple[int | None, int | None, str | None, float]] = []
        for eid in schema.member_episode_ids:
            evidence_rows.append((int(eid), None, None, 1.0))

        scope_id = self._scope_for_episodes(schema.member_episode_ids)

        # Near-duplicate guard: if an existing active schema is geometrically
        # almost identical (>= near_dup_guard_cosine, same threshold as
        # working-memory MMR dedup by default), strengthen it instead of
        # creating another copy. Without this, every consolidation pass
        # re-encoded each explicit remember into a duplicate summary schema.
        near_dup_cosine = getattr(self.geometric_judge.cfg, "near_dup_guard_cosine", 0.92)
        if claim_embedding is not None:
            near = self.schemas.search_embedding(claim_embedding, limit=1)
            if near and near[0][1] >= near_dup_cosine:
                try:
                    existing = self.schemas.get(near[0][0])
                except KeyError:
                    existing = None
                if existing is not None and existing.status == "active":
                    self.schemas.reinforce_schema(
                        existing.id,
                        prototype_ids=[prototype_id],
                        supporting_episode_ids=schema.member_episode_ids,
                        evidence=evidence_rows,
                        confidence=schema.confidence,
                    )
                    if scope_id and existing.scope_id and scope_id != existing.scope_id:
                        try:
                            self.schemas.increment_cross_scope_reinforcement(existing.id)
                        except Exception:
                            pass
                    if diag is not None:
                        diag["near_dup_intercepts"] += 1
                    return "reinforced", existing.id

            # Forgotten-schema guard: search_embedding() excludes non-active/
            # needs_review rows by default, so a forgotten schema never
            # surfaces in `near` above -- without this second check, every
            # consolidation pass would silently recreate it as a fresh
            # duplicate. Re-searching with include_inactive=True and checking
            # specifically for a forgotten match lets us skip instead, without
            # either reinforcing it back to active (would silently undo the
            # user's forget) or creating a duplicate (would defeat it).
            near_incl = self.schemas.search_embedding(
                claim_embedding, limit=1, include_inactive=True
            )
            if near_incl and near_incl[0][1] >= near_dup_cosine:
                try:
                    forgotten_existing = self.schemas.get(near_incl[0][0])
                except KeyError:
                    forgotten_existing = None
                if forgotten_existing is not None and forgotten_existing.status == "forgotten":
                    if diag is not None:
                        diag["near_dup_intercepts"] += 1
                    return "skipped", forgotten_existing.id

        related = self._best_related_schema(
            claim=claim_text,
            embedding=claim_embedding,
        )

        # Classify consolidated schema by provenance and structure;
        # tag schema_class so the eligibility gate can filter broad summaries.
        facets = dict(schema.facets) if schema.facets else {}
        source_kind = facets.get("source_kind") or "consolidation"
        schema_class = _classify_consolidated_schema(claim_text, source_kind)
        if schema_class is not None:
            facets["schema_class"] = schema_class

        new_schema_id = self.schemas.create(
            prototype_ids=[prototype_id],
            content_text=claim_text,
            facets=facets,
            tags=schema.tags,
            confidence=schema.confidence,
            salience=0.5 + schema.confidence,
            embedding=claim_embedding,
            scope_id=scope_id,
            supporting_episode_ids=schema.member_episode_ids,
            evidence=evidence_rows,
            facet_axes=schema.facet_axes,
            facet_strengths=schema.facet_strengths,
            logic_version=self.logic_version,
        )

        dedup_existing_id = self.schemas.last_create_reinforced_existing_id
        if dedup_existing_id is not None:
            return "reinforced", dedup_existing_id

        if related is None:
            if diag is not None:
                diag["verdict_counts"]["no_candidate"] += 1
            return "created", new_schema_id

        # Re-fetch the related schema's stored embedding from the DB. We
        # need it as a numpy array to compare centroids with the geometric
        # judge. A missing embedding is not evidence for contradiction or
        # supersession — fall back to creation without a geometric verdict.
        related_emb = self._fetch_schema_embedding(related.id)
        if related_emb is None:
            log.warning(
                "consolidation: schema %d has no stored embedding; "
                "skipping geometric verdict, creating new schema %d",
                related.id,
                new_schema_id,
            )
            if diag is not None:
                diag["verdict_counts"]["missing_embedding"] += 1
            return "created", new_schema_id

        # Real facet axes/strengths for the old schema, unpacked by
        # SchemaStore._row_to_schema from the persisted blobs (see
        # schema.sql). Falls back to an empty placeholder only when the
        # related schema genuinely has none (legacy row created before
        # this persistence was added, or it had too few members) — this
        # used to be an unconditional placeholder regardless of the old
        # schema's real facet data, which made the facet-distance-driven
        # verdict (then named "contradicts", since renamed -- see
        # GeometricContradictionJudge.judge()) provably unreachable
        # (fixed 2026-07-09, see PROGRESS.md and
        # tests/unit/test_contradicts_verdict_unreachable.py).
        related_facet_axes = related.facet_axes
        if not isinstance(related_facet_axes, np.ndarray) or related_facet_axes.ndim != 2:
            related_facet_axes = np.zeros((0, claim_embedding.shape[0]), dtype=np.float32)
        related_facet_strengths = related.facet_strengths
        if not isinstance(related_facet_strengths, np.ndarray) or related_facet_strengths.ndim != 1:
            related_facet_strengths = np.zeros((0,), dtype=np.float32)

        old_view = _LS(
            centroid=np.asarray(related_emb, dtype=np.float32),
            facet_axes=related_facet_axes,
            facet_strengths=related_facet_strengths,
            member_episode_ids=[],
            central_episode_id=0,
            central_episode_text=related.content_text or "",
            mean_ts=(
                int(related.facets.get("mean_ts", 0)) if isinstance(related.facets, dict) else 0
            ),
            ts_span_s=(
                int(related.facets.get("ts_span_s", 0)) if isinstance(related.facets, dict) else 0
            ),
            confidence=float(related.confidence),
            support_count=1,
        )
        verdict = self.geometric_judge.judge(old=old_view, new=schema)

        if verdict.verdict == "relates_to":
            # Cleared the same-topic floor. Cosine alone cannot distinguish
            # value-substitution from elaboration from restatement (per
            # 2026-07-22 semantic-relations findings), so the honest
            # verdict is the same for all same-topic pairs.
            if diag is not None:
                diag.setdefault("verdict_counts", {}).setdefault("relates_to", 0)
                diag["verdict_counts"]["relates_to"] += 1
            self.schemas.add_relation(
                src_schema_id=new_schema_id,
                dst_schema_id=related.id,
                relation="relates_to",
                confidence=schema.confidence,
                reason=f"geometric judge: cos={verdict.similarity:.3f} facet_dist={verdict.facet_distance:.3f}",
            )
            # Cross-scope offline reinforcement: when consolidation finds that a
            # newly-formed schema relates to an existing schema from a different
            # scope with high cosine similarity, record it as a distinct signal
            # (not as fabricated recall). This breaks the bootstrap deadlock
            # where stage-0 schemas can never accumulate cross-scope evidence
            # without already being promoted.
            if (
                verdict.similarity >= 0.90
                and scope_id
                and related.scope_id
                and scope_id != related.scope_id
            ):
                try:
                    self.schemas.increment_cross_scope_reinforcement(related.id)
                except Exception:
                    pass
            # Status transition: when cosine is very high and metadata
            # confirms this is a genuine update (enough support, meaningful
            # time gap), mark the older schema superseded/contradicted.
            # Previously this was gated on a `supersedes` verdict from
            # direction_score; now it's gated on the same non-geometric
            # signals (support + recency) that the old code already checked
            # AFTER the verdict, just triggered from relates_to instead.
            if verdict.similarity >= 0.90:
                min_support = getattr(self.geometric_judge.cfg, "min_support_to_supersede", 2)
                min_dt = getattr(self.geometric_judge.cfg, "min_time_delta_to_supersede_s", 3600.0)
                if schema.support_count >= min_support and not (0 < verdict.time_delta_s < min_dt):
                    old_status = "contradicted" if verdict.time_delta_s <= 0 else "superseded"
                    self.schemas.update_status(related.id, status=old_status, salience=0.05)
                    return "contradicted", new_schema_id
            return "reinforced", new_schema_id
        # unrelated
        if diag is not None:
            diag["verdict_counts"]["unrelated"] += 1
        return "created", new_schema_id

    def _record_debug(
        self, *, prototype_id: int, episode_ids: list[int], created_schema_ids: list[int]
    ) -> None:
        conn = self.db.connect()
        conn.execute(
            "INSERT INTO consolidation_debug "
            "(prototype_id, episode_ids, created_schema_ids, ts) "
            "VALUES (?, ?, ?, ?)",
            (
                int(prototype_id),
                dumps_json({"ids": [int(e) for e in episode_ids]}),
                dumps_json({"ids": [int(s) for s in created_schema_ids]}),
                int(time.time()),
            ),
        )
        conn.commit()

    def _embed(self, text: str) -> np.ndarray | None:
        if self.encoder is None:
            return None
        try:
            return self.encoder.encode(text)
        except Exception as e:
            log.warning("schema claim embedding failed: %s", e)
            return None

    def _fetch_schema_embedding(self, schema_id: int) -> np.ndarray | None:
        """Read a schema's stored embedding directly from the DB. Used by
        the latent path's geometric judge."""
        from slowave.utils.vec import unpack_f32

        conn = self.db.connect()
        row = conn.execute(
            "SELECT embedding, dim FROM schemas WHERE id = ?", (int(schema_id),)
        ).fetchone()
        if row is None or row["embedding"] is None:
            return None
        return unpack_f32(row["embedding"], int(row["dim"]))

    def _best_related_schema(self, *, claim: str, embedding: np.ndarray | None) -> Schema | None:
        related_cosine = getattr(self.geometric_judge.cfg, "related_schema_cosine", 0.72)
        if embedding is not None:
            scored = self.schemas.search_embedding(embedding, limit=5, include_inactive=False)
            if scored and scored[0][1] >= related_cosine:
                try:
                    return self.schemas.get(scored[0][0])
                except KeyError:
                    pass
        for sid in self.schemas.search_fts(claim, limit=3):
            try:
                return self.schemas.get(sid)
            except KeyError:
                continue
        return None

    def _schema_to_latent_view(self, schema: Schema):
        """Wrap a persisted Schema row as a LatentSchema so the geometric
        judge (built to compare a fresh replay candidate against an existing
        schema) can also compare two existing schemas against each other.
        Used by reconsolidate_labile_schemas(); mirrors the ad-hoc `old_view`
        construction in _write_latent_schema, generalized to either side."""
        from slowave.latent.schema import LatentSchema as _LS

        dim = schema.embedding.shape[0] if schema.embedding is not None else 0
        facet_axes = schema.facet_axes
        if not isinstance(facet_axes, np.ndarray) or facet_axes.ndim != 2:
            facet_axes = np.zeros((0, dim), dtype=np.float32)
        facet_strengths = schema.facet_strengths
        if not isinstance(facet_strengths, np.ndarray) or facet_strengths.ndim != 1:
            facet_strengths = np.zeros((0,), dtype=np.float32)
        facets = schema.facets if isinstance(schema.facets, dict) else {}
        support = len(schema.supporting_episode_ids or [])
        return _LS(
            centroid=np.asarray(schema.embedding, dtype=np.float32),
            facet_axes=facet_axes,
            facet_strengths=facet_strengths,
            member_episode_ids=list(schema.supporting_episode_ids or []),
            central_episode_id=0,
            central_episode_text=schema.content_text or "",
            mean_ts=int(facets.get("mean_ts", 0)) or int(schema.last_updated_ts),
            ts_span_s=int(facets.get("ts_span_s", 0)),
            confidence=float(schema.confidence),
            support_count=max(1, support),
        )

    def reconsolidate_labile_schemas(self, *, limit: int = 20) -> dict[str, int]:
        """Re-examine labile schemas (is_labile=True) by replaying each
        against its nearest active neighbor, reusing the same geometric
        judge and gates the fresh-schema consolidation path already uses.

        Terminology (see core/08-feedback.md's "Labile State & Reconsolidation"
        section for the full writeup): "labile" is the *state* — a reactivated
        memory trace that is temporarily uncertain and open to revision, the
        standard term in the reconsolidation literature. "Reconsolidation" is
        the *process* a labile trace goes through to resolve — restabilizing
        back to what it was, or being updated/replaced by better evidence.
        This method is this codebase's implementation of that process; the
        boolean flag it operates on marks the state. `decay_unused()`,
        `remember()`'s ambiguous-update case, and the feedback noise-demotion
        rule all flag schemas as labile; this method is what resolves them.

        Brain analogue: hippocampal replay reactivates a labile trace
        alongside related memories; if nothing challenges it, it
        restabilizes, and if it's contradicted or cleanly superseded by
        better-supported/more-recent evidence, it resolves that way instead.

        Only considers schemas that are still status="active" (already
        superseded/contradicted/archived/needs_review-string schemas are
        already resolved). Bounded by `limit` so one consolidation pass
        can't spend unbounded time on a backlog; any remainder is picked up
        on a later pass since the flag persists until resolved.

        Direction note: GeometricContradictionJudge.judge()'s `old`/`new`
        argument slots are NOT symmetric — which schema goes in which slot
        must be decided by actual chronology (last_updated_ts) before calling
        the judge. Always passing the labile schema as `new` would silently
        bias supersession in its favor purely because of which slot it
        occupies, regardless of which side the evidence actually favors.
        With the 2026-07-22 taxonomy collapse, the judge no longer returns
        "supersedes" — supersession is metadata-gated at the relates_to
        block below. The slot assignment still matters: it determines which
        schema's support/time_delta are checked against the gate thresholds.
        """
        related_cosine = getattr(self.geometric_judge.cfg, "related_schema_cosine", 0.72)
        min_support = getattr(self.geometric_judge.cfg, "min_support_to_supersede", 2)
        min_dt = getattr(self.geometric_judge.cfg, "min_time_delta_to_supersede_s", 3600.0)

        labile = self.schemas.list(is_labile=True, status="active", limit=limit)
        stats = {
            "examined": 0,
            "restabilized": 0,
            "superseded": 0,
            "contradicted": 0,
            "inconclusive": 0,
        }

        for schema in labile:
            stats["examined"] += 1
            if schema.embedding is None:
                stats["inconclusive"] += 1
                continue

            neighbors = self.schemas.search_embedding(
                schema.embedding, limit=5, include_inactive=False
            )
            neighbor = None
            for sid, cos in neighbors:
                if sid == schema.id:
                    continue
                if cos < related_cosine:
                    break
                try:
                    neighbor = self.schemas.get(sid)
                except KeyError:
                    continue
                break

            if neighbor is None:
                # No sufficiently related neighbor to replay against — the
                # brain analogue is passive extinction (a trace nothing ever
                # challenges or corroborates just fades), not resolution.
                # Left labile; decay_unused continues lowering its salience
                # each pass, and it will be picked up again if a related
                # memory ever does show up.
                stats["inconclusive"] += 1
                continue

            if schema.last_updated_ts >= neighbor.last_updated_ts:
                new_side, old_side = schema, neighbor
            else:
                new_side, old_side = neighbor, schema
            new_view = self._schema_to_latent_view(new_side)
            old_view = self._schema_to_latent_view(old_side)
            verdict = self.geometric_judge.judge(old=old_view, new=new_view)

            if verdict.verdict == "unrelated":
                # The nearest neighbor doesn't clear the same-topic floor --
                # no genuine conflict to resolve. Restabilize (the labile
                # flag was a false alarm).
                self.schemas.adjust_feedback_state(schema.id, is_labile=False)
                stats["restabilized"] += 1
                continue

            # Non-unrelated: the pair is about the same topic. Check if
            # this should trigger a status transition (supersession).
            # Previously gated on a `supersedes` verdict from direction_score;
            # now gated on cosine similarity + metadata (support + recency),
            # the same non-geometric signals the old code already checked
            # AFTER the verdict.
            if verdict.similarity >= 0.90:
                new_support = len(new_side.supporting_episode_ids or [])
                if new_support >= min_support and not (0 < verdict.time_delta_s < min_dt):
                    is_tie = verdict.time_delta_s <= 0
                    loser_status = "contradicted" if is_tie else "superseded"
                    self.schemas.update_status(old_side.id, status=loser_status, salience=0.05)
                    try:
                        self.schemas.add_relation(
                            src_schema_id=new_side.id,
                            dst_schema_id=old_side.id,
                            relation="relates_to",
                            confidence=max(new_side.confidence, 0.5),
                            reason=(
                                f"reconsolidation: cos={verdict.similarity:.3f} "
                                f"facet_dist={verdict.facet_distance:.3f} dt={verdict.time_delta_s}s"
                            ),
                        )
                    except ValueError as e:
                        log.warning("reconsolidation: refusing inconsistent relates_to edge: %s", e)
                    self.schemas.adjust_feedback_state(schema.id, is_labile=False)
                    if is_tie:
                        stats["contradicted"] += 1
                    else:
                        stats["superseded"] += 1
                    continue

            # Same topic but below the supersession cosine bar,
            # or gates didn't clear -- no status change needed.
            self.schemas.adjust_feedback_state(schema.id, is_labile=False)
            stats["restabilized"] += 1

        return stats

    def _looks_like_update(self, claim: str) -> bool:
        text = claim.lower()
        markers = (
            "now",
            "updated",
            "changed",
            "instead",
            "actually",
            "no longer",
            "rather than",
            "switched",
            "moved",
            "current",
            "latest",
        )
        return any(m in text for m in markers)

    def _episodes_for_prototype(self, prototype_id: int) -> list[int]:
        conn = self.db.connect()
        rows = conn.execute(
            "SELECT episode_id FROM episode_prototype_map WHERE prototype_id = ? ORDER BY episode_id DESC",
            (int(prototype_id),),
        ).fetchall()
        return [int(r["episode_id"]) for r in rows]

    def _episodes_all_explicit_remember(self, episode_ids: list[int]) -> bool:
        """True when every raw event behind these episodes is a remember event."""
        event_ids: list[int] = []
        for ep in self.episode_text.get_many(episode_ids):
            event_ids.extend(int(e) for e in (ep.event_ids or []))
        if not event_ids:
            return False
        ph = ",".join(["?"] * len(event_ids))
        conn = self.db.connect()
        row = conn.execute(
            f"SELECT COUNT(*) AS n FROM raw_events WHERE id IN ({ph}) "
            "AND type NOT LIKE 'remember:%'",
            tuple(event_ids),
        ).fetchone()
        return int(row["n"] or 0) == 0

    def _link_schemas_via_prototype_centroid(self, prototype_id: int, centroid: np.ndarray) -> None:
        """Link the two schemas closest to this prototype's centroid, using
        the real geometric judge to decide what relation (if any) actually
        holds between THEM -- not just each one's similarity to the shared
        centroid.

        Called when all episodes are explicit-remember so no new schema should
        be created, but the prototype cluster still signals that the two nearest
        schemas are worth relating. Populates schema_relations without
        touching schema counts.
        """
        top = self.schemas.search_embedding(centroid, limit=3)
        if len(top) < 2:
            return
        s1_id, s1_cos = top[0]
        s2_id, s2_cos = top[1]
        if s1_cos < 0.65 or s2_cos < 0.60:
            return
        try:
            s1 = self.schemas.get(s1_id)
            s2 = self.schemas.get(s2_id)
        except KeyError:
            return
        if s1.status not in ("active", "needs_review"):
            return
        if s2.status not in ("active", "needs_review"):
            return
        # Ranking by per-call cosine similarity to the prototype centroid is
        # unstable — the same pair can rank in either order across repeated
        # calls on the same prototype (centroid drifts as replay adds
        # members) or across different prototypes near the same two schemas.
        # Canonicalize on schema id before comparing so `old`/`new` (and thus
        # any directional verdict) are stable across repeat calls, and
        # symmetric relations collapse into one row via add_relation's
        # ON CONFLICT instead of producing both A->B and B->A.
        old_id, new_id = (s1_id, s2_id) if s1_id <= s2_id else (s2_id, s1_id)
        old_schema, new_schema = (s1, s2) if old_id == s1_id else (s2, s1)
        # "Both near the same reference point" does NOT imply "near each
        # other" in a moderately high-dimensional embedding space -- the
        # original version of this method skipped this comparison entirely
        # and unconditionally wrote an edge, which is exactly how a
        # confirmed production false positive (schema 153 linked to an
        # unrelated schema 154 at confidence 1.00) got created. Comparing
        # the two schemas directly via the same judge _write_latent_schema
        # uses closes that gap.
        old_view = self._schema_to_latent_view(old_schema)
        new_view = self._schema_to_latent_view(new_schema)
        verdict = self.geometric_judge.judge(old=old_view, new=new_view)

        if verdict.verdict == "unrelated":
            return
        # relates_to is symmetric -- canonicalize on schema id so the same
        # pair never produces both A->B and B->A.
        self.schemas.add_relation(
            src_schema_id=old_id,
            dst_schema_id=new_id,
            relation="relates_to",
            confidence=round(s2_cos, 3),
            reason=f"co-clustered via prototype {prototype_id}",
        )

    def _scope_for_episodes(self, episode_ids: list[int]) -> str | None:
        if not episode_ids:
            return None
        ph = ",".join(["?"] * len(episode_ids))
        conn = self.db.connect()
        row = conn.execute(
            "SELECT s.scope_id AS scope_id FROM episode_text et "
            "JOIN sessions s ON s.id = et.session_id "
            f"WHERE et.episode_id IN ({ph}) AND s.scope_id IS NOT NULL "
            "ORDER BY et.episode_id DESC LIMIT 1",
            tuple(int(e) for e in episode_ids),
        ).fetchone()
        return None if row is None else str(row["scope_id"])
