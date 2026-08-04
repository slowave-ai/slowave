"""Schema store: first-class symbolic semantic memories.

A schema is a durable typed claim consolidated from episodic traces. Unlike the
old one-schema-per-prototype model, schemas now have their own identity,
embedding, salience/status, normalized evidence links, prototype associations,
and relations to other schemas.
"""

from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from slowave.storage.sqlite_db import SQLiteDB
from slowave.utils.vec import (
    dumps_json,
    loads_json,
    pack_f32,
    pack_f32_matrix,
    unpack_f32,
    unpack_f32_matrix,
)

VALID_STATUS = (
    "active",
    "needs_review",
    "superseded",
    "contradicted",
    "archived",
    "forgotten",
)
# "forgotten" (2026-07-17) is a distinct status from "archived": archived means
# dedup_exact() folded this row into a canonical duplicate (see dedup_exact
# below); forgotten means a human explicitly asked to suppress this schema via
# CLI/dashboard. Keeping them separate preserves dedup_exact's duplicate-ratio
# stats and lets unforget() put a schema back to its real prior status instead
# of always guessing "active". Deliberately CLI/dashboard-only -- not exposed
# as an MCP tool, since forgetting requires a human looking at a specific
# schema id, not an agent inferring intent from conversational subtext.
# "contradicts" and the old "related_to" were removed (2026-07-14): contradicts
# was unreachable in practice (it required an exact time_delta_s<=0 tie, and
# every call site now writes "supersedes" for that case too -- see
# Consolidator._write_latent_schema / reconsolidate_labile_schemas), and
# related_to was only ever add_relation()'s own fallback for an invalid
# relation string, never triggered by any real caller. Both sat at 0 edges in
# production. The "contradicted" schema *status* (VALID_STATUS above) is
# unrelated and unaffected -- it still distinguishes a same-instant clash from
# an ordinary update, just no longer via a separate relation edge label.
#
# "relates_to" (2026-07-15) is a distinct, deliberate reintroduction, NOT a
# revival of the old related_to fallback -- do not conflate the two spellings.
# It is the taxonomy's honest catch-all: cosine cleared the same-topic floor
# (these two schemas are about the same thing) but neither containment,
# direction_score, nor facet_distance clears the bar for a stronger claim
# (part_of / supersedes / refines / reinforces). Without this bucket, the two
# geometry shortcuts that only ever detect loose similarity -- the centroid-
# proximity linker and GeometricContradictionJudge's old cos-only shortcut --
# had nowhere honest to put their output and mislabeled it "reinforces", a
# term that should mean "this specifically restates and strengthens that exact
# belief". See GeometricContradictionJudge.judge() and
# Consolidator._link_schemas_via_prototype_centroid, both of which now write
# this relation directly.
# (2026-07-22): the 4-relation taxonomy (refines, supersedes, part_of,
# relates_to) has been collapsed to a single content-relation type.
# "refines" and "supersedes" were removed because no geometric signal
# (direction_score, facet_distance) generalizes beyond synthetic seed
# sets, per independent measurement in the semantic-relations sibling
# repo. "part_of" (subspace containment) survived that round, but was
# removed 2026-07-23 after a hand-labeled audit of production edges found
# ~11% precision -- broad/generic schemas' wider facet-axis spread made
# them geometrically absorb containment edges from unrelated facts almost
# regardless of real hierarchy, and it carried no differential weight at
# retrieval time vs relates_to anyway. See
# private/docs/iterations/20260723_part_of_audit_and_brain_alignment_review.md.
# All same-topic pairs (cosine above same_topic_cosine) are now relates_to
# -- the only content-relation type left, and symmetric, so no relation
# in schema_relations carries directional meaning anymore.
VALID_RELATIONS = ("relates_to",)
DEDUP_ACTIVE_STATUSES = ("active", "needs_review")

# Shared salience ceiling for every reinforcement/feedback write path. The
# working-memory gate normalises salience by /20 (core/context.py), so growth
# past this is invisible to ranking and only distorts salience-ordered
# listings. Used by both reinforce() and adjust_feedback_state() so the two
# code paths stay bounded consistently instead of one capping and the other not.
SALIENCE_CEILING = 20.0

# A schema flagged is_labile is temporarily uncertain and open to revision
# (see core/08-feedback.md's "Labile State & Reconsolidation" section) —
# the flag is set by decay, remember()'s ambiguous-collision case, or
# feedback's noise-demotion/direct stale-wrong marks. A labile schema
# restabilizes on its own if it keeps getting genuinely reactivated
# afterward — sustained reactivation is itself evidence the memory is
# still good, mirroring how repeated recall drives real memory
# consolidation. This is a passive counterpart to
# Consolidator.reconsolidate_labile_schemas() (the active, replay-against-
# neighbor form of the same resolution process): _RECONSOLIDATION_RECOVERY_RECURRENCE
# recurrence hits *since being flagged* (tracked via the recurrence_count_at_flag
# facet, captured lazily the first time _update_utility_scores observes the
# flag) clears is_labile.
_RECONSOLIDATION_RECOVERY_RECURRENCE = 3

# ============================================================================
# Cross-scope generalization (Stage 11)
# ============================================================================


class GeneralizationConfig:
    """Thresholds for the 4-stage cross-scope generalization system.

    Stage promotion is driven by two relative signals computed against the
    scope_registry denominator (active scopes in the last ``active_window_days``):

    scope_breadth_pct       = distinct_scopes_recalled / total_active_scopes
    scope_kind_breadth_pct  = distinct_scope_kinds_recalled / total_active_scope_kinds

    Stages:
      0 SCOPED     — only returned within origin scope (default)
      1 PORTABLE   — returned across same scope_kind
      2 CONTEXTUAL — returned across all scopes with a score penalty
      3 GLOBAL     — returned everywhere, no penalty

    Hard floors prevent premature promotion:
      ``min_distinct_scopes``   — scope breadth guard (trivial on tiny systems)
      ``min_distinct_sessions`` — temporal spread guard (brain rationale: the
          hippocampus→neocortex transfer requires reactivation across separate
          waking/sleep cycles, not just repeated recall within one session;
          a schema recalled many times in one session is still episodic, not
          semantic).  Without this, self_supervise() rehearsal within a single
          consolidation pass can artificially inflate recurrence_count and
          salience, causing recently-formed schemas to behave as if they are
          cross-scope stable when they are not.
    """

    # Stage 1: requires BOTH >= 25% of active scopes AND >= 2 distinct scopes.
    # With N active scopes the raw count needed is ceil(0.25 * N), not just 2.
    # Example: 12 active scopes -> needs 3 distinct recalls (3/12 = 0.25), not 2.
    stage1_scope_breadth_pct: float = 0.25
    stage1_min_distinct_scopes: int = 2
    # Must have been recalled across at least this many distinct sessions before
    # promotion. Prevents within-session self_supervise rehearsal from driving
    # promotion of ephemeral/adversarial content.
    stage1_min_distinct_sessions: int = 2

    # Stage 2: >= 55% scope breadth (raised from 50% to compensate for removing
    # the hard scope-kind gate — see compute_stage for the brain-faithful rationale).
    stage2_scope_breadth_pct: float = 0.55
    stage2_scope_kind_breadth_pct: float = 0.40  # kept for the kind_bonus softener only
    stage2_min_distinct_scopes: int = 4
    stage2_min_distinct_sessions: int = 3

    # Stage 3: >= 78% scope breadth (raised from 75% for the same reason).
    stage3_scope_breadth_pct: float = 0.78
    stage3_scope_kind_breadth_pct: float = 0.75  # kept for the kind_bonus softener only
    stage3_min_distinct_scopes: int = 8
    stage3_min_distinct_sessions: int = 5

    # Score multiplier applied to Stage 2 schemas recalled outside their origin scope.
    # Stage 3 schemas receive no penalty (multiplier = 1.0).
    stage2_cross_scope_score_multiplier: float = 0.70

    # Minimum activation required for a cross-scope (Stage 1/2) schema to be
    # admitted when recalled outside its origin scope.  Prevents low-relevance
    # promoted schemas from surfacing on queries that merely share surface words
    # with the memory (the "pytest fixture on a password query" noise problem).
    # Applied in both recall() score ranking and WorkingMemoryGate admission.
    # Stage 3 (global) is exempt — it earned unrestricted retrieval.
    # Raised from 0.30 → 0.40: the extra 10pp tightens semantic relevance bar
    # so that loosely-associated cross-scope schemas (e.g. adversarial content
    # that shares surface tokens with a query) are blocked without affecting
    # genuinely useful cross-scope generalisation.
    cross_scope_min_score: float = 0.40

    # Only scopes active within this window count toward the denominator.
    active_window_days: int = 90

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)

    def compute_stage(
        self,
        distinct_scopes: int,
        distinct_scope_kinds: int,
        scope_breadth_pct: float,
        distinct_sessions: int = 0,
        total_active_scopes: int | None = None,
    ) -> int:
        """Return the promoted stage (0-3) given current breadth metrics.

        Brain-faithful design (hippocampus→neocortex transfer):

        The primary criterion is scope breadth — how many distinct contexts
        found this memory useful, weighted by validated use. Scope breadth is a
        proxy for semantic diversity: a memory useful in 78% of all active scopes
        has been tested across situations that are almost certainly semantically
        far apart, regardless of whether those situations share a categorical
        label (scope kind).

        Scope kind is no longer a hard gate. The original kind gate created an
        unreachable condition: stage-1 memories are only admitted in same-kind
        scopes, so cross-kind recall evidence can never accumulate, and the
        promotion ladder deadlocked at stage 1 for any single-kind store. The
        brain generalises based on semantic diversity of reactivation contexts,
        not their categorical type — "retry backoff with jitter" is a semantic
        memory whether it was recalled across projects, domains, or customers.

        Kind diversity acts instead as a session-floor softener: if the memory
        has been recalled (or evidence-linked via cross-scope remember) across
        >= 2 scope kinds, the temporal spread requirement relaxes by one session.
        This mirrors the hippocampus being more willing to transfer a schema
        whose recall pattern spans qualitatively different cognitive domains.

        distinct_sessions is the temporal spread guard — a memory must have
        survived multiple separate cognitive episodes (distinct waking/sleep
        cycles) before cross-scope visibility. Within-session rehearsal cannot
        drive promotion. Unlike scope breadth, this floor does NOT scale with
        total_active_scopes -- repeated confirmation over time is required
        regardless of how many scopes exist to spread across.

        total_active_scopes (2026-07-23): each stage's min_distinct_scopes was a
        fixed constant (2/4/8), which is unreachable for a user with fewer total
        scopes than that constant -- a fact confirmed in 100% of a 3-scope
        universe could never clear stage 2's floor of 4, permanently capping it
        below what its own evidence justifies. Passing total_active_scopes caps
        each floor at min(total_active_scopes, fixed_floor): for users with at
        least as many scopes as the largest floor (8), this is a no-op and
        behavior is identical to before. Omit (None) to keep the old fixed-floor
        behavior exactly, e.g. for tests exercising the thresholds in isolation.
        See private/docs/iterations/20260723_part_of_audit_and_brain_alignment_review.md
        for the full derivation and worked examples across scope counts.
        """
        # Cross-kind bonus: one fewer required session when the memory spans
        # at least 2 distinct scope kinds, reflecting genuine cross-domain use.
        kind_bonus = 1 if distinct_scope_kinds >= 2 else 0

        def _floor(fixed_floor: int) -> int:
            if total_active_scopes is None:
                return fixed_floor
            return min(total_active_scopes, fixed_floor)

        if (
            distinct_scopes >= _floor(self.stage3_min_distinct_scopes)
            and scope_breadth_pct >= self.stage3_scope_breadth_pct
            and distinct_sessions + kind_bonus >= self.stage3_min_distinct_sessions
        ):
            return 3
        if (
            distinct_scopes >= _floor(self.stage2_min_distinct_scopes)
            and scope_breadth_pct >= self.stage2_scope_breadth_pct
            and distinct_sessions + kind_bonus >= self.stage2_min_distinct_sessions
        ):
            return 2
        if (
            distinct_scopes >= _floor(self.stage1_min_distinct_scopes)
            and scope_breadth_pct >= self.stage1_scope_breadth_pct
            and distinct_sessions >= self.stage1_min_distinct_sessions
        ):
            return 1
        return 0


# Singleton default config — can be overridden via SchemaStore constructor.
_DEFAULT_GEN_CFG = GeneralizationConfig()


class ScopeRegistry:
    """Lightweight catalogue of known scopes.

    Updated on every session_start and activate call so the generalization
    denominator queries are cheap single-count SELECTs instead of full
    table scans over context_recall_events.
    """

    def __init__(self, db: "SQLiteDB"):
        self.db = db

    def record(
        self,
        scope_id: str,
        scope_kind: str | None,
        *,
        is_recall: bool = False,
    ) -> None:
        """Upsert a scope entry and bump the appropriate counter."""
        if not scope_id or not str(scope_id).strip():
            return
        now = int(time.time())
        session_inc = 0 if is_recall else 1
        recall_inc = 1 if is_recall else 0
        conn = self.db.connect()
        conn.execute(
            """
            INSERT INTO scope_registry (scope_id, scope_kind, first_seen_ts, last_active_ts,
                                        session_count, recall_count)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(scope_id) DO UPDATE SET
                last_active_ts = excluded.last_active_ts,
                scope_kind     = COALESCE(excluded.scope_kind, scope_kind),
                session_count  = session_count + ?,
                recall_count   = recall_count  + ?
            """,
            (scope_id, scope_kind, now, now, session_inc, recall_inc, session_inc, recall_inc),
        )
        conn.commit()

    def list_scope_ids(self) -> list[str]:
        """Return every distinct scope_id ever registered.

        Used by activate()'s cold-start path to warn about likely scope-string
        fragmentation (e.g. "project:my-repo" vs "project:my_repo" silently
        creating two isolated memory stores) -- the table is one row per
        scope, so a full scan here is cheap.
        """
        conn = self.db.connect()
        rows = conn.execute("SELECT scope_id FROM scope_registry").fetchall()
        return [str(r["scope_id"]) for r in rows]

    def active_counts(self, window_days: int = 90) -> tuple[int, int]:
        """Return (total_active_scopes, total_active_scope_kinds) within window."""
        cutoff = int(time.time()) - window_days * 86400
        conn = self.db.connect()
        row = conn.execute(
            """
            SELECT COUNT(DISTINCT scope_id)   AS n_scopes,
                   COUNT(DISTINCT scope_kind) AS n_kinds
            FROM scope_registry
            WHERE last_active_ts >= ?
            """,
            (cutoff,),
        ).fetchone()
        if row is None:
            return 0, 0
        return int(row["n_scopes"]), max(1, int(row["n_kinds"] or 1))


def normalize_schema_text(text: str) -> str:
    """Normalize schema claims for deterministic duplicate detection.

    This is deliberately conservative: it collapses whitespace and case, but
    does not strip semantic words or perform project-specific filtering. More
    aggressive semantic merging is handled by embeddings/relations elsewhere;
    this function exists to stop exact duplicate claims from multiplying during
    repeated consolidation passes.
    """
    return re.sub(r"\s+", " ", str(text).strip().lower())


@dataclass(frozen=True)
class Schema:
    id: int
    prototype_id: int | None
    content_text: str
    facets: dict[str, Any]
    tags: list[str]
    scope_id: str | None
    status: str
    confidence: float
    salience: float
    supporting_episode_ids: list[int]
    contradicting_episode_ids: list[int]
    is_labile: bool
    first_formed_ts: int
    last_updated_ts: int
    # Canonical embedding vector unpacked from the DB blob. Optional so
    # schemas without embeddings (old entries, failed encodings) degrade
    # gracefully. Default None keeps all existing construction sites valid.
    embedding: "np.ndarray | None" = None
    # Cross-scope generalization stage (Stage 11).
    # 0=scoped, 1=portable (same scope_kind), 2=contextual (all scopes, penalised),
    # 3=global (all scopes, no penalty). Promoted automatically by recall breadth.
    generalization_stage: int = 0
    # Facet axes / strengths unpacked from the DB blobs (see schema.sql). Shape
    # (0, dim) / (0,) when the schema has no facet data (legacy row, or fewer
    # than min_members_for_facets member episodes at creation time) rather than
    # None, so callers can treat "no facets" uniformly without a None-check.
    facet_axes: "np.ndarray | None" = None
    facet_strengths: "np.ndarray | None" = None
    # Logic version active when this schema was created (see schema.sql
    # comment on schemas.logic_version).
    logic_version: str = "0"


@dataclass(frozen=True)
class SchemaEvidence:
    schema_id: int
    episode_id: int | None
    raw_event_id: int | None
    quote: str | None
    weight: float


def canonical_schema_text(
    *,
    claim: str,
    facets: dict[str, Any] | None = None,
    tags: list[str] | None = None,
) -> str:
    """Return the semantic text representation used for schema similarity.

    The user-visible claim remains concise, but recall/relation matching should
    see the full flexible schema: scope, positive/negative affordances, salient
    entities, attributes, and tags. This is benchmark-agnostic: any schema with
    useful facets becomes easier to retrieve by its meaning rather than only by
    the wording of its claim.
    """
    facets = facets or {}
    tags = tags or []
    parts = [f"Claim: {claim.strip()}"]

    def add_value(label: str, value: Any) -> None:
        if value is None or value == "" or value == [] or value == {}:
            return
        if isinstance(value, list):
            text = ", ".join(str(v) for v in value if str(v).strip())
        elif isinstance(value, dict):
            text = dumps_json(value)
        else:
            text = str(value)
        if text.strip():
            parts.append(f"{label}: {text.strip()}")

    add_value("Class", facets.get("schema_class"))
    add_value("Scope", facets.get("scope"))
    add_value("Polarity", facets.get("polarity"))
    add_value("Stability", facets.get("stability"))
    add_value("Positive", facets.get("positive"))
    add_value("Negative", facets.get("negative"))
    add_value("Entities", facets.get("entities"))
    add_value("Attributes", facets.get("attributes"))
    add_value("Tags", tags)
    return "\n".join(parts)


class SchemaStore:
    def __init__(
        self,
        db: SQLiteDB,
        *,
        dim: int,
        gen_cfg: GeneralizationConfig | None = None,
    ):
        self.db = db
        self.dim = int(dim)
        self.last_create_reinforced_existing_id: int | None = None
        self._gen_cfg: GeneralizationConfig = gen_cfg or _DEFAULT_GEN_CFG
        self.scope_registry = ScopeRegistry(db)

    def create(
        self,
        *,
        content_text: str,
        facets: dict[str, Any] | None = None,
        tags: list[str] | None = None,
        embedding: np.ndarray | None,
        prototype_ids: list[int] | None = None,
        scope_id: str | None = None,
        status: str = "active",
        confidence: float = 1.0,
        salience: float = 1.0,
        supporting_episode_ids: list[int] | None = None,
        contradicting_episode_ids: list[int] | None = None,
        is_labile: bool = False,
        evidence: list[tuple[int | None, int | None, str | None, float]] | None = None,
        dedupe: bool = True,
        facet_axes: np.ndarray | None = None,
        facet_strengths: np.ndarray | None = None,
        logic_version: str = "0",
    ) -> int:
        self.last_create_reinforced_existing_id = None
        status = status if status in VALID_STATUS else "active"
        now = int(time.time())
        supporting = [int(x) for x in (supporting_episode_ids or [])]
        contradicting = [int(x) for x in (contradicting_episode_ids or [])]
        proto_ids = list(dict.fromkeys(int(p) for p in (prototype_ids or [])))
        primary_proto = proto_ids[0] if proto_ids else None

        if dedupe and status in DEDUP_ACTIVE_STATUSES:
            existing_id = self.find_duplicate(
                content_text=content_text,
                scope_id=scope_id,
                statuses=DEDUP_ACTIVE_STATUSES,
            )
            if existing_id is not None:
                self.reinforce_schema(
                    existing_id,
                    prototype_ids=proto_ids,
                    supporting_episode_ids=supporting,
                    contradicting_episode_ids=contradicting,
                    evidence=evidence,
                    salience_delta=max(0.05, min(float(salience) * 0.25, 0.5)),
                    confidence=confidence,
                    facets=facets,
                    tags=tags,
                )
                self.last_create_reinforced_existing_id = existing_id
                return existing_id

        emb_blob = None
        emb_dim = None
        if embedding is not None:
            vec = np.asarray(embedding, dtype=np.float32).reshape(-1)
            if vec.size != self.dim:
                raise ValueError(
                    f"schema embedding dim mismatch: expected {self.dim}, got {vec.size}"
                )
            emb_blob = pack_f32(vec)
            emb_dim = self.dim

        # Facet axes/strengths (see schema.sql comment on the `schemas` table):
        # persisted so a later comparison can reconstruct a real "old" view
        # instead of an always-empty placeholder.
        facet_axes_blob = None
        facet_strengths_blob = None
        n_facet_axes = 0
        if facet_axes is not None and facet_axes.size > 0:
            fa = np.asarray(facet_axes, dtype=np.float32)
            if fa.ndim != 2 or fa.shape[1] != self.dim:
                raise ValueError(
                    f"schema facet_axes shape mismatch: expected (k, {self.dim}), got {fa.shape}"
                )
            facet_axes_blob = pack_f32_matrix(fa)
            n_facet_axes = fa.shape[0]
            fs = np.asarray(
                facet_strengths if facet_strengths is not None else np.zeros(n_facet_axes),
                dtype=np.float32,
            ).reshape(-1)
            if fs.size != n_facet_axes:
                raise ValueError(
                    f"schema facet_strengths size mismatch: expected {n_facet_axes}, got {fs.size}"
                )
            facet_strengths_blob = pack_f32(fs)

        conn = self.db.connect()
        cur = conn.execute(
            """
            INSERT INTO schemas (
              prototype_id, content_text, facets_json, tags_json, scope_id, status, confidence,
              salience, embedding, dim, facet_axes, facet_strengths, n_facet_axes,
              supporting_episode_ids,
              contradicting_episode_ids, is_labile, first_formed_ts,
              last_updated_ts, logic_version
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                primary_proto,
                content_text,
                dumps_json(facets or {}),
                dumps_json({"tags": [str(t) for t in (tags or [])]}),
                scope_id,
                status,
                float(confidence),
                float(salience),
                emb_blob,
                emb_dim,
                facet_axes_blob,
                facet_strengths_blob,
                n_facet_axes,
                dumps_json({"ids": supporting}),
                dumps_json({"ids": contradicting}),
                1 if is_labile else 0,
                now,
                now,
                str(logic_version),
            ),
        )
        sid = int(cur.lastrowid or 0)
        conn.execute(
            "INSERT INTO schemas_fts (rowid, content_text) VALUES (?, ?)", (sid, content_text)
        )
        for pid in proto_ids:
            conn.execute(
                "INSERT INTO schema_prototype_map (schema_id, prototype_id, weight) VALUES (?, ?, ?) "
                "ON CONFLICT(schema_id, prototype_id) DO UPDATE SET weight=excluded.weight",
                (sid, pid, 1.0),
            )
        for episode_id, raw_event_id, quote, weight in evidence or []:
            conn.execute(
                "INSERT OR REPLACE INTO schema_evidence "
                "(schema_id, episode_id, raw_event_id, quote, weight) VALUES (?, ?, ?, ?, ?)",
                (sid, episode_id, raw_event_id, quote, float(weight)),
            )
        conn.commit()
        return sid

    def find_duplicate(
        self,
        *,
        content_text: str,
        scope_id: str | None,
        statuses: Iterable[str] = DEDUP_ACTIVE_STATUSES,
        exclude_id: int | None = None,
    ) -> int | None:
        """Return the strongest existing schema with identical normalized text.

        Duplicate identity is scope-aware: schemas are merged within the same
        scope_id (including NULL), but no scope values are special-cased.
        """
        wanted = normalize_schema_text(content_text)
        if not wanted:
            return None
        valid_statuses = [s for s in statuses if s in VALID_STATUS]
        if not valid_statuses:
            return None
        ph = ",".join(["?"] * len(valid_statuses))
        sql = "SELECT id, content_text FROM schemas " f"WHERE status IN ({ph}) "
        args: list[Any] = list(valid_statuses)
        if scope_id is None:
            sql += "AND scope_id IS NULL "
        else:
            sql += "AND scope_id = ? "
            args.append(scope_id)
        if exclude_id is not None:
            sql += "AND id != ? "
            args.append(int(exclude_id))
        sql += "ORDER BY salience DESC, last_updated_ts DESC, id ASC"
        conn = self.db.connect()
        for row in conn.execute(sql, tuple(args)).fetchall():
            if normalize_schema_text(row["content_text"]) == wanted:
                return int(row["id"])
        return None

    def reinforce_schema(
        self,
        schema_id: int,
        *,
        prototype_ids: list[int] | None = None,
        supporting_episode_ids: list[int] | None = None,
        contradicting_episode_ids: list[int] | None = None,
        evidence: list[tuple[int | None, int | None, str | None, float]] | None = None,
        salience_delta: float = 0.2,
        confidence: float | None = None,
        facets: dict[str, Any] | None = None,
        tags: list[str] | None = None,
    ) -> None:
        """Reinforce an existing schema with new provenance.

        This is the consolidation analogue of biological strengthening: repeat
        evidence should increase salience/support on the same semantic memory,
        not create another active copy of the same claim.
        """
        conn = self.db.connect()
        row = conn.execute(
            "SELECT supporting_episode_ids, contradicting_episode_ids, salience, "
            "confidence, facets_json, tags_json FROM schemas WHERE id = ?",
            (int(schema_id),),
        ).fetchone()
        if row is None:
            raise KeyError(f"No schema id={schema_id}")

        def merge_ids(current_json: str, extra: list[int] | None) -> list[int]:
            payload = loads_json(current_json)
            current = payload.get("ids", []) if isinstance(payload, dict) else []
            merged = list(
                dict.fromkeys([int(x) for x in current] + [int(x) for x in (extra or [])])
            )
            return merged

        supporting = merge_ids(row["supporting_episode_ids"], supporting_episode_ids)
        contradicting = merge_ids(row["contradicting_episode_ids"], contradicting_episode_ids)
        merged_confidence = max(float(row["confidence"]), float(confidence or 0.0))

        # Keep existing facets/tags stable, only filling missing keys/tags from
        # the incoming observation. This avoids oscillating canonical memories.
        merged_facets = loads_json(row["facets_json"])
        if isinstance(facets, dict):
            for key, value in facets.items():
                if key not in merged_facets or merged_facets[key] in (None, "", [], {}):
                    merged_facets[key] = value
        existing_tags = [str(t) for t in loads_json(row["tags_json"]).get("tags", [])]
        merged_tags = list(dict.fromkeys(existing_tags + [str(t) for t in (tags or [])]))

        conn.execute(
            """
            UPDATE schemas
            SET salience = min(salience + ?, 20.0),
                confidence = ?,
                supporting_episode_ids = ?,
                contradicting_episode_ids = ?,
                facets_json = ?,
                tags_json = ?,
                last_updated_ts = ?
            WHERE id = ?
            """,
            (
                float(salience_delta),
                merged_confidence,
                dumps_json({"ids": supporting}),
                dumps_json({"ids": contradicting}),
                dumps_json(merged_facets),
                dumps_json({"tags": merged_tags}),
                int(time.time()),
                int(schema_id),
            ),
        )
        for pid in list(dict.fromkeys(int(p) for p in (prototype_ids or []))):
            conn.execute(
                "INSERT INTO schema_prototype_map (schema_id, prototype_id, weight) VALUES (?, ?, ?) "
                "ON CONFLICT(schema_id, prototype_id) DO UPDATE SET weight=max(weight, excluded.weight)",
                (int(schema_id), pid, 1.0),
            )
        for episode_id, raw_event_id, quote, weight in evidence or []:
            conn.execute(
                "INSERT OR REPLACE INTO schema_evidence "
                "(schema_id, episode_id, raw_event_id, quote, weight) VALUES (?, ?, ?, ?, ?)",
                (int(schema_id), episode_id, raw_event_id, quote, float(weight)),
            )
        conn.commit()
        self._update_utility_scores(schema_id, recall_hit=False)

    def update_status(
        self,
        schema_id: int,
        *,
        status: str,
        is_labile: bool | None = None,
        salience: float | None = None,
    ) -> None:
        status = status if status in VALID_STATUS else "active"
        sets = ["status = ?", "last_updated_ts = ?"]
        args: list[Any] = [status, int(time.time())]
        if is_labile is not None:
            sets.append("is_labile = ?")
            args.append(1 if is_labile else 0)
        if salience is not None:
            sets.append("salience = ?")
            args.append(float(salience))
        args.append(int(schema_id))
        conn = self.db.connect()
        conn.execute(f"UPDATE schemas SET {', '.join(sets)} WHERE id = ?", tuple(args))
        conn.commit()

    def forget(self, schema_id: int, *, reason: str | None = None) -> None:
        """Suppress a schema from all retrieval (CLI/dashboard-initiated only).

        Records the schema's current status in schema_forget_log before
        overwriting it, so unforget() can reinstate the real prior status
        (e.g. "superseded") instead of unconditionally reactivating it.
        Idempotent: forgetting an already-forgotten schema is a no-op, so a
        repeated call can never clobber the recorded prior_status with
        "forgotten" itself.
        """
        schema = self.get(schema_id)
        if schema.status == "forgotten":
            return
        conn = self.db.connect()
        now = int(time.time())
        conn.execute(
            "INSERT INTO schema_forget_log (schema_id, action, prior_status, reason, created_ts) "
            "VALUES (?, 'forget', ?, ?, ?)",
            (int(schema_id), schema.status, reason, now),
        )
        conn.commit()
        self.update_status(schema_id, status="forgotten")

    def unforget(self, schema_id: int) -> str:
        """Undo a forget, returning the schema to its status prior to forgetting.

        Named "unforget" rather than "restore" to avoid colliding with the
        unrelated `slowave restore` DB-backup command (slowave/cli/backup.py).

        Returns the status it was restored to. Raises KeyError if the schema
        is not currently forgotten (either never forgotten, or already
        unforgotten) -- this only acts on status == "forgotten" so it can
        never be reapplied on top of an already-unforgotten schema.
        """
        schema = self.get(schema_id)
        if schema.status != "forgotten":
            raise KeyError(f"Schema id={schema_id} is not currently forgotten")
        conn = self.db.connect()
        row = conn.execute(
            "SELECT prior_status FROM schema_forget_log "
            "WHERE schema_id = ? AND action = 'forget' ORDER BY id DESC LIMIT 1",
            (int(schema_id),),
        ).fetchone()
        prior_status = row["prior_status"] if row is not None else "active"
        now = int(time.time())
        conn.execute(
            "INSERT INTO schema_forget_log (schema_id, action, prior_status, reason, created_ts) "
            "VALUES (?, 'unforget', ?, NULL, ?)",
            (int(schema_id), prior_status, now),
        )
        conn.commit()
        self.update_status(schema_id, status=prior_status)
        return prior_status

    def add_relation(
        self,
        *,
        src_schema_id: int,
        dst_schema_id: int,
        relation: str,
        confidence: float = 1.0,
        reason: str | None = None,
    ) -> None:
        """Write a schema_relations edge.

        relates_to is the only relation type and it's symmetric: callers
        MUST canonicalize src/dst as (min(id), max(id)) before calling, since
        this store has no way to recognize that A->B and B->A represent the
        same symmetric fact -- the ON CONFLICT dedup below is keyed on the
        literal (src, dst, relation) tuple. See
        Consolidator._link_schemas_via_prototype_centroid for the reference
        pattern.

        Historical note: part_of (removed 2026-07-23, see
        private/docs/iterations/20260723_part_of_audit_and_brain_alignment_review.md)
        was the last directional relation -- its removal took the
        reverse-edge consistency guard with it, since there is no longer any
        relation for which "A->B" and "B->A" existing simultaneously would be
        a contradiction rather than a duplicate of the same symmetric fact.
        """
        if relation not in VALID_RELATIONS:
            raise ValueError(
                f"add_relation: invalid relation {relation!r}, must be one of {VALID_RELATIONS}"
            )
        conn = self.db.connect()
        conn.execute(
            "INSERT INTO schema_relations "
            "(src_schema_id, dst_schema_id, relation, confidence, reason, created_ts) "
            "VALUES (?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(src_schema_id, dst_schema_id, relation) DO UPDATE SET "
            "confidence=excluded.confidence, reason=excluded.reason, created_ts=excluded.created_ts",
            (
                int(src_schema_id),
                int(dst_schema_id),
                relation,
                float(confidence),
                reason,
                int(time.time()),
            ),
        )
        conn.commit()

    def get_relations(self, schema_id: int) -> list[tuple[int, str, float]]:
        """Return `schema_id`'s schema_relations edges in EITHER direction as
        `(neighbor_id, relation, confidence)` tuples.

        A schema_relations edge is directed (src/dst), but for spreading
        activation (WorkingMemoryGate.expand_via_relations) the direction only
        ever mattered for who was created from whom -- activation should
        still spread along the edge regardless of which side `schema_id` is.
        """
        conn = self.db.connect()
        rows = conn.execute(
            "SELECT dst_schema_id AS neighbor_id, relation, confidence FROM schema_relations "
            "WHERE src_schema_id = ? "
            "UNION ALL "
            "SELECT src_schema_id AS neighbor_id, relation, confidence FROM schema_relations "
            "WHERE dst_schema_id = ?",
            (int(schema_id), int(schema_id)),
        ).fetchall()
        return [(int(r["neighbor_id"]), str(r["relation"]), float(r["confidence"])) for r in rows]

    # -- co-activation (Phase 2) -----------------------------------------------

    def get_coactivations(self, schema_id: int) -> list[tuple[int, str, float]]:
        """Return `schema_id`'s co-activation edges in EITHER direction as
        `(neighbor_id, 'coactivated_with', weight)` tuples.

        Same shape as get_relations() so both can be consumed identically
        by spread_relation_activation() via a composite fetch callback.
        """
        conn = self.db.connect()
        rows = conn.execute(
            "SELECT dst_schema_id AS neighbor_id, weight FROM schema_coactivation "
            "WHERE src_schema_id = ? "
            "UNION ALL "
            "SELECT src_schema_id AS neighbor_id, weight FROM schema_coactivation "
            "WHERE dst_schema_id = ?",
            (int(schema_id), int(schema_id)),
        ).fetchall()
        return [(int(r["neighbor_id"]), "coactivated_with", float(r["weight"])) for r in rows]

    def upsert_coactivation(
        self,
        src_schema_id: int,
        dst_schema_id: int,
        *,
        now_ts: int,
        half_life_s: float = 604800.0,
        boost: float = 1.0,
    ) -> None:
        """Strengthen the directional co-activation edge src -> dst.

        Applies Hebbian update: weight = weight * decay + boost
        where decay = exp(-lambda * dt) and lambda = ln(2) / half_life_s.

        `boost` (WP-6) lets a stronger, explicitly-grounded signal (a client
        reporting both schemas as `used` together) earn a bigger bump than
        ordinary same-call co-presentation, without a separate edge table --
        see ConsolidationService._write_coactivations.

        Self-loops (src == dst) are silently skipped.
        """
        if src_schema_id == dst_schema_id:
            return
        import math

        lam = math.log(2) / half_life_s
        conn = self.db.connect()
        conn.execute(
            "INSERT INTO schema_coactivation (src_schema_id, dst_schema_id, weight, last_touched_ts) "
            "VALUES (?, ?, ?, ?) "
            "ON CONFLICT(src_schema_id, dst_schema_id) DO UPDATE SET "
            "weight = weight * EXP(? * (? - last_touched_ts)) + ?, "
            "last_touched_ts = excluded.last_touched_ts",
            (
                int(src_schema_id),
                int(dst_schema_id),
                float(boost),
                int(now_ts),
                (-lam),
                int(now_ts),
                float(boost),
            ),
        )
        conn.commit()

    def decay_all_coactivations(
        self,
        *,
        now_ts: int,
        half_life_s: float = 604800.0,
    ) -> int:
        """Apply pure exponential decay to every co-activation row.

        Returns the number of rows decayed. Rows whose weight drops to
        or below 1e-6 are deleted (they'll never contribute meaningfully).
        """
        import math

        lam = math.log(2) / half_life_s
        conn = self.db.connect()
        # Decay in-place
        cur = conn.execute(
            "UPDATE schema_coactivation SET "
            "weight = weight * EXP(? * (? - last_touched_ts)), "
            "last_touched_ts = ? "
            "WHERE last_touched_ts < ?",
            ((-lam), int(now_ts), int(now_ts), int(now_ts)),
        )
        # conn.total_changes is cumulative for the connection's whole
        # lifetime, not this statement -- rowcount is the actual count
        # of rows this UPDATE touched.
        decayed = cur.rowcount
        # Prune near-zero edges
        conn.execute("DELETE FROM schema_coactivation WHERE weight <= 1e-6")
        conn.commit()
        return decayed

    def reinforce(
        self,
        schema_id: int,
        *,
        amount: float = 0.2,
        confidence_delta: float = 0.0,
        min_confidence: float = 0.0,
        max_confidence: float = 1.0,
        clear_labile: bool = False,
    ) -> None:
        conn = self.db.connect()
        if confidence_delta or clear_labile:
            row = conn.execute(
                "SELECT confidence FROM schemas WHERE id = ?", (int(schema_id),)
            ).fetchone()
            if row is None:
                raise KeyError(f"No schema id={schema_id}")
            sets = ["salience = min(salience + ?, ?)", "last_updated_ts = ?"]
            args: list[Any] = [float(amount), SALIENCE_CEILING, int(time.time())]
            if confidence_delta:
                new_confidence = max(
                    min_confidence, min(max_confidence, float(row["confidence"]) + confidence_delta)
                )
                sets.append("confidence = ?")
                args.append(new_confidence)
            if clear_labile:
                # An explicit useful/partially_useful mark is direct positive
                # evidence a labile schema is still good — clear it here
                # rather than leaving reconsolidation only to Consolidator.
                # reconsolidate_labile_schemas()'s replay or sustained
                # passive recurrence (core/08-feedback.md's "Labile State &
                # Reconsolidation" section).
                sets.append("is_labile = 0")
            args.append(int(schema_id))
            conn.execute(f"UPDATE schemas SET {', '.join(sets)} WHERE id = ?", tuple(args))
        else:
            conn.execute(
                "UPDATE schemas SET salience = min(salience + ?, ?), last_updated_ts = ? WHERE id = ?",
                (float(amount), SALIENCE_CEILING, int(time.time()), int(schema_id)),
            )
        conn.commit()
        self._update_utility_scores(schema_id, recall_hit=True, force_clear_labile=clear_labile)

    def increment_cross_scope_reinforcement(self, schema_id: int) -> None:
        """Increment cross_scope_reinforcement_count in facets for a schema.

        Called by the consolidation path when a new latent schema from a
        different scope reinforces an existing schema. This is a distinct
        signal from observed recall — offline reinforcement carries lower
        weight in generalization stage promotion.
        """
        conn = self.db.connect()
        row = conn.execute(
            "SELECT facets_json FROM schemas WHERE id = ?", (int(schema_id),)
        ).fetchone()
        if row is None:
            return
        facets = loads_json(row["facets_json"])
        if not isinstance(facets, dict):
            facets = {}
        facets["cross_scope_reinforcement_count"] = (
            int(facets.get("cross_scope_reinforcement_count", 0)) + 1
        )
        conn.execute(
            "UPDATE schemas SET facets_json = ?, last_updated_ts = ? WHERE id = ?",
            (dumps_json(facets), int(time.time()), int(schema_id)),
        )
        conn.commit()
        self._update_utility_scores(schema_id, recall_hit=False)

    def adjust_feedback_state(
        self,
        schema_id: int,
        *,
        salience_delta: float = 0.0,
        confidence_delta: float = 0.0,
        is_labile: bool | None = None,
        min_salience: float = 0.01,
        max_salience: float = SALIENCE_CEILING,
        min_confidence: float = 0.0,
        max_confidence: float = 1.0,
    ) -> None:
        """Apply feedback-driven state adjustments to a schema.

        Used by context_feedback to reinforce/suppress/review memories
        based on user feedback (useful, stale, wrong, irrelevant, etc).

        Args:
            schema_id: schema to adjust
            salience_delta: change in salience (can be positive or negative)
            confidence_delta: change in confidence (usually negative for errors)
            is_labile: set the labile flag (or None to leave unchanged)
            min_salience: floor for salience
            max_salience: ceiling for salience — defaults to SALIENCE_CEILING,
                the same ceiling reinforce() uses, so this path is bounded
                consistently with it.
            min_confidence: floor for confidence
            max_confidence: ceiling for confidence
        """
        conn = self.db.connect()
        row = conn.execute(
            "SELECT salience, confidence FROM schemas WHERE id = ?",
            (int(schema_id),),
        ).fetchone()
        if row is None:
            raise KeyError(f"No schema id={schema_id}")

        new_salience = min(max_salience, max(min_salience, float(row["salience"]) + salience_delta))
        new_confidence = max(
            min_confidence,
            min(
                max_confidence,
                float(row["confidence"]) + confidence_delta,
            ),
        )

        sets = ["salience = ?", "confidence = ?", "last_updated_ts = ?"]
        args: list[Any] = [new_salience, new_confidence, int(time.time())]

        if is_labile is not None:
            sets.append("is_labile = ?")
            args.append(1 if is_labile else 0)

        args.append(int(schema_id))
        conn.execute(f"UPDATE schemas SET {', '.join(sets)} WHERE id = ?", tuple(args))
        conn.commit()

    def refresh_utility(self, schema_id: int) -> None:
        """Recompute utility/generalization/noise facets for a schema.

        Public entry point for the feedback service: called after a feedback
        event is persisted so context_noise_score and the labile-demotion
        rule bite at the next retrieval, not at the next consolidation pass
        that happens to touch this schema.
        """
        self._update_utility_scores(schema_id, recall_hit=False)

    def full_generalization_sweep(self, *, limit: int = 2000) -> dict[str, Any]:
        """Recompute generalization_stage for every active/needs_review schema.

        Safety net for staleness the incremental refresh (schemas touched
        since the last worker run) can't catch: total_active_scopes is the
        denominator in scope_breadth_pct, and it changes as the user's scope
        universe grows or shrinks -- a schema promoted when there were 4 active
        scopes can sit at an inflated stage indefinitely if it's never
        recalled/reinforced again, since nothing schema-local ever triggers a
        recompute for it. This is meant to run rarely (e.g. once a day), not
        every consolidation cycle -- generalization_stage is a slow-moving
        stat, and this does one query per schema (_update_utility_scores joins
        context_recall_items/schema_evidence/scope_registry per call).

        Returns {"swept": <count>}.
        """
        conn = self.db.connect()
        rows = conn.execute(
            "SELECT id FROM schemas WHERE status IN ('active', 'needs_review') "
            "ORDER BY id LIMIT ?",
            (int(limit),),
        ).fetchall()
        for r in rows:
            self.refresh_utility(int(r["id"]))
        return {"swept": len(rows)}

    def _update_utility_scores(
        self, schema_id: int, *, recall_hit: bool = False, force_clear_labile: bool = False
    ) -> None:
        """Recompute and persist stability_score, recurrence_score, schema_utility.

        Called after every reinforce() and reinforce_schema() so the composite
        signal stays fresh without a separate background job.

        stability_score  = sigmoid-like score based on schema age (days since
                           first_formed_ts) and support count. Old, well-supported
                           schemas score near 1.0; brand-new schemas start near 0.

        recurrence_count = cumulative recall/reinforce hits (bumped each call).
        recurrence_score = soft-capped normalisation: count / (count + 5).

        schema_utility   = 0.5 * stability_score + 0.5 * recurrence_score.
        """
        conn = self.db.connect()
        row = conn.execute(
            "SELECT facets_json, first_formed_ts, last_updated_ts, supporting_episode_ids, "
            "scope_id, is_labile FROM schemas WHERE id = ?",
            (int(schema_id),),
        ).fetchone()
        if row is None:
            return

        now = int(time.time())
        facets = loads_json(row["facets_json"])
        if not isinstance(facets, dict):
            facets = {}

        # --- stability_score ---
        age_days = max(0.0, (now - int(row["first_formed_ts"])) / 86400.0)
        supporting = loads_json(row["supporting_episode_ids"])
        support_count = len(supporting.get("ids", [])) if isinstance(supporting, dict) else 0
        # age component: saturates at ~30 days (0→0, 7d→0.5, 30d→0.88)
        age_score = 1.0 - 1.0 / (1.0 + age_days / 7.0)
        # support component: saturates at ~10 episodes
        support_score = support_count / (support_count + 10.0)
        stability_score = round(0.5 * age_score + 0.5 * support_score, 4)

        # --- recurrence_score ---
        recurrence_count = int(facets.get("recurrence_count", 0))
        if recall_hit:
            recurrence_count += 1
        recurrence_score = round(recurrence_count / (recurrence_count + 5.0), 4)

        # --- reconsolidation via sustained recurrence: repeated genuine
        # reactivation restabilizes a labile schema on its own (see
        # _RECONSOLIDATION_RECOVERY_RECURRENCE above) ---
        recover = False
        was_flagged = bool(row["is_labile"])
        if was_flagged:
            baseline = facets.get("recurrence_count_at_flag")
            if baseline is None:
                # First time this function has observed the flag with no
                # recorded baseline — capture one now rather than counting
                # any pre-flagging history toward recovery.
                facets["recurrence_count_at_flag"] = recurrence_count
            elif (
                recall_hit
                and (recurrence_count - int(baseline)) >= _RECONSOLIDATION_RECOVERY_RECURRENCE
            ):
                recover = True
                facets.pop("recurrence_count_at_flag", None)

        # --- schema_utility ---
        schema_utility = round(0.5 * stability_score + 0.5 * recurrence_score, 4)

        facets["stability_score"] = stability_score
        facets["recurrence_count"] = recurrence_count
        facets["recurrence_score"] = recurrence_score
        facets["schema_utility"] = schema_utility

        # --- cross-scope generalization metrics (Stage 11) ---
        # Three sources:
        # 1. context_recall_items with admitted = 1: schemas actually surfaced
        #    into a context brief. The admitted filter is load-bearing — without
        #    it a schema merely evaluated-and-suppressed by the working-memory
        #    gate in a foreign scope counted as cross-scope usage, so gate
        #    rejections drove stage promotion (self-promoting noise loop).
        # 2. context_feedback_events: per-scope validation. A scope where the
        #    agent marked this schema used counts fully; exposure without
        #    feedback counts 0.25; a scope with only negative marks counts 0.
        #    Promotion therefore requires validated usefulness in a foreign
        #    scope, not mere exposure.
        # 3. schema_evidence from cross-scope remember (P4 in engine.py): same
        #    concept remembered from a different scope is strong evidence and
        #    counts fully; it breaks the bootstrap deadlock where stage-0
        #    schemas can't accumulate cross-scope recall events.
        schema_id_key = f"sch_{int(schema_id)}"
        token = f'"{schema_id_key}"'

        # This query does not filter on scope_id — total_negative_marks and
        # total_used_marks below count every matching feedback event
        # regardless of scope. Rows with no scope_id count normally;
        # `fb_scope` below becomes the literal string "None" for them, which
        # never collides with anything in `recall_rows` further down (that
        # query is independently filtered and always has a real scope_id),
        # so this has no effect on the separate cross-scope generalization
        # computation.
        fb_rows = conn.execute(
            """
            SELECT scope_id, used_memory_ids_json, irrelevant_memory_ids_json,
                   stale_memory_ids_json, wrong_memory_ids_json
            FROM context_feedback_events
            WHERE used_memory_ids_json LIKE ? OR irrelevant_memory_ids_json LIKE ?
               OR stale_memory_ids_json LIKE ? OR wrong_memory_ids_json LIKE ?
            """,
            (f"%{token}%",) * 4,
        ).fetchall()
        used_scopes: set[str] = set()
        negative_scopes: set[str] = set()
        total_used_marks = 0
        total_negative_marks = 0
        # Per-scope tallies (WP-7): same shape as the global counters above,
        # bucketed by fb_scope, so a per-scope noise ratio can be derived
        # without a second query. See context_noise_by_scope below.
        used_marks_by_scope: dict[str, int] = {}
        negative_marks_by_scope: dict[str, int] = {}
        for fb in fb_rows:
            fb_scope = str(fb["scope_id"])
            if token in (fb["used_memory_ids_json"] or ""):
                used_scopes.add(fb_scope)
                total_used_marks += 1
                used_marks_by_scope[fb_scope] = used_marks_by_scope.get(fb_scope, 0) + 1
            if any(
                token in (fb[col] or "")
                for col in (
                    "irrelevant_memory_ids_json",
                    "stale_memory_ids_json",
                    "wrong_memory_ids_json",
                )
            ):
                negative_scopes.add(fb_scope)
                total_negative_marks += 1
                negative_marks_by_scope[fb_scope] = negative_marks_by_scope.get(fb_scope, 0) + 1

        recall_rows = conn.execute(
            """
            SELECT cre.scope_id, cre.scope_kind, cre.session_id
            FROM context_recall_items cri
            JOIN context_recall_events cre ON cri.context_id = cre.context_id
            WHERE cri.memory_id = ?
              AND cri.admitted = 1
              AND cre.scope_id IS NOT NULL
            """,
            (schema_id_key,),
        ).fetchall()
        total_cross_recalls = len(recall_rows)

        # Two independent evidence-writers feed schema_evidence, and only one of
        # them ever sets raw_event_id: explicit remember() calls always attach a
        # raw_event_id, but the consolidation/prototype-clustering path writes
        # (episode_id, raw_event_id=None) rows (_write_latent_schema) since a
        # cluster of episodes has no single originating raw event. An INNER
        # JOIN on raw_event_id alone silently drops every consolidation-path
        # row from scope-breadth crediting, which is exactly the bug this
        # UNION closes: the second branch reaches scope via
        # episode_id -> episode_text.session_id -> sessions instead, the same
        # lookup _scope_for_episodes already trusts for scope resolution. The
        # two branches' WHERE clauses are mutually exclusive on raw_event_id's
        # nullity, so a single evidence row only ever matches one of them —
        # but two *separate* rows (one written by each path) can still resolve
        # to the same session if the same underlying activity produced both an
        # explicit remember() and a later consolidation cluster. Plain UNION
        # (not UNION ALL) collapses that case to one scope hit instead of
        # double-counting it in scope_weight below.
        evidence_rows = conn.execute(
            """
            SELECT ses.scope_id, ses.scope_kind, ses.id AS session_id
            FROM schema_evidence se
            JOIN raw_events re ON re.id = se.raw_event_id
            JOIN sessions ses ON ses.id = re.session_id
            WHERE se.schema_id = ?
              AND se.raw_event_id IS NOT NULL
              AND ses.scope_id IS NOT NULL
            UNION
            SELECT ses.scope_id, ses.scope_kind, ses.id AS session_id
            FROM schema_evidence se
            JOIN episode_text et ON et.episode_id = se.episode_id
            JOIN sessions ses ON ses.id = et.session_id
            WHERE se.schema_id = ?
              AND se.raw_event_id IS NULL
              AND se.episode_id IS NOT NULL
              AND ses.scope_id IS NOT NULL
            """,
            (int(schema_id), int(schema_id)),
        ).fetchall()

        # Per-scope weight: validated 1.0, exposure-only 0.25, net-negative 0.
        scope_weight: dict[str, float] = {}
        for r in recall_rows:
            r_scope = str(r["scope_id"])
            if r_scope in used_scopes:
                scope_weight[r_scope] = 1.0
            elif r_scope in negative_scopes:
                scope_weight.setdefault(r_scope, 0.0)
            else:
                scope_weight.setdefault(r_scope, 0.25)
        for r in evidence_rows:
            scope_weight[str(r["scope_id"])] = 1.0

        distinct_scopes = int(sum(scope_weight.values()) + 1e-9)
        counted_scopes = {s for s, w in scope_weight.items() if w > 0.0}
        # scope_kind breadth: count from recall rows AND evidence-path rows so
        # the kind_bonus in compute_stage can fire even when stage-1 gating
        # blocks cross-kind admission via the recall path.
        distinct_scope_kinds = len(
            {
                str(r["scope_kind"])
                for r in recall_rows
                if str(r["scope_id"]) in counted_scopes and r["scope_kind"]
            }
            | {str(r["scope_kind"]) for r in evidence_rows if r["scope_kind"]}
        )
        distinct_sessions = len(
            {str(r["session_id"]) for r in recall_rows if str(r["scope_id"]) in counted_scopes}
            | {str(r["session_id"]) for r in evidence_rows}
        )

        # Offline reinforcement bonus: each cross-scope reinforcement from
        # consolidation counts as 0.5 equivalent observed-recall scope.
        # Observed recall (context_recall_items) is the ground-truth signal;
        # offline reinforcement is weaker evidence and carries half the weight.
        reinforcement_count = int(facets.get("cross_scope_reinforcement_count", 0))
        distinct_scopes += reinforcement_count // 2  # integer, conservative

        total_active_scopes, total_active_scope_kinds = self.scope_registry.active_counts(
            self._gen_cfg.active_window_days
        )

        scope_breadth_pct = (
            round(distinct_scopes / total_active_scopes, 4) if total_active_scopes > 0 else 0.0
        )
        scope_kind_breadth_pct = (
            round(distinct_scope_kinds / total_active_scope_kinds, 4)
            if total_active_scope_kinds > 0
            else 0.0
        )

        new_stage = self._gen_cfg.compute_stage(
            distinct_scopes=distinct_scopes,
            distinct_scope_kinds=distinct_scope_kinds,
            scope_breadth_pct=scope_breadth_pct,
            distinct_sessions=distinct_sessions,
            total_active_scopes=total_active_scopes,
        )

        facets["cross_scope_recall_count"] = total_cross_recalls
        facets["distinct_scope_count"] = distinct_scopes
        facets["distinct_scope_kind_count"] = distinct_scope_kinds
        facets["distinct_session_count"] = distinct_sessions
        facets["scope_breadth_pct"] = scope_breadth_pct
        facets["scope_kind_breadth_pct"] = scope_kind_breadth_pct
        facets["generalization_stage"] = new_stage

        # --- context noise score (feedback-driven self-cleaning) ---
        # Ratio of negative to (weighted) positive feedback marks. Read by
        # WorkingMemoryGate._activation as a ranking penalty. A memory shown
        # repeatedly and marked irrelevant with never a used mark converges to
        # ~1.0; a single used mark outweighs three irrelevant marks.
        facets["context_shown_scoped_count"] = total_cross_recalls
        facets["context_used_count"] = total_used_marks
        facets["context_irrelevant_count"] = total_negative_marks
        facets["context_noise_score"] = round(
            total_negative_marks / (total_negative_marks + 3.0 * total_used_marks + 1.0), 4
        )

        # --- per-scope noise score (WP-7) ---
        # Same ratio as context_noise_score above, but computed independently
        # per scope so WorkingMemoryGate._activation can penalize a memory
        # only in the scope(s) that actually rejected it, instead of the
        # global aggregate leaking that penalty into every other scope's
        # ranking (see 20260728_retrieval_quality_execution_progress.md,
        # WP-7). Deliberately additive: context_noise_score, the is_labile
        # 3-strikes demotion below, and the direct salience_delta mutation in
        # FeedbackService.retrieval_feedback() are all left as schema-global
        # signals — only the ranking-time read in context.py is scoped.
        facets["context_noise_by_scope"] = {
            scope: round(
                negative_marks_by_scope.get(scope, 0)
                / (
                    negative_marks_by_scope.get(scope, 0)
                    + 3.0 * used_marks_by_scope.get(scope, 0)
                    + 1.0
                ),
                4,
            )
            for scope in set(used_marks_by_scope) | set(negative_marks_by_scope)
        }

        # Demotion: flags consistently-negative memories as labile, but this
        # alone does NOT exclude them from default-mode retrieval — only a
        # schema whose `status` column is set away from "active" is hard-
        # excluded there (see SchemaStore.update_status / context.py's
        # _eligible). This flag only feeds context_noise_score's soft ranking
        # penalty; a schema can sit at is_labile=1, status="active"
        # indefinitely, still fully eligible, just down-ranked.
        demote = total_negative_marks >= 3 and total_used_marks == 0

        # force_clear_labile wins unconditionally: it's set only by
        # reinforce() for an explicit "useful" mark that is *itself* in the
        # middle of being applied — the context_feedback_events row for it
        # hasn't been INSERTed yet (that happens later in
        # FeedbackService.retrieval_feedback(), after this call returns), so
        # `demote`'s recount above is blind to the very event that's
        # supposed to be clearing the flag and would otherwise immediately
        # re-set is_labile=1 using stale history.
        # Absent that, demote wins over recover on same-call conflict:
        # explicit accumulated negative feedback is stronger, more
        # deliberate evidence than passive reactivation, so a schema that
        # both qualifies for demotion and has crossed the recovery
        # threshold this call stays flagged.
        if force_clear_labile:
            labile_action: bool | None = False
        elif demote:
            labile_action = True
        elif recover:
            labile_action = False
        else:
            labile_action = None

        sets = "facets_json = ?, generalization_stage = ?"
        args: list[Any] = [dumps_json(facets), new_stage]
        if labile_action is not None:
            sets += ", is_labile = ?"
            args.append(1 if labile_action else 0)
        args.append(int(schema_id))
        conn.execute(f"UPDATE schemas SET {sets} WHERE id = ?", tuple(args))
        conn.commit()

    def get(self, schema_id: int) -> Schema:
        conn = self.db.connect()
        row = conn.execute("SELECT * FROM schemas WHERE id = ?", (int(schema_id),)).fetchone()
        if row is None:
            raise KeyError(f"No schema id={schema_id}")
        return self._row_to_schema(row)

    def get_many(self, schema_ids: Iterable[int]) -> list[Schema]:
        ids = list(dict.fromkeys(int(i) for i in schema_ids))
        if not ids:
            return []
        ph = ",".join(["?"] * len(ids))
        conn = self.db.connect()
        rows = conn.execute(f"SELECT * FROM schemas WHERE id IN ({ph})", tuple(ids)).fetchall()
        by_id = {int(r["id"]): r for r in rows}
        return [self._row_to_schema(by_id[i]) for i in ids if i in by_id]

    def get_by_prototypes(
        self, prototype_ids: Iterable[int], *, include_inactive: bool = False
    ) -> list[Schema]:
        ids = list(dict.fromkeys(int(i) for i in prototype_ids))
        if not ids:
            return []
        ph = ",".join(["?"] * len(ids))
        sql = (
            "SELECT DISTINCT s.* FROM schemas s "
            "JOIN schema_prototype_map m ON m.schema_id = s.id "
            f"WHERE m.prototype_id IN ({ph})"
        )
        args: list[Any] = list(ids)
        if not include_inactive:
            sql += " AND s.status IN ('active', 'needs_review')"
        sql += " ORDER BY s.salience DESC, s.last_updated_ts DESC"
        conn = self.db.connect()
        rows = conn.execute(sql, tuple(args)).fetchall()
        return [self._row_to_schema(r) for r in rows]

    # Backward-compatible method name for call sites; behavior is now many-per-prototype.
    def get_many_by_prototypes(self, prototype_ids: Iterable[int]) -> list[Schema]:
        return self.get_by_prototypes(prototype_ids)

    def list(
        self,
        *,
        is_labile: bool | None = None,
        scope_id: str | None = None,
        status: str | None = None,
        limit: int = 100,
    ) -> list[Schema]:
        conn = self.db.connect()
        sql = "SELECT * FROM schemas WHERE 1=1"
        args: list[Any] = []
        if is_labile is not None:
            sql += " AND is_labile = ?"
            args.append(1 if is_labile else 0)
        if scope_id is not None:
            sql += " AND scope_id = ?"
            args.append(scope_id)
        if status is not None:
            sql += " AND status = ?"
            args.append(status)
        sql += " ORDER BY salience DESC, last_updated_ts DESC LIMIT ?"
        args.append(int(limit))
        rows = conn.execute(sql, tuple(args)).fetchall()
        return [self._row_to_schema(r) for r in rows]

    def search_fts(
        self, query: str, limit: int = 20, *, include_inactive: bool = False
    ) -> list[int]:  # type: ignore[valid-type]
        conn = self.db.connect()
        try:
            if include_inactive:
                rows = conn.execute(
                    "SELECT rowid FROM schemas_fts WHERE schemas_fts MATCH ? ORDER BY rank LIMIT ?",
                    (query, int(limit)),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT rowid FROM schemas_fts "
                    "WHERE schemas_fts MATCH ? "
                    "AND rowid IN (SELECT id FROM schemas WHERE status IN ('active', 'needs_review')) "
                    "ORDER BY rank LIMIT ?",
                    (query, int(limit)),
                ).fetchall()
        except Exception:
            return []
        return [int(r["rowid"]) for r in rows]

    def search_embedding(
        self,
        query: np.ndarray,
        *,
        limit: int = 20,
        scope_id: str | None = None,
        include_inactive: bool = False,
    ) -> list[tuple[int, float]]:  # type: ignore[valid-type]
        q = np.asarray(query, dtype=np.float32).reshape(-1)
        qn = float(np.linalg.norm(q)) + 1e-12
        conn = self.db.connect()
        sql = "SELECT id, embedding, dim FROM schemas WHERE embedding IS NOT NULL"
        args: list[Any] = []
        if scope_id is not None:
            sql += " AND scope_id = ?"
            args.append(scope_id)
        if not include_inactive:
            sql += " AND status IN ('active', 'needs_review')"
        rows = conn.execute(sql, tuple(args)).fetchall()
        scored: list[tuple[int, float]] = []
        for r in rows:
            try:
                v = unpack_f32(r["embedding"], int(r["dim"]))
            except Exception:
                continue
            score = float(q.dot(v) / (qn * (float(np.linalg.norm(v)) + 1e-12)))
            scored.append((int(r["id"]), score))
        scored.sort(key=lambda t: t[1], reverse=True)
        return scored[: int(limit)]

    def schemas_for_episodes(
        self,
        episode_ids: Iterable[int],
        *,
        include_inactive: bool = True,
    ) -> dict[int, list[tuple[int, str, float, int]]]:  # type: ignore[valid-type]
        """Reverse index: episode_id -> list of (schema_id, status, confidence, last_updated_ts).

        Used by the schemas-as-priors retrieval step: a matched-query schema
        biases retrieval *toward* its evidence episodes, and a ``superseded``
        / ``contradicted`` schema silences them. Returning status, confidence
        and recency lets the caller weight the bias and the silence with a
        belief-revision-style freshness factor.

        Episodes are looked up via both ``schema_evidence`` (normalised
        table) and the legacy ``schemas.supporting_episode_ids`` JSON
        column, since older consolidations only populated the JSON column.
        """
        eids = list({int(e) for e in episode_ids})
        if not eids:
            return {}
        ph = ",".join(["?"] * len(eids))
        out: dict[int, list[tuple[int, str, float, int]]] = {e: [] for e in eids}
        conn = self.db.connect()

        # Normalised path: schema_evidence table.
        rows = conn.execute(
            f"""
            SELECT se.episode_id, s.id, s.status, s.confidence, s.last_updated_ts
            FROM schema_evidence se
            JOIN schemas s ON s.id = se.schema_id
            WHERE se.episode_id IN ({ph})
            """,
            tuple(eids),
        ).fetchall()
        for r in rows:
            eid = int(r["episode_id"])
            status = str(r["status"])
            if not include_inactive and status not in ("active", "needs_review"):
                continue
            out.setdefault(eid, []).append(
                (
                    int(r["id"]),
                    status,
                    float(r["confidence"]),
                    int(r["last_updated_ts"]),
                )
            )

        # Legacy JSON path: scan schemas with non-empty supporting_episode_ids.
        # Cheap enough for MVP scale (~thousands of schemas) and avoids a
        # silent recall regression for databases that pre-date schema_evidence.
        legacy_rows = conn.execute(
            "SELECT id, status, confidence, last_updated_ts, supporting_episode_ids "
            "FROM schemas WHERE supporting_episode_ids != '[]'"
        ).fetchall()
        target = set(eids)
        for r in legacy_rows:
            status = str(r["status"])
            if not include_inactive and status not in ("active", "needs_review"):
                continue
            payload = loads_json(r["supporting_episode_ids"])
            supporting = payload.get("ids", []) if isinstance(payload, dict) else []
            sid = int(r["id"])
            for eid in supporting:
                try:
                    eid_i = int(eid)
                except (TypeError, ValueError):
                    continue
                if eid_i in target:
                    entry = (sid, status, float(r["confidence"]), int(r["last_updated_ts"]))
                    # Dedupe with normalised-path entries.
                    if entry not in out.get(eid_i, []):
                        out.setdefault(eid_i, []).append(entry)
        return out

    def evidence_for_schema(self, schema_id: int, *, limit: int = 10) -> list[SchemaEvidence]:  # type: ignore[valid-type]
        conn = self.db.connect()
        rows = conn.execute(
            "SELECT * FROM schema_evidence WHERE schema_id = ? ORDER BY weight DESC LIMIT ?",
            (int(schema_id), int(limit)),
        ).fetchall()
        return [
            SchemaEvidence(
                schema_id=int(r["schema_id"]),
                episode_id=None if r["episode_id"] is None else int(r["episode_id"]),
                raw_event_id=None if r["raw_event_id"] is None else int(r["raw_event_id"]),
                quote=None if r["quote"] is None else str(r["quote"]),
                weight=float(r["weight"]),
            )
            for r in rows
        ]

    def dedup_exact(
        self,
        *,
        status: str = "active",
        dry_run: bool = True,
    ) -> dict[str, Any]:
        """Merge exact normalized duplicate schemas within each scope.

        Generic cleanup: groups by ``(scope_id, normalize_schema_text(content))``.
        The canonical row is the highest-salience, then oldest schema. Duplicate
        rows are marked ``archived`` and related to the canonical schema; their
        evidence/prototype links are moved onto the canonical row.
        """
        conn = self.db.connect()
        rows = conn.execute(
            "SELECT * FROM schemas WHERE status = ? ORDER BY scope_id, content_text, salience DESC, id ASC",
            (status,),
        ).fetchall()
        groups: dict[tuple[str | None, str], list[Any]] = {}
        for row in rows:
            norm = normalize_schema_text(row["content_text"])
            if not norm:
                continue
            sid = None if row["scope_id"] is None else str(row["scope_id"])
            groups.setdefault((sid, norm), []).append(row)

        duplicate_groups = [items for items in groups.values() if len(items) > 1]
        duplicate_rows = sum(len(items) - 1 for items in duplicate_groups)
        result: dict[str, Any] = {
            "status": status,
            "groups": len(duplicate_groups),
            "duplicate_rows": duplicate_rows,
            "canonical_rows": len(duplicate_groups),
            "dry_run": dry_run,
            "merged_rows": 0,
        }
        if dry_run or not duplicate_groups:
            return result

        now = int(time.time())
        for items in duplicate_groups:
            canonical = sorted(
                items,
                key=lambda r: (-float(r["salience"]), int(r["first_formed_ts"]), int(r["id"])),
            )[0]
            canonical_id = int(canonical["id"])
            dupes = [r for r in items if int(r["id"]) != canonical_id]
            for dupe in dupes:
                dupe_id = int(dupe["id"])
                supporting = loads_json(dupe["supporting_episode_ids"]).get("ids", [])
                contradicting = loads_json(dupe["contradicting_episode_ids"]).get("ids", [])
                proto_rows = conn.execute(
                    "SELECT prototype_id FROM schema_prototype_map WHERE schema_id = ?",
                    (dupe_id,),
                ).fetchall()
                evidence_rows = conn.execute(
                    "SELECT episode_id, raw_event_id, quote, weight FROM schema_evidence WHERE schema_id = ?",
                    (dupe_id,),
                ).fetchall()
                self.reinforce_schema(
                    canonical_id,
                    prototype_ids=[int(r["prototype_id"]) for r in proto_rows],
                    supporting_episode_ids=[int(x) for x in supporting],
                    contradicting_episode_ids=[int(x) for x in contradicting],
                    evidence=[
                        (r["episode_id"], r["raw_event_id"], r["quote"], float(r["weight"]))
                        for r in evidence_rows
                    ],
                    salience_delta=max(0.01, min(float(dupe["salience"]) * 0.1, 0.2)),
                    confidence=float(dupe["confidence"]),
                    facets=loads_json(dupe["facets_json"]),
                    tags=[str(t) for t in loads_json(dupe["tags_json"]).get("tags", [])],
                )
                conn.execute(
                    "UPDATE schemas SET status = 'archived', salience = 0.05, last_updated_ts = ? WHERE id = ?",
                    (now, dupe_id),
                )
                conn.execute(
                    "INSERT INTO schema_relations "
                    "(src_schema_id, dst_schema_id, relation, confidence, reason, created_ts) "
                    "VALUES (?, ?, 'relates_to', ?, ?, ?) "
                    "ON CONFLICT(src_schema_id, dst_schema_id, relation) DO UPDATE SET "
                    "confidence=excluded.confidence, reason=excluded.reason, created_ts=excluded.created_ts",
                    (
                        canonical_id,
                        dupe_id,
                        1.0,
                        "exact normalized duplicate archived after merging into canonical schema",
                        now,
                    ),
                )
                result["merged_rows"] += 1
        conn.commit()
        return result

    def health(self) -> dict[str, Any]:
        """Return lightweight schema quality metrics."""
        conn = self.db.connect()
        total = int(conn.execute("SELECT COUNT(*) AS n FROM schemas").fetchone()["n"])
        by_status = {
            str(r["status"]): int(r["n"])
            for r in conn.execute(
                "SELECT status, COUNT(*) AS n FROM schemas GROUP BY status ORDER BY n DESC"
            ).fetchall()
        }
        rows = conn.execute(
            "SELECT scope_id, content_text FROM schemas WHERE status = 'active'"
        ).fetchall()
        active = len(rows)
        unique_keys = {
            (
                None if r["scope_id"] is None else str(r["scope_id"]),
                normalize_schema_text(r["content_text"]),
            )
            for r in rows
        }
        exact_duplicate_rows = active - len(unique_keys)
        sal = conn.execute(
            "SELECT MIN(salience) AS min_sal, AVG(salience) AS avg_sal, MAX(salience) AS max_sal "
            "FROM schemas WHERE status = 'active'"
        ).fetchone()
        return {
            "schemas_total": total,
            "schemas_by_status": by_status,
            "active_schemas": active,
            "active_unique_exact_by_scope": len(unique_keys),
            "active_exact_duplicate_rows": exact_duplicate_rows,
            "active_exact_duplicate_ratio": (exact_duplicate_rows / active) if active else 0.0,
            "active_salience": {
                "min": None if sal["min_sal"] is None else float(sal["min_sal"]),
                "avg": None if sal["avg_sal"] is None else float(sal["avg_sal"]),
                "max": None if sal["max_sal"] is None else float(sal["max_sal"]),
            },
        }

    def decay_unused(
        self,
        *,
        idle_days: float = 30.0,
        decay_amount: float = 0.15,
        labile_threshold: float = 0.30,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """Decay salience of active schemas that have never been recalled.

        Brain analogue: memories that are never activated during waking or sleep
        phases weaken over time. Schemas with zero recurrence that are older than
        ``idle_days`` lose ``decay_amount`` salience per call. Those whose salience
        falls below ``labile_threshold`` are flagged labile (see core/08-feedback.md's
        "Labile State & Reconsolidation" section for what reads and resolves this flag).

        Only affects ``active`` schemas with ``recurrence_count == 0`` (or missing)
        and ``first_formed_ts`` older than ``idle_days``. Explicit-remember schemas
        (``source_kind == "explicit_remember"``) are intentionally excluded — the
        user asked us to keep those.

        Returns a stats dict with ``decayed``, ``flagged_labile``, ``dry_run``.
        """
        now = int(time.time())
        cutoff_ts = now - int(idle_days * 86400)
        conn = self.db.connect()
        rows = conn.execute(
            "SELECT id, salience, facets_json, first_formed_ts FROM schemas "
            "WHERE status = 'active' AND first_formed_ts < ?",
            (cutoff_ts,),
        ).fetchall()

        decayed = 0
        flagged = 0
        for row in rows:
            facets = loads_json(row["facets_json"])
            if not isinstance(facets, dict):
                facets = {}
            # Skip explicitly remembered schemas — user-authored memories are preserved.
            source_kind = str(facets.get("source_kind") or facets.get("source") or "")
            if source_kind == "explicit_remember":
                continue
            recurrence = int(facets.get("recurrence_count", 0))
            if recurrence > 0:
                continue  # schema has been recalled at least once — leave it alone

            sid = int(row["id"])
            new_salience = max(0.01, float(row["salience"]) - decay_amount)
            flag_labile = new_salience < labile_threshold

            if not dry_run:
                sets = ["salience = ?", "last_updated_ts = ?"]
                args: list[Any] = [new_salience, now]
                if flag_labile:
                    sets.append("is_labile = 1")
                args.append(sid)
                conn.execute(f"UPDATE schemas SET {', '.join(sets)} WHERE id = ?", tuple(args))
            decayed += 1
            if flag_labile:
                flagged += 1

        if not dry_run:
            conn.commit()

        return {
            "idle_days": idle_days,
            "decay_amount": decay_amount,
            "labile_threshold": labile_threshold,
            "decayed": decayed,
            "flagged_labile": flagged,
            "dry_run": dry_run,
        }

    def backfill_facet_axes(
        self,
        *,
        min_members: int = 3,
        n_facet_axes: int = 4,
        limit: int = 100,
    ) -> dict[str, Any]:
        """Retroactively compute and persist facet axes for schemas that have
        accumulated enough supporting episodes but lack facet data.

        ``engine.remember()`` creates schemas from single episodes, so facet
        axes (PCA of supporting episode embeddings) cannot be computed at
        creation time.  As schemas accumulate evidence via reinforces, they
        cross the ``min_members`` threshold but the axes are never backfilled.
        This method closes that gap by loading supporting-episode embeddings,
        running SVD, and persisting the top *n_facet_axes* components.

        Returns a dict with keys ``backfilled``, ``skipped_no_embeddings``,
        ``skipped_svd_failed``, ``backfilled_ids`` (schema ids that gained
        facet axes this call). backfill_part_of_edges used to consume
        ``backfilled_ids`` to restrict its subspace-containment comparison to
        newly-eligible schemas instead of re-scanning the whole pool; removed
        2026-07-23 (see
        private/docs/iterations/20260723_part_of_audit_and_brain_alignment_review.md).
        This method itself stays -- facet_distance (logged into relates_to's
        reason string) and the dormant VSA hypervector encoding
        (slowave/latent/vsa.py) both still read facet_axes off schemas.
        ``backfilled_ids`` currently has no caller; kept for whichever of
        those two picks it up next (e.g. a VSA re-encode step).
        """
        conn = self.db.connect()
        rows = conn.execute(
            """
            SELECT id, supporting_episode_ids
            FROM schemas
            WHERE (n_facet_axes IS NULL OR n_facet_axes = 0)
              AND json_array_length(supporting_episode_ids, '$.ids') >= ?
              AND status IN ('active', 'needs_review')
            ORDER BY json_array_length(supporting_episode_ids, '$.ids') DESC
            LIMIT ?
            """,
            (min_members, limit),
        ).fetchall()

        backfilled = 0
        skipped_no_embeddings = 0
        skipped_svd_failed = 0
        backfilled_ids: list[int] = []

        for row in rows:
            schema_id = int(row["id"])
            supp = loads_json(row["supporting_episode_ids"])
            episode_ids = [int(x) for x in supp.get("ids", [])]
            if len(episode_ids) < min_members:
                continue

            # Load embeddings for supporting episodes
            ph = ",".join(["?"] * len(episode_ids))
            ep_rows = conn.execute(
                f"SELECT embedding, dim FROM episodic_memories WHERE id IN ({ph})",
                tuple(episode_ids),
            ).fetchall()

            embs = []
            for er in ep_rows:
                try:
                    d = int(er["dim"])
                    emb = unpack_f32(er["embedding"], d)
                    embs.append(emb.astype(np.float32))
                except Exception:
                    pass

            if len(embs) < min_members:
                skipped_no_embeddings += 1
                continue

            embs_arr = np.stack(embs, axis=0)
            centroid = embs_arr.mean(axis=0)

            # Centre and run SVD (same as LatentSchemaBuilder.build)
            try:
                _, s, vh = np.linalg.svd(embs_arr - centroid, full_matrices=False)
                k = min(n_facet_axes, vh.shape[0])
                axes = vh[:k].astype(np.float32)
                strengths = s[:k].astype(np.float32)
            except np.linalg.LinAlgError:
                skipped_svd_failed += 1
                continue

            axes_blob = pack_f32_matrix(axes)
            strengths_blob = pack_f32(strengths)

            conn.execute(
                """
                UPDATE schemas
                SET facet_axes = ?, facet_strengths = ?, n_facet_axes = ?,
                    last_updated_ts = ?
                WHERE id = ?
                """,
                (axes_blob, strengths_blob, k, int(time.time()), schema_id),
            )
            backfilled += 1
            backfilled_ids.append(schema_id)

        conn.commit()
        return {
            "backfilled": backfilled,
            "skipped_no_embeddings": skipped_no_embeddings,
            "skipped_svd_failed": skipped_svd_failed,
            "backfilled_ids": backfilled_ids,
        }

    def count(self) -> int:
        conn = self.db.connect()
        row = conn.execute("SELECT COUNT(*) AS n FROM schemas").fetchone()
        return int(row["n"])

    def count_by_scope(self, scope_id: str | None) -> int:
        conn = self.db.connect()
        if scope_id is None:
            row = conn.execute(
                "SELECT COUNT(*) AS n FROM schemas "
                "WHERE scope_id IS NULL AND status IN ('active', 'needs_review')"
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT COUNT(*) AS n FROM schemas "
                "WHERE scope_id = ? AND status IN ('active', 'needs_review')",
                (scope_id,),
            ).fetchone()
        return int(row["n"])

    def _row_to_schema(self, row: Any) -> Schema:
        supporting = loads_json(row["supporting_episode_ids"]).get("ids", [])
        contradicting = loads_json(row["contradicting_episode_ids"]).get("ids", [])
        # Unpack stored embedding blob so the working-memory gate can score
        # activation geometrically (cosine) rather than purely lexically.
        emb: np.ndarray | None = None
        dim: int | None = None
        try:
            if row["dim"] is not None:
                dim = int(row["dim"])
            if row["embedding"] is not None and dim is not None:
                emb = unpack_f32(row["embedding"], dim)
        except Exception:
            emb = None
        try:
            gen_stage = int(row["generalization_stage"])
        except (KeyError, TypeError, IndexError):
            gen_stage = 0
        try:
            logic_version = str(row["logic_version"])
        except (KeyError, TypeError, IndexError):
            logic_version = "0"
        # Facet axes/strengths: NULL / n_facet_axes=0 (legacy row, or the schema
        # genuinely had fewer than min_members_for_facets members) both degrade
        # to an empty (0, dim) / (0,) matrix rather than None.
        n_facets = 0
        try:
            if row["n_facet_axes"] is not None:
                n_facets = int(row["n_facet_axes"])
        except (KeyError, TypeError, IndexError):
            n_facets = 0
        facet_axes = np.zeros((0, dim or 0), dtype=np.float32)
        facet_strengths = np.zeros((0,), dtype=np.float32)
        try:
            if n_facets > 0 and dim is not None and row["facet_axes"] is not None:
                facet_axes = unpack_f32_matrix(row["facet_axes"], n_facets, dim)
            if n_facets > 0 and row["facet_strengths"] is not None:
                facet_strengths = unpack_f32(row["facet_strengths"], n_facets)
        except (KeyError, TypeError, IndexError, ValueError):
            facet_axes = np.zeros((0, dim or 0), dtype=np.float32)
            facet_strengths = np.zeros((0,), dtype=np.float32)
        return Schema(
            id=int(row["id"]),
            prototype_id=None if row["prototype_id"] is None else int(row["prototype_id"]),
            content_text=str(row["content_text"]),
            facets=loads_json(row["facets_json"]),
            tags=[str(t) for t in loads_json(row["tags_json"]).get("tags", [])],
            scope_id=None if row["scope_id"] is None else str(row["scope_id"]),
            status=str(row["status"]),
            confidence=float(row["confidence"]),
            salience=float(row["salience"]),
            supporting_episode_ids=[int(x) for x in supporting],
            contradicting_episode_ids=[int(x) for x in contradicting],
            is_labile=bool(row["is_labile"]),
            first_formed_ts=int(row["first_formed_ts"]),
            last_updated_ts=int(row["last_updated_ts"]),
            embedding=emb,
            generalization_stage=gen_stage,
            facet_axes=facet_axes,
            facet_strengths=facet_strengths,
            logic_version=logic_version,
        )
