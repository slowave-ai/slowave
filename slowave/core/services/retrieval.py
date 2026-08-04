"""RetrievalService: multi-mechanism semantic recall and working-memory gating.

Previously implemented as methods on SlowaveEngine. Extracted so the retrieval
pipeline can be read, tested, and reasoned about independently.
"""

from __future__ import annotations

import dataclasses
import math
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from slowave.core.config import DEFAULT_RECALL_TOP_K
from slowave.core.context import (
    GatePolicy,
    MemoryCue,
    WorkingMemoryGate,
    WorkingMemoryState,
    _render,
    spread_relation_activation,
)
from slowave.core.scope import normalize_scope
from slowave.latent.episodic_store import EpisodicStore
from slowave.latent.graph_manager import GraphManager
from slowave.latent.retrieval import RetrievalConfig, RetrievalPipeline
from slowave.latent.semantic_store import SemanticStore
from slowave.latent.temporal import TemporalProbe
from slowave.latent.transition_model import TransitionModel
from slowave.latent.types import EpisodeDiagnostic, QueryDiagnostics, RetrievedMemorySet
from slowave.storage.sqlite_db import SQLiteDB
from slowave.symbolic.encoder import TextEncoder
from slowave.symbolic.episode_text import EpisodeTextStore
from slowave.symbolic.raw_log import RawLog
from slowave.symbolic.schema_store import Schema, SchemaStore

# Minimum injected activation (recall()'s own schema_scores scale -- cosine +
# a fixed bonus per candidate source, roughly 0.15-1.25, not the 0-1 scale
# WorkingMemoryGate uses) for a schema_relations-propagated neighbor to be
# worth surfacing. A single hop from a typical FTS-level match (~0.35) already
# loses ~40% to confidence*decay, so this is set proportionately lower than
# the direct-hit floor rather than reusing it verbatim -- see
# spread_relation_activation's docstring for why decay makes deep/weak paths
# self-limiting without needing this floor to also do that work.
_RECALL_GRAPH_MIN_ACTIVATION = 0.15

# A schema's most recent exposure history skews net-negative once explicit
# irrelevant/stale/wrong marks outnumber explicit useful marks by enough to
# push context_noise_score (negative / (negative + 3*used + 1), see
# SchemaStore._update_utility_scores) past this ratio. Below the threshold a
# single stray `irrelevant` mark against an otherwise well-used schema can't
# flip the gate -- only a genuine net-negative trend does.
_EXPOSURE_GATE_NOISE_THRESHOLD = 0.5

# Below this combined cue-text length (query + goal + task_type + situation +
# requirements + topics + entities, joined), context_brief() treats the call
# as carrying no real signal (a one-word ack, an empty follow-up) and shrinks
# its candidate-fetch limits accordingly. The call itself still always runs --
# this only makes the underlying work proportionate to the signal present,
# it does not skip or condition the call.
_TRIVIAL_CUE_MIN_CHARS = 12

# Default floor for recall()'s min_relevance param. As of WP-4 this is
# applied to schema_scores alone (topical/channel evidence — embedding,
# FTS, prototype, profile, promoted-cosine — NOT the same 0-1 scale as
# GatePolicy.min_activation/min_relevance). See recall()'s docstring for
# why the scales differ, and _rank_score's docstring for why the floor
# moved off the salience-blended rank score. Raised 0.20 -> 0.30 per the
# WP-4 replay-corpus sweep (run_relevance_floor_sweep.py): 0.3-0.5 gave an
# identical, maximal result (noise_free 18.2%->90.9%, empty_correct
# 0%->100%, zero all_relevant regression); 0.30 is the least-aggressive
# value in that plateau.
_RECALL_MIN_RELEVANCE_DEFAULT = 0.30

# Default floor for context_brief()/activate()'s min_relevance param, on
# WorkingMemoryGate's 0-1 relevance scale (cosine*0.40 + lexical_weight*
# overlap, pre-prior). See GatePolicy.min_relevance (library default stays
# 0.0/disabled -- this is a production-call-site override, same pattern as
# min_activation below). Calibrated against the WP-2 replay corpus via
# private/experiments/run_relevance_floor_sweep.py: floors from 0.05-0.30
# gave an identical, maximal result on that corpus (noise_free 18.2%->100%,
# empty_correct 0%->100%, zero all_relevant regression), so 0.10 is chosen
# as a mid-plateau value with margin under real (continuous, not
# axis-aligned) embedding similarity -- not the corpus-optimal edge value.
_ACTIVATE_MIN_RELEVANCE_DEFAULT = 0.10

# WP-5 (associative retrieval ablations, plan Phase 3): which schema_relations
# edge types are allowed to seed a graph-propagated neighbor at all, in both
# context_brief() and recall(). Valid values: "off" (no graph expansion --
# direct-only retrieval), "relates_to" (content/semantic relations only),
# "coactivated_with" (usage co-presentation only), "combined" (every edge
# type contributes -- the pre-WP-5 behavior).
#
# WP-5.1 (2026-07-29, post-execution validation R5 follow-up) flipped this
# from "combined" to "off". Three independent findings, not just absence of
# evidence:
#   1. Live production feedback on graph-pathway items: 0 used / 4 negative
#      in the post-boundary v3 cohort (post_execution_validation.md) -- every
#      graph item a human judged in production was rejected.
#   2. Same-scope graph rescue (a neighbor direct retrieval misses but the
#      graph channel recovers) is mathematically impossible at the shipped
#      floors, not merely unobserved: direct relevance's cosine contribution
#      is 0.40 * cosine gated at _ACTIVATE_MIN_RELEVANCE_DEFAULT (0.10);
#      _GRAPH_MIN_NEIGHBOR_RELEVANCE_DEFAULT gates raw cosine at 0.25;
#      0.40 * 0.25 == 0.10 -- the two floors share the exact same breakeven,
#      so anything clearing the graph gate necessarily clears direct
#      admission too, same-scope, always (empirically confirmed via a cosine
#      sweep, see private/experiments/find_graph_utility_evidence.py and the
#      validation doc's addendum). Cross-scope is symmetric for ordinary
#      (generalization_stage=0) schemas since _eligible() hard-blocks them
#      identically on both paths in strict_scope mode.
#   3. Until WP-5.1's ranking fix just above in expand_via_relations(), an
#      admitted graph neighbor's ranked activation was raw graph mass clamped
#      to 1.0 regardless of margin over the cosine floor, so it would
#      routinely outrank a genuinely relevant direct hit (whose activation is
#      a bounded sum that rarely nears 1.0) whenever both competed for a
#      limited response -- actively worse than merely "unproven benefit".
#
# The 11-case WP-2 replay corpus still can't distinguish channels
# (graph_relevant_hits is 0 for every channel/floor combination in
# run_graph_channel_sweep.py -- a corpus coverage gap, not evidence
# "relates_to" is safer than "coactivated_with"), so this default isn't a
# channel-restriction decision -- it's "no graph expansion at all until (1)'s
# net-negative live signal reverses or (2)'s floor relationship is
# recalibrated so same-scope rescue is possible in principle." Data
# collection is unaffected: the consolidation worker keeps writing
# coactivated_with edges regardless of this default, and any caller can still
# pass graph_channels="combined" (or another value) explicitly to opt back in
# per-call.
_GRAPH_CHANNELS_DEFAULT = "off"

# WP-5 dual gate (plan Phase 3, "association is mistaken for answer
# confidence"): a schema_relations edge means "these schemas are related or
# were recalled together" -- it says nothing about whether the neighbor
# answers THIS query. expand_via_relations()/recall()'s graph section
# previously admitted a neighbor purely on accumulated graph mass, with zero
# check that the neighbor is topically close to the cue at all (the
# graph_hub_saturation and budget_graph_overflow replay cases both reproduce
# this). When > 0.0, a graph-propagated neighbor must ALSO clear this cosine
# floor against the cue embedding, independent of its graph mass. Scale
# matches direct cosine similarity ([-1, 1], clamped to [0, 1] effectively
# since a negative match never clears a positive floor) -- NOT
# GatePolicy.min_relevance's blended 0-0.40+ scale.
#
# Calibrated via private/experiments/run_graph_channel_sweep.py against the
# WP-2 replay corpus (on top of the WP-4 direct-relevance defaults): with the
# gate disabled (0.0), the "coactivated_with" and "combined" channels each
# inject exactly 2 graph-attributable noise items (graph_hub_saturation,
# budget_graph_overflow -- both wired with coactivated_with edges in the
# corpus) and zero graph-attributable relevant hits on either channel at any
# floor. Floors 0.15-0.50 all fully eliminate that noise with no change to
# graph_relevant_hits (0 throughout) or any other metric -- a flat plateau,
# same shape as WP-4's threshold sweeps. 0.25 is chosen as a mid-plateau
# value, matching the existing cross-scope cosine gate at the same value
# (WorkingMemoryGate.select's `_cosine < 0.25` check) rather than the
# corpus-optimal edge, for the same reason WP-4 avoided edge values: real
# (non-axis-aligned) embeddings produce continuous similarity, not this
# corpus's clean 0/1 split.
_GRAPH_MIN_NEIGHBOR_RELEVANCE_DEFAULT = 0.25

_GRAPH_RELATION_FILTERS: dict[str, "frozenset[str] | None"] = {
    "relates_to": frozenset({"relates_to"}),
    "coactivated_with": frozenset({"coactivated_with"}),
    "combined": None,
}


def _relation_filter_for(graph_channels: str) -> "frozenset[str] | None":
    if graph_channels not in ("off", *_GRAPH_RELATION_FILTERS):
        raise ValueError(
            f"graph_channels must be one of 'off', {sorted(_GRAPH_RELATION_FILTERS)}, "
            f"got {graph_channels!r}"
        )
    return _GRAPH_RELATION_FILTERS.get(graph_channels)


@dataclass(frozen=True)
class RecallResult:
    """Recall result: schemas + episodes + raw events with provenance."""

    schemas: list[Schema]
    episode_texts: list[dict[str, Any]]
    raw_events: list[dict[str, Any]]
    expanded_neighbors: dict[int, list[tuple[int, float]]]
    schema_activations: dict[int, float] = field(
        default_factory=dict
    )  # schema_id -> raw channel/relevance score (schema_scores; what min_relevance gates on)
    schema_rank_scores: dict[int, float] = field(
        default_factory=dict
    )  # schema_id -> relevance + salience_weight*norm_salience (sort order, WP-4)
    episode_diagnostics: list[EpisodeDiagnostic] = field(default_factory=list)
    query_diagnostics: QueryDiagnostics | None = None
    # Stage 10 anchor diagnostics (plans/07-temporal.md Phase 4). Populated
    # unconditionally — TemporalProbe.estimate_anchor() runs on every recall()
    # regardless of RetrievalConfig.use_temporal (core/07-temporal.md Invariant 7).
    anchor_fired: bool = False  # True when estimate_anchor() returned something other than now_ts
    anchor_displacement_s: int = 0  # anchor_ts - now_ts; 0 when not fired
    # schema_relations-propagated schemas (spread_relation_activation) that
    # were NOT among the top_k direct hits. Deliberately kept OUT of `schemas`:
    # every benchmark script (retrieval_metrics.compute_recall_at_k_and_mrr,
    # dmr_original_eval.py, etc.) concatenates `schemas` assuming its length
    # is bounded by the `top_k` passed to this call -- merging graph winners
    # into that list would silently inflate recall@k/MRR/keyword-score by
    # smuggling in more than k schemas' worth of context.
    related_schemas: list[Schema] = field(default_factory=list)
    # schema_id -> relation type(s) it arrived via (e.g. ["relates_to"] or
    # ["coactivated_with"]), for related_schemas entries only -- lets callers
    # show/verify *why* a related schema surfaced instead of just that it did.
    related_schema_relations: dict[int, list[str]] = field(default_factory=dict)


def _prefix_date(text: str, ts: int) -> str:
    """Prepend an ISO date tag to an episode's text: "[YYYY-MM-DD] <text>"."""
    try:
        date_str = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")
        return f"[{date_str}] {text}" if text else f"[{date_str}]"
    except Exception:
        return text


def _normalize_episode_text(text: str) -> str:
    """Normalize episode text for deduplication.

    Strips date prefix ([YYYY-MM-DD]), role prefixes (user:, assistant:, etc.),
    and collapses whitespace so identical content with different formatting deduplicates.
    """
    text = text.strip().lower()
    # Strip date prefix: "[YYYY-MM-DD] "
    text = re.sub(r"^\[\d{4}-\d{2}-\d{2}\]\s*", "", text)
    # Strip role prefixes: "remember:", "user:", "assistant:", "system:", "note:"
    text = re.sub(r"^(remember|user|assistant|system|note):\s*", "", text)
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)
    return text


def _norm_salience(s: float) -> float:
    """Normalise raw salience [0, ∞) → [0, 1) via sigmoid.

    Fixes P1: raw salience ranges 0.01–4.0+ so salience_weight
    multiplication is unnormalised.  The sigmoid compresses the
    range while preserving monotonicity, giving salience a
    controlled contribution to the ranking score.
    """
    return 2.0 / (1.0 + math.exp(-s / 2.0)) - 1.0


class RetrievalService:
    """Multi-mechanism semantic recall and working-memory gating."""

    def __init__(
        self,
        *,
        episodic: EpisodicStore,
        semantic: SemanticStore,
        graph: GraphManager,
        schemas: SchemaStore,
        encoder: TextEncoder | None,
        episode_text: EpisodeTextStore,
        raw_log: RawLog,
        retrieval: RetrievalPipeline,
        transition_model: TransitionModel,
        temporal_probe: TemporalProbe | None,
        working_memory_gate: WorkingMemoryGate,
        db: SQLiteDB,
        retrieval_cfg: RetrievalConfig,
    ):
        self.episodic = episodic
        self.semantic = semantic
        self.graph = graph
        self.schemas = schemas
        self.encoder = encoder
        self.episode_text = episode_text
        self.raw_log = raw_log
        self.retrieval = retrieval
        self.transition_model = transition_model
        self._temporal_probe = temporal_probe
        self.working_memory_gate = working_memory_gate
        self.db = db
        self._retrieval_cfg = retrieval_cfg

    # ---- public API --------------------------------------------------------

    def refresh_indices(self) -> None:
        """Rebuild in-memory FAISS indices from SQLite."""
        self.episodic.reset_faiss_from_db()
        self.semantic.reset_faiss_from_db()

    def recall(
        self,
        query: str,
        *,
        top_k: int = DEFAULT_RECALL_TOP_K,
        evidence: bool = False,
        mode: str = "default",
        scope: str | None = None,
        diagnose: bool = False,
        refresh: bool = True,
        min_relevance: float = _RECALL_MIN_RELEVANCE_DEFAULT,
        graph_channels: str = _GRAPH_CHANNELS_DEFAULT,
        min_neighbor_relevance: float = _GRAPH_MIN_NEIGHBOR_RELEVANCE_DEFAULT,
    ) -> RecallResult:
        """
        refresh: rebuild the in-memory FAISS indices from SQLite before
        retrieving. This is an O(N) full-table scan (see refresh_indices),
        so callers who know no episodes/schemas were added since their last
        recall() on this engine (e.g. probing the same query at several
        top_k values back-to-back) can pass False to skip the redundant work.

        graph_channels / min_neighbor_relevance (WP-5): same associative-
        retrieval controls as context_brief() -- which schema_relations edge
        types may seed `related_schemas`, and the dual-gate cosine floor a
        graph neighbor must clear against the query embedding independent of
        its accumulated graph mass. "off" skips graph expansion entirely, so
        `related_schemas` is always empty. See _GRAPH_CHANNELS_DEFAULT /
        _GRAPH_MIN_NEIGHBOR_RELEVANCE_DEFAULT and run_graph_channel_sweep.py.

        min_relevance: candidates scoring below this on `schema_scores` --
        the per-channel topical/relevance evidence (cosine+0.25, FTS 0.35,
        prototype ~0.15-0.20, profile 0.30, promoted-cosine) BEFORE the
        salience-weighted rank adjustment -- are dropped rather than padding
        out top_k with weak matches. a thin/off-topic scope can now signal
        "nothing relevant" instead of confidently returning its top_k
        weakest matches.

        WP-4: this floor moved off `_rank_score` (schema_scores +
        salience_weight * normalized salience) onto `schema_scores` alone.
        Filtering on the salience-blended rank score let a high-salience,
        zero-relevance schema out-blend its way past the floor purely on
        salience -- exactly the failure mode this parameter exists to
        prevent. `_rank_score` still determines sort order (salience remains
        a legitimate tie-breaker among schemas that already cleared the
        relevance floor) but no longer decides admission. See
        RecallResult.schema_activations (relevance) vs
        RecallResult.schema_rank_scores (rank) for the same distinction
        surfaced to callers.

        NOTE: this is NOT the same 0-1 scale GatePolicy.min_activation/
        min_relevance use for activate()/WorkingMemoryGate -- recall()'s
        schema_scores mix cosine+0.25 with flat per-channel bonuses (FTS
        0.35, profile 0.30, prototype ~0.15-0.20), so reusing
        GatePolicy.min_activation's 0.20 verbatim would be a scale mismatch.
        _RECALL_MIN_RELEVANCE_DEFAULT is a first-pass estimate calibrated to
        reject the near-zero embedding-less fallback (0.10) and negative/weak
        cosine matches without discarding the legitimate flat-bonus channels
        -- treat it as a starting point to be tuned against the benchmark
        suite (Recall@K/MRR), not a final number. Pass 0.0 to disable the
        floor entirely.
        """
        if self.encoder is None:
            raise RuntimeError("recall requires an encoder; cfg.disable_encoder=True")
        if refresh:
            self.refresh_indices()
        q = self.encoder.encode(query)

        retrieval_pipeline = self.retrieval
        anchor_fired = False
        anchor_displacement_s = 0
        if self._temporal_probe is not None:
            now_ts = int(time.time())
            anchor_ts = self._temporal_probe.estimate_anchor(q, now_ts=now_ts)
            if anchor_ts != now_ts:
                anchor_fired = True
                anchor_displacement_s = anchor_ts - now_ts
                anchored_cfg = dataclasses.replace(
                    self._retrieval_cfg,
                    temporal_anchor_ts=anchor_ts,
                )
                retrieval_pipeline = RetrievalPipeline(
                    episodic=self.episodic,
                    semantic=self.semantic,
                    graph=self.graph,
                    cfg=anchored_cfg,
                    transition_model=self.transition_model,
                )

        retrieved: RetrievedMemorySet = retrieval_pipeline.retrieve(q, diagnose=diagnose)

        scope_id = normalize_scope(scope=scope) if scope else None

        schema_scores: dict[int, float] = {}
        for sid, score in self.schemas.search_embedding(
            q, limit=max(20, top_k * 4), scope_id=scope_id
        ):
            schema_scores[sid] = max(schema_scores.get(sid, -1e9), score + 0.25)

        # FTS candidates: scope-filter early to avoid collecting candidates
        # that will be discarded later. FTS doesn't support scope natively,
        # so we fetch schemas and filter by scope_id immediately.
        for sid in self.schemas.search_fts(query, limit=max(10, top_k * 2)):
            if scope_id:
                try:
                    s = self.schemas.get(sid)
                    if s.scope_id and s.scope_id != scope_id:
                        continue
                except KeyError:
                    continue
            schema_scores[sid] = max(schema_scores.get(sid, -1e9), 0.35)

        proto_ids = [p.id for p in retrieved.prototypes]
        for s in self.schemas.get_many_by_prototypes(proto_ids):
            if scope_id and s.scope_id and s.scope_id != scope_id:
                continue
            schema_scores[s.id] = max(schema_scores.get(s.id, -1e9), 0.15 + s.salience * 0.05)

        # Profile-layer injection: always included, but needs_review schemas only
        # surface in broad/debug modes (same gate as the rest of recall).
        conn = self.db.connect()
        profile_statuses = ("active",)
        if mode in ("broad", "debug"):
            profile_statuses = ("active", "needs_review")

        profile_sql = (
            "SELECT id FROM schemas "
            "WHERE status IN ('" + "','".join(profile_statuses) + "') "
            "AND json_extract(facets_json, '$.memory_layer') = 'profile' "
        )
        profile_args: list[Any] = []
        if scope_id:
            profile_sql += "AND (scope_id = ? OR scope_id IS NULL) "
            profile_args.append(scope_id)
        profile_sql += "ORDER BY salience DESC LIMIT ?"
        profile_args.append(top_k * 2)

        profile_rows = conn.execute(profile_sql, profile_args).fetchall()
        for row in profile_rows:
            sid = int(row["id"])
            if sid not in schema_scores:
                schema_scores[sid] = 0.30

        # Stage 11: inject promoted schemas that the scoped embedding search
        # would have missed.  search_embedding(scope_id=scope_id) only returns
        # schemas belonging to the current scope; promoted schemas (stage >= 1)
        # from other scopes are invisible to that search.
        # We score them using the same cosine + 0.25 formula as line 139 so
        # that natural paraphrases work without FTS overlap.  Schemas without
        # a stored embedding fall back to a small flat baseline (0.10) so they
        # can still enter the candidate set and be ranked by salience.
        if scope_id and mode == "strict_scope":
            import numpy as _np

            from slowave.utils.vec import unpack_f32

            _qn = float(_np.linalg.norm(q)) + 1e-12
            promoted_rows = conn.execute(
                "SELECT id, embedding, dim FROM schemas "
                "WHERE generalization_stage >= 1 "
                "AND status = 'active' "
                "AND (scope_id IS NOT NULL AND scope_id != ?)",
                (scope_id,),
            ).fetchall()
            for _row in promoted_rows:
                _sid = int(_row["id"])
                # Always compute the promoted embedding score and take max() against
                # any score already present (e.g. from FTS at 0.35).  Without this,
                # an FTS hit at 0.35 would block the cosine score: after the Stage 2
                # multiplier (0.70×) the FTS score lands at 0.245, which is below the
                # cross_scope_min_score floor of 0.30 and the schema is dropped.  The
                # cosine+0.25 score for the same lexically-matching query is typically
                # much higher and would pass — but it was never computed.
                # Using max() mirrors lines 139-144 where all three scoring paths
                # (embedding, FTS, prototype) compete and the best score wins.
                _score = 0.10  # fallback when no embedding stored
                if _row["embedding"] is not None and _row["dim"]:
                    try:
                        _v = unpack_f32(_row["embedding"], int(_row["dim"]))
                        _vn = float(_np.linalg.norm(_v)) + 1e-12
                        _cosine = float(q.dot(_v) / (_qn * _vn))
                        _score = max(0.0, _cosine) + 0.25
                    except Exception:
                        pass
                schema_scores[_sid] = max(schema_scores.get(_sid, -1e9), _score)

        schemas_all = self.schemas.get_many(schema_scores.keys())

        # Mode-gated status filter; when strict_scope, also enforce scope.
        if mode == "debug":
            recall_statuses = ("active", "needs_review", "superseded")
        elif mode == "broad":
            recall_statuses = ("active", "needs_review")
        else:  # default, strict_scope
            recall_statuses = ("active",)

        filtered_schemas = []

        for s in schemas_all:
            if s.status not in recall_statuses:
                continue

            if not self._cross_scope_gate(
                s, scope_id=scope_id, mode=mode, schema_scores=schema_scores
            ):
                continue

            # Belt-and-suspenders: apply score multiplier for labile schemas
            if s.is_labile and s.status == "active":
                schema_scores[s.id] = schema_scores.get(s.id, 0.0) * 0.20

            filtered_schemas.append(s)

        def _rank_score(s: Schema) -> float:
            return schema_scores.get(
                s.id, 0.0
            ) + self._retrieval_cfg.salience_weight * _norm_salience(s.salience)

        schemas = sorted(filtered_schemas, key=_rank_score, reverse=True)
        if min_relevance > 0.0:
            schemas = [s for s in schemas if schema_scores.get(s.id, 0.0) >= min_relevance]
        schemas = schemas[:top_k]

        for s in schemas:
            # Passive mere-exposure reinforcement on every top-k appearance is
            # gated off once a schema's exposure history skews net-negative --
            # otherwise an explicit `irrelevant`/`wrong` correction gets
            # silently erased by the very next exposure. Any subsequent
            # explicit positive mark (`useful`/`partially_useful`) raises
            # context_used_count and lifts the gate again; a positive mark can
            # only be recorded against a retrieval that surfaced this schema,
            # so this doubles as the "re-earned from a fresh query" reset.
            facets = s.facets or {}
            negative_marks = int(facets.get("context_irrelevant_count") or 0)
            used_marks = int(facets.get("context_used_count") or 0)
            noise = float(facets.get("context_noise_score") or 0.0)
            if negative_marks > used_marks and noise >= _EXPOSURE_GATE_NOISE_THRESHOLD:
                continue
            self.schemas.reinforce(s.id, amount=0.05)

        # Relation-graph spreading activation: schemas linked to one of the
        # above via schema_relations (relates_to) can still be worth
        # surfacing even though they didn't directly match the query —
        # see spread_relation_activation's docstring for the algorithm.
        # Kept OUT of `schemas`/`schema_scores`'s role as top_k results:
        # retrieval_metrics.compute_recall_at_k_and_mrr and every benchmark
        # script concatenate `schemas` assuming its length is bounded by the
        # top_k passed to this call; merging graph winners into that list
        # would silently inflate recall@k/MRR/keyword-score by smuggling in
        # more than k schemas' worth of context.
        related_schemas: list[Schema] = []
        related_schema_relations: dict[int, list[str]] = {}
        graph_winners = (
            {}
            if graph_channels == "off"
            else spread_relation_activation(
                {s.id: schema_scores.get(s.id, 0.0) for s in schemas},
                fetch_relations=self._fetch_all_relations,
                min_activation=_RECALL_GRAPH_MIN_ACTIVATION,
                relation_filter=_relation_filter_for(graph_channels),
            )
        )
        for neighbor_id, (activation, via) in graph_winners.items():
            try:
                neighbor_schema = self.schemas.get(neighbor_id)
            except KeyError:
                continue
            # Same status bar as the direct-hit candidates above (recall_statuses)
            # -- a graph-propagated neighbor is a bonus, not itself vetted by the
            # status/scope filtering loop, so a stale edge can't leak a
            # superseded/contradicted schema back into results.
            if neighbor_schema.status not in recall_statuses:
                continue
            # WP-5 dual gate: a graph edge is not proof the neighbor answers
            # THIS query (plan section 4.2) -- require it also clear a cosine
            # floor against the query embedding, independent of graph mass.
            if min_neighbor_relevance > 0.0 and neighbor_schema.embedding is not None:
                import numpy as _np

                _qn = float(_np.linalg.norm(q)) + 1e-12
                _v = _np.asarray(neighbor_schema.embedding, dtype=_np.float32)
                _vn = float(_np.linalg.norm(_v)) + 1e-12
                neighbor_cosine = float(q.dot(_v) / (_qn * _vn))
                if neighbor_cosine < min_neighbor_relevance:
                    continue
            # schema_relations is not a scope boundary -- relates_to edges can
            # legitimately form cross-scope (content similarity doesn't check
            # scope at write time), so reuse the exact same cross-scope gate
            # direct candidates go through above -- not a separate rule, and
            # not more permissive just because this candidate arrived via a
            # relation edge instead of FTS/embedding.
            schema_scores[neighbor_id] = activation
            if not self._cross_scope_gate(
                neighbor_schema, scope_id=scope_id, mode=mode, schema_scores=schema_scores
            ):
                continue
            related_schemas.append(neighbor_schema)
            related_schema_relations[neighbor_id] = sorted(via)

        prior_boost, silence_factor = self._schema_priors(
            candidate_episode_ids=[int(m.id) for m in retrieved.episodic],
            matched_schema_scores=schema_scores,
            matched_schemas=schemas_all,
        )

        ep_texts = self.episode_text.get_many([m.id for m in retrieved.episodic])
        ep_by_id = {e.episode_id: e for e in ep_texts}
        scored_pairs: list[tuple[float, Any]] = []
        n_ep = len(retrieved.episodic)
        for rank, m in enumerate(retrieved.episodic):
            base = 1.0 - (rank / max(1, n_ep))
            eid = int(m.id)
            score = (base + prior_boost.get(eid, 0.0)) * silence_factor.get(eid, 1.0)
            scored_pairs.append((score, m))
        scored_pairs.sort(key=lambda t: t[0], reverse=True)

        episode_dicts = []
        # Deduplication: track normalised episode texts already emitted.
        # Always dedup against active schema texts so episodes that merely
        # repeat an already-surfaced schema are suppressed regardless of the
        # evidence flag.
        seen_episodes: set[str] = set()
        schema_texts = {
            _normalize_episode_text(s.content_text or "") for s in schemas if s.content_text
        }

        for _score, m in scored_pairs[:top_k]:
            ep = ep_by_id.get(m.id)
            raw_text = ep.content_text if ep else ""
            dated_text = _prefix_date(raw_text, int(m.ts))

            # Skip if this episode content was already emitted (normalised).
            normalized = _normalize_episode_text(raw_text)
            if normalized and normalized in seen_episodes:
                continue

            # Skip if episode duplicates a schema already surfaced in results.
            if normalized and normalized in schema_texts:
                continue

            if normalized:
                seen_episodes.add(normalized)

            episode_dicts.append(
                {
                    "id": m.id,
                    "content_text": dated_text,
                    "salience": float(m.salience),
                    "ts": int(m.ts),
                    "schema_prior_boost": round(float(prior_boost.get(int(m.id), 0.0)), 4),
                    "schema_silence_factor": round(float(silence_factor.get(int(m.id), 1.0)), 4),
                }
            )

        raw_events_out: list[dict[str, Any]] = []
        if evidence:
            wanted: list[int] = []
            for s in schemas:
                for ev in self.schemas.evidence_for_schema(s.id, limit=5):
                    if ev.raw_event_id is not None:
                        wanted.append(ev.raw_event_id)
                    elif ev.episode_id is not None:
                        ep = self.episode_text.get(ev.episode_id)
                        if ep is not None:
                            wanted.extend(ep.event_ids[:3])
            for ep in ep_texts:
                wanted.extend(ep.event_ids[:3])
            seen: set[int] = set()
            for rid in wanted:
                if rid in seen:
                    continue
                seen.add(rid)
                try:
                    e = self.raw_log.get(rid)
                except KeyError:
                    continue
                raw_events_out.append(
                    {"id": e.id, "ts": e.ts, "type": e.type, "content": e.content}
                )

        return RecallResult(
            schemas=schemas,
            episode_texts=episode_dicts,
            raw_events=raw_events_out,
            expanded_neighbors=retrieved.expanded_neighbors,
            schema_activations=schema_scores,
            schema_rank_scores={s.id: _rank_score(s) for s in list(schemas) + related_schemas},
            episode_diagnostics=retrieved.episode_diagnostics,
            query_diagnostics=retrieved.query_diagnostics,
            anchor_fired=anchor_fired,
            anchor_displacement_s=anchor_displacement_s,
            related_schemas=related_schemas,
            related_schema_relations=related_schema_relations,
        )

    def context(self, *, scope: str | None = None, limit: int = 10) -> list[Schema]:
        """Return top active schemas, optionally scope-filtered."""
        return self.schemas.list(limit=limit, scope_id=scope, status="active")

    def _fetch_schema_or_none(self, schema_id: int) -> Schema | None:
        try:
            return self.schemas.get(schema_id)
        except KeyError:
            return None

    def _cross_scope_gate(
        self, schema: Schema, *, scope_id: str | None, mode: str, schema_scores: dict[int, float]
    ) -> bool:
        """Cross-scope admission gate, shared by recall()'s direct-candidate
        filtering AND its schema_relations graph-expansion step -- a single
        source of truth instead of two independently-drifting rules.

        Mirrors WorkingMemoryGate._eligible's stage-graduated rule exactly:
          Stage 0 (scoped)     : hard-blocked.
          Stage 1 (portable)   : allowed only within the same scope_kind.
          Stage 2 (contextual) : always admitted; score discounted + floored.
          Stage 3 (global)     : admitted without restriction or penalty.

        Cross-scope is only ever earned in strict_scope mode: every other
        mode's candidate-gathering above (embedding/FTS/prototype scoring)
        already scope-filters unconditionally when scope_id is set, so there
        is no cross-scope exception to grant outside strict_scope mode either
        -- a schema_relations neighbor doesn't get a more permissive rule just
        because it arrived via a different path.

        Mutates schema_scores[schema.id] in place for the Stage 2 discount;
        the id must already be present for the mutation and floor check to
        mean anything (callers set it before this check).
        """
        if not scope_id or not schema.scope_id or schema.scope_id == scope_id:
            return True
        if schema.scope_id in ("global", "user"):
            return True
        if mode != "strict_scope":
            return False

        from slowave.core.scope import scope_kind as _scope_kind

        gen_stage = getattr(schema, "generalization_stage", 0)
        gen_cfg = getattr(self.schemas, "_gen_cfg", None)
        if gen_stage >= 3:
            return True
        if gen_stage == 2:
            mult = gen_cfg.stage2_cross_scope_score_multiplier if gen_cfg else 0.70
            schema_scores[schema.id] = schema_scores.get(schema.id, 0.0) * mult
            floor = gen_cfg.cross_scope_min_score if gen_cfg else 0.30
            return schema_scores[schema.id] >= floor
        if gen_stage == 1:
            if _scope_kind(schema.scope_id) != _scope_kind(scope_id):
                return False
            floor = gen_cfg.cross_scope_min_score if gen_cfg else 0.30
            return schema_scores.get(schema.id, 0.0) >= floor
        return False  # stage 0: hard block

    def context_brief(
        self,
        *,
        query: str | None = None,
        scope: str | None = None,
        goal: str | None = None,
        task_type: str | None = None,
        situation: dict[str, Any] | None = None,
        requirements: list[str] | tuple[str, ...] | None = None,
        application: str | None = None,
        topics: list[str] | tuple[str, ...] | None = None,
        entities: list[str] | tuple[str, ...] | None = None,
        limit: int = 8,
        mode: str = "default",
        max_chars: int = 1800,
        include_peripheral: bool = True,
        min_relevance: float = _ACTIVATE_MIN_RELEVANCE_DEFAULT,
        graph_channels: str = _GRAPH_CHANNELS_DEFAULT,
        min_neighbor_relevance: float = _GRAPH_MIN_NEIGHBOR_RELEVANCE_DEFAULT,
    ) -> WorkingMemoryState:
        """Return a gated working-memory state for prompt injection.

        min_relevance: candidates whose topical evidence alone (cosine +
        lexical cue_overlap, computed before scope/identity priors — see
        WorkingMemoryGate._activation) falls below this floor are rejected
        regardless of scope match, salience, or class bonuses. Distinct from
        the fixed 0.20 min_activation floor below, which still gates the
        full prior-adjusted score. Default 0.0 (disabled) — see
        private/experiments/run_relevance_floor_sweep.py for calibration.

        include_peripheral: when False, disables the trailing
        salience-filled "exploration slots" (the ones rendered as
        "(peripheral)") by zeroing GatePolicy.exploration_slots, so only
        relevance-ranked items are returned.

        graph_channels (WP-5): "off" | "relates_to" | "coactivated_with" |
        "combined" — which schema_relations edge types may seed a
        graph-propagated neighbor. "off" skips expand_via_relations()
        entirely (direct-only retrieval). See _GRAPH_CHANNELS_DEFAULT.

        min_neighbor_relevance (WP-5): dual-gate cosine floor a graph
        neighbor must clear against the cue embedding, independent of its
        accumulated graph mass. See _GRAPH_MIN_NEIGHBOR_RELEVANCE_DEFAULT.
        """
        scope_id = normalize_scope(scope=scope)
        candidates_by_id: dict[int, Schema] = {}

        def add_many(schemas_list: list[Schema]) -> None:
            for schema in schemas_list:
                candidates_by_id[schema.id] = schema

        # Mode-gated status fetch: which schema statuses to include depends on mode.
        if mode in ("broad", "debug"):
            pass
        if mode == "debug":
            pass

        # Computed early (pure string join, no side effects) so the candidate
        # fetch below can size itself to how much real signal this call
        # carries. A trivial/near-empty cue (one-word ack, empty follow-up)
        # still runs the full call -- callers must keep calling this every
        # turn -- but doesn't need to fetch and rescore 60+ candidates to
        # answer it: the call stays mandatory, only the underlying work gets
        # cheaper.
        cue_text = " ".join(
            [
                query or "",
                application or "",
                scope_id or "",
                goal or "",
                task_type or "",
                " ".join(f"{k} {v}" for k, v in sorted((situation or {}).items())),
                " ".join(requirements or []),
                " ".join(topics or []),
                " ".join(entities or []),
            ]
        ).strip()
        trivial_cue = len(cue_text) < _TRIVIAL_CUE_MIN_CHARS

        scoped_fetch_limit = max(10, limit) if trivial_cue else max(60, limit * 6)
        global_fetch_limit = max(20, limit * 2) if trivial_cue else max(100, limit * 8)

        if scope_id:
            # Fetch active schemas for scope
            add_many(
                self.schemas.list(limit=scoped_fetch_limit, scope_id=scope_id, status="active")
            )
            # Fetch needs_review/superseded for scope if in appropriate mode
            if mode in ("broad", "debug"):
                add_many(
                    self.schemas.list(
                        limit=scoped_fetch_limit, scope_id=scope_id, status="needs_review"
                    )
                )
            if mode == "debug":
                add_many(
                    self.schemas.list(
                        limit=scoped_fetch_limit, scope_id=scope_id, status="superseded"
                    )
                )
            # Stage 11: also inject generalization-promoted schemas (Stage 2/3) as candidates.
            # Stage 1 (portable) is handled inside _eligible via scope_kind match.
            # Stage 2/3 are surfaced here so the working-memory gate can score them;
            # stage-penalty is applied inside WorkingMemoryGate._eligible.
            conn = self.db.connect()
            promoted_rows = conn.execute(
                "SELECT id FROM schemas WHERE generalization_stage >= 2 "
                "AND status = 'active' AND scope_id != ?",
                (scope_id,),
            ).fetchall()
            for r in promoted_rows:
                try:
                    candidates_by_id[int(r["id"])] = self.schemas.get(int(r["id"]))
                except KeyError:
                    pass

        # Fetch from global pool
        add_many(self.schemas.list(limit=global_fetch_limit, status="active"))
        if mode in ("broad", "debug"):
            add_many(self.schemas.list(limit=global_fetch_limit, status="needs_review"))
        if mode == "debug":
            add_many(self.schemas.list(limit=global_fetch_limit, status="superseded"))

        cue_embedding = None
        if cue_text:
            cue_fetch_limit = max(10, limit) if trivial_cue else max(50, limit * 8)
            for sid in self.schemas.search_fts(cue_text, limit=cue_fetch_limit):
                try:
                    candidates_by_id[sid] = self.schemas.get(sid)
                except KeyError:
                    continue
            if self.encoder is not None:
                cue_embedding = self.encoder.encode(cue_text)
                for sid, _score in self.schemas.search_embedding(
                    cue_embedding, limit=cue_fetch_limit
                ):
                    try:
                        candidates_by_id[sid] = self.schemas.get(sid)
                    except KeyError:
                        continue

        cue = MemoryCue(
            query=query,
            scope=scope_id,
            goal=goal,
            task_type=task_type,
            situation=situation or {},
            requirements=tuple(requirements or ()),
            application=application,
            topics=tuple(topics or ()),
            entities=tuple(entities or ()),
            mode=mode,
        )
        policy = GatePolicy(
            max_items=limit,
            max_chars=max_chars,
            min_activation=-999.0 if mode == "debug" else 0.20,
            min_relevance=0.0 if mode == "debug" else min_relevance,
            exploration_slots=0 if not include_peripheral else GatePolicy.exploration_slots,
        )
        state = self.working_memory_gate.select(
            candidates_by_id.values(), cue=cue, policy=policy, cue_embedding=cue_embedding
        )
        if graph_channels != "off":
            state = self.working_memory_gate.expand_via_relations(
                state,
                fetch_relations=self._fetch_all_relations,
                fetch_schema=self._fetch_schema_or_none,
                cue=cue,
                policy=policy,
                cue_embedding=cue_embedding,
                min_neighbor_relevance=min_neighbor_relevance,
                relation_filter=_relation_filter_for(graph_channels),
            )
        # When include_peripheral=False, strip ALL peripheral items — both
        # exploration slots (already handled by exploration_slots=0 above)
        # and graph-propagated neighbors — from the final context brief.
        if not include_peripheral:
            state = dataclasses.replace(
                state,
                items=[i for i in state.items if not i.peripheral],
                rendered=_render([i for i in state.items if not i.peripheral]),
            )
        return state

    # ---- private -----------------------------------------------------------

    def _fetch_all_relations(self, schema_id: int) -> list[tuple[int, str, float]]:
        """Composite: content relations + usage-based co-activation edges.

        Both sources return (neighbor_id, relation_label, weight)
        tuples — identical shape — so spread_relation_activation() consumes
        them transparently. Co-activation edges carry "coactivated_with"
        as their relation label.

        Co-activation weight is squashed to [0, 1) before being passed on
        (2026-07-23): schema_relations' confidence is bounded [0, 1], but
        raw Hebbian weight is an unbounded accumulating counter — under
        realistic daily-reinforcement cadence it converges to ~10 (7-day
        half-life), ~10x the maximum any content-relation edge can
        contribute. spread_relation_activation() multiplies this value
        directly into injected activation, so passing the raw weight would
        let a well-worn co-activation edge dominate the graph-expansion
        winners regardless of topical relevance. w/(w+1) keeps it
        monotonic (more co-occurrence still ranks higher) while capping it
        at the same ceiling content relations already have.
        """
        content = self.schemas.get_relations(schema_id)
        coact = self.schemas.get_coactivations(schema_id)
        normalized_coact = [(nid, rel, w / (w + 1.0)) for nid, rel, w in coact]
        return content + normalized_coact

    def _schema_priors(
        self,
        *,
        candidate_episode_ids: list[int],
        matched_schema_scores: dict[int, float],
        matched_schemas: list[Schema],
    ) -> tuple[dict[int, float], dict[int, float]]:
        """Compute schemas-as-priors boosts and belief-revision silences."""
        if not candidate_episode_ids:
            return {}, {}
        ep_schema_index = self.schemas.schemas_for_episodes(candidate_episode_ids)
        offsets = {"embed": 0.25, "fts": 0.35, "proto": 0.15}
        matched_q_score: dict[int, float] = {}
        for s in matched_schemas:
            raw = matched_schema_scores.get(int(s.id), 0.0)
            qsim = max(0.0, min(1.0, raw - offsets["embed"]))
            if qsim > 0.0:
                matched_q_score[int(s.id)] = qsim

        now_ts = int(time.time())
        silence_halflife_s = 14.0 * 86400.0
        prior_boost: dict[int, float] = {}
        silence_factor: dict[int, float] = {}
        for eid, entries in ep_schema_index.items():
            for sid, status, conf, last_ts in entries:
                if status in ("active", "needs_review"):
                    qsim = matched_q_score.get(sid)
                    if qsim is None:
                        continue
                    schema_obj = None
                    try:
                        schema_obj = self.schemas.get(sid)
                    except KeyError:
                        pass
                    utility = (
                        float((schema_obj.facets or {}).get("schema_utility", 0.0))
                        if schema_obj
                        else 0.0
                    )
                    utility_mult = 1.0 + 0.5 * utility
                    boost = 0.08 * float(qsim) * float(conf) * utility_mult
                    prior_boost[eid] = max(prior_boost.get(eid, 0.0), boost)
                elif status in ("superseded", "contradicted"):
                    age = max(0.0, float(now_ts - int(last_ts)))
                    fresh = 0.5 ** (age / silence_halflife_s)
                    damp = 0.6 * fresh * float(conf)
                    factor = max(0.05, 1.0 - damp)
                    silence_factor[eid] = min(silence_factor.get(eid, 1.0), factor)
        return prior_boost, silence_factor
