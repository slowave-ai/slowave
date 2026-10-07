"""RetrievalService: multi-mechanism semantic recall and working-memory gating.

Previously implemented as methods on SlowaveEngine. Extracted so the retrieval
pipeline can be read, tested, and reasoned about independently.
"""

from __future__ import annotations

import dataclasses
import logging
import re
import time
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Any

import numpy as np

from slowave.core import pool_relative
from slowave.core.activation_selection import (
    ActivationTask,
    ContributionCandidate,
    ContributionDecision,
    SelectionMode,
    select_complementary,
    specific_terms,
)
from slowave.core.applicability_ranking import (
    ApplicabilityConfig,
    PairScorer,
    applicability_order,
    query_needs,
)
from slowave.core.config import DEFAULT_RECALL_TOP_K, MAX_PREVIEW_CHARS
from slowave.core.context import (
    GatePolicy,
    MemoryCue,
    WorkingMemoryGate,
    WorkingMemoryState,
    _distinctive_terms,
    _render,
    _requires_distinctive_match,
    _schema_terms,
    spread_relation_activation,
)
from slowave.core.retrieval_matching import (
    CandidateSignals,
    MatchingConfig,
    match_candidates,
)
from slowave.core.scope import normalize_scope
from slowave.core.shared_retrieval import SharedRequest, assess
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


@dataclass(frozen=True)
class RelevantCatalogItem:
    """One relevance-qualified declarative memory before response paging."""

    schema: Schema
    score: float
    reason: str
    need_indexes: tuple[int, ...]
    # True when the contribution decision found a qualifying span. The paging
    # layer derives the strong-run first page from this flag.
    strong: bool = False


@dataclass(frozen=True)
class RelevantCatalog:
    """Stable, bounded discovery result shared by activation and recall.

    ``truncated`` describes candidate discovery, never response paging.  A
    caller may page every item in ``items`` while still knowing that a channel
    search hit its operational depth.
    """

    items: list[RelevantCatalogItem]
    truncated: bool
    candidate_count: int
    decisions: tuple[ContributionDecision, ...] = ()
    applicability_status: str = "disabled"


# Minimum injected activation (recall()'s own schema_scores scale -- cosine +
# a fixed bonus per candidate source, roughly 0.15-1.25, not the 0-1 scale
# WorkingMemoryGate uses) for a schema_relations-propagated neighbor to be
# worth surfacing. A single hop from a typical FTS-level match (~0.35) already
# loses ~40% to confidence*decay, so this is set proportionately lower than
# the direct-hit floor rather than reusing it verbatim -- see
# spread_relation_activation's docstring for why decay makes deep/weak paths
# self-limiting without needing this floor to also do that work.
_RECALL_GRAPH_MIN_ACTIVATION = 0.15

# Maximum independently searched task cues per catalog. Bounds the per-call
# FTS/embedding search count and keeps the unioned candidate load comfortably
# inside SQLite's host-parameter limit.
_MAX_CATALOG_CUES = 16

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
# Retained as a compatibility argument for library callers.  Schema admission
# now uses ``MatchingConfig.dense_relevance_floor`` plus lexical evidence,
# rather than a score floor whose units depended on the candidate source.
_RECALL_MIN_RELEVANCE_DEFAULT = 0.20

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
_ACTIVATE_MIN_RELEVANCE_DEFAULT = 0.20

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
        applicability_config: ApplicabilityConfig | None = None,
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
        self._applicability_config = applicability_config or ApplicabilityConfig()
        self._applicability_scorer: PairScorer | None = None

    def _delivery_facets(self, schema: Any) -> dict[str, Any]:
        """Facets enriched with the delivery-relevant episode-derived marker.

        Consolidation-derived schemas carry no explicit source_kind but do
        carry supporting episode ids; delivery treats them as artifacts
        without source provenance when pool-relative admission is enabled.
        """
        facets = dict(schema.facets or {})
        if not (facets.get("source_kind") or facets.get("source")):
            episodes = getattr(schema, "supporting_episode_ids", None) or []
            if episodes:
                facets["episode_derived"] = True
        return facets

    def _rank_applicability(
        self,
        items: list[RelevantCatalogItem],
        query: str,
        *,
        protected_ids: set[int] | None = None,
        minimum_logit: float | None = None,
        maximum_logit_gap: float | None = None,
        retain_weak_support: bool = True,
    ) -> tuple[list[RelevantCatalogItem], str]:
        if not self._applicability_config.enabled or not items:
            return items, "disabled" if not self._applicability_config.enabled else "empty"
        # Synthetic/no-encoder callers retain their explicit lexical contracts.
        # Tests of this stage inject a scorer rather than loading model weights.
        if self._applicability_scorer is None:
            if not isinstance(self.encoder, TextEncoder):
                return items, "no_model_encoder"
            from slowave.symbolic.applicability_encoder import ApplicabilityEncoder

            self._applicability_scorer = ApplicabilityEncoder()
        try:
            needs = query_needs(query)
            scores = self._applicability_scorer.score(
                needs, [item.schema.content_text or "" for item in items]
            )
            if scores.shape != (len(items), len(needs)):
                raise ValueError("incorrect applicability score shape")
            order = applicability_order(
                scores,
                minimum_logit=(
                    self._applicability_config.minimum_logit
                    if minimum_logit is None
                    else minimum_logit
                ),
                maximum_logit_gap=(
                    self._applicability_config.maximum_logit_gap
                    if maximum_logit_gap is None
                    else maximum_logit_gap
                ),
                retain_weak_support=retain_weak_support,
            )
            if protected_ids:
                order.extend(
                    sorted(
                        (
                            i
                            for i, item in enumerate(items)
                            if item.schema.id in protected_ids and i not in order
                        ),
                        key=lambda i: (-float(scores[i].max()), i),
                    )
                )
        except (OSError, RuntimeError, ValueError, ImportError) as exc:
            logging.getLogger(__name__).warning("Applicability ranking unavailable: %s", exc)
            return items, "unavailable"
        return [
            replace(
                items[index],
                score=float(1 / (1 + np.exp(-np.clip(float(scores[index].max()), -60, 60)))),
                reason="query-applicability-v1",
            )
            for index in order
        ], "applied"

    def recall_source_spans(self, text: str, query: str) -> tuple[tuple[int, int], ...]:
        """Choose an intact answer passage after admission, without changing rank.

        Reuse the source windows shared assessment uses, including overlapping
        preceding context. Never sever an oversized sentence or invent a claim.
        """
        from slowave.core.shared_retrieval import source_windows

        if len(text) <= MAX_PREVIEW_CHARS:
            return ((0, len(text)),) if text else ()
        spans = source_windows(text)
        if not spans:
            raise ValueError("selected memory has no safely bounded source passage")
        passages = [text[start:end] for start, end in spans]
        if self._applicability_scorer is not None:
            needs = query_needs(query)
            scores = self._applicability_scorer.score(needs, passages)
            if scores.shape != (len(spans), len(needs)) or not np.isfinite(scores).all():
                raise ValueError("invalid recall source passage scores")
            best = min(range(len(spans)), key=lambda i: (-float(scores[i].max()), i))
        else:
            # No model encoder: keep the existing lexical retrieval contract,
            # but require actual query overlap rather than silently taking a prefix.
            terms = specific_terms(query)
            overlaps = [len(terms & specific_terms(passage)) for passage in passages]
            if not max(overlaps):
                raise ValueError("selected memory has no query-matched source passage")
            best = max(range(len(spans)), key=lambda i: (overlaps[i], -i))
        if spans[best][1] - spans[best][0] > MAX_PREVIEW_CHARS:
            raise ValueError("required recall source passage exceeds preview bound")
        return (spans[best],)

    # ---- public API --------------------------------------------------------

    def _shared_scorer(self) -> PairScorer:
        if self._applicability_scorer is None:
            from slowave.symbolic.applicability_encoder import ApplicabilityEncoder

            self._applicability_scorer = ApplicabilityEncoder()
        return self._applicability_scorer

    def _multilingual_index(self, *, scope: str | None, mode: str) -> dict[str, Any] | None:
        """Role-aware multilingual passage index over the eligible universe.

        Returns ``None`` (with a logged warning) when the local retrieval
        encoder or its assets are unavailable; discovery then falls back to
        the stored dense and lexical channels alone.
        """
        try:
            from slowave.symbolic.retrieval_encoder import get_retrieval_encoder

            encoder = get_retrieval_encoder()
        except (OSError, RuntimeError, ImportError, ValueError) as exc:
            logging.getLogger(__name__).warning(
                "Multilingual retrieval channel unavailable: %s", exc
            )
            return None
        scope_id = normalize_scope(scope=scope)
        universe: list = []
        for schema in self.schemas.list(limit=100000):
            if not (
                scope_id is None
                or schema.scope_id in (scope_id, None, "global", "user")
                or schema.generalization_stage >= 2
            ):
                continue
            admitted, _ = self.working_memory_gate.eligible(
                schema,
                cue=MemoryCue(query="discovery", scope=scope_id, mode=mode),
                policy=replace(GatePolicy.catalog_bound(1), allow_multi_sentence=True),
            )
            if admitted:
                universe.append(schema)
        if not universe:
            return None
        vectors = encoder.passages([schema.content_text or "" for schema in universe])
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / np.clip(norms, 1e-12, None)
        return {
            "encoder": encoder,
            "ids": np.array([schema.id for schema in universe], dtype=np.int64),
            "vectors": vectors,
        }

    def _baseline_catalog(
        self, request: SharedRequest, scope: str | None, mode: str
    ) -> RelevantCatalog:
        from slowave.core.retrieval_baseline import (
            BaselineConfig,
            feedback_adjustments,
            rank_candidates,
        )
        from slowave.core.shared_retrieval import source_windows
        from slowave.symbolic.procedural_memory import _contexts_compatible
        from slowave.symbolic.retrieval_encoder import get_retrieval_encoder

        if not hasattr(self, "_baseline_encoder"):
            self._baseline_encoder = get_retrieval_encoder()
        encoder = self._baseline_encoder
        config = (
            BaselineConfig()
            if request.endpoint == "activate"
            else BaselineConfig(semantic_floor=0.80, max_memories=3)
        )
        scope_id = normalize_scope(scope=scope)
        schemas = self.schemas.list(limit=100000)
        cue = MemoryCue(query=request.task, scope=scope_id, mode=mode)
        policy = replace(GatePolicy.catalog_bound(len(schemas)), allow_multi_sentence=True)
        schemas = [
            schema
            for schema in schemas
            if (
                scope_id is None
                or schema.scope_id in (scope_id, None, "global", "user")
                or schema.generalization_stage >= 2
            )
            and self.working_memory_gate.eligible(schema, cue=cue, policy=policy)[0]
            and _contexts_compatible(
                (schema.facets or {}).get("applicability_context"), request.context
            )
        ]
        if not schemas:
            return RelevantCatalog([], False, 0)
        vectors = encoder.passages([schema.content_text or "" for schema in schemas])
        adjustments = feedback_adjustments(
            self.schemas.db.connect(), request.task, request.goal, request.context, scope_id
        )
        channels = []
        for query in [request.task] + ([request.goal] if request.goal else []):
            scores = vectors @ encoder.query(query)
            lexical = {
                sid: rank
                for sid, _, rank, _ in self.schemas.search_fts_candidates(
                    query,
                    limit=100000,
                    scope_id=scope_id,
                    content_weight=config.bm25_content_weight,
                )
            }
            channels.append(
                [
                    dict(
                        id=schema.id,
                        semantic=float(scores[index]),
                        lexical_rank=lexical.get(schema.id),
                        explicit=(schema.facets or {}).get("source_kind") == "explicit_remember",
                        feedback_adjustment=adjustments.get(schema.id, 0.0),
                    )
                    for index, schema in enumerate(schemas)
                ]
            )
        ranked = rank_candidates(channels[0], channels[1] if len(channels) > 1 else [], config)
        by_id = {schema.id: schema for schema in schemas}
        decisions = []
        selected = []
        for score, sid in ranked:
            text = by_id[sid].content_text or ""
            span = source_windows(text)[0]
            if span[1] - span[0] > MAX_PREVIEW_CHARS:
                continue
            selected.append((score, sid))
            decisions.append(
                ContributionDecision(
                    sid,
                    True,
                    "baseline_relevance",
                    contribution=text[span[0] : span[1]],
                    source_spans=(span,),
                    strong=True,
                    selected_position=len(selected) - 1,
                )
            )
        return RelevantCatalog(
            [
                RelevantCatalogItem(by_id[sid], score, "baseline_relevance", (), strong=True)
                for score, sid in selected
            ],
            len(schemas) >= 100000,
            len(schemas),
            tuple(decisions),
            "applied",
        )

    def baseline_procedure_filter(
        self, request: SharedRequest, hits: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Existing procedure candidates must also match the current task/goal."""
        if not hits:
            return hits
        encoder = self._baseline_encoder
        vectors = encoder.passages(
            [str(hit.get("goal", "")) + " " + str(hit.get("summary", "")) for hit in hits]
        )
        task = vectors @ encoder.query(request.task)
        goal = vectors @ encoder.query(request.goal) if request.goal else task
        return [
            hit for index, hit in enumerate(hits) if 0.75 * task[index] + 0.25 * goal[index] >= 0.86
        ]

    def _hybrid_dogfood_catalog(self, task: str, scope: str | None, mode: str) -> RelevantCatalog:
        from slowave.core.hybrid_selection import DOGFOOD_CONFIG, admit, discover
        from slowave.core.shared_retrieval import source_windows

        config = DOGFOOD_CONFIG
        scope_id = normalize_scope(scope=scope)
        lexical = self.schemas.search_fts_candidates(
            task,
            limit=config.depth + 1,
            scope_id=scope_id,
            content_weight=config.bm25_content_weight,
        )
        dense = (
            self.schemas.search_embedding(
                self.encoder.encode(task), limit=config.depth + 1, scope_id=scope_id
            )
            if self.encoder is not None
            else []
        )
        truncated = len(lexical) > config.depth or len(dense) > config.depth
        ranks = {sid: rank for sid, _, rank, _ in lexical[: config.depth]}
        cosines = {sid: (rank, score) for rank, (sid, score) in enumerate(dense[: config.depth], 1)}
        schemas = {s.id: s for s in self.schemas.get_many(ranks.keys() | cosines.keys())}
        vector = self.encoder.encode(task) if self.encoder is not None else None
        cue = MemoryCue(query=task, scope=scope_id, mode=mode)
        policy = replace(GatePolicy.catalog_bound(len(schemas)), allow_multi_sentence=True)
        rows = []
        for sid, schema in schemas.items():
            eligible, _ = self.working_memory_gate.eligible(schema, cue=cue, policy=policy)
            cosine = (
                float(
                    vector.dot(schema.embedding)
                    / (np.linalg.norm(vector) * np.linalg.norm(schema.embedding) + 1e-12)
                )
                if vector is not None and schema.embedding is not None
                else None
            )
            rows.append(
                dict(
                    memory_id=sid,
                    dense_rank=cosines.get(sid, (None, None))[0],
                    lexical_rank=ranks.get(sid),
                    cosine=cosine,
                    rare_terms=0,
                    eligible=eligible,
                )
            )
        pool = discover([rows], config)
        qualified = admit(pool, config)
        selected = qualified[:1]
        if selected:
            start, end = source_windows(schemas[selected[0]].content_text or "")[0]
            if end - start > MAX_PREVIEW_CHARS:
                selected = []
                truncated = True
        decisions = []
        for row in rows:
            sid = row["memory_id"]
            chosen = sid in selected
            text = schemas[sid].content_text or ""
            span = source_windows(text)[0] if text else (0, 0)
            decisions.append(
                ContributionDecision(
                    sid,
                    chosen,
                    "hybrid_relevance" if chosen else "hybrid_not_delivered",
                    contribution=text[span[0] : span[1]] if chosen else "",
                    source_spans=(span,) if chosen else (),
                    strong=chosen,
                    selected_position=0 if chosen else None,
                )
            )
        return RelevantCatalog(
            items=[
                RelevantCatalogItem(
                    schemas[sid], pool[sid]["rrf"], "hybrid_relevance", (), strong=True
                )
                for sid in selected
            ],
            truncated=truncated or len(qualified) > 1,
            candidate_count=len(schemas),
            decisions=tuple(decisions),
            applicability_status="disabled",
        )

    def _shared_catalog(
        self,
        request: SharedRequest,
        schemas: dict[int, Schema],
        per_query: list[dict[int, Any]],
        scope: str | None,
        mode: str,
        truncated: bool,
        e5_evidence: dict[int, tuple[float, int]] | None = None,
    ) -> RelevantCatalog:
        """One shared absolute relevance decision, independent of wording rules."""
        from slowave.symbolic.procedural_memory import _contexts_compatible

        cue = MemoryCue(query=request.task, scope=scope, mode=mode)
        policy = replace(GatePolicy.catalog_bound(len(schemas)), allow_multi_sentence=True)
        eligible: list[Schema] = []
        decisions: list[ContributionDecision] = []
        for schema in sorted(schemas.values(), key=lambda schema: schema.id):
            admitted, reason = self.working_memory_gate.eligible(schema, cue=cue, policy=policy)
            stored_context = (schema.facets or {}).get("applicability_context")
            if admitted and not _contexts_compatible(stored_context, request.context):
                admitted, reason = False, "context_conflict"
            if not admitted:
                decisions.append(
                    ContributionDecision(schema.id, False, reason, channel_evidence=())
                )
            else:
                eligible.append(schema)
        scores, spans = assess(
            self._shared_scorer(), request, [schema.content_text or "" for schema in eligible]
        )
        candidates = []
        for schema, logit, span in zip(eligible, scores, spans):
            evidence = tuple(
                {
                    "query_index": index,
                    "dense_cosine": match.dense_cosine,
                    "lexical_rank": match.lexical_rank,
                    "rrf_score": match.normalized_rank_score,
                    "applicability_score": logit,
                    "threshold": request.threshold,
                }
                for index, query in enumerate(per_query)
                if (match := query.get(schema.id)) is not None
            ) + (
                (
                    {
                        "query_index": e5_evidence[schema.id][1],
                        "e5_cosine": e5_evidence[schema.id][0],
                    },
                )
                if e5_evidence and schema.id in e5_evidence
                else ()
            )
            rrf = max((e.get("rrf_score", 0.0) for e in evidence), default=0.0)
            selected = logit >= request.threshold
            reason = "shared_relevance" if selected else "insufficient_applicability"
            if selected and span[1] - span[0] > MAX_PREVIEW_CHARS:
                selected, reason = False, "oversize_without_contribution_span"
            decision = ContributionDecision(
                schema.id,
                selected,
                reason,
                contribution=(schema.content_text or "")[span[0] : span[1]],
                source_spans=(span,) if selected else (),
                channel_evidence=evidence,
                strong=selected,
            )
            decisions.append(decision)
            if selected:
                candidates.append((logit, rrf, schema))
        candidates.sort(key=lambda item: (-item[0], -item[1], item[2].id))
        positions = {schema.id: index for index, (_, _, schema) in enumerate(candidates)}
        return RelevantCatalog(
            items=[
                RelevantCatalogItem(
                    schema,
                    float(1 / (1 + np.exp(-np.clip(logit, -60, 60)))),
                    "shared_relevance",
                    (),
                    strong=True,
                )
                for logit, _, schema in candidates
            ],
            truncated=truncated,
            candidate_count=len(schemas),
            decisions=tuple(
                replace(decision, selected_position=positions.get(decision.memory_id))
                for decision in decisions
            ),
            applicability_status="applied",
        )

    def shared_procedures(
        self, request: SharedRequest, procedures: list[dict[str, Any]], limit: int = 3
    ) -> list[dict[str, Any]]:
        """Procedures earn relevance independently of English action-intent regexes.

        Reuse evidence reuses the existing single helped/harmed utility formula
        and only orders candidates that relevance admission already accepted, so
        historical reward can never admit an irrelevant procedure and harmed
        guidance yields to an equally applicable alternative.
        """
        from slowave.symbolic.procedural_memory import (
            _contexts_compatible,
            procedure_feedback_utility,
        )

        compatible = [
            procedure
            for procedure in procedures
            if _contexts_compatible(procedure.get("context"), request.context)
        ]
        scores, _ = assess(
            self._shared_scorer(),
            request,
            [
                "\n".join([str(procedure.get("goal", "")), str(procedure.get("summary", ""))])
                for procedure in compatible
            ],
        )
        ranked = [
            (score, procedure)
            for score, procedure in zip(scores, compatible)
            if score >= request.threshold
        ]
        ranked.sort(
            key=lambda pair: (
                -(pair[0] + procedure_feedback_utility(pair[1])),
                str(pair[1]["id"]),
            )
        )
        return [
            dict(procedure, score=score, match={"admission": "shared_relevance", "logit": score})
            for score, procedure in ranked[:limit]
        ]

    def refresh_indices(self) -> None:
        """Rebuild in-memory FAISS indices from SQLite."""
        self.episodic.reset_faiss_from_db()
        self.semantic.reset_faiss_from_db()

    def relevant_catalog(
        self,
        cues: list[str],
        *,
        scope: str | None,
        mode: str = "strict_scope",
        candidate_limit: int = 256,
        min_relevance: float = _ACTIVATE_MIN_RELEVANCE_DEFAULT,
        focus_single_need: bool = True,
        selection_mode: SelectionMode | None = None,
        activation_task: ActivationTask | None = None,
        shared_request: SharedRequest | None = None,
        hybrid_task: str | None = None,
        baseline_request: SharedRequest | None = None,
    ) -> RelevantCatalog:
        """Build one declarative catalog before item and response budgets.

        Each cue is searched independently so a sparse explicit task need can
        recover a memory that a long whole-task cue would bury. IDs are then
        unioned, loaded once, passed through the existing evidence matcher and
        eligibility gate, and ordered deterministically.
        """
        from slowave.core.hybrid_selection import dogfood_enabled

        if baseline_request is not None:
            return self._baseline_catalog(baseline_request, scope, mode)
        if activation_task is not None and dogfood_enabled():
            return self._hybrid_dogfood_catalog(
                hybrid_task or activation_task.needs[0].text, scope, mode
            )
        e5_channel = None
        if shared_request is not None:
            clean_queries = shared_request.queries
            # Validate before discovery or state publication.
            shared_request.threshold
            candidate_limit = shared_request.candidate_depth
            cues = clean_queries
            activation_task = None
            selection_mode = "deliberate_recall"
            focus_single_need = False
            # Multilingual discovery channel: the stored dense index uses the
            # production encoder, which can miss cross-language evidence. The
            # shared pipeline therefore unions a role-aware multilingual
            # channel per query; failure degrades to the existing channels
            # with a visible warning instead of silently narrowing recall.
            e5_channel = self._multilingual_index(scope=scope, mode=mode)
        else:
            # The multilingual channel joins the discovery union so
            # cross-language evidence is structural, not threshold luck.
            # Failure degrades to the existing channels with the same visible
            # warning; task representation and selection mode are untouched.
            e5_channel = self._multilingual_index(scope=scope, mode=mode)
        resolved_mode = selection_mode or (
            "legacy_activation" if focus_single_need else "deliberate_recall"
        )
        if resolved_mode not in {
            "legacy_activation",
            "deliberate_recall",
            "complementary_activation",
        }:
            raise ValueError("unsupported declarative selection_mode")
        if resolved_mode == "complementary_activation" and activation_task is None:
            raise ValueError("complementary activation requires an explicit ActivationTask")
        clean_cues = list(dict.fromkeys(cue.strip() for cue in cues if cue and cue.strip()))
        if (
            activation_task is not None
            and resolved_mode == "complementary_activation"
            and clean_cues != activation_task.cues
        ):
            raise ValueError("candidate cues must match the activation task representation")
        truncated = False
        if len(clean_cues) > _MAX_CATALOG_CUES:
            # Bounded fan-out: each cue runs lexical and dense searches, and the
            # union feeds one chunked load; report honestly when the cap bit.
            # The task representation itself stays whole, so contribution
            # assessment is unaffected; needs beyond the cap are covered by the
            # whole-task cue instead of independent searches.
            truncated = True
            clean_cues = clean_cues[:_MAX_CATALOG_CUES]
        if not clean_cues:
            return RelevantCatalog(items=[], truncated=False, candidate_count=0)
        if candidate_limit < 1:
            raise ValueError("candidate_limit must be positive")

        scope_id = normalize_scope(scope=scope)
        cue_terms = [set(self.schemas.lexical_tokens(cue)) for cue in clean_cues]
        term_frequency: dict[str, int] = {}
        # The whole-task cue necessarily repeats terms from every explicit
        # item, so it must not make each item term look non-distinctive.
        frequency_terms = cue_terms[1:] if len(cue_terms) > 1 else cue_terms
        for terms in frequency_terms:
            for term in terms:
                term_frequency[term] = term_frequency.get(term, 0) + 1
        distinctive_terms = [
            {term for term in terms if term_frequency.get(term, 0) == 1} for terms in cue_terms
        ]
        per_need: list[dict[int, Any]] = []
        candidate_ids: set[int] = set()
        fetch_limit = candidate_limit + 1
        e5_evidence: dict[int, tuple[float, int]] = {}
        for cue_index, cue_text in enumerate(clean_cues):
            signals: dict[int, CandidateSignals] = {}
            lexical = self.schemas.search_fts_candidates(
                cue_text, limit=fetch_limit, scope_id=scope_id
            )
            if len(lexical) > candidate_limit:
                truncated = True
                lexical = lexical[:candidate_limit]
            for sid, bm25, rank, tokens in lexical:
                signals[sid] = CandidateSignals(
                    memory_id=sid,
                    lexical_score=bm25,
                    lexical_rank=rank,
                    lexical_specific=bool(tokens),
                )
            if self.encoder is not None:
                dense = self.schemas.search_embedding(
                    self.encoder.encode(cue_text), limit=fetch_limit, scope_id=scope_id
                )
                if len(dense) > candidate_limit:
                    truncated = True
                    dense = dense[:candidate_limit]
                for rank, (sid, score) in enumerate(dense, 1):
                    previous = signals.get(sid, CandidateSignals(memory_id=sid))
                    signals[sid] = CandidateSignals(
                        memory_id=sid,
                        dense_cosine=score,
                        dense_rank=rank,
                        lexical_score=previous.lexical_score,
                        lexical_rank=previous.lexical_rank,
                        lexical_specific=previous.lexical_specific,
                    )
            fused = match_candidates(
                signals.values(),
                config=MatchingConfig(dense_relevance_floor=min_relevance),
            )
            if e5_channel is not None:
                query_vector = np.asarray(e5_channel["encoder"].query(cue_text), dtype=np.float32)
                scores = e5_channel["vectors"] @ query_vector
                if len(scores) > candidate_limit:
                    # Truthful exhaustion: the slice cut a deeper candidate
                    # pool, so paging callers must be able to see it.
                    truncated = True
                order = np.argsort(-scores)[:candidate_limit]
                e5_ids: set[int] = set()
                for index in order:
                    sid = int(e5_channel["ids"][index])
                    score = float(scores[index])
                    e5_ids.add(sid)
                    if sid not in e5_evidence or score > e5_evidence[sid][0]:
                        e5_evidence[sid] = (score, cue_index)
                    if sid not in signals:
                        signals[sid] = CandidateSignals(memory_id=sid)
            if shared_request is not None and len(fused) > candidate_limit:
                truncated = True
                fused = fused[:candidate_limit]
            per_need.append({match.memory_id: match for match in fused})
            candidate_ids.update(per_need[-1])
            if e5_channel is not None:
                candidate_ids.update(e5_ids)

        schemas = {schema.id: schema for schema in self.schemas.get_many(candidate_ids)}
        if shared_request is not None:
            return self._shared_catalog(
                shared_request,
                schemas,
                per_need,
                scope_id,
                mode,
                truncated,
                e5_evidence=e5_evidence,
            )
        cue = MemoryCue(query=clean_cues[0], scope=scope_id, mode=mode)
        policy = GatePolicy.catalog_bound(len(schemas))
        items: list[RelevantCatalogItem] = []
        has_explicit_needs = len(clean_cues) > 1
        seen_content: set[str] = set()
        contribution_candidates: list[ContributionCandidate] = []
        for schema_id, schema in schemas.items():
            eligible, eligibility_reason = self.working_memory_gate.eligible(
                schema, cue=cue, policy=policy
            )
            matches = [need.get(schema_id) for need in per_need]
            schema_terms = set(self.schemas.lexical_tokens(schema.content_text or ""))
            covered = tuple(
                index
                for index, match in enumerate(matches)
                if match is not None
                and match.relevance_passed
                and (
                    index == 0
                    or not distinctive_terms[index]
                    or bool(distinctive_terms[index] & schema_terms)
                )
            )
            if resolved_mode == "complementary_activation":
                evidence = tuple(
                    {
                        "need_index": index,
                        "dense_cosine": match.dense_cosine,
                        "dense_rank": match.dense_rank,
                        "lexical_rank": match.lexical_rank,
                        "lexical_score": match.lexical_score,
                        "relevance_passed": match.relevance_passed,
                        "relevance_reason": match.relevance_reason,
                    }
                    for index, match in enumerate(matches)
                    if match is not None
                )
                best_score = max(
                    (matches[index].normalized_rank_score for index in covered), default=0.0
                )
                contribution_candidates.append(
                    ContributionCandidate(
                        memory_id=schema_id,
                        text=schema.content_text or "",
                        score=best_score,
                        eligible=eligible,
                        eligibility_reason=eligibility_reason,
                        need_indexes=covered,
                        channel_evidence=evidence,
                        facets=self._delivery_facets(schema),
                    )
                )
                continue
            if not eligible or not covered:
                continue
            # An explicit task list is a stronger declaration of independent
            # needs than the broad whole-task cue. Do not let an adjacent fact
            # through merely because it weakly matches the task introduction.
            if has_explicit_needs and not any(index > 0 for index in covered):
                continue
            content_key = " ".join((schema.content_text or "").casefold().split())
            if not content_key or content_key in seen_content:
                continue
            seen_content.add(content_key)
            best = max(
                (matches[index] for index in covered),
                key=lambda match: (match.normalized_rank_score, -match.memory_id),
            )
            items.append(
                RelevantCatalogItem(
                    schema=schema,
                    score=best.normalized_rank_score,
                    reason=f"{best.scoring_policy_version}:{best.relevance_reason}",
                    need_indexes=covered,
                )
            )
        if resolved_mode == "complementary_activation":
            if activation_task is None:
                raise ValueError("complementary activation requires an explicit ActivationTask")
            branch_pattern = r"\b(?:fix|feat|feature|hotfix|bugfix)/[\w.-]+"
            task_branches = set(re.findall(branch_pattern, "\n".join(activation_task.cues)))
            context_excluded = {}
            if task_branches:
                for candidate in contribution_candidates:
                    branches = set(re.findall(branch_pattern, candidate.text))
                    if (
                        candidate.eligible
                        and branches
                        and not branches & task_branches
                        and candidate.facets.get("schema_class")
                        not in {"warning", "lesson", "instruction", "constraint", "preference"}
                    ):
                        context_excluded[candidate.memory_id] = "structured_facet_conflict"
            # Explicit numbered targets are task facets, not merely topic words.
            # A hop-10 observation does not answer a hop-9 failure question.
            target_pattern = r"\b(hop|partition|shard)\s*[-#:]?\s*(\d+)\b"
            targets = {}
            for kind, value in re.findall(target_pattern, "\n".join(activation_task.cues), re.I):
                targets.setdefault(kind.lower(), set()).add(value)
            for candidate in contribution_candidates:
                candidate_targets = {}
                for kind, value in re.findall(target_pattern, candidate.text, re.I):
                    candidate_targets.setdefault(kind.lower(), set()).add(value)
                if candidate.eligible and any(
                    kind in candidate_targets and not values & candidate_targets[kind]
                    for kind, values in targets.items()
                ):
                    context_excluded[candidate.memory_id] = "structured_facet_conflict"
            # Source-independent action logs such as "Read the README" do
            # not convey a stored fact. Explicitly remembered knowledge and
            # branch-specific state remain assessable.
            for candidate in contribution_candidates:
                if (
                    candidate.eligible
                    and (candidate.facets.get("source_kind") or candidate.facets.get("source"))
                    != "explicit_remember"
                    and re.match(r"\s*(?:Read|Inspected|Reviewed)\b", candidate.text)
                    and len(re.findall(r".+?(?:[.!?](?=\s|$)|\n|$)", candidate.text)) == 1
                    and not re.search(
                        r"\b(?:found|confirmed|requires|must|because|failed|passed)\b",
                        candidate.text,
                        re.I,
                    )
                ):
                    context_excluded[candidate.memory_id] = "insufficient_evidence"
            for candidate in contribution_candidates:
                if candidate.eligible and candidate.facets.get("episode_derived"):
                    context_excluded[candidate.memory_id] = "episode_artifact_without_source"
            # Assess the eligible discovery pool before contribution filtering.
            # Goals can discover candidates, but only user/context cues justify exposure.
            user_indexes = [
                i
                for i, need in enumerate(activation_task.needs)
                if any(
                    source in need.provenance
                    for source in ("task", "explicit_list_item", "semantic_context", "task_context")
                )
                and specific_terms(need.text) - specific_terms((scope_id or "").split(":", 1)[-1])
            ]
            pool = [
                RelevantCatalogItem(
                    schema=schemas[c.memory_id],
                    score=c.score,
                    reason="discovered",
                    need_indexes=c.need_indexes,
                )
                for c in contribution_candidates
                if c.eligible
                and c.memory_id not in context_excluded
                and any(e.get("relevance_passed") for e in c.channel_evidence)
            ]
            # Pool-relative assessment: the per-column floor becomes
            # max(signal_floor, column_best - margin); abstention is
            # preserved by the signal floor.
            assessment_floor = pool_relative.signal_floor()
            assessment_gap: float | None = pool_relative.margin()
            ranked_pool, applicability_status = (
                self._rank_applicability(
                    pool,
                    "\n".join(activation_task.needs[i].text for i in user_indexes),
                    minimum_logit=assessment_floor,
                    maximum_logit_gap=assessment_gap,
                    retain_weak_support=False,
                )
                if user_indexes
                else ([], "empty")
            )
            applicable_ids = {item.schema.id for item in ranked_pool}
            assessed = applicability_status == "applied"
            contribution_floor = pool_relative.signal_floor()
            if assessed:
                updated = []
                for candidate in contribution_candidates:
                    if candidate.memory_id not in applicable_ids:
                        updated.append(candidate)
                        continue
                    text = candidate.text
                    # Short sources are cheap enough to preserve intact. Long
                    # sources need an exact contiguous source excerpt, assessed
                    # against the same user cues rather than a prefix truncation.
                    spans = [(0, len(text))]
                    if len(text) > MAX_PREVIEW_CHARS:
                        spans = []
                        sentences = list(re.finditer(r".+?(?:[.!?](?=\s|$)|\n|$)", text))
                        for index, match in enumerate(sentences):
                            start, end = match.start(), match.end()
                            if index and re.match(
                                r"\s*(?:For|If|When|Unless|Before|After)\b",
                                sentences[index - 1].group(),
                                re.I,
                            ):
                                start = sentences[index - 1].start()
                            for following in sentences[index + 1 :]:
                                if following.end() - start > MAX_PREVIEW_CHARS:
                                    break
                                end = following.end()
                            if end - start <= MAX_PREVIEW_CHARS:
                                spans.append((start, end))
                    if not spans:
                        updated.append(candidate)
                        continue
                    if len(text) <= MAX_PREVIEW_CHARS:
                        span_index, column = 0, 0
                        contribution_score = next(
                            item.score
                            for item in ranked_pool
                            if item.schema.id == candidate.memory_id
                        )
                    else:
                        try:
                            if self._applicability_scorer is None:
                                raise RuntimeError("applicability scorer unavailable")
                            scores = self._applicability_scorer.score(
                                [activation_task.needs[i].text for i in user_indexes],
                                [text[start:end] for start, end in spans],
                            )
                            if (
                                scores.shape != (len(spans), len(user_indexes))
                                or not np.isfinite(scores).all()
                            ):
                                raise ValueError("invalid contribution scores")
                            span_index, column = min(
                                (
                                    (i, j)
                                    for i in range(len(spans))
                                    for j in range(len(user_indexes))
                                ),
                                key=lambda pair: (
                                    -float(scores[pair]),
                                    spans[pair[0]][1] - spans[pair[0]][0],
                                    pair,
                                ),
                            )
                            contribution_score = float(scores[span_index, column])
                            if contribution_score < contribution_floor:
                                updated.append(candidate)
                                continue
                        except (OSError, RuntimeError, ValueError, ImportError) as exc:
                            logging.getLogger(__name__).warning(
                                "Contribution scoring unavailable: %s", exc
                            )
                            updated.append(candidate)
                            applicability_status = "unavailable"
                            continue
                    updated.append(
                        replace(
                            candidate,
                            channel_evidence=candidate.channel_evidence
                            + (
                                {
                                    "need_index": user_indexes[column],
                                    "applicability_passed": True,
                                    "applicability_score": contribution_score,
                                    "contribution_span": spans[span_index],
                                },
                            ),
                        )
                    )
                contribution_candidates = updated
            selection = select_complementary(
                activation_task, contribution_candidates, scope=scope_id
            )
            by_id = {candidate.memory_id: candidate for candidate in contribution_candidates}
            decisions = {
                decision.memory_id: (
                    replace(decision, selected=False, reason=context_excluded[decision.memory_id])
                    if decision.memory_id in context_excluded
                    else decision
                )
                for decision in selection.decisions
            }
            # Discovery is broad; only assessed task contributions are exposed.
            delivered: list[int] = []
            transport_excluded: dict[int, str] = {}
            for sid, candidate in by_id.items():
                decision = decisions[sid]
                if not candidate.eligible:
                    continue
                if not decision.selected:
                    continue
                if len(candidate.text) > MAX_PREVIEW_CHARS and not decision.source_spans:
                    # Transport exclusion: unrenderable without a validated
                    # contribution span; recorded in the decision trace.
                    transport_excluded[sid] = "oversize_without_contribution_span"
                    continue
                delivered.append(sid)
            ordered_ids = sorted(
                delivered,
                key=lambda sid: (-decisions[sid].shared_terms, -by_id[sid].score, sid),
            )
            delivered_set = set(ordered_ids)
            catalog_decisions = tuple(
                replace(
                    decisions[candidate.memory_id],
                    selected=candidate.memory_id in delivered_set,
                    reason=transport_excluded.get(
                        candidate.memory_id, decisions[candidate.memory_id].reason
                    ),
                )
                for candidate in contribution_candidates
            )
            delivery_query = "\n".join(activation_task.needs[i].text for i in user_indexes)
            delivery_floor: float | None = pool_relative.signal_floor()
            delivery_gap: float | None = pool_relative.margin()
            ranked_items, final_applicability_status = self._rank_applicability(
                [
                    RelevantCatalogItem(
                        schema=schemas[sid],
                        score=by_id[sid].score,
                        reason=decisions[sid].reason,
                        need_indexes=by_id[sid].need_indexes,
                        strong=decisions[sid].strong,
                    )
                    for sid in ordered_ids
                ],
                delivery_query,
                minimum_logit=delivery_floor,
                maximum_logit_gap=delivery_gap,
                retain_weak_support=False,
                protected_ids={
                    decision.memory_id
                    for decision in catalog_decisions
                    if decision.selected
                    and (
                        re.match(r"\s*For\s+[^,;:.]+[,;:]", decision.contribution or "", re.I)
                        or (
                            decision.supporting_need is not None
                            and "explicit_list_item"
                            in activation_task.needs[decision.supporting_need].provenance
                            and decision.shared_terms >= 2
                        )
                    )
                },
            )
            if applicability_status != "unavailable" and final_applicability_status != "empty":
                applicability_status = final_applicability_status
            # The applicability stage changes final exposure, not discovery.
            admitted = {item.schema.id for item in ranked_items}
            positions = {item.schema.id: position for position, item in enumerate(ranked_items)}
            catalog_decisions = tuple(
                replace(
                    decision,
                    selected=decision.memory_id in admitted,
                    reason=(
                        "insufficient_applicability"
                        if decision.selected and decision.memory_id not in admitted
                        else decision.reason
                    ),
                    selected_position=positions.get(decision.memory_id),
                )
                for decision in catalog_decisions
            )
            return RelevantCatalog(
                items=ranked_items,
                truncated=truncated,
                candidate_count=len(candidate_ids),
                decisions=catalog_decisions,
                applicability_status=applicability_status,
            )
        items.sort(key=lambda item: (-item.score, item.schema.id))
        if has_explicit_needs:
            # Greedy coverage is the first deterministic marginal-value rule:
            # a candidate must earn an uncovered user-declared need.  This
            # prevents a broad deployment-review fact from consuming a page
            # beside the rollback/configuration/validation answers. A later
            # calibrated distinct-contribution rule may add a second claim to
            # a covered need; it must prove more than adjacent lexical overlap.
            covered_needs: set[int] = set()
            selected: list[RelevantCatalogItem] = []
            for item in items:
                novel = set(item.need_indexes) - {0} - covered_needs
                if not novel:
                    continue
                selected.append(item)
                covered_needs.update(novel)
            items = selected
        # A single undivided request has one declared need. Until a candidate
        # demonstrates a distinct contribution rule, retain the established
        # focus behavior rather than treating rank two as evidence of a second
        # answer. Explicit task needs have no such ceiling.
        if resolved_mode == "legacy_activation" and not has_explicit_needs:
            items = items[:1]
        applicability_status = "not_requested"
        decisions: tuple[ContributionDecision, ...] = ()
        if resolved_mode == "deliberate_recall":
            before = items
            items, applicability_status = self._rank_applicability(
                items,
                clean_cues[0],
                minimum_logit=pool_relative.signal_floor(),
                maximum_logit_gap=pool_relative.margin(),
            )
            positions = {item.schema.id: position for position, item in enumerate(items)}
            decisions = tuple(
                ContributionDecision(
                    memory_id=item.schema.id,
                    selected=item.schema.id in positions,
                    reason=(
                        "discovery_ordering"
                        if applicability_status != "applied"
                        else (
                            "query_applicability"
                            if item.schema.id in positions
                            else "insufficient_applicability"
                        )
                    ),
                    selected_position=positions.get(item.schema.id),
                )
                for item in before
            )
        return RelevantCatalog(
            items=items,
            truncated=truncated,
            candidate_count=len(candidate_ids),
            applicability_status=applicability_status,
            decisions=decisions,
        )

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

        depth = max(20, top_k * 8)
        dense = self.schemas.search_embedding(q, limit=depth, scope_id=scope_id)
        lexical = self.schemas.search_fts_candidates(query, limit=depth, scope_id=scope_id)
        signals: dict[int, CandidateSignals] = {}
        for rank, (sid, cosine) in enumerate(dense, 1):
            signals[sid] = CandidateSignals(memory_id=sid, dense_cosine=cosine, dense_rank=rank)
        for sid, bm25, rank, tokens in lexical:
            prior = signals.get(sid, CandidateSignals(memory_id=sid))
            signals[sid] = CandidateSignals(
                memory_id=sid,
                dense_cosine=prior.dense_cosine,
                dense_rank=prior.dense_rank,
                lexical_score=bm25,
                lexical_rank=rank,
                lexical_specific=bool(tokens),
            )
        schemas_all = self.schemas.get_many(signals.keys())
        by_id = {schema.id: schema for schema in schemas_all}
        signals = {
            sid: CandidateSignals(**{**signal.__dict__, "salience": by_id[sid].salience})
            for sid, signal in signals.items()
            if sid in by_id
        }
        matches = {
            item.memory_id: item
            for item in match_candidates(
                signals.values(), config=MatchingConfig(dense_relevance_floor=min_relevance)
            )
        }
        # Raw cosine is passed only to consumers that explicitly need cosine;
        # fusion scores are never smuggled through the old semantic field.
        schema_scores: dict[int, float] = {
            sid: signal.dense_cosine or 0.0 for sid, signal in signals.items()
        }
        distinctive_terms = (
            _distinctive_terms(query) if _requires_distinctive_match(query) else set()
        )

        # Mode-gated status filter; when strict_scope, also enforce scope.
        if mode == "debug":
            recall_statuses = ("active", "needs_review", "stale")
        elif mode == "broad":
            recall_statuses = ("active", "needs_review")
        else:  # default, strict_scope
            recall_statuses = ("active",)

        filtered_schemas = []

        for s in schemas_all:
            if s.status not in recall_statuses:
                continue

            if distinctive_terms and not (distinctive_terms & _schema_terms(s)):
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
            match = matches.get(s.id)
            if match is not None:
                return match.final_rank_score
            # Graph-only neighbors have no direct lexical/dense rank. Their
            # propagated activation is the only meaningful ranking signal.
            return schema_scores.get(s.id, 0.0)

        schemas = [s for s in filtered_schemas if matches[s.id].relevance_passed]
        schemas.sort(key=lambda s: (-_rank_score(s), s.id))
        schemas = schemas[:top_k]

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
                {s.id: matches[s.id].normalized_rank_score for s in schemas},
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
        # The normal response avoids repeating a schema as an episode, but a
        # caller who requested evidence must still receive the ranked episode
        # and its source time. Otherwise temporal selection is invisible to
        # the public v9 contract whenever a schema has the same text.
        seen_episodes: set[str] = set()
        schema_texts = {
            _normalize_episode_text(s.content_text or "") for s in schemas if s.content_text
        }

        for _score, m in scored_pairs[:top_k]:
            ep = ep_by_id.get(m.id)
            raw_text = ep.content_text if ep else ""
            occurred_at = int(m.metadata.get("occurred_at", m.ts))
            dated_text = _prefix_date(raw_text, occurred_at)

            # Skip if this episode content was already emitted (normalised).
            normalized = _normalize_episode_text(raw_text)
            if normalized and normalized in seen_episodes:
                continue

            # Skip if episode duplicates a schema already surfaced in results.
            if normalized and normalized in schema_texts and not evidence:
                continue

            if normalized:
                seen_episodes.add(normalized)

            episode_dicts.append(
                {
                    "id": m.id,
                    "content_text": dated_text,
                    "salience": float(m.salience),
                    "ts": int(m.ts),
                    "recorded_at": int(m.ts),
                    "occurred_at": occurred_at,
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
                    {
                        "id": e.id,
                        "ts": e.ts,
                        "recorded_at": e.ts,
                        "occurred_at": int(e.metadata.get("occurred_at", e.ts)),
                        "type": e.type,
                        "content": e.content,
                    }
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
        semantic_context: str | None = None,
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

        # Computed early (pure string join, no side effects) so the candidate
        # lexical/dense fetch below can size itself to how much signal the cue
        # carries. Salience-only enumeration is unnecessary: select_matched()
        # requires lexical or dense evidence for every displayed memory.
        cue_text = " ".join(
            [
                query or "",
                semantic_context or "",
                application or "",
                goal or "",
                task_type or "",
                " ".join(f"{k} {v}" for k, v in sorted((situation or {}).items())),
                " ".join(requirements or []),
                " ".join(topics or []),
                " ".join(entities or []),
            ]
        ).strip()
        trivial_cue = len(cue_text) < _TRIVIAL_CUE_MIN_CHARS

        cue_embedding = None
        match_signals: dict[int, CandidateSignals] = {}
        if cue_text:
            cue_fetch_limit = max(10, limit) if trivial_cue else max(50, limit * 8)
            # Scope is an eligibility boundary, never text embedded in the
            # semantic query.  The store applies it before lexical truncation.
            for sid, bm25, rank, tokens in self.schemas.search_fts_candidates(
                cue_text, limit=cue_fetch_limit, scope_id=scope_id
            ):
                match_signals[sid] = CandidateSignals(
                    memory_id=sid,
                    lexical_score=bm25,
                    lexical_rank=rank,
                    lexical_specific=bool(tokens),
                )
            if self.encoder is not None:
                cue_embedding = self.encoder.encode(cue_text)
                for rank, (sid, score) in enumerate(
                    self.schemas.search_embedding(
                        cue_embedding, limit=cue_fetch_limit, scope_id=scope_id
                    ),
                    1,
                ):
                    previous = match_signals.get(sid, CandidateSignals(memory_id=sid))
                    match_signals[sid] = CandidateSignals(
                        memory_id=sid,
                        dense_cosine=score,
                        dense_rank=rank,
                        lexical_score=previous.lexical_score,
                        lexical_rank=previous.lexical_rank,
                        lexical_specific=previous.lexical_specific,
                    )

        candidates_by_id = {schema.id: schema for schema in self.schemas.get_many(match_signals)}

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
            require_explicit_multi_answer=limit == 2 and not include_peripheral,
        )
        match_signals = {
            sid: CandidateSignals(**{**signal.__dict__, "salience": candidates_by_id[sid].salience})
            for sid, signal in match_signals.items()
            if sid in candidates_by_id
        }
        matches = {
            item.memory_id: item
            for item in match_candidates(
                match_signals.values(), config=MatchingConfig(dense_relevance_floor=min_relevance)
            )
        }
        state = self.working_memory_gate.select_matched(
            candidates_by_id.values(), matches=matches, cue=cue, policy=policy
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
                elif status == "stale":
                    age = max(0.0, float(now_ts - int(last_ts)))
                    fresh = 0.5 ** (age / silence_halflife_s)
                    damp = 0.6 * fresh * float(conf)
                    factor = max(0.05, 1.0 - damp)
                    silence_factor[eid] = min(silence_factor.get(eid, 1.0), factor)
        return prior_boost, silence_factor
