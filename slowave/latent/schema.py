"""Stage 6 — latent schemas.

Brain analogue: cortical schema formation. The neocortex does not store
schemas as sentences; it stores them as activation patterns across
populations of neurons. A schema, in this implementation, is the
geometric fingerprint of a prototype:

  * centroid       : where the cluster lives in embedding space
  * facet_axes     : top principal directions of within-cluster spread
                     (= what dimensions the cluster's members differ on)
  * temporal_anchor: (mean_ts, span_ts) — when this concept is "alive"
  * member_episode_ids
  * confidence     : tightness of the cluster (1 - normalised variance)
  * salience       : reinforced by recall; decays with time

The crucial property: **no LLM call**. Schema formation is a pure
geometric operation over the prototype and its member episodes.

Schemas still hold a *text handle* — the most-central member episode's
content text — because external callers (the eval harness, MCP API,
human inspection) need something readable. The text is *derived* from
the latent state, not the substrate of it. If we later add Stage 11
(verbalisation at the boundary), the LLM can rewrite this handle into
a nicer sentence; until then, the central member's text is what the
schema "says".

Contradiction detection (Stage 6b) is also implemented here as a
pure geometric operation: two schemas conflict when their centroids
are close (same topic) but a measurable facet axis differs and the
newer one has higher confidence.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from slowave.core.judge_debug import emit_judge_signal
from slowave.symbolic.episode_text import EpisodeText

# ---- Data types ----------------------------------------------------------


@dataclass(frozen=True)
class LatentSchema:
    """The geometric fingerprint of a consolidated concept.

    Replaces `ExtractedSchema` (LLM-derived) for the brain-only
    architecture. Compatible enough with the existing
    ``Consolidator._create_and_relate_schema`` flow that the
    integration patch is small.
    """

    # Geometry
    centroid: np.ndarray  # mean embedding of member episodes
    facet_axes: np.ndarray  # top-k principal directions (k x dim)
    facet_strengths: np.ndarray  # variance explained per axis (k,)

    # Provenance
    member_episode_ids: list[int]
    central_episode_id: int  # closest member to centroid
    central_episode_text: str  # human-readable text of that member

    # Temporal
    mean_ts: int
    ts_span_s: int  # max - min of member timestamps

    # Statistics
    confidence: float  # 1.0 - normalised within-cluster variance
    support_count: int  # how many episodes back this schema

    # Lexical abstraction (Stage 7a) — contrastive TF-IDF over cluster texts.
    # A dict of {term: score} where score is the within-cluster term frequency
    # weighted by cluster distinctiveness vs the rest of the corpus.
    # Empty when the cluster has too few texts to compute.
    lexical_signature: dict = field(default_factory=dict)
    # Human-readable label derived from top lexical terms.
    # Example: "faiss / sqlite / local"  — deterministic, no LLM.
    display_label: str = ""

    # Hooks for the existing Consolidator API
    tags: list[str] = field(default_factory=list)
    facets: dict = field(default_factory=dict)
    evidence_indices: list[int] = field(default_factory=list)
    evidence_quote: Optional[str] = None

    # Aliases / adapters so the existing consolidation code path that
    # writes schemas into the DB doesn't have to know whether it was
    # produced by the LLM or by the latent builder.
    @property
    def claim(self) -> str:
        return self.central_episode_text


@dataclass(frozen=True)
class GeometricVerdict:
    """Result of a geometric contradiction comparison.

    Mirrors the shape of ``ContradictionJudge.judge``'s ``JudgeResult``
    so the Consolidator can route either backend through the same code
    path.
    """

    verdict: str  # 'unrelated' | 'relates_to'
    reasoning: str
    similarity: float
    facet_distance: float
    time_delta_s: int


# ---------------------------------------------------------------------------
# Lexical signature helpers (Stage 7a)
# ---------------------------------------------------------------------------

# Minimal English stopword list — no external dependency.
_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "with",
        "by",
        "from",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "shall",
        "can",
        "need",
        "dare",
        "it",
        "its",
        "this",
        "that",
        "these",
        "those",
        "i",
        "we",
        "you",
        "he",
        "she",
        "they",
        "me",
        "us",
        "him",
        "her",
        "them",
        "my",
        "our",
        "your",
        "his",
        "their",
        "what",
        "which",
        "who",
        "when",
        "where",
        "why",
        "how",
        "all",
        "each",
        "every",
        "both",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "no",
        "not",
        "only",
        "own",
        "same",
        "so",
        "than",
        "too",
        "very",
        "just",
        "user",
        "assistant",
        "remember",
    }
)


def _tokenize(text: str) -> list[str]:
    """Lowercase alphabetic tokens, length >= 3, not stopwords."""
    return [w for w in re.split(r"[^a-z]+", text.lower()) if len(w) >= 3 and w not in _STOPWORDS]


def _build_lexical_signature(
    cluster_texts: list[str],
    corpus_texts: list[str],
    top_n: int = 8,
) -> dict[str, float]:
    """Contrastive TF-IDF: terms that are common *within* this cluster
    but distinctive *against* the full episode corpus.

    Score formula::

        score(term) = tf_cluster(term) * log(1 + corpus_df / (1 + cluster_df))

    where ``tf_cluster`` is the normalised frequency inside the cluster and
    ``cluster_df`` / ``corpus_df`` are document frequencies (how many docs
    contain the term) in cluster vs corpus.  Contrastive scaling suppresses
    generic words that appear everywhere.

    Returns a dict {term: score} of at most ``top_n`` terms, sorted
    descending by score.
    """
    import math

    if not cluster_texts:
        return {}

    # Term frequencies within the cluster
    cluster_tf: dict[str, int] = {}
    cluster_df: dict[str, int] = {}
    for text in cluster_texts:
        tokens = _tokenize(text)
        for tok in tokens:
            cluster_tf[tok] = cluster_tf.get(tok, 0) + 1
        for tok in set(tokens):
            cluster_df[tok] = cluster_df.get(tok, 0) + 1

    if not cluster_tf:
        return {}

    # Document frequency over the full corpus
    corpus_df: dict[str, int] = {}
    for text in corpus_texts:
        for tok in set(_tokenize(text)):
            corpus_df[tok] = corpus_df.get(tok, 0) + 1

    n_corpus = max(1, len(corpus_texts))
    max(1, len(cluster_texts))
    total_cluster_terms = max(1, sum(cluster_tf.values()))

    scores: dict[str, float] = {}
    for term, tf in cluster_tf.items():
        tf_norm = tf / total_cluster_terms
        cdf = corpus_df.get(term, 0)
        # Boost terms that are rare in corpus but common in cluster
        idf_contrast = math.log(1.0 + n_corpus / (1.0 + cdf))
        scores[term] = tf_norm * idf_contrast

    # Sort and return top_n
    sorted_terms = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return {t: round(s, 4) for t, s in sorted_terms[:top_n]}


# ---- Builder -------------------------------------------------------------


@dataclass(frozen=True)
class LatentSchemaConfig:
    n_facet_axes: int = 4
    # Calibrated to 384-dim unit-norm embedding space.
    # Typical within-cluster variance for tight clusters is ~3e-4;
    # a floor of 1e-4 made the confidence formula always return 0.0 because
    # the ratio exceeded 1.0 for every real cluster. At 1e-2 a tight cluster
    # (within_var=3e-4) yields confidence≈0.97 and a loose cluster
    # (within_var≥1e-2) correctly yields confidence=0.0.
    variance_floor: float = 1e-2
    min_members_for_facets: int = 3


class LatentSchemaBuilder:
    """Build a ``LatentSchema`` from a prototype + its member episodes.

    Pure geometry. Zero LLM calls. Deterministic for a given set of inputs.

    vsa_mode controls how the VSA triple vector is built:
      "geometric" — current default: centroid + PCA axes (no encoder needed)
      "lexical"   — regex + lexical signature + encoder.encode_many()

    encoder must be supplied when vsa_mode != "geometric".
    """

    def __init__(
        self,
        cfg: Optional[LatentSchemaConfig] = None,
        vsa_mode: str = "geometric",
        encoder=None,
    ):
        self.cfg = cfg or LatentSchemaConfig()
        if vsa_mode not in ("geometric", "lexical"):
            raise ValueError(f"vsa_mode must be 'geometric' or 'lexical'; got {vsa_mode!r}")
        if vsa_mode != "geometric" and encoder is None:
            raise ValueError(f"vsa_mode={vsa_mode!r} requires an encoder")
        self.vsa_mode = vsa_mode
        self.encoder = encoder

    def build(
        self,
        *,
        centroid: np.ndarray,
        member_embeddings: np.ndarray,
        member_episodes: list[EpisodeText],
        member_episode_ids: list[int],
        member_timestamps: list[int] | None = None,
        # Optional background corpus for contrastive TF-IDF.
        # When provided, IDF is computed against this global corpus
        # instead of the cluster's own texts. This makes the lexical
        # signature surface terms distinctive to this cluster vs the
        # rest of the system, rather than penalising theme-defining
        # terms that appear across the cluster's own episodes.
        background_corpus_texts: list[str] | None = None,
    ) -> Optional[LatentSchema]:
        if len(member_episodes) == 0 or member_embeddings.size == 0:
            return None
        embs = np.asarray(member_embeddings, dtype=np.float32)
        if embs.ndim == 1:
            embs = embs.reshape(1, -1)
        cen = np.asarray(centroid, dtype=np.float32).reshape(-1)

        # temporal anchor
        if member_timestamps:
            ts_arr = np.asarray(member_timestamps, dtype=np.int64)
            mean_ts = int(ts_arr.mean())
            ts_span = int(ts_arr.max() - ts_arr.min())
        else:
            mean_ts = int(time.time())
            ts_span = 0

        # central member — prefer the most recent episode (highest timestamp)
        # so schema content_text reflects the latest information, not a
        # geometrically-average episode that can surface stale values when
        # old and new facts cluster together (e.g. knowledge updates).
        # Falls back to centroid-closest when timestamps are unavailable.
        if member_timestamps:
            central_idx = int(np.argmax(ts_arr))
        else:
            sims = (
                embs @ cen / (np.linalg.norm(embs, axis=1) * (np.linalg.norm(cen) + 1e-12) + 1e-12)
            )
            central_idx = int(np.argmax(sims))
        central_episode_id = int(member_episode_ids[central_idx])
        # Prefer source_content (raw, no role prefix) as the schema claim.
        # Falls back to content_text for legacy rows that predate source_content.
        central_ep = member_episodes[central_idx]
        central_text = str(
            central_ep.source_content if central_ep.source_content else central_ep.content_text
        )

        # facet axes (within-cluster principal directions)
        if len(member_episodes) >= self.cfg.min_members_for_facets:
            centered = embs - cen
            try:
                _, s, vh = np.linalg.svd(centered, full_matrices=False)
                k = min(self.cfg.n_facet_axes, vh.shape[0])
                facet_axes = vh[:k].astype(np.float32)
                facet_strengths = s[:k].astype(np.float32)
            except np.linalg.LinAlgError:
                facet_axes = np.zeros((0, embs.shape[1]), dtype=np.float32)
                facet_strengths = np.zeros((0,), dtype=np.float32)
        else:
            facet_axes = np.zeros((0, embs.shape[1]), dtype=np.float32)
            facet_strengths = np.zeros((0,), dtype=np.float32)

        # confidence (cluster tightness) and embedding variance (abstraction proxy)
        if embs.shape[0] >= 2:
            within_var = float(((embs - cen) ** 2).mean())
            confidence = 1.0 - min(1.0, within_var / max(self.cfg.variance_floor, 1e-6))
            confidence = max(0.0, min(1.0, confidence))
        else:
            within_var = 0.0
            confidence = 1.0

        # Lexical signature: contrastive TF-IDF against a global background
        # corpus when available, falling back to intra-cluster distinctiveness
        # if no background corpus is provided.
        cluster_texts = [str(ep.content_text) for ep in member_episodes if ep.content_text]
        if background_corpus_texts:
            corpus_texts = background_corpus_texts
        else:
            corpus_texts = cluster_texts  # fallback: intra-cluster
        lexical_sig = _build_lexical_signature(
            cluster_texts=cluster_texts,
            corpus_texts=corpus_texts,
            top_n=8,
        )
        top_terms = list(lexical_sig.keys())[:3]
        display_lbl = " / ".join(top_terms) if top_terms else ""

        # VSA binding: encode as a role-bound triple.
        # Mode is controlled by self.vsa_mode (set at builder construction).
        from slowave.latent.vsa import (
            build_schema_vsa,
            build_schema_vsa_lexical,
            vec_to_b64,
        )

        if self.vsa_mode == "lexical":
            vsa_vec = build_schema_vsa_lexical(
                cen,
                central_text,
                lexical_sig,
                self.encoder,
            )
        else:  # "geometric" — default, no encoder needed
            vsa_vec = build_schema_vsa(cen, facet_axes)
        vsa_b64 = vec_to_b64(vsa_vec)

        # Episode embedding variance: measures within-cluster semantic spread.
        # High variance = schema was recalled in diverse contexts = more abstract.
        # Low variance = specific fact recalled in similar contexts.
        # This is Phase 1 of gap 6 (abstraction quality) — measurement only.
        episode_embedding_variance = float(within_var) if embs.shape[0] >= 2 else 0.0

        return LatentSchema(
            centroid=cen,
            facet_axes=facet_axes,
            facet_strengths=facet_strengths,
            member_episode_ids=[int(i) for i in member_episode_ids],
            central_episode_id=central_episode_id,
            central_episode_text=central_text,
            mean_ts=mean_ts,
            ts_span_s=ts_span,
            confidence=confidence,
            support_count=int(embs.shape[0]),
            lexical_signature=lexical_sig,
            display_label=display_lbl,
            tags=[],
            facets={
                "schema_class": "latent",
                "confidence": float(confidence),
                "mean_ts": int(mean_ts),
                "ts_span_s": int(ts_span),
                "display_label": display_lbl,
                "vsa_vec": vsa_b64,
                "episode_embedding_variance": round(episode_embedding_variance, 6),
            },
            evidence_indices=list(range(1, len(member_episodes) + 1)),
            evidence_quote=None,
        )


# ---- Geometric contradiction judge --------------------------------------


@dataclass(frozen=True)
class GeometricJudgeConfig:
    # Centroid cosine similarity required for two schemas to be
    # "about the same thing" (the only floor below which no verdict
    # beyond "unrelated" is meaningful).
    same_topic_cosine: float = 0.75
    # Facet-axis distance above which two same-topic schemas are judged
    # to diverge enough in their facet structure to be worth noting,
    # though the verdict remains the honest catch-all.
    # Computed as 1 - mean(|cos(axis_old, axis_new)|).
    contradicts_facet_dist: float = 0.35
    # New schema must have at least this much support to supersede
    # an older one.
    min_support_to_supersede: int = 2
    # Minimum time (seconds) between old and new schema for the newer
    # one to supersede. Prevents rapid toggling where two schemas
    # contradict each other within a very short window.
    min_time_delta_to_supersede_s: float = 3600.0  # 1 hour
    # Cosine above which Consolidator._write_latent_schema's near-duplicate
    # guard reinforces the closest active schema instead of ever reaching
    # this judge. Set >= 1.0 to disable the guard (every candidate reaches
    # the judge, subject to related_schema_cosine below).
    near_dup_guard_cosine: float = 0.92
    # Cosine above which Consolidator._best_related_schema treats an
    # existing schema as related enough to compare via this judge.
    related_schema_cosine: float = 0.72


class GeometricContradictionJudge:
    """Latent-space relation detector.

    Emits exactly two verdicts: 'unrelated' (below the same-topic cosine
    floor) or 'relates_to' (cleared it). This is the end state of two
    rounds of cutting geometric discriminants that didn't hold up on real
    data, not the original design:

      * 2026-07-22: 'supersedes' and 'refines' removed. Both depended on
        direction_score, a single global SVD1 "value substitution" axis
        fit from a small synthetic seed set. Independent measurement in
        the sibling semantic-relations repo found it carried no
        separating signal on real held-out pairs, and got WORSE (not
        better) when refit on more real data -- evidence against the
        mechanism itself, not against that seed set. Supersession (status
        changes) is now gated on metadata (support count + time delta) at
        the consolidation call site instead.
      * 2026-07-23: 'part_of' removed. Its asymmetric-subspace-containment
        signal was structurally biased toward calling broad/generic
        schemas "containers" (their facet axes span more variance, so more
        things geometrically "fit inside" them regardless of real
        hierarchy) -- a hand-labeled audit of production edges found ~11%
        precision. It also carried no differential weight at retrieval
        time vs relates_to, so even a fixed detector wouldn't have changed
        behavior without also building consumption logic that never
        existed. See
        private/docs/iterations/20260723_part_of_audit_and_brain_alignment_review.md.

    Cosine alone cannot reliably distinguish value-substitution, elaboration,
    restatement, or specialization within a same-topic pair -- that's the
    throughline across both removals, not two unrelated incidents. If a
    genuinely reliable geometric discriminant is found later, add it back
    deliberately with its own validated evidence, not by reviving either of
    these.
    """

    def __init__(
        self,
        cfg: Optional[GeometricJudgeConfig] = None,
    ):
        self.cfg = cfg or GeometricJudgeConfig()

    def judge(self, *, old: LatentSchema, new: LatentSchema) -> GeometricVerdict:
        # Centroid similarity -- the ONLY topical gate. Below this floor the
        # two schemas simply aren't about the same thing, so no other signal
        # is worth computing.
        a = old.centroid.astype(np.float32)
        b = new.centroid.astype(np.float32)
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
        cos = float(a @ b / denom)

        # Time delta (always present, even if mean_ts==0)
        dt_s = int(new.mean_ts - old.mean_ts)

        def _emit(
            verdict: str,
            *,
            facet_distance: float | None = None,
            has_facet_signal: bool | None = None,
        ) -> None:
            emit_judge_signal(
                {
                    "code_path": "consolidation_judge",
                    "old_central_episode_id": old.central_episode_id,
                    "new_central_episode_id": new.central_episode_id,
                    "old_text": old.central_episode_text[:200],
                    "new_text": new.central_episode_text[:200],
                    "cos": cos,
                    "facet_distance": facet_distance,
                    "has_facet_signal": has_facet_signal,
                    "verdict": verdict,
                }
            )

        if cos < self.cfg.same_topic_cosine:
            _emit("unrelated")
            return GeometricVerdict(
                verdict="unrelated",
                reasoning=f"centroid_cos={cos:.3f}<same_topic",
                similarity=cos,
                facet_distance=0.0,
                time_delta_s=dt_s,
            )

        # Facet-axis distance = 1 - mean(|cos(axis_old_i, axis_new_i)|) over
        # the min(k_old, k_new) pairs. Bounded in [0, 1]. `has_facet_signal`
        # tracks whether that pairwise comparison actually happened -- with
        # no axes on one or both sides there is nothing to compare, and
        # facet_dist=0.0 in that case means "no data", not "these two
        # confirm each other". Diagnostic only (logged into the reason
        # string and the judge_debug signal below) -- does not gate the
        # verdict; see the class docstring for why.
        has_facet_signal = old.facet_axes.size > 0 and new.facet_axes.size > 0
        if has_facet_signal:
            k = min(old.facet_axes.shape[0], new.facet_axes.shape[0])
            pair_cos = []
            for i in range(k):
                pa = old.facet_axes[i]
                pb = new.facet_axes[i]
                pdenom = (np.linalg.norm(pa) * np.linalg.norm(pb)) + 1e-12
                pair_cos.append(abs(float(pa @ pb / pdenom)))
            facet_dist = 1.0 - float(np.mean(pair_cos)) if pair_cos else 0.0
        else:
            facet_dist = 0.0

        # Same-topic: the honest verdict is relates_to. Cosine alone cannot
        # distinguish value-substitution, elaboration, restatement, or
        # specialization within a same-topic pair -- see the class
        # docstring for the two rounds of geometric discriminants (facet
        # divergence/direction_score, then subspace containment) that were
        # tried and removed for exactly this reason. Supersession (status
        # changes) is gated on metadata (support count + time delta) at the
        # consolidation call site, not on a geometric classifier.
        _emit("relates_to", facet_distance=facet_dist, has_facet_signal=has_facet_signal)
        return GeometricVerdict(
            verdict="relates_to",
            reasoning=f"centroid_cos={cos:.3f} facet_dist={facet_dist:.3f}",
            similarity=cos,
            facet_distance=facet_dist,
            time_delta_s=dt_s,
        )
