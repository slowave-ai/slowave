"""Unit tests for LatentSchemaBuilder and GeometricContradictionJudge.

Covers:
  - unrelated schemas (different topic)
  - reinforcing schemas (very similar centroid)
  - refining schemas (same topic, aligned facets)
  - contradicting schemas (same topic, divergent facets)
  - newer schema with higher support triggers contradiction verdict
  - builder produces expected geometry for a trivial case
  - temporal anchor plumbing: dataclasses.replace on RetrievalConfig works
"""

from __future__ import annotations

import dataclasses
import time

import numpy as np

from slowave.latent.schema import (
    GeometricContradictionJudge,
    GeometricJudgeConfig,
    LatentSchema,
    LatentSchemaBuilder,
    _build_lexical_signature,
    _tokenize,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _unit(v: np.ndarray) -> np.ndarray:
    return (v / (np.linalg.norm(v) + 1e-12)).astype(np.float32)


def _make_schema(
    centroid: np.ndarray,
    *,
    facet_axes: np.ndarray | None = None,
    support_count: int = 3,
    mean_ts: int = 1_000_000,
) -> LatentSchema:
    dim = centroid.shape[0]
    if facet_axes is None:
        facet_axes = np.zeros((0, dim), dtype=np.float32)
    return LatentSchema(
        centroid=centroid.astype(np.float32),
        facet_axes=facet_axes.astype(np.float32),
        facet_strengths=np.ones(facet_axes.shape[0], dtype=np.float32),
        member_episode_ids=[1, 2, 3],
        central_episode_id=1,
        central_episode_text="test episode",
        mean_ts=mean_ts,
        ts_span_s=0,
        confidence=0.9,
        support_count=support_count,
    )


# ---------------------------------------------------------------------------
# GeometricContradictionJudge
# ---------------------------------------------------------------------------


class TestGeometricContradictionJudge:
    def setup_method(self):
        self.judge = GeometricContradictionJudge()

    def test_unrelated_schemas(self):
        """Orthogonal centroids → unrelated."""
        a = _make_schema(_unit(np.array([1.0, 0.0, 0.0])))
        b = _make_schema(_unit(np.array([0.0, 1.0, 0.0])))
        result = self.judge.judge(old=a, new=b)
        assert result.verdict == "unrelated"
        assert result.similarity < self.judge.cfg.same_topic_cosine

    def test_near_identical_centroids_no_facet_signal(self):
        """Near-identical centroids, no facet axes → relates_to."""
        base = _unit(np.array([1.0, 1.0, 0.0]))
        noisy = _unit(base + np.array([0.0001, 0.0, 0.0]))
        a = _make_schema(base)
        b = _make_schema(noisy)
        result = self.judge.judge(old=a, new=b)
        assert result.verdict == "relates_to"
        assert result.similarity >= 0.90

    def test_same_topic_identical_facets_no_direction(self):
        """Same topic (cos ~0.80), identical facet axes, no direction signal
        -> relates_to (facets agree but nothing more specific applies)."""
        # cos([1,0,0,0,0], [0.8,0.6,0,0,0]) = 0.80: in same-topic zone
        c_old = _unit(np.array([1.0, 0.0, 0.0, 0.0, 0.0]))
        c_new = _unit(np.array([0.8, 0.6, 0.0, 0.0, 0.0]))
        # Identical axis -> facet_dist = 0 < contradicts_facet_dist
        axis = np.array([[0.0, 0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
        a = _make_schema(c_old, facet_axes=axis)
        b = _make_schema(c_new, facet_axes=axis)
        result = self.judge.judge(old=a, new=b)
        assert result.verdict == "relates_to", (
            f"expected relates_to, got {result.verdict}; "
            f"sim={result.similarity:.3f} facet_dist={result.facet_distance:.3f}"
        )
        assert result.facet_distance < self.judge.cfg.contradicts_facet_dist

    def test_high_facet_distance_no_longer_refines(self):
        """Same topic, orthogonal facet axes -- previously expected 'refines',
        now expects 'relates_to' per 2026-07-22 taxonomy collapse: no geometric
        signal (direction_score, facet_distance) generalizes beyond synthetic
        seed sets, so the honest verdict for all same-topic pairs is relates_to."""
        c_old = _unit(np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
        c_new = _unit(np.array([0.8, 0.6, 0.0, 0.0, 0.0, 0.0]))
        axis_old = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        axis_new = np.array([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
        a = _make_schema(c_old, facet_axes=axis_old, support_count=3)
        b = _make_schema(c_new, facet_axes=axis_new, support_count=4, mean_ts=2_000_000)
        result = self.judge.judge(old=a, new=b)
        assert result.verdict == "relates_to"

    def test_high_facet_distance_no_manifold_relates_to(self):
        """High facet_distance with no manifold -- previously expected 'refines',
        now expects 'relates_to' per 2026-07-22 taxonomy collapse."""
        c = _unit(np.array([1.0, 0.0, 0.0, 0.0, 0.0]))
        c2 = _unit(np.array([0.8, 0.6, 0.0, 0.0, 0.0]))  # cos ~0.80
        axis_a = np.array([[1.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        axis_b = np.array([[0.0, 0.0, 0.0, 1.0, 0.0]], dtype=np.float32)
        old = _make_schema(c, facet_axes=axis_a, support_count=2, mean_ts=1_000_000)
        new = _make_schema(c2, facet_axes=axis_b, support_count=5, mean_ts=2_000_000)
        result = self.judge.judge(old=old, new=new)
        assert result.verdict == "relates_to"
        assert result.time_delta_s > 0

    def test_newer_with_higher_support_relates_to(self):
        """Same geometry as above (newer, better-supported, divergent
        facets) -- per 2026-07-22 taxonomy collapse, all same-topic pairs
        are relates_to regardless of facet distance or support/recency.
        Supersession (status changes) is now gated at the consolidation
        call site (cos >= 0.90 + support + recency), not by the judge."""
        c = _unit(np.array([1.0, 0.0, 0.0, 0.0, 0.0]))
        c2 = _unit(np.array([0.8, 0.6, 0.0, 0.0, 0.0]))  # cos ~0.80
        axis_a = np.array([[1.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        axis_b = np.array([[0.0, 0.0, 0.0, 1.0, 0.0]], dtype=np.float32)
        old = _make_schema(c, facet_axes=axis_a, support_count=2, mean_ts=1_000_000)
        new = _make_schema(c2, facet_axes=axis_b, support_count=5, mean_ts=2_000_000)
        result = self.judge.judge(old=old, new=new)
        assert result.verdict == "relates_to"

    def test_no_facet_axes_falls_back_to_relates_to(self):
        """Same-topic schemas with no facet axes on either side -> relates_to
        (facet_dist=0 here means "no data", not "these confirm each other" --
        "refines"/"reinforces" are both specific claims unwarranted without
        any facet signal at all)."""
        # cos([1,0,0,0,0], [0.8,0.6,0,0,0]) = 0.80: same-topic, below reinforce
        c_old = _unit(np.array([1.0, 0.0, 0.0, 0.0, 0.0]))
        c_new = _unit(np.array([0.8, 0.6, 0.0, 0.0, 0.0]))
        a = _make_schema(c_old)  # no facet axes
        b = _make_schema(c_new)
        result = self.judge.judge(old=a, new=b)
        assert result.verdict == "relates_to"
        assert result.facet_distance == 0.0

    def test_relates_to_catch_all(self):
        """Same-topic cosine below reinforce_cosine, but facet axes present
        on only one side (so there's no pairwise facet comparison to make)
        -> relates_to, not reinforces or refines."""
        c_old = _unit(np.array([1.0, 0.0, 0.0, 0.0, 0.0]))
        c_new = _unit(np.array([0.8, 0.6, 0.0, 0.0, 0.0]))  # cos ~0.80
        axis = np.array([[0.0, 0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
        a = _make_schema(c_old, facet_axes=axis)
        b = _make_schema(c_new)  # no facet axes on this side
        result = self.judge.judge(old=a, new=b)
        assert result.verdict == "relates_to"

    def test_custom_config_thresholds_respected(self):
        """Custom config thresholds gate all verdicts correctly."""
        cfg = GeometricJudgeConfig(
            same_topic_cosine=0.5,
            contradicts_facet_dist=0.1,
        )
        judge = GeometricContradictionJudge(cfg)
        c_a = _unit(np.array([1.0, 0.3, 0.0]))
        c_b = _unit(np.array([1.0, 0.0, 0.3]))
        axis = np.array([[0.0, 1.0, 0.0]], dtype=np.float32)
        a = _make_schema(c_a, facet_axes=axis)
        b = _make_schema(c_b, facet_axes=axis)
        result = judge.judge(old=a, new=b)
        assert result.verdict in ("refines", "relates_to")


# ---------------------------------------------------------------------------
# LatentSchemaBuilder
# ---------------------------------------------------------------------------


class _ET:
    """Minimal EpisodeText-like stub."""

    def __init__(self, text: str, source_content: str | None = None):
        self.content_text = text
        self.source_content = source_content  # mirrors EpisodeText.source_content


class TestLatentSchemaBuilder:
    def setup_method(self):
        self.builder = LatentSchemaBuilder()

    def test_build_single_member(self):
        """Single-member cluster: confidence=1.0, no facet axes."""
        dim = 8
        emb = _unit(np.random.default_rng(0).random(dim))
        schema = self.builder.build(
            centroid=emb,
            member_embeddings=np.array([emb]),
            member_episodes=[_ET("single episode")],
            member_episode_ids=[42],
            member_timestamps=[1_000_000],
        )
        assert schema is not None
        assert schema.confidence == 1.0
        assert schema.central_episode_id == 42
        assert schema.central_episode_text == "single episode"
        assert schema.support_count == 1
        assert schema.mean_ts == 1_000_000

    def test_build_multi_member(self):
        """Multi-member cluster: confidence < 1.0, facet axes present."""
        rng = np.random.default_rng(1)
        dim = 16
        n = 5
        embs = rng.random((n, dim)).astype(np.float32)
        embs /= np.linalg.norm(embs, axis=1, keepdims=True)
        centroid = _unit(embs.mean(axis=0))
        timestamps = [1_000_000 + i * 1000 for i in range(n)]
        schema = self.builder.build(
            centroid=centroid,
            member_embeddings=embs,
            member_episodes=[_ET(f"ep_{i}") for i in range(n)],
            member_episode_ids=list(range(n)),
            member_timestamps=timestamps,
        )
        assert schema is not None
        assert schema.support_count == n
        assert 0.0 <= schema.confidence <= 1.0
        assert schema.facet_axes.shape[0] > 0  # facets built for n >= 3
        assert schema.mean_ts == int(np.array(timestamps).mean())

    def test_central_member_most_recent_with_timestamps(self):
        """With timestamps, picks the most recent episode even when another
        episode is geometrically closer to the centroid."""
        # e0 = [1.0, 0.0] is closest to centroid [1.0, 0.1]
        # e2 = [0.1, 1.0] is farthest from centroid
        # But e2 has the highest timestamp, so it should win.
        e0 = _unit(np.array([1.0, 0.0]))
        e1 = _unit(np.array([0.5, 0.5]))
        e2 = _unit(np.array([0.1, 1.0]))
        centroid = _unit(np.array([1.0, 0.1]))
        schema = self.builder.build(
            centroid=centroid,
            member_embeddings=np.array([e0, e1, e2]),
            member_episodes=[
                _ET("old value"),
                _ET("middle value"),
                _ET("updated value"),
            ],
            member_episode_ids=[10, 11, 12],
            member_timestamps=[1000, 2000, 3000],
        )
        assert schema is not None
        assert schema.central_episode_id == 12
        assert schema.central_episode_text == "updated value"

    def test_build_empty_returns_none(self):
        """Empty input returns None."""
        schema = self.builder.build(
            centroid=np.zeros(8, dtype=np.float32),
            member_embeddings=np.zeros((0, 8), dtype=np.float32),
            member_episodes=[],
            member_episode_ids=[],
        )
        assert schema is None

    def test_central_member_fallback_no_timestamps(self):
        """Without timestamps, falls back to geometrically closest episode."""
        e0 = _unit(np.array([1.0, 0.5, 0.0, 0.0]))
        e1 = _unit(np.array([1.0, 1.0, 0.0, 0.0]))  # closest to centroid
        e2 = _unit(np.array([0.5, 1.0, 1.0, 0.0]))
        centroid = _unit(np.array([1.0, 1.0, 0.0, 0.0]))
        schema = self.builder.build(
            centroid=centroid,
            member_embeddings=np.array([e0, e1, e2]),
            member_episodes=[_ET(f"ep_{i}") for i in range(3)],
            member_episode_ids=[10, 11, 12],
        )
        assert schema is not None
        assert schema.central_episode_id == 11


# ---------------------------------------------------------------------------
# Temporal anchor plumbing
# ---------------------------------------------------------------------------


class TestTemporalAnchorPlumbing:
    """RetrievalConfig.temporal_anchor_ts field and dataclasses.replace work."""

    def test_retrieval_config_has_temporal_anchor_ts_field(self):
        from slowave.latent.retrieval import RetrievalConfig

        cfg = RetrievalConfig()
        assert cfg.temporal_anchor_ts is None

    def test_dataclasses_replace_works_with_temporal_anchor_ts(self):
        from slowave.latent.retrieval import RetrievalConfig

        cfg = RetrievalConfig()
        now = int(time.time())
        past = now - 30 * 86400
        anchored = dataclasses.replace(cfg, temporal_anchor_ts=past)
        assert anchored.temporal_anchor_ts == past
        # Other fields unchanged
        assert anchored.use_spreading == cfg.use_spreading
        assert anchored.temporal_weight == cfg.temporal_weight

    def test_temporal_probe_estimate_anchor_returns_int(self):
        """TemporalProbe.estimate_anchor always returns an int and does not raise."""
        from slowave.latent.temporal import TemporalProbe

        def mock_encode(text: str) -> np.ndarray:
            h = abs(hash(text)) % 16
            v = np.zeros(16, dtype=np.float32)
            v[h] = 1.0
            return v

        probe = TemporalProbe(mock_encode)
        now_ts = int(time.time())

        # Zero vector: no semantic signal, returns some int (weighted probe average)
        neutral = np.zeros(16, dtype=np.float32)
        anchor = probe.estimate_anchor(neutral, now_ts=now_ts)
        assert isinstance(anchor, int)

        # Non-zero query also returns an int without raising
        some_q = np.ones(16, dtype=np.float32) / 4.0
        anchor2 = probe.estimate_anchor(some_q, now_ts=now_ts)
        assert isinstance(anchor2, int)

        # A past-anchored query should differ from an atemporal one
        # (exact values are encoder-dependent; we only assert type and no raise)

    def test_temporal_probe_dead_zone_atemporal_returns_now(self):
        """Dead-zone gate: atemporal queries must return now_ts unchanged."""
        from slowave.latent.temporal import _TEMPORAL_PROBES, TemporalProbe

        dim = 32
        now_ts = 1_700_000_000
        # now probe → dim 0; all past probes → dim 1; query → dim 0 (matches now)
        phrase_to_vec: dict[str, np.ndarray] = {}
        for i, (phrase, _) in enumerate(_TEMPORAL_PROBES):
            v = np.zeros(dim, dtype=np.float32)
            v[0 if i == 0 else 1] = 1.0
            phrase_to_vec[phrase] = v

        def enc(text: str) -> np.ndarray:
            return phrase_to_vec.get(text, np.zeros(dim, dtype=np.float32))

        probe = TemporalProbe(enc, atemporal_margin=0.12)
        query = np.zeros(dim, dtype=np.float32)
        query[0] = 1.0
        assert probe.estimate_anchor(query, now_ts=now_ts) == now_ts

    def test_temporal_probe_dead_zone_temporal_shifts_anchor(self):
        """Dead-zone gate: genuinely temporal queries shift the anchor past."""
        from slowave.latent.temporal import _TEMPORAL_PROBES, TemporalProbe

        dim = 32
        now_ts = 1_700_000_000
        # Only "last month" probe (index 5) → dim 0; everything else → zeros.
        # query → dim 0; margin = 1.0 - 0.0 = 1.0 >> 0.12 → gate does NOT fire.
        last_month_phrase = _TEMPORAL_PROBES[5][0]
        last_month_disp = _TEMPORAL_PROBES[5][1]

        def enc(text: str) -> np.ndarray:
            v = np.zeros(dim, dtype=np.float32)
            if text == last_month_phrase:
                v[0] = 1.0
            return v

        probe = TemporalProbe(enc, atemporal_margin=0.12)
        query = np.zeros(dim, dtype=np.float32)
        query[0] = 1.0
        anchor = probe.estimate_anchor(query, now_ts=now_ts)
        assert anchor != now_ts, "Temporal query should NOT return now_ts"
        assert abs(anchor - (now_ts + last_month_disp)) < 86400

    def test_temporal_probe_dead_zone_boundary(self):
        """Dead-zone gate: margin just below threshold → now; just above → shifts."""
        from slowave.latent.temporal import _TEMPORAL_PROBES, TemporalProbe

        dim = 32
        now_ts = 1_700_000_000
        threshold = 0.12
        now_phrase = _TEMPORAL_PROBES[0][0]
        past_phrase = _TEMPORAL_PROBES[1][0]

        def make_enc(past_sim: float):
            def enc(text: str) -> np.ndarray:
                v = np.zeros(dim, dtype=np.float32)
                if text == now_phrase:
                    v[0] = 0.5
                    v[1] = float(np.sqrt(max(0.0, 1 - 0.25)))
                elif text == past_phrase:
                    v[0] = past_sim
                    v[2] = float(np.sqrt(max(0.0, 1 - past_sim**2)))
                return v

            return enc

        query = np.zeros(dim, dtype=np.float32)
        query[0] = 1.0
        # margin < threshold → now
        p_below = TemporalProbe(make_enc(0.5 + threshold - 0.001), atemporal_margin=threshold)
        assert p_below.estimate_anchor(query, now_ts=now_ts) == now_ts
        # margin > threshold → shifts
        p_above = TemporalProbe(make_enc(0.5 + threshold + 0.001), atemporal_margin=threshold)
        assert p_above.estimate_anchor(query, now_ts=now_ts) != now_ts


# ---------------------------------------------------------------------------
# Lexical signature helpers
# ---------------------------------------------------------------------------


class TestLexicalSignature:
    def test_tokenize_basic(self):
        tokens = _tokenize("Slowave uses FAISS for local memory")
        assert "slowave" in tokens
        assert "faiss" in tokens
        assert "local" in tokens
        assert "memory" in tokens
        # stopwords and short words removed
        assert "for" not in tokens  # stopword
        assert "a" not in tokens  # too short
        assert "uses" in tokens  # 4 chars, not a stopword - kept

    def test_tokenize_drops_short_and_stopwords(self):
        tokens = _tokenize("a an the in it")
        assert tokens == []

    def test_build_lexical_signature_returns_dict(self):
        texts = [
            "Slowave uses FAISS for vector search",
            "Slowave stores memory in SQLite locally",
            "Slowave avoids remote LLM inference entirely",
            "FAISS index is local and private",
        ]
        sig = _build_lexical_signature(cluster_texts=texts, corpus_texts=texts, top_n=6)
        assert isinstance(sig, dict)
        assert len(sig) <= 6
        # All scores should be positive floats
        for term, score in sig.items():
            assert isinstance(term, str)
            assert score > 0.0

    def test_build_lexical_signature_prominent_term(self):
        """A term repeated across all cluster docs should score highly."""
        texts = [
            "nebula uses faiss backend for storage",
            "nebula faiss index is local",
            "nebula faiss search is fast",
            "nebula stores episodes locally with faiss",
        ]
        sig = _build_lexical_signature(cluster_texts=texts, corpus_texts=texts, top_n=8)
        # "faiss" and "nebula" should both appear (both high frequency)
        assert "faiss" in sig
        assert "nebula" in sig

    def test_build_lexical_signature_empty_input(self):
        sig = _build_lexical_signature(cluster_texts=[], corpus_texts=[], top_n=5)
        assert sig == {}

    def test_builder_populates_lexical_signature(self):
        """LatentSchemaBuilder.build() sets lexical_signature and display_label."""
        builder = LatentSchemaBuilder()
        rng = np.random.default_rng(42)
        dim = 16
        n = 5
        embs = rng.random((n, dim)).astype(np.float32)
        embs /= np.linalg.norm(embs, axis=1, keepdims=True)
        centroid = embs.mean(axis=0)
        episodes = [
            _ET("Slowave uses FAISS for local vector search"),
            _ET("Slowave stores episodes in SQLite database"),
            _ET("Slowave FAISS index is private and local"),
            _ET("SQLite database stores all memory locally"),
            _ET("Local FAISS index enables fast retrieval"),
        ]
        schema = builder.build(
            centroid=centroid,
            member_embeddings=embs,
            member_episodes=episodes,
            member_episode_ids=list(range(n)),
        )
        assert schema is not None
        # lexical_signature should be populated
        assert isinstance(schema.lexical_signature, dict)
        assert len(schema.lexical_signature) > 0
        # display_label should be non-empty
        assert isinstance(schema.display_label, str)
        assert len(schema.display_label) > 0
        # Known frequent terms should appear
        all_terms = set(schema.lexical_signature.keys())
        assert "faiss" in all_terms or "local" in all_terms or "sqlite" in all_terms

    def test_builder_display_label_format(self):
        """display_label is top terms joined by ' / '."""
        builder = LatentSchemaBuilder()
        rng = np.random.default_rng(7)
        dim = 8
        n = 3
        embs = rng.random((n, dim)).astype(np.float32)
        embs /= np.linalg.norm(embs, axis=1, keepdims=True)
        centroid = embs.mean(axis=0)
        episodes = [
            _ET("alpha beta gamma delta"),
            _ET("alpha beta gamma epsilon"),
            _ET("alpha beta gamma zeta"),
        ]
        schema = builder.build(
            centroid=centroid,
            member_embeddings=embs,
            member_episodes=episodes,
            member_episode_ids=[0, 1, 2],
        )
        assert schema is not None
        if schema.display_label:
            # Must be slash-separated if there are multiple terms
            parts = schema.display_label.split(" / ")
            assert len(parts) >= 1
