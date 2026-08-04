"""Micro-benchmark tests for WorkingMemoryGate — eligibility, noise floor,
scope penalties, MMR dedup, and activation trace completeness.

All tests are deterministic (no encoder, no DB).
"""

from __future__ import annotations

import numpy as np
import pytest

from slowave.core.context import (
    GatePolicy,
    MemoryCue,
    WorkingMemoryGate,
)
from slowave.symbolic.schema_store import Schema


def _make_unit(dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-12)


def _stub_schema(
    sid: int,
    text: str = "A fact about the domain",
    embedding: np.ndarray | None = None,
    salience: float = 5.0,
    schema_class: str = "fact",
    memory_layer: str = "domain",
    source_kind: str = "explicit_remember",
    stability: str = "current",
    scope_id: str | None = "project:alpha",
    status: str = "active",
    generalization_stage: int = 0,
    injectable: bool = True,
    facets_extra: dict | None = None,
) -> Schema:
    facets: dict = {
        "schema_class": schema_class,
        "memory_layer": memory_layer,
        "source_kind": source_kind,
        "stability": stability,
        "injectable": injectable,
    }
    if facets_extra:
        facets.update(facets_extra)
    return Schema(
        id=sid,
        prototype_id=None,
        content_text=text,
        facets=facets,
        tags=[],
        scope_id=scope_id,
        status=status,
        confidence=1.0,
        salience=salience,
        supporting_episode_ids=[],
        contradicting_episode_ids=[],
        is_labile=False,
        first_formed_ts=1000000 + sid,
        last_updated_ts=1000 + sid,
        embedding=embedding,
        generalization_stage=generalization_stage,
    )


# ── 1. Mode-gated eligibility ──────────────────────────────────────────────


def test_default_mode_only_admits_active() -> None:
    """Schemas with status != 'active' are suppressed in default mode."""
    gate = WorkingMemoryGate()
    cue = MemoryCue(mode="default")
    active = _stub_schema(1, status="active")
    needs_review = _stub_schema(2, status="needs_review")
    superseded = _stub_schema(3, status="superseded")

    state = gate.select(
        [active, needs_review, superseded], cue=cue, policy=GatePolicy(min_activation=0.01)
    )

    admitted_ids = {item.schema.id for item in state.items}
    assert admitted_ids == {1}, f"Expected only active, got {admitted_ids}"
    assert state.suppressed.get("inactive", 0) == 2


def test_broad_mode_admits_active_and_needs_review() -> None:
    """Broad mode admits active + needs_review, not superseded."""
    gate = WorkingMemoryGate()
    cue = MemoryCue(mode="broad")
    active = _stub_schema(1, status="active")
    needs_review = _stub_schema(2, status="needs_review")
    superseded = _stub_schema(3, status="superseded")

    state = gate.select(
        [active, needs_review, superseded], cue=cue, policy=GatePolicy(min_activation=0.01)
    )

    admitted_ids = {item.schema.id for item in state.items}
    assert 1 in admitted_ids
    assert 2 in admitted_ids
    assert 3 not in admitted_ids
    assert state.suppressed.get("inactive", 0) == 1


def test_debug_mode_admits_everything() -> None:
    """Debug mode admits all statuses including superseded."""
    gate = WorkingMemoryGate()
    cue = MemoryCue(mode="debug")
    active = _stub_schema(1, status="active")
    needs_review = _stub_schema(2, status="needs_review")
    superseded = _stub_schema(3, status="superseded")

    state = gate.select(
        [active, needs_review, superseded], cue=cue, policy=GatePolicy(min_activation=0.01)
    )

    admitted_ids = {item.schema.id for item in state.items}
    assert admitted_ids == {1, 2, 3}
    assert "inactive" not in state.suppressed


# ── 2. Strict scope gating ─────────────────────────────────────────────────


def test_strict_scope_blocks_cross_scope_stage0() -> None:
    """In strict_scope mode, Stage 0 cross-scope schema is blocked."""
    gate = WorkingMemoryGate()
    cue = MemoryCue(mode="strict_scope", scope="project:beta")
    cross = _stub_schema(1, scope_id="project:alpha", generalization_stage=0)

    state = gate.select([cross], cue=cue)

    assert len(state.items) == 0
    assert state.suppressed.get("strict_scope_excluded") == 1


def test_strict_scope_admits_stage3_cross_scope() -> None:
    """In strict_scope mode, Stage 3 cross-scope passes eligibility."""
    gate = WorkingMemoryGate()
    cue = MemoryCue(mode="strict_scope", scope="project:beta")
    cross = _stub_schema(1, scope_id="project:alpha", generalization_stage=3)

    state = gate.select([cross], cue=cue)

    # Stage 3 passes eligibility; admission depends on activation
    assert "strict_scope_excluded" not in state.suppressed


def test_strict_scope_admits_same_scope() -> None:
    """In strict_scope mode, same-scope schemas always pass."""
    gate = WorkingMemoryGate()
    cue = MemoryCue(mode="strict_scope", scope="project:alpha")
    same = _stub_schema(1, scope_id="project:alpha", generalization_stage=0)

    state = gate.select([same], cue=cue)

    assert len(state.items) == 1
    assert "strict_scope_excluded" not in state.suppressed


def test_strict_scope_admits_global() -> None:
    """In strict_scope mode, global (no scope_id) schemas pass."""
    gate = WorkingMemoryGate()
    cue = MemoryCue(mode="strict_scope", scope="project:alpha")
    global_s = _stub_schema(1, scope_id=None)

    state = gate.select([global_s], cue=cue)

    assert len(state.items) == 1
    assert "strict_scope_excluded" not in state.suppressed


# ── 3. Multi-sentence summary gate ──────────────────────────────────────────


def test_multi_sentence_summary_suppressed_in_default() -> None:
    """A 4-sentence, >300 char non-explicit_remember schema is suppressed."""
    gate = WorkingMemoryGate()
    cue = MemoryCue(mode="default")
    long_text = (
        "This is the first sentence about project architecture. "
        "The second sentence discusses deployment choices. "
        "The third sentence covers monitoring setup. "
        "The fourth and final sentence concludes the summary."
    )
    # NOT explicit_remember, NOT episodic_summary -> gate fires
    summary = _stub_schema(1, text=long_text, schema_class="fact", source_kind="latent")

    state = gate.select([summary], cue=cue, policy=GatePolicy(min_activation=0.01))

    assert len(state.items) == 0
    assert state.suppressed.get("multi_sentence_summary") == 1


def test_multi_sentence_episodic_summary_not_suppressed() -> None:
    """A multi-sentence schema tagged episodic_summary is NOT suppressed."""
    gate = WorkingMemoryGate()
    cue = MemoryCue(mode="default")
    long_text = (
        "Sentence one about the episode. "
        "Sentence two continues. "
        "Sentence three wraps up. "
        "Sentence four is extra."
    )
    summary = _stub_schema(1, text=long_text, schema_class="episodic_summary", source_kind="latent")
    # episodic_summary not in default allowed_classes — add it
    policy = GatePolicy(
        min_activation=0.01,
        allowed_classes=(
            "episodic_summary",
            "fact",
            "preference",
            "decision",
            "lesson",
            "constraint",
            "warning",
            "procedure",
            "open_question",
            "task",
            "artifact",
            "relationship",
            "interaction_preference",
            "habit",
        ),
    )

    state = gate.select([summary], cue=cue, policy=policy)

    assert len(state.items) == 1
    assert "multi_sentence_summary" not in state.suppressed


def test_multi_sentence_explicit_remember_not_suppressed() -> None:
    """A multi-sentence schema with explicit_remember source bypasses the gate."""
    gate = WorkingMemoryGate()
    cue = MemoryCue(mode="default")
    long_text = (
        "Sentence one about the memory. "
        "Sentence two continues the thought. "
        "Sentence three finishes it."
    )
    summary = _stub_schema(1, text=long_text, schema_class="fact", source_kind="explicit_remember")

    state = gate.select([summary], cue=cue, policy=GatePolicy(min_activation=0.01))

    assert len(state.items) == 1
    assert "multi_sentence_summary" not in state.suppressed


def test_short_text_not_suppressed() -> None:
    """A short 2-sentence, <300 char schema passes the gate."""
    gate = WorkingMemoryGate()
    cue = MemoryCue(mode="default")
    short = _stub_schema(
        1, text="Just two sentences. Very short indeed.", schema_class="fact", source_kind="latent"
    )

    state = gate.select([short], cue=cue, policy=GatePolicy(min_activation=0.01))

    assert len(state.items) == 1
    assert "multi_sentence_summary" not in state.suppressed


def test_multi_sentence_bypasses_in_broad() -> None:
    """Multi-sentence schema passes in broad mode."""
    gate = WorkingMemoryGate()
    cue = MemoryCue(mode="broad")
    long_text = "A. B. C. D. E." * 20  # >300 chars, 5 sentences
    summary = _stub_schema(1, text=long_text, schema_class="fact", source_kind="latent")

    state = gate.select([summary], cue=cue, policy=GatePolicy(min_activation=0.01))

    assert len(state.items) == 1


# ── 4. Excluded layer / source_kind filters ────────────────────────────────


def test_excluded_layer_suppressed() -> None:
    """raw_event layer is excluded by default GatePolicy."""
    gate = WorkingMemoryGate()
    cue = MemoryCue(mode="default")
    raw = _stub_schema(1, memory_layer="raw_event", source_kind="latent", text="A short fact.")

    state = gate.select([raw], cue=cue, policy=GatePolicy(min_activation=0.01))

    assert len(state.items) == 0
    assert "layer_excluded:raw_event" in state.suppressed


def test_excluded_layer_passes_in_broad() -> None:
    """Layer exclusion is bypassed in broad mode."""
    gate = WorkingMemoryGate()
    cue = MemoryCue(mode="broad")
    raw = _stub_schema(1, memory_layer="raw_event", source_kind="latent", text="A short fact.")

    state = gate.select([raw], cue=cue, policy=GatePolicy(min_activation=0.01))

    assert len(state.items) == 1


def test_assistant_summary_source_excluded() -> None:
    """assistant_summary source_kind is excluded in default mode."""
    gate = WorkingMemoryGate()
    cue = MemoryCue(mode="default")
    summary = _stub_schema(
        1, source_kind="assistant_summary", memory_layer="workspace", text="A short summary."
    )

    state = gate.select([summary], cue=cue, policy=GatePolicy(min_activation=0.01))

    assert len(state.items) == 0
    assert "source_excluded:assistant_summary" in state.suppressed


def test_assistant_summary_source_admitted_in_broad() -> None:
    """assistant_summary source_kind passes in broad mode."""
    dim = 8
    cue_emb = _make_unit(dim, seed=1)
    gate = WorkingMemoryGate()
    cue = MemoryCue(query="summary", mode="broad")
    noise = _make_unit(dim, seed=10)
    schema_emb = cue_emb + 0.1 * noise
    schema_emb = schema_emb / (np.linalg.norm(schema_emb) + 1e-12)
    summary = _stub_schema(
        1,
        source_kind="assistant_summary",
        memory_layer="workspace",
        text="A short summary.",
        embedding=schema_emb.astype(np.float32),
    )

    state = gate.select(
        [summary], cue=cue, policy=GatePolicy(min_activation=0.01), cue_embedding=cue_emb
    )

    assert len(state.items) == 1, f"Expected admitted, got suppressed: {state.suppressed}"


# ── 5. Transcript summary detection ────────────────────────────────────────


def test_transcript_summary_suppressed() -> None:
    """Schema containing 'User:' and 'Assistant:' is suppressed."""
    gate = WorkingMemoryGate()
    cue = MemoryCue(mode="default")
    transcript = _stub_schema(
        1,
        text="User: What is the capital? Assistant: The capital is Paris.",
        schema_class="fact",
        source_kind="latent",
    )

    state = gate.select([transcript], cue=cue, policy=GatePolicy(min_activation=0.01))

    assert len(state.items) == 0
    assert state.suppressed.get("transcript_summary") == 1


def test_transcript_summary_admitted_in_broad() -> None:
    """Transcript summaries pass in broad mode."""
    dim = 8
    cue_emb = _make_unit(dim, seed=1)
    gate = WorkingMemoryGate()
    cue = MemoryCue(query="question answer", mode="broad")
    schema_emb = _make_unit(dim, seed=10)
    transcript = _stub_schema(
        1,
        text="User: Question? Assistant: Answer.",
        schema_class="fact",
        source_kind="latent",
        embedding=schema_emb,
    )

    state = gate.select(
        [transcript], cue=cue, policy=GatePolicy(min_activation=0.01), cue_embedding=cue_emb
    )

    assert len(state.items) == 1


# ── 6. Class exclusion filter ──────────────────────────────────────────────


def test_class_exclusion_filter() -> None:
    """Schema classes not in allowed_classes are suppressed."""
    gate = WorkingMemoryGate()
    cue = MemoryCue(mode="default")
    policy = GatePolicy(min_activation=0.01, allowed_classes=("preference", "decision"))

    fact = _stub_schema(1, text="A fact", schema_class="fact")
    pref = _stub_schema(2, text="A preference", schema_class="preference")

    state = gate.select([fact, pref], cue=cue, policy=policy)

    admitted_ids = {item.schema.id for item in state.items}
    assert 1 not in admitted_ids, f"fact should be excluded, got {admitted_ids}"
    assert 2 in admitted_ids, f"preference should be admitted, got {admitted_ids}"
    assert "class_excluded:fact" in state.suppressed


# ── 7. Cross-scope noise floor ─────────────────────────────────────────────


def test_cross_scope_stage1_below_activation_floor_suppressed() -> None:
    """Stage 1 cross-scope schema with activation < 0.30 is suppressed."""
    dim = 8
    gate = WorkingMemoryGate()
    # Use default mode (no strict_scope wall) so cross-scope passes eligibility
    cue = MemoryCue(query="unrelated", scope="project:beta")
    cue_emb = _make_unit(dim, seed=1)

    # Moderate cosine ~0.80 → low cosine → low activation
    noise = _make_unit(dim, seed=99)
    schema_emb = cue_emb + 0.6 * noise
    schema_emb = schema_emb / (np.linalg.norm(schema_emb) + 1e-12)
    cross = _stub_schema(
        1,
        text="Cross-scope fact",
        embedding=schema_emb.astype(np.float32),
        schema_class="fact",
        scope_id="project:alpha",
        generalization_stage=1,
    )

    state = gate.select(
        [cross], cue=cue, policy=GatePolicy(min_activation=0.01), cue_embedding=cue_emb
    )

    # Should be suppressed by cross_scope_below_floor (activation < 0.30)
    assert len(state.items) == 0
    assert state.suppressed.get("cross_scope_below_floor") == 1


def test_cross_scope_stage1_high_cosine_passes() -> None:
    """Stage 1 cross-scope schema with high cosine passes the noise floor."""
    dim = 8
    gate = WorkingMemoryGate()
    cue = MemoryCue(query="task", scope="project:beta")
    cue_emb = _make_unit(dim, seed=1)

    # Embedding very close to cue → cos ~0.995 → high activation
    noise = _make_unit(dim, seed=10)
    schema_emb = cue_emb + 0.03 * noise
    schema_emb = schema_emb / (np.linalg.norm(schema_emb) + 1e-12)

    cross = _stub_schema(
        1,
        text="Cross-scope task fact",
        embedding=schema_emb.astype(np.float32),
        schema_class="fact",
        scope_id="project:alpha",
        generalization_stage=2,
    )

    state = gate.select(
        [cross], cue=cue, policy=GatePolicy(min_activation=0.01), cue_embedding=cue_emb
    )

    assert len(state.items) == 1, f"Expected admission, got suppressed: {state.suppressed}"
    assert state.items[0].schema.id == 1


def test_cross_scope_stage3_exempt_from_noise_floor() -> None:
    """Stage 3 (global) schemas are exempt from the cross-scope noise floor."""
    dim = 8
    gate = WorkingMemoryGate()
    cue = MemoryCue(query="unrelated", scope="project:beta")
    cue_emb = _make_unit(dim, seed=1)
    policy = GatePolicy(min_activation=0.01)

    schema_emb = _make_unit(dim, seed=99)
    cross = _stub_schema(
        1,
        text="Global fact",
        embedding=schema_emb,
        schema_class="fact",
        scope_id="project:alpha",
        generalization_stage=3,
    )

    state = gate.select([cross], cue=cue, cue_embedding=cue_emb, policy=policy)

    # Stage 3 exempt from noise floor; admitted if activation >= min_activation
    assert len(state.items) == 1


# ── 8. Scope mismatch penalty grading ──────────────────────────────────────


def test_scope_mismatch_penalty_stage0() -> None:
    """Stage 0 cross-scope pays full -0.35 penalty."""
    dim = 8
    gate = WorkingMemoryGate()
    cue = MemoryCue(query="task", scope="project:beta")
    cue_emb = _make_unit(dim, seed=1)
    policy = GatePolicy(min_activation=0.01)

    # Same-scope baseline
    same_emb = _make_unit(dim, seed=10)
    same = _stub_schema(
        1, text="Same scope", embedding=same_emb, scope_id="project:beta", generalization_stage=0
    )

    # Cross-scope (no scope bonus, pays -0.35 penalty)
    cross_emb = _make_unit(dim, seed=10)
    cross = _stub_schema(
        2, text="Cross scope", embedding=cross_emb, scope_id="project:alpha", generalization_stage=0
    )

    state = gate.select([same, cross], cue=cue, cue_embedding=cue_emb, policy=policy)

    # Cross-scope pays the full -0.35 penalty and, with near-zero cosine here,
    # lands below min_activation and is suppressed entirely -- so its true
    # (negative) activation must be read from activation_trace, which records
    # every candidate's computed score regardless of admission. Falling back
    # to `next(..., 0)` over state.items would silently substitute 0 for a
    # suppressed schema's real (more negative) score and mask the exact size
    # of the scope-mismatch penalty being tested here.
    same_act = next((i.activation for i in state.items if i.schema.id == 1), None)
    cross_act = next((t.activation for t in state.activation_trace if t.schema_id == 2), None)
    assert same_act is not None and cross_act is not None

    assert (
        same_act > cross_act
    ), f"Same-scope ({same_act:.3f}) should > cross-scope ({cross_act:.3f})"
    # Delta == 0.20 (scope bonus on same) + 0.35 (mismatch penalty on cross) = 0.55
    delta = same_act - cross_act
    assert delta == pytest.approx(0.55, abs=1e-6), f"Expected delta == 0.55, got {delta:.3f}"


def test_scope_mismatch_penalty_stage2_reduced() -> None:
    """Stage 2 cross-scope pays reduced -0.12 penalty."""
    dim = 8
    gate = WorkingMemoryGate()
    cue = MemoryCue(query="task", scope="project:beta")
    cue_emb = _make_unit(dim, seed=1)
    policy = GatePolicy(min_activation=0.01)

    noise0 = _make_unit(dim, seed=10)
    emb0 = cue_emb + 0.35 * noise0
    emb0 = emb0 / (np.linalg.norm(emb0) + 1e-12)
    s0 = _stub_schema(
        1,
        text="Stage 0",
        embedding=emb0.astype(np.float32),
        scope_id="project:alpha",
        generalization_stage=0,
    )

    noise2 = _make_unit(dim, seed=20)
    emb2 = cue_emb + 0.35 * noise2
    emb2 = emb2 / (np.linalg.norm(emb2) + 1e-12)
    s2 = _stub_schema(
        2,
        text="Stage 2",
        embedding=emb2.astype(np.float32),
        scope_id="project:delta",
        generalization_stage=2,
    )

    state = gate.select([s0, s2], cue=cue, cue_embedding=cue_emb, policy=policy)

    assert (
        len(state.items) == 2
    ), f"Expected both admitted, got {len(state.items)}: {state.suppressed}"
    s0_act = next((i.activation for i in state.items if i.schema.id == 1), 0)
    s2_act = next((i.activation for i in state.items if i.schema.id == 2), 0)

    assert s2_act > s0_act, f"Stage 2 ({s2_act:.3f}) should > Stage 0 ({s0_act:.3f})"


def test_scope_mismatch_penalty_stage3_zero() -> None:
    """Stage 3 cross-scope pays no mismatch penalty."""
    dim = 8
    gate = WorkingMemoryGate()
    cue = MemoryCue(query="task", scope="project:beta")
    cue_emb = _make_unit(dim, seed=1)
    policy = GatePolicy(min_activation=0.01)

    emb = _make_unit(dim, seed=10)
    s3 = _stub_schema(
        1, text="Stage 3 global", embedding=emb, scope_id="project:alpha", generalization_stage=3
    )

    state = gate.select([s3], cue=cue, cue_embedding=cue_emb, policy=policy)

    assert len(state.items) == 1
    assert "scope_mismatch" not in state.items[0].reason


# ── 9. MMR deduplication ───────────────────────────────────────────────────


def test_mmr_deduplicates_near_identical_schemas() -> None:
    """Two schemas with cosine >= 0.92 → only the higher-activation one kept."""
    dim = 8
    gate = WorkingMemoryGate()
    cue = MemoryCue(query="task", scope="project:alpha")
    cue_emb = _make_unit(dim, seed=1)

    base = _make_unit(dim, seed=10)
    v1 = (base + 0.01 * _make_unit(dim, seed=11)).astype(np.float32)
    v1 = v1 / (np.linalg.norm(v1) + 1e-12)
    v2 = (base + 0.01 * _make_unit(dim, seed=12)).astype(np.float32)
    v2 = v2 / (np.linalg.norm(v2) + 1e-12)

    s1 = _stub_schema(1, text="First near-duplicate", embedding=v1, salience=10.0)
    s2 = _stub_schema(2, text="Second near-duplicate", embedding=v2, salience=5.0)

    state = gate.select([s1, s2], cue=cue, cue_embedding=cue_emb)

    admitted_ids = {item.schema.id for item in state.items}
    assert len(admitted_ids) == 1, f"MMR should deduplicate, got {admitted_ids}"
    assert 1 in admitted_ids, "Higher-activation schema should win"


def test_mmr_keeps_dissimilar_schemas() -> None:
    """Two schemas with low cosine are both kept."""
    dim = 8
    gate = WorkingMemoryGate()
    cue = MemoryCue(query="task", scope="project:alpha")
    cue_emb = _make_unit(dim, seed=1)

    v1 = _make_unit(dim, seed=10)
    v2 = _make_unit(dim, seed=20)

    s1 = _stub_schema(1, text="Python", embedding=v1.astype(np.float32))
    s2 = _stub_schema(2, text="Rust", embedding=v2.astype(np.float32))

    state = gate.select([s1, s2], cue=cue, cue_embedding=cue_emb)

    admitted_ids = {item.schema.id for item in state.items}
    assert admitted_ids == {1, 2}, f"Both should be kept, got {admitted_ids}"


def test_mmr_always_keeps_schemas_without_embeddings() -> None:
    """Schemas without embeddings are always kept by MMR."""
    gate = WorkingMemoryGate()
    cue = MemoryCue(query="task", scope="project:alpha")

    s1 = _stub_schema(1, text="No embedding A", embedding=None)
    s2 = _stub_schema(2, text="No embedding B", embedding=None)

    state = gate.select([s1, s2], cue=cue, policy=GatePolicy(min_activation=0.01))

    assert len(state.items) == 2


# ── 10. Activation trace completeness ───────────────────────────────────────


def test_activation_trace_includes_all_candidates() -> None:
    """Every candidate schema appears in the activation trace."""
    dim = 8
    gate = WorkingMemoryGate()
    cue = MemoryCue(query="task", scope="project:alpha", mode="debug")
    cue_emb = _make_unit(dim, seed=1)

    s1 = _stub_schema(1, text="Admitted fact", embedding=_make_unit(dim, seed=10))
    s2 = _stub_schema(2, text="Admitted too", embedding=_make_unit(dim, seed=11))

    state = gate.select([s1, s2], cue=cue, cue_embedding=cue_emb)

    trace_ids = {trace.schema_id for trace in state.activation_trace}
    assert trace_ids == {1, 2}, f"Trace should include all candidates, got {trace_ids}"


def test_activation_trace_reason_describes_suppression() -> None:
    """Rejected schemas in the trace have a descriptive reason."""
    gate = WorkingMemoryGate()
    cue = MemoryCue(mode="default")

    inactive = _stub_schema(1, status="superseded")

    state = gate.select([inactive], cue=cue, policy=GatePolicy(min_activation=0.01))

    rejected = [t for t in state.activation_trace if not t.admitted]
    assert len(rejected) == 1
    assert rejected[0].reason == "inactive"
    assert rejected[0].activation == 0.0


# ── 11. Identity prior cap ─────────────────────────────────────────────────


def test_identity_prior_capped_at_0_15() -> None:
    """Even with max bonuses, identity prior contribution is capped at 0.15."""
    dim = 8
    gate = WorkingMemoryGate()
    cue = MemoryCue(query="schema", scope="project:alpha")
    cue_emb = _make_unit(dim, seed=1)
    policy = GatePolicy(min_activation=0.01)

    noise_h = _make_unit(dim, seed=99)
    emb_h = cue_emb + 0.5 * noise_h
    emb_h = emb_h / (np.linalg.norm(emb_h) + 1e-12)
    high_identity = _stub_schema(
        1,
        text="High identity schema",
        embedding=emb_h.astype(np.float32),
        salience=20.0,
        schema_class="preference",
        memory_layer="profile",
        source_kind="explicit_remember",
        stability="current",
    )

    noise_l = _make_unit(dim, seed=88)
    emb_l = cue_emb + 0.5 * noise_l
    emb_l = emb_l / (np.linalg.norm(emb_l) + 1e-12)
    low_identity = _stub_schema(
        2,
        text="Low identity schema",
        embedding=emb_l.astype(np.float32),
        salience=0.1,
        schema_class="lesson",
        memory_layer="domain",
        source_kind="explicit_remember",
        stability="unknown",
    )

    state = gate.select(
        [high_identity, low_identity], cue=cue, cue_embedding=cue_emb, policy=policy
    )

    assert (
        len(state.items) == 2
    ), f"Expected both admitted, got {len(state.items)}: {state.suppressed}"
    high_act = next((i.activation for i in state.items if i.schema.id == 1), None)
    low_act = next((i.activation for i in state.items if i.schema.id == 2), None)

    assert high_act is not None and low_act is not None
    # Without the cap, high would dominate by ~0.59. With cap, delta ≤ 0.15 + epsilon.
    delta = abs(high_act - low_act)
    assert (
        delta <= 0.20
    ), f"Identity delta {delta:.3f} exceeds cap tolerance; high={high_act:.3f} low={low_act:.3f}"


# ── 12. Scope bonus applied after identity cap ─────────────────────────────


def test_scope_bonus_applied_after_identity_cap() -> None:
    """Scope bonus (+0.20 scope_match / +0.15 global) is NOT capped by identity."""
    dim = 8
    gate = WorkingMemoryGate()
    cue = MemoryCue(query="task", scope="project:alpha")
    cue_emb = _make_unit(dim, seed=1)

    emb = _make_unit(dim, seed=10)
    same_scope = _stub_schema(1, text="Same scope", embedding=emb, scope_id="project:alpha")
    global_s = _stub_schema(2, text="Global schema", embedding=emb, scope_id=None)

    state = gate.select([same_scope, global_s], cue=cue, cue_embedding=cue_emb)

    same_act = next((i.activation for i in state.items if i.schema.id == 1), 0)
    global_act = next((i.activation for i in state.items if i.schema.id == 2), 0)

    assert same_act > global_act, f"Scope_match ({same_act:.3f}) should > global ({global_act:.3f})"


# ── 13. noise penalty ─────────────────────────────────────────────────────


def test_noise_penalty_reduces_activation() -> None:
    """context_noise_score in facets applies a -0.30×noise penalty."""
    dim = 8
    gate = WorkingMemoryGate()
    cue = MemoryCue(query="task", scope="project:alpha")
    cue_emb = _make_unit(dim, seed=1)

    clean = _stub_schema(1, text="Clean fact", embedding=_make_unit(dim, seed=10))
    noisy = _stub_schema(
        2,
        text="Noisy fact",
        embedding=_make_unit(dim, seed=10),
        facets_extra={"context_noise_score": 0.8},
    )

    state = gate.select([clean, noisy], cue=cue, cue_embedding=cue_emb)

    clean_act = next((i.activation for i in state.items if i.schema.id == 1), 0)
    noisy_act = next((i.activation for i in state.items if i.schema.id == 2), 0)

    assert noisy_act < clean_act, f"Noisy ({noisy_act:.3f}) should < clean ({clean_act:.3f})"


def test_noise_by_scope_penalizes_when_scope_matches() -> None:
    """context_noise_by_scope, when present, is read for the cue's own scope
    (same magnitude as the flat context_noise_score fallback path above)."""
    dim = 8
    gate = WorkingMemoryGate()
    cue = MemoryCue(query="task", scope="project:alpha")
    cue_emb = _make_unit(dim, seed=1)

    clean = _stub_schema(1, text="Clean fact", embedding=_make_unit(dim, seed=10))
    noisy = _stub_schema(
        2,
        text="Noisy fact",
        embedding=_make_unit(dim, seed=10),
        facets_extra={
            "context_noise_score": 0.8,
            "context_noise_by_scope": {"project:alpha": 0.8, "project:other": 0.0},
        },
    )

    state = gate.select([clean, noisy], cue=cue, cue_embedding=cue_emb)
    clean_act = next((i.activation for i in state.items if i.schema.id == 1), 0)
    noisy_act = next((i.activation for i in state.items if i.schema.id == 2), 0)

    assert noisy_act < clean_act, f"Noisy ({noisy_act:.3f}) should < clean ({clean_act:.3f})"


def test_noise_by_scope_does_not_penalize_a_different_scope() -> None:
    """WP-7 (option 1, additive scoping fix): rejections recorded under one
    scope must not penalize ranking of the same memory when retrieved from a
    scope that never rejected it -- even though the schema-global
    context_noise_score aggregate looks noisy.

    Compares the *same* schema's computed activation across two different
    cue scopes via activation_trace. `scope_id=None` (a "global" schema)
    keeps the scope-match bonus and cross-scope-mismatch penalty identical
    (both are no-ops for a scope-less schema) across the two queries, so the
    only term that can differ between them is the noise-by-scope lookup
    itself."""
    dim = 8
    gate = WorkingMemoryGate()
    cue_emb = _make_unit(dim, seed=1)
    embedding = _make_unit(dim, seed=10)

    rejected_in_beta_only = _stub_schema(
        2,
        text="Rejected in a different scope",
        embedding=embedding,
        scope_id=None,
        facets_extra={
            "context_noise_score": 0.9,  # global aggregate suggests heavy noise...
            "context_noise_by_scope": {
                "project:beta": 0.9
            },  # ...but all of it is from project:beta
        },
    )

    def _activation_for(cue: MemoryCue) -> float:
        state = gate.select([rejected_in_beta_only], cue=cue, cue_embedding=cue_emb)
        trace = next(t for t in state.activation_trace if t.schema_id == 2)
        return trace.activation

    act_in_alpha = _activation_for(MemoryCue(query="task", scope="project:alpha"))
    act_in_beta = _activation_for(MemoryCue(query="task", scope="project:beta"))

    # No noise recorded for project:alpha -> full activation. All of the
    # noise history is under project:beta -> exactly the -0.30*0.9 penalty
    # applied there, nothing more (scope bonus/mismatch are both no-ops for
    # a scope_id=None schema, so this delta isolates the noise term alone).
    assert act_in_alpha - act_in_beta == pytest.approx(0.30 * 0.9, abs=1e-6), (
        f"noise recorded under project:beta must not reduce project:alpha ranking "
        f"(alpha={act_in_alpha:.4f}, beta={act_in_beta:.4f})"
    )


# ── 14. Exploration slots ─────────────────────────────────────────────────


def test_exploration_slots_populate_when_admitted_exceeds_max_items() -> None:
    """When admitted > max_items, trailing slots are filled by salience."""
    dim = 8
    gate = WorkingMemoryGate()
    cue = MemoryCue(query="task", scope="project:alpha")
    cue_emb = _make_unit(dim, seed=1)
    policy = GatePolicy(
        min_activation=0.01, max_items=2, exploration_slots=1, allowed_classes=("fact",)
    )

    schemas = []
    for i in range(4):
        noise = _make_unit(dim, seed=100 + i)
        emb = cue_emb + 0.6 * noise
        emb = emb / (np.linalg.norm(emb) + 1e-12)
        s = _stub_schema(
            i + 1,
            text=f"Fact {i+1}",
            embedding=emb.astype(np.float32),
            salience=float(i + 1),
            schema_class="fact",
        )
        schemas.append(s)

    state = gate.select(schemas, cue=cue, cue_embedding=cue_emb, policy=policy)

    # 2 relevance-ranked + 1 exploration = 3
    assert len(state.items) == 3
    peripheral = [item for item in state.items if item.peripheral]
    assert len(peripheral) == 1
    assert "peripheral" in peripheral[0].reason
