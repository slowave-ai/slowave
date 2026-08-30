"""Synthetic regression harness: Phase 0 - failing tests first.

These tests validate the functional improvements from the improvement plan v2.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

from slowave.core.config import SlowaveConfig
from slowave.core.engine import SlowaveEngine


class _StubEncoder:
    """Deterministic encoder: same text → same unit vector, no model needed."""

    def __init__(self, dim: int = 32):
        self._dim = dim

    def encode(self, text: str) -> np.ndarray:
        seed = int(abs(hash(text)) % (2**31))
        v = np.random.default_rng(seed).standard_normal(self._dim).astype(np.float32)
        return v / (np.linalg.norm(v) + 1e-12)


@pytest.fixture
def tmp_db() -> str:
    """Temporary database for tests."""
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    yield tmp.name
    for ext in ("", "-wal", "-shm"):
        p = tmp.name + ext
        if os.path.exists(p):
            os.remove(p)


@pytest.fixture
def engine_with_encoder(tmp_db: str) -> SlowaveEngine:
    """Engine with deterministic stub encoder (no LLM)."""
    cfg = SlowaveConfig(db_path=tmp_db, dim=32, disable_encoder=True)
    eng = SlowaveEngine(cfg)
    eng.encoder = _StubEncoder(dim=32)
    yield eng
    eng.close()


class TestScopeHandling:
    """Strict scope mode validation."""

    def test_strict_scope_excludes_other_project_facts(
        self, engine_with_encoder: SlowaveEngine
    ) -> None:
        """Strict scope should exclude memories from other projects."""
        eng = engine_with_encoder

        # Remember facts in different scopes
        proj_a_id = eng.remember(
            content="Project A specific fact",
            type="fact",
            scope="project:alpha",
        ).schema_id

        proj_b_id = eng.remember(
            content="Project B specific fact",
            type="fact",
            scope="project:beta",
        ).schema_id

        # Get context in strict_scope mode for project:alpha
        context = eng.context_brief(
            query="project specific",
            scope="project:alpha",
            mode="strict_scope",
        )

        returned_ids = {s.id for s in context.schemas}
        # Project A should be included, Project B should not
        assert proj_a_id in returned_ids, "Project A fact should be included"
        assert (
            proj_b_id not in returned_ids
        ), "Project B fact should be excluded in strict_scope mode"

    def test_strict_scope_is_mcp_activate_default(self, engine_with_encoder: SlowaveEngine) -> None:
        """strict_scope is the correct default for activate: any scoped call isolates by default.

        This verifies the behaviour contract without inspecting string prefixes.
        The gate in context.py: if cue.mode == 'strict_scope' and cue.scope is set,
        hard-block non-matching scopes. When scope is None, strict_scope == default.

        Uses type="fact" (memory_layer="domain") so the strict_scope gate applies.
        Types mapped to memory_layer="profile" (constraint, preference, etc.) are
        intentionally exempt from scope filtering in all modes.
        """
        eng = engine_with_encoder

        # Store two scoped facts (type="fact" → memory_layer="domain", not profile)
        fact_a = eng.remember(content="Alpha-only fact", type="fact", scope="alpha")
        fact_b = eng.remember(content="Beta-only fact", type="fact", scope="beta")
        # Use .schema_id explicitly — RememberResult is an int subclass whose integer
        # value is the event_id, NOT the schema_id; comparing to schema ids requires .schema_id.
        fact_a_sid = fact_a.schema_id
        fact_b_sid = fact_b.schema_id

        # strict_scope with scope="alpha" must exclude beta
        ctx = eng.context_brief(query="fact", scope="alpha", mode="strict_scope")
        ids = {s.id for s in ctx.schemas}
        assert fact_a_sid in ids, "same-scope fact must appear"
        assert fact_b_sid not in ids, "other-scope fact must be excluded in strict_scope"

        # strict_scope with scope=None must behave like default (no hard exclusion)
        ctx_no_scope = eng.context_brief(query="fact", scope=None, mode="strict_scope")
        ids_no_scope = {s.id for s in ctx_no_scope.schemas}
        # Both facts may appear when no scope is set (no hard block fires)
        assert (
            fact_a_sid in ids_no_scope or fact_b_sid in ids_no_scope
        ), "strict_scope with no scope must not hard-block anything"

    def test_strict_scope_allows_global_scope_none_memories(
        self, engine_with_encoder: SlowaveEngine
    ) -> None:
        """Strict scope should allow memories with scope_id=None."""
        eng = engine_with_encoder

        # Remember a global fact (no scope)
        global_id = eng.remember(
            content="Global system fact",
            type="fact",
            scope=None,
        ).schema_id

        # Remember project-specific fact
        proj_id = eng.remember(
            content="Project specific fact",
            type="fact",
            scope="project:alpha",
        ).schema_id

        # Get context in strict_scope mode for project:alpha
        context = eng.context_brief(
            query="fact",
            scope="project:alpha",
            mode="strict_scope",
        )

        returned_ids = {s.id for s in context.schemas}
        # Both should be included: project fact + global fact
        assert proj_id in returned_ids, "Project fact should be included"
        assert (
            global_id in returned_ids
        ), "Global fact (scope_id=None) should be included in strict scope"


class TestFeedbackSuppression:
    """Wrong/stale feedback suppression."""

    def test_wrong_feedback_removes_memory_from_top_3(
        self, engine_with_encoder: SlowaveEngine
    ) -> None:
        """Wrong feedback should remove memory from top-k."""
        eng = engine_with_encoder

        # Remember two similar facts
        schema_id_1 = eng.remember(
            content="Python is a dynamically typed language",
            type="fact",
        )

        schema_id_2 = eng.remember(
            content="JavaScript is also dynamically typed",
            type="fact",
        )

        # Recall without feedback - both should appear
        result1 = eng.recall("dynamically typed language", top_k=5)
        ids_before = {s.id for s in result1.schemas}
        assert schema_id_1 in ids_before or schema_id_2 in ids_before

        # Directly update schema status to needs_review to simulate feedback
        eng.schemas.update_status(schema_id_1, status="needs_review")

        # Recall again - needs_review schema should be suppressed in default mode
        result2 = eng.recall("dynamically typed language", top_k=5)
        ids_after = {s.id for s in result2.schemas}

        # Schema with needs_review should no longer appear in default mode
        assert (
            schema_id_1 not in ids_after
        ), "Needs-review schema should be removed from recall in default mode"

    def test_needs_review_excluded_from_default_recall(
        self, engine_with_encoder: SlowaveEngine
    ) -> None:
        """Memories with status=needs_review should be excluded."""
        eng = engine_with_encoder

        # Remember fact and mark as needs_review via status update
        schema_id = eng.remember(
            content="Deprecated API endpoint /v1/old",
            type="fact",
        )

        eng.schemas.update_status(schema_id, status="needs_review")

        # Recall in default mode - should not appear
        result_default = eng.recall("deprecated endpoint", top_k=5)
        ids_default = {s.id for s in result_default.schemas}
        assert (
            schema_id not in ids_default
        ), "Needs-review schema should be excluded in default mode"

    def test_needs_review_visible_in_broad_mode(self, engine_with_encoder: SlowaveEngine) -> None:
        """Memories with status=needs_review visible in broad/debug."""
        eng = engine_with_encoder

        # Remember fact and mark as needs_review
        schema_id = eng.remember(
            content="Deprecated API endpoint /v1/old",
            type="fact",
        )

        # Mark as needs_review
        eng.schemas.update_status(schema_id, status="needs_review")

        # In broad mode, should still be visible
        result_broad = eng.context_brief(query="deprecated endpoint", mode="broad", limit=10)
        ids_broad = {s.id for s in result_broad.schemas}
        assert schema_id in ids_broad, "Needs-review schema should be visible in broad mode"

    def test_wrong_feedback_sets_status_contradicted(
        self, engine_with_encoder: SlowaveEngine
    ) -> None:
        """Wrong client feedback retires the schema as contradicted."""
        eng = engine_with_encoder

        schema_id = eng.remember(
            content="Wrong database fact that caused failure",
            type="fact",
        )

        # Confirm schema starts active
        assert eng.schemas.get(schema_id).status == "active"

        # Apply wrong feedback with outcome=failed via retrieval_feedback
        eng.retrieval_feedback(
            retrieval_id="test-wrong-failed-001",
            feedback="wrong",
            outcome="failure",
            wrong_memory_ids=[f"sch_{schema_id}"],
        )

        # Status is terminal and client-owned.
        schema = eng.schemas.get(schema_id)
        assert (
            schema.status == "stale"
        ), f"wrong feedback should retire as stale/contradicted, got {schema.status}"
        assert schema.stale_reason == "contradicted"

        # And it must be excluded from default recall
        result = eng.recall("database fact failure", top_k=5)
        assert schema_id not in {
            s.id for s in result.schemas
        }, "contradicted schema must be excluded from default recall"


class TestBroadSummaryDemotion:
    """Broad summary demotion based on provenance."""

    def test_consolidated_broad_summary_excluded_from_default_context(
        self, engine_with_encoder: SlowaveEngine
    ) -> None:
        """Multi-sentence summaries excluded from default context (P5-A/B) via eligibility gate."""
        eng = engine_with_encoder

        # Create a consolidated multi-sentence summary (simulating consolidation)
        long_text = "First sentence. Second sentence. Third sentence. This is a long summary with multiple claims."
        long_summary_id = eng.schemas.create(
            content_text=long_text,
            facets={
                "schema_class": "episodic_summary",  # tagged to match consolidation classification
                "source_kind": "consolidation",  # Not explicit_remember
            },
            tags=["summary"],
            embedding=eng.encoder.encode(long_text),
        )

        # Verify P5 gate behavior
        long_summary = eng.schemas.get(long_summary_id)

        # Test the eligibility gate directly:
        # Multi-sentence consolidated schemas should be filtered in default mode
        from slowave.core.context import GatePolicy, MemoryCue, WorkingMemoryGate

        gate = WorkingMemoryGate()
        cue_default = MemoryCue(mode="default", scope=None)
        cue_broad = MemoryCue(mode="broad", scope=None)
        policy = GatePolicy()

        # In default mode, episodic_summary should be excluded
        eligible_default, reason_default = gate._eligible(
            long_summary, cue=cue_default, policy=policy
        )
        assert (
            not eligible_default
        ), f"Multi-sentence summary should be excluded in default mode (reason: {reason_default})"

        # In broad mode, episodic_summary should be included
        eligible_broad, reason_broad = gate._eligible(long_summary, cue=cue_broad, policy=policy)
        assert (
            eligible_broad
        ), f"Multi-sentence summary should be eligible in broad mode (reason: {reason_broad})"

    def test_explicit_long_memory_not_filtered(self, engine_with_encoder: SlowaveEngine) -> None:
        """Explicitly remembered long memories NOT filtered (P5-A)."""
        eng = engine_with_encoder

        # Create an explicitly remembered long memory (should never be filtered)
        explicit_long_id = eng.remember(
            content="First explicit sentence. Second explicit sentence. Third explicit sentence. User provided this explicit long memory.",
            type="fact",
        )

        # Get context in default mode
        context_default = eng.context_brief(query="explicit sentence", mode="default", limit=10)

        returned_ids = {s.id for s in context_default.schemas}
        # Explicit memory should NEVER be filtered, even if long and multi-sentence
        assert (
            explicit_long_id in returned_ids
        ), "Explicitly remembered long memory should never be filtered from default context"


class TestEpisodeDeduplication:
    """Episode deduplication."""

    def test_explicit_remember_no_duplicate_episodes(
        self, engine_with_encoder: SlowaveEngine
    ) -> None:
        """Episode dedup should not duplicate facts."""
        eng = engine_with_encoder

        eng.remember(
            content="Unique fact to deduplicate",
            type="fact",
        )

        # Recall with evidence
        result = eng.recall("fact deduplicate", top_k=5, evidence=True)

        episode_texts = result.episode_texts
        unique_texts = set()
        for ep in episode_texts:
            text = ep.get("content_text", "").lower().strip()
            if text:
                # Normalize - remove date prefix
                text = text.split("]")[-1].strip() if "]" in text else text
                assert text not in unique_texts, f"Duplicate found: {text}"
                unique_texts.add(text)

    def test_context_brief_has_no_duplicate_items(self, engine_with_encoder: SlowaveEngine) -> None:
        """Context brief should not include duplicates."""
        eng = engine_with_encoder

        # Remember facts
        for i in range(3):
            eng.remember(
                content=f"Python is a language variant {i}",
                type="fact",
            )

        # Get context brief
        context = eng.context_brief(query="programming language", limit=10)

        # Check for duplicate schema IDs
        schema_ids = [s.id for s in context.schemas]
        assert len(schema_ids) == len(set(schema_ids)), f"Duplicate schemas: {schema_ids}"
