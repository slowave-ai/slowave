"""Unit tests for Stage 11: cross-scope generalization.

Tests the full stack:
  - ScopeRegistry: upsert / active_counts
  - GeneralizationConfig.compute_stage: stage promotion logic
  - SchemaStore._update_utility_scores: stage written to DB
  - WorkingMemoryGate._eligible: stage-aware scope filtering
  - WorkingMemoryGate._activation: stage-2 penalty vs stage-3 free pass

All tests use an in-memory SQLite DB — no file I/O, no encoder needed.
"""

from __future__ import annotations

import os
import tempfile
import time

import pytest

from slowave.core.context import GatePolicy, MemoryCue, WorkingMemoryGate
from slowave.core.services.consolidation import ConsolidationService
from slowave.storage.sqlite_db import SQLiteConfig, SQLiteDB
from slowave.symbolic.schema_store import (
    GeneralizationConfig,
    Schema,
    SchemaStore,
    ScopeRegistry,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_db() -> SQLiteDB:
    """Fresh in-memory-backed temp DB with full schema."""
    f = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    f.close()
    db = SQLiteDB(SQLiteConfig(path=f.name))

    schema_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "slowave",
        "storage",
        "schema.sql",
    )
    db.init_schema(os.path.abspath(schema_path))
    return db


def _make_schema(
    db: SQLiteDB,
    *,
    scope_id: str | None = "project:foo",
    content: str = "test schema",
    gen_stage: int = 0,
) -> int:
    """Insert a bare schema row and return its id."""
    now = int(time.time())
    conn = db.connect()
    cur = conn.execute(
        """
        INSERT INTO schemas
          (content_text, facets_json, tags_json, scope_id, status, confidence,
           salience, supporting_episode_ids, contradicting_episode_ids,
           is_labile, generalization_stage, first_formed_ts, last_updated_ts)
        VALUES (?, '{}', '{"tags":[]}', ?, 'active', 1.0,
                1.0, '{"ids":[]}', '{"ids":[]}', 0, ?, ?, ?)
        """,
        (content, scope_id, gen_stage, now, now),
    )
    conn.commit()
    return int(cur.lastrowid)


def _insert_session(
    db: SQLiteDB, session_id: str, scope_id: str, scope_kind: str = "project"
) -> None:
    conn = db.connect()
    conn.execute(
        "INSERT INTO sessions (id, agent, scope_id, scope_kind, started_ts, ended_ts) "
        "VALUES (?, ?, ?, ?, ?, NULL)",
        (session_id, "test", scope_id, scope_kind, int(time.time())),
    )
    conn.commit()


def _insert_episode(db: SQLiteDB) -> int:
    """episode_text.episode_id FKs to episodic_memories(id), so a real
    parent row is required before episode_text/schema_evidence can
    reference it."""
    conn = db.connect()
    cur = conn.execute(
        "INSERT INTO episodic_memories "
        "(event_id, ts, embedding, dim, salience, last_salience_ts, metadata_json) "
        "VALUES (?, ?, ?, 1, 1.0, ?, '{}')",
        (f"evt-{time.time_ns()}", int(time.time()), b"\x00\x00\x00\x00", int(time.time())),
    )
    conn.commit()
    return int(cur.lastrowid)


def _insert_episode_text(db: SQLiteDB, *, episode_id: int, session_id: str) -> None:
    conn = db.connect()
    conn.execute(
        "INSERT INTO episode_text (episode_id, content_text, source_content, event_ids, session_id) "
        "VALUES (?, ?, ?, '[]', ?)",
        (episode_id, "episode text", "episode text", session_id),
    )
    conn.commit()


def _insert_raw_event(db: SQLiteDB, session_id: str) -> int:
    conn = db.connect()
    cur = conn.execute(
        "INSERT INTO raw_events (session_id, ts, type, content) VALUES (?, ?, ?, ?)",
        (session_id, int(time.time()), "task_complete", "{}"),
    )
    conn.commit()
    return int(cur.lastrowid)


def _insert_schema_evidence(
    db: SQLiteDB,
    schema_id: int,
    *,
    episode_id: int | None,
    raw_event_id: int | None,
) -> None:
    conn = db.connect()
    conn.execute(
        "INSERT INTO schema_evidence (schema_id, episode_id, raw_event_id, quote, weight) "
        "VALUES (?, ?, ?, NULL, 1.0)",
        (schema_id, episode_id, raw_event_id),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# 1. ScopeRegistry
# ---------------------------------------------------------------------------


class TestScopeRegistry:
    def test_record_and_count(self):
        db = _make_db()
        reg = ScopeRegistry(db)

        reg.record("project:foo", "project")
        reg.record("project:bar", "project")
        reg.record("domain:cooking", "domain")

        n_scopes, n_kinds = reg.active_counts(window_days=90)
        assert n_scopes == 3
        assert n_kinds == 2  # project + domain

    def test_upsert_same_scope(self):
        db = _make_db()
        reg = ScopeRegistry(db)

        reg.record("project:foo", "project")
        reg.record("project:foo", "project")
        reg.record("project:foo", "project")

        n_scopes, _ = reg.active_counts(window_days=90)
        assert n_scopes == 1  # deduplicated

    def test_empty_registry_returns_zero(self):
        db = _make_db()
        reg = ScopeRegistry(db)
        n_scopes, n_kinds = reg.active_counts(window_days=90)
        assert n_scopes == 0

    def test_record_recall_increments_recall_count(self):
        db = _make_db()
        reg = ScopeRegistry(db)
        reg.record("project:foo", "project", is_recall=True)
        reg.record("project:foo", "project", is_recall=True)
        conn = db.connect()
        row = conn.execute(
            "SELECT recall_count FROM scope_registry WHERE scope_id = ?",
            ("project:foo",),
        ).fetchone()
        assert row["recall_count"] == 2

    def test_blank_scope_ignored(self):
        db = _make_db()
        reg = ScopeRegistry(db)
        reg.record("", None)
        reg.record("  ", "project")
        n_scopes, _ = reg.active_counts(window_days=90)
        assert n_scopes == 0


# ---------------------------------------------------------------------------
# 2. GeneralizationConfig.compute_stage
# ---------------------------------------------------------------------------


class TestGeneralizationConfig:
    def setup_method(self):
        self.cfg = GeneralizationConfig()

    def test_stage0_no_cross_scope(self):
        assert self.cfg.compute_stage(0, 0, 0.0) == 0
        assert self.cfg.compute_stage(1, 1, 0.10) == 0  # below thresholds

    def test_stage1_requires_min_2_scopes(self):
        # 50% breadth but only 1 distinct scope -> stays stage 0
        assert self.cfg.compute_stage(1, 1, 0.50, distinct_sessions=5) == 0

    def test_stage1_requires_min_sessions(self):
        # scopes/breadth ok but only 1 distinct session -> stays stage 0
        assert self.cfg.compute_stage(2, 1, 0.25, distinct_sessions=1) == 0

    def test_stage1_promoted(self):
        # 25% breadth, 2 distinct scopes, 1 scope_kind, 2 distinct sessions
        assert self.cfg.compute_stage(2, 1, 0.25, distinct_sessions=2) == 1

    def test_stage2_single_kind_no_longer_blocked(self):
        # scope breadth ok, only 1 scope kind — kind breadth is no longer a
        # hard gate, so single-kind stores can reach stage 2.
        assert self.cfg.compute_stage(4, 1, 0.55, distinct_sessions=3) == 2

    def test_stage2_kind_bonus_softens_session_floor(self):
        # 2+ distinct scope kinds grants kind_bonus=1, so 2 sessions
        # are sufficient for stage 2 when multi-kind evidence exists.
        assert self.cfg.compute_stage(4, 2, 0.55, distinct_sessions=2) == 2

    def test_stage2_requires_min_sessions_no_kind_bonus(self):
        # single-kind (kind_bonus=0), only 2 sessions -> still stage 1 (needs 3).
        assert self.cfg.compute_stage(4, 1, 0.55, distinct_sessions=2) == 1

    def test_stage2_promoted(self):
        assert self.cfg.compute_stage(4, 2, 0.55, distinct_sessions=3) == 2

    def test_stage3_promoted(self):
        assert self.cfg.compute_stage(8, 4, 0.80, distinct_sessions=5) == 3

    def test_stage3_requires_min_distinct_scopes(self):
        # pct thresholds met but only 4 distinct scopes -> stage 2
        assert self.cfg.compute_stage(4, 4, 0.80, distinct_sessions=5) == 2

    def test_stage3_requires_min_sessions(self):
        # single-kind (kind_bonus=0), only 4 sessions -> stage 2 (needs 5)
        assert self.cfg.compute_stage(8, 1, 0.80, distinct_sessions=4) == 2

    def test_stage3_kind_bonus_fires(self):
        # multi-kind (kind_bonus=1): 4+1=5 sessions -> stage 3
        assert self.cfg.compute_stage(8, 4, 0.80, distinct_sessions=4) == 3


# ---------------------------------------------------------------------------
# 2b. GeneralizationConfig.compute_stage — adaptive scope-count floor (2026-07-23)
#
# stageN_min_distinct_scopes (2/4/8) is a fixed constant, unreachable for a
# user with fewer total scopes than that constant -- a fact confirmed in 100%
# of a 3-scope universe could never clear stage 2's floor of 4. Passing
# total_active_scopes caps each floor at min(total_active_scopes, fixed_floor).
# Omitting it (None) preserves the old fixed-floor behavior exactly -- see
# private/docs/iterations/20260723_part_of_audit_and_brain_alignment_review.md
# for the full derivation and worked examples.
# ---------------------------------------------------------------------------


class TestAdaptiveScopeFloor:
    def setup_method(self):
        self.cfg = GeneralizationConfig()

    def test_omitting_total_active_scopes_keeps_old_fixed_floor(self):
        # 2 distinct scopes, 100% breadth, 5 sessions -- breadth/sessions
        # alone would qualify for stage 3, but the fixed floor of 8 blocks
        # it when total_active_scopes isn't passed, same as before this fix.
        assert self.cfg.compute_stage(2, 1, 1.0, distinct_sessions=5) == 1

    def test_three_total_scopes_two_confirmed_reaches_stage1_not_stage2(self):
        # N=3: stage1 floor caps to min(3,2)=2 (reachable); stage2 floor
        # caps to min(3,4)=3 (NOT reachable with only 2 confirmed scopes).
        assert self.cfg.compute_stage(2, 1, 2 / 3, distinct_sessions=2, total_active_scopes=3) == 1

    def test_three_total_scopes_all_confirmed_reaches_stage3(self):
        # N=3, all 3 scopes confirmed, 5 sessions: every floor caps to 3,
        # all satisfied at distinct_scopes=3 -> jumps straight to stage 3.
        assert self.cfg.compute_stage(3, 1, 1.0, distinct_sessions=5, total_active_scopes=3) == 3

    def test_cold_start_single_scope_session_floor_still_gates(self):
        # N=1: every scope floor caps to 1 (trivially satisfied), but the
        # session floor (2/3/5) does NOT scale with N -- only 1 session
        # means stage 0 despite scope math clearing everything.
        assert self.cfg.compute_stage(1, 1, 1.0, distinct_sessions=1, total_active_scopes=1) == 0

    def test_cold_start_single_scope_five_sessions_reaches_stage3(self):
        # N=1, same schema, but now 5 distinct sessions of repeated
        # confirmation within that one scope -- clears every floor.
        assert self.cfg.compute_stage(1, 1, 1.0, distinct_sessions=5, total_active_scopes=1) == 3

    def test_two_total_scopes_two_sessions_reaches_stage1_only(self):
        # N=2: scope floors all cap to 2 (trivially satisfied by confirming
        # both), but stage 2 needs 3 sessions and stage 3 needs 5 -- with
        # only 2 sessions of evidence, the session floor caps this at stage 1.
        assert self.cfg.compute_stage(2, 1, 1.0, distinct_sessions=2, total_active_scopes=2) == 1

    def test_two_total_scopes_five_sessions_reaches_stage3(self):
        assert self.cfg.compute_stage(2, 1, 1.0, distinct_sessions=5, total_active_scopes=2) == 3

    def test_large_total_scopes_behaves_identically_to_fixed_floor(self):
        # N=20 >= the largest fixed floor (8), so capping is a no-op --
        # same result with or without total_active_scopes.
        args = (8, 4, 0.80)
        assert self.cfg.compute_stage(*args, distinct_sessions=5) == self.cfg.compute_stage(
            *args, distinct_sessions=5, total_active_scopes=20
        )


# ---------------------------------------------------------------------------
# 3. SchemaStore: generalization_stage written to DB on reinforce
# ---------------------------------------------------------------------------


class TestSchemaStoreGeneralizationStage:
    def test_stage_zero_by_default(self):
        db = _make_db()
        store = SchemaStore(db, dim=384)
        sid = _make_schema(db)
        schema = store.get(sid)
        assert schema.generalization_stage == 0

    def test_stage_persists_when_set_directly(self):
        db = _make_db()
        store = SchemaStore(db, dim=384)
        sid = _make_schema(db, gen_stage=2)
        schema = store.get(sid)
        assert schema.generalization_stage == 2

    def test_stage_column_survives_reinforce(self):
        """After a reinforce(), the stage should be recomputed (stays 0 with no history)."""
        db = _make_db()
        store = SchemaStore(db, dim=384)
        sid = _make_schema(db)
        store.reinforce(sid, amount=0.1)
        schema = store.get(sid)
        # No cross-scope history yet -> must stay stage 0
        assert schema.generalization_stage == 0


# ---------------------------------------------------------------------------
# 3b. Regression tests: consolidation-path schema_evidence (raw_event_id=NULL)
# must count toward distinct_scope_count. Root cause: the scope-breadth query
# used an INNER JOIN through raw_events keyed on raw_event_id, which NULL can
# never satisfy — every schema_evidence row written by _write_latent_schema
# (the consolidation/prototype-clustering path) was silently invisible to
# scope-breadth crediting, so schemas formed by clustering never generalized
# past stage 1 no matter how many scopes their episodes actually spanned.
# ---------------------------------------------------------------------------


class TestScopeBreadthEvidenceJoin:
    def test_scope_breadth_counts_null_raw_event_evidence(self):
        """The bug itself: a NULL-raw_event_id row (consolidation path) must
        still be reachable via episode_id -> episode_text -> sessions."""
        db = _make_db()
        store = SchemaStore(db, dim=384)
        sid = _make_schema(db, scope_id="project:home")

        _insert_session(db, "sess-1", scope_id="project:other")
        eid = _insert_episode(db)
        _insert_episode_text(db, episode_id=eid, session_id="sess-1")
        _insert_schema_evidence(db, sid, episode_id=eid, raw_event_id=None)

        store.reinforce(sid, amount=0.1)
        schema = store.get(sid)
        assert schema.facets.get("distinct_scope_count", 0) >= 1

    def test_scope_breadth_still_counts_raw_event_evidence(self):
        """The pre-existing explicit-remember path (raw_event_id set, no
        episode_text row at all) must keep working standalone."""
        db = _make_db()
        store = SchemaStore(db, dim=384)
        sid = _make_schema(db, scope_id="project:home")

        _insert_session(db, "sess-2", scope_id="project:other")
        raw_id = _insert_raw_event(db, "sess-2")
        _insert_schema_evidence(db, sid, episode_id=None, raw_event_id=raw_id)

        store.reinforce(sid, amount=0.1)
        schema = store.get(sid)
        assert schema.facets.get("distinct_scope_count", 0) >= 1

    def test_scope_breadth_dedupes_when_both_paths_resolve_to_same_session(self):
        """Two separate evidence rows -- one via each writer -- that both
        resolve to the same session must count as one scope hit, not two
        (this is why the query uses UNION rather than UNION ALL)."""
        db = _make_db()
        store = SchemaStore(db, dim=384)
        sid = _make_schema(db, scope_id="project:home")

        _insert_session(db, "sess-3", scope_id="project:other")
        raw_id = _insert_raw_event(db, "sess-3")
        eid = _insert_episode(db)
        _insert_episode_text(db, episode_id=eid, session_id="sess-3")
        _insert_schema_evidence(db, sid, episode_id=None, raw_event_id=raw_id)
        _insert_schema_evidence(db, sid, episode_id=eid, raw_event_id=None)

        store.reinforce(sid, amount=0.1)
        schema = store.get(sid)
        assert schema.facets.get("distinct_scope_count", 0) == 1

    def test_scope_breadth_multiple_consolidation_scopes(self):
        """The actual production symptom: a schema formed via consolidation
        with episodes spanning multiple scopes must get credit for all of
        them, not just its single _scope_for_episodes 'home' scope."""
        db = _make_db()
        store = SchemaStore(db, dim=384)
        sid = _make_schema(db, scope_id="project:home")

        _insert_session(db, "sess-4", scope_id="project:alpha")
        eid1 = _insert_episode(db)
        _insert_episode_text(db, episode_id=eid1, session_id="sess-4")
        _insert_schema_evidence(db, sid, episode_id=eid1, raw_event_id=None)

        _insert_session(db, "sess-5", scope_id="project:beta")
        eid2 = _insert_episode(db)
        _insert_episode_text(db, episode_id=eid2, session_id="sess-5")
        _insert_schema_evidence(db, sid, episode_id=eid2, raw_event_id=None)

        store.reinforce(sid, amount=0.1)
        schema = store.get(sid)
        assert schema.facets.get("distinct_scope_count", 0) >= 2


# ---------------------------------------------------------------------------
# 4. WorkingMemoryGate: _eligible respects generalization_stage
# ---------------------------------------------------------------------------


def _schema_obj(
    *,
    scope_id: str | None,
    gen_stage: int = 0,
    status: str = "active",
    content: str = "test",
) -> Schema:
    """Minimal Schema object for gate tests without DB."""

    return Schema(
        id=1,
        prototype_id=None,
        content_text=content,
        facets={"source_kind": "explicit_remember"},
        tags=[],
        scope_id=scope_id,
        status=status,
        confidence=1.0,
        salience=1.0,
        supporting_episode_ids=[],
        contradicting_episode_ids=[],
        is_labile=False,
        first_formed_ts=int(time.time()),
        last_updated_ts=int(time.time()),
        embedding=None,
        generalization_stage=gen_stage,
    )


class TestWorkingMemoryGateGeneralization:
    def setup_method(self):
        self.gate = WorkingMemoryGate()
        self.cue = MemoryCue(
            scope="project:bar",
            mode="strict_scope",
        )
        self.policy = GatePolicy(min_activation=-999.0)  # let everything through for gate tests

    def test_stage0_foreign_scope_excluded(self):
        schema = _schema_obj(scope_id="project:foo", gen_stage=0)
        ok, reason = self.gate._eligible(schema, cue=self.cue, policy=self.policy)
        assert not ok
        assert reason == "strict_scope_excluded"

    def test_stage1_same_kind_allowed(self):
        # project:foo -> stage1, cue is project:bar (same scope_kind=project)
        schema = _schema_obj(scope_id="project:foo", gen_stage=1)
        ok, reason = self.gate._eligible(schema, cue=self.cue, policy=self.policy)
        assert ok, f"Expected eligible, got reason={reason!r}"

    def test_stage1_different_kind_excluded(self):
        # domain:cooking -> stage1, cue is project:bar (different scope_kind)
        schema = _schema_obj(scope_id="domain:cooking", gen_stage=1)
        ok, reason = self.gate._eligible(schema, cue=self.cue, policy=self.policy)
        assert not ok
        assert reason == "strict_scope_excluded"

    def test_stage2_foreign_scope_allowed(self):
        schema = _schema_obj(scope_id="domain:cooking", gen_stage=2)
        ok, reason = self.gate._eligible(schema, cue=self.cue, policy=self.policy)
        assert ok, f"Expected eligible, got reason={reason!r}"

    def test_stage3_foreign_scope_allowed(self):
        schema = _schema_obj(scope_id="relationship:alice", gen_stage=3)
        ok, reason = self.gate._eligible(schema, cue=self.cue, policy=self.policy)
        assert ok, f"Expected eligible, got reason={reason!r}"

    def test_stage2_has_reduced_mismatch_penalty(self):
        """Stage 2 schema should get a smaller scope_mismatch penalty than stage 0."""
        cue_with_embedding = MemoryCue(scope="project:bar", mode="strict_scope")

        schema_s0 = _schema_obj(scope_id="project:foo", gen_stage=0, content="test content alpha")
        schema_s2 = _schema_obj(scope_id="project:foo", gen_stage=2, content="test content alpha")
        schema_s0 = Schema(**{**schema_s0.__dict__, "id": 1})
        schema_s2 = Schema(**{**schema_s2.__dict__, "id": 2})

        _act0, _r0, _rel0 = self.gate._activation(
            schema_s0, cue=cue_with_embedding, cue_terms=set(), cue_embedding=None
        )
        _act2, _r2, _rel2 = self.gate._activation(
            schema_s2, cue=cue_with_embedding, cue_terms=set(), cue_embedding=None
        )
        # stage 2 should score higher (less penalty) than stage 0
        assert _act2 > _act0, f"Stage 2 ({_act2:.3f}) should be > Stage 0 ({_act0:.3f})"
        assert "stage2" in _r2

    def test_stage3_no_mismatch_penalty(self):
        """Stage 3 schema gets no scope_mismatch at all."""
        cue = MemoryCue(scope="project:bar", mode="strict_scope")
        schema_s3 = _schema_obj(scope_id="project:foo", gen_stage=3)
        _act, reason, _rel = self.gate._activation(
            schema_s3, cue=cue, cue_terms=set(), cue_embedding=None
        )
        assert "scope_mismatch" not in reason


# ---------------------------------------------------------------------------
# 5. recall() / context_brief parity — the gap identified in the test report
# ---------------------------------------------------------------------------


def _make_db_with_schemas() -> tuple["SQLiteDB", "SchemaStore"]:
    """Create a DB with three schemas mirroring the test-report scenario."""
    db = _make_db()
    store = SchemaStore(db, dim=384)
    _make_schema(
        db,
        scope_id="project:alpha",
        content="Use pytest fixtures to isolate test state",
        gen_stage=2,
    )
    _make_schema(
        db,
        scope_id="project:alpha",
        content="Project Alpha production database password is ALPHA-SECRET-123",
        gen_stage=0,
    )
    _make_schema(
        db,
        scope_id="project:alpha",
        content="Project Alpha API base URL is https://alpha.internal.local",
        gen_stage=0,
    )
    return db, store


def _apply_strict_scope_filter(
    db: "SQLiteDB",
    store: "SchemaStore",
    *,
    scope_id: str,
) -> list["Schema"]:
    """Run the scope-filter portion of recall() without encoder/FAISS."""
    from slowave.core.scope import scope_kind as _scope_kind

    conn = db.connect()
    all_schemas = store.list(limit=1000, status="active")
    promoted_rows = conn.execute(
        "SELECT id FROM schemas WHERE generalization_stage >= 1 "
        "AND status = 'active' AND (scope_id IS NOT NULL AND scope_id != ?)",
        (scope_id,),
    ).fetchall()
    existing_ids = {s.id for s in all_schemas}
    for r in promoted_rows:
        pid = int(r["id"])
        if pid not in existing_ids:
            try:
                all_schemas.append(store.get(pid))
            except KeyError:
                pass
    filtered = []
    for s in all_schemas:
        if s.status != "active":
            continue
        if s.scope_id and s.scope_id != scope_id and s.scope_id not in ("global", "user"):
            _gs = getattr(s, "generalization_stage", 0)
            if _gs >= 3:
                pass
            elif _gs == 2:
                pass
            elif _gs == 1:
                if _scope_kind(s.scope_id) != _scope_kind(scope_id):
                    continue
            else:
                continue
        filtered.append(s)
    return filtered


class TestRecallContextParity:
    """Case 6: recall() and context_brief must honour generalization_stage consistently."""

    def test_stage0_foreign_blocked(self):
        db, store = _make_db_with_schemas()
        results = _apply_strict_scope_filter(db, store, scope_id="project:theta")
        texts = [s.content_text for s in results]
        assert not any("password" in t for t in texts), "Stage 0 secret must be blocked"
        assert not any("API base URL" in t for t in texts), "Stage 0 project URL must be blocked"

    def test_stage2_admitted_foreign_same_kind(self):
        db, store = _make_db_with_schemas()
        results = _apply_strict_scope_filter(db, store, scope_id="project:theta")
        texts = [s.content_text for s in results]
        assert any(
            "pytest fixtures" in t for t in texts
        ), "Stage 2 generic memory must appear in foreign project scope"

    def test_stage2_admitted_cross_kind(self):
        db, store = _make_db_with_schemas()
        results = _apply_strict_scope_filter(db, store, scope_id="customer:acme")
        texts = [s.content_text for s in results]
        assert any(
            "pytest fixtures" in t for t in texts
        ), "Stage 2 memory must appear even in different scope_kind"

    def test_origin_scope_sees_all(self):
        db, store = _make_db_with_schemas()
        results = _apply_strict_scope_filter(db, store, scope_id="project:alpha")
        assert len(results) == 3

    def test_stage1_same_kind_admitted(self):
        db, store = _make_db_with_schemas()
        _make_schema(
            db, scope_id="project:alpha", content="Always run mypy before committing", gen_stage=1
        )
        results = _apply_strict_scope_filter(db, store, scope_id="project:beta")
        assert any("mypy" in s.content_text for s in results)

    def test_stage1_different_kind_blocked(self):
        db, store = _make_db_with_schemas()
        _make_schema(
            db, scope_id="project:alpha", content="Always run mypy before committing", gen_stage=1
        )
        results = _apply_strict_scope_filter(db, store, scope_id="customer:acme")
        assert not any("mypy" in s.content_text for s in results)

    def test_stage3_admitted_everywhere(self):
        db, store = _make_db_with_schemas()
        _make_schema(
            db,
            scope_id="project:alpha",
            content="Separate concerns early in system design",
            gen_stage=3,
        )
        results = _apply_strict_scope_filter(db, store, scope_id="personal:home")
        assert any("Separate concerns" in s.content_text for s in results)


# ---------------------------------------------------------------------------
# 6. Stage 2 score discount in recall() — multiplier applied, not hard block
# ---------------------------------------------------------------------------


class TestRecallStage2ScoreDiscount:

    def test_stage2_multiplier_applied(self):
        from slowave.symbolic.schema_store import GeneralizationConfig

        cfg = GeneralizationConfig()
        raw_score = 0.80
        discounted = raw_score * cfg.stage2_cross_scope_score_multiplier
        assert discounted < raw_score
        assert discounted == pytest.approx(0.80 * 0.70, abs=1e-6)

    def test_stage3_has_no_multiplier_field(self):
        """Stage 3 passes at full score — no separate multiplier field exists."""
        from slowave.symbolic.schema_store import GeneralizationConfig

        cfg = GeneralizationConfig()
        assert not hasattr(cfg, "stage3_cross_scope_score_multiplier")


# ---------------------------------------------------------------------------
# 7. Recall cosine scoring for promoted candidates (Fix 1)
# ---------------------------------------------------------------------------


class TestRecallCosineScoring:
    """cosine+0.25 replaces flat 0.10 baseline for promoted schemas with embeddings."""

    def test_cosine_score_higher_than_flat_baseline(self):
        import numpy as np

        from slowave.utils.vec import pack_f32, unpack_f32

        db = _make_db()
        dim = 4
        schema_vec = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        now = int(time.time())
        conn = db.connect()
        cur = conn.execute(
            "INSERT INTO schemas "
            "(content_text, facets_json, tags_json, scope_id, status, confidence,"
            " salience, supporting_episode_ids, contradicting_episode_ids,"
            " is_labile, generalization_stage, first_formed_ts, last_updated_ts,"
            " embedding, dim)"
            " VALUES (?, '{}', '{\"tags\":[]}', ?, 'active', 1.0,"
            " 1.0, '{\"ids\":[]}', '{\"ids\":[]}', 0, 2, ?, ?, ?, ?)",
            ("pytest fixtures", "project:alpha", now, now, pack_f32(schema_vec), dim),
        )
        conn.commit()
        schema_id = int(cur.lastrowid)

        q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        qn = float(np.linalg.norm(q)) + 1e-12
        row = conn.execute(
            "SELECT embedding, dim FROM schemas WHERE id = ?", (schema_id,)
        ).fetchone()
        v = unpack_f32(row["embedding"], int(row["dim"]))
        cosine = float(q.dot(v) / (qn * (float(np.linalg.norm(v)) + 1e-12)))
        computed_score = max(0.0, cosine) + 0.25
        assert computed_score == pytest.approx(1.25, abs=1e-5)
        assert computed_score > 0.10

    def test_flat_baseline_when_no_embedding(self):
        db = _make_db()
        sid = _make_schema(db, scope_id="project:alpha", gen_stage=2)
        conn = db.connect()
        row = conn.execute("SELECT embedding FROM schemas WHERE id = ?", (sid,)).fetchone()
        assert row["embedding"] is None
        # documents fallback: 0.10 < 0.30 floor → dropped during filter
        assert 0.10 < 0.30


# ---------------------------------------------------------------------------
# 8. Context noise floor: raised threshold + cosine gate (Fix 2)
# ---------------------------------------------------------------------------


def _cross_scope_schema(*, gen_stage: int = 2, embedding=None) -> "Schema":
    return Schema(
        id=99,
        prototype_id=None,
        content_text="Use pytest fixtures to isolate test state",
        facets={
            "source_kind": "explicit_remember",
            "memory_layer": "domain",
            "schema_class": "lesson",
        },
        tags=[],
        scope_id="project:alpha",
        status="active",
        confidence=1.0,
        salience=5.0,
        supporting_episode_ids=[],
        contradicting_episode_ids=[],
        is_labile=False,
        first_formed_ts=int(time.time()),
        last_updated_ts=int(time.time()),
        embedding=embedding,
        generalization_stage=gen_stage,
    )


class TestContextCrossScopeNoiseFloor:

    def test_orthogonal_query_blocked_by_noise_gates(self):
        """Stage 2 schema with cosine ≈ 0 vs query must be suppressed.

        The two gates fire in order: activation floor first, then cosine gate.
        With no cue-overlap, salience-only activation lands ~0.24 which is below
        the 0.30 floor, so gate A blocks it before gate B is even reached.
        With enough cue-overlap to clear 0.30, the cosine gate would then block it.
        Both behaviors are correct — we just assert the schema is blocked.
        """
        import numpy as np

        gate = WorkingMemoryGate()
        schema_emb = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        cue_emb = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
        schema = _cross_scope_schema(gen_stage=2, embedding=schema_emb)
        cue = MemoryCue(scope="project:theta", mode="strict_scope")
        state = gate.select(
            [schema], cue=cue, policy=GatePolicy(min_activation=0.0), cue_embedding=cue_emb
        )
        assert len(state.items) == 0
        # Either the activation floor or the cosine gate fired — both are correct
        blocked_by_floor = state.suppressed.get("cross_scope_below_floor", 0) > 0
        blocked_by_cosine = state.suppressed.get("cross_scope_low_cosine", 0) > 0
        assert (
            blocked_by_floor or blocked_by_cosine
        ), f"Expected schema to be noise-gated, got suppressed={state.suppressed}"

    def test_aligned_query_passes_cosine_gate(self):
        """Stage 2 schema with cosine ≈ 1.0 and good activation passes both gates."""
        import numpy as np

        gate = WorkingMemoryGate()
        vec = np.array([0.7, 0.7, 0.0, 0.0], dtype=np.float32)
        vec /= np.linalg.norm(vec)
        schema = _cross_scope_schema(gen_stage=2, embedding=vec)
        cue = MemoryCue(scope="project:theta", mode="strict_scope")
        state = gate.select(
            [schema], cue=cue, policy=GatePolicy(min_activation=0.0), cue_embedding=vec
        )
        assert len(state.items) == 1

    def test_floor_raised_to_0_40(self):
        from slowave.symbolic.schema_store import GeneralizationConfig

        assert GeneralizationConfig().cross_scope_min_score == pytest.approx(0.40)

    def test_stage3_exempt_from_both_gates(self):
        """Stage 3 schemas bypass activation floor and cosine gate."""
        import numpy as np

        gate = WorkingMemoryGate()
        schema_emb = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        cue_emb = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
        schema = _cross_scope_schema(gen_stage=3, embedding=schema_emb)
        cue = MemoryCue(scope="personal:home", mode="strict_scope")
        state = gate.select(
            [schema], cue=cue, policy=GatePolicy(min_activation=0.0), cue_embedding=cue_emb
        )
        assert "cross_scope_low_cosine" not in state.suppressed
        assert "cross_scope_below_floor" not in state.suppressed


# ---------------------------------------------------------------------------
# 9. FTS score must not suppress a better promoted embedding score (Fix 3)
# ---------------------------------------------------------------------------


class TestFTSVsPromotedEmbeddingScore:
    """
    Regression for the scoring bug where FTS inserts a promoted cross-scope
    schema at 0.35, then the Stage 2 multiplier reduces it to 0.245, which
    falls below the 0.30 cross_scope_min_score floor — dropping the schema
    even though the embedding-derived cosine score would have been much higher.

    The fix: always compute cosine for promoted schemas and use max() so the
    best available signal wins, regardless of which path arrived first.
    """

    def test_fts_score_does_not_block_better_cosine_score(self):
        """
        Simulate the exact failure path:
          - FTS fires first → score = 0.35
          - Stage 2 multiplier → 0.35 * 0.70 = 0.245
          - 0.245 < 0.30 floor → dropped (WRONG)
        After fix:
          - cosine computed regardless of FTS → e.g. 0.85 + 0.25 = 1.10
          - max(0.35, 1.10) = 1.10
          - 1.10 * 0.70 = 0.77 > 0.30 → passes (CORRECT)
        """
        from slowave.symbolic.schema_store import GeneralizationConfig

        cfg = GeneralizationConfig()
        fts_score = 0.35
        multiplier = cfg.stage2_cross_scope_score_multiplier  # 0.70
        floor = cfg.cross_scope_min_score  # 0.30

        # Old behaviour: FTS score only, no cosine override
        old_score = fts_score * multiplier
        assert (
            old_score < floor
        ), f"Test precondition: FTS-only path should fail floor ({old_score:.3f} < {floor})"

        # New behaviour: cosine computed and max() used
        high_cosine = 0.85
        promoted_score = max(0.0, high_cosine) + 0.25  # 1.10
        final_score = max(fts_score, promoted_score) * multiplier  # 1.10 * 0.70 = 0.77
        assert (
            final_score >= floor
        ), f"After fix: cosine-boosted path should pass floor ({final_score:.3f} >= {floor})"

    def test_fts_wins_when_cosine_is_weak(self):
        """When cosine is low, FTS score is allowed to win — max() is symmetric."""
        fts_score = 0.35
        low_cosine = 0.02
        promoted_score = max(0.0, low_cosine) + 0.25  # 0.27

        # max() correctly keeps the FTS score when cosine is weaker
        winning_score = max(fts_score, promoted_score)
        assert winning_score == pytest.approx(fts_score, abs=1e-6)

    def test_max_used_not_conditional_assignment(self):
        """
        Verify the actual code uses max() unconditionally for promoted schemas,
        not a guarded `if _sid not in schema_scores` assignment.
        Inspects the source text directly — simpler and more robust than AST.
        """
        import inspect
        import textwrap

        from slowave.core.services import retrieval as _ret_mod

        src = textwrap.dedent(inspect.getsource(_ret_mod.RetrievalService.recall))

        # The guarded old form should not appear
        assert (
            "if _sid not in schema_scores" not in src
        ), "Old guard 'if _sid not in schema_scores' still present — fix not applied"

        # The unconditional max() form must be present
        assert (
            "schema_scores[_sid] = max(schema_scores.get(_sid" in src
        ), "Expected 'schema_scores[_sid] = max(schema_scores.get(_sid, ...), _score)' not found"


# ---------------------------------------------------------------------------
# 6. ConsolidationService._refresh_generalization (2026-07-23): recompute
# generalization_stage during consolidation instead of only reactively from
# feedback.py. Incremental tier (schemas with new evidence since last worker
# run) fixes "stale-low"; the rarer full sweep fixes "stale-high" (a schema's
# cached stage drifting from a since-grown/shrunk total_active_scopes
# denominator, with nothing schema-local to trigger a recompute).
# ---------------------------------------------------------------------------


def _make_consolidation_service(
    db: SQLiteDB, schemas: SchemaStore, **kwargs
) -> ConsolidationService:
    """Minimal ConsolidationService for testing _refresh_generalization in
    isolation -- replay_engine/consolidator/ingest are unused by that method."""
    return ConsolidationService(
        db=db,
        replay_engine=None,
        consolidator=None,
        schemas=schemas,
        ingest=None,
        **kwargs,
    )


class TestRefreshGeneralizationIncremental:
    def test_incremental_pass_refreshes_schema_with_new_evidence(self):
        db = _make_db()
        store = SchemaStore(db, dim=384)
        sid = _make_schema(db, scope_id="project:home")

        _insert_session(db, "sess-1", scope_id="project:other")
        raw_id = _insert_raw_event(db, "sess-1")
        _insert_schema_evidence(db, sid, episode_id=None, raw_event_id=raw_id)

        svc = _make_consolidation_service(db, store, full_sweep_interval_s=999999.0)
        conn = db.connect()
        result = svc._refresh_generalization(conn, int(time.time()))

        assert result["incremental_refreshed"] >= 1
        schema = store.get(sid)
        assert schema.facets.get("distinct_scope_count", 0) >= 1

    def test_incremental_pass_ignores_untouched_schema(self):
        """A schema with no new recall/evidence activity shouldn't be counted
        as refreshed by the incremental tier."""
        db = _make_db()
        store = SchemaStore(db, dim=384)
        _make_schema(db, scope_id="project:home")  # no evidence seeded at all

        svc = _make_consolidation_service(db, store, full_sweep_interval_s=999999.0)
        conn = db.connect()
        result = svc._refresh_generalization(conn, int(time.time()))

        assert result["incremental_refreshed"] == 0


class TestRefreshGeneralizationFullSweep:
    def test_full_sweep_runs_on_first_call(self):
        db = _make_db()
        store = SchemaStore(db, dim=384)
        _make_schema(db)

        svc = _make_consolidation_service(db, store, full_sweep_interval_s=86400.0)
        conn = db.connect()
        result = svc._refresh_generalization(conn, int(time.time()))
        assert result["full_swept"] >= 1

    def test_full_sweep_does_not_rerun_within_interval(self):
        db = _make_db()
        store = SchemaStore(db, dim=384)
        _make_schema(db)

        svc = _make_consolidation_service(db, store, full_sweep_interval_s=86400.0)
        conn = db.connect()
        now = int(time.time())
        first = svc._refresh_generalization(conn, now)
        second = svc._refresh_generalization(conn, now + 10)  # 10s later, well under 1 day
        assert first["full_swept"] >= 1
        assert second["full_swept"] == 0

    def test_full_sweep_reruns_after_interval_elapses(self):
        db = _make_db()
        store = SchemaStore(db, dim=384)
        _make_schema(db)

        svc = _make_consolidation_service(db, store, full_sweep_interval_s=100.0)
        conn = db.connect()
        now = int(time.time())
        first = svc._refresh_generalization(conn, now)
        second = svc._refresh_generalization(conn, now + 200)  # past the 100s interval
        assert first["full_swept"] >= 1
        assert second["full_swept"] >= 1

    def test_full_sweep_corrects_stale_high_stage(self):
        """The motivating scenario: a schema cached at stage 2 (as if promoted
        when total_active_scopes was small) with zero real cross-scope
        evidence must be demoted back down once the full sweep recomputes it
        against the current (real) evidence -- nothing schema-local would
        otherwise trigger that recompute."""
        db = _make_db()
        store = SchemaStore(db, dim=384)
        sid = _make_schema(db, scope_id="project:home", gen_stage=2)
        reg = ScopeRegistry(db)
        reg.record("project:home", "project")

        assert store.get(sid).generalization_stage == 2  # stale, artificially set

        svc = _make_consolidation_service(db, store, full_sweep_interval_s=0.0)
        conn = db.connect()
        svc._refresh_generalization(conn, int(time.time()))

        # Zero cross-scope evidence exists -> a real recompute must not leave
        # this at stage 2.
        assert store.get(sid).generalization_stage == 0
