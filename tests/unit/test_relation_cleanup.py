"""Regression tests for the 2026-07-14 dead-relation cleanup, plus the
2026-07-15 taxonomy update that reintroduces "relates_to" as a distinct,
actively-used relation (not a revival of the old "related_to" fallback --
that spelling stays dead).

"contradicts" and "related_to" were removed from VALID_RELATIONS: both sat at
0 edges in production (contradicts required an exact time_delta_s<=0 tie that
every call site now records as "supersedes" too; related_to was only ever
add_relation()'s own silent fallback for an invalid relation string, never
triggered by a real caller). See schema_store.py's VALID_RELATIONS comment.
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from slowave.core.consolidation import Consolidator
from slowave.latent.schema import GeometricVerdict
from slowave.symbolic.schema_store import VALID_RELATIONS, SchemaStore

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SCHEMA_PATH = str(REPO_ROOT / "slowave" / "storage" / "schema.sql")
DIM = 8


@pytest.fixture()
def store():
    db_path = str(Path(tempfile.mkdtemp()) / "test.db")
    from slowave.storage.sqlite_db import SQLiteConfig, SQLiteDB

    db = SQLiteDB(SQLiteConfig(path=db_path))
    db.init_schema(SCHEMA_PATH)
    conn = db.connect()
    conn.execute("PRAGMA foreign_keys = OFF")
    s = SchemaStore(db, dim=DIM)
    yield s
    db.close()


def test_valid_relations_matches_current_taxonomy():
    assert "contradicts" not in VALID_RELATIONS
    assert "related_to" not in VALID_RELATIONS  # old dead fallback spelling
    assert "relates_to" in VALID_RELATIONS  # reintroduced 2026-07-15, distinct spelling
    assert "part_of" not in VALID_RELATIONS  # removed 2026-07-23, see
    # private/docs/iterations/20260723_part_of_audit_and_brain_alignment_review.md
    assert set(VALID_RELATIONS) == {"relates_to"}


def test_add_relation_raises_on_invalid_relation(store):
    emb = np.ones(DIM, dtype=np.float32) / np.sqrt(DIM)
    id_a = store.create(content_text="A", embedding=emb, dedupe=False)
    id_b = store.create(content_text="B", embedding=emb, dedupe=False)

    with pytest.raises(ValueError):
        store.add_relation(src_schema_id=id_a, dst_schema_id=id_b, relation="contradicts")
    with pytest.raises(ValueError):
        store.add_relation(src_schema_id=id_a, dst_schema_id=id_b, relation="related_to")
    # Contrast case: "relates_to" (new spelling) is a valid relation and must
    # NOT be rejected, despite looking almost identical to the dead fallback.
    store.add_relation(src_schema_id=id_a, dst_schema_id=id_b, relation="relates_to")


def test_add_relation_still_accepts_valid_relations(store):
    emb = np.ones(DIM, dtype=np.float32) / np.sqrt(DIM)
    id_a = store.create(content_text="A", embedding=emb, dedupe=False)
    id_b = store.create(content_text="B", embedding=emb, dedupe=False)

    for relation in VALID_RELATIONS:
        store.add_relation(src_schema_id=id_a, dst_schema_id=id_b, relation=relation)

    rows = (
        store.db.connect()
        .execute(
            "SELECT relation FROM schema_relations WHERE src_schema_id=? AND dst_schema_id=?",
            (id_a, id_b),
        )
        .fetchall()
    )
    assert {r["relation"] for r in rows} == set(VALID_RELATIONS)


# ---------------------------------------------------------------------------
# _link_schemas_via_prototype_centroid: regression tests for the confirmed
# production false positive (schema 153 linked to unrelated schema 154 at
# confidence 1.00). The centroid-proximity linker used to write an
# unconditional "reinforces" edge from each pair's similarity to the shared
# prototype centroid alone, without ever comparing the two schemas to EACH
# OTHER -- "both near the same reference point" does not imply "near each
# other". The fix makes it call the real geometric judge on the pair
# directly and dispatch on the judge's actual verdict.
# ---------------------------------------------------------------------------


def _consolidator_with_mocked_judge(store: SchemaStore, verdict: str) -> Consolidator:
    judge = MagicMock()
    judge.judge.return_value = GeometricVerdict(
        verdict=verdict,
        reasoning="test",
        similarity=0.9,
        facet_distance=0.0,
        time_delta_s=0,
    )
    return Consolidator(
        db=store.db,
        semantic=MagicMock(),
        episode_text=MagicMock(),
        schemas=store,
        encoder=None,
        latent_builder=MagicMock(),
        geometric_judge=judge,
    )


def _seed_centroid_pair(store: SchemaStore) -> tuple[int, int, np.ndarray]:
    """Two schemas both close enough to a shared centroid to clear the
    linker's 0.65/0.60 proximity gate."""
    rng = np.random.default_rng(0)
    centroid = rng.standard_normal(DIM).astype(np.float32)
    centroid /= np.linalg.norm(centroid)
    emb_a = centroid + rng.standard_normal(DIM).astype(np.float32) * 0.05
    emb_a /= np.linalg.norm(emb_a)
    emb_b = centroid + rng.standard_normal(DIM).astype(np.float32) * 0.05
    emb_b /= np.linalg.norm(emb_b)
    id_a = store.create(content_text="schema A", embedding=emb_a, dedupe=False)
    id_b = store.create(content_text="schema B", embedding=emb_b, dedupe=False)
    return id_a, id_b, centroid


def test_centroid_linker_calls_judge_not_unconditional_reinforces(store):
    """The judge disagrees ('relates_to') with what the old unconditional
    behavior would have written ('reinforces') -- the linker must defer to
    the judge, not the centroid-proximity gate alone. Direct regression test
    for the 153->154 production false positive."""
    id_a, id_b, centroid = _seed_centroid_pair(store)
    cons = _consolidator_with_mocked_judge(store, verdict="relates_to")

    cons._link_schemas_via_prototype_centroid(1, centroid)

    rows = (
        store.db.connect()
        .execute(
            "SELECT relation FROM schema_relations WHERE src_schema_id IN (?, ?) "
            "AND dst_schema_id IN (?, ?)",
            (id_a, id_b, id_a, id_b),
        )
        .fetchall()
    )
    assert [r["relation"] for r in rows] == ["relates_to"]


def test_centroid_linker_downgrades_reinforces_to_relates_to(store):
    """reinforces is now a directional relation. The centroid linker has no
    consolidation-time notion of which schema is new, so like all directional
    verdicts, reinforces from the judge is downgraded to relates_to."""
    id_a, id_b, centroid = _seed_centroid_pair(store)
    cons = _consolidator_with_mocked_judge(store, verdict="reinforces")

    cons._link_schemas_via_prototype_centroid(1, centroid)

    rows = (
        store.db.connect()
        .execute(
            "SELECT relation FROM schema_relations WHERE src_schema_id IN (?, ?) "
            "AND dst_schema_id IN (?, ?)",
            (id_a, id_b, id_a, id_b),
        )
        .fetchall()
    )
    assert [r["relation"] for r in rows] == ["relates_to"]


def test_centroid_linker_writes_nothing_when_unrelated(store):
    id_a, id_b, centroid = _seed_centroid_pair(store)
    cons = _consolidator_with_mocked_judge(store, verdict="unrelated")

    cons._link_schemas_via_prototype_centroid(1, centroid)

    rows = (
        store.db.connect()
        .execute(
            "SELECT relation FROM schema_relations WHERE src_schema_id IN (?, ?) "
            "AND dst_schema_id IN (?, ?)",
            (id_a, id_b, id_a, id_b),
        )
        .fetchall()
    )
    assert rows == []


# ---------------------------------------------------------------------------
# add_relation: relates_to is the only relation type left, and it's
# symmetric -- "A->B" and "B->A" are the same fact, not a contradiction.
# The directional-relation reverse-edge guard (originally covering refines,
# supersedes, part_of) was removed 2026-07-23 along with part_of, the last
# directional relation -- see
# private/docs/iterations/20260723_part_of_audit_and_brain_alignment_review.md.
# ---------------------------------------------------------------------------


def test_add_relation_relates_to_not_blocked_by_guard(store):
    emb = np.ones(DIM, dtype=np.float32) / np.sqrt(DIM)
    id_a = store.create(content_text="A", embedding=emb, dedupe=False)
    id_b = store.create(content_text="B", embedding=emb, dedupe=False)

    store.add_relation(src_schema_id=id_a, dst_schema_id=id_b, relation="relates_to")
    # relates_to is the only symmetric relation — reverse write must not raise.
    store.add_relation(src_schema_id=id_b, dst_schema_id=id_a, relation="relates_to")


# -- co-activation (Phase 2) tests -------------------------------------------


def test_coactivation_write_and_read(store):
    """upsert_coactivation writes directional edge; get_coactivations reads from either side."""
    emb = np.ones(DIM, dtype=np.float32) / np.sqrt(DIM)
    id_a = store.create(content_text="A", embedding=emb, dedupe=False)
    id_b = store.create(content_text="B", embedding=emb, dedupe=False)

    now = int(time.time())
    store.upsert_coactivation(id_a, id_b, now_ts=now)

    # Read from src side
    edges_a = store.get_coactivations(id_a)
    assert len(edges_a) == 1
    assert edges_a[0] == (id_b, "coactivated_with", 1.0)

    # Read from dst side (via UNION ALL)
    edges_b = store.get_coactivations(id_b)
    assert len(edges_b) == 1
    assert edges_b[0] == (id_a, "coactivated_with", 1.0)

    # Self-loop is silently skipped
    store.upsert_coactivation(id_a, id_a, now_ts=now)
    edges_a2 = store.get_coactivations(id_a)
    assert len(edges_a2) == 1  # unchanged


def test_coactivation_hebbian_update(store):
    """Repeated upserts strengthen via Hebbian w_new = w_old * decay + 1.0."""
    emb = np.ones(DIM, dtype=np.float32) / np.sqrt(DIM)
    id_a = store.create(content_text="A", embedding=emb, dedupe=False)
    id_b = store.create(content_text="B", embedding=emb, dedupe=False)

    t0 = 1_000_000
    # First write: weight = 1.0
    store.upsert_coactivation(id_a, id_b, now_ts=t0)

    # Second write at same time (dt=0 -> decay=1.0): weight = 1.0 + 1.0 = 2.0
    store.upsert_coactivation(id_a, id_b, now_ts=t0)
    edges = store.get_coactivations(id_a)
    assert edges[0][2] == pytest.approx(2.0, abs=0.01)

    # Third write after exactly one half-life: decay = 0.5
    t1 = t0 + 604800
    # weight = 2.0 * 0.5 + 1.0 = 2.0
    store.upsert_coactivation(id_a, id_b, now_ts=t1)
    edges = store.get_coactivations(id_a)
    assert edges[0][2] == pytest.approx(2.0, abs=0.01)

    # After two more half-lives without touch
    t2 = t1 + 604800 * 2
    # decay = exp(-ln(2) * 1209600 / 604800) = 0.25
    # weight = 2.0 * 0.25 + 1.0 = 1.5
    store.upsert_coactivation(id_a, id_b, now_ts=t2)
    edges = store.get_coactivations(id_a)
    assert edges[0][2] == pytest.approx(1.5, abs=0.01)


def test_coactivation_decay_and_prune(store):
    """decay_all_coactivations decays all rows; near-zero rows are pruned."""
    emb = np.ones(DIM, dtype=np.float32) / np.sqrt(DIM)
    id_a = store.create(content_text="A", embedding=emb, dedupe=False)
    id_b = store.create(content_text="B", embedding=emb, dedupe=False)
    id_c = store.create(content_text="C", embedding=emb, dedupe=False)

    t_fresh = 1_000_000
    t_old = t_fresh - 604800 * 20  # 20 half-lives ago
    store.upsert_coactivation(id_a, id_b, now_ts=t_old)
    store.upsert_coactivation(id_a, id_c, now_ts=t_fresh)

    # Decay at t_fresh: the old edge should be pruned (weight <= 1e-6)
    decayed = store.decay_all_coactivations(now_ts=t_fresh)
    assert decayed > 0  # some rows were touched

    # Old edge (a->b) should be gone
    edges_a = store.get_coactivations(id_a)
    neighbor_ids = {e[0] for e in edges_a}
    assert id_b not in neighbor_ids  # pruned
    assert id_c in neighbor_ids  # still fresh
