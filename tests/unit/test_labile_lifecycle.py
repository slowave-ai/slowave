"""Tests for the "labile" lifecycle introduced 2026-07-10 (see
core/08-feedback.md, core/05-consolidation.md, outcomes/08-feedback.md):
sustained recurrence restabilizing a flagged schema, and
Consolidator.reconsolidate_labile_schemas() replaying flagged schemas against
their nearest neighbor via the existing geometric contradiction judge.

tests/unit/test_feedback_review_gating.py covers the third recovery channel
(explicit useful/partially_useful feedback clearing is_labile) and the
scope_id fix.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np

from slowave.core.config import SlowaveConfig
from slowave.core.engine import SlowaveEngine

DIM = 8


def _engine() -> tuple[SlowaveEngine, str]:
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    cfg = SlowaveConfig(db_path=tmp.name, dim=DIM, disable_encoder=True)
    return SlowaveEngine(cfg), tmp.name


def _cleanup(path: str) -> None:
    for ext in ("", "-wal", "-shm"):
        p = path + ext
        if os.path.exists(p):
            os.remove(p)


def _set_last_updated_ts(eng: SlowaveEngine, schema_id: int, ts: int) -> None:
    conn = eng.db.connect()
    conn.execute("UPDATE schemas SET last_updated_ts = ? WHERE id = ?", (ts, schema_id))
    conn.commit()


class _StubManifold:
    """Fixed direction_score, for tests that need a genuine value-substitution
    signal (see GeometricContradictionJudge.judge()'s "supersedes" branch) --
    the engine fixture here runs with disable_encoder=True, so the real
    SupersessionManifold is never constructed and the judge's manifold stays
    None. Facet-axis divergence alone (no direction signal) now resolves to
    "refines"/restabilize, not "supersedes" -- these tests specifically
    exercise the supersession gates (support/recency/tie-status), so they
    inject this stub to reach that branch honestly instead of relying on the
    retired cos-and-facet-distance-only path."""

    def __init__(self, score: float) -> None:
        self._score = score

    def direction_score(self, emb_new, emb_old) -> float:
        return self._score


def _same_topic_centroids(cos_target: float) -> tuple[np.ndarray, np.ndarray]:
    a = np.zeros(DIM, dtype=np.float32)
    a[0] = 1.0
    orth = np.zeros(DIM, dtype=np.float32)
    orth[1] = 1.0
    b = cos_target * a + float(np.sqrt(1 - cos_target**2)) * orth
    b = (b / np.linalg.norm(b)).astype(np.float32)
    return a, b


def _create(
    eng: SlowaveEngine,
    *,
    text: str,
    embedding: np.ndarray,
    facet_axes: np.ndarray | None = None,
    facet_strengths: np.ndarray | None = None,
    supporting_episode_ids: list[int] | None = None,
    is_labile: bool = False,
) -> int:
    return eng.schemas.create(
        content_text=text,
        facets={},
        tags=[],
        embedding=embedding,
        confidence=0.9,
        salience=1.0,
        facet_axes=facet_axes,
        facet_strengths=facet_strengths,
        supporting_episode_ids=supporting_episode_ids or [],
        is_labile=is_labile,
        dedupe=False,
    )


class TestRecurrenceClearsLability:
    """Sustained genuine reactivation (recall_hit=True reinforcement) after
    a schema is flagged restabilizes it — a passive form of reconsolidation,
    mirroring how repeated reactivation drives real memory consolidation.
    Requires _RECONSOLIDATION_RECOVERY_RECURRENCE (3) hits *since* the flag,
    not lifetime recurrence accumulated before flagging."""

    def test_three_reinforcements_after_flag_clear_it(self) -> None:
        eng, path = _engine()
        try:
            sid = _create(eng, text="recurrence recovery schema", embedding=self._emb())
            eng.schemas.adjust_feedback_state(sid, is_labile=True)
            eng.schemas.refresh_utility(sid)  # lazily captures recurrence_count_at_flag
            assert eng.schemas.get(sid).is_labile is True

            for _ in range(3):
                eng.schemas.reinforce(sid, amount=0.05)

            assert eng.schemas.get(sid).is_labile is False
        finally:
            eng.close()
            _cleanup(path)

    def test_two_reinforcements_after_flag_do_not_clear_it(self) -> None:
        eng, path = _engine()
        try:
            sid = _create(eng, text="insufficient recurrence schema", embedding=self._emb())
            eng.schemas.adjust_feedback_state(sid, is_labile=True)
            eng.schemas.refresh_utility(sid)
            for _ in range(2):
                eng.schemas.reinforce(sid, amount=0.05)
            assert eng.schemas.get(sid).is_labile is True
        finally:
            eng.close()
            _cleanup(path)

    def test_recurrence_before_flag_does_not_count_toward_recovery(self) -> None:
        """A schema already reinforced many times before ever being flagged
        must not instantly recover the moment it's flagged."""
        eng, path = _engine()
        try:
            sid = _create(eng, text="pre-flag recurrence schema", embedding=self._emb())
            for _ in range(10):
                eng.schemas.reinforce(sid, amount=0.05)
            eng.schemas.adjust_feedback_state(sid, is_labile=True)
            eng.schemas.refresh_utility(sid)  # captures baseline at (post pre-flag) count
            assert eng.schemas.get(sid).is_labile is True
            # One more reinforcement since the flag — nowhere near the
            # threshold of 3 *since* flagging, even though lifetime
            # recurrence is 11.
            eng.schemas.reinforce(sid, amount=0.05)
            assert eng.schemas.get(sid).is_labile is True
        finally:
            eng.close()
            _cleanup(path)

    @staticmethod
    def _emb() -> np.ndarray:
        rng = np.random.default_rng(1)
        v = rng.normal(size=(DIM,)).astype(np.float32)
        return v / (np.linalg.norm(v) + 1e-12)


class TestReconsolidateLabileSchemas:
    """Consolidator.reconsolidate_labile_schemas(): replays flagged (labile)
    schemas against their nearest active neighbor via the real, unmocked
    GeometricContradictionJudge — the active, judge-driven form of
    reconsolidation (see TestRecurrenceClearsLability for the passive
    form)."""

    def _judge_cfg(self, eng: SlowaveEngine):
        return eng._consolidation.consolidator.geometric_judge.cfg

    def test_restabilizes_when_no_conflict_with_neighbor(self) -> None:
        """cos=0.85 (facet-comparison band) with IDENTICAL facet axes on
        both sides -> facet_distance == 0 -> verdict 'refines' -> the
        labile schema restabilizes."""
        eng, path = _engine()
        try:
            old_c, new_c = _same_topic_centroids(0.85)
            axes = np.eye(2, DIM, dtype=np.float32)
            strengths = np.array([1.0, 0.5], dtype=np.float32)

            neighbor_id = _create(
                eng,
                text="deployment target is kubernetes",
                embedding=old_c,
                facet_axes=axes,
                facet_strengths=strengths,
                supporting_episode_ids=[1, 2, 3],
            )
            labile_id = _create(
                eng,
                text="deployment target is kubernetes cluster v2",
                embedding=new_c,
                facet_axes=axes,
                facet_strengths=strengths,
                supporting_episode_ids=[4, 5, 6],
                is_labile=True,
            )
            _set_last_updated_ts(eng, neighbor_id, 1000)
            _set_last_updated_ts(eng, labile_id, 2000)

            stats = eng._consolidation.consolidator.reconsolidate_labile_schemas()

            assert stats["restabilized"] == 1
            assert eng.schemas.get(labile_id).is_labile is False
            assert eng.schemas.get(labile_id).status == "active"
            assert eng.schemas.get(neighbor_id).status == "active"
        finally:
            eng.close()
            _cleanup(path)

    def test_high_cosine_triggers_supersession_in_reconsolidation(self) -> None:
        """cos >= 0.90 with enough support and timestamp gap triggers
        supersession during reconsolidation, even though the judge only
        returns 'relates_to'. Per 2026-07-22 taxonomy collapse,
        supersession is gated on metadata (cos >= 0.90 + support + recency)
        at the consolidation call site, not on a geometric classifier."""
        eng, path = _engine()
        try:
            old_c, new_c = _same_topic_centroids(0.92)
            neighbor_axes = np.zeros((2, DIM), dtype=np.float32)
            neighbor_axes[0, 4] = 1.0
            neighbor_axes[1, 5] = 1.0
            labile_axes = np.zeros((2, DIM), dtype=np.float32)
            labile_axes[0, 2] = 1.0
            labile_axes[1, 3] = 1.0
            strengths = np.array([1.0, 0.5], dtype=np.float32)

            neighbor_id = _create(
                eng,
                text="preferred database is postgres",
                embedding=old_c,
                facet_axes=neighbor_axes,
                facet_strengths=strengths,
                supporting_episode_ids=[1, 2],
            )
            labile_id = _create(
                eng,
                text="preferred database is mysql now",
                embedding=new_c,
                facet_axes=labile_axes,
                facet_strengths=strengths,
                supporting_episode_ids=[3, 4],
                is_labile=True,
            )
            _set_last_updated_ts(eng, neighbor_id, 1000)
            _set_last_updated_ts(
                eng, labile_id, 1000 + int(self._judge_cfg(eng).min_time_delta_to_supersede_s) + 100
            )

            stats = eng._consolidation.consolidator.reconsolidate_labile_schemas()

            assert stats["superseded"] == 1
            assert eng.schemas.get(neighbor_id).status == "superseded"
            assert eng.schemas.get(labile_id).status == "active"
            assert eng.schemas.get(labile_id).is_labile is False
        finally:
            eng.close()
            _cleanup(path)

    def test_contradicted_when_timestamps_effectively_simultaneous(self) -> None:
        """cos >= 0.90 with equal timestamps (time_delta_s == 0) triggers
        'contradicted' status (not 'superseded') during reconsolidation.
        Per 2026-07-22 taxonomy collapse, supersession is gated on metadata
        at the consolidation call site, not on a geometric classifier."""
        eng, path = _engine()
        try:
            old_c, new_c = _same_topic_centroids(0.92)
            # e4, e5 -- see the sibling test above for why not e0/e1.
            neighbor_axes = np.zeros((2, DIM), dtype=np.float32)
            neighbor_axes[0, 4] = 1.0
            neighbor_axes[1, 5] = 1.0
            labile_axes = np.zeros((2, DIM), dtype=np.float32)
            labile_axes[0, 2] = 1.0
            labile_axes[1, 3] = 1.0
            strengths = np.array([1.0, 0.5], dtype=np.float32)

            neighbor_id = _create(
                eng,
                text="team standup is at 9am",
                embedding=old_c,
                facet_axes=neighbor_axes,
                facet_strengths=strengths,
                supporting_episode_ids=[1, 2],
            )
            labile_id = _create(
                eng,
                text="team standup is at 10am",
                embedding=new_c,
                facet_axes=labile_axes,
                facet_strengths=strengths,
                supporting_episode_ids=[3, 4],
                is_labile=True,
            )
            _set_last_updated_ts(eng, neighbor_id, 5000)
            _set_last_updated_ts(eng, labile_id, 5000)

            stats = eng._consolidation.consolidator.reconsolidate_labile_schemas()

            assert stats["contradicted"] == 1
            assert eng.schemas.get(neighbor_id).status == "contradicted"
            assert eng.schemas.get(labile_id).is_labile is False
        finally:
            eng.close()
            _cleanup(path)

    def test_stays_labile_when_no_related_neighbor_exists(self) -> None:
        eng, path = _engine()
        try:
            rng = np.random.default_rng(7)
            v = rng.normal(size=(DIM,)).astype(np.float32)
            isolated_emb = (v / (np.linalg.norm(v) + 1e-12)).astype(np.float32)
            labile_id = _create(
                eng,
                text="an isolated memory about beekeeping",
                embedding=isolated_emb,
                is_labile=True,
            )

            stats = eng._consolidation.consolidator.reconsolidate_labile_schemas()

            assert stats["inconclusive"] == 1
            assert eng.schemas.get(labile_id).is_labile is True
        finally:
            eng.close()
            _cleanup(path)

    def test_stays_labile_when_support_gate_downgrades_contradiction(self) -> None:
        """Same setup as the supersede test (stubbed direction_score), but
        the labile (newer) side has only 1 supporting episode -- below
        min_support_to_supersede=2 -- so the supersession is downgraded to
        inconclusive rather than acted on."""
        eng, path = _engine()
        try:
            eng._consolidation.consolidator.geometric_judge.manifold = _StubManifold(0.5)
            old_c, new_c = _same_topic_centroids(0.85)
            # e4, e5 -- see the sibling test above for why not e0/e1.
            neighbor_axes = np.zeros((2, DIM), dtype=np.float32)
            neighbor_axes[0, 4] = 1.0
            neighbor_axes[1, 5] = 1.0
            labile_axes = np.zeros((2, DIM), dtype=np.float32)
            labile_axes[0, 2] = 1.0
            labile_axes[1, 3] = 1.0
            strengths = np.array([1.0, 0.5], dtype=np.float32)

            neighbor_id = _create(
                eng,
                text="release cadence is weekly",
                embedding=old_c,
                facet_axes=neighbor_axes,
                facet_strengths=strengths,
                supporting_episode_ids=[1, 2],
            )
            labile_id = _create(
                eng,
                text="release cadence is biweekly now",
                embedding=new_c,
                facet_axes=labile_axes,
                facet_strengths=strengths,
                supporting_episode_ids=[3],  # below min_support_to_supersede
                is_labile=True,
            )
            _set_last_updated_ts(eng, neighbor_id, 1000)
            _set_last_updated_ts(
                eng, labile_id, 1000 + int(self._judge_cfg(eng).min_time_delta_to_supersede_s) + 100
            )

            stats = eng._consolidation.consolidator.reconsolidate_labile_schemas()

            assert stats["restabilized"] == 1
            assert eng.schemas.get(neighbor_id).status == "active"
            assert eng.schemas.get(labile_id).is_labile is False
        finally:
            eng.close()
            _cleanup(path)

    def test_ignores_non_active_labile_schemas(self) -> None:
        """list(is_labile=True, status="active") should never surface an
        already-resolved (non-active) schema even if its is_labile flag
        was never cleared."""
        eng, path = _engine()
        try:
            rng = np.random.default_rng(3)
            v = rng.normal(size=(DIM,)).astype(np.float32)
            emb = (v / (np.linalg.norm(v) + 1e-12)).astype(np.float32)
            sid = _create(eng, text="already superseded schema", embedding=emb, is_labile=True)
            eng.schemas.update_status(sid, status="superseded")

            stats = eng._consolidation.consolidator.reconsolidate_labile_schemas()

            assert stats["examined"] == 0
        finally:
            eng.close()
            _cleanup(path)
