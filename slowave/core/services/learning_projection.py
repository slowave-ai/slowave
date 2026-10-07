"""Canonical declarative learning projection (WP-2).

Public feedback appends immutable rows to ``feedback_events``. Declarative
learning (salience/confidence) must be a deterministic function of that
ledger, not an in-place tally: increments are clamped by caps and lowered by
decay, so subtracting an earlier increment cannot reliably undo it. This
module therefore recomputes each schema's learning effect from the canonical
view and applies it as an absolute target value.

Semantics (execution spec rev 2, §2/WP-2 and §3):

- Identity ``(retrieval_id, target_kind='memory', target_id)``; the latest
  accepted row wins per identity, folded in ``(created_at, rowid)`` order.
  Resubmitting identical feedback is a no-op; a refinement (used ->
  irrelevant) replaces the observation's contribution in one deterministic
  step; a distinct retrieval observing the same schema is a new observation
  and applies once.
- Exactly one reinforcement per independent retrieval: the contribution is
  the number of distinct retrieval ids whose winning accepted row assesses
  the schema ``used``, times the caller-provided per-use deltas (taken from
  the existing ``FeedbackConfig``; this module defines no second constants).
- Adoption: the first time a schema is touched, the view attribution of all
  pre-existing observations except the triggering one is recorded as already
  reflected in the stored value, so historical effects are not re-applied.
  Pre-WP-2 tallies may have been capped in ways that cannot be reconstructed;
  that residual imprecision is accepted as documented legacy contamination and
  never amplifies: later applies recompute from the current view.
- Bookkeeping stores the *effective* applied deltas (post-clamp), so caps and
  any foreign movement (worker decay) between applies are preserved and the
  fold converges: incremental application equals full replay, and a retried
  apply after a crash is a no-op.
- Shadow-mode rows (legacy internal CLI normalization) are excluded: their
  learning was applied by the legacy path with its own weights.
- Terminal ``stale`` transitions are applied by the feedback service with the
  existing first-terminal-wins semantics; this module only resets that
  schema's bookkeeping so no stale contribution is later subtracted.
- Procedures are out of scope: procedural evidence is already read from the
  canonical ledger with latest-accepted-wins by ``load_procedures``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from slowave.storage.sqlite_db import SQLiteDB
from slowave.symbolic.schema_store import SALIENCE_CEILING, SchemaStore


@dataclass(frozen=True)
class ApplyResult:
    """Outcome of one projection apply attempt."""

    status: str  # "applied" | "noop" | "skipped_status"
    salience_delta: float = 0.0
    confidence_delta: float = 0.0
    observations: int = 0


_EPS = 1e-9


class LearningProjectionService:
    """Apply declarative learning as a deterministic function of the ledger."""

    def __init__(self, db: SQLiteDB, schemas: SchemaStore):
        self.db = db
        self.schemas = schemas

    # ---- canonical view ----------------------------------------------------

    def _winning_observations(
        self, conn: Any, schema_id: int, *, exclude_retrieval_id: str | None = None
    ) -> dict[str, int]:
        """Per-retrieval usage units from winning accepted rows (section-3).

        Rows are folded in ``(created_at, rowid)`` order; the latest accepted
        row per retrieval id wins. Rejected and shadow-mode rows never count.
        Unit weights: ``used`` with no effect or ``helped`` -> +1 (an absent
        effect is the legacy surface and means helped); ``used`` with
        ``harmed`` -> -1; ``used`` with ``no_effect`` -> 0;
        ``already_known`` -> 0 (dedup observation, no reinforcement).
        """
        rows = conn.execute(
            "SELECT retrieval_id, assessment, effect FROM feedback_events "
            "WHERE target_kind = 'memory' AND target_id = ? AND status = 'accepted' "
            "AND mutation_mode = 'active' "
            "ORDER BY created_at, rowid",
            (f"sch_{int(schema_id)}",),
        ).fetchall()
        winning: dict[str, tuple[str | None, str | None]] = {}
        for row in rows:
            retrieval_id = str(row["retrieval_id"])
            if exclude_retrieval_id is not None and retrieval_id == exclude_retrieval_id:
                continue
            winning[retrieval_id] = (row["assessment"], row["effect"])
        units: dict[str, int] = {}
        for retrieval_id, (assessment, effect) in winning.items():
            if assessment == "used":
                if effect == "harmed":
                    units[retrieval_id] = -1
                elif effect == "no_effect":
                    units[retrieval_id] = 0
                else:
                    units[retrieval_id] = 1
            else:
                # already_known / irrelevant: no reinforcement contribution.
                units[retrieval_id] = 0
        return units

    def _view_total(
        self,
        conn: Any,
        schema_id: int,
        delta_per_use: float,
        *,
        exclude_retrieval_id: str | None = None,
    ) -> float:
        units = self._winning_observations(
            conn, schema_id, exclude_retrieval_id=exclude_retrieval_id
        )
        return sum(units.values()) * float(delta_per_use)

    # ---- application -------------------------------------------------------

    def apply_memory_observation(
        self,
        schema_id: int,
        *,
        trigger_retrieval_id: str,
        salience_delta_per_use: float,
        confidence_delta_per_use: float,
        min_salience: float = 0.01,
        min_confidence: float = 0.0,
        max_confidence: float = 1.0,
        ceiling: float = SALIENCE_CEILING,
        mode: str = "observation",
    ) -> ApplyResult:
        """Recompute the schema's learning effect and apply it as a target value.

        The schema update and the bookkeeping update share one transaction, so
        a crash leaves both unchanged and a retry converges to the same state.

        ``mode="observation"`` (the feedback path) adopts pre-existing
        observations except the triggering one as already reflected in the
        stored value. ``mode="replay"`` treats the stored value as virgin
        (rebuild scenario: the ledger's learning was never applied) and
        applies the full view; it refuses when bookkeeping already exists —
        delete the state rows first (the WP-7 rebuild procedure).
        """
        if mode not in ("observation", "replay"):
            raise ValueError("mode must be observation or replay")
        conn = self.db.connect()
        schema_row = conn.execute(
            "SELECT salience, confidence, status FROM schemas WHERE id = ?",
            (int(schema_id),),
        ).fetchone()
        if schema_row is None:
            raise KeyError(f"No schema id={schema_id}")
        if schema_row["status"] not in ("active", "needs_review"):
            return ApplyResult(status="skipped_status")

        state = conn.execute(
            "SELECT adopted_salience_attribution, adopted_confidence_attribution, "
            "applied_extra_salience, applied_extra_confidence "
            "FROM learning_projection_state WHERE schema_id = ?",
            (int(schema_id),),
        ).fetchone()
        if mode == "replay":
            if state is not None:
                raise ValueError(
                    "replay requires no existing projection bookkeeping; "
                    "delete learning_projection_state rows to rebuild"
                )
            adopted_sal = 0.0
            adopted_conf = 0.0
            extra_sal = 0.0
            extra_conf = 0.0
        elif state is None:
            adopted_sal = self._view_total(
                conn,
                schema_id,
                salience_delta_per_use,
                exclude_retrieval_id=str(trigger_retrieval_id),
            )
            adopted_conf = self._view_total(
                conn,
                schema_id,
                confidence_delta_per_use,
                exclude_retrieval_id=str(trigger_retrieval_id),
            )
            extra_sal = 0.0
            extra_conf = 0.0
        else:
            adopted_sal = float(state["adopted_salience_attribution"])
            adopted_conf = float(state["adopted_confidence_attribution"])
            extra_sal = float(state["applied_extra_salience"])
            extra_conf = float(state["applied_extra_confidence"])
        prev_sal_total = adopted_sal + extra_sal
        prev_conf_total = adopted_conf + extra_conf

        used_observations = self._winning_observations(conn, schema_id)
        new_sal_total = sum(used_observations.values()) * float(salience_delta_per_use)
        new_conf_total = sum(used_observations.values()) * float(confidence_delta_per_use)
        now = int(time.time())

        salience = float(schema_row["salience"])
        confidence = float(schema_row["confidence"])
        salience_target = min(ceiling, max(min_salience, salience - prev_sal_total + new_sal_total))
        confidence_target = min(
            max_confidence,
            max(min_confidence, confidence - prev_conf_total + new_conf_total),
        )
        salience_delta = salience_target - salience
        confidence_delta = confidence_target - confidence

        if mode != "replay" and abs(salience_delta) < _EPS and abs(confidence_delta) < _EPS:
            # The view's learning effect is already fully reflected in the
            # stored value (duplicate submission, or a delta a cap absorbs).
            # No writes: the state is already at the computed target.
            return ApplyResult(status="noop", observations=len(used_observations))

        if state is None:
            conn.execute(
                "INSERT INTO learning_projection_state ("
                "schema_id, adopted_at, adopted_salience_attribution, "
                "adopted_confidence_attribution, applied_extra_salience, "
                "applied_extra_confidence, last_applied_ts) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    int(schema_id),
                    now,
                    adopted_sal,
                    adopted_conf,
                    salience_delta,
                    confidence_delta,
                    now,
                ),
            )
        else:
            conn.execute(
                "UPDATE learning_projection_state SET applied_extra_salience = ?, "
                "applied_extra_confidence = ?, last_applied_ts = ? WHERE schema_id = ?",
                (extra_sal + salience_delta, extra_conf + confidence_delta, now, int(schema_id)),
            )
        net_positive = salience_delta > 0
        if net_positive:
            # Explicit positive evidence can clear decay/review lability,
            # matching the reinforce() path this projection replaces.
            conn.execute(
                "UPDATE schemas SET salience = ?, confidence = ?, last_updated_ts = ?, "
                "is_labile = 0 WHERE id = ?",
                (salience_target, confidence_target, now, int(schema_id)),
            )
        else:
            conn.execute(
                "UPDATE schemas SET salience = ?, confidence = ?, last_updated_ts = ? "
                "WHERE id = ?",
                (salience_target, confidence_target, now, int(schema_id)),
            )
        conn.commit()
        # Mirror reinforce()'s structure: derived utility facets refresh after
        # the learning transaction commits.
        self.schemas._update_utility_scores(
            int(schema_id), recall_hit=True, force_clear_labile=net_positive
        )
        return ApplyResult(
            status="applied",
            salience_delta=salience_delta,
            confidence_delta=confidence_delta,
            observations=len(used_observations),
        )

    def reset_schema(self, schema_id: int) -> None:
        """Drop bookkeeping after a terminal retirement (stale transition)."""
        conn = self.db.connect()
        conn.execute("DELETE FROM learning_projection_state WHERE schema_id = ?", (int(schema_id),))
        conn.commit()
