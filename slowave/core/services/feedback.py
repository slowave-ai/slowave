"""FeedbackService: records retrieval snapshots and applies learning signals.

Previously scattered as methods on SlowaveEngine. Extracted so it can be
instantiated, tested, and reasoned about independently.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import time
from typing import Any, Callable

from slowave.core.feedback import FeedbackConfig
from slowave.core.scope import scope_kind as _scope_kind
from slowave.lifecycle import LIFECYCLE_VERSION
from slowave.storage.sqlite_db import SQLiteDB
from slowave.symbolic.schema_store import SchemaStore
from slowave.utils.vec import dumps_json

log = logging.getLogger(__name__)


class FeedbackService:
    """Records retrieval snapshots and applies learning signals to schemas."""

    def __init__(
        self,
        *,
        db: SQLiteDB,
        schemas: SchemaStore,
        cfg: FeedbackConfig,
    ):
        self.db = db
        self.schemas = schemas
        self._parse_procedure_ids: Callable[[list[str]], list[str]] = (
            lambda ids: []
        )  # removed Phase 1 P1
        self.cfg = cfg

    # ---- public API --------------------------------------------------------

    def record_retrieval(
        self,
        *,
        retrieval_id: str,
        retrieval_type: str = "context",
        session_id: str | None = None,
        scope_id: str | None = None,
        scope_kind: str | None = None,
        application: str | None = None,
        query: str | None = None,
        goal: str | None = None,
        task_type: str | None = None,
        situation: dict[str, Any] | None = None,
        requirements: list[str] | None = None,
        mode: str = "default",
        limit: int = 8,
        topics: list[str] | None = None,
        entities: list[str] | None = None,
        cue_terms: list[str] | None = None,
        suppressed: dict[str, int] | None = None,
        response: dict[str, Any] | None = None,
        filtered_items: list[dict[str, Any]] | None = None,
        lifecycle_version: str | None = None,
    ) -> None:
        """Record a retrieval response snapshot for feedback correlation.

        filtered_items: list of items the working-memory gate evaluated but did NOT
        admit into context. Each item is a dict with at least 'memory_id' and
        optionally 'activation' and 'reason'.
        These are persisted to context_recall_items with admitted=0 so the full
        candidate pool (admitted + filtered) is queryable for trace analysis and
        future implicit signal learning.

        lifecycle_version: the lifecycle-instructions contract version in
        effect for this call (WP-8 telemetry) -- defaults to the server's
        current contract (slowave.lifecycle.LIFECYCLE_VERSION). Stamped here
        rather than derived from the session because recall() has no session
        concept at all (ops.recall() never sets session_id), so this is the
        only reliable per-call attribution point covering both activate's
        "context" and recall's "recall" retrieval_type rows.
        """
        if not self.cfg.enabled or not self.cfg.persist_context_snapshots:
            return

        conn = self.db.connect()
        now = int(time.time())

        memory_ids = []
        response_json_text = None
        if response:
            memory_ids = response.get("memory_ids", []) + response.get("procedure_ids", [])
            if self.cfg.persist_response_json:
                import json

                response_json_text = json.dumps(response)[: self.cfg.max_response_json_chars]

        conn.execute(
            """
            INSERT INTO context_recall_events (
              context_id, retrieval_type, session_id, scope_id, scope_kind,
              application, query, goal, task_type, situation_json, requirements_json,
              mode, limit_n, count_n, topics_json, entities_json,
              cue_terms_json, suppressed_json, memory_ids_json,
              response_json, created_at, lifecycle_version
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(retrieval_id),
                retrieval_type,
                session_id,
                scope_id,
                scope_kind,
                application,
                query,
                goal,
                task_type,
                dumps_json(situation or {}),
                dumps_json(requirements or []),
                mode,
                int(limit),
                len(memory_ids),
                dumps_json(topics or []),
                dumps_json(entities or []),
                dumps_json(cue_terms or []),
                dumps_json(suppressed or {}),
                dumps_json(memory_ids),
                response_json_text,
                now,
                LIFECYCLE_VERSION if lifecycle_version is None else (lifecycle_version or None),
            ),
        )

        items: list[tuple[str, str, dict[str, Any]]] = []
        if response:
            for schema_item in response.get("schemas", []):
                items.append(
                    ("schema", schema_item.get("id") or schema_item.get("memory_id"), schema_item)
                )
            for ep_item in response.get("episodes", []):
                items.append(("episode", ep_item.get("id") or ep_item.get("memory_id"), ep_item))
            for event_item in response.get("raw_events", []):
                items.append(
                    ("raw_event", event_item.get("id") or event_item.get("memory_id"), event_item)
                )
            for proc_item in response.get("procedures", []):
                items.append(
                    (
                        "procedural_memory",
                        proc_item.get("id") or proc_item.get("memory_id"),
                        proc_item,
                    )
                )
            for related_item in response.get("related_schemas", []):
                items.append(("related", related_item.get("id"), related_item))

        if items:
            for rank, (memory_type, memory_id, item) in enumerate(items):
                if memory_id:
                    conn.execute(
                        """
                        INSERT INTO context_recall_items (
                          context_id, memory_id, retrieval_type, memory_type, rank,
                          activation, reason, content_text, status,
                          salience, confidence, admitted, pathway, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            str(retrieval_id),
                            str(memory_id),
                            retrieval_type,
                            memory_type,
                            rank,
                            item.get("activation") or item.get("score"),
                            item.get("reason"),
                            str(item.get("content", ""))[: self.cfg.max_memory_content_chars],
                            item.get("status"),
                            item.get("salience"),
                            item.get("confidence"),
                            1,  # admitted=1: item was selected into context
                            item.get("pathway", "direct"),
                            now,
                        ),
                    )

        # Phase 1: persist filtered items (admitted=0) so the full candidate pool
        # is queryable. These are items the working-memory gate evaluated but dropped.
        for f_item in filtered_items or []:
            f_memory_id = f_item.get("memory_id")
            if not f_memory_id:
                continue
            # Use INSERT OR IGNORE: if by any chance the same memory_id was
            # already inserted as admitted=1, don't overwrite it.
            conn.execute(
                """
                INSERT OR IGNORE INTO context_recall_items (
                  context_id, memory_id, retrieval_type, memory_type, rank,
                  activation, reason, content_text, status,
                  salience, confidence, admitted, pathway, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(retrieval_id),
                    str(f_memory_id),
                    retrieval_type,
                    f_item.get("memory_type", "schema"),
                    -1,  # rank=-1 signals filtered
                    f_item.get("activation"),
                    f_item.get("reason"),
                    str(f_item.get("content", ""))[: self.cfg.max_memory_content_chars],
                    f_item.get("status"),
                    f_item.get("salience"),
                    f_item.get("confidence"),
                    0,  # admitted=0: item was filtered by working-memory gate
                    f_item.get("pathway", "direct"),
                    now,
                ),
            )

        conn.commit()

    def record_context_recall(self, *, context_id: str, **kwargs: Any) -> None:
        """Backward-compatible wrapper for context retrieval snapshots."""
        self.record_retrieval(retrieval_id=context_id, retrieval_type="context", **kwargs)

    def _derive_context_fields(self, retrieval_id: str) -> dict[str, Any]:
        """Auto-derive goal/task_type/scope_id/session_id/situation/requirements/retrieval_type
        from context_recall_events JOIN on retrieval_id.

        Returns a dict with the derived values (or None for each if not found).
        This is the Step-5 auto-derive: agents no longer need to re-supply these.

        DB source: context_recall_events.context_id = retrieval_id
          - retrieval_type: context_recall_events.retrieval_type
          - session_id:     context_recall_events.session_id
          - scope_id:       context_recall_events.scope_id
          - goal:           context_recall_events.goal
          - task_type:      context_recall_events.task_type
          - situation:      context_recall_events.situation_json (parsed)
          - requirements:   context_recall_events.requirements_json (parsed)
        """
        conn = self.db.connect()
        row = conn.execute(
            """
            SELECT retrieval_type, session_id, scope_id, goal, task_type,
                   situation_json, requirements_json
            FROM context_recall_events
            WHERE context_id = ?
            """,
            (str(retrieval_id),),
        ).fetchone()
        if row is None:
            return {}
        try:
            situation = json.loads(row["situation_json"] or "{}")
        except (json.JSONDecodeError, TypeError):
            situation = {}
        try:
            requirements = json.loads(row["requirements_json"] or "[]")
        except (json.JSONDecodeError, TypeError):
            requirements = []
        return {
            "retrieval_type": row["retrieval_type"],
            "session_id": row["session_id"],
            "scope_id": row["scope_id"],
            "goal": row["goal"],
            "task_type": row["task_type"],
            "situation": situation,
            "requirements": requirements,
        }

    def _valid_schema_ids_for_context(self, retrieval_id: str) -> set[int] | None:
        """Schema ids actually surfaced (admitted=1) by this retrieval.

        Returns None (meaning "can't validate, don't restrict") only when no
        snapshot rows exist at all for this retrieval_id -- e.g. persistence
        was disabled, or a caller-supplied retrieval_id that predates this
        check. When rows do exist, feedback calls must not be allowed to
        mutate schemas the client never actually saw: reinforce/reinforce-like
        calls have no scope filter of their own (schema_store.reinforce and
        friends do a bare `WHERE id = ?`), so an ID copied or hallucinated
        from a different scope/retrieval would otherwise silently mutate
        another project's memory.
        """
        conn = self.db.connect()
        rows = conn.execute(
            "SELECT memory_id FROM context_recall_items "
            "WHERE context_id = ? AND memory_type IN ('schema', 'related') AND admitted = 1",
            (str(retrieval_id),),
        ).fetchall()
        if not rows:
            return None
        ids: set[int] = set()
        for row in rows:
            mid = row["memory_id"]
            if isinstance(mid, str) and mid.startswith("sch_"):
                try:
                    ids.add(int(mid[4:]))
                except (ValueError, IndexError):
                    pass
        return ids

    def retrieval_feedback(
        self,
        *,
        retrieval_id: str,
        retrieval_type: str = "recall",
        feedback: str,
        outcome: str = "unknown",
        session_id: str | None = None,
        scope_id: str | None = None,
        goal: str | None = None,
        task_type: str | None = None,
        situation: dict[str, Any] | None = None,
        requirements: list[str] | None = None,
        used_memory_ids: list[str] | None = None,
        irrelevant_memory_ids: list[str] | None = None,
        stale_memory_ids: list[str] | None = None,
        wrong_memory_ids: list[str] | None = None,
        used_procedure_ids: list[str] | None = None,
        irrelevant_procedure_ids: list[str] | None = None,
        stale_procedure_ids: list[str] | None = None,
        wrong_procedure_ids: list[str] | None = None,
        missing_context: str | None = None,
        notes: str | None = None,
    ) -> dict[str, Any]:
        """Accept and learn from post-retrieval feedback.

        Auto-derives the following fields from context_recall_events (keyed on
        retrieval_id) so callers no longer need to re-supply them:
          - retrieval_type  (from DB column, inferred by id prefix as fallback)
          - session_id
          - scope_id
          - goal
          - task_type
          - situation
          - requirements

        Caller-supplied values (if not None) override the DB-derived values,
        preserving backward compatibility.

        `scope_id` (supplied here or derivable from the original
        `record_retrieval()` snapshot) is optional — `irrelevant`/`stale`/
        `wrong` marks accumulate into `context_noise_score` regardless of
        whether a scope is present (fixed 2026-07-10; the counting query
        used to require a non-null scope, which silently excluded scopeless
        events with no warning — see outcomes/08-feedback.md). `scope_id` is
        still used to attribute cross-scope generalization signal, which is
        a separate, unrelated computation from noise tracking.
        """
        if not self.cfg.enabled:
            return {
                "retrieval_id": retrieval_id,
                "feedback": feedback,
                "outcome": outcome,
                "enabled": False,
            }

        from slowave.core.feedback import (
            feedback_signal_for,
            normalize_feedback_label,
            normalize_outcome_label,
        )

        try:
            fb_label = normalize_feedback_label(feedback)
        except ValueError as e:
            return {"retrieval_id": retrieval_id, "error": str(e)}

        # Auto-derive context fields from the stored snapshot.
        # Caller-supplied non-None values always win.
        derived = self._derive_context_fields(retrieval_id)
        if derived:
            if retrieval_type == "recall" and derived.get("retrieval_type"):
                # Use DB-stored retrieval_type (fix D7: prefer DB column over prefix heuristics)
                retrieval_type = derived["retrieval_type"]
            if session_id is None:
                session_id = derived.get("session_id")
            if scope_id is None:
                scope_id = derived.get("scope_id")
            if goal is None:
                goal = derived.get("goal")
            if task_type is None:
                task_type = derived.get("task_type")
            if situation is None:
                situation = derived.get("situation")
            if requirements is None:
                requirements = derived.get("requirements")

        outcome = normalize_outcome_label(outcome)
        retrieval_type = retrieval_type if retrieval_type in ("context", "recall") else "recall"
        source_weight = (
            self.cfg.context_feedback_weight
            if retrieval_type == "context"
            else self.cfg.recall_feedback_weight
        )

        signal = feedback_signal_for(fb_label, outcome, self.cfg)
        useful_signal = feedback_signal_for("useful", outcome, self.cfg)
        partial_signal = feedback_signal_for("partially_useful", outcome, self.cfg)
        irrelevant_signal = feedback_signal_for("irrelevant", outcome, self.cfg)
        stale_signal = feedback_signal_for("stale", outcome, self.cfg)
        wrong_signal = feedback_signal_for("wrong", outcome, self.cfg)

        def _parse_schema_ids(ids: list[str] | None) -> list[int]:
            result = []
            for mid in ids or []:
                if isinstance(mid, str) and mid.startswith("sch_"):
                    try:
                        result.append(int(mid[4:]))
                    except (ValueError, IndexError):
                        pass
            return result

        used_ids = _parse_schema_ids(used_memory_ids)
        irrelevant_ids = _parse_schema_ids(irrelevant_memory_ids)
        stale_ids = _parse_schema_ids(stale_memory_ids)
        wrong_ids = _parse_schema_ids(wrong_memory_ids)

        applied: dict[str, list] = {
            "reinforced": [],
            "penalized": [],
            "marked_review": [],
            "procedures": [],
            "rejected": [],
        }

        # Restrict mutations to schema ids this retrieval actually surfaced.
        # See _valid_schema_ids_for_context docstring for why this matters.
        valid_ids = self._valid_schema_ids_for_context(retrieval_id)
        if valid_ids is not None:

            def _authorize(ids: list[int]) -> list[int]:
                kept = []
                for i in ids:
                    if i in valid_ids:
                        kept.append(i)
                    else:
                        applied["rejected"].append(f"sch_{i}")
                return kept

            used_ids = _authorize(used_ids)
            irrelevant_ids = _authorize(irrelevant_ids)
            stale_ids = _authorize(stale_ids)
            wrong_ids = _authorize(wrong_ids)

        if self.cfg.apply_learning:
            if self.cfg.apply_positive_learning and fb_label in ("useful", "partially_useful"):
                for schema_id in used_ids:
                    try:
                        if self.schemas.get(schema_id).status == "forgotten":
                            continue
                        if fb_label == "useful":
                            self.schemas.reinforce(
                                schema_id,
                                amount=useful_signal.salience_delta * source_weight,
                                confidence_delta=useful_signal.confidence_delta * source_weight,
                                min_confidence=self.cfg.min_confidence,
                                max_confidence=self.cfg.max_confidence,
                                clear_labile=True,
                            )
                        else:
                            self.schemas.adjust_feedback_state(
                                schema_id,
                                salience_delta=partial_signal.salience_delta * source_weight,
                                confidence_delta=partial_signal.confidence_delta * source_weight,
                                is_labile=False,
                                min_salience=self.cfg.min_salience,
                                min_confidence=self.cfg.min_confidence,
                                max_confidence=self.cfg.max_confidence,
                            )
                        applied["reinforced"].append(f"sch_{schema_id}")
                    except KeyError:
                        pass

            if self.cfg.apply_negative_learning:
                for schema_id in irrelevant_ids:
                    try:
                        if self.schemas.get(schema_id).status == "forgotten":
                            continue
                        self.schemas.adjust_feedback_state(
                            schema_id,
                            salience_delta=irrelevant_signal.salience_delta * source_weight,
                            confidence_delta=0.0,
                            min_salience=self.cfg.min_salience,
                            min_confidence=self.cfg.min_confidence,
                            max_confidence=self.cfg.max_confidence,
                        )
                        applied["penalized"].append(f"sch_{schema_id}")
                    except KeyError:
                        pass

            if self.cfg.apply_stale_wrong_review:
                for schema_id in stale_ids:
                    try:
                        if self.schemas.get(schema_id).status == "forgotten":
                            continue
                        self.schemas.adjust_feedback_state(
                            schema_id,
                            salience_delta=stale_signal.salience_delta * source_weight,
                            confidence_delta=stale_signal.confidence_delta * source_weight,
                            is_labile=True,
                            min_salience=self.cfg.min_salience,
                            min_confidence=self.cfg.min_confidence,
                            max_confidence=self.cfg.max_confidence,
                        )
                        applied["marked_review"].append(f"sch_{schema_id}")
                    except KeyError:
                        pass

                for schema_id in wrong_ids:
                    try:
                        schema = self.schemas.get(schema_id)
                        if schema.status == "forgotten":
                            continue
                        self.schemas.adjust_feedback_state(
                            schema_id,
                            salience_delta=wrong_signal.salience_delta * source_weight,
                            confidence_delta=wrong_signal.confidence_delta * source_weight,
                            is_labile=True,
                            min_salience=self.cfg.min_salience,
                            min_confidence=self.cfg.min_confidence,
                            max_confidence=self.cfg.max_confidence,
                        )
                        # wrong + outcome=failed: escalate status to needs_review so the
                        # mode-gated filter in recall() fully excludes the schema in
                        # default mode rather than relying on score-penalisation alone.
                        # Only escalate from "active" -- schemas already in a resolved
                        # state (superseded/contradicted/archived) must not be pulled
                        # back into needs_review visibility by an ordinary feedback call;
                        # that resolution should only be undone by whatever process
                        # reached it (e.g. a new supersession, or explicit unforget).
                        if outcome == "failure" and schema.status == "active":
                            self.schemas.update_status(schema_id, status="needs_review")
                        applied["marked_review"].append(f"sch_{schema_id}")
                    except KeyError:
                        pass

        # Procedure feedback removed in Phase 1 P1

        # Persist feedback event; ensure parent FK row exists first.
        conn = self.db.connect()
        now = int(time.time())
        parent = conn.execute(
            "SELECT context_id FROM context_recall_events WHERE context_id = ?",
            (str(retrieval_id),),
        ).fetchone()
        if parent is None:
            conn.execute(
                """
                INSERT INTO context_recall_events (
                  context_id, retrieval_type, session_id, scope_id, scope_kind,
                  application, query, goal, task_type, situation_json, requirements_json,
                  mode, limit_n, count_n, topics_json, entities_json,
                  cue_terms_json, suppressed_json, memory_ids_json,
                  response_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(retrieval_id),
                    retrieval_type,
                    session_id,
                    scope_id,
                    _scope_kind(scope_id),
                    None,
                    None,
                    goal,
                    task_type,
                    dumps_json(situation or {}),
                    dumps_json(requirements or []),
                    "unknown",
                    0,
                    0,
                    "[]",
                    "[]",
                    "[]",
                    "{}",
                    "[]",
                    None,
                    now,
                ),
            )
        conn.execute(
            """
            INSERT INTO context_feedback_events (
              context_id, retrieval_type, session_id, scope_id, scope_kind,
              goal, task_type, situation_json, requirements_json,
              feedback, outcome, feedback_signal_json, outcome_reward,
              used_memory_ids_json, irrelevant_memory_ids_json,
              stale_memory_ids_json, wrong_memory_ids_json,
              used_procedure_ids_json, irrelevant_procedure_ids_json,
              stale_procedure_ids_json, wrong_procedure_ids_json,
              missing_context, notes, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(retrieval_id),
                retrieval_type,
                session_id,
                scope_id,
                _scope_kind(scope_id),
                goal,
                task_type,
                dumps_json(situation or {}),
                dumps_json(requirements or []),
                fb_label,
                outcome,
                dumps_json(dataclasses.asdict(signal)),
                signal.outcome_reward,
                dumps_json([f"sch_{i}" for i in used_ids]),
                dumps_json([f"sch_{i}" for i in irrelevant_ids]),
                dumps_json([f"sch_{i}" for i in stale_ids]),
                dumps_json([f"sch_{i}" for i in wrong_ids]),
                dumps_json(used_procedure_ids or []),
                dumps_json(irrelevant_procedure_ids or []),
                dumps_json(stale_procedure_ids or []),
                dumps_json(wrong_procedure_ids or []),
                missing_context,
                notes,
                now,
            ),
        )
        conn.commit()

        # Refresh noise/utility facets now that the event row is persisted —
        # the per-schema adjustments above ran before this insert and would
        # otherwise lag one feedback event behind.
        for schema_id in set(used_ids + irrelevant_ids + stale_ids + wrong_ids):
            try:
                self.schemas.refresh_utility(schema_id)
            except KeyError:
                pass

        return {
            "retrieval_id": retrieval_id,
            "context_id": retrieval_id if retrieval_type == "context" else None,
            "recall_id": retrieval_id if retrieval_type == "recall" else None,
            "retrieval_type": retrieval_type,
            "feedback": fb_label,
            "outcome": outcome,
            "applied": applied,
            "signal": dataclasses.asdict(signal),
            "source_weight": source_weight,
        }

    def context_feedback(self, *, context_id: str, **kwargs: Any) -> dict[str, Any]:
        """Backward-compatible wrapper for context/gating feedback."""
        return self.retrieval_feedback(retrieval_id=context_id, retrieval_type="context", **kwargs)
