"""FeedbackService: records retrieval snapshots and applies learning signals.

Previously scattered as methods on SlowaveEngine. Extracted so it can be
instantiated, tested, and reasoned about independently.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import time
from typing import Any, Callable

from slowave.core.feedback import FeedbackConfig, feedback_signal_for
from slowave.core.scope import scope_kind as _scope_kind
from slowave.core.services.feedback_events import FeedbackEventService
from slowave.core.services.learning_projection import LearningProjectionService
from slowave.core.services.retrieval_access import (
    RetrievalAccessEvidenceStore,
    canonical_cue_text,
    packed_cue_embedding,
)
from slowave.lifecycle import LIFECYCLE_VERSION
from slowave.storage.sqlite_db import SQLiteDB
from slowave.symbolic.schema_store import SchemaStore
from slowave.utils.vec import dumps_json

log = logging.getLogger(__name__)


def _bounded_response_json(response: dict[str, Any], max_chars: int) -> str:
    """Serialize a snapshot without ever cutting JSON in the middle.

    Oversized snapshots retain as many complete top-level list entries as fit
    and declare what was omitted. Normalized retrieval-item rows remain the
    authoritative complete item inventory.
    """
    encoded = json.dumps(response)
    if len(encoded) <= max_chars:
        return encoded

    bounded: dict[str, Any] = {
        "_truncated": True,
        "original_chars": len(encoded),
        "omitted_counts": {},
    }
    for key in ("memory_ids", "procedure_ids"):
        if key in response:
            bounded[key] = response[key]

    for key, value in response.items():
        if key in bounded or key in {"memory_ids", "procedure_ids"}:
            continue
        if not isinstance(value, list):
            candidate = {**bounded, key: value}
            if len(json.dumps(candidate)) <= max_chars:
                bounded = candidate
            continue
        kept: list[Any] = []
        bounded[key] = kept
        for entry in value:
            kept.append(entry)
            if len(json.dumps(bounded)) > max_chars:
                kept.pop()
                break
        omitted = len(value) - len(kept)
        if omitted:
            bounded["omitted_counts"][key] = omitted

    result = json.dumps(bounded)
    if len(result) <= max_chars:
        return result
    # Extremely small caller-provided caps cannot carry metadata. A minimal
    # valid object is still preferable to malformed JSON.
    minimal = json.dumps({"_truncated": True})
    return minimal if len(minimal) <= max_chars else "{}"


class FeedbackService:
    """Records retrieval snapshots and applies learning signals to schemas."""

    def __init__(
        self,
        *,
        db: SQLiteDB,
        schemas: SchemaStore,
        cfg: FeedbackConfig,
        encoder=None,
    ):
        self.db = db
        self.schemas = schemas
        self.encoder = encoder
        self.access_evidence = RetrievalAccessEvidenceStore(db)
        self.feedback_events = FeedbackEventService(db)
        self.learning_projection = LearningProjectionService(db=db, schemas=schemas)
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
        decision_candidates: list[dict[str, Any]] | None = None,
        shadow_interpretation: dict[str, Any] | None = None,
        lifecycle_version: str | None = None,
        retrieval_policy_version: str | None = None,
        continuity_state: str | None = None,
        cue_embedding=None,
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
        cue_text = canonical_cue_text(
            query=query,
            goal=goal,
            task_type=task_type,
            situation=situation,
            requirements=requirements,
            topics=topics,
            entities=entities,
        )
        cue = packed_cue_embedding(self.encoder, cue_text, vector=cue_embedding)

        memory_ids = []
        response_json_text = None
        if response:
            memory_ids = response.get("memory_ids", []) + response.get("procedure_ids", [])
            if self.cfg.persist_response_json:
                response_json_text = _bounded_response_json(
                    response, self.cfg.max_response_json_chars
                )

        conn.execute(
            """
            INSERT INTO context_recall_events (
              context_id, retrieval_type, session_id, scope_id, scope_kind,
              application, query, goal, task_type, situation_json, requirements_json,
              mode, limit_n, count_n, topics_json, entities_json,
              cue_terms_json, suppressed_json, memory_ids_json,
              response_json, cue_embedding, cue_dim, retrieval_policy_version,
              continuity_state, created_at, lifecycle_version
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                cue[0] if cue else None,
                cue[1] if cue else None,
                retrieval_policy_version,
                continuity_state,
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
                          salience, confidence, admitted, pathway, topical_relevance,
                          final_rank_score, score_margin, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                            item.get("topical_relevance", item.get("activation")),
                            item.get("final_rank_score", item.get("score")),
                            item.get("score_margin"),
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
                  salience, confidence, admitted, pathway, topical_relevance,
                  final_rank_score, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(retrieval_id),
                    str(f_memory_id),
                    retrieval_type,
                    f_item.get("memory_type", "schema"),
                    int(f_item["rank"]) if f_item.get("rank") is not None else -1,
                    f_item.get("activation"),
                    f_item.get("reason"),
                    str(f_item.get("content", ""))[: self.cfg.max_memory_content_chars],
                    f_item.get("status"),
                    f_item.get("salience"),
                    f_item.get("confidence"),
                    0,  # admitted=0: item was filtered by working-memory gate
                    f_item.get("pathway", "direct"),
                    f_item.get("topical_relevance", f_item.get("activation")),
                    f_item.get("final_rank_score", f_item.get("activation")),
                    now,
                ),
            )

        conn.commit()
        # Decision traces are advisory audit data. They are intentionally
        # fail-open: a storage problem must not affect the retrieval response
        # or the context_recall_items exposure ledger.
        try:
            self._record_decision_trace(
                retrieval_id=str(retrieval_id),
                response=response or {},
                filtered_items=filtered_items or [],
                decision_candidates=decision_candidates or [],
                retrieval_policy_version=retrieval_policy_version or "strict-v9",
                mode=mode,
                limit=limit,
            )
        except Exception:
            log.exception("failed to persist retrieval decision trace")
        if shadow_interpretation:
            try:
                self._record_shadow_interpretation(
                    retrieval_id=str(retrieval_id),
                    interpretation=shadow_interpretation,
                    mode=mode,
                    limit=limit,
                )
            except Exception:
                log.exception("failed to persist retrieval shadow interpretation")

    def _record_shadow_interpretation(
        self, *, retrieval_id: str, interpretation: dict[str, Any], mode: str, limit: int
    ) -> None:
        policy = "phase2-intent-shadow-v1"
        config_hash = hashlib.sha256(
            dumps_json({"policy_version": policy, "mode": mode, "limit": int(limit)}).encode()
        ).hexdigest()
        self.db.connect().execute(
            "INSERT OR REPLACE INTO retrieval_decisions "
            "(retrieval_id, policy_version, policy_role, trace_origin, task_needs_json, "
            "action_intent, action_intent_confidence, action_intent_reason, encoder_id, config_hash, "
            "frozen_portfolio_json, catalog_truncated, trace_complete, created_at) "
            "VALUES (?, ?, 'shadow', 'prospective', ?, ?, ?, ?, 'not_applicable', ?, '[]', 0, 1, ?)",
            (
                retrieval_id,
                policy,
                dumps_json(interpretation["task_needs"]),
                interpretation["action_intent"],
                interpretation["action_intent_confidence"],
                interpretation["action_intent_reason"],
                config_hash,
                int(time.time()),
            ),
        )
        self.db.connect().commit()

    def _record_decision_trace(
        self,
        *,
        retrieval_id: str,
        response: dict[str, Any],
        filtered_items: list[dict[str, Any]],
        decision_candidates: list[dict[str, Any]],
        retrieval_policy_version: str,
        mode: str,
        limit: int,
    ) -> None:
        """Persist a compact, current-policy trace after the snapshot exists."""
        now = int(time.time())
        policy = retrieval_policy_version
        config_hash = hashlib.sha256(
            dumps_json({"policy_version": policy, "mode": mode, "limit": int(limit)}).encode()
        ).hexdigest()
        encoder_id = (
            "no_encoder"
            if self.encoder is None
            else (f"{self.encoder.__class__.__module__}.{self.encoder.__class__.__qualname__}")
        )
        if policy == "multilingual-retrieval-baseline-v1":
            from dataclasses import asdict

            from slowave.core.retrieval_baseline import BaselineConfig
            from slowave.symbolic.retrieval_encoder import MODEL, REVISION

            encoder_id = f"{MODEL}@{REVISION}"
            config_hash = hashlib.sha256(
                dumps_json(
                    {
                        "policy_version": policy,
                        "activate": asdict(BaselineConfig()),
                        "recall": asdict(BaselineConfig(semantic_floor=0.80, max_memories=3)),
                        "model": encoder_id,
                    }
                ).encode()
            ).hexdigest()
        selected: list[tuple[str, str, dict[str, Any]]] = []
        selected.extend(
            ("memory", str(item.get("id") or item.get("memory_id")), item)
            for item in response.get("schemas", [])
            if item.get("id") or item.get("memory_id")
        )
        selected.extend(
            ("procedure", str(item.get("id") or item.get("procedure_id")), item)
            for item in response.get("procedures", [])
            if item.get("id") or item.get("procedure_id")
        )
        portfolio = [
            {"position": index, "kind": kind, "id": candidate_id}
            for index, (kind, candidate_id, _item) in enumerate(selected)
        ]
        conn = self.db.connect()
        with conn:
            conn.execute(
                "INSERT OR REPLACE INTO retrieval_decisions "
                "(retrieval_id, policy_version, policy_role, trace_origin, task_needs_json, "
                "action_intent, action_intent_confidence, action_intent_reason, encoder_id, config_hash, "
                "frozen_portfolio_json, catalog_truncated, trace_complete, created_at) "
                "VALUES (?, ?, 'current', 'prospective', '[]', 'uncertain', NULL, "
                "'not_derived_phase_1', ?, ?, ?, ?, 1, ?)",
                (
                    retrieval_id,
                    policy,
                    encoder_id,
                    config_hash,
                    dumps_json(portfolio),
                    int(bool(response.get("catalog_truncated"))),
                    now,
                ),
            )
            for position, (kind, candidate_id, item) in enumerate(selected):
                conn.execute(
                    "INSERT OR REPLACE INTO retrieval_candidate_decisions "
                    "(retrieval_id, policy_version, candidate_kind, candidate_id, eligible, "
                    "context_compatibility, dense_category, lexical_category, applicability_category, "
                    "novelty_category, evidence_category, covered_need_indexes_json, marginal_contribution, "
                    "decision, reason_code, redundant_with, portfolio_position, created_at) "
                    "VALUES (?, ?, ?, ?, 1, 'unknown', 'unknown', 'unknown', 'unknown', 'unknown', "
                    "'unknown', '[]', 'selected', 'selected', ?, NULL, ?, ?)",
                    (
                        retrieval_id,
                        policy,
                        kind,
                        candidate_id,
                        str(item.get("reason") or "selected"),
                        position,
                        now,
                    ),
                )
            for item in filtered_items:
                candidate_id = str(item.get("memory_id") or "")
                if not candidate_id:
                    continue
                conn.execute(
                    "INSERT OR IGNORE INTO retrieval_candidate_decisions "
                    "(retrieval_id, policy_version, candidate_kind, candidate_id, eligible, "
                    "context_compatibility, dense_category, lexical_category, applicability_category, "
                    "novelty_category, evidence_category, covered_need_indexes_json, marginal_contribution, "
                    "decision, reason_code, redundant_with, portfolio_position, created_at) "
                    "VALUES (?, ?, 'memory', ?, 0, 'unknown', 'unknown', 'unknown', 'unknown', 'unknown', "
                    "'unknown', '[]', 'not_selected', 'rejected', ?, NULL, NULL, ?)",
                    (
                        retrieval_id,
                        policy,
                        candidate_id,
                        str(item.get("reason") or "filtered"),
                        now,
                    ),
                )
            for item in decision_candidates:
                candidate_id = str(item.get("candidate_id") or "")
                if not candidate_id:
                    continue
                conn.execute(
                    "INSERT OR IGNORE INTO retrieval_candidate_decisions "
                    "(retrieval_id, policy_version, candidate_kind, candidate_id, eligible, "
                    "context_compatibility, dense_category, lexical_category, applicability_category, "
                    "novelty_category, evidence_category, covered_need_indexes_json, marginal_contribution, "
                    "decision, reason_code, redundant_with, portfolio_position, created_at) "
                    "VALUES (?, ?, ?, ?, ?, 'unknown', 'unknown', 'unknown', 'unknown', 'unknown', "
                    "'unknown', '[]', 'not_selected', ?, ?, ?, NULL, ?)",
                    (
                        retrieval_id,
                        policy,
                        str(item.get("candidate_kind") or "memory"),
                        candidate_id,
                        int(bool(item.get("eligible", False))),
                        str(item.get("decision") or "rejected"),
                        str(item.get("reason_code") or "not_selected"),
                        item.get("redundant_with"),
                        now,
                    ),
                )

    def backfill_legacy_decision_traces(self, *, batch_size: int = 100) -> int:
        """Record observed legacy exposure without inventing historical decisions.

        This metadata-only migration is deliberately bounded and idempotent.
        It never creates context_recall_items, never authorizes feedback, and
        never claims that unpersisted candidates or intent were observed.
        """
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        conn = self.db.connect()
        rows = conn.execute(
            "SELECT context_id, retrieval_policy_version, memory_ids_json, created_at "
            "FROM context_recall_events r WHERE NOT EXISTS ("
            "SELECT 1 FROM retrieval_decisions d WHERE d.retrieval_id = r.context_id"
            ") ORDER BY created_at, context_id LIMIT ?",
            (batch_size,),
        ).fetchall()
        now = int(time.time())
        with conn:
            for row in rows:
                retrieval_id = str(row["context_id"])
                source_policy = str(row["retrieval_policy_version"] or "unknown")
                policy = f"legacy-observed:{source_policy}"
                items = conn.execute(
                    "SELECT memory_id, memory_type, admitted, reason, rank FROM context_recall_items "
                    "WHERE context_id = ? ORDER BY admitted DESC, rank, memory_id",
                    (retrieval_id,),
                ).fetchall()
                portfolio = [
                    {
                        "position": index,
                        "kind": (
                            "procedure"
                            if item["memory_type"] in {"procedure", "procedural_memory"}
                            else "memory"
                        ),
                        "id": item["memory_id"],
                    }
                    for index, item in enumerate(items)
                    if item["admitted"]
                ]
                conn.execute(
                    "INSERT INTO retrieval_decisions "
                    "(retrieval_id, policy_version, policy_role, trace_origin, task_needs_json, "
                    "action_intent, action_intent_reason, encoder_id, config_hash, frozen_portfolio_json, "
                    "catalog_truncated, trace_complete, source_policy_version, source_created_at, "
                    "reconstruction_reason, observed_at, created_at) "
                    "VALUES (?, ?, 'historical', 'legacy_observed', '[]', 'uncertain', "
                    "'not_historically_persisted', 'unknown', 'legacy_observed', ?, 0, 0, ?, ?, ?, ?, ?)",
                    (
                        retrieval_id,
                        policy,
                        dumps_json(portfolio),
                        source_policy,
                        int(row["created_at"]),
                        "historical_candidate_decisions_not_persisted",
                        int(row["created_at"]),
                        now,
                    ),
                )
                for position, item in enumerate(items):
                    kind = (
                        "procedure"
                        if item["memory_type"] in {"procedure", "procedural_memory"}
                        else "memory"
                    )
                    decision = "selected" if item["admitted"] else "rejected"
                    conn.execute(
                        "INSERT INTO retrieval_candidate_decisions "
                        "(retrieval_id, policy_version, candidate_kind, candidate_id, eligible, "
                        "context_compatibility, dense_category, lexical_category, applicability_category, "
                        "novelty_category, evidence_category, covered_need_indexes_json, marginal_contribution, "
                        "decision, reason_code, redundant_with, portfolio_position, created_at) "
                        "VALUES (?, ?, ?, ?, ?, 'unknown', 'unknown', 'unknown', 'unknown', 'unknown', "
                        "'unknown', '[]', 'not_historically_persisted', ?, ?, NULL, ?, ?)",
                        (
                            retrieval_id,
                            policy,
                            kind,
                            item["memory_id"],
                            int(bool(item["admitted"])),
                            decision,
                            str(item["reason"] or "legacy_observed"),
                            position if item["admitted"] else None,
                            now,
                        ),
                    )
        return len(rows)

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

            if self.cfg.apply_stale_wrong_review:
                for schema_id in stale_ids:
                    try:
                        schema = self.schemas.get(schema_id)
                        if schema.status not in ("active", "needs_review"):
                            continue
                        self.schemas.adjust_feedback_state(
                            schema_id,
                            salience_delta=stale_signal.salience_delta * source_weight,
                            confidence_delta=stale_signal.confidence_delta * source_weight,
                            is_labile=False,
                            min_salience=self.cfg.min_salience,
                            min_confidence=self.cfg.min_confidence,
                            max_confidence=self.cfg.max_confidence,
                        )
                        self.schemas.update_status(
                            schema_id,
                            status="stale",
                            stale_reason="superseded",
                            is_labile=False,
                            salience=0.05,
                        )
                        applied["marked_review"].append(f"sch_{schema_id}")
                    except KeyError:
                        pass

                for schema_id in wrong_ids:
                    try:
                        schema = self.schemas.get(schema_id)
                        if schema.status not in ("active", "needs_review"):
                            continue
                        self.schemas.adjust_feedback_state(
                            schema_id,
                            salience_delta=wrong_signal.salience_delta * source_weight,
                            confidence_delta=wrong_signal.confidence_delta * source_weight,
                            is_labile=False,
                            min_salience=self.cfg.min_salience,
                            min_confidence=self.cfg.min_confidence,
                            max_confidence=self.cfg.max_confidence,
                        )
                        self.schemas.update_status(
                            schema_id,
                            status="stale",
                            stale_reason="contradicted",
                            is_labile=False,
                            salience=0.05,
                        )
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
        cursor = conn.execute(
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
                # Canonicalize legacy wrong ids into stale ids with reason
                # ``contradicted``; do not write new wrong-memory fields.
                dumps_json([f"sch_{i}" for i in sorted(set(stale_ids + wrong_ids))]),
                dumps_json([]),
                dumps_json(used_procedure_ids or []),
                dumps_json(irrelevant_procedure_ids or []),
                dumps_json(stale_procedure_ids or []),
                dumps_json(wrong_procedure_ids or []),
                missing_context,
                notes,
                now,
            ),
        )
        # FDB-1 shadow normalization.  It intentionally excludes task outcome:
        # the legacy row/mutations remain replayable, while the new event stream
        # models retrieval evidence independently of task success.
        self.feedback_events.record_legacy_reinforce(
            retrieval_id=str(retrieval_id),
            feedback=fb_label,
            used_memory_ids=[f"sch_{i}" for i in used_ids],
            irrelevant_memory_ids=[f"sch_{i}" for i in irrelevant_ids],
            stale_memory_ids=[f"sch_{i}" for i in stale_ids],
            wrong_memory_ids=[f"sch_{i}" for i in wrong_ids],
            used_procedure_ids=used_procedure_ids,
            irrelevant_procedure_ids=irrelevant_procedure_ids,
            stale_procedure_ids=stale_procedure_ids,
            wrong_procedure_ids=wrong_procedure_ids,
            missing_context=missing_context,
            source_feedback_id=int(cursor.lastrowid) if cursor.lastrowid is not None else None,
            conn=conn,
        )
        access_evidence = self.access_evidence.record_feedback(
            conn,
            retrieval_id=str(retrieval_id),
            useful_ids=used_ids if fb_label == "useful" else [],
            irrelevant_ids=irrelevant_ids if fb_label == "irrelevant" else [],
        )
        conn.commit()

        # Refresh noise/utility facets now that the event row is persisted —
        # the per-schema adjustments above ran before this insert and would
        # otherwise lag one feedback event behind.
        for schema_id in set(used_ids + stale_ids + wrong_ids):
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
            "access_evidence": access_evidence,
        }

    def context_feedback(self, *, context_id: str, **kwargs: Any) -> dict[str, Any]:
        """Backward-compatible wrapper for context/gating feedback."""
        return self.retrieval_feedback(retrieval_id=context_id, retrieval_type="context", **kwargs)

    def feedback(
        self,
        *,
        retrieval_id: str,
        memory_feedback: list[dict[str, Any]] | None = None,
        procedure_feedback: list[dict[str, Any]] | None = None,
        retrieval_quality: str | None = None,
        missing: list[str] | None = None,
        coverage: str = "partial",
    ) -> dict[str, Any]:
        """Record v9 feedback and apply only explicit declarative assessments.

        Task outcome is intentionally absent. Procedure evidence is persisted
        append-only for later outcome joining; it never enters declarative
        salience/confidence updates.
        """
        result = self.feedback_events.record(
            retrieval_id=retrieval_id,
            memory_feedback=memory_feedback,
            procedure_feedback=procedure_feedback,
            retrieval_quality=retrieval_quality,
            missing=missing,
            coverage=coverage,
            source_contract="slowave_feedback:v9",
            mutation_mode="active",
        )
        # Drive mutations from the accepted ledger rows, not submitted IDs:
        # one rejected duplicate must not hide another accepted observation.
        accepted_memory = []
        conn = self.db.connect()
        for event_id in result["accepted_event_ids"]:
            row = conn.execute(
                "SELECT target_id AS memory_id, assessment, stale_reason, "
                "replacement_target_id AS replacement_memory_id FROM feedback_events "
                "WHERE event_id = ? AND target_kind = 'memory'",
                (event_id,),
            ).fetchone()
            if row is not None:
                accepted_memory.append(dict(row))
        applied: dict[str, list[str]] = {
            "strengthened": [],
            "already_known": [],
            "not_used": [],
            "unassessable": [],
            "unchanged": [],
            "weakened": [],
            "superseded": [],
            "contradicted": [],
            "outdated": [],
            "unsupported": [],
            "withdrawn": [],
            "replacements": [],
            "access_evidence": [],
        }
        for item in accepted_memory:
            memory_id = str(item["memory_id"])
            try:
                schema_id = int(memory_id.removeprefix("sch_"))
            except ValueError:
                continue
            assessment = item["assessment"]
            stale_reason = item.get("stale_reason")
            schema = self.schemas.get(schema_id)
            if assessment == "used":
                # Conflicting feedback is append-only evidence, not authority
                # to resurrect a memory already retired by stale/wrong input.
                if schema.status not in ("active", "needs_review"):
                    continue
                signal = feedback_signal_for("useful", "unknown", self.cfg)
                # WP-2: learning is recomputed from the canonical ledger and
                # applied as a target value (idempotent per retrieval
                # identity), replacing the direct in-place tally.
                observation = self.learning_projection.apply_memory_observation(
                    schema_id,
                    trigger_retrieval_id=str(retrieval_id),
                    salience_delta_per_use=signal.salience_delta,
                    confidence_delta_per_use=signal.confidence_delta,
                    min_salience=self.cfg.min_salience,
                    min_confidence=self.cfg.min_confidence,
                    max_confidence=self.cfg.max_confidence,
                )
                bucket = (
                    "strengthened"
                    if observation.salience_delta > 0
                    else "weakened" if observation.salience_delta < 0 else "unchanged"
                )
                applied[bucket].append(memory_id)
            elif assessment in {"already_known", "not_used", "unassessable"}:
                # Section-3: a dedup observation — the client already knew
                # this. It is recorded evidence with no reinforcement and no
                # suppression, but it still replaces prior observations for
                # this identity, so the projection recomputes (e.g. a
                # used -> already_known refinement drops that contribution).
                if schema.status in ("active", "needs_review"):
                    useful_signal = feedback_signal_for("useful", "unknown", self.cfg)
                    self.learning_projection.apply_memory_observation(
                        schema_id,
                        trigger_retrieval_id=str(retrieval_id),
                        salience_delta_per_use=useful_signal.salience_delta,
                        confidence_delta_per_use=useful_signal.confidence_delta,
                        min_salience=self.cfg.min_salience,
                        min_confidence=self.cfg.min_confidence,
                        max_confidence=self.cfg.max_confidence,
                    )
                applied[assessment].append(memory_id)
            elif assessment == "stale":
                # First accepted terminal assessment wins. A later conflicting
                # assessment is still persisted above but cannot oscillate the
                # lifecycle state or resurrect an already retired memory.
                if schema.status not in ("active", "needs_review"):
                    continue
                signal = feedback_signal_for(assessment, "unknown", self.cfg)
                self.schemas.adjust_feedback_state(
                    schema_id,
                    salience_delta=signal.salience_delta,
                    confidence_delta=signal.confidence_delta,
                    is_labile=False,
                    min_salience=self.cfg.min_salience,
                    min_confidence=self.cfg.min_confidence,
                    max_confidence=self.cfg.max_confidence,
                )
                terminal_status = "stale"
                stale_reason = stale_reason or "superseded"
                self.schemas.update_status(
                    schema_id,
                    status=terminal_status,
                    stale_reason=stale_reason,
                    is_labile=False,
                    salience=0.05,
                )
                applied[stale_reason].append(memory_id)
                replacement_memory_id = item.get("replacement_memory_id")
                if replacement_memory_id:
                    applied["replacements"].append(f"{memory_id}->{replacement_memory_id}")
                # WP-2: the terminal transition overwrites the learning
                # contribution; drop the projection bookkeeping so a later
                # re-adoption cannot subtract a stale contribution.
                self.learning_projection.reset_schema(schema_id)
            elif assessment == "irrelevant":
                applied["access_evidence"].append(memory_id)
                # WP-2: a refinement (used -> irrelevant) changes the canonical
                # view; recompute so the prior contribution is removed in one
                # deterministic step. Neutral when there was no prior use.
                if schema.status in ("active", "needs_review"):
                    useful_signal = feedback_signal_for("useful", "unknown", self.cfg)
                    self.learning_projection.apply_memory_observation(
                        schema_id,
                        trigger_retrieval_id=str(retrieval_id),
                        salience_delta_per_use=useful_signal.salience_delta,
                        confidence_delta_per_use=useful_signal.confidence_delta,
                        min_salience=self.cfg.min_salience,
                        min_confidence=self.cfg.min_confidence,
                        max_confidence=self.cfg.max_confidence,
                    )
        # Reconcile corrections and transport retries, not just new negatives.
        if accepted_memory:
            conn = self.db.connect()
            self.access_evidence.reconcile_feedback(
                conn,
                retrieval_id=retrieval_id,
                memory_ids=[
                    int(str(item["memory_id"]).removeprefix("sch_")) for item in accepted_memory
                ],
            )
            conn.commit()
        result["applied"] = applied
        return result
