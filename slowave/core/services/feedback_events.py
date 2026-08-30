"""Append-only feedback-event foundation.

FDB-1 stores the v9 ``slowave_feedback`` stream directly. Legacy internal CLI
events may still be normalized in shadow mode so immutable history remains
replayable, but no compatibility MCP tool is registered.
"""

from __future__ import annotations

import time
import uuid
from typing import Any

from slowave.storage.sqlite_db import SQLiteDB
from slowave.utils.vec import dumps_json

# Lifecycle status is intentionally separate from the client's semantic reason.
MEMORY_ASSESSMENTS = frozenset({"used", "irrelevant", "stale"})
STALE_REASONS = frozenset({"contradicted", "superseded", "outdated", "unsupported", "withdrawn"})
PROCEDURE_USES = frozenset({"used", "not_used"})
PROCEDURE_EFFECTS = frozenset({"helped", "no_effect", "harmed", "unknown"})
COVERAGE_VALUES = frozenset({"partial", "complete"})


class FeedbackEventService:
    """Validate and append neutral feedback records without applying learning."""

    def __init__(self, db: SQLiteDB):
        self.db = db

    @staticmethod
    def _clean_text(value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    def _exposure(self, conn, retrieval_id: str) -> tuple[dict[str, str], set[str], set[str]]:
        parent = conn.execute(
            "SELECT session_id, scope_id FROM context_recall_events WHERE context_id = ?",
            (retrieval_id,),
        ).fetchone()
        if parent is None:
            raise ValueError(f"unknown retrieval_id: {retrieval_id}")
        rows = conn.execute(
            "SELECT memory_id, memory_type FROM context_recall_items "
            "WHERE context_id = ? AND admitted = 1",
            (retrieval_id,),
        ).fetchall()
        memories = {
            str(row["memory_id"]) for row in rows if row["memory_type"] in ("schema", "related")
        }
        procedures = {
            str(row["memory_id"]) for row in rows if row["memory_type"] == "procedural_memory"
        }
        return dict(parent), memories, procedures

    @staticmethod
    def _latest_event_id(conn, retrieval_id: str, target_kind: str, target_id: str) -> str | None:
        row = conn.execute(
            "SELECT event_id FROM feedback_events WHERE retrieval_id = ? "
            "AND target_kind = ? AND target_id = ? AND status = 'accepted' "
            "ORDER BY created_at DESC, rowid DESC LIMIT 1",
            (retrieval_id, target_kind, target_id),
        ).fetchone()
        return str(row["event_id"]) if row else None

    def record(
        self,
        *,
        retrieval_id: str,
        memory_feedback: list[dict[str, Any]] | None = None,
        procedure_feedback: list[dict[str, Any]] | None = None,
        retrieval_quality: str | None = None,
        missing: list[str] | None = None,
        coverage: str = "partial",
        source_contract: str = "internal:v1",
        source_feedback_id: int | None = None,
        mutation_mode: str = "active",
        conn=None,
    ) -> dict[str, Any]:
        """Append one retrieval declaration plus independently validated targets.

        Invalid or unauthorized targets are retained as rejected audit rows, but
        are never treated as accepted evidence.  Existing accepted rows are
        referenced by later rows instead of being updated or deleted.
        """
        if coverage not in COVERAGE_VALUES:
            raise ValueError("coverage must be partial or complete")
        if mutation_mode not in {"shadow", "active"}:
            raise ValueError("mutation_mode must be shadow or active")
        own_transaction = conn is None
        conn = conn or self.db.connect()
        parent, exposed_memories, exposed_procedures = self._exposure(conn, str(retrieval_id))
        now = int(time.time())
        accepted: list[str] = []
        rejected: list[dict[str, str]] = []
        assessed_memories = {
            str(item.get("memory_id", "")).strip() for item in (memory_feedback or [])
        }
        assessed_procedures = {
            str(item.get("procedure_id", "")).strip() for item in (procedure_feedback or [])
        }
        previous = conn.execute(
            "SELECT target_kind, target_id FROM feedback_events WHERE retrieval_id = ? "
            "AND target_kind IN ('memory', 'procedure') AND status = 'accepted'",
            (str(retrieval_id),),
        ).fetchall()
        assessed_memories.update(
            str(row["target_id"]) for row in previous if row["target_kind"] == "memory"
        )
        assessed_procedures.update(
            str(row["target_id"]) for row in previous if row["target_kind"] == "procedure"
        )
        outstanding = {
            "memory_ids": sorted(exposed_memories - assessed_memories),
            "procedure_ids": sorted(exposed_procedures - assessed_procedures),
        }
        coverage_error = coverage == "complete" and any(outstanding.values())

        def append(
            *,
            target_kind: str,
            target_id: str,
            replacement_target_id: str | None = None,
            assessment: str | None = None,
            stale_reason: str | None = None,
            effect: str | None = None,
            contribution: str | None = None,
            reason: str | None = None,
            status: str = "accepted",
            rejection_reason: str | None = None,
            quality: str | None = None,
            missing_items: list[str] | None = None,
        ) -> None:
            event_id = f"fbe_{uuid.uuid4().hex}"
            refines = self._latest_event_id(conn, str(retrieval_id), target_kind, target_id)
            conn.execute(
                """
                INSERT INTO feedback_events (
                  event_id, retrieval_id, session_id, scope_id, target_kind, target_id,
                  replacement_target_id,
                  assessment, stale_reason, effect, contribution, reason, coverage, retrieval_quality,
                  missing_json, status, rejection_reason, source_contract,
                  source_feedback_id, refines_event_id, mutation_mode, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event_id,
                    str(retrieval_id),
                    parent.get("session_id"),
                    parent.get("scope_id"),
                    target_kind,
                    target_id,
                    replacement_target_id,
                    assessment,
                    stale_reason,
                    effect,
                    contribution,
                    reason,
                    coverage,
                    quality,
                    dumps_json(missing_items or []),
                    status,
                    rejection_reason,
                    source_contract,
                    source_feedback_id,
                    refines,
                    mutation_mode,
                    now,
                ),
            )
            if status == "accepted":
                accepted.append(event_id)
            else:
                rejected.append({"target_id": target_id, "reason": rejection_reason or "rejected"})

        append(
            target_kind="retrieval",
            target_id=str(retrieval_id),
            quality=self._clean_text(retrieval_quality),
            missing_items=[str(item).strip() for item in (missing or []) if str(item).strip()],
            status="rejected" if coverage_error else "accepted",
            rejection_reason="incomplete_coverage" if coverage_error else None,
        )

        for item in memory_feedback or []:
            target_id = self._clean_text(item.get("memory_id")) or ""
            replacement_target_id = self._clean_text(item.get("replacement_memory_id"))
            assessment = self._clean_text(item.get("assessment"))
            stale_reason = self._clean_text(item.get("stale_reason"))
            reason = self._clean_text(item.get("reason"))
            error = None
            if target_id not in exposed_memories:
                error = "target_not_exposed"
            elif assessment not in MEMORY_ASSESSMENTS:
                error = "invalid_memory_assessment"
            elif assessment == "stale":
                if stale_reason not in STALE_REASONS:
                    error = "stale_requires_valid_stale_reason"
                elif not reason:
                    error = "stale_requires_reason"
                elif stale_reason == "superseded" and replacement_target_id is None:
                    error = "superseded_requires_replacement_memory_id"
            elif stale_reason is not None:
                error = "stale_reason_requires_stale_assessment"
            if error is None and replacement_target_id is not None:
                if assessment != "stale":
                    error = "replacement_requires_stale_assessment"
                elif replacement_target_id == target_id:
                    error = "replacement_matches_retired_memory"
                else:
                    try:
                        replacement_schema_id = int(replacement_target_id.removeprefix("sch_"))
                    except ValueError:
                        error = "invalid_replacement_memory_id"
                    else:
                        replacement = conn.execute(
                            "SELECT scope_id, status FROM schemas WHERE id = ?",
                            (replacement_schema_id,),
                        ).fetchone()
                        if replacement is None:
                            error = "replacement_not_found"
                        elif replacement["scope_id"] != parent.get("scope_id"):
                            error = "replacement_scope_mismatch"
                        elif replacement["status"] != "active":
                            error = "replacement_not_current"
            append(
                target_kind="memory",
                target_id=target_id,
                replacement_target_id=replacement_target_id,
                assessment=assessment,
                stale_reason=stale_reason,
                reason=reason,
                status="rejected" if error else "accepted",
                rejection_reason=error,
            )

        for item in procedure_feedback or []:
            target_id = self._clean_text(item.get("procedure_id")) or ""
            use = self._clean_text(item.get("use"))
            effect = self._clean_text(item.get("effect")) or "unknown"
            contribution = self._clean_text(item.get("contribution"))
            reason = self._clean_text(item.get("reason"))
            error = None
            if target_id not in exposed_procedures:
                error = "target_not_exposed"
            elif use not in PROCEDURE_USES:
                error = "invalid_procedure_use"
            elif effect not in PROCEDURE_EFFECTS:
                error = "invalid_procedure_effect"
            elif use == "used" and contribution is None:
                error = "used_procedure_requires_contribution"
            elif use == "not_used" and (effect != "unknown" or contribution is not None):
                error = "not_used_requires_unknown_effect_and_no_contribution"
            append(
                target_kind="procedure",
                target_id=target_id,
                assessment=use,
                effect=effect,
                contribution=contribution,
                reason=reason,
                status="rejected" if error else "accepted",
                rejection_reason=error,
            )

        if own_transaction:
            conn.commit()
        return {
            "retrieval_id": str(retrieval_id),
            "coverage": coverage,
            "outstanding": (
                outstanding if coverage_error else {"memory_ids": [], "procedure_ids": []}
            ),
            "accepted_event_ids": accepted,
            "rejected": rejected,
        }

    def incomplete_for_session(self, session_id: str) -> list[dict[str, Any]]:
        """Return machine-actionable outstanding exposure coverage for a session."""
        conn = self.db.connect()
        retrievals = conn.execute(
            "SELECT context_id FROM context_recall_events WHERE session_id = ? ORDER BY created_at",
            (session_id,),
        ).fetchall()
        incomplete: list[dict[str, Any]] = []
        for retrieval in retrievals:
            retrieval_id = str(retrieval["context_id"])
            exposure_rows = conn.execute(
                "SELECT memory_id, memory_type FROM context_recall_items "
                "WHERE context_id = ? AND admitted = 1",
                (retrieval_id,),
            ).fetchall()
            exposed_memories = {
                str(row["memory_id"])
                for row in exposure_rows
                if row["memory_type"] in {"schema", "related"}
            }
            exposed_procedures = {
                str(row["memory_id"])
                for row in exposure_rows
                if row["memory_type"] == "procedural_memory"
            }
            assessed_rows = conn.execute(
                "SELECT target_kind, target_id FROM feedback_events WHERE retrieval_id = ? "
                "AND status = 'accepted' AND target_kind IN ('memory', 'procedure')",
                (retrieval_id,),
            ).fetchall()
            assessed_memories = {
                str(row["target_id"]) for row in assessed_rows if row["target_kind"] == "memory"
            }
            assessed_procedures = {
                str(row["target_id"]) for row in assessed_rows if row["target_kind"] == "procedure"
            }
            complete = conn.execute(
                "SELECT 1 FROM feedback_events WHERE retrieval_id = ? "
                "AND target_kind = 'retrieval' AND coverage = 'complete' "
                "AND status = 'accepted' LIMIT 1",
                (retrieval_id,),
            ).fetchone()
            missing_memories = sorted(exposed_memories - assessed_memories)
            missing_procedures = sorted(exposed_procedures - assessed_procedures)
            if complete is None or missing_memories or missing_procedures:
                incomplete.append(
                    {
                        "retrieval_id": retrieval_id,
                        "memory_ids": missing_memories,
                        "procedure_ids": missing_procedures,
                        "coverage_declared_complete": complete is not None,
                    }
                )
        return incomplete

    def record_legacy_reinforce(
        self,
        *,
        retrieval_id: str,
        feedback: str,
        used_memory_ids: list[str],
        irrelevant_memory_ids: list[str],
        stale_memory_ids: list[str],
        wrong_memory_ids: list[str],
        used_procedure_ids: list[str] | None,
        irrelevant_procedure_ids: list[str] | None,
        stale_procedure_ids: list[str] | None,
        wrong_procedure_ids: list[str] | None,
        missing_context: str | None,
        source_feedback_id: int | None,
        conn,
    ) -> dict[str, Any]:
        """Normalize the legacy surface without importing task outcome."""
        memories: list[dict[str, Any]] = [
            *({"memory_id": mid, "assessment": "used"} for mid in used_memory_ids),
            *({"memory_id": mid, "assessment": "irrelevant"} for mid in irrelevant_memory_ids),
            *(
                {
                    "memory_id": mid,
                    "assessment": "stale",
                    "stale_reason": "superseded",
                    "replacement_memory_id": None,
                    "reason": "Legacy stale feedback",
                }
                for mid in stale_memory_ids
            ),
            *(
                {
                    "memory_id": mid,
                    "assessment": "stale",
                    "stale_reason": "contradicted",
                    "reason": "Legacy contradicted feedback",
                }
                for mid in wrong_memory_ids
            ),
        ]
        procedures = [
            *(
                {"procedure_id": pid, "use": "used", "effect": "unknown"}
                for pid in (used_procedure_ids or [])
            ),
            *(
                {"procedure_id": pid, "use": "not_used", "effect": "unknown"}
                for pid in (irrelevant_procedure_ids or [])
            ),
            *(
                {"procedure_id": pid, "use": "legacy_stale", "effect": "unknown"}
                for pid in (stale_procedure_ids or [])
            ),
            *(
                {"procedure_id": pid, "use": "legacy_wrong", "effect": "unknown"}
                for pid in (wrong_procedure_ids or [])
            ),
        ]
        return self.record(
            retrieval_id=retrieval_id,
            memory_feedback=memories,
            procedure_feedback=procedures,
            retrieval_quality=feedback,
            missing=[missing_context] if missing_context else None,
            coverage="partial",
            source_contract="slowave_reinforce:legacy",
            source_feedback_id=source_feedback_id,
            mutation_mode="shadow",
            conn=conn,
        )
