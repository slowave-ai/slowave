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
MEMORY_ASSESSMENTS = frozenset(
    {"used", "not_used", "unassessable", "irrelevant", "already_known", "stale"}
)
RELEVANCE_VALUES = frozenset({"relevant", "irrelevant", "uncertain"})
STALE_REASONS = frozenset({"contradicted", "superseded", "outdated", "unsupported", "withdrawn"})
PROCEDURE_USES = frozenset({"used", "not_used", "unassessable"})
PROCEDURE_EFFECTS = frozenset({"helped", "no_effect", "harmed", "unknown"})
COVERAGE_VALUES = frozenset({"partial", "complete"})


# Additive recovery guidance: reason codes remain stable for existing clients.
_FEEDBACK_HINTS = {
    "neutral_requires_unknown_or_absent_effect": (
        "effect",
        "For not_used/unassessable omit effect or use unknown.",
    ),
    "unassessable_requires_reason": ("reason", "Explain why usage cannot be assessed."),
    "target_not_exposed": ("target_id", "Use an assessed target ID returned by this retrieval."),
    "invalid_memory_assessment": (
        "assessment",
        "Use used|not_used|unassessable|irrelevant|already_known|stale.",
    ),
    "invalid_relevance": ("relevance", "Use relevant|irrelevant|uncertain, or omit relevance."),
    "used_requires_helped_no_effect_or_harmed_effect": (
        "effect",
        "For used memories use helped|no_effect|harmed, or omit effect (defaults to helped).",
    ),
    "already_known_requires_unknown_or_absent_effect": (
        "effect",
        "For already_known omit effect or use unknown.",
    ),
    "irrelevant_requires_unknown_or_absent_effect": (
        "effect",
        "For irrelevant omit effect or use unknown.",
    ),
    "stale_requires_unknown_or_absent_effect": ("effect", "For stale omit effect or use unknown."),
    "already_known_requires_no_replacement": (
        "replacement_memory_id",
        "Omit replacement_memory_id for already_known.",
    ),
    "stale_requires_valid_stale_reason": (
        "stale_reason",
        "Use contradicted|superseded|outdated|unsupported|withdrawn with assessment=stale.",
    ),
    "stale_requires_reason": ("reason", "Provide a nonblank explanation for stale."),
    "superseded_requires_replacement_memory_id": (
        "replacement_memory_id",
        "Provide a different active memory ID in this retrieval's scope.",
    ),
    "stale_reason_requires_stale_assessment": (
        "stale_reason",
        "Omit stale_reason unless assessment is stale.",
    ),
    "replacement_requires_stale_assessment": (
        "replacement_memory_id",
        "Omit replacement_memory_id unless assessment is stale.",
    ),
    "replacement_matches_retired_memory": (
        "replacement_memory_id",
        "Choose a replacement different from memory_id.",
    ),
    "invalid_replacement_memory_id": (
        "replacement_memory_id",
        "Use a canonical sch_<integer> memory ID.",
    ),
    "replacement_not_found": (
        "replacement_memory_id",
        "Use an existing active memory in this retrieval's scope.",
    ),
    "replacement_scope_mismatch": (
        "replacement_memory_id",
        "Use an active replacement in this retrieval's scope.",
    ),
    "replacement_not_current": ("replacement_memory_id", "Use an active replacement memory."),
    "invalid_procedure_use": ("use", "Use used|not_used|unassessable."),
    "invalid_procedure_effect": ("effect", "Use helped|no_effect|harmed|unknown, or omit effect."),
    "used_procedure_requires_contribution": (
        "contribution",
        "Provide a nonblank description of how the used procedure contributed.",
    ),
    "not_used_requires_unknown_effect_and_no_contribution": (
        "effect,contribution",
        "For not_used omit contribution and omit effect or use unknown.",
    ),
    "incomplete_coverage": (
        "coverage",
        "Assess all outstanding targets, then resend coverage=complete for this retrieval.",
    ),
}


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
        assessed_memories: set[str] = set()
        assessed_procedures: set[str] = set()
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

        def append(
            *,
            target_kind: str,
            target_id: str,
            replacement_target_id: str | None = None,
            assessment: str | None = None,
            relevance: str | None = None,
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
                  assessment, relevance, stale_reason, effect, contribution, reason, coverage,
                  retrieval_quality,
                  missing_json, status, rejection_reason, source_contract,
                  source_feedback_id, refines_event_id, mutation_mode, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    relevance,
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
                if target_kind == "memory":
                    assessed_memories.add(target_id)
                elif target_kind == "procedure":
                    assessed_procedures.add(target_id)
            else:
                code = rejection_reason or "rejected"
                field, hint = _FEEDBACK_HINTS.get(
                    code, ("target_id", "Correct this target and resend feedback.")
                )
                rejected.append(
                    {"target_id": target_id, "reason": code, "field": field, "hint": hint}
                )

        for item in memory_feedback or []:
            target_id = self._clean_text(item.get("memory_id")) or ""
            replacement_target_id = self._clean_text(item.get("replacement_memory_id"))
            assessment = self._clean_text(item.get("assessment"))
            relevance = self._clean_text(item.get("relevance"))
            stale_reason = self._clean_text(item.get("stale_reason"))
            effect = self._clean_text(item.get("effect"))
            reason = self._clean_text(item.get("reason"))
            error = None
            if target_id not in exposed_memories:
                error = "target_not_exposed"
            elif assessment not in MEMORY_ASSESSMENTS:
                error = "invalid_memory_assessment"
            elif relevance is not None and relevance not in RELEVANCE_VALUES:
                error = "invalid_relevance"
            elif assessment == "used":
                # Section-3: a used mark carries an explicit effect; an absent
                # effect is the legacy surface and means helped. unknown is
                # invalid with used (it would silently zero the observation).
                if effect is not None and effect not in ("helped", "no_effect", "harmed"):
                    error = "used_requires_helped_no_effect_or_harmed_effect"
            elif assessment in {"not_used", "unassessable"}:
                if effect is not None and effect != "unknown":
                    error = "neutral_requires_unknown_or_absent_effect"
                elif assessment == "unassessable" and not reason:
                    error = "unassessable_requires_reason"
            elif assessment == "already_known":
                # Section-3: a dedup observation — no effect, no replacement.
                if effect is not None and effect != "unknown":
                    error = "already_known_requires_unknown_or_absent_effect"
                elif replacement_target_id is not None:
                    error = "already_known_requires_no_replacement"
            elif assessment == "irrelevant":
                if effect is not None and effect != "unknown":
                    error = "irrelevant_requires_unknown_or_absent_effect"
            elif assessment == "stale":
                if effect is not None and effect != "unknown":
                    error = "stale_requires_unknown_or_absent_effect"
                elif stale_reason not in STALE_REASONS:
                    error = "stale_requires_valid_stale_reason"
                elif not reason:
                    error = "stale_requires_reason"
                elif stale_reason == "superseded" and replacement_target_id is None:
                    error = "superseded_requires_replacement_memory_id"
            if error is None and stale_reason is not None and assessment != "stale":
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
                relevance=relevance,
                stale_reason=stale_reason,
                effect=effect,
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
            elif use in {"not_used", "unassessable"} and (
                effect != "unknown" or contribution is not None
            ):
                error = "not_used_requires_unknown_effect_and_no_contribution"
            elif use == "unassessable" and not reason:
                error = "unassessable_requires_reason"
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

        outstanding = {
            "memory_ids": sorted(exposed_memories - assessed_memories),
            "procedure_ids": sorted(exposed_procedures - assessed_procedures),
        }
        coverage_error = coverage == "complete" and (bool(rejected) or any(outstanding.values()))
        append(
            target_kind="retrieval",
            target_id=str(retrieval_id),
            quality=self._clean_text(retrieval_quality),
            missing_items=[str(item).strip() for item in (missing or []) if str(item).strip()],
            status="rejected" if coverage_error else "accepted",
            rejection_reason="incomplete_coverage" if coverage_error else None,
        )

        if coverage_error:
            # Target observations can be accepted within an incomplete request,
            # but must not advertise successful complete coverage to consumers.
            conn.executemany(
                "UPDATE feedback_events SET coverage = 'partial' WHERE event_id = ?",
                [(event_id,) for event_id in accepted],
            )
        if own_transaction:
            conn.commit()
        return {
            "retrieval_id": str(retrieval_id),
            "coverage": "partial" if coverage_error else coverage,
            "requested_coverage": coverage,
            "outstanding": outstanding,
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
