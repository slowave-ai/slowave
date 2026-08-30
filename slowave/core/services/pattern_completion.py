"""PatternCompletionService: encoding-time familiarity check against existing schemas.

Brain framing: this is the fast familiarity check that happens at encoding
time. It may reinforce independently observed cross-scope evidence, but it
never changes the truth or lifecycle state of a same-scope memory.

Extracted from ``SlowaveEngine.remember()``. Encoding-time similarity is not a
semantic-truth classifier; lifecycle decisions are made by client feedback.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from slowave.symbolic.schema_store import SchemaStore

if TYPE_CHECKING:
    from slowave.storage.sqlite_db import SQLiteDB

log = logging.getLogger(__name__)

# Empirically calibrated cross-scope near-duplicate floor. This is an evidence
# association threshold, not a semantic-truth classifier.
_CROSS_SCOPE_REINFORCEMENT_COSINE = 0.78


class PatternCompletionService:
    """Checks a freshly-remembered schema against its near-embedding neighbors.

    remember() is deliberately encoding-only for same-scope pairs. Similarity
    cannot decide whether a nearby claim replaces, contradicts, elaborates, or
    merely restates another claim, so same-scope candidates are left untouched
    until an exposed memory receives explicit client feedback.

    Cross-scope handling below never writes a schema_relations edge or a
    status change. It feeds the promotion ladder instead —
    schema_evidence rows from independent cross-scope attestations of the
    same fact are what _update_utility_scores reads to advance
    generalization_stage (see
    private/docs/iterations/20260715_promotion_ladder_and_relation_taxonomy_review.md).

    Cross-scope reinforcement no longer gates on direction_score (2026-07-23):
    it used to skip reinforcement when direction_score read as "value
    diverged across scopes", on the theory that same-concept-vs-diverged was
    a different question from supersedes/refines/relates_to. But it's the
    same underlying discrimination task (raw direction_score, no facet
    signal) that the 2026-07-22 cross-repo measurement showed doesn't
    generalize — see
    private/docs/iterations/20260723_part_of_audit_and_brain_alignment_review.md.
    Once a candidate clears the cross-scope cosine floor it now reinforces
    unconditionally, the same principle Phase 1 applied to relates_to:
    cosine alone can't reliably tell restatement from value-substitution,
    and a modest salience nudge + evidence record is low-stakes enough (no
    status change, no relation written) that there's no honest signal left
    worth gating it on.
    """

    def __init__(
        self,
        *,
        schemas: SchemaStore,
        db: "SQLiteDB",
    ):
        self.schemas = schemas
        self.db = db

    # ---- public API ---------------------------------------------------------

    def process_candidates(
        self,
        *,
        new_schema_id: int,
        emb: np.ndarray,
        event_id: int,
        content: str,
        scope_id: str | None,
    ) -> None:
        """Check ``new_schema_id``'s near-embedding neighbors and act.

        Same-scope neighbors are never mutated. Cross-scope neighbors above
        the evidence-association floor are reinforced (see class docstring).
        """
        try:
            # Explicit annotation works around SchemaStore.search_embedding's
            # own broken return-type resolution (its `list[tuple[int, float]]`
            # annotation is shadowed by SchemaStore's own `list()` method
            # earlier in the same class body, silenced there with `# type:
            # ignore[valid-type]` but left broken for any external caller
            # that destructures the result — this is the first one to).
            candidates: list[tuple[int, float]] = self.schemas.search_embedding(
                emb, limit=10, scope_id=None
            )
            for candidate_id, score in candidates:
                if candidate_id == new_schema_id or score < _CROSS_SCOPE_REINFORCEMENT_COSINE:
                    continue
                try:
                    candidate = self.schemas.get(candidate_id)
                except KeyError:
                    continue
                if candidate.status not in ("active", "needs_review"):
                    continue

                if candidate.scope_id == scope_id:
                    continue

                # Cross-scope: re-apply the cross-scope floor now that
                # scope is known (the loop-level pre-filter above uses
                # the lower COS_THRESHOLD_EXTENDED_SAME_SCOPE). Reinforce
                # unconditionally once cleared -- see the class docstring
                # for why this no longer gates on direction_score.
                try:
                    self.schemas.reinforce_schema(
                        candidate_id,
                        salience_delta=0.05,
                        evidence=[(None, event_id, content, 0.5)],
                    )
                except Exception as e:
                    log.warning(
                        "remember: cross-scope reinforce failed for schema %d: %s",
                        candidate_id,
                        e,
                    )
        except Exception as e:
            log.warning("remember: candidate loop failed for schema %d: %s", new_schema_id, e)
