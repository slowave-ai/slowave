"""PatternCompletionService: encoding-time familiarity check against existing schemas.

Brain framing: this is the fast, hippocampal familiarity check that happens
*at encoding time* — "does this resemble something I already know, and if so,
should that existing trace be marked provisionally uncertain pending
reconsolidation?" That's a distinct cognitive step from ingest (encoding a new
experience), consolidation (offline replay/abstraction), retrieval (spreading
activation), or feedback (reinforcement from outcome).

Extracted from ``SlowaveEngine.remember()`` — see
``private/docs/iterations/20260721_engine_complexity_review_and_simplification_plan.md``
(Fix 3) for the rationale, and
``private/docs/iterations/20260720_supersession_classification_investigation.md``
for why ``remember()`` no longer classifies supersedes/refines/relates_to
itself.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from slowave.core.supersession_manifold import (
    COS_THRESHOLD_CROSS_SCOPE,
    COS_THRESHOLD_EXTENDED_SAME_SCOPE,
)
from slowave.symbolic.schema_store import SchemaStore

if TYPE_CHECKING:
    from slowave.storage.sqlite_db import SQLiteDB

log = logging.getLogger(__name__)


class PatternCompletionService:
    """Checks a freshly-remembered schema against its near-embedding neighbors.

    remember() is deliberately encoding-only for same-scope pairs: it stores
    the new schema and, here, on the pattern-completion side, flags any close
    same-scope neighbor as labile — it no longer decides
    supersedes/refines/relates_to itself.

    Two independent classifiers used to exist for that taxonomy: this one
    (raw SupersessionManifold.direction_score, no facet or containment
    signal) and GeometricContradictionJudge (facet_distance + containment +
    direction_score jointly, used by consolidation). remember()'s copy was
    structurally the weaker of the two, and on single-episode schemas (the
    common case for a fresh remember) it had no facet signal to fall back on
    — see
    private/docs/iterations/20260720_supersession_classification_investigation.md
    for the incident and measurement work that motivated removing it rather
    than continuing to patch it.

    `is_labile=True` already discounts a schema's default recall score 5x
    (retrieval.py) — a retrieval-time "this trace is temporarily uncertain"
    signal, the same role a reactivated-but-not-yet-reconsolidated trace
    plays in the brain — and is picked up by reconsolidate_labile_schemas()
    on the next consolidation pass, which runs the real judge (facet-aware,
    chronology-correct old/new) and writes whatever typed relation actually
    holds. Nothing on the same-scope side writes to schema_relations or
    changes a candidate's status.

    Cross-scope handling below is NOT part of the relation-classification
    taxonomy described above: it never writes a schema_relations edge or a
    status change either. It feeds the promotion ladder instead —
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

        Same scope, non-profile neighbor above the extended-range cosine
        floor: flagged ``is_labile=True`` (see class docstring).
        Cross-scope neighbor above the cross-scope cosine floor: reinforced
        unconditionally (see class docstring).
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
                if candidate_id == new_schema_id or score < COS_THRESHOLD_EXTENDED_SAME_SCOPE:
                    continue
                try:
                    candidate = self.schemas.get(candidate_id)
                except KeyError:
                    continue
                if candidate.status not in ("active", "needs_review"):
                    continue

                # Profile-layer memories (preferences, constraints, habits)
                # must never be flagged labile — a preference flipping from
                # "dark mode" to "light mode" is a divergence, not a
                # candidate for reconsolidation to weigh as a possible
                # supersession. GeometricContradictionJudge has no
                # profile-layer awareness of its own, so this guard has to
                # live at the one place deciding whether a schema becomes
                # eligible for reconsolidation in the first place. Cross-scope
                # reinforcement below no longer distinguishes profile from
                # non-profile candidates (see class docstring).
                _mem_layer = str(candidate.facets.get("memory_layer", "")).lower()
                _mem_class = str(candidate.facets.get("schema_class", "")).lower()
                _is_profile = _mem_layer == "profile" or _mem_class in {
                    "preference",
                    "interaction_preference",
                    "constraint",
                    "habit",
                    "relationship",
                }

                if candidate.scope_id == scope_id:
                    if _is_profile:
                        continue
                    try:
                        self.schemas.adjust_feedback_state(candidate_id, is_labile=True)
                    except Exception as e:
                        log.warning(
                            "remember: adjust_feedback_state failed for schema %d: %s",
                            candidate_id,
                            e,
                        )
                    continue

                # Cross-scope: re-apply the cross-scope floor now that
                # scope is known (the loop-level pre-filter above uses
                # the lower COS_THRESHOLD_EXTENDED_SAME_SCOPE). Reinforce
                # unconditionally once cleared -- see the class docstring
                # for why this no longer gates on direction_score.
                if score < COS_THRESHOLD_CROSS_SCOPE:
                    continue

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
