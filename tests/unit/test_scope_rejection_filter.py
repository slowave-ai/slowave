"""Regression test for ops.py's broken scope-rejection exclusion (2026-07-23).

activate()'s filtered_items list is meant to exclude scope-related rejections
from context_recall_items (they aren't an interesting "close but filtered"
candidate worth persisting). The original check (`reason != "scope_mismatch"`)
was an exact-match test against a bare string that WorkingMemoryGate never
actually produces -- real reason strings are either a compound diagnostic
ending in ",scope_mismatch" (from _activation()) or the unrelated literal
"strict_scope_excluded" (from _eligible()'s hard scope wall). The broken
check let every scope-rejected candidate through, which is what fed the
cross-scope co-activation leak.
"""

from __future__ import annotations

from slowave.ops import _is_scope_rejection


def test_compound_diagnostic_reason_ending_in_scope_mismatch():
    # Exact reason string observed in the live DB for a rejected candidate.
    reason = "cosine=0.22,cue_overlap=0.04,salience=0.07,constraint,utility=0.18,profile,explicit,scope_mismatch"
    assert _is_scope_rejection(reason)


def test_strict_scope_excluded_literal():
    assert _is_scope_rejection("strict_scope_excluded")


def test_cross_scope_below_floor():
    assert _is_scope_rejection("cross_scope_below_floor")


def test_cross_scope_low_cosine_prefix():
    assert _is_scope_rejection("cross_scope_low_cosine:0.19")


def test_non_scope_rejections_are_not_excluded():
    assert not _is_scope_rejection("below_activation")
    assert not _is_scope_rejection("inactive")
    assert not _is_scope_rejection("class_excluded:latent")


def test_old_exact_match_check_would_have_missed_the_real_reason_strings():
    """Sanity-check documenting the actual bug: the old check
    (`reason != "scope_mismatch"`) never matched the real compound string."""
    reason = "cosine=0.22,cue_overlap=0.04,salience=0.07,constraint,utility=0.18,profile,explicit,scope_mismatch"
    assert reason != "scope_mismatch"  # old check thought this WASN'T a scope rejection
    assert _is_scope_rejection(reason)  # new check correctly identifies it as one
