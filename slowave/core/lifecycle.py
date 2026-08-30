"""Lifecycle-event classification for Slowave's own operations.

Slowave distinguishes *memories of the world/task* (which may be consolidated
into episodic/declarative memory) from *bookkeeping about Slowave's own
operations* (which must never become durable semantic content). The brain
analogue: the hippocampus encodes perceived, external sequences as episodic
trace, while internally-generated control/self-monitoring signals are not
transferred to the neocortex as semantic facts.

Events classified here are Slowave's *own* lifecycle vocabulary, and only
that vocabulary. The detector is deliberately narrow and exact — it matches
server-owned patterns (synthetic recall-cue logs, the activate ``context_query``
type) and a small set of canonical lifecycle phrases a client might narrate in
a commit trajectory. It is **not** free-text NLP: a genuine task observation
(such as "Added a slowave_recall tool test" or "Fixed the slowave session
resolver") does not match any of these patterns, so it remains eligible.

Where this runs:
  - ``IngestService.form_episodes`` excludes lifecycle events from episode
    formation regardless of their stored ``memory_role``. This is what lets a
    logic-version rebuild decontaminate historical events (already persisted
    as ``experience``) without rewriting ``raw_events``.
  - ``ops.commit`` drops lifecycle entries from a client-supplied trajectory
    before they are persisted, so they never enter ``raw_events`` as
    ``experience`` in the first place.
"""

from __future__ import annotations

# Event types that are purely Slowave-internal cues or markers, never episodic
# content.
#   ``context_query``  — the server's own ``slowave_activate`` log (a retrieval
#                       cue, already excluded from episode formation by type).
#   ``task_complete``  — the server's own ``slowave_commit`` marker. It is
#                       written as ``procedural_evidence`` (never embedded) by
#                       current code, but legacy events from before that fix
#                       can carry a stored embedding and no ``memory_role``
#                       (so they default to ``experience``). Excluding the type
#                       unconditionally keeps them out of episodic/declarative
#                       consolidation regardless of stored role or embedding —
#                       this is what lets a logic-version rebuild decontaminate
#                       historical ``outcome=...`` schemas.
_LIFECYCLE_EVENT_TYPES = frozenset({"context_query", "task_complete"})

# Synthetic recall-cue logs produced by the MCP lifecycle tools. The server
# writes these as ``trajectory:action`` events with content ``slowave_recall:
# <query>``; the query is a *cue*, not a learned fact or observation.
_LIFECYCLE_TOOL_PREFIXES = (
    "slowave_activate:",
    "slowave_remember:",
    "slowave_recall:",
    "slowave_feedback:",
    "slowave_commit:",
)

# Canonical lifecycle phrases a client might narrate as commit-trajectory
# entries. Drawn from observed live contamination (2026-08-18): clients have
# described their own Slowave lifecycle steps instead of their task. Matching
# is exact-after-normalization, so ordinary task wording is unaffected.
_LIFECYCLE_PHRASES = frozenset(
    {
        "activated the slowave session",
        "committed the slowave session",
        "committed the session",
        "the slowave session was activated",
        "the slowave session was committed",
        "submitted complete retrieval feedback",
        "submitted retrieval feedback",
        "completed retrieval feedback",
        "completed lifecycle feedback",
    }
)


def normalize(text: str) -> str:
    """Lowercase, collapse whitespace, and strip trailing punctuation."""
    if not text:
        return ""
    return " ".join(str(text).strip().lower().split()).strip(" .,;:!?")


def is_slowave_lifecycle(event_type: str | None, content: str | None) -> bool:
    """True iff an event is Slowave's own lifecycle bookkeeping (not memory).

    Matches the closed set of server-owned lifecycle patterns. Any event
    returning True must be kept out of episodic/declarative consolidation but
    may remain auditable raw history.
    """
    if event_type in _LIFECYCLE_EVENT_TYPES:
        return True
    text = str(content or "")
    if not text.strip():
        return False
    lowered = text.strip().lower()
    for prefix in _LIFECYCLE_TOOL_PREFIXES:
        if lowered.startswith(prefix):
            return True
    return normalize(text) in _LIFECYCLE_PHRASES
