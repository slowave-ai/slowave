"""Single source of truth for the Slowave lifecycle-instructions version.

"Lifecycle" here means the 5-verb cognitive-cycle contract (activate /
remember / recall / reinforce / commit) that clients are told to follow --
both via the injected block in CLAUDE.md/.clinerules/etc.
(slowave/cli/setup.py's ``_LIFECYCLE_BLOCK_TEMPLATE``) and via the MCP tool
docstrings themselves. Bump this constant whenever that contract changes in
a way rollout telemetry should be able to distinguish (new/renamed verb,
changed call order, changed required fields).

Two independent things read this constant:
  - slowave/cli/setup.py stamps it into the injected block's HTML markers,
    and slowave/cli/clients.py compares an installed block's marker against
    it to detect a stale (un-upgraded) client integration.
  - slowave/core/engine.py stamps it onto every new session row, so
    activate/recall/feedback counts can later be grouped by which lifecycle
    contract version was in effect when the session ran (see WP-8,
    private/docs/iterations/20260728_retrieval_quality_execution_progress.md).

The two uses are independent: a session's stamped version reflects the
*server's* current contract, not necessarily what's physically written into
that machine's client instruction file (which can lag until `slowave setup`
is re-run) -- `slowave doctor` reports the latter.
"""

from __future__ import annotations

LIFECYCLE_VERSION = "v3"
