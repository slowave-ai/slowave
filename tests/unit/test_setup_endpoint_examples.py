"""Keep global client instructions compact."""

from slowave.cli.setup import _clients, _lifecycle_block
from slowave.mcp.activation_catalog import DEFAULT_MEMORY_PAGE_SIZE


def test_all_clients_receive_compact_lifecycle_rules():
    for client in _clients():
        block = _lifecycle_block(client.lifecycle_agent)
        # Keep global rules compact and below the pre-fix v13 block (3901 chars).
        assert len(block) <= 3500
        assert "```json" not in block
        assert "schemas" in block and "conditional requirements" in block
        assert "**Cold start:** read one stable context document" in block
        assert "Do not scan the whole codebase" in block
        assert "including continuations and empty results" in block
        assert 'coverage="complete"' in block
        assert "each batch item's" in block
        assert f"server default ({DEFAULT_MEMORY_PAGE_SIZE})" in block
        assert "maximum memories/page" in block and "(not guaranteed)" in block
        assert "omit with `continue_from`" in block


def test_compaction_preserves_lifecycle_behavior_and_safeguards():
    block = _lifecycle_block("codex")
    required = [
        "Call `slowave_activate` once",
        "verbatim task",
        "concise action-led `initial_goal`",
        "stable scope",
        "falling back to `project:<basename(cwd)>`",
        "resend it unchanged on later tasks in this conversation only",
        "materially new questions",
        "Call `slowave_remember` without an explicit request",
        "when you discover new knowledge worth retaining as durable memory",
        "because it could help you with future tasks",
        "Save standalone facts, preferences, decisions, constraints",
        "skip duplicates, speculation, progress notes, pending steps",
        "and temporary task state",
        "First save any missed useful claims",
        "if none qualify, no write is needed",
        "including continuations and empty results",
        'coverage="complete"',
        "for all returned targets",
        "after the work, before commit",
        "Send `slowave_feedback`",
        "`used` requires actual influence",
        "follow the tool's conditional rules",
        "Report `partial` or `failure` honestly",
        "specific, standalone future-facing knowledge",
        "attempted reusable multi-step method",
        "two ordered task actions",
        "omit it for trivial/answer-only work",
        "entries task-only",
        "Verify `feedback_status` and `verification_status`",
        "**Cold start:** read one stable context document",
        "remember selectively using",
        "Do not scan the whole codebase",
        "Account for every warning",
        "Preserve cursors exactly",
        "continue only when more context would help",
        "A continuation sends only\n`session_id`, `scope`, and `continue_from`",
        "Never activate because of a hook, stop event, system reminder, or injected follow-up",
        "Do not invent IDs, scope, continuity, cursors, or success",
        '`{"ok":false,...}` as failure',
        "each batch item's `ok`/data",
        "outer success is insufficient",
        "For feedback batches use `items` alone, with coverage inside each item",
        "Correct\nrejected/outstanding feedback before committing",
        "`incomplete_feedback`",
        "missing assessments with complete coverage and retry",
    ]
    for requirement in required:
        assert requirement in block, requirement
