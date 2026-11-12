"""Activation must degrade gracefully, never fail, on any valid task text."""

from __future__ import annotations

import asyncio
import random
from types import SimpleNamespace

import pytest
from mcp.server.fastmcp import FastMCP

from slowave import ops
from slowave.core.config import SlowaveConfig
from slowave.core.engine import SlowaveEngine
from slowave.mcp.activation_catalog import (
    MAX_PREVIEW_CHARS,
    CatalogPreparationError,
    FrozenActivationCatalog,
    degraded_source_preview,
)
from slowave.mcp.tools import register_tools

BULLETED_TASK = (
    "1 - its ok as long as we cannot reach a better result. If >0.8 precision is not "
    "reachable via similarity how would you then design the retrieval algorithm? "
    "Performance is not negotiable (Slowave is already too slow)\n"
    "2 - confirm, some noise is acceptable as long as retrieval works,it should just "
    "minimised\n"
    "3 - any design is ok as long as fixes the above, no new tools preferred\n"
    "4 - no carry on with multi agents, as many as you need to deliver the optimal "
    "feature. Time is not also a blocker\n"
    "more points\n- feel free to apply fixes and evaluate them\n"
)

# One sentence (no terminator) far beyond the preview bound that matches the task.
LONG_RUN_ON = ", ".join(
    f"retrieval design note {i}: if precision above 0.8 is not reachable via similarity, "
    "design the retrieval algorithm with multi agents and keep performance, some noise "
    "is acceptable"
    for i in range(40)
)


def _ctx() -> SimpleNamespace:
    return SimpleNamespace(request_id="req", client_id=None, session=None)


def _setup(tmp_path):
    eng = SlowaveEngine(
        SlowaveConfig(db_path=str(tmp_path / "robust.db"), dim=8, disable_encoder=True)
    )
    mcp = FastMCP("test")
    register_tools(mcp, lambda **_: eng)
    return eng, mcp._tool_manager.get_tool("slowave_activate")


def _activate(tool, task: str, scope: str = "project:p", **extra):
    return asyncio.run(tool.fn(ctx=_ctx(), task=task, initial_goal="work", scope=scope, **extra))


def _seed(eng: SlowaveEngine, scope: str, *contents: str) -> None:
    session = eng.session_start(agent="seed", scope=scope, goal="seed")
    for content in contents:
        ops.remember(eng, content=content, memory_type="decision", scope=scope, session_id=session)


def test_prepare_stays_strict_by_default():
    with pytest.raises(CatalogPreparationError, match="exceed preview"):
        FrozenActivationCatalog.prepare(
            retrieval_id="ctx_1",
            session_id="sess_1",
            scope="project:p",
            memories=[{"memory_id": "sch_1", "content": "x" * 3000}],
            procedures=[],
            contribution_spans={"sch_1": ((0, 3000),)},
            catalog_truncated=False,
        )


def test_degraded_preview_keeps_whole_spans_then_truncates_at_word_boundary():
    source = ("alpha " * 300) + "tail"
    preview, indication, truncated = degraded_source_preview(source, ((0, len(source)),))
    assert truncated and len(preview) <= MAX_PREVIEW_CHARS
    assert indication["truncated"] is True and indication["source_chars"] == len(source)
    assert not preview.endswith("alph")  # cut at a word boundary
    # Whole spans that fit are kept; the one that does not is omitted.
    preview, indication, truncated = degraded_source_preview(
        source, ((0, 600), (700, 1700), (1750, 1780))
    )
    assert truncated and indication["source_spans"] == [[0, 600], [1750, 1780]]
    # Invalid spans never raise.
    degraded_source_preview(source, ((-5, 3), (10, 10**9)))


def test_unicode_heavy_preview_is_shrunk_to_fit_the_record_bound():
    source = "\U0001f600" * 5000  # 12 escaped bytes per character
    catalog = FrozenActivationCatalog.prepare(
        retrieval_id="ctx_1",
        session_id="sess_1",
        scope="project:p",
        memories=[{"memory_id": "sch_1", "content": source, "pathway": "direct"}],
        procedures=[],
        contribution_spans={"sch_1": ((0, 5000),)},
        catalog_truncated=False,
        degrade=True,
    )
    page, _ = catalog.page()
    assert page["memories"] and 0 < len(page["memories"][0]["content"]) <= MAX_PREVIEW_CHARS
    assert any(w["code"] == "memory_excerpt_truncated" for w in page["warnings"])


def test_oversized_procedure_is_skipped_with_a_warning_when_degrading():
    catalog = FrozenActivationCatalog.prepare(
        retrieval_id="ctx_1",
        session_id="sess_1",
        scope="project:p",
        memories=[],
        procedures=[{"procedure_id": "proc_1", "caveats": ["x" * 20000]}],
        contribution_spans={},
        catalog_truncated=False,
        degrade=True,
    )
    page, _ = catalog.page()
    assert page["procedures"] == []
    assert [w["code"] for w in page["warnings"]] == ["procedure_omitted"]


def test_bulleted_task_against_long_run_on_memory_activates(tmp_path):
    eng, tool = _setup(tmp_path)
    try:
        _seed(eng, "project:p", LONG_RUN_ON, "Performance is not negotiable.")
        response = _activate(tool, BULLETED_TASK)
        assert response["ok"] is True, response
        contents = [m["content"] for m in response["data"]["memories"]]
        assert contents and all(len(c) <= MAX_PREVIEW_CHARS for c in contents)
        assert any(w["code"] == "memory_excerpt_truncated" for w in response["data"]["warnings"])
    finally:
        eng.close()


_FRAGMENTS = [
    "retrieval precision similarity threshold",
    "1. fix the cache",
    "- performance is not negotiable",
    "* [ ] run tests",
    "```python\nprint('x')\n```",
    "# Heading",
    "naïve café 日本語 \U0001f600 Привет",
    "If this is unreachable, how would you design it?",
    "\t  ",
    "do not run migration; timeout must stay at 12 seconds.",
]


def _random_task(rng: random.Random) -> str:
    size = rng.choice([1, 5, 40, 200, 700, 3000, 20_000])
    parts: list[str] = []
    total = 0
    while total < size:
        piece = rng.choice(_FRAGMENTS) + rng.choice(["\n", " ", "\n\n", "\r\n", ""])
        parts.append(piece)
        total += len(piece)
    return "".join(parts)[:size] or "x"


def test_fuzzed_tasks_never_fail_against_short_and_huge_memories(tmp_path):
    eng, tool = _setup(tmp_path)
    try:
        rng = random.Random(20260804)
        huge_a = " ".join(rng.choice(_FRAGMENTS) for _ in range(900))[:25_000]
        huge_b = LONG_RUN_ON * 7
        for scope in ("project:p", "project:q"):
            _seed(
                eng,
                scope,
                "Performance is not negotiable.",
                "Run the cache fix tests before deploy.",
                huge_a,
                huge_b[:25_000],
            )
        _seed(eng, "project:q", "Heading retrieval precision similarity threshold design.")
        for i in range(24):
            scope = ("project:p", "project:q", "project:empty")[i % 3]
            response = _activate(tool, _random_task(rng), scope=scope)
            assert response["ok"] is True, (i, scope, response)
            for memory in response["data"]["memories"]:
                assert len(memory["content"]) <= MAX_PREVIEW_CHARS
    finally:
        eng.close()
