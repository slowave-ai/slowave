"""Input-size bounds: MCP argument limits, warmup and lexical-token scaling."""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from slowave.mcp import tools


@pytest.mark.parametrize(
    "model,payload",
    [
        (
            tools.ActivateArguments,
            {"task": "x" * (tools.MAX_TASK_CHARS + 1), "initial_goal": "g", "scope": "project:p"},
        ),
        (
            tools.ActivateArguments,
            {"task": "t", "initial_goal": "g" * (tools.MAX_GOAL_CHARS + 1), "scope": "project:p"},
        ),
        (
            tools.ActivateArguments,
            {
                "task": "t",
                "initial_goal": "g",
                "scope": "project:p",
                "semantic_context": "x" * (tools.MAX_TASK_CHARS + 1),
            },
        ),
        (
            tools.RecallArguments,
            {
                "session_id": "s",
                "scope": "project:p",
                "query": "x" * (tools.MAX_TASK_CHARS + 1),
            },
        ),
        (
            tools.RememberArguments,
            {
                "scope": "project:p",
                "session_id": "s",
                "content": "x" * (tools.MAX_MEMORY_CHARS + 1),
                "type": "fact",
            },
        ),
        (
            tools.RememberArguments,
            {
                "scope": "project:p",
                "session_id": "s",
                "memories": [{"content": "x" * (tools.MAX_MEMORY_CHARS + 1), "type": "fact"}],
            },
        ),
    ],
)
def test_oversized_inputs_are_rejected_with_structured_error(model, payload):
    with pytest.raises(ValidationError) as exc:
        model.model_validate(payload)
    message, field_errors = tools._format_validation_error(exc.value)
    assert field_errors
    assert "too_long" in str(exc.value) or "at most" in str(exc.value)


def test_normal_inputs_are_accepted_at_the_limit():
    tools.ActivateArguments.model_validate(
        {"task": "x" * tools.MAX_TASK_CHARS, "initial_goal": "g", "scope": "project:p"}
    )
    tools.RememberArguments.model_validate(
        {
            "scope": "project:p",
            "session_id": "s",
            "content": "x" * tools.MAX_MEMORY_CHARS,
            "type": "fact",
        }
    )


def test_warmup_prepares_retrieval_in_background_and_can_be_disabled(monkeypatch):
    import threading

    from slowave.mcp import server

    calls = []
    release = threading.Event()

    class Engine:
        def prepare_retrieval(self, *, wait):
            calls.append(wait)
            release.wait(5)
            return {"ready": True}

    monkeypatch.delenv("SLOWAVE_MCP_WARMUP", raising=False)
    thread = server.start_warmup(Engine)
    assert thread is not None and thread.is_alive()  # returned without blocking
    release.set()
    thread.join(5)
    assert calls == [True]

    monkeypatch.setenv("SLOWAVE_MCP_WARMUP", "0")
    assert server.start_warmup(Engine) is None


def test_warmup_failure_is_swallowed(monkeypatch):
    from slowave.mcp import server

    def boom():
        raise RuntimeError("no models")

    monkeypatch.delenv("SLOWAVE_MCP_WARMUP", raising=False)
    thread = server.start_warmup(boom)
    thread.join(5)
    assert not thread.is_alive()


def _reference_lexical_tokens(query: str) -> list[str]:
    """The original quadratic implementation, kept as the behavioural oracle."""
    import re

    tokens: list[str] = []
    for raw in re.findall(r"[^\W_]+(?:[_./:-][^\W_]+)*", query, flags=re.UNICODE):
        variants = [raw]
        variants.extend(part for part in re.split(r"[_./:-]+", raw) if part)
        for token in variants:
            token = token.strip()
            if len(token) >= 2 and token not in tokens:
                tokens.append(token)
    return tokens


@pytest.mark.parametrize(
    "text",
    [
        "",
        "a",
        "Fix the refund_webhook retry.cache in ledger/service a-b-c x:y 日本語 テスト",
        "Fix the Refund fix the refund a b c 12 12 ab ab-cd ab_cd ab.cd " * 40,
        " ".join(f"tok{i % 700}_{i % 13}" for i in range(6000)),  # > cache threshold
    ],
)
def test_lexical_tokens_matches_reference_and_is_not_aliased(text):
    from slowave.symbolic.schema_store import SchemaStore

    expected = _reference_lexical_tokens(text)
    first = SchemaStore.lexical_tokens(text)
    assert first == expected
    first.append("mutated")  # callers own the returned list
    assert SchemaStore.lexical_tokens(text) == expected


def test_lexical_tokens_scales_linearly_for_long_input():
    import time

    from slowave.symbolic.schema_store import SchemaStore

    text = " ".join(f"uniq{i}" for i in range(60_000))  # ~420 KB, 60k distinct tokens
    started = time.perf_counter()
    assert len(SchemaStore.lexical_tokens(text)) == 60_000
    # The quadratic membership test needed minutes; linear is well under a second.
    assert time.perf_counter() - started < 20
