"""Session setup for the black-box MCP acceptance suite."""

from __future__ import annotations

import os

import pytest


@pytest.fixture(scope="session", autouse=True)
def _prepare_production_encoder() -> None:
    """Download the production model once, then make MCP children cache-only.

    Every scenario gets its own MCP subprocess and database so that tests are
    isolated.  The subprocess boundary cannot share an in-memory encoder, but
    it can share the Hugging Face cache.  A single preflight avoids repeated
    network metadata checks and makes missing model files fail at suite start.
    """
    if os.environ.get("SLOWAVE_ACCEPTANCE_ENCODER", "deterministic") != "production":
        return

    if os.environ.get("SLOWAVE_ACCEPTANCE_VERBOSE") != "1":
        os.environ.setdefault("SLOWAVE_ACCEPTANCE_QUIET", "1")

    # Configure dependency verbosity before importing the tokenizer stack; the
    # transformers import itself can emit an advisory when torch is absent.
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    os.environ["TRANSFORMERS_VERBOSITY"] = "error"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    from slowave.symbolic.encoder import TextEncoder

    try:
        # Accessing dim forces lazy ONNX model and tokenizer initialization.
        TextEncoder().dim
    except Exception as exc:  # pragma: no cover - exercised by environment failures
        raise pytest.UsageError(
            "The production acceptance encoder could not be prepared. "
            "Check model dependencies/network access and retry."
        ) from exc

    # Child MCP servers must reuse the files prepared above and must not make
    # one network request per test.  These variables are inherited by the
    # stdio subprocesses created by tests/acceptance/mcp_harness.py.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"


def pytest_report_header() -> str:
    encoder = os.environ.get("SLOWAVE_ACCEPTANCE_ENCODER", "deterministic")
    return f"Slowave MCP acceptance scenarios (encoder: {encoder}; isolated server per test)"
