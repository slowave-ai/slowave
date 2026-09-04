"""Stdio MCP server using real tools and a selected acceptance encoder."""

from __future__ import annotations

import hashlib
import logging
import os
import sqlite3
from collections.abc import Iterable
from functools import wraps
from typing import Any, cast

import numpy as np
from mcp.server.fastmcp import FastMCP

import slowave.mcp.tools as mcp_tools
from slowave.core.config import SlowaveConfig
from slowave.core.engine import SlowaveEngine
from slowave.mcp import session_reaper
from slowave.mcp.tools import register_tools
from slowave.symbolic.encoder import TextEncoder

_TOKEN_HASH_END = 32
_TEMPORAL_CUES: tuple[tuple[str, ...], ...] = (
    ("right now", "today", "at the moment"),
    ("yesterday", "day before"),
    ("a few days ago", "several days ago"),
    (
        "last week",
        "week ago",
        "la settimana scorsa",
        "la semana pasada",
        "la semaine dernière",
        "letzte woche",
        "na semana passada",
    ),
    ("two weeks ago", "fortnight ago"),
    ("last month", "month ago", "recently"),
    ("two months ago", "couple of months ago"),
    ("three months ago", "several months ago"),
    ("six months ago", "half a year ago"),
    ("last year", "year ago"),
    ("two years ago",),
    ("a long time ago", "years ago", "long ago"),
)
DIM = _TOKEN_HASH_END + len(_TEMPORAL_CUES)

CONCEPTS: tuple[tuple[str, ...], ...] = (
    ("database", "postgres", "postgresql", "sql", "ledger"),
    ("authentication", "credential", "credentials", "login", "token", "tokens"),
    ("expire", "expires", "expiry", "valid", "lifetime", "minutes"),
    ("paint", "color", "colour", "palette"),
    ("deployment", "deploy", "release", "production"),
    ("billing", "invoice", "payment"),
    ("cache", "redis"),
    ("queue", "worker", "jobs"),
)


class DeterministicEncoder:
    """Small semantic encoder for transport/contract acceptance, not quality claims."""

    @property
    def dim(self) -> int:
        return DIM

    def encode(self, text: str) -> np.ndarray:
        lowered = text.casefold()
        vector = np.zeros(DIM, dtype=np.float32)
        for index, terms in enumerate(CONCEPTS):
            if any(term in lowered for term in terms):
                vector[index] += 1.0
        for token in _tokens(lowered):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=2).digest()
            vector[8 + int.from_bytes(digest, "big") % (_TOKEN_HASH_END - 8)] += 0.08
        # The compact transport encoder otherwise represents temporal language
        # only through hash buckets. Reserve one deterministic dimension per
        # temporal-probe band so acceptance scenarios exercise a meaningful,
        # stable temporal anchor instead of a collision-selected interval.
        for index, cues in enumerate(_TEMPORAL_CUES):
            if any(cue in lowered for cue in cues):
                vector[_TOKEN_HASH_END + index] += 4.0
        norm = float(np.linalg.norm(vector))
        if norm == 0.0:
            vector[-1] = 1.0
            return vector
        return vector / norm


def _tokens(text: str) -> Iterable[str]:
    token = ""
    for char in text:
        if char.isalnum() or char in "_-.":
            token += char
        elif token:
            if len(token) >= 4:
                yield token
            token = ""
    if len(token) >= 4:
        yield token


_ACTIVATE_LIMIT = int(os.environ.get("SLOWAVE_ACCEPTANCE_ACTIVATE_LIMIT", "2"))
if _ACTIVATE_LIMIT < 1:
    raise ValueError("SLOWAVE_ACCEPTANCE_ACTIVATE_LIMIT must be positive")
# Acceptance-only override: exercise the real public tool with the selected
# compact-context budget without changing the shipped MCP default.
mcp_tools._MCP_ACTIVATE_LIMIT_DEFAULT = _ACTIVATE_LIMIT

_ENGINES: dict[bool, SlowaveEngine] = {}

_ACCEPTANCE_MUTATIONS = frozenset(
    {
        "scope_filtering",
        "stale_suppression",
        "feedback_enforcement",
        "activation_budget",
        "relevance_admission",
    }
)


def _apply_acceptance_mutation(name: str) -> None:
    """Inject one named regression into this acceptance server only.

    The mutation registry exists solely to prove that launch-critical public
    scenarios fail when their corresponding implementation invariant breaks.
    It is never loaded by the shipped MCP server or production code paths.
    """
    if name not in _ACCEPTANCE_MUTATIONS:
        raise ValueError(f"unknown acceptance mutation: {name}")

    import slowave.mcp.tools as mcp_tools

    original_activate = mcp_tools.ops.activate
    original_recall = mcp_tools.ops.recall
    original_commit = mcp_tools.ops.commit
    if name == "stale_suppression":
        original_canonical = mcp_tools._canonical_activation_result

        @wraps(original_canonical)
        def mutated_canonical(result: dict[str, Any], *, scope: str) -> dict[str, Any]:
            data = original_canonical(result, scope=scope)
            # Re-introduce the actual retired row into the current response.
            # This models stale suppression being removed while keeping the
            # returned ID valid for the normal feedback lifecycle.
            connection = sqlite3.connect(os.environ["SLOWAVE_DB"])
            row = connection.execute(
                "SELECT id, content_text FROM schemas WHERE status = 'stale' ORDER BY id LIMIT 1"
            ).fetchone()
            connection.close()
            if row is not None:
                data["memories"].append(
                    {
                        "memory_id": f"sch_{row[0]}",
                        "content": str(row[1]),
                        "pathway": "direct",
                    }
                )
            return data

        mcp_tools._canonical_activation_result = mutated_canonical

    @wraps(original_activate)
    def mutated_activate(*args: Any, **kwargs: Any) -> dict[str, Any]:
        if name == "scope_filtering":
            kwargs["scope"] = None
        elif name == "activation_budget":
            kwargs["limit"] = max(int(kwargs.get("limit", 0)), 100)
        elif name == "relevance_admission":
            kwargs["min_relevance"] = 0.0
        return original_activate(*args, **kwargs)

    @wraps(original_recall)
    def mutated_recall(*args: Any, **kwargs: Any) -> dict[str, Any]:
        if name == "scope_filtering":
            kwargs["scope"] = None
        elif name == "stale_suppression":
            kwargs["mode"] = "debug"
        elif name == "relevance_admission":
            kwargs["min_relevance"] = 0.0
        return original_recall(*args, **kwargs)

    @wraps(original_commit)
    def mutated_commit(*args: Any, **kwargs: Any) -> dict[str, Any]:
        if name == "feedback_enforcement":
            kwargs["enforce_feedback"] = False
        return original_commit(*args, **kwargs)

    mcp_tools.ops.activate = mutated_activate
    mcp_tools.ops.recall = mutated_recall
    mcp_tools.ops.commit = mutated_commit


def build_engine(disable_encoder: bool = False) -> SlowaveEngine:
    existing = _ENGINES.get(disable_encoder)
    if existing is not None:
        return existing
    db_path = os.environ["SLOWAVE_DB"]
    encoder_mode = os.environ.get("SLOWAVE_ACCEPTANCE_ENCODER", "deterministic")
    if encoder_mode not in {"deterministic", "production"}:
        raise ValueError(f"unknown SLOWAVE_ACCEPTANCE_ENCODER: {encoder_mode}")
    if encoder_mode == "production":
        engine = SlowaveEngine(
            SlowaveConfig(db_path=db_path, dim=384, disable_encoder=disable_encoder)
        )
        _ENGINES[disable_encoder] = engine
        return engine
    encoder = None if disable_encoder else DeterministicEncoder()
    engine = SlowaveEngine(
        SlowaveConfig(db_path=db_path, dim=DIM, disable_encoder=True),
        shared_encoder=cast(TextEncoder, encoder),
    )
    _ENGINES[disable_encoder] = engine
    return engine


mcp = FastMCP("slowave-retrieval-acceptance")
register_tools(mcp, build_engine)


if __name__ == "__main__":
    if os.environ.get("SLOWAVE_ACCEPTANCE_QUIET") == "1":
        # FastMCP emits request-level INFO logs to stderr by default.  They
        # obscure the acceptance scenario being exercised when -s is used.
        # Keep errors visible; opt into full child diagnostics explicitly.
        logging.basicConfig(level=logging.ERROR, force=True)
        for logger_name in ("transformers", "huggingface_hub"):
            logging.getLogger(logger_name).setLevel(logging.ERROR)
        # Negative contract cases intentionally trigger handled tool errors
        # (for example invalid scope and occurred_at).  Their MCP envelopes
        # are asserted by the tests, so ERROR logbacks only obscure progress.
        logging.getLogger("slowave.mcp.tools").setLevel(logging.CRITICAL)
        try:
            from transformers.utils import logging as transformers_logging

            transformers_logging.set_verbosity_error()
        except ImportError:
            pass
    # The production daemon runs this reaper as well.  Acceptance tests can
    # shorten its polling interval and timeout to exercise abandoned-session
    # closure without waiting for the production one-hour defaults.
    poll_interval = float(os.environ.get("SLOWAVE_ACCEPTANCE_REAPER_POLL_INTERVAL", "120"))
    session_reaper.start(build_engine=build_engine, poll_interval_s=poll_interval)
    mutation = os.environ.get("SLOWAVE_ACCEPTANCE_MUTATION")
    if mutation:
        _apply_acceptance_mutation(mutation)
    mcp.run(transport="stdio")
