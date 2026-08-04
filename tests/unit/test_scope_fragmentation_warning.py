"""Regression test for the 2026-07-24 Tier-0 audit finding: scope strings are
never validated or canonicalized (slowave/core/scope.py only strips
whitespace), so "project:my-repo" and "project:my_repo" silently fragment
into two isolated memory stores with no warning. ops.activate() now surfaces
an advisory `scope_warning` on cold start when an existing scope looks like a
case/separator variant of the "new" one — it never rejects or rewrites the
scope, only flags the collision.
"""

from __future__ import annotations

import os

import numpy as np

from slowave.core.config import SlowaveConfig
from slowave.core.engine import SlowaveEngine
from slowave.ops import activate


def _fake_encoder_engine(db_path: str, dim: int = 16) -> SlowaveEngine:
    cfg = SlowaveConfig(db_path=db_path, dim=dim, disable_encoder=True)
    eng = SlowaveEngine(cfg)

    class _StubEncoder:
        def encode(self, text: str) -> np.ndarray:
            seed = int(abs(hash(text)) % (2**31))
            r = np.random.default_rng(seed)
            v = r.standard_normal(dim).astype(np.float32)
            return v / (np.linalg.norm(v) + 1e-12)

    eng.encoder = _StubEncoder()
    return eng


def _cleanup(path: str) -> None:
    for ext in ("", "-wal", "-shm"):
        p = path + ext
        if os.path.exists(p):
            os.remove(p)


def test_no_warning_for_the_first_scope_ever_seen(tmp_path):
    db = str(tmp_path / "test.db")
    eng = _fake_encoder_engine(db)
    try:
        result = activate(eng, query="hello", scope="project:my-repo")
        assert result["cold_start"] is True
        assert "scope_warning" not in result
    finally:
        eng.close()
        _cleanup(db)


def test_warns_on_separator_variant_of_existing_scope(tmp_path):
    db = str(tmp_path / "test.db")
    eng = _fake_encoder_engine(db)
    try:
        activate(eng, query="hello", scope="project:my-repo")
        result = activate(eng, query="hello again", scope="project:my_repo")
        assert result["cold_start"] is True
        assert "scope_warning" in result
        assert "project:my-repo" in result["scope_warning"]
        assert "project:my_repo" in result["scope_warning"]
    finally:
        eng.close()
        _cleanup(db)


def test_warns_on_case_variant_of_existing_scope(tmp_path):
    db = str(tmp_path / "test.db")
    eng = _fake_encoder_engine(db)
    try:
        activate(eng, query="hello", scope="project:Slowave")
        result = activate(eng, query="hello again", scope="project:slowave")
        assert "scope_warning" in result
    finally:
        eng.close()
        _cleanup(db)


def test_no_warning_for_a_genuinely_different_scope(tmp_path):
    db = str(tmp_path / "test.db")
    eng = _fake_encoder_engine(db)
    try:
        activate(eng, query="hello", scope="project:my-repo")
        result = activate(eng, query="hello", scope="project:totally-different")
        assert result["cold_start"] is True
        assert "scope_warning" not in result
    finally:
        eng.close()
        _cleanup(db)


def test_no_warning_once_scope_is_no_longer_cold_start(tmp_path):
    """The warning only fires on cold start -- once a scope has its own
    memories, repeated activate() calls into it shouldn't keep re-flagging
    a collision the user has presumably already seen (or accepted)."""
    db = str(tmp_path / "test.db")
    eng = _fake_encoder_engine(db)
    try:
        activate(eng, query="hello", scope="project:my-repo")
        eng.remember(content="a fact for my-repo", type="fact", scope="project:my-repo")
        result = activate(eng, query="hello again", scope="project:my-repo")
        assert result["cold_start"] is False
        assert "scope_warning" not in result
    finally:
        eng.close()
        _cleanup(db)
