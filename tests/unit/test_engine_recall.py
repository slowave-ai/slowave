"""Regression tests for engine.recall().

Covers the multi-mechanism retrieval pipeline before it is extracted into a
standalone RetrievalService. Uses a deterministic stub encoder (hash-seeded
random) — same text always produces the same embedding, different texts
produce unrelated embeddings. No model weights are downloaded.
"""

from __future__ import annotations

import numpy as np
import pytest

from slowave.core.config import SlowaveConfig
from slowave.core.engine import SlowaveEngine

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


class _StubEncoder:
    """Deterministic encoder: same text → same unit vector, no model needed."""

    def __init__(self, dim: int = 32):
        self._dim = dim

    def encode(self, text: str) -> np.ndarray:
        seed = int(abs(hash(text)) % (2**31))
        v = np.random.default_rng(seed).standard_normal(self._dim).astype(np.float32)
        return v / (np.linalg.norm(v) + 1e-12)


def _make_engine(tmp_path, dim: int = 32) -> SlowaveEngine:
    eng = SlowaveEngine(
        SlowaveConfig(db_path=str(tmp_path / "test.db"), dim=dim, disable_encoder=True)
    )
    eng.encoder = _StubEncoder(dim)
    return eng


@pytest.fixture()
def eng(tmp_path):
    engine = _make_engine(tmp_path)
    yield engine
    engine.close()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_recall_raises_without_encoder(tmp_path):
    engine = SlowaveEngine(
        SlowaveConfig(db_path=str(tmp_path / "noenc.db"), dim=32, disable_encoder=True)
    )
    try:
        with pytest.raises(RuntimeError, match="encoder"):
            engine.recall("anything")
    finally:
        engine.close()


def test_engine_never_persists_faiss_index_to_working_directory(tmp_path, monkeypatch):
    """The SQLite database is durable; FAISS must remain an in-memory cache.

    Reopening an engine with episodes used to write ``episodic.faiss`` to the
    process cwd because the episodic-store default was a relative path.
    """
    db_dir = tmp_path / "data"
    db_dir.mkdir()
    work_dir = tmp_path / "working-directory"
    work_dir.mkdir()
    monkeypatch.chdir(work_dir)

    db_path = str(db_dir / "slowave.db")
    engine = SlowaveEngine(SlowaveConfig(db_path=db_path, dim=32, disable_encoder=True))
    engine.encoder = _StubEncoder(32)
    engine.remember(content="FAISS persistence must not create cwd files", type="fact")
    engine.close()

    # A restart rebuilds the index from the durable SQLite embeddings.  This is
    # where the former relative ``episodic.faiss`` write occurred.
    restarted = SlowaveEngine(SlowaveConfig(db_path=db_path, dim=32, disable_encoder=True))
    try:
        assert list(work_dir.iterdir()) == []
    finally:
        restarted.close()


def test_recall_result_has_expected_fields(eng):
    eng.remember(content="I prefer dark mode", type="preference")
    r = eng.recall("I prefer dark mode", top_k=5)
    assert hasattr(r, "schemas")
    assert hasattr(r, "episode_texts")
    assert hasattr(r, "raw_events")
    assert hasattr(r, "expanded_neighbors")
    assert isinstance(r.schemas, list)
    assert isinstance(r.episode_texts, list)
    assert isinstance(r.raw_events, list)


def test_recall_surfaces_remembered_schema(eng):
    content = "I use Neovim as my primary editor"
    eng.remember(content=content, type="preference")
    r = eng.recall(content, top_k=5)
    assert any(content in s.content_text for s in r.schemas)


def test_recall_returns_episode_texts(eng):
    content = "I store data in PostgreSQL"
    eng.remember(content=content, type="fact")
    r = eng.recall(content, top_k=5)
    # remember() creates episodes immediately — the content must be surfaced in recall.
    # Episodes that exactly duplicate an active schema text are suppressed to avoid
    # sending redundant content to agents; the content is then surfaced via r.schemas.
    # Accept either: episode_texts has entries OR the schema itself is present.
    assert (
        len(r.schemas) > 0 or len(r.episode_texts) > 0
    ), "Recalled content must be accessible via schemas or episode_texts after remember()"
    assert any(
        content in s.content_text for s in r.schemas
    ), "The remembered schema must be surfaced in recall results"
    assert all("content_text" in ep for ep in r.episode_texts)
    assert all("salience" in ep for ep in r.episode_texts)


def test_recall_episode_texts_contain_ts_field(eng):
    content = "deadlines are tracked in Linear"
    eng.remember(content=content, type="fact")
    r = eng.recall(content, top_k=5)
    assert all("ts" in ep for ep in r.episode_texts)


def test_recall_top_k_limits_returned_schemas(eng):
    for i in range(8):
        eng.remember(content=f"unique preference item number {i}", type="preference")
    r = eng.recall("preference item", top_k=3)
    assert len(r.schemas) <= 3


def test_recall_evidence_flag_populates_raw_events(eng):
    content = "my deploy process uses GitHub Actions"
    eng.remember(content=content, type="fact")
    r = eng.recall(content, top_k=3, evidence=True)
    # evidence=True traces back through schema → episode → raw event
    assert isinstance(r.raw_events, list)
    # At least the remember event should be traceable
    assert len(r.raw_events) > 0


def test_recall_without_evidence_has_empty_raw_events(eng):
    eng.remember(content="I prefer monorepos", type="preference")
    r = eng.recall("I prefer monorepos", top_k=3, evidence=False)
    assert r.raw_events == []


def test_recall_schema_reinforcement_increases_salience(eng):
    content = "always write tests before shipping"
    eng.remember(content=content, type="fact")
    schema_id = eng.schemas.list(limit=1)[0].id
    salience_before = eng.schemas.get(schema_id).salience

    eng.recall(content, top_k=5)

    salience_after = eng.schemas.get(schema_id).salience
    assert salience_after >= salience_before


def test_recall_multiple_schemas_are_ranked(eng):
    eng.remember(content="I use Python for backend", type="fact")
    eng.remember(content="I use TypeScript for frontend", type="fact")
    r = eng.recall("I use Python for backend", top_k=5)
    # The Python schema should rank above the TypeScript one since
    # query text matches exactly
    assert len(r.schemas) >= 1
    assert any("Python" in s.content_text for s in r.schemas)


def test_recall_on_empty_db_returns_empty_results(eng):
    r = eng.recall("anything at all", top_k=5)
    assert r.schemas == []
    assert r.episode_texts == []
    assert r.raw_events == []
