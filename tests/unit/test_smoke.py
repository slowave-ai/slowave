"""Smoke test: import everything, build the engine, run a trivial end-to-end pass.

Uses synthetic numpy embeddings (no sentence-transformers download). Verifies the latent substrate + schema tables
are wired correctly even without external services.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner

from slowave.cli.main import cli
from slowave.core.config import SlowaveConfig
from slowave.core.engine import SlowaveEngine
from slowave.latent.synthetic import SyntheticConfig, generate_synthetic_events


def test_imports() -> None:
    # Touched all the main modules.
    import slowave  # noqa
    import slowave.cli.main  # noqa
    import slowave.core.consolidation  # noqa
    import slowave.core.engine  # noqa
    import slowave.latent.episodic_store  # noqa
    import slowave.latent.replay_engine  # noqa


def test_cli_uses_default_db_from_env_without_db_arg(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """CLI commands should not require --db for normal local usage."""
    db_path = tmp_path / "nested" / "slowave.db"
    monkeypatch.delenv("SLOWAVE_HOME", raising=False)
    monkeypatch.setenv("SLOWAVE_DB", str(db_path))

    runner = CliRunner()
    result = runner.invoke(cli, ["--json", "stats"])

    assert result.exit_code == 0, result.output
    assert db_path.exists()


def test_engine_latent_only_synthetic() -> None:
    """Build engine without encoder; ingest via the original latent path.

    This validates SlowWave's substrate still works under the new package name.
    """
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    try:
        cfg = SlowaveConfig(
            db_path=tmp.name,
            dim=64,
            disable_encoder=True,
        )
        eng = SlowaveEngine(cfg)

        # Generate synthetic events and feed them directly into the latent store.
        synth = SyntheticConfig(dim=64, seed=11)
        events = generate_synthetic_events(200, synth)
        for ev in events:
            # Mirror SlowWave's old direct-to-latent path.
            if eng.episodic.count() >= 1:
                scores, _ = eng.episodic.search(ev.embedding, top_k=1)
                nn_sim = float(scores[0])
            else:
                nn_sim = -1.0
            s = eng.salience.compute_novelty_salience(nn_similarity=nn_sim)
            eng.episodic.add(
                event_id=ev.event_id,
                ts=ev.timestamp,
                embedding=ev.embedding,
                salience=s,
                metadata={"type": ev.type, "entities": ev.entities, **ev.metadata},
            )

        replay_stats = eng.replay_engine.replay_once()
        assert eng.episodic.count() == 200
        assert eng.semantic.count() > 0
        assert "transition_loss" in replay_stats

        # Query: nearest to a perturbed embedding of an existing event.
        q = events[-1].embedding + 0.05 * np.random.default_rng(42).normal(size=(64,)).astype(
            np.float32
        )
        q = q / (np.linalg.norm(q) + 1e-12)
        retrieved = eng.retrieval.retrieve(q)
        assert len(retrieved.episodic) > 0
        # No schemas until consolidation runs.
        assert eng.schemas.count() == 0

        eng.close()
    finally:
        for ext in ("", "-wal", "-shm"):
            p = tmp.name + ext
            if os.path.exists(p):
                os.remove(p)


def test_engine_symbolic_no_llm() -> None:
    """Engine + raw_events + episode_text path with latent schemas.

    Feeds synthetic embeddings directly via the engine's symbolic surface to
    confirm raw_log + episode_text + session lifecycle all behave.
    """
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    try:
        cfg = SlowaveConfig(
            db_path=tmp.name,
            dim=64,
            disable_encoder=True,
        )
        eng = SlowaveEngine(cfg)

        sid = eng.session_start(agent="test", scope="project:smoke")
        rng = np.random.default_rng(0)
        for i in range(10):
            emb = rng.normal(size=(64,)).astype(np.float32)
            emb = emb / (np.linalg.norm(emb) + 1e-12)
            eng.raw_log.append(
                session_id=sid,
                type="user_message",
                content=f"test message {i}",
                embedding=emb,
            )
        result = eng.session_end(sid, consolidate=True)
        # Multi-scale strategy: 9 two-turn micro episodes + 1 macro episode.
        assert result["episodes_formed"] == 10
        # Default schema mode is latent: schemas are formed without an LLM.
        assert eng.schemas.count() > 0
        assert eng.episodic.count() == 10
        eng.close()
    finally:
        for ext in ("", "-wal", "-shm"):
            p = tmp.name + ext
            if os.path.exists(p):
                os.remove(p)
