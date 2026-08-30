"""Regression tests for IngestService.form_episodes()'s macro-episode branch.

Root cause (2026-07-15): macro_emb was computed from `embeddable` events only
(those with a non-null embedding), and event_ids was built from `embeddable`
too, but macro_text/macro_source were built from `events` -- the unfiltered
list. A non-embeddable event's text (e.g. slowave_commit's task_complete
"outcome=X" marker, always logged with disable_encoder=True) still leaked
into macro_text/macro_source even though it contributed nothing to macro_emb.
When a session had exactly one embeddable event, macro_emb ended up being
that single event's embedding, unchanged -- while macro_text carried an
unrelated non-embeddable event's content on top of it. This produced
persisted episodes (and the schemas consolidated from them) whose text
described more than their embedding represented, confirmed in production as
27 schema pairs sharing byte-identical embeddings despite different content.
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

import numpy as np
import pytest

from slowave.core.services.ingest import IngestService
from slowave.latent.episodic_store import EpisodicStore, EpisodicStoreConfig
from slowave.latent.salience import SalienceConfig, SalienceEngine
from slowave.latent.transition_model import TransitionModel, TransitionModelConfig
from slowave.storage.sqlite_db import SQLiteConfig, SQLiteDB
from slowave.symbolic.episode_text import EpisodeTextStore
from slowave.symbolic.raw_log import RawLog

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SCHEMA_PATH = str(REPO_ROOT / "slowave" / "storage" / "schema.sql")
DIM = 8


@pytest.fixture()
def ingest():
    db_path = str(Path(tempfile.mkdtemp()) / "test.db")
    db = SQLiteDB(SQLiteConfig(path=db_path))
    db.init_schema(SCHEMA_PATH)
    conn = db.connect()
    conn.execute(
        "INSERT INTO sessions (id, agent, started_ts) VALUES (?, ?, ?)",
        ("sess-1", "test", int(time.time())),
    )
    conn.commit()

    raw_log = RawLog(db)
    episodic = EpisodicStore(db, EpisodicStoreConfig(dim=DIM, db_path=db_path))
    episode_text = EpisodeTextStore(db)
    salience = SalienceEngine(SalienceConfig())
    transition_model = TransitionModel(TransitionModelConfig(dim=DIM))
    svc = IngestService(
        raw_log=raw_log,
        episodic=episodic,
        episode_text=episode_text,
        salience=salience,
        transition_model=transition_model,
        db=db,
    )
    yield svc, raw_log, episode_text, episodic
    db.close()


def _unit(rng: np.random.Generator) -> np.ndarray:
    v = rng.standard_normal(DIM).astype(np.float32)
    return v / np.linalg.norm(v)


def _macro_episode_id(episodic: EpisodicStore, episode_ids: list[int]) -> int:
    for eid in episode_ids:
        if episodic.get(eid).metadata.get("kind") == "macro":
            return eid
    raise AssertionError("no macro episode found among formed episodes")


def _macro_episode_text(
    episodic: EpisodicStore, episode_text: EpisodeTextStore, episode_ids: list[int]
) -> tuple[str, str]:
    macro_id = _macro_episode_id(episodic, episode_ids)
    ep = next(e for e in episode_text.get_many(episode_ids) if e.episode_id == macro_id)
    return ep.content_text, ep.source_content


def test_non_embeddable_event_text_excluded_from_macro_episode(ingest):
    """The bug itself: a single embeddable event plus a non-embeddable
    task_complete marker must NOT leak the marker's text into macro_text,
    and macro_emb must still just be that one embeddable event's vector."""
    svc, raw_log, episode_text, episodic = ingest
    rng = np.random.default_rng(0)
    emb = _unit(rng)

    raw_log.append(
        session_id="sess-1", type="remember:fact", content="the important fact", embedding=emb
    )
    raw_log.append(
        session_id="sess-1", type="task_complete", content="outcome=success", embedding=None
    )

    episode_ids = svc.form_episodes("sess-1")
    assert episode_ids  # at least the macro episode was formed

    content_text, source_content = _macro_episode_text(episodic, episode_text, episode_ids)
    assert "important fact" in content_text
    assert "outcome=success" not in content_text
    assert "outcome=success" not in source_content


def test_macro_embedding_matches_embeddable_events_only(ingest):
    """macro_emb must be the mean of embeddable events' vectors regardless of
    how many non-embeddable events are interleaved -- confirms the embedding
    side was already correct and stays correct after the text-side fix."""
    svc, raw_log, episode_text, episodic = ingest
    rng = np.random.default_rng(1)
    emb = _unit(rng)

    raw_log.append(
        session_id="sess-1", type="remember:fact", content="the only real fact", embedding=emb
    )
    raw_log.append(
        session_id="sess-1", type="context_query", content="some cue text", embedding=None
    )
    raw_log.append(
        session_id="sess-1", type="task_complete", content="outcome=success", embedding=None
    )

    episode_ids = svc.form_episodes("sess-1")
    macro_id = _macro_episode_id(episodic, episode_ids)
    macro_ep_row = episodic.get(macro_id)
    np.testing.assert_allclose(macro_ep_row.embedding, emb, atol=1e-6)


def test_multiple_embeddable_events_still_all_included_in_macro_text(ingest):
    """The fix must not accidentally exclude embeddable events -- only
    non-embeddable ones. With two embeddable events, both texts must appear."""
    svc, raw_log, episode_text, episodic = ingest
    rng = np.random.default_rng(2)
    emb1, emb2 = _unit(rng), _unit(rng)

    raw_log.append(session_id="sess-1", type="remember:fact", content="fact one", embedding=emb1)
    raw_log.append(session_id="sess-1", type="remember:fact", content="fact two", embedding=emb2)
    raw_log.append(
        session_id="sess-1", type="task_complete", content="outcome=success", embedding=None
    )

    episode_ids = svc.form_episodes("sess-1")
    content_text, _source = _macro_episode_text(episodic, episode_text, episode_ids)
    assert "fact one" in content_text
    assert "fact two" in content_text
    assert "outcome=success" not in content_text


def test_control_role_is_excluded_even_if_an_embedding_was_supplied(ingest):
    """Role eligibility is a defense-in-depth boundary, independent of content."""
    svc, raw_log, episode_text, episodic = ingest
    rng = np.random.default_rng(3)
    experience_emb, control_emb = _unit(rng), _unit(rng)
    raw_log.append(
        session_id="sess-1",
        type="observation",
        content="a meaningful observation",
        embedding=experience_emb,
        metadata={"memory_role": "experience"},
    )
    raw_log.append(
        session_id="sess-1",
        type="arbitrary_control_type",
        content="generic repeated transport marker",
        embedding=control_emb,
        metadata={"memory_role": "control"},
    )

    episode_ids = svc.form_episodes("sess-1")
    content_text, source_content = _macro_episode_text(episodic, episode_text, episode_ids)
    assert "meaningful observation" in content_text
    assert "generic repeated transport marker" not in content_text
    assert "generic repeated transport marker" not in source_content


def test_recall_cue_does_not_form_episodes_despite_experience_role(ingest):
    """The MCP recall log ``slowave_recall: <query>`` is a cue, not a memory —
    it must not form episodes even if a (legacy) event stored it as
    ``experience`` with an embedding. This is what a logic-version rebuild
    relies on to decontaminate historical records."""
    svc, raw_log, episode_text, episodic = ingest
    rng = np.random.default_rng(4)
    raw_log.append(
        session_id="sess-1",
        type="trajectory:action",
        content="slowave_recall: Explicit user preferences for an AI assistant",
        embedding=_unit(rng),
        metadata={"memory_role": "experience"},
    )
    # A genuine task event must still form episodes alongside the excluded cue.
    raw_log.append(
        session_id="sess-1",
        type="trajectory:action",
        content="Implemented the retry backoff",
        embedding=_unit(rng),
        metadata={"memory_role": "experience"},
    )
    episode_ids = svc.form_episodes("sess-1")
    assert episode_ids
    content_text, source_content = _macro_episode_text(episodic, episode_text, episode_ids)
    assert "retry backoff" in content_text
    assert "slowave_recall" not in content_text
    assert "slowave_recall" not in source_content


def test_lifecycle_only_trajectory_forms_no_episodes(ingest):
    """A trajectory that narrates only Slowave lifecycle steps (the sch_624
    pattern) forms NO episodes; the session is effectively empty of memory."""
    svc, raw_log, _episode_text, episodic = ingest
    rng = np.random.default_rng(5)
    for summary in (
        "Activated the Slowave session.",
        "Submitted complete retrieval feedback.",
        "Committed the session.",
    ):
        raw_log.append(
            session_id="sess-1",
            type="trajectory:action",
            content=summary,
            embedding=_unit(rng),
            metadata={"memory_role": "experience"},
        )
    episode_ids = svc.form_episodes("sess-1")
    assert not episode_ids


def test_legacy_embedded_task_complete_does_not_form_episodes(ingest):
    """A legacy task_complete event (stored embedding, no memory_role -> defaults
    to experience) must not form episodes: the type is Slowave's own commit
    marker and is excluded regardless of role/embedding. This is what a
    logic-version rebuild relies on to drop historical ``outcome=...`` schemas."""
    svc, raw_log, episode_text, episodic = ingest
    rng = np.random.default_rng(6)
    raw_log.append(
        session_id="sess-1",
        type="task_complete",
        content="outcome=success",
        embedding=_unit(rng),  # legacy: a stored embedding with no memory_role
    )
    # A genuine task event alongside must still form episodes.
    raw_log.append(
        session_id="sess-1",
        type="remember:fact",
        content="the real fact",
        embedding=_unit(rng),
    )
    episode_ids = svc.form_episodes("sess-1")
    assert episode_ids
    content_text, source_content = _macro_episode_text(episodic, episode_text, episode_ids)
    assert "real fact" in content_text
    assert "outcome=success" not in content_text
    assert "outcome=success" not in source_content
