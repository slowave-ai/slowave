import json

import numpy as np
import pytest

from slowave import ops
from slowave.core.config import SlowaveConfig
from slowave.core.engine import SlowaveEngine


@pytest.fixture()
def eng(tmp_path):
    engine = SlowaveEngine(SlowaveConfig(db_path=str(tmp_path / "test.db"), disable_encoder=True))
    yield engine
    engine.close()


def test_commit_persists_bounded_trajectory_with_server_owned_provenance(eng):
    class StubEncoder:
        def encode(self, text):
            return np.ones(384, dtype=np.float32)

    eng.encoder = StubEncoder()
    session_id = eng.session_start(agent="mcp:test-client", scope="project:test")

    result = ops.commit(
        eng,
        session_id=session_id,
        outcome="success",
        final_goal="verify trajectory capture",
        outcome_summary="trajectory was stored",
        verification={"status": "verified", "summary": "rows inspected"},
        trajectory=[
            {"kind": "action", "summary": "Ran the focused test", "status": "succeeded"},
            {"kind": "observation", "summary": "The test passed"},
        ],
        provenance={"source_kind": "integration", "integration": "test-client"},
    )
    assert result["episodes_formed"] > 0

    rows = (
        eng.db.connect()
        .execute(
            "SELECT type, content, metadata_json FROM raw_events "
            "WHERE session_id = ? AND type LIKE 'trajectory:%' ORDER BY id",
            (session_id,),
        )
        .fetchall()
    )
    assert [(row["type"], row["content"]) for row in rows] == [
        ("trajectory:action", "Ran the focused test"),
        ("trajectory:observation", "The test passed"),
    ]
    metadata = json.loads(rows[0]["metadata_json"])
    assert metadata.pop("memory_role") == "experience"
    assert metadata["provenance"] == {
        "source_kind": "agent_inference",
        "integration": "test-client",
        "observed": False,
    }

    completion = (
        eng.db.connect()
        .execute(
            "SELECT content, metadata_json, embedding, dim FROM raw_events "
            "WHERE session_id = ? AND type = 'task_complete'",
            (session_id,),
        )
        .fetchone()
    )
    assert completion["content"] == "outcome=success"
    assert json.loads(completion["metadata_json"])["memory_role"] == "procedural_evidence"
    assert completion["embedding"] is None
    assert completion["dim"] is None

    episode_texts = [
        row["content_text"]
        for row in eng.db.connect()
        .execute(
            "SELECT et.content_text FROM episode_text et WHERE et.session_id = ?",
            (session_id,),
        )
        .fetchall()
    ]
    assert episode_texts
    assert all("outcome=success" not in text for text in episode_texts)


def test_trajectory_rejects_unbounded_or_source_claiming_entries_before_writes(eng):
    session_id = eng.session_start(agent="mcp:test-client", scope="project:test")

    with pytest.raises(ValueError, match="at most 32"):
        ops.commit(
            eng,
            session_id=session_id,
            trajectory=[{"kind": "action", "summary": str(i)} for i in range(33)],
        )
    with pytest.raises(ValueError, match="only kind, summary, and status"):
        ops.commit(
            eng,
            session_id=session_id,
            trajectory=[
                {
                    "kind": "observation",
                    "summary": "Claimed external observation",
                    "source_kind": "human",
                }
            ],
        )

    count = (
        eng.db.connect()
        .execute("SELECT COUNT(*) AS count FROM raw_events WHERE session_id = ?", (session_id,))
        .fetchone()["count"]
    )
    assert count == 0


def test_commit_update_clears_a_legacy_completion_embedding(eng):
    class StubEncoder:
        def encode(self, text):
            return np.ones(384, dtype=np.float32)

    eng.encoder = StubEncoder()
    session_id = eng.session_start(agent="mcp:test-client", scope="project:test")
    event_id = eng.event_append(
        session_id=session_id,
        type="task_complete",
        content="legacy completion transport text",
    )
    assert eng.raw_log.get(event_id).embedding is not None

    ops.commit(
        eng,
        session_id=session_id,
        outcome="partial",
        final_goal="normalize completion evidence",
        outcome_summary="The completion record was normalized.",
        verification={"status": "verified", "summary": "row inspected"},
    )

    updated = eng.raw_log.get(event_id)
    assert updated.content == "outcome=partial"
    assert updated.embedding is None
    assert updated.metadata["memory_role"] == "procedural_evidence"


def test_remember_carries_integration_provenance_into_source_event(eng):
    session_id = eng.session_start(agent="mcp:test-client", scope="project:test")
    result = ops.remember(
        eng,
        content="A sourced fact",
        memory_type="fact",
        scope="project:test",
        session_id=session_id,
        provenance={"source_kind": "integration", "integration": "test-client"},
    )

    row = (
        eng.db.connect()
        .execute(
            "SELECT metadata_json FROM raw_events WHERE id = ?",
            (int(result["source_event_id"].removeprefix("evt_")),),
        )
        .fetchone()
    )
    assert json.loads(row["metadata_json"])["provenance"] == {
        "source_kind": "integration",
        "integration": "test-client",
    }
    schema_id = int(result["memory_id"].removeprefix("sch_"))
    assert eng.schemas.get(schema_id).facets["source_provenance"] == {
        "source_kind": "integration",
        "integration": "test-client",
    }


def test_commit_drops_lifecycle_trajectory_entries_and_reports_count(eng):
    """Lifecycle narration ("Activated the Slowave session.") must be dropped
    from the trajectory so it is never stored as episodic experience; a
    genuine task entry stays. The drop is surfaced to the client."""

    class StubEncoder:
        def encode(self, text):
            return np.ones(384, dtype=np.float32)

    eng.encoder = StubEncoder()
    session_id = eng.session_start(agent="mcp:test-client", scope="project:test")

    result = ops.commit(
        eng,
        session_id=session_id,
        outcome="success",
        final_goal="verify lifecycle filtering",
        outcome_summary="lifecycle entries dropped",
        verification={"status": "verified", "summary": "rows inspected"},
        trajectory=[
            {"kind": "action", "summary": "Activated the Slowave session.", "status": "succeeded"},
            {"kind": "action", "summary": "Added conditional Labs UI", "status": "succeeded"},
            {"kind": "observation", "summary": "The live DB showed zero used marks"},
        ],
    )
    assert result["trajectory_lifecycle_filtered"] == 1
    assert result["episodes_formed"] > 0

    rows = (
        eng.db.connect()
        .execute(
            "SELECT type, content FROM raw_events "
            "WHERE session_id = ? AND type LIKE 'trajectory:%' ORDER BY id",
            (session_id,),
        )
        .fetchall()
    )
    assert [(row["type"], row["content"]) for row in rows] == [
        ("trajectory:action", "Added conditional Labs UI"),
        ("trajectory:observation", "The live DB showed zero used marks"),
    ]


def test_commit_lifecycle_only_trajectory_forms_no_episodes(eng):
    """If the only trajectory content is lifecycle bookkeeping, no episodic
    memory is formed from it — equivalent to the client having omitted it."""
    session_id = eng.session_start(agent="mcp:test-client", scope="project:test")
    result = ops.commit(
        eng,
        session_id=session_id,
        outcome="failure",
        final_goal="does not matter",
        outcome_summary="no task work recorded",
        verification={"status": "unverified", "summary": "n/a"},
        trajectory=[
            {"kind": "action", "summary": "Activated the Slowave session."},
            {"kind": "action", "summary": "Committed the session."},
        ],
    )
    assert result["trajectory_lifecycle_filtered"] == 2
    traj_rows = (
        eng.db.connect()
        .execute(
            "SELECT COUNT(*) AS n FROM raw_events "
            "WHERE session_id = ? AND type LIKE 'trajectory:%'",
            (session_id,),
        )
        .fetchone()
    )
    assert traj_rows["n"] == 0
