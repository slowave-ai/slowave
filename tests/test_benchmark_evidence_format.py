from __future__ import annotations

from types import SimpleNamespace

from tests.benchmarks.evidence_format import format_retrieved_evidence
from tests.benchmarks.longmemeval_eval import _parse_longmemeval_ts


def test_parse_longmemeval_timestamp_preserves_date_and_time() -> None:
    ts = _parse_longmemeval_ts("2023/03/15 (Wed) 10:31")

    from datetime import datetime, timezone

    parsed = datetime.fromtimestamp(ts, tz=timezone.utc)
    assert (parsed.year, parsed.month, parsed.day, parsed.hour, parsed.minute) == (
        2023,
        3,
        15,
        10,
        31,
    )


def test_format_retrieved_evidence_preserves_boundaries_dates_and_scores() -> None:
    schema = SimpleNamespace(
        id=7,
        content_text="The preferred editor is Zed.",
        first_formed_ts=1_700_000_000,
        last_updated_ts=1_700_003_600,
    )
    result = SimpleNamespace(
        schemas=[schema],
        schema_rank_scores={7: 0.875},
        episode_texts=[
            {
                "id": 3,
                "ts": 1_700_007_200,
                "content_text": "[2023-11-15] User: I switched to Zed.",
            }
        ],
    )

    rendered = format_retrieved_evidence(result)

    assert "=== SCHEMAS" in rendered
    assert "[SCHEMA 1" in rendered
    assert "source_date=unknown" in rendered
    assert "formed=" not in rendered
    assert "score=0.8750" in rendered
    assert "=== EPISODES" in rendered
    assert "[EPISODE 1" in rendered
    assert "I switched to Zed" in rendered
    assert "date=2023-11-15 00:13Z" in rendered


def test_format_retrieved_evidence_can_emit_episode_only() -> None:
    result = SimpleNamespace(
        schemas=[
            SimpleNamespace(
                id=7,
                content_text="schema text",
                first_formed_ts=1,
                last_updated_ts=1,
            )
        ],
        schema_rank_scores={7: 0.5},
        episode_texts=[{"ts": 1_700_007_200, "content_text": "episode text"}],
    )

    rendered = format_retrieved_evidence(result, include_schemas=False)

    assert "SCHEMAS" not in rendered
    assert "episode text" in rendered


def test_format_retrieved_evidence_can_put_episodes_first() -> None:
    result = SimpleNamespace(
        schemas=[
            SimpleNamespace(
                id=7,
                content_text="schema text",
                first_formed_ts=1,
                last_updated_ts=1,
            )
        ],
        schema_rank_scores={7: 0.5},
        episode_texts=[{"ts": 1_700_007_200, "content_text": "episode text"}],
    )

    rendered = format_retrieved_evidence(result, episodes_first=True)

    assert rendered.index("EPISODES") < rendered.index("SCHEMAS")
