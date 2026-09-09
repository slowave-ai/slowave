from __future__ import annotations

from slowave.core.config import SlowaveConfig
from slowave.core.engine import SlowaveEngine
from slowave.mcp.tools import (
    _MCP_CONTINUATION_RESPONSE_CHARS,
    _accessible_field,
    _attach_frozen_tail,
    _continuation_page,
    _freeze_continuation,
    _serialized_chars,
)


def _engine(tmp_path) -> SlowaveEngine:
    return SlowaveEngine(
        SlowaveConfig(db_path=str(tmp_path / "continuation.db"), dim=8, disable_encoder=True)
    )


def _seed_retrieval(eng: SlowaveEngine, retrieval_id: str, session_id: str, scope: str) -> None:
    eng.record_retrieval(
        retrieval_id=retrieval_id,
        retrieval_type="recall",
        session_id=session_id,
        query="frozen candidates",
        scope_id=scope,
        response={"schemas": []},
    )


def test_frozen_cursor_is_repeatable_nonduplicating_and_reaches_end(tmp_path) -> None:
    eng = _engine(tmp_path)
    try:
        scope = "project:test"
        session_id = eng.session_start(agent="test", scope=scope, goal="test frozen continuation")
        retrieval_id = "rec_frozen"
        _seed_retrieval(eng, retrieval_id, session_id, scope)
        candidates = [
            {
                "kind": "memory",
                "value": {
                    "memory_id": f"sch_{index}",
                    "content": f"candidate {index}",
                    "pathway": "direct",
                },
            }
            for index in range(1, 6)
        ]
        cursor = _freeze_continuation(
            eng,
            retrieval_id=retrieval_id,
            session_id=session_id,
            scope=scope,
            candidates=candidates,
            offset=0,
        )
        assert cursor

        first = _continuation_page(eng, cursor=cursor, session_id=session_id, scope=scope)
        repeated = _continuation_page(eng, cursor=cursor, session_id=session_id, scope=scope)
        assert first == repeated
        assert [item["memory_id"] for item in first["memories"]] == ["sch_1", "sch_2"]

        seen = [item["memory_id"] for item in first["memories"]]
        page = first
        while page["more_available"]:
            page = _continuation_page(
                eng,
                cursor=page["continue_from"],
                session_id=session_id,
                scope=scope,
            )
            seen.extend(item["memory_id"] for item in page["memories"])
            assert (
                _serialized_chars(
                    {key: value for key, value in page.items() if key != "accessible_field"}
                )
                <= _MCP_CONTINUATION_RESPONSE_CHARS
            )
        assert seen == ["sch_1", "sch_2", "sch_3", "sch_4", "sch_5"]
        assert len(seen) == len(set(seen))
        assert page["accessible_field"] == {}
    finally:
        eng.close()


def test_accessible_field_contains_only_counts_and_kinds() -> None:
    candidates = [
        {
            "kind": "memory",
            "value": {
                "memory_id": "sch_1",
                "content": "secret candidate content",
                "pathway": "associated",
            },
        },
        {
            "kind": "procedure",
            "value": {"procedure_id": "proc_1", "summary": "secret procedure"},
        },
    ]
    field = _accessible_field(candidates)
    assert field["extra_candidates"] == 2
    assert field["kinds"] == ["associated", "procedure"]
    assert field["approx_extra_tokens"] > 0
    assert "secret" not in str(field)
    assert _accessible_field([]) == {}


def test_oversized_candidate_never_advertises_an_unusable_cursor(tmp_path) -> None:
    eng = _engine(tmp_path)
    try:
        scope = "project:test"
        session_id = eng.session_start(
            agent="test", scope=scope, goal="test oversized continuation"
        )
        retrieval_id = "rec_oversized"
        _seed_retrieval(eng, retrieval_id, session_id, scope)
        data = {"retrieval_id": retrieval_id, "memories": [], "procedures": []}
        candidates = [
            {
                "kind": "procedure",
                "value": {
                    "procedure_id": "proc_large",
                    "summary": "x" * (_MCP_CONTINUATION_RESPONSE_CHARS * 2),
                },
            }
        ]
        _attach_frozen_tail(
            eng,
            data=data,
            candidates=candidates,
            session_id=session_id,
            scope=scope,
            exposed_ids=set(),
        )
        assert data["more_available"] is False
        assert "continue_from" not in data
        assert data["accessible_field"] == {}
    finally:
        eng.close()
