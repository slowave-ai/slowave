from tests.benchmarks.longmemeval_oracle_export import (
    build_payload,
    format_oracle_context,
)


def _question():
    return {
        "question_id": "q1",
        "question_type": "temporal-reasoning",
        "question": "Which came first?",
        "answer": "alpha",
        "answer_session_ids": ["s2"],
        "haystack_session_ids": ["s1", "s2"],
        "haystack_dates": ["2023/01/02 (Mon) 10:00", "2023/01/01 (Sun) 09:00"],
        "haystack_sessions": [
            [{"role": "user", "content": "noise"}],
            [
                {"role": "user", "content": "alpha happened"},
                {"role": "assistant", "content": "noted"},
            ],
        ],
    }


def test_format_oracle_context_keeps_only_complete_answer_sessions():
    context = format_oracle_context(_question())
    assert "SESSION s2" in context
    assert "2023/01/01 (Sun) 09:00" in context
    assert "USER: alpha happened" in context
    assert "ASSISTANT: noted" in context
    assert "noise" not in context


def test_build_payload_is_aml_compatible(tmp_path):
    payload = build_payload([_question()], tmp_path / "dataset.json", "temporal-reasoning")
    assert payload["meta"]["evidence_format"] == "structured_v1"
    assert payload["meta"]["oracle_context"] is True
    assert payload["results"][0]["expected_answer"] == "alpha"
