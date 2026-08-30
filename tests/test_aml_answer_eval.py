from __future__ import annotations

import json
from types import SimpleNamespace

from tests.benchmarks import aml_answer_eval
from tests.benchmarks.llm_judge import estimate_cost_usd


def test_gpt_4o_mini_pricing() -> None:
    assert estimate_cost_usd("openai/gpt-4o-mini", 1_000_000, 1_000_000) == 0.75


def test_render_answer_prompt_matches_aml_shape() -> None:
    prompt = aml_answer_eval.render_answer_prompt("Where?", "Alice moved to Sweden.")

    assert "Memories for user speaker 1:" in prompt
    assert "Alice moved to Sweden." in prompt
    assert "Question: Where?" in prompt
    assert "shortest correct phrase or sentence" in prompt


def test_render_accuracy_prompt_substitutes_values_without_touching_json() -> None:
    prompt = aml_answer_eval.render_accuracy_prompt("Where?", "Sweden", "Sweden")

    assert "Question: Where?" in prompt
    assert "Gold answer: Sweden" in prompt
    assert "Generated answer: Sweden" in prompt
    assert '"label": "CORRECT" or "WRONG"' in prompt


def test_parse_aml_label_accepts_fenced_or_plain_json() -> None:
    assert aml_answer_eval.parse_aml_label('{"label":"CORRECT"}') == "CORRECT"
    assert aml_answer_eval.parse_aml_label('```json\n{"label": "wrong"}\n```') == "WRONG"
    assert aml_answer_eval.parse_aml_label("CORRECT") is None


def test_eligible_rows_excludes_errors_and_locomo_adversarial() -> None:
    payload = {
        "results": [
            {"question": "q1", "expected": "a1", "hypothesis": "h1", "category": 1},
            {"question": "q2", "expected": "a2", "hypothesis": "h2", "category": 5},
            {"question": "q3", "expected": "a3", "hypothesis": "h3", "error": "boom"},
        ]
    }

    rows = aml_answer_eval.eligible_rows(payload)

    assert len(rows) == 1
    assert rows[0]["_source_index"] == 0


def test_eligible_rows_normalizes_longmemeval_expected_answer() -> None:
    rows = aml_answer_eval.eligible_rows(
        {
            "results": [
                {
                    "question": "q1",
                    "expected_answer": "a1",
                    "hypothesis": "h1",
                    "question_type": "knowledge-update",
                }
            ]
        }
    )

    assert rows[0]["expected"] == "a1"


def test_save_output_tracks_answer_and_judge_costs(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        aml_answer_eval,
        "estimate_cost_usd",
        lambda model, prompt, completion: {"answer": 1.25, "judge": 0.75}[model],
    )
    output = tmp_path / "aml.json"
    args = SimpleNamespace(answer_model="answer", judge_model="judge")
    results = [
        {
            "parse_ok": True,
            "is_correct": True,
            "answer_prompt_tokens": 10,
            "answer_completion_tokens": 2,
            "judge_prompt_tokens": 5,
            "judge_completion_tokens": 1,
        }
    ]

    aml_answer_eval.save_output(
        output,
        tmp_path / "source.json",
        args,
        {"meta": {"top_k": 20}},
        results,
        partial=False,
        started=0.0,
    )
    payload = json.loads(output.read_text())

    assert payload["summary"]["score_pct"] == 100.0
    assert payload["summary"]["answer_cost_usd"] == 1.25
    assert payload["summary"]["judge_cost_usd"] == 0.75
    assert payload["summary"]["total_cost_usd"] == 2.0
