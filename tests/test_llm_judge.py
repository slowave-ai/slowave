from __future__ import annotations

from tests.benchmarks.llm_judge import parse_judge_response


def test_parse_plain_json() -> None:
    assert parse_judge_response('{"score": 1.0, "reason": "present"}') == (1.0, "present")


def test_parse_code_fenced_json() -> None:
    raw = '```json\n{"score": 0.0, "reason": "absent"}\n```'
    assert parse_judge_response(raw) == (0.0, "absent")


def test_parse_score_clamped_to_unit_interval() -> None:
    assert parse_judge_response('{"score": 7, "reason": "x"}')[0] == 1.0


def test_parse_prose_with_stray_digit_is_not_a_score() -> None:
    # A judge that emits prose containing a digit must NOT have that digit
    # silently treated as a score (regression guard for the removed
    # "find any number in [0,1]" fallback).
    assert parse_judge_response("I found 0 of 3 facts, so it is wrong") is None


def test_parse_empty_is_none() -> None:
    assert parse_judge_response("") is None
