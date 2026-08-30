from tests.benchmarks.additive_turn_retrieval import (
    fixed_budget_addition,
    split_evidence_units,
)
from tests.benchmarks.export_additive_aml_arms import arm_payload


def test_fixed_budget_addition_repairs_missing_turn_without_growth() -> None:
    baseline = "[2024-01-01] first gold\n\n[2024-01-02] low ranked noise"
    evidence, metrics = fixed_budget_addition(
        baseline,
        [("gold2", "second gold"), ("noise", "other noise")],
        additions=1,
    )

    assert "first gold" in evidence
    assert "second gold" in evidence
    assert "low ranked noise" not in evidence
    assert len(evidence) <= len(baseline)
    assert metrics["added_turns"] == 1
    assert metrics["evicted_units"] == 1


def test_existing_turn_is_not_added_twice() -> None:
    baseline = "[2024-01-01] already present\n\n[2024-01-02] removable padding"

    evidence, metrics = fixed_budget_addition(
        baseline,
        [("existing", "already present"), ("new", "new")],
        additions=1,
    )

    assert evidence.count("already present") == 1
    assert "new" in evidence
    assert metrics["added_turns"] == 1


def test_split_preserves_unstructured_prefix() -> None:
    units = split_evidence_units("stable context\n[2024-01-01] event")

    assert units == ["stable context", "[2024-01-01] event"]


def test_arm_payload_exports_paired_aml_rows() -> None:
    source = {
        "meta": {
            "format": "additive_turn_retrieval_fixed_budget_v1",
            "split": "development",
            "query_method": "single full-query multilingual embedding",
            "captured_hypotheses": True,
        },
        "rows": [
            {
                "id": "long-id",
                "dataset": "longmemeval",
                "question": "When?",
                "expected": "Tuesday",
                "category": None,
                "baseline": {"hypothesis": "baseline evidence"},
                "arms": {
                    "1": {
                        "hypothesis": "arm evidence",
                        "first_candidate_query_cosine": 0.6,
                    }
                },
            }
        ],
    }

    baseline = arm_payload(source, "baseline")
    arm = arm_payload(source, "1")

    assert baseline["results"][0]["hypothesis"] == "baseline evidence"
    assert arm["results"][0]["hypothesis"] == "arm evidence"
    assert arm["results"][0]["question_id"] == "long-id"
    assert arm["meta"]["evidence_format"] == "structured_v1"
    assert arm_payload(source, "gated-1")["results"][0]["hypothesis"] == "arm evidence"
