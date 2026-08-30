import numpy as np

from tests.benchmarks.turn_level_retrieval import _locomo_cases, depth_metrics, rank_turns


def test_rank_turns_uses_full_query_cosine_order() -> None:
    query = np.array([1.0, 0.0], dtype=np.float32)
    turns = np.array([[0.2, 0.98], [1.0, 0.0], [0.8, 0.6]], dtype=np.float32)

    assert rank_turns(query, turns).tolist() == [1, 2, 0]


def test_depth_metrics_distinguishes_any_and_complete_recovery() -> None:
    metrics = depth_metrics(["noise", "a", "noise2", "b"], {"a", "b"}, (2, 4))

    assert metrics["2"]["any_gold"] is True
    assert metrics["2"]["all_gold"] is False
    assert metrics["2"]["gold_recall"] == 0.5
    assert metrics["4"]["all_gold"] is True
    assert metrics["first_gold_rank"] == 2


def test_locomo_combined_evidence_references_are_split() -> None:
    dataset = [
        {
            "sample_id": "conv-1",
            "conversation": {
                "session_1": [
                    {"dia_id": "D8:6", "text": "first"},
                    {"dia_id": "D9:17", "text": "second"},
                ]
            },
            "qa": [{"question": "Q?", "evidence": ["D8:6; D9:17"]}],
        }
    ]

    case = next(_locomo_cases(dataset, {("conv-1", "Q?")}))

    assert case[-1] == {"D8:6", "D9:17"}
