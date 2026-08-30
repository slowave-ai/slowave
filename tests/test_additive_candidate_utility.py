from tests.benchmarks.analyze_additive_candidate_utility import _auc, candidate_parts


def test_candidate_parts_recovers_appended_and_evicted_units() -> None:
    baseline = "[2024-01-01] retained\n\n[2024-01-02] evicted"
    additive = "[2024-01-01] retained\n\nnew candidate"

    candidate, retained, evicted = candidate_parts(baseline, additive)

    assert candidate == "new candidate"
    assert retained == ["[2024-01-01] retained"]
    assert evicted == ["[2024-01-02] evicted"]


def test_auc_handles_order_and_ties() -> None:
    assert _auc([1.0, 2.0, 0.0, 1.0], [True, True, False, False]) == 0.875
