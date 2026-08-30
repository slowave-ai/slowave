from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

_SCRIPT = Path(__file__).parents[2] / "private/experiments/grid_search_procedural_discovery.py"
_SPEC = importlib.util.spec_from_file_location("grid_search_procedural_discovery", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

_clusters_at = _MODULE._clusters_at
_greedy = _MODULE._greedy
_hierarchy = _MODULE._hierarchy
_monotonic = _MODULE._monotonic


def test_monotonic_alignment_penalizes_reversed_action_order() -> None:
    first = np.asarray([1.0, 0.0], dtype=np.float32)
    second = np.asarray([0.0, 1.0], dtype=np.float32)

    assert _greedy([first, second], [second, first]) == 1.0
    assert _monotonic([first, second], [second, first]) == 0.5


def test_complete_linkage_rejects_a_weak_cluster_member() -> None:
    ids = ["a", "b", "c"]
    scores = {("a", "b"): 0.9, ("a", "c"): 0.6, ("b", "c"): 0.4}

    average_groups = _clusters_at(ids, _hierarchy(scores, ids, "average"), 0.45)
    complete_groups = _clusters_at(ids, _hierarchy(scores, ids, "complete"), 0.45)

    assert {frozenset(group) for group in average_groups} == {frozenset(ids)}
    assert {frozenset(group) for group in complete_groups} == {
        frozenset({"a", "b"}),
        frozenset({"c"}),
    }
