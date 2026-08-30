from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

MODULE = Path(__file__).parents[2] / "private" / "experiments" / "after_procedural_contract_gate.py"
SPEC = importlib.util.spec_from_file_location("after_procedural_contract_gate", MODULE)
assert SPEC and SPEC.loader
gate = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = gate
SPEC.loader.exec_module(gate)


def test_offline_contract_gate_passes_without_model_calls() -> None:
    result = gate.run_gate()
    assert result["model_calls"] == 0
    assert result["formation"] == {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    assert result["emitter_partition_agreement"] is True
    assert result["retrieval"]["hard_negative_false_positives"] == 0
    assert result["retrieval"]["correct_hits"] == result["retrieval"]["expected_hits"] == 1
    assert result["authorize_transfer_trials"] is False
    assert "need at least 2 supported families" in result["transfer_blocker"]
