from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

MODULE_PATH = (
    Path(__file__).resolve().parents[2] / "private" / "experiments" / "procedural_identity_mvp.py"
)
SPEC = importlib.util.spec_from_file_location("procedural_identity_mvp", MODULE_PATH)
assert SPEC and SPEC.loader
mvp = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = mvp
SPEC.loader.exec_module(mvp)


def sig(context, actions, outcomes):
    return mvp.ProcedureSignature.create(
        contexts=context,
        actions=[mvp.Action(*action) for action in actions],
        outcomes=outcomes,
    )


def test_rejects_unknown_action_operator():
    with pytest.raises(ValueError, match="unsupported operator"):
        mvp.Action("deploy_magic", "service_system")


def test_same_procedure_transfers_across_domains():
    # Same incident option, different verticals: reproduce -> inspect -> repair -> verify.
    aiops = sig(
        {"service_system", "runtime_failure", "observable_evidence"},
        [
            ("reproduce", "runtime_behavior"),
            ("inspect", "evidence"),
            ("repair", "root_cause"),
            ("test", "runtime_behavior"),
        ],
        {"defect_repaired", "behavior_verified"},
    )
    coding = sig(
        {"service_system", "runtime_failure", "observable_evidence"},
        [
            ("reproduce", "runtime_behavior"),
            ("inspect", "evidence"),
            ("repair", "root_cause"),
            ("test", "runtime_behavior"),
        ],
        {"defect_repaired", "behavior_verified"},
    )
    assert mvp.compatible(aiops, coding).compatible


def test_similar_topic_does_not_override_different_policy():
    validate_config = sig(
        {"configuration", "data_contract"},
        [("parse", "configuration"), ("validate", "contract"), ("report", "artifacts")],
        {"differences_reported"},
    )
    repair_config = sig(
        {"configuration", "runtime_failure"},
        [
            ("reproduce", "runtime_behavior"),
            ("inspect", "configuration"),
            ("repair", "root_cause"),
            ("test", "runtime_behavior"),
        ],
        {"defect_repaired", "behavior_verified"},
    )
    result = mvp.compatible(validate_config, repair_config)
    assert not result.compatible
    assert "actions" in result.reason and "outcomes" in result.reason


def test_complete_link_prevents_chaining():
    a = sig(
        {"configuration", "runtime_failure"},
        [("inspect", "configuration"), ("repair", "configuration")],
        {"defect_repaired", "behavior_verified"},
    )
    b = sig(
        {"configuration"},
        [("inspect", "configuration"), ("repair", "configuration")],
        {"behavior_verified"},
    )
    c = sig(
        {"configuration", "contract_mismatch"},
        [("inspect", "configuration"), ("repair", "configuration")],
        {"behavior_verified", "contract_satisfied"},
    )
    attempts = [mvp.SignedAttempt(name, value) for name, value in (("a", a), ("b", b), ("c", c))]
    assert mvp.form_families(attempts) == [["a", "b"]]


def test_applicability_abstains_on_hard_negative():
    repair = sig(
        {"service_system", "runtime_failure", "observable_evidence"},
        [
            ("reproduce", "runtime_behavior"),
            ("inspect", "evidence"),
            ("repair", "root_cause"),
            ("test", "runtime_behavior"),
        ],
        {"defect_repaired", "behavior_verified"},
    )
    family = [[mvp.SignedAttempt("one", repair), mvp.SignedAttempt("two", repair)]]
    reconcile = sig(
        {"structured_records", "inconsistent_sources", "data_contract"},
        [("parse", "records"), ("compare", "records"), ("report", "artifacts")],
        {"differences_reported", "data_preserved"},
    )
    assert mvp.applicable_families(reconcile, family) == []


def test_consolidated_core_generalizes_without_topic_matching():
    first = sig(
        {"configuration", "service_system", "contract_mismatch"},
        [
            ("parse", "configuration"),
            ("inspect", "dependency_graph"),
            ("repair", "configuration"),
            ("validate", "configuration"),
        ],
        {"defect_repaired", "service_stable", "artifacts_valid"},
    )
    second = sig(
        {"configuration", "service_system", "runtime_failure"},
        [
            ("inspect", "dependency_graph"),
            ("compare", "configuration"),
            ("repair", "configuration"),
            ("validate", "configuration"),
            ("report", "root_cause"),
        ],
        {"defect_repaired", "service_stable", "differences_reported"},
    )
    family = mvp.consolidate_family(
        [mvp.SignedAttempt("one", first), mvp.SignedAttempt("two", second)]
    )
    adjacent = sig(
        {"configuration", "service_system", "runtime_failure", "observable_evidence"},
        [
            ("inspect", "dependency_graph"),
            ("compare", "evidence"),
            ("repair", "configuration"),
            ("validate", "configuration"),
            ("report", "root_cause"),
        ],
        {"defect_repaired", "service_stable", "artifacts_valid"},
    )
    assert mvp.applicable_prototypes(adjacent, [family])
