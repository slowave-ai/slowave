"""Acceptance test configuration.

The end-to-end tests are stateful: each phase builds on the DB state left
by the previous one and MUST run in definition order.

Run with:
    pytest tests/acceptance/ -v -p no:randomly
or simply:
    pytest tests/acceptance/test_e2e.py -v
"""


def pytest_collection_modifyitems(config, items):
    """Re-sort acceptance tests back to definition order after any randomisation."""
    acceptance = [i for i in items if i.fspath.dirpath().basename == "acceptance"]
    if len(acceptance) < 2:
        return
    acceptance.sort(key=lambda i: (str(i.fspath), i.function.__code__.co_firstlineno))
    non_acceptance = [i for i in items if i.fspath.dirpath().basename != "acceptance"]
    items[:] = non_acceptance + acceptance


_PHASES: dict[str, str] = {
    "test_phase0_register_scopes": "Phase  0 — Register 10 scopes",
    "test_phase1_inject_dataset": "Phase  1 — Ingest dataset + cross-scope remember L1",
    "test_phase2_context_ranking": "Phase  2 — Context ranking (P@1 = 3/3)",
    "test_phase3_recall": "Phase  3 — Semantic recall + cross-scope isolation",
    "test_phase4_demotion": "Phase  4 — Noise demotion (S1 → needs_review) + recovery via feedback",
    "test_phase4b_scope_independent_noise_tracking": "Phase 4b — Noise demotion with no scope_id (D2)",
    "test_phase5_consolidation_hygiene": "Phase  5 — Consolidation hygiene + reconsolidation (D2)",
    "test_phase6_promotion_ladder": "Phase  6 — Promotion ladder (0 → 1 → 2 → 3)",
    "test_phase7_decay": "Phase  7 — Salience decay",
    "test_relations_schema_evidence": "Relations — Schema evidence links",
    "test_relations_evidence_credits_consolidation_path_across_scopes": "Relations — Consolidation-path evidence crediting",
    "test_relations_cross_scope_isolation": "Relations — Cross-scope isolation",
    "test_metadata_gated_supersession": "Relations — Metadata-gated supersession edges",
    "test_relations_prototype_coactivation": "Relations — Prototype-level co-activation edges",
    "test_relations_schema_coactivation": "Relations — Schema-level co-activation edges + cross-scope isolation",
    "test_relations_graph_expansion_respects_cross_scope_isolation": "Relations — Graph-expansion cross-scope isolation",
    "test_forget_unforget_lifecycle": "Forget / unforget lifecycle",
}


def pytest_runtest_logstart(nodeid: str, location) -> None:
    """Print a one-line phase description before each acceptance test."""
    test_name = nodeid.split("::")[-1]
    desc = _PHASES.get(test_name)
    if desc:
        print(f"\n  {desc}", flush=True)
