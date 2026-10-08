"""Paging foundations: pure preparation cannot publish phantom exposure."""

import json

import pytest

from slowave.mcp.activation_catalog import (
    MAX_MEMORY_RECORD_CHARS,
    MAX_PAGE_CHARS,
    MAX_PROCEDURE_RECORD_CHARS,
    CatalogCompatibilityError,
    CatalogPreparationError,
    FrozenActivationCatalog,
)


def prepare(count, procedures=(), memories=None, spans=None, **kwargs):
    return FrozenActivationCatalog.prepare(
        retrieval_id="ctx_test",
        session_id="sess_test",
        scope="project:test",
        memories=(
            memories
            if memories is not None
            else [
                {
                    "memory_id": f"sch_{i}",
                    "content": f"Shard {i} must use port {9000+i}.",
                    "pathway": "direct",
                }
                for i in range(count)
            ]
        ),
        procedures=list(procedures),
        contribution_spans=spans or {},
        catalog_truncated=False,
        first_page_metadata={
            "continuity_id": "cont_test",
            "continuity_state": "started",
            "warnings": [],
        },
        **kwargs,
    )


@pytest.mark.parametrize(
    "count,sizes",
    [
        (0, [0]),
        (2, [2]),
        (4, [4]),
        (8, [5, 3]),
        (10, [5, 5]),
        (13, [5, 5, 3]),
        (23, [5, 5, 5, 5, 3]),
        (100, [5] * 20),
    ],
)
def test_memory_page_contract_and_duplicate_free_replay(count, sizes):
    catalog = prepare(count)
    offset = (0, 0)
    observed = []
    ids = []
    while True:
        page, successor = catalog.page(*offset)
        assert (page, successor) == catalog.page(*offset)
        assert len(json.dumps(page, ensure_ascii=True)) < MAX_PAGE_CHARS
        observed.append(len(page["memories"]))
        ids.extend(m["memory_id"] for m in page["memories"])
        assert page["relevant_total"] == count
        if successor is None:
            break
        assert successor != offset
        offset = successor
    assert observed == sizes
    assert ids == [f"sch_{i}" for i in range(count)]


@pytest.mark.parametrize("size", [1, 5, 7, 10])
def test_custom_page_size_is_frozen_across_restart(size):
    catalog = FrozenActivationCatalog.read(prepare(23, page_size=size).snapshot_json)
    offset = (0, 0)
    seen = []
    while True:
        page, successor = catalog.page(*offset)
        assert len(page["memories"]) <= size
        seen.extend(m["memory_id"] for m in page["memories"])
        if successor is None:
            break
        offset = successor
    assert seen == [f"sch_{i}" for i in range(23)]


def test_existing_v4_snapshot_retains_ten_memory_pages():
    snapshot = json.loads(prepare(23).snapshot_json)
    snapshot["paging_contract_version"] = "ten-memory-budget-pages-v4"
    snapshot.pop("requested_page_size")
    old = FrozenActivationCatalog.read(json.dumps(snapshot))
    first, successor = old.page()
    assert len(first["memories"]) == 10
    assert len(old.page(*successor)[0]["memories"]) == 10


@pytest.mark.parametrize(
    "first_page_size,sizes",
    [(1, [1, 5, 5, 5, 4]), (7, [7, 5, 5, 3]), (10, [10, 5, 5])],
)
def test_strong_run_first_page_with_five_memory_continuation(first_page_size, sizes):
    catalog = prepare(20, first_page_size=first_page_size)
    offset = (0, 0)
    observed = []
    ids = []
    while True:
        page, successor = catalog.page(*offset)
        observed.append(len(page["memories"]))
        ids.extend(m["memory_id"] for m in page["memories"])
        if successor is None:
            break
        offset = successor
    assert observed == sizes
    assert ids == [f"sch_{i}" for i in range(20)]


def test_first_page_size_is_capped_and_validated():
    with pytest.raises(CatalogPreparationError):
        prepare(20, first_page_size=0)
    with pytest.raises(CatalogPreparationError):
        prepare(20, first_page_size=11)
    with pytest.raises(CatalogPreparationError):
        prepare(20, first_page_size="7")


def test_snapshot_without_first_page_size_serves_the_legacy_five_memory_page():
    catalog = prepare(8)
    legacy = json.loads(catalog.snapshot_json)
    legacy.pop("first_page_size")
    legacy["paging_contract_version"] = "five-memory-pages-v1"
    replayed = FrozenActivationCatalog.read(json.dumps(legacy, sort_keys=True))
    page, successor = replayed.page()
    assert len(page["memories"]) == 5
    assert successor == (5, 0)


def test_procedures_never_compete_with_memory_slots_and_tail_caveats_survive():
    procedures = [
        {
            "procedure_id": f"proc_{i}",
            "summary": "Migrate safely",
            "caveats": ["Do not stop the billing worker before its queue drains."],
        }
        for i in range(10)
    ]
    catalog = prepare(8, procedures, first_page_size=5)
    offset = (0, 0)
    sizes = []
    seen = []
    while True:
        page, successor = catalog.page(*offset)
        sizes.append(len(page["memories"]))
        seen.extend(page["procedures"])
        if successor is None:
            break
        offset = successor
    assert sizes == [5, 3, 0, 0]
    assert seen == procedures
    assert page["accessible_field"] == {}


@pytest.mark.parametrize("position", [0, 7])
def test_required_late_source_contributions_survive_initial_tail_and_source_mutation(position):
    clause = "For Atlas deployment, if the billing worker is active, do not run migration; the lock timeout must stay at 12 seconds."
    source = ("Historical garden notes are irrelevant to this task. " * 40) + clause
    start = source.index(clause)
    assert start > 500
    memories = [
        {
            "memory_id": f"sch_{i}",
            "content": f"Shard {i} must use port {9000+i}.",
            "pathway": "direct",
        }
        for i in range(8)
    ]
    memories[position]["content"] = source
    catalog = prepare(
        8, memories=memories, spans={f"sch_{position}": ((start, len(source)),)}, first_page_size=5
    )
    memories[position]["content"] = "mutated source"
    page, next_offset = catalog.page()
    if position >= 5:
        page, _ = catalog.page(*next_offset)
    delivered = next(item for item in page["memories"] if item["memory_id"] == f"sch_{position}")
    assert delivered["content"] == clause
    assert delivered["provenance"]["source_excerpt"]["version"] == 1
    assert delivered["provenance"]["source_excerpt"]["source_spans"] == [[start, len(source)]]


def test_snapshot_survives_restart_and_config_rollback_without_reinterpretation(tmp_path):
    catalog = prepare(13, first_page_size=5)
    first, offset = catalog.page()
    path = tmp_path / "snapshot.json"
    path.write_text(catalog.snapshot_json)
    restored = FrozenActivationCatalog.read(path.read_text())
    assert restored.page() == catalog.page()
    second, offset = restored.page(*offset)
    third, offset = restored.page(*offset)
    assert [len(p["memories"]) for p in (first, second, third)] == [5, 5, 3]
    assert all(
        p["retrieval_policy_version"] == "activation-pool-relative-v1"
        for p in (first, second, third)
    )
    assert offset is None


def test_unknown_versions_and_legacy_lists_need_their_own_reader():
    catalog = prepare(8)
    changed = json.loads(catalog.snapshot_json)
    changed["snapshot_format_version"] = 99
    with pytest.raises(CatalogCompatibilityError, match="unsupported"):
        FrozenActivationCatalog.read(json.dumps(changed))
    with pytest.raises(CatalogCompatibilityError, match="legacy"):
        FrozenActivationCatalog.read("[]")


@pytest.mark.parametrize("position", [0, 7])
def test_oversized_first_or_tail_record_fails_before_any_page_is_available(position):
    memories = [
        {"memory_id": f"sch_{i}", "content": "required fact", "pathway": "direct"} for i in range(8)
    ]
    memories[position]["provenance"] = {"unsupported": "x" * MAX_MEMORY_RECORD_CHARS}
    with pytest.raises(CatalogPreparationError, match="oversized memory"):
        prepare(8, memories=memories)


def test_procedure_caveats_are_never_silently_dropped():
    with pytest.raises(CatalogPreparationError, match="oversized procedure"):
        prepare(8, [{"procedure_id": "proc_1", "caveats": ["x" * MAX_PROCEDURE_RECORD_CHARS]}])


def test_escape_and_unicode_heavy_provenance_still_fit_five_records():
    memories = [
        {
            "memory_id": f"sch_{i}",
            "content": '\\"\n☃' * 150,
            "pathway": "direct",
            "provenance": {"observed": True},
        }
        for i in range(5)
    ]
    page, offset = prepare(5, memories=memories).page()
    assert len(page["memories"]) == 5
    assert offset is None
    assert len(json.dumps(page, ensure_ascii=True)) < MAX_PAGE_CHARS


def test_serialization_failures_and_incomplete_or_oversized_contributions_are_explicit():
    with pytest.raises(TypeError):
        prepare(
            1, memories=[{"memory_id": "sch_1", "content": "fact", "provenance": {"bad": object()}}]
        )
    with pytest.raises(CatalogPreparationError, match="requires contribution"):
        prepare(1, memories=[{"memory_id": "sch_1", "content": "x" * 2000}])
    with pytest.raises(CatalogPreparationError, match="exceed preview"):
        prepare(
            1,
            memories=[{"memory_id": "sch_1", "content": "x" * 2000}],
            spans={"sch_1": ((0, 2000),)},
        )


def test_metadata_cannot_override_the_selected_catalog_or_page_contract():
    with pytest.raises(CatalogPreparationError, match="cannot override"):
        FrozenActivationCatalog.prepare(
            retrieval_id="ctx_test",
            session_id="sess_test",
            scope="project:test",
            memories=[],
            procedures=[],
            contribution_spans={},
            catalog_truncated=False,
            first_page_metadata={"memories": [{"memory_id": "unselected"}]},
        )


def test_even_short_sources_require_valid_assessment_offsets():
    with pytest.raises(CatalogPreparationError, match="invalid contribution"):
        prepare(
            1,
            memories=[{"memory_id": "sch_1", "content": "short fact"}],
            spans={"sch_1": ((0, 99),)},
        )


@pytest.mark.parametrize("count", [0, 1, 2, 4, 7, 11, 23, 100])
def test_payload_pages_default_to_five_without_capping_the_catalog(count):
    page, successor = prepare(count).page()
    assert len(page["memories"]) == min(count, 5)
    assert (successor is not None) == (count > 5)


def test_payload_budget_varies_counts_with_content_and_survives_restart():
    memories = [
        {"memory_id": f"sch_{i}", "content": "x" * 1000, "pathway": "direct"} for i in range(29)
    ]
    catalog = prepare(29, memories=memories)
    restored = FrozenActivationCatalog.read(catalog.snapshot_json)
    offset = (0, 0)
    pages, ids = [], []
    while True:
        page, successor = restored.page(*offset)
        assert restored.page(*offset) == catalog.page(*offset)
        pages.append(len(page["memories"]))
        ids.extend(m["memory_id"] for m in page["memories"])
        if successor is None:
            break
        assert successor[0] > offset[0]
        offset = successor
    assert pages == [5, 5, 5, 5, 5, 4]
    assert ids == [m["memory_id"] for m in memories]


def test_existing_v3_snapshot_keeps_its_original_budget_only_pages():
    snapshot = json.loads(prepare(23).snapshot_json)
    snapshot["paging_contract_version"] = "payload-budget-pages-v3"
    old = FrozenActivationCatalog.read(json.dumps(snapshot))
    page, successor = old.page()
    assert len(page["memories"]) == 23
    assert successor is None


def test_recall_evidence_has_separate_budget_and_does_not_displace_memories():
    evidence = {
        "evidence": [{"content": "e" * 1000} for _ in range(8)],
        "evidence_mode": "full",
        "evidence_truncated": False,
    }
    catalog = prepare(23, policy_version="recall-pool-relative-v1", recall_evidence=evidence)
    page, successor = catalog.page()
    assert len(page["memories"]) == 5
    assert page["evidence"] == evidence["evidence"]
    restored = FrozenActivationCatalog.read(catalog.snapshot_json)
    tail, _ = restored.page(*successor)
    assert len(tail["memories"]) == 5
    assert tail["evidence"] == []
    assert tail["evidence_mode"] == "full"


def test_payload_guard_may_return_fewer_than_ten_without_losing_records():
    memories = [
        {
            "memory_id": f"sch_{i}",
            "content": "fact",
            "pathway": "direct",
            "provenance": {"detail": "x" * 6000},
        }
        for i in range(3)
    ]
    catalog = prepare(3, memories=memories)
    first, offset = catalog.page()
    assert len(first["memories"]) == 2
    second, successor = catalog.page(*offset)
    assert len(second["memories"]) == 1
    assert successor is None


def test_unicode_evidence_is_budgeted_without_silently_dropping_memories():
    evidence = {
        "evidence": [{"content": "☃" * 1000} for _ in range(8)],
        "evidence_mode": "full",
        "evidence_truncated": False,
    }
    page, _ = prepare(10, recall_evidence=evidence).page()
    assert len(page["memories"]) == 5
    assert 0 < len(page["evidence"]) < 8
    assert page["evidence_truncated"] is True
    assert len(evidence["evidence"]) == 8
