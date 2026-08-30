from tests.benchmarks.evidence_bundle import (
    assemble_evidence_bundle,
    bundle_metrics,
    parse_structured_evidence,
)


def _episode(rank: int, date: str, text: str) -> str:
    return f"[EPISODE {rank} | date={date} 00:00Z]\n{text}"


def test_bundle_selects_relevant_distinct_events_and_orders_them() -> None:
    evidence = "\n\n".join(
        [
            "=== EPISODES (raw conversation excerpts) ===",
            _episode(1, "2024-03-03", "I planted basil in the herb garden."),
            _episode(2, "2024-01-01", "Unrelated discussion about a movie."),
            _episode(3, "2024-03-27", "I harvested basil from the herb garden."),
            _episode(4, "2024-03-03", "I planted basil in the herb garden."),
        ]
    )

    bundle = assemble_evidence_bundle(
        "How long between planting and harvesting basil?", evidence, max_items=2
    )

    assert "planted basil" in bundle
    assert "harvested basil" in bundle
    assert "Unrelated" not in bundle
    assert bundle.index("2024-03-03") < bundle.index("2024-03-27")


def test_parser_and_metrics_preserve_source_boundaries() -> None:
    evidence = "\n\n".join(
        [_episode(1, "2024-01-01", "alpha beta"), _episode(2, "2024-01-02", "alpha beta")]
    )

    items = parse_structured_evidence(evidence)
    metrics = bundle_metrics(evidence, expected="alpha")

    assert len(items) == 2
    assert items[0].date == "2024-01-01"
    assert metrics["expected_token_coverage"] == 1.0
    assert metrics["mean_pairwise_jaccard"] == 1.0


def test_parser_supports_legacy_dated_episode_blocks() -> None:
    evidence = (
        "schema prefix without a boundary [2024-01-01] User: first\n[2024-02-01] User: second"
    )

    items = parse_structured_evidence(evidence)

    assert [item.date for item in items] == ["2024-01-01", "2024-02-01"]
    assert items[0].text == "[2024-01-01] User: first"
