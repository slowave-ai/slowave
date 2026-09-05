"""Coverage for the worker dashboard payload and presentation."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import slowave.dashboard.app as dashboard_app

_WORKER_RUNS_DDL = """
    CREATE TABLE worker_runs (
      id INTEGER PRIMARY KEY,
      started_ts INTEGER NOT NULL,
      ended_ts INTEGER,
      duration_ms INTEGER,
      triggered_by TEXT NOT NULL,
      prototypes_processed INTEGER NOT NULL DEFAULT 0,
      episodes_processed INTEGER NOT NULL DEFAULT 0,
      schemas_created INTEGER NOT NULL DEFAULT 0,
      schemas_reinforced INTEGER NOT NULL DEFAULT 0,
      schemas_skipped INTEGER NOT NULL DEFAULT 0,
      schemas_decayed INTEGER NOT NULL DEFAULT 0,
      error_text TEXT
    )
    """


def _make_worker_db(tmp_path: Path) -> Path:
    db_path = tmp_path / "worker.sqlite3"
    conn = sqlite3.connect(db_path)
    conn.execute(_WORKER_RUNS_DDL)
    conn.commit()
    conn.close()
    return db_path


def test_worker_payload_separates_pass_health_from_successful_work(
    tmp_path: Path, monkeypatch
) -> None:
    db_path = tmp_path / "worker.sqlite3"
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE worker_runs (
          id INTEGER PRIMARY KEY,
          started_ts INTEGER NOT NULL,
          ended_ts INTEGER,
          duration_ms INTEGER,
          triggered_by TEXT NOT NULL,
          prototypes_processed INTEGER NOT NULL DEFAULT 0,
          episodes_processed INTEGER NOT NULL DEFAULT 0,
          schemas_created INTEGER NOT NULL DEFAULT 0,
          schemas_reinforced INTEGER NOT NULL DEFAULT 0,
          schemas_skipped INTEGER NOT NULL DEFAULT 0,
          schemas_decayed INTEGER NOT NULL DEFAULT 0,
          error_text TEXT
        )
        """)
    conn.executemany(
        "INSERT INTO worker_runs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (1, 100, 101, 1000, "worker", 3, 5, 1, 1, 1, 2, None),
            (2, 200, 201, 1000, "session_end", 9, 9, 9, 9, 9, 9, "failed"),
            (3, 300, None, None, "worker", 0, 0, 0, 0, 0, 0, None),
        ],
    )
    conn.commit()
    conn.close()
    monkeypatch.setattr(
        dashboard_app,
        "_slowave_processes",
        lambda: [{"kind": "worker", "pid": 42, "age_seconds": 60}],
    )

    payload = dashboard_app._worker_runs_payload(str(db_path), {"limit": ["10"]})

    assert payload["worker"] == {
        "running": True,
        "process_count": 1,
        "processes": [{"kind": "worker", "pid": 42, "age_seconds": 60}],
    }
    assert payload["trigger_counts"] == {"worker": 2, "session_end": 1}
    assert payload["summary"]["total_passes"] == 3
    assert payload["summary"]["successful_passes"] == 1
    assert payload["summary"]["failed_passes"] == 1
    assert payload["summary"]["incomplete_passes"] == 1
    assert payload["summary"]["total_episodes_processed"] == 5
    assert payload["summary"]["total_prototypes_processed"] == 3
    assert payload["summary"]["total_schemas_created"] == 1


def test_worker_payload_filters_runs_by_time_range(tmp_path, monkeypatch) -> None:
    db_path = tmp_path / "worker_range.sqlite3"
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE worker_runs (
          id INTEGER PRIMARY KEY,
          started_ts INTEGER NOT NULL,
          ended_ts INTEGER,
          duration_ms INTEGER,
          triggered_by TEXT NOT NULL,
          prototypes_processed INTEGER NOT NULL DEFAULT 0,
          episodes_processed INTEGER NOT NULL DEFAULT 0,
          schemas_created INTEGER NOT NULL DEFAULT 0,
          schemas_reinforced INTEGER NOT NULL DEFAULT 0,
          schemas_skipped INTEGER NOT NULL DEFAULT 0,
          schemas_decayed INTEGER NOT NULL DEFAULT 0,
          error_text TEXT
        )
        """)
    conn.executemany(
        "INSERT INTO worker_runs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (1, 900_000, 900_001, 1000, "worker", 1, 1, 1, 1, 1, 1, None),
            (2, 395_200, 395_201, 1000, "worker", 2, 2, 2, 2, 2, 2, None),
            (3, 100_000, 100_001, 1000, "session_end", 3, 3, 3, 3, 3, 3, None),
        ],
    )
    conn.commit()
    conn.close()
    monkeypatch.setattr(dashboard_app.time, "time", lambda: 1_000_000)

    # now=1_000_000 → a 1w window starts at 395_200, so run 3 is excluded.
    one_week = dashboard_app._worker_runs_payload(str(db_path), {"limit": ["100"], "range": ["1w"]})
    assert [r["id"] for r in one_week["runs"]] == [1, 2]

    all_runs = dashboard_app._worker_runs_payload(
        str(db_path), {"limit": ["100"], "range": ["all"]}
    )
    assert [r["id"] for r in all_runs["runs"]] == [1, 2, 3]

    # No range param (or an unknown one) falls back to an unbounded window.
    default = dashboard_app._worker_runs_payload(str(db_path), {"limit": ["100"]})
    assert [r["id"] for r in default["runs"]] == [1, 2, 3]


def test_worker_chart_buckets_aggregate_outcomes_over_range(tmp_path: Path, monkeypatch) -> None:
    db_path = _make_worker_db(tmp_path)
    conn = sqlite3.connect(db_path)
    conn.executemany(
        "INSERT INTO worker_runs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            # id, started_ts, ended_ts, dur, trigger, pro, epi, created, reinf, skip, decayed, err
            (1, 100_000, 100_001, 1000, "worker", 1, 1, 10, 1, 0, 1, None),
            (2, 500_000, 500_001, 1000, "worker", 1, 1, 1, 20, 0, 2, None),
            (3, 900_000, 900_001, 1000, "worker", 1, 1, 3, 0, 4, 0, "boom"),
        ],
    )
    conn.commit()
    conn.close()
    monkeypatch.setattr(dashboard_app.time, "time", lambda: 1_000_000)

    all_chart = dashboard_app._worker_runs_payload(
        str(db_path), {"limit": ["100"], "range": ["all"]}
    )["chart"]
    assert all_chart["range"] == "all"
    assert all_chart["pass_count_total"] == 3
    assert all_chart["bucket_seconds"] == 86400
    # Outcomes must be summed per time bucket across all three passes.
    assert sum(b["created"] for b in all_chart["buckets"]) == 10 + 1 + 3
    assert sum(b["reinforced"] for b in all_chart["buckets"]) == 1 + 20
    assert sum(b["skipped"] for b in all_chart["buckets"]) == 4
    assert sum(b["errors"] for b in all_chart["buckets"]) == 1
    # Zero-filling: every bucket carries all keys.
    assert all({"ts", "created", "reinforced", "skipped"} <= set(b) for b in all_chart["buckets"])

    # A 1w window (now=1_000_000) starts at 395_200, dropping the 100_000 run.
    week_chart = dashboard_app._worker_runs_payload(
        str(db_path), {"limit": ["100"], "range": ["1w"]}
    )["chart"]
    assert week_chart["pass_count_total"] == 2
    assert week_chart["buckets"][0]["ts"] == 345_600
    assert sum(b["created"] for b in week_chart["buckets"]) == 1 + 3


def test_maintenance_history_is_moved_to_diagnostics_without_decorative_chart() -> None:
    source = Path(__file__).parents[2] / "slowave/dashboard/ui/src/pages.tsx"
    app = source.read_text()
    for label in ("Diagnostics", "Consolidation history", "Maintenance runs", "No maintenance passes recorded"):
        assert label in app
    assert "maintenanceExpanded ? 50 : 10" in app
    assert '"Show 50 runs"' in app
    assert '"Show recent 10"' in app
    assert "schemas_created" in app and "schemas_reinforced" in app
    assert "Consolidation activity chart" not in app
    assert "LoadingRows" in app and "InlineError" in app and "EmptyState" in app
