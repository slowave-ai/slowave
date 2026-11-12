"""Regression checks for the handover benchmark accounting."""

import importlib.util
import json
from pathlib import Path

import pytest

# The benchmark tooling lives under private/, which the public release never
# ships, so skip at collection instead of failing the whole suite there.
BENCH = Path(__file__).resolve().parents[2] / "private/handover/bench_dev"
if not (BENCH / "run_bench.py").is_file():
    pytest.skip("private benchmark tooling is not published", allow_module_level=True)

spec = importlib.util.spec_from_file_location("handover_bench", BENCH / "run_bench.py")
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


def test_missing_and_failed_tasks_count_as_misses():
    tasks = [
        dict(id=i, scope="project:p", cat="normal", must=["a"], should=[], ok=[], noise=[])
        for i in ("failed", "missing")
    ]
    raw = [dict(id="failed", act=[], rec=[], payload_bytes=0, calls=1, error="failure")]
    results = bench.complete_task_results(tasks, raw)
    scored = [bench.score_task(t, results[t["id"]], {}, "act") for t in tasks]
    assert len(scored) == 2
    assert all(x["must_total"] == 1 and x["must_hit"] == 0 for x in scored)


def test_multiturn_counts_standing_and_failed_turns():
    sessions = [dict(id="s", turns=[dict(id="t1", must=["a"]), dict(id="t2", must=["a"])])]
    raw = [dict(id="t1", act=[], stand=[dict(id="sch_1", keys=["a"])])]
    turns = bench.multiturn(raw, sessions, {})["sessions"][0]["turns"]
    assert len(turns) == 2
    assert turns[0]["fresh_cov"] == 1
    assert turns[1]["fresh_cov"] == 0 and turns[1]["cum_cov"] == 1


def test_ad6_must_and_noise_are_in_task_scope():
    task = next(
        t for t in json.loads((BENCH / "bench/tasks_adv.json").read_text()) if t["id"] == "AD6"
    )
    memories = {m["key"]: m for m in json.loads((BENCH / "bench/memories.json").read_text())}
    assert all(memories[k]["scope"] == task["scope"] for k in task["must"] + task["noise"])
