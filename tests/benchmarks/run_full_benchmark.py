#!/usr/bin/env python3
"""Run all Slowave benchmarks in sequence and print a consolidated summary.

Benchmarks (execution order matters — see note below):
  - LoCoMo         (1986 q, ~3 min)
  - LongMemEval    (500 q, ~10 min with consolidation)
  - DMR            (500 q, ~9 min)
  - StaleMemory    (1200 scenarios, ~15 min)
  - BEAM           (optional, ~2000 q, needs OPENROUTER_API_KEY + HF datasets)

NOTE: LoCoMo must run before LongMemEval. Running LME first degrades LoCoMo
by ~5-6 pp due to OS memory pressure after LME's large DB operations
(confirmed Jun 12 2026: standalone=83-84%, post-LME=77.5%).

Usage:
  # Full suite (~40 min, external datasets must be present):
  python tests/benchmarks/run_full_benchmark.py

  # Quick smoke - 5 questions/scenarios per category:
  python tests/benchmarks/run_full_benchmark.py --limit 5

  # Skip benchmarks you don't have data for:
  python tests/benchmarks/run_full_benchmark.py --skip stalememory locomo

  # Include BEAM (requires OPENROUTER_API_KEY; runs with --beam-workers
  # parallel conversation workers by default — BEAM is ~99% LLM-round-trip
  # time, so this matters a lot). Each worker loads its own encoder/engine
  # (~1.3-1.4GB RSS minimum, more once it's ingested a conversation) — size
  # --beam-workers to your available RAM, not just CPU count:
  python tests/benchmarks/run_full_benchmark.py --beam
  python tests/benchmarks/run_full_benchmark.py --beam --beam-workers 6

  # Custom output directory:
  python tests/benchmarks/run_full_benchmark.py --out-dir data/suite_runs/2026-06-06

Each benchmark writes its own JSON to --out-dir, then a combined
summary JSON is written to --out-dir/summary.json.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
INTEG = REPO_ROOT / "tests" / "benchmarks"
sys.path.insert(0, str(REPO_ROOT))

from tests.benchmarks.llm_judge import confirm_paid_run  # noqa: E402

# (skip-key, display name, output filename) — same order the benchmarks run in.
BENCHMARK_PLAN = [
    ("locomo", "LoCoMo", "locomo.json"),
    ("longmemeval", "LongMemEval", "longmemeval.json"),
    ("dmr", "DMR", "dmr.json"),
    ("stalememory", "StaleMemory", "stalememory.json"),
    ("beam", "BEAM", "beam.json"),
]
_ELAPSED_KEYS = ("total_elapsed_s", "elapsed_s")


def _run(cmd: list[str], label: str) -> bool:
    print("\n" + "=" * 60)
    print(f"  {label}")
    print("=" * 60)
    t0 = time.time()
    # Pass parent environment to subprocess so OPENROUTER_API_KEY is inherited
    r = subprocess.run(cmd, cwd=str(REPO_ROOT), env=os.environ)
    elapsed = time.time() - t0
    if r.returncode != 0:
        print(f"[ERROR] {label} exited {r.returncode}")
        return False
    print(f"[OK] {label} finished in {elapsed:.0f}s")
    return True


def _load(path: Path) -> dict | None:
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        print(f"[WARN] could not load {path}: {e}")
        return None


def _fmt_duration(seconds: float) -> str:
    seconds = int(round(seconds))
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def _find_recent_elapsed(runs_dir: Path, filename: str) -> tuple[float, str] | None:
    """Most recent past run's elapsed time for one benchmark output file, or None."""
    if not runs_dir.exists():
        return None
    candidates: list[tuple[str, float, str]] = []
    for d in runs_dir.iterdir():
        if not d.is_dir():
            continue
        p = d / filename
        if not p.exists():
            continue
        data = _load(p)
        if not data:
            continue
        meta = data.get("meta", {})
        elapsed = next((meta[k] for k in _ELAPSED_KEYS if k in meta), None)
        if elapsed is None:
            continue
        candidates.append((meta.get("created_at", d.name), float(elapsed), d.name))
    if not candidates:
        return None
    candidates.sort(key=lambda c: c[0])
    _created, elapsed, run_name = candidates[-1]
    return elapsed, run_name


def _print_plan(planned: list[tuple[str, str]], runs_dir: Path, limit: int) -> None:
    """planned: list of (display_name, output_filename) for benchmarks about to run."""
    print("\nBenchmarks to run:", flush=True)
    total = 0.0
    missing = False
    for display_name, filename in planned:
        found = _find_recent_elapsed(runs_dir, filename)
        if found is None:
            missing = True
            print(f"  {display_name:<15} no history yet", flush=True)
            continue
        elapsed, run_name = found
        total += elapsed
        caveat = "" if limit == 0 else f" — full run, --limit {limit} will differ"
        print(
            f"  {display_name:<15} ~{_fmt_duration(elapsed):<8} (from {run_name}{caveat})",
            flush=True,
        )
    if total:
        tail = "  (+ some benchmarks have no history)" if missing else ""
        print(f"  {'Total estimated:':<15} ~{_fmt_duration(total)}{tail}", flush=True)
    print(flush=True)


def _summary_qa(d: dict) -> dict:
    s = d.get("summary", {})
    out = {
        "n": s.get("n", 0),
        "score_pct": s.get("score_pct", 0.0),
    }
    # keyword-overlap benchmarks only — BEAM's summary has no recall_at_k/mrr
    if "recall_at_k" in s:
        out["recall_at_k"] = s["recall_at_k"]
        out["mrr"] = s.get("mrr", 0.0)
    # Only present when --judge-model was passed (LoCoMo/LongMemEval only)
    if s.get("llm_judge_score_pct") is not None:
        out["llm_judge_score_pct"] = s["llm_judge_score_pct"]
        out["llm_judge_cost_usd"] = s.get("llm_judge_cost_usd")
    return out


def _summary_stale(d: dict) -> dict:
    s = d.get("summary", {})
    return {
        "n": s.get("n", 0),
        "detection_rate_pct": round(s.get("detection_rate", 0) * 100, 1),
        "stale_rate_pct": round(s.get("stale_rate", 0) * 100, 1),
        "no_answer_rate_pct": round(s.get("no_answer_rate", 0) * 100, 1),
        "recall_at_k": s.get("recall_at_k", {}),
        "mrr": s.get("mrr", 0.0),
    }


def _load_prev_summaries(runs_dir: Path, current_dir: Path, limit: int, n: int = 5) -> list[dict]:
    """Return up to n previous summary.json dicts, oldest-first, matching limit setting.

    Ordered by each summary's own `meta.created_at` timestamp, not by directory
    name — a custom `--out-dir` (e.g. "baseline") sorts lexicographically before
    or after timestamped run dirs regardless of when it was actually created,
    which previously could put stale runs where the most recent one belongs.
    """
    if not runs_dir.exists():
        return []
    candidates = []
    for d in runs_dir.iterdir():
        if d == current_dir or not d.is_dir():
            continue
        summary_path = d / "summary.json"
        if not summary_path.exists():
            continue
        try:
            with open(summary_path) as f:
                data = json.load(f)
            if data.get("meta", {}).get("limit", 0) != limit:
                continue
            candidates.append(data)
        except Exception:
            continue
    candidates.sort(key=lambda d: d.get("meta", {}).get("created_at", ""))
    return candidates[-n:]


def _print_trend(current: dict[str, dict], prev_summaries: list[dict]) -> None:
    """Print a trend table comparing current results against previous runs."""
    if not prev_summaries:
        return

    def _primary(name: str, r: dict | None) -> float | None:
        if r is None:
            return None
        return r.get("detection_rate_pct" if name == "stalememory" else "score_pct")

    def _short_date(summary: dict) -> str:
        ts = summary.get("meta", {}).get("created_at", "")
        try:
            dt = datetime.fromisoformat(ts)
            return dt.strftime("%b %d")
        except Exception:
            return "?"

    W = 9  # column width for score values
    print()
    label_count = len(prev_summaries)
    print(f"  TREND  (vs {label_count} previous comparable run{'s' if label_count > 1 else ''})")
    sep = "  " + "─" * (17 + (W + 2) * label_count + W + 2 + 7)
    print(sep)
    header = f"  {'benchmark':<15}"
    for s in prev_summaries:
        header += f"  {_short_date(s):>{W}}"
    header += f"  {'now':>{W}}  {'Δ':>5}"
    print(header)
    print(sep)

    for name, r in current.items():
        cur_val = _primary(name, r)
        if cur_val is None:
            continue
        prev_vals = [_primary(name, s.get("results", {}).get(name)) for s in prev_summaries]
        row = f"  {name:<15}"
        for v in prev_vals:
            row += f"  {(f'{v:.1f}%') if v is not None else 'n/a':>{W}}"
        row += f"  {cur_val:.1f}%"
        last = next((v for v in reversed(prev_vals) if v is not None), None)
        if last is not None:
            delta = cur_val - last
            if abs(delta) < 0.5:
                marker = f"  ~{abs(delta):.1f}"
            elif delta > 0:
                marker = f"  ↑{delta:.1f}"
            else:
                marker = f"  ↓{abs(delta):.1f}"
            row += marker
        print(row)

    print(sep)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all Slowave benchmarks")
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Max questions/scenarios per category (0=full run)",
    )
    parser.add_argument(
        "--skip",
        nargs="+",
        default=[],
        choices=["longmemeval", "locomo", "dmr", "stalememory", "beam"],
        help="Benchmarks to skip",
    )
    parser.add_argument(
        "--out-dir",
        default="",
        help="Output directory for results (default: data/suite_runs/<timestamp>)",
    )
    parser.add_argument(
        "--beam",
        action="store_true",
        help="Include BEAM benchmark (requires OPENROUTER_API_KEY and datasets)",
    )
    parser.add_argument(
        "--beam-model",
        default="deepseek/deepseek-v4-flash",
        help="LLM model for BEAM answer/judge (default: deepseek/deepseek-v4-flash)",
    )
    parser.add_argument(
        "--beam-workers",
        type=int,
        default=4,
        help="Parallel conversation workers for BEAM (default: 4; BEAM is "
        "otherwise sequential and ~100%% LLM-round-trip-bound). Each worker "
        "loads its own encoder + engine (~1.3-1.4GB RSS minimum per worker, "
        "measured) — size this to available RAM, not just CPU count",
    )
    parser.add_argument(
        "--no-consolidate",
        action="store_true",
        help="Skip consolidation (faster, lower scores)",
    )
    parser.add_argument(
        "--judge-model",
        default="",
        help="LLM-judge model for LoCoMo + LongMemEval (the only two benchmarks "
        "that support it). Costs real API tokens; unset by default so the "
        "suite stays free.",
    )
    parser.add_argument(
        "--save-full-hypotheses",
        action="store_true",
        help="Persist complete LoCoMo and LongMemEval retrieved contexts for "
        "offline judging after the suite finishes.",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Hard guarantee of zero API calls for this run: forces --judge-model "
        "off and BEAM off, even if --judge-model or --beam were also passed "
        "(a mistake, since --no-llm means you explicitly want a free run).",
    )
    parser.add_argument(
        "--yes",
        "-y",
        action="store_true",
        help="Skip the confirmation prompt before paid runs (--judge-model or --beam).",
    )
    args = parser.parse_args()

    if args.no_llm:
        if args.judge_model or args.beam:
            print("[WARN] --no-llm overrides --judge-model/--beam — forcing both off.")
        args.judge_model = ""
        args.beam = False

    judge_flag = ["--judge-model", args.judge_model] if args.judge_model else []
    full_hypotheses_flag = ["--save-full-hypotheses"] if args.save_full_hypotheses else []
    yes_flag = ["--yes"] if args.yes else []

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) if args.out_dir else REPO_ROOT / "data" / "suite_runs" / stamp
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {out_dir}", flush=True)

    py = sys.executable
    skip = set(args.skip)
    cons_flag = ["--no-consolidate"] if args.no_consolidate else []
    lim_flag = ["--limit", str(args.limit)] if args.limit else []

    planned = [
        (display_name, filename)
        for key, display_name, filename in BENCHMARK_PLAN
        if key not in skip and (key != "beam" or args.beam)
    ]
    _print_plan(planned, runs_dir=REPO_ROOT / "data" / "suite_runs", limit=args.limit)

    if judge_flag or args.beam:
        parts = []
        if judge_flag:
            parts.append(f"LoCoMo+LongMemEval judged with {args.judge_model}")
        if args.beam:
            parts.append(f"BEAM answered+judged with {args.beam_model}")
        confirm_paid_run(
            "This suite run will make paid API calls: " + "; ".join(parts) + ". "
            "Sub-benchmarks won't prompt again — this is the one confirmation for the whole suite.",
            None,
            assume_yes=args.yes,
        )
        yes_flag = ["--yes"]  # sub-scripts must not double-prompt after this

    results: dict[str, dict] = {}
    suite_start = time.time()

    # LoCoMo runs FIRST (before LME) to keep the run order consistent across benchmarks.
    # NOTE: empirical data (Jun 2026, 14 runs) shows LoCoMo scores HIGHER when LME ran
    # just before it in the same OS session (~81-83%) than when it runs cold (~76-81%,
    # mean 78.8%, stdev 2.3%). The earlier comment claiming post-LME hurts LoCoMo was
    # incorrect — the effect is the opposite, likely ONNX/encoder warmup benefits.
    # Cold-run variance of ~5 pp is normal; don't treat it as a regression signal.
    if "locomo" not in skip:
        ds = REPO_ROOT / "data" / "locomo" / "locomo10.json"
        if not ds.exists():
            print(f"[SKIP] LoCoMo: {ds} not found")
        else:
            out = out_dir / "locomo.json"
            cmd = (
                [
                    py,
                    str(INTEG / "locomo_eval.py"),
                    "--dataset",
                    str(ds),
                    "--assignment-threshold",
                    "0.85",
                    "--out",
                    str(out),
                ]
                + cons_flag
                + lim_flag
                + judge_flag
                + full_hypotheses_flag
                + (yes_flag if judge_flag else [])
            )
            if _run(cmd, "LoCoMo"):
                d = _load(out)
                if d:
                    results["locomo"] = _summary_qa(d)

    # LongMemEval
    if "longmemeval" not in skip:
        ds = REPO_ROOT / "data" / "longmemeval" / "longmemeval_oracle.json"
        if not ds.exists():
            print(f"[SKIP] LongMemEval: {ds} not found")
        else:
            out = out_dir / "longmemeval.json"
            cmd = (
                [
                    py,
                    str(INTEG / "longmemeval_eval.py"),
                    "--dataset",
                    str(ds),
                    "--assignment-threshold",
                    "0.85",
                    "--out",
                    str(out),
                ]
                + cons_flag
                + lim_flag
                + judge_flag
                + full_hypotheses_flag
                + (yes_flag if judge_flag else [])
            )
            if _run(cmd, "LongMemEval"):
                d = _load(out)
                if d:
                    results["longmemeval"] = _summary_qa(d)

    # DMR
    if "dmr" not in skip:
        ds = REPO_ROOT / "data" / "dmr_original" / "msc_self_instruct.jsonl"
        if not ds.exists():
            print(f"[SKIP] DMR: {ds} not found")
        else:
            out = out_dir / "dmr.json"
            cmd = [
                py,
                str(INTEG / "dmr_original_eval.py"),
                "--dataset",
                str(ds),
                "--out",
                str(out),
            ] + lim_flag
            if _run(cmd, "DMR (MSC Self-Instruct)"):
                d = _load(out)
                if d:
                    results["dmr"] = _summary_qa(d)

    # StaleMemory
    if "stalememory" not in skip:
        ds = REPO_ROOT / "data" / "stalememory" / "scenarios.jsonl"
        if not ds.exists():
            print(f"[SKIP] StaleMemory: {ds} not found")
        else:
            out = out_dir / "stalememory.json"
            cmd = (
                [
                    py,
                    str(INTEG / "stalememory_eval.py"),
                    "--dataset",
                    str(ds),
                    "--assignment-threshold",
                    "0.85",
                    "--out",
                    str(out),
                ]
                + cons_flag
                + lim_flag
            )
            if _run(cmd, "StaleMemory"):
                d = _load(out)
                if d:
                    results["stalememory"] = _summary_stale(d)

    # BEAM (optional — requires OPENROUTER_API_KEY + HuggingFace datasets)
    if "beam" not in skip and args.beam:
        if not os.environ.get("OPENROUTER_API_KEY"):
            print("[SKIP] BEAM: OPENROUTER_API_KEY not set")
        else:
            out = out_dir / "beam.json"
            cmd = (
                [
                    py,
                    str(INTEG / "beam_eval.py"),
                    "--chat-sizes",
                    "1M",
                    "--answerer-model",
                    args.beam_model,
                    "--judge-model",
                    args.beam_model,
                    "--workers",
                    str(args.beam_workers),
                    "--out",
                    str(out),
                ]
                + cons_flag
                + lim_flag
                + yes_flag
            )
            if _run(cmd, "BEAM"):
                d = _load(out)
                if d:
                    results["beam"] = _summary_qa(d)

    # Summary
    total_elapsed = time.time() - suite_start
    summary = {
        "meta": {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "total_elapsed_s": round(total_elapsed, 1),
            "limit": args.limit,
            "consolidate": not args.no_consolidate,
            "skipped": list(skip),
        },
        "results": results,
    }
    summary_path = out_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print(f"  SUITE COMPLETE  ({total_elapsed:.0f}s)")
    print("=" * 60)
    for name, r in results.items():
        if name == "stalememory":
            det = r["detection_rate_pct"]
            stl = r["stale_rate_pct"]
            n = r["n"]
            print(f"  {name:<15} n={n:<6} detection={det}%  stale={stl}%")
        else:
            sc = r["score_pct"]
            n = r["n"]
            print(f"  {name:<15} n={n:<6} score={sc}%")
        if r.get("recall_at_k"):
            rk = "  ".join(f"{k}={v}%" for k, v in r["recall_at_k"].items())
            print(f"  {'':<15} recall@k: {rk}  mrr={r.get('mrr', 0.0)}")
        if r.get("llm_judge_score_pct") is not None:
            cost = r.get("llm_judge_cost_usd")
            cost_s = f"  cost=${cost:.4f}" if cost is not None else "  cost=unknown pricing"
            print(f"  {'':<15} llm_judge_score: {r['llm_judge_score_pct']}%{cost_s}")
    print(f"  Summary: {summary_path}")

    prev = _load_prev_summaries(
        runs_dir=REPO_ROOT / "data" / "suite_runs",
        current_dir=out_dir,
        limit=args.limit,
    )
    _print_trend(results, prev)


if __name__ == "__main__":
    main()
