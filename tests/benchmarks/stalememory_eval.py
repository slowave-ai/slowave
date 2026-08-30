#!/usr/bin/env python3
"""StaleMemory evaluation harness for Slowave.

Evaluates Slowave's ability to detect implicit belief drift (staleness).
Each scenario has multi-session traces where a user preference changes
implicitly (through behavior, never explicitly stated after session 0).
Slowave must recall the *post-drift* (current) preference, not the
originally-established stale one.

Based on: "When Agent Memory Anchors on What Users Said: Implicit Belief
Staleness and Behavioral Belief Tracking" (StaleMemory benchmark, EMNLP 2026).
Dataset: data/stalememory/scenarios.jsonl  (1,200 scenarios)

Metrics (zero LLM calls):
  - detection_rate:    recalled post-drift value (hit)
  - stale_rate:        recalled pre-drift value (anchor pull)
  - no_answer_rate:    neither value found

Usage:
  # Quick smoke (50 per attribute/pattern combo):
  python tests/benchmarks/stalememory_eval.py --limit 5

  # Full run (1200 scenarios):
  python tests/benchmarks/stalememory_eval.py
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s %(message)s")
for noisy in (
    "sentence_transformers",
    "transformers",
    "httpx",
    "httpcore",
    "huggingface_hub",
    "filelock",
    "tqdm",
):
    logging.getLogger(noisy).setLevel(logging.ERROR)

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from slowave.core.config import SlowaveConfig
from slowave.core.engine import SlowaveEngine
from slowave.latent.replay_engine import ReplayConfig
from slowave.latent.retrieval import RetrievalConfig
from slowave.latent.salience import SalienceConfig
from slowave.latent.schema import GeometricRelationConfig
from slowave.symbolic.encoder import EncoderConfig, TextEncoder
from tests.benchmarks.report_format import print_footer, print_header, print_table
from tests.benchmarks.retrieval_metrics import (
    aggregate_recall_at_k_mrr,
    compute_recall_at_k_and_mrr,
)

ALL_ATTRIBUTES = [
    "programming_language",
    "output_format",
    "communication_style",
    "naming_convention",
    "error_handling",
    "explanation_approach",
    "example_scope",
    "tool_preference",
]
DRIFT_PATTERNS = ["abrupt", "gradual", "noisy"]


# ── Scoring ──────────────────────────────────────────────────────────────────


def _word_present(text_lower: str, token_lower: str) -> bool:
    """Word-boundary match: alnum characters immediately before/after the
    token disqualify it. Plain substring matching (the previous behavior)
    let short/common values like "cli" match inside unrelated words like
    "right-click" -- confirmed on the gui->cli tool_preference scenarios,
    see PROGRESS.md 2026-07-09."""
    pattern = r"(?<![a-z0-9])" + re.escape(token_lower) + r"(?![a-z0-9])"
    return re.search(pattern, text_lower) is not None


def _value_present(text: str, value: str) -> bool:
    text_lower = text.lower()
    value_lower = value.lower()
    if _word_present(text_lower, value_lower):
        return True
    if "_" in value:
        if _word_present(text_lower, value.replace("_", " ").lower()):
            return True
        if _word_present(text_lower, value.replace("_", "").lower()):
            return True
    return False


def score_recall(hypothesis: str, post_val: str, pre_val: str) -> tuple[bool, bool, bool]:
    """Return (detected, stale, no_answer)."""
    detected = _value_present(hypothesis, post_val)
    if detected:
        return True, False, False
    if _value_present(hypothesis, pre_val):
        return False, True, False
    return False, False, True


def _current_value_score(hypothesis: str, post_val: str) -> float:
    """Adapts _value_present's boolean check to the keyword_score_fn shape
    Recall@K/MRR expect: 1.0 if the current (post-drift) value is present."""
    return 1.0 if _value_present(hypothesis, post_val) else 0.0


# ── Consolidation diagnostics (plans/05-consolidation.md Phase 4) ────────────


def _new_consolidation_diag_accumulator() -> dict[str, Any]:
    return {
        "prototypes_processed": 0,
        "schemas_created": 0,
        "schemas_reinforced": 0,
        "schemas_skipped": 0,
        "near_dup_intercepts": 0,
        "verdict_counts": {},
        "confidence_histogram": [],
    }


def _accumulate_consolidation_diag(acc: dict[str, Any], diag: dict[str, Any] | None) -> None:
    """Sum one session_end(consolidate=True) diagnostics dict into a
    scenario- or run-level accumulator built by _new_consolidation_diag_accumulator."""
    if not diag:
        return
    for key in (
        "prototypes_processed",
        "schemas_created",
        "schemas_reinforced",
        "schemas_skipped",
        "near_dup_intercepts",
    ):
        acc[key] += int(diag.get(key, 0) or 0)
    for key in ("verdict_counts",):
        for k, v in (diag.get(key) or {}).items():
            acc[key][k] = acc[key].get(k, 0) + int(v)
    acc["confidence_histogram"].extend(diag.get("confidence_histogram") or [])


# ── Per-scenario runner ───────────────────────────────────────────────────────


@dataclass
class ScenarioResult:
    scenario_id: str
    attribute: str
    drift_pattern: str
    pre_drift_value: str
    post_drift_value: str
    probe_question: str
    expected_answer: str
    hypothesis: str
    detected: bool
    stale: bool
    no_answer: bool
    n_schemas: int
    n_episodes: int
    consolidate: bool
    latency_ingest_s: float
    latency_recall_s: float
    n_sessions_ingested: int
    error: str | None = None
    # Consolidation diagnostics summed across every session_end(consolidate=True)
    # call in this scenario (plans/05-consolidation.md Phase 4, Q1/Q4).
    consolidation_diag: dict = field(default_factory=dict)
    # Schema-only scoring variant (episodes excluded from the hypothesis) --
    # isolates consolidation's contribution from episodic retrieval's.
    # See PROGRESS.md 2026-07-09 "Fix the scorer" entry.
    detected_schema_only: bool = False
    stale_schema_only: bool = False
    no_answer_schema_only: bool = False
    # Recall@K / MRR for the current (post-drift) value only — "stale" vs.
    # "no_answer" aren't distinguished at intermediate K, only detected-or-not.
    recall_at_k: dict = field(default_factory=dict)
    mrr: float = 0.0


def run_scenario(
    scenario: dict[str, Any],
    *,
    consolidate: bool,
    assignment_threshold: float,
    shared_encoder: TextEncoder,
    top_k: int = 10,
    tau_seconds: float = 86400.0,
    salience_weight: float = 0.5,
    surprise_weight: float = 0.3,
    judge_overrides: dict[str, Any] | None = None,
) -> ScenarioResult:
    sid = scenario["scenario_id"]
    attribute = scenario["attribute"]
    pre_val = scenario["pre_drift_value"]
    post_val = scenario["post_drift_value"]
    drift_pattern = scenario["drift_pattern"]
    probe_question = scenario["probe_question"]
    expected = scenario["expected_answer"]
    sessions = scenario["sessions"]

    # Deterministic seed per scenario so consolidation sampling is reproducible.
    import hashlib as _hashlib

    import numpy as _np

    _seed = int.from_bytes(_hashlib.sha256(str(sid).encode("utf-8")).digest()[:4], "big") % (2**31)
    _np.random.seed(_seed)

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    try:
        cfg = SlowaveConfig(
            db_path=db_path,
            dim=shared_encoder.dim,
            encoder=EncoderConfig(),
            salience=SalienceConfig(tau_seconds=tau_seconds, surprise_weight=surprise_weight),
            replay=ReplayConfig(
                assignment_threshold=assignment_threshold,
                sample_size=256,
                max_prototypes_per_replay=32,
                use_multi_scale=True,
            ),
            retrieval=RetrievalConfig(
                salience_weight=salience_weight,
                neighbor_top_k=6,
                use_multi_scale=True,
            ),
            relation=GeometricRelationConfig(**(judge_overrides or {})),
            disable_encoder=False,
        )
        eng = SlowaveEngine(cfg, shared_encoder=shared_encoder)

        # Compute synthetic timestamps: spread non-probe sessions evenly
        # over a 180-day window ending 1 day ago, so salience decay and
        # the temporal recency bonus operate on realistic time separations.
        # Pre-drift sessions are months old; post-drift sessions are recent.
        _now = int(time.time())
        _window_end = _now - 86400  # 1 day ago
        _window_start = _now - 180 * 86400  # 180 days ago
        _non_probe = [s for s in sessions if not s["is_probe"]]
        _n_non_probe = max(1, len(_non_probe))

        def _session_ts(idx: int) -> int:
            frac = idx / (_n_non_probe - 1) if _n_non_probe > 1 else 0.0
            return int(_window_start + frac * (_window_end - _window_start))

        n_ingested = 0
        consolidation_diag = _new_consolidation_diag_accumulator()
        t_ingest_start = time.time()
        for sess_idx, sess in enumerate(_non_probe):
            sess_ts = _session_ts(sess_idx)
            session_id = eng.session_start(agent="stalememory", ts=sess_ts)
            for turn in sess["turns"]:
                role = str(turn.get("role", "user"))
                content = str(turn.get("content", "")).strip()
                if not content:
                    continue
                etype = "user_message" if role == "user" else "assistant_message"
                eng.event_append(session_id=session_id, type=etype, content=content, ts=sess_ts)
            end_stats = eng.session_end(session_id, consolidate=consolidate, ts=sess_ts)
            _accumulate_consolidation_diag(consolidation_diag, end_stats.get("consolidation"))
            n_ingested += 1
        latency_ingest = time.time() - t_ingest_start

        t_recall_start = time.time()
        result = eng.recall(probe_question, top_k=top_k, evidence=False)
        latency_recall = time.time() - t_recall_start

        schemas_text = " ".join(s.content_text for s in result.schemas)
        episodes_text = " ".join(
            ep["content_text"] for ep in result.episode_texts if ep["content_text"]
        )
        hypothesis = " ".join([schemas_text, episodes_text]).strip()

        detected, stale, no_answer = score_recall(hypothesis, post_val, pre_val)
        # Schema-only variant: episodes are immutable and untouched by any
        # Consolidator/judge threshold (see PROGRESS.md 2026-07-09), so the
        # combined metric above can't isolate consolidation's contribution.
        # Scored in parallel, not as a replacement.
        detected_so, stale_so, no_answer_so = score_recall(schemas_text.strip(), post_val, pre_val)
        recall_at_k, mrr = compute_recall_at_k_and_mrr(
            eng,
            probe_question,
            post_val,
            keyword_score_fn=_current_value_score,
            hit_threshold=0.5,
            recall_kwargs={"evidence": False},
        )
        eng.close()

        return ScenarioResult(
            scenario_id=sid,
            attribute=attribute,
            drift_pattern=drift_pattern,
            pre_drift_value=pre_val,
            post_drift_value=post_val,
            probe_question=probe_question,
            expected_answer=expected,
            hypothesis=hypothesis[:400],
            detected=detected,
            stale=stale,
            no_answer=no_answer,
            n_schemas=len(result.schemas),
            n_episodes=len(result.episode_texts),
            consolidate=consolidate,
            latency_ingest_s=round(latency_ingest, 2),
            latency_recall_s=round(latency_recall, 4),
            n_sessions_ingested=n_ingested,
            consolidation_diag=consolidation_diag,
            detected_schema_only=detected_so,
            stale_schema_only=stale_so,
            no_answer_schema_only=no_answer_so,
            recall_at_k=recall_at_k,
            mrr=round(mrr, 4),
        )

    except Exception as e:
        return ScenarioResult(
            scenario_id=sid,
            attribute=attribute,
            drift_pattern=drift_pattern,
            pre_drift_value=pre_val,
            post_drift_value=post_val,
            probe_question=probe_question,
            expected_answer=expected,
            hypothesis="",
            detected=False,
            stale=False,
            no_answer=True,
            n_schemas=0,
            n_episodes=0,
            consolidate=consolidate,
            latency_ingest_s=0.0,
            latency_recall_s=0.0,
            n_sessions_ingested=0,
            error=str(e),
        )
    finally:
        for ext in ("", "-wal", "-shm"):
            p = db_path + ext
            if os.path.exists(p):
                os.remove(p)


# ── Report ────────────────────────────────────────────────────────────────────


def print_report(results: list[ScenarioResult], *, total_elapsed: float = 0.0) -> None:
    print_header(
        "StaleMemory Evaluation Report",
        [
            "Dataset : StaleMemory (implicit belief staleness, 1,200 scenarios)",
            f"Mode    : {'consolidation on (replay + geometric schema extraction, zero LLM by default)' if (results[0].consolidate if results else True) else 'consolidation off'}",
            "Scorer  : keyword value-match (zero LLM calls)",
            f"Total   : {len(results)} scenarios",
        ],
    )
    valid = [r for r in results if not r.error]
    n = len(valid)
    detected = sum(1 for r in valid if r.detected)
    stale = sum(1 for r in valid if r.stale)
    no_ans = sum(1 for r in valid if r.no_answer)
    errors = sum(1 for r in results if r.error)
    print_table(
        ["Metric", "Count", "Rate"],
        [
            ["Detection Rate (post-drift)", str(detected), f"{100 * detected / max(1, n):.1f}%"],
            ["Stale Persistence (anchor pull)", str(stale), f"{100 * stale / max(1, n):.1f}%"],
            ["No Answer (neither found)", str(no_ans), f"{100 * no_ans / max(1, n):.1f}%"],
            ["Errors", str(errors), f"{100 * errors / max(1, len(results)):.1f}%"],
        ],
    )
    print()

    pattern_rows = []
    for pattern in DRIFT_PATTERNS:
        rs = [r for r in valid if r.drift_pattern == pattern]
        if not rs:
            continue
        pn = len(rs)
        d = sum(1 for r in rs if r.detected)
        s = sum(1 for r in rs if r.stale)
        na = sum(1 for r in rs if r.no_answer)
        pattern_rows.append(
            [
                pattern,
                str(pn),
                f"{100 * d / pn:.1f}%",
                f"{100 * s / pn:.1f}%",
                f"{100 * na / pn:.1f}%",
            ]
        )
    print_table(["Drift Pattern", "N", "Detect", "Stale", "NoAns"], pattern_rows)
    print()

    attr_rows = []
    for attr in ALL_ATTRIBUTES:
        rs = [r for r in valid if r.attribute == attr]
        if not rs:
            continue
        an = len(rs)
        d = sum(1 for r in rs if r.detected)
        s = sum(1 for r in rs if r.stale)
        na = sum(1 for r in rs if r.no_answer)
        attr_rows.append(
            [
                attr,
                str(an),
                f"{100 * d / an:.1f}%",
                f"{100 * s / an:.1f}%",
                f"{100 * na / an:.1f}%",
            ]
        )
    print_table(["Attribute", "N", "Detect", "Stale", "NoAns"], attr_rows)
    print()
    print(" Timing")
    print(f"  total:  {total_elapsed:.1f}s  ({len(results)} scenarios)")
    if valid:
        ingests = sorted(r.latency_ingest_s for r in valid)
        recalls = sorted(r.latency_recall_s for r in valid)
        print(
            f"  ingest: sum={sum(ingests):.1f}s  mean={sum(ingests)/len(ingests):.2f}s  "
            f"p50={ingests[len(ingests)//2]:.2f}s  max={ingests[-1]:.2f}s"
        )
        print(
            f"  recall: sum={sum(recalls):.1f}s  mean={sum(recalls)/len(recalls)*1000:.1f}ms  "
            f"p50={recalls[len(recalls)//2]*1000:.1f}ms  max={recalls[-1]*1000:.1f}ms"
        )
    print()
    recall_at_k_pct, mrr = aggregate_recall_at_k_mrr(
        [r.recall_at_k for r in valid],
        [r.mrr for r in valid],
    )
    print(
        " Recall@K (post-drift value): "
        + "  ".join(f"{k}={v}%" for k, v in recall_at_k_pct.items())
    )
    print(f" MRR: {mrr}")
    print_footer()


# ── Payload helpers ───────────────────────────────────────────────────────────


def _result_row(r: ScenarioResult) -> dict[str, Any]:
    return {
        "scenario_id": r.scenario_id,
        "attribute": r.attribute,
        "drift_pattern": r.drift_pattern,
        "pre_drift_value": r.pre_drift_value,
        "post_drift_value": r.post_drift_value,
        "probe_question": r.probe_question,
        "expected_answer": r.expected_answer,
        "hypothesis": r.hypothesis,
        "detected": r.detected,
        "stale": r.stale,
        "no_answer": r.no_answer,
        "n_schemas": r.n_schemas,
        "n_episodes": r.n_episodes,
        "latency_ingest_s": r.latency_ingest_s,
        "latency_recall_s": r.latency_recall_s,
        "n_sessions_ingested": r.n_sessions_ingested,
        "detected_schema_only": r.detected_schema_only,
        "stale_schema_only": r.stale_schema_only,
        "no_answer_schema_only": r.no_answer_schema_only,
        "recall_at_k": r.recall_at_k,
        "mrr": r.mrr,
        "error": r.error,
    }


def _build_payload(
    *,
    results: list[ScenarioResult],
    dataset_path: Path,
    args: argparse.Namespace,
    total_elapsed: float,
    partial: bool,
) -> dict[str, Any]:
    valid = [r for r in results if not r.error]
    n = len(valid)
    by_pattern: dict[str, dict] = {}
    for pat in DRIFT_PATTERNS:
        rs = [r for r in valid if r.drift_pattern == pat]
        if not rs:
            continue
        pn = len(rs)
        by_pattern[pat] = {
            "n": pn,
            "detection_rate": round(sum(1 for r in rs if r.detected) / pn, 4),
            "stale_rate": round(sum(1 for r in rs if r.stale) / pn, 4),
            "no_answer_rate": round(sum(1 for r in rs if r.no_answer) / pn, 4),
        }
    by_attribute: dict[str, dict] = {}
    by_attribute_schema_only: dict[str, dict] = {}
    for attr in ALL_ATTRIBUTES:
        rs = [r for r in valid if r.attribute == attr]
        if not rs:
            continue
        an = len(rs)
        by_attribute[attr] = {
            "n": an,
            "detection_rate": round(sum(1 for r in rs if r.detected) / an, 4),
            "stale_rate": round(sum(1 for r in rs if r.stale) / an, 4),
            "no_answer_rate": round(sum(1 for r in rs if r.no_answer) / an, 4),
        }
        by_attribute_schema_only[attr] = {
            "n": an,
            "detection_rate": round(sum(1 for r in rs if r.detected_schema_only) / an, 4),
            "stale_rate": round(sum(1 for r in rs if r.stale_schema_only) / an, 4),
            "no_answer_rate": round(sum(1 for r in rs if r.no_answer_schema_only) / an, 4),
        }
    detected = sum(1 for r in valid if r.detected)
    stale_c = sum(1 for r in valid if r.stale)
    no_ans = sum(1 for r in valid if r.no_answer)
    detected_so = sum(1 for r in valid if r.detected_schema_only)
    stale_so = sum(1 for r in valid if r.stale_schema_only)
    no_ans_so = sum(1 for r in valid if r.no_answer_schema_only)
    recall_at_k_pct, mrr = aggregate_recall_at_k_mrr(
        [r.recall_at_k for r in valid],
        [r.mrr for r in valid],
    )

    # Split consolidation diagnostics by outcome: if near_dup_intercepts is
    # markedly higher among stale (anchor-pull) scenarios than detected ones,
    # that's direct evidence for the Priority Finding in plans/05-consolidation.md
    # (Q1) — the near-dup guard is absorbing post-drift updates as reinforcement
    # of the pre-drift schema before the geometric judge ever sees them.
    diag_all = _new_consolidation_diag_accumulator()
    diag_stale = _new_consolidation_diag_accumulator()
    diag_detected = _new_consolidation_diag_accumulator()
    for r in valid:
        _accumulate_consolidation_diag(diag_all, r.consolidation_diag)
        if r.stale:
            _accumulate_consolidation_diag(diag_stale, r.consolidation_diag)
        elif r.detected:
            _accumulate_consolidation_diag(diag_detected, r.consolidation_diag)

    return {
        "meta": {
            "benchmark": "StaleMemory",
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "partial": partial,
            "dataset": str(dataset_path),
            "limit": args.limit,
            "attributes": args.attributes,
            "drift_patterns": args.drift_patterns,
            "consolidate": not args.no_consolidate,
            "assignment_threshold": args.assignment_threshold,
            "judge_overrides": args.judge_overrides,
            "top_k": args.top_k,
            "total_elapsed_s": round(total_elapsed, 2),
            "llm_calls": 0,
        },
        "summary": {
            "n": len(results),
            "n_valid": n,
            "detection_rate": round(detected / max(1, n), 4),
            "stale_rate": round(stale_c / max(1, n), 4),
            "no_answer_rate": round(no_ans / max(1, n), 4),
            "recall_at_k": recall_at_k_pct,
            "mrr": mrr,
            "by_drift_pattern": by_pattern,
            "by_attribute": by_attribute,
            # Schema-only variant (episodes excluded from the hypothesis) --
            # isolates consolidation's contribution. See PROGRESS.md 2026-07-09.
            "schema_only": {
                "detection_rate": round(detected_so / max(1, n), 4),
                "stale_rate": round(stale_so / max(1, n), 4),
                "no_answer_rate": round(no_ans_so / max(1, n), 4),
                "by_attribute": by_attribute_schema_only,
            },
        },
        "diagnostics": {
            "consolidation": {
                "all": diag_all,
                "stale_scenarios": diag_stale,
                "detected_scenarios": diag_detected,
            },
        },
        "results": [_result_row(r) for r in results],
    }


def _write_payload(out_path: Path, payload: dict[str, Any]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, out_path)


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="StaleMemory evaluation harness for Slowave")
    parser.add_argument("--dataset", default="data/stalememory/scenarios.jsonl")
    parser.add_argument("--attributes", nargs="+", default=ALL_ATTRIBUTES)
    parser.add_argument(
        "--drift-patterns", nargs="+", default=DRIFT_PATTERNS, dest="drift_patterns"
    )
    parser.add_argument(
        "--limit", type=int, default=0, help="Max scenarios per attribute×pattern (0=all)"
    )
    parser.add_argument("--no-consolidate", action="store_true")
    parser.add_argument("--assignment-threshold", type=float, default=0.85)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--tau-seconds", type=float, default=86400.0)
    parser.add_argument("--salience-weight", type=float, default=0.5)
    parser.add_argument("--surprise-weight", type=float, default=0.3)
    parser.add_argument(
        "--judge-overrides",
        default="",
        help="JSON dict of GeometricRelationConfig field overrides "
        "(plans/05-consolidation.md Threshold Ablation Matrix), e.g. "
        "'{\"near_dup_guard_cosine\": 1.01}'.",
    )
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    judge_overrides = json.loads(args.judge_overrides) if args.judge_overrides else {}

    dataset_path = Path(args.dataset)
    if not dataset_path.is_absolute():
        dataset_path = REPO_ROOT / dataset_path

    print(f"Loading dataset: {dataset_path}")
    scenarios: list[dict] = []
    with open(dataset_path) as f:
        for line in f:
            line = line.strip()
            if line:
                scenarios.append(json.loads(line))
    print(f"Total in file: {len(scenarios)}")

    selected: list[dict] = []
    combo_count: dict[tuple[str, str], int] = {}
    for s in scenarios:
        attr = s["attribute"]
        pattern = s["drift_pattern"]
        if attr not in args.attributes or pattern not in args.drift_patterns:
            continue
        key = (attr, pattern)
        combo_count[key] = combo_count.get(key, 0)
        if args.limit > 0 and combo_count[key] >= args.limit:
            continue
        selected.append(s)
        combo_count[key] += 1

    print(f"Selected: {len(selected)} scenarios")
    if args.limit:
        print(f"(capped at {args.limit} per attribute×pattern)")
    print(
        f"consolidate={not args.no_consolidate}  threshold={args.assignment_threshold}  top_k={args.top_k}"
    )

    print("Loading encoder (paraphrase-multilingual-MiniLM-L12-v2)...", end=" ", flush=True)
    enc_cfg = EncoderConfig()
    shared_enc = TextEncoder(enc_cfg)
    _ = shared_enc.dim
    print(f"OK (dim={shared_enc.dim})")
    print()

    if args.out:
        out_path = Path(args.out)
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mode_tag = "with_consolidation" if not args.no_consolidate else "no_consolidation"
        out_path = Path(f"data/stalememory/runs/{stamp}_{mode_tag}.json")
    if not out_path.is_absolute():
        out_path = REPO_ROOT / out_path

    results: list[ScenarioResult] = []
    t_start = time.time()
    last_print = t_start
    print_every = max(1, len(selected) // 20)  # ~20 updates over a full run
    try:
        for i, scenario in enumerate(selected):
            r = run_scenario(
                scenario,
                consolidate=not args.no_consolidate,
                assignment_threshold=args.assignment_threshold,
                shared_encoder=shared_enc,
                top_k=args.top_k,
                tau_seconds=args.tau_seconds,
                salience_weight=args.salience_weight,
                surprise_weight=args.surprise_weight,
                judge_overrides=judge_overrides,
            )
            results.append(r)

            if r.error:
                print(
                    f"[{i+1:>4}/{len(selected)}] {r.attribute:<25} {r.drift_pattern:<8} ERROR: {r.error[:60]}",
                    flush=True,
                )

            done = i + 1
            now = time.time()
            if (
                r.error
                or done == len(selected)
                or done % print_every == 0
                or now - last_print >= 15.0
            ):
                detected = sum(1 for x in results if x.detected)
                stale = sum(1 for x in results if x.stale)
                elapsed = now - t_start
                rate = done / elapsed if elapsed > 0 else 0.0
                eta_s = (len(selected) - done) / rate if rate > 0 else None
                eta = f"{eta_s:.0f}s" if eta_s is not None else "?"
                print(
                    f"[{done:>4}/{len(selected)}] ({100 * done / len(selected):>3.0f}%)  "
                    f"detect={detected}/{done} ({100 * detected / done:.1f}%)  stale={stale}/{done}  "
                    f"elapsed={elapsed:.0f}s  rate={rate:.2f}/s  eta={eta}",
                    flush=True,
                )
                last_print = now
            _write_payload(
                out_path,
                _build_payload(
                    results=results,
                    dataset_path=dataset_path,
                    args=args,
                    total_elapsed=time.time() - t_start,
                    partial=True,
                ),
            )
    except KeyboardInterrupt:
        _write_payload(
            out_path,
            _build_payload(
                results=results,
                dataset_path=dataset_path,
                args=args,
                total_elapsed=time.time() - t_start,
                partial=True,
            ),
        )
        print(f"\nInterrupted. Partial results saved to: {out_path}")
        raise

    total_elapsed = time.time() - t_start
    print(f"\nCompleted {len(results)} scenarios in {total_elapsed:.1f}s")
    print_report(results, total_elapsed=total_elapsed)
    payload = _build_payload(
        results=results,
        dataset_path=dataset_path,
        args=args,
        total_elapsed=total_elapsed,
        partial=False,
    )
    _write_payload(out_path, payload)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
