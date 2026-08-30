#!/usr/bin/env python3
"""LongMemEval evaluation harness for Slowave.

Runs Slowave against the LongMemEval benchmark and reports per-category
precision. Uses a fresh SQLite DB per question (required: memories must not
leak across questions).

Usage:
  python tests/benchmarks/longmemeval_eval.py \\
      --dataset data/longmemeval/longmemeval_oracle.json \\
      --categories knowledge-update single-session-preference multi-session \\
      --limit 50 \\
      --out data/longmemeval/results_oracle.json

With no --limit it runs all questions in the selected categories.
With --no-consolidate it skips consolidation (fast smoke check, embedding recall only).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Suppress noisy logs from sentence-transformers and other libraries
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

# Ensure repo root is on path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from slowave.core.config import SlowaveConfig
from slowave.core.engine import SlowaveEngine
from slowave.latent.replay_engine import ReplayConfig
from slowave.latent.retrieval import RetrievalConfig
from slowave.latent.salience import SalienceConfig
from slowave.symbolic.encoder import EncoderConfig, TextEncoder
from tests.benchmarks.evidence_format import format_retrieved_evidence
from tests.benchmarks.llm_judge import (
    confirm_paid_run,
    estimate_cost_usd,
    get_openai_client,
    judge_batch_concurrent,
)
from tests.benchmarks.report_format import print_footer, print_header, print_table
from tests.benchmarks.retrieval_metrics import (
    aggregate_recall_at_k_mrr,
    compute_recall_at_k_and_mrr,
)

# ---- categories of interest ----

ALL_CATEGORIES = [
    "knowledge-update",
    "single-session-preference",
    "multi-session",
    "single-session-user",
    "single-session-assistant",
    "temporal-reasoning",
]

# ---- scoring ----


def keyword_score(hypothesis: str, answer: str) -> float:
    """Simple keyword overlap score in [0, 1].

    Splits the expected answer into content tokens and checks what fraction
    appear in the hypothesis. Works well for factual answers.
    Avoids the GPT-4o judge cost for iteration purposes.
    """
    import re

    stop = {
        "the",
        "a",
        "an",
        "is",
        "was",
        "were",
        "are",
        "i",
        "my",
        "me",
        "it",
        "its",
        "of",
        "in",
        "on",
        "at",
        "to",
        "for",
        "and",
        "or",
        "that",
        "this",
        "with",
        "be",
        "have",
        "has",
        "had",
    }

    def tokens(s: str) -> set[str]:
        return {
            w
            for w in re.findall(r"[a-z0-9]+", s.lower())
            if w not in stop and (len(w) > 1 or w.isdigit())
        }

    answer_tokens = tokens(answer)
    if not answer_tokens:
        return 0.0
    hyp_tokens = tokens(hypothesis)
    matched = answer_tokens & hyp_tokens
    return len(matched) / len(answer_tokens)


HIT_THRESHOLD = 0.5  # fraction of answer keywords that must appear in recall


def _parse_longmemeval_ts(value: str) -> int:
    """Parse LongMemEval's ``YYYY/MM/DD (Day) HH:MM`` session timestamps."""
    cleaned = re.sub(r"\s*\([^)]*\)\s*", " ", str(value)).strip()
    parsed = datetime.strptime(cleaned, "%Y/%m/%d %H:%M").replace(tzinfo=timezone.utc)
    return int(parsed.timestamp())


# ---- per-question runner ----


@dataclass
class QuestionResult:
    question_id: str
    question_type: str
    question: str
    expected_answer: str
    hypothesis: str
    keyword_score: float
    hit: bool
    n_schemas: int
    n_episodes: int
    consolidate: bool
    latency_ingest_s: float
    latency_recall_s: float
    recall_at_k: dict[str, bool] = field(default_factory=dict)
    mrr: float = 0.0
    # Stage-comparison instrumentation (zero in latent / no-LLM mode).
    # n_llm_calls and llm_tokens_* measure the actual cost of forming
    # the memory for this question. db_size_bytes measures the storage
    # footprint after consolidation.
    n_llm_calls: int = 0
    llm_prompt_tokens: int = 0
    llm_completion_tokens: int = 0
    db_size_bytes: int = 0
    error: str | None = None
    component_scores: dict[str, Any] = field(default_factory=dict)
    debug: dict[str, Any] = field(default_factory=dict)
    # Temporal anchor diagnostics (plans/07-temporal.md Phase 4).
    anchor_fired: bool = False
    anchor_displacement_s: int = 0
    # Optional LLM-judge pass (Phase 3.5, --judge-model flag) — None unless requested.
    llm_judge_score: float | None = None
    llm_judge_reason: str = ""
    llm_judge_parse_ok: bool = True


def run_question(
    question: dict[str, Any],
    *,
    consolidate: bool,
    assignment_threshold: float,
    shared_encoder: TextEncoder,
    recall_mode: str = "hybrid",
    top_k: int = 5,
    debug: bool = False,
    keep_debug_dbs: bool = False,
    # ablation flags
    no_salience_rerank: bool = False,
    no_graph_expansion: bool = False,
    no_temporal: bool = False,
    replay_only: bool = False,
    no_multi_scale: bool = False,
    tau_seconds: float = 86400.0,
    salience_weight: float = 0.5,
    surprise_weight: float = 0.3,
    judge_model: str | None = None,
    openai_client: Any = None,
    save_full_hypothesis: bool = False,
) -> tuple[QuestionResult, tuple[str, str, str] | None]:
    """Returns (result, pending_judge_job). `pending_judge_job` is
    (question, hypothesis, expected_answer) if judge_model was set, to be
    judged in main()'s concurrent batch pass — else None."""
    qid = str(question["question_id"])
    qtype = str(question["question_type"])
    qtext = str(question["question"])
    expected = str(question["answer"])
    sessions = question["haystack_sessions"]  # list of lists of {role, content}
    session_dates = question.get("haystack_dates", [])
    if len(session_dates) != len(sessions):
        raise ValueError(
            f"haystack_dates/session mismatch for {qid}: "
            f"{len(session_dates)} dates for {len(sessions)} sessions"
        )

    # Deterministic seed per question so consolidation sampling is reproducible.
    import hashlib as _hashlib

    import numpy as _np

    _seed = int.from_bytes(_hashlib.sha256(qid.encode("utf-8")).digest()[:4], "big") % (2**31)
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
                use_multi_scale=not no_multi_scale,
            ),
            retrieval=RetrievalConfig(
                salience_weight=0.0 if no_salience_rerank else salience_weight,
                neighbor_top_k=0 if no_graph_expansion else 6,
                use_multi_scale=not no_multi_scale,
                use_temporal=not no_temporal,
                temporal_weight=0.0 if no_temporal else 0.25,
            ),
            disable_encoder=False,
        )
        eng = SlowaveEngine(cfg, shared_encoder=shared_encoder)

        t_ingest_start = time.time()
        for i, (session_turns, session_date) in enumerate(zip(sessions, session_dates)):
            session_ts = _parse_longmemeval_ts(str(session_date))
            sid = eng.session_start(agent="longmemeval")
            conn = eng.db.connect()
            conn.execute("UPDATE sessions SET started_ts=? WHERE id=?", (session_ts, sid))
            conn.commit()
            # Keep only the first 10 turns per session to bound ingest time.
            # The evidence turns that contain the answer are typically early.
            #
            # No per-turn char cap. Root cause analysis (2026-06-06) showed
            # the original 500-char cap was the dominant cause of the
            # single-session-assistant regression (92.9% → 66.1%): long
            # assistant responses in LongMemEval frequently contain the answer
            # beyond char 500 (p50 turn length = 546, p90 = 2821, p99 = 3568;
            # 50.6% of turns exceed 500 chars). The cap was introduced as a
            # performance bound for LLM-mode ingest, but LLM ingest was
            # removed in commit a330abe. Without LLM, there is no context-
            # window pressure: the harness uses pure embedding ingest which
            # scales linearly with token count but has no hard limit.
            # V4 (no cap): 78.0% total vs 61.0% with cap=500.
            for turn in session_turns[:10]:
                role = str(turn.get("role", "user"))
                content = str(turn.get("content", "")).strip()
                if not content:
                    continue
                etype = "user_message" if role == "user" else "assistant_message"
                embedding = shared_encoder.encode(content)
                eng.raw_log.append(
                    session_id=sid,
                    ts=session_ts,
                    type=etype,
                    content=content,
                    embedding=embedding,
                )
            eng.session_end(sid, consolidate=False)
            conn = eng.db.connect()
            conn.execute(
                "UPDATE sessions SET started_ts=?, ended_ts=? WHERE id=?",
                (session_ts, session_ts, sid),
            )
            conn.execute(
                "UPDATE episodic_memories SET ts=?, last_salience_ts=? "
                "WHERE event_id LIKE ? OR event_id LIKE ?",
                (session_ts, session_ts, f"micro_{sid}_%", f"macro_{sid}"),
            )
            conn.commit()
        if consolidate:
            eng.consolidate_once(triggered_by="longmemeval_eval")
        latency_ingest = time.time() - t_ingest_start

        # Stage 3/5 latent-only mode: run replay to train the transition model,
        # then self-supervise to reinforce the prototype graph based on its own
        # retrieval failures. Without LLM consolidation.
        if replay_only and not consolidate:
            eng.replay_engine.replay_once()
            eng.replay_engine.self_supervise()

        t_recall_start = time.time()
        result = eng.recall(qtext, top_k=top_k, evidence=False)
        latency_recall = time.time() - t_recall_start

        # Instrumentation snapshot — taken BEFORE eng.close() (the close
        # paths below close the DB and we cannot stat it after that).
        # ``_counting_llm`` only exists when the engine built an LLM
        # backend; in latent / no-LLM modes it is absent and all
        # counters stay at 0.
        counting = getattr(eng, "_counting_llm", None)
        if counting is not None:
            usage_snap = counting.snapshot()
        else:
            usage_snap = {"n_calls": 0, "prompt_tokens": 0, "completion_tokens": 0}
        # Sum the SQLite main file plus its WAL/SHM sidecars — in WAL
        # mode the bulk of recent writes lives in the -wal file until
        # checkpoint, so reading just the main file under-reports.
        db_size_bytes = 0
        for ext in ("", "-wal", "-shm"):
            try:
                db_size_bytes += int(os.path.getsize(db_path + ext))
            except OSError:
                pass

        schemas_hypothesis = " ".join(s.content_text for s in result.schemas)
        episodes_hypothesis = " ".join(
            ep["content_text"] for ep in result.episode_texts if ep["content_text"]
        )
        if recall_mode == "schemas":
            hypothesis = schemas_hypothesis
        elif recall_mode == "episodes":
            hypothesis = episodes_hypothesis
        else:
            hypothesis = " ".join([schemas_hypothesis, episodes_hypothesis]).strip()
        structured_hypothesis = format_retrieved_evidence(
            result,
            include_schemas=recall_mode in {"schemas", "hybrid"},
            include_episodes=recall_mode in {"episodes", "hybrid"},
            episodes_first=recall_mode == "hybrid",
        )

        component_scores = {
            "schemas": round(keyword_score(schemas_hypothesis, expected), 3),
            "episodes": round(keyword_score(episodes_hypothesis, expected), 3),
            "hybrid": round(
                keyword_score(" ".join([schemas_hypothesis, episodes_hypothesis]), expected),
                3,
            ),
            "recall_mode": recall_mode,
        }
        ks = float(component_scores[recall_mode])
        hit = ks >= HIT_THRESHOLD

        # Always the hybrid (schemas+episodes) definition, regardless of
        # --recall-mode, to stay consistent with the other benchmarks.
        recall_at_k, mrr = compute_recall_at_k_and_mrr(
            eng,
            qtext,
            expected,
            keyword_score_fn=keyword_score,
            hit_threshold=HIT_THRESHOLD,
            recall_kwargs={"evidence": False},
        )

        # Judging is deferred to main()'s per-batch concurrent pass — it's a
        # pure network call over already-computed text, never touches the
        # engine, so running it inline here would serialize every question's
        # judge call behind the previous one for no reason.
        pending_judge_job = (qtext, hypothesis, expected) if judge_model else None

        debug_payload = _build_debug_payload(eng, result, expected) if debug else {}
        if keep_debug_dbs and (debug or not hit):
            debug_dir = REPO_ROOT / "data" / "longmemeval" / "debug_dbs"
            debug_dir.mkdir(parents=True, exist_ok=True)
            dest = debug_dir / f"{qid}_{qtype}_{'hit' if hit else 'miss'}.db"
            eng.close()
            for ext in ("", "-wal", "-shm"):
                src = Path(db_path + ext)
                if src.exists():
                    shutil.copy2(src, Path(str(dest) + ext))
            debug_payload["debug_db"] = str(dest)
        else:
            eng.close()

        return (
            QuestionResult(
                question_id=qid,
                question_type=qtype,
                question=qtext,
                expected_answer=expected,
                hypothesis=structured_hypothesis if save_full_hypothesis else hypothesis[:400],
                keyword_score=round(ks, 3),
                hit=hit,
                n_schemas=len(result.schemas),
                n_episodes=len(result.episode_texts),
                consolidate=consolidate,
                latency_ingest_s=round(latency_ingest, 2),
                latency_recall_s=round(latency_recall, 3),
                recall_at_k=recall_at_k,
                mrr=round(mrr, 4),
                n_llm_calls=int(usage_snap["n_calls"]),
                llm_prompt_tokens=int(usage_snap["prompt_tokens"]),
                llm_completion_tokens=int(usage_snap["completion_tokens"]),
                db_size_bytes=int(db_size_bytes),
                component_scores=component_scores,
                debug=debug_payload,
                anchor_fired=result.anchor_fired,
                anchor_displacement_s=result.anchor_displacement_s,
            ),
            pending_judge_job,
        )

    except Exception as e:
        return (
            QuestionResult(
                question_id=qid,
                question_type=qtype,
                question=qtext,
                expected_answer=expected,
                hypothesis="",
                keyword_score=0.0,
                hit=False,
                n_schemas=0,
                n_episodes=0,
                consolidate=consolidate,
                latency_ingest_s=0.0,
                latency_recall_s=0.0,
                error=str(e),
            ),
            None,
        )
    finally:
        for ext in ("", "-wal", "-shm"):
            p = db_path + ext
            if os.path.exists(p):
                os.remove(p)


def _answer_presence(text: str, expected: str) -> bool:
    return keyword_score(text, expected) >= HIT_THRESHOLD


def _schema_dict(s: Any) -> dict[str, Any]:
    return {
        "id": s.id,
        "prototype_id": s.prototype_id,
        "content_text": s.content_text,
        "facets": s.facets,
        "tags": s.tags,
        "scope_id": s.scope_id,
        "status": s.status,
        "confidence": s.confidence,
        "salience": s.salience,
        "supporting_episode_ids": s.supporting_episode_ids,
        "needs_review": s.is_labile,
        "first_formed_ts": s.first_formed_ts,
        "last_updated_ts": s.last_updated_ts,
    }


def _build_debug_payload(eng: SlowaveEngine, result: Any, expected: str) -> dict[str, Any]:
    conn = eng.db.connect()
    all_schema_rows = eng.schemas.list(limit=1000)
    all_schemas = [_schema_dict(s) for s in all_schema_rows]
    retrieved_schemas = [_schema_dict(s) for s in result.schemas]

    episode_rows = conn.execute(
        "SELECT em.id, em.salience, em.recalled_count, em.metadata_json, et.content_text, et.event_ids, et.session_id "
        "FROM episodic_memories em LEFT JOIN episode_text et ON et.episode_id = em.id "
        "ORDER BY em.id LIMIT 1000"
    ).fetchall()
    all_episodes = []
    for r in episode_rows:
        metadata = json.loads(r["metadata_json"] or "{}")
        all_episodes.append(
            {
                "id": int(r["id"]),
                "kind": metadata.get("kind"),
                "salience": float(r["salience"]),
                "recalled_count": int(r["recalled_count"]),
                "prediction_error": metadata.get("prediction_error"),
                "session_id": r["session_id"],
                "content_text": r["content_text"] or "",
                "event_ids": json.loads(r["event_ids"] or '{"ids":[]}').get("ids", []),
            }
        )

    relations = [
        {
            "src_schema_id": int(r["src_schema_id"]),
            "dst_schema_id": int(r["dst_schema_id"]),
            "relation": str(r["relation"]),
            "confidence": float(r["confidence"]),
            "reason": None if r["reason"] is None else str(r["reason"]),
        }
        for r in conn.execute(
            "SELECT * FROM schema_relations ORDER BY created_ts, src_schema_id"
        ).fetchall()
    ]
    consolidation_debug = []
    try:
        rows = conn.execute("SELECT * FROM consolidation_debug ORDER BY id LIMIT 200").fetchall()
        for r in rows:
            consolidation_debug.append(
                {
                    "id": int(r["id"]),
                    "prototype_id": None if r["prototype_id"] is None else int(r["prototype_id"]),
                    "episode_ids": json.loads(r["episode_ids"] or '{"ids":[]}').get("ids", []),
                    "prompt_text": str(r["prompt_text"]),
                    "response_json": json.loads(r["response_json"] or "{}"),
                    "extracted_claims": json.loads(
                        r["extracted_claims_json"] or '{"claims":[]}'
                    ).get("claims", []),
                    "created_schema_ids": json.loads(r["created_schema_ids"] or '{"ids":[]}').get(
                        "ids", []
                    ),
                    "ts": int(r["ts"]),
                }
            )
    except Exception:
        consolidation_debug = []

    retrieved_episodes = result.episode_texts
    retrieved_schema_text = " ".join(s["content_text"] for s in retrieved_schemas)
    retrieved_episode_text = " ".join(ep.get("content_text", "") for ep in retrieved_episodes)
    all_schema_text = " ".join(s["content_text"] for s in all_schemas)
    all_episode_text = " ".join(ep["content_text"] for ep in all_episodes)

    return {
        "retrieved_schemas": retrieved_schemas,
        "retrieved_episodes": retrieved_episodes,
        "all_schemas": all_schemas,
        "all_episodes": all_episodes,
        "schema_relations": relations,
        "consolidation_debug": consolidation_debug,
        "answer_presence": {
            "in_retrieved_schemas": _answer_presence(retrieved_schema_text, expected),
            "in_retrieved_episodes": _answer_presence(retrieved_episode_text, expected),
            "in_all_schemas": _answer_presence(all_schema_text, expected),
            "in_all_episodes": _answer_presence(all_episode_text, expected),
        },
        "answer_scores": {
            "retrieved_schemas": round(keyword_score(retrieved_schema_text, expected), 3),
            "retrieved_episodes": round(keyword_score(retrieved_episode_text, expected), 3),
            "all_schemas": round(keyword_score(all_schema_text, expected), 3),
            "all_episodes": round(keyword_score(all_episode_text, expected), 3),
        },
    }


# ---- report ----


def print_report(
    results: list[QuestionResult],
    consolidate: bool,
    *,
    judge_model: str | None = None,
    total_elapsed: float = 0.0,
    judge_elapsed: float = 0.0,
) -> None:
    print_header(
        "LongMemEval Evaluation Report",
        [
            "Dataset : LongMemEval Oracle",
            f"Mode    : {'consolidation on (replay + geometric schema extraction, zero LLM by default)' if consolidate else 'consolidation off (recall only)'}",
            f"Scorer  : keyword overlap (threshold={HIT_THRESHOLD})"
            + (f" + LLM-judge ({judge_model})" if judge_model else ""),
            f"Total   : {len(results)} questions",
        ],
    )

    cats: dict[str, list[QuestionResult]] = {}
    for r in results:
        cats.setdefault(r.question_type, []).append(r)
    has_judge = any(r.llm_judge_score is not None for r in results)

    headers = ["Category", "N", "Hit%"]
    if has_judge:
        headers.append("Judge%")
    table_rows = []
    total_hits = 0
    total_n = 0
    all_judge_rs: list[QuestionResult] = []
    for cat, rs in sorted(cats.items()):
        hits = sum(1 for r in rs if r.hit)
        pct = 100.0 * hits / max(1, len(rs))
        total_hits += hits
        total_n += len(rs)
        judge_rs = [r for r in rs if r.llm_judge_score is not None]
        all_judge_rs.extend(judge_rs)
        j_pct = 100 * sum(r.llm_judge_score for r in judge_rs) / len(judge_rs) if judge_rs else None

        row = [cat, str(len(rs)), f"{pct:.1f}%"]
        if has_judge:
            row.append(f"{j_pct:.1f}%" if j_pct is not None else "n/a")
        table_rows.append(row)

    overall = 100.0 * total_hits / max(1, total_n)
    total_row = ["TOTAL", str(total_n), f"{overall:.1f}%"]
    if has_judge:
        j_overall = 100 * sum(r.llm_judge_score for r in all_judge_rs) / max(1, len(all_judge_rs))
        total_row.append(f"{j_overall:.1f}%")

    print_table(headers, table_rows, total_row=total_row)

    recall_at_k_pct, mrr = aggregate_recall_at_k_mrr(
        [r.recall_at_k for r in results if not r.error],
        [r.mrr for r in results if not r.error],
    )
    print()
    print(" Recall@K: " + "  ".join(f"{k}={v}%" for k, v in recall_at_k_pct.items()))
    print(f" MRR     : {mrr}")
    judge_rows = [r for r in results if not r.error and r.llm_judge_score is not None]
    if judge_rows:
        judge_errs = sum(1 for r in judge_rows if not r.llm_judge_parse_ok)
        print(f" Judge parse errors: {judge_errs}/{len(judge_rows)}")

    # Timing — total wall-clock plus the phases that make it up, so it's
    # obvious where the time actually went instead of one opaque number.
    print()
    print(" Timing")
    all_ingest = [r.latency_ingest_s for r in results if not r.error]
    all_recall = [r.latency_recall_s for r in results if not r.error]
    print(f"  total:  {total_elapsed:.1f}s  ({len(results)} questions)")
    if judge_model:
        other = total_elapsed - judge_elapsed
        print(f"  ingest+recall phase: {other:.1f}s   judge phase: {judge_elapsed:.1f}s")
    if all_ingest:
        all_ingest_s = sorted(all_ingest)
        all_recall_s = sorted(all_recall)
        print(
            f"  ingest: sum={sum(all_ingest):.1f}s  mean={sum(all_ingest) / len(all_ingest):.2f}s  "
            f"p50={all_ingest_s[len(all_ingest_s) // 2]:.2f}s  "
            f"p95={all_ingest_s[int(0.95 * (len(all_ingest_s) - 1))]:.2f}s  "
            f"max={all_ingest_s[-1]:.2f}s"
        )
        print(
            f"  recall: sum={sum(all_recall):.1f}s  mean={sum(all_recall) / len(all_recall) * 1000:.1f}ms  "
            f"p50={all_recall_s[len(all_recall_s) // 2] * 1000:.1f}ms  "
            f"max={all_recall_s[-1] * 1000:.1f}ms"
        )

    print()
    print(" Cost summary")
    valid = [r for r in results if not r.error]
    if valid:
        sizes_mb = [r.db_size_bytes / (1024 * 1024) for r in valid]
        print(f"  db size/q: mean={sum(sizes_mb) / len(sizes_mb):.2f}MB  max={max(sizes_mb):.2f}MB")

        calls = [r.n_llm_calls for r in valid]
        if sum(calls) > 0:
            prompt = [r.llm_prompt_tokens for r in valid]
            compl = [r.llm_completion_tokens for r in valid]
            total = [p + c for p, c in zip(prompt, compl)]
            print(
                f"  consolidation llm_calls/q: mean={sum(calls) / len(calls):.1f}  max={max(calls)}  total={sum(calls)}"
            )
            print(
                f"  consolidation tokens/q:     mean={sum(total) / len(total):.0f}  total={sum(total)}"
            )
        else:
            print("  consolidation: zero LLM calls (brain-only, geometric schema extraction)")
    if judge_model:
        judge_pt = sum(r.debug.get("llm_judge_tokens", {}).get("prompt", 0) for r in valid)
        judge_ct = sum(r.debug.get("llm_judge_tokens", {}).get("completion", 0) for r in valid)
        judge_cost = estimate_cost_usd(judge_model, judge_pt, judge_ct)
        cost_s = f"${judge_cost:.2f}" if judge_cost is not None else "unknown pricing"
        print(f"  judge ({judge_model}): prompt={judge_pt}  completion={judge_ct}  cost={cost_s}")

    print_footer()


def _result_row(r: QuestionResult) -> dict[str, Any]:
    return {
        "question_id": r.question_id,
        "question_type": r.question_type,
        "question": r.question,
        "expected_answer": r.expected_answer,
        "hypothesis": r.hypothesis,
        "keyword_score": r.keyword_score,
        "hit": r.hit,
        "n_schemas": r.n_schemas,
        "n_episodes": r.n_episodes,
        "latency_ingest_s": r.latency_ingest_s,
        "latency_recall_s": r.latency_recall_s,
        "recall_at_k": r.recall_at_k,
        "mrr": r.mrr,
        "llm_judge_score": r.llm_judge_score,
        "llm_judge_reason": r.llm_judge_reason,
        "llm_judge_parse_ok": r.llm_judge_parse_ok,
        "n_llm_calls": r.n_llm_calls,
        "llm_prompt_tokens": r.llm_prompt_tokens,
        "llm_completion_tokens": r.llm_completion_tokens,
        "db_size_bytes": r.db_size_bytes,
        "component_scores": r.component_scores,
        "debug": r.debug,
        "anchor_fired": r.anchor_fired,
        "anchor_displacement_s": r.anchor_displacement_s,
        "error": r.error,
    }


def _build_payload(
    *,
    results: list[QuestionResult],
    dataset_path: Path,
    args: argparse.Namespace,
    total_elapsed: float,
    partial: bool,
) -> dict[str, Any]:
    by_cat: dict[str, dict[str, Any]] = {}
    anchor_by_cat: dict[str, dict[str, Any]] = {}
    for cat in sorted({r.question_type for r in results}):
        rs = [r for r in results if r.question_type == cat]
        hits = sum(1 for r in rs if r.hit)
        cat_recall_at_k, cat_mrr = aggregate_recall_at_k_mrr(
            [r.recall_at_k for r in rs if not r.error],
            [r.mrr for r in rs if not r.error],
        )
        cat_judge_rows = [r for r in rs if not r.error and r.llm_judge_score is not None]
        by_cat[cat] = {
            "n": len(rs),
            "hits": hits,
            "score_pct": round(100.0 * hits / max(1, len(rs)), 2),
            "avg_keyword_score": round(sum(r.keyword_score for r in rs) / max(1, len(rs)), 4),
            "errors": sum(1 for r in rs if r.error),
            "recall_at_k": cat_recall_at_k,
            "mrr": cat_mrr,
            "llm_judge_score_pct": (
                round(
                    100 * sum(r.llm_judge_score for r in cat_judge_rows) / len(cat_judge_rows),
                    2,
                )
                if cat_judge_rows
                else None
            ),
        }
        fired = [r for r in rs if r.anchor_fired]
        anchor_by_cat[cat] = {
            "n": len(rs),
            "anchor_fired_n": len(fired),
            "anchor_fired_rate": round(len(fired) / max(1, len(rs)), 4),
            "mean_displacement_s": (
                round(sum(r.anchor_displacement_s for r in fired) / max(1, len(fired)), 1)
                if fired
                else 0.0
            ),
        }
    total_hits = sum(1 for r in results if r.hit)
    total_recall_at_k, total_mrr = aggregate_recall_at_k_mrr(
        [r.recall_at_k for r in results if not r.error],
        [r.mrr for r in results if not r.error],
    )
    judge_rows = [r for r in results if not r.error and r.llm_judge_score is not None]
    total_judge_score_pct = (
        round(100 * sum(r.llm_judge_score for r in judge_rows) / len(judge_rows), 2)
        if judge_rows
        else None
    )
    total_judge_parse_errors = sum(1 for r in judge_rows if not r.llm_judge_parse_ok)
    fired_all = [r for r in results if r.anchor_fired]
    temporal_diag = {
        "n": len(results),
        "anchor_fired_n": len(fired_all),
        "anchor_fired_rate": round(len(fired_all) / max(1, len(results)), 4),
        "mean_displacement_s": (
            round(
                sum(r.anchor_displacement_s for r in fired_all) / max(1, len(fired_all)),
                1,
            )
            if fired_all
            else 0.0
        ),
        "by_category": anchor_by_cat,
    }
    valid_results = [r for r in results if not r.error]
    cost_summary: dict[str, Any] = {
        "n_llm_calls_total": sum(r.n_llm_calls for r in valid_results),
        "llm_calls_per_q_mean": round(
            sum(r.n_llm_calls for r in valid_results) / max(1, len(valid_results)), 2
        ),
        "prompt_tokens_total": sum(r.llm_prompt_tokens for r in valid_results),
        "completion_tokens_total": sum(r.llm_completion_tokens for r in valid_results),
        "tokens_per_q_mean": round(
            sum(r.llm_prompt_tokens + r.llm_completion_tokens for r in valid_results)
            / max(1, len(valid_results)),
            1,
        ),
        "db_size_mb_mean": round(
            sum(r.db_size_bytes for r in valid_results)
            / max(1, len(valid_results))
            / (1024 * 1024),
            3,
        ),
    }
    return {
        "meta": {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "partial": partial,
            "dataset": str(dataset_path),
            "categories": args.categories,
            "limit_per_category": args.limit,
            "consolidate": not args.no_consolidate,
            "assignment_threshold": args.assignment_threshold,
            "hit_threshold": HIT_THRESHOLD,
            "recall_mode": args.recall_mode,
            "top_k": args.top_k,
            "debug": args.debug,
            "keep_debug_dbs": args.keep_debug_dbs,
            "no_salience_rerank": args.no_salience_rerank,
            "no_graph_expansion": args.no_graph_expansion,
            "no_temporal": args.no_temporal,
            "judge_model": args.judge_model or None,
            "full_hypotheses": args.save_full_hypotheses,
            "evidence_format": "structured_v1" if args.save_full_hypotheses else "preview",
            "total_elapsed_s": round(total_elapsed, 2),
        },
        "summary": {
            "n": len(results),
            "hits": total_hits,
            "score_pct": round(100.0 * total_hits / max(1, len(results)), 2),
            "recall_at_k": total_recall_at_k,
            "mrr": total_mrr,
            "llm_judge_score_pct": total_judge_score_pct,
            "llm_judge_n": len(judge_rows),
            "llm_judge_parse_errors": total_judge_parse_errors,
            "llm_judge_prompt_tokens": sum(
                r.debug.get("llm_judge_tokens", {}).get("prompt", 0) for r in valid_results
            ),
            "llm_judge_completion_tokens": sum(
                r.debug.get("llm_judge_tokens", {}).get("completion", 0) for r in valid_results
            ),
            "llm_judge_cost_usd": (
                estimate_cost_usd(
                    args.judge_model,
                    sum(
                        r.debug.get("llm_judge_tokens", {}).get("prompt", 0) for r in valid_results
                    ),
                    sum(
                        r.debug.get("llm_judge_tokens", {}).get("completion", 0)
                        for r in valid_results
                    ),
                )
                if args.judge_model
                else None
            ),
            "llm_judge_note": (
                "Grades the same retrieved-context hypothesis the keyword scorer "
                "uses, not a separately generated answer — a retrieval-quality "
                "measure, not an end-to-end (retrieval+generation) score."
            ),
            "by_category": by_cat,
            "cost": cost_summary,
        },
        "diagnostics": {
            "temporal": temporal_diag,
        },
        "results": [_result_row(r) for r in results],
    }


def _write_payload(out_path: Path, payload: dict[str, Any]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    with open(tmp_path, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp_path, out_path)


# ---- main ----


def main() -> None:
    parser = argparse.ArgumentParser(description="LongMemEval evaluation harness for Slowave")
    parser.add_argument(
        "--dataset",
        default="data/longmemeval/longmemeval_oracle.json",
        help="Path to longmemeval JSON file",
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        default=[
            "knowledge-update",
            "single-session-preference",
            "multi-session",
            "single-session-user",
            "single-session-assistant",
            "temporal-reasoning",
        ],
        help="Question types to evaluate",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Max questions per category (0 = all)",
    )
    parser.add_argument(
        "--no-consolidate",
        action="store_true",
        help="Skip consolidation (fast smoke check with embedding recall only)",
    )
    parser.add_argument(
        "--replay-only",
        action="store_true",
        help="Run replay/transition training without consolidation (Stage 3 latent-only test)",
    )
    parser.add_argument(
        "--no-multi-scale",
        action="store_true",
        help="Stage 9 ablation: disable CA3+CA1 dual-scale prototypes. "
        "Default keeps both scales active.",
    )
    parser.add_argument(
        "--assignment-threshold",
        type=float,
        default=0.65,
        help="Cosine similarity threshold for prototype clustering (default 0.65)",
    )
    parser.add_argument(
        "--out",
        default="",
        help="Output JSON path. Default: timestamped file under data/longmemeval/runs/",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Include retrieved/all schemas, episodes, relations, and answer-presence diagnostics.",
    )
    parser.add_argument(
        "--keep-debug-dbs",
        action="store_true",
        help="Copy per-question temp DBs for misses (or all debug runs) to data/longmemeval/debug_dbs/.",
    )
    parser.add_argument(
        "--recall-mode",
        choices=["hybrid", "schemas", "episodes"],
        default="hybrid",
        help="Which recalled components to score as the hypothesis. Debug always records component scores.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Number of schemas/episodes to request from recall.",
    )
    # ablation flags
    parser.add_argument(
        "--no-salience-rerank",
        action="store_true",
        help="Ablation: disable salience-weighted episode reranking (salience_weight=0).",
    )
    parser.add_argument("--salience-weight", type=float, default=0.5)
    parser.add_argument("--tau-seconds", type=float, default=86400.0)
    parser.add_argument("--surprise-weight", type=float, default=0.3)
    parser.add_argument(
        "--no-graph-expansion",
        action="store_true",
        help="Ablation: disable prototype graph expansion at recall (neighbor_top_k=0).",
    )
    parser.add_argument(
        "--no-temporal",
        action="store_true",
        help="Ablation: disable temporal context weighting at recall (use_temporal=False).",
    )
    parser.add_argument(
        "--judge-model",
        default="",
        help="If set, also score with a lenient LLM-judge semantic-equivalence "
        "pass (via OpenRouter, needs OPENROUTER_API_KEY) alongside the default "
        "zero-cost keyword-overlap score. Costs real API tokens; unset by "
        "default so the benchmark stays free.",
    )
    parser.add_argument(
        "--save-full-hypotheses",
        action="store_true",
        help="Persist complete retrieved contexts instead of 400-character previews. "
        "Required when the artifact will be passed to aml_answer_eval.py.",
    )
    parser.add_argument(
        "--yes",
        "-y",
        action="store_true",
        help="Skip the confirmation prompt before paid --judge-model runs.",
    )
    args = parser.parse_args()
    openai_client = get_openai_client() if args.judge_model else None

    dataset_path = Path(args.dataset)
    if not dataset_path.is_absolute():
        dataset_path = REPO_ROOT / dataset_path

    print(f"Loading dataset: {dataset_path}")
    with open(dataset_path) as f:
        all_questions = json.load(f)
    print(f"Total in file: {len(all_questions)}")

    # filter and limit
    selected: list[dict] = []
    per_cat: dict[str, int] = {}
    for q in all_questions:
        qt = q["question_type"]
        if qt not in args.categories:
            continue
        per_cat[qt] = per_cat.get(qt, 0)
        if args.limit > 0 and per_cat[qt] >= args.limit:
            continue
        selected.append(q)
        per_cat[qt] += 1

    print(f"Selected: {len(selected)} questions from categories {args.categories}")
    if args.limit:
        print(f"(capped at {args.limit} per category)")

    if args.judge_model:
        # Empirically measured avg per-question judge token usage (2026-07-13
        # real full run, top_k=20) — pre-run estimate only, every selected
        # question gets judged here (no category exclusion, unlike LoCoMo).
        n_judged = len(selected)
        est_cost = estimate_cost_usd(args.judge_model, n_judged * 12_900, n_judged * 477)
        confirm_paid_run(
            f"LongMemEval will call {args.judge_model} as an LLM judge for ~{n_judged} questions.",
            est_cost,
            assume_yes=args.yes,
        )

    print(
        f"consolidate: {not args.no_consolidate}  "
        f"threshold: {args.assignment_threshold}  recall_mode: {args.recall_mode} top_k: {args.top_k}"
    )

    # Pre-load the encoder once; shared across all questions to avoid
    # reloading weights for every fresh per-question engine instance.
    print(
        "Loading encoder (paraphrase-multilingual-MiniLM-L12-v2)...",
        end=" ",
        flush=True,
    )
    enc_cfg = EncoderConfig()
    shared_enc = TextEncoder(enc_cfg)
    _ = shared_enc.dim  # force model load now
    print(f"OK (dim={shared_enc.dim})")
    print()

    # Determine output path before running so partial results can be saved after each question.
    if args.out:
        out_path = Path(args.out)
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mode = "with_consolidation" if not args.no_consolidate else "no_consolidation"
        parts = [stamp, mode]
        if sorted(args.categories) != sorted(ALL_CATEGORIES):
            parts.append("cats-" + "-".join(args.categories))
        if args.judge_model:
            parts.append("judge-" + args.judge_model.rsplit("/", 1)[-1])
        parts.append("debug" if args.debug else "run")
        out_path = Path(f"data/longmemeval/runs/{'_'.join(parts)}.json")
    if not out_path.is_absolute():
        out_path = REPO_ROOT / out_path

    results: list[QuestionResult] = []
    # (result_index, (question, hypothesis, expected_answer)) — judged in a
    # concurrent batch after the loop; see run_question's docstring.
    pending_judge: list[tuple[int, tuple[str, str, str]]] = []
    t_start = time.time()
    last_print = t_start
    print_every = max(1, len(selected) // 20)  # ~20 updates over a full run
    try:
        for i, q in enumerate(selected):
            r, judge_job = run_question(
                q,
                consolidate=not args.no_consolidate,
                assignment_threshold=args.assignment_threshold,
                shared_encoder=shared_enc,
                recall_mode=args.recall_mode,
                top_k=args.top_k,
                debug=args.debug,
                keep_debug_dbs=args.keep_debug_dbs,
                no_salience_rerank=args.no_salience_rerank,
                no_graph_expansion=args.no_graph_expansion,
                no_temporal=args.no_temporal,
                replay_only=args.replay_only,
                no_multi_scale=args.no_multi_scale,
                tau_seconds=args.tau_seconds,
                salience_weight=args.salience_weight,
                surprise_weight=args.surprise_weight,
                judge_model=args.judge_model or None,
                openai_client=openai_client,
                save_full_hypothesis=args.save_full_hypotheses,
            )
            results.append(r)
            if judge_job is not None:
                pending_judge.append((len(results) - 1, judge_job))

            if r.error:
                print(
                    f"[{i + 1:>4}/{len(selected)}] {r.question_type:<26} ERROR: {r.error[:80]}",
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
                hits = sum(1 for x in results if x.hit)
                elapsed = now - t_start
                rate = done / elapsed if elapsed > 0 else 0.0
                eta_s = (len(selected) - done) / rate if rate > 0 else None
                eta = f"{eta_s:.0f}s" if eta_s is not None else "?"
                print(
                    f"[{done:>4}/{len(selected)}] ({100 * done / len(selected):>3.0f}%)  "
                    f"hits={hits}/{done} ({100 * hits / done:.1f}%)  "
                    f"elapsed={elapsed:.0f}s  rate={rate:.2f}q/s  eta={eta}",
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

    judge_elapsed = 0.0
    if args.judge_model and pending_judge:
        print(f"\nJudging {len(pending_judge)} questions concurrently...", flush=True)
        t_judge_start = time.time()
        jobs = [job for _idx, job in pending_judge]
        judged = judge_batch_concurrent(openai_client, args.judge_model, jobs)
        judge_elapsed = time.time() - t_judge_start
        for (idx, _job), (score, reason, pt, ct, parse_ok) in zip(pending_judge, judged):
            row = results[idx]
            row.llm_judge_score = score
            row.llm_judge_reason = reason
            row.llm_judge_parse_ok = parse_ok
            row.debug["llm_judge_tokens"] = {"prompt": pt, "completion": ct}

    total_elapsed = time.time() - t_start
    print(f"\nCompleted {len(results)} questions in {total_elapsed:.1f}s")

    print_report(
        results,
        consolidate=not args.no_consolidate,
        judge_model=args.judge_model or None,
        total_elapsed=total_elapsed,
        judge_elapsed=judge_elapsed,
    )

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
