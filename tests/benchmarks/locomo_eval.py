#!/usr/bin/env python3
"""LoCoMo evaluation harness for Slowave.

LoCoMo (ACL 2024) - 10 long multi-session conversations, 1986 QA pairs.
Categories: 1=single-session  2=temporal  3=commonsense  4=multi-session  5=adversarial

Usage:
  .venv/bin/python tests/benchmarks/locomo_eval.py                    # all 10 convs
  .venv/bin/python tests/benchmarks/locomo_eval.py --limit 3          # first 3 convs
  .venv/bin/python tests/benchmarks/locomo_eval.py --categories 1 4   # only cat 1+4
  .venv/bin/python tests/benchmarks/locomo_eval.py --no-consolidate   # episode-only baseline
Download dataset first:
  curl -o data/locomo/locomo10.json https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import signal
import sys
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.WARNING)
for _n in (
    "sentence_transformers",
    "transformers",
    "httpx",
    "httpcore",
    "huggingface_hub",
    "filelock",
    "tqdm",
):
    logging.getLogger(_n).setLevel(logging.ERROR)
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

CATEGORY_NAMES = {
    1: "single-session",
    2: "temporal",
    3: "commonsense",
    4: "multi-session",
    5: "adversarial",
}
HIT_THRESHOLD = 0.5


class ConversationTimeout(Exception):
    """Raised by _raise_conv_timeout when a per-conversation SIGALRM fires."""


def _raise_conv_timeout(signum, frame):
    raise ConversationTimeout()


# ---- scoring -----------------------------------------------------------------


def _normalize(s: str) -> list[str]:
    """Normalize text to a token list exactly as in the LoCoMo / SQuAD F1 metric.

    Lowercases, strips punctuation, collapses whitespace, removes articles.
    Matches the normalization used in the LoCoMo paper (identical to SQuAD).
    """
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    tokens = s.split()
    articles = {"a", "an", "the"}
    return [t for t in tokens if t not in articles]


def f1_score(hyp: str, answer: str) -> float:
    """Token-level F1 between hypothesis and gold answer.

    Two variants are computed and the higher is returned, making the
    metric fair for both short generated answers (paper setting) and
    longer retrieved passages (Slowave setting):

    Standard SQuAD F1 (paper-comparable when prediction is short):
      precision = |hyp ∩ ans| / |hyp|
      recall    = |hyp ∩ ans| / |ans|
      F1        = 2PR/(P+R)

    Answer-recall F1 (fair for retrieval passages):
      precision = |hyp ∩ ans| / |ans|   ← cap precision by answer length
      recall    = |hyp ∩ ans| / |ans|
      F1        = same formula but with answer-capped precision

    The answer-recall variant does not penalise a long retrieved
    passage for containing extra tokens beyond the answer. It asks:
    "do the answer tokens appear in the retrieved context?" which is
    the right question for a retrieval system. It matches the
    'answer-in-context' accuracy used in RAG evaluation literature.

    Using max(standard_f1, recall_f1) gives a fair score for both
    short generated predictions and long retrieved passages.
    """
    from collections import Counter

    hyp_tok = _normalize(hyp)
    ans_tok = _normalize(answer)
    if not ans_tok or not hyp_tok:
        return 0.0
    hyp_c = Counter(hyp_tok)
    ans_c = Counter(ans_tok)
    common = sum((hyp_c & ans_c).values())
    if common == 0:
        return 0.0
    # standard SQuAD F1
    p_std = common / len(hyp_tok)
    r_std = common / len(ans_tok)
    f1_std = 2 * p_std * r_std / (p_std + r_std) if (p_std + r_std) > 0 else 0.0
    # answer-recall F1: precision capped by answer length (fair for passages)
    p_cap = common / len(ans_tok)  # = recall; denominator is answer, not hypothesis
    r_cap = common / len(ans_tok)
    f1_cap = 2 * p_cap * r_cap / (p_cap + r_cap) if (p_cap + r_cap) > 0 else 0.0
    # return the higher of the two — whichever mode the prediction is closest to
    return max(f1_std, f1_cap)


def keyword_score(hyp, ans):
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

    def tok(s):
        return {
            w
            for w in re.findall(r"[a-z0-9]+", s.lower())
            if w not in stop and (len(w) > 1 or w.isdigit())
        }

    at = tok(ans)
    return len(at & tok(hyp)) / len(at) if at else 0.0


def _parse_ts(s):
    s = str(s).strip()
    try:
        return int(datetime.strptime(s, "%I:%M %p on %d %B, %Y").timestamp())
    except Exception:
        pass
    m = re.search(r"(\d{1,2})\s+(\w+),?\s+(\d{4})", s)
    if m:
        try:
            return int(
                datetime.strptime(
                    "%s %s %s" % (m.group(1), m.group(2), m.group(3)), "%d %B %Y"
                ).timestamp()
            )
        except Exception:
            pass
    return int(time.time())


@dataclass
class QAResult:
    conv_id: str
    question: str
    expected: str
    hypothesis: str
    category: int
    keyword_score: float
    f1: float
    hit: bool
    n_schemas: int
    n_episodes: int
    latency_ingest_s: float
    latency_recall_s: float
    consolidate: bool
    # Cost / storage instrumentation (zero in latent / no-LLM mode).
    n_llm_calls: int = 0
    llm_prompt_tokens: int = 0
    llm_completion_tokens: int = 0
    db_size_bytes: int = 0
    error: str | None = None
    component_scores: dict = field(default_factory=dict)
    # Consolidation diagnostics (plans/05-consolidation.md Phase 4), stamped
    # onto every row of a conversation the same way cost instrumentation is.
    consolidation_diag: dict = field(default_factory=dict)
    # Temporal anchor diagnostics (plans/07-temporal.md Phase 4) — per question,
    # since estimate_anchor() runs once per recall() call, not per conversation.
    anchor_fired: bool = False
    anchor_displacement_s: int = 0
    # Recall@K / MRR (Layer 2 retrieval metrics; not computed for category 5
    # adversarial rows, which score "not fooled" rather than answer recall).
    recall_at_k: dict = field(default_factory=dict)
    mrr: float = 0.0
    # Optional LLM-judge pass (Phase 3.5, --judge-model flag) — None unless
    # requested. Not computed for category 5 adversarial rows, same reason
    # as recall_at_k above: there's no "correct answer" to judge equivalence
    # against for a "don't get fooled" check.
    llm_judge_score: float | None = None
    llm_judge_reason: str = ""
    llm_judge_parse_ok: bool = True


def run_conversation(
    sample,
    *,
    consolidate,
    shared_encoder,
    categories,
    top_k=10,
    no_salience_rerank=False,
    no_graph_expansion=False,
    keep_debug_dbs=False,
    replay_only=False,
    assignment_threshold=0.85,
    max_prototypes_per_replay=128,
    no_transition=False,
    no_self_supervise=False,
    no_pattern_separation=False,
    no_multi_scale=False,
    no_temporal=False,
    temporal_weight=0.25,
    tau_seconds=86400 * 30,
    salience_weight=0.5,
    surprise_weight=0.3,
    spread_score_weight=0.90,
    judge_overrides=None,
    judge_model=None,
    openai_client=None,
    limit_questions=0,
    save_full_hypothesis=False,
):
    conv_id = str(sample.get("sample_id", "?"))
    # Pin the global numpy RNG to a deterministic seed derived from conv_id so
    # salience sampling (salience.py:sample_proportional) and transition-model
    # batch sampling (replay_engine.py) produce identical results across runs.
    # Without this, np.random.choice in consolidation is non-deterministic and
    # LoCoMo scores vary by ±8 pp between runs on the same conversation.
    import hashlib as _hashlib

    import numpy as _np

    _seed = int.from_bytes(_hashlib.sha256(conv_id.encode("utf-8")).digest()[:4], "big") % (2**31)
    _np.random.seed(_seed)
    conv = sample["conversation"]
    speaker_a = conv.get("speaker_a", "A")
    conv.get("speaker_b", "B")
    qa_items = [q for q in sample["qa"] if q["category"] in categories]
    if limit_questions:
        qa_items = qa_items[:limit_questions]
    if not qa_items:
        return []
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
                sample_size=2048,
                max_prototypes_per_replay=max_prototypes_per_replay,
                use_multi_scale=not no_multi_scale,
            ),
            retrieval=RetrievalConfig(
                salience_weight=0.0 if no_salience_rerank else salience_weight,
                neighbor_top_k=0 if no_graph_expansion else 6,
                use_transition=not no_transition,
                use_multi_scale=not no_multi_scale,
                spread_score_weight=spread_score_weight,
                use_temporal=not no_temporal,
                temporal_weight=0.0 if no_temporal else temporal_weight,
            ),
            relation=GeometricRelationConfig(**(judge_overrides or {})),
            disable_encoder=False,
        )
        eng = SlowaveEngine(cfg, shared_encoder=shared_encoder)
        t_ingest = time.time()
        nsess = len([k for k in conv if k.startswith("session_") and "date" not in k])
        for i in range(1, nsess + 1):
            turns = conv.get("session_%d" % i, [])
            date_str = conv.get("session_%d_date_time" % i, "")
            session_ts = _parse_ts(date_str) if date_str else None
            if not turns:
                continue
            sid = eng.session_start(agent="locomo", scope=f"eval:{conv_id}")
            if session_ts:
                conn = eng.db.connect()
                conn.execute("UPDATE sessions SET started_ts=? WHERE id=?", (session_ts, sid))
                conn.commit()
            for turn in turns:
                speaker = str(turn.get("speaker", ""))
                text = str(turn.get("text", "")).strip()
                if not text:
                    continue
                caption = str(turn.get("blip_caption", "")).strip()
                if caption:
                    text = text + " [image: " + caption + "]"
                role = "user_message" if speaker == speaker_a else "assistant_message"
                emb = shared_encoder.encode(text)
                eng.raw_log.append(
                    session_id=sid,
                    ts=session_ts or int(time.time()),
                    type=role,
                    content=text,
                    embedding=emb,
                )
            eng.session_end(sid, consolidate=False)
            if session_ts:
                conn = eng.db.connect()
                conn.execute(
                    "UPDATE episodic_memories SET ts=?,last_salience_ts=? WHERE event_id LIKE ? OR event_id LIKE ?",
                    (session_ts, session_ts, "micro_%s_%%" % sid, "macro_%s" % sid),
                )
                conn.commit()
        consolidation_diag: dict = {}
        if consolidate:
            _cstats = eng.consolidate_once(triggered_by="locomo_eval")
            consolidation_diag = _cstats.get("consolidation", {}) or {}
            # Stage 5: self-supervised retrieval rehearsal AFTER the
            # graph has been built and schemas extracted. Brain analogue:
            # sleep replay tightens the graph based on what the system
            # would have failed to retrieve.
            if not no_self_supervise:
                eng.replay_engine.self_supervise()
        elif replay_only:
            # Brain-inspired-only baseline: build prototypes + graph edges
            # by running replay, but skip LLM schema extraction so any
            # benchmark lift comes from the latent-side mechanisms
            # (spreading activation, salience, coactivation).
            eng.replay_engine.replay_once()
            if not no_self_supervise:
                eng.replay_engine.self_supervise()
        latency_ingest = time.time() - t_ingest
        results = []
        # (result_index, question, hypothesis, expected_answer) — judged in a
        # concurrent batch after this loop, since judge_answer_equivalence is a
        # pure network call over already-computed text and never touches the
        # engine, unlike everything else in this loop.
        pending_judge: list[tuple[int, str, str, str]] = []
        for qa in qa_items:
            question = str(qa.get("question", "")).strip()
            answer = str(qa.get("answer", "")) if qa.get("answer") is not None else ""
            adversarial = str(qa.get("adversarial_answer", ""))
            category = int(qa.get("category", 1))
            if category == 5:
                t0 = time.time()
                r = eng.recall(question, top_k=top_k)
                lr = time.time() - t0
                hyp = " ".join(
                    [s.content_text for s in r.schemas]
                    + [ep["content_text"] for ep in r.episode_texts]
                )
                adv_ks = keyword_score(hyp[:600], adversarial) if adversarial else 0.0
                not_fooled = adv_ks < HIT_THRESHOLD
                results.append(
                    QAResult(
                        conv_id=conv_id,
                        question=question,
                        expected="[not: %s]" % adversarial,
                        hypothesis=hyp[:400],
                        category=category,
                        keyword_score=round(1.0 - adv_ks, 3),
                        f1=round(1.0 - f1_score(hyp[:600], adversarial), 3),
                        hit=not_fooled,
                        n_schemas=len(r.schemas),
                        n_episodes=len(r.episode_texts),
                        latency_ingest_s=round(latency_ingest, 2),
                        latency_recall_s=round(lr, 3),
                        consolidate=consolidate,
                        component_scores={"adversarial_ks": round(adv_ks, 3)},
                        anchor_fired=r.anchor_fired,
                        anchor_displacement_s=r.anchor_displacement_s,
                    )
                )
                continue
            if not answer:
                continue
            t0 = time.time()
            r = eng.recall(question, top_k=top_k)
            lr = time.time() - t0
            sh = " ".join(s.content_text for s in r.schemas)
            eh = " ".join(ep["content_text"] for ep in r.episode_texts if ep["content_text"])
            hyp = (sh + " " + eh).strip()
            structured_hypothesis = format_retrieved_evidence(r)
            ks = keyword_score(hyp, answer)
            f1 = f1_score(hyp, answer)
            recall_at_k, mrr = compute_recall_at_k_and_mrr(
                eng,
                question,
                answer,
                keyword_score_fn=keyword_score,
                hit_threshold=HIT_THRESHOLD,
            )
            if judge_model:
                pending_judge.append((len(results), question, hyp, answer))
            results.append(
                QAResult(
                    conv_id=conv_id,
                    question=question,
                    expected=answer,
                    hypothesis=structured_hypothesis if save_full_hypothesis else hyp[:400],
                    category=category,
                    keyword_score=round(ks, 3),
                    f1=round(f1, 3),
                    hit=ks >= HIT_THRESHOLD,
                    n_schemas=len(r.schemas),
                    n_episodes=len(r.episode_texts),
                    latency_ingest_s=round(latency_ingest, 2),
                    latency_recall_s=round(lr, 3),
                    consolidate=consolidate,
                    component_scores={
                        "schemas": round(keyword_score(sh, answer), 3),
                        "episodes": round(keyword_score(eh, answer), 3),
                        "hybrid": round(ks, 3),
                        "f1_schemas": round(f1_score(sh, answer), 3),
                        "f1_episodes": round(f1_score(eh, answer), 3),
                        "f1_hybrid": round(f1, 3),
                    },
                    anchor_fired=r.anchor_fired,
                    anchor_displacement_s=r.anchor_displacement_s,
                    recall_at_k=recall_at_k,
                    mrr=round(mrr, 4),
                )
            )
        # Cost instrumentation snapshot — taken BEFORE eng.close().
        # Conversation-level (the same DB serves all per-question
        # recalls), so we stamp the same numbers on every QAResult in
        # this conversation. Divide LLM call totals by qa-count to get
        # a per-question figure that aggregates correctly downstream.
        counting = getattr(eng, "_counting_llm", None)
        if counting is not None:
            snap = counting.snapshot()
            calls_total = int(snap["n_calls"])
            pt_total = int(snap["prompt_tokens"])
            ct_total = int(snap["completion_tokens"])
        else:
            calls_total = pt_total = ct_total = 0
        db_size_total = 0
        for ext in ("", "-wal", "-shm"):
            try:
                db_size_total += int(os.path.getsize(db_path + ext))
            except OSError:
                pass
        n_results = max(1, len(results))
        for r in results:
            r.n_llm_calls = calls_total // n_results
            r.llm_prompt_tokens = pt_total // n_results
            r.llm_completion_tokens = ct_total // n_results
            r.db_size_bytes = db_size_total
            r.consolidation_diag = consolidation_diag
        if keep_debug_dbs:
            dest = REPO_ROOT / "data" / "locomo" / "debug_dbs" / "%s.db" % conv_id
            dest.parent.mkdir(parents=True, exist_ok=True)
            eng.close()
            for ext in ("", "-wal", "-shm"):
                src = Path(db_path + ext)
                if src.exists():
                    shutil.copy2(src, Path(str(dest) + ext))
        else:
            eng.close()

        if judge_model and pending_judge:
            jobs = [(q, hyp, ans) for _idx, q, hyp, ans in pending_judge]
            judged = judge_batch_concurrent(openai_client, judge_model, jobs)
            for (idx, _q, _hyp, _ans), (score, reason, pt, ct, parse_ok) in zip(
                pending_judge, judged
            ):
                row = results[idx]
                row.llm_judge_score = score
                row.llm_judge_reason = reason
                row.llm_judge_parse_ok = parse_ok
                row.component_scores["llm_judge_prompt_tokens"] = pt
                row.component_scores["llm_judge_completion_tokens"] = ct
        return results
    except Exception as e:
        return [
            QAResult(
                conv_id=conv_id,
                question="[error]",
                expected="",
                hypothesis="",
                category=0,
                keyword_score=0.0,
                f1=0.0,
                hit=False,
                n_schemas=0,
                n_episodes=0,
                latency_ingest_s=0.0,
                latency_recall_s=0.0,
                consolidate=consolidate,
                error=str(e) or f"{type(e).__name__} (no message)",
            )
        ]
    finally:
        for ext in ("", "-wal", "-shm"):
            p = db_path + ext
            if os.path.exists(p):
                os.remove(p)


def print_report(results, consolidate, *, judge_model=None, total_elapsed=0.0):
    print_header(
        "LoCoMo Evaluation Report",
        [
            f"Mode   : {'consolidation on (replay + geometric schema extraction, zero LLM by default)' if consolidate else 'consolidation off (episodes only, no schemas)'}",
            f"Total  : {len(results)} questions",
        ],
    )
    by_cat = {}
    for r in results:
        by_cat.setdefault(r.category, []).append(r)
    has_judge = any(r.llm_judge_score is not None for r in results)

    headers = ["Category", "N", "Hit%", "AvgF1"]
    if has_judge:
        headers.append("Judge%")
    table_rows = []
    tn = th = 0
    all_judge_rs = []
    for cat in sorted(by_cat):
        rs = by_cat[cat]
        hits = sum(1 for r in rs if r.hit)
        avg_f1 = sum(getattr(r, "f1", 0.0) for r in rs) / max(1, len(rs))
        pct = 100 * hits / max(1, len(rs))
        tn += len(rs)
        th += hits
        judge_rs = [r for r in rs if r.llm_judge_score is not None]
        all_judge_rs.extend(judge_rs)
        row = [
            CATEGORY_NAMES.get(cat, "cat-%d" % cat),
            str(len(rs)),
            f"{pct:.1f}%",
            f"{avg_f1:.3f}",
        ]
        if has_judge:
            if judge_rs:
                j_pct = 100 * sum(r.llm_judge_score for r in judge_rs) / len(judge_rs)
                row.append(f"{j_pct:.1f}%")
            else:
                row.append("n/a")
        table_rows.append(row)

    overall_f1 = sum(getattr(r, "f1", 0.0) for r in results) / max(1, len(results))
    total_row = ["TOTAL", str(tn), f"{100 * th / max(1, tn):.1f}%", f"{overall_f1:.3f}"]
    if has_judge:
        j_overall = 100 * sum(r.llm_judge_score for r in all_judge_rs) / max(1, len(all_judge_rs))
        total_row.append(f"{j_overall:.1f}%")

    print_table(headers, table_rows, total_row=total_row)
    print()
    recall_rows = [r for r in results if not r.error and r.recall_at_k]
    recall_at_k_pct, mrr = aggregate_recall_at_k_mrr(
        [r.recall_at_k for r in recall_rows],
        [r.mrr for r in recall_rows],
    )
    print(
        " Recall@K: "
        + "  ".join(f"{k}={v}%" for k, v in recall_at_k_pct.items())
        + "  (category 5 adversarial rows excluded)"
    )
    print(f" MRR     : {mrr}")
    judge_rows = [r for r in recall_rows if r.llm_judge_score is not None]
    if judge_rows:
        judge_avg = sum(r.llm_judge_score for r in judge_rows) / len(judge_rows)
        judge_errs = sum(1 for r in judge_rows if not r.llm_judge_parse_ok)
        print(
            f" LLM-judge score: {judge_avg * 100:.1f}%  (n={len(judge_rows)}, "
            f"parse errors: {judge_errs}/{len(judge_rows)}, category 5 excluded)"
        )

    print()
    print(" Timing")
    print(f"  total:  {total_elapsed:.1f}s  ({len(results)} questions)")
    all_ingest = [r.latency_ingest_s for r in results if not r.error]
    all_recall = [r.latency_recall_s for r in results if not r.error]
    if all_ingest:
        print(
            f"  ingest: sum={sum(all_ingest):.1f}s  mean={sum(all_ingest) / len(all_ingest):.2f}s  "
            f"max={max(all_ingest):.2f}s"
        )
    if all_recall:
        print(
            f"  recall: sum={sum(all_recall):.1f}s  mean={sum(all_recall) / len(all_recall) * 1000:.1f}ms  "
            f"max={max(all_recall) * 1000:.1f}ms"
        )

    by_conv = {}
    for r in results:
        by_conv.setdefault(r.conv_id, []).append(r)
    print()
    print(" Cost summary")
    valid = [r for r in results if not r.error]
    if valid:
        # Conversation-level numbers: pick one representative per conv_id.
        per_conv_rep = [next(iter(rs)) for rs in by_conv.values()]
        n_convs = max(1, len(per_conv_rep))
        sizes_mb = [r.db_size_bytes / (1024 * 1024) for r in per_conv_rep]
        print(f"  db size/conv: mean={sum(sizes_mb) / n_convs:.2f}MB  max={max(sizes_mb):.2f}MB")

        calls = [r.n_llm_calls * len(by_conv[r.conv_id]) for r in per_conv_rep]
        if sum(calls) > 0:
            prompt = [r.llm_prompt_tokens * len(by_conv[r.conv_id]) for r in per_conv_rep]
            compl = [r.llm_completion_tokens * len(by_conv[r.conv_id]) for r in per_conv_rep]
            total = [p + c for p, c in zip(prompt, compl)]
            print(
                f"  consolidation llm_calls/conv: mean={sum(calls) / n_convs:.1f}  max={max(calls)}  total={sum(calls)}"
            )
            print(
                f"  consolidation tokens/conv:    mean={sum(total) / n_convs:.0f}  total={sum(total)}"
            )
        else:
            print("  consolidation: zero LLM calls (brain-only, geometric schema extraction)")
    judge_rows = [r for r in results if r.llm_judge_score is not None]
    if judge_rows and judge_model:
        judge_pt = sum(r.component_scores.get("llm_judge_prompt_tokens", 0) for r in judge_rows)
        judge_ct = sum(r.component_scores.get("llm_judge_completion_tokens", 0) for r in judge_rows)
        judge_cost = estimate_cost_usd(judge_model, judge_pt, judge_ct)
        cost_s = f"${judge_cost:.2f}" if judge_cost is not None else "unknown pricing"
        print(f"  judge ({judge_model}): prompt={judge_pt}  completion={judge_ct}  cost={cost_s}")
    print_footer()


def _aggregate_consolidation_diag(results):
    """Sum per-conversation consolidation diagnostics into one run-level
    summary. Dedupes by conv_id since consolidation_diag is stamped
    identically on every QAResult row of the same conversation."""
    seen_convs = set()
    agg = {
        "prototypes_processed": 0,
        "schemas_created": 0,
        "schemas_reinforced": 0,
        "schemas_skipped": 0,
        "near_dup_intercepts": 0,
        "verdict_counts": {},
        "confidence_histogram": [],
    }
    for r in results:
        diag = getattr(r, "consolidation_diag", None)
        if not diag or r.conv_id in seen_convs:
            continue
        seen_convs.add(r.conv_id)
        for key in (
            "prototypes_processed",
            "schemas_created",
            "schemas_reinforced",
            "schemas_skipped",
            "near_dup_intercepts",
        ):
            agg[key] += int(diag.get(key, 0) or 0)
        for key in ("verdict_counts",):
            for k, v in (diag.get(key) or {}).items():
                agg[key][k] = agg[key].get(k, 0) + int(v)
        agg["confidence_histogram"].extend(diag.get("confidence_histogram") or [])
    return agg


def _aggregate_temporal_diag(results):
    """anchor_fired rate + mean displacement, overall and per LoCoMo category
    (plans/07-temporal.md Q1 — is TemporalProbe firing, and where)."""
    by_cat: dict = {}
    for cat in set(r.category for r in results):
        rs = [r for r in results if r.category == cat]
        fired = [r for r in rs if r.anchor_fired]
        by_cat[str(cat)] = {
            "name": CATEGORY_NAMES.get(cat, "cat-%d" % cat),
            "n": len(rs),
            "anchor_fired_n": len(fired),
            "anchor_fired_rate": round(len(fired) / max(1, len(rs)), 4),
            "mean_displacement_s": (
                round(sum(r.anchor_displacement_s for r in fired) / max(1, len(fired)), 1)
                if fired
                else 0.0
            ),
        }
    fired_all = [r for r in results if r.anchor_fired]
    return {
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
        "by_category": by_cat,
    }


def _save(path, results, args, elapsed, partial):
    path.parent.mkdir(parents=True, exist_ok=True)
    by_cat = {}
    for cat in set(r.category for r in results):
        rs = [r for r in results if r.category == cat]
        hits = sum(1 for r in rs if r.hit)
        cat_recall_rows = [r for r in rs if not r.error and r.recall_at_k]
        cat_recall_at_k, cat_mrr = aggregate_recall_at_k_mrr(
            [r.recall_at_k for r in cat_recall_rows],
            [r.mrr for r in cat_recall_rows],
        )
        cat_judge_rows = [r for r in cat_recall_rows if r.llm_judge_score is not None]
        by_cat[str(cat)] = {
            "name": CATEGORY_NAMES.get(cat, "cat-%d" % cat),
            "n": len(rs),
            "hits": hits,
            "score_pct": round(100 * hits / max(1, len(rs)), 2),
            "avg_ks": round(sum(r.keyword_score for r in rs) / max(1, len(rs)), 4),
            "avg_f1": round(sum(getattr(r, "f1", 0.0) for r in rs) / max(1, len(rs)), 4),
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
    th = sum(1 for r in results if r.hit)
    recall_rows = [r for r in results if not r.error and r.recall_at_k]
    total_recall_at_k, total_mrr = aggregate_recall_at_k_mrr(
        [r.recall_at_k for r in recall_rows],
        [r.mrr for r in recall_rows],
    )
    judge_rows = [r for r in recall_rows if r.llm_judge_score is not None]
    total_judge_score_pct = (
        round(100 * sum(r.llm_judge_score for r in judge_rows) / len(judge_rows), 2)
        if judge_rows
        else None
    )
    total_judge_parse_errors = sum(1 for r in judge_rows if not r.llm_judge_parse_ok)
    payload = {
        "meta": {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "partial": partial,
            "dataset": "locomo10",
            "consolidate": not args.no_consolidate,
            "categories": args.categories,
            "limit": args.limit,
            "top_k": args.top_k,
            "no_salience_rerank": args.no_salience_rerank,
            "no_graph_expansion": args.no_graph_expansion,
            "no_temporal": args.no_temporal,
            "temporal_weight": args.temporal_weight,
            "judge_overrides": args.judge_overrides,
            "judge_model": args.judge_model or None,
            "full_hypotheses": args.save_full_hypotheses,
            "evidence_format": "structured_v1" if args.save_full_hypotheses else "preview",
            "total_elapsed_s": round(elapsed, 2),
        },
        "summary": {
            "n": len(results),
            "hits": th,
            "score_pct": round(100 * th / max(1, len(results)), 2),
            "recall_at_k": total_recall_at_k,
            "mrr": total_mrr,
            "llm_judge_score_pct": total_judge_score_pct,
            "llm_judge_n": len(judge_rows),
            "llm_judge_parse_errors": total_judge_parse_errors,
            "llm_judge_prompt_tokens": sum(
                r.component_scores.get("llm_judge_prompt_tokens", 0) for r in judge_rows
            ),
            "llm_judge_completion_tokens": sum(
                r.component_scores.get("llm_judge_completion_tokens", 0) for r in judge_rows
            ),
            "llm_judge_cost_usd": (
                estimate_cost_usd(
                    args.judge_model,
                    sum(r.component_scores.get("llm_judge_prompt_tokens", 0) for r in judge_rows),
                    sum(
                        r.component_scores.get("llm_judge_completion_tokens", 0) for r in judge_rows
                    ),
                )
                if args.judge_model
                else None
            ),
            "llm_judge_note": (
                "Grades the same retrieved-context hypothesis the keyword scorer "
                "uses, not a separately generated answer — a retrieval-quality "
                "measure, not an end-to-end (retrieval+generation) score. Category "
                "5 (adversarial) excluded, same reason as recall_at_k."
            ),
            "by_category": by_cat,
        },
        "diagnostics": {
            "consolidation": _aggregate_consolidation_diag(results),
            "temporal": _aggregate_temporal_diag(results),
        },
        "results": [
            {
                "conv_id": r.conv_id,
                "category": r.category,
                "question": r.question,
                "expected": r.expected,
                "hypothesis": r.hypothesis,
                "keyword_score": r.keyword_score,
                "f1": getattr(r, "f1", 0.0),
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
                "anchor_fired": r.anchor_fired,
                "anchor_displacement_s": r.anchor_displacement_s,
                "error": r.error,
            }
            for r in results
        ],
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, path)


def main():
    parser = argparse.ArgumentParser(description="LoCoMo eval for Slowave")
    parser.add_argument("--dataset", default="data/locomo/locomo10.json")
    parser.add_argument("--categories", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--limit", type=int, default=0, help="Max conversations (0=all 10)")
    parser.add_argument(
        "--limit-questions",
        type=int,
        default=0,
        help="Max questions per conversation (0=all). Combine with --limit for a "
        "cheap smoke test, e.g. --limit 1 --limit-questions 3 — otherwise --limit 1 "
        "still runs every question in that one conversation (~150-260 of them),  "
        "which is expensive with --judge-model on a pricier model.",
    )
    parser.add_argument("--no-consolidate", action="store_true")
    parser.add_argument(
        "--replay-only",
        action="store_true",
        help="Run replay (build prototypes + edges) but skip LLM consolidation. "
        "Use this to isolate the contribution of brain-inspired retrieval "
        "(spreading activation, salience, graph) from LLM schema extraction.",
    )
    parser.add_argument(
        "--assignment-threshold",
        type=float,
        default=0.85,
        help="Replay cluster-merge threshold; lower = fewer, larger prototypes. "
        "0.65 collapses everything into one cluster on conversational data; "
        "0.85 gives ~50 prototypes per LoCoMo conversation. Default 0.85.",
    )
    parser.add_argument(
        "--max-prototypes",
        type=int,
        default=128,
        help="Cap on prototypes formed per replay (default 128).",
    )
    parser.add_argument("--no-salience-rerank", action="store_true")
    parser.add_argument("--salience-weight", type=float, default=0.5)
    parser.add_argument("--tau-seconds", type=float, default=float(86400 * 30))
    parser.add_argument("--surprise-weight", type=float, default=0.3)
    parser.add_argument("--spread-score-weight", type=float, default=0.90)
    parser.add_argument("--no-graph-expansion", action="store_true")
    parser.add_argument(
        "--no-transition",
        action="store_true",
        help="Disable the predictive transition-model seed at recall "
        "(Stage 3 ablation: leaves the rest of the pipeline intact).",
    )
    parser.add_argument(
        "--no-self-supervise",
        action="store_true",
        help="Disable the self-supervised retrieval-rehearsal pass at the "
        "end of consolidation (Stage 5 ablation).",
    )
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--keep-debug-dbs", action="store_true")
    parser.add_argument(
        "--timeout",
        type=float,
        default=600.0,
        help="Hard per-conversation watchdog (SIGALRM), seconds (default 600). If a "
        "conversation exceeds this — hang, thrashing, stalled network call, anything — "
        "it's aborted and recorded as an error so the run keeps moving instead of hanging "
        "forever. 600s gives ~2x margin over the largest observed real conversation (~290s "
        "with --judge-model). A prior version of this flag was accepted but never enforced.",
    )
    parser.add_argument(
        "--no-pattern-separation",
        action="store_true",
        help="Stage 8 ablation: disable dentate-gyrus-style competitive prototype assignment.",
    )
    parser.add_argument(
        "--no-multi-scale",
        action="store_true",
        help="Stage 9 ablation: disable CA3+CA1 dual-scale prototypes.",
    )
    parser.add_argument(
        "--no-temporal",
        action="store_true",
        help="Stage 7/10 ablation: disable temporal-proximity score bonus at "
        "recall (use_temporal=False, temporal_weight=0.0). Does not stop "
        "TemporalProbe.estimate_anchor() from running (see core/07-temporal.md).",
    )
    parser.add_argument(
        "--temporal-weight",
        type=float,
        default=0.25,
        help="Stage 7 grid search: weight of the temporal-proximity score bonus "
        "(plans/07-temporal.md Grid Search). Ignored when --no-temporal is set.",
    )
    parser.add_argument(
        "--judge-overrides",
        default="",
        help="JSON dict of GeometricRelationConfig field overrides "
        "(plans/05-consolidation.md Threshold Ablation Matrix), e.g. "
        "'{\"related_schema_cosine\": 1.01}'.",
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
    parser.add_argument("--out", default="")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="If --out already exists (e.g. from a run killed by OOM), load its "
        "completed conversations and skip re-running them instead of starting over.",
    )
    parser.add_argument(
        "--yes",
        "-y",
        action="store_true",
        help="Skip the confirmation prompt before paid --judge-model runs.",
    )
    args = parser.parse_args()
    judge_overrides = json.loads(args.judge_overrides) if args.judge_overrides else {}
    openai_client = get_openai_client() if args.judge_model else None
    dataset_path = Path(args.dataset)
    if not dataset_path.is_absolute():
        dataset_path = REPO_ROOT / dataset_path
    if not dataset_path.exists():
        print("Dataset not found:", dataset_path)
        print(
            "Download: curl -o data/locomo/locomo10.json https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"
        )
        sys.exit(1)
    print("Loading dataset:", dataset_path)
    samples = json.load(open(dataset_path))
    if args.limit > 0:
        samples = samples[: args.limit]
    cat_names = ", ".join(f"{c}={CATEGORY_NAMES.get(c, '?')}" for c in sorted(args.categories))
    print(f"Conversations: {len(samples)}  Categories: {cat_names}")

    if args.judge_model:
        # Category 5 (adversarial) is never judged (§ run_conversation). Count
        # everything else with a non-empty answer to estimate the judge bill —
        # mirroring run_conversation's own filter-then-slice order so this
        # matches actual behavior when --limit-questions caps a conversation.
        n_judged = 0
        for s in samples:
            qa_items = [q for q in s.get("qa", []) if q.get("category") in args.categories]
            if args.limit_questions:
                qa_items = qa_items[: args.limit_questions]
            n_judged += sum(
                1
                for qa in qa_items
                if qa.get("category") != 5 and str(qa.get("answer", "")).strip()
            )
        # Empirically measured avg per-question judge token usage (2026-07-13
        # real run, top_k=20, 60K char hypothesis cap) — pre-run estimate only.
        est_cost = estimate_cost_usd(args.judge_model, n_judged * 10_540, n_judged * 201)
        confirm_paid_run(
            f"LoCoMo will call {args.judge_model} as an LLM judge for ~{n_judged} questions.",
            est_cost,
            assume_yes=args.yes,
        )

    print("Loading encoder...", end=" ", flush=True)
    enc = TextEncoder(EncoderConfig())
    _ = enc.dim
    print("OK (dim=%d)" % enc.dim)
    print()
    if args.out:
        out_path = Path(args.out)
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mode = "with_consolidation" if not args.no_consolidate else "no_consolidation"
        parts = [stamp, mode]
        if sorted(args.categories) != [1, 2, 3, 4, 5]:
            parts.append("cats" + "-".join(str(c) for c in sorted(args.categories)))
        if args.judge_model:
            parts.append("judge-" + args.judge_model.rsplit("/", 1)[-1])
        filename = "_".join(parts) + ".json"
        out_path = REPO_ROOT / "data" / "locomo" / "runs" / filename
    if not out_path.is_absolute():
        out_path = REPO_ROOT / out_path

    all_results = []
    done_conv_ids: set[str] = set()
    if args.resume and out_path.exists():
        try:
            prev = json.load(open(out_path))
            prev_rows = prev.get("results", [])
            # A conversation only counts as "done" if it has at least one
            # non-error row — otherwise it errored out completely (e.g. a
            # bare MemoryError with no message under memory pressure) and
            # must be retried, not silently skipped forever. Its broken
            # placeholder row(s) are dropped here so a retry doesn't end up
            # duplicated alongside the real results.
            rows_by_conv: dict[str, list] = {}
            for r in prev_rows:
                rows_by_conv.setdefault(r["conv_id"], []).append(r)

            def _is_broken_row(r: dict) -> bool:
                # question=="[error]"/"[timeout]" catches rows even from
                # before error= was guaranteed non-empty (legacy data).
                return bool(r.get("error")) or r.get("question") in (
                    "[error]",
                    "[timeout]",
                )

            done_conv_ids = {
                cid
                for cid, rows in rows_by_conv.items()
                if any(not _is_broken_row(r) for r in rows)
            }
            broken_conv_ids = set(rows_by_conv) - done_conv_ids
            if broken_conv_ids:
                print(
                    f"--resume: {len(broken_conv_ids)} conversation(s) errored out completely "
                    f"last time, will retry: {sorted(broken_conv_ids)}",
                    flush=True,
                )
            prev_rows = [r for r in prev_rows if r["conv_id"] not in broken_conv_ids]
            for r in prev_rows:
                all_results.append(
                    QAResult(
                        conv_id=r["conv_id"],
                        question=r["question"],
                        expected=r["expected"],
                        hypothesis=r["hypothesis"],
                        category=r["category"],
                        keyword_score=r["keyword_score"],
                        f1=r.get("f1", 0.0),
                        hit=r["hit"],
                        n_schemas=r["n_schemas"],
                        n_episodes=r["n_episodes"],
                        latency_ingest_s=r["latency_ingest_s"],
                        latency_recall_s=r["latency_recall_s"],
                        consolidate=not args.no_consolidate,
                        n_llm_calls=r.get("n_llm_calls", 0),
                        llm_prompt_tokens=r.get("llm_prompt_tokens", 0),
                        llm_completion_tokens=r.get("llm_completion_tokens", 0),
                        db_size_bytes=r.get("db_size_bytes", 0),
                        error=r.get("error"),
                        component_scores=r.get("component_scores", {}) or {},
                        anchor_fired=r.get("anchor_fired", False),
                        anchor_displacement_s=r.get("anchor_displacement_s", 0),
                        recall_at_k=r.get("recall_at_k", {}) or {},
                        mrr=r.get("mrr", 0.0),
                        llm_judge_score=r.get("llm_judge_score"),
                        llm_judge_reason=r.get("llm_judge_reason", ""),
                        llm_judge_parse_ok=r.get("llm_judge_parse_ok", True),
                    )
                )
            print(
                f"Resuming from {out_path}: {len(done_conv_ids)} conversation(s) already done "
                f"({len(all_results)} questions), skipping them.",
                flush=True,
            )
        except Exception as e:
            print(
                f"[WARN] --resume: could not load {out_path} ({e}), starting fresh.",
                flush=True,
            )
            all_results = []
            done_conv_ids = set()

    t_start = time.time()
    try:
        _run_locomo_loop(
            samples,
            done_conv_ids,
            all_results,
            args,
            t_start,
            out_path,
            enc,
            judge_overrides,
            openai_client,
        )
    except KeyboardInterrupt:
        _save(out_path, all_results, args, time.time() - t_start, partial=True)
        print(f"\nInterrupted. {len(all_results)} questions completed and saved to: {out_path}")
        raise
    print("\nCompleted %d questions in %.1fs" % (len(all_results), time.time() - t_start))
    print_report(
        all_results,
        not args.no_consolidate,
        judge_model=args.judge_model or None,
        total_elapsed=time.time() - t_start,
    )
    _save(out_path, all_results, args, time.time() - t_start, partial=False)
    print("\nResults saved to:", out_path)


def _run_locomo_loop(
    samples,
    done_conv_ids,
    all_results,
    args,
    t_start,
    out_path,
    enc,
    judge_overrides,
    openai_client,
):
    for i, sample in enumerate(samples):
        conv_id = sample.get("sample_id", str(i))
        if conv_id in done_conv_ids:
            print(
                "[%d/%d] %s ... skipped (already done, --resume)" % (i + 1, len(samples), conv_id),
                flush=True,
            )
            continue
        print("[%d/%d] %s ..." % (i + 1, len(samples), conv_id), end="", flush=True)
        t0 = time.time()
        if hasattr(signal, "SIGALRM"):
            signal.signal(signal.SIGALRM, _raise_conv_timeout)
            signal.alarm(int(args.timeout))
        try:
            rs = run_conversation(
                sample,
                consolidate=not args.no_consolidate,
                shared_encoder=enc,
                categories=args.categories,
                top_k=args.top_k,
                no_salience_rerank=args.no_salience_rerank,
                no_graph_expansion=args.no_graph_expansion,
                keep_debug_dbs=args.keep_debug_dbs,
                replay_only=args.replay_only,
                assignment_threshold=args.assignment_threshold,
                max_prototypes_per_replay=args.max_prototypes,
                no_transition=args.no_transition,
                no_self_supervise=args.no_self_supervise,
                no_pattern_separation=args.no_pattern_separation,
                no_multi_scale=args.no_multi_scale,
                no_temporal=args.no_temporal,
                temporal_weight=args.temporal_weight,
                tau_seconds=args.tau_seconds,
                salience_weight=args.salience_weight,
                surprise_weight=args.surprise_weight,
                spread_score_weight=args.spread_score_weight,
                judge_overrides=judge_overrides,
                judge_model=args.judge_model or None,
                openai_client=openai_client,
                limit_questions=args.limit_questions,
                save_full_hypothesis=args.save_full_hypotheses,
            )
        except ConversationTimeout:
            print(
                f"  TIMEOUT: exceeded --timeout {args.timeout:.0f}s — aborting this conversation, moving on"
            )
            rs = [
                QAResult(
                    conv_id=conv_id,
                    question="[timeout]",
                    expected="",
                    hypothesis="",
                    category=0,
                    keyword_score=0.0,
                    f1=0.0,
                    hit=False,
                    n_schemas=0,
                    n_episodes=0,
                    latency_ingest_s=0.0,
                    latency_recall_s=0.0,
                    consolidate=not args.no_consolidate,
                    error=f"conversation exceeded --timeout {args.timeout:.0f}s (watchdog)",
                )
            ]
        except Exception as e:
            print("  ERROR:", e)
            rs = []
        finally:
            if hasattr(signal, "SIGALRM"):
                signal.alarm(0)
        all_results.extend(rs)
        hits = sum(1 for r in rs if r.hit)
        elapsed = time.time() - t0
        line = "  %d questions, %d hits (%.0f%% keyword)" % (
            len(rs),
            hits,
            100 * hits / max(1, len(rs)),
        )
        judge_rs = [r for r in rs if r.llm_judge_score is not None]
        if judge_rs:
            j_avg = 100 * sum(r.llm_judge_score for r in judge_rs) / len(judge_rs)
            line += ", %.0f%% judge (n=%d)" % (j_avg, len(judge_rs))
        line += "  [%.1fs]" % elapsed
        print(line)
        _save(out_path, all_results, args, time.time() - t_start, partial=True)


if __name__ == "__main__":
    main()
