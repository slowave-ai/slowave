"""Run AML's public answerer -> judge contract over a Slowave result artifact.

This is an offline evaluation adapter, not part of Slowave's runtime memory
loop. It consumes the retrieved ``hypothesis`` already saved by LoCoMo or
LongMemEval, asks an answer model to produce a minimal answer using AML's
published prompt, and asks a judge model to grade that answer using AML's
published binary accuracy prompt.

Example:

    python tests/benchmarks/aml_answer_eval.py \
      --input data/locomo/runs/example.json \
      --answer-model openai/gpt-4o-mini \
      --judge-model openai/gpt-4o-mini
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from tests.benchmarks.llm_judge import (  # noqa: E402
    call_llm,
    confirm_paid_run,
    estimate_cost_usd,
    get_openai_client,
)

# Public AML LoCoMo-Refined / LongMemEval-S contract, retrieved 2026-08-11:
# https://github.com/AML-memory/agent-memory-leaderboard/tree/main/data
AML_OPEN_ENDED_ANSWER_TEMPLATE = """You are asked to answer a question based on your memories of a conversation.

<instructions>
1. Use only the provided memories. Prefer the memory that answers the question most directly.
2. Your memories are episodic raw observations. Reason about what they imply. Do not refuse just because the answer is not stated verbatim.
3. The question may contain typos. Match it to the most relevant memory even if the wording differs.
4. When multiple answers are possible, list all supported answers, not just the first.
5. For counts or time intervals, enumerate carefully before answering.
6. Preserve specific names, titles, places, and labels from the memories. Use "Rob" not "a colleague", "Sweden" not "home country".
7. Convert relative times like "yesterday", "last month", and "last year" into dates, months, or years when the memory timestamp makes it clear. Keep week-based expressions relative.
8. If memories conflict, prefer the most recent supported memory.
9. For list questions, include all required items and no extras.
10. Keep the final answer minimal. Do not add explanation, background, or extra dates unless needed for correctness.
</instructions>

<memories>
Memories for user {{speaker_1_name}}:

{{speaker_1_memories}}

Memories for user {{speaker_2_name}}:

{{speaker_2_memories}}
</memories>

Question: {{question}}
Answer with the shortest correct phrase or sentence. No preamble, no fluff:"""


AML_ACCURACY_PROMPT = """Your task is to label an answer as ’CORRECT’ or ’WRONG’ given:
(1) a question,
(2) a gold (ground truth) answer,
(3) a generated answer.

Core principle — Inclusion + Non-contradiction
- Be GENEROUS: if the generated answer clearly includes the gold’s key content (or a clear paraphrase of the same content) and does not contradict it, mark CORRECT — even if extra details are added.
- Mark WRONG only when the generated answer does not include the gold’s content, changes it, or contradicts it.

TIME (strict granularity; relative form equivalence; no calendar math)
- Granularity must match exactly: HOUR↔HOUR, DAY↔DAY, MONTH↔MONTH, YEAR↔YEAR.
  Do not answer a gold at a different time unit — even if the numeric value overlaps. Do not answer a month-level gold with a specific day, nor a year with a specific month/day/hour, etc.
  (e.g., gold = "July 26, 2019" [DAY]; generated = "2019-07-26 08:09:17" [includes Second] → WRONG)
- Do NOT convert relative ↔ absolute. If the gold uses a relative time expression, the generated answer must also use a relative form (or a clear paraphrase of that same form), not a computed date/range.
- Treat harmless modifiers in relative forms (e.g., “the/last/previous/just prior”) as equivalent when both the anchor date and the time unit are the same.

- Lists of DISTINCT facts:
- If the gold answer lists multiple distinct facts (joined by "and", commas, or slashes), the generated answer must cover **all** of them.
- Extra non-contradictory items **generally count as WRONG**.
    - Example: gold = A, B, C ; gen = A, B, C → CORRECT
    - Example: gold = A, B, C ; gen = A, B, C, D → WRONG
- Exception: If a gold element is elaborated or split into finer details in the generated answer (e.g., C → C, C′), it is still considered CORRECT.

Preference/Benefit Questions (e.g., "what X likes/values most")
- If gold lists multiple reasons/aspects, the generated answer only needs to include **any one** of them without contradiction to be CORRECT.

Now it's time for the real question:
Question: {question}
Gold answer: {gold_answer}
Generated answer: {generated_answer}

First, provide a short (one sentence) explanation of your reasoning, then finish with CORRECT or WRONG.
Do NOT include both CORRECT and WRONG in your response, or it will break the evaluation script.

Just return the label CORRECT or WRONG in a json format with the key as "label":

```json
{{
    "label": "CORRECT" or "WRONG"
}}
```"""


def render_answer_prompt(
    question: str,
    memories: str,
    *,
    speaker_1_name: str = "speaker 1",
    speaker_2_name: str = "speaker 2",
    speaker_2_memories: str = "",
) -> str:
    values = {
        "speaker_1_name": speaker_1_name,
        "speaker_1_memories": memories,
        "speaker_2_name": speaker_2_name,
        "speaker_2_memories": speaker_2_memories,
        "question": question,
    }
    return re.sub(
        r"\{\{(speaker_1_name|speaker_1_memories|speaker_2_name|speaker_2_memories|question)\}\}",
        lambda match: values[match.group(1)],
        AML_OPEN_ENDED_ANSWER_TEMPLATE,
    )


def render_accuracy_prompt(question: str, gold_answer: str, generated_answer: str) -> str:
    values = {
        "question": question,
        "gold_answer": gold_answer,
        "generated_answer": generated_answer,
    }
    return re.sub(
        r"\{(question|gold_answer|generated_answer)\}",
        lambda match: values[match.group(1)],
        AML_ACCURACY_PROMPT,
    )


def parse_aml_label(raw: str) -> str | None:
    """Return AML's CORRECT/WRONG label, or None for an invalid response."""
    for match in re.finditer(r"\{.*?\}", raw, re.DOTALL):
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            continue
        label = str(payload.get("label", "")).upper()
        if label in {"CORRECT", "WRONG"}:
            return label
    return None


def evaluate_row(
    client: Any,
    row: dict[str, Any],
    answer_model: str,
    judge_model: str,
) -> dict[str, Any]:
    question = str(row["question"])
    gold = str(row["expected"])
    memories = str(row["hypothesis"])
    answer_prompt = render_answer_prompt(question, memories)
    generated, answer_pt, answer_ct = call_llm(
        client, answer_model, "", answer_prompt, temperature=0.0, max_tokens=256
    )
    if not generated:
        return {
            "generated_answer": "",
            "label": "WRONG",
            "is_correct": False,
            "judge_response": "",
            "parse_ok": False,
            "error": "answer generation returned no content",
            "answer_prompt_tokens": answer_pt,
            "answer_completion_tokens": answer_ct,
            "judge_prompt_tokens": 0,
            "judge_completion_tokens": 0,
        }

    judge_prompt = render_accuracy_prompt(question, gold, generated)
    raw, judge_pt, judge_ct = call_llm(
        client, judge_model, "", judge_prompt, temperature=0.0, max_tokens=256
    )
    label = parse_aml_label(raw)
    return {
        "generated_answer": generated,
        "label": label or "WRONG",
        "is_correct": label == "CORRECT",
        "judge_response": raw,
        "parse_ok": label is not None,
        "error": None if label is not None else "unparseable judge response",
        "answer_prompt_tokens": answer_pt,
        "answer_completion_tokens": answer_ct,
        "judge_prompt_tokens": judge_pt,
        "judge_completion_tokens": judge_ct,
    }


def eligible_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Select answerable retrieval rows from LoCoMo/LongMemEval artifacts."""
    rows = []
    for index, row in enumerate(payload.get("results", [])):
        expected = row.get("expected", row.get("expected_answer"))
        if row.get("error") or not row.get("question") or not expected:
            continue
        # LoCoMo category 5 is an adversarial retrieval check without a normal
        # reference-answer contract, matching locomo_eval's existing exclusion.
        if row.get("category") == 5:
            continue
        copied = dict(row)
        copied["expected"] = expected
        copied["_source_index"] = index
        rows.append(copied)
    return rows


def _cost(model: str, prompt: int, completion: int) -> float | None:
    return estimate_cost_usd(model, prompt, completion)


def save_output(
    path: Path,
    source: Path,
    args: argparse.Namespace,
    source_payload: dict[str, Any],
    results: list[dict[str, Any]],
    *,
    partial: bool,
    started: float,
) -> None:
    answer_pt = sum(r.get("answer_prompt_tokens", 0) for r in results)
    answer_ct = sum(r.get("answer_completion_tokens", 0) for r in results)
    judge_pt = sum(r.get("judge_prompt_tokens", 0) for r in results)
    judge_ct = sum(r.get("judge_completion_tokens", 0) for r in results)
    answer_cost = _cost(args.answer_model, answer_pt, answer_ct)
    judge_cost = _cost(args.judge_model, judge_pt, judge_ct)
    valid = [r for r in results if r.get("parse_ok")]
    payload = {
        "meta": {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "partial": partial,
            "source_artifact": str(source),
            "source_meta": source_payload.get("meta", {}),
            "answer_model": args.answer_model,
            "judge_model": args.judge_model,
            "protocol": "AML public LoCoMo-Refined/LongMemEval-S answer+accuracy prompts",
            "top_k": source_payload.get("meta", {}).get("top_k"),
            "elapsed_s": round(time.time() - started, 2),
        },
        "summary": {
            "n": len(results),
            "valid_n": len(valid),
            "correct": sum(1 for r in valid if r["is_correct"]),
            "score_pct": (
                round(100 * sum(1 for r in valid if r["is_correct"]) / len(valid), 2)
                if valid
                else None
            ),
            "parse_errors": sum(1 for r in results if not r.get("parse_ok")),
            "answer_prompt_tokens": answer_pt,
            "answer_completion_tokens": answer_ct,
            "judge_prompt_tokens": judge_pt,
            "judge_completion_tokens": judge_ct,
            "answer_cost_usd": answer_cost,
            "judge_cost_usd": judge_cost,
            "total_cost_usd": (
                answer_cost + judge_cost
                if answer_cost is not None and judge_cost is not None
                else None
            ),
        },
        "results": results,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def default_output_path(source: Path, answer_model: str, judge_model: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    answer_name = answer_model.rsplit("/", 1)[-1]
    judge_name = judge_model.rsplit("/", 1)[-1]
    return source.parent / f"{source.stem}_aml_{answer_name}_{judge_name}_{stamp}.json"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply AML's public answerer+judge contract to a Slowave benchmark artifact"
    )
    parser.add_argument("--input", required=True, help="LoCoMo or LongMemEval result JSON")
    parser.add_argument("--out", default="")
    parser.add_argument("--answer-model", default="openai/gpt-4o-mini")
    parser.add_argument("--judge-model", default="openai/gpt-4o-mini")
    parser.add_argument("--limit", type=int, default=0, help="Maximum eligible rows (0=all)")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--yes", "-y", action="store_true")
    args = parser.parse_args()

    source = Path(args.input).resolve()
    source_payload = json.loads(source.read_text(encoding="utf-8"))
    rows = eligible_rows(source_payload)
    if args.limit > 0:
        rows = rows[: args.limit]
    if not rows:
        raise SystemExit("No eligible answerable rows in input artifact")
    if source_payload.get("meta", {}).get("evidence_format") != "structured_v1":
        raise SystemExit(
            "Input does not use structured_v1 dated evidence. Re-run the source "
            "benchmark with the current code and --save-full-hypotheses before "
            "AML-style evaluation."
        )
    if (
        not source_payload.get("meta", {}).get("full_hypotheses")
        and max(len(str(row["hypothesis"])) for row in rows) <= 400
    ):
        raise SystemExit(
            "Input contains only 400-character hypothesis previews. Re-run the source "
            "benchmark with --save-full-hypotheses before AML-style evaluation."
        )

    # Conservative preflight estimate. Actual token usage and separate model
    # costs are persisted in the output artifact.
    avg_context_tokens = max(1, sum(len(str(r["hypothesis"])) for r in rows) // 4 // len(rows))
    answer_cost = _cost(args.answer_model, len(rows) * (avg_context_tokens + 700), len(rows) * 100)
    judge_cost = _cost(args.judge_model, len(rows) * 900, len(rows) * 40)
    estimated = (
        answer_cost + judge_cost if answer_cost is not None and judge_cost is not None else None
    )
    confirm_paid_run(
        f"AML-style evaluation will make {len(rows)} answer calls with {args.answer_model} "
        f"and {len(rows)} judge calls with {args.judge_model}.",
        estimated,
        assume_yes=args.yes,
    )

    client = get_openai_client()
    output = (
        Path(args.out).resolve()
        if args.out
        else default_output_path(source, args.answer_model, args.judge_model)
    )
    started = time.time()
    completed: list[dict[str, Any] | None] = [None] * len(rows)
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(evaluate_row, client, row, args.answer_model, args.judge_model): i
                for i, row in enumerate(rows)
            }
            for done, future in enumerate(as_completed(futures), start=1):
                index = futures[future]
                result = future.result()
                source_row = rows[index]
                completed[index] = {
                    "source_index": source_row["_source_index"],
                    "conv_id": source_row.get("conv_id"),
                    "question_id": source_row.get("question_id", source_row.get("id")),
                    "category": source_row.get("category"),
                    "question": source_row["question"],
                    "expected": source_row["expected"],
                    "hypothesis": source_row["hypothesis"],
                    **result,
                }
                if done == len(rows) or done % max(1, len(rows) // 10) == 0:
                    print(
                        f"  [AML] {done}/{len(rows)} ({100 * done / len(rows):.0f}%)",
                        flush=True,
                    )
    except (KeyboardInterrupt, Exception):
        partial_results = [r for r in completed if r is not None]
        save_output(
            output,
            source,
            args,
            source_payload,
            partial_results,
            partial=True,
            started=started,
        )
        print(f"Partial results saved to {output}", file=sys.stderr)
        raise

    results = [r for r in completed if r is not None]
    save_output(output, source, args, source_payload, results, partial=False, started=started)
    valid = [r for r in results if r["parse_ok"]]
    correct = sum(1 for r in valid if r["is_correct"])
    print(f"AML-style score: {100 * correct / max(1, len(valid)):.2f}% ({correct}/{len(valid)})")
    print(f"Results saved to: {output}")


if __name__ == "__main__":
    main()
