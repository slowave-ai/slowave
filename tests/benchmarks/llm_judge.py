"""Shared LLM-judge plumbing for benchmark scripts that score with an LLM.

Used by beam_eval.py (per-rubric-nugget judging) and by locomo_eval.py /
longmemeval_eval.py's optional --judge-model LLM-equivalence scoring pass
(Phase 3.5 — see private/docs/iterations/20260712_benchmarking_strategy_review.md).
Extracted here so both share one implementation instead of drifting.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from typing import Any


class QuotaExhausted(Exception):
    """Raised when the LLM API returns 402 (insufficient credits)."""


class InvalidModel(Exception):
    """Raised when the provider rejects the requested model ID (a permanent
    config error — retrying will not help)."""


# $ per million tokens (prompt, completion) on OpenRouter, for models actually
# used as judge/answerer in this project. Verified 2026-08-11 via openrouter.ai
# — re-check there if pricing looks stale by the time you read this.
KNOWN_MODEL_PRICING_PER_MTOK: dict[str, tuple[float, float]] = {
    "openai/gpt-5.6-luna": (0.20, 1.20),
    "openai/gpt-4o-mini": (0.15, 0.60),
    "deepseek/deepseek-v4-flash": (0.077, 0.154),
    "deepseek/deepseek-v4-pro": (0.435, 0.87),
}


def confirm_paid_run(
    description: str, estimated_cost_usd: float | None, *, assume_yes: bool
) -> None:
    """Block on an interactive y/N confirmation before a run that makes paid
    API calls. Exits the process (code 1) if the user declines. `assume_yes`
    is a no-op passthrough for --yes / -y flags, and for when a parent
    orchestrator (e.g. run_full_benchmark.py) already confirmed once for the
    whole suite and doesn't want each sub-script to prompt again."""
    if assume_yes:
        return
    print()
    print("=" * 72)
    print(" PAID API CALLS AHEAD")
    print("=" * 72)
    print(f" {description}")
    if estimated_cost_usd is not None:
        print(f" Estimated cost: ${estimated_cost_usd:.2f}")
    else:
        print(
            " Estimated cost: unknown (model not in local pricing table — check openrouter.ai/models)"
        )
    print("=" * 72)
    try:
        answer = input(" Continue? [y/N] ").strip().lower()
    except EOFError:
        answer = ""
    if answer not in ("y", "yes"):
        print(" Aborted — no API calls made.")
        sys.exit(1)


def estimate_cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float | None:
    """Real $ cost for `model` at known OpenRouter rates, or None if `model`
    isn't in KNOWN_MODEL_PRICING_PER_MTOK. Returning None (rather than $0 or
    a same-shape number priced at an unrelated model) is deliberate — a run's
    console output should never imply "you spent nothing" when the model's
    price just isn't in this table."""
    pricing = KNOWN_MODEL_PRICING_PER_MTOK.get(model)
    if pricing is None:
        return None
    in_rate, out_rate = pricing
    return (prompt_tokens * in_rate + completion_tokens * out_rate) / 1_000_000


def _obfuscate_api_key(key: str) -> str:
    """Show enough of a key to tell accounts apart without exposing it."""
    if len(key) <= 12:
        return "*" * len(key)
    return f"{key[:8]}...{key[-4:]}"


def get_openai_client() -> Any:
    """Lazily import and return an OpenAI client configured for OpenRouter."""
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        print("ERROR: OPENROUTER_API_KEY env var is not set.", file=sys.stderr)
        print("  export OPENROUTER_API_KEY=sk-or-...", file=sys.stderr)
        sys.exit(1)
    print(f"Using OPENROUTER_API_KEY: {_obfuscate_api_key(api_key)}", flush=True)
    try:
        from openai import OpenAI  # type: ignore[import-untyped]
    except ImportError:
        print("ERROR: openai package not installed.", file=sys.stderr)
        print("  pip install openai", file=sys.stderr)
        sys.exit(1)
    return OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        timeout=120.0,
        max_retries=3,
    )


def call_llm(
    client: Any,
    model: str,
    system: str,
    user: str,
    *,
    temperature: float = 0.0,
    max_tokens: int = 512,
) -> tuple[str, int, int]:
    """Call LLM, return (text, prompt_tokens, completion_tokens).

    Retries transient API errors (malformed JSON, timeouts) up to 3 times
    with exponential backoff. Raises ``QuotaExhausted`` on 402; returns empty
    result on 400 (context too long) so the question is skipped gracefully.
    """
    from json import JSONDecodeError

    messages: list[dict] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})

    last_error = None
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            choice = resp.choices[0]
            text = choice.message.content or ""
            usage = resp.usage
            pt = usage.prompt_tokens if usage else 0
            ct = usage.completion_tokens if usage else 0
            return text.strip(), pt, ct
        except (JSONDecodeError, OSError, ConnectionError, TimeoutError) as e:
            last_error = e
            if attempt < 2:
                wait = 2**attempt
                time.sleep(wait)
                continue
        except Exception as e:
            # Check for known API error codes from openai package
            err_str = str(e)
            if "402" in err_str or "Insufficient credits" in err_str:
                raise QuotaExhausted(str(e)) from e
            if "not a valid model" in err_str.lower() or "model id" in err_str.lower():
                raise InvalidModel(
                    f"Model '{model}' was rejected by the provider ({err_str}). "
                    "Check the model ID — e.g. 'deepseek/deepseek-v4-flash' "
                    "(no '-latest' suffix)."
                ) from e
            if "400" in err_str and (
                "context length" in err_str.lower() or "maximum context" in err_str.lower()
            ):
                # Context too long — return empty, don't crash the whole run
                print(
                    "\n  [WARN] Context too long for model, skipping question",
                    file=sys.stderr,
                    flush=True,
                )
                return "", 0, 0
            # Other non-retriable errors — re-raise
            raise

    raise RuntimeError(
        f"call_llm failed after 3 retries: {type(last_error).__name__}: {last_error}"
    ) from last_error


def parse_judge_response(raw: str) -> tuple[float, str] | None:
    """Extract (score, reason) from a judge completion, or None if unparseable."""
    # Strategy 1: find first balanced {..} JSON block
    try:
        depth = 0
        start = -1
        for i, ch in enumerate(raw):
            if ch == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0 and start >= 0:
                    json_str = raw[start : i + 1]
                    parsed = json.loads(json_str)
                    score = float(parsed.get("score", 0.0))
                    reason = str(parsed.get("reason", ""))
                    return max(0.0, min(1.0, score)), reason
    except (json.JSONDecodeError, ValueError):
        pass

    # Strategy 2: regex-based JSON extraction (handles code fences)
    for pattern in [
        r"```(?:json)?\s*(\{.*?\})\s*```",  # code-fenced JSON
        r'(\{\s*"score"\s*:\s*[\d.]+\s*,\s*"reason"\s*:\s*".*?"\s*\})',  # strict format
    ]:
        match = re.search(pattern, raw, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(1))
                score = float(parsed.get("score", 0.0))
                reason = str(parsed.get("reason", ""))
                return max(0.0, min(1.0, score)), reason
            except (json.JSONDecodeError, ValueError):
                continue

    # Strategy 3: find any {..} with a "score" key
    for m in re.finditer(r'\{[^{}]*"score"\s*:\s*([\d.]+)[^{}]*\}', raw, re.DOTALL):
        try:
            score = float(m.group(1))
            if 0.0 <= score <= 1.0:
                return score, raw[:200]
        except ValueError:
            continue

    # No further fallback. The prompt demands a strict JSON object, and a judge
    # that returns prose with a stray "0"/"1" (e.g. "0 of 3 facts present") must
    # not have that digit silently treated as a score. Anything that doesn't
    # parse as a `score` field is an explicit parse failure, which the caller
    # records (parse_ok=False) and excludes from the score average.

    return None


# ── LoCoMo / LongMemEval: single-reference-answer equivalence judging ──

ANSWER_EQUIVALENCE_JUDGE_SYSTEM_PROMPT = """Evaluate whether the RETRIEVED CONTEXT below correctly supports answering the QUESTION, by comparing it to the REFERENCE ANSWER (known correct).

SCORING:
- 1.0: The retrieved context contains information that correctly answers the question — semantically equivalent to the reference answer. Paraphrases, synonyms, different units/formats, and extra irrelevant content are all fine.
- 0.0: The retrieved context does not contain the correct answer, is off-topic, or contradicts the reference answer.

RULES:
1. Semantic tolerance: paraphrases and synonyms are acceptable.
2. Numeric/date equivalence: "$68,000" = "68k", "2 years" = "24 months".
3. Case/punctuation/whitespace differences: ignore.
4. The context may contain lots of irrelevant extra material — grade only on whether the correct answer's information is present and correct somewhere in it.
5. This is a binary judgment — score exactly 0.0 or 1.0, no partial credit.

Output ONLY the JSON object below as your entire response — no reasoning,
no chain-of-thought, no preamble before it. Reasoning-heavy models that
think out loud before the JSON risk exhausting their token budget and
returning nothing parseable; keep any reasoning inside the "reason" field.

Return EXACTLY this JSON format (no markdown, no code fences):
{"score": <0.0|1.0>, "reason": "<one sentence>"}"""


def judge_answer_equivalence(
    client: Any,
    judge_model: str,
    question: str,
    hypothesis: str,
    expected_answer: str,
    *,
    max_hypothesis_chars: int = 60_000,
) -> tuple[float, str, int, int, bool]:
    """Judge whether `hypothesis` (Slowave's retrieved context) semantically
    supports the correct answer to `question`, vs. `expected_answer`.

    Mirrors the lenient semantic-equivalence grading Mem0/Zep use, as opposed
    to Slowave's default strict keyword-overlap scorer — this grades the same
    retrieved-context "hypothesis" the keyword scorer already computes (up to
    a char cap — see below), not a separately generated answer, so it stays a
    retrieval-quality measure, not an end-to-end (retrieval + generation) one
    like Mem0/Zep's published numbers. See Phase 3.5 in the benchmarking
    strategy doc for the caveat.

    `hypothesis` is `schemas_text + " " + episodes_text` — schemas first.
    An earlier version of this function capped at 8,000 chars for cost
    predictability, without checking that schemas alone can already exceed
    that (measured: 8,007 chars of schemas alone on a real LoCoMo question at
    top_k=20) — meaning the judge saw zero episode content while the keyword
    scorer saw everything, and every fact that only lived in an episode was
    scored as "not present" by the judge. Confirmed on a real run: 964/1542
    questions had keyword_hit=True but llm_judge_score=0.0, almost all citing
    missing content that was actually in the (never-seen) episodes. Real
    measured full-hypothesis lengths run 43K-50K+ chars at top_k=20, so the
    cap is now 60K — comfortably above the measured range rather than a
    guess. The keyword-overlap scorer still sees the full untruncated text;
    only the LLM-judge pass is capped, and only as an outlier safety net now.

    Retries once with a larger token budget on parse failure — same pattern
    as BEAM's per-nugget judging, where the dominant failure mode was an
    empty completion from the judge exhausting its token budget on reasoning
    before the JSON payload.

    Returns (score, reason, prompt_tokens, completion_tokens, parse_ok).
    """
    if len(hypothesis) > max_hypothesis_chars:
        hypothesis = hypothesis[:max_hypothesis_chars] + "…"
    user = (
        "QUESTION:\n" + question + "\n\n"
        "REFERENCE ANSWER (known correct):\n" + expected_answer + "\n\n"
        "RETRIEVED CONTEXT TO GRADE:\n" + hypothesis
    )
    raw, pt, ct = call_llm(
        client,
        judge_model,
        ANSWER_EQUIVALENCE_JUDGE_SYSTEM_PROMPT,
        user,
        temperature=0.0,
        max_tokens=256,
    )
    parsed = parse_judge_response(raw)
    if parsed is not None:
        score, reason = parsed
        return score, reason, pt, ct, True

    raw2, pt2, ct2 = call_llm(
        client,
        judge_model,
        ANSWER_EQUIVALENCE_JUDGE_SYSTEM_PROMPT,
        user,
        temperature=0.0,
        max_tokens=2048,
    )
    total_pt, total_ct = pt + pt2, ct + ct2
    parsed2 = parse_judge_response(raw2)
    if parsed2 is not None:
        score, reason = parsed2
        return score, reason, total_pt, total_ct, True

    return 0.0, f"parse error: {raw2[:100]}", total_pt, total_ct, False


def judge_batch_concurrent(
    client: Any,
    judge_model: str,
    jobs: list[tuple[str, str, str]],
    *,
    max_workers: int = 8,
    min_interval_s: float = 15.0,
) -> list[tuple[float, str, int, int, bool]]:
    """Run `judge_answer_equivalence` over `jobs` (question, hypothesis,
    expected_answer) concurrently via a thread pool, returning results in the
    same order as `jobs`.

    The judge call is a pure network round-trip over already-computed text —
    it never touches the Slowave engine — so it's safe to parallelize even
    though the retrieval/consolidation work that produced `hypothesis` had to
    run sequentially. Without this, a judge-enabled run is effectively
    single-threaded network I/O: one question waits for the previous
    question's judge call to finish before starting its own, turning what
    should be a few-minute benchmark into tens of minutes.

    Uses threads, not processes — the OpenAI SDK client is safe to share
    across threads, and this is I/O-bound (waiting on HTTP responses), so
    there's no GIL contention to worry about.

    Prints a progress line with elapsed time, throughput, and an ETA — every
    ~10% of the batch or every `min_interval_s` seconds, whichever comes
    first. LongMemEval defers all judging to a single end-of-run batch (up
    to 500 jobs), so without periodic, informative output this phase runs
    silently for many minutes with no way to tell it apart from a hang.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from time import time as _now

    if not jobs:
        return []

    total = len(jobs)
    results: list[tuple[float, str, int, int, bool] | None] = [None] * total
    print_every = max(1, total // 10)
    start = _now()
    last_print = start
    done = 0

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(judge_answer_equivalence, client, judge_model, q, h, a): i
            for i, (q, h, a) in enumerate(jobs)
        }
        for future in as_completed(futures):
            i = futures[future]
            results[i] = future.result()
            done += 1
            now = _now()
            if done == total or done % print_every == 0 or now - last_print >= min_interval_s:
                elapsed = now - start
                rate = done / elapsed if elapsed > 0 else 0.0
                eta_s = (total - done) / rate if rate > 0 else None
                eta = f"{eta_s:.0f}s" if eta_s is not None else "?"
                print(
                    f"  [judge] {done}/{total} ({100 * done / total:.0f}%)  "
                    f"elapsed={elapsed:.0f}s  rate={rate:.1f}/s  eta={eta}",
                    flush=True,
                )
                last_print = now

    return results  # type: ignore[return-value]
