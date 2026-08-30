#!/usr/bin/env python3
"""BEAM evaluation harness for Slowave.

BEAM (ICLR 2026) — 100 conversations, 2,000 probing questions across
10 memory ability categories at 4 chat sizes (100K–10M tokens).

Unlike other Slowave benchmarks, BEAM requires LLM API calls for answer
generation and LLM-as-judge scoring. Reads OPENROUTER_API_KEY from env.

Cost estimate (BEAM 1M, ~2,000 questions):
  GPT-5:         ~$25–35  (answerer + judge, via OpenRouter)
  GPT-5-mini:    ~$2–4    (answerer + judge, via OpenRouter)

Usage:
  # Full BEAM 1M run (~2,000 questions, ~$30 GPT-5 cost)
  python tests/benchmarks/beam_eval.py

  # Smoke test — 3 conversations, GPT-5-mini (~$0.50)
  python tests/benchmarks/beam_eval.py --limit 3 --answerer-model deepseek/deepseek-v4-flash --judge-model deepseek/deepseek-v4-flash

  # Specific chat sizes
  python tests/benchmarks/beam_eval.py --chat-sizes 500K,1M --limit 5

  # Episode-only baseline (no consolidation)
  python tests/benchmarks/beam_eval.py --no-consolidate
"""

from __future__ import annotations

import argparse
import ast
import dataclasses
import hashlib
import json
import logging
import os
import re
import sys
import tempfile
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s %(message)s")
for _noisy in (
    "sentence_transformers",
    "transformers",
    "httpx",
    "httpcore",
    "huggingface_hub",
    "filelock",
    "tqdm",
    "openai",
):
    logging.getLogger(_noisy).setLevel(logging.ERROR)

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# Prevent ONNX Runtime from saturating all CPUs during model optimization.
# onnx_encoder.py hardcodes intra_op_num_threads=os.cpu_count();
# ORT inter-op threads control the outer thread pool and cap total concurrency.
os.environ.setdefault("ORT_NUM_THREADS", "2")

# Load .env from repo root (e.g. OPENROUTER_API_KEY)
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_env = REPO_ROOT / ".env"
if _env.exists():
    try:
        from dotenv import load_dotenv

        load_dotenv(_env, override=True)
    except ImportError:
        pass
sys.path.insert(0, str(REPO_ROOT))

from slowave.core.config import SlowaveConfig
from slowave.core.engine import SlowaveEngine
from slowave.latent.replay_engine import ReplayConfig
from slowave.latent.retrieval import RetrievalConfig
from slowave.latent.salience import SalienceConfig
from slowave.symbolic.encoder import EncoderConfig, TextEncoder
from tests.benchmarks.llm_judge import (
    QuotaExhausted,
    call_llm,
    confirm_paid_run,
    estimate_cost_usd,
    get_openai_client,
    parse_judge_response,
)
from tests.benchmarks.report_format import print_footer, print_header
from tests.benchmarks.retrieval_metrics import (
    aggregate_recall_at_k_mrr,
    compute_recall_at_k_and_mrr,
)

# ── Constants ─────────────────────────────────────────────────────────

BEAM_QUESTION_TYPES: dict[str, str] = {
    "abstention": "Withholding answers when evidence is absent",
    "contradiction_resolution": "Detecting and reconciling inconsistent statements",
    "event_ordering": "Reconstructing chronological sequence of events",
    "information_extraction": "Recalling specific entities, dates, numbers, facts",
    "instruction_following": "Sustained adherence to user constraints and formatting",
    "knowledge_update": "Revising stored facts when new information appears",
    "multi_session_reasoning": "Integrating evidence across non-adjacent segments",
    "preference_following": "Adapting responses to evolving user preferences",
    "summarization": "Abstracting and compressing dialogue content",
    "temporal_reasoning": "Reasoning about time relations, durations, sequences",
}

HF_DATASET_1M = "Mohammadta/BEAM"
HF_DATASET_10M = "Mohammadta/BEAM-10M"
VALID_CHAT_SIZES = ["100K", "500K", "1M", "10M"]

RECALL_HIT_THRESHOLD = 0.5
_RECALL_STOP = {
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


def _recall_keyword_score(hypothesis: str, answer: str) -> float:
    """Same keyword-overlap scorer used by locomo/longmemeval/dmr_original_eval.py.

    Only used for the retrieval-side Recall@K/MRR metric below, not for the
    LLM-judge score BEAM actually reports — kept separate so a change to one
    doesn't silently affect the other.
    """

    def tokens(s: str) -> set[str]:
        return {
            w
            for w in re.findall(r"[a-z0-9]+", str(s).lower())
            if w not in _RECALL_STOP and (len(w) > 1 or w.isdigit())
        }

    answer_tokens = tokens(answer)
    if not answer_tokens:
        return 0.0
    return len(answer_tokens & tokens(hypothesis)) / len(answer_tokens)


# ── LLM Prompts ───────────────────────────────────────────────────────

ANSWER_SYSTEM_PROMPT = """You are an AI assistant with access to a memory system.
You will receive memories retrieved from prior conversations along with a question.

IMPORTANT RULES:
1. Scan ALL provided memories carefully before answering.
2. If multiple memories contain relevant information, combine them.
3. If memories contradict each other, prefer the more recent one.
4. If memories don't contain enough information to answer, say exactly:
   "I don't have enough information to answer this question."
5. For temporal questions: pay close attention to dates and relative time references.
6. For ordering questions: present events in chronological order.
7. For preference questions: use the most recently stated preference.
8. Be specific — include exact names, dates, and numbers from the memories.
9. Do NOT invent or assume information that isn't in the memories.
10. Answer the question directly. Do not describe what the memories contain —
    use them to answer."""

ANSWER_USER_TEMPLATE = """QUESTION:
{question}

RETRIEVED MEMORIES:
{memories}

ANSWER:"""


def _format_memories(
    schemas: list,
    episode_texts: list[dict],
    max_chars: int = 40_000,
) -> str:
    """Format retrieved schemas and episodes as a structured, readable block.

    Applies a HARD character budget across all content. Schemas get up to 4K
    chars each (they're more information-dense than raw episodes). Episodes get
    up to 2K chars each. The total output never exceeds ``max_chars``.
    """
    MAX_SCHEMA_CHARS = 4_000  # per-schema cap
    MAX_EPISODE_CHARS = 2_000  # per-episode cap

    all_items: list[tuple[str, int]] = []  # (text, char_limit)

    if schemas:
        all_items.append(("─── SCHEMAS (stable extracted knowledge) ───\n", 0))
        for i, s in enumerate(schemas, 1):
            raw = (s.content_text or "").strip()
            if raw:
                text = raw if len(raw) <= MAX_SCHEMA_CHARS else raw[:MAX_SCHEMA_CHARS] + "…"
                all_items.append((f"  [{i}] {text}\n", MAX_SCHEMA_CHARS))

    if episode_texts:
        all_items.append(("\n─── EPISODES (raw conversation excerpts) ───\n", 0))
        for i, ep in enumerate(episode_texts, 1):
            raw = (ep.get("content_text") or "").strip()
            if raw:
                text = raw if len(raw) <= MAX_EPISODE_CHARS else raw[:MAX_EPISODE_CHARS] + "…"
                all_items.append((f"  [{i}] {text}\n", MAX_EPISODE_CHARS))

    if not all_items:
        return "(no stored memories found for this query)"

    # Build output with hard budget — headers always included, entries truncated
    result: list[str] = []
    chars_used = 0
    emitted_entries = 0
    skipped = 0
    budget = max_chars - 200  # reserve for truncation notice

    for text, _limit in all_items:
        if chars_used + len(text) <= budget:
            result.append(text)
            chars_used += len(text)
            if _limit > 0:  # not a header
                emitted_entries += 1
        elif _limit == 0:
            # Header that doesn't fit — still emit it (headers are small)
            result.append(text)
            chars_used += len(text)
        else:
            skipped += 1

    if skipped > 0 and emitted_entries > 0:
        result.append(f"  … ({skipped} more entries omitted, limit {max_chars} chars)")

    out = "".join(result).rstrip()
    return out if out else "(no stored memories found for this query)"


JUDGE_SYSTEM_PROMPT = """Evaluate whether the LLM response complies with the RUBRIC CRITERION below.

SCORING:
**POSITIVE requirement** (response SHOULD include something):
- 1.0: Required element present, accurate, complete.
- 0.5: Partially present, minor inaccuracies, or incomplete.
- 0.0: Missing, incorrect, or off-topic.

**NEGATIVE constraint** (response SHOULD NOT include something):
- 1.0: Responsive AND prohibited element absent.
- 0.5: Responsive but borderline reference to prohibited element.
- 0.0: Prohibited element present, OR non-responsive.

**Compound statements** (multiple elements joined by "and" or commas):
- All present = 1.0, some present = 0.5, none present = 0.0.

RULES:
1. Semantic tolerance: paraphrases and synonyms are acceptable.
2. Numeric/date equivalence: "$68,000" = "68k", "2 years" = "24 months".
3. Case/punctuation/whitespace differences: ignore.
4. Hedging tolerance: "I think", "probably" are fine if substance is correct.
5. Style neutrality: don't penalize tone, formatting, or length.
6. If response is off-topic or refuses to answer → score 0.0.
7. Evaluate this criterion in isolation.
8. Vague answers score lower than specific, detailed answers.

Return EXACTLY this JSON format (no markdown, no code fences):
{"score": <0.0|0.5|1.0>, "reason": "<one sentence>"}"""


def _judge_user_prompt(question: str, response: str, rubric: str) -> str:
    return (
        "QUESTION:\n" + question + "\n\n"
        "LLM RESPONSE:\n" + response + "\n\n"
        "RUBRIC CRITERION:\n" + rubric
    )


# ── BEAM dataset loading ───────────────────────────────────────────────


def _load_beam_dataset(
    chat_sizes: list[str],
    cache_dir: str,
) -> dict[str, list[dict]]:
    """Load BEAM from HuggingFace Parquet with per-conversation JSON cache.

    Uses huggingface-hub + pyarrow to download parquet files and caches
    each conversation as a separate JSON file to avoid loading 173MB at once.

    Returns dict: chat_size -> list[conversation_dict].
    """
    os.makedirs(cache_dir, exist_ok=True)
    dataset: dict[str, list[dict]] = {}

    for size in chat_sizes:
        size_dir = Path(cache_dir) / f"beam_{size}"
        # Check if per-conversation cache exists
        if size_dir.exists() and list(size_dir.glob("*.json")):
            convs: list[dict] = []
            for cf in sorted(size_dir.glob("*.json")):
                with open(cf, encoding="utf-8") as f:
                    convs.append(json.load(f))
            dataset[size] = convs
            print(f"  Loaded {len(convs)} cached {size} conversations")
            continue

        # Check legacy single-file cache, migrate to per-conversation
        legacy = Path(cache_dir) / f"beam_{size}.json"
        if legacy.exists():
            print(f"  Loading cached {size} from {legacy}, migrating to per-file...")
            with open(legacy, encoding="utf-8") as f:
                convs = json.load(f)
            size_dir.mkdir(parents=True, exist_ok=True)
            for i, c in enumerate(convs):
                with open(size_dir / f"{i:03d}.json", "w", encoding="utf-8") as cf:
                    json.dump(c, cf, ensure_ascii=False)
            dataset[size] = convs
            print(f"    Migrated {len(convs)} conversations to {size_dir}")
            continue

        print(f"  Downloading BEAM {size} from HuggingFace...")
        repo = HF_DATASET_10M if size == "10M" else HF_DATASET_1M
        split_name = "10M" if size == "10M" else size

        from huggingface_hub import hf_hub_download, list_repo_files

        try:
            import pyarrow.parquet as pq
        except ImportError:
            print("ERROR: pyarrow not installed. Run: pip install pyarrow", file=sys.stderr)
            sys.exit(1)

        files = list_repo_files(repo, repo_type="dataset")
        parquet_files = sorted(
            f for f in files if f.endswith(".parquet") and split_name.lower() in f.lower()
        )
        if not parquet_files:
            parquet_files = sorted(f for f in files if f.endswith(".parquet"))

        convs = []
        for pf in parquet_files:
            local = hf_hub_download(repo, pf, repo_type="dataset")
            table = pq.read_table(local)
            for row in table.to_pylist():
                convs.append(dict(row))

        dataset[size] = convs
        size_dir.mkdir(parents=True, exist_ok=True)
        for i, c in enumerate(convs):
            with open(size_dir / f"{i:03d}.json", "w", encoding="utf-8") as cf:
                json.dump(c, cf, ensure_ascii=False)
        print(f"    Cached {len(convs)} conversations to {size_dir}")

    return dataset


# ── Embedding cache ─────────────────────────────────────────────────────


def _get_embedding_cache_path(cache_dir: str, size: str) -> Path:
    """Return the path to the pre-computed embedding cache for a chat size."""
    return Path(cache_dir) / f"beam_{size.lower()}_embeddings.npz"


def _compute_content_hash(convs: list[dict]) -> str:
    """Stable hash of conversation data to detect changes and trigger rebuild."""
    # Serialize to canonical JSON (sorted keys) for stable ordering
    payload = json.dumps(convs, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _build_embedding_cache(
    convs: list[dict],
    enc: TextEncoder,
    cache_path: Path,
) -> dict[str, np.ndarray]:
    """Batch-encode all conversation messages and save to .npz cache.

    Returns a dict mapping ``{conv_id}:msg_{idx}`` → embedding array.
    """
    import numpy as _np

    texts: list[str] = []
    keys: list[str] = []

    for conv in convs:
        conv_id = str(conv.get("conversation_id", "?"))
        chat = conv.get("chat", [])
        msg_idx = 0
        for turn in chat:
            if not isinstance(turn, (list, tuple)):
                continue
            for msg in turn:
                if isinstance(msg, dict):
                    text = str(msg.get("content", "")).strip()
                else:
                    text = str(msg).strip()
                if not text:
                    continue
                texts.append(text)
                keys.append(f"{conv_id}:msg_{msg_idx}")
                msg_idx += 1

        # Metadata: user_profile and conversation_plan (answer-critical facts)
        up = conv.get("user_profile", {})
        if isinstance(up, dict):
            up_text = up.get("user_info", "")
            if isinstance(up_text, str) and up_text.strip():
                texts.append(up_text.strip())
                keys.append(f"{conv_id}:user_profile")

        plan = conv.get("conversation_plan", "")
        if isinstance(plan, str) and plan.strip():
            for pi, line in enumerate(plan.split("\n")):
                line = line.strip()
                if len(line) > 40:
                    texts.append(line)
                    keys.append(f"{conv_id}:plan_{pi}")

    if not texts:
        # Empty dataset: save a minimal placeholder
        content_hash = _compute_content_hash(convs)
        _np.savez(
            cache_path,
            content_hash=np.array([content_hash], dtype="<U128"),
            model_name=np.array([enc.cfg.model_name], dtype="<U256"),
            dim=np.array([enc.dim], dtype=np.int32),
            keys=np.array(["__empty__"]),
            embeddings=_np.zeros((1, enc.dim), dtype=np.float32),
        )
        return {}

    print(f"  Building embedding cache ({len(texts)} messages)...", flush=True)
    t0 = time.time()

    # Encode in chunks to avoid OOM (BEAM 1M = ~75K messages)
    # ONNX hidden state is [batch, 512, 384] ~ batch * 0.8 MB — keep batches small
    BATCH_SIZE = 200
    import gc as _gc

    all_embeddings: list[np.ndarray] = []
    for start in range(0, len(texts), BATCH_SIZE):
        end = min(start + BATCH_SIZE, len(texts))
        chunk_emb = enc.encode_many(texts[start:end])
        all_embeddings.append(chunk_emb)
        elapsed = time.time() - t0
        rate = end / elapsed if elapsed > 0 else 0
        print(
            f"\r  Building embedding cache ({len(texts)} messages)... "
            f"{end}/{len(texts)} ({rate:.0f} msg/s)",
            end="",
            flush=True,
        )
        _gc.collect()
    embeddings = np.concatenate(all_embeddings, axis=0)
    t1 = time.time() - t0
    print(f"\r  Building embedding cache ({len(texts)} messages)... done ({t1:.1f}s)")

    content_hash = _compute_content_hash(convs)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    _np.savez(
        cache_path,
        content_hash=np.array([content_hash], dtype="<U128"),
        model_name=np.array([enc.cfg.model_name], dtype="<U256"),
        dim=np.array([enc.dim], dtype=np.int32),
        keys=np.array(keys),
        embeddings=embeddings,
    )
    print(f"  Embedding cache saved → {cache_path}")

    # Build lookup dict
    return dict(zip(keys, embeddings))


def _load_embedding_cache(cache_path: Path) -> dict[str, np.ndarray] | None:
    """Load a pre-built embedding cache from .npz file. Returns None on failure."""
    import numpy as _np

    try:
        data = _np.load(cache_path, allow_pickle=False)
    except (FileNotFoundError, OSError, ValueError):
        return None

    keys = data["keys"]
    embeddings = data["embeddings"]
    # Handle empty placeholder
    if len(keys) == 1 and str(keys[0]) == "__empty__":
        return {}
    return dict(zip(keys, embeddings))


def _load_or_build_embedding_cache(
    convs: list[dict],
    enc: TextEncoder,
    size: str,
    cache_dir: str,
    force_rebuild: bool = False,
    skip_cache: bool = False,
) -> dict[str, np.ndarray] | None:
    """Auto-managed embedding cache: load if valid, build if missing/stale.

    Returns:
        ``dict`` of pre-computed embeddings, or ``None`` when ``skip_cache=True``.
    """
    if skip_cache:
        return None

    cache_path = _get_embedding_cache_path(cache_dir, size)
    current_hash = _compute_content_hash(convs)

    if force_rebuild:
        print(f"  Force-rebuilding embedding cache for BEAM {size}...")
        return _build_embedding_cache(convs, enc, cache_path)

    # Try loading existing cache
    cache = _load_embedding_cache(cache_path)
    if cache is not None:
        try:
            data = np.load(cache_path, allow_pickle=False)
            cached_hash = str(data["content_hash"][0])
            cached_model = str(data["model_name"][0])
            cached_dim = int(data["dim"][0])
        except (KeyError, IndexError, OSError, ValueError):
            print(f"  Cache corrupted, rebuilding for BEAM {size}...")
            return _build_embedding_cache(convs, enc, cache_path)

        hash_match = cached_hash == current_hash
        model_match = cached_model == enc.cfg.model_name
        dim_match = cached_dim == enc.dim

        if hash_match and model_match and dim_match:
            print(f"  Using cached embeddings ({len(cache)} messages) from {cache_path.name}")
            return cache
        else:
            reasons = []
            if not hash_match:
                reasons.append("content changed")
            if not model_match:
                reasons.append(f"model changed ({cached_model} → {enc.cfg.model_name})")
            if not dim_match:
                reasons.append(f"dim mismatch ({cached_dim} → {enc.dim})")
            print(f"  Cache stale ({', '.join(reasons)}), rebuilding for BEAM {size}...")
            return _build_embedding_cache(convs, enc, cache_path)

    # Cache file doesn't exist
    print(f"  No embedding cache found, building for BEAM {size}...")
    return _build_embedding_cache(convs, enc, cache_path)


# ── Probing questions ───────────────────────────────────────────────────


def _parse_probing_questions(raw: str) -> dict[str, list[dict]]:
    """Parse probing_questions string into dict[ability_type, list[question]]."""
    if not raw or not raw.strip():
        return {}
    try:
        return ast.literal_eval(raw)  # type: ignore[no-any-return]
    except (ValueError, SyntaxError):
        pass
    try:
        return json.loads(raw)  # type: ignore[no-any-return]
    except (json.JSONDecodeError, ValueError):
        pass
    print(f"WARNING: could not parse probing_questions: {raw[:120]}...", file=sys.stderr)
    return {}


def _flatten_questions(parsed: dict[str, list[dict]]) -> list[dict]:
    """Flatten ability-keyed dict to a single annotated list."""
    out: list[dict] = []
    for ability, qs in parsed.items():
        for q in qs:
            q = dict(q)
            q.setdefault("question_type", ability)
            out.append(q)
    return out


def _get_nuggets(question: dict) -> list[str]:
    """Extract rubric nuggets from a BEAM question."""
    rubric = question.get("rubric") or question.get("answer", "")
    if isinstance(rubric, list):
        return [str(r) for r in rubric if str(r).strip()]
    if isinstance(rubric, str) and rubric.strip():
        return [rubric]
    return []


# ── Ingestion ──────────────────────────────────────────────────────────


def _ingest_conversation(
    eng: SlowaveEngine,
    conv: dict,
    shared_encoder: TextEncoder,
    embedding_cache: dict[str, np.ndarray] | None = None,
) -> float:
    """Ingest a BEAM conversation into Slowave. Returns elapsed seconds.

    Ingests both chat messages AND metadata (user_profile, conversation_plan)
    since BEAM probing questions reference facts from all three fields.

    When ``embedding_cache`` is provided, embeddings are looked up by key
    instead of being recomputed.
    """
    t0 = time.time()
    conv_id = str(conv.get("conversation_id", "?"))

    # Collect all texts to encode (chat messages + metadata)
    to_encode: list[tuple[str, str, str]] = []  # (cache_key, event_type, text)

    chat = conv.get("chat", [])
    if chat:
        msg_idx = 0
        for turn in chat:
            if not isinstance(turn, (list, tuple)):
                continue
            for msg in turn:
                if isinstance(msg, dict):
                    text = str(msg.get("content", "")).strip()
                    role = str(msg.get("role", "")).strip()
                else:
                    text = str(msg).strip()
                    role = ""
                if not text:
                    continue
                if not role:
                    role = "user_message"
                to_encode.append((f"{conv_id}:msg_{msg_idx}", role, text))
                msg_idx += 1

    # Metadata: user_profile (has name, age, location, preferences)
    up = conv.get("user_profile", {})
    if isinstance(up, dict):
        up_text = up.get("user_info", "")
        if isinstance(up_text, str) and up_text.strip():
            to_encode.append((f"{conv_id}:user_profile", "user_profile", up_text.strip()))

    # Metadata: conversation_plan (has timeline, deadlines, milestones, dates)
    # Split into individual lines — each bullet point is a self-contained fact
    plan = conv.get("conversation_plan", "")
    if isinstance(plan, str) and plan.strip():
        for pi, line in enumerate(plan.split("\n")):
            line = line.strip()
            if len(line) > 40:  # skip trivial fragments and blank lines
                to_encode.append((f"{conv_id}:plan_{pi}", "conversation_plan", line))

    if not to_encode:
        return 0.0

    sid = eng.session_start(agent="beam_eval", scope=f"eval:beam:{conv_id}")
    for cache_key, ev_type, text in to_encode:
        if embedding_cache is not None and cache_key in embedding_cache:
            emb = embedding_cache[cache_key]
        else:
            emb = shared_encoder.encode(text)
        eng.raw_log.append(
            session_id=sid,
            type=ev_type,
            content=text,
            embedding=emb,
        )
    eng.session_end(sid, consolidate=False)
    return time.time() - t0


# ── Judging ────────────────────────────────────────────────────────────


def _judge_nugget(
    client: Any,
    judge_model: str,
    question: str,
    response: str,
    rubric: str,
) -> tuple[float, str, int, int, bool]:
    """Judge a rubric nugget.

    Returns (score, reason, prompt_tokens, completion_tokens, parse_ok).

    Retries once with a larger token budget on parse failure. In practice the
    dominant failure mode is an *empty* completion (not truncated JSON) —
    consistent with the judge model exhausting its token budget on reasoning
    before emitting the JSON payload — so a bare retry at the same max_tokens
    would likely reproduce the same empty output (temperature=0.0).
    """
    user = _judge_user_prompt(question, response, rubric)
    raw, pt, ct = call_llm(
        client,
        judge_model,
        JUDGE_SYSTEM_PROMPT,
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
        JUDGE_SYSTEM_PROMPT,
        user,
        temperature=0.0,
        max_tokens=768,
    )
    total_pt, total_ct = pt + pt2, ct + ct2
    parsed2 = parse_judge_response(raw2)
    if parsed2 is not None:
        score, reason = parsed2
        return score, reason, total_pt, total_ct, True

    return 0.0, f"parse error: {raw2[:100]}", total_pt, total_ct, False


# ── Result type ────────────────────────────────────────────────────────


@dataclass
class BeamResult:
    conv_id: str = ""
    chat_size: str = ""
    question_idx: int = -1
    question: str = ""
    question_type: str = ""
    expected_rubrics: list[str] = field(default_factory=list)
    hypothesis: str = ""
    nugget_scores: list[float] = field(default_factory=list)
    nugget_reasons: list[str] = field(default_factory=list)
    nugget_parse_errors: int = 0
    avg_score: float = 0.0
    hit: bool = False
    n_schemas: int = 0
    n_episodes: int = 0
    latency_ingest_s: float = 0.0
    latency_recall_s: float = 0.0
    n_llm_calls: int = 0
    llm_prompt_tokens: int = 0
    llm_completion_tokens: int = 0
    consolidate: bool = False
    error: str | None = None
    # Retrieval-side Recall@K/MRR (keyword-overlap vs. the rubric text, not
    # the LLM-judge score) — see _recall_keyword_score's docstring caveat.
    recall_at_k: dict = field(default_factory=dict)
    mrr: float = 0.0


# ── Main eval per conversation ─────────────────────────────────────────


def run_conversation(
    conv: dict,
    chat_size: str,
    *,
    shared_encoder: TextEncoder,
    openai_client: Any,
    answerer_model: str,
    judge_model: str,
    consolidate: bool,
    consolidation_passes: int = 5,
    top_k: int = 20,
    limit_questions: int = 0,
    embedding_cache: dict[str, np.ndarray] | None = None,
) -> list[BeamResult]:
    """Run all probing questions for one BEAM conversation."""
    conv_id = str(conv.get("conversation_id", "?"))

    import hashlib as _hashlib

    import numpy as _np

    _seed = int.from_bytes(
        _hashlib.sha256(f"beam:{chat_size}:{conv_id}".encode()).digest()[:4], "big"
    ) % (2**31)
    _np.random.seed(_seed)

    results: list[BeamResult] = []

    raw_q = conv.get("probing_questions", "")
    if isinstance(raw_q, dict):
        parsed = raw_q
    elif isinstance(raw_q, str):
        parsed = _parse_probing_questions(raw_q)
    else:
        return results

    all_qs = _flatten_questions(parsed)
    if limit_questions and limit_questions < len(all_qs):
        all_qs = all_qs[:limit_questions]
    if not all_qs:
        return results

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    try:
        cfg = SlowaveConfig(
            db_path=db_path,
            dim=shared_encoder.dim,
            encoder=EncoderConfig(),
            salience=SalienceConfig(),
            # BEAM conversations are ~1,700 messages — need more prototype
            # clusters than the defaults (designed for small sessions).
            replay=ReplayConfig(
                sample_size=256,
                max_prototypes_per_replay=64,
                assignment_threshold=0.65,  # tighter clusters → more prototypes → more schemas
                use_multi_scale=True,
            ),
            retrieval=RetrievalConfig(
                salience_weight=0.5,
                use_transition=True,
                use_multi_scale=True,
                use_temporal=True,
                temporal_weight=0.25,
            ),
            disable_encoder=False,
        )
        eng = SlowaveEngine(cfg, shared_encoder=shared_encoder)
        t_ingest = _ingest_conversation(eng, conv, shared_encoder, embedding_cache)

        if consolidate:
            # BEAM conversations are ~1,700 dense messages — multiple passes
            # needed since sample_size=256 and max_prototypes_per_replay=64.
            # 5 passes covers ~1,280 episodes and up to 320 prototype slots.
            for _pass in range(consolidation_passes):
                eng.consolidate_once(triggered_by="beam_eval")
            eng.replay_engine.self_supervise()

        for qi, q in enumerate(all_qs):
            q_text = str(q.get("question", "")).strip()
            q_type = str(q.get("question_type", "unknown"))
            nuggets = _get_nuggets(q)
            if not q_text:
                continue

            t_rec = time.time()
            r = eng.recall(q_text, top_k=top_k)
            t_rec = time.time() - t_rec

            # Retrieval-side Recall@K/MRR vs. the rubric text — cheap (no LLM
            # calls), doesn't touch the judge pipeline. See _recall_keyword_score.
            recall_at_k, mrr = compute_recall_at_k_and_mrr(
                eng,
                q_text,
                " ".join(nuggets),
                keyword_score_fn=_recall_keyword_score,
                hit_threshold=RECALL_HIT_THRESHOLD,
            )

            # Format memories with structure (schemas + episodes labelled)
            mem_text = _format_memories(r.schemas, r.episode_texts)

            total_pt = 0
            total_ct = 0
            n_calls = 0
            # System: instructions only. User: question + retrieved memories.
            ans_user = ANSWER_USER_TEMPLATE.format(
                question=q_text,
                memories=mem_text,
            )
            ans_text, apt, act = call_llm(
                openai_client,
                answerer_model,
                system=ANSWER_SYSTEM_PROMPT,
                user=ans_user,
                temperature=0.0,
                max_tokens=512,
            )
            total_pt += apt
            total_ct += act
            n_calls += 1

            nscores: list[float] = []
            nreasons: list[str] = []
            n_parse_errors = 0
            for nugget in nuggets:
                score, reason, npt, nct, parse_ok = _judge_nugget(
                    openai_client,
                    judge_model,
                    q_text,
                    ans_text,
                    nugget,
                )
                nscores.append(score)
                nreasons.append(reason)
                if not parse_ok:
                    n_parse_errors += 1
                total_pt += npt
                total_ct += nct
                n_calls += 1

            avg = sum(nscores) / len(nscores) if nscores else 0.0
            results.append(
                BeamResult(
                    conv_id=conv_id,
                    chat_size=chat_size,
                    question_idx=qi,
                    question=q_text,
                    question_type=q_type,
                    expected_rubrics=nuggets,
                    hypothesis=ans_text,
                    nugget_scores=nscores,
                    nugget_reasons=nreasons,
                    nugget_parse_errors=n_parse_errors,
                    avg_score=round(avg, 4),
                    hit=avg >= 0.5,
                    n_schemas=len(r.schemas),
                    n_episodes=len(r.episode_texts),
                    latency_ingest_s=round(t_ingest, 2),
                    latency_recall_s=round(t_rec, 4),
                    n_llm_calls=n_calls,
                    llm_prompt_tokens=total_pt,
                    llm_completion_tokens=total_ct,
                    consolidate=consolidate,
                    recall_at_k=recall_at_k,
                    mrr=round(mrr, 4),
                )
            )

    except Exception as e:
        import traceback

        print(f"\n  [ERROR] {conv_id}: {e}", file=sys.stderr, flush=True)
        traceback.print_exc(file=sys.stderr)
        results.append(
            BeamResult(
                conv_id=conv_id,
                chat_size=chat_size,
                question_type="error",
                error=str(e),
                consolidate=consolidate,
            )
        )
    finally:
        try:
            os.unlink(db_path)
        except OSError:
            pass

    return results


# ── Reporting ──────────────────────────────────────────────────────────


def print_report(results: list[BeamResult], consolidate: bool) -> None:
    """Print BEAM results in standard Slowave benchmark format."""
    mode = "with_consolidation" if consolidate else "episode-only"
    errors = [r for r in results if r.error]
    total_tokens = sum(r.llm_prompt_tokens + r.llm_completion_tokens for r in results)
    total_calls = sum(r.n_llm_calls for r in results)

    print_header(
        "BEAM Benchmark",
        [
            f"mode: {mode}",
            f"questions: {len(results)} ({len(errors)} errors)",
            f"LLM calls: {total_calls}  tokens: {total_tokens}",
        ],
    )

    by_type: dict[str, list[float]] = defaultdict(list)
    parse_errors_by_type: dict[str, int] = defaultdict(int)
    nuggets_by_type: dict[str, int] = defaultdict(int)
    for r in results:
        if r.question_type != "error" and r.nugget_scores:
            by_type[r.question_type].append(r.avg_score)
            parse_errors_by_type[r.question_type] += r.nugget_parse_errors
            nuggets_by_type[r.question_type] += len(r.nugget_scores)

    print(f"{'Ability':<30} {'n':>5} {'Score':>8} {'ParseErr':>9}  Description")
    print("-" * 72)
    total_scores: list[float] = []
    total_parse_errors = 0
    total_nuggets = 0
    for qtype in sorted(by_type):
        scores = by_type[qtype]
        avg = sum(scores) / len(scores)
        desc = BEAM_QUESTION_TYPES.get(qtype, "")
        pe_rate = (
            parse_errors_by_type[qtype] / nuggets_by_type[qtype] if nuggets_by_type[qtype] else 0.0
        )
        print(f"  {qtype:<28} {len(scores):>5} {avg:>7.1%} {pe_rate:>8.1%}  {desc}")
        total_scores.extend(scores)
        total_parse_errors += parse_errors_by_type[qtype]
        total_nuggets += nuggets_by_type[qtype]

    if total_scores:
        overall = sum(total_scores) / len(total_scores)
        overall_pe_rate = total_parse_errors / total_nuggets if total_nuggets else 0.0
        print("-" * 72)
        print(f"  {'OVERALL':<28} {len(total_scores):>5} {overall:>7.1%} {overall_pe_rate:>8.1%}")
        print(f"  (judge parse errors: {total_parse_errors}/{total_nuggets} nuggets)")
        print()

    valid = [r for r in results if r.question_type != "error"]
    recall_at_k_pct, mrr = aggregate_recall_at_k_mrr(
        [r.recall_at_k for r in valid],
        [r.mrr for r in valid],
    )
    print(
        " Recall@K (keyword-overlap vs. rubric text, not the LLM-judge score — "
        "noisier for negation-style rubrics like abstention): "
        + "  ".join(f"{k}={v}%" for k, v in recall_at_k_pct.items())
    )
    print(f" MRR: {mrr}")
    print()

    print_footer()


def _build_payload(
    results: list[BeamResult],
    chat_sizes: list[str],
    args: argparse.Namespace,
    total_elapsed: float,
) -> dict:
    """Build standard Slowave benchmark JSON payload."""
    by_type: dict[str, dict] = {}
    for r in results:
        if r.question_type == "error":
            continue
        t = r.question_type
        if t not in by_type:
            by_type[t] = {
                "n": 0,
                "sum_score": 0.0,
                "n_nuggets": 0,
                "n_parse_errors": 0,
                "recall_at_k_rows": [],
                "mrrs": [],
            }
        by_type[t]["n"] += 1
        by_type[t]["sum_score"] += r.avg_score
        by_type[t]["n_nuggets"] += len(r.nugget_scores)
        by_type[t]["n_parse_errors"] += r.nugget_parse_errors
        by_type[t]["recall_at_k_rows"].append(r.recall_at_k)
        by_type[t]["mrrs"].append(r.mrr)

    cat_summary = {}
    for t, d in sorted(by_type.items()):
        avg = d["sum_score"] / d["n"] if d["n"] else 0.0
        parse_error_rate = round(d["n_parse_errors"] / d["n_nuggets"], 4) if d["n_nuggets"] else 0.0
        cat_recall_at_k, cat_mrr = aggregate_recall_at_k_mrr(d["recall_at_k_rows"], d["mrrs"])
        cat_summary[t] = {
            "n": d["n"],
            "score_pct": round(avg * 100, 1),
            "n_nuggets": d["n_nuggets"],
            "n_parse_errors": d["n_parse_errors"],
            "parse_error_rate": parse_error_rate,
            "recall_at_k": cat_recall_at_k,
            "mrr": cat_mrr,
        }

    total_n = sum(d["n"] for d in by_type.values())
    total_score = sum(d["sum_score"] for d in by_type.values())
    overall_pct = round(total_score / total_n * 100, 1) if total_n else 0.0
    total_nuggets = sum(d["n_nuggets"] for d in by_type.values())
    total_parse_errors = sum(d["n_parse_errors"] for d in by_type.values())
    overall_parse_error_rate = (
        round(total_parse_errors / total_nuggets, 4) if total_nuggets else 0.0
    )
    overall_recall_at_k, overall_mrr = aggregate_recall_at_k_mrr(
        [r.recall_at_k for r in results if r.question_type != "error"],
        [r.mrr for r in results if r.question_type != "error"],
    )

    return {
        "meta": {
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "benchmark": "beam",
            "chat_sizes": chat_sizes,
            "answerer_model": getattr(args, "answerer_model", "unknown"),
            "judge_model": getattr(args, "judge_model", "unknown"),
            "consolidate": not getattr(args, "no_consolidate", False),
            "consolidation_passes": getattr(args, "consolidation_passes", 5),
            "top_k": getattr(args, "top_k", 10),
            "limit": getattr(args, "limit", 0),
            "total_elapsed_s": round(total_elapsed, 1),
        },
        "summary": {
            "n": total_n,
            "score_pct": overall_pct,
            "n_nuggets": total_nuggets,
            "n_parse_errors": total_parse_errors,
            "parse_error_rate": overall_parse_error_rate,
            "recall_at_k": overall_recall_at_k,
            "mrr": overall_mrr,
            "recall_at_k_note": "keyword-overlap vs. rubric text, not the LLM-judge "
            "score — noisier for negation-style rubrics (e.g. abstention)",
            "by_type": cat_summary,
        },
        "results": [
            {
                "conv_id": r.conv_id,
                "chat_size": r.chat_size,
                "question_idx": r.question_idx,
                "question": r.question[:500],
                "question_type": r.question_type,
                "expected_rubrics": r.expected_rubrics,
                "hypothesis": r.hypothesis[:500],
                "nugget_scores": r.nugget_scores,
                "nugget_reasons": r.nugget_reasons,
                "nugget_parse_errors": r.nugget_parse_errors,
                "recall_at_k": r.recall_at_k,
                "mrr": r.mrr,
                "avg_score": r.avg_score,
                "hit": r.hit,
                "n_schemas": r.n_schemas,
                "n_episodes": r.n_episodes,
                "latency_ingest_s": r.latency_ingest_s,
                "latency_recall_s": r.latency_recall_s,
                "n_llm_calls": r.n_llm_calls,
                "llm_prompt_tokens": r.llm_prompt_tokens,
                "llm_completion_tokens": r.llm_completion_tokens,
                "consolidate": r.consolidate,
                "error": r.error,
            }
            for r in results
        ],
    }


# ── Parallel worker ─────────────────────────────────────────────────────


def _worker_run_conv(task: dict) -> dict:
    """Run one BEAM conversation in a worker process.

    Each worker creates its own encoder + OpenAI client to avoid
    pickling/serialization issues with C extensions (ONNX Runtime, PyTorch).
    """
    conv = task["conv"]
    chat_size = task["chat_size"]
    conv_id = str(conv.get("conversation_id", task.get("conv_idx", "?")))

    # Load embedding cache if path was provided (workers can't share Python objects)
    embedding_cache = None
    cache_path = task.get("embedding_cache_path")
    if cache_path:
        embedding_cache = _load_embedding_cache(cache_path)

    try:
        enc = TextEncoder(EncoderConfig())
        _ = enc.dim  # force load
        from openai import OpenAI

        api_key = os.environ.get("OPENROUTER_API_KEY", "")
        client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            timeout=120.0,
            max_retries=3,
        )
        results = run_conversation(
            conv,
            chat_size,
            shared_encoder=enc,
            openai_client=client,
            answerer_model=task["answerer_model"],
            judge_model=task["judge_model"],
            consolidate=task["consolidate"],
            top_k=task["top_k"],
            consolidation_passes=task.get("consolidation_passes", 5),
            limit_questions=task["limit_questions"],
            embedding_cache=embedding_cache,
        )
        return {
            "conv_id": conv_id,
            "chat_size": chat_size,
            "results": [dataclasses.asdict(r) for r in results],
            "error": None,
        }
    except Exception as e:
        return {
            "conv_id": conv_id,
            "chat_size": chat_size,
            "results": [],
            "error": str(e),
        }


# ── Main ───────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="BEAM benchmark for Slowave (ICLR 2026)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--chat-sizes",
        default="1M",
        help="Comma-separated BEAM splits: 100K,500K,1M,10M (default: 1M)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit to N conversations per chat size (0 = all)",
    )
    parser.add_argument(
        "--limit-questions",
        type=int,
        default=0,
        help="Limit to N questions per conversation (0 = all)",
    )
    parser.add_argument(
        "--answerer-model",
        default="deepseek/deepseek-v4-flash",
        help="LLM for answer generation (default: deepseek/deepseek-v4-flash)",
    )
    parser.add_argument(
        "--judge-model",
        default="deepseek/deepseek-v4-flash",
        help="LLM for judging (default: deepseek/deepseek-v4-flash)",
    )
    parser.add_argument(
        "--no-consolidate",
        action="store_true",
        help="Skip consolidation (episode-only baseline)",
    )
    parser.add_argument(
        "--consolidation-passes",
        type=int,
        default=5,
        help="Number of consolidation passes per conversation (default: 5)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Recall top-k (default: 20)",
    )
    parser.add_argument("--out", type=str, default="", help="Output JSON path")
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="",
        help="BEAM cache directory (default: data/beam/)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1 = sequential)",
    )
    parser.add_argument(
        "--no-embedding-cache",
        action="store_true",
        help="Skip embedding cache: encode all texts at runtime",
    )
    parser.add_argument(
        "--rebuild-embedding-cache",
        action="store_true",
        help="Force rebuild embedding cache even if valid cache exists",
    )
    parser.add_argument(
        "--yes",
        "-y",
        action="store_true",
        help="Skip the confirmation prompt before this paid run.",
    )

    args = parser.parse_args()

    chat_sizes = [s.strip() for s in args.chat_sizes.split(",") if s.strip()]
    for sz in chat_sizes:
        if sz not in VALID_CHAT_SIZES:
            print(f"ERROR: unknown chat size '{sz}'. Valid: {VALID_CHAT_SIZES}", file=sys.stderr)
            sys.exit(1)

    if not os.environ.get("OPENROUTER_API_KEY"):
        print("ERROR: OPENAI_API_KEY env var not set.", file=sys.stderr)
        print("  export OPENROUTER_API_KEY=sk-or-...", file=sys.stderr)
        sys.exit(1)

    print("Loading BEAM dataset...")
    cache_dir = args.cache_dir or str(REPO_ROOT / "data" / "beam")
    dataset = _load_beam_dataset(chat_sizes, cache_dir=cache_dir)
    total_convs = sum(len(v) for v in dataset.values())
    print(f"  {total_convs} conversations across {list(dataset.keys())}")

    # ── Embedding cache ─────────────────────────────────────────────
    skip_embedding_cache = args.no_embedding_cache
    force_rebuild_cache = args.rebuild_embedding_cache
    embedding_caches: dict[str, dict[str, np.ndarray] | None] = {}

    if skip_embedding_cache:
        print("  Embedding cache: disabled (--no-embedding-cache)")
        for size in dataset:
            embedding_caches[size] = None
        if args.workers > 1:
            enc = None
            openai_client = None
            print(f"  Parallel mode: each of {args.workers} workers loads its own encoder + client")
        else:
            print("Loading encoder...", end=" ", flush=True)
            enc = TextEncoder(EncoderConfig())
            _ = enc.dim
            print(f"OK (dim={enc.dim})")
            print(
                f"Init OpenAI client (answerer={args.answerer_model}, judge={args.judge_model})..."
            )
            openai_client = get_openai_client()
    elif args.workers > 1:
        # Parallel mode: pre-build caches in main process, pass paths to workers
        print("Loading encoder for cache build...", end=" ", flush=True)
        cache_enc = TextEncoder(EncoderConfig())
        _ = cache_enc.dim
        print(f"OK (dim={cache_enc.dim})")
        for size, convs in dataset.items():
            if not convs:
                embedding_caches[size] = None
                continue
            ec = _load_or_build_embedding_cache(
                convs,
                cache_enc,
                size,
                cache_dir,
                force_rebuild=force_rebuild_cache,
            )
            embedding_caches[size] = ec
        # Workers don't need the main encoder; they create their own
        enc = None
        openai_client = None
        print(f"  Parallel mode: each of {args.workers} workers loads its own encoder + client")
    else:
        # Sequential mode: load encoder, then build/load caches
        print("Loading encoder...", end=" ", flush=True)
        enc = TextEncoder(EncoderConfig())
        _ = enc.dim
        print(f"OK (dim={enc.dim})")
        print(f"Init OpenAI client (answerer={args.answerer_model}, judge={args.judge_model})...")
        openai_client = get_openai_client()
        for size, convs in dataset.items():
            if not convs:
                embedding_caches[size] = None
                continue
            embedding_caches[size] = _load_or_build_embedding_cache(
                convs,
                enc,
                size,
                cache_dir,
                force_rebuild=force_rebuild_cache,
            )

    if args.out:
        out_path = Path(args.out)
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mode = "with_consolidation" if not args.no_consolidate else "no_consolidation"
        sizes = "-".join(chat_sizes)
        out_path = REPO_ROOT / "data" / "beam" / "runs" / f"{stamp}_{mode}_{sizes}.json"
    if not out_path.is_absolute():
        out_path = REPO_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    consolidate = not args.no_consolidate
    all_results: list[BeamResult] = []
    t_start = time.time()
    t_start_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n=== BEAM run started at {t_start_str} ===\n")

    # Flatten conversation list (may span multiple chat sizes)
    all_convs: list[tuple[str, dict]] = []
    for size in chat_sizes:
        convs = dataset.get(size, [])
        if args.limit and args.limit < len(convs):
            convs = convs[: args.limit]
        for ci, conv in enumerate(convs):
            all_convs.append((size, ci, conv))

    total = len(all_convs)

    # BEAM always makes paid API calls (answerer + judge, no free path) —
    # unlike LoCoMo/LongMemEval this always prompts, no --judge-model gate.
    n_questions = 0
    for _size, _ci, conv in all_convs:
        raw_q = conv.get("probing_questions", "")
        parsed = (
            raw_q
            if isinstance(raw_q, dict)
            else (_parse_probing_questions(raw_q) if isinstance(raw_q, str) else {})
        )
        qs = _flatten_questions(parsed)
        if args.limit_questions and args.limit_questions < len(qs):
            qs = qs[: args.limit_questions]
        n_questions += len(qs)
    # Historical measured total (2026-07-13, n=700 questions, deepseek-v4-flash,
    # answerer+judge combined) ≈ 13,029 tokens/question mixed prompt+completion.
    # Split roughly 90/10 prompt/completion as a rough approximation — BEAM's
    # per-nugget judge calls are individually small but numerous.
    est_cost = None
    if args.answerer_model == args.judge_model:
        est_cost = estimate_cost_usd(
            args.answerer_model, int(n_questions * 13_029 * 0.9), int(n_questions * 13_029 * 0.1)
        )
    confirm_paid_run(
        f"BEAM will answer+judge ~{n_questions} questions across {total} conversations "
        f"using answerer={args.answerer_model}, judge={args.judge_model}.",
        est_cost,
        assume_yes=args.yes,
    )

    if args.workers > 1:
        # ── Parallel ──────────────────────────────────────────────
        from multiprocessing import Pool

        workers = min(args.workers, total)
        tasks = [
            {
                "conv": conv,
                "chat_size": size,
                "conv_idx": ci,
                "answerer_model": args.answerer_model,
                "judge_model": args.judge_model,
                "consolidate": consolidate,
                "top_k": args.top_k,
                "consolidation_passes": args.consolidation_passes,
                "limit_questions": args.limit_questions,
                "embedding_cache_path": (
                    str(_get_embedding_cache_path(cache_dir, size))
                    if not skip_embedding_cache and embedding_caches.get(size) is not None
                    else None
                ),
            }
            for size, ci, conv in all_convs
        ]
        print(f"\n  Running {total} conversations across {workers} workers ...\n")
        done = 0
        with Pool(processes=workers) as pool:
            for wr in pool.imap_unordered(_worker_run_conv, tasks):
                done += 1
                if wr["error"]:
                    all_results.append(
                        BeamResult(
                            conv_id=wr["conv_id"],
                            chat_size=wr["chat_size"],
                            question_type="error",
                            error=wr["error"],
                            consolidate=consolidate,
                        )
                    )
                else:
                    for rd in wr["results"]:
                        all_results.append(BeamResult(**rd))
                rs = (
                    [BeamResult(**rd) for rd in wr["results"]]
                    if not wr["error"]
                    else [
                        BeamResult(
                            conv_id=wr["conv_id"],
                            chat_size=wr["chat_size"],
                            question_type="error",
                            error=wr["error"],
                            consolidate=consolidate,
                        ),
                    ]
                )
                valid = sum(1 for r in rs if r.question_type != "error")
                hits = sum(1 for r in rs if r.hit)
                avg_sc = sum(r.avg_score for r in rs if r.question_type != "error") / max(valid, 1)
                t_tok = sum(r.llm_prompt_tokens + r.llm_completion_tokens for r in rs)
                err_suffix = f"  ERR: {wr['error']}" if wr["error"] else ""
                print(
                    f"  [{done:>3}/{total}] {wr['conv_id']:<20} "
                    f"{valid:>3}q  {hits:>3}hits  avg={avg_sc:.2f}  "
                    f"tok={t_tok}{err_suffix}",
                    flush=True,
                )
                # Save checkpoint
                payload = _build_payload(all_results, chat_sizes, args, time.time() - t_start)
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(payload, f, indent=2, ensure_ascii=False)
    else:
        # ── Sequential ────────────────────────────────────────────
        from collections import defaultdict

        by_size: dict[str, list] = defaultdict(list)
        for size, ci, conv in all_convs:
            by_size[size].append((ci, conv))

        for size in by_size:
            convs = by_size[size]
            print(f"\n{'='*60}")
            print(f"  BEAM {size} — {len(convs)} conversations")
            print(f"{'='*60}")

            for ci, conv in convs:
                conv_id = str(conv.get("conversation_id", ci))
                t0 = time.time()
                try:
                    rs = run_conversation(
                        conv,
                        size,
                        shared_encoder=enc,
                        openai_client=openai_client,
                        answerer_model=args.answerer_model,
                        judge_model=args.judge_model,
                        consolidate=consolidate,
                        top_k=args.top_k,
                        consolidation_passes=args.consolidation_passes,
                        limit_questions=args.limit_questions,
                        embedding_cache=embedding_caches.get(size),
                    )
                except QuotaExhausted as e:
                    print(f"\n  [ABORT] Credits exhausted: {e}")
                    rs = [
                        BeamResult(
                            conv_id=conv_id,
                            chat_size=size,
                            question_type="error",
                            error=f"quota_exhausted: {e}",
                            consolidate=consolidate,
                        )
                    ]
                    all_results.extend(rs)
                    break  # stop processing further conversations
                except Exception as e:
                    print(f"  [{ci+1}/{len(convs)}] {conv_id} ERROR: {e}")
                    rs = [
                        BeamResult(
                            conv_id=conv_id,
                            chat_size=size,
                            question_type="error",
                            error=str(e),
                            consolidate=consolidate,
                        )
                    ]

                all_results.extend(rs)
                elapsed = time.time() - t0
                hits = sum(1 for r in rs if r.hit)
                valid = sum(1 for r in rs if r.question_type != "error")
                avg_sc = sum(r.avg_score for r in rs if r.question_type != "error") / max(valid, 1)
                t_tok = sum(r.llm_prompt_tokens + r.llm_completion_tokens for r in rs)
                print(
                    f"  [{ci+1:>3}/{len(convs)}] {conv_id:<20} "
                    f"{valid:>3}q  {hits:>3}hits  avg={avg_sc:.2f}  "
                    f"tok={t_tok}  {elapsed:.1f}s",
                    flush=True,
                )

                # Save checkpoint
                payload = _build_payload(all_results, chat_sizes, args, time.time() - t_start)
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(payload, f, indent=2, ensure_ascii=False)

    total_elapsed = time.time() - t_start
    t_end_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    h, remainder = divmod(int(total_elapsed), 3600)
    m, s = divmod(remainder, 60)
    elapsed_str = f"{h}h {m}m {s}s" if h else f"{m}m {s}s"
    print(f"\n=== BEAM run finished at {t_end_str} — total time: {elapsed_str} ===\n")
    print_report(all_results, consolidate)

    payload = _build_payload(all_results, chat_sizes, args, total_elapsed)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"Results saved to: {out_path}")

    total_prompt = sum(r.llm_prompt_tokens for r in all_results)
    total_completion = sum(r.llm_completion_tokens for r in all_results)
    print(f"\nToken usage: {total_prompt:,} prompt + {total_completion:,} completion")
    if args.answerer_model == args.judge_model:
        cost = estimate_cost_usd(args.answerer_model, total_prompt, total_completion)
        cost_s = f"${cost:.2f}" if cost is not None else "unknown pricing"
        print(f"Cost ({args.answerer_model}): {cost_s}")
    else:
        print(
            f"Cost: unknown split between answerer ({args.answerer_model}) and judge "
            f"({args.judge_model}) token usage — check openrouter.ai/models for current rates"
        )


if __name__ == "__main__":
    main()
