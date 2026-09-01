# Benchmarks

Slowave is a local memory layer: it maintains and retrieves memory without LLM
calls for ingestion, consolidation, or recall. These benchmarks test whether
the evidence returned by Slowave contains what is needed to answer a question.
They do not test an LLM-generated final answer.

## Current results

The headline metric is binary LLM-judge evidence containment. The external
judge was `deepseek/deepseek-v4-flash` at temperature 0; it is evaluation
infrastructure, not part of Slowave.

| Benchmark | Result | What was tested | What the result says |
| --- | ---: | --- | --- |
| LoCoMo | **71.84%** | 1,534 answerable questions in categories 1–4; consolidated top-20 retrieval | Slowave retrieves explicit information well across long conversations. The LoCoMo multi-session category reached **87.04%**. |
| LongMemEval oracle | **65.20%** | All 500 questions; consolidated hybrid top-20 retrieval; evidence sessions only | Slowave can retain and combine supplied relevant sessions, but this is not a distractor-retrieval test. |

The reported values were calculated from complete local evaluation records.
Those raw records are not part of the public repository; the settings and
commands below are sufficient to reproduce a new run locally.

## What Slowave does well today

- **Long-conversation fact retrieval.** LoCoMo's multi-session category scored
  87.04%. Most of those questions ask for one explicitly stated fact located
  somewhere in a long, multi-session conversation.
- **Direct user and assistant facts.** LongMemEval oracle scored 91.07% for
  single-session assistant information and 87.14% for single-session user
  information.
- **Knowledge updates.** LongMemEval oracle scored 83.33% on knowledge-update
  questions, a useful signal for maintaining changing facts.
- **Local, LLM-free memory operations.** These retrieval results came without
  an LLM call for ingest, consolidation, or recall. That gives Slowave local
  control, inspectability, predictable memory-operation cost, and independence
  from a model provider.

## What the lower-scoring categories ask of a system

A memory layer does **not** need to embed a reasoning layer. In Slowave's
architecture, the connected agent consumes retrieved evidence and decides how
to answer. The categories below mix memory retrieval with other capabilities,
so a lower score is not automatically a pure memory failure.

| Area | Judge score | Primary task demand | What this score can and cannot tell us |
| --- | ---: | --- | --- |
| Temporal reasoning | 55.03% LoCoMo; 45.86% LongMemEval oracle | Retrieve dated evidence and interpret or calculate over time | It can reflect either missing dates or a need for temporal reasoning. Inspect evidence coverage before changing memory retrieval. |
| Cross-session aggregation | 53.38% LongMemEval multi-session | Retrieve every required item, then count or combine them | This is both a memory challenge and a synthesis challenge. One missing item makes the binary result fail. |
| Implicit preferences | 56.67% LongMemEval single-session preference | Infer a preference from statements or behavior | The needed conclusion may not be stored verbatim, so an agent-level reasoner has a legitimate role. |
| Commonsense | 35.79% LoCoMo | Combine retrieved evidence with world knowledge or interpretation | This is not a pure memory-recall task and should not be used alone to judge the memory layer. |

This distinction suggests two complementary evaluations:

1. **Memory quality:** measure whether all gold evidence is retrieved and well
   ranked (for example, recall@K, MRR, and evidence coverage).
2. **Assistant quality:** give the same retrieved evidence to one fixed reader
   model, then judge its final answer.

The completed runs deliberately measure the first boundary—whether Slowave's
raw context supports an answer—without adding a hidden answerer. That is the
right design for an LLM-free memory layer, but it does not resolve every
question about end-to-end assistant performance. LLM-managed systems may score
well on the latter because their pipeline includes an extractor, reranker, or
answerer; Slowave keeps those reasoning decisions with the connected agent.

## Read the two benchmarks correctly

The word “multi-session” does not mean the same thing in both datasets.

| Property | LoCoMo category 4 | LongMemEval multi-session |
| --- | ---: | ---: |
| Judge score | 87.04% | 53.38% |
| Gold evidence per question | 1.07 turns on average | 2.59 sessions on average |
| Need more than one evidence unit | 46 / 841 (5.5%) | 133 / 133 (100%) |
| Typical task | Find one fact in a long conversation history | Combine facts or counts across 2–5 sessions |

This explains the apparent inversion. In LoCoMo, a single retrieved episode is
usually enough. In LongMemEval, returning four out of five relevant facts still
fails the binary check if the required total is five.

LongMemEval oracle deserves an additional caveat: the upstream dataset defines
it as **oracle retrieval**, with only evidence sessions in the history. The
local oracle file contains no distractor sessions for any of its 500 questions.
LongMemEval-S is the strict setting, with roughly 40 sessions and about 115k
tokens of history. Do not compare the 65.20% oracle result to LongMemEval-S
leaderboards. The [upstream LongMemEval documentation](https://github.com/xiaowu0162/LongMemEval)
defines both settings.

## What the judge actually does

For each question, Slowave produces the retrieved schema and episode text. The
judge receives:

1. The question.
2. The benchmark reference answer.
3. The retrieved text, capped at 60,000 characters.

It returns `1.0` only when the retrieved text supports the reference answer,
and `0.0` when the answer is absent, off-topic, or contradicted. It accepts
paraphrases and equivalent dates or numbers, allows irrelevant extra context,
and gives no partial credit. The judge does not receive a generated Slowave
answer, source ranks, or ground-truth evidence locations.

The required output is strict JSON. A malformed response is retried once with
a larger output budget; a second failure is recorded as a parse failure and is
excluded from the mean. The LoCoMo run had 6 parse failures out of 1,540 judge
calls; LongMemEval had none out of 500.

This is a semantic **evidence-containment** metric, not end-to-end QA accuracy.
It also remains an LLM-judge metric: it can make mistakes, has access to the
reference answer, and can change with model serving. Review a stratified sample
of pass and fail decisions before making a stronger public claim.

## Comparison with other memory systems

These numbers are not directly rankable against vendor leaderboard claims yet.
Those claims can differ in dataset split, retrieval budget, reader/answerer,
judge, prompt, and scoring denominator. Most importantly, many evaluate an
LLM-generated answer, while Slowave's completed runs judge raw retrieved
evidence.

The right comparative claim is: **Slowave already shows strong LLM-free
retrieval for explicit facts, especially in LoCoMo's multi-session category;
strict end-to-end and strict-haystack comparisons remain to be run.** A fair
head-to-head evaluation must use the same split, context budget, answerer,
judge, prompt, temperature, and denominator for every system.

## Reproduce the completed runs

### 1. Download the datasets

Run these commands from the repository root. They download the exact files
used by the completed runs, plus the strict LongMemEval-S file for the pending
evaluation:

```bash
# LoCoMo
bash scripts/download_datasets.sh locomo

# LongMemEval oracle and strict-S
bash scripts/download_datasets.sh lme
```

The script downloads LoCoMo from the [official LoCoMo repository](https://github.com/snap-research/locomo)
and the cleaned LongMemEval files from the [upstream LongMemEval dataset](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned).
Check the upstream dataset licenses before redistributing them.

Verify that the required files exist:

```bash
test -f data/locomo/locomo10.json
test -f data/longmemeval/longmemeval_oracle.json
```

### 2. Configure the external judge

Use Python 3.11+ with the project dependencies installed. The judge needs an
OpenRouter key; Slowave itself does not.

```bash
export OPENROUTER_API_KEY='...'
```

### 3. Run LoCoMo

```bash
.venv/bin/python tests/benchmarks/locomo_eval.py \
  --dataset data/locomo/locomo10.json \
  --assignment-threshold 0.85 \
  --top-k 20 \
  --judge-model deepseek/deepseek-v4-flash \
  --save-full-hypotheses \
  --out data/suite_runs/locomo_judge_deepseek_rerun.json
```

### 4. Run LongMemEval oracle

```bash
.venv/bin/python tests/benchmarks/longmemeval_eval.py \
  --dataset data/longmemeval/longmemeval_oracle.json \
  --assignment-threshold 0.65 \
  --recall-mode hybrid \
  --top-k 20 \
  --judge-model deepseek/deepseek-v4-flash \
  --save-full-hypotheses \
  --out data/longmemeval/runs/longmemeval_oracle_judge_deepseek_rerun.json
```

Confirm the paid prompt, or use `--yes` in non-interactive automation.
`--save-full-hypotheses` makes the new artifact auditable without changing
retrieval or the judge input.

The historical artifacts do not record a Git revision, dataset checksum,
dependency lock state, or complete retrieved evidence. Their results can be
reproduced approximately, not forensically byte-for-byte. Preserve those items
with future runs, and report LongMemEval-S separately with its exact command,
artifact, judge count, and parse-failure count.
