# Design Rationale

Slowave is a centralized, adaptive memory substrate shared across AI tools.

It gives different assistants, agents, chat clients, and MCP-compatible tools access to the same persistent memory layer instead of each tool keeping its own isolated memory.

It is built around one core idea:

> **Memory is a latent process before it is a reasoning process.**

Slowave stores and updates memory through local embeddings, timestamps, scopes, salience, reinforcement, decay, supersession, and graph relationships. Only after retrieval does it render selected memory as natural language for a human, agent, chatbot, or language model.

The architectural separation is simple:

> Use language models for reasoning.
>
> Use latent space mechanisms for memory.

Slowave is not a replacement for a language model, a reasoning engine, or an autonomous agent framework. It is the persistent memory layer those systems can use. The downstream client remains responsible for reasoning, planning, answer construction, tool execution, and final user-facing behavior.

This document explains *why* Slowave exists and what it chooses to be. For *how* the memory model works — the two learning systems, the lifecycle, and the brain analogy they're built on — see [architecture.md](architecture.md).

---

## Problem Statement

Most AI tools still treat memory as one of a few familiar things:

- a transcript of previous messages;
- a static note store;
- an LLM-generated summary of past interactions;
- a tool-specific memory silo that disappears when the user switches clients.

Imagine switching from Claude Code to Cursor mid-task. In most setups, the new tool knows nothing — not your project conventions, not the decision you just made, not the bug you were tracking. You re-explain. You rebuild context from scratch.

Those approaches can work, but they have drawbacks. They often depend on remote model calls, grow with conversation length, are difficult to inspect, and are tied to one assistant or vendor.

Slowave takes a different path.

It treats memory as a local adaptive system. Events are encoded, associated, reinforced, weakened, revised, consolidated, and retrieved before they are verbalized.

This is the central product idea behind Slowave: memory should live outside any single tool.

A chat client, coding assistant, terminal agent, desktop assistant, or future model should be able to connect to the same memory substrate. The user should not lose continuity just because they switch from one tool to another.

The design target is not to maximize every benchmark score. The design target is to provide a private, local, inspectable, reusable memory substrate that improves continuity over repeated use.

---

## What Slowave Is

Slowave is a shared memory substrate for repeated AI use across multiple tools.

It is designed to help AI tools remember context that remains useful across sessions, such as:

- project decisions;
- user preferences;
- recurring workflows, from release checklists to monthly reporting cycles;
- team, client, or tool conventions;
- architectural and organizational choices;
- prior troubleshooting context;
- long-running task history.

The important point is that these memories are not locked inside one assistant. A decision remembered through one client can later be recalled by another client, as long as both use the same Slowave memory store.

Instead of replaying entire histories into every prompt, Slowave retrieves a compact working-memory brief for the current task.

The goal is not to remember everything with equal priority.

The goal is to remember what remains useful.

---

## Boundaries

Slowave intentionally keeps memory separate from reasoning.

It is not:

- a language model;
- a general reasoning engine;
- a full autonomous agent framework;
- a natural-language summarization engine;
- a static knowledge base retrieval system
- a markdown file manager

Higher-order reasoning, planning, synthesis, and final answer construction still belong to the downstream model or application.

Slowave provides persistent context. The client decides how to use it.

---

## The LLM's Role: Author and Consumer, Not Operator

Many modern memory systems use language models as memory *operators*. A language model is asked to summarize conversations, merge memories, reflect on past sessions, rewrite stored knowledge, or rerank retrieved context.

Slowave approach is different.

The language model **is** part of the boundary of the system:

- as an **author**, when a client decides something is worth remembering and phrases the claim it stores;
- as a **consumer**, when recalled context is injected into its prompt;
- as a **critic**, when it labels retrieved memories as useful, stale, or wrong.

The language model is **not** part of memory maintenance:

- consolidation, reinforcement, decay, supersession, ranking, and retrieval never require an LLM call;
- no model rewrites, merges, or summarizes stored memory;
- the memory layer does not depend on an LLM provider, API key, hosted model, or cloud memory service.

This keeps the maintenance loop:

- local-first;
- low-latency;
- reproducible;
- inspectable;
- inexpensive to run;
- portable across tools;
- independent from any specific model vendor.

The quality of what enters memory still benefits from a capable client writing clear claims — but once a memory exists, its evolution is governed by deterministic local mechanisms, not by another model call.

---

## Memory Before Language

Human memory is not an append-only transcript of sentences.

Experiences are encoded, associated, reinforced, reorganized, weakened, and recalled before they are verbalized.

Slowave follows that principle at the system level.

Incoming events are converted into local memory representations. Retrieval is shaped by semantic similarity, time, scope, salience, reinforcement, decay, supersession, and graph relationships. Only after recall does Slowave render selected memory into language, usually as a compact working-memory brief.

This keeps the memory layer independent from the reasoning layer. The same memory store can support different clients, models, and tools without being tied to one assistant or one LLM provider.

How this is structured — the episodic and schema layers, offline consolidation, scoped recall, and the working-memory brief — is described in [architecture.md](architecture.md).

---

## Behavioral Patterns

Not all useful memory is factual.

Some memory is behavioral: repeated ways of doing things that shouldn't need to be re-stated every session — how a project is usually tested, how a monthly report is assembled, how a client onboarding is run, how a recurring troubleshooting workflow unfolds.

Slowave captures these patterns implicitly, not as an explicit procedural store. Repetition strengthens the paths between consolidated patterns, and recall can surface "what has tended to come next" as a predictive signal alongside regular retrieval.

Explicit instructions ("run tests before pushing", "send the recap after every meeting") are stored as constraints and recalled when relevant. Observed repetition reinforces the associative structure. Over time both signals converge: the recalled constraint and the observed tendency point in the same direction.

One rule keeps this honest: behavioral memory *explains*, it never *prescribes*. Slowave supplies context about what has tended to work; the LLM remains the decision-maker.

---

## Benefits of the Approach

**Cross-tool continuity.** Memory is centralized outside individual tools. A user can remember something from one assistant, retrieve it from another, and continue work without rebuilding context from scratch.

**Predictable cost.** Recall and context generation do not require per-query LLM calls. Memory cost is not tied to model pricing, remote inference, or context-window replay.

**Privacy.** Memory can stay entirely in the local environment. Slowave does not require sending stored memories to a hosted memory provider. Local-first does not mean encrypted-by-default: users should protect the local database, backups, logs, and exported artifacts according to their security needs.

**Low latency.** Recall runs through local retrieval and deterministic ranking rather than remote model inference — fast enough for interactive use.

**Reproducibility.** Because retrieval is based on local state and deterministic ranking signals, behavior is easier to inspect and reproduce than LLM-mediated memory rewriting.

**Vendor independence.** The memory layer does not depend on a specific hosted model, API key, or cloud memory service. The reasoning layer can change while the memory layer remains persistent.

---

## Positioning

Slowave is a centralized, reusable memory layer for systems that need persistent context across sessions, tools, and models. Context is organized by flexible scopes — projects, domains, workflows, clients, relationships, or unscoped general memory — not hardcoded to one domain such as coding.

The guiding principles are few:

- Evolve memory through use: strengthen what keeps helping, let stale information lose priority.
- Keep memory local, inspectable, and portable — independent of any model vendor.
- Inject context selectively instead of replaying history.
- Support many tools through one shared substrate, and keep the reasoning layer interchangeable.

The client can change. The model can change. The interface can change. The memory remains available.

This separation is deliberate. It allows Slowave to act as a local, adaptive second brain for agents, assistants, and tools without turning memory itself into another LLM-dependent pipeline.

Slowave should therefore be evaluated as a memory substrate: by continuity, retrieval quality, suppression of stale context, scope behavior, feedback adaptation, portability, and operational reliability.

That is the design goal behind Slowave: useful memory should strengthen, stale memory should fade, outdated memory should be revised, and relevant context should be retrievable without replaying everything that ever happened.
