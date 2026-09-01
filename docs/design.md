# Design Rationale

Slowave is a centralized, adaptive memory substrate shared across AI tools. It gives assistants, agents, chat clients, and MCP-compatible tools access to one persistent memory layer rather than isolated, tool-specific memories.

> **Memory is a latent process before it is a reasoning process.**

Slowave stores and updates memory through local embeddings, timestamps, scopes, salience, reinforcement, decay, supersession, and graph relationships. It renders selected memory as natural language only after retrieval, for a person, agent, chatbot, or language model to use.

For the system's lifecycle, components, and memory model, see [architecture.md](architecture.md).

---

## Overview

The architectural separation is simple:

> Use language models for reasoning.
>
> Use latent space mechanisms for memory.

Slowave is not a replacement for a language model, a reasoning engine, or an autonomous agent framework. It is the persistent memory layer those systems can use. The downstream client remains responsible for reasoning, planning, answer construction, tool execution, and final user-facing behavior.

The design target is a private, local, inspectable, reusable memory substrate that improves continuity through repeated use—not a system that maximizes every benchmark score.

---

## The problem

Most AI tools still treat memory as one of a few familiar things:

| Pattern | Limitation |
|---|---|
| Previous-message transcript | Context grows with the conversation and is costly to replay. |
| Static note store | Knowledge does not adapt through use. |
| LLM-generated history summary | Maintenance depends on model calls and can be difficult to inspect. |
| Tool-specific memory silo | Continuity disappears when the user changes clients. |

Imagine switching from Claude Code to Cursor mid-task. In most setups, the new tool knows nothing — not your project conventions, not the decision you just made, not the bug you were tracking. You re-explain. You rebuild context from scratch.

Those approaches can work, but they have drawbacks. They often depend on remote model calls, grow with conversation length, are difficult to inspect, and are tied to one assistant or vendor.

Slowave takes a different path: it treats memory as a local adaptive system. Events are encoded, associated, reinforced, weakened, revised, consolidated, and retrieved before they are verbalized.

The central product idea is that memory should live outside any single tool. A chat client, coding assistant, terminal agent, desktop assistant, or future model can connect to the same memory substrate, so users do not lose continuity merely by changing interfaces.

---

## What Slowave remembers

Slowave is a shared memory substrate for repeated AI use across multiple tools. It is designed to retain context that remains useful across sessions, such as:

- project decisions;
- user preferences;
- recurring workflows, from release checklists to monthly reporting cycles;
- team, client, or tool conventions;
- architectural and organizational choices;
- prior troubleshooting context;
- long-running task history.

The important point is that these memories are not locked inside one assistant. A decision remembered through one client can later be recalled by another client, as long as both use the same Slowave memory store.

Instead of replaying entire histories into every prompt, Slowave retrieves a compact working-memory brief for the current task. The goal is not to remember everything with equal priority; it is to remember what remains useful.

---

## Boundaries

Slowave intentionally keeps memory separate from reasoning.

It is not:

- a language model;
- a general reasoning engine;
- a full autonomous agent framework;
- a natural-language summarization engine;
- a static knowledge base retrieval system; or
- a Markdown file manager.

Higher-order reasoning, planning, synthesis, and final answer construction still belong to the downstream model or application.

Slowave provides persistent context. The client decides how to use it.

---

## Language models at the boundary

Many modern memory systems use language models as memory *operators*. A language model is asked to summarize conversations, merge memories, reflect on past sessions, rewrite stored knowledge, or rerank retrieved context.

Slowave's approach is different. The language model is part of the system boundary:

| Role | What the model does |
|---|---|
| **Author** | A client decides what is worth remembering and writes a clear claim. |
| **Consumer** | Recalled context is injected into a prompt for reasoning. |
| **Critic** | Retrieved memories are labelled useful, stale, or wrong. |

The language model is **not** part of memory maintenance:

- consolidation, reinforcement, decay, supersession, ranking, and retrieval never require an LLM call;
- no model rewrites, merges, or summarizes stored memory;
- the memory layer does not depend on an LLM provider, API key, hosted model, or cloud memory service.

This keeps the maintenance loop local-first, low-latency, reproducible, inspectable, inexpensive to run, portable across tools, and independent of a specific model vendor. The quality of what enters memory still benefits from a capable client writing clear claims—but once a memory exists, deterministic local mechanisms govern its evolution.

---

## Memory before language

Human memory is not an append-only transcript of sentences. Experiences are encoded, associated, reinforced, reorganized, weakened, and recalled before they are verbalized.

Slowave follows that principle at the system level.

Incoming events are converted into local memory representations. Retrieval is shaped by semantic similarity, time, scope, salience, reinforcement, decay, supersession, and graph relationships. Only after recall does Slowave render selected memory into language, usually as a compact working-memory brief.

This keeps the memory layer independent from the reasoning layer. The same memory store can support different clients, models, and tools without being tied to one assistant or LLM provider. The episodic and schema layers, offline consolidation, scoped recall, and working-memory brief are described in [architecture.md](architecture.md).

---

## Behavioral patterns

Not all useful memory is factual. Some memory is behavioral: repeated ways of doing things that should not need to be restated every session—how a project is tested, a monthly report is assembled, a client onboarding is run, or a recurring troubleshooting workflow unfolds.

Slowave captures these patterns implicitly, not as an explicit procedural store. Repetition strengthens the paths between consolidated patterns, and recall can surface "what has tended to come next" as a predictive signal alongside regular retrieval.

Explicit instructions ("run tests before pushing", "send the recap after every meeting") are stored as constraints and recalled when relevant. Observed repetition reinforces the associative structure. Over time both signals converge: the recalled constraint and the observed tendency point in the same direction.

> **Behavioral memory explains; it never prescribes.** Slowave supplies context about what has tended to work, while the language model remains the decision-maker.

---

## Benefits

| Benefit | Why it matters |
|---|---|
| **Cross-tool continuity** | Memory is centralized outside individual tools, so work can continue across assistants without rebuilding context. |
| **Predictable cost** | Recall and context generation do not need per-query LLM calls, model pricing, remote inference, or context-window replay. |
| **Privacy** | Memory can remain entirely local, without a hosted memory provider. Local-first does not mean encrypted by default: users should protect the database, backups, logs, and exports. |
| **Low latency** | Local retrieval and deterministic ranking support interactive use without remote inference. |
| **Reproducibility** | Local state and deterministic ranking are easier to inspect and reproduce than LLM-mediated memory rewriting. |
| **Vendor independence** | The reasoning layer can change while the persistent memory layer remains available. |

---

## Positioning

Slowave is a centralized, reusable memory layer for systems that need persistent context across sessions, tools, and models. Context is organized by flexible scopes — projects, domains, workflows, clients, relationships, or unscoped general memory — not hardcoded to one domain such as coding.

The guiding principles are:

- evolve memory through use: strengthen what keeps helping and let stale information lose priority;
- keep memory local, inspectable, and portable—independent of any model vendor;
- inject context selectively instead of replaying history; and
- support many tools through one shared substrate while keeping the reasoning layer interchangeable.

The client can change. The model can change. The interface can change. The memory remains available.

This separation lets Slowave act as a local, adaptive second brain for agents, assistants, and tools without turning memory itself into another LLM-dependent pipeline.

Slowave should be evaluated as a memory substrate: by continuity, retrieval quality, suppression of stale context, scope behavior, feedback adaptation, portability, and operational reliability. Its goal is simple: useful memory should strengthen, stale memory should fade, outdated memory should be revised, and relevant context should be retrievable without replaying everything that ever happened.
