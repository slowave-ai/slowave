# Architecture

This document describes the current Slowave memory system and its five-verb MCP lifecycle. It explains the public behavior and the major local components; it is not a specification for internal ranking constants or database schema.

For product rationale, boundaries, and trade-offs, see [design.md](design.md).

---

## Overview

Slowave keeps durable memory in a local SQLite database. An MCP client opens a task session, retrieves relevant memory, records durable claims when needed, assesses retrieved evidence, and closes the task with an outcome. Background consolidation turns eligible session activity into longer-lived memory structures without calling an LLM.

```mermaid
flowchart LR
    A[Agent task] --> B[activate]
    B --> C[Scoped retrieval and session]
    C --> D[Agent reasoning]
    D --> E[remember durable claims]
    D --> F[recall mid-task lookup]
    C --> G[feedback target assessments]
    F --> G
    E --> H[commit outcome and verification]
    G --> H
    H --> I[(Local SQLite raw events and evidence)]
    I --> J[Offline consolidation]
    J --> K[(Episodes, prototypes, schemas, relations)]
    K --> C
```

The agent remains the reasoning layer. Slowave returns memory and evidence; it does not independently decide what the final answer or action should be.

---

## The public cognitive cycle

The MCP interface intentionally exposes five task-level verbs.

| Verb | Purpose | Important constraint |
|---|---|---|
| slowave_activate | Starts a task session and primes working memory with scoped memories and relevant procedures. | task, initial_goal, and scope are required. |
| slowave_remember | Stores a durable typed claim. | It must use the active session and matching scope. |
| slowave_recall | Performs a deliberate semantic lookup during the same task. | It is session-bound and scope-bound. |
| slowave_feedback | Records target-specific assessments of retrieved memories and procedures. | Task outcome belongs in commit, not feedback. |
| slowave_commit | Closes the task with its outcome, verification, and, when applicable, a procedure. | Complete feedback is required for every exposed retrieval target. |

Activation returns an opaque continuity ID. A client omits it for a new conversation and reuses the returned value unchanged for later tasks in that same conversation. Continuity correlates related sessions; it does not relax the scope boundary.

### Activate

Activation receives the verbatim task, a concise initial goal, and a scope. It opens an implicit task session and returns a compact set of directly relevant or associated memories, relevant execution-backed procedures, warnings, and a retrieval ID for later feedback.

On a cold start, the response signals that no memory exists yet for the scope. The client should read one stable context document, preserve only durable facts that are not already observable, and then continue normally.

### Remember

Remember is for durable, standalone claims such as facts, preferences, decisions, constraints, lessons, warnings, and durable tasks. It deliberately does not store transient work state or a narration of the current task.

Claims may include occurred_at when they describe a specific past event. Slowave separately records the time it received the call, preserving the real ordering of session activity.

### Recall

Recall is a mid-task lookup when the question changes or activation did not surface enough context. It returns canonical memories and procedures plus bounded provenance references; full evidence includes bounded source evidence for inspection.

### Feedback

Feedback records what actually happened after retrieval. A memory assessment is used, irrelevant, or stale. Stale feedback must name its reason; a superseded memory also names the active replacement. Procedure feedback records both whether the procedure was used and whether it helped, had no effect, caused harm, or remains unknown.

Feedback is append-only. A complete-coverage declaration assesses every target exposed by that retrieval; an incomplete declaration leaves unassessed targets unknown rather than treating silence as negative evidence.

### Commit

Commit stores the final goal, honest outcome, and structured verification. When the work produced a clear reusable method, it can also capture a procedure with context, steps, and caveats. The commit preflight rejects closure with a retryable error until required retrieval feedback is complete.

---

## Local memory components

| Component | Role |
|---|---|
| Raw events and sessions | Preserve the ordered record of tool activity, task outcome, verification, and provenance. |
| Episodic memory | Represents eligible individual experiences with scope, time, salience, and embeddings. |
| Prototypes and associations | Group related episodes and retain local semantic, temporal, and co-occurrence structure. |
| Schemas | Provide durable, searchable memory records with evidence and lifecycle state. |
| Procedures | Store execution-backed methods captured when a task closes. |
| Retrieval snapshots and feedback events | Preserve what was exposed, how it was reached, and what the client later reported. |

SQLite is the source of durable state. The dashboard and CLI expose this state for inspection; they do not rely on a hosted control plane.

---

## From activity to durable memory

Session activity is first recorded as raw events. Eligible events are encoded as episodes. Offline consolidation groups related episodes into prototypes and maintains associations, salience, and searchable schema records.

The system uses local embeddings and deterministic operations in this path. It does not ask an LLM to summarize or rewrite memory. A readable memory record is kept at the boundary so agents and people can inspect what the system returned.

Geometry can establish topical association, but it is not authoritative for semantic truth. Contradiction and supersession are recorded through explicit client feedback with evidence, not inferred solely from embedding similarity.

---

## Retrieval and working memory

Retrieval begins with the current task or recall query. It considers semantic relevance, scope eligibility, temporal context, salience, and bounded associations. Direct matches and associated results are distinguished in the returned data.

The result is intentionally bounded: Slowave returns a compact working-memory set rather than an ever-growing transcript. The client chooses whether and how to use that set in its own prompt and reasoning process.

### Scopes and generalization

Ordinary MCP retrieval uses strict scope matching. Some schemas can earn a broader generalization stage after use across distinct scopes and sessions:

| Stage | Visibility |
|---|---|
| Scoped | Origin scope only. |
| Portable | Related scopes of the same kind. |
| Contextual | Other scopes with a retrieval penalty. |
| Global | Broad visibility after strong cross-scope evidence. |

This mechanism reduces accidental context leakage during ordinary work, but it is not an authorization system. Separate sensitive projects, users, or tenants into separate stores when hard isolation is required.

### Time

Each stored experience has a recording time and can retain an optional source time for an earlier real-world event. Temporal context is a ranking signal, not a promise of exact natural-language date interpretation. The client can inspect source evidence when date provenance matters.

---

## Memory lifecycle states and human control

Active memory normally participates in retrieval. Feedback can mark memory stale or record a superseding replacement, while deduplication can archive an exact duplicate. A person can suppress a memory through the CLI or dashboard; suppression is reversible and does not delete its source evidence.

There is intentionally no agent-facing MCP forget tool. Forgetting is a human decision made after inspecting a specific memory, not an inference from conversation text.

---

## Procedures

Procedures are separate from ordinary remembered claims. They are explicit records captured at commit time, including a summary, durable context, ordered steps, and caveats. Retrieval may surface a procedure that matches the current task; later feedback records its observed usefulness.

A procedure is evidence from past work, not an instruction that the agent must follow. Failed attempts remain available as warnings when relevant.

---

## Operational boundaries

Slowave is public beta software. Its APIs, configuration, and storage schema may change; migrations are not guaranteed before stable release. The local database is plaintext by default and should be protected accordingly.

The architecture is inspired by episodic memory, consolidation, and associative recall. These are design analogies, not claims that Slowave is a biological simulation or that the analogy establishes retrieval quality.
