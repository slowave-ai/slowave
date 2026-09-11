[![PyPI](https://img.shields.io/pypi/v/slowave?color=2f6f4e)](https://pypi.org/project/slowave/)
[![Python](https://img.shields.io/badge/python-3.11%2B-4c6f91)](https://pypi.org/project/slowave/)
[![PyPI Status](https://img.shields.io/pypi/status/slowave?color=orange)](https://pypi.org/project/slowave/)
[![License: AGPL-3.0-or-later](https://img.shields.io/badge/license-AGPL--3.0--or--later-blue.svg)](LICENSE)

---

<img src="img/slowave-logo-text.jpeg" alt="Slowave" width="300"/>

**Living memory layer across your AI tools.**

---

AI agents have large context windows, but that context ends with your current session.
Open a new session, switch from Claude Code to Codex, and you have to restate the same decisions, constraints, and failed attempts.

Slowave gives your agents one local, shared memory.

Many AI memory systems focus on storing information and retrieving semantically relevant context.

Slowave starts from a different premise:

> **Agent memory isn't only a retrieval problem. Memory quality is driven by how facts retained in memory can help an agent achieve its future goals.**

Slowave addresses this with a continuous feedback loop between the agent and its memory:

> **remember → recall → use → feedback → reinforce / weaken → decay**

Over time, the agent’s feedback shapes what Slowave returns without needing a separate LLM judge inside the memory layer.

Memory becomes something continuously shaped by use rather than a static collection of facts waiting to be retrieved.

- **Keep context across tasks**: Your agents can reuse recorded decisions, preferences, constraints, and lessons instead of making you repeat them.
- **Improves with use:** Useful memories strengthen, irrelevant ones lose priority, stale knowledge can be suppressed or superseded.
- **Learns from experience:** Decisions, outcomes, and multi-step solutions can become reusable memories and procedures.
- **Runs locally:** Slowave stores memory in SQLite and does not send it to a hosted memory service.
- **No LLM API key:** The memory core performs maintenance and retrieval without LLM calls or an LLM API key.
- **Inspectable:** Review memories, retrievals, feedback, procedures, and system activity in the local dashboard.

Supported integrations include Claude Code, Codex, Cursor, Cline, Windsurf / Devin Desktop, OpenCode, and Claude Desktop. See [platform coverage and manual steps](#supported-clients).

## Why Slowave?

Every new agent session has the codebase but not necessarily the context behind
it: previous decisions, constraints, failed attempts, the thinking process.

The **missing context** is what makes an agent repeat a rejected design,
miss an operational constraint, or make you explain (again and again) the project principles.

Slowave turns those durable judgments into a shared project memory. It is not a
markdown-transcript replay system or another LLM layer on top of your agent.

The connected agent already understands the conversation, task, and outcome.
It decides what is worth preserving and whether retrieved context helped,
was irrelevant, or became stale. Slowave makes those decisions persistent,
inspectable, and useful in later tasks—without LLM calls inside its memory core.

Install Slowave if you feel you're **missing the continuity** in your daily work with your LLM agent.

The first useful payoff is simply not having to repeat the same constraint in the next task.

Over time, the way you work becomes **reusable context** for your agent.

## Installation


### Quick start
```bash
pipx install slowave
slowave setup --dry-run
slowave setup
```

The quick start configures every detected client. To configure just one client at a time, see the [installation reference](docs/install.md).

> [!IMPORTANT]
> **No LLM API key required.**

To remove Slowave, see the [removal guide](docs/install.md#remove-slowave).

## What changes in your workflow?

Slowave is transparent to your work.

You still ask your agent to do normal work: fix a regression, add a migration,
review a pull request. When it encounters a durable fact or decision, the
installed lifecycle instructs your agent to preserve that claim. 
On a later task, Slowave can return a compact, scoped set of relevant recorded memories to your agent, so that it can act upon its own memories. 

What you will see while working with your agent:
- your agent activating Slowave for the current task and goal, 
- Slowave retrieving relevant context to your agent, 
- your agent sending feedback to Slowave on what was retrieved.
- your agent committing a Slowave session.

Optionally you will see:
- your agent invoking Slowave to remember durable facts.
- your agent invoking Slowave to recall something critical for the current task or goal.

Slowave does not decide whether a claim is true or important. Your agent makes
that judgment and reports whether retrieved memory helped, was irrelevant, or
became stale. Slowave maintains the resulting local memory.


## Dashboard

Start the local dashboard with:

```bash
slowave dashboard
```

Opens the dashboard in your browser, where you can inspect:

- **Memories:** browse saved decisions, constraints, and lessons.
- **Procedures:** review reusable step-by-step methods from past work.
- **Retrievals:** see what memory Slowave returned for each task.
- **Activity:** follow recent sessions, memory updates, and feedback.
- **Memory graph:** explore connections between related memories.
- **System health:** check the database, worker, backups, and local services.

<p align="center">
  <a href="img/overview.jpg">
    <img src="img/overview.jpg" alt="Slowave local dashboard" width="80%">
  </a>
</p>

<p align="center">
    <a href="img/schemas.jpg"><img src="img/schemas.jpg" alt="Memory detail" width="16%"></a>
    <a href="img/procedures.jpg"><img src="img/procedures.jpg" alt="Procedures" width="16%"></a>
    <a href="img/retrieval.jpg"><img src="img/retrieval.jpg" alt="Retrieval" width="16%"></a>
    <a href="img/activity.jpg"><img src="img/activity.jpg" alt="Activity" width="16%"></a>
    <a href="img/graph.jpg"><img src="img/graph.jpg" alt="Memory graph" width="16%"></a>
</p>


## Supported clients

Client coverage is actively expanding. Suggest more integrations or report broken ones with setup details.

✅ = manually verified · ⬜ = pending verification

| Client         | macOS | Linux | Windows | Setup                                    |
|----------------|--|--|--|------------------------------------------|
| [Claude Code](integrations/claude-code/README.md) | ✅ | ✅ | ✅ | `slowave setup --client claude-code` |
| [Cline](integrations/cline/README.md) | ✅ | ✅ | ✅ | `slowave setup --client cline` |
| [Cursor](integrations/cursor/README.md) | ✅ | ✅ | ✅ | `slowave setup --client cursor` ¹ |
| [Windsurf](integrations/windsurf/README.md) | ✅ | ✅ | ✅ | `slowave setup --client windsurf` |
| [Claude Desktop](integrations/claude-desktop/README.md) | ✅ | ✅ | ✅ | `slowave setup --client claude-desktop` ¹ |
| [OpenCode](integrations/opencode/README.md) | ✅ | ✅ | ✅ | `slowave setup --client opencode` |
| [Codex](integrations/codex/README.md) | ✅ | ✅ | ✅ | `slowave setup --client codex` |
| All the above |  |  |  | `slowave setup` |

¹ requires one manual paste after setup

> [!IMPORTANT]
> The default embedding model downloads from Hugging Face on first use (~45 MB, cached locally). Subsequent runs work offline.
>
> Memory is stored in plaintext in the current OS user's application-data directory. Slowave does not send it to a hosted memory service. See [runtime data and migration](docs/install.md#runtime-data-location-and-migration).


## How Slowave memory works

Slowave works through 5 simple MCP tools:

- `Activate`: start a task and load relevant memory.
- `Remember`: save a fact, decision, preference, or instruction.
- `Recall`: search memory during a task.
- `Feedback`: mark retrieved memory as useful, irrelevant, or stale.
- `Commit`: save the task outcome and any reusable procedure.

A **background worker** consolidates relevant memories and procedures.

See [architecture.md](docs/architecture.md) and [design.md](docs/design.md) for more details.

### Slowave MCP lifecycle

```mermaid
flowchart LR
    A[Agent task] --> B[1. <i>activate</i><br/>start session]
    B --> C[Scoped retrieval<br/>and session]
    C --> D[Agent reasoning]
    D --> E[2. <i>remember</i><br/>durable claims]
    D --> F[3. <i>recall</i><br/>mid-task lookup]
    C --> G[4. <i>feedback</i><br/>target assessments]
    F --> G
    E --> H[5. <i>commit</i><br/>outcome and verification]
    G --> H
    H --> I[(Local SQLite<br/>raw events and evidence)]
    I --> J[Offline consolidation]
    J --> K[(Episodes, prototypes,<br/>schemas, relations)]
    K --> C
```

See [architecture.md](docs/architecture.md) and [design.md](docs/design.md) for details.


## Boundaries

- Slowave is a memory layer, not a reasoning engine.
- It cannot recall information that was never recorded.
- It supplies relevant context, but the connected agent decides how to interpret and use it.
- Memory quality depends on the client agent and the feedback it provides.
- Scopes reduce accidental context leakage; use separate stores when hard isolation is required.
- Slowave adds token overhead from tool calls and retrieved context.
- The local SQLite database is plaintext by default; protect it with OS permissions or full-disk encryption.
- Slowave is public beta software. APIs, configuration, and storage schema may change, and migrations are not guaranteed before stable release.

## Evaluation

The current evaluation notes report preliminary retrieval-evidence results,
methodology, limitations, and commands for running new evaluations. They do not
claim end-to-end agent accuracy or a comparison against other memory systems.
See [benchmarks.md](docs/benchmarks.md) before treating any result as a
production-quality claim.


## Documentation

- <a href="" target="_blank">Mintlify</a>: full auto-generated documentation 
- [design.md](docs/design.md): design rationale, boundaries, and positioning
- [architecture.md](docs/architecture.md): brain-inspired memory model and lifecycle
- [install.md](docs/install.md): installation, setup, lifecycle instructions, modified files, and removal
- [benchmarks.md](docs/benchmarks.md): benchmark results, methodology, and reproduction
- [troubleshooting.md](docs/troubleshooting.md): daemon, worker, dashboard, client integration, database, backup/restore


## Contributing

Slowave is open source under the AGPL-3.0-or-later license.

Contributions are welcome, especially in:

- installation and setup quality
- client integrations
- performance optimization

See [CONTRIBUTING.md](./CONTRIBUTING.md) before submitting a pull request.


## License

Slowave is open source under the [GNU AGPL-3.0-or-later](LICENSE) license.
