<!-- mcp-name: io.github.slowave-ai/slowave -->

---

<p align="center">
  <img src="img/slowave-logo-text.jpeg" alt="Slowave" width="450"/>
</p>
<p align="center">
  <b>Living memory layer across your coding agents and AI tools.</b>
</p>
<p align="center">
  <a href="https://pypi.org/project/slowave/"><img src="https://img.shields.io/pypi/v/slowave?color=2f6f4e" alt="PyPI version" /></a>
  <img src="https://img.shields.io/badge/python-3.11%2B-4c6f91" />
  <img src="https://img.shields.io/pypi/status/slowave?color=orange" />
  <img src="https://img.shields.io/badge/license-AGPL--3.0--or--later-blue.svg" />
</p>
<p align="center">
  Supports: Claude Code, Copilot CLI, Codex, Cursor, OpenCode, Windsurf/Devin Desktop, Cline, Claude Desktop
</p>


---
AI agents have large context windows, but that context ends with your current session.
Open a new session, switch from Claude Code to Codex, and you have to restate the same decisions, constraints, over and over.

Slowave gives your agents one local, shared memory, without requiring a separate LLM for memory maintenance.

At the core Slowave is built around a simple concept:

> **Reasoning and memory feed each other in a continuous loop.**

```mermaid
graph LR
    LLM["Reasoning<br/>(Claude, Codex, Copilot, etc.)"]
    MEM["Memory<br/>(Slowave)"]
    
    LLM -->|Feedback| MEM
    MEM -->|Retrieve| LLM
    MEM -->|Adapt| MEM
```

Slowave retains what helps agents achieve their goals, weakens what does not, and continuously adapts based on use. It does this through a continuous feedback loop between your agent and its memory:

> **remember → recall → use → feedback → reinforce / weaken → decay**


Memory is continuously reshaped by use rather than a static collection of facts waiting to be retrieved.

- **Keep context across tasks:** Your agents can reuse recorded decisions, preferences, constraints, and lessons.
- **Evolves through feedback:** Useful memories grow stronger, irrelevant ones lose priority, stale knowledge can be superseded.
- **Learns from experience:** Decisions, outcomes, and multi-step solutions can become reusable context your agents can act on.
- **Runs locally:** Slowave stores memory in SQLite and does not send it to a hosted memory service.
- **No LLM API key:** The memory core performs maintenance and retrieval without LLM calls or an LLM API key.
- **Inspectable and measurable:** Review memories, retrievals, feedback, procedures, and system performance in the local dashboard.

The first useful payoff is simply not having to repeat the same constraint in the next task. 

Over time, the way you work becomes reusable context for your agent.


<p align="center">
  <sub><a href="#supported-clients">Platform coverage and setup details</a></sub>
</p>

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

For upgrades and service management, see [Install, upgrade, run, and remove Slowave](docs/install.md).
Use `slowave start`, `slowave stop`, and `slowave restart` for installed services;
use `slowave status --services` and `slowave doctor` to diagnose problems.
See [Troubleshooting](docs/troubleshooting.md) for recovery and logs, or run
`slowave docs troubleshooting` to open it from your terminal.

## Workflow

You keep working with your agent as usual, nothing changes in your workflow. You can switch agents or start new sessions. 

Slowave enriches your agent context with relevant memories and procedures. Your agent feedbacks Slwoave on what was useful and what was not. Slowave keeps building relevant context for your agents in the background.



```mermaid
sequenceDiagram
    actor User as You
    participant Agent as Your Agent<br/>(Claude Code, Codex, etc.)
    participant Slowave
    participant Worker as Slowave Worker

    User->>Agent: Prompt
    Agent->>Slowave: Activate: Start task (set goal)
    Slowave-->>Agent: Relevant memories and procedures

    opt 
        Agent->>Slowave: Remember: Save new knowledge
    end

    opt 
        Agent->>Slowave: Recall: Look up more context
        Slowave-->>Agent: Matching memories and procedures
    end

    Agent->>Agent: Reason upon enriched context, work on the task
    Agent->>Slowave: Feedback: Assess retrieved context
    Agent->>Slowave: Commit: Record outcome
    Slowave-->>Agent: Task outcome saved, session closed
    Agent-->>User: Deliver the result

    Worker->>Slowave: Consolidate, reinforce, decay memories
```


## Dashboard

The local dashboard shows what Slowave is actually doing with your agent's memory over time: what it remembers, what it retrieves, and what your agent reports using.

As your history grows, you can follow a memory from where it came from to the tasks where it was later retrieved and used. You can inspect learned procedures, see when retrieved context was useful or irrelevant, and understand whether Slowave is helping your agent reuse past work rather than simply accumulating memories.

Start the dashboard with:

```bash
slowave dashboard
```

From the dashboard you can:

- **Inspect memories:** see what was saved, where it came from, its current state, and when it was later retrieved or used.
- **Inspect procedures:** review reusable methods captured from previous work and see when they are retrieved and reused.
- **Trace retrievals:** see the request, the context Slowave actually returned, and the feedback reported by the agent.
- **Follow activity:** inspect tasks, retrievals, feedback, memory updates, and their links to each other.
- **Explore memory connections:** browse relationships and associations between memories.
- **Check system health:** inspect the database, background maintenance, backups, and local services.

The overview also helps answer whether the memory loop is working: how many memories and procedures exist, how often retrieved context is reported as used, how much is marked irrelevant, whether feedback is being recorded, and how creation, retrieval, and use change over time.

Slowave also shows a rough estimate of the context returned through all MCP endpoints, so you can keep an eye on memory overhead.

<p align="center">
    <a href="img/overview.jpg"><img src="img/overview.jpg" alt="Slowave local dashboard" width="90%"></a>
</p>

<p align="center">
    <a href="img/memories.jpg"><img src="img/memories.jpg" alt="Memory detail" width="17%"></a>
    <a href="img/procedures.jpg"><img src="img/procedures.jpg" alt="Procedures" width="17%"></a>
    <a href="img/retrieval.jpg"><img src="img/retrieval.jpg" alt="Retrieval" width="17%"></a>
    <a href="img/graph.jpg"><img src="img/graph.jpg" alt="Memory graph" width="17%"></a>
    <a href="img/activity.jpg"><img src="img/activity.jpg" alt="Activity" width="17"></a>
</p>


## Supported clients

Client coverage is actively expanding. Suggest more integrations or report broken ones with setup details.

✅ = supported. Copilot validation limits are listed below.

| Client         | macOS | Linux | Windows | Setup                                    |
|----------------|--|--|--|------------------------------------------|
| <img src="img/clients/claude.svg" alt="" title="Claude Code" width="20" height="20" align="center" />&nbsp; [Claude Code](integrations/claude-code/README.md) | ✅ | ✅ | ✅ | `slowave setup --client claude-code` |
| <picture><source media="(prefers-color-scheme: dark)" srcset="img/clients/copilot-dark.svg" /><img src="img/clients/copilot.svg" alt="" title="Copilot CLI" width="20" height="20" align="center" /></picture>&nbsp; [Copilot CLI](integrations/copilot/README.md) | ✅ | ✅ | ✅ | `slowave setup --client copilot` |
| <picture><source media="(prefers-color-scheme: dark)" srcset="img/clients/openai-dark.svg" /><img src="img/clients/openai.svg" alt="" title="Codex" width="20" height="20" align="center" /></picture>&nbsp; [Codex](integrations/codex/README.md) | ✅ | ✅ | ✅ | `slowave setup --client codex` |
| <img src="img/clients/cursor.svg" alt="" title="Cursor" width="20" height="20" align="center" />&nbsp; [Cursor](integrations/cursor/README.md) | ✅ | ✅ | ✅ | `slowave setup --client cursor` ¹ |
| <img src="img/clients/opencode-mark.svg" alt="" title="OpenCode" width="20" height="20" align="center" />&nbsp; [OpenCode](integrations/opencode/README.md) | ✅ | ✅ | ✅ | `slowave setup --client opencode` |
| <img src="img/clients/windsurf.svg" alt="" title="Windsurf" width="20" height="20" align="center" />&nbsp; [Windsurf](integrations/windsurf/README.md) | ✅ | ✅ | ✅ | `slowave setup --client windsurf` |
| <picture><source media="(prefers-color-scheme: dark)" srcset="img/clients/cline-dark.svg" /><img src="img/clients/cline.svg" alt="" title="Cline" width="20" height="20" align="center" /></picture>&nbsp; [Cline](integrations/cline/README.md) | ✅ | ✅ | ✅ | `slowave setup --client cline` |
| <img src="img/clients/claude.svg" alt="" title="Claude Desktop" width="20" height="20" align="center" />&nbsp; [Claude Desktop](integrations/claude-desktop/README.md) | ✅ | ✅ | ✅ | `slowave setup --client claude-desktop` ¹ |
| All the above |  |  |  | `slowave setup` |

¹ requires one manual paste after setup

Copilot support targets local CLI execution. Automated integration tests pass, and
native configuration discovery is verified on macOS. A user also confirmed memory
reads and writes through Copilot CLI on macOS. Full model-driven lifecycle and native
Windows/Linux qualification remain pending. IDE, desktop, and hosted/cloud
Copilot support is deferred.

> [!IMPORTANT]
> The default embedding model downloads from Hugging Face on first use (~45 MB, cached locally). Subsequent runs work offline.
>
> Memory is stored in plaintext in the current OS user's application-data directory. Slowave does not send it to a hosted memory service. See [runtime data location](docs/install.md#runtime-data-location).


## How Slowave memory works

Slowave works through 5 simple MCP tools:

- `Activate`: start a task and load relevant memory.
- `Remember`: save a fact, decision, preference, or instruction.
- `Recall`: search memory during a task.
- `Feedback`: assess retrieved memories as `used`, `not_used`, `unassessable`, `irrelevant`, `already_known`, or `stale`.
- `Commit`: save the task outcome and any reusable procedure.

A **background worker** consolidates relevant memories and procedures.

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
- Memory quality depends on the client agent and the feedback it provides.
- Slowave adds token overhead from tool calls and retrieved context.
- The local SQLite database is plaintext by default; protect it with OS permissions or full-disk encryption.

> [!IMPORTANT]
> Slowave is public beta software. APIs, configuration, and storage schema may change, and migrations are not guaranteed before stable release.

## Evaluation

The current evaluation notes report preliminary retrieval-evidence results,
methodology, limitations, and commands for running new evaluations. They do not
claim end-to-end agent accuracy or a comparison against other memory systems.
See [benchmarks.md](docs/benchmarks.md) before treating any result as a
production-quality claim.


## Documentation

- [Mintlify documentation](https://slowave-ai.mintlify.app/): full auto-generated documentation
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
