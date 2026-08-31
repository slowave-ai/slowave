[![PyPI](https://img.shields.io/pypi/v/slowave?color=2f6f4e)](https://pypi.org/project/slowave/)
[![Python](https://img.shields.io/badge/python-3.11%2B-4c6f91)](https://pypi.org/project/slowave/)
[![PyPI Status](https://img.shields.io/pypi/status/slowave?color=orange)](https://pypi.org/project/slowave/)
[![License: AGPL-3.0-or-later](https://img.shields.io/badge/license-AGPL--3.0--or--later-blue.svg)](LICENSE)

---

<img src="img/slowave-logo-text.jpeg" alt="Slowave" width="300"/>

**A living memory layer across your AI tools.**

---

Slowave keeps useful decisions, context, procedures, while lets stale memories fade.

- Pick up where you left off, even when you switch AI tools.
- Useful memories are reinforced. Irrelevant or outdated memories fade.
- Past solutions and failures can become reusable procedures.
- Memory is stored locally in SQLite. No data leaves your machine. 
- Slowave makes no LLM calls. No LLM API key is required. 
- Inspect and manage memory through the local dashboard.

Works on Claude Code, Codex, Curson, Cline, Windsurf, OpenCode, Claude Desktop. 

---

## How Slowave memory works

Slowave works through 5 simple MCP tools:

- **Activate** — start a task and load relevant memory.
- **Remember** — save a fact, decision, preference, or instruction.
- **Recall** — search memory during a task.
- **Feedback** — mark retrieved memory as useful, irrelevant, or stale.
- **Commit** — save the task outcome and any reusable procedure.

Remember, recall, and feedback can be called more than once during a task.

A **background worker** consolidates your memories and procedures.

See [design.md](docs/design.md) and [architecture.md](docs/architecture.md) for more details.

---

## Installation

Install Slowave:

```bash
pipx install slowave
```

Preview the changes, configure detected clients, and verify the installation:

```bash
slowave setup --dry-run   # preview
slowave setup             # configure detected clients
slowave doctor            # verify the installation
```

`slowave setup` is idempotent, safe to run more than once. It 
- configures MCP.
- installs the lifecycle instructions.
- starts the daemon.
- starts the background worker.



> [!IMPORTANT]
> **No LLM API key required.**


Configure one client with `slowave setup --client <name>`. See [supported clients](#supported-clients) and the [installation reference](docs/install.md).

---

## Dashboard

Start the local dashboard:

```bash
slowave dashboard
```

Open [http://127.0.0.1:8765](http://127.0.0.1:8765) to inspect memories, retrievals, procedures, activity, and system health.

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


---

## Design choices

- **Local** — memory stays on your machine in SQLite.
- **Shared** — supported AI tools can use the same memory.
- **Agent-driven** — your LLM agent decides what to save and rates what it retrieves. 
- **Feedback-based** — memories can be reinforced, ignored, marked stale, or replaced.
- **Deterministic** - no LLM in the loop, memory is stored in latent space and governed by deterministic algorithms.
- **Inspectable** — the dashboard shows what was stored and why it was retrieved.

The consolidation model is inspired by episodic memory, associative recall, and slow-wave sleep. See [design.md](docs/design.md) and [architecture.md](docs/architecture.md) for details.

---

## Supported clients

Work in progress — suggest more integrations or report broken ones with setup details.

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
> Memory is stored in plaintext at `~/.slowave/slowave.db`. Slowave does not send it to a hosted memory service. Protect sensitive data with OS permissions or full-disk encryption.

---

## Honest limits

- Slowave can only recall information the agent saved.
- Retrieval may omit relevant memories or surface irrelevant ones.
- Slowave does not verify that stored information is true.
- Memory quality depends on what the agent saves and how it rates retrievals.

---

## Documentation

- [design.md](docs/design.md) — design rationale, boundaries, and positioning
- [architecture.md](docs/architecture.md) — brain-inspired memory model and lifecycle
- [install.md](docs/install.md) — install & setup reference, lifecycle block, files modified

---

## Contributing

Slowave is open source under the AGPL-3.0-or-later license.

Contributions are welcome, especially in:

- client integrations
- recall quality improvements
- evaluation datasets
- performance optimization

See [CONTRIBUTING.md](./CONTRIBUTING.md) before submitting a pull request.

---

## License

Slowave is open source under the [GNU AGPL-3.0-or-later](LICENSE) license.
