[![PyPI](https://img.shields.io/pypi/v/slowave?color=2f6f4e)](https://pypi.org/project/slowave/)
[![Python](https://img.shields.io/badge/python-3.11%2B-4c6f91)](https://pypi.org/project/slowave/)
[![PyPI Status](https://img.shields.io/pypi/status/slowave?color=orange)](https://pypi.org/project/slowave/)
[![License: AGPL-3.0-or-later](https://img.shields.io/badge/license-AGPL--3.0--or--later-blue.svg)](LICENSE)

<img src="img/slowave-logo-text.jpeg" alt="Slowave" width="400"/>

**A living local memory layer across your AI tools.**

- Slowave keeps useful decisions, context, procedures, while lets stale memories fade.
- Memory evolves over time, following your work.
- Fully local, no data leaves your machine. 
- Inspectable through a local dashboard.
- No LLM API key required.

---

## How it feels

You work daily with your AI tools:

- **Day 1** — cold start: Slowave bootstraps memory, initializing the embedding-based memory state.
- **Week 1** — emerging patterns: new interactions begin reinforcing relevant signals, forming stable associations.
- **Month 1** — context consolidates: frequently reinforced information becomes consistently retrievable, low-signal data fades.

Multiple AI clients continuously build and reuse the same evolving memory over time:
- not a markdown manager
- not static RAG retrieval system
- not an extra LLM layer over your agent

---

## What you gain over time

Slowave becomes more useful the more you use it.

* **Continuity** — pick up projects where you left off
* **Clarity** — your AI understands you without repeated explanation
* **Consistency** — keep your context across AI tools
* **Retention** — retain decisions, patterns, and preferences over time
* **Focus** — spend time creating instead of managing context

Slowave does not just store information — it compounds it into usable context.

The result is a continuous working context that follows you across tools and time.

---

## Installation

### Setup all clients in one go

Install Slowave and configure every detected client in one go:

```bash
pipx install slowave
```

Then wire everything up:

```bash
slowave setup --dry-run   # preview what will change
slowave setup             # apply: MCP configs, lifecycle instructions, hooks, services
slowave doctor            # verify: daemon health, client detection
```

`slowave setup` is idempotent and safe to run multiple times. The HTTP MCP daemon and background consolidation worker start automatically as system services.

> [!IMPORTANT]
> **Public beta.** APIs and the storage schema may change. Your memory lives in a local plaintext SQLite database by default; protect it with OS permissions or full-disk encryption.


### Per-client setup

To configure a single client, or to find client-specific details:

| Client | Integration doc |
|---|---|
| Claude Code | [integrations/claude-code/README.md](integrations/claude-code/README.md) |
| Claude Desktop ¹ | [integrations/claude-desktop/README.md](integrations/claude-desktop/README.md) |
| Cline | [integrations/cline/README.md](integrations/cline/README.md) |
| Cursor ¹ | [integrations/cursor/README.md](integrations/cursor/README.md) |
| OpenCode | [integrations/opencode/README.md](integrations/opencode/README.md) |
| Windsurf | [integrations/windsurf/README.md](integrations/windsurf/README.md) |
| Codex | [integrations/codex/README.md](integrations/codex/README.md) |
| Gemini CLI | coming soon |

¹ requires one manual paste after setup

See the complete install & setup reference: [docs/install.md](docs/install.md)

### Storage

The default embedding model downloads from Hugging Face on first use (~45 MB, cached locally). Subsequent runs work offline.

Memory is stored in a local SQLite database at `~/.slowave/slowave.db` — fully inspectable, never leaves your machine. Not encrypted by default; protect sensitive data with OS permissions or full-disk encryption.


---

## Why Slowave is different

Slowave is a local, feedback-driven long-term memory layer for AI agents — not a transcript replay, static RAG file, or LLM summarization pipeline.

- **Local and model-independent** — memory stays in a local SQLite database; it needs no memory-service API key or internal LLM calls.
- **Shared across clients** — one scoped store provides continuity across supported MCP tools.
- **Adaptive, not append-only** — explicit feedback can reinforce useful memory, suppress noise, and record stale or superseded information with provenance.
- **Selective context** — Slowave retrieves a compact, task-relevant brief instead of replaying conversation history.
- **Inspectable and controllable** — you can inspect evidence, review retrievals, and suppress a memory; execution-backed procedures preserve reusable work patterns.

The architecture draws inspiration from episodic memory, offline consolidation, and associative recall. Read the [design rationale](docs/design.md) or [architecture guide](docs/architecture.md) for the full model.

---

## Dashboard

Monitor Slowave’s memory health, incoming events and memory consolidation in real time.

<p align="center">
    <a href="img/overview.jpg"><img src="img/overview.jpg" alt="Dashboard overview" width="33%"></a>
    <a href="img/schemas.jpg"><img src="img/schemas.jpg" alt="Memory detail" width="33%"></a>
    <a href="img/procedures.jpg"><img src="img/procedures.jpg" alt="Procedures" width="33%"></a><br>
    <a href="img/retrieval.jpg"><img src="img/retrieval.jpg" alt="Retrieval" width="33%"></a>
    <a href="img/activity.jpg"><img src="img/activity.jpg" alt="Activity" width="33%"></a>
    <a href="img/graph.jpg"><img src="img/graph.jpg" alt="Memory graph" width="33%"></a>
</p>


---

## Supported clients

Work in progress — suggest more integrations or report broken ones with setup details.

✅ = manually verified · ⬜ = pending verification

| Client         | macOS | Linux | Windows | Setup                                    |
|----------------|--|--|--|------------------------------------------|
| Claude Code    | ✅ | ✅ | ✅ | `slowave setup --client claude-code`     |
| Cline          | ✅ | ✅ | ✅ | `slowave setup --client cline`           |
| Cursor         | ✅ | ✅ | ✅ | `slowave setup --client cursor` ¹        |
| Windsurf (Devin)    | ✅ | ✅ | ✅ | `slowave setup --client windsurf`        |
| Claude Desktop | ✅ | ✅ | ✅ | `slowave setup --client claude-desktop` ¹ |
| OpenCode       | ✅ | ✅ | ✅ | `slowave setup --client opencode`        |
| Codex          | ✅ | ✅ | ✅ | `slowave setup --client codex`           |
| Gemini CLI     | ⬜ | ⬜ | ⬜ | `slowave setup --client gemini`          |
| All the above  |  |  |  | `slowave setup`                          |

¹ requires one manual paste after setup

---

## Honest limits

- It recalls stored information; it does not infer missing preferences.
- It retrieves relevant memories; it does not perform reasoning.
- Memory quality (definition, feedback, classification, etc) depend on your agent capabilities.

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