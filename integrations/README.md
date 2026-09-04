# Slowave — client integrations

Full install and setup guide: **[../docs/install.md](../docs/install.md)**

---

## Quick-start (all clients)

```bash
pipx install slowave
slowave setup
slowave doctor
```

`slowave setup` auto-configures every client it detects, injects the same Slowave lifecycle instructions into each client's instruction surface, installs the background worker, and starts the HTTP MCP daemon. It is idempotent — safe to re-run.

**Remove Slowave:**
```bash
slowave uninstall       # remove integrations and services; keep memories
slowave purge           # remove integrations, services, and local data
pipx uninstall slowave  # remove the installed package
```

Run `--dry-run` first. `purge` is destructive; see the [complete removal
guide](../docs/install.md#remove-slowave) for preserved database archives and
the manual Claude Desktop and Cursor steps.

> **Claude Desktop & Cursor:** after `slowave setup`, paste the lifecycle block into the client UI — see [docs/install.md#lifecycle-instruction-block](../docs/install.md#lifecycle-instruction-block).

---

## Client quick-ref cards

| Client | Quick-ref | What's different |
|---|---|---|
| Claude Desktop | [claude-desktop/README.md](claude-desktop/README.md) | Requires Custom Instructions (one manual paste) |
| Claude Code | [claude-code/README.md](claude-code/README.md) | CLAUDE.md + enforcement hooks |
| Cline | [cline/README.md](cline/README.md) | Same lifecycle block injected into `.clinerules` |
| Cursor | [cursor/README.md](cursor/README.md) | Requires Rules for AI (one manual paste); MCP config at `~/.cursor/mcp.json` |
| Windsurf | [windsurf/README.md](windsurf/README.md) | Fully automated; MCP config at `~/.codeium/windsurf/mcp_config.json`, lifecycle block injected into `global_rules.md` |
| OpenCode | [opencode/README.md](opencode/README.md) | Uses local MCP (`mcp` key, not `mcpServers`); lifecycle via Slowave-owned file registered in `instructions` array |
| Codex | [codex/README.md](codex/README.md) | TOML config (`~/.codex/config.toml`) shared with Codex Desktop and the IDE extension; enforcement hooks like Claude Code, lifecycle via `AGENTS.md` |

Each card is a one-screen reminder of where that client stores MCP config and lifecycle instructions. The full guide, including the exact lifecycle block, is in [docs/install.md](../docs/install.md).
