# Slowave — client integrations

Full install and setup guide: **[../docs/install.md](../docs/install.md)**

---

## Quick-start (all clients)

```bash
pipx install slowave
slowave setup
slowave doctor
```

`slowave setup` auto-configures every client it detects, injects the same Slowave lifecycle instructions into each client's instruction surface, installs the background worker, and starts the HTTP MCP daemon. It is safe to repeat and restarts managed services on every run.

For upgrades and service commands, see the [installation guide](../docs/install.md#upgrade).
For OS-specific logs and recovery, see [troubleshooting](../docs/troubleshooting.md).

**Remove Slowave (choose uninstall or purge, then remove the package):**
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
| Claude Code | [claude-code/README.md](claude-code/README.md) | CLAUDE.md lifecycle instructions |
| Copilot CLI | [copilot/README.md](copilot/README.md) | Local CLI only; automatic discovery through `copilot` on PATH or its config directory; personal lifecycle instructions |
| Codex | [codex/README.md](codex/README.md) | TOML config (`~/.codex/config.toml`) shared with Codex Desktop and the IDE extension; lifecycle via `AGENTS.md` |
| Cursor | [cursor/README.md](cursor/README.md) | Requires Rules for AI (one manual paste); MCP config at `~/.cursor/mcp.json` |
| OpenCode | [opencode/README.md](opencode/README.md) | Uses local MCP (`mcp` key, not `mcpServers`); lifecycle via Slowave-owned file registered in `instructions` array |
| Windsurf | [windsurf/README.md](windsurf/README.md) | Fully automated; MCP config at `~/.codeium/windsurf/mcp_config.json`, lifecycle block injected into `global_rules.md` |
| Cline | [cline/README.md](cline/README.md) | Same lifecycle block injected into `.clinerules` |
| Claude Desktop | [claude-desktop/README.md](claude-desktop/README.md) | Requires Custom Instructions (one manual paste) |

Each card is a one-screen reminder of where that client stores MCP config and lifecycle instructions. The full guide, including the exact lifecycle block, is in [docs/install.md](../docs/install.md).
