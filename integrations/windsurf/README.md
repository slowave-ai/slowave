# Windsurf + Slowave — quick-ref

Full guide: **[../../docs/install.md](../../docs/install.md)**

For upgrades, service management, and removal, see the [installation guide](../../docs/install.md).
For recovery and OS-specific logs, see [troubleshooting](../../docs/troubleshooting.md).

---

## Setup

```bash
pipx install slowave
slowave setup --client windsurf
```

`slowave setup` handles everything automatically:
- Patches Windsurf's MCP config to connect to the Slowave HTTP daemon
- Injects the lifecycle instruction block into `~/.codeium/windsurf/memories/global_rules.md`
- Installs and starts the background worker and HTTP daemon as system services

Restart / reload Windsurf.

---

## What gets configured

| What | Where |
|---|---|
| MCP server | `~/.codeium/windsurf/mcp_config.json` |
| Lifecycle instructions | `~/.codeium/windsurf/memories/global_rules.md` |
| Background worker | launchd (macOS) / systemd (Linux) / Task Scheduler (Windows) |

---

## Lifecycle instructions

`slowave setup` injects the lifecycle block into `~/.codeium/windsurf/memories/global_rules.md`.
This file is Windsurf's **global rules** surface — always-on, injected into every Cascade conversation.

**Full lifecycle documentation:** [docs/install.md#lifecycle-instruction-block](../../docs/install.md#lifecycle-instruction-block)

---

## Manual MCP config (if `slowave setup` didn't work)

Edit `~/.codeium/windsurf/mcp_config.json` (create it if missing):

```jsonc
{
  "mcpServers": {
    "slowave": {
      "type": "http",
      "url": "http://127.0.0.1:8766/mcp"
    }
  }
}
```

Make sure the daemon is running (`slowave status --services`). Restart Windsurf after editing.

---

## Verify

Open Windsurf and start a Cascade conversation. If Slowave is configured correctly, the `slowave_*` tools appear in the tool list and the lifecycle (activate → commit) runs automatically on every session — no manual invocation needed.

To confirm from the terminal:

```bash
slowave stats     # shows session/event counts
slowave doctor    # shows client detection and daemon health
```

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| Tools don't appear | Run `slowave status --services`; restart Windsurf |
| Tools appear but aren't called | `global_rules.md` block missing — re-run `slowave setup` |
| Sessions are empty | Verify `global_rules.md` has the Slowave lifecycle block — re-run `slowave setup` |

