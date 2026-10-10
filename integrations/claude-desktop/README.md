# Claude Desktop + Slowave — quick-ref

Full guide: **[../../docs/install.md](../../docs/install.md)**

For upgrades, service management, and removal, see the [installation guide](../../docs/install.md).
For recovery and OS-specific logs, see [troubleshooting](../../docs/troubleshooting.md).

---

## Setup

```bash
pipx install slowave
slowave setup --client claude-desktop
```

`slowave setup` configures the MCP server and background worker automatically.
One manual step remains: add the lifecycle instructions to Claude Desktop.

---

## Step 1 — Add Custom Instructions (required)

Open Claude Desktop → **Settings → General → Instructions for Claude**

**Paste the lifecycle instructions from:** [docs/install.md#lifecycle-instruction-block](../../docs/install.md#lifecycle-instruction-block)

> **Why Custom Instructions, not a Skill?**
> Claude Desktop Skills fire from turn 2 onward. Custom Instructions fire before turn 1.
> The Custom Instructions field is stored server-side — `slowave setup` cannot patch it automatically.

---

## What `slowave setup` configures automatically

| What | Where |
|---|---|
| MCP server | `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS) |
| Background worker | launchd user service (macOS) / systemd (Linux) / Task Scheduler (Windows) |

---

## Manual MCP config (if `slowave setup` didn't work)

Edit `~/Library/Application Support/Claude/claude_desktop_config.json`:

```jsonc
{
  "mcpServers": {
    "slowave": {
      "command": "/path/to/slowave-mcp"
    }
  }
}
```

Replace `/path/to/slowave-mcp` with the actual path — run `which slowave-mcp` to find it (e.g. `~/.local/bin/slowave-mcp`). Restart Claude Desktop after editing.

> **Note:** Claude Desktop uses stdio transport (command-based). It does NOT support the `url` or `type: "http"` formats — those are for Claude Code / Cline only.

---

## Verify

Open Claude Desktop and start a conversation. If Slowave is configured correctly, the `slowave_*` tools appear in the tool list and the lifecycle (activate → commit) runs automatically — no manual invocation needed.

To confirm from the terminal:

```bash
slowave stats     # shows session/event counts
slowave doctor    # shows client detection and daemon health
```

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| Tools don\'t appear | Check MCP path (`slowave setup --dry-run`), check JSON syntax, restart Claude Desktop |
| Tools appear but aren\'t called | Custom Instructions not set — complete Step 1 above |
| Only works from turn 2 | Custom Instructions not set (Skills fire too late) — complete Step 1 above |
| Sessions are empty | Confirm Custom Instructions are set — complete Step 1 above |

