# Cursor + Slowave — quick-ref

Full guide: **[../../docs/install.md](../../docs/install.md)**

For upgrades, service management, and removal, see the [installation guide](../../docs/install.md).
For recovery and OS-specific logs, see [troubleshooting](../../docs/troubleshooting.md).

---

## Setup

```bash
pipx install slowave
slowave setup --client cursor
```

`slowave setup` handles the MCP server and background worker automatically.
One manual step remains: add the lifecycle instructions to Cursor.

---

## Step 1 — Add Rules for AI (required)

Open Cursor → **Settings → Rules for AI** (or add a `.cursorrules` file at the root of your project).

**Paste the lifecycle instructions from:** [docs/install.md#lifecycle-instruction-block](../../docs/install.md#lifecycle-instruction-block)

> **Why Rules for AI, not just the MCP config?**
> Cursor's MCP integration exposes tools but does not automatically instruct the AI _when_ to call them. The Rules for AI field fires on every request and ensures Slowave's session/event lifecycle is honoured from the first message.

---

## What `slowave setup` configures automatically

| What | Where |
|---|---|
| MCP server | `~/.cursor/mcp.json` |
| Background worker | launchd user service (macOS) / systemd (Linux) / Task Scheduler (Windows) |

---

## Manual MCP config (if `slowave setup` didn't work)

Edit `~/.cursor/mcp.json` (create it if missing):

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

Make sure the daemon is running (`slowave status --services`). Restart Cursor after editing.

---

## Verify

Open Cursor and start a conversation with a model that supports tool use. If Slowave is configured correctly, the `slowave_*` tools appear in the tool list and the lifecycle (activate → commit) runs automatically — no manual invocation needed.

To confirm from the terminal:

```bash
slowave stats     # shows session/event counts
slowave doctor    # shows client detection and daemon health
```

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| Tools don't appear | Run `slowave status --services`; restart Cursor |
| Tools appear but aren't called | Rules for AI block not set — complete Step 1 above |
| Sessions are empty | Confirm Rules for AI block is set — complete Step 1 above |

