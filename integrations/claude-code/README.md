# Claude Code + Slowave — quick-ref

Full guide: **[../../docs/install.md](../../docs/install.md)**

---

## Setup

```bash
pipx install slowave
slowave setup --client claude-code
```

`slowave setup` handles everything automatically:
- Adds the MCP server entry to `~/.claude.json` (user-scope MCP registry)
- Injects the lifecycle instruction block into `~/.claude/CLAUDE.md`
- Installs and starts the background worker and HTTP daemon as system services

Restart Claude Code.

---

## What gets configured

| What | Where |
|---|---|
| MCP server | `~/.claude.json` (user-scope MCP registry) |
| Lifecycle instructions | `~/.claude/CLAUDE.md` |
| Background worker | launchd (macOS) / systemd (Linux) / Task Scheduler (Windows) |

---

## Lifecycle instructions

`slowave setup` injects the lifecycle block into `~/.claude/CLAUDE.md`.

**Full lifecycle documentation:** [docs/install.md#lifecycle-instruction-block](../../docs/install.md#lifecycle-instruction-block)

---

## Manual MCP config (if `slowave setup` didn't work)

Edit `~/.claude.json` (create it if it doesn't exist):

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

Make sure the daemon is running (`slowave serve status`). Restart Claude Code after editing.

---

## Verify

Open Claude Code and start any coding task. If Slowave is configured correctly, the `slowave_*` tools appear in the tool list and the lifecycle (activate → commit) runs automatically on every session — no manual invocation needed.

To confirm from the terminal:

```bash
slowave stats     # shows session/event counts
slowave doctor    # shows client detection and daemon health
```

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| Tools don't appear | Run `slowave serve status`; restart Claude Code |
| Tools appear but aren't called | The `CLAUDE.md` lifecycle block may be missing or stale — re-run `slowave setup` |
| Sessions are empty | Check that the lifecycle block is installed and the `slowave_*` tools are available |
