# Install & Setup

The complete reference for installing, setting up, and uninstalling Slowave — what `slowave setup` does, what files it touches, and how to undo it.

## Installation

### Global setup

Install Slowave and configure every detected client in one go:

```bash
pipx install slowave

# or

brew tap mrsalty/slowave https://github.com/mrsalty/slowave
brew install slowave
```

Then wire everything up:

```bash
slowave setup --dry-run   # preview what will change
slowave setup             # apply: MCP configs, lifecycle instructions, hooks, services
slowave doctor            # verify: daemon health, client detection
```

`slowave setup` is idempotent and safe to run multiple times. The HTTP MCP daemon and background consolidation worker start automatically as system services.

Claude Desktop and Cursor require one manual paste after setup because their instruction surfaces cannot be modified programmatically. `slowave setup` prints the exact text and path.
### Per-client setup

To configure a single client, or to find client-specific details:

| Client | Integration doc |
|---|---|
| Claude Code | [integrations/claude-code/README.md](../integrations/claude-code/README.md) |
| Claude Desktop ¹ | [integrations/claude-desktop/README.md](../integrations/claude-desktop/README.md) |
| Cline | [integrations/cline/README.md](../integrations/cline/README.md) |
| Cursor ¹ | [integrations/cursor/README.md](../integrations/cursor/README.md) |
| OpenCode | [integrations/opencode/README.md](../integrations/opencode/README.md) |
| Windsurf | [integrations/windsurf/README.md](../integrations/windsurf/README.md) |
| Codex ² | [integrations/codex/README.md](../integrations/codex/README.md) |

¹ requires one manual paste after setup
² also configures Codex Desktop (ChatGPT app) and the Codex IDE extension — all three share `~/.codex/config.toml`

## What `slowave setup` does

| Action | Clients | Detail |
|---|---|---|
| MCP config | All | Patches each client's MCP config so `slowave_*` tools appear |
| Lifecycle instructions | Claude Code, Cline, Windsurf, OpenCode, Codex | Injects the mandatory Slowave block automatically |
| Lifecycle instructions | Claude Desktop, Cursor | Prints the block to paste — requires one manual step |
| Enforcement hooks | Claude Code, Codex | Adds `UserPromptSubmit` + `Stop` hooks so the client calls Slowave every turn |
| HTTP daemon | All | Installs as launchd/systemd/Task Scheduler — auto-starts |
| Background worker | All | Installs as launchd/systemd/Task Scheduler — consolidates events |
| Daily backup | All | Installs as launchd/systemd/Task Scheduler — gzip snapshot of the database |

Options:

```
slowave setup --client [claude-code|claude-desktop|cline|cursor|opencode|windsurf|codex|all]
              --no-worker       # skip worker service install
              --no-hooks        # skip Claude Code / Codex hooks
              --dry-run         # preview without writing
```

---

## Client detection

`slowave setup` only configures clients it detects on your machine. Detection checks whether each client's config directory exists — if the directory isn't there, the client is skipped silently.

| Client | Detection |
|---|---|
| Claude Code | `~/.claude/` exists |
| Claude Desktop | `~/Library/Application Support/Claude/` (macOS) or equivalent exists |
| Cline | `~/.cline/` (CLI/TUI) or `~/Library/Application Support/Code/.../cline_mcp_settings.json` parent exists |
| Cursor | `~/.cursor/` exists |
| OpenCode | `~/.config/opencode/` exists |
| Windsurf | `~/.codeium/windsurf/` exists |
| Codex | `~/.codex/` (or `$CODEX_HOME`) exists |

Clients not detected are omitted from the setup output.

---

## Backup files

Before overwriting any config file, `slowave setup` creates a timestamped copy next to the original:

```
~/.claude.json.bak.20260611_142300
~/.claude/settings.json.bak.20260611_142300
~/.claude/CLAUDE.md.bak.20260611_142300
```

- The backup path is printed during setup.
- Only **one backup per file** — re-running replaces the previous backup.
- `slowave purge` removes all `*.bak.*` files. (`slowave cleanup` is a compatibility alias.)

To restore: `cp ~/.claude.json.bak.20260611_142300 ~/.claude.json`

---

## Files modified (by platform)

### macOS

| File | Purpose | What Changes |
|---|---|---|
| `~/.claude.json` | Claude Code MCP config | Adds `mcpServers.slowave` entry (user-scope MCP registry) |
| `~/.claude/settings.json` | Claude Code hooks | Adds `hooks.UserPromptSubmit` and `hooks.Stop` |
| `~/.claude/CLAUDE.md` | Claude Code instructions | Prepends lifecycle block |
| `~/Library/Application Support/Claude/claude_desktop_config.json` | Claude Desktop MCP config | Adds `mcpServers.slowave` entry |
| `~/.cline/rules/slowave.md` | Cline instructions | Prepends lifecycle block |
| `~/.cline/data/settings/cline_mcp_settings.json` | Cline MCP config (CLI/TUI) | Adds `mcpServers.slowave` entry |
| `~/.config/Code/User/globalStorage/.../cline_mcp_settings.json` | Cline MCP config (VS Code) | Adds `mcpServers.slowave` entry |
| `~/.config/Cursor/User/globalStorage/.../cline_mcp_settings.json` | Cline MCP config (Cursor) | Adds `mcpServers.slowave` entry |
| `~/.cursor/mcp.json` | Cursor native MCP config | Adds `mcpServers.slowave` entry |
| `~/.codeium/windsurf/mcp_config.json` | Windsurf MCP config | Adds `mcpServers.slowave` entry |
| `~/.codeium/windsurf/memories/global_rules.md` | Windsurf global rules | Prepends lifecycle block |
| `~/.config/opencode/opencode.json` | OpenCode MCP + instructions config | Adds `mcp.slowave` and registers instructions file |
| `~/.config/opencode/slowave-instructions.md` | OpenCode lifecycle instructions | Creates Slowave-owned instruction file |
| `~/.codex/config.toml` | Codex MCP config + hooks | Adds `[mcp_servers.slowave]` and `[[hooks.UserPromptSubmit/Stop]]` (single combined write) |
| `~/.codex/AGENTS.md` | Codex instructions | Prepends lifecycle block |
| `~/Library/LaunchAgents/com.slowave.worker.plist` | Background worker | launchd plist, loads with `launchctl` |
| `~/Library/LaunchAgents/com.slowave.daemon.plist` | HTTP MCP daemon | launchd plist, auto-starts on load |
| `~/Library/LaunchAgents/com.slowave.backup.plist` | Daily backup | launchd plist with `StartCalendarInterval` |

### Linux

| File | Purpose | What Changes |
|---|---|---|
| `~/.claude.json` | Claude Code MCP config | Same as macOS |
| `~/.claude/settings.json` | Claude Code hooks | Same as macOS |
| `~/.claude/CLAUDE.md` | Claude Code instructions | Same as macOS |
| `~/.config/Claude/claude_desktop_config.json` | Claude Desktop MCP config | Adds `mcpServers.slowave` entry |
| `~/.cline/rules/slowave.md` | Cline instructions | Same as macOS |
| `~/.cline/data/settings/cline_mcp_settings.json` | Cline MCP config (CLI/TUI) | Same as macOS |
| `~/.config/Code/User/globalStorage/.../cline_mcp_settings.json` | Cline MCP config | Same as macOS |
| `~/.config/systemd/user/slowave-worker.service` | Background worker | systemd user service, enabled with `systemctl --user` |
| `~/.config/systemd/user/slowave-daemon.service` | HTTP MCP daemon | systemd user service, auto-starts |
| `~/.config/systemd/user/slowave-backup.service` | Daily backup | systemd oneshot service |
| `~/.config/systemd/user/slowave-backup.timer` | Daily backup timer | Triggers backup daily |
| `~/.config/opencode/opencode.json` | OpenCode MCP + instructions config | Same as macOS |
| `~/.config/opencode/slowave-instructions.md` | OpenCode lifecycle instructions | Same as macOS |
| `~/.cursor/mcp.json` | Cursor native MCP config | Same as macOS |
| `~/.codeium/windsurf/mcp_config.json` | Windsurf MCP config | Same as macOS |
| `~/.codeium/windsurf/memories/global_rules.md` | Windsurf global rules | Same as macOS |
| `~/.codex/config.toml` | Codex MCP config + hooks | Same as macOS |
| `~/.codex/AGENTS.md` | Codex instructions | Same as macOS |

### Windows

| File | Purpose | What Changes |
|---|---|---|
| `%USERPROFILE%\.claude.json` | Claude Code MCP config | Same as macOS |
| `%USERPROFILE%\.claude\settings.json` | Claude Code hooks | Same as macOS |
| `%USERPROFILE%\.claude\CLAUDE.md` | Claude Code instructions | Same as macOS |
| `%APPDATA%\Claude\claude_desktop_config.json` | Claude Desktop MCP config | Adds `mcpServers.slowave` entry |
| `%USERPROFILE%\.clinerules` | Cline instructions | Same as macOS |
| `%USERPROFILE%\.cline\data\settings\cline_mcp_settings.json` | Cline MCP config (CLI/TUI) | Same as macOS |
| `%APPDATA%\Code\User\globalStorage\.../cline_mcp_settings.json` | Cline MCP config | Same as macOS |
| Task Scheduler | Background worker | Registers `SlowaveWorker` task |
| Task Scheduler | HTTP MCP daemon | Registers `SlowaveDaemon` task |
| Task Scheduler | Daily backup | Registers `SlowaveBackup` task |
| `%USERPROFILE%\.config\opencode\opencode.json` | OpenCode MCP + instructions config | Same as macOS |
| `%USERPROFILE%\.config\opencode\slowave-instructions.md` | OpenCode lifecycle instructions | Same as macOS |
| `%USERPROFILE%\.cursor\mcp.json` | Cursor native MCP config | Same as macOS |
| `%APPDATA%\Codeium\windsurf\mcp_config.json` | Windsurf MCP config | Same as macOS |
| `%APPDATA%\Codeium\windsurf\memories\global_rules.md` | Windsurf global rules | Same as macOS |
| `%USERPROFILE%\.codex\config.toml` | Codex MCP config + hooks | Same as macOS |
| `%USERPROFILE%\.codex\AGENTS.md` | Codex instructions | Same as macOS |

---

## Services

### HTTP MCP daemon

Serves the `slowave_*` tools at `http://127.0.0.1:8766/mcp`. All clients connect to it.

| Platform | Service | Verify |
|---|---|---|
| macOS | `~/Library/LaunchAgents/com.slowave.daemon.plist` | `launchctl list \| grep slowave` |
| Linux | `~/.config/systemd/user/slowave-daemon.service` | `systemctl --user status slowave-daemon` |
| Windows | Task Scheduler: `SlowaveDaemon` | `Get-ScheduledTask -TaskName SlowaveDaemon` |

### Background worker

Runs consolidation offline — transforms raw events into searchable schemas.

| Platform | Service | Verify |
|---|---|---|
| macOS | `~/Library/LaunchAgents/com.slowave.worker.plist` | `launchctl list \| grep slowave` |
| Linux | `~/.config/systemd/user/slowave-worker.service` | `systemctl --user status slowave-worker` |
| Windows | Task Scheduler: `SlowaveWorker` | `Get-ScheduledTask -TaskName SlowaveWorker` |

### Daily backup

Gzip snapshot of the SQLite database. Keeps the last 7 backups in the effective
runtime root's `backups/` directory.

| Platform | Service | Verify |
|---|---|---|
| macOS | `~/Library/LaunchAgents/com.slowave.backup.plist` | `launchctl list com.slowave.backup` |
| Linux | `~/.config/systemd/user/slowave-backup.timer` | `systemctl --user status slowave-backup.timer` |
| Windows | Task Scheduler: `SlowaveBackup` | `Get-ScheduledTask -TaskName SlowaveBackup` |

### Runtime data location and migration

Slowave isolates runtime data by operating-system user. The default root is
the native application-data directory selected by `platformdirs`:

| Platform | Typical default runtime root |
|---|---|
| macOS | `~/Library/Application Support/slowave` |
| Linux/Unix | `$XDG_DATA_HOME/slowave`, normally `~/.local/share/slowave` |
| Windows | `%LOCALAPPDATA%\slowave` |

The database, SQLite sidecars, daemon PID, logs, backups, setup sentinel, and
diagnostic logs stay beneath that root. `slowave doctor` prints the effective
root and database path.

Set `SLOWAVE_HOME` to relocate the complete runtime tree for CI, containers,
portable installs, or an intentionally shared operator-managed deployment.
`SLOWAVE_DB` remains a legacy exact-database override; its parent becomes the
runtime root. Setting both variables is an error. A shared root is not
multi-tenant isolation: run one daemon under the intended service account and
protect the directory with OS permissions.

After upgrading an installation that used `~/.slowave`, migrate explicitly:

```bash
slowave migrate-data --dry-run
slowave migrate-data
slowave doctor
```

Migration stops a legacy daemon if necessary, copies SQLite through its online
backup API, validates `PRAGMA integrity_check`, and promotes a staged directory.
It refuses to merge into a non-empty destination and preserves `~/.slowave` for
rollback. To roll back, stop the new daemon and run commands with
`SLOWAVE_HOME=~/.slowave` (or point `SLOWAVE_DB` at the exact legacy database).

---

## What does NOT get modified

| ❌ Never touched | Why |
|---|---|
| Python packages | Installed via pip/pipx — no auto-upgrades |
| Shell profiles (`.bashrc`, `.zshrc`, etc.) | No PATH modifications |
| System-wide configs (`/etc`, `/usr/local`) | User-scoped only |
| VSCode/Cursor settings.json | Only Cline's dedicated MCP settings file |
| Claude Desktop Custom Instructions | Server-side, cannot be automated |
| Existing file content (outside markers) | Lifecycle blocks use markers |

---

## Verification

```bash
slowave doctor          # shows detected clients and config status
slowave setup --dry-run # preview without writing
```

---

## Remove Slowave

Slowave has three distinct removal operations. Choose the smallest one that
matches your goal. Run the dry run first whenever possible.

### Stop using Slowave but keep its memories

```bash
slowave uninstall --dry-run
slowave uninstall
```

`slowave uninstall` removes every Slowave-managed client integration: MCP
entries, generated lifecycle instructions and hooks, plus the HTTP daemon,
background worker, and daily-backup services. It preserves:

- the effective runtime root, including the SQLite database and archives
- setup-created `*.bak.*` configuration backups
- unrelated client configuration, MCP entries, hooks, and instruction content
- the installed Slowave package

Claude Desktop and Cursor lifecycle instructions are pasted into their UI and
cannot be removed by a command. Remove the Slowave text manually from **Claude
Desktop → Settings → General → Instructions for Claude** and **Cursor →
Settings → Rules for AI**.

### Remove integrations and local data

```bash
slowave purge --dry-run
slowave purge
```

`slowave purge` performs `uninstall` and then removes local Slowave data from
the effective runtime root, including the SQLite database. It also removes setup-created
`*.bak.*` configuration backups. This is destructive and asks for confirmation.

Database archives already stored in the runtime root's `backups/` directory are intentionally
retained so memories can be recovered. Delete that directory yourself only if
you have confirmed that those archives are no longer needed.

`slowave cleanup` is retained as a compatibility alias for `slowave purge`.
Use `purge` in new scripts and documentation.

### Remove the installed package

Slowave does not remove its own Python package. Use the same installer used to
install it, after `uninstall` or `purge` as appropriate:

```bash
pipx uninstall slowave
# or, if installed with Homebrew:
brew uninstall slowave
```

---

## Trust & Transparency

- ✅ **Open Source** — [github.com/mrsalty/slowave](https://github.com/mrsalty/slowave)
- ✅ **Idempotent** — safe to re-run
- ✅ **Dry-run mode** — `slowave setup --dry-run`
- ✅ **Verification** — `slowave doctor` shows state
- ✅ **Reversible setup removal** — `slowave uninstall` preserves local memories
- ✅ **No telemetry** — no analytics, no data collection
- ✅ **Local-first** — all data stays on your machine

---

## Reference

### MCP config block

All clients except Claude Desktop use this HTTP transport block:

```json
{
  "mcpServers": {
    "slowave": {
      "type": "http",
      "url": "http://127.0.0.1:8766/mcp"
    }
  }
}
```

Claude Desktop uses stdio transport. OpenCode uses the `mcp` key (not `mcpServers`). Codex
uses TOML, not JSON:

```toml
[mcp_servers.slowave]
url = "http://127.0.0.1:8766/mcp"
```

| Client | Config file |
|---|---|
| Claude Code | `~/.claude.json` |
| Claude Desktop (macOS) | `~/Library/Application Support/Claude/claude_desktop_config.json` |
| Claude Desktop (Windows) | `%APPDATA%\Claude\claude_desktop_config.json` |
| Claude Desktop (Linux) | `~/.config/Claude/claude_desktop_config.json` |
| Cline (VS Code / Cursor) | `.../cline_mcp_settings.json` |
| Cursor | `~/.cursor/mcp.json` |
| OpenCode | `~/.config/opencode/opencode.json` |
| Windsurf | `~/.codeium/windsurf/mcp_config.json` |
| Codex | `~/.codex/config.toml` (shared by the CLI, Codex Desktop, and the IDE extension) |

### Lifecycle instruction block

`slowave setup` installs the current lifecycle instructions for clients it can
configure. Claude Desktop and Cursor require a manual paste because their
instruction surfaces cannot be changed programmatically; setup prints the
current text and destination. The generated instructions are the authoritative
contract. They require the connected agent to:

1. activate a scoped task session and receive relevant recorded memory;
2. remember durable claims when appropriate;
3. recall during a task when the question changes;
4. assess every retrieved memory or procedure as used, irrelevant, or stale;
5. commit an honest outcome and any reusable procedure.

See [architecture.md](architecture.md) for the current tool contracts and
feedback requirements.

| Client | Location | `agent` value |
|---|---|---|
| Claude Code | `~/.claude/CLAUDE.md` (global) or repo `CLAUDE.md` | `claude-code` |
| Claude Desktop | **Settings → General → Instructions for Claude** | `claude-desktop` |
| Cline | `~/.cline/rules/slowave.md` or repo `.clinerules` | `cline-tui` |
| Cursor | **Settings → Rules for AI** (or repo `.cursorrules`) | `cursor` |
| OpenCode | `~/.config/opencode/slowave-instructions.md` | `opencode` |
| Windsurf | `~/.codeium/windsurf/memories/global_rules.md` | `windsurf` |
| Codex | `~/.codex/AGENTS.md` | `codex` |

Claude Code, Cline, Windsurf, OpenCode, and Codex are injected automatically.
Claude Desktop and Cursor require manual paste.

---

## Questions?

- 🩺 Run `slowave doctor` to check status
- 🐛 Report issues on [GitHub](https://github.com/mrsalty/slowave/issues)
