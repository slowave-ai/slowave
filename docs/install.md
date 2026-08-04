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
| Claude Code | [integrations/claude-code/README.md](integrations/claude-code/README.md) |
| Claude Desktop ¹ | [integrations/claude-desktop/README.md](integrations/claude-desktop/README.md) |
| Cline | [integrations/cline/README.md](integrations/cline/README.md) |
| Cursor ¹ | [integrations/cursor/README.md](integrations/cursor/README.md) |
| OpenCode | [integrations/opencode/README.md](integrations/opencode/README.md) |
| Windsurf | [integrations/windsurf/README.md](integrations/windsurf/README.md) |
| Codex ² | [integrations/codex/README.md](integrations/codex/README.md) |

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
| Cline | `~/Library/Application Support/Code/.../cline_mcp_settings.json` parent exists |
| Cursor | `~/.cursor/` exists |
| OpenCode | `~/.config/opencode/` exists |
| Windsurf | `~/.codeium/windsurf/` exists |
| Codex | `~/.codex/` (or `$CODEX_HOME`) exists |

Clients not detected are listed as "Not installed — skipping" in the setup output.

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
- `slowave cleanup` removes all `*.bak.*` files.

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

Gzip snapshot of the SQLite database. Keeps the last 7 backups in `~/.slowave/backups/`.

| Platform | Service | Verify |
|---|---|---|
| macOS | `~/Library/LaunchAgents/com.slowave.backup.plist` | `launchctl list com.slowave.backup` |
| Linux | `~/.config/systemd/user/slowave-backup.timer` | `systemctl --user status slowave-backup.timer` |
| Windows | Task Scheduler: `SlowaveBackup` | `Get-ScheduledTask -TaskName SlowaveBackup` |

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

## Uninstall

```bash
slowave cleanup --dry-run  # preview
slowave cleanup            # remove all config (keeps database)
pipx uninstall slowave     # remove package
rm -rf ~/.slowave          # optional: delete all memories
```

`slowave cleanup` removes:
- All `"slowave"` MCP server entries from client configs
- Lifecycle instruction blocks (between `<!-- slowave-lifecycle-start/end -->` markers)
- Enforcement hooks (Claude Code)
- Worker, daemon, and backup services
- All `*.bak.*` backup files

**Preserved:** other MCP servers, other hooks, all other config keys, the SQLite database.

Claude Desktop and Cursor: also manually clear the lifecycle block from Settings.

---

## Trust & Transparency

- ✅ **Open Source** — [github.com/mrsalty/slowave](https://github.com/mrsalty/slowave)
- ✅ **Idempotent** — safe to re-run
- ✅ **Dry-run mode** — `slowave setup --dry-run`
- ✅ **Verification** — `slowave doctor` shows state
- ✅ **Reversible** — `slowave cleanup`
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

This is the exact block `slowave setup` injects into each client's instruction surface. Claude Desktop and Cursor users must paste it manually.

<details>
<summary>Click to expand</summary>

```md
<!-- slowave-lifecycle-start v4 -->
## MANDATORY — Slowave memory (5-verb cognitive cycle)

You are the reasoning module; Slowave is the memory module. Give it honest signals — what you encoded, what helped, what was noise, the outcome — and trust consolidation to do the rest. Do not respond until step 1 completes. Do not end the task without step 5.

**1 — `slowave_activate` (before your first response)**
`slowave_activate(query="<verbatim task>", goal="<short goal>", scope="project:<basename(cwd)>")` → store `retrieval_id`.
- `query`: the task verbatim — do not summarize (raw text drives retrieval).
- `goal`: 3–6 word verb-noun phrase (e.g. `"fix auth null pointer"`). Phrase it naturally; it is folded into the retrieval cue, so roughly consistent wording for the same kind of task gives a small overlap boost. Exact matching is NOT needed.
- `scope`: `project:<name>` (or `user:<id>` / `domain:<topic>`). Never omit.
- Call ONCE.

   **Cold start gate — if the response contains `cold_start: true`:**
   - Read the full `cold_start_hints` field in the response. It contains a numbered checklist.
   - Follow EVERY step in order. Do NOT skip steps or stop after the first file.
   - The gate is satisfied only when ALL listed files are exhausted AND slowave_remember
     has been called for every durable fact found (one call per fact, never grouped).
   - Do NOT respond to the user until the self-verification step passes.

**2 — `slowave_remember` (encode durable knowledge)**
`slowave_remember(content, type, scope="project:<basename(cwd)>")` — call per durable fact.
- Novelty gate — skip if it already surfaced in activate/recall, is reconstructible from current context, or is transient/session-only state.
- ONE fact per call (never bundle — it blurs the embedding).
- Blank-slate phrasing: write so a reader with zero session context understands it. WRONG: `"fixed it by adding the field"`. RIGHT: `"SessionReaper idle timeout defaults to 3600s; the HTTP daemon disables it (0)"`.
- `type` (pick the most specific; default `decision`): `fact` · `preference` (how the user wants things) · `decision` (choice + reason) · `constraint` (invariant) · `procedure` (repeatable steps) · `lesson` (from failure/surprise) · `warning` (hazard) · `open_question` · `task` (durable to-do) · `artifact` (produced/external ref).
- If a remembered fact changed: remember the corrected version AND flag the old one via `stale_memory_ids`/`wrong_memory_ids` in step 4.
- Never encode: what is observable right now, transient state, vague impressions, or what you did this session (step 5 captures that).

**3 — `slowave_recall` (mid-task lookups — call whenever you pivot to a new sub-question, not only on failure)**
`slowave_recall(query, scope="project:<basename(cwd)>")` — specific, semantic query. WRONG: `"what about auth"`. RIGHT: `"decision on daemon single-instance enforcement"`. Always pass `scope` (omitting returns ALL projects). Store the returned `retrieval_id`. Not a substitute for activate — a deliberate lookup you reach for as often as the task's sub-questions change, not a fallback for when activate came up empty.

**4 — `slowave_reinforce` (after ANY retrieval — reward hits, suppress noise)**
Call whenever activate/recall returned memories — not only when you used some. Penalizing noise is how the store stays clean.
`slowave_reinforce(retrieval_id=<id>, feedback="useful|partially_useful|irrelevant|stale|wrong|missing|too_much_context", outcome="success|partial|failure|unknown", used_memory_ids=[...], irrelevant_memory_ids=[...], stale_memory_ids=[...], wrong_memory_ids=[...])`
- `used_memory_ids`: IDs you actually relied on (strengthens them).
- `irrelevant`/`stale`/`wrong_memory_ids`: IDs that were noise, outdated, or incorrect (this is how the store self-cleans). Use real IDs only — never invent.
- `feedback` and `outcome`: honest, not optimistic. Use `missing` to flag a needed-but-absent memory.

**5 — `slowave_commit` (session close — always)**
`slowave_commit(scope="project:<basename(cwd)>", outcome="success|partial|failure")`. Non-negotiable. Scope must match activate; outcome honest (`partial` if anything was incomplete). Skipping = no episodes form; the session lingers until the idle reaper closes it with no outcome.

Anti-patterns: skip activate · `remember` without `scope` · bundle facts in one call · context-dependent phrasing · re-encode facts already surfaced · leave a superseded fact unflagged · reinforce only hits and never penalize noise · default feedback to `useful` · invent memory IDs · report `success` when partial/failed · skip reinforce or commit · use deleted tools (`slowave_context`, `slowave_session_start/end`, `slowave_event`, `slowave_retrieval_feedback`, `slowave_context_feedback`).
<!-- slowave-lifecycle-end v4 -->
```

</details>

| Client | Location | `agent` value |
|---|---|---|
| Claude Code | `~/.claude/CLAUDE.md` (global) or repo `CLAUDE.md` | `claude-code` |
| Claude Desktop | **Settings → General → Instructions for Claude** | `claude-desktop` |
| Cline | `~/.cline/rules/slowave.md` or repo `.clinerules` | `cline-tui` |
| Cursor | **Settings → Rules for AI** (or repo `.cursorrules`) | `cursor` |
| OpenCode | `~/.config/opencode/slowave-instructions.md` | `opencode` |
| Windsurf | `~/.codeium/windsurf/memories/global_rules.md` | `windsurf` |

Claude Code, Cline, Windsurf, and OpenCode are injected automatically.
Claude Desktop and Cursor require manual paste.

---

## Questions?

- 🩺 Run `slowave doctor` to check status
- 🐛 Report issues on [GitHub](https://github.com/mrsalty/slowave/issues)
