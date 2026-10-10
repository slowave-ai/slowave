# Install, upgrade, and manage Slowave

Use **pipx, pip in a dedicated virtual environment, or Homebrew** to manage the
installed package. Use **Slowave commands** to configure clients and manage its
user services. Use the same installer, Python environment, OS account, and
runtime settings throughout. Do not run user-service commands with sudo or from
a different administrator account.

- [Install](#installation)
- [Start, stop, restart, and inspect](#manage-services)
- [Upgrade and recover from a failed upgrade](#upgrade)
- [Remove integrations, data, or the package](#remove-slowave)
- [Troubleshooting and logs](troubleshooting.md)
- [CLI reference](cli.md)

Run `slowave docs` to open this guide, or `slowave docs troubleshooting` for
recovery steps. Add `--no-open` to print the link without opening a browser.

## Installation

### Requirements by OS

Python **3.11 or newer** is required. Install [pipx](https://pipx.pypa.io/latest/how-to/install-pipx.html)
if it is not already available, run `pipx ensurepath`, and reopen your terminal
before installing Slowave.

| OS | Terminal and service requirements |
|---|---|
| macOS | Terminal in your logged-in desktop account; services use launchd's GUI user domain |
| Linux | Terminal with a working **systemd user manager** (`systemctl --user`); services run in your user session |
| Windows | **PowerShell** in your logged-in account with Task Scheduler available; tasks use interactive logon and do not run while logged out |

Containers, Linux without systemd, and headless macOS sessions need
[manual service management](#manual-service-management). WSL follows the Linux
path and requires a working systemd user manager; it does not use Windows tasks.
On managed machines, OS policy may restrict service/task registration.
Windows task names include the logged-in account's SID, so multiple accounts
can each manage their own service set. Setup also migrates that account's older
root-level tasks when their recorded owner matches; it leaves other accounts'
tasks alone.
On every OS, one managed service set is shared by clients in the selected account;
multiple Python installations do not create independent managed runtimes.

### Recommended: pipx (macOS, Linux, Windows)

These commands work in a Unix terminal or Windows PowerShell:

```text
pipx install slowave
slowave setup --dry-run
slowave setup
slowave status --services
slowave doctor
```

Setup configures detected clients, restarts daemon/worker, and enables the daily
backup schedule. Complete the manual instruction paste when prompted for
Claude Desktop or Cursor, then restart/reload your MCP clients.

If `slowave` is not found, run `pipx ensurepath` and reopen the terminal. Use
`pipx list` to confirm the installation. Do not install a second copy just to
solve a PATH problem.

### Alternative: pip in a dedicated virtual environment

Use a permanent directory: registered services refer to this environment by
absolute path. Moving or deleting it breaks service startup. The following
examples use Python 3.11 or newer already installed on your system.

**macOS / Linux:**

```bash
python3 -m venv "$HOME/.venvs/slowave"
"$HOME/.venvs/slowave/bin/python" -m pip install slowave
"$HOME/.venvs/slowave/bin/slowave" setup
```

**Windows PowerShell:**

```powershell
py -m venv "$env:LOCALAPPDATA\slowave-venv"
& "$env:LOCALAPPDATA\slowave-venv\Scripts\python.exe" -m pip install slowave
& "$env:LOCALAPPDATA\slowave-venv\Scripts\slowave.exe" setup
```

For the commands below, activate that environment or substitute the full
`slowave` executable path shown above. Activation is optional when using full
paths. Upgrade with the **same environment's Python**, not a different global
`pip` or `py` installation. Avoid modifying an OS-managed Python installation.

### Alternative: Homebrew (macOS)

```bash
brew tap mrsalty/slowave https://github.com/mrsalty/slowave
brew install slowave
slowave setup
slowave status --services
slowave doctor
```

Use Homebrew's upgrade/uninstall commands for this installation. The pipx path
above is the common documented path across all three operating systems.

## Manage services

| Command | Purpose |
|---|---|
| `slowave setup` | Configure detected clients, install/reapply daemon, worker and backup registrations, restart services, and check the daemon version |
| `slowave setup --dry-run` | Preview configuration and service work |
| `slowave start` | Start registered daemon and worker; resume the backup schedule |
| `slowave stop` | Stop registered services and suspend automatic recovery/backup scheduling |
| `slowave restart` | Stop and start registered services through their OS supervisor |
| `slowave status --services` | Inspect supervisor state, installed/live daemon versions, and runtime/log paths without opening the database |
| `slowave doctor` | Diagnose runtime and client configuration |
| `slowave dashboard` | Run the dashboard in this terminal; Ctrl+C stops it |
| `slowave backup` | Create an immediate consistent database snapshot |
| `slowave restore FILE.db.gz` | Restore a snapshot after stopping services; services stay stopped until `start` |
| `slowave uninstall` | Remove integrations/services, preserving memories and the installed package |
| `slowave purge` | Remove integrations/services and local data; retain database archives |
| `slowave docs [TOPIC]` | Open this guide, troubleshooting, or the CLI reference |

`start` and `restart` need service registrations from `setup`. They do not update
client configuration. Use `setup` after a package upgrade or when repairing
client integration. `setup --force` explicitly reapplies client configuration;
it is **not required to apply upgraded service code**. Every normal setup run
reapplies/restarts services, even when client configuration is unchanged; expect
a brief MCP interruption. `--client codex` selects client configuration only:
it still reapplies the shared services.

`start` and `restart` check the daemon's running package version and database path. They do not
prove worker consolidation or backup completion; inspect supervisor state and
run history separately. Reinstalling a development build with the same version
can change its code: restart services even if the version strings match.

The dashboard, manual workers, and foreground servers are owned by their
terminal or custom supervisor. Stop those with Ctrl+C before upgrades,
restores, or removal. Service commands do not kill arbitrary processes by name.
Refresh the dashboard browser tab after relaunching its Python process.

The daily backup is scheduled, not immediately executed by `start`/`restart`.
On macOS/Linux, enabled services may start again at your next login. Windows
`stop` disables tasks until `start` or `setup`. Use `uninstall` for permanent
removal of registrations. Windows stopping may terminate in-progress work;
create a backup before upgrading.

### Manual service management

If the OS user-service manager is unavailable, use:

```text
slowave setup --no-worker
slowave serve start
```

Run `slowave worker --interval 300` in another terminal and `slowave backup`
manually or with your own scheduler. `--no-worker` skips **daemon, worker, and
backup registration**, despite the historical option name. Stop/relaunch these
foreground processes yourself; top-level service commands manage registered
OS services. `serve start` and `worker` are advanced foreground commands.
Legacy `serve stop/restart` control only the registered daemon through its
supervisor; use top-level `stop/restart` for the whole managed runtime.

## Upgrade

Stop any foreground dashboard, worker, or server with Ctrl+C. Then run:

```text
slowave backup
slowave stop
```

Keep the printed backup path. If there is no database yet, there are no memories
to back up; continue after confirming that this is expected. If backup or stop
fails for another reason, resolve that failure before upgrading.

Upgrade using **one** matching installer:

| Installation | Upgrade command |
|---|---|
| pipx, all OS | `pipx upgrade slowave` |
| pip venv, macOS/Linux example above | `"$HOME/.venvs/slowave/bin/python" -m pip install --upgrade slowave` |
| pip venv, Windows PowerShell example above | `& "$env:LOCALAPPDATA\slowave-venv\Scripts\python.exe" -m pip install --upgrade slowave` |
| Homebrew, macOS | `brew upgrade slowave` |

Then:

```text
slowave setup
slowave status --services
slowave doctor
```

Restart/reload connected MCP clients to refresh cached tool definitions and
instructions. Relaunch `slowave dashboard` if wanted. Confirm installed and
live daemon versions match and the worker is active. A scheduled backup can be
idle between runs; a running daemon alone does not prove all services work.

Stopping services before replacing files avoids mixed old/new Python imports
and Windows file locks. `setup` rebinds service registrations to the current
installation and refreshes client instructions. Use the same `SLOWAVE_HOME` or
legacy `SLOWAVE_DB` override used for the original setup, if any.

### Upgrading from a release without `slowave stop`

Stop the old jobs through their OS supervisor first. On macOS, run
`launchctl bootout gui/$(id -u)/com.slowave.daemon`, then the same command for
`com.slowave.worker` and `com.slowave.backup`. A not-loaded job needs no stop.
On Linux, run `systemctl --user stop slowave-daemon.service slowave-worker.service slowave-backup.timer slowave-backup.service`.

On Windows PowerShell, disable recovery triggers before stopping tasks:

```powershell
$sid = [System.Security.Principal.WindowsIdentity]::GetCurrent().User.Value
$tasks = Get-ScheduledTask | Where-Object {
    ($_.TaskName -like "Slowave*-$sid") -or
    (($_.TaskName -in @('SlowaveDaemon', 'SlowaveWorker', 'SlowaveBackup')) -and
     $_.Principal.UserId -and
     ([System.Security.Principal.NTAccount]::new($_.Principal.UserId).Translate(
       [System.Security.Principal.SecurityIdentifier]).Value -eq $sid))
}
$tasks | ForEach-Object {
    Disable-ScheduledTask -InputObject $_ | Out-Null
    Stop-ScheduledTask -InputObject $_
}
```

Confirm those jobs have stopped, upgrade with the original installer, and run
`slowave setup`. This fallback is for old versions; use `slowave stop` afterward.

### Failed upgrade or wrong running version

If installation fails, services remain stopped. Repair the original environment
with the same package manager and rerun `setup`. Do not start a partly upgraded
environment. For a known previous release, pip can install `slowave==VERSION`;
pipx supports `pipx install --force 'slowave==VERSION'`. Stop services first.
An older release may not read a newer database. Keep a pre-upgrade snapshot and
consult release notes before downgrading or restoring.

If you already upgraded while services were running, use `slowave setup`, then
compare installed/live versions with `slowave status --services`. A CLI version
or browser refresh alone does not prove running Python processes were replaced.
See [troubleshooting](troubleshooting.md#old-version-after-an-upgrade) if they
still differ.

On Windows, setup registers the Worker and HTTP daemon with a logon trigger
and a five-minute recovery trigger. While a task is running, `IgnoreNew` skips
the recovery launches; the Worker's `--interval 300` controls consolidation
inside that process. Setup launches `pythonw.exe` directly when available,
preserving the configured runtime directory and port without a PowerShell
action that can briefly open Windows Terminal. Rerun `slowave setup` after
upgrading to replace older task actions. Custom Python installations without
`pythonw.exe` retain a PowerShell fallback and may show a console.

### Local retrieval models

The embedding model and the pinned multilingual applicability model run locally.
Missing applicability assets are downloaded from Hugging Face on first use;
cached installations reuse them without a network request. Only model assets
are downloaded: task and memory text stays on your computer. The initial
request can take longer while assets are downloaded and loaded.

For an offline installation, provision the model cache beforehand. Set
`HF_HUB_OFFLINE=1` to prohibit downloads. If applicability assets are missing
or unavailable, retrieval uses lexical/semantic fallback and returns an
`applicability_unavailable` warning; relevance can differ in that mode.

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

Setup installs compact lifecycle rules. Endpoint fields and validation rules come
from the connected MCP tool definitions. After upgrading,
rerun setup and refresh/restart the client to load current tool definitions.

| Action | Clients | Detail |
|---|---|---|
| MCP config | All | Patches each client's MCP config so `slowave_*` tools appear |
| Lifecycle instructions | Claude Code, Cline, Windsurf, OpenCode, Codex | Injects the mandatory Slowave block automatically |
| Lifecycle instructions | Claude Desktop, Cursor | Prints the block to paste — requires one manual step |
| HTTP daemon | All | Installs as launchd/systemd/Task Scheduler — auto-starts |
| Background worker | All | Installs as launchd/systemd/Task Scheduler — consolidates events |
| Daily backup | All | Installs as launchd/systemd/Task Scheduler — gzip snapshot of the database |

Options:

```
slowave setup --client [claude-code|claude-desktop|cline|cursor|opencode|windsurf|codex|all]
              --no-worker       # skip daemon, worker, and backup registration
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
| `~/.claude/CLAUDE.md` | Claude Code instructions | Prepends lifecycle block |
| `~/Library/Application Support/Claude/claude_desktop_config.json` | Claude Desktop MCP config | Adds `mcpServers.slowave` entry |
| `~/.cline/rules/slowave.md` | Cline instructions | Prepends lifecycle block |
| `~/.cline/data/settings/cline_mcp_settings.json` | Cline MCP config (CLI/TUI) | Adds `mcpServers.slowave` entry |
| `~/Library/Application Support/Code/User/globalStorage/.../cline_mcp_settings.json` | Cline MCP config (VS Code) | Adds `mcpServers.slowave` entry |
| `~/Library/Application Support/Cursor/User/globalStorage/.../cline_mcp_settings.json` | Cline MCP config (Cursor) | Adds `mcpServers.slowave` entry |
| `~/.cursor/mcp.json` | Cursor native MCP config | Adds `mcpServers.slowave` entry |
| `~/.codeium/windsurf/mcp_config.json` | Windsurf MCP config | Adds `mcpServers.slowave` entry |
| `~/.codeium/windsurf/memories/global_rules.md` | Windsurf global rules | Prepends lifecycle block |
| `~/.config/opencode/opencode.json` | OpenCode MCP + instructions config | Adds `mcp.slowave` and registers instructions file |
| `~/.config/opencode/slowave-instructions.md` | OpenCode lifecycle instructions | Creates Slowave-owned instruction file |
| `~/.codex/config.toml` | Codex MCP config | Adds `[mcp_servers.slowave]` |
| `~/.codex/AGENTS.md` | Codex instructions | Prepends lifecycle block |
| `~/Library/LaunchAgents/com.slowave.worker.plist` | Background worker | launchd plist, loads with `launchctl` |
| `~/Library/LaunchAgents/com.slowave.daemon.plist` | HTTP MCP daemon | launchd plist, auto-starts on load |
| `~/Library/LaunchAgents/com.slowave.backup.plist` | Daily backup | launchd plist with `StartCalendarInterval` |

### Linux

| File | Purpose | What Changes |
|---|---|---|
| `~/.claude.json` | Claude Code MCP config | Same as macOS |
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
| `~/.codex/config.toml` | Codex MCP config | Same as macOS |
| `~/.codex/AGENTS.md` | Codex instructions | Same as macOS |

### Windows

| File | Purpose | What Changes |
|---|---|---|
| `%USERPROFILE%\.claude.json` | Claude Code MCP config | Same as macOS |
| `%USERPROFILE%\.claude\CLAUDE.md` | Claude Code instructions | Same as macOS |
| `%APPDATA%\Claude\claude_desktop_config.json` | Claude Desktop MCP config | Adds `mcpServers.slowave` entry |
| `%USERPROFILE%\.cline\rules\slowave.md` | Cline instructions | Same as macOS |
| `%USERPROFILE%\.cline\data\settings\cline_mcp_settings.json` | Cline MCP config (CLI/TUI) | Same as macOS |
| `%APPDATA%\Code\User\globalStorage\.../cline_mcp_settings.json` | Cline MCP config | Same as macOS |
| Task Scheduler | Background worker | Registers account-scoped `SlowaveWorker-<SID>` task |
| Task Scheduler | HTTP MCP daemon | Registers account-scoped `SlowaveDaemon-<SID>` task |
| Task Scheduler | Daily backup | Registers account-scoped `SlowaveBackup-<SID>` task |
| `%USERPROFILE%\.config\opencode\opencode.json` | OpenCode MCP + instructions config | Same as macOS |
| `%USERPROFILE%\.config\opencode\slowave-instructions.md` | OpenCode lifecycle instructions | Same as macOS |
| `%USERPROFILE%\.cursor\mcp.json` | Cursor native MCP config | Same as macOS |
| `%USERPROFILE%\.codeium\windsurf\mcp_config.json` | Windsurf MCP config | Same as macOS |
| `%USERPROFILE%\.codeium\windsurf\memories\global_rules.md` | Windsurf global rules | Same as macOS |
| `%USERPROFILE%\.codex\config.toml` | Codex MCP config | Same as macOS |
| `%USERPROFILE%\.codex\AGENTS.md` | Codex instructions | Same as macOS |

---

## Services

### HTTP MCP daemon

Serves HTTP MCP tools on the assigned loopback port (8766 upward). Use `slowave serve status` for the effective URL. HTTP-capable clients connect to this daemon; stdio clients start their own MCP process and must also be restarted after upgrading.

| Platform | Service | Verify |
|---|---|---|
| macOS | `~/Library/LaunchAgents/com.slowave.daemon.plist` | `launchctl list \| grep slowave` |
| Linux | `~/.config/systemd/user/slowave-daemon.service` | `systemctl --user status slowave-daemon` |
| Windows | Task Scheduler: daemon | `Get-ScheduledTask | Where-Object TaskName -like 'SlowaveDaemon-*'` |

### Background worker

Runs consolidation offline — transforms raw events into searchable schemas.

| Platform | Service | Verify |
|---|---|---|
| macOS | `~/Library/LaunchAgents/com.slowave.worker.plist` | `launchctl list \| grep slowave` |
| Linux | `~/.config/systemd/user/slowave-worker.service` | `systemctl --user status slowave-worker` |
| Windows | Task Scheduler: worker | `Get-ScheduledTask | Where-Object TaskName -like 'SlowaveWorker-*'` |

### Daily backup

Gzip snapshot of the SQLite database. Keeps the last 7 backups in the effective
runtime root's `backups/` directory.

| Platform | Service | Verify |
|---|---|---|
| macOS | `~/Library/LaunchAgents/com.slowave.backup.plist` | `launchctl list com.slowave.backup` |
| Linux | `~/.config/systemd/user/slowave-backup.timer` | `systemctl --user status slowave-backup.timer` |
| Windows | Task Scheduler: backup | `Get-ScheduledTask | Where-Object TaskName -like 'SlowaveBackup-*'` |

### Runtime data location

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
matches your goal. Stop foreground dashboard/worker/server terminals with Ctrl+C.
Run the dry run first whenever possible. Remove integrations **before** removing
the package, so the cleanup command remains available. If removal reports an
error, resolve it and rerun; do not remove the package while jobs remain registered.

### Stop using Slowave but keep its memories

```bash
slowave uninstall --dry-run
slowave uninstall
```

`slowave uninstall` removes every Slowave-managed client integration: MCP
entries, generated lifecycle instructions, plus the HTTP daemon,
background worker, and daily-backup services. It preserves:

- the effective runtime root, including the SQLite database and archives
- setup-created `*.bak.*` configuration backups
- unrelated client configuration, MCP entries, and instruction content
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
# pip venv: use that environment's Python
python -m pip uninstall slowave
# Homebrew (macOS)
brew uninstall slowave
```

---

## Trust & Transparency

- ✅ **Open Source** — [github.com/mrsalty/slowave](https://github.com/mrsalty/slowave)
- ✅ **Repeatable setup** — preserves unrelated configuration; restarts managed services
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
lifecycle guidance; the connected MCP schemas and descriptions define endpoint
contracts.

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
