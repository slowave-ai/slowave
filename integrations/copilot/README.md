# Copilot CLI + Slowave — quick-ref

Full guide: **[../../docs/install.md](../../docs/install.md)**

For upgrades, service management, and removal, see the [installation guide](../../docs/install.md).
For recovery and OS-specific logs, see [troubleshooting](../../docs/troubleshooting.md#github-copilot-cli).

---

## Scope

This integration targets **GitHub Copilot CLI running locally**, using the
`copilot` command. Copilot IDE extensions, desktop app, cloud agent, cloud
sandboxes, and CLI running in hosted Actions jobs are outside this release’s
support scope. A shared configuration may be discovered by another Copilot
surface; that does not establish lifecycle support there.

Automated setup and diagnostic tests cover this integration. Configuration and
personal instruction discovery were checked with Copilot CLI **1.0.95 on macOS**.
A user also confirmed memory reads and writes through Copilot CLI on macOS.
Full model-driven lifecycle validation and native Windows/Linux client qualification
remain pending; see the verification steps below.

---

## Setup

### 1. Install and sign in to Copilot CLI

Install the current Copilot CLI using
[GitHub’s installation guide](https://docs.github.com/en/copilot/how-tos/copilot-cli/set-up-copilot-cli/install-copilot-cli).
For example, with Node.js 22 or newer:

```text
npm install -g @github/copilot
copilot --version
copilot
```

Follow Copilot’s sign-in prompt (`/login` when needed), then exit the session.
Copilot needs its own account/plan access; managed accounts may require an
administrator to enable CLI and MCP access. Slowave does not install, authenticate,
or change organization permissions for Copilot.

Use the standalone `copilot` command, rather than the older `gh copilot`
extension. Reopen the terminal if a new executable is not yet on PATH.

### 2. Install Slowave and configure clients

Python 3.11 or newer and pipx are required for this installation method. Run these
commands in the same OS account and environment used for Copilot:

```text
pipx install slowave
slowave setup --dry-run
slowave setup
slowave status --services
slowave doctor
```

**Normal `slowave setup` discovers Copilot automatically** when `copilot` is on
PATH or its configuration directory exists. A fresh CLI install is detected even
before its first launch. Review the preview, then accept setup’s confirmation.

To configure only Copilot’s client files instead:

```text
slowave setup --client copilot --dry-run
slowave setup --client copilot
```

Both commands use the shared service setup: daemon, worker and daily backup.
Other clients’ files are unchanged by `--client copilot`, but their shared daemon
briefly restarts. No API key or manual lifecycle paste is needed for Slowave.

If Slowave is already installed, upgrade it using the original installer, then
rerun setup. For pipx:

```text
pipx upgrade slowave
slowave setup
```

Copilot support requires a Slowave build containing this integration. If
`slowave setup --help` does not list `copilot`, the installed build is too old.
These changes must be published before a normal package upgrade can install them.
For development, use the current branch in a permanent dedicated virtual
environment and install with `python -m pip install -e .`; run that environment’s
`slowave` executable. Do not move/delete the environment after service setup.

### 3. Restart Copilot CLI

Exit any open CLI session and run `copilot` again so it loads the new tools and
instructions. Keep the same `COPILOT_HOME`, if set.

---

## What gets configured

| What | Where |
|---|---|
| HTTP MCP server | `~/.copilot/mcp-config.json` → `mcpServers.slowave` |
| Personal lifecycle instructions | `~/.copilot/copilot-instructions.md` |
| Shared daemon, worker, daily backup | launchd (macOS) / systemd user services (Linux) / Task Scheduler (Windows) |

On Windows, `~/.copilot` means `%USERPROFILE%\.copilot`. Setting `COPILOT_HOME`
replaces this directory for both files. Use the same setting for setup, doctor,
uninstall, and Copilot; setup does not edit your shell startup files. For example:

**macOS / Linux:**

```bash
export COPILOT_HOME="$HOME/copilot-personal"
slowave setup --client copilot
copilot
```

**Windows PowerShell:**

```powershell
$env:COPILOT_HOME = "$env:USERPROFILE\copilot-personal"
slowave setup --client copilot
copilot
```

Use an absolute directory, as in these examples, so changing repositories does
not change the configuration location. These examples set the variable in the
current shell only. A CLI-specific
`--config-dir` passed only to Copilot cannot be inferred by Slowave; use a
consistent `COPILOT_HOME` instead. Multiple independent Copilot directories must
be configured and removed separately with their respective environment settings.

Setup preserves unrelated MCP servers and personal text outside Slowave’s
versioned lifecycle markers. Existing files receive timestamped backups.
Setup manages Slowave’s transport/URL, including migration from stdio to HTTP;
it preserves extra options such as timeout, tool selection and disabled state.
Copilot also records disabled servers in `settings.json` (`disabledMcpServers`).
Slowave leaves that file intact; doctor reports when it disables `slowave`.
Run `copilot mcp enable slowave` to re-enable it deliberately.
`--force` does not override an intentional tool restriction. Configuration must
be valid JSON with object-shaped `mcpServers` and `slowave` entries.

The daemon URL uses the assigned per-user port. Setup prints the actual endpoint;
`slowave serve status` reports it later. Do not assume port 8766 on every machine.

---

## Lifecycle instructions

Setup injects **the same lifecycle block as the other Slowave integrations** into
Copilot’s personal instruction file. It applies across repositories, including
research tasks that do not edit files. No hooks or custom agents are installed.

**Full lifecycle documentation:**
[docs/install.md#lifecycle-instruction-block](../../docs/install.md#lifecycle-instruction-block).

Copilot can combine personal and repository instructions. Keep older Slowave
blocks consistent if a repository also contains them. Disabling custom instructions
or excluding Slowave’s tools prevents the automatic cycle. Check the files loaded
in the current session with `/instructions`.
[GitHub instruction reference](https://docs.github.com/en/copilot/how-tos/copilot-cli/customize-copilot/add-custom-instructions).

---

## Verify and use

Start Copilot inside the repository you intend to work on:

```text
copilot
```

1. Run `/mcp` and check that `slowave` is connected and all five `slowave_*`
   tools are available. Approve connection/tool requests when Copilot asks.
2. Run `/instructions` and confirm your personal `copilot-instructions.md` is
   loaded and enabled.
3. Start a normal coding or research task. Check that Slowave activates before
   work, receives feedback for the retrieval, and commits before the final answer.
   Remember and recall are conditional; every simple task need not call all five
   tools.

For a deliberate memory check, use a disposable repository and ask:

> Remember this project preference in Slowave: use the phrase “orchid-check” for
> the example acceptance label. Complete the task using the installed lifecycle.

Exit Copilot, start a fresh conversation in the **same repository**, and ask for
the example acceptance label. Inspect that the answer comes from a Slowave
retrieval rather than relying on the previous chat. The same scoped knowledge
can be used by another supported agent connected to this store. Remove the test
memory through the dashboard afterward if you no longer need it.

Check records with `slowave stats` or `slowave dashboard`. `slowave doctor` checks
the installed user configuration and lifecycle block; it cannot prove which
instructions/tools a running Copilot session actually selected, or that its model
followed the lifecycle. A healthy daemon alone is insufficient verification.

For daily use, keep working normally; the installed instructions guide memory
use. No manual CLI memory commands are required. Storage stays local, but retrieved
context is given to Copilot and can be sent to its model service.

---

## Manual MCP config (if `slowave setup` did not work)

Prefer fixing setup first: registering an MCP server alone does not install the
lifecycle instructions. A default-port example for `mcp-config.json` is:

```json
{
  "mcpServers": {
    "slowave": {
      "type": "http",
      "url": "http://127.0.0.1:8766/mcp",
      "tools": ["*"]
    }
  }
}
```

Merge only the `slowave` entry into existing configuration; preserve other servers.
Replace the URL with the actual endpoint from `slowave serve status`. Local Slowave
does not need GitHub tokens or authentication headers. Keep all five tools enabled,
and rerun `slowave setup --client copilot` to install the canonical instructions.
Restart Copilot afterward.

For Copilot-side server management, see
[GitHub’s MCP guide](https://docs.github.com/en/copilot/how-tos/copilot-cli/customize-copilot/add-mcp-servers).

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `copilot` is absent from Slowave’s `--client` choices | Upgrade/install the Slowave build containing this integration, using the same package environment |
| Copilot is not detected | Verify `copilot --version`; reopen the shell after installation, or launch Copilot once; check `COPILOT_HOME` |
| Malformed JSON or object-shape error | Repair `mcp-config.json` first; setup stops without overwriting it |
| Connection refused or wrong URL | Check `slowave status --services`, start registered services, and rerun setup to refresh the assigned endpoint |
| Tools missing despite healthy user config | Inspect `/mcp`; check disabled servers, tool selection, repository/plugin overrides and organization policy |
| Only some Slowave tools available | Set the entry’s `tools` to `["*"]` or list all five lifecycle tools; setup preserves restrictions |
| Doctor reports Slowave disabled in settings | Run `copilot mcp enable slowave`, then restart the session; rerunning setup preserves this preference |
| Tools available but never called | Check `/instructions`, remove `--no-custom-instructions`, enable the personal file and restart the session |
| Old behavior after upgrading | Rerun setup and restart Copilot to refresh instructions and MCP definitions |
| Memory differs across agents | Use the same OS account, runtime/store settings and stable project scope; different machines/accounts have separate stores |
| WSL/container/SSH cannot reach Slowave | Run Copilot and Slowave in the same environment; loopback refers to that environment, not automatically your host |
| Service/model startup fails | See [service startup](../../docs/troubleshooting.md#services-will-not-start); first embedding-model use needs download access |
| Outstanding feedback blocks commit | Follow the tool’s returned rejected/outstanding targets, correct feedback and retry; reinstalling is unnecessary |

Repository/plugin definitions named `slowave` may override the user entry. Setup
does not rewrite these files or relax Copilot trust, sandbox or organization
policies. Diagnostic OK means the user files match this installation, not that
all higher-priority configuration is known to Slowave.

---

## Removal

Use the standard [removal workflow](../../docs/install.md#remove-slowave):

```text
slowave uninstall --dry-run
slowave uninstall
```

Uninstall removes **all managed client integrations and shared services**, keeping
memories and the package. For Copilot it removes `mcpServers.slowave` and the
marker-bounded lifecycle block, preserving other servers and personal text.
An instruction file containing only Slowave’s block is deleted. Use the same
`COPILOT_HOME` as setup. Restart Copilot to discard cached tools. Uninstall does
not remove manually added repository/plugin entries; remove those separately.
