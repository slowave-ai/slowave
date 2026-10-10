# Install, upgrade, run, and remove Slowave

Use your package manager to install or upgrade the Python package. Use Slowave
commands to configure clients and manage the installed user services. Run these
commands as your normal OS user, in the same Python environment used to install
Slowave. Do not use sudo for user services.

## Command reference

| Command | Purpose |
|---|---|
| `slowave setup` | Configure detected clients, install/reapply daemon, worker and backup schedule, restart services, and verify the daemon's installed version |
| `slowave setup --dry-run` | Preview configuration and service changes |
| `slowave start` | Start installed daemon and worker; enable the backup schedule |
| `slowave stop` | Stop installed services and suspend the backup schedule |
| `slowave restart` | Stop and start installed services through their OS supervisor; verify the daemon version |
| `slowave status --services` | Show supervisor state, installed/live daemon versions, and runtime location without opening the database |
| `slowave docs [TOPIC]` | Open a guide in your browser; `--no-open` prints its link |
| `slowave doctor` | Diagnose client configuration and runtime health |
| `slowave dashboard` | Run the dashboard in the foreground; Ctrl+C stops it |
| `slowave uninstall` | Remove integrations and service registrations, preserving memories and the Python package |
| `slowave purge` | Remove local data too; see `slowave purge --help` before use |

`setup --force` remains available to reapply client configuration. It is no
longer necessary to load upgraded service code. Every setup run with services
enabled reapplies and restarts them, including when the client configuration
is unchanged. Expect a brief MCP interruption. `setup --no-worker` skips daemon,
worker, and backup services, for advanced manually managed deployments.

`start` and `restart` require installed service registrations; use `setup` first.
The daily backup is scheduled, not run immediately by `start` or `restart`.
Use `slowave backup` for an immediate snapshot.

Before `slowave restore`, run `slowave stop` and close foreground workers and
dashboards. Restore validates the compressed backup before replacing the database
and keeps a consistent snapshot of the previous database as `slowave.db.bak`.
Run `slowave start` afterward. Restore refuses detected active processes rather
than killing them while their supervisor can restart them.

## First installation

```bash
pipx install slowave
slowave setup --dry-run
slowave setup
slowave status --services
slowave doctor
```

Alternatively, install in a dedicated virtual environment using
`python -m pip install slowave`, then run the same setup commands from that
environment. Use one installation method consistently: upgrading a different
Python environment does not update the package used by registered services.

## Upgrade

Stop the foreground dashboard with Ctrl+C, if running. Then:

```bash
slowave backup
slowave stop
pipx upgrade slowave
slowave setup
slowave status --services
slowave doctor
```

For a pip installation, replace `pipx upgrade slowave` with
`python -m pip install --upgrade slowave` using the original environment.
For Homebrew, use `brew upgrade slowave`.

Stopping first prevents old processes from importing partially replaced code
and avoids Windows locks on loaded package files. `setup` updates client
instructions and service definitions and launches the installed version.
Afterward, restart or refresh your MCP clients to reload cached tool definitions.
Run `slowave dashboard` again if wanted, and refresh its browser tab.

If you already upgraded while services were running, run `slowave setup` and
verify the reported installed/live versions. A browser refresh alone does not
restart the dashboard's Python process.

If the package upgrade fails, services remain stopped. Repair the package in
the original environment, then rerun setup. To return to a known version with
pip, install `slowave==VERSION`. Do not assume an older release can read a
newer database; preserve a pre-upgrade backup and consult release notes before
restoring or downgrading data.

## Remove

Stop any foreground dashboard, then remove integrations **before** removing
the executable:

```bash
slowave uninstall --dry-run
slowave uninstall
pipx uninstall slowave
```

For pip use `python -m pip uninstall slowave`; for Homebrew use
`brew uninstall slowave`. Memories remain in the runtime directory. To delete
memories too, review `slowave purge --help` and run purge while the package is
still installed. Remove manually pasted instructions from Cursor and Claude
Desktop as prompted by uninstall.

## How services work

| OS | Supervisor | Start/stop behavior |
|---|---|---|
| macOS | Per-user launchd agents | Start bootstraps saved plists; stop boots out jobs so KeepAlive cannot immediately relaunch them |
| Linux | systemd user units | Start/stop daemon and worker; also start/stop backup timer and stop any dispatched backup |
| Windows | Task Scheduler | Stop disables triggers and stops tasks; start enables tasks and launches daemon/worker |

On macOS and Linux, a subsequent login may start enabled services again.
Windows tasks stay disabled until `start` or `setup`. Use `uninstall` to remove
registrations permanently. Windows Task Scheduler stopping can terminate an
in-progress task; avoid stopping during an important write or backup. Keep a
backup before upgrading.

The dashboard and advanced foreground `worker` / `serve start` processes are
owned by the terminal or custom supervisor that launched them. Stop and restart
those separately. Service commands do not kill arbitrary processes by name.
`serve start` is an advanced foreground server command, not the normal start
command. Legacy `serve stop` and `serve restart` now control only the registered
daemon through the OS supervisor; use top-level commands for all services.

## Troubleshooting and support

Start with:

```bash
slowave status --services
slowave doctor --verbose
```

Compare `installed_version` with `daemon_health.version`. Inspect the supervisor
state for the worker and backup schedule; daemon health alone does not verify
those jobs. `slowave --version` reports only the CLI's installed version.

See [Troubleshooting](troubleshooting.md) for logs, failed startup, ports, database
issues, and client configuration. See [installation details](install.md) for
runtime paths and files setup changes. Include status and doctor output, OS,
installation method, and upgrade command when reporting an issue; redact private
paths or data before sharing.

Service-manager references: [systemd control semantics](https://github.com/systemd/systemd/blob/main/man/systemctl.xml)
and [Windows task stopping](https://learn.microsoft.com/en-us/powershell/module/scheduledtasks/stop-scheduledtask).
