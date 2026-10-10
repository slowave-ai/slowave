# Troubleshooting Slowave

For installation, service commands, upgrades, and removal, start with the
[installation guide](install.md). Run `slowave docs troubleshooting` to open this
page from your terminal.

## Start here

```text
slowave --version
slowave status --services
slowave doctor --verbose
```

`status --services` reads the OS supervisor and daemon health without opening the
database. It reports installed/live versions and effective runtime/log paths.
`doctor` performs deeper checks and may initialize database/model state. Run
commands as the same user, with the same Python environment and runtime overrides
used by setup. If one command fails, retain its error and continue with the
OS-specific checks below; do not treat partial output as a healthy installation.

| Symptom | Next step |
|---|---|
| `slowave` command not found | [PATH and Python environment](#command-not-found-or-wrong-python-environment) |
| Old version after upgrading | [Upgrade checks](#old-version-after-an-upgrade) |
| Missing, stopped, or repeatedly failing jobs | [Service startup](#services-will-not-start) and [logs](#logs-by-os) |
| MCP tools missing or stale | [Client integration](#client-integration) |
| Dashboard old, blank, or occupied port | [Dashboard](#dashboard) |
| Database locked, migration failure, or corruption | [Database and recovery](#database-and-recovery) |
| Uninstall/purge failed | [Removal](#removal-problems) |

## Command not found or wrong Python environment

For pipx, run `pipx list`, then `pipx ensurepath` and reopen the terminal. For a
virtual environment, use its full `slowave` executable path from the
[installation examples](install.md#alternative-pip-in-a-dedicated-virtual-environment).
Do not install a second copy to work around PATH problems.

Inspect the executable selected by your shell:

**macOS / Linux:**

```bash
command -v slowave
```

**Windows PowerShell:**

```powershell
Get-Command slowave | Select-Object Source
```

Compare it with the service's registered executable/action (see below). Pipx,
Homebrew, and pip environments can coexist, so upgrading one does not update
another. Run `setup` from the intended installation to rebind registrations.
Do not move a virtual environment after setup.

## Old version after an upgrade

Run `slowave setup`, then `slowave status --services`. Normal setup reapplies
services without requiring `--force`. To restart without changing client
configuration, run `slowave restart`.

Compare `installed_version` with `daemon_health.version` and check
`daemon_database_matches`. If either mismatches, inspect
the registered executable and environment. Check that the package upgrade used
the same installation and `SLOWAVE_HOME`/`SLOWAVE_DB` settings. Version equality
alone cannot distinguish two development builds bearing the same version;
restart after reinstalling either build.

Stop foreground dashboard/worker/server terminals with Ctrl+C and relaunch them.
Reload MCP clients too: stdio servers and cached tool definitions belong to those
clients and are not replaced by restarting the HTTP daemon. See the complete
[upgrade workflow](install.md#upgrade), including older releases without `stop`.

## Services will not start

Use `slowave setup` if registrations are missing or their executable path no
longer exists. Use `slowave start` when registered services are simply stopped.
Do not use foreground `serve start` to compete with an installed daemon.

`setup`, `start`, and `restart` allow up to two minutes for the daemon health
check on every OS, returning immediately when its version and database match
the installation. Cold imports and antivirus scanning after an upgrade can
delay startup. Setup applies daemon, worker, and backup registrations before
checking daemon readiness, so a health timeout does not skip the worker or
backup registration. Registration errors themselves still stop setup.

After a health timeout, inspect `slowave status --services` and the daemon logs
in its reported `logs_dir`. The daemon may still be starting; a timeout does
not stop it. If the registrations exist and the daemon remains unhealthy,
resolve the logged error and retry `slowave start` without repeating setup.

### macOS

Run in the logged-in desktop account. These commands inspect jobs and their
registered program/environment:

```bash
launchctl print gui/$(id -u)/com.slowave.daemon
launchctl print gui/$(id -u)/com.slowave.worker
launchctl print gui/$(id -u)/com.slowave.backup
```

A loaded job is not necessarily a running process. Check its state, PID, and last
exit status. Backup jobs normally sit idle between scheduled runs. A missing GUI
user domain in a headless session requires [manual management](install.md#manual-service-management).
Use `setup` after editing service definitions; simply signaling an old process
does not reload its registration.

### Linux

```bash
systemctl --user status slowave-daemon.service slowave-worker.service slowave-backup.timer
systemctl --user cat slowave-daemon.service slowave-worker.service
systemctl --user list-timers slowave-backup.timer
```

If the user bus/systemd user manager is unavailable, check your login/session
configuration. Containers and non-systemd systems need
[manual management](install.md#manual-service-management). WSL needs systemd
support enabled and a working user manager. Do not replace `--user` with sudo:
that controls a different service scope.

Services normally follow your user session. Running them after logout requires
an explicit OS policy choice, such as user lingering; setup does not enable it.
See [systemd's loginctl documentation](https://www.freedesktop.org/software/systemd/man/252/loginctl.html).

### Windows PowerShell

```powershell
Get-ScheduledTask | Where-Object TaskName -like 'Slowave*' | Select-Object TaskName,TaskPath,State
Get-ScheduledTask | Where-Object TaskName -like 'Slowave*' | Select-Object TaskName,Actions
Get-ScheduledTaskInfo -TaskName (Get-ScheduledTask | Where-Object TaskName -like 'SlowaveDaemon-*' | Select-Object -First 1 -ExpandProperty TaskName)
Get-ScheduledTaskInfo -TaskName (Get-ScheduledTask | Where-Object TaskName -like 'SlowaveWorker-*' | Select-Object -First 1 -ExpandProperty TaskName)
Get-ScheduledTaskInfo -TaskName (Get-ScheduledTask | Where-Object TaskName -like 'SlowaveBackup-*' | Select-Object -First 1 -ExpandProperty TaskName)
```

Inspect Task Scheduler's History tab and last-run results for immediate exits.
The jobs run as your logged-in user. `stop` disables tasks to prevent recovery
triggers from relaunching them; `start`/`setup` re-enable them. A backup task
normally reports Ready between runs. The daemon and worker should remain Running.
Their five-minute recovery trigger is not the worker's consolidation interval.

Each account's tasks have a SID suffix. Setup and uninstall manage only the
current account's tasks, plus that account's older root-level tasks during
migration/removal. A root-level task owned by another account is left alone.

Task registration can be blocked by organizational policy. An access-denied error
requires correcting that policy/account issue or using manual management; do
not switch to a different administrator account and create a second installation.

### Port conflict or stale PID

Use `slowave serve status` for the daemon URL and PID-file path. Slowave assigns
loopback ports per user (daemon from 8766, dashboard from 8765); do not assume the
default port when investigating a conflict.

For an explicit port, inspect its owner before stopping anything:

**macOS / Linux:** `lsof -nP -iTCP:8766 -sTCP:LISTEN` (or Linux `ss -ltnp`).

**Windows PowerShell:**

```powershell
Get-NetTCPConnection -LocalPort 8766 -State Listen | Select-Object LocalPort,OwningProcess
```

If it is a registered Slowave job, use `slowave stop`, then `slowave start`. If it
is a manual foreground Slowave process, stop its own terminal. Resolve a genuine
third-party port conflict by choosing another explicit port and rerunning setup
with `SLOWAVE_MCP_HTTP_PORT` set. A stale PID file is normally cleaned on daemon
startup; remove it manually only after confirming that PID is no longer a live
Slowave process. Never delete the database or WAL files to repair a PID problem.

## Logs by OS

Use the `logs_dir` printed by `slowave status --services`. Runtime overrides can
change these paths; old `/tmp/slowave-*.err` examples do not apply.

| OS | Logs / diagnostics |
|---|---|
| macOS | Under `logs_dir`: `daemon.log`, `daemon.err`, `worker.log`, `worker.err`, `backup.log`, `backup.err` |
| Linux | `journalctl --user -u slowave-daemon -u slowave-worker -u slowave-backup --since today` |
| Windows | Under `logs_dir`, normally `pythonw-serve.log`, `pythonw-worker.log`, `pythonw-backup.log`; also Task Scheduler history |

To read a file, use `tail -n 100 "/actual/log/path"` on macOS/Linux or
`Get-Content "C:\actual\log\path" -Tail 100` in PowerShell. A custom Windows
installation without `pythonw.exe` may need foreground execution to expose
startup output. Files need not exist before the job's first launch.

## Worker and backups

Check `status --services` and worker logs. Worker process detection in `doctor`
is best-effort; verify supervisor state and the dashboard's worker run history
before concluding consolidation is healthy. Stop the managed worker before an
advanced one-off test (`slowave worker --once`), then restart services when done.

For backup failures, run `slowave backup` and inspect its error. The database may
not exist before the first memory operation. Backups use SQLite's consistent
online snapshot API, so routine backups can run with services active. The
command prints its destination; by default the runtime `backups/` directory
retains seven snapshots. `SLOWAVE_BACKUP_DIR`/`SLOWAVE_BACKUP_KEEP` can override
manual backup settings. Custom scheduler environments may differ from your shell.

## Dashboard

`slowave dashboard` prints its actual URL. It runs in the foreground. If a
previous copy owns the port, stop that copy with Ctrl+C and relaunch; a browser
refresh alone does not load new Python code. Do not kill processes solely by
matching the word `slowave`.

For a blank page, inspect the browser console and the dashboard terminal output,
then verify the database/runtime path with `slowave doctor`. Published packages
should include frontend assets. If a published wheel lacks them, report the
packaging issue. For a source checkout, use `npm ci` and `npm run build` in
`slowave/dashboard/ui`, then relaunch the dashboard.

The dashboard enables its mutating actions by default. Use
`slowave dashboard --no-allow-actions` for read-only inspection; use
`--allow-actions` to explicitly enable actions. Stale memory data may reflect missing client
writes or worker consolidation, rather than a browser problem.

## Client integration

Run `slowave doctor` and `slowave setup --client CLIENT`. Open the corresponding
[client guide](../integrations/README.md) for transport/configuration details.
Setup skips clients it does not detect; launch/install the client first if its
configuration directory has not been created. Setup prints required manual
instruction steps for Cursor and Claude Desktop.

After upgrades, reload/restart clients to refresh cached schemas/instructions
and client-owned stdio processes. HTTP daemon health alone does not verify a
client's tool connection. Lifecycle validation errors, including outstanding
feedback targets, should be handled using the connected MCP tool schema and
error response; reinstalling does not fix missing task feedback.

## Database and recovery

Use `slowave doctor` for effective paths and reported database checks; the
dashboard's Database Health view also exposes SQLite checks. Keep `SLOWAVE_HOME`
for a complete runtime root, or legacy `SLOWAVE_DB` for an exact database path;
do not set both. Plan legacy migration with `slowave migrate-data --dry-run`.
Migration preserves the source and refuses a non-empty destination.

For database locks, stop foreground processes and run `slowave stop`. If stopping
fails, resolve it before replacing/deleting any database files. Keep the `-wal`
and `-shm` sidecars intact: committed data can still be in WAL. Avoid blanket
`pkill` or deletion of SQLite sidecars as a repair step.

For a failed migration or confirmed corruption, preserve the current files and
use a known good backup. Restore deliberately replaces memories:

```text
slowave backup
slowave stop
slowave restore /actual/path/slowave-TIMESTAMP.db.gz
slowave start
slowave doctor
```

Use your real `.db.gz` filename, quoted if its path contains spaces. On Windows,
use the Windows path to that file. If the current database is unreadable and
backup fails, preserve its files separately before restoring. Restore refuses to
overwrite a database it cannot snapshot: after preserving the original database
and its sidecars, move those original files aside and retry the restore. Do not restart
until restore succeeds. Restore does not refresh package versions or client
instructions. An older package may not read a database migrated by a newer one.

Engine initialization after an upgrade can replay raw events to rebuild derived
state; inspect logs before interrupting it. Report persistent migration/rebuild
errors with their exact text rather than repeatedly deleting runtime data.

## Removal problems

Use `slowave uninstall --dry-run` before removal. Uninstall removes integrations
and service registrations; it retains memories and the package. `purge` also
removes local data and setup configuration backups, but retains database archives
in `backups/`. Review [removal choices](install.md#remove-slowave) before deleting.
Stop foreground processes first. If a service cannot be stopped or removed,
resolve that error before removing the package or purging data. Package removal
comes last, using the original installer.

## Report an issue

Include OS/version, installation method, the failing command and exact error,
`slowave --version`, `slowave status --services`, and relevant log excerpts.
Include `slowave doctor --verbose` when it can complete. Inspect output before
sharing: supervisor details and logs may contain private paths or environment
values. Report issues at [GitHub](https://github.com/mrsalty/slowave/issues).
