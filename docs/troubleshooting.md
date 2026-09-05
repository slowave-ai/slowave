# Troubleshooting

If something isn't working, start with `slowave doctor` — it checks every
component, detects stale lifecycle blocks, and points you at specific issues:

```bash
slowave doctor
```

```bash
slowave doctor --verbose
```

The sections below cover common failure modes for each component.

---

## Daemon (HTTP MCP server)

The daemon is a long-running process that serves MCP tools to clients over HTTP
or SSE. It runs as a user service.

### Daemon won't start

**Port already in use.** The daemon binds to `127.0.0.1:8766` by default.
If something else is on that port:

```bash
lsof -i :8766
```

```bash
kill <PID>
```

```bash
slowave serve start
```

**Stale PID file.** If the daemon was killed ungracefully, the `daemon.pid`
file beneath the effective runtime root may prevent it from restarting.
`slowave serve start` detects and cleans stale entries automatically. Use
`slowave serve status` to print the exact PID-file path before removing it
manually.

```bash
slowave serve status
```

```bash
slowave serve start
```

**Slow Python import.** On Windows the health check waits up to 45 seconds for
imports to finish. If it still fails, inspect `logs/daemon.err` beneath the
runtime root shown by `slowave doctor`.

**Missing dependencies.** Run `slowave doctor` — it validates that all required
packages are installed and the embedding model can load.

### Daemon running but MCP tools not reachable

Check that the daemon is listening:

```bash
slowave serve status
```

```bash
curl http://127.0.0.1:8766/health
```

A `200 OK` means the daemon is alive. If the health endpoint hangs, the engine
may be warming up (models load lazily on first tool call).

### Daemon process is a zombie

If the daemon process exists but doesn't respond, force-kill and restart:

```bash
pkill -f 'slowave serve'
```

```bash
pkill -f 'slowave.mcp.http_server'
```

```bash
slowave serve start
```

---

## Background Worker

The worker consolidates raw events into episodic memories on a 5-minute
interval. It runs as a user service alongside the daemon.

### Worker is not consolidating

Check whether the worker process is running:

```bash
slowave status | grep worker
```

If it's not detected, check the supervisor:

```bash
launchctl list | grep slowave
```

```bash
systemctl --user status slowave-worker
```

```bash
Get-ScheduledTask -TaskName SlowaveWorker
```

Check the worker log for errors:

```bash
cat /tmp/slowave-worker.err
```

### Manual test

Run a single consolidation pass to verify the worker logic works:

```bash
slowave worker --once
```

### Recent runs

The dashboard shows worker run history, or from the CLI:

```bash
slowave status --verbose | grep worker
```

### Worker conflicting with daemon

Worker and daemon share the same SQLite database. WAL mode handles concurrent
access. If you see `database is locked` errors in the logs, an orphaned worker
from a prior session may be holding stale WAL state. Restart both:

```bash
slowave serve restart
```

```bash
pkill -f 'slowave worker'
```

---

## Dashboard

The dashboard is a local web UI on `127.0.0.1:8765`.

### Dashboard won't start

**Port conflict.** Port 8765 is the default. If something else is on it:

```bash
lsof -i :8765
```

```bash
pkill -f 'slowave dashboard'
```

```bash
slowave dashboard
```

**Missing static assets.** If installed from source without building the
frontend, the dashboard serves a broken page. Verify the assets exist:

```bash
ls slowave/dashboard/static/index.html
```

If missing, build them:

```bash
cd slowave/dashboard/ui
```

```bash
npm install
```

```bash
npm run build
```

### Dashboard loads but is blank

Open the browser's developer console. API fetch errors indicate the dashboard's
DB connection may be failing. Check the database path:

```bash
slowave doctor
```

```bash
slowave status
```

If the database file doesn't exist, start using Slowave with an MCP client
first — the dashboard reads from the same database.

### Dashboard shows stale data

The dashboard queries the database directly on every request. If data appears
stale, the consolidation worker may not have run recently, or the daemon has
not written events yet.

### Forget/Unforget buttons not working

Start the dashboard with the actions flag:

```bash
slowave dashboard --allow-actions
```

---

## Client Integration (MCP tools)

If MCP tools (`slowave_activate`, `slowave_remember`, etc.) do not appear in
your agent, the client configuration is likely wrong or missing.

### MCP tools not appearing

Run the diagnostic:

```bash
slowave doctor
```

This checks every supported client's config file and reports which ones are
correctly configured, misconfigured, or missing.

Common issues per client:

| Client | Most common issue |
|---|---|
| Claude Code | Config file at `~/.claude.json` has wrong `mcpServers` key. Re-run `slowave setup` |
| Claude Desktop | Requires one manual paste of the lifecycle block. Check `claude_desktop_config.json` |
| Cursor | Its MCP config at `~/.cursor/mcp.json` uses a different key format. Re-run `slowave setup --client cursor` |
| OpenCode | Uses the `mcp` key, not `mcpServers`. Check `~/.config/opencode/opencode.json` |

### Tools appear but return errors

**Lifecycle version mismatch.** The lifecycle contract (`activate → remember →
recall → feedback → commit`) evolves between releases. `slowave doctor` reports
if the version in `CLAUDE.md` or equivalent instruction files doesn't match the
installed version. Fix by re-running setup:

```bash
slowave setup
```

**Feedback completeness enforcement.** `slowave_commit` fails if feedback was
not provided for every retrieved memory or procedure. This is intentional. The
error response lists the outstanding targets.

### Hooks not firing

Claude Code and Codex use `UserPromptSubmit` and `Stop` hooks to call Slowave
on every turn. If they aren't firing:

```bash
slowave setup --client claude-code
```

```bash
slowave setup --client codex
```

Check that `~/.claude/settings.json` contains the hook configuration.

### Scope fragmentation

If memory is split across two similar scopes (e.g., `project:my-repo` and
`project:my_repo`), the system creates separate memory silos. A cold-start
warning is logged when a new scope is detected. Use consistent scope names
across sessions.

---

## Database

Slowave uses a local SQLite database beneath the OS user's native application-
data directory. `slowave doctor` prints the exact path. `SLOWAVE_HOME` moves
the complete runtime tree; legacy `SLOWAVE_DB` selects an exact database path.
Do not set both.

### Runtime root or migration issues

Use these commands to inspect the effective paths and safely plan a legacy
data migration:

```bash
slowave doctor
slowave migrate-data --dry-run
```

`SLOWAVE_HOME` and `SLOWAVE_DB` cannot be set together. Unset one of them, then
use `SLOWAVE_HOME` for a complete relocated runtime tree or `SLOWAVE_DB` only
when an integration needs an exact legacy database path. Migration refuses a
non-empty destination and leaves `~/.slowave` intact for rollback.

### Database integrity

Run the built-in health check:

```bash
slowave status --verbose
```

Or use the dashboard at `/diagnostics`.

The dashboard exposes `PRAGMA integrity_check` and `PRAGMA quick_check` results
under the Database Health section.

### Schema errors on startup

Slowave applies schema migrations automatically (pre-migrations → DDL →
post-migrations). If migration fails, the database may be left in an
inconsistent state:

1. Run `slowave doctor` to see which step failed
2. Restore from your latest backup (see Backup/Restore below)
3. If no backup exists, file an issue with the error message

### Slow performance

- The `WAL` journal mode keeps write performance consistent. The `-wal` and
  `-shm` sidecar files are normal and auto-checkpointed.
- Large databases (thousands of sessions) may slow down schema listing. The
  dashboard paginates results.
- The auto-rebuild on logic version bump can take minutes. This is a one-time
  cost.

### Auto-rebuild is slow

When `current_logic_version` changes, the engine replays all raw events to
rebuild derived state. This is normal after an upgrade. Progress is logged at
the INFO level. If the rebuild appears stuck:

1. Check `log_versions` and `replay_checkpoints` table — one process holds a
   `claimed_ts` lock.
2. If the lock is stale (older than 180 seconds), wait — the next retry will
   claim it.
3. After 5 failed claim attempts the rebuild stops. Restart the daemon to
   retry.

### Database file locked

SQLite in WAL mode supports concurrent reads and one writer. If you see
`database is locked` errors, an orphaned process may hold the write lock.
Kill all slowave processes and retry:

```bash
pkill -f slowave
```

```bash
slowave serve start
```

---

## Backup & Restore

### Backup fails

Backup is a daily scheduled task. If it fails:

Check the backup log:

```bash
cat /tmp/slowave-backup.err
```

Run manually:

```bash
slowave backup
```

### Restore doesn't work

`slowave restore` stops the daemon and worker, swaps the database file, and
deletes stale WAL sidecars. If the restored database triggers an auto-rebuild
(version mismatch), the engine may be temporarily unavailable. This is normal.

Create a fresh backup first:

```bash
slowave backup
```

Then restore:

```bash
slowave restore /path/to/backup.sqlite.gz
```

---

## General Diagnostics

`slowave doctor` runs all checks in sequence:

| Check | What it validates |
|---|---|
| Python version | >= 3.11 |
| Package version | Installed vs latest |
| Database | File exists, accessible, integrity |
| Daemon | PID file, process running, health endpoint |
| Worker | Process detected |
| Client configs | MCP config, instructions, hooks per client |
| Lifecycle version | Installed vs instruction files |
| Feedback health | Feedback events per retrieval ratio |
| Embedding model | Model loads and encodes |

If none of the above helps, run the verbose diagnostic and include its output
when filing an issue:

```bash
slowave doctor --verbose 2>&1 | tee /tmp/slowave-doctor.log
```
