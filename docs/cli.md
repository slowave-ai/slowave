# Slowave CLI

The Slowave CLI is for local setup, inspection, maintenance, backups, and
manual experiments. Its JSON output makes it useful in scripts and CI:

```bash
slowave --json status
```

Slowave stores data in a local SQLite database. Set `SLOWAVE_DB` to use a
different database; otherwise it uses `~/.slowave/slowave.db`.

> [!IMPORTANT]
> The database is plaintext by default. Store it on an encrypted volume or
> protect it with operating-system permissions if its contents are sensitive.

## MCP lifecycle versus CLI workflow

Agents integrated through MCP use the current five-verb lifecycle:
`activate → remember → recall → feedback → commit`. See
[Architecture](architecture.md#the-public-cognitive-cycle) for that contract.

The CLI has older, manual lifecycle commands that use
`activate → remember → recall → reinforce → commit`. In particular,
`slowave reinforce` is a CLI compatibility and experimentation command; it is
**not** an MCP tool and does not replace MCP `feedback`. Do not use CLI
examples below as an agent integration contract.

## Manual lifecycle commands

These commands are helpful when testing a local database or building a
scripted workflow. Prefer an MCP client for normal agent use.

### Activate

Open a task session and retrieve relevant context:

```bash
slowave --json activate \
  --query "fix the session reaper race condition" \
  --scope "project:my-repo" \
  --initial-goal "Fix the session reaper race" \
  --mode strict_scope
```

The response includes a `session_id` and `retrieval_id`. `--query` is
required. Use `--scope` to isolate memory; `--task-type`, `--situation`,
repeatable `--requirement`, `--topic`, and `--entity` flags add retrieval
cues. `--mode` accepts `default`, `strict_scope`, `broad`, or `debug`;
`strict_scope` is the default. `--limit` defaults to 8.

### Remember and recall

```bash
slowave --json remember "SQLite is preferred for small local deployments." \
  --type decision --scope "project:my-repo" --session sess_abc123

slowave --json recall "database choice" \
  --scope "project:my-repo" --top-k 5 --evidence
```

`remember` records a typed durable claim. Its `--session` flag is optional.
`recall` is a semantic lookup; `--scope` is strongly recommended, and
`--evidence` includes raw-event citations. Its default `--top-k` is 20.

### Reinforce and commit

```bash
slowave --json reinforce ctx_abc123 \
  --feedback useful --outcome success --used sch_5 --irrelevant sch_7

slowave --json commit sess_abc123 \
  --outcome success \
  --final-goal "Fix the session reaper race" \
  --outcome-summary "Added the lock and verified concurrent cleanup."
```

`reinforce` records feedback for a CLI `activate` or `recall` retrieval.
It accepts `useful`, `partially_useful`, `irrelevant`, `stale`, `wrong`,
`missing`, and `too_much_context`, plus repeatable `--used`,
`--irrelevant`, `--stale`, and `--wrong` schema references.

`commit` closes the session. Supply an accurate `--outcome`
(`success`, `partial`, `failure`, or `unknown`) and, when available,
`--final-goal`, `--outcome-summary`, `--step`, and `--procedure-json`.

## Lower-level session controls

For raw event ingestion, use `session` and `event` directly:

```bash
SID=$(slowave --json session start --scope "project:my-repo" \
  | python3 -c 'import json,sys; print(json.load(sys.stdin)["session_id"])')

slowave event --session "$SID" --type user_message \
  --content "I prefer SQLite for MVPs."
slowave session end "$SID"
```

`session end` is a lower-level close operation; use `commit` for the
manual lifecycle above. `context` produces a working-memory brief and may
open an implicit session when given a scope. It also belongs to the legacy CLI
workflow and returns a retrieval ID for `reinforce`.

## Operations and maintenance

| Command | Purpose |
|---|---|
| `slowave status` | Check database, memory health, and local process status. |
| `slowave stats [--scope S] [--verbose] [--graph]` | Inspect storage and graph statistics. |
| `slowave schema [--needs-review]` | List schemas. |
| `slowave show sch_N\|epi_N\|evt_N` | Inspect one schema, episode, or raw event. |
| `slowave dashboard` | Start the local dashboard (default `127.0.0.1:8765`). Use `--no-open` for a headless host. |
| `slowave forget sch_N` / `unforget sch_N` | Suppress or restore a reviewed schema. These are intentionally human-only; MCP has no equivalent. |
| `slowave consolidate` | Run one replay and latent-consolidation pass. |
| `slowave worker --once` | Run one consolidation pass; omit `--once` for the background loop. |
| `slowave dedup-schemas` | Preview exact duplicate schemas; add `--apply` to merge them. |
| `slowave rebuild --force` | Force a rebuild of derived state from raw events. Support/debugging only; normal version upgrades rebuild automatically. |

Use `slowave dashboard --no-allow-actions` when the dashboard must be
strictly read-only. The dashboard's Forget and Unforget buttons are enabled by
default.

## Setup and diagnostics

```bash
slowave setup --client codex
slowave doctor --verbose
slowave serve status
```

`setup` configures detected clients, lifecycle instructions, enforcement
hooks where supported, and daemon/worker services. Choose one client with
`--client claude-code|claude-desktop|cline|cursor|windsurf|opencode|codex|all`.
Use `--dry-run` before changing configuration, or `--no-worker` to skip
worker-service installation. `doctor` verifies the local environment.

`serve start|stop|restart|status` manages the HTTP MCP daemon. By default it
listens at `http://127.0.0.1:8766/mcp`.

## Backups, removal, and recovery

```bash
slowave backup --dir ~/.slowave/backups --keep 14
slowave restore ~/.slowave/backups/slowave-YYYYMMDD_HHMMSS.db.gz
```

`backup` uses SQLite's online backup API and is safe while Slowave services
run. It writes gzip-compressed database snapshots, retaining seven by default.
`restore` stops the worker, replaces the database, preserves the previous
database as `slowave.db.bak`, then restarts the worker. Review the target
backup carefully; add `--yes` only in an unattended script.

Slowave and its package manager have separate responsibilities. Slowave removes
the client integrations and local state it owns; the package manager removes the
installed executable and its dependencies. Two Slowave removal commands have
deliberately different scopes:

| Command | What it removes |
|---|---|
| `slowave uninstall [--dry-run]` | Slowave MCP entries, generated lifecycle instructions, hooks, and daemon, worker, and backup services. It preserves `~/.slowave`, database archives, setup backups, and the installed package. |
| `slowave purge [--dry-run]` | Everything removed by `uninstall`, plus local data in `~/.slowave` and setup-created `*.bak.*` configuration backups. Database archives in `~/.slowave/backups` are retained. This is destructive and asks for confirmation. |

`slowave cleanup` remains a compatibility alias for `slowave purge`; use
`purge` in new scripts and documentation. To remove the Python application
after either command, use the same installer that installed it, for example
`pipx uninstall slowave`.

## Environment variables

| Variable | Default | Purpose |
|---|---:|---|
| `SLOWAVE_DB` | `~/.slowave/slowave.db` | SQLite database path |
| `SLOWAVE_MCP_HTTP_PORT` | `8766` | HTTP MCP daemon port |
| `SLOWAVE_MCP_HOST` | `127.0.0.1` | HTTP MCP bind host |
| `SLOWAVE_DAEMON_PID` | `~/.slowave/daemon.pid` | Daemon PID-file path |
| `SLOWAVE_SESSION_IDLE_TIMEOUT` | `3600` | Idle-session timeout in seconds |
| `SLOWAVE_BACKUP_DIR` | `~/.slowave/backups` | Default backup directory |
| `SLOWAVE_BACKUP_KEEP` | `7` | Number of backups retained |
| `KMP_DUPLICATE_LIB_OK` | — | Set to `TRUE` on macOS only if FAISS and ONNX otherwise segfault |

Run `slowave <command> --help` for the complete, installed-version reference.
