# Slowave local dashboard

Slowave includes a local web dashboard for inspecting the memory
database, MCP/server processes, schema health, schema relations, and schema graph.

The dashboard is intended for local development and operational hygiene. The
only way to mutate memory content from it is forgetting a schema (see
[Forgetting a memory](#forgetting-a-memory) below), enabled by default; pass
`--no-allow-actions` for a strictly read-only dashboard, e.g. before sharing
a screen or a port.

## Launch

```bash
slowave dashboard
```

Then open:

```text
http://127.0.0.1:8765
```

Common options:

```bash
# Use a different port.
slowave dashboard --port 8766

# Do not open the browser automatically.
slowave dashboard --no-open

# Refresh visible Home and Diagnostics observations every 5 seconds.
slowave dashboard --refresh-ms 5000

# Disable the Forget/Unforget buttons for a strictly read-only dashboard.
slowave dashboard --no-allow-actions

# Enable creator-only diagnostics and experimental measurements.
slowave dashboard --experimental
```

The default DB is `~/.slowave/slowave.db`. Use `SLOWAVE_DB` or the global
`--db /path/to/slowave.db` option only when you need to inspect another DB.

The dashboard binds to `127.0.0.1` by default. Binding to a non-localhost address
prints a warning because Slowave memory content may contain private project or
user information.

## What it shows

### Home

The default landing view separates three operational observations: MCP daemon
availability, SQLite integrity, and the last recorded maintenance result. It
then shows actionable exceptions and a chronological feed of conservatively
observed memory, retrieval, activity, and procedure changes for the selected
period. It does not claim to know when a particular viewer last looked.

When enough recent data exists, Home shows separate lanes for captured activity,
episodes formed, and memories formed. It never sums these unlike records into a
single health or quality signal. A new installation instead shows the ordinary
lifecycle steps that will populate the workbench.

### Memory

The **Memory** page is a server-paginated library. Search text stays in local page
state rather than browser history. Structural state, scope, date, sort, and page
filters are bookmarkable. Open a memory to inspect its current retrieval effect,
evidence, recorded exposures, feedback or replacement observations, relations,
and advanced metadata.

Use **Suppress memory** or **Restore memory** only from detail. The confirmation
states the scope and reversibility and confirms that source evidence remains.

### Retrieval

The **Retrieval** page lists activation and recall snapshots. Detail shows the
exact admitted items, their persisted pathway, the product-level Direct or
Associated grouping, and explicit target feedback. Exposure, use, effect, and
task outcome remain separate; missing feedback is shown as unknown, not negative.

### Procedures

The **Procedures** page lists execution-backed records captured at session close.
Detail links verification, source activity, context, steps, caveats, retrieval
exposures, and explicit use/effect feedback. A capture is not presented as a
proven general playbook.

### Activity

The **Activity** page lists recorded task sessions and opens a bookmarkable,
chronological detail view. It links events, episodes, retrievals, feedback,
supported memory-formation evidence, and a captured procedure. Related sessions
share a continuity identifier; the UI explicitly treats that as correlation,
not a complete work-attempt model.

### Diagnostics

**Diagnostics** contains service probes, maintenance-run history, database
integrity, storage sizes, and collapsed raw SQLite details. These are operational
observations, not measures of memory value. No repair, restart, or graph mutation
is available from the beta dashboard.

### Labs (creator-only)

Pass `--experimental` to expose **Labs** under Diagnostics. It contains lifecycle
cohort measurements, procedure-exposure diagnostics, and the bounded graph
explorer with an accessible table alternative. Every section states its
population and limitation and is labelled as not a product metric.


## Forgetting a memory

If you spot a memory in the Memories tab that's wrong, stale, or something you
just don't want influencing future recall, you can suppress it — expand the
schema and click **Forget** (shown by default). This sets the schema's
status to `forgotten`, which hides it
from `activate`/`recall` in every retrieval mode (`strict_scope`, `broad`,
`debug`). It's reversible: click **Unforget** on a forgotten schema to restore
it to whatever status it had before (not always `active` — a schema that was
already `superseded` or `contradicted` goes back to that, not `active`).
Forgetting is logged with an optional reason to `schema_forget_log` for audit,
and the underlying episodes/raw events/evidence are never touched — only the
schema row's status changes.

Forget/Unforget are deliberately **CLI and dashboard only** — there is no MCP
tool for this, unlike `remember`/`recall`/`reinforce`/`commit`. Forgetting is
meant to be a deliberate action a human takes after looking at a specific
memory, not something an AI agent infers from conversational subtext (which
could also make it a prompt-injection target if it were a callable tool). See
`slowave forget`/`slowave unforget` in [`docs/cli.md`](cli.md) for the CLI
equivalent, which works without a running dashboard.

## Local JSON API

The dashboard serves a small JSON API on the same local HTTP server:

| Endpoint | Purpose |
|---|---|
| `GET /api/home?hours=24&scope=...` | Home availability sources, attention observations, recent changes, and separate activity lanes |
| `GET /api/status` | Installation snapshot, scopes, recent sessions, and service observations |
| `GET /api/db/health` | SQLite pragmas, integrity check, FK check, table counts |
| `GET /api/schemas?states=active,needs_review&page=1&per_page=50` | Paginated memory list, state counts, structural filters, and server sort |
| `GET /api/schemas/123` | Memory detail, evidence, retrieval exposures, feedback, audit, and relations |
| `GET /api/retrievals?page=1&per_page=50` | Paginated retrieval snapshots and denominator-safe observed summary |
| `GET /api/retrievals/:id` | Exact exposed items, stored pathways, feedback, and source activity |
| `GET /api/activity?page=1&per_page=50` | Paginated session activity list |
| `GET /api/activity/:session_id` | Chronological activity detail with retrieval, feedback, memory, and procedure links |
| `GET /api/procedures/:id` | Captured procedure detail, verification, exposures, and explicit effects |
| `GET /api/graph/schemas?limit=120&min_salience=2.5` | Schema graph data |
| `GET /api/procedural-memory?page=1&per_page=50` | Paginated captured procedures and observed transfer fields |
| `GET /api/labs/rollout` | Experimental retrieval, lifecycle, and feedback diagnostics; available only with `--experimental` |
| `POST /api/schemas/123/forget` | Suppress schema 123 (status → `forgotten`). Body: optional JSON `{"reason": "..."}`. `403` if started with `--no-allow-actions`. |
| `POST /api/schemas/123/unforget` | Undo a forget, restoring schema 123's prior status. `403` if started with `--no-allow-actions`. |

Example Labs graph request:

```bash
curl 'http://127.0.0.1:8765/api/graph/schemas?limit=120&statuses=active,needs_review&min_salience=2.5'
```

## Safety and limitations

- The only mutating action is forgetting/unforgetting a schema. It's enabled
  by default; pass `--no-allow-actions` for a strictly read-only dashboard —
  see [Forgetting a memory](#forgetting-a-memory).
- It is local-first and binds to `127.0.0.1` by default.
- It uses Python stdlib HTTP serving and packaged Vite-built React/TypeScript
  assets. Users do not need Node.js at runtime; contributors can rebuild the UI
  from `slowave/dashboard/ui` with `npm run build`.
- Labs measurements are exploratory diagnostics, not claims about retrieval
  quality, causal usefulness, or system-wide reliability.
