# Slowave MCP acceptance tests

This directory is the executable description of Slowave's public memory contract.
Each test starts a real MCP server, sends only public MCP requests, and checks only
public responses. Tests do not seed or inspect SQLite directly.

## Run the suite

```bash
uv run pytest tests/acceptance -vv -p no:randomly
```

Use the production encoder before merging changes that affect semantic retrieval:

```bash
SLOWAVE_ACCEPTANCE_ENCODER=production \
uv run pytest tests/acceptance -vv -p no:randomly
```

Use `-vv` rather than `-v`: the repository defaults to `-q` in
`pyproject.toml`, so `-v` only cancels that quiet setting. `-vv` shows each
scenario's complete test name and result.

The production model is prepared once at session start and then loaded from
the local cache by each isolated MCP server. This prevents repeated downloads
and Hugging Face network checks. Add `-s` only when debugging server output;
normal runs keep diagnostics captured and show the scenario names and results.
For child-server diagnostics with `-s`, also set
`SLOWAVE_ACCEPTANCE_VERBOSE=1`.

The deterministic encoder is the fast, repeatable contract gate. The production
encoder is the semantic-quality gate. A test marked `xfail` under the deterministic
encoder must pass with the production encoder.

## The stories

| File | What a client learns from it |
|---|---|
| `test_mcp_contract.py` | The five lifecycle tools reject invalid boundaries safely, require complete feedback, leave a valid session usable after a rejected request, and fail when feedback enforcement is deliberately disabled. |
| `test_mcp_lifecycle.py` | Context and continuity, overlapping sessions, separate client connections, abandoned-session reaping, and a clean-install first lifecycle work through public MCP calls. |
| `test_memory_lifecycle.py` | A client can store, retrieve, assess, correct, and date memories without scope leaks, stale current guidance, or irrelevant context padding; deliberate scope, stale, budget, and relevance regressions are caught. |
| `test_procedure_lifecycle.py` | A completed procedure returns as verified guidance; a failed procedure returns with a system warning that a client can decline to follow. |

## What does not belong here

Acceptance tests protect what an MCP client can observe. Internal details such as
SQLite rows, salience values, labile state, graph edges, and promotion counters are
covered by unit tests. They must not become acceptance assertions unless Slowave
adds a corresponding public product promise.
