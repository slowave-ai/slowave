# Antigravity + Slowave

## Setup

```bash
slowave setup --client antigravity
```

`slowave setup` also configures Antigravity among all detected clients. Use `--dry-run` to preview changes. Setup starts the shared Slowave services unless `--no-worker` is specified.

Antigravity is detected when `agy` or `antigravity` is on PATH, `~/.gemini/antigravity/` exists, or its global MCP config exists.

| What | Global path |
|---|---|
| MCP server | `~/.gemini/config/mcp_config.json` |
| Lifecycle instructions | `~/.gemini/config/AGENTS.md` |

Setup uses Antigravity's `serverUrl` field to connect to the local HTTP daemon and injects the same lifecycle instructions as other Slowave clients. An empty MCP config from Antigravity's first launch is supported. These global paths are documented for the Antigravity app, IDE and CLI. Restart Antigravity or reload MCP servers after setup.

## Verify

```bash
agy mcp list
slowave doctor
```

Start a conversation, ask Antigravity to remember a project fact, then open a new conversation in the same project and ask for that fact. The lifecycle should activate memory, save useful claims, assess retrievals, and commit before answering. Approve Slowave MCP tools when Antigravity requests permission.

For headless tests, grant permission to Slowave's tools for that invocation:

```bash
agy --settings '{"permissions":{"allow":["mcp(slowave/*)"]}}' -p "What project guidance do you remember?"
```

Antigravity headless mode can exit successfully even when a tool is denied. Check `denied_actions` and tool results in `--output-format stream-json` output, rather than relying on the exit code alone. See [Antigravity headless permissions](https://antigravity.google/docs/cli/headless/).

## Removal

Run `slowave uninstall` to remove the integration while keeping memories, or `slowave cleanup` to remove Slowave data too. Other MCP servers and instructions are preserved. See the [installation guide](../../docs/install.md), [Antigravity MCP documentation](https://antigravity.google/docs/mcp), and [rules documentation](https://antigravity.google/docs/rules/).

Live validation: Antigravity CLI 1.4.12 on macOS, using the shared Slowave HTTP daemon. A fact saved in one CLI conversation was retrieved in a fresh conversation, with complete feedback and successful commits. The desktop app, IDE, Linux and Windows have not been tested live.
