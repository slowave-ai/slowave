# Security Policy

## Reporting a Vulnerability

Please **do not** open a public GitHub issue for security-sensitive reports.

Security-sensitive examples include:

- memory data exposure or leakage between scopes
- unsafe file permissions on the database or config files
- path traversal in file handling
- prompt or context injection via stored memory
- accidental logging of private memory content
- unsafe default storage behavior

### How to report

Open a **minimal public issue** stating only that you found a security-sensitive problem and request a private contact channel. A maintainer will respond with a way to share details privately.

Alternatively, use [GitHub's private vulnerability reporting](https://github.com/mrsalty/slowave/security/advisories/new) if you prefer a fully private channel from the start.

## Scope

Slowave stores memory locally in plaintext beneath the current OS user's native
application-data directory. The runtime root contains the SQLite database,
SQLite sidecars, logs, backups, and daemon state. `slowave doctor` prints the
effective root and database path. The default root is per OS user;
`SLOWAVE_HOME` may intentionally select another complete runtime tree, while
legacy `SLOWAVE_DB` selects an exact database file. The data is unencrypted, so
protect sensitive information with OS-level permissions or full-disk encryption.

The MCP server listens on localhost only (`127.0.0.1`) and is not intended to be exposed to a network.

## Response

This is an early-stage open source project maintained by a small team. We will acknowledge reports promptly and aim to ship fixes as quickly as the severity warrants.
