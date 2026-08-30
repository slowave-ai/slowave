# Contributing to Slowave

Thanks for your interest in contributing to Slowave.

Slowave is an experimental, local-first memory engine for AI agents. The project is currently in beta, so APIs, storage layout, configuration, and behavior may still change as the design evolves.

This document explains how to contribute, what kinds of contributions are most useful, and how licensing works.

## Project status

Slowave is currently beta software.

That means:

- the core ideas are implemented and usable;
- the public API may still change;
- storage schema and migrations are not yet guaranteed stable;
- benchmarks should be treated as directional rather than definitive;
- feedback, testing, bug reports, and design critique are highly valuable.

Please open an issue before starting large changes.

## Good ways to contribute

The most useful contributions at this stage are:

- bug reports with clear reproduction steps;
- installation feedback on macOS, Linux, and Windows;
- dependency and packaging fixes;
- documentation improvements;
- small usability improvements to the CLI;
- integration examples for agent frameworks and MCP clients;
- benchmark and evaluation scripts;
- tests for recall, consolidation, replay, and storage behavior;
- reports of confusing or noisy memory retrieval results.

Design feedback is also welcome, especially around:

- memory evolution;
- recall ranking;
- salience and decay;
- schema and prototype formation;
- context injection;
- local-first privacy and storage behavior.

## Before opening a pull request

Before submitting a PR:

1. Check whether there is already an open issue or PR for the same topic.
2. For substantial changes, open an issue first and describe the proposed approach.
3. Keep PRs focused and reasonably small.
4. Include tests or a clear manual validation note when possible.
5. Update documentation if the behavior, CLI, configuration, or public API changes.

## Development setup

Clone the repository:

```bash
git clone https://github.com/mrsalty/slowave.git
cd slowave
```

The project uses [uv](https://github.com/astral-sh/uv) for dependency management.
Install all dependencies (including dev tools) with:

```bash
uv sync
```

Run the unit tests:

```bash
uv run pytest
```

All tests should pass.  Integration benchmarks (under `tests/integration/`) require
additional datasets and are skipped automatically in a clean checkout.

Run the CLI locally:

```bash
uv run slowave --help
uv run slowave doctor
```

**Alternative: plain pip**

If you prefer a plain virtual environment, install the `dev` extra:

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest
```

If dependency installation fails, please open an issue with:

- operating system;
- Python version (`python --version`);
- installation command used;
- full error message;
- whether the failure involves `faiss-cpu`, `onnxruntime`, or `spacy`.

## Reporting bugs

A good bug report includes:

- what you expected to happen;
- what actually happened;
- exact command or integration path used;
- Slowave version;
- operating system and Python version;
- relevant logs or traceback;
- whether the issue reproduces on a fresh database.

Please avoid sharing private memory data in public issues. Redact personal content before posting logs, database snippets, or recall outputs.

## Feature requests

Feature requests are welcome, but please describe the use case rather than only the implementation.

Useful format:

```text
Problem:
What I tried:
What I expected:
Why this matters:
Possible solution:
```

## Pull request guidelines

Please keep PRs focused.

Good PRs usually:

- solve one problem;
- include a clear description;
- include tests or manual validation steps;
- avoid unrelated formatting changes;
- update docs when behavior changes;
- preserve local-first behavior unless explicitly discussed.

For larger changes, please discuss the design first.

Examples of larger changes:

- storage schema changes;
- migration logic;
- embedding model changes;
- recall ranking changes;
- replay or consolidation behavior;
- public API changes;
- MCP/tooling changes;
- dependency changes involving `torch`, `sentence-transformers`, `faiss`, or `spacy`.

## Licensing

Slowave is released under the GNU Affero General Public License v3.0 or later, unless otherwise stated.

The AGPL license is used to keep improvements to the open memory engine available to the community, especially when modified versions are operated as network services.

Commercial licensing may be offered in the future for organizations that want to embed, distribute, or operate Slowave under different terms.

## Contributor License Agreement

Substantial code contributions may require acceptance of a Contributor License Agreement, or CLA, before they can be merged.

The purpose of the CLA is to allow Slowave to remain open source while preserving the option to offer commercial licensing in the future.

For now:

- bug reports are welcome and do not require a CLA;
- feature requests are welcome and do not require a CLA;
- small documentation fixes may not require a CLA;
- substantial code contributions may require CLA acceptance before merge;
- maintainers may defer large PRs until the CLA process is in place.

By submitting a pull request, you confirm that:

- you have the right to submit the contribution;
- your contribution does not knowingly include code you are not allowed to contribute;
- you understand that substantial contributions may require CLA acceptance before merging.

## Security and privacy reports

Please do not open public issues for security-sensitive problems.

See [SECURITY.md](./SECURITY.md) for the full security policy and reporting instructions.

## Code of conduct

Be respectful and constructive.

Slowave is an experimental project. Strong technical critique is welcome, but keep discussions focused on the design, implementation, and evidence.

## Maintainer expectations

Slowave is early-stage. Some proposed changes may be declined or postponed even if they are technically valid.

Reasons may include:

- the API is not stable yet;
- the storage model is still evolving;
- the change increases dependency complexity;
- the change conflicts with local-first goals;
- the change is better suited for a later server or SaaS layer;
- the project needs more evaluation before committing to the behavior.

When possible, maintainers will explain the reason and suggest a better path.
