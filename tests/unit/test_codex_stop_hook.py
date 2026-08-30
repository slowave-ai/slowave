from __future__ import annotations

import json

from click.testing import CliRunner

from slowave.cli.main import cli


def test_codex_stop_hook_blocks_with_structured_continuation() -> None:
    result = CliRunner().invoke(cli, ["hook", "codex-stop"], input='{"stop_hook_active":false}')
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["decision"] == "block"
    assert payload["reason"].startswith("SLOWAVE MANDATORY:")


def test_codex_stop_hook_allows_loop_safe_second_stop() -> None:
    result = CliRunner().invoke(cli, ["hook", "codex-stop"], input='{"stop_hook_active":true}')
    assert result.exit_code == 0
    assert json.loads(result.output) == {}
