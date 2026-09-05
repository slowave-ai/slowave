"""Black-box proof that distinct runtime roots cannot retrieve each other's data."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def _cli(root: Path, *args: str) -> dict:
    env = os.environ.copy()
    env.pop("SLOWAVE_DB", None)
    env["SLOWAVE_HOME"] = str(root)
    result = subprocess.run(
        [sys.executable, "-m", "slowave", "--json", *args],
        env=env,
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
    )
    return json.loads(result.stdout)


def test_two_runtime_roots_do_not_cross_retrieve(tmp_path):
    root_a = tmp_path / "user-a"
    root_b = tmp_path / "user-b"
    scope = "project:runtime-isolation"

    _cli(root_a, "remember", "secret-A", "--type", "fact", "--scope", scope)
    from_b = _cli(root_b, "recall", "secret-A", "--scope", scope)
    assert all(memory["content_text"] != "secret-A" for memory in from_b["memories"])

    _cli(root_b, "remember", "secret-B", "--type", "fact", "--scope", scope)
    from_a = _cli(root_a, "recall", "secret-B", "--scope", scope)
    assert all(memory["content_text"] != "secret-B" for memory in from_a["memories"])

    assert (root_a / "slowave.db").is_file()
    assert (root_b / "slowave.db").is_file()
    assert (root_a / "slowave.db").resolve() != (root_b / "slowave.db").resolve()
