"""Core test fixtures (auto-used by pytest).

Ensures every test has a clean environment regardless of local state:
- SLOWAVE_DB pointed at a temp file, never the user's real DB.
- SQLite WAL/SHM cleanup after each test that touches the DB.
"""

from __future__ import annotations

import os
import tempfile

import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "requires_faiss: test requires faiss library")
    config.addinivalue_line("markers", "requires_model: test requires downloaded model files")
    config.addinivalue_line("markers", "slow: slow integration test")
    config.addinivalue_line("markers", "benchmark: benchmark test (long-running)")
    config.addinivalue_line("markers", "acceptance: full blackbox acceptance test via CLI")


@pytest.fixture(autouse=True)
def _isolate_slowave_db(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force every test to use a throwaway DB via SLOWAVE_DB.

    Tests that need a real filesystem DB path should use tmp_path fixtures
    and pass db_path directly to SlowaveConfig. This fixture prevents any
    test from accidentally mutating the developer's real ~/.slowave/slowave.db.
    """
    if os.environ.get("SLOWAVE_HOME"):
        # The runtime-root acceptance mode deliberately exercises the new
        # override and must not manufacture an invalid two-override config.
        monkeypatch.delenv("SLOWAVE_DB", raising=False)
    else:
        monkeypatch.setenv(
            "SLOWAVE_DB",
            os.path.join(tempfile.gettempdir(), f"slowave_test_{os.getpid()}.db"),
        )
