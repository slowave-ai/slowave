"""Regression tests for the 2026-07-24 Tier-0 audit dashboard findings:

- `_schemas_payload` used to do raw `int((qs.get("limit") or [100])[0])` with
  no guard -- a non-numeric `limit` query param raised an uncaught ValueError,
  caught only by do_GET's blanket `except Exception`, returning an HTTP 500
  with a raw Python message instead of a clean 400. Now uses `_qs_int`, which
  falls back to the default instead of raising.
- An unknown `status` filter used to silently no-op (skip the WHERE clause
  entirely, returning every status) instead of erroring -- now raises
  ValueError, which do_GET/do_POST translate into a clean 400 response.
"""

from __future__ import annotations

import os
import tempfile

import pytest

from slowave.core.config import SlowaveConfig
from slowave.core.engine import SlowaveEngine
from slowave.dashboard.app import _schemas_payload


def _tmp_engine() -> tuple[SlowaveEngine, str]:
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    cfg = SlowaveConfig(db_path=tmp.name, dim=8, disable_encoder=True)
    return SlowaveEngine(cfg), tmp.name


def _cleanup(path: str) -> None:
    for ext in ("", "-wal", "-shm"):
        p = path + ext
        if os.path.exists(p):
            os.remove(p)


def test_non_numeric_limit_falls_back_to_default_instead_of_raising():
    eng, path = _tmp_engine()
    try:
        eng.schemas.create(
            content_text="a schema", facets={}, tags=[], embedding=None, dedupe=False
        )
        payload = _schemas_payload(path, {"limit": ["abc"]})
        assert len(payload["schemas"]) == 1
    finally:
        eng.close()
        _cleanup(path)


def test_unknown_status_filter_raises_value_error():
    eng, path = _tmp_engine()
    try:
        with pytest.raises(ValueError):
            _schemas_payload(path, {"status": ["bogus_status"]})
    finally:
        eng.close()
        _cleanup(path)


def test_known_status_filter_still_works():
    eng, path = _tmp_engine()
    try:
        sid = eng.schemas.create(
            content_text="an active schema", facets={}, tags=[], embedding=None, dedupe=False
        )
        eng.schemas.forget(sid)
        payload = _schemas_payload(path, {"status": ["forgotten"]})
        assert len(payload["schemas"]) == 1
        assert payload["schemas"][0]["id"] == f"sch_{sid}"
    finally:
        eng.close()
        _cleanup(path)
