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

import json
import os
import sqlite3
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


def test_schemas_are_ordered_by_creation_date_descending():
    eng, path = _tmp_engine()
    try:
        older_id = eng.schemas.create(
            content_text="older schema", facets={}, tags=[], embedding=None, dedupe=False
        )
        newer_id = eng.schemas.create(
            content_text="newer schema", facets={}, tags=[], embedding=None, dedupe=False
        )
        with sqlite3.connect(path) as conn:
            conn.execute("UPDATE schemas SET first_formed_ts = ? WHERE id = ?", (100, older_id))
            conn.execute("UPDATE schemas SET first_formed_ts = ? WHERE id = ?", (200, newer_id))

        payload = _schemas_payload(path, {})

        assert [schema["id"] for schema in payload["schemas"]] == [
            f"sch_{newer_id}",
            f"sch_{older_id}",
        ]
    finally:
        eng.close()
        _cleanup(path)


def test_schema_sort_ranks_full_result_set_server_side():
    eng, path = _tmp_engine()
    try:
        ids = [
            eng.schemas.create(
                content_text=f"schema {i}", facets={}, tags=[], embedding=None, dedupe=False
            )
            for i in range(3)
        ]
        with sqlite3.connect(path) as conn:
            for sid, sal in zip(ids, [0.9, 0.1, 0.5]):
                conn.execute("UPDATE schemas SET salience = ? WHERE id = ?", (sal, sid))

        asc = _schemas_payload(path, {"sort": ["salience"], "dir": ["asc"]})
        assert [s["schema_id"] for s in asc["schemas"]] == [ids[1], ids[2], ids[0]]

        desc = _schemas_payload(path, {"sort": ["salience"], "dir": ["desc"]})
        assert [s["schema_id"] for s in desc["schemas"]] == [ids[0], ids[2], ids[1]]

        by_id = _schemas_payload(path, {"sort": ["id"], "dir": ["desc"]})
        assert [s["schema_id"] for s in by_id["schemas"]] == list(reversed(ids))
    finally:
        eng.close()
        _cleanup(path)


def test_schema_sort_tiebreak_reverses_on_direction_toggle():
    eng, path = _tmp_engine()
    try:
        ids = [
            eng.schemas.create(
                content_text=f"s{i}", facets={}, tags=[], embedding=None, dedupe=False
            )
            for i in range(4)
        ]
        with sqlite3.connect(path) as conn:
            for sid in ids:
                conn.execute("UPDATE schemas SET status = 'active' WHERE id = ?", (sid,))

        # All rows share status='active', so the id tiebreak must swap asc/desc.
        asc = _schemas_payload(path, {"sort": ["status"], "dir": ["asc"]})
        desc = _schemas_payload(path, {"sort": ["status"], "dir": ["desc"]})
        assert [s["schema_id"] for s in asc["schemas"]] == sorted(ids)
        assert [s["schema_id"] for s in desc["schemas"]] == sorted(ids, reverse=True)
    finally:
        eng.close()
        _cleanup(path)


def test_schema_sort_by_support_uses_ids_object_and_reverses():
    eng, path = _tmp_engine()
    try:
        ids = [
            eng.schemas.create(
                content_text=f"s{i}", facets={}, tags=[], embedding=None, dedupe=False
            )
            for i in range(4)
        ]
        with sqlite3.connect(path) as conn:
            # supporting_episode_ids is stored as a JSON object {"ids": [...]}.
            conn.execute(
                "UPDATE schemas SET supporting_episode_ids = ? WHERE id = ?",
                (json.dumps({"ids": [1, 2]}), ids[0]),
            )
            conn.execute(
                "UPDATE schemas SET supporting_episode_ids = ? WHERE id = ?",
                (json.dumps({"ids": [3, 4, 5]}), ids[1]),
            )
            conn.execute(
                "UPDATE schemas SET supporting_episode_ids = ? WHERE id = ?",
                (json.dumps({"ids": [7]}), ids[2]),
            )
            conn.execute(
                "UPDATE schemas SET supporting_episode_ids = ? WHERE id = ?",
                (json.dumps([]), ids[3]),
            )

        # support counts via _ids_from_json: ids[1]=3, ids[0]=2, ids[2]=1, ids[3]=0.
        desc = _schemas_payload(path, {"sort": ["support"], "dir": ["desc"]})
        assert [s["schema_id"] for s in desc["schemas"]] == [ids[1], ids[0], ids[2], ids[3]]
        asc = _schemas_payload(path, {"sort": ["support"], "dir": ["asc"]})
        assert [s["schema_id"] for s in asc["schemas"]] == [ids[3], ids[2], ids[0], ids[1]]
    finally:
        eng.close()
        _cleanup(path)


def test_schema_sort_expressions_do_not_raise_for_derived_columns():
    eng, path = _tmp_engine()
    try:
        eng.schemas.create(
            content_text="a schema",
            facets={"schema_class": "fact"},
            tags=[],
            embedding=None,
            dedupe=False,
        )
        # Derived/computed columns and JSON-extracted columns must not raise.
        for col in (
            "id",
            "status",
            "salience",
            "confidence",
            "stage",
            "class",
            "scope",
            "support",
            "content",
        ):
            payload = _schemas_payload(path, {"sort": [col], "dir": ["asc"]})
            assert len(payload["schemas"]) == 1
        # Unknown sort column falls back to the default ordering, no error.
        payload = _schemas_payload(path, {"sort": ["bogus"], "dir": ["desc"]})
        assert len(payload["schemas"]) == 1
    finally:
        eng.close()
        _cleanup(path)
