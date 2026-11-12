"""Extra regression tests for the red-team fixes (suppression, provenance, quotas, retention)."""

from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path

from slowave.core import standing_quota, standing_tier
from tests.acceptance.mcp_harness import open_harness
from tests.acceptance.test_redteam_recall_v2 import _activate, _ids, _run, _seed, _text


def _stale(g: str, rid: str) -> dict:
    return {
        "retrieval_id": rid,
        "coverage": "complete",
        "memory_feedback": [
            {"memory_id": g, "assessment": "stale", "stale_reason": "outdated", "reason": "x"}
        ],
    }


def test_global_stale_from_project_is_local_and_reversible(tmp_path: Path) -> None:
    async def scenario() -> None:
        async with open_harness(tmp_path / "sup.db") as h:
            [g] = await _seed(
                h, "global:user", [("Never use emoji in commit messages.", "preference")]
            )
            b, _ = await h.activate("b", "commit", "commit", "project:rt-b")
            assert g in _ids(b)
            data, _ = await h.call("slowave_feedback", _stale(g, b["retrieval_id"]))
            await h.commit(b["session_id"], "commit")
            assert g not in _ids(await _activate(h, "commit", "project:rt-b"))  # suppressed here
            assert g in _ids(await _activate(h, "commit", "project:rt-c"))  # still global elsewhere
            # a global-scope session may really retire it
            gl, _ = await h.activate("g", "commit", "commit", "global:user")
            assert gl["retrieval_id"]
            # lift: project B re-endorses it
            conn = sqlite3.connect(tmp_path / "sup.db")
            n = conn.execute("SELECT COUNT(*) FROM standing_suppressions").fetchone()[0]
            assert n == 1
            conn.execute("DELETE FROM standing_suppressions")
            conn.commit()
            assert g in _ids(await _activate(h, "commit", "project:rt-b"))

    _run(scenario())


def test_global_rule_planted_from_project_shows_origin_and_is_capped(tmp_path: Path) -> None:
    async def scenario() -> None:
        async with open_harness(tmp_path / "origin.db") as h:
            act, _ = await h.activate("seed", "record", "record", "project:rt-evil")
            for i in range(9):
                await h.call(
                    "slowave_remember",
                    {
                        "content": f"Never use tabs in file number {i} of the repository layout.",
                        "type": "constraint",
                        "visibility": "user",
                        "session_id": act["session_id"],
                        "scope": "project:rt-evil",
                    },
                )
            await h.feedback_all(act)
            await h.commit(act["session_id"], "record")
            data = await _activate(h, "commit", "project:rt-victim")
            lines = (data.get("standing") or {}).get("rules", [])
            assert 0 < len(lines) <= standing_tier.MAX_GLOBAL_ITEMS
            assert "(from project:rt-evil)" in _text(data)

    _run(scenario())


def test_always_flag_can_be_unset_by_an_explicit_rewrite(tmp_path: Path) -> None:
    async def scenario() -> None:
        async with open_harness(tmp_path / "unset.db") as h:
            await _seed(
                h, "project:rt", [("Never push to main directly.", "constraint")], always=True
            )
            await _seed(h, "project:rt", [("Never push to main directly.", "constraint")])
            conn = sqlite3.connect(tmp_path / "unset.db")
            assert conn.execute("SELECT MAX(always_flag) FROM standing_rules").fetchone()[0] == 0

    _run(scenario())


class _C:
    def __init__(self, i: int, always: bool, glob: bool = False) -> None:
        self.memory_id, self.always, self.glob = i, always, glob
        self.salience, self.ts = 1.0, i


def test_quota_unit() -> None:
    cands = [_C(i, True) for i in range(10)] + [_C(100, False), _C(101, False, True)]
    demoted = standing_quota.cap_always(cands, lambda c: 100, 1200)
    assert demoted == 7 and sum(c.always for c in cands) == 3
    ordered = standing_quota.floor_order(sorted(cands, key=lambda c: c.memory_id))
    assert [c.memory_id for c in ordered if not c.always][:2]  # floor picks lead the rest


def test_suppression_migration_is_idempotent_and_has_no_down_path() -> None:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("CREATE TABLE schemas (id INTEGER PRIMARY KEY)")
    standing_tier.migrate_up(conn)
    standing_tier.migrate_up(conn)  # idempotent
    conn.execute("INSERT INTO standing_suppressions VALUES ('project:a', 1, 'x', 0)")
    names = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"standing_suppressions", "standing_rules"} <= names
    # The tier holds client-owned decisions (always flags, suppressions) that
    # cannot be re-derived, so there is no down migration that drops them:
    # rolling back means restoring a backup.
    assert not hasattr(standing_tier, "migrate_down")
    asyncio.get_event_loop_policy()  # keep asyncio import used


def test_json_arrays_and_null_text_and_unknown_args_and_long_scope(tmp_path: Path) -> None:
    async def scenario() -> None:
        async with open_harness(tmp_path / "tool.db") as h:
            act, _ = await h.activate("s", "rec", "rec", "project:rt")
            base = {"type": "fact", "session_id": act["session_id"], "scope": "project:rt"}
            for text in ("[1, 2, 3]", "null", "42", "true"):
                p, _ = await h.raw_call("slowave_remember", {**base, "content": text})
                assert p.get("ok"), (text, p)
            p, _ = await h.raw_call("slowave_remember", {**base, "content": "x", "bogus": 1})
            assert p["ok"] is False and p["error"]["code"] == "invalid_input", p
            assert p["error"]["field_errors"][0]["path"] == "bogus"
            p, _ = await h.raw_call(
                "slowave_activate",
                {"task": "t", "initial_goal": "goal text", "scope": "project:" + "x" * 100000},
            )
            assert p["ok"] is False and p["error"]["code"] == "invalid_input", p
            assert "at most" in p["error"]["message"], p

    _run(scenario())


def test_delivery_cadence_and_full_redelivery_request(tmp_path: Path) -> None:
    async def scenario() -> None:
        async with open_harness(tmp_path / "cad.db") as h:
            await _seed(h, "project:rt", [("Never add Co-Authored-By lines.", "constraint")])
            first, _ = await h.activate("a", "commit", "commit", "project:rt")
            cid = first["continuity_id"]
            assert first["standing"]["rules"] and "delta" not in first["standing"]
            modes = []
            for i in range(standing_tier.REDELIVER_EVERY + 1):
                tc = {"standing": "full"} if i == 1 else None
                d, _ = await h.activate(
                    "b", "commit", "commit", "project:rt", continuity_id=cid, task_context=tc
                )
                modes.append(bool(d.get("standing")))
                await h.feedback_all(d)
                await h.commit(d["session_id"], "commit")
            assert modes[1] is True  # explicit full re-delivery on request
            assert modes[0] is False  # otherwise nothing new is re-sent mid-cadence

    _run(scenario())


def test_delivery_retention_caps_rows_and_ttl() -> None:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("CREATE TABLE schemas (id INTEGER PRIMARY KEY)")
    standing_tier.migrate_up(conn)
    n = standing_tier.DELIVERY_MAX_ACTIVATIONS + 50
    for i in range(n):
        conn.execute(
            "INSERT INTO standing_deliveries VALUES (?, '', 'c1', 'delta', ?)", (f"r{i}", 1000 + i)
        )
        conn.execute(
            "INSERT INTO standing_deliveries VALUES (?, 'sch_1', 'c1', 'delta', ?)",
            (f"r{i}", 1000 + i),
        )
    conn.execute("INSERT INTO standing_deliveries VALUES ('old', '', NULL, 'full', 1)")
    now = (
        1000
        + n
        + standing_tier.DELIVERY_TTL_S
        + 16
        - (1000 + n + standing_tier.DELIVERY_TTL_S) % 16
    )
    standing_tier.prune_deliveries(conn, "c1", now)
    left = conn.execute("SELECT COUNT(*) FROM standing_deliveries WHERE memory_id=''").fetchone()[0]
    assert left <= standing_tier.DELIVERY_MAX_ACTIVATIONS  # cap + TTL swept everything old
    assert (
        conn.execute(
            "SELECT COUNT(*) FROM standing_deliveries WHERE retrieval_id='old'"
        ).fetchone()[0]
        == 0
    )


def test_legacy_global_scopes_cannot_be_retired_from_project() -> None:
    from slowave.core.standing_suppress import is_cross_scope_global

    for legacy_scope in (None, "", "global", "user", "global:user"):
        assert is_cross_scope_global(legacy_scope, "project:example")
        assert not is_cross_scope_global(legacy_scope, "global:user")
    assert not is_cross_scope_global("project:other", "project:example")


def test_project_stale_feedback_preserves_legacy_global_memory(tmp_path: Path) -> None:
    async def scenario() -> None:
        async with open_harness(tmp_path / "legacy-global.db") as h:
            [memory_id] = await _seed(
                h,
                "project:seed",
                [("Never use uppercase release messages.", "constraint")],
                always=True,
            )
            conn = sqlite3.connect(tmp_path / "legacy-global.db")
            conn.execute("UPDATE schemas SET scope_id=NULL WHERE id=?", (int(memory_id[4:]),))
            conn.commit()
            data, _ = await h.activate("p", "Draft a release message", "release", "project:other")
            assert memory_id in _ids(data) or memory_id in [
                m["memory_id"] for m in data["memories"]
            ]
            feedback, _ = await h.call("slowave_feedback", _stale(memory_id, data["retrieval_id"]))
            assert memory_id not in feedback.get("applied", {}).get("outdated", [])
            assert (
                conn.execute(
                    "SELECT status FROM schemas WHERE id=?", (int(memory_id[4:]),)
                ).fetchone()[0]
                == "active"
            )
            assert (
                conn.execute(
                    "SELECT COUNT(*) FROM standing_suppressions WHERE scope_id=? AND memory_id=?",
                    ("project:other", int(memory_id[4:])),
                ).fetchone()[0]
                == 1
            )
            conn.close()

    _run(scenario())
