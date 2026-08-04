"""Tests for the dashboard's graph payload endpoint (2026-07-23 dangling-edge fix).

_schema_graph_payload() must never return an edge whose source/target isn't
also present in the returned nodes list. Cytoscape.js throws when constructed
with an edge referencing a nonexistent node id, and drawGraph() (_js.py) has
no try/catch around that call -- so a single dangling edge blanks the entire
graph, not just that edge. schema_relations edges already required both ends
to be in the filtered node set; schema_coactivation edges only required one
end, since co-activation isn't scope-restricted and can link schemas that a
salience/scope/status filter excludes on one side.
"""

from __future__ import annotations

import os
import tempfile

from slowave.core.config import SlowaveConfig
from slowave.core.engine import SlowaveEngine
from slowave.dashboard.app import _schema_graph_payload


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


def _no_dangling_edges(payload: dict) -> bool:
    node_ids = {n["id"] for n in payload["nodes"]}
    return all(e["source"] in node_ids and e["target"] in node_ids for e in payload["edges"])


def test_coactivation_edge_excluded_when_salience_filter_hides_one_endpoint():
    eng, path = _tmp_engine()
    try:
        low_id = eng.schemas.create(
            content_text="low salience schema",
            facets={},
            tags=[],
            embedding=None,
            salience=0.1,
            dedupe=False,
        )
        high_id = eng.schemas.create(
            content_text="high salience schema",
            facets={},
            tags=[],
            embedding=None,
            salience=5.0,
            dedupe=False,
        )
        eng.schemas.upsert_coactivation(low_id, high_id, now_ts=1_000_000)
        eng.close()

        # min_salience excludes low_id but not high_id -- the co-activation
        # edge between them must not appear at all now.
        payload = _schema_graph_payload(path, {"min_salience": ["1.0"]})
        node_ids = {n["schema_id"] for n in payload["nodes"]}
        assert low_id not in node_ids
        assert high_id in node_ids
        assert _no_dangling_edges(payload)
        assert not any(e["relation"] == "coactivated_with" for e in payload["edges"])
    finally:
        _cleanup(path)


def test_coactivation_edge_excluded_when_scope_filter_hides_one_endpoint():
    eng, path = _tmp_engine()
    try:
        alpha_id = eng.schemas.create(
            content_text="alpha schema",
            facets={},
            tags=[],
            embedding=None,
            scope_id="project:alpha",
            dedupe=False,
        )
        beta_id = eng.schemas.create(
            content_text="beta schema",
            facets={},
            tags=[],
            embedding=None,
            scope_id="project:beta",
            dedupe=False,
        )
        eng.schemas.upsert_coactivation(alpha_id, beta_id, now_ts=1_000_000)
        eng.close()

        payload = _schema_graph_payload(path, {"scope": ["project:alpha"]})
        node_ids = {n["schema_id"] for n in payload["nodes"]}
        assert alpha_id in node_ids
        assert beta_id not in node_ids
        assert _no_dangling_edges(payload)
        assert not any(e["relation"] == "coactivated_with" for e in payload["edges"])
    finally:
        _cleanup(path)


def test_coactivation_edge_included_when_both_endpoints_visible():
    eng, path = _tmp_engine()
    try:
        id_a = eng.schemas.create(
            content_text="A", facets={}, tags=[], embedding=None, dedupe=False
        )
        id_b = eng.schemas.create(
            content_text="B", facets={}, tags=[], embedding=None, dedupe=False
        )
        eng.schemas.upsert_coactivation(id_a, id_b, now_ts=1_000_000)
        eng.close()

        payload = _schema_graph_payload(path, {})
        assert _no_dangling_edges(payload)
        coact_edges = [e for e in payload["edges"] if e["relation"] == "coactivated_with"]
        assert len(coact_edges) == 1
    finally:
        _cleanup(path)
