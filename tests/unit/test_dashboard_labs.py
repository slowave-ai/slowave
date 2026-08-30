from __future__ import annotations

import json
import os
import tempfile
import threading
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import urlopen

from slowave.core.config import SlowaveConfig
from slowave.core.engine import SlowaveEngine
from slowave.dashboard._html import render_index_html
from slowave.dashboard.app import _labs_rollout_payload, _make_handler
from slowave.lifecycle import LIFECYCLE_VERSION


def test_labs_ui_is_disabled_by_default() -> None:
    stable = render_index_html()
    experimental = render_index_html(experimental=True)

    assert 'data-experimental="false"' in stable
    assert 'data-experimental="true"' in experimental
    assert "__EXPERIMENTAL__" not in stable


def test_react_ui_keeps_labs_gated_but_exposes_the_experimental_tab() -> None:
    source_dir = os.path.join(
        os.path.dirname(__file__), "..", "..", "slowave", "dashboard", "ui", "src"
    )
    app = "\n".join(
        open(os.path.join(source_dir, name), encoding="utf-8").read()
        for name in ("components.tsx", "pages.tsx")
    )
    assert "experimental && (" in app
    assert 'to="/diagnostics/labs"' in app
    assert "Experimental — not a product metric" in app


def test_labs_payload_is_explicitly_current_lifecycle_scoped() -> None:
    handle = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    handle.close()
    engine = SlowaveEngine(SlowaveConfig(db_path=handle.name, dim=8, disable_encoder=True))
    try:
        payload = _labs_rollout_payload(handle.name)
        assert payload["status"] == "experimental"
        assert payload["cohort"]["lifecycle_version"] == LIFECYCLE_VERSION
        assert payload["cohort"]["sessions"] == 0
        assert payload["provenance"]["eligible_events"] == 0
        assert payload["retrieval"]["procedure_exposures"] == 0
        assert payload["retrieval"]["no_match_retrievals"] == 0
        assert payload["retrieval"]["response_chars"] == {
            "observed": 0,
            "total": 0,
            "average": None,
        }
        assert payload["retrieval"]["estimated_tokens"] == {
            "observed": 0,
            "total": 0,
            "average": None,
        }
        assert payload["retrieval"]["latency"] == "not_persisted"
        assert payload["retrieval"]["memory_feedback"] == {
            "irrelevant": 0,
            "stale": 0,
            "contradicted": 0,
            "accepted": 0,
        }
        assert payload["truth_maintenance"]["schemas_by_status"] == {}
        assert payload["truth_maintenance"]["v9_feedback"]["stale"] == 0
        assert payload["truth_maintenance"]["sample"] == []
    finally:
        engine.close()
        for suffix in ("", "-wal", "-shm"):
            if os.path.exists(handle.name + suffix):
                os.remove(handle.name + suffix)


def test_labs_payload_reports_stale_reasons_and_replacement_lineage() -> None:
    handle = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    handle.close()
    engine = SlowaveEngine(SlowaveConfig(db_path=handle.name, dim=8, disable_encoder=True))
    try:
        session_id = engine.session_start(
            agent="dashboard-test", scope="project:test", goal="truth"
        )
        old_id = engine.remember(content="old claim", type="fact")
        new_id = engine.remember(content="replacement claim", type="fact")
        retrieval_id = "ctx_labs_truth"
        engine.record_retrieval(
            retrieval_id=retrieval_id,
            session_id=session_id,
            response={"schemas": [{"id": f"sch_{old_id}"}]},
        )
        engine.feedback(
            retrieval_id=retrieval_id,
            memory_feedback=[
                {
                    "memory_id": f"sch_{old_id}",
                    "assessment": "stale",
                    "stale_reason": "superseded",
                    "replacement_memory_id": f"sch_{new_id}",
                    "reason": "Replacement claim is current.",
                }
            ],
            coverage="complete",
        )
        payload = _labs_rollout_payload(handle.name)
        truth = payload["truth_maintenance"]
        assert truth["schemas_by_status"]["stale"] == 1
        assert truth["v9_feedback"]["superseded"] == 1
        assert truth["v9_feedback"]["with_replacement"] == 1
        assert truth["sample"][0]["stale_reason"] == "superseded"
        assert truth["sample"][0]["replacement_memory_id"] == f"sch_{new_id}"
    finally:
        engine.close()
        for suffix in ("", "-wal", "-shm"):
            if os.path.exists(handle.name + suffix):
                os.remove(handle.name + suffix)


def test_labs_payload_separates_closure_states_and_hook_usefulness() -> None:
    handle = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    handle.close()
    engine = SlowaveEngine(SlowaveConfig(db_path=handle.name, dim=8, disable_encoder=True))
    try:
        complete_session = engine.session_start(
            agent="dashboard-test", scope="project:test", goal="complete"
        )
        from slowave import ops

        ops.commit(
            engine,
            session_id=complete_session,
            outcome="success",
            final_goal="complete",
            outcome_summary="Complete feedback closure.",
            enforce_feedback=True,
        )
        incomplete_session = engine.session_start(
            agent="dashboard-test", scope="project:test", goal="incomplete"
        )
        ops.commit(
            engine,
            session_id=incomplete_session,
            outcome="partial",
            final_goal="incomplete",
            outcome_summary="Closed without feedback enforcement.",
        )
        active_session = engine.session_start(
            agent="dashboard-test", scope="project:test", goal="active"
        )
        for retrieval_id, query, procedure_id, use, effect in (
            ("ctx_hook", "<hook_prompt> SLOWAVE MANDATORY:", "proc_hook", "not_used", "unknown"),
            ("ctx_task", "repair the production configuration", "proc_task", "used", "helped"),
        ):
            engine.record_retrieval(
                retrieval_id=retrieval_id,
                session_id=active_session,
                scope_id="project:test",
                query=query,
                response={
                    "procedure_ids": [procedure_id],
                    "procedures": [{"id": procedure_id}],
                },
            )
            engine._feedback.feedback_events.record(
                retrieval_id=retrieval_id,
                procedure_feedback=[
                    {
                        "procedure_id": procedure_id,
                        "use": use,
                        "effect": effect,
                        **(
                            {"contribution": "The repair order transferred."}
                            if use == "used"
                            else {}
                        ),
                    }
                ],
                coverage="complete",
                mutation_mode="active",
            )

        payload = _labs_rollout_payload(handle.name)
        cohort = payload["cohort"]
        retrieval = payload["retrieval"]
        feedback = retrieval["procedure_feedback"]

        assert cohort["sessions"] == 3
        assert cohort["completed_sessions"] == 2
        assert cohort["feedback_complete"] == 1
        assert cohort["feedback_incomplete"] == 1
        assert cohort["active_pending"] == 1
        assert retrieval["procedure_exposures"] == 2
        assert retrieval["hook_procedure_exposures"] == 1
        assert retrieval["non_hook_procedure_exposures"] == 1
        assert feedback["used"] == 1
        assert feedback["not_used"] == 1
        assert feedback["non_hook_used"] == 1
        assert feedback["non_hook_not_used"] == 0
        assert feedback["non_hook_helped"] == 1
        assert feedback["non_hook_unknown"] == 0
    finally:
        engine.close()
        for suffix in ("", "-wal", "-shm"):
            if os.path.exists(handle.name + suffix):
                os.remove(handle.name + suffix)


def test_labs_api_is_not_exposed_without_flag(tmp_path) -> None:
    engine = SlowaveEngine(
        SlowaveConfig(db_path=str(tmp_path / "labs.db"), dim=8, disable_encoder=True)
    )
    engine.close()

    def request(experimental: bool) -> tuple[int, dict[str, object]]:
        handler = _make_handler(
            db_path=str(tmp_path / "labs.db"),
            refresh_ms=2000,
            allow_actions=False,
            experimental_dashboard=experimental,
        )
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            url = f"http://127.0.0.1:{server.server_port}/api/labs/rollout"
            try:
                response = urlopen(url)
                return response.status, json.load(response)
            except HTTPError as error:
                return error.code, json.load(error)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    disabled_status, disabled_payload = request(False)
    enabled_status, enabled_payload = request(True)

    assert disabled_status == 404
    assert disabled_payload["error"] == "not found"
    assert enabled_status == 200
    assert enabled_payload["status"] == "experimental"
