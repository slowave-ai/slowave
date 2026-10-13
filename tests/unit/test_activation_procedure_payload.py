import pytest

from slowave.mcp.activation_catalog import CatalogPreparationError, FrozenActivationCatalog
from slowave.mcp.tools import _compact_activation_procedure


def test_activation_procedure_includes_complete_method_and_context_not_reuse_history():
    source = dict(
        id="proc_test",
        goal="Diagnose stale assets",
        summary="Verify served bundle",
        context={"environment": "local"},
        steps=[{"summary": "Compare bundle hashes"}, {"summary": "Check serving process"}],
        caveats=["Do not restart production."],
        outcome="success",
        outcome_summary="Served expected bundle",
        evidence={"used": 3},
        contributions=[{"contribution": "Earlier use"}],
    )
    delivered = _compact_activation_procedure(source, "project:test")
    assert delivered["steps"] == source["steps"]
    assert delivered["context"] == source["context"]
    assert delivered["caveats"] == source["caveats"]
    assert "contributions" not in delivered and "evidence" not in delivered
    catalog = FrozenActivationCatalog.prepare(
        retrieval_id="ctx_test",
        session_id="sess_test",
        scope="project:test",
        memories=[],
        procedures=[delivered],
        contribution_spans={},
        catalog_truncated=False,
        first_page_metadata={},
    )
    page, _ = catalog.page()
    assert page["procedures"][0]["steps"] == source["steps"]
    delivered["steps"][0]["summary"] = "x" * 20000
    with pytest.raises(CatalogPreparationError):
        FrozenActivationCatalog.prepare(
            retrieval_id="ctx_test",
            session_id="sess_test",
            scope="project:test",
            memories=[],
            procedures=[delivered],
            contribution_spans={},
            catalog_truncated=False,
            first_page_metadata={},
        )
