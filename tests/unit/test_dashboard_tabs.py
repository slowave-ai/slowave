from pathlib import Path

from slowave.dashboard._html import _INDEX_HTML, render_index_html

ROOT = Path(__file__).parents[2]
UI_SOURCE = ROOT / "slowave" / "dashboard" / "ui" / "src"
STATIC_INDEX = ROOT / "slowave" / "dashboard" / "static" / "index.html"


def test_python_shell_is_a_small_static_react_bootstrap() -> None:
    assert 'id="root"' in _INDEX_HTML
    assert 'data-experimental="__EXPERIMENTAL__"' in _INDEX_HTML
    assert "/assets/index.js" in _INDEX_HTML
    assert "cytoscape" not in _INDEX_HTML.lower()


def test_react_source_contains_the_supported_dashboard_surfaces() -> None:
    source = "\n".join(path.read_text() for path in UI_SOURCE.glob("*.tsx"))
    for label in (
        "Home",
        "Memory",
        "Retrieval",
        "Procedures",
        "Activity",
        "Diagnostics",
        "Graph explorer",
    ):
        assert label in source
    for endpoint in (
        "/api/home",
        "/api/status",
        "/api/schemas",
        "/api/schemas/",
        "/api/retrievals",
        "/api/activity",
        "/api/worker/runs",
        "/api/procedural-memory",
        "/api/graph/schemas",
        "/api/db/health",
        "/api/labs/rollout",
    ):
        assert endpoint in source
    assert "allowActions" in source
    assert "Exposure is not proof it was used" in source
    assert "not necessarily a proven general playbook" in source
    assert "state-summary" not in source
    assert "compact-state-chips" not in source
    assert "rounded === 0 || rounded === 100" in source
    assert "percent.toFixed(1)" in source
    assert "formatRateParts" in source
    assert "metric-rate-ratio" in source
    assert "metric-rate-percent" in source
    styles = (UI_SOURCE / "styles.css").read_text()
    assert "text-transform: uppercase" in styles
    assert "numerator === denominator" in source
    assert "numerator === 0" in source
    assert "denominator === null" in source
    assert "home-metric-card-grid" in source
    assert "activity-metric-card-grid" in source
    assert "ErrorState" in source
    assert "Memory results unavailable" in source
    assert "Retrieval details unavailable" in source
    assert "Procedure not found" in source
    assert "Activity not found" in source
    assert "No memory metrics available" in source
    assert "No retrieval summary available" in source
    assert "No procedure summary available" in source
    assert "No activity summary available" in source
    assert "Evidence quality for the retrieval metrics above" not in source
    assert "Active memories is the current library denominator" not in source
    assert 'title="Assessed memories used"' in source
    assert 'title="Demonstrated value"' in source
    assert 'title="Helpful assessments"' in source
    assert 'title="Used context"' in source
    assert "Historical feedback is incomplete" in source
    assert "Used among assessed retrieved memories" not in source
    assert "Retrievals with demonstrated value" not in source
    assert "Helpful assessments when assessed" not in source
    assert "Activities with used context" not in source
    assert "currently visible bounded subset" not in source
    assert "limit reached" in source and "limit not reached" in source
    assert ".home-metric-card-grid { grid-template-columns: repeat(3" in styles
    assert ".activity-metric-card-grid { grid-template-columns: repeat(2" in styles
    assert ".metric-card {\n  background: var(--surface-background);" in styles
    assert ".metric-card:hover," in styles
    assert "createPortal" in source
    assert "definition-tooltip-popover" in source
    assert "position: fixed" in styles
    assert "effectiveness-card-title-text" in source


def test_built_static_index_is_present_and_has_hashed_assets() -> None:
    assert STATIC_INDEX.is_file()
    html = STATIC_INDEX.read_text()
    assert "/assets/index-" in html
    assert "__EXPERIMENTAL__" in html


def test_experimental_flag_is_encoded_in_fallback_shell() -> None:
    assert 'data-experimental="false"' in render_index_html()
    assert 'data-experimental="true"' in render_index_html(experimental=True)
