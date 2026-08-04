"""Regression tests for the dashboard's visible navigation."""

from slowave.dashboard._html import _INDEX_HTML


def test_procedures_tab_is_temporarily_hidden() -> None:
    """Keep the unfinished Procedures view out of the public dashboard UI."""
    assert 'data-tab="procedures"' not in _INDEX_HTML
