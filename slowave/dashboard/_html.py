"""Compatibility shell for the packaged React dashboard.

The browser application is authored in ``dashboard/ui`` and built into
``dashboard/static``. Keeping this tiny renderer preserves the import used by
older integrations and gives the HTTP handler a useful fallback if a source
checkout has not built the frontend yet.
"""

_HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Slowave Dashboard</title>
</head>
<body>
<div id="root" data-refresh-ms="__REFRESH_MS__" data-allow-actions="__ALLOW_ACTIONS__" data-version="__SLOWAVE_VERSION__" data-lifecycle-version="__LIFECYCLE_VERSION__"></div>
<script type="module" src="/assets/index.js"></script>
</body>
</html>"""

_INDEX_HTML = _HTML_TEMPLATE


def render_index_html() -> str:
    """Render the minimal dashboard bootstrap shell."""
    return _HTML_TEMPLATE
