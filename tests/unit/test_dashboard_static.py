from __future__ import annotations

import threading
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from slowave.dashboard.app import _make_handler


def _server(tmp_path: Path, *, allow_actions: bool = True, experimental: bool = False):
    handler = _make_handler(
        db_path=str(tmp_path / "dashboard.sqlite3"),
        refresh_ms=1234,
        allow_actions=allow_actions,
        experimental_dashboard=experimental,
    )
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def test_static_shell_and_assets_are_served_with_safe_paths(tmp_path: Path) -> None:
    server = _server(tmp_path, allow_actions=False, experimental=True)
    try:
        base = f"http://127.0.0.1:{server.server_port}"
        with urlopen(base + "/") as response:
            html = response.read().decode()
            assert response.headers["Content-Type"].startswith("text/html")
        assert 'data-experimental="true"' in html
        assert 'data-refresh-ms="1234"' in html
        assert 'data-allow-actions="false"' in html

        with urlopen(base + "/img/slowave-logo-small.jpeg") as response:
            assert response.headers["Content-Type"].startswith("image/jpeg")
            assert response.read(2) == b"\xff\xd8"

        asset = next(
            line.split('src="', 1)[1].split('"', 1)[0]
            for line in html.splitlines()
            if "assets/index-" in line and "src=" in line
        )
        with urlopen(base + asset) as response:
            content = response.read()
            assert response.headers["Content-Type"].startswith(
                ("application/javascript", "text/javascript")
            )
            assert b"Slowave" in content

        for path in ("/assets/%2e%2e/%2e%2e/pyproject.toml",):
            try:
                urlopen(base + path)
            except HTTPError as error:
                assert error.code == 404
            else:
                raise AssertionError(f"path traversal unexpectedly served: {path}")
    finally:
        server.shutdown()
        server.server_close()


def test_static_shell_preserves_action_gate(tmp_path: Path) -> None:
    server = _server(tmp_path, allow_actions=False)
    try:
        request = Request(
            f"http://127.0.0.1:{server.server_port}/api/schemas/1/forget",
            method="POST",
        )
        try:
            urlopen(request)
        except HTTPError as error:
            assert error.code == 403
            assert b"mutating actions disabled" in error.read()
        else:
            raise AssertionError("disabled action unexpectedly succeeded")
    finally:
        server.shutdown()
        server.server_close()


def test_canonical_product_routes_refresh_to_react_shell(tmp_path: Path) -> None:
    server = _server(tmp_path, experimental=True)
    try:
        base = f"http://127.0.0.1:{server.server_port}"
        for path in (
            "/memory",
            "/memory/sch_1",
            "/retrieval",
            "/retrieval/ctx_1",
            "/procedures",
            "/procedures/proc_1",
            "/activity",
            "/activity/sess_1",
            "/diagnostics",
            "/diagnostics/labs",
        ):
            with urlopen(base + path) as response:
                assert response.headers["Content-Type"].startswith("text/html")
                assert b'id="root"' in response.read()

        for path in ("/unknown", "/api/not-real", "/missing.js"):
            try:
                urlopen(base + path)
            except HTTPError as error:
                assert error.code == 404
                assert not error.read().lstrip().startswith(b"<!doctype html")
            else:
                raise AssertionError(f"unknown path unexpectedly served the React shell: {path}")
    finally:
        server.shutdown()
        server.server_close()
