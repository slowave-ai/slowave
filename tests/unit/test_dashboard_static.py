from __future__ import annotations

import threading
import zipfile
from http.server import ThreadingHTTPServer
from pathlib import Path
from shutil import copy2, copytree
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from setuptools.build_meta import build_wheel

from slowave.dashboard.app import _make_handler


def _server(tmp_path: Path, *, allow_actions: bool = True):
    handler = _make_handler(
        db_path=str(tmp_path / "dashboard.sqlite3"),
        refresh_ms=1234,
        allow_actions=allow_actions,
    )
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def test_static_shell_and_assets_are_served_with_safe_paths(tmp_path: Path) -> None:
    server = _server(tmp_path, allow_actions=False)
    try:
        base = f"http://127.0.0.1:{server.server_port}"
        with urlopen(base + "/") as response:
            html = response.read().decode()
            assert response.headers["Content-Type"].startswith("text/html")
        assert 'data-refresh-ms="1234"' in html
        assert 'data-allow-actions="false"' in html

        with urlopen(base + "/img/slowave-logo-small.jpeg") as response:
            assert response.headers["Content-Type"].startswith("image/jpeg")
            assert response.read(2) == b"\xff\xd8"

        with urlopen(base + "/img/slowave-logo-text-small.jpeg") as response:
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


def test_wheel_contains_dashboard_shell_and_referenced_assets(tmp_path: Path, monkeypatch) -> None:
    """A released wheel must be able to serve its Vite dashboard on its own."""
    source_root = Path(__file__).parents[2]
    package_root = tmp_path / "package"
    package_root.mkdir()
    for name in ("pyproject.toml", "README.md", "LICENSE"):
        copy2(source_root / name, package_root / name)
    copytree(source_root / "slowave", package_root / "slowave")

    wheel_dir = tmp_path / "wheel"
    wheel_dir.mkdir()
    monkeypatch.chdir(package_root)
    wheel_name = build_wheel(str(wheel_dir))
    with zipfile.ZipFile(wheel_dir / wheel_name) as wheel:
        files = set(wheel.namelist())
        index = wheel.read("slowave/dashboard/static/index.html").decode()

    assert "slowave/dashboard/static/index.html" in files
    assert "slowave/dashboard/static/img/slowave-logo-text-small.jpeg" in files
    assets = [value for value in index.split('"') if value.startswith(("/assets/", "/img/"))]
    assert assets
    for asset in assets:
        assert f"slowave/dashboard/static/{asset.lstrip('/')}" in files


def test_canonical_product_routes_refresh_to_react_shell(tmp_path: Path) -> None:
    server = _server(tmp_path)
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
            "/graph",
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
