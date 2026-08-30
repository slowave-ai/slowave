"""Test: HTTP daemon ASGI response double-send guard.

The daemon historically crashed request handling with uvicorn's
``RuntimeError: Expected ASGI message 'http.response.body', but got
'http.response.start'`` whenever a client disconnected / a session was
terminated mid-stream and Starlette's error middleware tried to emit a second
``http.response.start``.

These tests verify ``_wrap_no_double_send``:
1. passes a normal request through unchanged.
2. drops a duplicate ``http.response.start`` sent on an already-started response.
3. swallows a post-start error (only ONE ``http.response.start`` reaches the client).
4. still lets a pre-start error raise (real 500s are preserved).
"""

import asyncio

import pytest

from slowave.mcp.http_server import _wrap_no_double_send


def _run(app, scope=None):
    """Drive an ASGI app and collect the messages it sends to the client."""
    sent = []

    async def send(message):
        sent.append(message)

    async def receive():
        return {"type": "http.disconnect"}

    async def drive():
        await app(scope or {"type": "http", "method": "POST", "path": "/mcp"}, receive, send)
        return sent

    return drive, sent


# --- inner app variations ------------------------------------------------


async def _normal_app(scope, receive, send):
    await send({"type": "http.response.start", "status": 200, "headers": []})
    await send({"type": "http.response.body", "body": b"ok", "more_body": False})


async def _double_start_app(scope, receive, send):
    # Misbehaving downstream app: sends start twice (what Starlette's error
    # middleware does after a mid-stream failure).
    await send({"type": "http.response.start", "status": 200, "headers": []})
    await send({"type": "http.response.start", "status": 500, "headers": []})
    await send({"type": "http.response.body", "body": b"ok", "more_body": False})


async def _raises_after_start_app(scope, receive, send):
    await send({"type": "http.response.start", "status": 200, "headers": []})
    raise RuntimeError("boom after start")


async def _raises_before_start_app(scope, receive, send):
    raise RuntimeError("boom before start")


# --- tests ---------------------------------------------------------------


def test_normal_request_passes_through():
    drive, sent = _run(_wrap_no_double_send(_normal_app))
    asyncio.run(drive())
    assert [m["type"] for m in sent] == [
        "http.response.start",
        "http.response.body",
    ]
    assert sent[0]["status"] == 200


def test_duplicate_response_start_is_dropped():
    drive, sent = _run(_wrap_no_double_send(_double_start_app))
    asyncio.run(drive())
    starts = [m for m in sent if m["type"] == "http.response.start"]
    # Only the first start reaches the client; the duplicate 500 start is dropped.
    assert len(starts) == 1
    assert starts[0]["status"] == 200


def test_post_start_error_is_swallowed():
    drive, sent = _run(_wrap_no_double_send(_raises_after_start_app))
    # Must not raise (uvicorn's double-send crash source).
    asyncio.run(drive())
    starts = [m for m in sent if m["type"] == "http.response.start"]
    assert len(starts) == 1
    assert starts[0]["status"] == 200


def test_pre_start_error_still_raises():
    drive, _ = _run(_wrap_no_double_send(_raises_before_start_app))
    with pytest.raises(RuntimeError, match="boom before start"):
        asyncio.run(drive())
