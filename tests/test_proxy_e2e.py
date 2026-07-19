"""End-to-end passthrough tests: real fake upstream over a socket, proxy app
driven via httpx ASGITransport (the proxy's own outbound calls are real HTTP).
"""

from __future__ import annotations

import asyncio
import socket
import threading
import time

import httpx
import pytest
import uvicorn
from starlette.applications import Starlette
from starlette.responses import JSONResponse, StreamingResponse
from starlette.routing import Route

from mmu.config import Config
from mmu.proxy.app import make_app

ANTHROPIC_JSON = {
    "id": "msg_01",
    "type": "message",
    "role": "assistant",
    "content": [{"type": "text", "text": "hi"}],
    "usage": {"input_tokens": 120, "output_tokens": 7, "cache_read_input_tokens": 90},
}

SSE_BODY = (
    b'event: message_start\n'
    b'data: {"type":"message_start","message":{"usage":{"input_tokens":200,'
    b'"cache_read_input_tokens":150,"cache_creation_input_tokens":10}}}\n\n'
    b'event: content_block_delta\n'
    b'data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"hello"}}\n\n'
    b'event: message_delta\n'
    b'data: {"type":"message_delta","usage":{"output_tokens":42}}\n\n'
    b'event: message_stop\ndata: {"type":"message_stop"}\n\n'
)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


async def _messages(request):
    body = await request.json()
    if body.get("stream"):
        async def gen():
            for i in range(0, len(SSE_BODY), 40):  # deliberately awkward chunking
                yield SSE_BODY[i : i + 40]
        return StreamingResponse(gen(), media_type="text/event-stream")
    return JSONResponse(ANTHROPIC_JSON)


async def _models(request):
    return JSONResponse({"data": [{"id": "claude-fable-5"}]})


@pytest.fixture(scope="module")
def upstream():
    port = _free_port()
    app = Starlette(routes=[
        Route("/v1/messages", _messages, methods=["POST"]),
        Route("/v1/models", _models, methods=["GET"]),
    ])
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 5
    while not server.started and time.time() < deadline:
        time.sleep(0.02)
    assert server.started
    yield f"http://127.0.0.1:{port}"
    server.should_exit = True
    thread.join(timeout=5)


@pytest.fixture()
def cfg(tmp_path, upstream):
    c = Config()
    c.proxy.anthropic_upstream = upstream
    c.proxy.openai_upstream = upstream
    c.store.path = tmp_path / "store"
    return c


def _client(app):
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://mmu.local"
    )


REQ = {
    "model": "claude-fable-5",
    "max_tokens": 100,
    "messages": [{"role": "user", "content": "hi"}],
}


@pytest.mark.asyncio
async def test_non_streaming_passthrough(cfg):
    app = make_app(cfg)
    async with _client(app) as client:
        r = await client.post("/v1/messages", json=REQ, headers={"x-api-key": "sk-t"})
        assert r.status_code == 200
        assert r.json() == ANTHROPIC_JSON
    proxy = app.state.proxy
    s = proxy.store.stats()
    assert s["requests"] == 1
    assert s["tokens_in"] == 120 and s["tokens_out"] == 7 and s["cache_read"] == 90
    await proxy.shutdown()


@pytest.mark.asyncio
async def test_streaming_bytes_verbatim_and_usage(cfg):
    app = make_app(cfg)
    async with _client(app) as client:
        body = b""
        async with client.stream(
            "POST", "/v1/messages", json={**REQ, "stream": True},
            headers={"x-api-key": "sk-t"},
        ) as r:
            assert r.status_code == 200
            assert "text/event-stream" in r.headers["content-type"]
            async for chunk in r.aiter_bytes():
                body += chunk
        assert body == SSE_BODY  # byte-transparent invariant
    proxy = app.state.proxy
    await asyncio.sleep(0.05)  # let finally-branch record land
    s = proxy.store.stats()
    assert s["tokens_in"] == 200 and s["tokens_out"] == 42 and s["cache_read"] == 150
    await proxy.shutdown()


@pytest.mark.asyncio
async def test_unknown_path_forwarded(cfg):
    app = make_app(cfg)
    async with _client(app) as client:
        r = await client.get("/v1/models")
        assert r.status_code == 200
        assert r.json()["data"][0]["id"] == "claude-fable-5"
    await app.state.proxy.shutdown()


@pytest.mark.asyncio
async def test_session_continuity_through_proxy(cfg):
    app = make_app(cfg)
    async with _client(app) as client:
        await client.post("/v1/messages", json=REQ, headers={"x-api-key": "sk-t"})
        followup = {
            **REQ,
            "messages": REQ["messages"]
            + [
                {"role": "assistant", "content": [{"type": "text", "text": "hi"}]},
                {"role": "user", "content": "again"},
            ],
        }
        await client.post("/v1/messages", json=followup, headers={"x-api-key": "sk-t"})
    proxy = app.state.proxy
    s = proxy.store.stats()
    assert s["requests"] == 2 and s["sessions"] == 1
    await proxy.shutdown()


@pytest.mark.asyncio
async def test_malformed_body_still_forwarded(cfg):
    app = make_app(cfg)
    async with _client(app) as client:
        r = await client.post(
            "/v1/messages", content=b"not json{{",
            headers={"content-type": "application/json", "x-api-key": "sk-t"},
        )
        # Upstream chokes (500 from fake), but the proxy itself must not error out
        # before forwarding; any status is acceptable as long as we got a response.
        assert r.status_code in (200, 400, 422, 500)
    await app.state.proxy.shutdown()
