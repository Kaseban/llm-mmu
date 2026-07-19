"""End-to-end paging tests: proxy in mode="paging" against a capturing fake
upstream. Verifies the three core invariants of M1:

  1. Cold tool results are replaced by stubs *upstream only* — the client
     response bytes are untouched and tokens_saved lands in accounting.
  2. A recall marker in assistant history faults the page back in (upstream
     sees the full text again) and the fault is counted.
  3. Paging failures / small sessions leave the request byte-identical.
"""

from __future__ import annotations

import json
import socket
import threading
import time

import httpx
import pytest
import uvicorn
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route

from mmu.config import Config
from mmu.paging.engine import page_id_for
from mmu.proxy.app import make_app

ANTHROPIC_JSON = {
    "id": "msg_01",
    "type": "message",
    "role": "assistant",
    "content": [{"type": "text", "text": "ok"}],
    "usage": {"input_tokens": 100, "output_tokens": 5},
}

CAPTURED: list[dict] = []


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


async def _messages(request):
    CAPTURED.append(await request.json())
    return JSONResponse(ANTHROPIC_JSON)


@pytest.fixture(scope="module")
def upstream():
    port = _free_port()
    app = Starlette(routes=[Route("/v1/messages", _messages, methods=["POST"])])
    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    )
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
    c.proxy.mode = "paging"
    c.store.path = tmp_path / "store"
    c.paging.budget_tokens = 2000
    c.paging.low_watermark = 0.5
    c.paging.min_page_tokens = 50
    c.paging.pin_recent_turns = 2
    return c


def _client(app):
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://mmu.local"
    )


TOOL_TEXT = "".join(
    f"line {i} of synthetic build output with some variety\n" for i in range(200)
)


def _history_with_big_tool_result():
    return [
        {"role": "user", "content": "run the build"},
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "tu_1", "name": "bash", "input": {}}
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "tu_1", "content": TOOL_TEXT}
            ],
        },
        {"role": "assistant", "content": [{"type": "text", "text": "build done"}]},
        {"role": "user", "content": "now summarize the failures"},
        {"role": "assistant", "content": [{"type": "text", "text": "working on it"}]},
        {"role": "user", "content": "go on"},
    ]


def _req(messages):
    return {"model": "claude-fable-5", "max_tokens": 100, "messages": messages}


@pytest.mark.asyncio
async def test_eviction_upstream_only_and_accounting(cfg):
    CAPTURED.clear()
    app = make_app(cfg)
    async with _client(app) as client:
        r = await client.post(
            "/v1/messages",
            json=_req(_history_with_big_tool_result()),
            headers={"x-api-key": "sk-t"},
        )
        assert r.status_code == 200
        assert r.json() == ANTHROPIC_JSON  # client bytes untouched
    sent = CAPTURED[-1]["messages"]
    tool_block = sent[2]["content"][0]
    assert tool_block["type"] == "tool_result"
    assert "mmu: evicted tool_result page" in tool_block["content"]
    assert TOOL_TEXT not in tool_block["content"]  # cold page not sent upstream
    # pinned tail untouched
    assert sent[-1] == {"role": "user", "content": "go on"}
    proxy = app.state.proxy
    s = proxy.store.stats()
    assert s["tokens_saved"] > 500 and s["faults"] == 0
    await proxy.shutdown()


@pytest.mark.asyncio
async def test_recall_faults_page_back_in(cfg):
    CAPTURED.clear()
    app = make_app(cfg)
    pid = page_id_for(TOOL_TEXT)
    history = _history_with_big_tool_result()
    async with _client(app) as client:
        await client.post(
            "/v1/messages", json=_req(history), headers={"x-api-key": "sk-t"}
        )
        assert "mmu: evicted" in json.dumps(CAPTURED[-1])
        history = history + [
            {
                "role": "assistant",
                "content": [{"type": "text", "text": f"mmu recall {pid}"}],
            },
            {"role": "user", "content": "continue"},
        ]
        await client.post(
            "/v1/messages", json=_req(history), headers={"x-api-key": "sk-t"}
        )
    sent = CAPTURED[-1]["messages"]
    # faulted back: full text goes upstream again
    assert sent[2]["content"][0]["content"] == TOOL_TEXT
    proxy = app.state.proxy
    s = proxy.store.stats()
    assert s["faults"] == 1
    row = proxy.store.conn.execute(
        "SELECT state, fault_count FROM pages WHERE page_id=?", (pid,)
    ).fetchone()
    assert row["state"] == "resident" and row["fault_count"] == 1
    await proxy.shutdown()


@pytest.mark.asyncio
async def test_small_session_byte_identical(cfg):
    CAPTURED.clear()
    app = make_app(cfg)
    req = _req([{"role": "user", "content": "hi"}])
    async with _client(app) as client:
        r = await client.post("/v1/messages", json=req, headers={"x-api-key": "sk-t"})
        assert r.status_code == 200
    assert CAPTURED[-1] == req  # under budget: untouched
    proxy = app.state.proxy
    assert proxy.store.stats()["tokens_saved"] == 0
    await proxy.shutdown()
