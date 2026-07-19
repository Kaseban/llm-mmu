"""mmu proxy server (M0: transparent passthrough with observation).

Stack: starlette + uvicorn + httpx (per plan; no FastAPI).

Design invariants:
- Client branch re-emits upstream bytes verbatim; observer-branch parser errors
  are logged, never propagated to the client.
- Unknown paths are forwarded untouched to the Anthropic upstream by default
  (so /v1/models, count_tokens etc. keep working).
- API keys are forwarded in headers, hashed (sha256) for namespacing, and never
  written to the store or logs.
"""

from __future__ import annotations

import json
import logging
import time
from contextlib import asynccontextmanager
from typing import AsyncIterator

import httpx
import uvicorn
from starlette.applications import Starlette
from starlette.background import BackgroundTask
from starlette.requests import Request
from starlette.responses import Response, StreamingResponse
from starlette.routing import Route

from mmu.config import Config
from mmu.paging.engine import PageStats, PagingEngine
from mmu.proxy.adapters import Usage, adapter_for_path
from mmu.proxy.sse import SSEParser
from mmu.sessions import resolve_session, system_hash
from mmu.store.db import Store

log = logging.getLogger("mmu")

HOP_BY_HOP = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailers", "transfer-encoding", "upgrade", "host", "content-length",
}


def _forward_headers(headers) -> dict[str, str]:
    return {k: v for k, v in headers.items() if k.lower() not in HOP_BY_HOP}


class ProxyApp:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.store = Store(cfg.store.path)
        self.engine = PagingEngine(self.store, cfg.paging)
        self.http = httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=15.0))

    def upstream_for(self, path: str) -> str:
        if path.startswith("/v1/chat/completions") or path.startswith("/v1/responses"):
            return self.cfg.proxy.openai_upstream
        return self.cfg.proxy.anthropic_upstream

    async def shutdown(self) -> None:
        await self.http.aclose()
        self.store.close()

    # -- handler --------------------------------------------------------------

    async def handle(self, request: Request) -> Response:
        t0 = time.monotonic()
        path = request.url.path
        adapter = adapter_for_path(path)
        raw = await request.body()

        session_id = None
        parsed = None
        pstats = PageStats()
        if adapter and raw:
            try:
                body = json.loads(raw)
                parsed = adapter.parse_request(body)
                auth = request.headers.get("x-api-key") or request.headers.get(
                    "authorization"
                )
                sys_h = system_hash(parsed.system, parsed.tools)
                session_id = resolve_session(
                    self.store, parsed.messages, auth, adapter.provider, sys_h
                )
                if self.cfg.proxy.mode == "paging":
                    try:  # paging must fail open to plain forwarding
                        new_body, pstats = self.engine.rewrite(session_id, body)
                        if pstats.evicted or pstats.faults:
                            raw = json.dumps(new_body, ensure_ascii=False).encode()
                    except Exception:
                        pstats = PageStats()
                        log.exception("paging: rewrite failed (forwarding original)")
            except Exception:
                log.exception("observer: request parse failed (forwarding anyway)")

        url = self.upstream_for(path).rstrip("/") + path
        if request.url.query:
            url += "?" + request.url.query

        upstream_req = self.http.build_request(
            request.method,
            url,
            headers=_forward_headers(request.headers),
            content=raw or None,
        )
        t_up = time.monotonic()
        up = await self.http.send(upstream_req, stream=True)

        resp_headers = _forward_headers(up.headers)
        is_sse = "text/event-stream" in up.headers.get("content-type", "")

        if is_sse:
            return StreamingResponse(
                self._tee(up, adapter, parsed, session_id, t0, t_up, pstats),
                status_code=up.status_code,
                headers=resp_headers,
                background=BackgroundTask(up.aclose),
            )

        body_bytes = await up.aread()
        await up.aclose()
        usage = Usage()
        if adapter:
            try:
                usage = adapter.usage_from_json(json.loads(body_bytes))
            except Exception:
                pass
        self._record(
            adapter, parsed, session_id, up.status_code, usage, False, t0, t_up, pstats
        )
        return Response(body_bytes, status_code=up.status_code, headers=resp_headers)

    async def _tee(
        self, up: httpx.Response, adapter, parsed, session_id, t0, t_up, pstats
    ) -> AsyncIterator[bytes]:
        parser = SSEParser()
        usage = Usage()
        try:
            async for chunk in up.aiter_raw():
                yield chunk  # client branch: verbatim bytes
                if adapter:
                    try:  # observer branch: never break the stream
                        for ev in parser.feed(chunk):
                            adapter.usage_from_event(ev, usage)
                    except Exception:
                        log.exception("observer: SSE parse failed")
        finally:
            self._record(
                adapter, parsed, session_id, up.status_code, usage, True, t0, t_up, pstats
            )

    def _record(
        self, adapter, parsed, session_id, status, usage, streamed, t0, t_up, pstats=None
    ):
        try:
            total_ms = int((time.monotonic() - t0) * 1000)
            upstream_ms = int((time.monotonic() - t_up) * 1000)
            self.store.record_request(
                session_id=session_id,
                provider=adapter.provider if adapter else "unknown",
                model=parsed.model if parsed else None,
                streamed=1 if streamed else 0,
                status_code=status,
                tokens_in=usage.tokens_in,
                tokens_out=usage.tokens_out,
                cache_read=usage.cache_read,
                cache_write=usage.cache_write,
                tokens_saved=pstats.tokens_saved if pstats else 0,
                faults=pstats.faults if pstats else 0,
                latency_ms_upstream=upstream_ms,
                latency_ms_overhead=max(0, total_ms - upstream_ms),
            )
        except Exception:
            log.exception("observer: record failed")


def make_app(cfg: Config) -> Starlette:
    proxy = ProxyApp(cfg)

    @asynccontextmanager
    async def lifespan(app: Starlette):
        try:
            yield
        finally:
            await proxy.shutdown()

    app = Starlette(
        routes=[
            Route(
                "/{tail:path}",
                proxy.handle,
                methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD"],
            )
        ],
        lifespan=lifespan,
    )
    app.state.proxy = proxy
    return app


def run(cfg: Config) -> None:
    if cfg.proxy.listen_host not in ("127.0.0.1", "localhost", "::1") and not (
        cfg.proxy.allow_non_loopback
    ):
        raise SystemExit(
            "mmu: refusing to listen on a non-loopback address without "
            "proxy.allow_non_loopback = true (the proxy forwards API keys)"
        )
    uvicorn.run(
        make_app(cfg),
        host=cfg.proxy.listen_host,
        port=cfg.proxy.listen_port,
        log_level="warning",
    )
