"""E2E benchmark: passthrough vs paging over real HTTP.

Simulates a Claude-Code-style agent loop: the client keeps full history and
resends it every turn; each turn adds a tool_use + large tool_result + short
user message. A fake Anthropic upstream measures what actually arrives.

Usage: .venv/bin/python bench/e2e_bench.py [--turns 30] [--tool-tokens 2000]
"""

from __future__ import annotations

import argparse
import json
import socket
import statistics
import sys
import tempfile
import threading
import time
from pathlib import Path

import httpx
import uvicorn
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mmu.config import Config  # noqa: E402
from mmu.proxy.app import make_app  # noqa: E402
from mmu.tokens import estimate_request_tokens  # noqa: E402


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Upstream:
    """Fake Anthropic endpoint that records the token size of each request."""

    def __init__(self):
        self.request_tokens: list[int] = []

    async def messages(self, request):
        body = await request.json()
        est = estimate_request_tokens(body.get("messages", []), body.get("system"))
        self.request_tokens.append(est)
        return JSONResponse(
            {
                "id": f"msg_{len(self.request_tokens)}",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": "ack"}],
                "usage": {"input_tokens": est, "output_tokens": 5},
            }
        )


def start_server(app, port: int) -> uvicorn.Server:
    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 10
    while not server.started and time.time() < deadline:
        time.sleep(0.02)
    assert server.started, "server failed to start"
    return server


def tool_text(turn: int, tokens: int) -> str:
    line = f"turn {turn} synthetic tool output line %d: status ok, nothing unusual\n"
    n = max(1, (tokens * 4) // len(line))
    return "".join(line % i for i in range(n))


def run_mode(mode: str, turns: int, tool_tokens: int) -> dict:
    up = Upstream()
    up_port, px_port = free_port(), free_port()
    up_server = start_server(
        Starlette(routes=[Route("/v1/messages", up.messages, methods=["POST"])]),
        up_port,
    )

    tmp = tempfile.mkdtemp(prefix=f"mmu-bench-{mode}-")
    cfg = Config()
    cfg.proxy.anthropic_upstream = f"http://127.0.0.1:{up_port}"
    cfg.proxy.mode = mode
    cfg.store.path = Path(tmp)
    cfg.paging.budget_tokens = 12_000
    cfg.paging.low_watermark = 0.85
    cfg.paging.min_page_tokens = 256
    cfg.paging.pin_recent_turns = 4
    px_app = make_app(cfg)
    px_server = start_server(px_app, px_port)

    history: list[dict] = []
    latencies: list[float] = []
    t_wall = time.monotonic()
    with httpx.Client(base_url=f"http://127.0.0.1:{px_port}", timeout=60) as client:
        for turn in range(turns):
            history += [
                {"role": "user", "content": f"do step {turn}"},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": f"tu_{turn}",
                            "name": "bash",
                            "input": {"cmd": f"step {turn}"},
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": f"tu_{turn}",
                            "content": tool_text(turn, tool_tokens),
                        }
                    ],
                },
            ]
            t0 = time.monotonic()
            r = client.post(
                "/v1/messages",
                json={
                    "model": "claude-fable-5",
                    "max_tokens": 100,
                    "messages": history,
                },
                headers={"x-api-key": "sk-bench"},
            )
            latencies.append((time.monotonic() - t0) * 1000)
            r.raise_for_status()
            history.append(
                {"role": "assistant", "content": r.json()["content"]}
            )
    wall_s = time.monotonic() - t_wall

    proxy = px_app.state.proxy
    stats = proxy.store.stats()
    client_tokens = sum(
        estimate_request_tokens(m, None)
        for m in [history]  # final history ~ what client held; per-turn sum below
    )
    px_server.should_exit = True
    up_server.should_exit = True
    time.sleep(0.1)
    return {
        "mode": mode,
        "upstream_tokens": sum(up.request_tokens),
        "peak_request_tokens": max(up.request_tokens),
        "final_request_tokens": up.request_tokens[-1],
        "tokens_saved": stats["tokens_saved"],
        "faults": stats["faults"],
        "avg_overhead_ms": stats["avg_overhead_ms"],
        "p50_ms": statistics.median(latencies),
        "p95_ms": sorted(latencies)[int(0.95 * len(latencies)) - 1],
        "wall_s": wall_s,
        "client_final_tokens": client_tokens,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--turns", type=int, default=30)
    ap.add_argument("--tool-tokens", type=int, default=2000)
    args = ap.parse_args()

    results = [run_mode(m, args.turns, args.tool_tokens) for m in ("passthrough", "paging")]
    base, paged = results

    print(f"\nMMU e2e benchmark — {args.turns} turns, ~{args.tool_tokens} tok/tool result")
    print(f"{'metric':<28}{'passthrough':>14}{'paging':>14}")
    for key in (
        "upstream_tokens",
        "peak_request_tokens",
        "final_request_tokens",
        "tokens_saved",
        "faults",
        "avg_overhead_ms",
        "p50_ms",
        "p95_ms",
        "wall_s",
    ):
        b, p = base[key], paged[key]
        fmt = "{:>14.1f}" if isinstance(b, float) else "{:>14}"
        print(f"{key:<28}" + fmt.format(b) + fmt.format(p))
    saved_pct = 100 * (1 - paged["upstream_tokens"] / base["upstream_tokens"])
    print(f"\nupstream input tokens saved: {saved_pct:.1f}%")
    print(json.dumps(results, indent=2), file=open("bench/last_run.json", "w"))


if __name__ == "__main__":
    main()
