"""Accuracy / context-integrity benchmark (NIAH + RULER style).

Methodology — adapted from Needle-in-a-Haystack and RULER multi-needle
retrieval: every turn plants a unique random "needle" fact
(`SECRET[step-N] = <16-hex>`) inside that turn's large tool_result, buried
mid-content in realistic filler. The context grows far past the paging
budget, so old needles get evicted into the page store. Periodically (and in
a final sweep) the client probes: "what was the SECRET in step N's log?"

Two upstreams:

* oracle (default, offline, deterministic): simulates a competent model.
  If the literal secret is present anywhere in the request it answers
  `VALUE: <secret>` (resident hit). If not, but a stub whose preview
  mentions `step-N log` is present, it replies with the stub's
  `mmu recall <id>` marker (page fault) and the client re-sends — exactly
  the designed fault path. If neither is present, the fact is LOST.
  Because the oracle only answers when the *literal bytes* are in the
  request, eventual accuracy == byte-level context integrity: 100% means
  the paging layer never made a fact unreachable.

* live (--live, needs ANTHROPIC_API_KEY, costs money): same protocol
  against the real API; grades whether the model's reply contains the
  secret. Measures true end-to-end model accuracy through the proxy.

Usage:
  .venv/bin/python bench/accuracy_bench.py [--turns 120] [--tool-tokens 2000]
      [--probe-every 10] [--sweep-every 5] [--modes passthrough,paging]
      [--live --model claude-haiku-4-5-20251001]
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import tempfile
import time
from pathlib import Path

import httpx
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from e2e_bench import free_port, start_server  # noqa: E402

from mmu.config import Config  # noqa: E402
from mmu.proxy.app import make_app  # noqa: E402
from mmu.tokens import estimate_request_tokens  # noqa: E402

RECALL_RE = re.compile(r"mmu\s+recall\s+([0-9a-f]{8,16})")
PROBE_RE = re.compile(r"PROBE step-(\d+)\b")
MAX_ROUND_TRIPS = 4


def make_tool_output(step: int, secret: str, tokens: int) -> str:
    """Filler log with the needle buried ~40% deep (NIAH-style)."""
    line = f"=== step-{step} log === line %d: build ok, cache warm, nothing unusual\n"
    n = max(4, (tokens * 4) // len(line))
    lines = [line % i for i in range(n)]
    lines.insert(max(1, int(n * 0.4)), f"note: SECRET[step-{step}] = {secret}\n")
    return f"=== step-{step} log ===\n" + "".join(lines)


def iter_strings(node):
    if isinstance(node, str):
        yield node
    elif isinstance(node, list):
        for x in node:
            yield from iter_strings(x)
    elif isinstance(node, dict):
        for k in ("content", "text"):
            if k in node:
                yield from iter_strings(node[k])


def last_user_text(messages: list[dict]) -> str:
    for m in reversed(messages):
        if m.get("role") == "user":
            return " ".join(iter_strings(m.get("content")))
    return ""


class OracleUpstream:
    """Deterministic stand-in for a competent model (see module docstring)."""

    def __init__(self, secrets: dict[int, str]):
        self.secrets = secrets
        self.requests = 0

    def answer(self, body: dict) -> str:
        probe = PROBE_RE.search(last_user_text(body.get("messages", [])))
        if not probe:
            return "ack"
        step = int(probe.group(1))
        secret = self.secrets[step]
        blobs = list(iter_strings([m.get("content") for m in body["messages"]]))
        if any(secret in b for b in blobs):
            return f"VALUE: {secret}"
        for b in blobs:  # stub whose preview names this step's log?
            if "[mmu: evicted" in b and f"step-{step} log" in b:
                m = RECALL_RE.search(b)
                if m:
                    return f"I need that log restored. mmu recall {m.group(1)}"
        return "LOST: fact not reachable"

    async def messages(self, request):
        body = await request.json()
        self.requests += 1
        return JSONResponse(
            {
                "id": f"msg_{self.requests}",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": self.answer(body)}],
                "usage": {"input_tokens": 0, "output_tokens": 5},
            }
        )


LIVE_SYSTEM = (
    "You are being benchmarked. Tool logs may have been swapped for "
    '[mmu: evicted …] stubs. When asked "PROBE step-N", reply exactly '
    "'VALUE: <secret>' using SECRET[step-N] from that step's log. If the "
    "log is a stub, reply with its literal 'mmu recall <id>' text instead "
    "and you will be re-asked. Otherwise reply 'ack'."
)


def run_mode(mode: str, args) -> dict:
    rng = random.Random(args.seed)
    secrets = {i: f"{rng.getrandbits(64):016x}" for i in range(args.turns)}

    if args.live:
        upstream_url = "https://api.anthropic.com"
        api_key = os.environ["ANTHROPIC_API_KEY"]
        up_server = None
    else:
        oracle = OracleUpstream(secrets)
        up_port = free_port()
        up_server = start_server(
            Starlette(routes=[Route("/v1/messages", oracle.messages, methods=["POST"])]),
            up_port,
        )
        upstream_url = f"http://127.0.0.1:{up_port}"
        api_key = "sk-bench"

    cfg = Config()
    cfg.proxy.anthropic_upstream = upstream_url
    cfg.proxy.mode = mode
    cfg.store.path = Path(tempfile.mkdtemp(prefix=f"mmu-acc-{mode}-"))
    cfg.paging.budget_tokens = args.budget_tokens
    cfg.paging.low_watermark = 0.85
    cfg.paging.min_page_tokens = 256
    cfg.paging.pin_recent_turns = 4
    px_app = make_app(cfg)
    px_port = free_port()
    px_server = start_server(px_app, px_port)

    history: list[dict] = []
    m = {"probes": 0, "first_try": 0, "recovered": 0, "lost": 0, "round_trips": []}
    probed: set[int] = set()

    def post(client) -> str:
        body = {
            "model": args.model,
            "max_tokens": 200,
            "messages": history,
        }
        if args.live:
            body["system"] = LIVE_SYSTEM
        r = client.post("/v1/messages", json=body, headers={"x-api-key": api_key})
        r.raise_for_status()
        content = r.json()["content"]
        history.append({"role": "assistant", "content": content})
        return " ".join(iter_strings(content))

    def probe(client, step: int) -> None:
        probed.add(step)
        m["probes"] += 1
        history.append(
            {
                "role": "user",
                "content": f"PROBE step-{step}: what is SECRET[step-{step}] "
                f"from the step-{step} log? Reply 'VALUE: <secret>'.",
            }
        )
        for rt in range(1, MAX_ROUND_TRIPS + 1):
            reply = post(client)
            if secrets[step] in reply:
                m["round_trips"].append(rt)
                m["first_try" if rt == 1 else "recovered"] += 1
                return
            if "mmu recall" not in reply:
                break
            history.append(
                {
                    "role": "user",
                    "content": f"PROBE step-{step}: the log should be restored now — "
                    f"what is SECRET[step-{step}]? Reply 'VALUE: <secret>'.",
                }
            )
        m["lost"] += 1
        print(f"  !! step-{step} LOST in mode={mode}", file=sys.stderr)

    t0 = time.monotonic()
    with httpx.Client(base_url=f"http://127.0.0.1:{px_port}", timeout=120) as client:
        for turn in range(args.turns):
            history += [
                {"role": "user", "content": f"do step {turn}"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "tool_use", "id": f"tu_{turn}", "name": "bash",
                         "input": {"cmd": f"step {turn}"}}
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": f"tu_{turn}",
                         "content": make_tool_output(turn, secrets[turn], args.tool_tokens)}
                    ],
                },
            ]
            post(client)
            # periodic probe of a random already-planted, unprobed old step
            if turn and turn % args.probe_every == 0:
                candidates = [s for s in range(turn - 4) if s not in probed]
                if candidates:
                    probe(client, rng.choice(candidates))
        # final sweep: every Nth needle plus the very oldest (worst case)
        for step in sorted({0, *range(0, args.turns, args.sweep_every)} - probed):
            probe(client, step)
    wall_s = time.monotonic() - t0

    stats = px_app.state.proxy.store.stats()
    result = {
        "mode": mode + ("+live" if args.live else "+oracle"),
        "turns": args.turns,
        "planted_tokens": args.turns * args.tool_tokens,
        "final_context_tokens": estimate_request_tokens(history, None),
        "probes": m["probes"],
        "first_try_hits": m["first_try"],
        "recovered_via_fault": m["recovered"],
        "lost": m["lost"],
        "eventual_accuracy": (m["probes"] - m["lost"]) / m["probes"] if m["probes"] else 1.0,
        "avg_round_trips": (sum(m["round_trips"]) / len(m["round_trips"]))
        if m["round_trips"] else 0.0,
        "engine_faults": stats["faults"],
        "tokens_saved": stats["tokens_saved"],
        "wall_s": round(wall_s, 1),
    }
    px_server.should_exit = True
    if up_server:
        up_server.should_exit = True
    time.sleep(0.1)
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--turns", type=int, default=120)
    ap.add_argument("--tool-tokens", type=int, default=2000)
    ap.add_argument("--budget-tokens", type=int, default=12_000)
    ap.add_argument("--probe-every", type=int, default=10)
    ap.add_argument("--sweep-every", type=int, default=5)
    ap.add_argument("--modes", default="passthrough,paging")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--model", default="claude-fable-5")
    args = ap.parse_args()

    results = [run_mode(mode, args) for mode in args.modes.split(",")]
    print(f"\nMMU accuracy benchmark — {args.turns} turns, "
          f"~{args.turns * args.tool_tokens:,} planted tokens")
    keys = [k for k in results[0] if k != "mode"]
    print(f"{'metric':<24}" + "".join(f"{r['mode']:>22}" for r in results))
    for k in keys:
        row = "".join(
            f"{r[k]:>22.3f}" if isinstance(r[k], float) else f"{r[k]:>22}"
            for r in results
        )
        print(f"{k:<24}" + row)
    json.dump(results, open("bench/accuracy_last_run.json", "w"), indent=2)
    if any(r["lost"] for r in results):
        sys.exit("INTEGRITY FAILURE: some facts were unreachable")


if __name__ == "__main__":
    main()
