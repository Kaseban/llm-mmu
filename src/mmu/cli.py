"""mmu command-line interface.

Commands (M0):
  mmu proxy    run the proxy in the foreground
  mmu stats    print accounting from the store
  mmu env      print shell exports to point a client at the proxy
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from mmu.config import Config


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="mmu", description="Memory-management unit proxy for LLM agents")
    ap.add_argument("--config", type=Path, default=None, help="path to mmu.toml")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_proxy = sub.add_parser("proxy", help="run the proxy (foreground)")
    p_proxy.add_argument("--listen", default=None, help="host:port (default 127.0.0.1:4004)")
    p_proxy.add_argument("--mode", choices=["passthrough", "paging"], default=None)

    p_stats = sub.add_parser("stats", help="show token/latency accounting")
    p_stats.add_argument("--since", default=None, help="window, e.g. 24h, 7d")
    p_stats.add_argument("--json", action="store_true")

    sub.add_parser("env", help="print exports for pointing clients at the proxy")

    args = ap.parse_args(argv)
    cfg = Config.load(args.config)

    if args.cmd == "proxy":
        if args.listen:
            host, _, port = args.listen.rpartition(":")
            cfg.proxy.listen_host = host or "127.0.0.1"
            cfg.proxy.listen_port = int(port)
        if args.mode:
            cfg.proxy.mode = args.mode
        if cfg.proxy.mode == "paging":
            raise SystemExit("mmu: paging mode lands in M1; run passthrough for now")
        from mmu.proxy.app import run

        base = f"http://{cfg.proxy.listen_host}:{cfg.proxy.listen_port}"
        print(f"mmu proxy [{cfg.proxy.mode}] listening on {base}")
        print(f"  anthropic upstream: {cfg.proxy.anthropic_upstream}")
        print(f"  openai upstream:    {cfg.proxy.openai_upstream}")
        print(f"  store:              {cfg.store.path}")
        run(cfg)
        return 0

    if args.cmd == "stats":
        from mmu.store.db import Store

        since_ts = None
        if args.since:
            unit = args.since[-1]
            n = int(args.since[:-1])
            mult = {"h": 3600, "d": 86400, "m": 60}.get(unit)
            if not mult:
                raise SystemExit(f"mmu: bad --since value: {args.since}")
            since_ts = int(time.time()) - n * mult
        store = Store(cfg.store.path)
        s = store.stats(since_ts)
        store.close()
        if args.json:
            print(json.dumps(s, indent=2))
        else:
            print(f"requests:        {s['requests']}  (sessions: {s['sessions']})")
            print(f"tokens in/out:   {s['tokens_in']:,} / {s['tokens_out']:,}")
            print(f"cache read:      {s['cache_read']:,}")
            print(f"tokens saved:    {s['tokens_saved']:,}")
            print(f"faults:          {s['faults']}")
            print(f"avg overhead:    {s['avg_overhead_ms']:.1f} ms")
        return 0

    if args.cmd == "env":
        base = f"http://{cfg.proxy.listen_host}:{cfg.proxy.listen_port}"
        print(f"export ANTHROPIC_BASE_URL={base}")
        print(f"export OPENAI_BASE_URL={base}/v1")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
