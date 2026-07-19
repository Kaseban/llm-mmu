# llm-mmu

**Virtual memory for LLM agents.** A transparent local proxy that pages cold
context out of the window — and faults it back in only when the model asks.
**No SDK. No framework. No code changes.**

[![CI](https://github.com/kaseban/llm-mmu/actions/workflows/ci.yml/badge.svg)](https://github.com/kaseban/llm-mmu/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/mmu-proxy)](https://pypi.org/project/mmu-proxy/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](https://pypi.org/project/mmu-proxy/)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)

```
┌────────────┐      ┌─────────────────────────┐      ┌─────────────┐
│ your agent │ ───► │  mmu (localhost:4004)   │ ───► │  model API  │
│ (unchanged)│ ◄─── │  RAM ⇄ SQLite "disk"    │ ◄─── │             │
└────────────┘      └─────────────────────────┘      └─────────────┘
        cold tool results paged out · faulted back in one round trip
```

Agent sessions resend hundreds of thousands of tokens every turn — mostly old
tool output the model never looks at again. `mmu` treats the context window
like RAM: cold blocks are evicted to a local SQLite store and replaced with a
tiny self-describing stub; if the model needs one, it writes `mmu recall <id>`
and the full content is back on the very next request.

**Result: 74% fewer upstream input tokens** on a 30-turn agent benchmark, ~2 ms
added latency, zero behavioral difference for the client.

| 30-turn agent loop (e2e, real HTTP) | passthrough | paging |
|-------------------------------------|------------:|-------:|
| upstream input tokens               |     972,030 | **248,930 (−74.4%)** |
| peak request tokens                 |      62,709 |  9,847 |
| avg proxy overhead (ms)             |         0.9 |    2.6 |

## Quick start

```sh
pip install mmu-proxy        # or: pipx install mmu-proxy / uv tool install mmu-proxy
mmu proxy                    # starts on 127.0.0.1:4004
```

In another shell, point any client at it:

```sh
eval "$(mmu env)"            # exports ANTHROPIC_BASE_URL / OPENAI_BASE_URL
claude                       # or aider, or your own loop — anything
mmu stats --since 24h        # tokens saved, sessions, faults, overhead
```

That's it. Your API keys are forwarded, never stored. Any internal error
fails open to plain forwarding.

## How it works

Clients like Claude Code are stateless — they resend the full history each
turn. `mmu` rewrites only the **upstream** copy:

1. **Track** sessions across requests via message-prefix hashing (no cookies,
   no client state).
2. **Evict** when a request exceeds `low_watermark × budget_tokens`: oldest
   cold blocks first (tool results and giant user dumps — never assistant
   turns, never the pinned recent tail). Pages persist to SQLite (WAL).
3. **Stub** each evicted block in place:

   ```
   [mmu: evicted tool_result page fb2a88a05fe1 (~1898 tokens).
    Preview: "line 0 of synthetic build output…" — to restore, include the
    literal text "mmu recall fb2a88a05fe1" in your reply.]
   ```

4. **Fault** — if the model emits `mmu recall <id>`, the marker lands in the
   resent history, the page is marked resident, and the next request carries
   the full content. One round trip, no SDK, no client-side state.

## Configuration

`mmu.toml` in the working directory or `~/.mmu/mmu.toml`; `MMU_*` env vars
override:

```toml
[proxy]
listen = "127.0.0.1:4004"
mode = "paging"           # or "passthrough" (observe/measure only)

[paging]
budget_tokens = 120000    # eviction kicks in above low_watermark * budget
low_watermark = 0.85
policy = "lru"
pin_recent_turns = 6      # never page the freshest N messages
min_page_tokens = 512     # smaller blocks are never worth a stub
```

## Design invariants

- **Byte-transparent by default** — passthrough mode returns upstream bytes
  verbatim; the observer branch (SSE parsing, accounting) can never break the
  stream.
- **Keys are never stored** — forwarded in headers, hashed (sha256) only to
  namespace sessions.
- **Loopback only** unless explicitly configured otherwise.
- **Fail open** — any mmu-internal error degrades to plain forwarding.

## Benchmarks

Reproduce the table above (fake instrumented upstream, real HTTP through the
real proxy, client transcript verified byte-identical in both modes):

```sh
git clone https://github.com/kaseban/llm-mmu && cd llm-mmu
pip install -e ".[dev]"
python bench/e2e_bench.py --turns 30 --tool-tokens 2000
```

## Project layout

```
src/mmu/
  cli.py            # mmu proxy | stats | env
  config.py         # mmu.toml + MMU_* env
  sessions.py       # session identity via prefix-hash chains
  tokens.py         # token estimation heuristics
  proxy/app.py      # starlette + httpx proxy w/ SSE tee
  proxy/adapters.py # anthropic / openai request+usage normalization
  proxy/sse.py      # incremental SSE parser (observer branch)
  paging/engine.py  # eviction/stub/fault engine (fail-open rewrite hook)
  store/            # SQLite (WAL) schema + accessors
bench/e2e_bench.py  # passthrough-vs-paging benchmark over real HTTP
```

## Roadmap

- [x] **M0** — transparent proxy, session tracking, token/latency accounting
- [x] **M1** — LRU demand paging, stub/recall page faults, e2e benchmark
- [ ] **M2** — summarize tier (compress instead of stub), per-block TTLs
- [ ] **M3** — smarter eviction (attention-proxy scoring), multi-session store GC
- [ ] Streaming-tool-use edge cases, OpenAI Responses API, provider matrix

## Contributing

Issues and PRs welcome — see [CONTRIBUTING.md](CONTRIBUTING.md). Good first
contributions: trying it against your agent stack and filing transcripts where
paging misbehaves, new eviction policies (`src/mmu/paging/policies/`), and
provider adapter coverage.

```sh
pip install -e ".[dev]" && pytest -q   # 17 tests, <1s
```

## License

[Apache-2.0](LICENSE)
