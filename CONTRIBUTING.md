# Contributing to llm-mmu

Thanks for your interest! This project is young and contributions of every
size are welcome.

## Dev setup

```sh
git clone https://github.com/kaseban/llm-mmu && cd llm-mmu
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -q                  # full suite, <1s
ruff check src tests       # lint
```

## What we're looking for

- **Field reports.** Run `mmu` in front of your agent (Claude Code, aider,
  custom loops) in `passthrough` mode first, then `paging`. If anything
  behaves differently, open an issue with `mmu stats` output and, if possible,
  a redacted transcript.
- **Eviction policies.** `src/mmu/paging/policies/` — LRU is the baseline;
  smarter scoring (recency × size × block type) is an open area.
- **Provider coverage.** `src/mmu/proxy/adapters.py` — Anthropic Messages and
  OpenAI Chat Completions are supported; Responses API and others are not yet.
- **Benchmarks.** More realistic replay corpora for `bench/`.

## Ground rules

- Keep the core invariants: byte-transparent passthrough, fail-open on any
  internal error, keys never persisted, loopback-only by default.
- Every behavior change needs a test. e2e tests (real sockets) preferred over
  mocks — see `tests/test_paging_e2e.py` for the pattern.
- `ruff check` and `pytest -q` must pass; CI runs both on 3.11–3.13.
- One logical change per PR; small PRs merge fast.

## Reporting security issues

Please do not open public issues for security problems — email the maintainer
(see `pyproject.toml`) instead.
