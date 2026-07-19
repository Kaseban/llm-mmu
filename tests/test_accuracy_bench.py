"""Tiny-scale run of the NIAH/RULER-style accuracy bench as a CI gate.

Asserts context integrity: with paging on and a budget small enough to force
heavy eviction, every planted needle must remain reachable (resident hit or
recovered via one `mmu recall` fault) — eventual accuracy 100%, zero lost.
"""

import sys
from argparse import Namespace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))

from accuracy_bench import run_mode  # noqa: E402


def _args(**over):
    base = dict(
        turns=12,
        tool_tokens=600,
        budget_tokens=5000,
        probe_every=4,
        sweep_every=2,
        seed=7,
        live=False,
        model="claude-fable-5",
    )
    base.update(over)
    return Namespace(**base)


def test_paging_context_integrity():
    r = run_mode("paging", _args())
    assert r["lost"] == 0
    assert r["eventual_accuracy"] == 1.0
    # eviction actually happened and the fault path was exercised
    assert r["tokens_saved"] > 0
    assert r["recovered_via_fault"] > 0


def test_passthrough_all_first_try():
    r = run_mode("passthrough", _args())
    assert r["lost"] == 0
    assert r["first_try_hits"] == r["probes"]
