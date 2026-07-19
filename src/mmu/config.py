"""mmu configuration: mmu.toml + MMU_* environment overrides."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ProxyConfig:
    listen_host: str = "127.0.0.1"
    listen_port: int = 4004
    anthropic_upstream: str = "https://api.anthropic.com"
    openai_upstream: str = "https://api.openai.com"
    mode: str = "passthrough"  # "passthrough" | "paging" (paging lands in M1)
    allow_non_loopback: bool = False


@dataclass
class StoreConfig:
    path: Path = field(default_factory=lambda: Path("~/.mmu").expanduser())
    retention_days: int = 14


@dataclass
class PagingConfig:
    budget_tokens: int = 120_000
    low_watermark: float = 0.85
    policy: str = "lru"
    pin_recent_turns: int = 6
    min_page_tokens: int = 512
    epoch_min_turns: int = 8


@dataclass
class Config:
    proxy: ProxyConfig = field(default_factory=ProxyConfig)
    store: StoreConfig = field(default_factory=StoreConfig)
    paging: PagingConfig = field(default_factory=PagingConfig)

    @classmethod
    def load(cls, path: Path | None = None) -> "Config":
        cfg = cls()
        candidates = [path] if path else [Path("mmu.toml"), Path("~/.mmu/mmu.toml").expanduser()]
        for cand in candidates:
            if cand and cand.exists():
                data = tomllib.loads(cand.read_text())
                _apply_toml(cfg, data)
                break
        _apply_env(cfg)
        return cfg


def _apply_toml(cfg: Config, data: dict) -> None:
    p = data.get("proxy", {})
    listen = p.get("listen")
    if listen:
        host, _, port = listen.rpartition(":")
        cfg.proxy.listen_host = host or "127.0.0.1"
        cfg.proxy.listen_port = int(port)
    for key in ("anthropic_upstream", "openai_upstream", "mode", "allow_non_loopback"):
        if key in p:
            setattr(cfg.proxy, key, p[key])
    s = data.get("store", {})
    if "path" in s:
        cfg.store.path = Path(s["path"]).expanduser()
    if "retention_days" in s:
        cfg.store.retention_days = int(s["retention_days"])
    g = data.get("paging", {})
    for key in (
        "budget_tokens",
        "low_watermark",
        "policy",
        "pin_recent_turns",
        "min_page_tokens",
        "epoch_min_turns",
    ):
        if key in g:
            setattr(cfg.paging, key, g[key])


def _apply_env(cfg: Config) -> None:
    env = os.environ
    if v := env.get("MMU_LISTEN"):
        host, _, port = v.rpartition(":")
        cfg.proxy.listen_host = host or "127.0.0.1"
        cfg.proxy.listen_port = int(port)
    if v := env.get("MMU_ANTHROPIC_UPSTREAM"):
        cfg.proxy.anthropic_upstream = v
    if v := env.get("MMU_OPENAI_UPSTREAM"):
        cfg.proxy.openai_upstream = v
    if v := env.get("MMU_MODE"):
        cfg.proxy.mode = v
    if v := env.get("MMU_STORE_PATH"):
        cfg.store.path = Path(v).expanduser()
