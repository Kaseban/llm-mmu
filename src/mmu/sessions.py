"""Session identity via message-prefix hash chains.

Stateless clients (Claude Code, most agent harnesses) resend the full message
list each turn. We compute a rolling hash chain over canonicalized messages;
if an incoming request's chain matches a stored session's chain for some
prefix, it is the same session continuing (the usual case: stored chain is a
proper prefix of the incoming one, extended by one assistant + one user turn).

Canonicalization must be stable across turns even though the client may render
identically (e.g. re-serialized JSON key order): we hash a sorted-key JSON dump
of a reduced view (role + content) and ignore volatile fields.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from typing import Any

from mmu.store.db import Store


def _canonical_message(msg: dict[str, Any]) -> bytes:
    view = {"role": msg.get("role"), "content": msg.get("content")}
    return json.dumps(view, sort_keys=True, ensure_ascii=False).encode()


def chain_hashes(messages: list[dict[str, Any]]) -> list[str]:
    prev = b""
    out: list[str] = []
    for msg in messages:
        h = hashlib.sha256(prev + _canonical_message(msg)).hexdigest()
        out.append(h)
        prev = h.encode()
    return out


def system_hash(system: Any, tools: Any) -> str:
    blob = json.dumps({"system": system, "tools": tools}, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def api_hash(auth_header: str | None) -> str:
    return hashlib.sha256((auth_header or "anon").encode()).hexdigest()[:16]


def resolve_session(
    store: Store,
    messages: list[dict[str, Any]],
    auth_header: str | None,
    provider: str,
    sys_hash: str,
) -> str:
    """Return a session id for this request, creating one if needed."""
    ah = api_hash(auth_header)
    chain = chain_hashes(messages)
    session_id: str | None = None
    # Match on the deepest hash first; a returning session's stored chain tip
    # appears somewhere in our new chain (client appended turns since).
    for h in reversed(chain):
        session_id = store.find_session_by_prefix(h, ah)
        if session_id:
            break
    if not session_id:
        session_id = uuid.uuid4().hex[:12]
    store.upsert_session(session_id, ah, provider, sys_hash)
    store.replace_prefix_chain(session_id, chain)
    return session_id
