"""M1 demand-paging engine.

Key property: the *client* keeps its full history (it is stateless and resends
everything each turn). The proxy rewrites only the upstream payload, replacing
cold blocks with self-describing stubs. Because eviction is recomputed from the
full history on every request, no client-side state or SDK is needed, and a
page fault resolves in exactly one round trip:

  1. model output contains "mmu recall <id>"
  2. client appends that assistant turn and resends history
  3. engine sees the recall marker, exempts the page, upstream gets full body

Eviction: oldest-first (LRU by construction — the resent history is in time
order) over tool results and oversized user dumps, never touching the last
`pin_recent_turns` messages, until the estimate is under
`low_watermark * budget_tokens`.
"""

from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass, field
from typing import Any

from mmu.config import PagingConfig
from mmu.store.db import Store
from mmu.tokens import estimate_request_tokens, estimate_text_tokens

RECALL_RE = re.compile(r"mmu\s+recall\s+([0-9a-f]{8,16})")

STUB_TMPL = (
    '[mmu: evicted {kind} page {page_id} (~{tokens} tokens) to save context. '
    'Preview: "{preview}…" — to restore the full content, include the literal '
    "text \"mmu recall {page_id}\" in your reply.]"
)


def page_id_for(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def _block_text(content: Any) -> str:
    """Flatten a content value (str or block list) to text for hashing/size."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for b in content:
            if isinstance(b, dict):
                parts.append(b.get("text") or b.get("content") or str(b))
            else:
                parts.append(str(b))
        return "\n".join(str(p) for p in parts)
    return str(content)


@dataclass
class Candidate:
    msg_index: int
    block_index: int | None  # None => whole-message content
    kind: str                # 'tool_result' | 'file_dump'
    text: str
    tokens: int
    tool_use_id: str | None = None


@dataclass
class PageStats:
    evicted: int = 0
    tokens_saved: int = 0
    faults: int = 0
    recalled: list[str] = field(default_factory=list)


class PagingEngine:
    def __init__(self, store: Store, cfg: PagingConfig):
        self.store = store
        self.cfg = cfg

    # -- public ---------------------------------------------------------------

    def rewrite(self, session_id: str | None, body: dict[str, Any]) -> tuple[dict, PageStats]:
        stats = PageStats()
        messages = body.get("messages")
        if not isinstance(messages, list) or not messages:
            return body, stats

        recalls = self._find_recalls(messages)
        stats.faults = self._register_faults(recalls, session_id)
        stats.recalled = sorted(recalls)

        est = estimate_request_tokens(messages, body.get("system"))
        target = int(self.cfg.budget_tokens * self.cfg.low_watermark)
        if est <= target:
            return body, stats

        candidates = self._candidates(messages)
        new_messages = list(messages)
        for cand in candidates:  # oldest first
            if est <= target:
                break
            pid = page_id_for(cand.text)
            if pid in recalls:
                continue  # faulted back in — keep resident
            stub = STUB_TMPL.format(
                kind=cand.kind,
                page_id=pid,
                tokens=cand.tokens,
                preview=cand.text[:120].replace('"', "'").replace("\n", " "),
            )
            tok_stub = estimate_text_tokens(stub)
            saved = cand.tokens - tok_stub
            if saved <= 0:
                continue
            self._persist_page(pid, session_id, cand, tok_stub)
            new_messages[cand.msg_index] = self._apply_stub(
                new_messages[cand.msg_index], cand, stub
            )
            est -= saved
            stats.evicted += 1
            stats.tokens_saved += saved

        if not stats.evicted:
            return body, stats
        return {**body, "messages": new_messages}, stats

    # -- internals ------------------------------------------------------------

    def _find_recalls(self, messages: list[dict]) -> set[str]:
        found: set[str] = set()
        for msg in messages:
            if not isinstance(msg, dict) or msg.get("role") != "assistant":
                continue
            found.update(RECALL_RE.findall(_block_text(msg.get("content"))))
        return found

    def _register_faults(self, recalls: set[str], session_id: str | None) -> int:
        faults = 0
        for pid in recalls:
            row = self.store.conn.execute(
                "SELECT state FROM pages WHERE page_id=?", (pid,)
            ).fetchone()
            if row and row["state"] == "stub":
                self.store.conn.execute(
                    "UPDATE pages SET state='resident', fault_count=fault_count+1, "
                    "last_access_turn=? WHERE page_id=?",
                    (int(time.time()), pid),
                )
                faults += 1
        if faults:
            self.store.conn.commit()
        return faults

    def _candidates(self, messages: list[dict]) -> list[Candidate]:
        limit = max(0, len(messages) - self.cfg.pin_recent_turns)
        out: list[Candidate] = []
        for i, msg in enumerate(messages[:limit]):
            if not isinstance(msg, dict):
                continue
            role = msg.get("role")
            content = msg.get("content")
            # OpenAI tool message: whole content is a tool result.
            if role == "tool" and isinstance(content, str):
                tok = estimate_text_tokens(content)
                if tok >= self.cfg.min_page_tokens:
                    out.append(Candidate(i, None, "tool_result", content, tok,
                                         msg.get("tool_call_id")))
                continue
            if role != "user":
                continue  # never page assistant turns (coherence)
            if isinstance(content, str):
                tok = estimate_text_tokens(content)
                if tok >= self.cfg.min_page_tokens:
                    out.append(Candidate(i, None, "file_dump", content, tok))
                continue
            if isinstance(content, list):
                for j, block in enumerate(content):
                    if not isinstance(block, dict):
                        continue
                    if block.get("type") == "tool_result":
                        text = _block_text(block.get("content"))
                        kind = "tool_result"
                    elif block.get("type") == "text":
                        text = block.get("text") or ""
                        kind = "file_dump"
                    else:
                        continue
                    tok = estimate_text_tokens(text)
                    if tok >= self.cfg.min_page_tokens:
                        out.append(Candidate(i, j, kind, text, tok,
                                             block.get("tool_use_id")))
        return out

    def _persist_page(self, pid: str, session_id: str | None, cand: Candidate,
                      tok_stub: int) -> None:
        now = int(time.time())
        self.store.conn.execute(
            """INSERT INTO pages(page_id, session_id, kind, state, msg_index,
                                 tool_use_id, body, body_sha, tok_full, tok_stub,
                                 created_turn, last_access_turn)
               VALUES(?,?,?,'stub',?,?,?,?,?,?,?,?)
               ON CONFLICT(page_id) DO UPDATE SET
                 state=CASE WHEN pages.state='resident' THEN 'resident' ELSE 'stub' END,
                 last_access_turn=excluded.last_access_turn""",
            (pid, session_id or "unknown", cand.kind, cand.msg_index,
             cand.tool_use_id, cand.text.encode(), pid, cand.tokens, tok_stub,
             now, now),
        )
        self.store.conn.commit()

    @staticmethod
    def _apply_stub(msg: dict, cand: Candidate, stub: str) -> dict:
        if cand.block_index is None:
            return {**msg, "content": stub}
        content = list(msg["content"])
        block = dict(content[cand.block_index])
        if block.get("type") == "tool_result":
            block["content"] = stub
        else:
            block["text"] = stub
        content[cand.block_index] = block
        return {**msg, "content": content}
