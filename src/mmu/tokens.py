"""Token estimation.

For M0 accounting we prefer authoritative numbers from provider usage blocks.
Estimation is only used for `tokens_saved` projections and paging budgets; a
chars/4 heuristic is deliberately dependency-free. tiktoken can be added as an
optional extra later without changing callers.
"""

from __future__ import annotations

import json
from typing import Any


def estimate_text_tokens(text: str) -> int:
    # Cheap, provider-agnostic heuristic; ~4 chars/token for English + code.
    return max(1, len(text) // 4)


def estimate_message_tokens(message: dict[str, Any]) -> int:
    content = message.get("content")
    if isinstance(content, str):
        return estimate_text_tokens(content) + 4
    if isinstance(content, list):
        total = 4
        for block in content:
            if isinstance(block, dict):
                total += estimate_text_tokens(json.dumps(block, ensure_ascii=False))
            else:
                total += estimate_text_tokens(str(block))
        return total
    return 4


def estimate_request_tokens(messages: list[dict[str, Any]], system: Any = None) -> int:
    total = sum(estimate_message_tokens(m) for m in messages if isinstance(m, dict))
    if isinstance(system, str):
        total += estimate_text_tokens(system)
    elif isinstance(system, list):
        total += estimate_text_tokens(json.dumps(system, ensure_ascii=False))
    return total
