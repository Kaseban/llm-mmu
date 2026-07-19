"""Provider adapters: normalize request/response shapes for observation.

The proxy never rewrites bytes in passthrough mode; adapters only *read* the
request body and response events to extract messages, model, and usage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from mmu.proxy.sse import SSEEvent


@dataclass
class Usage:
    tokens_in: int | None = None
    tokens_out: int | None = None
    cache_read: int | None = None
    cache_write: int | None = None


@dataclass
class ParsedRequest:
    provider: str
    model: str | None
    stream: bool
    messages: list[dict[str, Any]] = field(default_factory=list)
    system: Any = None
    tools: Any = None


class AnthropicAdapter:
    """/v1/messages — request has top-level `system`, `messages`, `tools`.

    Streaming usage: `message_start` carries input_tokens + cache counters,
    `message_delta` carries final output_tokens. Non-streaming: `usage` on the
    response object.
    """

    provider = "anthropic"
    paths = ("/v1/messages",)

    def parse_request(self, body: dict[str, Any]) -> ParsedRequest:
        return ParsedRequest(
            provider=self.provider,
            model=body.get("model"),
            stream=bool(body.get("stream")),
            messages=body.get("messages") or [],
            system=body.get("system"),
            tools=body.get("tools"),
        )

    def usage_from_json(self, body: dict[str, Any]) -> Usage:
        return self._usage(body.get("usage") or {})

    def usage_from_event(self, ev: SSEEvent, acc: Usage) -> None:
        obj = ev.json()
        if not obj:
            return
        etype = obj.get("type")
        if etype == "message_start":
            u = self._usage((obj.get("message") or {}).get("usage") or {})
            acc.tokens_in = u.tokens_in
            acc.cache_read = u.cache_read
            acc.cache_write = u.cache_write
        elif etype == "message_delta":
            u = (obj.get("usage") or {})
            if "output_tokens" in u:
                acc.tokens_out = u["output_tokens"]

    @staticmethod
    def _usage(u: dict[str, Any]) -> Usage:
        return Usage(
            tokens_in=u.get("input_tokens"),
            tokens_out=u.get("output_tokens"),
            cache_read=u.get("cache_read_input_tokens"),
            cache_write=u.get("cache_creation_input_tokens"),
        )


class OpenAIAdapter:
    """/v1/chat/completions — system prompt is messages[0] when role=system.

    Streaming usage arrives in the final chunk only when the client sends
    stream_options.include_usage; otherwise we fall back to estimates.
    """

    provider = "openai"
    paths = ("/v1/chat/completions", "/v1/responses")

    def parse_request(self, body: dict[str, Any]) -> ParsedRequest:
        messages = body.get("messages") or body.get("input") or []
        system = None
        if messages and isinstance(messages[0], dict) and messages[0].get("role") in (
            "system",
            "developer",
        ):
            system = messages[0].get("content")
            messages = messages[1:]
        return ParsedRequest(
            provider=self.provider,
            model=body.get("model"),
            stream=bool(body.get("stream")),
            messages=messages,
            system=system,
            tools=body.get("tools"),
        )

    def usage_from_json(self, body: dict[str, Any]) -> Usage:
        return self._usage(body.get("usage") or {})

    def usage_from_event(self, ev: SSEEvent, acc: Usage) -> None:
        if ev.data.strip() == "[DONE]":
            return
        obj = ev.json()
        if not obj:
            return
        u = obj.get("usage")
        if u:
            parsed = self._usage(u)
            acc.tokens_in = parsed.tokens_in
            acc.tokens_out = parsed.tokens_out
            acc.cache_read = parsed.cache_read

    @staticmethod
    def _usage(u: dict[str, Any]) -> Usage:
        details = u.get("prompt_tokens_details") or {}
        return Usage(
            tokens_in=u.get("prompt_tokens") or u.get("input_tokens"),
            tokens_out=u.get("completion_tokens") or u.get("output_tokens"),
            cache_read=details.get("cached_tokens"),
        )


ADAPTERS = [AnthropicAdapter(), OpenAIAdapter()]


def adapter_for_path(path: str):
    for a in ADAPTERS:
        if any(path.startswith(p) for p in a.paths):
            return a
    return None
