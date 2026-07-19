"""SSE primitives: incremental parser used on the observer branch of the tee.

The client branch in passthrough mode re-emits raw bytes untouched; this parser
only feeds observation (usage accounting, later fault detection). Parser errors
must never affect the client stream.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field


@dataclass
class SSEEvent:
    event: str | None = None
    data: str = ""

    def json(self) -> dict | None:
        try:
            obj = json.loads(self.data)
        except (json.JSONDecodeError, ValueError):
            return None
        return obj if isinstance(obj, dict) else None


@dataclass
class SSEParser:
    """Incremental, tolerant SSE parser. Feed raw bytes, get completed events."""

    _buf: bytes = b""
    _event: str | None = None
    _data_lines: list[str] = field(default_factory=list)

    def feed(self, chunk: bytes) -> list[SSEEvent]:
        self._buf += chunk
        events: list[SSEEvent] = []
        while True:
            # Normalize on \n; SSE allows \r\n and \r but upstreams use \n.
            idx = self._buf.find(b"\n")
            if idx < 0:
                break
            line = self._buf[:idx].rstrip(b"\r")
            self._buf = self._buf[idx + 1 :]
            if not line:
                if self._data_lines or self._event is not None:
                    events.append(SSEEvent(self._event, "\n".join(self._data_lines)))
                self._event = None
                self._data_lines = []
                continue
            if line.startswith(b":"):
                continue  # comment / keepalive
            field_name, _, value = line.partition(b":")
            if value.startswith(b" "):
                value = value[1:]
            try:
                name = field_name.decode("utf-8")
                text = value.decode("utf-8")
            except UnicodeDecodeError:
                continue
            if name == "event":
                self._event = text
            elif name == "data":
                self._data_lines.append(text)
        return events
