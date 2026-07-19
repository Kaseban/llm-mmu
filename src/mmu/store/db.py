"""SQLite store for mmu.

Single-writer (the proxy process). WAL mode so `mmu stats` can read
concurrently. Bodies are zstd-compressed in M1; M0 keeps accounting only.
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Any


class Store:
    def __init__(self, root: Path):
        self.root = root
        root.mkdir(parents=True, exist_ok=True)
        self.db_path = root / "mmu.db"
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        schema = (Path(__file__).parent / "schema.sql").read_text()
        self.conn.executescript(schema)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    # -- sessions -------------------------------------------------------------

    def upsert_session(
        self, session_id: str, api_hash: str, provider: str, system_hash: str
    ) -> None:
        now = int(time.time())
        self.conn.execute(
            """INSERT INTO sessions(session_id, api_hash, provider, system_hash,
                                    created_at, last_seen_at, turn_count)
               VALUES(?,?,?,?,?,?,1)
               ON CONFLICT(session_id) DO UPDATE SET
                 last_seen_at=excluded.last_seen_at,
                 turn_count=turn_count+1""",
            (session_id, api_hash, provider, system_hash, now, now),
        )
        self.conn.commit()

    def get_prefix_chain(self, session_id: str) -> list[str]:
        rows = self.conn.execute(
            "SELECT chain_hash FROM prefix_hashes WHERE session_id=? ORDER BY msg_index",
            (session_id,),
        ).fetchall()
        return [r["chain_hash"] for r in rows]

    def replace_prefix_chain(self, session_id: str, chain: list[str]) -> None:
        self.conn.execute("DELETE FROM prefix_hashes WHERE session_id=?", (session_id,))
        self.conn.executemany(
            "INSERT INTO prefix_hashes(session_id, msg_index, chain_hash) VALUES(?,?,?)",
            [(session_id, i, h) for i, h in enumerate(chain)],
        )
        self.conn.commit()

    def find_session_by_prefix(self, chain_hash: str, api_hash: str) -> str | None:
        """Find the session whose chain contains this hash (longest match wins
        upstream; here any session owning the hash under the same api_hash)."""
        row = self.conn.execute(
            """SELECT p.session_id FROM prefix_hashes p
               JOIN sessions s ON s.session_id = p.session_id
               WHERE p.chain_hash=? AND s.api_hash=?
               ORDER BY s.last_seen_at DESC LIMIT 1""",
            (chain_hash, api_hash),
        ).fetchone()
        return row["session_id"] if row else None

    # -- requests -------------------------------------------------------------

    def record_request(self, **kw: Any) -> None:
        cols = (
            "session_id", "ts", "provider", "model", "streamed", "status_code",
            "tokens_in", "tokens_out", "cache_read", "cache_write", "tokens_saved",
            "latency_ms_upstream", "latency_ms_overhead", "faults",
        )
        kw.setdefault("ts", int(time.time()))
        kw["faults"] = kw.get("faults") or 0
        kw["tokens_saved"] = kw.get("tokens_saved") or 0
        vals = [kw.get(c) for c in cols]
        self.conn.execute(
            f"INSERT INTO requests({','.join(cols)}) VALUES({','.join('?' * len(cols))})",
            vals,
        )
        self.conn.commit()

    def stats(self, since_ts: int | None = None) -> dict[str, Any]:
        where = "WHERE ts >= ?" if since_ts else ""
        args = (since_ts,) if since_ts else ()
        row = self.conn.execute(
            f"""SELECT COUNT(*) AS requests,
                       COUNT(DISTINCT session_id) AS sessions,
                       COALESCE(SUM(tokens_in),0) AS tokens_in,
                       COALESCE(SUM(tokens_out),0) AS tokens_out,
                       COALESCE(SUM(cache_read),0) AS cache_read,
                       COALESCE(SUM(tokens_saved),0) AS tokens_saved,
                       COALESCE(SUM(faults),0) AS faults,
                       COALESCE(AVG(latency_ms_overhead),0) AS avg_overhead_ms
                FROM requests {where}""",
            args,
        ).fetchone()
        return dict(row)
