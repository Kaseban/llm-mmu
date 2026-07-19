-- mmu store schema (M0: sessions, prefix_hashes, requests; M1+: pages, embeddings)
PRAGMA journal_mode = WAL;

CREATE TABLE IF NOT EXISTS sessions (
  session_id    TEXT PRIMARY KEY,
  api_hash      TEXT NOT NULL,          -- sha256(api key): namespace only, never the key
  provider      TEXT NOT NULL,          -- 'anthropic' | 'openai'
  system_hash   TEXT NOT NULL,          -- hash of system prompt + tools signature
  created_at    INTEGER NOT NULL,
  last_seen_at  INTEGER NOT NULL,
  turn_count    INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS prefix_hashes (
  session_id  TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
  msg_index   INTEGER NOT NULL,
  chain_hash  TEXT NOT NULL,            -- H(prev_chain_hash || canonical(msg))
  PRIMARY KEY (session_id, msg_index)
);
CREATE INDEX IF NOT EXISTS idx_prefix ON prefix_hashes(chain_hash);

CREATE TABLE IF NOT EXISTS requests (
  id                   INTEGER PRIMARY KEY AUTOINCREMENT,
  session_id           TEXT,
  ts                   INTEGER NOT NULL,
  provider             TEXT NOT NULL,
  model                TEXT,
  streamed             INTEGER NOT NULL DEFAULT 0,
  status_code          INTEGER,
  tokens_in            INTEGER,
  tokens_out           INTEGER,
  cache_read           INTEGER,
  cache_write          INTEGER,
  tokens_saved         INTEGER NOT NULL DEFAULT 0,
  latency_ms_upstream  INTEGER,
  latency_ms_overhead  INTEGER,
  faults               INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_requests_session ON requests(session_id, ts);

CREATE TABLE IF NOT EXISTS pages (
  page_id          TEXT PRIMARY KEY,
  session_id       TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
  kind             TEXT NOT NULL,       -- 'tool_result' | 'turn' | 'file_dump' | ...
  state            TEXT NOT NULL,       -- 'resident' | 'summary' | 'stub' | 'disk'
  pinned           INTEGER NOT NULL DEFAULT 0,
  gc_flag          INTEGER NOT NULL DEFAULT 0,
  msg_index        INTEGER,
  block_span       TEXT,
  tool_use_id      TEXT,
  body             BLOB,
  body_sha         TEXT,
  summary          TEXT,
  tok_full         INTEGER,
  tok_summary      INTEGER,
  tok_stub         INTEGER,
  created_turn     INTEGER,
  last_access_turn INTEGER,
  ref_count        INTEGER NOT NULL DEFAULT 0,
  fault_count      INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_pages_session ON pages(session_id, state);

CREATE TABLE IF NOT EXISTS embeddings (
  page_id TEXT PRIMARY KEY REFERENCES pages(page_id) ON DELETE CASCADE,
  model   TEXT,
  vec     BLOB
);
