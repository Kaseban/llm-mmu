from pathlib import Path

from mmu.sessions import chain_hashes, resolve_session, system_hash
from mmu.store.db import Store


def _store(tmp_path: Path) -> Store:
    return Store(tmp_path / "store")


def test_chain_is_prefix_stable():
    msgs = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
        {"role": "user", "content": "more"},
    ]
    full = chain_hashes(msgs)
    prefix = chain_hashes(msgs[:2])
    assert full[:2] == prefix


def test_same_session_across_turns(tmp_path):
    store = _store(tmp_path)
    sys_h = system_hash("you are helpful", None)
    turn1 = [{"role": "user", "content": "hi"}]
    sid1 = resolve_session(store, turn1, "sk-test", "anthropic", sys_h)
    # Client resends full history + new turns (stateless harness pattern).
    turn2 = turn1 + [
        {"role": "assistant", "content": "hello"},
        {"role": "user", "content": "do the thing"},
    ]
    sid2 = resolve_session(store, turn2, "sk-test", "anthropic", sys_h)
    assert sid1 == sid2
    store.close()


def test_different_key_is_different_session(tmp_path):
    store = _store(tmp_path)
    sys_h = system_hash(None, None)
    msgs = [{"role": "user", "content": "hi"}]
    sid_a = resolve_session(store, msgs, "sk-a", "anthropic", sys_h)
    sid_b = resolve_session(store, msgs, "sk-b", "anthropic", sys_h)
    assert sid_a != sid_b
    store.close()


def test_fresh_conversation_is_new_session(tmp_path):
    store = _store(tmp_path)
    sys_h = system_hash(None, None)
    sid1 = resolve_session(
        store, [{"role": "user", "content": "topic one"}], "sk", "anthropic", sys_h
    )
    sid2 = resolve_session(
        store, [{"role": "user", "content": "topic two"}], "sk", "anthropic", sys_h
    )
    assert sid1 != sid2
    store.close()
