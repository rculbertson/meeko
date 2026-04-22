"""Tests for the SQLite-backed SessionStore."""

import json
import sqlite3

import pytest

from meeko.sessions import SessionStore, default_db_path


@pytest.fixture
def store(tmp_path):
    s = SessionStore.open(tmp_path / "meeko.db")
    yield s
    s.close()


async def test_open_creates_schema(tmp_path):
    db = tmp_path / "meeko.db"
    s = SessionStore.open(db)
    s.close()

    conn = sqlite3.connect(str(db))
    try:
        tables = {
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
    finally:
        conn.close()
    assert {"sessions", "turns"}.issubset(tables)


async def test_open_creates_parent_dir(tmp_path):
    nested = tmp_path / "a" / "b" / "meeko.db"
    s = SessionStore.open(nested)
    try:
        assert nested.exists()
    finally:
        s.close()


async def test_create_session_persists_row(tmp_path, store):
    session_id = await store.create_session("default")
    assert isinstance(session_id, str) and len(session_id) == 36

    conn = sqlite3.connect(str(tmp_path / "meeko.db"))
    try:
        row = conn.execute(
            "SELECT id, profile_name, created_at, last_active FROM sessions"
        ).fetchone()
    finally:
        conn.close()
    assert row[0] == session_id
    assert row[1] == "default"
    assert row[2] == row[3]  # created_at == last_active at creation


async def test_persist_turn_roundtrip_string_content(tmp_path, store):
    session_id = await store.create_session("default")
    await store.persist_turn(session_id, "user", "hello world")

    conn = sqlite3.connect(str(tmp_path / "meeko.db"))
    try:
        rows = conn.execute(
            "SELECT role, content FROM turns WHERE session_id = ? ORDER BY id",
            (session_id,),
        ).fetchall()
    finally:
        conn.close()
    assert rows == [("user", json.dumps("hello world"))]


async def test_persist_turn_roundtrip_block_list_content(tmp_path, store):
    session_id = await store.create_session("default")
    blocks = [
        {"type": "text", "text": "Hi."},
        {"type": "tool_use", "id": "t1", "name": "timer", "input": {"s": 60}},
    ]
    await store.persist_turn(session_id, "assistant", blocks)

    conn = sqlite3.connect(str(tmp_path / "meeko.db"))
    try:
        content = conn.execute(
            "SELECT content FROM turns WHERE session_id = ?", (session_id,)
        ).fetchone()[0]
    finally:
        conn.close()
    assert json.loads(content) == blocks


async def test_persist_turn_updates_last_active(tmp_path, store):
    session_id = await store.create_session("default")

    conn = sqlite3.connect(str(tmp_path / "meeko.db"))
    try:
        before = conn.execute(
            "SELECT last_active FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()[0]
    finally:
        conn.close()

    await store.persist_turn(session_id, "user", "hi")

    conn = sqlite3.connect(str(tmp_path / "meeko.db"))
    try:
        after = conn.execute(
            "SELECT last_active FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()[0]
    finally:
        conn.close()
    assert after >= before


async def test_multiple_sessions_isolated(tmp_path, store):
    a = await store.create_session("default")
    b = await store.create_session("pirate")
    await store.persist_turn(a, "user", "in a")
    await store.persist_turn(b, "user", "in b")
    await store.persist_turn(a, "assistant", [{"type": "text", "text": "ack a"}])

    conn = sqlite3.connect(str(tmp_path / "meeko.db"))
    try:
        a_rows = conn.execute(
            "SELECT role FROM turns WHERE session_id = ? ORDER BY id", (a,)
        ).fetchall()
        b_rows = conn.execute(
            "SELECT role FROM turns WHERE session_id = ? ORDER BY id", (b,)
        ).fetchall()
    finally:
        conn.close()
    assert [r[0] for r in a_rows] == ["user", "assistant"]
    assert [r[0] for r in b_rows] == ["user"]


def test_default_db_path_honors_env_var(monkeypatch, tmp_path):
    monkeypatch.setenv("MEEKO_DB_PATH", str(tmp_path / "custom.db"))
    assert default_db_path() == tmp_path / "custom.db"


def test_default_db_path_defaults_to_home(monkeypatch):
    monkeypatch.delenv("MEEKO_DB_PATH", raising=False)
    path = default_db_path()
    assert path.name == "meeko.db"
    assert path.parent.name == ".meeko"
