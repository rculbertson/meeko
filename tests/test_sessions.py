"""Tests for the SQLite-backed SessionStore."""

import json
import sqlite3

import pytest

from meeko.sessions import SessionStore, default_db_path


@pytest.fixture
async def store(tmp_path):
    s = SessionStore.open(tmp_path / "meeko.db")
    yield s
    await s.close()


async def test_open_creates_schema(tmp_path):
    db = tmp_path / "meeko.db"
    s = SessionStore.open(db)
    await s.close()

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
        await s.close()


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


async def test_get_latest_returns_none_when_empty(store):
    assert await store.get_latest_session() is None


async def test_get_latest_returns_most_recent_by_last_active(store):
    a = await store.create_session("default")
    b = await store.create_session("pirate")
    # b was created later, so b should be latest
    latest = await store.get_latest_session()
    assert latest["id"] == b

    # Persisting a turn on `a` moves it to the top.
    await store.persist_turn(a, "user", "ping")
    latest = await store.get_latest_session()
    assert latest["id"] == a
    assert latest["profile_name"] == "default"


async def test_get_session_by_id(store):
    sid = await store.create_session("default")
    row = await store.get_session(sid)
    assert row["id"] == sid
    assert row["profile_name"] == "default"
    assert await store.get_session("no-such-id") is None


async def test_list_sessions_orders_and_counts(store):
    a = await store.create_session("default")
    b = await store.create_session("pirate")
    await store.persist_turn(a, "user", "hi")
    await store.persist_turn(a, "assistant", [{"type": "text", "text": "hey"}])
    await store.persist_turn(b, "user", "yo")

    rows = await store.list_sessions()
    # b was created after a and has the later turn write; but a's last
    # turn is the most recent. Whichever has higher last_active comes
    # first — so just check ordering by last_active and turn counts.
    assert len(rows) == 2
    by_id = {r["id"]: r for r in rows}
    assert by_id[a]["turn_count"] == 2
    assert by_id[b]["turn_count"] == 1
    # Ordered by last_active DESC.
    assert rows[0]["last_active"] >= rows[1]["last_active"]


async def test_load_turns_roundtrips_mixed_content(store):
    sid = await store.create_session("default")
    await store.persist_turn(sid, "user", "hello")
    blocks = [
        {"type": "text", "text": "Hi."},
        {"type": "tool_use", "id": "t1", "name": "echo", "input": {"v": 1}},
    ]
    await store.persist_turn(sid, "assistant", blocks)

    turns = await store.load_turns(sid)
    assert turns == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": blocks},
    ]


async def test_touch_session_updates_last_active(store):
    sid = await store.create_session("default")
    before = (await store.get_session(sid))["last_active"]
    await store.touch_session(sid)
    after = (await store.get_session(sid))["last_active"]
    assert after >= before


async def test_update_session_metadata_writes_row_and_fts(tmp_path, store):
    sid = await store.create_session("default")
    await store.update_session_metadata(
        sid,
        title="Todo app prototype",
        summary="Compared Supabase and SQLite as backends for a todo app.",
        transcript="USER: we should build a todo app\nASSISTANT: sounds good",
    )

    conn = sqlite3.connect(str(tmp_path / "meeko.db"))
    try:
        row = conn.execute(
            "SELECT title, summary FROM sessions WHERE id = ?", (sid,)
        ).fetchone()
        fts_rows = conn.execute(
            "SELECT session_id, title, summary, transcript FROM sessions_fts "
            "WHERE session_id = ?",
            (sid,),
        ).fetchall()
    finally:
        conn.close()

    assert row == (
        "Todo app prototype",
        "Compared Supabase and SQLite as backends for a todo app.",
    )
    assert len(fts_rows) == 1
    assert fts_rows[0] == (
        sid,
        "Todo app prototype",
        "Compared Supabase and SQLite as backends for a todo app.",
        "USER: we should build a todo app\nASSISTANT: sounds good",
    )


async def test_update_session_metadata_is_idempotent(tmp_path, store):
    """Re-summarizing a session must replace the FTS row, not duplicate."""
    sid = await store.create_session("default")
    await store.update_session_metadata(
        sid, title="first", summary="old", transcript="old text"
    )
    await store.update_session_metadata(
        sid, title="second", summary="new", transcript="new text"
    )

    conn = sqlite3.connect(str(tmp_path / "meeko.db"))
    try:
        count = conn.execute(
            "SELECT COUNT(*) FROM sessions_fts WHERE session_id = ?", (sid,)
        ).fetchone()[0]
        latest = conn.execute(
            "SELECT title, summary, transcript FROM sessions_fts WHERE session_id = ?",
            (sid,),
        ).fetchone()
    finally:
        conn.close()
    assert count == 1
    assert latest == ("second", "new", "new text")


async def test_fts_match_finds_by_summary_and_transcript(tmp_path, store):
    sid_a = await store.create_session("default")
    sid_b = await store.create_session("default")
    await store.update_session_metadata(
        sid_a,
        title="Todo app",
        summary="Compared Supabase and SQLite.",
        transcript="discussed Redis as a cache layer",
    )
    await store.update_session_metadata(
        sid_b,
        title="Marketing brainstorm",
        summary="Ideas for the spring launch.",
        transcript="email campaigns and landing pages",
    )

    conn = sqlite3.connect(str(tmp_path / "meeko.db"))
    try:
        # Summary column match
        by_summary = conn.execute(
            "SELECT session_id FROM sessions_fts WHERE sessions_fts MATCH 'Supabase'"
        ).fetchall()
        # Transcript-only match (word only appears in transcript col)
        by_transcript = conn.execute(
            "SELECT session_id FROM sessions_fts WHERE sessions_fts MATCH 'Redis'"
        ).fetchall()
    finally:
        conn.close()

    assert [r[0] for r in by_summary] == [sid_a]
    assert [r[0] for r in by_transcript] == [sid_a]


async def test_search_sessions_returns_ranked_results(store):
    sid_a = await store.create_session("default")
    sid_b = await store.create_session("default")
    await store.update_session_metadata(
        sid_a,
        title="Todo app prototype",
        summary="Compared Supabase and SQLite as backends.",
        transcript="lots of back and forth about the schema",
    )
    await store.update_session_metadata(
        sid_b,
        title="Marketing brainstorm",
        summary="Ideas for the spring launch campaign.",
        transcript="email flows and landing page copy",
    )

    results = await store.search_sessions("todo app")
    assert len(results) >= 1
    assert results[0]["session_id"] == sid_a
    assert results[0]["title"] == "Todo app prototype"
    assert "last_active" in results[0]


async def test_search_sessions_title_outranks_transcript(store):
    """Title matches must rank above transcript-only matches.

    Regression: bm25() weights are positional across *all* columns including
    UNINDEXED ones, so a missing leading 0.0 for session_id silently shifts
    every weight one column to the left.
    """
    sid_title = await store.create_session("default")
    sid_transcript = await store.create_session("default")
    await store.update_session_metadata(
        sid_title,
        title="Redis caching strategy",
        summary="Picking a cache layer.",
        transcript="discussed several options at length",
    )
    await store.update_session_metadata(
        sid_transcript,
        title="Weekend trip planning",
        summary="Where to go and what to pack.",
        transcript="Redis came up briefly as an aside",
    )
    results = await store.search_sessions("Redis")
    assert [r["session_id"] for r in results] == [sid_title, sid_transcript]


async def test_search_sessions_transcript_fallback(store):
    sid = await store.create_session("default")
    await store.update_session_metadata(
        sid,
        title="Project planning",
        summary="High-level roadmap discussion.",
        transcript="we also considered using Redis for caching",
    )
    results = await store.search_sessions("Redis")
    assert len(results) == 1
    assert results[0]["session_id"] == sid


async def test_search_sessions_no_match(store):
    sid = await store.create_session("default")
    await store.update_session_metadata(
        sid,
        title="Cooking tips",
        summary="How to make pasta.",
        transcript="pasta water",
    )
    results = await store.search_sessions("blockchain")
    assert results == []


async def test_search_sessions_empty_query_returns_empty(store):
    sid = await store.create_session("default")
    await store.update_session_metadata(
        sid, title="Something", summary="Stuff.", transcript="things"
    )
    results = await store.search_sessions("")
    assert results == []


async def test_search_sessions_strips_fts_operators(store):
    """Query characters like * " : ^ must not cause an FTS5 syntax error."""
    sid = await store.create_session("default")
    await store.update_session_metadata(
        sid, title="Supabase chat", summary="Database discussion.", transcript="schema"
    )
    # These would explode if passed raw to FTS5.
    for query in ["supabase*", '"supabase"', "title:supabase", "^supabase"]:
        results = await store.search_sessions(query)
        assert isinstance(results, list), (
            f"query {query!r} raised instead of returning list"
        )


async def test_search_sessions_respects_limit(store):
    for i in range(6):
        sid = await store.create_session("default")
        await store.update_session_metadata(
            sid,
            title=f"Session {i}",
            summary="about widgets and gadgets",
            transcript="widgets",
        )
    results = await store.search_sessions("widgets", limit=3)
    assert len(results) <= 3


def test_default_db_path_honors_env_var(monkeypatch, tmp_path):
    monkeypatch.setenv("MEEKO_DB_PATH", str(tmp_path / "custom.db"))
    assert default_db_path() == tmp_path / "custom.db"


def test_default_db_path_defaults_to_home(monkeypatch):
    monkeypatch.delenv("MEEKO_DB_PATH", raising=False)
    path = default_db_path()
    assert path.name == "meeko.db"
    assert path.parent.name == ".meeko"
