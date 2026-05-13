"""Unit tests for `meeko.tools.session`.

Covers the `SessionManager` flags, the tool definition shape, and the
handler dispatching path for all session tools.
"""

import pytest

from meeko.sessions import SessionStore
from meeko.tools.session import (
    SessionManager,
    get_tool_definitions,
    handle,
)


def test_session_manager_defaults_to_neither_flag_set():
    manager = SessionManager()
    assert manager.should_end() is False
    assert manager.should_start_new() is False
    assert manager.should_load() is False
    assert manager.get_load_target() is None


def test_request_end_sets_end_flag_only():
    manager = SessionManager()
    manager.request_end()
    assert manager.should_end() is True
    assert manager.should_start_new() is False


def test_request_new_sets_new_flag_only():
    manager = SessionManager()
    manager.request_new()
    assert manager.should_start_new() is True
    assert manager.should_end() is False


def test_clear_resets_both_flags():
    manager = SessionManager()
    manager.request_end()
    manager.clear()
    manager.request_new()
    manager.clear()
    assert manager.should_end() is False
    assert manager.should_start_new() is False


def test_request_new_overrides_prior_request_end():
    """end/new contradict each other; the later request wins so the
    orchestrator never sees both flags set at once."""
    manager = SessionManager()
    manager.request_end()
    manager.request_new()
    assert manager.should_end() is False
    assert manager.should_start_new() is True


def test_request_end_overrides_prior_request_new():
    manager = SessionManager()
    manager.request_new()
    manager.request_end()
    assert manager.should_start_new() is False
    assert manager.should_end() is True


def test_request_load_sets_load_target():
    manager = SessionManager()
    manager.request_load("abc-123")
    assert manager.should_load() is True
    assert manager.get_load_target() == "abc-123"
    assert manager.should_start_new() is False


def test_request_load_clears_new_flag():
    manager = SessionManager()
    manager.request_new()
    manager.request_load("abc-123")
    assert manager.should_start_new() is False
    assert manager.should_load() is True


def test_request_load_coexists_with_end_flag():
    """end+load can both be set: Sonnet chains end_session then load_session."""
    manager = SessionManager()
    manager.request_end()
    manager.request_load("abc-123")
    assert manager.should_end() is True
    assert manager.should_load() is True


def test_request_new_clears_load_target():
    manager = SessionManager()
    manager.request_load("abc-123")
    manager.request_new()
    assert manager.should_load() is False
    assert manager.get_load_target() is None


def test_clear_resets_load_target():
    manager = SessionManager()
    manager.request_load("abc-123")
    manager.clear()
    assert manager.should_load() is False
    assert manager.get_load_target() is None


def test_tool_definition_shape():
    defs = get_tool_definitions()
    assert len(defs) == 4
    by_name = {d["name"]: d for d in defs}
    assert set(by_name) == {
        "end_session",
        "new_session",
        "list_sessions",
        "load_session",
    }
    for d in defs:
        assert "description" in d and d["description"]
    assert by_name["end_session"]["input_schema"] == {
        "type": "object",
        "properties": {},
    }
    assert by_name["new_session"]["input_schema"] == {
        "type": "object",
        "properties": {},
    }
    assert "query" in by_name["list_sessions"]["input_schema"]["properties"]
    assert "id" in by_name["load_session"]["input_schema"]["properties"]


async def test_handle_end_session_flips_flag_and_returns_ack():
    manager = SessionManager()
    result = await handle("end_session", {}, manager=manager)
    assert manager.should_end() is True
    assert manager.should_start_new() is False
    assert "Session ended" in result


async def test_handle_new_session_flips_flag_and_returns_ack():
    manager = SessionManager()
    result = await handle("new_session", {}, manager=manager)
    assert manager.should_start_new() is True
    assert manager.should_end() is False
    assert "fresh session" in result.lower()


async def test_handle_unknown_function_returns_descriptive_string():
    manager = SessionManager()
    result = await handle("not_a_real_tool", {}, manager=manager)
    assert manager.should_end() is False
    assert manager.should_start_new() is False
    assert "Unknown session function" in result


@pytest.mark.parametrize("fn", ["end_session", "new_session"])
@pytest.mark.parametrize("args", [{}, {"unused": "value"}])
async def test_handle_ignores_args(fn, args):
    """Both tools take no arguments; extra ones must not break dispatch."""
    manager = SessionManager()
    await handle(fn, args, manager=manager)
    if fn == "end_session":
        assert manager.should_end() is True
    else:
        assert manager.should_start_new() is True


async def test_handle_list_sessions_no_store_returns_error():
    manager = SessionManager()
    result = await handle(
        "list_sessions", {"query": "todo"}, manager=manager, store=None
    )
    assert "not available" in result.lower()


async def test_handle_list_sessions_empty_query_returns_prompt():
    manager = SessionManager()
    result = await handle("list_sessions", {"query": ""}, manager=manager)
    assert "query" in result.lower() or "date" in result.lower()


async def test_handle_list_sessions_no_args_prompts_user():
    """At least one of query/since/until is required."""
    manager = SessionManager()
    result = await handle("list_sessions", {}, manager=manager)
    assert "query" in result.lower() or "date" in result.lower()


async def test_handle_list_sessions_invalid_date_returns_error(tmp_path):
    store = SessionStore.open(tmp_path / "meeko.db")
    try:
        manager = SessionManager()
        result = await handle(
            "list_sessions",
            {"since": "yesterday"},
            manager=manager,
            store=store,
        )
        assert "invalid" in result.lower()
    finally:
        await store.close()


async def test_handle_list_sessions_date_only_finds_unfinalized(tmp_path):
    """A date-only call must find an active session that hasn't been
    summarized into the FTS table yet — this is the bug the feature fixes."""
    from datetime import date, timedelta

    store = SessionStore.open(tmp_path / "meeko.db")
    try:
        sid = await store.create_session("query")
        await store.persist_turn(sid, "user", "hi")
        today = date.today().isoformat()
        tomorrow = (date.today() + timedelta(days=1)).isoformat()
        manager = SessionManager()
        result = await handle(
            "list_sessions",
            {"since": today, "until": tomorrow},
            manager=manager,
            store=store,
        )
        assert sid in result
        assert "Found 1" in result
    finally:
        await store.close()


async def test_handle_list_sessions_date_no_results(tmp_path):
    store = SessionStore.open(tmp_path / "meeko.db")
    try:
        manager = SessionManager()
        result = await handle(
            "list_sessions",
            {"since": "1970-01-01", "until": "1971-01-01"},
            manager=manager,
            store=store,
        )
        assert "No sessions found" in result
    finally:
        await store.close()


async def test_handle_list_sessions_no_results(tmp_path):
    store = SessionStore.open(tmp_path / "meeko.db")
    try:
        manager = SessionManager()
        result = await handle(
            "list_sessions", {"query": "xyzzy"}, manager=manager, store=store
        )
        assert "No sessions found" in result
    finally:
        await store.close()


async def test_handle_list_sessions_returns_formatted_results(tmp_path):
    store = SessionStore.open(tmp_path / "meeko.db")
    try:
        sid = await store.create_session("query")
        await store.update_session_metadata(
            sid,
            title="Supabase planning",
            summary="Discussed Supabase vs SQLite.",
            transcript="supabase schema design",
        )
        manager = SessionManager()
        result = await handle(
            "list_sessions", {"query": "supabase"}, manager=manager, store=store
        )
        assert "Supabase planning" in result
        assert sid in result
    finally:
        await store.close()


async def test_handle_load_session_no_store_returns_error():
    manager = SessionManager()
    result = await handle("load_session", {"id": "abc"}, manager=manager, store=None)
    assert "not available" in result.lower()


async def test_handle_load_session_unknown_id_returns_error(tmp_path):
    store = SessionStore.open(tmp_path / "meeko.db")
    try:
        manager = SessionManager()
        result = await handle(
            "load_session", {"id": "does-not-exist"}, manager=manager, store=store
        )
        assert "No session found" in result
        assert manager.should_load() is False
    finally:
        await store.close()


async def test_handle_load_session_sets_flag(tmp_path):
    store = SessionStore.open(tmp_path / "meeko.db")
    try:
        sid = await store.create_session("query")
        await store.update_session_metadata(
            sid, title="Todo app", summary="s", transcript="t"
        )
        manager = SessionManager()
        result = await handle(
            "load_session",
            {"id": sid},
            manager=manager,
            store=store,
            current_session_id="other-session",
        )
        assert manager.should_load() is True
        assert manager.get_load_target() == sid
        # Tool result should reference the title, not the raw UUID.
        assert "Todo app" in result
        assert sid not in result
    finally:
        await store.close()


async def test_handle_load_session_rejects_current_session(tmp_path):
    """Loading the already-active session is a no-op with a friendly reply."""
    store = SessionStore.open(tmp_path / "meeko.db")
    try:
        sid = await store.create_session("query")
        manager = SessionManager()
        result = await handle(
            "load_session",
            {"id": sid},
            manager=manager,
            store=store,
            current_session_id=sid,
        )
        assert manager.should_load() is False
        assert "already loaded" in result.lower()
    finally:
        await store.close()


async def test_handle_load_session_untitled_session_uses_placeholder(tmp_path):
    """Sessions without a title (never summarized) still get a readable result."""
    store = SessionStore.open(tmp_path / "meeko.db")
    try:
        sid = await store.create_session("query")
        manager = SessionManager()
        result = await handle(
            "load_session",
            {"id": sid},
            manager=manager,
            store=store,
            current_session_id="other",
        )
        assert "untitled" in result.lower()
    finally:
        await store.close()
