"""Unit tests for `meeko.tools.session`.

Covers the `SessionManager` flags, the tool definition shape, and the
handler dispatching path for all session tools.
"""

from datetime import datetime

import pytest

from meeko.sessions import UNTITLED, SessionStore
from meeko.tools.session import (
    SessionManager,
    _format_results,
    _parse_list_args,
    get_tool_definitions,
    handle,
)


@pytest.fixture
async def store(tmp_path):
    """An open SessionStore on a temp database, closed on teardown.

    Every handler takes a real store: the session tools are only ever
    reached through `main.py`, which always has one open.
    """
    store = SessionStore.open(tmp_path / "meeko.db")
    try:
        yield store
    finally:
        await store.close()


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
        assert d.get("description")
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


async def test_handle_end_session_flips_flag_and_returns_ack(store):
    manager = SessionManager()
    result = await handle("end_session", {}, manager=manager, store=store)
    assert manager.should_end() is True
    assert manager.should_start_new() is False
    assert "Session ended" in result


async def test_handle_new_session_flips_flag_and_returns_ack(store):
    manager = SessionManager()
    result = await handle("new_session", {}, manager=manager, store=store)
    assert manager.should_start_new() is True
    assert manager.should_end() is False
    assert "fresh session" in result.lower()


async def test_handle_unknown_function_returns_descriptive_string(store):
    manager = SessionManager()
    result = await handle("not_a_real_tool", {}, manager=manager, store=store)
    assert manager.should_end() is False
    assert manager.should_start_new() is False
    assert "Unknown session function" in result


@pytest.mark.parametrize("fn", ["end_session", "new_session"])
@pytest.mark.parametrize("args", [{}, {"unused": "value"}])
async def test_handle_ignores_args(fn, args, store):
    """Both tools take no arguments; extra ones must not break dispatch."""
    manager = SessionManager()
    await handle(fn, args, manager=manager, store=store)
    if fn == "end_session":
        assert manager.should_end() is True
    else:
        assert manager.should_start_new() is True


async def test_handle_list_sessions_empty_query_returns_prompt(store):
    manager = SessionManager()
    result = await handle("list_sessions", {"query": ""}, manager=manager, store=store)
    assert "query" in result.lower() or "date" in result.lower()


async def test_handle_list_sessions_no_args_prompts_user(store):
    """At least one of query/since/until is required."""
    manager = SessionManager()
    result = await handle("list_sessions", {}, manager=manager, store=store)
    assert "query" in result.lower() or "date" in result.lower()


async def test_handle_list_sessions_invalid_date_returns_error(store):
    manager = SessionManager()
    result = await handle(
        "list_sessions",
        {"since": "yesterday"},
        manager=manager,
        store=store,
    )
    assert "invalid" in result.lower()


async def test_handle_list_sessions_date_only_finds_unfinalized(store):
    """A date-only call must find an active session that hasn't been
    summarized into the FTS table yet — this is the bug the feature fixes."""
    from datetime import date, timedelta

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


async def test_handle_list_sessions_date_no_results(store):
    manager = SessionManager()
    result = await handle(
        "list_sessions",
        {"since": "1970-01-01", "until": "1971-01-01"},
        manager=manager,
        store=store,
    )
    assert "No sessions found" in result


async def test_handle_list_sessions_no_results(store):
    manager = SessionManager()
    result = await handle(
        "list_sessions", {"query": "xyzzy"}, manager=manager, store=store
    )
    assert "No sessions found" in result


async def test_handle_list_sessions_returns_formatted_results(store):
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


async def test_handle_load_session_unknown_id_returns_error(store):
    manager = SessionManager()
    result = await handle(
        "load_session", {"id": "does-not-exist"}, manager=manager, store=store
    )
    assert "No session found" in result
    assert manager.should_load() is False


async def test_handle_load_session_sets_flag(store):
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


async def test_handle_load_session_rejects_current_session(store):
    """Loading the already-active session is a no-op with a friendly reply."""
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


async def test_handle_list_sessions_untitled_session_uses_placeholder(store):
    """A date-range search reads the base table, so it can return a session
    that hasn't been summarized yet. It must use the same placeholder as the
    load path and the CLI (these had drifted to "(no title)" vs "(untitled)")."""
    sid = await store.create_session("query")
    await store.persist_turn(sid, "user", "hi")
    result = await handle(
        "list_sessions",
        {"since": datetime.now().astimezone().strftime("%Y-%m-%d")},
        manager=SessionManager(),
        store=store,
    )
    assert f'"{UNTITLED}"' in result
    assert sid in result


async def test_handle_load_session_untitled_session_uses_placeholder(store):
    """Sessions without a title (never summarized) still get a readable result."""
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


# --- _parse_list_args / _format_results ---------------------------------
#
# Sonnet fills these arguments in from speech, so the parser is the place
# sloppy input gets turned into either usable criteria or something Meeko
# can say back.


def test_parse_list_args_strips_and_treats_empty_as_absent():
    parsed = _parse_list_args({"query": "  supabase  ", "since": "  ", "until": ""})
    assert not isinstance(parsed, str)
    assert parsed.query == "supabase"
    assert parsed.since is None and parsed.until is None
    assert parsed.since_iso is None and parsed.until_iso is None


def test_parse_list_args_coerces_non_strings():
    """Tool arguments arrive as JSON, so a number is possible."""
    parsed = _parse_list_args({"query": 42})
    assert not isinstance(parsed, str)
    assert parsed.query == "42"


def test_parse_list_args_reads_null_as_absent():
    """Sonnet may send an explicit null for an optional field rather
    than omitting it. Coerced with `str()` that becomes the literal
    "None": a bogus search term, or a date that fails to parse."""
    parsed = _parse_list_args({"query": "todo", "since": None, "until": None})
    assert not isinstance(parsed, str), parsed
    assert parsed.query == "todo"
    assert parsed.since is None and parsed.since_iso is None
    assert parsed.until is None and parsed.until_iso is None


def test_parse_list_args_null_query_is_not_a_search_term():
    parsed = _parse_list_args({"query": None, "since": "2026-09-01"})
    assert not isinstance(parsed, str)
    assert parsed.query is None


def test_parse_list_args_all_null_prompts_for_criteria():
    """The prompt, not the invalid-date message — which is what three
    nulls coerced to the string "None" would produce."""
    result = _parse_list_args({"query": None, "since": None, "until": None})
    assert result == "Please provide a search query or date range."


def test_parse_list_args_requires_at_least_one_criterion():
    result = _parse_list_args({})
    assert isinstance(result, str)
    assert "query" in result.lower() or "date" in result.lower()


def test_parse_list_args_rejects_an_unparseable_date():
    result = _parse_list_args({"since": "yesterday"})
    assert isinstance(result, str)
    assert "invalid" in result.lower()


def test_parse_list_args_converts_dates_to_a_utc_window():
    """The raw local dates are kept for the spoken description; the ISO
    pair is what the query compares against."""
    parsed = _parse_list_args({"since": "2026-09-01", "until": "2026-09-30"})
    assert not isinstance(parsed, str)
    assert parsed.since == "2026-09-01" and parsed.until == "2026-09-30"
    assert parsed.since_iso is not None and parsed.until_iso is not None
    assert parsed.since_iso < parsed.until_iso
    assert parsed.since_iso.endswith("+00:00")


def test_format_results_numbers_from_one_and_includes_ids():
    results = [
        {
            "title": "Supabase planning",
            "last_active": "2026-09-01T12:00:00+00:00",
            "session_id": "sid-a",
        },
        {
            "title": "Timer bugs",
            "last_active": "2026-09-02T12:00:00+00:00",
            "session_id": "sid-b",
        },
    ]
    out = _format_results(results, "matching 'x'")
    assert out.startswith("Found 2 session(s) matching 'x':")
    lines = out.split("\n")[1:]
    assert lines[0].startswith('1. "Supabase planning"')
    assert lines[1].startswith('2. "Timer bugs"')
    assert "sid-a" in lines[0] and "sid-b" in lines[1]


def test_format_results_uses_the_untitled_placeholder():
    """A session that was never summarized has no title."""
    out = _format_results(
        [{"title": None, "last_active": None, "session_id": "sid-a"}],
        "since 2026-09-01",
    )
    assert f'"{UNTITLED}"' in out
