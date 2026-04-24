"""Unit tests for `meeko.tools.session`.

Covers the `SessionManager` flags, the tool definition shape, and the
handler dispatching path for both `end_session` and `new_session`.
"""

import pytest

from meeko.tools.session import (
    SessionManager,
    get_tool_definitions,
    handle,
)


def test_session_manager_defaults_to_neither_flag_set():
    manager = SessionManager()
    assert manager.should_end() is False
    assert manager.should_start_new() is False


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
    manager.request_new()
    manager.clear()
    assert manager.should_end() is False
    assert manager.should_start_new() is False


def test_tool_definition_shape():
    defs = get_tool_definitions()
    assert len(defs) == 2
    by_name = {d["name"]: d for d in defs}
    assert set(by_name) == {"end_session", "new_session"}
    for d in defs:
        assert "description" in d and d["description"]
        assert d["input_schema"] == {"type": "object", "properties": {}}


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
