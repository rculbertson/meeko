"""Unit tests for `meeko.tools.session`.

Covers the `SessionManager` flag, the tool definition shape, and the
handler dispatching path.
"""

import pytest

from meeko.tools.session import (
    SessionManager,
    get_tool_definitions,
    handle,
)


def test_session_manager_defaults_to_not_ending():
    manager = SessionManager()
    assert manager.should_end() is False


def test_request_end_sets_flag():
    manager = SessionManager()
    manager.request_end()
    assert manager.should_end() is True


def test_clear_resets_flag():
    manager = SessionManager()
    manager.request_end()
    manager.clear()
    assert manager.should_end() is False


def test_tool_definition_shape():
    defs = get_tool_definitions()
    assert len(defs) == 1
    end = defs[0]
    assert end["name"] == "end_session"
    assert "description" in end and end["description"]
    assert end["input_schema"] == {"type": "object", "properties": {}}


async def test_handle_end_session_flips_flag_and_returns_ack():
    manager = SessionManager()
    result = await handle("end_session", {}, manager=manager)
    assert manager.should_end() is True
    assert "Session ended" in result


async def test_handle_unknown_function_returns_descriptive_string():
    manager = SessionManager()
    result = await handle("not_a_real_tool", {}, manager=manager)
    assert manager.should_end() is False
    assert "Unknown session function" in result


@pytest.mark.parametrize("args", [{}, {"unused": "value"}])
async def test_handle_ignores_args(args):
    """The tool takes no arguments; extra ones must not break dispatch."""
    manager = SessionManager()
    await handle("end_session", args, manager=manager)
    assert manager.should_end() is True
