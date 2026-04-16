from unittest.mock import AsyncMock, MagicMock

from deepgram.agent.v1.types import AgentV1SettingsAgentThinkOneItemFunctionsItem

from meeko.tools.dispatch import ToolDispatcher


def _make_definition(name: str) -> AgentV1SettingsAgentThinkOneItemFunctionsItem:
    return AgentV1SettingsAgentThinkOneItemFunctionsItem(
        name=name,
        description=f"Test tool: {name}",
        parameters={"type": "object", "properties": {}},
    )


def _make_function_call_request(fn_name: str, fn_id: str = "123", args: str = "{}"):
    """Create a mock FunctionCallRequest message."""
    fn = MagicMock()
    fn.id = fn_id
    fn.name = fn_name
    fn.arguments = args

    msg = MagicMock()
    msg.type = "FunctionCallRequest"
    msg.functions = [fn]
    return msg


async def test_dispatcher_routes_to_correct_handler():
    """Each function name routes to the handler registered with it."""
    dispatcher = ToolDispatcher()

    handler_a = AsyncMock(return_value="result_a")
    handler_b = AsyncMock(return_value="result_b")

    dispatcher.register([_make_definition("tool_a")], handler_a)
    dispatcher.register([_make_definition("tool_b")], handler_b)

    conn = MagicMock()
    conn.send_function_call_response = AsyncMock()

    # Call tool_a
    msg = _make_function_call_request("tool_a")
    await dispatcher.handle_function_call_request(msg, conn)

    handler_a.assert_called_once_with("tool_a", {}, conn)
    handler_b.assert_not_called()

    # Verify response was sent with correct content
    response = conn.send_function_call_response.call_args[0][0]
    assert response.content == "result_a"
    assert response.name == "tool_a"


async def test_dispatcher_unknown_function():
    """Calling an unregistered function returns an error message."""
    dispatcher = ToolDispatcher()

    conn = MagicMock()
    conn.send_function_call_response = AsyncMock()

    msg = _make_function_call_request("nonexistent")
    await dispatcher.handle_function_call_request(msg, conn)

    response = conn.send_function_call_response.call_args[0][0]
    assert "Unknown function" in response.content


async def test_dispatcher_get_all_definitions():
    """get_all_definitions returns definitions from all registered modules."""
    dispatcher = ToolDispatcher()

    handler = AsyncMock()
    dispatcher.register(
        [_make_definition("tool_a"), _make_definition("tool_b")], handler
    )
    dispatcher.register([_make_definition("tool_c")], AsyncMock())

    defs = dispatcher.get_all_definitions()
    names = [d.name for d in defs]
    assert names == ["tool_a", "tool_b", "tool_c"]


async def test_dispatcher_parses_json_arguments():
    """Arguments JSON string is parsed into a dict before calling the handler."""
    dispatcher = ToolDispatcher()
    handler = AsyncMock(return_value="ok")
    dispatcher.register([_make_definition("my_tool")], handler)

    conn = MagicMock()
    conn.send_function_call_response = AsyncMock()

    msg = _make_function_call_request("my_tool", args='{"key": "value"}')
    await dispatcher.handle_function_call_request(msg, conn)

    handler.assert_called_once_with("my_tool", {"key": "value"}, conn)
