from unittest.mock import AsyncMock

from meeko.tools.dispatch import ToolDispatcher


def _defn(name: str) -> dict:
    return {
        "name": name,
        "description": f"Test tool: {name}",
        "input_schema": {"type": "object", "properties": {}},
    }


async def test_dispatcher_routes_to_correct_handler():
    dispatcher = ToolDispatcher()

    handler_a = AsyncMock(return_value="result_a")
    handler_b = AsyncMock(return_value="result_b")

    dispatcher.register([_defn("tool_a")], handler_a)
    dispatcher.register([_defn("tool_b")], handler_b)

    result = await dispatcher.dispatch("tool_a", {})

    assert result == "result_a"
    handler_a.assert_awaited_once_with("tool_a", {})
    handler_b.assert_not_awaited()


async def test_dispatcher_unknown_function():
    dispatcher = ToolDispatcher()

    result = await dispatcher.dispatch("nonexistent", {})

    assert "Unknown function" in result


async def test_dispatcher_get_all_definitions():
    dispatcher = ToolDispatcher()

    dispatcher.register([_defn("tool_a"), _defn("tool_b")], AsyncMock())
    dispatcher.register([_defn("tool_c")], AsyncMock())

    names = [d["name"] for d in dispatcher.get_all_definitions()]
    assert names == ["tool_a", "tool_b", "tool_c"]


async def test_dispatcher_forwards_args():
    dispatcher = ToolDispatcher()
    handler = AsyncMock(return_value="ok")
    dispatcher.register([_defn("my_tool")], handler)

    await dispatcher.dispatch("my_tool", {"key": "value"})

    handler.assert_awaited_once_with("my_tool", {"key": "value"})
