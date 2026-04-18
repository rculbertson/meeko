"""Unit tests for the ClaudeClient tool-use loop.

Exercises the core Claude turn flow with the Anthropic client mocked,
so we can verify tool_use routing + message-history bookkeeping without
making real API calls. End-to-end voice tests (mic → STT → Claude → TTS
→ speaker) are manual for now.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from meeko.claude_client import ClaudeClient
from meeko.tools.dispatch import ToolDispatcher


def _text_block(text: str) -> SimpleNamespace:
    ns = SimpleNamespace(type="text", text=text)
    ns.model_dump = lambda: {"type": "text", "text": text}
    return ns


def _tool_use_block(*, id: str, name: str, input: dict) -> SimpleNamespace:
    ns = SimpleNamespace(type="tool_use", id=id, name=name, input=input)
    ns.model_dump = lambda: {
        "type": "tool_use",
        "id": id,
        "name": name,
        "input": input,
    }
    return ns


def _response(stop_reason: str, content: list) -> SimpleNamespace:
    return SimpleNamespace(stop_reason=stop_reason, content=content)


def _build_client(responses, tool_handler):
    dispatcher = ToolDispatcher()
    dispatcher.register(
        [
            {
                "name": "echo",
                "description": "echo input",
                "input_schema": {"type": "object", "properties": {}},
            }
        ],
        tool_handler,
    )
    mock_anthropic = MagicMock()
    with patch(
        "meeko.claude_client.anthropic.AsyncAnthropic", return_value=mock_anthropic
    ):
        client = ClaudeClient(api_key="x", system_prompt="sys", dispatcher=dispatcher)
    create = AsyncMock(side_effect=responses)
    client._client.messages.create = create
    return client, create


async def test_turn_without_tool_use_returns_text():
    responses = [_response("end_turn", [_text_block("hello there")])]
    client, create = _build_client(responses, AsyncMock(return_value="_"))

    reply = await client.turn("hi")

    assert reply == "hello there"
    create.assert_awaited_once()
    # user + assistant recorded for next turn
    assert client._messages[0] == {"role": "user", "content": "hi"}
    assert client._messages[1]["role"] == "assistant"


async def test_turn_executes_tool_use_then_returns_text():
    responses = [
        _response(
            "tool_use",
            [_tool_use_block(id="t1", name="echo", input={"v": 1})],
        ),
        _response("end_turn", [_text_block("done")]),
    ]
    handler = AsyncMock(return_value="tool-result-body")
    client, create = _build_client(responses, handler)

    reply = await client.turn("please run a tool")

    assert reply == "done"
    assert create.await_count == 2
    handler.assert_awaited_once_with("echo", {"v": 1})
    # messages: user, assistant(tool_use), user(tool_result), assistant(text)
    assert len(client._messages) == 4
    tool_result_msg = client._messages[2]
    assert tool_result_msg["role"] == "user"
    assert tool_result_msg["content"][0]["type"] == "tool_result"
    assert tool_result_msg["content"][0]["tool_use_id"] == "t1"
    assert tool_result_msg["content"][0]["content"] == "tool-result-body"


async def test_set_system_prompt_takes_effect_on_next_call():
    responses = [_response("end_turn", [_text_block("ok")])]
    client, create = _build_client(responses, AsyncMock())
    client.set_system_prompt("new system")

    await client.turn("hi")

    kwargs = create.await_args.kwargs
    assert kwargs["system"] == "new system"
