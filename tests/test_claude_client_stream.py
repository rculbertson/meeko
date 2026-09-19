"""Unit tests for the ClaudeClient streaming tool-use loop.

Exercises stream_turn() with the Anthropic messages.stream context
manager mocked, so we can verify sentence yielding, tool_use routing,
and message-history bookkeeping without real API calls.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

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


def _final_message(stop_reason: str, content: list) -> SimpleNamespace:
    return SimpleNamespace(stop_reason=stop_reason, content=content, usage=None)


class _FakeStream:
    """Mimics the async context manager returned by messages.stream()."""

    def __init__(self, deltas: list[str], final: SimpleNamespace):
        self._deltas = deltas
        self._final = final

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return None

    @property
    def text_stream(self):
        async def _iter():
            for d in self._deltas:
                yield d

        return _iter()

    async def get_final_message(self):
        return self._final


def _build_client(rounds: list[tuple[list[str], SimpleNamespace]], tool_handler):
    """rounds[i] = (text_deltas, final_message) for each Claude round."""
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

    rounds_iter = iter(rounds)
    stream_mock = MagicMock(side_effect=lambda **kw: _FakeStream(*next(rounds_iter)))
    client._client.beta.messages.stream = stream_mock
    return client, stream_mock


async def _collect(aiter):
    return [x async for x in aiter]


async def test_stream_turn_yields_sentences_as_they_complete():
    deltas = ["Hello", " there", ". How", " are you", "?"]
    final = _final_message("end_turn", [_text_block("Hello there. How are you?")])
    client, _ = _build_client([(deltas, final)], AsyncMock())

    sentences = await _collect(client.stream_turn("hi"))

    assert sentences == ["Hello there.", "How are you?"]
    # user + assistant recorded for next turn
    assert client._messages[0] == {"role": "user", "content": "hi"}
    assert client._messages[1]["role"] == "assistant"


async def test_stream_turn_does_not_split_on_abbreviation():
    """Periods inside acronyms like "U.S." must not trigger a split
    when followed by a lowercase word."""
    deltas = ["In 1974, the U.S. president was Nixon."]
    final = _final_message(
        "end_turn", [_text_block("In 1974, the U.S. president was Nixon.")]
    )
    client, _ = _build_client([(deltas, final)], AsyncMock())

    sentences = await _collect(client.stream_turn("hi"))

    assert sentences == ["In 1974, the U.S. president was Nixon."]


async def test_stream_turn_flushes_tail_without_terminal_punctuation():
    deltas = ["Just a fragment"]
    final = _final_message("end_turn", [_text_block("Just a fragment")])
    client, _ = _build_client([(deltas, final)], AsyncMock())

    sentences = await _collect(client.stream_turn("hi"))

    assert sentences == ["Just a fragment"]


async def test_stream_turn_executes_tool_use_then_yields_text():
    round1_final = _final_message(
        "tool_use",
        [_tool_use_block(id="t1", name="echo", input={"v": 1})],
    )
    round2_final = _final_message("end_turn", [_text_block("Done.")])
    rounds = [
        ([], round1_final),
        (["Done."], round2_final),
    ]
    handler = AsyncMock(return_value="tool-result-body")
    client, stream_mock = _build_client(rounds, handler)

    sentences = await _collect(client.stream_turn("please run a tool"))

    assert sentences == ["Done."]
    assert stream_mock.call_count == 2
    handler.assert_awaited_once_with("echo", {"v": 1})
    # messages: user, assistant(tool_use), user(tool_result), assistant(text)
    assert len(client._messages) == 4
    tool_result_msg = client._messages[2]
    assert tool_result_msg["role"] == "user"
    assert tool_result_msg["content"][0]["type"] == "tool_result"
    assert tool_result_msg["content"][0]["tool_use_id"] == "t1"
    assert tool_result_msg["content"][0]["content"] == "tool-result-body"


async def test_raising_tool_becomes_an_error_result_and_the_turn_continues():
    """A handler that raises must not end the turn with the tool_use
    unanswered — that history would 400 every later request. Sonnet gets
    the failure as the tool's result and replies to it."""
    round1_final = _final_message(
        "tool_use", [_tool_use_block(id="t1", name="echo", input={})]
    )
    round2_final = _final_message("end_turn", [_text_block("Sorry, that failed.")])
    handler = AsyncMock(side_effect=RuntimeError("fts5: syntax error"))
    client, stream_mock = _build_client(
        [([], round1_final), (["Sorry, that failed."], round2_final)], handler
    )

    sentences = await _collect(client.stream_turn("find the todo-app chat"))

    assert sentences == ["Sorry, that failed."]
    result = client._messages[2]["content"][0]
    assert result["type"] == "tool_result"
    assert result["tool_use_id"] == "t1"
    assert result["is_error"] is True
    assert "fts5" not in result["content"]  # internals stay in the log
    sent = stream_mock.call_args.kwargs["messages"]
    assert sent[2]["content"][0]["tool_use_id"] == "t1"


async def test_barge_in_during_a_tool_does_not_break_the_next_turn():
    """Cancelling while a tool runs leaves the committed tool_use without a
    result. The next request must still pair it, or the API rejects it."""
    round1_final = _final_message(
        "tool_use", [_tool_use_block(id="t1", name="echo", input={})]
    )
    next_final = _final_message("end_turn", [_text_block("Okay.")])
    started = asyncio.Event()

    async def slow_tool(name, args):
        started.set()
        await asyncio.sleep(10)
        return "late"

    client, stream_mock = _build_client(
        [([], round1_final), (["Okay."], next_final)], slow_tool
    )

    turn = asyncio.create_task(_collect(client.stream_turn("what's the weather?")))
    await started.wait()
    turn.cancel()
    with pytest.raises(asyncio.CancelledError):
        await turn

    assert await _collect(client.stream_turn("never mind")) == ["Okay."]
    sent = stream_mock.call_args.kwargs["messages"]
    assert sent[1]["content"][0]["type"] == "tool_use"
    first_block = sent[2]["content"][0]
    assert first_block["type"] == "tool_result"
    assert first_block["tool_use_id"] == "t1"
    assert first_block["is_error"] is True


async def test_barge_in_mid_dispatch_keeps_results_of_tools_that_ran():
    """Two tools, interrupted during the second: the first already did its
    work (a timer really started), so Sonnet must get its real result, not
    the "not run" filler. Only the unfinished tool is marked interrupted."""
    round1_final = _final_message(
        "tool_use",
        [
            _tool_use_block(id="t1", name="echo", input={"n": 1}),
            _tool_use_block(id="t2", name="echo", input={"n": 2}),
        ],
    )
    next_final = _final_message("end_turn", [_text_block("Okay.")])
    second_started = asyncio.Event()

    async def tools(name, args):
        if args["n"] == 1:
            return "Timer 'pasta' set."
        second_started.set()
        await asyncio.sleep(10)
        return "late"

    client, stream_mock = _build_client(
        [([], round1_final), (["Okay."], next_final)], tools
    )

    turn = asyncio.create_task(_collect(client.stream_turn("pasta timer, and rain?")))
    await second_started.wait()
    turn.cancel()
    with pytest.raises(asyncio.CancelledError):
        await turn

    results = client._messages[-1]["content"]
    assert results[0] == {
        "type": "tool_result",
        "tool_use_id": "t1",
        "content": "Timer 'pasta' set.",
    }
    assert results[1]["tool_use_id"] == "t2"
    assert results[1]["is_error"] is True

    await _collect(client.stream_turn("never mind"))
    sent = stream_mock.call_args.kwargs["messages"]
    assert sent[2]["content"] == results  # nothing re-filled at send time


async def test_stream_turn_strips_extra_fields_from_stored_blocks():
    """The streaming SDK attaches fields like parsed_output to text
    blocks; those must not be replayed on later turns or Anthropic
    rejects the request with a 400."""
    text = SimpleNamespace(type="text", text="Hi.")
    text.model_dump = lambda: {"type": "text", "text": "Hi.", "parsed_output": None}
    final = _final_message("end_turn", [text])
    client, _ = _build_client([(["Hi."], final)], AsyncMock())

    await _collect(client.stream_turn("hello"))

    stored = client._messages[1]
    assert stored["role"] == "assistant"
    assert stored["content"] == [{"type": "text", "text": "Hi."}]


async def test_stream_turn_persists_user_and_assistant_turns(tmp_path):
    from meeko.sessions import SessionStore

    store = SessionStore.open(tmp_path / "m.db")
    try:
        session_id = await store.create_session("query")

        dispatcher = ToolDispatcher()
        mock_anthropic = MagicMock()
        with patch(
            "meeko.claude_client.anthropic.AsyncAnthropic",
            return_value=mock_anthropic,
        ):
            client = ClaudeClient(
                api_key="x",
                system_prompt="sys",
                dispatcher=dispatcher,
                store=store,
                session_id=session_id,
            )
        final = _final_message("end_turn", [_text_block("Hi.")])
        client._client.beta.messages.stream = MagicMock(
            return_value=_FakeStream(["Hi."], final)
        )

        await _collect(client.stream_turn("hello"))

        import json
        import sqlite3

        conn = sqlite3.connect(str(tmp_path / "m.db"))
        try:
            rows = conn.execute(
                "SELECT role, content FROM turns WHERE session_id = ? ORDER BY id",
                (session_id,),
            ).fetchall()
        finally:
            conn.close()
        assert len(rows) == 2
        assert rows[0][0] == "user"
        assert json.loads(rows[0][1]) == "hello"
        assert rows[1][0] == "assistant"
        assert json.loads(rows[1][1]) == [{"type": "text", "text": "Hi."}]
    finally:
        await store.close()


async def test_stream_turn_persists_tool_round_messages(tmp_path):
    from meeko.sessions import SessionStore

    store = SessionStore.open(tmp_path / "m.db")
    try:
        session_id = await store.create_session("query")

        round1_final = _final_message(
            "tool_use",
            [_tool_use_block(id="t1", name="echo", input={"v": 1})],
        )
        round2_final = _final_message("end_turn", [_text_block("Done.")])

        dispatcher = ToolDispatcher()
        dispatcher.register(
            [
                {
                    "name": "echo",
                    "description": "echo",
                    "input_schema": {"type": "object", "properties": {}},
                }
            ],
            AsyncMock(return_value="res"),
        )
        mock_anthropic = MagicMock()
        with patch(
            "meeko.claude_client.anthropic.AsyncAnthropic",
            return_value=mock_anthropic,
        ):
            client = ClaudeClient(
                api_key="x",
                system_prompt="sys",
                dispatcher=dispatcher,
                store=store,
                session_id=session_id,
            )
        rounds = iter([([], round1_final), (["Done."], round2_final)])
        client._client.beta.messages.stream = MagicMock(
            side_effect=lambda **kw: _FakeStream(*next(rounds))
        )

        await _collect(client.stream_turn("run it"))

        import sqlite3

        conn = sqlite3.connect(str(tmp_path / "m.db"))
        try:
            roles = [
                r[0]
                for r in conn.execute(
                    "SELECT role FROM turns WHERE session_id = ? ORDER BY id",
                    (session_id,),
                ).fetchall()
            ]
        finally:
            conn.close()
        assert roles == ["user", "assistant", "user", "assistant"]
    finally:
        await store.close()


async def test_load_history_preloads_messages_sent_on_next_turn():
    final = _final_message("end_turn", [_text_block("ok")])
    client, stream_mock = _build_client([(["ok"], final)], AsyncMock())
    prior = [
        {"role": "user", "content": "first thing"},
        {"role": "assistant", "content": [{"type": "text", "text": "reply"}]},
    ]
    client.load_history(prior)

    await _collect(client.stream_turn("follow up"))

    sent = stream_mock.call_args.kwargs["messages"]
    # Prior turns appear first, then the new user turn. The send-time view
    # adds a cache_control breakpoint to the tail block of the latest
    # message; earlier messages stay in their canonical shape.
    assert sent[:2] == prior
    assert sent[2] == {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "follow up",
                "cache_control": {"type": "ephemeral"},
            }
        ],
    }


async def test_set_system_prompt_takes_effect_on_next_call():
    final = _final_message("end_turn", [_text_block("ok")])
    client, stream_mock = _build_client([(["ok"], final)], AsyncMock())
    client.set_system_prompt("new system")

    await _collect(client.stream_turn("hi"))

    kwargs = stream_mock.call_args.kwargs
    # The profile-prompt block carries the cache breakpoint; a small
    # uncached today block follows for date-aware reasoning.
    blocks = kwargs["system"]
    assert blocks[0] == {
        "type": "text",
        "text": "new system",
        "cache_control": {"type": "ephemeral"},
    }
    assert blocks[-1]["type"] == "text"
    assert "Today is" in blocks[-1]["text"]
    assert "cache_control" not in blocks[-1]
