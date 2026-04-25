"""Tests for ClaudeClient prompt-caching breakpoints.

Faking the Anthropic streaming API surface (`messages.stream(...)`
context manager → `text_stream` async iterator + `get_final_message`)
keeps these tests offline. The `_FakeStream` records the kwargs each
call was made with so we can assert on the cache_control placement in
the message payload that hits the wire.
"""

from __future__ import annotations

import copy
import logging
import os
import re
from types import SimpleNamespace
from typing import Any

import pytest

from meeko import claude_client as claude_client_module
from meeko.claude_client import ClaudeClient, _with_cache_breakpoint
from meeko.tools.dispatch import ToolDispatcher


class _FakeStream:
    def __init__(self, captured: list[dict[str, Any]], kwargs: dict[str, Any]):
        self._captured = captured
        self._kwargs = kwargs

    async def __aenter__(self):
        # Snapshot the kwargs at enter time — claude_client builds the
        # send-time message view inside the call, so the values we want
        # to assert on are already realized in `kwargs`.
        self._captured.append(copy.deepcopy(self._kwargs))
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    @property
    def text_stream(self):
        async def _iter():
            yield "Hi there. "
            yield "How are you?"

        return _iter()

    async def get_final_message(self):
        text_block = SimpleNamespace(type="text", text="Hi there. How are you?")
        usage = SimpleNamespace(
            input_tokens=42,
            output_tokens=8,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
        )
        return SimpleNamespace(
            content=[text_block],
            stop_reason="end_turn",
            usage=usage,
        )


class _FakeMessages:
    def __init__(self, captured: list[dict[str, Any]]):
        self._captured = captured

    def stream(self, **kwargs):
        return _FakeStream(self._captured, kwargs)


class _FakeAsyncAnthropic:
    def __init__(self, *_, **__):
        self.captured: list[dict[str, Any]] = []
        self.messages = _FakeMessages(self.captured)


@pytest.fixture
def fake_anthropic(monkeypatch):
    """Patch anthropic.AsyncAnthropic so ClaudeClient gets the fake.

    We retain a reference to the constructed instance so tests can
    inspect captured kwargs."""
    holder: dict[str, _FakeAsyncAnthropic] = {}

    def factory(*args, **kwargs):
        client = _FakeAsyncAnthropic(*args, **kwargs)
        holder["client"] = client
        return client

    monkeypatch.setattr(claude_client_module.anthropic, "AsyncAnthropic", factory)
    return holder


def _make_client() -> ClaudeClient:
    return ClaudeClient(
        api_key="test-key",
        system_prompt="You are Meeko.",
        dispatcher=ToolDispatcher(),
    )


async def _drain(agen):
    async for _ in agen:
        pass


def test_system_prompt_has_cache_control(fake_anthropic):
    client = _make_client()
    assert isinstance(client._system, list)
    assert client._system[-1]["cache_control"] == {"type": "ephemeral"}
    assert client._system[-1]["text"] == "You are Meeko."


def test_set_system_prompt_rewraps_with_cache_control(fake_anthropic):
    client = _make_client()
    client.set_system_prompt("New persona.")
    assert client._system == [
        {
            "type": "text",
            "text": "New persona.",
            "cache_control": {"type": "ephemeral"},
        }
    ]


def test_with_cache_breakpoint_wraps_string_user_message():
    msgs = [{"role": "user", "content": "hello"}]
    out = _with_cache_breakpoint(msgs)
    assert out[-1]["content"] == [
        {"type": "text", "text": "hello", "cache_control": {"type": "ephemeral"}}
    ]
    # Source untouched — persistence stays clean.
    assert msgs == [{"role": "user", "content": "hello"}]


def test_with_cache_breakpoint_annotates_block_list_tail():
    msgs = [
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "first"},
                {"type": "text", "text": "second"},
            ],
        }
    ]
    out = _with_cache_breakpoint(msgs)
    blocks = out[-1]["content"]
    assert blocks[0] == {"type": "text", "text": "first"}
    assert blocks[1] == {
        "type": "text",
        "text": "second",
        "cache_control": {"type": "ephemeral"},
    }
    # Original blocks untouched.
    assert "cache_control" not in msgs[-1]["content"][1]


@pytest.mark.asyncio
async def test_stream_turn_sets_cache_breakpoint_on_last_message(fake_anthropic):
    client = _make_client()
    await _drain(client.stream_turn("Hello!"))

    captured = fake_anthropic["client"].captured
    assert len(captured) == 1
    sent_msgs = captured[0]["messages"]
    assert len(sent_msgs) == 1
    last_block = sent_msgs[-1]["content"][-1]
    assert last_block["cache_control"] == {"type": "ephemeral"}

    # System breakpoint is also present.
    sent_system = captured[0]["system"]
    assert sent_system[-1]["cache_control"] == {"type": "ephemeral"}

    # Stored history kept its canonical plain-string shape.
    assert client._messages[0] == {"role": "user", "content": "Hello!"}
    # Assistant turn appended without cache_control leaking in.
    assert client._messages[1]["role"] == "assistant"
    for block in client._messages[1]["content"]:
        assert "cache_control" not in block


@pytest.mark.asyncio
async def test_cache_breakpoint_moves_to_newest_message_each_turn(fake_anthropic):
    client = _make_client()
    await _drain(client.stream_turn("Turn one."))
    await _drain(client.stream_turn("Turn two."))

    captured = fake_anthropic["client"].captured
    assert len(captured) == 2

    # Second call: tail is the new user message; earlier messages are clean.
    sent = captured[1]["messages"]
    # ['user: turn one', 'assistant: ...', 'user: turn two']
    assert len(sent) == 3
    # Turn-1 user msg stays in canonical plain-string form (the breakpoint
    # only ever lives on the latest message, applied at send time).
    assert sent[0] == {"role": "user", "content": "Turn one."}
    # Turn-1 assistant blocks must be free of stale cache_control keys.
    assert isinstance(sent[1]["content"], list)
    for block in sent[1]["content"]:
        assert "cache_control" not in block, (
            "stale cache_control on prior-turn assistant message"
        )
    # The new turn-2 user message carries the breakpoint on its tail block.
    assert sent[-1]["content"][-1]["cache_control"] == {"type": "ephemeral"}


class _ToolUseStream(_FakeStream):
    """First call emits a tool_use; second call (after tool_result is
    appended) emits a normal text response."""

    def __init__(self, captured, kwargs, *, tool_use: bool):
        super().__init__(captured, kwargs)
        self._tool_use = tool_use

    @property
    def text_stream(self):
        async def _iter():
            if not self._tool_use:
                yield "Done."

        return _iter()

    async def get_final_message(self):
        usage = SimpleNamespace(
            input_tokens=10,
            output_tokens=2,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
        )
        if self._tool_use:
            tool_block = SimpleNamespace(
                type="tool_use",
                id="toolu_1",
                name="set_timer",
                input={"seconds": 30},
            )
            return SimpleNamespace(
                content=[tool_block], stop_reason="tool_use", usage=usage
            )
        text_block = SimpleNamespace(type="text", text="Done.")
        return SimpleNamespace(
            content=[text_block], stop_reason="end_turn", usage=usage
        )


@pytest.mark.asyncio
async def test_breakpoint_applied_each_round_in_tool_use_loop(monkeypatch):
    captured: list[dict[str, Any]] = []
    call_count = {"n": 0}

    class _Messages:
        def stream(self, **kwargs):
            call_count["n"] += 1
            tool_use = call_count["n"] == 1
            return _ToolUseStream(captured, kwargs, tool_use=tool_use)

    class _Client:
        def __init__(self, *_, **__):
            self.messages = _Messages()

    monkeypatch.setattr(claude_client_module.anthropic, "AsyncAnthropic", _Client)

    dispatcher = ToolDispatcher()

    async def _handle(name, args):
        return "timer set"

    dispatcher.register(
        [
            {
                "name": "set_timer",
                "description": "set a timer",
                "input_schema": {"type": "object", "properties": {}},
            }
        ],
        _handle,
    )
    client = ClaudeClient(api_key="k", system_prompt="sys", dispatcher=dispatcher)
    await _drain(client.stream_turn("Set a 30-second timer."))

    assert call_count["n"] == 2
    # Round 1: tail is the user text.
    msgs_round1 = captured[0]["messages"]
    assert msgs_round1[-1]["content"][-1]["cache_control"] == {"type": "ephemeral"}
    # Round 2: tail is the tool_result block appended after the tool_use round.
    msgs_round2 = captured[1]["messages"]
    tail = msgs_round2[-1]["content"][-1]
    assert tail["type"] == "tool_result"
    assert tail["cache_control"] == {"type": "ephemeral"}


# Anthropic's cache minimum for Sonnet is 1024 tokens — pad the system prompt
# well past that so the first turn is guaranteed to create a cache entry.
_INTEGRATION_SYSTEM_PROMPT = (
    "You are a terse test assistant. Reply in three words or fewer.\n\n"
    "Background context (padding so the system prompt clears the prompt-cache "
    "minimum for Sonnet): "
    + ("Meeko is a long-running brainstorming voice assistant. " * 200)
)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_second_turn_hits_prompt_cache(caplog):
    """End-to-end check that our cache_control placement actually produces
    a cache hit on turn 2.

    Costs ~2 short Sonnet calls (a fraction of a cent). Skipped without
    ANTHROPIC_API_KEY so unit-only runs stay offline. The skip happens
    inside the body (not via skipif) because conftest's autouse
    load_dotenv fixture runs after collection-time decorators."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        pytest.skip("ANTHROPIC_API_KEY not set")
    client = ClaudeClient(
        api_key=api_key,
        system_prompt=_INTEGRATION_SYSTEM_PROMPT,
        dispatcher=ToolDispatcher(),
    )

    with caplog.at_level(logging.DEBUG, logger="meeko"):
        async for _ in client.stream_turn("Say 'one'."):
            pass
        async for _ in client.stream_turn("Say 'two'."):
            pass

    cache_reads: list[int] = []
    cache_creates: list[int] = []
    for rec in caplog.records:
        msg = rec.getMessage()
        read = re.search(r"cache_read=(\d+)", msg)
        create = re.search(r"cache_create=(\d+)", msg)
        if read and create:
            cache_reads.append(int(read.group(1)))
            cache_creates.append(int(create.group(1)))

    assert len(cache_reads) >= 2, (
        f"expected at least 2 timing log records, got {len(cache_reads)}"
    )
    # Turn 2 must hit the cache for at least the system+tools+turn-1 prefix.
    # We can't assert turn 1 is a cold miss — repeat test runs within the
    # 5-minute cache TTL will already find the system prompt cached, which
    # is fine (it just proves caching is working from a prior run too).
    assert cache_reads[1] > 0, (
        f"turn 2 should hit the cache, got cache_read={cache_reads[1]}"
    )
    # Turn 2's cached prefix must include turn 1's user+assistant content,
    # so its cache read should be strictly larger than turn 1's.
    assert cache_reads[1] > cache_reads[0], (
        f"turn 2 cache_read ({cache_reads[1]}) should exceed turn 1's "
        f"({cache_reads[0]}) — the breakpoint moves forward each turn"
    )
