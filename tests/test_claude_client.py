"""Tests for ClaudeClient prompt-caching breakpoints.

Faking the Anthropic streaming API surface (`messages.stream(...)`
context manager → `text_stream` async iterator + `get_final_message`)
keeps these tests offline. The `_FakeStream` records the kwargs each
call was made with so we can assert on the cache_control placement in
the message payload that hits the wire.
"""

from __future__ import annotations

import asyncio
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
        # Production code calls `client.beta.messages.stream(...)` for the
        # compaction beta; expose the same fake under both surfaces.
        self.beta = SimpleNamespace(messages=self.messages)


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
            self.beta = SimpleNamespace(messages=self.messages)

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


@pytest.mark.asyncio
async def test_stream_turn_sends_compaction_context_management(fake_anthropic):
    client = _make_client()
    await _drain(client.stream_turn("hello"))

    sent = fake_anthropic["client"].captured[0]
    assert sent["betas"] == [claude_client_module.COMPACTION_BETA]
    assert sent["context_management"] == {
        "edits": [
            {
                "type": claude_client_module.COMPACTION_STRATEGY,
                "trigger": {
                    "type": "input_tokens",
                    "value": claude_client_module.COMPACTION_TRIGGER_TOKENS,
                },
            }
        ]
    }


class _CompactionStream(_FakeStream):
    """Streams a compaction summary alongside a normal text response — the
    shape the API returns when server-side compaction fires mid-turn."""

    @property
    def text_stream(self):
        async def _iter():
            yield "Compacted reply."

        return _iter()

    async def get_final_message(self):
        compaction_block = SimpleNamespace(
            type="compaction",
            content="Earlier turns: user asked things, assistant answered.",
        )
        text_block = SimpleNamespace(type="text", text="Compacted reply.")
        usage = SimpleNamespace(
            input_tokens=42,
            output_tokens=8,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
            iterations=[
                SimpleNamespace(
                    type="compaction", input_tokens=180000, output_tokens=3500
                ),
                SimpleNamespace(type="message", input_tokens=23000, output_tokens=8),
            ],
        )
        return SimpleNamespace(
            content=[compaction_block, text_block],
            stop_reason="end_turn",
            usage=usage,
        )


@pytest.mark.asyncio
async def test_compaction_block_kept_in_memory_but_not_persisted(monkeypatch, caplog):
    captured: list[dict[str, Any]] = []

    class _Messages:
        def stream(self, **kwargs):
            return _CompactionStream(captured, kwargs)

    class _Client:
        def __init__(self, *_, **__):
            self.messages = _Messages()
            self.beta = SimpleNamespace(messages=self.messages)

    monkeypatch.setattr(claude_client_module.anthropic, "AsyncAnthropic", _Client)

    persisted: list[tuple[str, Any]] = []

    class _FakeStore:
        async def persist_turn(self, session_id, role, content):
            persisted.append((role, content))

    client = ClaudeClient(
        api_key="k",
        system_prompt="sys",
        dispatcher=ToolDispatcher(),
        store=_FakeStore(),
        session_id="sess-1",
    )

    with caplog.at_level(logging.INFO, logger="meeko"):
        await _drain(client.stream_turn("Tell me a long story."))

    # In-memory: the compaction block is preserved so the next turn's
    # `messages=` payload carries it (otherwise the server would re-summarize).
    assert client._messages[-1]["role"] == "assistant"
    types = [b["type"] for b in client._messages[-1]["content"]]
    assert types == ["compaction", "text"]
    compaction_block = client._messages[-1]["content"][0]
    assert compaction_block == {
        "type": "compaction",
        "content": "Earlier turns: user asked things, assistant answered.",
    }

    # SQLite: only the user line + the assistant's text block were persisted.
    # The compaction summary stays out of the on-disk transcript.
    assert persisted[0] == ("user", "Tell me a long story.")
    assert persisted[1][0] == "assistant"
    persisted_blocks = persisted[1][1]
    assert [b["type"] for b in persisted_blocks] == ["text"]

    # The new info-level compaction log line fired with the iteration counts.
    compaction_logs = [
        rec for rec in caplog.records if rec.getMessage().startswith("[compaction]")
    ]
    assert len(compaction_logs) == 1
    msg = compaction_logs[0].getMessage()
    assert "in_tok=180000" in msg
    assert "out_tok=3500" in msg


@pytest.mark.asyncio
async def test_compaction_block_round_trips_into_next_turn(monkeypatch):
    """After compaction fires on turn 1, turn 2's outgoing `messages=` must
    include the compaction block on the prior assistant turn — otherwise the
    server has no record that compaction happened and will redo it."""
    captured: list[dict[str, Any]] = []
    call_count = {"n": 0}

    class _Messages:
        def stream(self, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return _CompactionStream(captured, kwargs)
            return _FakeStream(captured, kwargs)

    class _Client:
        def __init__(self, *_, **__):
            self.messages = _Messages()
            self.beta = SimpleNamespace(messages=self.messages)

    monkeypatch.setattr(claude_client_module.anthropic, "AsyncAnthropic", _Client)

    client = ClaudeClient(api_key="k", system_prompt="sys", dispatcher=ToolDispatcher())
    await _drain(client.stream_turn("Turn one."))
    await _drain(client.stream_turn("Turn two."))

    # Second call's outgoing messages: the prior assistant turn carries the
    # compaction block as its first content entry.
    sent = captured[1]["messages"]
    prior_assistant = sent[1]
    assert prior_assistant["role"] == "assistant"
    assert prior_assistant["content"][0]["type"] == "compaction"


def test_compaction_trigger_env_var_override(monkeypatch):
    """Reloading the module with the env var set picks up a new threshold
    and threads it into `_CONTEXT_MANAGEMENT`."""
    import importlib

    monkeypatch.setenv("MEEKO_COMPACTION_TRIGGER_TOKENS", "12345")
    reloaded = importlib.reload(claude_client_module)
    try:
        assert reloaded.COMPACTION_TRIGGER_TOKENS == 12345
        assert reloaded._CONTEXT_MANAGEMENT["edits"][0]["trigger"]["value"] == 12345
    finally:
        # Restore the module so later tests in the session see the default.
        monkeypatch.delenv("MEEKO_COMPACTION_TRIGGER_TOKENS", raising=False)
        importlib.reload(claude_client_module)


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


# Padding sized to push the system prompt past the API's 50 000-token
# minimum compaction trigger. Each repeat is ~13 tokens, so 4500 repeats
# clears 55k. Used only by the live compaction test below.
_COMPACTION_SYSTEM_PROMPT = (
    "You are a terse test assistant. Reply in three words or fewer.\n\n"
    "Background context (padding so total input crosses the API's 50k "
    "compaction-trigger floor): "
    + ("Meeko is a long-running brainstorming voice assistant. " * 4500)
)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_server_side_compaction_fires_on_long_prefix(monkeypatch, caplog):
    """End-to-end check that the compaction beta is wired up correctly.

    EXPENSIVE — disabled by default. The Anthropic API enforces a minimum
    compaction `trigger.value` of 50 000 input tokens, so to force compaction
    the test ships ~55k tokens of padded system prompt across two turns plus
    a server-side summarization pass over the same prefix. That works out to
    roughly $0.30–0.40 per run on Sonnet 4.6 — too expensive to leave on the
    default `-m integration` runs. Set `RUN_EXPENSIVE_TESTS=1` to opt in
    (`ANTHROPIC_API_KEY` is loaded from `.env` by `tests/conftest.py`):

        RUN_EXPENSIVE_TESTS=1 uv run pytest -m integration \\
            tests/test_claude_client.py::test_server_side_compaction_fires_on_long_prefix
    """
    if not os.environ.get("RUN_EXPENSIVE_TESTS"):
        pytest.skip("expensive (~$0.35/run); set RUN_EXPENSIVE_TESTS=1 to enable")
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        pytest.skip("ANTHROPIC_API_KEY not set")

    # 50000 is the API floor; the padded system prompt above crosses it on
    # turn 2 (system + turn-1 history + new user msg).
    import importlib

    monkeypatch.setenv("MEEKO_COMPACTION_TRIGGER_TOKENS", "50000")
    reloaded = importlib.reload(claude_client_module)
    try:
        client = reloaded.ClaudeClient(
            api_key=api_key,
            system_prompt=_COMPACTION_SYSTEM_PROMPT,
            dispatcher=ToolDispatcher(),
        )

        with caplog.at_level(logging.INFO, logger="meeko"):
            async for _ in client.stream_turn("Say 'one'."):
                pass
            async for _ in client.stream_turn("Say 'two'."):
                pass

        compaction_logs = [
            rec for rec in caplog.records if rec.getMessage().startswith("[compaction]")
        ]
        compaction_in_history = any(
            isinstance(msg.get("content"), list)
            and any(b.get("type") == "compaction" for b in msg["content"])
            for msg in client._messages
            if msg.get("role") == "assistant"
        )

        assert compaction_logs or compaction_in_history, (
            "expected server-side compaction to fire with trigger=50000 "
            "and a ~55k-token padded system prompt"
        )
    finally:
        monkeypatch.delenv("MEEKO_COMPACTION_TRIGGER_TOKENS", raising=False)
        importlib.reload(claude_client_module)


class _CancellableStream:
    """A streaming response that yields a few deltas, then awaits a
    gate so the consumer can be cancelled mid-stream."""

    def __init__(
        self,
        captured: list[dict[str, Any]],
        kwargs: dict[str, Any],
        gate: asyncio.Event,
        early_deltas: list[str],
    ):
        self._captured = captured
        self._kwargs = kwargs
        self._gate = gate
        self._early_deltas = early_deltas

    async def __aenter__(self):
        self._captured.append(copy.deepcopy(self._kwargs))
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    @property
    def text_stream(self):
        gate = self._gate
        deltas = self._early_deltas

        async def _iter():
            for d in deltas:
                yield d
            await gate.wait()  # block until test cancels us
            yield "never reached"

        return _iter()

    async def get_final_message(self):  # pragma: no cover - never called
        raise AssertionError("stream cancelled before final message")


async def test_cancelled_stream_appends_partial_assistant_turn(monkeypatch):
    """When stream_turn is cancelled mid-stream (barge-in), it must
    commit a partial assistant message so the next turn doesn't append
    a second consecutive user message."""
    captured: list[dict[str, Any]] = []
    gate = asyncio.Event()

    class _Messages:
        def stream(self, **kwargs):
            return _CancellableStream(
                captured, kwargs, gate, ["Hello there. ", "I was about"]
            )

    class _Client:
        def __init__(self, *_, **__):
            self.messages = _Messages()
            self.beta = SimpleNamespace(messages=self.messages)

    monkeypatch.setattr(claude_client_module.anthropic, "AsyncAnthropic", _Client)

    persisted: list[tuple[str, Any]] = []

    class _FakeStore:
        async def persist_turn(self, session_id, role, content):
            persisted.append((role, content))

    client = ClaudeClient(
        api_key="k",
        system_prompt="sys",
        dispatcher=ToolDispatcher(),
        store=_FakeStore(),
        session_id="sess-1",
    )

    yielded: list[str] = []

    async def consume():
        async for s in client.stream_turn("Hi Claude"):
            yielded.append(s)

    task = asyncio.create_task(consume())
    # Wait until at least one delta has been processed (sentence yielded).
    for _ in range(100):
        if yielded:
            break
        await asyncio.sleep(0)
    assert yielded, "expected at least one sentence before cancel"

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    # In-memory history is balanced: user followed by partial assistant.
    assert client._messages[-2] == {"role": "user", "content": "Hi Claude"}
    last = client._messages[-1]
    assert last["role"] == "assistant"
    assert last["content"][0]["type"] == "text"
    assert "Hello there" in last["content"][0]["text"]

    # SQLite persisted both the user turn and the partial assistant turn.
    roles = [r for r, _ in persisted]
    assert roles == ["user", "assistant"]
    assert "Hello there" in persisted[1][1][0]["text"]


async def test_cancelled_stream_with_no_output_uses_placeholder(monkeypatch):
    """If the cancel arrives before any tokens streamed, the partial
    assistant turn is a placeholder ellipsis so history stays valid."""
    captured: list[dict[str, Any]] = []
    gate = asyncio.Event()

    class _Messages:
        def stream(self, **kwargs):
            return _CancellableStream(captured, kwargs, gate, [])

    class _Client:
        def __init__(self, *_, **__):
            self.messages = _Messages()
            self.beta = SimpleNamespace(messages=self.messages)

    monkeypatch.setattr(claude_client_module.anthropic, "AsyncAnthropic", _Client)

    client = ClaudeClient(
        api_key="k",
        system_prompt="sys",
        dispatcher=ToolDispatcher(),
    )

    async def consume():
        async for _ in client.stream_turn("hello"):
            pass

    task = asyncio.create_task(consume())
    # Yield enough times for the stream to enter the gate.wait().
    for _ in range(20):
        await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    last = client._messages[-1]
    assert last["role"] == "assistant"
    assert last["content"] == [{"type": "text", "text": "…"}]
