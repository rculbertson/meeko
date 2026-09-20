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
from meeko.claude_client import (
    _INTERRUPTED_TOOL_RESULT,
    ClaudeClient,
    _drop_stranded_server_tool_uses,
    _lead_with_results,
    _missing_results,
    _pair_orphan_tool_uses,
    _repaired,
    _with_cache_breakpoint,
)
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


def _install_stream_factory(monkeypatch, make_stream) -> list[dict[str, Any]]:
    """Patch anthropic.AsyncAnthropic with a client whose `messages.stream(**kw)`
    returns `make_stream(captured, kw)`, and return `captured`.

    The stream fakes append their kwargs to `captured` on enter, and rounds
    run one at a time, so `len(captured)` inside `make_stream` is the
    zero-based index of the call being made.
    """
    captured: list[dict[str, Any]] = []

    class _Messages:
        def stream(self, **kwargs):
            return make_stream(captured, kwargs)

    class _Client:
        def __init__(self, *_, **__):
            self.messages = _Messages()
            self.beta = SimpleNamespace(messages=self.messages)

    monkeypatch.setattr(claude_client_module.anthropic, "AsyncAnthropic", _Client)
    return captured


class _RecordingStore:
    """Session store fake that records every `(session_id, role, content)`."""

    def __init__(self) -> None:
        self.persisted: list[tuple[str, str, Any]] = []

    async def persist_turn(self, session_id, role, content):
        self.persisted.append((session_id, role, content))


def _make_client(**kwargs: Any) -> ClaudeClient:
    return ClaudeClient(
        **{
            "api_key": "test-key",
            "system_prompt": "You are Meeko.",
            "dispatcher": ToolDispatcher(),
            **kwargs,
        }
    )


async def _drain(agen):
    async for _ in agen:
        pass


def test_system_prompt_has_cache_control(fake_anthropic):
    from meeko.claude_client import _system_blocks

    blocks = _system_blocks("You are Meeko.")
    assert blocks[0]["text"] == "You are Meeko."
    assert blocks[0]["cache_control"] == {"type": "ephemeral"}
    # Trailing today block sits past the cache breakpoint and is uncached.
    assert "cache_control" not in blocks[-1]
    assert "Today is" in blocks[-1]["text"]


def test_system_blocks_include_location_when_coords_set(fake_anthropic):
    from meeko.claude_client import _system_blocks

    blocks = _system_blocks("You are Meeko.", 40.7484, -73.9857)
    # Location block precedes the cached profile prompt and is itself
    # uncached (the cache breakpoint sits on the profile-prompt block
    # below, which covers everything before it).
    assert "40.7484" in blocks[0]["text"]
    assert "-73.9857" in blocks[0]["text"]
    assert "cache_control" not in blocks[0]
    assert blocks[1]["text"] == "You are Meeko."
    assert blocks[1]["cache_control"] == {"type": "ephemeral"}
    assert "Today is" in blocks[-1]["text"]


def test_system_blocks_omit_location_when_coords_unset(fake_anthropic):
    from meeko.claude_client import _system_blocks

    blocks = _system_blocks("You are Meeko.", None, None)
    # No location block — first block is the cached profile prompt.
    assert blocks[0]["text"] == "You are Meeko."
    assert blocks[0]["cache_control"] == {"type": "ephemeral"}
    assert len(blocks) == 2  # profile prompt + today


def test_system_blocks_omit_location_when_only_one_coord_set(fake_anthropic):
    from meeko.claude_client import _system_blocks

    blocks = _system_blocks("You are Meeko.", 40.0, None)
    assert blocks[0]["text"] == "You are Meeko."
    assert len(blocks) == 2


def test_set_system_prompt_updates_profile_prompt(fake_anthropic):
    client = _make_client()
    client.set_system_prompt("New persona.")
    assert client._profile_prompt == "New persona."


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


def _tool_use(tid: str) -> dict:
    return {"type": "tool_use", "id": tid, "name": "echo", "input": {}}


def _tool_result(tid: str, content: str = "ok") -> dict:
    return {"type": "tool_result", "tool_use_id": tid, "content": content}


def _interrupted(tid: str) -> dict:
    return {
        "type": "tool_result",
        "tool_use_id": tid,
        "content": _INTERRUPTED_TOOL_RESULT,
        "is_error": True,
    }


def test_pair_orphan_tool_uses_leaves_answered_history_alone():
    msgs = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": [_tool_use("t1")]},
        {"role": "user", "content": [_tool_result("t1")]},
        {"role": "assistant", "content": [{"type": "text", "text": "done"}]},
    ]
    assert _pair_orphan_tool_uses(msgs) == msgs


def test_pair_orphan_tool_uses_leads_the_next_user_message_with_results():
    """The barge-in case: the next turn's plain-string user message follows
    the orphan. The results must come first in that message."""
    msgs = [
        {"role": "user", "content": "weather?"},
        {"role": "assistant", "content": [_tool_use("t1"), _tool_use("t2")]},
        {"role": "user", "content": "never mind"},
    ]
    out = _pair_orphan_tool_uses(msgs)
    assert out[2] == {
        "role": "user",
        "content": [
            _interrupted("t1"),
            _interrupted("t2"),
            {"type": "text", "text": "never mind"},
        ],
    }
    # Send-time view only: stored history stays verbatim.
    assert msgs[2] == {"role": "user", "content": "never mind"}


def test_pair_orphan_tool_uses_fills_only_the_missing_results():
    msgs = [
        {"role": "assistant", "content": [_tool_use("t1"), _tool_use("t2")]},
        {"role": "user", "content": [_tool_result("t1")]},
    ]
    out = _pair_orphan_tool_uses(msgs)
    assert out[1]["content"] == [_interrupted("t2"), _tool_result("t1")]


def test_pair_orphan_tool_uses_inserts_a_user_message_between_assistants():
    msgs = [
        {"role": "assistant", "content": [_tool_use("t1")]},
        {"role": "assistant", "content": [{"type": "text", "text": "…"}]},
    ]
    out = _pair_orphan_tool_uses(msgs)
    assert out == [
        msgs[0],
        {"role": "user", "content": [_interrupted("t1")]},
        msgs[1],
    ]


def test_pair_orphan_tool_uses_answers_a_trailing_orphan():
    msgs = [{"role": "assistant", "content": [_tool_use("t1")]}]
    out = _pair_orphan_tool_uses(msgs)
    assert out[-1] == {"role": "user", "content": [_interrupted("t1")]}


@pytest.mark.asyncio
async def test_persist_lazily_creates_session_on_first_turn(fake_anthropic):
    """No session row is created up-front. The first persisted turn
    triggers create_session_fn; subsequent turns reuse the id."""
    store = _RecordingStore()
    creates: list[int] = []

    async def create_fn() -> str:
        creates.append(1)
        return f"sess-{len(creates)}"

    client = _make_client(store=store, create_session_fn=create_fn)

    assert client.session_id is None
    assert len(creates) == 0
    assert store.persisted == []

    await _drain(client.stream_turn("first"))
    assert client.session_id == "sess-1"
    assert len(creates) == 1
    # User + assistant turns both went to sess-1.
    assert {sid for sid, _, _ in store.persisted} == {"sess-1"}

    await _drain(client.stream_turn("second"))
    # No second create — id reused.
    assert len(creates) == 1
    assert client.session_id == "sess-1"


@pytest.mark.asyncio
async def test_reset_session_clears_id_and_next_turn_lazily_creates(fake_anthropic):
    """reset_session() clears the bound id so the next persisted turn
    invokes create_session_fn for a fresh row."""
    creates: list[str] = []

    async def create_fn() -> str:
        creates.append("x")
        return f"sess-{len(creates)}"

    client = _make_client(store=_RecordingStore(), create_session_fn=create_fn)

    await _drain(client.stream_turn("first"))
    assert client.session_id == "sess-1"

    client.reset_session()
    assert client.session_id is None
    assert client._messages == []

    await _drain(client.stream_turn("after reset"))
    assert client.session_id == "sess-2"
    assert len(creates) == 2


@pytest.mark.asyncio
async def test_persist_skips_when_no_session_id_and_no_create_fn(fake_anthropic):
    """Store provided but no create_session_fn and no session_id — _persist
    should silently skip rather than crash."""
    store = _RecordingStore()
    client = _make_client(store=store)  # no session_id, no create_session_fn

    await _drain(client.stream_turn("hello"))
    assert store.persisted == []


@pytest.mark.asyncio
async def test_persist_skips_when_create_fn_returns_empty(fake_anthropic, caplog):
    """If create_session_fn returns an empty string, _persist logs an error
    and skips writing the turn rather than persisting under a blank id."""
    store = _RecordingStore()

    async def bad_create_fn() -> str:
        return ""

    client = _make_client(store=store, create_session_fn=bad_create_fn)

    with caplog.at_level(logging.ERROR, logger="meeko"):
        await _drain(client.stream_turn("hello"))

    assert store.persisted == []
    assert any("empty id" in r.getMessage() for r in caplog.records)
    assert client.session_id is None


@pytest.mark.asyncio
async def test_persist_logs_error_when_create_fn_raises(fake_anthropic, caplog):
    """If create_session_fn raises, _persist logs an error and skips writing
    the turn rather than propagating the exception."""
    store = _RecordingStore()

    async def failing_create_fn() -> str:
        raise RuntimeError("DB connection lost")

    client = _make_client(store=store, create_session_fn=failing_create_fn)

    with caplog.at_level(logging.ERROR, logger="meeko"):
        await _drain(client.stream_turn("hello"))

    assert store.persisted == []
    assert any("raised" in r.getMessage() for r in caplog.records)
    assert client.session_id is None


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

    # System breakpoint sits on the profile-prompt block; the trailing
    # today block is uncached (past the breakpoint).
    sent_system = captured[0]["system"]
    assert sent_system[0]["cache_control"] == {"type": "ephemeral"}
    assert "cache_control" not in sent_system[-1]

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
    # Only the first round asks for a tool.
    captured = _install_stream_factory(
        monkeypatch,
        lambda captured, kw: _ToolUseStream(captured, kw, tool_use=not captured),
    )

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
    client = _make_client(dispatcher=dispatcher)
    await _drain(client.stream_turn("Set a 30-second timer."))

    assert len(captured) == 2
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
                    "value": claude_client_module.DEFAULT_COMPACTION_TRIGGER_TOKENS,
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
    _install_stream_factory(monkeypatch, _CompactionStream)
    store = _RecordingStore()
    client = _make_client(store=store, session_id="sess-1")

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
    (_, user_role, user_content), (_, asst_role, asst_blocks) = store.persisted
    assert (user_role, user_content) == ("user", "Tell me a long story.")
    assert asst_role == "assistant"
    assert [b["type"] for b in asst_blocks] == ["text"]

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
    # Compaction fires on the first call only.
    captured = _install_stream_factory(
        monkeypatch,
        lambda captured, kw: (_FakeStream if captured else _CompactionStream)(
            captured, kw
        ),
    )
    client = _make_client()
    await _drain(client.stream_turn("Turn one."))
    await _drain(client.stream_turn("Turn two."))

    # Second call's outgoing messages: the prior assistant turn carries the
    # compaction block as its first content entry.
    sent = captured[1]["messages"]
    prior_assistant = sent[1]
    assert prior_assistant["role"] == "assistant"
    assert prior_assistant["content"][0]["type"] == "compaction"


def test_compaction_trigger_kwarg_threads_into_context_management(fake_anthropic):
    """The `compaction_trigger_tokens` kwarg threads the value into the
    per-instance context-management dict that gets sent with each turn."""
    client = ClaudeClient(
        api_key="test-key",
        system_prompt="sys",
        dispatcher=ToolDispatcher(),
        compaction_trigger_tokens=12345,
    )
    assert client._context_management["edits"][0]["trigger"]["value"] == 12345


def test_serialize_server_tool_use_strips_helper_fields():
    from meeko.claude_client import _serialize_block

    block = SimpleNamespace(
        type="server_tool_use",
        id="srvtoolu_abc",
        name="web_search",
        input={"query": "claude shannon"},
        # Mimic an SDK helper field that the Messages API rejects on input.
        parsed_output={"foo": "bar"},
    )
    assert _serialize_block(block) == {
        "type": "server_tool_use",
        "id": "srvtoolu_abc",
        "name": "web_search",
        "input": {"query": "claude shannon"},
    }


# `web_search_20260209` runs dynamic filtering: Sonnet calls `web_search`
# from inside `code_execution`, and each nested block carries a `caller`
# pointing back at the code_execution tool_use. Dropping it on replay
# makes the API report the outer code_execution's result as missing and
# 400 the request — confirmed against the live API by replaying a
# captured paused turn with and without the field.
_CALLER = {"tool_id": "srvtoolu_outer", "type": "code_execution_20260120"}


def _caller_obj():
    """A `caller` the way the SDK hands it over: a model, not a dict."""
    return SimpleNamespace(model_dump=lambda: dict(_CALLER))


def test_serialize_server_tool_use_preserves_caller():
    from meeko.claude_client import _serialize_block

    block = SimpleNamespace(
        type="server_tool_use",
        id="srvtoolu_nested",
        name="web_search",
        input={"query": "tallest building in Dallas"},
        caller=_caller_obj(),
    )
    assert _serialize_block(block) == {
        "type": "server_tool_use",
        "id": "srvtoolu_nested",
        "name": "web_search",
        "input": {"query": "tallest building in Dallas"},
        "caller": _CALLER,
    }


def test_serialize_web_search_tool_result_preserves_caller():
    from meeko.claude_client import _serialize_block

    block = SimpleNamespace(
        type="web_search_tool_result",
        tool_use_id="srvtoolu_nested",
        content=[],
        caller=_caller_obj(),
    )
    serialized = _serialize_block(block)
    assert serialized["caller"] == _CALLER
    assert serialized["tool_use_id"] == "srvtoolu_nested"


def test_serialize_caller_accepts_a_plain_dict():
    """`_with_caller` takes the field as-is when it isn't an SDK model.

    Nothing feeds it a plain dict today — `load_history` puts stored
    blocks straight into `_messages` without re-serializing — so this
    covers the fallback branch, not a live path."""
    from meeko.claude_client import _serialize_block

    block = SimpleNamespace(
        type="server_tool_use",
        id="srvtoolu_nested",
        name="web_search",
        input={},
        caller=dict(_CALLER),
    )
    assert _serialize_block(block)["caller"] == _CALLER


def test_serialize_nested_search_turn_keeps_every_caller():
    """The real shape of a dynamic-filtering turn: an outer
    code_execution with no caller of its own, then nested web_search
    calls and their results, each tagged with the outer call's id. Every
    tag must survive, or the replay 400s."""
    from meeko.claude_client import _serialize_block

    turn = [
        SimpleNamespace(
            type="server_tool_use",
            id="srvtoolu_outer",
            name="code_execution",
            input={"code": "web_search(...)"},
        ),
        SimpleNamespace(
            type="server_tool_use",
            id="srvtoolu_nested",
            name="web_search",
            input={"query": "x"},
            caller=_caller_obj(),
        ),
        SimpleNamespace(
            type="web_search_tool_result",
            tool_use_id="srvtoolu_nested",
            content=[],
            caller=_caller_obj(),
        ),
    ]
    serialized = [_serialize_block(b) for b in turn]

    assert "caller" not in serialized[0]
    assert all(b["caller"] == _CALLER for b in serialized[1:])


def test_serialize_web_search_tool_result_preserves_encrypted_content():
    from meeko.claude_client import _serialize_block

    result_item = SimpleNamespace(
        type="web_search_result",
        url="https://example.com/a",
        title="Example",
        encrypted_content="ENC1",
        page_age="April 30, 2025",
        # SDK helper that must not survive into the next request.
        extra_helper="strip-me",
    )
    block = SimpleNamespace(
        type="web_search_tool_result",
        tool_use_id="srvtoolu_abc",
        content=[result_item],
    )
    serialized = _serialize_block(block)
    assert serialized["type"] == "web_search_tool_result"
    assert serialized["tool_use_id"] == "srvtoolu_abc"
    [item] = serialized["content"]
    assert item == {
        "type": "web_search_result",
        "url": "https://example.com/a",
        "title": "Example",
        "encrypted_content": "ENC1",
        "page_age": "April 30, 2025",
    }


def test_serialize_web_search_tool_result_error_shape():
    from meeko.claude_client import _serialize_block

    error_content = SimpleNamespace(
        type="web_search_tool_result_error",
        error_code="max_uses_exceeded",
        model_dump=lambda: {
            "type": "web_search_tool_result_error",
            "error_code": "max_uses_exceeded",
        },
    )
    block = SimpleNamespace(
        type="web_search_tool_result",
        tool_use_id="srvtoolu_err",
        content=error_content,
    )
    serialized = _serialize_block(block)
    assert serialized["content"] == {
        "type": "web_search_tool_result_error",
        "error_code": "max_uses_exceeded",
    }


def test_web_search_tool_included_by_default(fake_anthropic):
    client = _make_client()
    types = [t.get("type") for t in client._tools]
    assert "web_search_20260209" in types


def test_web_search_tool_disabled_via_kwarg(fake_anthropic):
    """web_search_enabled=False omits the server tool from the tool list."""
    client = ClaudeClient(
        api_key="k",
        system_prompt="sys",
        dispatcher=ToolDispatcher(),
        web_search_enabled=False,
    )
    types = [t.get("type") for t in client._tools]
    assert "web_search_20260209" not in types


def test_web_search_max_uses_kwarg_threads_into_tool_definition(fake_anthropic):
    client = ClaudeClient(
        api_key="k",
        system_prompt="sys",
        dispatcher=ToolDispatcher(),
        web_search_max_uses=7,
    )
    ws = next(t for t in client._tools if t.get("type") == "web_search_20260209")
    assert ws["max_uses"] == 7


class _PauseThenEndStream(_FakeStream):
    """First-round stream returns stop_reason=pause_turn so the loop must
    re-enter without dispatching tools; second-round behaves normally."""

    def __init__(
        self,
        captured: list[dict[str, Any]],
        kwargs: dict[str, Any],
        round_counter: dict[str, int],
    ):
        super().__init__(captured, kwargs)
        self._round = round_counter

    @property
    def text_stream(self):
        async def _iter():
            if self._round["n"] == 0:
                yield "Searching. "
            else:
                yield "Done."

        return _iter()

    async def get_final_message(self):
        if self._round["n"] == 0:
            self._round["n"] += 1
            text_block = SimpleNamespace(type="text", text="Searching. ")
            return SimpleNamespace(
                content=[text_block],
                stop_reason="pause_turn",
                usage=SimpleNamespace(
                    input_tokens=10,
                    output_tokens=2,
                    cache_creation_input_tokens=0,
                    cache_read_input_tokens=0,
                ),
            )
        text_block = SimpleNamespace(type="text", text="Done.")
        return SimpleNamespace(
            content=[text_block],
            stop_reason="end_turn",
            usage=SimpleNamespace(
                input_tokens=12,
                output_tokens=2,
                cache_creation_input_tokens=0,
                cache_read_input_tokens=0,
            ),
        )


@pytest.mark.asyncio
async def test_pause_turn_continues_loop_without_dispatch(monkeypatch):
    round_counter = {"n": 0}
    captured = _install_stream_factory(
        monkeypatch,
        lambda captured, kw: _PauseThenEndStream(captured, kw, round_counter),
    )
    client = _make_client()
    await _drain(client.stream_turn("Search the web."))

    # Two rounds: first returned pause_turn, second returned end_turn.
    assert len(captured) == 2

    # Round 2's request ends with the paused content — committed now,
    # not carried in a transient message — which is what tells the
    # server where to resume.
    round2_msgs = captured[1]["messages"]
    assert round2_msgs[-1]["role"] == "assistant"
    paused_content = round2_msgs[-1]["content"]
    paused_texts = [b.get("text") for b in paused_content if b.get("type") == "text"]
    assert any(t and "Searching" in t for t in paused_texts)

    # Each round commits its own blocks, so the paused round and its
    # continuation are two assistant messages. The API combines
    # consecutive same-role turns, so they read as one turn
    # (tests/test_claude_api_contract.py).
    roles = [m["role"] for m in client._messages]
    assert roles == ["user", "assistant", "assistant"]
    text = "".join(
        b["text"]
        for m in client._messages
        if m["role"] == "assistant"
        for b in m["content"]
        if b.get("type") == "text"
    )
    assert "Searching" in text and "Done" in text


class _AlwaysPauseStream(_FakeStream):
    """Stream that always returns stop_reason=pause_turn so the round
    loop exhausts its budget."""

    @property
    def text_stream(self):
        async def _iter():
            yield "thinking. "

        return _iter()

    async def get_final_message(self):
        text_block = SimpleNamespace(type="text", text="thinking. ")
        return SimpleNamespace(
            content=[text_block],
            stop_reason="pause_turn",
            usage=SimpleNamespace(
                input_tokens=10,
                output_tokens=2,
                cache_creation_input_tokens=0,
                cache_read_input_tokens=0,
            ),
        )


@pytest.mark.asyncio
async def test_every_paused_round_is_committed_as_it_arrives(monkeypatch):
    """Each pause_turn round commits its own blocks, so exhausting the
    round budget needs no flush at loop exit — nothing is left
    uncommitted to flush."""
    captured = _install_stream_factory(monkeypatch, _AlwaysPauseStream)
    client = _make_client()
    await _drain(client.stream_turn("hello"))

    assert len(captured) == claude_client_module.MAX_TOOL_ROUNDS
    roles = [m["role"] for m in client._messages]
    assert roles == ["user"] + ["assistant"] * claude_client_module.MAX_TOOL_ROUNDS


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
async def test_server_side_compaction_fires_on_long_prefix(caplog):
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
    client = ClaudeClient(
        api_key=api_key,
        system_prompt=_COMPACTION_SYSTEM_PROMPT,
        dispatcher=ToolDispatcher(),
        compaction_trigger_tokens=50000,
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
    gate = asyncio.Event()
    _install_stream_factory(
        monkeypatch,
        lambda captured, kw: _CancellableStream(
            captured, kw, gate, ["Hello there. ", "I was about"]
        ),
    )
    store = _RecordingStore()
    client = _make_client(store=store, session_id="sess-1")

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
    roles = [r for _, r, _ in store.persisted]
    assert roles == ["user", "assistant"]
    assert "Hello there" in store.persisted[1][2][0]["text"]


async def test_cancelled_stream_with_no_output_commits_nothing(monkeypatch):
    """A cancel before any token streamed has nothing worth recording.
    History keeps the lone user message, and the next user message
    merges with it — which the API accepts, so the old "…" placeholder
    turn was cosmetic (tests/test_claude_api_contract.py)."""
    gate = asyncio.Event()
    _install_stream_factory(
        monkeypatch, lambda captured, kw: _CancellableStream(captured, kw, gate, [])
    )
    client = _make_client()

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

    assert [m["role"] for m in client._messages] == ["user"]


class _RaisingStream:
    """A streaming response that yields a couple of deltas, then raises
    a non-cancel exception to simulate a network/API error mid-stream."""

    def __init__(
        self,
        captured: list[dict[str, Any]],
        kwargs: dict[str, Any],
        early_deltas: list[str],
        exc: BaseException,
    ):
        self._captured = captured
        self._kwargs = kwargs
        self._early_deltas = early_deltas
        self._exc = exc

    async def __aenter__(self):
        self._captured.append(copy.deepcopy(self._kwargs))
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    @property
    def text_stream(self):
        deltas = self._early_deltas
        exc = self._exc

        async def _iter():
            for d in deltas:
                yield d
            raise exc

        return _iter()

    async def get_final_message(self):  # pragma: no cover - never reached
        raise AssertionError("stream raised before final message")


class _TailOnlyStream:
    """Stream that yields a single delta with no sentence terminator,
    so the stream context exits cleanly and stream_turn's only yield
    is the trailing buffer at `yield tail` — outside the try/except."""

    def __init__(self, captured: list[dict[str, Any]], kwargs: dict[str, Any]):
        self._captured = captured
        self._kwargs = kwargs

    async def __aenter__(self):
        self._captured.append(copy.deepcopy(self._kwargs))
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    @property
    def text_stream(self):
        async def _iter():
            yield "Partial reply"  # no terminator → buffered into tail

        return _iter()

    async def get_final_message(self):
        text_block = SimpleNamespace(type="text", text="Partial reply")
        usage = SimpleNamespace(
            input_tokens=1,
            output_tokens=1,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
        )
        return SimpleNamespace(
            content=[text_block],
            stop_reason="end_turn",
            usage=usage,
        )


async def test_cancel_at_tail_yield_keeps_history_balanced(monkeypatch):
    """If the consumer cancels while stream_turn is suspended at the
    final `yield tail` (which is outside the try/except wrapping the
    stream context), history must already be balanced — the assistant
    turn must be appended *before* the yield, not after, otherwise a
    barge-in landing on this exact suspension point leaves an orphan
    user message."""
    _install_stream_factory(monkeypatch, _TailOnlyStream)
    store = _RecordingStore()
    client = _make_client(store=store, session_id="sess-1")

    agen = client.stream_turn("Hi Claude")
    # Drive the generator to its first (and only) yield — `yield tail`.
    first = await agen.__anext__()
    assert first == "Partial reply"

    # At this suspension point, before the consumer closes us, the
    # assistant turn must already be in history. Without the fix this
    # assertion fails: append happens after the yield.
    assert client._messages[-2] == {"role": "user", "content": "Hi Claude"}
    assert client._messages[-1]["role"] == "assistant"
    assert client._messages[-1]["content"][0]["text"] == "Partial reply"
    assert [r for _, r, _ in store.persisted] == ["user", "assistant"]

    # Simulate barge-in cancelling us at this suspension point.
    # Should unwind cleanly without rolling back or double-appending.
    await agen.aclose()

    assert client._messages[-1]["role"] == "assistant"
    assert client._messages[-1]["content"][0]["text"] == "Partial reply"
    # No duplicate assistant append from the cancel path.
    assert sum(1 for m in client._messages if m["role"] == "assistant") == 1


async def test_stream_error_commits_partial_assistant_turn(monkeypatch):
    """A non-cancel error mid-stream (e.g. a network drop) must still
    commit a partial assistant turn — otherwise the next stream_turn
    call appends a second user message and the API 400s on consecutive
    user roles."""
    _install_stream_factory(
        monkeypatch,
        lambda captured, kw: _RaisingStream(
            captured,
            kw,
            ["Hello there. ", "I was about"],
            RuntimeError("network dropped"),
        ),
    )
    store = _RecordingStore()
    client = _make_client(store=store, session_id="sess-1")

    with pytest.raises(RuntimeError, match="network dropped"):
        async for _ in client.stream_turn("Hi Claude"):
            pass

    # History must be balanced: user followed by partial assistant,
    # so a follow-up turn doesn't stack a second user message.
    assert client._messages[-2] == {"role": "user", "content": "Hi Claude"}
    last = client._messages[-1]
    assert last["role"] == "assistant"
    assert "Hello there" in last["content"][0]["text"]

    # SQLite mirrors in-memory: both turns persisted.
    assert [r for _, r, _ in store.persisted] == ["user", "assistant"]


# --- stranded server tool uses ------------------------------------------
#
# A `pause_turn` round ends with a `server_tool_use` that has no result —
# that's what the server is still working on. Keeping it is required to
# resume; keeping it once a user turn follows makes the API 400 the whole
# request (both verified against the live API).

_PAUSED_CONTENT = [
    {"type": "text", "text": "Let me look that up."},
    {
        "type": "server_tool_use",
        "id": "srvtoolu_outer",
        "name": "code_execution",
        "input": {"code": "web_search(...)"},
    },
    {
        "type": "server_tool_use",
        "id": "srvtoolu_nested",
        "name": "web_search",
        "input": {"query": "x"},
        "caller": {"tool_id": "srvtoolu_outer", "type": "code_execution_20260120"},
    },
]


def test_stranded_server_tool_use_is_kept_when_it_trails_the_history():
    """The resume case: the paused content is the last message, and its
    trailing orphan is what tells the server where to pick up."""
    from meeko.claude_client import _drop_stranded_server_tool_uses

    messages = [
        {"role": "user", "content": "what's the weather?"},
        {"role": "assistant", "content": _PAUSED_CONTENT},
    ]
    assert _drop_stranded_server_tool_uses(messages) == messages


def test_stranded_server_tool_use_is_dropped_once_a_turn_follows():
    """The barge-in case: a user turn after the paused content would
    otherwise 400 the request, and every later turn of the session."""
    from meeko.claude_client import _drop_stranded_server_tool_uses

    messages = [
        {"role": "user", "content": "what's the weather?"},
        {"role": "assistant", "content": _PAUSED_CONTENT},
        {"role": "user", "content": "never mind"},
    ]
    out = _drop_stranded_server_tool_uses(messages)

    # The nested call goes with the outer one it was made from.
    assert out[1]["content"] == [{"type": "text", "text": "Let me look that up."}]
    # Everything else is untouched, and the input list isn't mutated.
    assert out[0] == messages[0] and out[2] == messages[2]
    assert len(messages[1]["content"]) == 3


def test_answered_server_tool_uses_survive():
    from meeko.claude_client import _drop_stranded_server_tool_uses

    content = [
        {
            "type": "server_tool_use",
            "id": "srvtoolu_a",
            "name": "web_search",
            "input": {},
        },
        {"type": "web_search_tool_result", "tool_use_id": "srvtoolu_a", "content": []},
    ]
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": content},
        {"role": "user", "content": "and then?"},
    ]
    assert _drop_stranded_server_tool_uses(messages) == messages


def test_message_of_only_a_stranded_call_keeps_a_placeholder():
    """The API rejects an empty content list."""
    from meeko.claude_client import _drop_stranded_server_tool_uses

    messages = [
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "server_tool_use",
                    "id": "srvtoolu_a",
                    "name": "web_search",
                    "input": {},
                }
            ],
        },
        {"role": "user", "content": "never mind"},
    ]
    assert _drop_stranded_server_tool_uses(messages)[1]["content"] == [
        {"type": "text", "text": "…"}
    ]


def test_client_tool_use_is_left_to_the_pairing_pass():
    """Client `tool_use` gets an interrupted `tool_result` instead —
    that repair is `_pair_orphan_tool_uses`'s job, not this one."""
    from meeko.claude_client import _drop_stranded_server_tool_uses

    content = [{"type": "tool_use", "id": "toolu_a", "name": "set_timer", "input": {}}]
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": content},
        {"role": "user", "content": "never mind"},
    ]
    assert _drop_stranded_server_tool_uses(messages) == messages


@pytest.mark.asyncio
async def test_next_turn_after_a_paused_round_omits_the_stranded_call(fake_anthropic):
    """End-to-end: a session whose history carries a paused round (a
    barge-in mid-web-search, or a resumed one from SQLite) must still
    send a request the API accepts. The stranded call is filtered from
    the send-time view while stored history stays verbatim."""
    client = _make_client()
    client.load_history(
        [
            {"role": "user", "content": "what's the weather?"},
            {"role": "assistant", "content": _PAUSED_CONTENT},
        ]
    )

    await _drain(client.stream_turn("never mind, say hi"))

    sent = fake_anthropic["client"].captured[0]["messages"]
    assistant_blocks = sent[1]["content"]
    assert [b["type"] for b in assistant_blocks] == ["text"]

    # Stored history is untouched — SQLite stays the verbatim record.
    assert len(client._messages[1]["content"]) == 3


class _PauseThenHangStream(_FakeStream):
    """Pauses on the first round, then hangs mid-stream on the resume so
    the test can cancel exactly where a barge-in would land."""

    def __init__(
        self,
        captured: list[dict[str, Any]],
        kwargs: dict[str, Any],
        round_counter: dict[str, int],
        gate: asyncio.Event,
    ):
        super().__init__(captured, kwargs)
        self._round = round_counter
        self._gate = gate

    @property
    def text_stream(self):
        first = self._round["n"] == 0
        gate = self._gate

        async def _iter():
            if first:
                yield "Let me look that up. "
                return
            yield "The answer is"
            await gate.wait()  # block until the test cancels us
            yield "never reached"

        return _iter()

    async def get_final_message(self):
        assert self._round["n"] == 0
        self._round["n"] += 1
        return SimpleNamespace(
            content=[
                SimpleNamespace(type="text", text="Let me look that up. "),
                SimpleNamespace(
                    type="server_tool_use",
                    id="srvtoolu_paused",
                    name="web_search",
                    input={"query": "x"},
                ),
            ],
            stop_reason="pause_turn",
            usage=SimpleNamespace(
                input_tokens=10,
                output_tokens=2,
                cache_creation_input_tokens=0,
                cache_read_input_tokens=0,
            ),
        )


@pytest.mark.asyncio
async def test_paused_round_is_persisted_before_the_resume_request(monkeypatch):
    """The paused round is committed on arrival, so it reaches SQLite
    even if the resume never completes."""
    round_counter = {"n": 0}
    gate = asyncio.Event()
    captured = _install_stream_factory(
        monkeypatch,
        lambda cap, kw: _PauseThenHangStream(cap, kw, round_counter, gate),
    )
    store = _RecordingStore()
    client = _make_client(store=store, session_id="sess-1")

    task = asyncio.create_task(_drain(client.stream_turn("what's the weather?")))
    for _ in range(100):
        if len(captured) == 2:
            break
        await asyncio.sleep(0)
    assert len(captured) == 2, "expected the resume request to have gone out"

    # The paused round hit SQLite before the resume was sent.
    assert [r for _, r, _ in store.persisted] == ["user", "assistant"]

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_barge_in_during_a_resume_keeps_both_rounds(monkeypatch):
    """Cancelling mid-resume leaves the committed paused round plus a
    partial for what the resume got through — and the next request must
    still be one the API accepts, so the paused round's unanswered
    server_tool_use is filtered out of the send-time view."""
    round_counter = {"n": 0}
    gate = asyncio.Event()
    captured = _install_stream_factory(
        monkeypatch,
        lambda cap, kw: _PauseThenHangStream(cap, kw, round_counter, gate),
    )
    client = _make_client()

    task = asyncio.create_task(_drain(client.stream_turn("what's the weather?")))
    for _ in range(100):
        if len(captured) == 2:
            break
        await asyncio.sleep(0)
    # Let the resume stream its first delta before cancelling.
    for _ in range(10):
        await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    roles = [m["role"] for m in client._messages]
    assert roles == ["user", "assistant", "assistant"]
    # The paused round kept its blocks verbatim, including the orphan.
    assert [b["type"] for b in client._messages[1]["content"]] == [
        "text",
        "server_tool_use",
    ]
    assert client._messages[2]["content"] == [{"type": "text", "text": "The answer is"}]

    # The send-time view of the next turn drops the stranded call.
    sent = _drop_stranded_server_tool_uses(client._messages)
    assert [b["type"] for b in sent[1]["content"]] == ["text"]


# A resumed pause_turn's continuation *starts* with the result for the
# tool use the previous message left unanswered — verified against the
# live API. With per-round commits the pair therefore spans two
# assistant messages, so "answered" has to be judged conversation-wide.
_RESUME_CONTINUATION = [
    {
        "type": "code_execution_tool_result",
        "tool_use_id": "srvtoolu_paused",
        "content": [],
    },
    {"type": "text", "text": "It's 54 and raining."},
]


def test_a_use_answered_by_the_next_message_is_not_stranded():
    """The pair spans two messages after a resume. Dropping the call
    here would strand its result instead — the mirror-image 400."""
    messages = [
        {"role": "user", "content": "what's the weather?"},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Let me look."},
                {
                    "type": "server_tool_use",
                    "id": "srvtoolu_paused",
                    "name": "code_execution",
                    "input": {},
                },
            ],
        },
        {"role": "assistant", "content": _RESUME_CONTINUATION},
        {"role": "user", "content": "thanks"},
    ]
    assert _drop_stranded_server_tool_uses(messages) == messages


def test_only_the_still_unanswered_call_is_dropped_across_rounds():
    """Two paused rounds: the first round's call gets answered by the
    second, whose own trailing call never does. Only the latter goes."""
    messages = [
        {"role": "user", "content": "research this"},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "server_tool_use",
                    "id": "srvtoolu_one",
                    "name": "code_execution",
                    "input": {},
                }
            ],
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "code_execution_tool_result",
                    "tool_use_id": "srvtoolu_one",
                    "content": [],
                },
                {
                    "type": "server_tool_use",
                    "id": "srvtoolu_two",
                    "name": "code_execution",
                    "input": {},
                },
            ],
        },
        {"role": "user", "content": "never mind"},
    ]
    out = _drop_stranded_server_tool_uses(messages)

    # Round 1's call survives: round 2 answers it.
    assert out[1] == messages[1]
    # Round 2's own trailing call goes, and its result for round 1 stays.
    assert [b["type"] for b in out[2]["content"]] == ["code_execution_tool_result"]


def test_a_result_addressed_to_a_dropped_call_goes_with_it():
    """Nothing may be left pointing at a block that isn't there."""
    messages = [
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "server_tool_use",
                    "id": "srvtoolu_outer",
                    "name": "code_execution",
                    "input": {},
                },
                {
                    "type": "server_tool_use",
                    "id": "srvtoolu_nested",
                    "name": "web_search",
                    "input": {},
                    "caller": {
                        "tool_id": "srvtoolu_outer",
                        "type": "code_execution_20260120",
                    },
                },
                {
                    "type": "web_search_tool_result",
                    "tool_use_id": "srvtoolu_nested",
                    "content": [],
                    "caller": {
                        "tool_id": "srvtoolu_outer",
                        "type": "code_execution_20260120",
                    },
                },
            ],
        },
        {"role": "user", "content": "never mind"},
    ]
    # The outer call is never answered, so the whole group goes and the
    # message falls back to the placeholder.
    assert _drop_stranded_server_tool_uses(messages)[1]["content"] == [
        {"type": "text", "text": "…"}
    ]


def test_non_dict_blocks_do_not_crash_the_filter():
    """Stored history is JSON cast in unchecked by `load_history`."""
    messages = [
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": [
                "not a block",
                {
                    "type": "server_tool_use",
                    "id": "srvtoolu_a",
                    "name": "web_search",
                    "input": {},
                },
            ],
        },
        {"role": "user", "content": "never mind"},
    ]
    assert _drop_stranded_server_tool_uses(messages)[1]["content"] == ["not a block"]


class _PauseThenResolveStream(_FakeStream):
    """A faithful resume: round 1 pauses with an unanswered call, and
    round 2 opens with that call's result, the way the live API does."""

    def __init__(self, captured, kwargs, round_counter):
        super().__init__(captured, kwargs)
        self._round = round_counter

    @property
    def text_stream(self):
        first = self._round["n"] == 0

        async def _iter():
            yield "Let me look. " if first else "It's raining."

        return _iter()

    async def get_final_message(self):
        first = self._round["n"] == 0
        self._round["n"] += 1
        usage = SimpleNamespace(
            input_tokens=10,
            output_tokens=2,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
        )
        if first:
            return SimpleNamespace(
                content=[
                    SimpleNamespace(type="text", text="Let me look. "),
                    SimpleNamespace(
                        type="server_tool_use",
                        id="srvtoolu_paused",
                        name="code_execution",
                        input={},
                    ),
                ],
                stop_reason="pause_turn",
                usage=usage,
            )
        return SimpleNamespace(
            content=[
                SimpleNamespace(
                    type="code_execution_tool_result",
                    tool_use_id="srvtoolu_paused",
                    content=[],
                    model_dump=lambda: {
                        "type": "code_execution_tool_result",
                        "tool_use_id": "srvtoolu_paused",
                        "content": [],
                    },
                ),
                SimpleNamespace(type="text", text="It's raining."),
            ],
            stop_reason="end_turn",
            usage=usage,
        )


@pytest.mark.asyncio
async def test_turn_after_a_completed_pause_keeps_the_answered_call(monkeypatch):
    """End-to-end for the two-message pair: the paused round's call is
    answered by the continuation round, so the next turn must send both
    — dropping the call would strand its result and 400."""
    round_counter = {"n": 0}
    captured = _install_stream_factory(
        monkeypatch,
        lambda cap, kw: _PauseThenResolveStream(cap, kw, round_counter),
    )
    client = _make_client()
    await _drain(client.stream_turn("what's the weather?"))
    round_counter["n"] = 1  # keep the follow-up on the resolved branch
    await _drain(client.stream_turn("thanks"))

    sent = captured[-1]["messages"]
    paused_msg = next(
        m
        for m in sent
        if m["role"] == "assistant"
        and any(b.get("type") == "server_tool_use" for b in m["content"])
    )
    ids = [b["id"] for b in paused_msg["content"] if b.get("type") == "server_tool_use"]
    assert ids == ["srvtoolu_paused"]
    # And its result is still there, in the following message.
    assert any(
        b.get("tool_use_id") == "srvtoolu_paused"
        for m in sent
        if isinstance(m["content"], list)
        for b in m["content"]
    )


# --- the two repairs, composed -------------------------------------------


def test_missing_results_returns_nothing_when_nothing_is_pending():
    assert _missing_results([], {"role": "user", "content": "hi"}) == []


def test_missing_results_skips_ids_the_message_already_answers():
    msg = {
        "role": "user",
        "content": [
            {"type": "tool_result", "tool_use_id": "toolu_a", "content": "done"}
        ],
    }
    filler = _missing_results(["toolu_a", "toolu_b"], msg)
    assert [f["tool_use_id"] for f in filler] == ["toolu_b"]


def test_lead_with_results_normalizes_a_plain_string_message():
    """A user turn is stored as a bare string; results have to lead it,
    so the string becomes a text block behind them."""
    filler = [{"type": "tool_result", "tool_use_id": "toolu_a", "content": "x"}]
    out = _lead_with_results({"role": "user", "content": "never mind"}, filler)
    assert out["content"] == [
        filler[0],
        {"type": "text", "text": "never mind"},
    ]


def test_repaired_applies_both_repairs_to_one_history():
    """A barge-in during a paused web search can leave both kinds of
    orphan in the same session: a client `tool_use` with no result, and
    a server one the pause never finished. Each needs the opposite
    repair, and the send-time view needs both."""
    messages = [
        {"role": "user", "content": "timer and weather?"},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Looking."},
                {
                    "type": "server_tool_use",
                    "id": "srvtoolu_paused",
                    "name": "code_execution",
                    "input": {},
                },
            ],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "toolu_a", "name": "set_timer", "input": {}}
            ],
        },
        {"role": "user", "content": "never mind"},
    ]
    out = _repaired(messages)

    # The stranded server call is gone; its message keeps its text.
    assert out[1]["content"] == [{"type": "text", "text": "Looking."}]
    # The client tool_use is answered, and the results lead the user turn.
    answered = out[3]["content"]
    assert answered[0]["type"] == "tool_result"
    assert answered[0]["tool_use_id"] == "toolu_a"
    assert answered[0]["is_error"] is True
    assert answered[1] == {"type": "text", "text": "never mind"}
    # Input untouched — stored history stays verbatim.
    assert len(messages[1]["content"]) == 2
