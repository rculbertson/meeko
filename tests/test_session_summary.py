"""Tests for `meeko.session_summary`.

Covers transcript assembly (skipping tool blocks, decoding JSON), the
happy-path write to `update_session_metadata`, and defensive behavior
when the Anthropic call fails or returns garbage."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from meeko.session_summary import (
    MAX_TOKENS,
    MODEL,
    _build_transcript,
    summarize_session,
)
from meeko.sessions import SessionStore


@pytest.fixture
async def store(tmp_path):
    s = SessionStore.open(tmp_path / "meeko.db")
    yield s
    await s.close()


def _text_response(text: str):
    """Build a fake `messages.create` response with a single text block."""
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)],
        stop_reason="end_turn",
    )


def _fake_anthropic_client(response) -> MagicMock:
    client = MagicMock()
    client.messages = MagicMock()
    client.messages.create = AsyncMock(return_value=response)
    return client


def test_build_transcript_joins_user_and_assistant_turns():
    turns = [
        {"role": "user", "content": "hello there"},
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "hi, what are we building?"}],
        },
        {"role": "user", "content": "a todo app"},
    ]
    out = _build_transcript(turns)
    assert "USER: hello there" in out
    assert "ASSISTANT: hi, what are we building?" in out
    assert "USER: a todo app" in out


def test_build_transcript_skips_tool_use_and_tool_result_blocks():
    """tool_use and tool_result content must not leak into the FTS index."""
    turns = [
        {"role": "user", "content": "set a timer for 5 minutes"},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Setting a 5 minute timer."},
                {
                    "type": "tool_use",
                    "id": "tu_1",
                    "name": "start_timer",
                    "input": {"seconds": 300},
                },
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "tu_1", "content": "ok"}
            ],
        },
        {"role": "assistant", "content": [{"type": "text", "text": "Done."}]},
    ]
    out = _build_transcript(turns)
    assert "tu_1" not in out
    assert "start_timer" not in out
    assert "tool_result" not in out
    assert "Setting a 5 minute timer." in out
    assert "Done." in out


async def test_summarize_session_writes_title_summary_transcript(store):
    sid = await store.create_session("query")
    await store.persist_turn(sid, "user", "let's build a todo app")
    await store.persist_turn(
        sid,
        "assistant",
        [{"type": "text", "text": "sounds good, what backend?"}],
    )

    client = _fake_anthropic_client(
        _text_response(
            json.dumps(
                {
                    "title": "Todo app backend",
                    "summary": "Started scoping a todo app and discussed backends.",
                }
            )
        )
    )

    await summarize_session(store, sid, client)

    # Verify the Anthropic call shape.
    client.messages.create.assert_awaited_once()
    kwargs = client.messages.create.await_args.kwargs
    assert kwargs["model"] == MODEL
    assert kwargs["max_tokens"] == MAX_TOKENS
    prompt = kwargs["messages"][0]["content"]
    assert "USER: let's build a todo app" in prompt
    assert "ASSISTANT: sounds good, what backend?" in prompt

    # Verify the sessions row and FTS row were updated. Reaching into
    # the store's connection is a test-only shortcut; the public
    # get_session() doesn't return title/summary.
    title_summary = store._conn.execute(
        "SELECT title, summary FROM sessions WHERE id = ?", (sid,)
    ).fetchone()
    assert title_summary == (
        "Todo app backend",
        "Started scoping a todo app and discussed backends.",
    )

    fts_row = store._conn.execute(
        "SELECT title, summary, transcript FROM sessions_fts WHERE session_id = ?",
        (sid,),
    ).fetchone()
    assert fts_row is not None
    assert fts_row[0] == "Todo app backend"
    assert "todo app" in fts_row[2]


async def test_summarize_session_skips_when_no_turns(store):
    sid = await store.create_session("query")
    client = _fake_anthropic_client(_text_response("{}"))
    await summarize_session(store, sid, client)
    client.messages.create.assert_not_awaited()
    row = store._conn.execute(
        "SELECT title, summary FROM sessions WHERE id = ?", (sid,)
    ).fetchone()
    assert row == (None, None)


async def test_summarize_session_handles_markdown_fenced_json(store):
    """Sonnet sometimes wraps the JSON in a ```json fence despite the
    prompt asking for plain JSON. Extract the object instead of failing."""
    sid = await store.create_session("query")
    await store.persist_turn(sid, "user", "hi")
    fenced = (
        "```json\n"
        '{\n  "title": "Kitchen gifts",\n  "summary": "Narrowed on a meat '
        'thermometer."\n}\n'
        "```"
    )
    client = _fake_anthropic_client(_text_response(fenced))
    await summarize_session(store, sid, client)
    row = store._conn.execute(
        "SELECT title, summary FROM sessions WHERE id = ?", (sid,)
    ).fetchone()
    assert row == ("Kitchen gifts", "Narrowed on a meat thermometer.")


async def test_summarize_session_handles_prose_wrapped_json(store):
    """Also tolerant of 'Sure, here's the summary:' prefixes."""
    sid = await store.create_session("query")
    await store.persist_turn(sid, "user", "hi")
    prose = (
        "Sure, here's the JSON summary:\n"
        '{"title": "Quick chat", "summary": "Said hi, that was it."}\n'
        "Hope that helps!"
    )
    client = _fake_anthropic_client(_text_response(prose))
    await summarize_session(store, sid, client)
    row = store._conn.execute(
        "SELECT title, summary FROM sessions WHERE id = ?", (sid,)
    ).fetchone()
    assert row == ("Quick chat", "Said hi, that was it.")


async def test_summarize_session_survives_invalid_json(store):
    sid = await store.create_session("query")
    await store.persist_turn(sid, "user", "hi")
    client = _fake_anthropic_client(
        _text_response("sure, here's a summary: not-valid-json")
    )
    # Should not raise; should leave metadata NULL.
    await summarize_session(store, sid, client)
    row = store._conn.execute(
        "SELECT title, summary FROM sessions WHERE id = ?", (sid,)
    ).fetchone()
    assert row == (None, None)


async def test_summarize_session_survives_missing_keys(store):
    sid = await store.create_session("query")
    await store.persist_turn(sid, "user", "hi")
    # Valid JSON but missing the required keys.
    client = _fake_anthropic_client(_text_response('{"title": "x"}'))
    await summarize_session(store, sid, client)
    row = store._conn.execute(
        "SELECT title, summary FROM sessions WHERE id = ?", (sid,)
    ).fetchone()
    assert row == (None, None)


async def test_summarize_session_survives_anthropic_error(store):
    sid = await store.create_session("query")
    await store.persist_turn(sid, "user", "hi")
    client = MagicMock()
    client.messages = MagicMock()
    client.messages.create = AsyncMock(side_effect=RuntimeError("api down"))
    # Must not propagate — summarization is best-effort.
    await summarize_session(store, sid, client)
    row = store._conn.execute(
        "SELECT title, summary FROM sessions WHERE id = ?", (sid,)
    ).fetchone()
    assert row == (None, None)


async def test_summarize_session_survives_non_text_first_block(store):
    """If the first content block is not a text block, walk the list —
    never index or attribute-access blindly."""
    sid = await store.create_session("query")
    await store.persist_turn(sid, "user", "hi")
    response = SimpleNamespace(
        content=[
            SimpleNamespace(type="tool_use", id="x", name="y", input={}),
            SimpleNamespace(
                type="text",
                text=json.dumps({"title": "OK", "summary": "all fine"}),
            ),
        ],
        stop_reason="end_turn",
    )
    client = _fake_anthropic_client(response)
    await summarize_session(store, sid, client)
    row = store._conn.execute(
        "SELECT title, summary FROM sessions WHERE id = ?", (sid,)
    ).fetchone()
    assert row == ("OK", "all fine")
