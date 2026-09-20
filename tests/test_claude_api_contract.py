"""Live checks of the Messages API contract that `ClaudeClient`'s history
bookkeeping is built on.

`stream_turn` goes to some length to keep the in-memory message array
strictly user/assistant-alternating: it carries `pause_turn` content
forward in `paused_blocks` rather than committing it, commits a partial
assistant turn on every early exit, and substitutes a "…" placeholder
when nothing streamed. The stated reason (see `fddc387`) is that
consecutive same-role messages 400.

The API docs say the opposite — consecutive same-role messages are
combined into a single turn — and nothing in the git history shows the
400 was ever observed. These tests settle it against the live API, using
the same model, betas and context-management config the client sends, so
a simplification can be built on the answer (or dropped if it fails).

Marked `integration`; a few cents per run. Test 3 issues one real web
search.
"""

import os

import anthropic
import pytest

from meeko.claude_client import (
    COMPACTION_BETA,
    DEFAULT_COMPACTION_TRIGGER_TOKENS,
    MODEL,
    _context_management,
    _serialize_block,
    _web_search_tool,
)

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]

SYSTEM = "You are a terse test assistant. Reply in three words or fewer."


def _client() -> anthropic.AsyncAnthropic:
    """Skip inside the body, not via skipif: conftest's autouse
    load_dotenv fixture runs after collection-time decorators."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        pytest.skip("ANTHROPIC_API_KEY not set")
    return anthropic.AsyncAnthropic(api_key=api_key)


async def _send(client, messages, **kwargs):
    """Send exactly what ClaudeClient sends, minus the cache breakpoint."""
    return await client.beta.messages.create(
        model=MODEL,
        max_tokens=512,
        system=SYSTEM,
        messages=messages,
        betas=[COMPACTION_BETA],
        context_management=_context_management(DEFAULT_COMPACTION_TRIGGER_TOKENS),
        **kwargs,
    )


async def test_consecutive_user_messages_are_accepted():
    """Two user messages in a row — what a barge-in leaves behind if the
    partial-assistant commit is removed."""
    client = _client()
    response = await _send(
        client,
        [
            {"role": "user", "content": "Say 'one'."},
            {"role": "user", "content": "Actually, say 'two'."},
        ],
    )
    assert response.stop_reason in ("end_turn", "max_tokens")
    assert any(b.type == "text" and b.text.strip() for b in response.content)


async def test_consecutive_assistant_messages_are_accepted():
    """Two assistant messages in a row followed by a user turn — what
    committing each `pause_turn` round separately would produce."""
    client = _client()
    response = await _send(
        client,
        [
            {"role": "user", "content": "Count to two, one number per message."},
            {"role": "assistant", "content": [{"type": "text", "text": "One."}]},
            {"role": "assistant", "content": [{"type": "text", "text": "Two."}]},
            {"role": "user", "content": "What number came last?"},
        ],
    )
    assert response.stop_reason in ("end_turn", "max_tokens")
    assert any(b.type == "text" and b.text.strip() for b in response.content)


async def test_split_web_search_turn_is_accepted():
    """The `pause_turn` shape, as closely as it can be staged offline of a
    real pause: take a genuine web-search assistant turn and split it at
    the `web_search_tool_result` boundary into two assistant messages,
    the way committing a paused round and its continuation separately
    would, then continue the conversation.

    A real `pause_turn` can't be triggered on demand, so this stands in
    for it: same block types, same split point, same trailing-user
    follow-up.
    """
    client = _client()
    tools = [_web_search_tool(2)]

    query = "Search the web: what is the capital of France?"
    first = await _send(
        client,
        [{"role": "user", "content": query}],
        tools=tools,
    )
    blocks = [_serialize_block(b) for b in first.content]
    split = next(
        (i for i, b in enumerate(blocks) if b.get("type") == "web_search_tool_result"),
        None,
    )
    if split is None:
        types = [b.get("type") for b in blocks]
        pytest.skip(f"model did not search; block types={types}")
    head, tail = blocks[: split + 1], blocks[split + 1 :]
    if not tail:
        pytest.skip("no content after the search result to split off")

    response = await _send(
        client,
        [
            {"role": "user", "content": query},
            {"role": "assistant", "content": head},
            {"role": "assistant", "content": tail},
            {"role": "user", "content": "Name that city again."},
        ],
        tools=tools,
    )
    assert response.stop_reason in ("end_turn", "max_tokens", "pause_turn")
    assert any(b.type == "text" and b.text.strip() for b in response.content)


async def test_nested_web_search_turn_replays_on_the_next_turn():
    """End-to-end regression for the dropped `caller` field.

    A web-search turn whose searches ran inside `code_execution`
    (dynamic filtering) must replay on the following turn. Before the
    fix, `_serialize_block` dropped each nested block's `caller`, and
    the API answered the *next* turn with

        `code_execution` tool use with id ... was found without a
        corresponding `code_execution_tool_result` block

    which killed every later turn of that session.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        pytest.skip("ANTHROPIC_API_KEY not set")

    from meeko.claude_client import ClaudeClient
    from meeko.tools.dispatch import ToolDispatcher

    client = ClaudeClient(
        api_key=api_key,
        system_prompt="You are a terse research assistant. Two sentences maximum.",
        dispatcher=ToolDispatcher(),
    )

    async for _ in client.stream_turn(
        "Search the web for the height of the tallest building in Dallas, "
        "and separately the tallest in Atlanta."
    ):
        pass

    nested = [
        b
        for m in client._messages
        if m["role"] == "assistant" and isinstance(m["content"], list)
        for b in m["content"]
        if isinstance(b, dict) and b.get("caller")
    ]
    if not nested:
        pytest.skip("model did not nest its searches inside code_execution")

    # The turn that used to 400: it replays the whole nested turn.
    replies = [s async for s in client.stream_turn("Which of the two is taller?")]
    assert any(s.strip() for s in replies)
