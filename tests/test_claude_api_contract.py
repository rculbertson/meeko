"""Live checks of the Messages API contract `ClaudeClient` relies on.

`stream_turn` commits each round's blocks as they arrive and lets the
API combine consecutive same-role turns, rather than keeping history
strictly user/assistant-alternating. That rests on a documented
guarantee — "consecutive `user` or `assistant` turns in your request
will be combined into a single turn" — which these tests hold the API
to, using the same model, betas and context-management config the
client sends.

The history bookkeeping here was originally built on the opposite
assumption, so these run against the live API rather than a fake: a
premise this load-bearing is worth re-checking, and the fake would just
agree with whatever we believed.

Not covered here: the `pause_turn` shapes. A pause needs the server's
10-iteration loop limit, which no test can provoke on demand, so those
were verified by replaying a captured paused turn and are modelled
offline in tests/test_claude_client.py.

Marked `integration`; a few cents per run. Two of these issue real web
searches.
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


def _group_boundary(blocks: list) -> int | None:
    """Pick a split index that doesn't cut a server-tool group apart.

    With dynamic filtering the model calls `web_search` from inside
    `code_execution`, so the blocks come in groups: the outer
    `server_tool_use`, the nested calls and their results, then the
    outer result. Splitting mid-group strands a tool use without its
    result, and the API rejects that — which would fail this test for a
    reason that has nothing to do with the alternation it checks. Only
    offer boundaries where every tool use so far has been answered.
    """
    open_ids: set[str] = set()
    points: list[int] = []
    for i, b in enumerate(blocks):
        block_type = b.get("type")
        if block_type == "server_tool_use":
            open_ids.add(b["id"])
        elif block_type and block_type.endswith("tool_result"):
            open_ids.discard(b.get("tool_use_id"))
        if not open_ids and 0 < i + 1 < len(blocks):
            points.append(i + 1)
    return points[-1] if points else None


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

    query = (
        "Search the web for the height of the tallest building in Dallas "
        "and separately the tallest in Atlanta."
    )
    first = await _send(
        client,
        [{"role": "user", "content": query}],
        tools=tools,
    )
    blocks = [_serialize_block(b) for b in first.content]
    split = _group_boundary(blocks)
    if split is None:
        types = [b.get("type") for b in blocks]
        pytest.skip(f"no safe split point; block types={types}")
    head, tail = blocks[:split], blocks[split:]

    response = await _send(
        client,
        [
            {"role": "user", "content": query},
            {"role": "assistant", "content": head},
            {"role": "assistant", "content": tail},
            {"role": "user", "content": "Which building was taller?"},
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
