"""Thin wrapper around the Anthropic async client.

Maintains the in-memory message history for a single Meeko session and
runs the tool-use loop against Claude's streaming API, yielding
sentence-sized text chunks as they are generated so the caller can
begin TTS before the full reply is ready.
"""

import asyncio
import logging
import re
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from datetime import datetime
from typing import Any

import anthropic

from meeko.sessions import SessionStore
from meeko.tools.dispatch import ToolDispatcher

logger = logging.getLogger("meeko")

MODEL = "claude-sonnet-4-6"
MAX_TOKENS = 8192
MAX_TOOL_ROUNDS = 10

# Server-side compaction (beta `compact-2026-01-12`). When the API sees
# input tokens cross this threshold it summarizes the early portion of the
# message array in-flight and prepends a `compaction` block to the assistant
# response. SQLite remains the verbatim source of truth — the summary lives
# only in the in-memory message array (see `_persist`).
COMPACTION_BETA = "compact-2026-01-12"
COMPACTION_STRATEGY = "compact_20260112"
DEFAULT_COMPACTION_TRIGGER_TOKENS = 150000

# Anthropic-hosted web search server tool. When enabled, Sonnet decides
# per-turn whether to issue searches; results are inlined into the
# assistant message as `server_tool_use` + `web_search_tool_result`
# blocks without any client-side dispatch. `max_uses` caps worst-case
# latency and cost per turn ($10 per 1k searches).
DEFAULT_WEB_SEARCH_ENABLED = True
DEFAULT_WEB_SEARCH_MAX_USES = 2


def _context_management(trigger_tokens: int) -> dict:
    return {
        "edits": [
            {
                "type": COMPACTION_STRATEGY,
                "trigger": {"type": "input_tokens", "value": trigger_tokens},
            }
        ]
    }


def _web_search_tool(max_uses: int) -> dict:
    return {
        "type": "web_search_20260209",
        "name": "web_search",
        "max_uses": max_uses,
    }


# Split on terminal punctuation that looks like a real sentence break:
#   - not preceded by a capital letter (skips "U.S.", "N.Y.", "Ph.D.")
#   - followed by whitespace + a capital letter (skips "U.S. president",
#     "i.e. something", ellipses mid-thought)
# Still false-splits rare cases like "Mr. Smith", which the
# inter-sentence pause smooths over.
_SENTENCE_END_RE = re.compile(r"(?<![A-Z])[.!?](?=\s+[A-Z])")


def _serialize_block(block: Any) -> dict[str, Any]:
    """Serialize an assistant content block for replay in later turns.

    The streaming SDK attaches helper fields (e.g. ``parsed_output``) to
    text blocks that the Messages API rejects on input, so we emit only
    the canonical fields per block type.
    """
    if block.type == "text":
        return {"type": "text", "text": block.text}
    if block.type == "tool_use":
        return {
            "type": "tool_use",
            "id": block.id,
            "name": block.name,
            "input": block.input,
        }
    if block.type == "server_tool_use":
        return {
            "type": "server_tool_use",
            "id": block.id,
            "name": block.name,
            "input": block.input,
        }
    if block.type == "web_search_tool_result":
        # `content` can be either a list of web_search_result blocks
        # (success) or a single error dict (e.g. max_uses_exceeded).
        # Preserve whichever the server sent — `encrypted_content` on
        # each result is required for citation continuity in later
        # turns.
        return {
            "type": "web_search_tool_result",
            "tool_use_id": block.tool_use_id,
            "content": _serialize_web_search_content(block.content),
        }
    if block.type == "compaction":
        return {"type": "compaction", "content": block.content}
    return block.model_dump()


def _serialize_web_search_content(content: Any) -> Any:
    """Strip helper fields from web_search_tool_result.content so the
    Messages API accepts it as input on the next turn."""
    if isinstance(content, list):
        out: list[dict[str, Any]] = []
        for item in content:
            item_type = getattr(item, "type", None)
            if item_type == "web_search_result":
                out.append(
                    {
                        "type": "web_search_result",
                        "url": item.url,
                        "title": item.title,
                        "encrypted_content": item.encrypted_content,
                        "page_age": getattr(item, "page_age", None),
                    }
                )
            else:
                out.append(item.model_dump() if hasattr(item, "model_dump") else item)
        return out
    # Error shape: {"type": "web_search_tool_result_error", "error_code": ...}
    if hasattr(content, "model_dump"):
        return content.model_dump()
    return content


def _pop_sentences(buffer: str) -> tuple[list[str], str]:
    """Return (complete_sentences, remaining_buffer) — splits on
    terminal punctuation and keeps any trailing partial sentence."""
    sentences: list[str] = []
    last_end = 0
    for m in _SENTENCE_END_RE.finditer(buffer):
        end = m.end()
        piece = buffer[last_end:end].strip()
        if piece:
            sentences.append(piece)
        last_end = end
    return sentences, buffer[last_end:]


_CACHE_CONTROL: dict[str, str] = {"type": "ephemeral"}


def _location_block(latitude: float, longitude: float) -> dict[str, Any]:
    """A small system block carrying the user's home coordinates.

    Stable per host, so it lives inside the cached prefix (no
    `cache_control` of its own — the breakpoint on the profile-prompt
    block covers it). Sonnet reverse-geocodes the coords from training
    data when needed, so a single (lat, lon) suffices for any
    location-aware question.
    """
    return {
        "type": "text",
        "text": (
            f"The user's home location is approximately {latitude}, "
            f"{longitude} (decimal degrees). Use this when answering "
            "questions that depend on where the user is — local "
            "weather, daylight, regional references, and similar."
        ),
    }


def _today_block() -> dict[str, Any]:
    """A small, uncached system block carrying the user's local date and time.

    Sonnet uses this to interpret relative time references like "yesterday"
    in `list_sessions` calls and to answer "what time is it?" directly.
    Recomputed per turn so the clock stays fresh and long-running sessions
    that span midnight don't see a stale date.
    """
    now = datetime.now().astimezone()
    tz = now.tzname() or "local time"
    return {
        "type": "text",
        "text": (
            f"Today is {now.strftime('%Y-%m-%d (%A)')} and the current "
            f"local time is {now.strftime('%-I:%M %p')} {tz}. "
            "Use this when interpreting relative time references "
            "or when asked about the current date or time."
        ),
    }


def _system_blocks(
    prompt: str,
    latitude: float | None = None,
    longitude: float | None = None,
) -> list[dict[str, Any]]:
    """Wrap the profile's system prompt with a cache breakpoint and append
    a small dynamic block carrying today's local date.

    When `latitude` and `longitude` are both set, a stable location
    block is prepended so Sonnet can answer location-aware questions
    (sunset, regional context, etc.) without the user mentioning where
    they are. That block sits inside the cached prefix — coords are
    stable per host, so they ride the same cache as the profile prompt.

    The profile prompt is stable per session, so a cache_control on it
    caches the entire `tools + profile prompt` prefix. The trailing
    today block is recomputed per turn (see ``_today_block``); it sits
    after the cache breakpoint so cache hits aren't invalidated by the
    daily date change.
    """
    blocks: list[dict[str, Any]] = []
    if latitude is not None and longitude is not None:
        blocks.append(_location_block(latitude, longitude))
    blocks.append({"type": "text", "text": prompt, "cache_control": _CACHE_CONTROL})
    blocks.append(_today_block())
    return blocks


def _with_cache_breakpoint(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return a shallow copy of `messages` with a cache breakpoint on the
    last block of the last message.

    On turn N+1 the prior turn's tail becomes the longest cached prefix,
    so we move the breakpoint forward each call. The stored messages stay
    in their canonical (str | list[block]) shape — annotation lives only
    on the send-time view, so SQLite persistence stays clean.
    """
    if not messages:
        return messages
    out = list(messages)
    last = dict(out[-1])
    content = last["content"]
    if isinstance(content, str):
        new_content = [
            {"type": "text", "text": content, "cache_control": _CACHE_CONTROL}
        ]
    else:
        # list[block] — copy the list and re-emit the final block with cache_control.
        new_content = list(content)
        tail = dict(new_content[-1])
        tail["cache_control"] = _CACHE_CONTROL
        new_content[-1] = tail
    last["content"] = new_content
    out[-1] = last
    return out


class ClaudeClient:
    def __init__(
        self,
        *,
        api_key: str,
        system_prompt: str,
        dispatcher: ToolDispatcher,
        store: SessionStore | None = None,
        session_id: str | None = None,
        create_session_fn: Callable[[], Awaitable[str]] | None = None,
        compaction_trigger_tokens: int = DEFAULT_COMPACTION_TRIGGER_TOKENS,
        web_search_enabled: bool = DEFAULT_WEB_SEARCH_ENABLED,
        web_search_max_uses: int = DEFAULT_WEB_SEARCH_MAX_USES,
        latitude: float | None = None,
        longitude: float | None = None,
    ):
        self._client = anthropic.AsyncAnthropic(api_key=api_key)
        self._profile_prompt = system_prompt
        self._latitude = latitude
        self._longitude = longitude
        self._dispatcher = dispatcher
        self._tools = dispatcher.get_all_definitions()
        if web_search_enabled:
            self._tools = self._tools + [_web_search_tool(web_search_max_uses)]
        self._messages: list[dict[str, Any]] = []
        self._store = store
        self._session_id = session_id
        self._create_session_fn = create_session_fn
        self._session_create_lock = asyncio.Lock()
        self._context_management = _context_management(compaction_trigger_tokens)

    @property
    def session_id(self) -> str | None:
        return self._session_id

    def set_system_prompt(self, prompt: str) -> None:
        self._profile_prompt = prompt

    def load_history(self, messages: list[dict[str, Any]]) -> None:
        self._messages = list(messages)

    def reset_session(self) -> None:
        """Drop in-memory history and clear the bound SQLite session_id.

        Called by `apply_post_turn_session_change` after `end_session` /
        `new_session` so
        subsequent turns persist to a fresh row (lazily created on first
        persist) and don't carry the prior conversation into a new wake
        cycle."""
        self._messages = []
        self._session_id = None

    def rebind_session(self, session_id: str) -> None:
        """Rebind to a different SQLite session_id without touching history.

        Called by `apply_post_turn_session_change` after `load_history`
        swaps the message array so subsequent turns persist to the loaded
        session's row."""
        self._session_id = session_id

    async def stream_turn(self, user_text: str) -> AsyncIterator[str]:
        """Yield sentence chunks as Claude generates them, running the
        tool-use loop across rounds. The caller drives TTS per chunk."""
        self._messages.append({"role": "user", "content": user_text})
        await self._persist("user", user_text)

        # When a round returns stop_reason=pause_turn (long-running
        # server tool like web_search), we must replay the partial
        # assistant content back to the API to resume. But the API
        # requires user/assistant alternation across rounds, so we
        # can't commit the partial as its own message and then commit
        # the continuation as a second assistant message — the next
        # user turn would produce [..., assistant, assistant, user]
        # and 400. Carry the paused content here, send it as a
        # transient assistant message on the next round, and merge it
        # with the continuation's blocks into a single committed turn
        # once the server finishes.
        paused_blocks: list[dict[str, Any]] = []

        for round_idx in range(MAX_TOOL_ROUNDS):
            api_start = time.perf_counter()
            ttft_ms: int | None = None
            buffer = ""
            # Track text streamed in this round so a barge-in cancel can
            # commit a partial assistant message and keep alternating
            # user/assistant history valid. Reset per round — completed
            # rounds commit full assistant_blocks via the normal path.
            streamed_text = ""

            request_messages = self._messages
            if paused_blocks:
                request_messages = self._messages + [
                    {"role": "assistant", "content": paused_blocks}
                ]

            try:
                async with self._client.beta.messages.stream(
                    model=MODEL,
                    max_tokens=MAX_TOKENS,
                    system=_system_blocks(
                        self._profile_prompt, self._latitude, self._longitude
                    ),
                    tools=self._tools,
                    messages=_with_cache_breakpoint(request_messages),
                    betas=[COMPACTION_BETA],
                    context_management=self._context_management,
                ) as stream:
                    async for delta in stream.text_stream:
                        if ttft_ms is None:
                            ttft_ms = int((time.perf_counter() - api_start) * 1000)
                            logger.debug(
                                "[timing] claude round=%d ttft=%dms",
                                round_idx,
                                ttft_ms,
                            )
                        buffer += delta
                        streamed_text += delta
                        sentences, buffer = _pop_sentences(buffer)
                        for s in sentences:
                            yield s
                    final = await stream.get_final_message()
            except asyncio.CancelledError, GeneratorExit, Exception:
                await self._commit_partial_assistant(streamed_text, paused_blocks)
                raise

            new_blocks = [_serialize_block(b) for b in final.content]
            combined_blocks = paused_blocks + new_blocks

            if final.stop_reason == "pause_turn":
                # Server-side tool still working. Carry the partial
                # content forward; defer the commit until the resume
                # completes so history stays user/assistant-alternating.
                paused_blocks = combined_blocks
                tail = buffer.strip()
                if tail:
                    yield tail
                self._log_round_usage(round_idx, api_start, final)
                continue

            await self._commit_full_assistant(combined_blocks)
            paused_blocks = []

            tail = buffer.strip()
            if tail:
                yield tail

            self._log_round_usage(round_idx, api_start, final)

            if final.stop_reason != "tool_use":
                return

            await self._dispatch_tool_calls(final)

        # Exhausted the round budget. If we exited mid-pause (every
        # round returned pause_turn) the paused content was never
        # committed — flush it now so the next user turn doesn't stack
        # on an orphan user message and 400 the API.
        if paused_blocks:
            await self._commit_full_assistant(paused_blocks)
        logger.warning("Exceeded MAX_TOOL_ROUNDS without a text response")

    async def _commit_partial_assistant(
        self,
        streamed_text: str,
        paused_blocks: list[dict[str, Any]] | None = None,
    ) -> None:
        # Anything that terminates the stream early — barge-in
        # (CancelledError), consumer-driven GeneratorExit, or a
        # network/API error (Exception) — leaves the user message
        # already appended at the top of stream_turn without a paired
        # assistant message. Commit a partial assistant turn so the
        # next user turn doesn't produce two consecutive user messages
        # and trip a 400 from the API. Empty stream gets a "…"
        # placeholder rather than an empty text block (which the API
        # rejects). If we cancelled mid-resume after a pause_turn,
        # prepend the carried blocks so the partial covers the whole
        # paused turn.
        text = streamed_text.strip() or "…"
        partial: list[dict[str, Any]] = list(paused_blocks or [])
        partial.append({"type": "text", "text": text})
        self._messages.append({"role": "assistant", "content": partial})
        # Best-effort persistence: during process shutdown asyncio
        # cleans up pending async generators after the SessionStore
        # has already been closed, so the persist would crash on a
        # closed DB. The in-memory commit above is what matters for
        # next-turn correctness; the on-disk record is nice-to-have.
        try:
            await self._persist("assistant", partial)
        except Exception:
            logger.debug(
                "partial-turn persist skipped (store likely closed)",
                exc_info=True,
            )

    async def _commit_full_assistant(
        self, assistant_blocks: list[dict[str, Any]]
    ) -> None:
        # Commit the full assistant turn to history *before* the caller
        # yields the trailing partial sentence. If the consumer cancels
        # while suspended at `yield tail`, GeneratorExit fires outside
        # stream_turn's try/except — without this ordering, the next
        # user turn would stack on an orphaned user message and 400
        # from the API. SQLite is the verbatim source of truth — strip
        # the server's compaction summary before persisting so on-disk
        # transcripts never carry derived state. The block stays
        # in-memory so the next turn's `messages=` payload includes it
        # and the server doesn't re-summarize the prefix.
        self._messages.append({"role": "assistant", "content": assistant_blocks})
        persisted_blocks = [
            b for b in assistant_blocks if b.get("type") != "compaction"
        ]
        await self._persist("assistant", persisted_blocks)

    def _log_round_usage(self, round_idx: int, api_start: float, final: Any) -> None:
        usage = getattr(final, "usage", None)
        logger.debug(
            "[timing] claude round=%d api_total=%dms in_tok=%s out_tok=%s "
            "cache_create=%s cache_read=%s stop=%s",
            round_idx,
            int((time.perf_counter() - api_start) * 1000),
            getattr(usage, "input_tokens", None),
            getattr(usage, "output_tokens", None),
            getattr(usage, "cache_creation_input_tokens", None),
            getattr(usage, "cache_read_input_tokens", None),
            final.stop_reason,
        )
        for it in getattr(usage, "iterations", None) or []:
            if getattr(it, "type", None) == "compaction":
                logger.info(
                    "[compaction] round=%d in_tok=%s out_tok=%s",
                    round_idx,
                    getattr(it, "input_tokens", None),
                    getattr(it, "output_tokens", None),
                )

    async def _dispatch_tool_calls(self, final: Any) -> None:
        tool_results = []
        for block in final.content:
            if block.type != "tool_use":
                continue
            result = await self._dispatcher.dispatch(block.name, block.input or {})
            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result,
                }
            )
        self._messages.append({"role": "user", "content": tool_results})
        await self._persist("user", tool_results)

    async def _persist(self, role: str, content: str | list[dict[str, Any]]) -> None:
        if self._store is None:
            return
        if self._session_id is None:
            if self._create_session_fn is None:
                return
            async with self._session_create_lock:
                if self._session_id is None:
                    try:
                        sid = await self._create_session_fn()
                    except Exception:
                        logger.error(
                            "create_session_fn raised; skipping persist",
                            exc_info=True,
                        )
                        return
                    if not sid:
                        logger.error(
                            "create_session_fn returned empty id; skipping persist"
                        )
                        return
                    self._session_id = sid
        await self._store.persist_turn(self._session_id, role, content)
