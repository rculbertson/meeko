"""Thin wrapper around the Anthropic async client.

Maintains the in-memory message history for a single Meeko session and
runs the tool-use loop against Claude's streaming API, yielding
sentence-sized text chunks as they are generated so the caller can
begin TTS before the full reply is ready.
"""

import logging
import re
import time
from collections.abc import AsyncIterator
from typing import Any

import anthropic

from meeko.sessions import SessionStore
from meeko.tools.dispatch import ToolDispatcher

logger = logging.getLogger("meeko")

MODEL = "claude-sonnet-4-6"
MAX_TOKENS = 8192
MAX_TOOL_ROUNDS = 5

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
    return block.model_dump()


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


class ClaudeClient:
    def __init__(
        self,
        *,
        api_key: str,
        system_prompt: str,
        dispatcher: ToolDispatcher,
        store: SessionStore | None = None,
        session_id: str | None = None,
    ):
        self._client = anthropic.AsyncAnthropic(api_key=api_key)
        self._system = system_prompt
        self._dispatcher = dispatcher
        self._tools = dispatcher.get_all_definitions()
        self._messages: list[dict[str, Any]] = []
        self._store = store
        self._session_id = session_id

    def set_system_prompt(self, prompt: str) -> None:
        self._system = prompt

    def load_history(self, messages: list[dict[str, Any]]) -> None:
        self._messages = list(messages)

    async def stream_turn(self, user_text: str) -> AsyncIterator[str]:
        """Yield sentence chunks as Claude generates them, running the
        tool-use loop across rounds. The caller drives TTS per chunk."""
        self._messages.append({"role": "user", "content": user_text})
        await self._persist("user", user_text)

        for round_idx in range(MAX_TOOL_ROUNDS):
            api_start = time.perf_counter()
            ttft_ms: int | None = None
            buffer = ""

            async with self._client.messages.stream(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                system=self._system,
                tools=self._tools,
                messages=self._messages,
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
                    sentences, buffer = _pop_sentences(buffer)
                    for s in sentences:
                        yield s
                final = await stream.get_final_message()

            tail = buffer.strip()
            if tail:
                yield tail

            usage = getattr(final, "usage", None)
            logger.debug(
                "[timing] claude round=%d api_total=%dms in_tok=%s out_tok=%s stop=%s",
                round_idx,
                int((time.perf_counter() - api_start) * 1000),
                getattr(usage, "input_tokens", None),
                getattr(usage, "output_tokens", None),
                final.stop_reason,
            )

            assistant_blocks = [_serialize_block(block) for block in final.content]
            self._messages.append({"role": "assistant", "content": assistant_blocks})
            await self._persist("assistant", assistant_blocks)

            if final.stop_reason != "tool_use":
                return

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

        logger.warning("Exceeded MAX_TOOL_ROUNDS without a text response")

    async def _persist(self, role: str, content: str | list[dict[str, Any]]) -> None:
        if self._store is None or self._session_id is None:
            return
        await self._store.persist_turn(self._session_id, role, content)
