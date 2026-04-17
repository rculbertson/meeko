"""Thin wrapper around the Anthropic async client.

Maintains the in-memory message history for a single Meeko session and
runs the tool-use loop. Deliberately does not add prompt caching,
auto compaction, or persistence — those are separate steps.
"""

import logging
from typing import Any

import anthropic

from meeko.tools.dispatch import ToolDispatcher

logger = logging.getLogger("meeko")

MODEL = "claude-sonnet-4-6"
MAX_TOKENS = 1024
MAX_TOOL_ROUNDS = 5


class ClaudeClient:
    def __init__(
        self,
        *,
        api_key: str,
        system_prompt: str,
        dispatcher: ToolDispatcher,
    ):
        self._client = anthropic.AsyncAnthropic(api_key=api_key)
        self._system = system_prompt
        self._dispatcher = dispatcher
        self._tools = dispatcher.get_all_definitions()
        self._messages: list[dict[str, Any]] = []

    def set_system_prompt(self, prompt: str) -> None:
        self._system = prompt

    async def turn(self, user_text: str) -> str:
        """Run one user turn: append the message, loop through any tool
        calls, return the final assistant text to be spoken."""
        self._messages.append({"role": "user", "content": user_text})
        return await self._run_until_text()

    async def _run_until_text(self) -> str:
        for _ in range(MAX_TOOL_ROUNDS):
            response = await self._client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                system=self._system,
                tools=self._tools,
                messages=self._messages,
            )

            # Persist the assistant turn verbatim so future turns see tool_use
            # blocks paired with matching tool_result blocks.
            assistant_blocks = [block.model_dump() for block in response.content]
            self._messages.append({"role": "assistant", "content": assistant_blocks})

            if response.stop_reason != "tool_use":
                return _extract_text(response.content)

            tool_results = []
            for block in response.content:
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

        logger.warning("Exceeded MAX_TOOL_ROUNDS without a text response")
        return ""


def _extract_text(blocks) -> str:
    parts = [b.text for b in blocks if getattr(b, "type", None) == "text"]
    return " ".join(p.strip() for p in parts if p).strip()
