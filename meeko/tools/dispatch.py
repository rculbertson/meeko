"""Generic tool dispatcher.

Collects Anthropic-style tool definitions from registered modules and routes
tool_use invocations to the correct handler.
"""

import logging
from collections.abc import Awaitable, Callable
from typing import Any

logger = logging.getLogger("meeko")

# A tool definition is a plain dict in Anthropic's tool-use format:
#   {"name": str, "description": str, "input_schema": {...}}
ToolDefinition = dict[str, Any]

# A handler is an async callable: (fn_name, args_dict) -> result_string.
ToolHandler = Callable[[str, dict], Awaitable[str]]


class ToolDispatcher:
    """Routes tool_use invocations to registered handlers."""

    def __init__(self):
        self._handlers: dict[str, ToolHandler] = {}
        self._definitions: list[ToolDefinition] = []

    def register(
        self,
        definitions: list[ToolDefinition],
        handler: ToolHandler,
    ):
        """Register a tool module's definitions and handler."""
        self._definitions.extend(definitions)
        for defn in definitions:
            self._handlers[defn["name"]] = handler

    def get_all_definitions(self) -> list[ToolDefinition]:
        return list(self._definitions)

    async def dispatch(self, name: str, args: dict) -> str:
        """Look up the handler for `name` and invoke it with `args`."""
        logger.info("Function call: %s(%s)", name, args)
        handler = self._handlers.get(name)
        if handler is None:
            logger.warning("Unknown function call: %s", name)
            return f"Unknown function: {name}"
        return await handler(name, args)
