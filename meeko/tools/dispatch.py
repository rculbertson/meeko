"""Generic tool dispatcher for the voice agent.

Collects tool definitions from registered modules and routes incoming
FunctionCallRequest messages to the correct handler.
"""

import json
import logging
from collections.abc import Awaitable, Callable

from deepgram.agent.v1.types import (
    AgentV1SendFunctionCallResponse,
    AgentV1SettingsAgentThinkOneItemFunctionsItem,
)

logger = logging.getLogger("meeko")

# Type alias for a tool handler: async (fn_name, args_dict, connection) -> str
ToolHandler = Callable[[str, dict, object], Awaitable[str]]


def _get(obj, key, default=""):
    """Get a value from a dict or Pydantic model.

    The Deepgram SDK sometimes deserializes messages as the wrong Pydantic
    type (AgentV1PromptUpdated), storing actual fields as extra dict data.
    This helper handles both dict and attribute access.
    """
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


class ToolDispatcher:
    """Routes function calls to registered tool handlers."""

    def __init__(self):
        self._handlers: dict[str, ToolHandler] = {}
        self._definitions: list[AgentV1SettingsAgentThinkOneItemFunctionsItem] = []

    def register(
        self,
        definitions: list[AgentV1SettingsAgentThinkOneItemFunctionsItem],
        handler: ToolHandler,
    ):
        """Register a tool module's definitions and handler."""
        self._definitions.extend(definitions)
        for defn in definitions:
            self._handlers[defn.name] = handler

    def get_all_definitions(
        self,
    ) -> list[AgentV1SettingsAgentThinkOneItemFunctionsItem]:
        return list(self._definitions)

    async def handle_function_call_request(self, message, connection):
        """Parse a FunctionCallRequest and route to the registered handler."""
        functions = _get(message, "functions", [])
        for fn in functions:
            fn_id = _get(fn, "id")
            fn_name = _get(fn, "name")
            raw_args = _get(fn, "arguments", "{}")

            try:
                args = json.loads(raw_args)
            except json.JSONDecodeError, TypeError:
                args = {}

            logger.info("Function call: %s(%s) id=%s", fn_name, args, fn_id)

            handler = self._handlers.get(fn_name)
            if handler:
                result = await handler(fn_name, args, connection)
            else:
                result = f"Unknown function: {fn_name}"
                logger.warning("Unknown function call: %s", fn_name)

            await connection.send_function_call_response(
                AgentV1SendFunctionCallResponse(
                    type="FunctionCallResponse",
                    id=fn_id,
                    name=fn_name,
                    content=result,
                )
            )
