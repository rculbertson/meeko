"""Session-management tools exposed to Sonnet.

Currently provides a single tool, `end_session`, which Sonnet calls when
the user signals they want to stop the conversation ("stop", "goodnight",
"that's enough for today", etc.). The handler flips a flag on the shared
``SessionManager``; the orchestrator checks the flag after the SPEAKING
phase completes and only then tears the session down — so Sonnet's verbal
acknowledgement plays in full before Meeko returns to IDLE.

Future tools (`new_session`, `list_sessions`, `load_session`) will live
here too. `load_session` is the one with non-trivial orchestrator
semantics — see `private/meeko-design.md` §4.5.
"""

import logging

from meeko.tools.dispatch import ToolDefinition

logger = logging.getLogger("meeko")


class SessionManager:
    """Shared state between the `end_session` tool handler and the
    orchestrator's post-SPEAKING shutdown hook.

    The handler only sets a flag — it does NOT tear down state directly.
    The orchestrator drains Sonnet's acknowledgement via TTS first and
    then, after SPEAKING ends, performs the actual session reset."""

    def __init__(self) -> None:
        self._should_end = False

    def request_end(self) -> None:
        self._should_end = True

    def should_end(self) -> bool:
        return self._should_end

    def clear(self) -> None:
        self._should_end = False


def get_tool_definitions() -> list[ToolDefinition]:
    return [
        {
            "name": "end_session",
            "description": (
                "End the current conversation and return Meeko to idle. "
                "Call this when the user clearly indicates they want to "
                "stop — e.g. 'stop', 'goodnight', 'that's enough for "
                "today', 'let's pick this up later'. Before calling this "
                "tool, respond with a brief verbal acknowledgement (e.g. "
                "'Goodnight!'). Do not call this tool when the user is "
                "ambiguous or immediately walks back the signal "
                "('stop interrupting me', 'that's enough about X, let's "
                "talk about Y')."
            ),
            "input_schema": {"type": "object", "properties": {}},
        }
    ]


async def handle(
    fn_name: str,
    args: dict,
    *,
    manager: SessionManager,
) -> str:
    if fn_name == "end_session":
        manager.request_end()
        logger.info("end_session tool called; will transition after SPEAKING")
        return "Session ended. Meeko will return to idle after this turn."
    return f"Unknown session function: {fn_name}"
