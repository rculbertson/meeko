"""Session-management tools exposed to Sonnet.

Provides two tools today:

- ``end_session``: user signals they want to stop ("stop", "goodnight",
  "that's enough for today", etc.). After SPEAKING the orchestrator
  finalizes the session and returns to IDLE, re-arming the wake word.
- ``new_session``: user wants to start a fresh thread without stopping
  Meeko ("let's start fresh", "different topic"). After SPEAKING the
  orchestrator finalizes the current session, allocates a new one, and
  stays in LISTENING.

Both handlers only set a flag on the shared ``SessionManager``; the
orchestrator checks the flags after the SPEAKING phase completes so
Sonnet's verbal acknowledgement plays in full before any reset.

Future tools (`list_sessions`, `load_session`) will live here too.
`load_session` is the one with non-trivial orchestrator semantics — see
`private/meeko-design.md` §4.5.
"""

import logging

from meeko.tools.dispatch import ToolDefinition

logger = logging.getLogger("meeko")


class SessionManager:
    """Shared state between the session-tool handlers and the
    orchestrator's post-SPEAKING hook.

    Handlers only set flags — they do NOT tear down state directly.
    The orchestrator drains Sonnet's acknowledgement via TTS first and
    then, after SPEAKING ends, performs the actual session transition."""

    def __init__(self) -> None:
        self._should_end = False
        self._should_start_new = False

    def request_end(self) -> None:
        self._should_end = True

    def should_end(self) -> bool:
        return self._should_end

    def request_new(self) -> None:
        self._should_start_new = True

    def should_start_new(self) -> bool:
        return self._should_start_new

    def clear(self) -> None:
        self._should_end = False
        self._should_start_new = False


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
        },
        {
            "name": "new_session",
            "description": (
                "Finalize the current conversation and start a fresh one, "
                "without stopping Meeko. Call this when the user clearly "
                "wants to switch to a new thread — e.g. 'let's start "
                "fresh', 'different topic', 'new conversation'. Before "
                "calling this tool, confirm briefly that the user wants "
                "to discard the current thread (e.g. 'Got it, starting "
                "fresh.'). Do not call this tool when the user is just "
                "changing subjects within the same conversation "
                "('that's enough about X, let's talk about Y') — only "
                "when they explicitly want to wipe the slate."
            ),
            "input_schema": {"type": "object", "properties": {}},
        },
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
    if fn_name == "new_session":
        manager.request_new()
        logger.info("new_session tool called; will rotate session after SPEAKING")
        return "Starting a fresh session. Meeko will swap the context after this turn."
    return f"Unknown session function: {fn_name}"
