"""Timer tools for the voice agent.

Provides set_timer, list_timers, and cancel_timer functionality.
Timers run as asyncio tasks and announce expiry via InjectAgentMessage.
"""

import asyncio
import json
import logging
import time

from deepgram.agent.v1.types import (
    AgentV1InjectAgentMessage,
    AgentV1SendFunctionCallResponse,
    AgentV1SettingsAgentThinkOneItemFunctionsItem,
)

logger = logging.getLogger("meeko")

_PLURAL_UNITS = {"seconds": "second", "minutes": "minute", "hours": "hour"}


def _singularize_units(display: str) -> str:
    """Convert plural time units to singular, e.g. '10 minutes' -> '10 minute'."""
    for plural, singular in _PLURAL_UNITS.items():
        display = display.replace(plural, singular)
    return display


class TimerManager:
    """Manages concurrent timers as asyncio tasks."""

    def __init__(self):
        # label -> (task, start_time, duration_seconds)
        self._timers: dict[str, tuple[asyncio.Task, float, float]] = {}
        self._counter = 0  # for auto-labeling

    async def set_timer(
        self,
        duration_seconds: float,
        duration_display: str,
        label: str | None,
        connection,
    ):
        duration_display = _singularize_units(duration_display)
        custom_label = label
        if not label:
            self._counter += 1
            label = f"timer {self._counter}"

        # Cancel existing timer with same label
        if label in self._timers:
            self._timers[label][0].cancel()

        task = asyncio.create_task(
            self._run_timer(
                label, custom_label, duration_seconds, duration_display, connection
            )
        )
        self._timers[label] = (task, time.monotonic(), duration_seconds)
        return f"Timer '{label}' set for {duration_display}."

    async def _run_timer(
        self,
        label: str,
        custom_label: str | None,
        duration_seconds: float,
        duration_display: str,
        connection,
    ):
        try:
            await asyncio.sleep(duration_seconds)
            logger.info("Timer '%s' expired", label)
            if custom_label:
                message = f"The {duration_display} {custom_label} timer is done!"
            else:
                message = f"The {duration_display} timer is done!"
            await connection.send_inject_agent_message(
                AgentV1InjectAgentMessage(
                    type="InjectAgentMessage",
                    message=message,
                )
            )
        except asyncio.CancelledError:
            logger.info("Timer '%s' cancelled", label)
        finally:
            self._timers.pop(label, None)

    def list_timers(self) -> str:
        if not self._timers:
            return "No active timers."
        lines = []
        now = time.monotonic()
        for label, (_, start, duration) in self._timers.items():
            elapsed = now - start
            remaining = max(0, duration - elapsed)
            mins = int(remaining // 60)
            secs = int(remaining % 60)
            lines.append(f"{label}: {mins}m {secs}s remaining")
        return "\n".join(lines)

    def cancel_timer(self, label: str) -> str:
        if label not in self._timers:
            return f"No active timer named '{label}'."
        self._timers[label][0].cancel()
        self._timers.pop(label, None)
        return f"Timer '{label}' cancelled."

    def cancel_all_timers(self) -> str:
        if not self._timers:
            return "No active timers."
        count = len(self._timers)
        for task, _, _ in self._timers.values():
            task.cancel()
        self._timers.clear()
        return f"All {count} timer{'s' if count != 1 else ''} cancelled."


# Singleton instance
timer_manager = TimerManager()


def get_tool_definitions() -> list[AgentV1SettingsAgentThinkOneItemFunctionsItem]:
    return [
        AgentV1SettingsAgentThinkOneItemFunctionsItem(
            name="set_timer",
            description=(
                "Set a countdown timer. When the timer expires, "
                "the assistant will announce it aloud, including the "
                "timer name (if given) and duration."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "duration_seconds": {
                        "type": "number",
                        "description": "Duration of the timer in seconds",
                    },
                    "duration_display": {
                        "type": "string",
                        "description": (
                            "The timer duration using singular time units, "
                            "e.g. '30 second', '10 minute', '1 hour'. "
                            "Used verbatim in the expiry announcement."
                        ),
                    },
                    "label": {
                        "type": "string",
                        "description": (
                            "Optional name for the timer, e.g. pasta, "
                            "oven. Included in the expiry announcement."
                        ),
                    },
                },
                "required": ["duration_seconds", "duration_display"],
            },
        ),
        AgentV1SettingsAgentThinkOneItemFunctionsItem(
            name="list_timers",
            description="List all active timers with their remaining time.",
            parameters={"type": "object", "properties": {}},
        ),
        AgentV1SettingsAgentThinkOneItemFunctionsItem(
            name="cancel_timer",
            description="Cancel an active timer by its label.",
            parameters={
                "type": "object",
                "properties": {
                    "label": {
                        "type": "string",
                        "description": "The label of the timer to cancel",
                    },
                },
                "required": ["label"],
            },
        ),
        AgentV1SettingsAgentThinkOneItemFunctionsItem(
            name="cancel_all_timers",
            description="Cancel all active timers at once.",
            parameters={"type": "object", "properties": {}},
        ),
    ]


def _get(obj, key, default=""):
    """Get a value from a dict or Pydantic model.

    The Deepgram SDK sometimes deserializes messages as the wrong Pydantic
    type (AgentV1PromptUpdated), storing actual fields as extra dict data.
    This helper handles both dict and attribute access.
    """
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


async def handle_function_call_request(message, connection):
    """Handle a FunctionCallRequest from the Voice Agent.

    Iterates the functions array, routes each to the appropriate handler,
    and sends a FunctionCallResponse with the matching id.
    """
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

        if fn_name == "set_timer":
            result = await timer_manager.set_timer(
                duration_seconds=args.get("duration_seconds", 60),
                duration_display=args.get("duration_display", "1 minute"),
                label=args.get("label"),
                connection=connection,
            )
        elif fn_name == "list_timers":
            result = timer_manager.list_timers()
        elif fn_name == "cancel_timer":
            result = timer_manager.cancel_timer(
                label=args.get("label", ""),
            )
        elif fn_name == "cancel_all_timers":
            result = timer_manager.cancel_all_timers()
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
