"""Timer tools for the voice assistant.

Provides set_timer, list_timers, cancel_timer, and cancel_all_timers.
Timers run as asyncio tasks and announce expiry via a speak_callback
supplied by meeko/main.py at startup (typically: synthesize TTS + play
through the speaker, respecting the SPEAKING state so the mic stays
muted).
"""

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable

from meeko.tools.dispatch import ToolDefinition

logger = logging.getLogger("meeko")

_PLURAL_UNITS = {"seconds": "second", "minutes": "minute", "hours": "hour"}

SpeakCallback = Callable[[str], Awaitable[None]]


def _singularize_units(display: str) -> str:
    """Convert plural time units to singular, e.g. '10 minutes' -> '10 minute'."""
    for plural, singular in _PLURAL_UNITS.items():
        display = display.replace(plural, singular)
    return display


class TimerManager:
    """Manages concurrent timers as asyncio tasks.

    On expiry, calls ``speak_callback`` with a fixed announcement string.
    The callback is set via ``set_speak_callback`` during wire-up in
    ``meeko/main.py``; handlers run before wire-up (e.g. in unit tests) will no-op
    on expiry unless a callback is supplied.
    """

    def __init__(self, speak_callback: SpeakCallback | None = None):
        # label -> (task, start_time, duration_seconds)
        self._timers: dict[str, tuple[asyncio.Task, float, float]] = {}
        self._counter = 0  # for auto-labeling
        self._speak: SpeakCallback | None = speak_callback

    def set_speak_callback(self, speak_callback: SpeakCallback) -> None:
        self._speak = speak_callback

    async def set_timer(
        self,
        duration_seconds: float,
        duration_display: str,
        label: str | None,
    ) -> str:
        duration_display = _singularize_units(duration_display)
        custom_label = label
        if not label:
            self._counter += 1
            label = f"timer {self._counter}"

        # Cancel existing timer with same label
        if label in self._timers:
            self._timers[label][0].cancel()

        task = asyncio.create_task(
            self._run_timer(label, custom_label, duration_seconds, duration_display)
        )
        self._timers[label] = (task, time.monotonic(), duration_seconds)
        return f"Timer '{label}' set for {duration_display}."

    # NOTE: expiry speaks a fixed announcement directly via TTS and does NOT
    # tell the Claude model that the timer fired. Sonnet has no idea the timer
    # it set ever went off, so it can't refer back to it.
    #
    # POSSIBLE FOLLOW-UP: append a synthetic user-role message to the
    # ClaudeClient's message history (e.g. {"role": "user", "content": "The 10
    # minute pasta timer just finished."}) and trigger a Claude turn, so the
    # assistant can weave the expiry into the ongoing conversation, persist it,
    # and react with context. That path requires the timer to hold a reference
    # to ClaudeClient + the speak path, and to coordinate with whatever turn is
    # in flight (don't interrupt the user; queue if the assistant is speaking).
    # The original blocker (session persistence / prompt caching / compaction
    # not yet built) is gone — all three have since landed — so this is now a
    # design question about turn coordination, not a sequencing one.
    async def _run_timer(
        self,
        label: str,
        custom_label: str | None,
        duration_seconds: float,
        duration_display: str,
    ) -> None:
        try:
            await asyncio.sleep(duration_seconds)
            logger.info("Timer '%s' expired", label)
            if custom_label:
                message = f"The {duration_display} {custom_label} timer is done!"
            else:
                message = f"The {duration_display} timer is done!"
            if self._speak is not None:
                try:
                    await self._speak(message)
                except Exception:
                    # Log it here: an exception escaping this task would only
                    # surface as "Task exception was never retrieved" whenever
                    # the task happened to be garbage-collected.
                    logger.exception("Timer '%s' announcement failed", label)
            else:
                logger.warning(
                    "Timer '%s' expired but no speak_callback is configured", label
                )
        except asyncio.CancelledError:
            logger.info("Timer '%s' cancelled", label)
        finally:
            # Only remove our own entry. Re-setting a label cancels this
            # task and stores its replacement under the same key before
            # this finally runs; popping unconditionally would drop the
            # new timer, leaving it running but invisible to list/cancel.
            entry = self._timers.get(label)
            if entry is not None and entry[0] is asyncio.current_task():
                del self._timers[label]

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


# Singleton instance. meeko/main.py installs a speak_callback on startup.
timer_manager = TimerManager()


def get_tool_definitions() -> list[ToolDefinition]:
    return [
        {
            "name": "set_timer",
            "description": (
                "Set a countdown timer. When the timer expires, "
                "the assistant will announce it aloud, including the "
                "timer name (if given) and duration."
            ),
            "input_schema": {
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
        },
        {
            "name": "list_timers",
            "description": "List all active timers with their remaining time.",
            "input_schema": {"type": "object", "properties": {}},
        },
        {
            "name": "cancel_timer",
            "description": "Cancel an active timer by its label.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "label": {
                        "type": "string",
                        "description": "The label of the timer to cancel",
                    },
                },
                "required": ["label"],
            },
        },
        {
            "name": "cancel_all_timers",
            "description": "Cancel all active timers at once.",
            "input_schema": {"type": "object", "properties": {}},
        },
    ]


async def handle(fn_name: str, args: dict) -> str:
    """Handle a timer-related function call."""
    if fn_name == "set_timer":
        return await timer_manager.set_timer(
            duration_seconds=args.get("duration_seconds", 60),
            duration_display=args.get("duration_display", "1 minute"),
            label=args.get("label"),
        )
    elif fn_name == "list_timers":
        return timer_manager.list_timers()
    elif fn_name == "cancel_timer":
        return timer_manager.cancel_timer(label=args.get("label", ""))
    elif fn_name == "cancel_all_timers":
        return timer_manager.cancel_all_timers()
    return f"Unknown timer function: {fn_name}"
