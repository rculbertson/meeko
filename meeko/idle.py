"""Idle and post-wake silence windows.

Owns the timers that close a session when the user stops talking. Two
windows, one task slot:

* **Post-turn** — armed after each turn that left the session in
  LISTENING. Mode-driven (see ``run_idle_window``): ``query`` closes
  silently, ``conversation`` speaks a check-in first.
* **Post-wake** — armed when the wake word fires, covering the gap
  before the user's *first* turn. Mode-independent and always silent:
  "Hey Meeko" with no follow-up returns to IDLE and re-arms the wake
  word.

Only one can be running at a time, so they share a single task slot and
every teardown path (user activity, barge-in, the next turn dequeuing,
shutdown) is a plain ``cancel()``.

Neither window ends the session itself. On expiry they set the
``SessionManager`` end flag and post ``IDLE_TIMEOUT_SENTINEL`` to the
turn queue; the orchestrator's turn worker picks it up and runs the
normal post-turn session handling, minus the Claude/TTS round-trip.
"""

import asyncio
import logging
from collections.abc import Awaitable, Callable

from meeko.config import Profile
from meeko.tools.profile import ProfileManager
from meeko.tools.session import SessionManager

logger = logging.getLogger("meeko")

# Sentinel posted to the turn queue by the idle-timeout monitor to wake
# drive_turns and run post-turn session handling (which finalizes the
# session) without spending a Claude/TTS round-trip first.
IDLE_TIMEOUT_SENTINEL: object = object()

CONVERSATION_PROMPT_TEXT = (
    "Would you like to continue, or should we end the session now?"
)
CONVERSATION_CLOSE_TEXT = "Okay, ending the session now."


async def run_idle_window(
    profile: Profile,
    on_timeout: Callable[[], None],
    speak: Callable[[str], Awaitable[None]] | None = None,
) -> None:
    """Wait for the active mode's idle window, then call on_timeout.

    Cancellation at any await bails cleanly without firing. Called after
    each turn that left the state in LISTENING.

    Query mode: silent close after `idle_timeout_seconds`.
    Conversation mode: after `conversation_idle_seconds`, speak a prompt;
    if no response within `conversation_close_seconds` after the prompt
    finishes, speak a closing line and end the session.

    A non-positive timeout (`idle_timeout_seconds` for query,
    `conversation_idle_seconds` for conversation) disables the monitor —
    used as a test escape hatch and an operator override.
    """
    if profile.mode == "query":
        if profile.idle_timeout_seconds <= 0:
            return
        await asyncio.sleep(profile.idle_timeout_seconds)
        on_timeout()
        return

    elif profile.mode == "conversation":
        if profile.conversation_idle_seconds <= 0:
            return
        if speak is None:
            raise ValueError("conversation mode requires a speak callback")
        await asyncio.sleep(profile.conversation_idle_seconds)
        await speak(CONVERSATION_PROMPT_TEXT)
        # Full close window after the prompt finishes — Meeko's own
        # talking does not eat into the user's response time.
        await asyncio.sleep(profile.conversation_close_seconds)
        await speak(CONVERSATION_CLOSE_TEXT)
        on_timeout()
        return


class IdleController:
    """Arms, cancels and fires the two silence windows.

    ``is_listening`` is a callable rather than the orchestrator's
    ``StateManager`` so this module doesn't import ``meeko.main`` (which
    imports this one). Same shape as ``STTSupervisor``'s ``is_speaking``.
    """

    def __init__(
        self,
        *,
        is_listening: Callable[[], bool],
        profile_manager: ProfileManager,
        session_manager: SessionManager,
        turn_queue: asyncio.Queue,
        speak: Callable[[str], Awaitable[None]],
    ) -> None:
        self._is_listening = is_listening
        self._profiles = profile_manager
        self._sessions = session_manager
        self._turn_queue = turn_queue
        self._speak = speak
        self._task: asyncio.Task | None = None

    def cancel(self) -> None:
        """Tear down whichever window is running, if any."""
        if self._task is not None and not self._task.done():
            self._task.cancel()
        self._task = None

    async def aclose(self) -> None:
        """Cancel and await the running window. For shutdown only.

        ``cancel()`` drops the reference immediately, which is what the
        hot paths want but leaves nothing to await; at shutdown we want
        the task actually finished before the event loop goes away, or
        asyncio logs "Task was destroyed but it is pending".
        """
        task = self._task
        self._task = None
        if task is None:
            return
        if not task.done():
            task.cancel()
        try:
            await task
        except asyncio.CancelledError, Exception:
            pass

    def start_post_turn(self) -> None:
        """Arm the post-turn idle window for the active profile's mode."""
        self.cancel()
        self._task = asyncio.create_task(
            run_idle_window(
                self._profiles.active_profile,
                self._on_post_turn_timeout,
                speak=self._speak,
            )
        )

    def start_post_wake(self) -> None:
        """Arm the silence window running from the wake word to the
        user's first turn. Non-positive timeout disables it."""
        self.cancel()
        timeout = self._profiles.active_profile.post_wake_timeout_seconds
        if timeout <= 0:
            return

        async def _monitor() -> None:
            await asyncio.sleep(timeout)
            self._on_post_wake_timeout()

        self._task = asyncio.create_task(_monitor())

    def _should_fire(self) -> bool:
        """Race guard: a turn may have arrived in the gap between
        asyncio.sleep waking and the timeout callback running. Skip if
        the session is no longer LISTENING (turn already in flight) or
        if a user transcript is already queued ahead of us — letting the
        sentinel land behind real text would end a session right after a
        successful turn."""
        return self._is_listening() and self._turn_queue.empty()

    def _on_post_turn_timeout(self) -> None:
        if not self._should_fire():
            return
        active = self._profiles.active_profile
        if active.mode == "query":
            logger.info(
                "Idle timeout (%.1fs, mode=query); ending session",
                active.idle_timeout_seconds,
            )
        elif active.mode == "conversation":
            logger.info(
                "Idle timeout (%.1fs after prompt, mode=conversation); ending session",
                active.conversation_close_seconds,
            )
        else:
            logger.warning(
                "Idle timeout for unknown mode: %s; ending session", active.mode
            )
        self._end()

    def _on_post_wake_timeout(self) -> None:
        if not self._should_fire():
            return
        logger.info(
            "Post-wake timeout (%.1fs, no speech); ending session",
            self._profiles.active_profile.post_wake_timeout_seconds,
        )
        self._end()

    def _end(self) -> None:
        """Flag the session for teardown and wake the turn worker."""
        self._sessions.request_end()
        self._turn_queue.put_nowait(IDLE_TIMEOUT_SENTINEL)
