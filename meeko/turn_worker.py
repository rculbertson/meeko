"""The turn worker and barge-in.

`TurnWorker.run()` is the long-lived consumer on the far side of the turn
queue: it takes one user transcript at a time, runs the Claude+TTS turn,
then applies whatever session change the turn requested. It lives at
`run()` scope rather than per STT session, so an STT reconnect mid-reply
doesn't cut TTS off mid-sentence or lose the in-flight turn.

`request_barge_in()` is how `SttEventRouter` interrupts it. The turn runs
as its own sub-task so a barge-in can cancel just that turn and leave the
worker running; a shutdown cancel of the worker itself must still
propagate. Both arrive in the worker as the same `CancelledError`, and the
only thing that tells them apart is the flag `request_barge_in()` sets
before cancelling. The two share that flag and the sub-task handle, which
is why they live on one object.
"""

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable, Coroutine
from typing import Any

from meeko.idle import IDLE_TIMEOUT_SENTINEL, IdleController
from meeko.leds import LedController
from meeko.state import State, StateManager

logger = logging.getLogger("meeko")


class TurnWorker:
    """Drives queued user turns and cancels the in-flight one on barge-in.

    ``start_turn(text)`` must return the turn's coroutine without awaiting
    it (in production, ``speaker.speak_stream(claude.stream_turn(text))``);
    it becomes the cancellable sub-task. ``apply_session_change()`` runs
    the post-turn session handling and returns the next State.
    """

    def __init__(
        self,
        *,
        turn_queue: asyncio.Queue,
        state_manager: StateManager,
        idle: IdleController,
        leds: LedController,
        start_turn: Callable[[str], Coroutine[Any, Any, None]],
        apply_session_change: Callable[[], Awaitable[State]],
        stop_event: asyncio.Event,
    ) -> None:
        self._turn_queue = turn_queue
        self._state = state_manager
        self._idle = idle
        self._leds = leds
        self._start_turn = start_turn
        self._apply_session_change = apply_session_change
        self._stop_event = stop_event
        # Set in run() while a Claude+TTS turn is running so
        # request_barge_in() can cancel just that turn without tearing down
        # the long-lived worker.
        self._current_speak_task: asyncio.Task | None = None
        # Set by request_barge_in() so run() can tell the difference
        # between a real barge-in (continue worker) and a shutdown cancel
        # (re-raise). Checking stop_event isn't reliable because asyncio
        # shutdown cancels the worker task directly, before meeko.main
        # run()'s finally has a chance to set stop_event.
        self._barge_in_requested = False

    def request_barge_in(self) -> None:
        """Cancel the in-flight speak task, if any. Called from
        SttEventRouter when StartOfTurn fires during SPEAKING (the
        assistant is talking) or PROCESSING (the assistant's reply is
        still being generated; user has changed their mind)."""
        # Flip state synchronously so any EndOfTurn arriving before the
        # cancel propagates through run() isn't dropped as echo
        # by the router. This must happen even when there's no
        # current speak task (e.g. a timer's expiry announcement is playing
        # via speaker.speak() — that path enters SPEAKING but is not
        # cancellable from here): the announcement keeps playing, but the
        # user's interruption is at least captured into turn_queue
        # instead of silently dropped. run() and speak_stream's
        # exit_speaking will re-assert LISTENING when they unwind; the
        # brief PROCESSING window in between is harmless (PROCESSING-
        # state EndOfTurns are queued normally).
        self._idle.cancel()
        self._state.set(State.LISTENING)
        if self._current_speak_task is not None and not self._current_speak_task.done():
            logger.info("Barge-in: cancelling in-flight reply")
            self._barge_in_requested = True
            self._current_speak_task.cancel()

    async def run(self) -> None:
        """Drive Claude + TTS for queued user turns, one at a time."""
        while not self._stop_event.is_set():
            item = await self._turn_queue.get()
            self._idle.cancel()
            if item is IDLE_TIMEOUT_SENTINEL:
                # Idle window expired without user activity; IdleController
                # already set session_manager.request_end(). Run the
                # post-turn block to finalize and (with wake word) return
                # to IDLE. No Claude/TTS round-trip on this path, so we
                # do not start a fresh idle window after.
                new_state = await self._apply_session_change()
                self._state.set(new_state)
                continue
            text = item
            assert isinstance(text, str)
            logger.info("[user] %s", text)
            self._state.set(State.PROCESSING)
            t_turn = time.perf_counter()
            # Run the turn as a sub-task so request_barge_in() can
            # cancel just this turn without tearing down the worker.
            self._current_speak_task = asyncio.create_task(self._start_turn(text))
            try:
                await self._current_speak_task
            except asyncio.CancelledError:
                # speak_stream's finally already flushed TTS subtasks
                # and exit_speaking() restored state. Distinguish a real
                # barge-in (continue worker) from a shutdown cancel
                # (re-raise) by the explicit flag — stop_event is racy
                # because asyncio's shutdown cancels the worker task
                # before meeko.main run()'s finally has set it.
                self._state.set(State.LISTENING)
                self._current_speak_task = None
                if not self._barge_in_requested:
                    raise
                self._barge_in_requested = False
                logger.info("Barge-in: turn cancelled, returning to LISTENING")
                continue
            except Exception:
                logger.exception("Claude turn failed")
                # Set state first so the worker applies LISTENING before
                # the error animation, and the post-flash restore picks
                # up LISTENING as _current_state. Reversing the order
                # makes _sleep_or_interrupt see the queued state action
                # and abort the breath immediately.
                self._state.set(State.LISTENING)
                self._leds.error()
                self._current_speak_task = None
                continue
            self._current_speak_task = None
            logger.debug(
                "[timing] turn_total_eot_to_speak_done=%dms",
                int((time.perf_counter() - t_turn) * 1000),
            )
            new_state = await self._apply_session_change()
            # Start the post-turn idle window. Only when state is
            # LISTENING — IDLE means the session already ended and the
            # next interaction needs the wake word.
            if new_state == State.LISTENING:
                self._idle.start_post_turn()
            self._state.set(new_state)
