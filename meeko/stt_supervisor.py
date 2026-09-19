"""STT session lifecycle: reconnect, backoff, grace cutoff.

Owns the outer `while not stop_event` loop that enters a Deepgram STT
session, runs a caller-supplied per-session coroutine, and reconnects
on failure with exponential backoff. During a brief outage the mic
stays hot so the user's speech isn't dropped; after
RECONNECT_GRACE_S the mic is stopped and the queue drained until a
clean reconnect.
"""

import asyncio
import logging
from collections.abc import Awaitable, Callable, Coroutine
from typing import Any

from websockets.exceptions import ConnectionClosed

from meeko.audio_io import AudioIO

logger = logging.getLogger("meeko")

# How often to send a Deepgram KeepAlive text frame while we're not
# streaming mic audio. Deepgram documents 3–5s as the recommended cadence.
# This pump only prevents server-side idle close; dead-connection
# detection is handled by the websockets library's auto-ping, tuned in
# meeko/deepgram_stt.py (ping_interval / ping_timeout).
KEEPALIVE_INTERVAL_S = 5

# During an STT outage we keep capturing mic audio so a brief blip
# doesn't drop the user's speech. After this many seconds we give up,
# stop capture, and drain the queue until a clean reconnect.
RECONNECT_GRACE_S = 10

# Delay before each reconnect attempt: 0.5s → 1 → 2 → 4 → 8 → 16 → 30
# (cap), reset on a successful connect. The first failure in a streak
# logs the full traceback; subsequent failures log one line so a
# prolonged outage doesn't flood the logs.
RECONNECT_BACKOFF_S = (0.5, 1, 2, 4, 8, 16, 30)


class SttSessionEndedError(RuntimeError):
    """A session ended without an error while Meeko wasn't stopping.

    Deepgram closing the socket cleanly, or keepalive_pump seeing the
    connection close, ends a session this way. It goes through the same
    backoff as any other failure; returning normally would reconnect at
    once, with no delay and no failure count.
    """


async def keepalive_pump(stt_session, stop_event: asyncio.Event) -> None:
    """Send a Deepgram KeepAlive every few seconds so the session stays
    open when we're not streaming audio.

    Returns (rather than raising) on ConnectionClosed: the supervisor's
    reconnect loop owns that failure, and this pump is one of the
    session-scoped tasks it tears down and recreates around it.
    """
    while not stop_event.is_set():
        try:
            await asyncio.sleep(KEEPALIVE_INTERVAL_S)
            await stt_session.send_keepalive()
        except ConnectionClosed:
            return


async def run_session_workers(*workers: Coroutine[Any, Any, None]) -> None:
    """Run one STT session's workers until the first of them finishes.

    Every worker needs the same live session, so any one ending means
    the session is over: the others are cancelled and awaited, and then
    the first finisher's outcome becomes this call's outcome.

    That last part is the contract the supervisor depends on. If the
    first finisher raised, the exception propagates, and ``run()`` backs
    off, arms the grace cutoff and reconnects. Swallowing it would turn
    a dead connection into a normal return, which ``run()`` can only
    report as the session having "ended unexpectedly", with the real
    cause lost.

    Exceptions raised by the *other* workers while they're being
    cancelled are discarded, so they can't mask the one that ended the
    session. If this call is itself cancelled (shutdown), every worker
    is cancelled and awaited before the CancelledError propagates.
    """
    tasks = [asyncio.create_task(w) for w in workers]
    try:
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for t in pending:
            t.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        for t in done:
            exc = t.exception()
            if exc is not None:
                raise exc
    finally:
        for t in tasks:
            if not t.done():
                t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


class STTSupervisor:
    def __init__(
        self,
        stt,
        audio: AudioIO,
        stop_event: asyncio.Event,
        on_session: Callable[[Any], Awaitable[None]],
        is_speaking: Callable[[], bool],
    ):
        self._stt = stt
        self._audio = audio
        self._stop_event = stop_event
        self._on_session = on_session
        self._is_speaking = is_speaking

    async def run(self) -> None:
        """Connect, run `on_session` for the life of the connection, and
        reconnect with backoff whenever it fails, until `stop_event` is set.

        Any exception out of the session (including `on_session`) counts as
        a failure; CancelledError propagates.
        """
        # On a brief STT outage we keep the mic running and buffer audio
        # so the user's speech isn't dropped. If the outage exceeds
        # RECONNECT_GRACE_S we stop capture and drain the queue until we
        # reconnect cleanly. If we disconnected mid-SPEAKING we drain on
        # reconnect so buffered TTS echo isn't flushed to the new session
        # as a phantom user turn. Longer-term, hardware AEC (ReSpeaker
        # XVF3800) will remove the echo path entirely and enable barge-in.
        self._grace_task: asyncio.Task | None = None
        self._was_speaking_at_disconnect = False
        self._consecutive_failures = 0

        while not self._stop_event.is_set():
            logger.info("Connecting to Deepgram STT (Flux)...")
            try:
                async with self._stt.session() as stt_session:
                    await self._on_connected()
                    await self._on_session(stt_session)
                    if not self._stop_event.is_set():
                        raise SttSessionEndedError(
                            "STT session ended unexpectedly (connection "
                            "closed or event stream ended)"
                        )
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                await asyncio.sleep(self._on_failure(exc))

        await self._cancel_grace()

    async def _on_connected(self) -> None:
        """Undo the outage handling once a session is open again."""
        await self._cancel_grace()
        if not self._audio.mic_capturing:
            self._audio.start_mic()
            logger.info("Mic capture resumed.")
        if self._was_speaking_at_disconnect:
            self._audio.drain_mic_queue()
            self._was_speaking_at_disconnect = False
        if self._consecutive_failures > 0:
            logger.info(
                "STT reconnected after %d attempt(s)",
                self._consecutive_failures + 1,
            )
        self._consecutive_failures = 0

    def _on_failure(self, exc: Exception) -> float:
        """Record a failed session, arm the grace cutoff, log, and return
        how long to wait before the next attempt."""
        if self._is_speaking():
            self._was_speaking_at_disconnect = True
        if self._grace_task is None:
            self._grace_task = asyncio.create_task(self._grace_cutoff())
        if self._consecutive_failures == 0:
            if isinstance(exc, SttSessionEndedError):
                # Raised by run() itself: the traceback would add nothing.
                logger.warning("%s; reconnecting", exc)
            elif isinstance(exc, ConnectionClosed):
                logger.exception("STT websocket closed; reconnecting")
            else:
                logger.exception("STT session failed; reconnecting")
        else:
            logger.warning(
                "STT reconnect failed (attempt %d): %s",
                self._consecutive_failures + 1,
                exc,
            )
        delay = RECONNECT_BACKOFF_S[
            min(self._consecutive_failures, len(RECONNECT_BACKOFF_S) - 1)
        ]
        self._consecutive_failures += 1
        return delay

    async def _grace_cutoff(self) -> None:
        """Stop mic capture once an outage outlasts RECONNECT_GRACE_S."""
        await asyncio.sleep(RECONNECT_GRACE_S)
        logger.error(
            "STT disconnected for %ds; stopping mic capture until reconnect",
            RECONNECT_GRACE_S,
        )
        self._audio.stop_mic()
        self._audio.drain_mic_queue()

    async def _cancel_grace(self) -> None:
        """Cancel a pending grace cutoff, if any, and wait for it to unwind."""
        if self._grace_task is not None and not self._grace_task.done():
            self._grace_task.cancel()
            try:
                await self._grace_task
            except asyncio.CancelledError:
                # Expected: the grace task's own cancellation. But a cancel
                # aimed at run() (shutdown) that lands during this await
                # arrives as the same exception; only cancelling() tells
                # them apart, and swallowing it would ignore Ctrl-C.
                current = asyncio.current_task()
                if current is not None and current.cancelling():
                    raise
            except Exception:  # noqa: BLE001
                pass
        self._grace_task = None
