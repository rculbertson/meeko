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
from collections.abc import Awaitable, Callable
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
        # Reconnect backoff: 0.5s → 1 → 2 → 4 → 8 → 16 → 30 (cap), reset
        # on a successful connect. The first failure in a streak logs the
        # full traceback; subsequent failures log one line so a prolonged
        # outage doesn't flood the logs.
        backoff_schedule = [0.5, 1, 2, 4, 8, 16, 30]
        consecutive_failures = 0

        # On a brief STT outage we keep the mic running and buffer audio
        # so the user's speech isn't dropped. If the outage exceeds
        # RECONNECT_GRACE_S we stop capture and drain the queue until we
        # reconnect cleanly. If we disconnected mid-SPEAKING we drain on
        # reconnect so buffered TTS echo isn't flushed to the new session
        # as a phantom user turn. Longer-term, hardware AEC (ReSpeaker
        # XVF3800) will remove the echo path entirely and enable barge-in.
        grace_task: asyncio.Task | None = None
        was_speaking_at_disconnect = False

        async def grace_cutoff() -> None:
            await asyncio.sleep(RECONNECT_GRACE_S)
            logger.error(
                "STT disconnected for %ds; stopping mic capture until reconnect",
                RECONNECT_GRACE_S,
            )
            self._audio.stop_mic()
            self._audio.drain_mic_queue()

        while not self._stop_event.is_set():
            logger.info("Connecting to Deepgram STT (Flux)...")
            try:
                async with self._stt.session() as stt_session:
                    if grace_task is not None:
                        grace_task.cancel()
                        try:
                            await grace_task
                        except asyncio.CancelledError, Exception:
                            pass
                        grace_task = None
                    if not self._audio.mic_capturing:
                        self._audio.start_mic()
                        logger.info("Mic capture resumed.")
                    if was_speaking_at_disconnect:
                        self._audio.drain_mic_queue()
                        was_speaking_at_disconnect = False
                    if consecutive_failures > 0:
                        logger.info(
                            "STT reconnected after %d attempt(s)",
                            consecutive_failures + 1,
                        )
                    consecutive_failures = 0
                    await self._on_session(stt_session)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if self._is_speaking():
                    was_speaking_at_disconnect = True
                if grace_task is None:
                    grace_task = asyncio.create_task(grace_cutoff())
                if consecutive_failures == 0:
                    if isinstance(exc, ConnectionClosed):
                        logger.exception("STT websocket closed; reconnecting")
                    else:
                        logger.exception("STT session failed; reconnecting")
                else:
                    logger.warning(
                        "STT reconnect failed (attempt %d): %s",
                        consecutive_failures + 1,
                        exc,
                    )
                delay = backoff_schedule[
                    min(consecutive_failures, len(backoff_schedule) - 1)
                ]
                consecutive_failures += 1
                try:
                    await asyncio.sleep(delay)
                except asyncio.CancelledError:
                    raise
                continue

        if grace_task is not None and not grace_task.done():
            grace_task.cancel()
            try:
                await grace_task
            except asyncio.CancelledError, Exception:
                pass
