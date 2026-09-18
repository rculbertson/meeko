"""Mic chunk routing: wake-word gate in, Deepgram STT out.

`AudioIO`'s callback thread fills a queue with 50ms PCM chunks; this
drains it and decides where each chunk goes, which depends entirely on
the session state:

* **IDLE** — chunks go to the on-device wake-word detector instead of
  Deepgram, so nothing is streamed (or billed) until "Hey Meeko". On a
  detection the session moves to LISTENING and the post-wake silence
  window is armed.
* **SPEAKING** with `mute_mic_while_speaking` — chunks are dropped. This
  is the no-hardware-AEC path (Mac dev, generic USB mics): without it,
  Deepgram hears the assistant's own voice and reports it as a user
  turn. It costs barge-in, which is why it isn't the default.
* **otherwise** — forwarded to STT, relying on the XVF3800's hardware
  AEC to keep playback out of the mic signal.

The chunk that triggers the wake word is deliberately not forwarded: it
contains the wake phrase itself, which isn't part of what the user is
asking.
"""

import asyncio
import logging

from meeko.audio_io import AudioIO
from meeko.idle import IdleController
from meeko.state import State, StateManager
from meeko.wake_word import WakeWordDetector

logger = logging.getLogger("meeko")

# How long to wait on the mic queue before looping. Bounds how long the
# pump can sit blocked while a shutdown is in progress — the queue can
# legitimately stay empty (mic stopped during an STT outage), so the
# stop_event check has to be reachable without a chunk arriving.
_QUEUE_POLL_TIMEOUT_S = 0.1


class MicPump:
    """Drains the mic queue into the wake detector or the STT session."""

    def __init__(
        self,
        *,
        audio: AudioIO,
        state_manager: StateManager,
        idle: IdleController,
        wake_detector: WakeWordDetector | None,
        stop_event: asyncio.Event,
        mute_mic_while_speaking: bool = False,
    ) -> None:
        self._audio = audio
        self._state = state_manager
        self._idle = idle
        self._wake_detector = wake_detector
        self._stop_event = stop_event
        self._mute_mic_while_speaking = mute_mic_while_speaking

    async def run(self, stt_session) -> None:
        """Pump until shutdown. One pump per STT session."""
        while not self._stop_event.is_set():
            try:
                data = await asyncio.wait_for(
                    self._audio.mic_queue.get(), timeout=_QUEUE_POLL_TIMEOUT_S
                )
            except TimeoutError:
                continue
            await self.route_chunk(data, stt_session)

    async def route_chunk(self, data: bytes, stt_session) -> None:
        """Send one mic chunk wherever the current state says it goes."""
        if self._state.state == State.IDLE:
            if self._wake_detector is None:
                # Unreachable today: IDLE is only ever entered with the wake
                # word enabled (_create_wake_detector and
                # _apply_post_turn_session_change in meeko/main.py). An
                # explicit raise rather than an assert, because `python -O`
                # strips asserts, and because this propagates to
                # STTSupervisor, which logs str(exc) on every retry: an
                # AssertionError's empty message would read as an
                # unexplained STT failure.
                raise RuntimeError(
                    "Mic pump reached IDLE with no wake-word detector; IDLE "
                    "requires the wake word to be enabled"
                )
            if self._wake_detector.process(data):
                self._state.set(State.LISTENING)
                logger.info("Wake word accepted; entering LISTENING")
                self._idle.start_post_wake()
            return
        if self._mute_mic_while_speaking and self._state.state == State.SPEAKING:
            return
        await stt_session.send_audio(data)
