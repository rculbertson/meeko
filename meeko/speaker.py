"""Pipelined TTS playback + SPEAKING state transition.

Drains an async iterator of sentence-sized text chunks through Deepgram
TTS and plays the resulting PCM on the speaker. Sentences are
synthesized concurrently (up to `pipeline_depth`) so sentence N+1's
bytes are ready by the time sentence N finishes playing — this
eliminates the TTS time-to-first-byte gap at sentence boundaries.
"""

import asyncio
import logging
import time
from collections.abc import AsyncIterator, Callable
from typing import Any

from meeko.audio_io import RATE, AudioIO
from meeko.config import Profile
from meeko.deepgram_tts import DEFAULT_VOICE, DeepgramTTS

logger = logging.getLogger("meeko")

# Silence inserted between pipelined TTS sentences so back-to-back
# synthesis doesn't run into the next sentence without a natural pause.
# 200ms at 16kHz 16-bit mono = 6400 bytes.
INTER_SENTENCE_PAUSE_MS = 200
INTER_SENTENCE_SILENCE = b"\x00" * (INTER_SENTENCE_PAUSE_MS * RATE * 2 // 1000)


class Speaker:
    def __init__(
        self,
        tts: DeepgramTTS,
        audio: AudioIO,
        profile: Profile,
        enter_speaking: Callable[[], Any],
        exit_speaking: Callable[[Any], None],
        mute_mic_while_speaking: bool = False,
    ):
        """
        `enter_speaking()` is called when a speak starts; its return
        value is passed back to `exit_speaking(prev)` when playback
        completes, letting the caller restore the prior state.

        When `mute_mic_while_speaking` is true (Mac / no-AEC dev path),
        `speak_stream` sleeps briefly after the last write and drains
        the mic queue so buffered echo doesn't leak into STT.
        """
        self._tts = tts
        self._audio = audio
        self._profile = profile
        self._enter_speaking = enter_speaking
        self._exit_speaking = exit_speaking
        self._mute_mic_while_speaking = mute_mic_while_speaking
        self._speak_lock = asyncio.Lock()

    def set_profile(self, profile: Profile) -> None:
        """Swap the active profile so subsequent speaks use its voice."""
        self._profile = profile

    async def speak_stream(self, texts: AsyncIterator[str]) -> None:
        pipeline_depth = 2
        end_marker = object()

        async with self._speak_lock:
            # `prev` stays None until the first PCM chunk is about to be
            # written — that's when we transition the state machine to
            # SPEAKING. Entering earlier (at speak_stream's top) makes
            # the SPEAKING LEDs appear during Claude TTFT + TTS first-
            # byte, hiding PROCESSING and causing a perceived gap
            # between LED change and audible speech.
            prev: Any = None
            try:
                self._audio.reset_speaker_buffer()
                voice = self._profile.voice or DEFAULT_VOICE
                t_start = time.perf_counter()
                t_first_play: float | None = None

                outer: asyncio.Queue = asyncio.Queue(maxsize=pipeline_depth)
                tts_tasks: list[asyncio.Task] = []

                async def tts_into(sentence: str, inner: asyncio.Queue) -> None:
                    try:
                        async for chunk in self._tts.stream(sentence, voice=voice):
                            await inner.put(chunk)
                    finally:
                        await inner.put(None)

                async def produce() -> None:
                    async for sentence in texts:
                        if not sentence:
                            continue
                        logger.info("[assistant] %s", sentence)
                        inner: asyncio.Queue = asyncio.Queue()
                        tts_tasks.append(asyncio.create_task(tts_into(sentence, inner)))
                        await outer.put(inner)
                    await outer.put(end_marker)

                async def consume() -> None:
                    nonlocal t_first_play, prev
                    first_sentence = True
                    while True:
                        inner = await outer.get()
                        if inner is end_marker:
                            return
                        if not first_sentence:
                            await self._audio.write_speaker(INTER_SENTENCE_SILENCE)
                        first_sentence = False
                        while True:
                            chunk = await inner.get()
                            if chunk is None:
                                break
                            if t_first_play is None:
                                t_first_play = time.perf_counter()
                                logger.debug(
                                    "[timing] first_audio_to_speaker=%dms",
                                    int((t_first_play - t_start) * 1000),
                                )
                                prev = self._enter_speaking()
                            await self._audio.write_speaker(chunk)

                cancelled = False
                try:
                    await asyncio.gather(produce(), consume())
                except asyncio.CancelledError:
                    cancelled = True
                    raise
                finally:
                    if cancelled:
                        # Flush the output device first so barge-in is
                        # audible immediately — otherwise buffered PCM
                        # keeps playing for the duration of the TTS
                        # subtask drain below.
                        self._audio.abort_speaker()
                    for t in tts_tasks:
                        if not t.done():
                            t.cancel()
                    await asyncio.gather(*tts_tasks, return_exceptions=True)

                logger.debug(
                    "[timing] speak_total=%dms",
                    int((time.perf_counter() - t_start) * 1000),
                )
                if self._mute_mic_while_speaking:
                    # Tail-drain: speaker buffer may still be flushing,
                    # and any residual echo captured then would feed STT
                    # as a phantom user turn.
                    await asyncio.sleep(1.0)
                    self._audio.drain_mic_queue()
            finally:
                if prev is not None:
                    self._exit_speaking(prev)
                logger.debug("speak complete")

    async def speak(self, text: str) -> None:
        """Single-utterance convenience wrapper (e.g. timer chime)."""
        if not text:
            return

        async def _one() -> AsyncIterator[str]:
            yield text

        await self.speak_stream(_one())
