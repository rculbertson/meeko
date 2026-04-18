"""Deepgram TTS helper.

Streams audio from Deepgram's speak REST endpoint as raw linear16 16 kHz
mono PCM bytes. Callers are expected to forward chunks straight to the
speaker so playback starts as soon as the first chunk arrives.
"""

import logging
import time
from collections.abc import AsyncIterator

from deepgram import AsyncDeepgramClient

logger = logging.getLogger("meeko")

DEFAULT_VOICE = "asteria"
ENCODING = "linear16"
CONTAINER = "none"
SAMPLE_RATE = 16000


class DeepgramTTS:
    def __init__(self, api_key: str):
        self._client = AsyncDeepgramClient(api_key=api_key)

    async def stream(
        self, text: str, voice: str = DEFAULT_VOICE
    ) -> AsyncIterator[bytes]:
        """Yield PCM chunks for ``text`` as they arrive from Deepgram."""
        model = f"aura-2-{voice}-en"
        t_start = time.perf_counter()
        t_first: float | None = None
        total_bytes = 0
        async for chunk in self._client.speak.v1.audio.generate(
            text=text,
            model=model,
            encoding=ENCODING,
            container=CONTAINER,
            sample_rate=SAMPLE_RATE,
        ):
            if t_first is None:
                t_first = time.perf_counter()
                logger.debug(
                    "[timing] tts ttfb=%dms chars=%d",
                    int((t_first - t_start) * 1000),
                    len(text),
                )
            total_bytes += len(chunk)
            yield chunk
        total_ms = int((time.perf_counter() - t_start) * 1000)
        audio_ms = int(total_bytes / 2 / SAMPLE_RATE * 1000)
        logger.debug(
            "[timing] tts synth_total=%dms bytes=%d audio_duration=%dms",
            total_ms,
            total_bytes,
            audio_ms,
        )
