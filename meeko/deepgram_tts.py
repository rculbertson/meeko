"""Deepgram TTS helper.

Synthesizes a full assistant utterance via the speak REST endpoint and
returns raw linear16 16 kHz mono PCM bytes suitable for direct write to
a PyAudio output stream. Token-level streaming is deferred.
"""

import logging

from deepgram import AsyncDeepgramClient

logger = logging.getLogger("meeko")

DEFAULT_VOICE = "asteria"
ENCODING = "linear16"
CONTAINER = "none"
SAMPLE_RATE = 16000


class DeepgramTTS:
    def __init__(self, api_key: str):
        self._client = AsyncDeepgramClient(api_key=api_key)

    async def synthesize(self, text: str, voice: str = DEFAULT_VOICE) -> bytes:
        """Return the full PCM audio for ``text``."""
        model = f"aura-2-{voice}-en"
        chunks: list[bytes] = []
        async for chunk in self._client.speak.v1.audio.generate(
            text=text,
            model=model,
            encoding=ENCODING,
            container=CONTAINER,
            sample_rate=SAMPLE_RATE,
        ):
            chunks.append(chunk)
        audio = b"".join(chunks)
        logger.debug("TTS synthesized %d bytes for %d chars", len(audio), len(text))
        return audio
