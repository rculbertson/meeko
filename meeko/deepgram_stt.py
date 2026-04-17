"""Async wrapper around Deepgram's v2 live STT (Flux) websocket.

Feeds mic audio in and yields typed events out (StartOfTurn, Update,
EndOfTurn, ...). The orchestrator owns the lifecycle via
``async with DeepgramSTT(...).session() as stt: ...``.
"""

import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass

from deepgram import AsyncDeepgramClient

logger = logging.getLogger("meeko")

MODEL = "flux-general-en"
ENCODING = "linear16"
SAMPLE_RATE = 16000


@dataclass
class TurnEvent:
    event: str  # "StartOfTurn" | "Update" | "EagerEndOfTurn" | "EndOfTurn" | ...
    transcript: str


class DeepgramSTT:
    """Small facade over ``client.listen.v2.connect(...)``."""

    def __init__(self, api_key: str):
        self._client = AsyncDeepgramClient(api_key=api_key)

    @asynccontextmanager
    async def session(self):
        async with self._client.listen.v2.connect(
            model=MODEL,
            encoding=ENCODING,
            sample_rate=SAMPLE_RATE,
        ) as socket:
            yield _Session(socket)


class _Session:
    def __init__(self, socket):
        self._socket = socket

    async def send_audio(self, pcm: bytes) -> None:
        await self._socket.send_media(pcm)

    async def events(self):
        """Yield TurnEvent objects for each turn_info message.

        Non-turn-info messages (Connected, FatalError, stray bytes) are
        logged and skipped so callers only see conversational events.
        """
        async for message in self._socket:
            if isinstance(message, bytes):
                continue
            msg_type = getattr(message, "type", None)
            if msg_type == "TurnInfo":
                event = getattr(message, "event", "")
                transcript = getattr(message, "transcript", "") or ""
                yield TurnEvent(event=event, transcript=transcript)
            elif msg_type == "FatalError":
                logger.error("STT fatal error: %s", message)
                return
            else:
                logger.debug("STT message: %s", message)
