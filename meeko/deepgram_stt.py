"""Async wrapper around Deepgram's v2 live STT (Flux) websocket.

Feeds mic audio in and yields typed events out (StartOfTurn, Update,
EndOfTurn, ...). ``STTSupervisor`` (meeko/stt_supervisor.py) owns the
lifecycle via ``async with DeepgramSTT(...).session() as stt: ...``.
"""

import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass

from deepgram import AsyncDeepgramClient
from deepgram.listen.v2 import client as _dg_client
from deepgram.listen.v2 import raw_client as _dg_raw_client

logger = logging.getLogger("meeko")

MODEL = "flux-general-en"
ENCODING = "linear16"
SAMPLE_RATE = 16000

# Tighten the websockets library's auto-ping watchdog for STT connections.
# Defaults are ping_interval=20s / ping_timeout=20s, which means a silently
# dead TCP connection takes up to ~40s to detect — and any mic audio sent
# during that window is buffered into the dead socket and lost (the user's
# speech never reaches Deepgram). The Deepgram SDK hard-codes its
# ``websockets_client_connect(...)`` call without forwarding kwargs, so we
# patch the symbol on every SDK module that imports it. Both client.py
# (AsyncV2Client.connect, the public path used by self._client.listen.v2)
# and raw_client.py (AsyncRawV2Client, used via .with_raw_response) hold
# their own module-level reference to the function via ``from ... import
# connect as websockets_client_connect``, so each must be patched
# independently. Drop this patch once the SDK exposes these settings via
# request_options.
_STT_PING_INTERVAL_S = 5
_STT_PING_TIMEOUT_S = 5

_PATCH_TARGETS = (_dg_client, _dg_raw_client)
_orig_ws_connect = _dg_raw_client.websockets_client_connect  # pyright: ignore[reportPrivateImportUsage]


def _connect_with_tight_pings(*args, **kwargs):
    kwargs.setdefault("ping_interval", _STT_PING_INTERVAL_S)
    kwargs.setdefault("ping_timeout", _STT_PING_TIMEOUT_S)
    return _orig_ws_connect(*args, **kwargs)


for _mod in _PATCH_TARGETS:
    _mod.websockets_client_connect = _connect_with_tight_pings  # pyright: ignore[reportAttributeAccessIssue]


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

    async def send_keepalive(self) -> None:
        """Send a standard websocket ping frame to keep Deepgram from
        idle-closing the session when we're not streaming audio. v2 Flux
        has no application-level keepalive (unlike v1's
        ``{"type":"KeepAlive"}``); the Deepgram JS SDK's v2 example uses
        raw websocket pings for the same purpose. This is fire-and-forget
        — dead-connection detection is the websockets library's job, via
        the ping_interval / ping_timeout we set at module load."""
        await self._socket._websocket.ping()

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
