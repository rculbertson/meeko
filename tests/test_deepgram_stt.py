"""Unit tests for the Deepgram STT wrapper."""

import asyncio
import os
from contextlib import asynccontextmanager
from unittest.mock import MagicMock, patch

import pytest
from deepgram.listen.v2 import client as _dg_client
from deepgram.listen.v2 import raw_client as _dg_raw_client

import meeko.deepgram_stt as deepgram_stt
from meeko.deepgram_stt import DeepgramSTT, TurnEvent
from meeko.deepgram_tts import DeepgramTTS


def _make_msg(msg_type: str, **attrs):
    m = MagicMock()
    m.type = msg_type
    for k, v in attrs.items():
        setattr(m, k, v)
    return m


class _FakeWebsocket:
    def __init__(self):
        self.pings = 0

    async def ping(self):
        self.pings += 1


class _FakeSocket:
    def __init__(self, messages: list):
        self.messages = messages
        self.sent: list[bytes] = []
        self._websocket = _FakeWebsocket()

    async def send_media(self, pcm: bytes) -> None:
        self.sent.append(pcm)

    def __aiter__(self):
        return self._gen()

    async def _gen(self):
        for m in self.messages:
            yield m


def _build_stt(messages: list) -> tuple[DeepgramSTT, _FakeSocket]:
    socket = _FakeSocket(messages)
    mock_client = MagicMock()

    @asynccontextmanager
    async def _fake_connect(**kwargs):
        yield socket

    mock_client.listen.v2.connect = _fake_connect
    with patch("meeko.deepgram_stt.AsyncDeepgramClient", return_value=mock_client):
        stt = DeepgramSTT(api_key="x")
    return stt, socket


async def test_events_yields_turn_info():
    msg = _make_msg("TurnInfo", event="EndOfTurn", transcript="hello world")
    stt, _ = _build_stt([msg])

    async with stt.session() as sess:
        events = [e async for e in sess.events()]

    assert events == [TurnEvent(event="EndOfTurn", transcript="hello world")]


async def test_events_skips_bytes():
    msg = _make_msg("TurnInfo", event="StartOfTurn", transcript="hi")
    stt, _ = _build_stt([b"\x00\x01", msg])

    async with stt.session() as sess:
        events = [e async for e in sess.events()]

    assert len(events) == 1
    assert events[0].event == "StartOfTurn"


async def test_events_skips_non_turn_info_messages():
    other = _make_msg("Connected")
    turn = _make_msg("TurnInfo", event="Update", transcript="hey")
    stt, _ = _build_stt([other, turn])

    async with stt.session() as sess:
        events = [e async for e in sess.events()]

    assert events == [TurnEvent(event="Update", transcript="hey")]


async def test_events_raises_on_fatal_error():
    """A FatalError ends the session as a failure, so the supervisor backs
    off (and logs why) instead of reconnecting straight away."""
    fatal = _make_msg("Error")  # ListenV2FatalError's type, per the SDK
    turn = _make_msg("TurnInfo", event="EndOfTurn", transcript="unreachable")
    stt, _ = _build_stt([fatal, turn])

    events = []
    async with stt.session() as sess:
        with pytest.raises(RuntimeError, match="STT fatal error"):
            async for e in sess.events():
                events.append(e)

    assert events == []


async def test_events_coerces_none_transcript():
    msg = _make_msg("TurnInfo", event="EagerEndOfTurn", transcript=None)
    stt, _ = _build_stt([msg])

    async with stt.session() as sess:
        events = [e async for e in sess.events()]

    assert events == [TurnEvent(event="EagerEndOfTurn", transcript="")]


async def test_send_audio_forwards_bytes():
    stt, socket = _build_stt([])

    async with stt.session() as sess:
        await sess.send_audio(b"\xde\xad\xbe\xef")

    assert socket.sent == [b"\xde\xad\xbe\xef"]


async def test_send_keepalive_sends_websocket_ping():
    stt, socket = _build_stt([])

    async with stt.session() as sess:
        await sess.send_keepalive()

    assert socket._websocket.pings == 1


def test_patch_installed_on_every_sdk_module_that_imports_connect():
    """Both client.py (the public AsyncV2Client.connect path) and
    raw_client.py (AsyncRawV2Client) hold their own module-level
    reference to ``websockets_client_connect``. If we patch only one,
    the other's call site silently uses the un-tuned default and our
    fix is dead code at runtime — exactly the bug this guards against."""
    assert (
        _dg_client.websockets_client_connect is deepgram_stt._connect_with_tight_pings
    )
    assert (
        _dg_raw_client.websockets_client_connect
        is deepgram_stt._connect_with_tight_pings
    )


def test_websocket_connect_is_patched_with_tight_pings():
    """The patched function injects ping_interval and ping_timeout so
    dead TCP connections are detected in seconds, not ~17-40s."""
    captured: dict = {}

    def fake_orig(*args, **kwargs):
        captured.update(kwargs)
        return MagicMock()

    with patch.object(deepgram_stt, "_orig_ws_connect", fake_orig):
        _dg_client.websockets_client_connect("wss://example", extra_headers={})

    assert captured.get("ping_interval") == deepgram_stt._STT_PING_INTERVAL_S
    assert captured.get("ping_timeout") == deepgram_stt._STT_PING_TIMEOUT_S


def test_websocket_connect_patch_does_not_clobber_explicit_kwargs():
    """If a future SDK version starts forwarding ping settings, we must
    not overwrite them — ``setdefault`` semantics."""
    captured: dict = {}

    def fake_orig(*args, **kwargs):
        captured.update(kwargs)
        return MagicMock()

    with patch.object(deepgram_stt, "_orig_ws_connect", fake_orig):
        _dg_client.websockets_client_connect(
            "wss://example", extra_headers={}, ping_interval=2, ping_timeout=3
        )

    assert captured["ping_interval"] == 2
    assert captured["ping_timeout"] == 3


async def test_multiple_turn_events_in_order():
    msgs = [
        _make_msg("TurnInfo", event="StartOfTurn", transcript=""),
        _make_msg("TurnInfo", event="Update", transcript="hel"),
        _make_msg("TurnInfo", event="EndOfTurn", transcript="hello"),
    ]
    stt, _ = _build_stt(msgs)

    async with stt.session() as sess:
        events = [e async for e in sess.events()]

    assert [e.event for e in events] == ["StartOfTurn", "Update", "EndOfTurn"]
    assert events[-1].transcript == "hello"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_live_stt_session_transcribes_audio():
    api_key = os.environ.get("DEEPGRAM_API_KEY")
    if not api_key:
        pytest.skip("DEEPGRAM_API_KEY not set")

    tts = DeepgramTTS(api_key=api_key)
    chunks = [c async for c in tts.stream("Testing one two three.")]
    assert chunks

    stt = DeepgramSTT(api_key=api_key)
    async with stt.session() as sess:
        await sess.send_keepalive()

        async def feeder():
            for c in chunks:
                await sess.send_audio(c)
                await asyncio.sleep(0.05)
            # Send trailing silence so Flux detects EndOfTurn
            for _ in range(10):
                await sess.send_audio(b"\x00" * 3200)
                await asyncio.sleep(0.1)

        async def reader():
            events = []
            async for e in sess.events():
                events.append(e)
                if e.event == "EndOfTurn":
                    break
            return events

        feed_task = asyncio.create_task(feeder())
        read_task = asyncio.create_task(reader())
        done, _ = await asyncio.wait([read_task], timeout=10.0)
        feed_task.cancel()
        read_task.cancel()
        await asyncio.gather(feed_task, read_task, return_exceptions=True)

        assert read_task in done, "timed out waiting for EndOfTurn"
        events = read_task.result()
        assert any(e.event == "StartOfTurn" for e in events)
        assert any(e.event == "EndOfTurn" for e in events)
        end_event = next(e for e in reversed(events) if e.event == "EndOfTurn")
        assert "testing" in end_event.transcript.lower()
