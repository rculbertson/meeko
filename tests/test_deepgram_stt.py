"""Unit tests for the Deepgram STT wrapper."""

from contextlib import asynccontextmanager
from unittest.mock import MagicMock, patch

from meeko.deepgram_stt import DeepgramSTT, TurnEvent


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


async def test_events_stops_on_fatal_error():
    fatal = _make_msg("FatalError")
    turn = _make_msg("TurnInfo", event="EndOfTurn", transcript="unreachable")
    stt, _ = _build_stt([fatal, turn])

    async with stt.session() as sess:
        events = [e async for e in sess.events()]

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
