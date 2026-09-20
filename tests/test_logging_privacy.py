"""Conversation content must not reach the system log.

Meeko runs under systemd with stderr going to journald, so anything the
`meeko` logger emits at the configured level is captured by the system
journal and retained there. The rule is that transcripts, assistant
replies, tool arguments and session titles are logged at DEBUG only, and
`log_level` defaults to INFO — so a default install journals state
transitions, tool names and errors but nothing that was said.

Each test drives the real log site twice: once with the logger at INFO
(the secret must be absent) and once at DEBUG (it must be present, so a
line that was deleted outright rather than demoted still fails).
"""

import asyncio
import json
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from meeko import session_summary
from meeko.config import MeekoConfig, Profile
from meeko.leds import LedState
from meeko.orchestrator.state import State, StateManager
from meeko.orchestrator.stt_events import SttEventRouter
from meeko.orchestrator.turn_worker import TurnWorker
from meeko.sessions import SessionStore
from meeko.speaker import Speaker
from meeko.tools.dispatch import ToolDispatcher
from meeko.tools.timer import TimerManager
from meeko.tools.weather import WeatherClient

SECRET = "my therapist said something unrepeatable"


class _NullLeds:
    def set_state(self, state: LedState) -> None:
        pass

    def error(self) -> None:
        pass


class _FakeIdle:
    def cancel(self) -> None:
        pass

    def start_post_turn(self) -> None:
        pass


def test_default_log_level_is_info():
    """The default that keeps DEBUG content out of journald.
    `setup_logging`'s own default is covered in tests/test_main.py."""
    assert MeekoConfig().log_level == "INFO"


@pytest.mark.parametrize("level", [logging.INFO, logging.DEBUG])
async def test_user_transcript_only_logged_at_debug(caplog, level):
    """turn_worker logs the user's utterance as it picks up the turn."""
    queue: asyncio.Queue = asyncio.Queue()
    stop = asyncio.Event()
    worker = TurnWorker(
        turn_queue=queue,
        state_manager=StateManager(State.LISTENING, _NullLeds()),
        idle=_FakeIdle(),
        leds=_NullLeds(),
        start_turn=lambda text: asyncio.sleep(0),
        apply_session_change=AsyncMock(return_value=State.LISTENING),
        stop_event=stop,
    )
    with caplog.at_level(level, logger="meeko"):
        task = asyncio.create_task(worker.run())
        await queue.put(SECRET)
        while not queue.empty():
            await asyncio.sleep(0)
        await asyncio.sleep(0)
        stop.set()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    assert (SECRET in caplog.text) is (level == logging.DEBUG)


@pytest.mark.parametrize("level", [logging.INFO, logging.DEBUG])
async def test_echo_transcript_only_logged_at_debug(caplog, level):
    """A transcript arriving during SPEAKING is dropped as echo — and
    logged, since it's the only record that STT heard it at all."""
    router = SttEventRouter(
        state_manager=StateManager(State.SPEAKING, _NullLeds()),
        idle=_FakeIdle(),
        turn_queue=asyncio.Queue(),
        request_barge_in=lambda: None,
        stop_event=asyncio.Event(),
    )
    ev = MagicMock()
    ev.event = "EndOfTurn"
    ev.transcript = SECRET

    with caplog.at_level(level, logger="meeko"):
        await router.handle(ev)

    assert (SECRET in caplog.text) is (level == logging.DEBUG)


@pytest.mark.parametrize("level", [logging.INFO, logging.DEBUG])
async def test_assistant_sentences_only_logged_at_debug(caplog, level):
    """The speaker logs each sentence it sends to TTS."""
    tts = MagicMock()

    def stream(text, voice):
        async def _gen():
            yield b""

        return _gen()

    tts.stream.side_effect = stream
    audio = MagicMock()
    audio.write_speaker = AsyncMock()
    audio.drain_mic_queue = MagicMock()
    speaker = Speaker(
        tts,
        audio,
        Profile(name="query", prompt="sys", voice=None),
        enter_speaking=lambda: None,
        exit_speaking=lambda prev: None,
    )

    async def _sentences():
        yield SECRET

    with caplog.at_level(level, logger="meeko"):
        await speaker.speak_stream(_sentences())

    assert (SECRET in caplog.text) is (level == logging.DEBUG)


@pytest.mark.parametrize("level", [logging.INFO, logging.DEBUG])
async def test_tool_arguments_only_logged_at_debug(caplog, level):
    """Tool arguments carry conversation content (session search queries,
    timer labels, place names); the tool name alone stays at INFO."""
    dispatcher = ToolDispatcher()
    dispatcher.register(
        [{"name": "list_sessions", "description": "d", "input_schema": {}}],
        AsyncMock(return_value="ok"),
    )

    with caplog.at_level(level, logger="meeko"):
        await dispatcher.dispatch("list_sessions", {"query": SECRET})

    assert "list_sessions" in caplog.text
    assert (SECRET in caplog.text) is (level == logging.DEBUG)


@pytest.mark.parametrize("level", [logging.INFO, logging.DEBUG])
async def test_session_title_only_logged_at_debug(tmp_path, caplog, level):
    """The summarizer's title is derived from the conversation. This path
    is fire-and-forget after a session ends, so a regression here would
    otherwise go unnoticed."""
    store = SessionStore.open(tmp_path / "meeko.db")
    try:
        session_id = await store.create_session("query")
        await store.persist_turn(session_id, "user", SECRET)
        client = MagicMock()
        client.messages = MagicMock()
        client.messages.create = AsyncMock(
            return_value=SimpleNamespace(
                content=[
                    SimpleNamespace(
                        type="text",
                        text=json.dumps({"title": SECRET, "summary": SECRET}),
                    )
                ],
                stop_reason="end_turn",
            )
        )
        with caplog.at_level(level, logger="meeko"):
            await session_summary.summarize_session(
                store=store, anthropic_client=client, session_id=session_id
            )
    finally:
        await store.close()

    assert "wrote summary" in caplog.text
    assert (SECRET in caplog.text) is (level == logging.DEBUG)


@pytest.mark.parametrize("level", [logging.INFO, logging.DEBUG])
async def test_summarizer_raw_response_only_logged_at_debug(tmp_path, caplog, level):
    """A response that fails to parse still paraphrases the conversation."""
    store = SessionStore.open(tmp_path / "meeko.db")
    try:
        session_id = await store.create_session("query")
        await store.persist_turn(session_id, "user", "hello")
        client = MagicMock()
        client.messages = MagicMock()
        client.messages.create = AsyncMock(
            return_value=SimpleNamespace(
                content=[SimpleNamespace(type="text", text=f"not json: {SECRET}")],
                stop_reason="end_turn",
            )
        )
        with caplog.at_level(level, logger="meeko"):
            await session_summary.summarize_session(
                store=store, anthropic_client=client, session_id=session_id
            )
    finally:
        await store.close()

    assert "invalid JSON" in caplog.text
    assert (SECRET in caplog.text) is (level == logging.DEBUG)


@pytest.mark.parametrize("level", [logging.INFO, logging.DEBUG])
async def test_timer_label_only_logged_at_debug(caplog, level):
    """Timer labels are user-chosen ("call the clinic"), including on the
    announcement-failure path."""
    speak = AsyncMock(side_effect=RuntimeError("TTS unavailable"))
    manager = TimerManager(speak_callback=speak)
    with pytest.MonkeyPatch.context() as mp, caplog.at_level(level, logger="meeko"):
        mp.setattr(asyncio, "sleep", AsyncMock())
        await manager.set_timer(60, "1 minute", SECRET)
        await manager._timers[SECRET][0]

    assert "Timer announcement failed" in caplog.text
    assert (SECRET in caplog.text) is (level == logging.DEBUG)


@pytest.mark.parametrize("level", [logging.INFO, logging.DEBUG])
async def test_weather_failure_does_not_log_coordinates(caplog, level):
    """httpx puts the request URL — and so the coordinates, which are the
    user's home location by default — in the error message and traceback."""
    client = WeatherClient()
    client.configure(latitude=40.7484, longitude=-73.9857, units="imperial")

    request = httpx.Request(
        "GET",
        "https://api.open-meteo.com/v1/forecast?latitude=40.7484&longitude=-73.9857",
    )
    response = httpx.Response(503, request=request)

    async def _boom(self, params):
        raise httpx.HTTPStatusError(
            f"Server error '503' for url '{request.url}'",
            request=request,
            response=response,
        )

    with pytest.MonkeyPatch.context() as mp, caplog.at_level(level, logger="meeko"):
        mp.setattr(WeatherClient, "_get", _boom)
        out = await client.get_weather(None, None, None)

    # The user still gets a friendly answer, and the failure is still
    # visible in the journal — with the status code but not the URL.
    assert "weather" in out.lower()
    assert "HTTP 503" in caplog.text
    assert ("40.7484" in caplog.text) is (level == logging.DEBUG)
