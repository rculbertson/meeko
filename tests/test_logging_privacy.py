"""Conversation content must not reach the system log.

Meeko runs under systemd with stderr going to journald, so anything the
`meeko` logger emits at the configured level is captured by the system
journal and retained there. The rule is that transcripts, assistant
replies, tool arguments and session titles are logged at DEBUG only, and
`log_level` defaults to INFO — so a default install journals state
transitions and errors but nothing that was said.

Each test drives the real log site twice: once with the logger at INFO
(the secret must be absent) and once at DEBUG (it must be present, so a
line that was deleted outright rather than demoted still fails).
"""

import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock

import pytest

from meeko.config import MeekoConfig, Profile
from meeko.leds import LedState
from meeko.orchestrator.state import State, StateManager
from meeko.orchestrator.stt_events import SttEventRouter
from meeko.orchestrator.turn_worker import TurnWorker
from meeko.speaker import Speaker
from meeko.tools.dispatch import ToolDispatcher

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
    """The default that keeps DEBUG content out of journald."""
    assert MeekoConfig().log_level == "INFO"


def test_setup_logging_defaults_to_info():
    from meeko.main import setup_logging

    logger = logging.getLogger("meeko")
    handlers = list(logger.handlers)
    level = logger.level
    try:
        setup_logging()
        assert logger.level == logging.INFO
        # An unrecognized level must not fall back to DEBUG.
        setup_logging(log_level="nonsense")
        assert logger.level == logging.INFO
    finally:
        logger.handlers[:] = handlers
        logger.setLevel(level)


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
        Profile(name="query", wake_word="meeko", prompt="sys", voice=None),
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
