"""Tests for the `meeko.main` orchestrator module.

Covers `setup_logging`, the `State` enum, the sync `main()` entrypoint,
and a happy-path drive-through of `run()` with all external services
(PyAudio, Deepgram STT/TTS, Anthropic) mocked.
"""

import asyncio
import contextlib
import logging
import logging.handlers
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from meeko import main as meeko_main
from meeko.main import State, setup_logging
from meeko.profiles import Profile


@pytest.fixture
def clean_logger():
    logger = logging.getLogger("meeko")
    saved_handlers = logger.handlers[:]
    saved_level = logger.level
    logger.handlers = []
    yield logger
    logger.handlers = saved_handlers
    logger.setLevel(saved_level)


def test_setup_logging_defaults_to_stream_handler(monkeypatch, clean_logger):
    monkeypatch.delenv("MEEKO_LOG_TARGET", raising=False)
    monkeypatch.delenv("MEEKO_LOG_LEVEL", raising=False)
    setup_logging()
    assert len(clean_logger.handlers) == 1
    assert isinstance(clean_logger.handlers[0], logging.StreamHandler)
    assert clean_logger.level == logging.DEBUG


def test_setup_logging_file_target(monkeypatch, tmp_path, clean_logger):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MEEKO_LOG_TARGET", "file")
    monkeypatch.setenv("MEEKO_LOG_LEVEL", "INFO")
    setup_logging()
    assert isinstance(clean_logger.handlers[0], logging.handlers.RotatingFileHandler)
    assert clean_logger.level == logging.INFO
    # Close the file handler so tmp_path can be cleaned up on Windows/macOS.
    clean_logger.handlers[0].close()


def test_setup_logging_invalid_level_falls_back_to_debug(monkeypatch, clean_logger):
    monkeypatch.delenv("MEEKO_LOG_TARGET", raising=False)
    monkeypatch.setenv("MEEKO_LOG_LEVEL", "NOT_A_REAL_LEVEL")
    setup_logging()
    assert clean_logger.level == logging.DEBUG


def test_state_enum_members():
    assert {s.name for s in State} == {"LISTENING", "PROCESSING", "SPEAKING"}


def test_main_drives_run_to_completion():
    ran = {"count": 0}

    async def fake_run():
        ran["count"] += 1

    with patch("meeko.main.run", new=fake_run):
        meeko_main.main()

    assert ran["count"] == 1


def test_main_swallows_keyboard_interrupt():
    async def raising_run():
        raise KeyboardInterrupt

    with patch("meeko.main.run", new=raising_run):
        # Should not raise.
        meeko_main.main()


# --------------------------------------------------------------------------
# run() happy-path drive-through
# --------------------------------------------------------------------------


class _FakeSTTSession:
    def __init__(self, events):
        self._events = events
        self.sent_audio_count = 0
        self.keepalive_count = 0

    async def send_audio(self, pcm: bytes) -> None:
        self.sent_audio_count += 1

    async def send_keepalive(self) -> None:
        self.keepalive_count += 1

    async def events(self):
        for ev in self._events:
            yield ev
        # Hold open so pump_mic keeps running until the test cancels
        # the task; otherwise handle_turns returns and asyncio.gather
        # still waits on pump_mic forever.
        await asyncio.sleep(10)


class _FakeSTTClient:
    def __init__(self, api_key):
        self.session_obj = None
        self.events = [
            SimpleNamespace(event="StartOfTurn", transcript=""),
            SimpleNamespace(event="EndOfTurn", transcript="hello"),
        ]

    @contextlib.asynccontextmanager
    async def session(self):
        self.session_obj = _FakeSTTSession(self.events)
        yield self.session_obj


class _FakeTTSClient:
    def __init__(self, api_key):
        self.calls: list[tuple[str, str]] = []

    async def stream(self, text, voice):
        self.calls.append((text, voice))
        yield b"\x01\x02\x03\x04"


class _FakeClaudeClient:
    def __init__(self, api_key, system_prompt, dispatcher):
        self.api_key = api_key
        self.system_prompt = system_prompt
        self.dispatcher = dispatcher
        self.turns: list[str] = []

    def stream_turn(self, text):
        self.turns.append(text)

        async def _gen():
            yield "Hi."

        return _gen()


@pytest.fixture
def fake_profiles():
    return {
        "default": Profile(
            name="default",
            wake_word="meeko",
            prompt="system",
            greeting="",  # skip greeting to avoid an extra speak round
            voice=None,
        )
    }


async def test_run_drives_one_turn_end_to_end(monkeypatch, fake_profiles):
    monkeypatch.setenv("DEEPGRAM_API_KEY", "dg-test")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-test")

    # Mic callback is never exercised here; mic_queue stays empty and
    # pump_mic just spins on its 100ms timeout.
    pa_instance = MagicMock()
    mic_stream = MagicMock()
    speaker_stream = MagicMock()

    first_write = asyncio.Event()

    def on_speaker_write(data):
        first_write.set()

    speaker_stream.write.side_effect = on_speaker_write
    pa_instance.open.side_effect = [mic_stream, speaker_stream]

    fake_stt = _FakeSTTClient("dg-test")
    fake_tts = _FakeTTSClient("dg-test")
    fake_claude_holder: dict = {}

    def make_claude(api_key, system_prompt, dispatcher):
        c = _FakeClaudeClient(api_key, system_prompt, dispatcher)
        fake_claude_holder["client"] = c
        return c

    # Short-circuit the 1s tail-drain so the test doesn't drag.
    real_sleep = asyncio.sleep

    async def fast_sleep(delay, *a, **kw):
        if delay >= 1.0:
            return await real_sleep(0)
        return await real_sleep(delay, *a, **kw)

    with (
        patch("meeko.main.pyaudio.PyAudio", return_value=pa_instance),
        patch("meeko.main.load_profiles", return_value=fake_profiles),
        patch("meeko.main.load_dotenv"),
        patch("meeko.main.DeepgramSTT", return_value=fake_stt),
        patch("meeko.main.DeepgramTTS", return_value=fake_tts),
        patch("meeko.main.ClaudeClient", side_effect=make_claude),
        patch("meeko.main.asyncio.sleep", new=fast_sleep),
        patch("meeko.main.setup_logging"),  # avoid mutating the real logger
    ):
        task = asyncio.create_task(meeko_main.run())
        try:
            await asyncio.wait_for(first_write.wait(), timeout=5)
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task

    claude = fake_claude_holder["client"]
    assert claude.turns == ["hello"]
    assert fake_tts.calls == [("Hi.", "asteria")]
    # First speaker.write call received the TTS chunk we yielded.
    speaker_stream.write.assert_any_call(b"\x01\x02\x03\x04")
    # Cleanup was executed in run()'s finally block.
    mic_stream.stop_stream.assert_called()
    mic_stream.close.assert_called()
    speaker_stream.stop_stream.assert_called()
    speaker_stream.close.assert_called()
    pa_instance.terminate.assert_called()


class _ReconnectSTTClient:
    """STT client whose first session's events() raises ConnectionClosed,
    forcing the orchestrator to reconnect. The second session completes a
    turn normally."""

    def __init__(self, api_key):
        self.session_count = 0

    @contextlib.asynccontextmanager
    async def session(self):
        self.session_count += 1
        count = self.session_count
        session = _ReconnectSTTSession(fail_on_events=(count == 1))
        yield session


class _ReconnectSTTSession:
    def __init__(self, fail_on_events: bool):
        self._fail = fail_on_events

    async def send_audio(self, pcm: bytes) -> None:
        return None

    async def send_keepalive(self) -> None:
        return None

    async def events(self):
        if self._fail:
            from websockets.exceptions import ConnectionClosedError

            raise ConnectionClosedError(None, None)
        yield SimpleNamespace(event="EndOfTurn", transcript="hello")
        await asyncio.sleep(10)


async def test_run_reconnects_stt_after_connection_closed(monkeypatch, fake_profiles):
    monkeypatch.setenv("DEEPGRAM_API_KEY", "dg-test")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-test")

    pa_instance = MagicMock()
    mic_stream = MagicMock()
    speaker_stream = MagicMock()

    first_write = asyncio.Event()
    speaker_stream.write.side_effect = lambda data: first_write.set()
    pa_instance.open.side_effect = [mic_stream, speaker_stream]

    fake_stt = _ReconnectSTTClient("dg-test")
    fake_tts = _FakeTTSClient("dg-test")

    def make_claude(api_key, system_prompt, dispatcher):
        return _FakeClaudeClient(api_key, system_prompt, dispatcher)

    real_sleep = asyncio.sleep

    async def fast_sleep(delay, *a, **kw):
        if delay >= 1.0:
            return await real_sleep(0)
        return await real_sleep(delay, *a, **kw)

    with (
        patch("meeko.main.pyaudio.PyAudio", return_value=pa_instance),
        patch("meeko.main.load_profiles", return_value=fake_profiles),
        patch("meeko.main.load_dotenv"),
        patch("meeko.main.DeepgramSTT", return_value=fake_stt),
        patch("meeko.main.DeepgramTTS", return_value=fake_tts),
        patch("meeko.main.ClaudeClient", side_effect=make_claude),
        patch("meeko.main.asyncio.sleep", new=fast_sleep),
        patch("meeko.main.setup_logging"),
    ):
        task = asyncio.create_task(meeko_main.run())
        try:
            await asyncio.wait_for(first_write.wait(), timeout=5)
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task

    # Session was re-established after the ConnectionClosed on the first.
    assert fake_stt.session_count >= 2


async def test_run_backs_off_on_repeated_stt_failures(monkeypatch, fake_profiles):
    """Repeated connect failures should sleep with increasing delays
    (0.5, 1, 2, ...) and only log a full traceback on the first failure."""
    monkeypatch.setenv("DEEPGRAM_API_KEY", "dg-test")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-test")

    pa_instance = MagicMock()
    pa_instance.open.side_effect = [MagicMock(), MagicMock()]

    class _AlwaysFailSTT:
        def __init__(self, api_key):
            self.attempts = 0

        @contextlib.asynccontextmanager
        async def session(self):
            self.attempts += 1
            raise OSError("dns down")
            yield  # pragma: no cover

    fake_stt = _AlwaysFailSTT("dg-test")
    sleep_calls: list[float] = []
    real_sleep = asyncio.sleep

    async def recording_sleep(delay, *a, **kw):
        sleep_calls.append(delay)
        # collapse real sleep so the test finishes quickly
        return await real_sleep(0)

    with (
        patch("meeko.main.pyaudio.PyAudio", return_value=pa_instance),
        patch("meeko.main.load_profiles", return_value=fake_profiles),
        patch("meeko.main.load_dotenv"),
        patch("meeko.main.DeepgramSTT", return_value=fake_stt),
        patch("meeko.main.DeepgramTTS", return_value=_FakeTTSClient("x")),
        patch(
            "meeko.main.ClaudeClient",
            side_effect=lambda api_key, system_prompt, dispatcher: _FakeClaudeClient(
                api_key, system_prompt, dispatcher
            ),
        ),
        patch("meeko.main.asyncio.sleep", new=recording_sleep),
        patch("meeko.main.setup_logging"),
    ):
        task = asyncio.create_task(meeko_main.run())
        # Let the loop churn through several reconnect attempts.
        for _ in range(50):
            if fake_stt.attempts >= 4:
                break
            await real_sleep(0)
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await task

    # Reconnect delays grow: 0.5, 1, 2, 4, ...
    assert sleep_calls[:4] == [0.5, 1, 2, 4]


async def test_run_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("DEEPGRAM_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with (
        patch("meeko.main.load_dotenv"),
        patch("meeko.main.setup_logging"),
    ):
        with pytest.raises(KeyError):
            await meeko_main.run()
