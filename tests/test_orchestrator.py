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

    async def send_audio(self, pcm: bytes) -> None:
        self.sent_audio_count += 1

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


async def test_run_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("DEEPGRAM_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with (
        patch("meeko.main.load_dotenv"),
        patch("meeko.main.setup_logging"),
    ):
        with pytest.raises(KeyError):
            await meeko_main.run()
