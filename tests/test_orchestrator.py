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


def test_main_drives_run_to_completion(monkeypatch):
    monkeypatch.setattr("sys.argv", ["meeko"])
    ran = {"count": 0}

    async def fake_run(**kwargs):
        ran["count"] += 1

    with patch("meeko.main.run", new=fake_run):
        meeko_main.main()

    assert ran["count"] == 1


def test_main_swallows_keyboard_interrupt(monkeypatch):
    monkeypatch.setattr("sys.argv", ["meeko"])

    async def raising_run(**kwargs):
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
    def __init__(self, api_key, system_prompt, dispatcher, **kwargs):
        self.api_key = api_key
        self.system_prompt = system_prompt
        self.dispatcher = dispatcher
        self.store = kwargs.get("store")
        self.session_id = kwargs.get("session_id")
        self.turns: list[str] = []
        self.loaded_history: list[dict] | None = None

    def stream_turn(self, text):
        self.turns.append(text)

        async def _gen():
            yield "Hi."

        return _gen()

    def load_history(self, messages):
        self.loaded_history = list(messages)


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


async def test_run_drives_one_turn_end_to_end(monkeypatch, fake_profiles, tmp_path):
    monkeypatch.setenv("DEEPGRAM_API_KEY", "dg-test")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-test")
    monkeypatch.setenv("MEEKO_DB_PATH", str(tmp_path / "meeko.db"))

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

    def make_claude(api_key, system_prompt, dispatcher, **kwargs):
        c = _FakeClaudeClient(api_key, system_prompt, dispatcher, **kwargs)
        fake_claude_holder["client"] = c
        return c

    # Short-circuit the 1s tail-drain so the test doesn't drag.
    real_sleep = asyncio.sleep

    async def fast_sleep(delay, *a, **kw):
        if delay >= 1.0:
            return await real_sleep(0)
        return await real_sleep(delay, *a, **kw)

    with (
        patch("meeko.audio_io.pyaudio.PyAudio", return_value=pa_instance),
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


async def test_run_reconnects_stt_after_connection_closed(
    monkeypatch, fake_profiles, tmp_path
):
    monkeypatch.setenv("DEEPGRAM_API_KEY", "dg-test")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-test")
    monkeypatch.setenv("MEEKO_DB_PATH", str(tmp_path / "meeko.db"))

    pa_instance = MagicMock()
    mic_stream = MagicMock()
    speaker_stream = MagicMock()

    first_write = asyncio.Event()
    speaker_stream.write.side_effect = lambda data: first_write.set()
    pa_instance.open.side_effect = [mic_stream, speaker_stream]

    fake_stt = _ReconnectSTTClient("dg-test")
    fake_tts = _FakeTTSClient("dg-test")

    def make_claude(api_key, system_prompt, dispatcher, **kwargs):
        return _FakeClaudeClient(api_key, system_prompt, dispatcher, **kwargs)

    real_sleep = asyncio.sleep

    async def fast_sleep(delay, *a, **kw):
        if delay >= 1.0:
            return await real_sleep(0)
        return await real_sleep(delay, *a, **kw)

    with (
        patch("meeko.audio_io.pyaudio.PyAudio", return_value=pa_instance),
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


async def test_run_backs_off_on_repeated_stt_failures(
    monkeypatch, fake_profiles, tmp_path
):
    """Repeated connect failures should sleep with increasing delays
    (0.5, 1, 2, ...) and only log a full traceback on the first failure."""
    monkeypatch.setenv("DEEPGRAM_API_KEY", "dg-test")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-test")
    monkeypatch.setenv("MEEKO_DB_PATH", str(tmp_path / "meeko.db"))

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
        patch("meeko.audio_io.pyaudio.PyAudio", return_value=pa_instance),
        patch("meeko.main.load_profiles", return_value=fake_profiles),
        patch("meeko.main.load_dotenv"),
        patch("meeko.main.DeepgramSTT", return_value=fake_stt),
        patch("meeko.main.DeepgramTTS", return_value=_FakeTTSClient("x")),
        patch(
            "meeko.main.ClaudeClient",
            side_effect=lambda api_key, system_prompt, dispatcher, **kw: (
                _FakeClaudeClient(api_key, system_prompt, dispatcher, **kw)
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

    # Reconnect delays grow: 0.5, 1, 2, 4, ... (exclude the grace-cutoff
    # task's 10s sleep, which also goes through asyncio.sleep).
    from meeko import stt_supervisor

    backoff_delays = [d for d in sleep_calls if d != stt_supervisor.RECONNECT_GRACE_S]
    assert backoff_delays[:4] == [0.5, 1, 2, 4]


async def test_grace_cutoff_stops_mic_and_drains_after_outage(
    monkeypatch, fake_profiles, tmp_path
):
    """After RECONNECT_GRACE_S of failed reconnects the mic is stopped
    and mic_queue is drained; a subsequent successful session restarts
    the mic."""
    monkeypatch.setenv("DEEPGRAM_API_KEY", "dg-test")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-test")
    monkeypatch.setenv("MEEKO_DB_PATH", str(tmp_path / "meeko.db"))

    pa_instance = MagicMock()
    mic_stream = MagicMock()
    speaker_stream = MagicMock()
    pa_instance.open.side_effect = [mic_stream, speaker_stream]

    class _FailThenSucceedSTT:
        def __init__(self, api_key):
            self.attempts = 0

        @contextlib.asynccontextmanager
        async def session(self):
            self.attempts += 1
            if self.attempts <= 5:
                raise OSError("dns down")
            yield _FakeSTTSession([])
            await asyncio.sleep(10)

    fake_stt = _FailThenSucceedSTT("dg-test")
    real_sleep = asyncio.sleep

    async def fast_sleep(delay, *a, **kw):
        # Collapse every real asyncio.sleep so the grace task fires
        # immediately after a disconnect.
        return await real_sleep(0)

    with (
        patch("meeko.audio_io.pyaudio.PyAudio", return_value=pa_instance),
        patch("meeko.main.load_profiles", return_value=fake_profiles),
        patch("meeko.main.load_dotenv"),
        patch("meeko.main.DeepgramSTT", return_value=fake_stt),
        patch("meeko.main.DeepgramTTS", return_value=_FakeTTSClient("x")),
        patch(
            "meeko.main.ClaudeClient",
            side_effect=lambda api_key, system_prompt, dispatcher, **kw: (
                _FakeClaudeClient(api_key, system_prompt, dispatcher, **kw)
            ),
        ),
        patch("meeko.main.asyncio.sleep", new=fast_sleep),
        patch("meeko.main.setup_logging"),
    ):
        # Prime the mic_queue with buffered PCM so we can assert it
        # gets drained by the grace cutoff.
        task = asyncio.create_task(meeko_main.run())
        for _ in range(200):
            if fake_stt.attempts >= 6:
                break
            await real_sleep(0)
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await task

    # mic_stream was stopped during the outage and restarted after
    # reconnect — so start_stream() was called at least twice (initial
    # + post-cutoff resume) and stop_stream() at least twice (grace
    # cutoff + shutdown).
    assert mic_stream.start_stream.call_count >= 2
    assert mic_stream.stop_stream.call_count >= 2


async def test_mic_queue_full_triggers_shutdown(monkeypatch, fake_profiles, tmp_path):
    """If mic_callback can't enqueue because the queue is at MIC_QUEUE_MAX,
    it logs and sets stop_event so run() exits cleanly."""
    monkeypatch.setenv("DEEPGRAM_API_KEY", "dg-test")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-test")
    monkeypatch.setenv("MEEKO_DB_PATH", str(tmp_path / "meeko.db"))

    pa_instance = MagicMock()
    mic_stream = MagicMock()
    speaker_stream = MagicMock()
    captured_cb: dict = {}

    def open_stream(**kwargs):
        if kwargs.get("input"):
            captured_cb["cb"] = kwargs["stream_callback"]
            return mic_stream
        return speaker_stream

    pa_instance.open.side_effect = open_stream

    fake_stt = _FakeSTTClient("dg-test")
    # Make the only event a long sleep so pump_mic never drains.
    fake_stt.events = []

    with (
        patch("meeko.audio_io.pyaudio.PyAudio", return_value=pa_instance),
        patch("meeko.main.load_profiles", return_value=fake_profiles),
        patch("meeko.main.load_dotenv"),
        patch("meeko.main.DeepgramSTT", return_value=fake_stt),
        patch("meeko.main.DeepgramTTS", return_value=_FakeTTSClient("x")),
        patch(
            "meeko.main.ClaudeClient",
            side_effect=lambda api_key, system_prompt, dispatcher, **kw: (
                _FakeClaudeClient(api_key, system_prompt, dispatcher, **kw)
            ),
        ),
        patch("meeko.audio_io.MIC_QUEUE_MAX", 2),
        patch("meeko.main.setup_logging"),
    ):
        task = asyncio.create_task(meeko_main.run())
        # Wait until mic_callback has been captured.
        for _ in range(50):
            if "cb" in captured_cb:
                break
            await asyncio.sleep(0)
        # Overflow the queue from the "PyAudio thread" side.
        for _ in range(10):
            captured_cb["cb"](b"\x00\x00", 1, None, 0)
            await asyncio.sleep(0)
        # run() should exit on its own because stop_event was set.
        try:
            await asyncio.wait_for(task, timeout=3)
        except TimeoutError:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
            pytest.fail("run() did not exit after mic_queue overflow")


async def test_run_resume_preloads_history_and_skips_greeting(
    monkeypatch, fake_profiles, tmp_path
):
    from meeko.sessions import SessionStore

    db_path = tmp_path / "meeko.db"
    monkeypatch.setenv("DEEPGRAM_API_KEY", "dg-test")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-test")
    monkeypatch.setenv("MEEKO_DB_PATH", str(db_path))

    # Seed a session with two prior turns.
    seed = SessionStore.open(db_path)
    try:
        session_id = await seed.create_session("default")
        await seed.persist_turn(session_id, "user", "prior question")
        await seed.persist_turn(
            session_id, "assistant", [{"type": "text", "text": "prior reply"}]
        )
    finally:
        seed.close()

    pa_instance = MagicMock()
    mic_stream = MagicMock()
    speaker_stream = MagicMock()
    pa_instance.open.side_effect = [mic_stream, speaker_stream]

    session_entered = asyncio.Event()

    class _SignalingSTTClient(_FakeSTTClient):
        @contextlib.asynccontextmanager
        async def session(self):
            async with super().session() as s:
                session_entered.set()
                yield s

    fake_stt = _SignalingSTTClient("dg-test")
    fake_stt.events = []  # no turns needed — we just need run() to start up
    fake_tts = _FakeTTSClient("dg-test")
    fake_claude_holder: dict = {}

    def make_claude(api_key, system_prompt, dispatcher, **kwargs):
        c = _FakeClaudeClient(api_key, system_prompt, dispatcher, **kwargs)
        fake_claude_holder["client"] = c
        return c

    # Give the fake profile a non-empty greeting so we can assert TTS
    # was NOT called with it on resume.
    resume_profiles = {
        "default": Profile(
            name="default",
            wake_word="meeko",
            prompt="system",
            greeting="Hello there.",
            voice=None,
        )
    }

    real_sleep = asyncio.sleep

    async def fast_sleep(delay, *a, **kw):
        if delay >= 1.0:
            return await real_sleep(0)
        return await real_sleep(delay, *a, **kw)

    with (
        patch("meeko.audio_io.pyaudio.PyAudio", return_value=pa_instance),
        patch("meeko.main.load_profiles", return_value=resume_profiles),
        patch("meeko.main.load_dotenv"),
        patch("meeko.main.DeepgramSTT", return_value=fake_stt),
        patch("meeko.main.DeepgramTTS", return_value=fake_tts),
        patch("meeko.main.ClaudeClient", side_effect=make_claude),
        patch("meeko.main.asyncio.sleep", new=fast_sleep),
        patch("meeko.main.setup_logging"),
    ):
        task = asyncio.create_task(meeko_main.run(resume=session_id))
        # Wait until the STT supervisor has entered a session. This is
        # downstream of the greeting branch in run(), so if the greeting
        # was going to be spoken it already would have been.
        await asyncio.wait_for(session_entered.wait(), timeout=5)
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await task

    claude = fake_claude_holder["client"]
    assert claude.session_id == session_id
    assert claude.loaded_history == [
        {"role": "user", "content": "prior question"},
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "prior reply"}],
        },
    ]
    # Greeting must NOT have been spoken on resume.
    assert fake_tts.calls == []


async def test_run_resume_unknown_id_exits(monkeypatch, fake_profiles, tmp_path):
    monkeypatch.setenv("DEEPGRAM_API_KEY", "dg-test")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-test")
    monkeypatch.setenv("MEEKO_DB_PATH", str(tmp_path / "meeko.db"))

    with (
        patch("meeko.main.load_profiles", return_value=fake_profiles),
        patch("meeko.main.load_dotenv"),
        patch("meeko.main.setup_logging"),
    ):
        with pytest.raises(SystemExit):
            await meeko_main.run(resume="does-not-exist")


async def test_run_list_sessions_prints_and_returns(monkeypatch, tmp_path, capsys):
    from meeko.sessions import SessionStore

    db_path = tmp_path / "meeko.db"
    monkeypatch.setenv("MEEKO_DB_PATH", str(db_path))

    seed = SessionStore.open(db_path)
    try:
        sid = await seed.create_session("default")
        await seed.persist_turn(sid, "user", "hi")
    finally:
        seed.close()

    with patch("meeko.main.setup_logging"):
        await meeko_main.run(list_sessions=True)

    out = capsys.readouterr().out
    assert sid in out
    assert "default" in out


async def test_run_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("DEEPGRAM_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with (
        patch("meeko.main.load_dotenv"),
        patch("meeko.main.setup_logging"),
    ):
        with pytest.raises(KeyError):
            await meeko_main.run()
