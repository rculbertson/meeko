"""Tests for the `meeko.main` orchestrator module.

Covers `setup_logging`, the `State` enum, the sync `main()` entrypoint,
and a happy-path drive-through of `run()` with all external services
(PyAudio, Deepgram STT/TTS, Anthropic) mocked.
"""

import asyncio
import contextlib
import json
import logging
import logging.handlers
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from meeko import main as meeko_main
from meeko.config import Profile
from meeko.main import State, setup_logging


@pytest.fixture
def clean_logger():
    logger = logging.getLogger("meeko")
    saved_handlers = logger.handlers[:]
    saved_level = logger.level
    logger.handlers = []
    yield logger
    logger.handlers = saved_handlers
    logger.setLevel(saved_level)


def test_setup_logging_defaults_to_stream_handler(clean_logger):
    setup_logging()
    assert len(clean_logger.handlers) == 1
    assert isinstance(clean_logger.handlers[0], logging.StreamHandler)
    assert clean_logger.level == logging.DEBUG


def test_setup_logging_file_target(monkeypatch, tmp_path, clean_logger):
    monkeypatch.chdir(tmp_path)
    setup_logging(log_level="INFO", log_target="file")
    assert isinstance(clean_logger.handlers[0], logging.handlers.RotatingFileHandler)
    assert clean_logger.level == logging.INFO
    # Close the file handler so tmp_path can be cleaned up on Windows/macOS.
    clean_logger.handlers[0].close()


def test_setup_logging_invalid_level_falls_back_to_debug(clean_logger):
    setup_logging(log_level="NOT_A_REAL_LEVEL")
    assert clean_logger.level == logging.DEBUG


def test_state_enum_members():
    assert {s.name for s in State} == {"IDLE", "LISTENING", "PROCESSING", "SPEAKING"}


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
        # Hold open so the session doesn't tear down once we run out of
        # canned events — pull_stt_events returning would otherwise let
        # on_session's FIRST_COMPLETED cancel an in-flight drive_turns
        # before its TTS chunk reaches the speaker. asyncio.sleep can't
        # be used here because tests patch meeko.main.asyncio.sleep
        # (which is the global asyncio.sleep) to short-circuit long
        # delays. An un-set Event isn't affected by that patch.
        await asyncio.Event().wait()


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
        self._create_session_fn = kwargs.get("create_session_fn")
        self.turns: list[str] = []
        self.loaded_history: list[dict] | None = None
        self.reset_count = 0  # incremented each time reset_session() is called

    def stream_turn(self, text):
        self.turns.append(text)
        store = self.store
        create_fn = self._create_session_fn

        async def _gen():
            # Mirror the real ClaudeClient's lazy session creation + persistence
            # so downstream consumers (end-of-session summarization) see a
            # turn log and tests cover the lazy-create path.
            if store is not None:
                if self.session_id is None and create_fn is not None:
                    self.session_id = await create_fn()
                if self.session_id is not None:
                    await store.persist_turn(self.session_id, "user", text)
            yield "Hi."
            if store is not None and self.session_id is not None:
                await store.persist_turn(
                    self.session_id, "assistant", [{"type": "text", "text": "Hi."}]
                )

        return _gen()

    def load_history(self, messages):
        self.loaded_history = list(messages)

    def reset_session(self):
        self.session_id = None
        self.loaded_history = None
        self.reset_count += 1

    def rebind_session(self, session_id):
        self.session_id = session_id

    def set_system_prompt(self, prompt):
        self.system_prompt = prompt


@pytest.fixture(autouse=True)
def _disable_wake_word_by_default(monkeypatch):
    """Existing orchestrator tests predate wake-word gating — default them
    into the legacy flow. Tests that exercise the wake-word path opt back
    in by deleting this env var in their own body."""
    monkeypatch.setenv("MEEKO_WAKE_WORD_DISABLED", "1")


class _StubAsyncAnthropic:
    """Stand-in for `anthropic.AsyncAnthropic` in orchestrator tests.

    Real construction trips on env-configured proxies in some dev
    environments; production also does real network I/O we don't want
    in unit tests. The default `messages.create` returns a parseable
    summary response so the background `summarize_session` task
    completes successfully — tests that want to assert on the payload
    can replace `_STUB_SUMMARY_TITLE` / `_STUB_SUMMARY_BODY` on the
    instance before triggering a session rotation."""

    _STUB_SUMMARY_TITLE = "stub title"
    _STUB_SUMMARY_BODY = "stub summary for tests"

    def __init__(self, *args, **kwargs):
        body = json.dumps(
            {
                "title": self._STUB_SUMMARY_TITLE,
                "summary": self._STUB_SUMMARY_BODY,
            }
        )
        response = SimpleNamespace(
            content=[SimpleNamespace(type="text", text=body)],
            stop_reason="end_turn",
        )
        self.messages = MagicMock()
        self.messages.create = AsyncMock(return_value=response)


@pytest.fixture(autouse=True)
def _stub_summary_anthropic_client():
    """Every orchestrator test constructs a summary `AsyncAnthropic`
    client at startup; stub it unless the test opts into a custom fake."""
    with patch("meeko.main.anthropic.AsyncAnthropic", _StubAsyncAnthropic):
        yield


@pytest.fixture
def fake_profiles():
    return {
        "query": Profile(
            name="query",
            wake_word="meeko",
            prompt="system",
            voice=None,
            # Disable the post-turn idle monitor: orchestrator tests
            # patch asyncio.sleep, which would otherwise short-circuit
            # the idle timer and trigger spurious session ends.
            idle_timeout_seconds=0,
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
        patch("meeko.main.load_profiles", return_value=(fake_profiles, "query")),
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
    # Speaker.write receives the TTS chunk duplicated into stereo
    # (L/R from the mono TTS sample).
    speaker_stream.write.assert_any_call(b"\x01\x02\x01\x02\x03\x04\x03\x04")
    # Cleanup was executed in run()'s finally block.
    mic_stream.stop_stream.assert_called()
    mic_stream.close.assert_called()
    speaker_stream.stop_stream.assert_called()
    speaker_stream.close.assert_called()
    pa_instance.terminate.assert_called()


async def test_mute_mic_while_speaking_drops_chunks_during_speaking(
    monkeypatch, fake_profiles, tmp_path
):
    """With MEEKO_MUTE_MIC_WHILE_SPEAKING=1, mic audio captured while
    the assistant is SPEAKING must not reach the STT session."""
    monkeypatch.setenv("DEEPGRAM_API_KEY", "dg-test")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-test")
    monkeypatch.setenv("MEEKO_DB_PATH", str(tmp_path / "meeko.db"))
    monkeypatch.setenv("MEEKO_MUTE_MIC_WHILE_SPEAKING", "1")

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

    first_speaker_write = asyncio.Event()
    release_tts = asyncio.Event()
    speaker_stream.write.side_effect = lambda data: first_speaker_write.set()

    fake_stt = _FakeSTTClient("dg-test")

    class _SlowTTS:
        """Yield one chunk, then block until the test releases us —
        keeps the state machine in SPEAKING while we push mic audio."""

        def __init__(self, api_key):
            self.calls: list[tuple[str, str]] = []

        async def stream(self, text, voice):
            self.calls.append((text, voice))
            yield b"\x01\x02\x03\x04"
            await release_tts.wait()

    fake_tts = _SlowTTS("dg-test")

    def make_claude(api_key, system_prompt, dispatcher, **kwargs):
        return _FakeClaudeClient(api_key, system_prompt, dispatcher, **kwargs)

    real_sleep = asyncio.sleep

    async def fast_sleep(delay, *a, **kw):
        if delay >= 1.0:
            return await real_sleep(0)
        return await real_sleep(delay, *a, **kw)

    with (
        patch("meeko.audio_io.pyaudio.PyAudio", return_value=pa_instance),
        patch("meeko.main.load_profiles", return_value=(fake_profiles, "query")),
        patch("meeko.main.load_dotenv"),
        patch("meeko.main.DeepgramSTT", return_value=fake_stt),
        patch("meeko.main.DeepgramTTS", return_value=fake_tts),
        patch("meeko.main.ClaudeClient", side_effect=make_claude),
        patch("meeko.main.asyncio.sleep", new=fast_sleep),
        patch("meeko.main.setup_logging"),
    ):
        task = asyncio.create_task(meeko_main.run())
        try:
            await asyncio.wait_for(first_speaker_write.wait(), timeout=5)
            # We're now inside SPEAKING. Snapshot STT send count, then
            # push mic chunks via the captured PyAudio callback and
            # give pump_mic a chance to see them.
            session = fake_stt.session_obj
            baseline = session.sent_audio_count
            # 4 bytes = one stereo int16 frame, matches audio_io callback.
            for _ in range(5):
                captured_cb["cb"](b"\x00\x00\x00\x00", 1, None, 0)
            for _ in range(50):
                await real_sleep(0)
            assert session.sent_audio_count == baseline
        finally:
            release_tts.set()
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task


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
        # See _FakeSTTSession.events() for why this isn't asyncio.sleep.
        await asyncio.Event().wait()


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
        patch("meeko.main.load_profiles", return_value=(fake_profiles, "query")),
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
        patch("meeko.main.load_profiles", return_value=(fake_profiles, "query")),
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
        # Let the loop churn through several reconnect attempts. Use a
        # small real sleep (not sleep(0)) so a slow CI runner gets enough
        # scheduling time to advance run() past the retry branch.
        for _ in range(200):
            if fake_stt.attempts >= 4:
                break
            await real_sleep(0.01)
        assert fake_stt.attempts >= 4, (
            f"run() did not reach 4 reconnect attempts (got {fake_stt.attempts})"
        )
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
        patch("meeko.main.load_profiles", return_value=(fake_profiles, "query")),
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
            await real_sleep(0.01)
        assert fake_stt.attempts >= 6, (
            f"run() did not reach 6 reconnect attempts (got {fake_stt.attempts})"
        )
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
        patch("meeko.main.load_profiles", return_value=(fake_profiles, "query")),
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
        # Wait until mic_callback has been captured. Use a small real
        # sleep (not sleep(0)) so a slow CI runner gets enough
        # scheduling time to actually open the mic stream.
        for _ in range(200):
            if "cb" in captured_cb:
                break
            await asyncio.sleep(0.01)
        assert "cb" in captured_cb, "mic stream was never opened"
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


async def test_run_resume_preloads_history(monkeypatch, fake_profiles, tmp_path):
    from meeko.sessions import SessionStore

    db_path = tmp_path / "meeko.db"
    monkeypatch.setenv("DEEPGRAM_API_KEY", "dg-test")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-test")
    monkeypatch.setenv("MEEKO_DB_PATH", str(db_path))

    # Seed a session with two prior turns.
    seed = SessionStore.open(db_path)
    try:
        session_id = await seed.create_session("query")
        await seed.persist_turn(session_id, "user", "prior question")
        await seed.persist_turn(
            session_id, "assistant", [{"type": "text", "text": "prior reply"}]
        )
    finally:
        await seed.close()

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

    resume_profiles = {
        "query": Profile(
            name="query",
            wake_word="meeko",
            prompt="system",
            voice=None,
            idle_timeout_seconds=0,
        )
    }

    real_sleep = asyncio.sleep

    async def fast_sleep(delay, *a, **kw):
        if delay >= 1.0:
            return await real_sleep(0)
        return await real_sleep(delay, *a, **kw)

    with (
        patch("meeko.audio_io.pyaudio.PyAudio", return_value=pa_instance),
        patch("meeko.main.load_profiles", return_value=(resume_profiles, "query")),
        patch("meeko.main.load_dotenv"),
        patch("meeko.main.DeepgramSTT", return_value=fake_stt),
        patch("meeko.main.DeepgramTTS", return_value=fake_tts),
        patch("meeko.main.ClaudeClient", side_effect=make_claude),
        patch("meeko.main.asyncio.sleep", new=fast_sleep),
        patch("meeko.main.setup_logging"),
    ):
        task = asyncio.create_task(meeko_main.run(resume=session_id))
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
    assert fake_tts.calls == []


async def test_run_resume_unknown_id_exits(monkeypatch, fake_profiles, tmp_path):
    monkeypatch.setenv("DEEPGRAM_API_KEY", "dg-test")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-test")
    monkeypatch.setenv("MEEKO_DB_PATH", str(tmp_path / "meeko.db"))

    with (
        patch("meeko.main.load_profiles", return_value=(fake_profiles, "query")),
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
        sid = await seed.create_session("query")
        await seed.persist_turn(sid, "user", "hi")
    finally:
        await seed.close()

    with patch("meeko.main.setup_logging"):
        await meeko_main.run(list_sessions=True)

    out = capsys.readouterr().out
    assert sid in out
    assert "query" in out


async def test_run_backfills_untitled_sessions_on_startup(
    monkeypatch, fake_profiles, tmp_path
):
    """A session left untitled-but-non-empty by a prior killed run should
    be summarized at the next startup, so `list_sessions` stops reading
    it back as '(no title)'."""
    from meeko.sessions import SessionStore

    db_path = tmp_path / "meeko.db"
    monkeypatch.setenv("DEEPGRAM_API_KEY", "dg-test")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-test")
    monkeypatch.setenv("MEEKO_DB_PATH", str(db_path))

    seed = SessionStore.open(db_path)
    try:
        untitled_sid = await seed.create_session("query")
        await seed.persist_turn(untitled_sid, "user", "leftover question")
        await seed.persist_turn(
            untitled_sid,
            "assistant",
            [{"type": "text", "text": "leftover answer"}],
        )
    finally:
        await seed.close()

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
    fake_stt.events = []
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
        patch("meeko.main.load_profiles", return_value=(fake_profiles, "query")),
        patch("meeko.main.load_dotenv"),
        patch("meeko.main.DeepgramSTT", return_value=fake_stt),
        patch("meeko.main.DeepgramTTS", return_value=fake_tts),
        patch("meeko.main.ClaudeClient", side_effect=make_claude),
        patch("meeko.main.asyncio.sleep", new=fast_sleep),
        patch("meeko.main.setup_logging"),
    ):
        task = asyncio.create_task(meeko_main.run())
        try:
            await asyncio.wait_for(session_entered.wait(), timeout=5)
            # Backfill is fire-and-forget — poll the DB until the title
            # appears or we give up.
            for _ in range(200):
                check = SessionStore.open(db_path)
                try:
                    row = await check.get_session(untitled_sid)
                finally:
                    await check.close()
                if row is not None and row.get("title") is not None:
                    break
                await real_sleep(0.01)
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task

    store = SessionStore.open(db_path)
    try:
        row = await store.get_session(untitled_sid)
    finally:
        await store.close()
    assert row is not None
    # _StubAsyncAnthropic returns "stub title" for any summary call.
    assert row["title"] == "stub title"


async def test_run_gates_stt_on_wake_word(monkeypatch, fake_profiles, tmp_path):
    """With wake word enabled the orchestrator starts in IDLE and holds
    audio back from STT until the detector fires. After the detector
    fires, mic audio flows to STT (so a question spoken right after
    the wake word is transcribed) and no greeting TTS is emitted."""
    monkeypatch.delenv("MEEKO_WAKE_WORD_DISABLED", raising=False)
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
    speaker_stream.write.side_effect = lambda data: None

    wake_profiles = {
        "query": Profile(
            name="query",
            wake_word="meeko",
            prompt="system",
            voice=None,
            idle_timeout_seconds=0,
        )
    }

    # Fake detector: records every process() call. The first call also
    # captures the idle-state snapshot (STT count, TTS calls) so the
    # test can assert those on the pre-wake path without racing the
    # event loop. The second call fires the wake word.
    class _FakeDetector:
        def __init__(self, *a, **kw):
            self.calls = 0
            self.fired = asyncio.Event()
            self.first_call_snapshot: dict | None = None

        def process(self, pcm: bytes) -> bool:
            self.calls += 1
            if self.calls == 1:
                self.first_call_snapshot = {
                    "stt": fake_stt.session_obj.sent_audio_count,
                    "tts": list(fake_tts.calls),
                }
                return False
            self.fired.set()
            return True

    detector_holder: dict = {}

    def make_detector(*a, **kw):
        d = _FakeDetector()
        detector_holder["d"] = d
        return d

    # Use a holding STT client so the session stays open while the test
    # pushes post-wake chunks — otherwise the empty events list combined
    # with fast_sleep-patched asyncio.sleep causes the supervisor to
    # churn through reconnects and drop the mic chunks mid-cycle.
    fake_stt = _HoldingSTTClient("dg-test")
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
        patch("meeko.main.load_profiles", return_value=(wake_profiles, "query")),
        patch("meeko.main.load_dotenv"),
        patch("meeko.main.DeepgramSTT", return_value=fake_stt),
        patch("meeko.main.DeepgramTTS", return_value=fake_tts),
        patch("meeko.main.ClaudeClient", side_effect=make_claude),
        patch("meeko.main.WakeWordDetector", side_effect=make_detector),
        patch("meeko.main.asyncio.sleep", new=fast_sleep),
        patch("meeko.main.setup_logging"),
    ):
        task = asyncio.create_task(meeko_main.run())
        try:
            # Wait for detector + mic callback + STT session to be live.
            for _ in range(500):
                if (
                    "cb" in captured_cb
                    and "d" in detector_holder
                    and fake_stt.session_obj is not None
                ):
                    break
                await real_sleep(0.01)
            assert "cb" in captured_cb
            assert "d" in detector_holder
            detector = detector_holder["d"]

            # Push chunks until the detector fires on the second call.
            # A small amount of real time keeps pump_mic ticking under
            # load.
            for _ in range(20):
                if detector.fired.is_set():
                    break
                captured_cb["cb"](b"\x00\x00\x00\x00", 1, None, 0)
                await real_sleep(0.02)

            await asyncio.wait_for(detector.fired.wait(), timeout=5)

            # First call happened while IDLE — snapshot proves STT and
            # TTS were both untouched up to that point.
            assert detector.first_call_snapshot == {"stt": 0, "tts": []}

            # After the wake word fires, mic chunks should flow through
            # to STT so a question spoken right after "Hey Meeko" gets
            # transcribed. No greeting TTS should be emitted.
            for _ in range(50):
                captured_cb["cb"](b"\x00\x00\x00\x00", 1, None, 0)
                await real_sleep(0.02)
                if fake_stt.session_obj.sent_audio_count > 0:
                    break
            assert fake_stt.session_obj.sent_audio_count > 0
            assert fake_tts.calls == []
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task


class _SessionToolClaudeClient(_FakeClaudeClient):
    """Simulates Sonnet calling a session-management tool mid-turn: yields a
    brief acknowledgement AND dispatches the named tool. Mirrors the real
    flow where the assistant acknowledges verbally before the tool fires.

    Subclasses override ``TOOL_NAME`` / ``ACK`` to pick which tool to call.
    """

    TOOL_NAME = "end_session"
    ACK = "Goodnight!"

    def stream_turn(self, text):
        self.turns.append(text)
        dispatcher = self.dispatcher
        tool_name = self.TOOL_NAME
        ack = self.ACK
        store = self.store
        create_fn = self._create_session_fn

        async def _gen():
            if store is not None:
                if self.session_id is None and create_fn is not None:
                    self.session_id = await create_fn()
                if self.session_id is not None:
                    await store.persist_turn(self.session_id, "user", text)
            yield ack
            if store is not None and self.session_id is not None:
                await store.persist_turn(
                    self.session_id, "assistant", [{"type": "text", "text": ack}]
                )
            await dispatcher.dispatch(tool_name, {})

        return _gen()


class _EndSessionClaudeClient(_SessionToolClaudeClient):
    TOOL_NAME = "end_session"
    ACK = "Goodnight!"


class _NewSessionClaudeClient(_SessionToolClaudeClient):
    TOOL_NAME = "new_session"
    ACK = "Starting fresh."


class _SwitchThenNewSessionClaudeClient(_FakeClaudeClient):
    """Simulates Sonnet dispatching `switch_profile` and then
    `new_session` in a single turn. Used to verify the fresh SQLite row
    records the switched profile, not the original one."""

    SWITCH_TO = "pirate"

    def stream_turn(self, text):
        self.turns.append(text)
        dispatcher = self.dispatcher
        switch_to = self.SWITCH_TO
        store = self.store
        create_fn = self._create_session_fn

        async def _gen():
            if store is not None and self.session_id is None and create_fn is not None:
                self.session_id = await create_fn()
            if store is not None and self.session_id is not None:
                await store.persist_turn(self.session_id, "user", text)
            yield "Okay, switching and starting fresh."
            await dispatcher.dispatch("switch_profile", {"profile_name": switch_to})
            await dispatcher.dispatch("new_session", {})

        return _gen()


class _HoldingSTTClient:
    """Holds a single EndOfTurn until a `ready` event is set, then blocks
    the events() generator until the test cancels the task. Prevents
    spurious reconnects from replaying the same event when fast_sleep
    collapses the normal 10s hold; the `ready` gate lets the test fire
    the EndOfTurn only after wake-word detection has transitioned the
    orchestrator out of IDLE."""

    def __init__(self, api_key):
        self.session_obj = None
        self.transcript = "stop"
        self.ready = asyncio.Event()

    @contextlib.asynccontextmanager
    async def session(self):
        self.session_obj = _HoldingSTTSession(self.transcript, self.ready)
        try:
            yield self.session_obj
        finally:
            self.session_obj.release.set()


class _HoldingSTTSession:
    def __init__(self, transcript, ready):
        self._transcript = transcript
        self._ready = ready
        self.sent_audio_count = 0
        self.keepalive_count = 0
        self.release = asyncio.Event()

    async def send_audio(self, pcm):
        self.sent_audio_count += 1

    async def send_keepalive(self):
        self.keepalive_count += 1

    async def events(self):
        await self._ready.wait()
        yield SimpleNamespace(event="EndOfTurn", transcript=self._transcript)
        # Hold indefinitely; released when the surrounding session
        # context manager exits (i.e. the test cancels the task).
        await self.release.wait()


async def test_end_session_tool_returns_to_idle_with_fresh_session(
    monkeypatch, fake_profiles, tmp_path
):
    """User says 'stop' → fake Claude dispatches end_session → after
    SPEAKING completes, state returns to IDLE, a fresh SQLite session is
    created, and the Claude client's in-memory history is dropped."""
    from meeko.sessions import SessionStore

    db_path = tmp_path / "meeko.db"
    monkeypatch.delenv("MEEKO_WAKE_WORD_DISABLED", raising=False)
    monkeypatch.setenv("DEEPGRAM_API_KEY", "dg-test")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-test")
    monkeypatch.setenv("MEEKO_DB_PATH", str(db_path))

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
    speaker_stream.write.side_effect = lambda data: None

    wake_profiles = {
        "query": Profile(
            name="query",
            wake_word="meeko",
            prompt="system",
            voice=None,
            idle_timeout_seconds=0,
        )
    }

    # Wake detector fires on the first process() call so we get into
    # LISTENING quickly, then never looked at again until we re-enter
    # IDLE after end_session.
    class _FakeDetector:
        def __init__(self, *a, **kw):
            self.calls = 0
            self.resets = 0

        def process(self, pcm: bytes) -> bool:
            self.calls += 1
            return True  # fire on first chunk

        def reset(self) -> None:
            self.resets += 1

    detector_holder: dict = {}

    def make_detector(*a, **kw):
        d = _FakeDetector()
        detector_holder["d"] = d
        return d

    fake_stt = _HoldingSTTClient("dg-test")
    fake_stt.transcript = "Meeko stop"
    fake_tts = _FakeTTSClient("dg-test")
    fake_claude_holder: dict = {}

    def make_claude(api_key, system_prompt, dispatcher, **kwargs):
        c = _EndSessionClaudeClient(api_key, system_prompt, dispatcher, **kwargs)
        fake_claude_holder["client"] = c
        return c

    real_sleep = asyncio.sleep

    async def fast_sleep(delay, *a, **kw):
        if delay >= 1.0:
            return await real_sleep(0)
        return await real_sleep(delay, *a, **kw)

    with (
        patch("meeko.audio_io.pyaudio.PyAudio", return_value=pa_instance),
        patch("meeko.main.load_profiles", return_value=(wake_profiles, "query")),
        patch("meeko.main.load_dotenv"),
        patch("meeko.main.DeepgramSTT", return_value=fake_stt),
        patch("meeko.main.DeepgramTTS", return_value=fake_tts),
        patch("meeko.main.ClaudeClient", side_effect=make_claude),
        patch("meeko.main.WakeWordDetector", side_effect=make_detector),
        patch("meeko.main.asyncio.sleep", new=fast_sleep),
        patch("meeko.main.setup_logging"),
    ):
        task = asyncio.create_task(meeko_main.run())
        try:
            # Drive: first mic chunk fires the wake detector → LISTENING;
            # EndOfTurn("Meeko stop") fires end_session; we then wait for
            # the detector to be reset (our signal that the IDLE
            # transition ran).
            for _ in range(500):
                if "cb" in captured_cb and "d" in detector_holder:
                    break
                await real_sleep(0.01)
            assert "cb" in captured_cb
            # Push a mic frame so the wake detector fires → state
            # transitions to LISTENING. Only then do we release the
            # EndOfTurn from the fake STT; otherwise it'd arrive while
            # still in IDLE and get discarded by the safety belt.
            captured_cb["cb"](b"\x00\x00\x00\x00", 1, None, 0)
            for _ in range(200):
                if detector_holder["d"].calls >= 1:
                    break
                await real_sleep(0.01)
            fake_stt.ready.set()

            detector = detector_holder["d"]
            for _ in range(500):
                if detector.resets >= 1:
                    break
                await real_sleep(0.01)
            assert detector.resets >= 1, "wake detector was never reset"
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task

    # Claude fake got the user's stop message and its session_id was
    # cleared by reset_session (lazy creation: the next user turn would
    # create a fresh row; none happens in this test).
    claude = fake_claude_holder["client"]
    assert claude.turns == ["Meeko stop"]
    assert claude.session_id is None

    # Only the original session row exists — the one the user spoke into.
    # No empty stub for the "fresh" post-rotation session under lazy
    # creation.
    store = SessionStore.open(db_path)
    try:
        rows = await store.list_sessions()
    finally:
        await store.close()
    assert len(rows) == 1
    assert rows[0]["turn_count"] >= 1


async def test_end_session_without_wake_word_transitions_to_listening(
    monkeypatch, fake_profiles, tmp_path
):
    """MEEKO_WAKE_WORD_DISABLED=1: end_session still resets the session
    but state transitions to LISTENING (no wake gate to re-arm)."""
    from meeko.sessions import SessionStore

    db_path = tmp_path / "meeko.db"
    monkeypatch.setenv("DEEPGRAM_API_KEY", "dg-test")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-test")
    monkeypatch.setenv("MEEKO_DB_PATH", str(db_path))
    # Autouse fixture already sets MEEKO_WAKE_WORD_DISABLED=1.

    pa_instance = MagicMock()
    mic_stream = MagicMock()
    speaker_stream = MagicMock()
    pa_instance.open.side_effect = [mic_stream, speaker_stream]
    speaker_stream.write.side_effect = lambda data: None

    fake_stt = _HoldingSTTClient("dg-test")
    fake_stt.transcript = "stop"
    fake_tts = _FakeTTSClient("dg-test")
    fake_claude_holder: dict = {}

    def make_claude(api_key, system_prompt, dispatcher, **kwargs):
        c = _EndSessionClaudeClient(api_key, system_prompt, dispatcher, **kwargs)
        fake_claude_holder["client"] = c
        return c

    real_sleep = asyncio.sleep

    async def fast_sleep(delay, *a, **kw):
        if delay >= 1.0:
            return await real_sleep(0)
        return await real_sleep(delay, *a, **kw)

    with (
        patch("meeko.audio_io.pyaudio.PyAudio", return_value=pa_instance),
        patch("meeko.main.load_profiles", return_value=(fake_profiles, "query")),
        patch("meeko.main.load_dotenv"),
        patch("meeko.main.DeepgramSTT", return_value=fake_stt),
        patch("meeko.main.DeepgramTTS", return_value=fake_tts),
        patch("meeko.main.ClaudeClient", side_effect=make_claude),
        patch("meeko.main.asyncio.sleep", new=fast_sleep),
        patch("meeko.main.setup_logging"),
    ):
        task = asyncio.create_task(meeko_main.run())
        try:
            # Wake word disabled → state starts in LISTENING, so we can
            # release the EndOfTurn immediately.
            for _ in range(500):
                if fake_stt.session_obj is not None:
                    break
                await real_sleep(0.01)
            fake_stt.ready.set()

            # Wait for end_session to run: stable signal is reset_count
            # incrementing, which happens after the turn completes.
            claude = None
            for _ in range(500):
                c = fake_claude_holder.get("client")
                if c is not None and c.reset_count >= 1:
                    claude = c
                    break
                await real_sleep(0.01)
            assert claude is not None, "end_session never called reset_session"
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task

    store = SessionStore.open(db_path)
    try:
        rows = await store.list_sessions()
    finally:
        await store.close()
    # Only the original (user-spoken) session row exists. Lazy creation
    # means the post-rotation session has no row until the next user turn.
    assert len(rows) == 1


async def test_new_session_tool_rotates_session_and_stays_listening(
    monkeypatch, fake_profiles, tmp_path
):
    """User says 'let's start fresh' → fake Claude dispatches new_session →
    after SPEAKING completes, a fresh SQLite session is created, claude
    client rebinds to it, and state stays in LISTENING (wake detector is
    NOT reset — Meeko keeps listening without requiring the wake word)."""
    from meeko.sessions import SessionStore

    db_path = tmp_path / "meeko.db"
    monkeypatch.delenv("MEEKO_WAKE_WORD_DISABLED", raising=False)
    monkeypatch.setenv("DEEPGRAM_API_KEY", "dg-test")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-test")
    monkeypatch.setenv("MEEKO_DB_PATH", str(db_path))

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
    speaker_stream.write.side_effect = lambda data: None

    wake_profiles = {
        "query": Profile(
            name="query",
            wake_word="meeko",
            prompt="system",
            voice=None,
            idle_timeout_seconds=0,
        )
    }

    class _FakeDetector:
        def __init__(self, *a, **kw):
            self.calls = 0
            self.resets = 0

        def process(self, pcm: bytes) -> bool:
            self.calls += 1
            return True  # fire on first chunk

        def reset(self) -> None:
            self.resets += 1

    detector_holder: dict = {}

    def make_detector(*a, **kw):
        d = _FakeDetector()
        detector_holder["d"] = d
        return d

    fake_stt = _HoldingSTTClient("dg-test")
    fake_stt.transcript = "let's start fresh"
    fake_tts = _FakeTTSClient("dg-test")
    fake_claude_holder: dict = {}

    def make_claude(api_key, system_prompt, dispatcher, **kwargs):
        c = _NewSessionClaudeClient(api_key, system_prompt, dispatcher, **kwargs)
        fake_claude_holder["client"] = c
        return c

    real_sleep = asyncio.sleep

    async def fast_sleep(delay, *a, **kw):
        if delay >= 1.0:
            return await real_sleep(0)
        return await real_sleep(delay, *a, **kw)

    with (
        patch("meeko.audio_io.pyaudio.PyAudio", return_value=pa_instance),
        patch("meeko.main.load_profiles", return_value=(wake_profiles, "query")),
        patch("meeko.main.load_dotenv"),
        patch("meeko.main.DeepgramSTT", return_value=fake_stt),
        patch("meeko.main.DeepgramTTS", return_value=fake_tts),
        patch("meeko.main.ClaudeClient", side_effect=make_claude),
        patch("meeko.main.WakeWordDetector", side_effect=make_detector),
        patch("meeko.main.asyncio.sleep", new=fast_sleep),
        patch("meeko.main.setup_logging"),
    ):
        task = asyncio.create_task(meeko_main.run())
        try:
            for _ in range(500):
                if "cb" in captured_cb and "d" in detector_holder:
                    break
                await real_sleep(0.01)
            assert "cb" in captured_cb
            captured_cb["cb"](b"\x00\x00\x00\x00", 1, None, 0)
            for _ in range(200):
                if detector_holder["d"].calls >= 1:
                    break
                await real_sleep(0.01)
            fake_stt.ready.set()

            # Wait for new_session to run: stable signal is reset_count
            # incrementing after the turn completes.
            claude = None
            for _ in range(500):
                c = fake_claude_holder.get("client")
                if c is not None and c.reset_count >= 1:
                    claude = c
                    break
                await real_sleep(0.01)
            assert claude is not None, "new_session never called reset_session"
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task

    # Wake detector must NOT have been reset — new_session stays in
    # LISTENING, not IDLE.
    assert detector_holder["d"].resets == 0

    # Only the original (user-spoken) session row exists. Lazy creation
    # means the rotated session has no row until the next user turn.
    store = SessionStore.open(db_path)
    try:
        rows = await store.list_sessions()
    finally:
        await store.close()
    assert len(rows) == 1
    assert claude.session_id is None


async def test_end_session_fires_background_summary_for_finalized_session(
    monkeypatch, fake_profiles, tmp_path
):
    """After end_session rotates the session, the background summary
    task should run and write title/summary/FTS for the *finalized*
    session (not the freshly-created one)."""
    from meeko.sessions import SessionStore

    db_path = tmp_path / "meeko.db"
    monkeypatch.setenv("DEEPGRAM_API_KEY", "dg-test")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-test")
    monkeypatch.setenv("MEEKO_DB_PATH", str(db_path))
    # Autouse fixture already sets MEEKO_WAKE_WORD_DISABLED=1.

    pa_instance = MagicMock()
    mic_stream = MagicMock()
    speaker_stream = MagicMock()
    pa_instance.open.side_effect = [mic_stream, speaker_stream]
    speaker_stream.write.side_effect = lambda data: None

    fake_stt = _HoldingSTTClient("dg-test")
    fake_stt.transcript = "stop"
    fake_tts = _FakeTTSClient("dg-test")
    fake_claude_holder: dict = {}

    def make_claude(api_key, system_prompt, dispatcher, **kwargs):
        c = _EndSessionClaudeClient(api_key, system_prompt, dispatcher, **kwargs)
        fake_claude_holder["client"] = c
        return c

    real_sleep = asyncio.sleep

    async def fast_sleep(delay, *a, **kw):
        if delay >= 1.0:
            return await real_sleep(0)
        return await real_sleep(delay, *a, **kw)

    with (
        patch("meeko.audio_io.pyaudio.PyAudio", return_value=pa_instance),
        patch("meeko.main.load_profiles", return_value=(fake_profiles, "query")),
        patch("meeko.main.load_dotenv"),
        patch("meeko.main.DeepgramSTT", return_value=fake_stt),
        patch("meeko.main.DeepgramTTS", return_value=fake_tts),
        patch("meeko.main.ClaudeClient", side_effect=make_claude),
        patch("meeko.main.asyncio.sleep", new=fast_sleep),
        patch("meeko.main.setup_logging"),
    ):
        task = asyncio.create_task(meeko_main.run())
        try:
            for _ in range(500):
                if fake_stt.session_obj is not None:
                    break
                await real_sleep(0.01)
            fake_stt.ready.set()

            # Wait for end_session to run via the stable reset_count signal,
            # then read the finalized session id from the DB (lazy creation
            # means it's the only row that was written).
            for _ in range(500):
                c = fake_claude_holder.get("client")
                if c is not None and c.reset_count >= 1:
                    break
                await real_sleep(0.01)
            assert c.reset_count >= 1, "end_session never called reset_session"

            # The finalized row is the only session in the DB at this point.
            check = SessionStore.open(db_path)
            try:
                rows = await check.list_sessions()
            finally:
                await check.close()
            assert len(rows) == 1, "expected exactly one finalized session row"
            original_sid = rows[0]["id"]

            # Give the detached summary task a chance to complete.
            # The background summarize_session call awaits the stubbed
            # anthropic client (immediate) and then writes SQLite.
            for _ in range(200):
                store_check = SessionStore.open(db_path)
                try:
                    row = store_check._conn.execute(
                        "SELECT title, summary FROM sessions WHERE id = ?",
                        (original_sid,),
                    ).fetchone()
                finally:
                    await store_check.close()
                if row is not None and row[0] is not None:
                    break
                await real_sleep(0.01)
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task

    store = SessionStore.open(db_path)
    try:
        row = store._conn.execute(
            "SELECT title, summary FROM sessions WHERE id = ?", (original_sid,)
        ).fetchone()
        fts = store._conn.execute(
            "SELECT session_id FROM sessions_fts WHERE sessions_fts MATCH ?",
            ("stub",),
        ).fetchall()
    finally:
        await store.close()
    assert row == ("stub title", "stub summary for tests")
    # FTS row exists and matches the finalized session.
    assert [r[0] for r in fts] == [original_sid]


async def test_new_session_after_profile_switch_records_active_profile(
    monkeypatch, tmp_path
):
    """If Sonnet switches profiles and then calls new_session in the same
    turn, the fresh SQLite row must carry the switched profile — not the
    one active at startup. This is the regression guard for using
    `profile_manager.active_profile.name` in the post-SPEAKING hook."""
    from meeko.sessions import SessionStore

    db_path = tmp_path / "meeko.db"
    monkeypatch.setenv("DEEPGRAM_API_KEY", "dg-test")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-test")
    monkeypatch.setenv("MEEKO_DB_PATH", str(db_path))
    # Autouse fixture already sets MEEKO_WAKE_WORD_DISABLED=1.

    pa_instance = MagicMock()
    mic_stream = MagicMock()
    speaker_stream = MagicMock()
    pa_instance.open.side_effect = [mic_stream, speaker_stream]
    speaker_stream.write.side_effect = lambda data: None

    two_profiles = {
        "query": Profile(
            name="query",
            wake_word="meeko",
            prompt="default system",
            voice=None,
            idle_timeout_seconds=0,
        ),
        "pirate": Profile(
            name="pirate",
            wake_word="meeko",
            prompt="pirate system",
            voice=None,
            idle_timeout_seconds=0,
        ),
    }

    fake_stt = _HoldingSTTClient("dg-test")
    fake_stt.transcript = "switch to pirate and start fresh"
    fake_tts = _FakeTTSClient("dg-test")
    fake_claude_holder: dict = {}

    def make_claude(api_key, system_prompt, dispatcher, **kwargs):
        c = _SwitchThenNewSessionClaudeClient(
            api_key, system_prompt, dispatcher, **kwargs
        )
        fake_claude_holder["client"] = c
        return c

    real_sleep = asyncio.sleep

    async def fast_sleep(delay, *a, **kw):
        if delay >= 1.0:
            return await real_sleep(0)
        return await real_sleep(delay, *a, **kw)

    with (
        patch("meeko.audio_io.pyaudio.PyAudio", return_value=pa_instance),
        patch("meeko.main.load_profiles", return_value=(two_profiles, "query")),
        patch("meeko.main.load_dotenv"),
        patch("meeko.main.DeepgramSTT", return_value=fake_stt),
        patch("meeko.main.DeepgramTTS", return_value=fake_tts),
        patch("meeko.main.ClaudeClient", side_effect=make_claude),
        patch("meeko.main.asyncio.sleep", new=fast_sleep),
        patch("meeko.main.setup_logging"),
    ):
        task = asyncio.create_task(meeko_main.run())
        try:
            for _ in range(500):
                if fake_stt.session_obj is not None:
                    break
                await real_sleep(0.01)
            fake_stt.ready.set()

            # Wait for new_session to run via the stable reset_count signal.
            claude = None
            for _ in range(500):
                c = fake_claude_holder.get("client")
                if c is not None and c.reset_count >= 1:
                    claude = c
                    break
                await real_sleep(0.01)
            assert claude is not None, "new_session never called reset_session"

            # Drive a second lazy creation by invoking the create_session_fn
            # the orchestrator wired up. After switch_profile + new_session,
            # the next user-turn's lazy create must use the *switched*
            # profile.
            new_sid = await claude._create_session_fn()
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task

    store = SessionStore.open(db_path)
    try:
        row = await store.get_session(new_sid)
    finally:
        await store.close()
    assert row is not None
    assert row["profile_name"] == "pirate", (
        f"fresh session recorded stale profile {row['profile_name']!r}; "
        "expected 'pirate' after switch_profile"
    )


class _EndSessionNoTurnClaudeClient(_FakeClaudeClient):
    """Dispatches end_session without creating a session row first.

    Simulates Sonnet ending the session before any user turn is persisted
    (lazy creation never fires). Used to verify fire_summary no-ops on None."""

    def stream_turn(self, text):
        self.turns.append(text)
        dispatcher = self.dispatcher

        async def _gen():
            yield "Goodbye."
            await dispatcher.dispatch("end_session", {})

        return _gen()


class _LoadSessionWithTurnClaudeClient(_FakeClaudeClient):
    """Lazily creates a session, persists a turn, then dispatches load_session.

    Ensures session_id is non-None when the post-turn hook runs so the
    `if session_id is not None: logger.info(...)` branch in
    _apply_post_turn_session_change is exercised."""

    def __init__(self, *args, target_session_id="", **kwargs):
        super().__init__(*args, **kwargs)
        self.target_session_id = target_session_id

    def stream_turn(self, text):
        self.turns.append(text)
        dispatcher = self.dispatcher
        target_id = self.target_session_id
        store = self.store
        create_fn = self._create_session_fn

        async def _gen():
            if store is not None and self.session_id is None and create_fn is not None:
                self.session_id = await create_fn()
            if store is not None and self.session_id is not None:
                await store.persist_turn(self.session_id, "user", text)
            yield "Let me pull that up."
            if store is not None and self.session_id is not None:
                await store.persist_turn(
                    self.session_id,
                    "assistant",
                    [{"type": "text", "text": "Let me pull that up."}],
                )
            await dispatcher.dispatch("load_session", {"id": target_id})

        return _gen()


class _LoadSessionClaudeClient(_SessionToolClaudeClient):
    """Simulates Sonnet calling load_session with a specific target id.

    The target session_id must be set on the instance before use."""

    TOOL_NAME = "load_session"
    ACK = "Picking up where we left off."

    def __init__(self, *args, target_session_id="", **kwargs):
        super().__init__(*args, **kwargs)
        self.target_session_id = target_session_id

    def stream_turn(self, text):
        self.turns.append(text)
        dispatcher = self.dispatcher
        target_id = self.target_session_id
        ack = self.ACK
        store = self.store
        sid = self.session_id

        async def _gen():
            if store is not None and sid is not None:
                await store.persist_turn(sid, "user", text)
            yield ack
            if store is not None and sid is not None:
                await store.persist_turn(
                    sid, "assistant", [{"type": "text", "text": ack}]
                )
            await dispatcher.dispatch("load_session", {"id": target_id})

        return _gen()


class _EndThenLoadSessionClaudeClient(_FakeClaudeClient):
    """Simulates Sonnet chaining end_session + load_session in one turn.

    Used to verify that the summary fires for the current session before
    history is swapped to the target."""

    ACK = "Wrapping up and switching over."

    def __init__(self, *args, target_session_id="", **kwargs):
        super().__init__(*args, **kwargs)
        self.target_session_id = target_session_id

    def stream_turn(self, text):
        self.turns.append(text)
        dispatcher = self.dispatcher
        target_id = self.target_session_id
        ack = self.ACK
        store = self.store
        sid = self.session_id

        async def _gen():
            if store is not None and sid is not None:
                await store.persist_turn(sid, "user", text)
            yield ack
            if store is not None and sid is not None:
                await store.persist_turn(
                    sid, "assistant", [{"type": "text", "text": ack}]
                )
            await dispatcher.dispatch("end_session", {})
            await dispatcher.dispatch("load_session", {"id": target_id})

        return _gen()


async def test_load_session_tool_swaps_history_and_stays_listening(
    monkeypatch, fake_profiles, tmp_path
):
    """User says 'go back to the todo session' → Sonnet calls load_session →
    after SPEAKING, claude.load_history is populated from the target session's
    turns, the client is rebound to target's session_id, and the abandoned
    session is summarized in the background so it stays in the recall index."""
    from meeko.sessions import SessionStore

    db_path = tmp_path / "meeko.db"
    monkeypatch.setenv("DEEPGRAM_API_KEY", "dg-test")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-test")
    monkeypatch.setenv("MEEKO_DB_PATH", str(db_path))

    seed = SessionStore.open(db_path)
    target_id = None
    try:
        target_id = await seed.create_session("query")
        await seed.persist_turn(target_id, "user", "let's plan a todo app")
        await seed.persist_turn(
            target_id, "assistant", [{"type": "text", "text": "Sure, let's start!"}]
        )
        await seed.update_session_metadata(
            target_id,
            title="Todo app planning",
            summary="Discussed architecture options.",
            transcript="USER: let's plan a todo app\nASSISTANT: Sure, let's start!",
        )
    finally:
        await seed.close()

    pa_instance = MagicMock()
    mic_stream = MagicMock()
    speaker_stream = MagicMock()
    pa_instance.open.side_effect = [mic_stream, speaker_stream]
    speaker_stream.write.side_effect = lambda data: None

    fake_stt = _HoldingSTTClient("dg-test")
    fake_stt.transcript = "go back to the todo session"
    fake_tts = _FakeTTSClient("dg-test")
    fake_claude_holder: dict = {}

    def make_claude(api_key, system_prompt, dispatcher, **kwargs):
        c = _LoadSessionClaudeClient(
            api_key, system_prompt, dispatcher, target_session_id=target_id, **kwargs
        )
        fake_claude_holder["client"] = c
        return c

    real_sleep = asyncio.sleep

    async def fast_sleep(delay, *a, **kw):
        if delay >= 1.0:
            return await real_sleep(0)
        return await real_sleep(delay, *a, **kw)

    with (
        patch("meeko.audio_io.pyaudio.PyAudio", return_value=pa_instance),
        patch("meeko.main.load_profiles", return_value=(fake_profiles, "query")),
        patch("meeko.main.load_dotenv"),
        patch("meeko.main.DeepgramSTT", return_value=fake_stt),
        patch("meeko.main.DeepgramTTS", return_value=fake_tts),
        patch("meeko.main.ClaudeClient", side_effect=make_claude),
        patch("meeko.main.asyncio.sleep", new=fast_sleep),
        patch("meeko.main.setup_logging"),
    ):
        task = asyncio.create_task(meeko_main.run())
        try:
            for _ in range(500):
                if fake_stt.session_obj is not None:
                    break
                await real_sleep(0.01)
            fake_stt.ready.set()

            # Wait for claude to be rebound to the target session.
            original_sid = None
            for _ in range(500):
                c = fake_claude_holder.get("client")
                if c is not None:
                    if original_sid is None:
                        original_sid = c.session_id
                    if c.session_id == target_id:
                        break
                await real_sleep(0.01)
            assert fake_claude_holder.get("client") is not None
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task

    claude = fake_claude_holder["client"]
    assert claude.session_id == target_id
    assert claude.loaded_history == [
        {"role": "user", "content": "let's plan a todo app"},
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "Sure, let's start!"}],
        },
    ]


async def test_apply_post_turn_session_change_rebinds_profile_on_load(tmp_path):
    """Direct test of the should_load branch in _apply_post_turn_session_change.

    When the loaded session was created under a different profile than
    the one currently active, the runtime profile (system prompt + voice
    via ProfileManager) must be rebound before history is swapped, so
    subsequent turns persist into a row whose profile_name still matches
    what is driving the model. Regression test for #50.
    """
    from meeko.main import _apply_post_turn_session_change
    from meeko.sessions import SessionStore
    from meeko.tools.profile import ProfileManager
    from meeko.tools.session import SessionManager

    profiles = {
        "query": Profile(
            name="query",
            wake_word="meeko",
            prompt="QUERY-PROMPT",
            voice="thalia",
        ),
        "conversation": Profile(
            name="conversation",
            wake_word="meeko",
            prompt="CONVERSATION-PROMPT",
            voice="orion",
        ),
    }

    db_path = tmp_path / "meeko.db"
    store = SessionStore.open(db_path)
    try:
        # Target session was created under the *conversation* profile —
        # different from the "query" profile we'll have active.
        target_id = await store.create_session("conversation")
        await store.persist_turn(target_id, "user", "let's keep chatting")
        await store.persist_turn(
            target_id, "assistant", [{"type": "text", "text": "Sure!"}]
        )

        claude = MagicMock()
        claude.session_id = None
        speaker = MagicMock()

        profile_manager = ProfileManager(
            profiles, claude_client=claude, active_name="query"
        )
        profile_manager.set_speaker(speaker)

        session_manager = SessionManager()
        session_manager.request_load(target_id)

        summary_calls: list[str | None] = []

        def fire_summary(sid: str | None) -> None:
            summary_calls.append(sid)

        new_state = await _apply_post_turn_session_change(
            session_manager=session_manager,
            profile_manager=profile_manager,
            claude=claude,
            store=store,
            wake_detector=None,
            session_id=None,
            fire_summary=fire_summary,
        )

        assert new_state == State.LISTENING
        # Profile was rebound to the loaded session's profile.
        assert profile_manager.active_profile.name == "conversation"
        claude.set_system_prompt.assert_called_once_with("CONVERSATION-PROMPT")
        speaker.set_profile.assert_called_once_with(profiles["conversation"])
        # History was loaded and the session id rebound to the target.
        claude.load_history.assert_called_once()
        claude.rebind_session.assert_called_once_with(target_id)
    finally:
        await store.close()


async def test_apply_post_turn_session_change_aborts_load_on_missing_row(
    tmp_path, caplog
):
    """If Sonnet hands us a bogus session id (or the row was deleted),
    abort the load rather than binding Claude to a nonexistent session."""
    from meeko.main import _apply_post_turn_session_change
    from meeko.sessions import SessionStore
    from meeko.tools.profile import ProfileManager
    from meeko.tools.session import SessionManager

    profiles = {
        "query": Profile(
            name="query", wake_word="meeko", prompt="QUERY-PROMPT", voice=None
        ),
    }

    db_path = tmp_path / "meeko.db"
    store = SessionStore.open(db_path)
    try:
        claude = MagicMock()
        claude.session_id = None
        speaker = MagicMock()
        profile_manager = ProfileManager(
            profiles, claude_client=claude, active_name="query"
        )
        profile_manager.set_speaker(speaker)

        session_manager = SessionManager()
        bogus_id = "00000000-0000-0000-0000-000000000000"
        session_manager.request_load(bogus_id)

        with caplog.at_level("ERROR", logger="meeko"):
            new_state = await _apply_post_turn_session_change(
                session_manager=session_manager,
                profile_manager=profile_manager,
                claude=claude,
                store=store,
                wake_detector=None,
                session_id=None,
                fire_summary=lambda _sid: None,
            )

        assert new_state == State.LISTENING
        claude.load_history.assert_not_called()
        claude.rebind_session.assert_not_called()
        # Pending load flag cleared so we don't loop on the bogus id.
        assert not session_manager.should_load()
        assert any("not found" in r.getMessage() for r in caplog.records)
    finally:
        await store.close()


async def test_end_then_load_session_fires_summary_for_current_session(
    monkeypatch, fake_profiles, tmp_path
):
    """Sonnet chains end_session + load_session in one turn.

    Equivalent in effect to load_session alone (the orchestrator always
    summarizes the abandoned session on load), but the chain must still
    succeed: claude ends up bound to the target session and the original
    session has been summarized in the background."""
    from meeko.sessions import SessionStore

    db_path = tmp_path / "meeko.db"
    monkeypatch.setenv("DEEPGRAM_API_KEY", "dg-test")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-test")
    monkeypatch.setenv("MEEKO_DB_PATH", str(db_path))

    seed = SessionStore.open(db_path)
    target_id = None
    try:
        target_id = await seed.create_session("query")
        await seed.persist_turn(target_id, "user", "prior question")
        await seed.persist_turn(
            target_id, "assistant", [{"type": "text", "text": "prior answer"}]
        )
    finally:
        await seed.close()

    pa_instance = MagicMock()
    mic_stream = MagicMock()
    speaker_stream = MagicMock()
    pa_instance.open.side_effect = [mic_stream, speaker_stream]
    speaker_stream.write.side_effect = lambda data: None

    fake_stt = _HoldingSTTClient("dg-test")
    fake_stt.transcript = "wrap up and go back to the prior session"
    fake_tts = _FakeTTSClient("dg-test")
    fake_claude_holder: dict = {}

    def make_claude(api_key, system_prompt, dispatcher, **kwargs):
        c = _EndThenLoadSessionClaudeClient(
            api_key, system_prompt, dispatcher, target_session_id=target_id, **kwargs
        )
        fake_claude_holder["client"] = c
        return c

    real_sleep = asyncio.sleep

    async def fast_sleep(delay, *a, **kw):
        if delay >= 1.0:
            return await real_sleep(0)
        return await real_sleep(delay, *a, **kw)

    with (
        patch("meeko.audio_io.pyaudio.PyAudio", return_value=pa_instance),
        patch("meeko.main.load_profiles", return_value=(fake_profiles, "query")),
        patch("meeko.main.load_dotenv"),
        patch("meeko.main.DeepgramSTT", return_value=fake_stt),
        patch("meeko.main.DeepgramTTS", return_value=fake_tts),
        patch("meeko.main.ClaudeClient", side_effect=make_claude),
        patch("meeko.main.asyncio.sleep", new=fast_sleep),
        patch("meeko.main.setup_logging"),
    ):
        task = asyncio.create_task(meeko_main.run())
        original_sid = None
        try:
            for _ in range(500):
                if fake_stt.session_obj is not None:
                    break
                await real_sleep(0.01)
            fake_stt.ready.set()

            # Capture original_sid before the swap, then wait for claude
            # to be rebound to the target session.
            for _ in range(500):
                c = fake_claude_holder.get("client")
                if c is not None:
                    if original_sid is None:
                        original_sid = c.session_id
                    if c.session_id == target_id:
                        break
                await real_sleep(0.01)

            # Give the background summary task time to write.
            assert original_sid is not None
            for _ in range(200):
                store_check = SessionStore.open(db_path)
                try:
                    row = store_check._conn.execute(
                        "SELECT title FROM sessions WHERE id = ?",
                        (original_sid,),
                    ).fetchone()
                finally:
                    await store_check.close()
                if row is not None and row[0] is not None:
                    break
                await real_sleep(0.01)
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task

    claude = fake_claude_holder["client"]
    assert claude.session_id == target_id
    assert claude.loaded_history == [
        {"role": "user", "content": "prior question"},
        {"role": "assistant", "content": [{"type": "text", "text": "prior answer"}]},
    ]

    # The original session (not target) got a summary from the stub.
    store = SessionStore.open(db_path)
    try:
        row = store._conn.execute(
            "SELECT title FROM sessions WHERE id = ?", (original_sid,)
        ).fetchone()
    finally:
        await store.close()
    assert row is not None and row[0] == _StubAsyncAnthropic._STUB_SUMMARY_TITLE


# --------------------------------------------------------------------------
# Decoupled puller / worker behavior — guards the queue-backpressure fix
# --------------------------------------------------------------------------


class _QueueDrivenSTTSession:
    """STT session whose events() yields whatever the test puts on
    ``event_queue``. Lets the test push events at precise moments and
    inspect what the puller has actually consumed via ``observed``."""

    def __init__(self):
        self.event_queue: asyncio.Queue = asyncio.Queue()
        self.observed: list = []
        self.sent_audio_count = 0
        self.keepalive_count = 0

    async def send_audio(self, pcm):
        self.sent_audio_count += 1

    async def send_keepalive(self):
        self.keepalive_count += 1

    async def events(self):
        while True:
            ev = await self.event_queue.get()
            self.observed.append(ev)
            yield ev


class _QueueDrivenSTTClient:
    def __init__(self, api_key):
        self.session_obj: _QueueDrivenSTTSession | None = None

    @contextlib.asynccontextmanager
    async def session(self):
        self.session_obj = _QueueDrivenSTTSession()
        yield self.session_obj


async def _wait_until(predicate, *, timeout=5.0, real_sleep=None):
    """Poll until predicate() is true, or raise TimeoutError. Bypasses
    any monkeypatched asyncio.sleep so it works alongside fast_sleep."""
    sleep = real_sleep if real_sleep is not None else asyncio.sleep
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not predicate():
        if loop.time() >= deadline:
            raise TimeoutError("predicate never became true")
        await sleep(0.005)


async def test_puller_keeps_draining_events_while_worker_is_in_tts(
    monkeypatch, fake_profiles, tmp_path
):
    """Regression test for the queue-backpressure bug. While
    drive_turns is parked awaiting TTS, pull_stt_events must keep
    consuming events from stt_session.events() so the underlying
    websockets recv queue doesn't fill and starve pong frames."""
    monkeypatch.setenv("DEEPGRAM_API_KEY", "dg-test")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-test")
    monkeypatch.setenv("MEEKO_DB_PATH", str(tmp_path / "meeko.db"))

    pa_instance = MagicMock()
    mic_stream = MagicMock()
    speaker_stream = MagicMock()
    first_write = asyncio.Event()
    speaker_stream.write.side_effect = lambda data: first_write.set()
    pa_instance.open.side_effect = [mic_stream, speaker_stream]

    fake_stt = _QueueDrivenSTTClient("dg-test")

    # TTS that yields one chunk and then blocks until the test releases
    # it — keeps drive_turns parked in speak_stream.
    release_tts = asyncio.Event()

    class _SlowTTS:
        def __init__(self, api_key):
            self.calls: list[tuple[str, str]] = []

        async def stream(self, text, voice):
            self.calls.append((text, voice))
            yield b"\x01\x02\x03\x04"
            await release_tts.wait()

    fake_tts = _SlowTTS("dg-test")

    fake_claude_holder: dict = {}

    def make_claude(api_key, system_prompt, dispatcher, **kwargs):
        c = _FakeClaudeClient(api_key, system_prompt, dispatcher, **kwargs)
        fake_claude_holder["client"] = c
        return c

    real_sleep = asyncio.sleep

    async def fast_sleep(delay, *a, **kw):
        if delay >= 1.0:
            return await real_sleep(0)
        return await real_sleep(delay, *a, **kw)

    with (
        patch("meeko.audio_io.pyaudio.PyAudio", return_value=pa_instance),
        patch("meeko.main.load_profiles", return_value=(fake_profiles, "query")),
        patch("meeko.main.load_dotenv"),
        patch("meeko.main.DeepgramSTT", return_value=fake_stt),
        patch("meeko.main.DeepgramTTS", return_value=fake_tts),
        patch("meeko.main.ClaudeClient", side_effect=make_claude),
        patch("meeko.main.asyncio.sleep", new=fast_sleep),
        patch("meeko.main.setup_logging"),
    ):
        task = asyncio.create_task(meeko_main.run())
        try:
            # Wait until the session is up.
            await _wait_until(
                lambda: fake_stt.session_obj is not None, real_sleep=real_sleep
            )
            session = fake_stt.session_obj

            # Drive a turn.
            await session.event_queue.put(
                SimpleNamespace(event="StartOfTurn", transcript="")
            )
            await session.event_queue.put(
                SimpleNamespace(event="EndOfTurn", transcript="hello")
            )

            # Wait until drive_turns has actually entered TTS — the
            # first chunk reaching the speaker proves it.
            await asyncio.wait_for(first_write.wait(), timeout=5)

            # While drive_turns is parked awaiting `release_tts`, push a
            # batch of events. If pull_stt_events were parked too (the
            # old behavior), `observed` would not grow; the queue would
            # fill and block on `put`. With the fix, all events are
            # consumed promptly.
            baseline = len(session.observed)
            for _ in range(50):
                await session.event_queue.put(
                    SimpleNamespace(event="Update", transcript="...")
                )
            await _wait_until(
                lambda: len(session.observed) >= baseline + 50,
                real_sleep=real_sleep,
            )
            assert len(session.observed) >= baseline + 50

            # drive_turns must still be in TTS (not advanced past it).
            assert fake_claude_holder["client"].turns == ["hello"]
        finally:
            release_tts.set()
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task


async def test_endofturn_during_speaking_is_logged_as_echo(
    monkeypatch, fake_profiles, tmp_path, caplog
):
    """An EndOfTurn observed while state==SPEAKING must be logged as
    `[echo?]` and NOT drive a second Claude turn (AEC observation
    mode). The check lives in the puller — decided synchronously at
    observation time so it can't race with the worker."""
    monkeypatch.setenv("DEEPGRAM_API_KEY", "dg-test")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-test")
    monkeypatch.setenv("MEEKO_DB_PATH", str(tmp_path / "meeko.db"))

    pa_instance = MagicMock()
    mic_stream = MagicMock()
    speaker_stream = MagicMock()
    first_write = asyncio.Event()
    speaker_stream.write.side_effect = lambda data: first_write.set()
    pa_instance.open.side_effect = [mic_stream, speaker_stream]

    fake_stt = _QueueDrivenSTTClient("dg-test")

    release_tts = asyncio.Event()

    class _SlowTTS:
        def __init__(self, api_key):
            self.calls: list[tuple[str, str]] = []

        async def stream(self, text, voice):
            self.calls.append((text, voice))
            yield b"\x01\x02\x03\x04"
            await release_tts.wait()

    fake_tts = _SlowTTS("dg-test")

    fake_claude_holder: dict = {}

    def make_claude(api_key, system_prompt, dispatcher, **kwargs):
        c = _FakeClaudeClient(api_key, system_prompt, dispatcher, **kwargs)
        fake_claude_holder["client"] = c
        return c

    real_sleep = asyncio.sleep

    async def fast_sleep(delay, *a, **kw):
        if delay >= 1.0:
            return await real_sleep(0)
        return await real_sleep(delay, *a, **kw)

    caplog.set_level(logging.INFO, logger="meeko")

    with (
        patch("meeko.audio_io.pyaudio.PyAudio", return_value=pa_instance),
        patch("meeko.main.load_profiles", return_value=(fake_profiles, "query")),
        patch("meeko.main.load_dotenv"),
        patch("meeko.main.DeepgramSTT", return_value=fake_stt),
        patch("meeko.main.DeepgramTTS", return_value=fake_tts),
        patch("meeko.main.ClaudeClient", side_effect=make_claude),
        patch("meeko.main.asyncio.sleep", new=fast_sleep),
        patch("meeko.main.setup_logging"),
    ):
        task = asyncio.create_task(meeko_main.run())
        try:
            await _wait_until(
                lambda: fake_stt.session_obj is not None, real_sleep=real_sleep
            )
            session = fake_stt.session_obj

            await session.event_queue.put(
                SimpleNamespace(event="EndOfTurn", transcript="hello")
            )

            # Wait until SPEAKING (first TTS chunk reached the speaker).
            await asyncio.wait_for(first_write.wait(), timeout=5)

            # Now fire a second EndOfTurn — should be suppressed as echo.
            await session.event_queue.put(
                SimpleNamespace(event="EndOfTurn", transcript="echo leakage")
            )
            await _wait_until(lambda: len(session.observed) >= 2, real_sleep=real_sleep)

            # Echo turn must be logged as such, never reach Claude.
            echo_logged = any(
                "[echo?] echo leakage" in rec.getMessage() for rec in caplog.records
            )
            assert echo_logged, "echo EndOfTurn was not logged as [echo?]"
            assert fake_claude_holder["client"].turns == ["hello"]
        finally:
            release_tts.set()
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task


async def test_endofturn_while_idle_does_not_drive_a_turn(
    monkeypatch, fake_profiles, tmp_path
):
    """When the wake-word gate has not fired (state==IDLE), any
    EndOfTurn that somehow arrives must not drive a Claude turn — the
    puller's IDLE safety belt drops it."""
    monkeypatch.delenv("MEEKO_WAKE_WORD_DISABLED", raising=False)
    monkeypatch.setenv("DEEPGRAM_API_KEY", "dg-test")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-test")
    monkeypatch.setenv("MEEKO_DB_PATH", str(tmp_path / "meeko.db"))

    pa_instance = MagicMock()
    mic_stream = MagicMock()
    speaker_stream = MagicMock()
    pa_instance.open.side_effect = [mic_stream, speaker_stream]

    wake_profiles = {
        "query": Profile(
            name="query",
            wake_word="meeko",
            prompt="system",
            voice=None,
            idle_timeout_seconds=0,
        )
    }

    # Detector that never fires — keeps state pinned at IDLE.
    class _NeverFires:
        def __init__(self, *a, **kw):
            self.calls = 0

        def process(self, pcm: bytes) -> bool:
            self.calls += 1
            return False

    fake_stt = _QueueDrivenSTTClient("dg-test")
    fake_tts = _FakeTTSClient("dg-test")

    fake_claude_holder: dict = {}

    def make_claude(api_key, system_prompt, dispatcher, **kwargs):
        c = _FakeClaudeClient(api_key, system_prompt, dispatcher, **kwargs)
        fake_claude_holder["client"] = c
        return c

    real_sleep = asyncio.sleep

    async def fast_sleep(delay, *a, **kw):
        if delay >= 1.0:
            return await real_sleep(0)
        return await real_sleep(delay, *a, **kw)

    with (
        patch("meeko.audio_io.pyaudio.PyAudio", return_value=pa_instance),
        patch("meeko.main.load_profiles", return_value=(wake_profiles, "query")),
        patch("meeko.main.load_dotenv"),
        patch("meeko.main.DeepgramSTT", return_value=fake_stt),
        patch("meeko.main.DeepgramTTS", return_value=fake_tts),
        patch("meeko.main.ClaudeClient", side_effect=make_claude),
        patch("meeko.main.WakeWordDetector", side_effect=_NeverFires),
        patch("meeko.main.asyncio.sleep", new=fast_sleep),
        patch("meeko.main.setup_logging"),
    ):
        task = asyncio.create_task(meeko_main.run())
        try:
            await _wait_until(
                lambda: fake_stt.session_obj is not None, real_sleep=real_sleep
            )
            session = fake_stt.session_obj

            await session.event_queue.put(
                SimpleNamespace(event="EndOfTurn", transcript="ignored while idle")
            )
            await _wait_until(lambda: len(session.observed) >= 1, real_sleep=real_sleep)

            # Give drive_turns a few scheduling cycles to be wrong if it
            # were going to. It shouldn't run — turn was filtered.
            for _ in range(20):
                await real_sleep(0.001)

            assert fake_claude_holder["client"].turns == []
            assert fake_tts.calls == []
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task


async def test_start_of_turn_during_speaking_triggers_barge_in(
    monkeypatch, fake_profiles, tmp_path, caplog
):
    """A StartOfTurn observed while state==SPEAKING must cancel the
    in-flight speak task, flush the speaker buffer, and return the
    session to LISTENING. A subsequent EndOfTurn drives a fresh turn."""
    monkeypatch.setenv("DEEPGRAM_API_KEY", "dg-test")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-test")
    monkeypatch.setenv("MEEKO_DB_PATH", str(tmp_path / "meeko.db"))

    pa_instance = MagicMock()
    mic_stream = MagicMock()
    speaker_stream = MagicMock()
    first_write = asyncio.Event()
    speaker_stream.write.side_effect = lambda data: first_write.set()
    pa_instance.open.side_effect = [mic_stream, speaker_stream]

    fake_stt = _QueueDrivenSTTClient("dg-test")

    release_tts = asyncio.Event()

    class _SlowTTS:
        def __init__(self, api_key):
            self.calls: list[tuple[str, str]] = []

        async def stream(self, text, voice):
            self.calls.append((text, voice))
            yield b"\x01\x02\x03\x04"
            await release_tts.wait()

    fake_tts = _SlowTTS("dg-test")

    fake_claude_holder: dict = {}

    def make_claude(api_key, system_prompt, dispatcher, **kwargs):
        c = _FakeClaudeClient(api_key, system_prompt, dispatcher, **kwargs)
        fake_claude_holder["client"] = c
        return c

    real_sleep = asyncio.sleep

    async def fast_sleep(delay, *a, **kw):
        if delay >= 1.0:
            return await real_sleep(0)
        return await real_sleep(delay, *a, **kw)

    caplog.set_level(logging.INFO, logger="meeko")

    with (
        patch("meeko.audio_io.pyaudio.PyAudio", return_value=pa_instance),
        patch("meeko.main.load_profiles", return_value=(fake_profiles, "query")),
        patch("meeko.main.load_dotenv"),
        patch("meeko.main.DeepgramSTT", return_value=fake_stt),
        patch("meeko.main.DeepgramTTS", return_value=fake_tts),
        patch("meeko.main.ClaudeClient", side_effect=make_claude),
        patch("meeko.main.asyncio.sleep", new=fast_sleep),
        patch("meeko.main.setup_logging"),
    ):
        task = asyncio.create_task(meeko_main.run())
        try:
            await _wait_until(
                lambda: fake_stt.session_obj is not None, real_sleep=real_sleep
            )
            session = fake_stt.session_obj

            # Drive the first turn into SPEAKING.
            await session.event_queue.put(
                SimpleNamespace(event="EndOfTurn", transcript="hello")
            )
            await asyncio.wait_for(first_write.wait(), timeout=5)

            # User barges in: StartOfTurn while SPEAKING. The speak task
            # should be cancelled even though release_tts is still
            # blocking the TTS stream.
            await session.event_queue.put(
                SimpleNamespace(event="StartOfTurn", transcript="")
            )
            await _wait_until(
                lambda: any(
                    "Barge-in: turn cancelled" in rec.getMessage()
                    for rec in caplog.records
                ),
                real_sleep=real_sleep,
            )

            # Barge-in mutes further writes via the audio_io flag rather
            # than touching the PortAudio stream — stop_stream from the
            # asyncio thread races a blocking write_stream still running
            # in the to_thread executor and corrupts the stream on ALSA.
            assert not speaker_stream.stop_stream.called
            assert not speaker_stream.start_stream.called

            # Now the user finishes their interruption. The EndOfTurn
            # arrives in LISTENING (not SPEAKING) and drives a fresh turn —
            # the barge-in utterance is the new instruction and reaches Claude.
            release_tts.set()  # unblock any residual TTS so new turn can run
            await session.event_queue.put(
                SimpleNamespace(event="EndOfTurn", transcript="wait, actually")
            )
            await _wait_until(
                lambda: (
                    fake_claude_holder["client"].turns == ["hello", "wait, actually"]
                ),
                real_sleep=real_sleep,
            )
        finally:
            release_tts.set()
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task


async def test_start_of_turn_during_processing_triggers_barge_in(
    monkeypatch, fake_profiles, tmp_path, caplog
):
    """A StartOfTurn observed while state==PROCESSING (after EndOfTurn,
    before first audio plays — the window covers Claude TTFT plus
    Deepgram TTS first-byte synthesis) must cancel the in-flight turn
    and return the session to LISTENING so the user's interruption is
    captured rather than silently dropped."""
    monkeypatch.setenv("DEEPGRAM_API_KEY", "dg-test")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-test")
    monkeypatch.setenv("MEEKO_DB_PATH", str(tmp_path / "meeko.db"))

    pa_instance = MagicMock()
    mic_stream = MagicMock()
    speaker_stream = MagicMock()
    pa_instance.open.side_effect = [mic_stream, speaker_stream]

    fake_stt = _QueueDrivenSTTClient("dg-test")
    fake_tts = _FakeTTSClient("dg-test")

    # Block Claude before it yields any text — keeps the turn in
    # PROCESSING (no TTS first-byte → no SPEAKING transition).
    release_claude = asyncio.Event()
    claude_called = asyncio.Event()

    class _SlowClaude(_FakeClaudeClient):
        def stream_turn(self, text):
            self.turns.append(text)

            async def _gen():
                claude_called.set()
                await release_claude.wait()
                yield "Hi."

            return _gen()

    fake_claude_holder: dict = {}

    def make_claude(api_key, system_prompt, dispatcher, **kwargs):
        c = _SlowClaude(api_key, system_prompt, dispatcher, **kwargs)
        fake_claude_holder["client"] = c
        return c

    real_sleep = asyncio.sleep

    async def fast_sleep(delay, *a, **kw):
        if delay >= 1.0:
            return await real_sleep(0)
        return await real_sleep(delay, *a, **kw)

    caplog.set_level(logging.INFO, logger="meeko")

    with (
        patch("meeko.audio_io.pyaudio.PyAudio", return_value=pa_instance),
        patch("meeko.main.load_profiles", return_value=(fake_profiles, "query")),
        patch("meeko.main.load_dotenv"),
        patch("meeko.main.DeepgramSTT", return_value=fake_stt),
        patch("meeko.main.DeepgramTTS", return_value=fake_tts),
        patch("meeko.main.ClaudeClient", side_effect=make_claude),
        patch("meeko.main.asyncio.sleep", new=fast_sleep),
        patch("meeko.main.setup_logging"),
    ):
        task = asyncio.create_task(meeko_main.run())
        try:
            await _wait_until(
                lambda: fake_stt.session_obj is not None, real_sleep=real_sleep
            )
            session = fake_stt.session_obj

            # Drive the first turn into PROCESSING.
            await session.event_queue.put(
                SimpleNamespace(event="EndOfTurn", transcript="hello")
            )
            await asyncio.wait_for(claude_called.wait(), timeout=5)

            # No audio has played — state is still PROCESSING. User
            # barges in: StartOfTurn must cancel the in-flight turn.
            await session.event_queue.put(
                SimpleNamespace(event="StartOfTurn", transcript="")
            )
            await _wait_until(
                lambda: any(
                    "Barge-in: turn cancelled" in rec.getMessage()
                    for rec in caplog.records
                ),
                real_sleep=real_sleep,
            )

            # Release Claude in case any partial cleanup is awaiting it.
            release_claude.set()

            # The user finishes their interruption. The EndOfTurn arrives
            # in LISTENING (not PROCESSING/SPEAKING) and drives a fresh turn
            # — the user's change-of-mind was captured, not silently dropped.
            await session.event_queue.put(
                SimpleNamespace(event="EndOfTurn", transcript="wait, actually")
            )
            await _wait_until(
                lambda: (
                    fake_claude_holder["client"].turns == ["hello", "wait, actually"]
                ),
                real_sleep=real_sleep,
            )
        finally:
            release_claude.set()
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task


async def test_end_of_turn_immediately_after_barge_in_is_not_dropped(
    monkeypatch, fake_profiles, tmp_path, caplog
):
    """An EndOfTurn arriving before the barge-in cancel has propagated
    through drive_turns must still be processed as a real turn, not
    dropped as `[echo?]`. request_barge_in() flips state to LISTENING
    synchronously so pull_stt_events sees the right state when it drains
    the EndOfTurn that follows StartOfTurn."""
    monkeypatch.setenv("DEEPGRAM_API_KEY", "dg-test")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-test")
    monkeypatch.setenv("MEEKO_DB_PATH", str(tmp_path / "meeko.db"))

    pa_instance = MagicMock()
    mic_stream = MagicMock()
    speaker_stream = MagicMock()
    first_write = asyncio.Event()
    speaker_stream.write.side_effect = lambda data: first_write.set()
    pa_instance.open.side_effect = [mic_stream, speaker_stream]

    fake_stt = _QueueDrivenSTTClient("dg-test")

    release_tts = asyncio.Event()

    class _SlowTTS:
        def __init__(self, api_key):
            self.calls: list[tuple[str, str]] = []

        async def stream(self, text, voice):
            self.calls.append((text, voice))
            yield b"\x01\x02\x03\x04"
            await release_tts.wait()

    fake_tts = _SlowTTS("dg-test")

    fake_claude_holder: dict = {}

    def make_claude(api_key, system_prompt, dispatcher, **kwargs):
        c = _FakeClaudeClient(api_key, system_prompt, dispatcher, **kwargs)
        fake_claude_holder["client"] = c
        return c

    real_sleep = asyncio.sleep

    async def fast_sleep(delay, *a, **kw):
        if delay >= 1.0:
            return await real_sleep(0)
        return await real_sleep(delay, *a, **kw)

    caplog.set_level(logging.INFO, logger="meeko")

    with (
        patch("meeko.audio_io.pyaudio.PyAudio", return_value=pa_instance),
        patch("meeko.main.load_profiles", return_value=(fake_profiles, "query")),
        patch("meeko.main.load_dotenv"),
        patch("meeko.main.DeepgramSTT", return_value=fake_stt),
        patch("meeko.main.DeepgramTTS", return_value=fake_tts),
        patch("meeko.main.ClaudeClient", side_effect=make_claude),
        patch("meeko.main.asyncio.sleep", new=fast_sleep),
        patch("meeko.main.setup_logging"),
    ):
        task = asyncio.create_task(meeko_main.run())
        try:
            await _wait_until(
                lambda: fake_stt.session_obj is not None, real_sleep=real_sleep
            )
            session = fake_stt.session_obj

            # Drive into SPEAKING.
            await session.event_queue.put(
                SimpleNamespace(event="EndOfTurn", transcript="hello")
            )
            await asyncio.wait_for(first_write.wait(), timeout=5)

            # Queue StartOfTurn and EndOfTurn back-to-back so the
            # EndOfTurn is available the moment pull_stt_events finishes
            # handling the StartOfTurn. Without the synchronous state
            # flip in request_barge_in, the EndOfTurn would be processed
            # while state is still SPEAKING and dropped as `[echo?]`.
            await session.event_queue.put(
                SimpleNamespace(event="StartOfTurn", transcript="")
            )
            await session.event_queue.put(
                SimpleNamespace(event="EndOfTurn", transcript="stop")
            )
            release_tts.set()  # let the cancelled speak unwind

            await _wait_until(
                lambda: fake_claude_holder["client"].turns == ["hello", "stop"],
                real_sleep=real_sleep,
            )

            # Defensive: ensure the interruption was not logged as echo.
            assert not any(
                "[echo?] stop" in rec.getMessage() for rec in caplog.records
            ), "EndOfTurn that arrived during cancel propagation was dropped as echo"
        finally:
            release_tts.set()
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task


async def test_start_of_turn_while_listening_does_not_cancel(
    monkeypatch, fake_profiles, tmp_path, caplog
):
    """StartOfTurn while LISTENING must not trigger barge-in (there is
    nothing to cancel) and must not log a barge-in line. PROCESSING is
    a separate case — it does trigger barge-in, covered by
    test_start_of_turn_during_processing_triggers_barge_in."""
    monkeypatch.setenv("DEEPGRAM_API_KEY", "dg-test")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-test")
    monkeypatch.setenv("MEEKO_DB_PATH", str(tmp_path / "meeko.db"))

    pa_instance = MagicMock()
    mic_stream = MagicMock()
    speaker_stream = MagicMock()
    pa_instance.open.side_effect = [mic_stream, speaker_stream]

    fake_stt = _QueueDrivenSTTClient("dg-test")
    fake_tts = _FakeTTSClient("dg-test")

    fake_claude_holder: dict = {}

    def make_claude(api_key, system_prompt, dispatcher, **kwargs):
        c = _FakeClaudeClient(api_key, system_prompt, dispatcher, **kwargs)
        fake_claude_holder["client"] = c
        return c

    real_sleep = asyncio.sleep

    async def fast_sleep(delay, *a, **kw):
        if delay >= 1.0:
            return await real_sleep(0)
        return await real_sleep(delay, *a, **kw)

    caplog.set_level(logging.INFO, logger="meeko")

    with (
        patch("meeko.audio_io.pyaudio.PyAudio", return_value=pa_instance),
        patch("meeko.main.load_profiles", return_value=(fake_profiles, "query")),
        patch("meeko.main.load_dotenv"),
        patch("meeko.main.DeepgramSTT", return_value=fake_stt),
        patch("meeko.main.DeepgramTTS", return_value=fake_tts),
        patch("meeko.main.ClaudeClient", side_effect=make_claude),
        patch("meeko.main.asyncio.sleep", new=fast_sleep),
        patch("meeko.main.setup_logging"),
    ):
        task = asyncio.create_task(meeko_main.run())
        try:
            await _wait_until(
                lambda: fake_stt.session_obj is not None, real_sleep=real_sleep
            )
            session = fake_stt.session_obj

            # StartOfTurn arriving while LISTENING — no barge-in.
            await session.event_queue.put(
                SimpleNamespace(event="StartOfTurn", transcript="")
            )
            await session.event_queue.put(
                SimpleNamespace(event="EndOfTurn", transcript="hi")
            )
            await _wait_until(
                lambda: fake_claude_holder["client"].turns == ["hi"],
                real_sleep=real_sleep,
            )

            assert not any("Barge-in" in rec.getMessage() for rec in caplog.records)
            assert not speaker_stream.stop_stream.called
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task


async def test_run_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("DEEPGRAM_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with (
        patch("meeko.main.load_dotenv"),
        patch("meeko.main.setup_logging"),
    ):
        with pytest.raises(KeyError):
            await meeko_main.run()


async def test_end_session_before_any_turn_fire_summary_is_noop(
    monkeypatch, fake_profiles, tmp_path
):
    """If end_session fires before any user turn is persisted, session_id
    is None and fire_summary must no-op (lazy creation never ran).
    Covers the `if finalized_sid is None: return` branch in fire_summary."""
    from meeko.sessions import SessionStore

    db_path = tmp_path / "meeko.db"
    monkeypatch.setenv("DEEPGRAM_API_KEY", "dg-test")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-test")
    monkeypatch.setenv("MEEKO_DB_PATH", str(db_path))

    pa_instance = MagicMock()
    mic_stream = MagicMock()
    speaker_stream = MagicMock()
    pa_instance.open.side_effect = [mic_stream, speaker_stream]
    speaker_stream.write.side_effect = lambda data: None

    fake_stt = _HoldingSTTClient("dg-test")
    fake_stt.transcript = "stop"
    fake_tts = _FakeTTSClient("dg-test")
    fake_claude_holder: dict = {}

    def make_claude(api_key, system_prompt, dispatcher, **kwargs):
        c = _EndSessionNoTurnClaudeClient(api_key, system_prompt, dispatcher, **kwargs)
        fake_claude_holder["client"] = c
        return c

    real_sleep = asyncio.sleep

    async def fast_sleep(delay, *a, **kw):
        if delay >= 1.0:
            return await real_sleep(0)
        return await real_sleep(delay, *a, **kw)

    with (
        patch("meeko.audio_io.pyaudio.PyAudio", return_value=pa_instance),
        patch("meeko.main.load_profiles", return_value=(fake_profiles, "query")),
        patch("meeko.main.load_dotenv"),
        patch("meeko.main.DeepgramSTT", return_value=fake_stt),
        patch("meeko.main.DeepgramTTS", return_value=fake_tts),
        patch("meeko.main.ClaudeClient", side_effect=make_claude),
        patch("meeko.main.asyncio.sleep", new=fast_sleep),
        patch("meeko.main.setup_logging"),
    ):
        task = asyncio.create_task(meeko_main.run())
        try:
            for _ in range(500):
                if fake_stt.session_obj is not None:
                    break
                await real_sleep(0.01)
            fake_stt.ready.set()

            claude = None
            for _ in range(500):
                c = fake_claude_holder.get("client")
                if c is not None and c.reset_count >= 1:
                    claude = c
                    break
                await real_sleep(0.01)
            assert claude is not None, "end_session never called reset_session"
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task

    # No session row was created — lazy creation never fired and
    # fire_summary(None) was a no-op rather than crashing.
    store = SessionStore.open(db_path)
    try:
        rows = await store.list_sessions()
    finally:
        await store.close()
    assert len(rows) == 0


async def test_load_session_when_current_session_has_turns_fires_summary(
    monkeypatch, fake_profiles, tmp_path
):
    """load_session fires after the current session already has turns.
    session_id is non-None so the summary log branch (line 332 in
    _apply_post_turn_session_change) and fire_summary are both exercised."""
    from meeko.sessions import SessionStore

    db_path = tmp_path / "meeko.db"
    monkeypatch.setenv("DEEPGRAM_API_KEY", "dg-test")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-test")
    monkeypatch.setenv("MEEKO_DB_PATH", str(db_path))

    seed = SessionStore.open(db_path)
    target_id = None
    try:
        target_id = await seed.create_session("query")
        await seed.persist_turn(target_id, "user", "prior question")
        await seed.persist_turn(
            target_id, "assistant", [{"type": "text", "text": "prior answer"}]
        )
        await seed.update_session_metadata(
            target_id,
            title="Prior session",
            summary="We discussed things.",
            transcript="USER: prior question\nASSISTANT: prior answer",
        )
    finally:
        await seed.close()

    pa_instance = MagicMock()
    mic_stream = MagicMock()
    speaker_stream = MagicMock()
    pa_instance.open.side_effect = [mic_stream, speaker_stream]
    speaker_stream.write.side_effect = lambda data: None

    fake_stt = _HoldingSTTClient("dg-test")
    fake_stt.transcript = "go back to the prior session"
    fake_tts = _FakeTTSClient("dg-test")
    fake_claude_holder: dict = {}

    def make_claude(api_key, system_prompt, dispatcher, **kwargs):
        c = _LoadSessionWithTurnClaudeClient(
            api_key, system_prompt, dispatcher, target_session_id=target_id, **kwargs
        )
        fake_claude_holder["client"] = c
        return c

    real_sleep = asyncio.sleep

    async def fast_sleep(delay, *a, **kw):
        if delay >= 1.0:
            return await real_sleep(0)
        return await real_sleep(delay, *a, **kw)

    with (
        patch("meeko.audio_io.pyaudio.PyAudio", return_value=pa_instance),
        patch("meeko.main.load_profiles", return_value=(fake_profiles, "query")),
        patch("meeko.main.load_dotenv"),
        patch("meeko.main.DeepgramSTT", return_value=fake_stt),
        patch("meeko.main.DeepgramTTS", return_value=fake_tts),
        patch("meeko.main.ClaudeClient", side_effect=make_claude),
        patch("meeko.main.asyncio.sleep", new=fast_sleep),
        patch("meeko.main.setup_logging"),
    ):
        task = asyncio.create_task(meeko_main.run())
        original_sid = None
        try:
            for _ in range(500):
                if fake_stt.session_obj is not None:
                    break
                await real_sleep(0.01)
            fake_stt.ready.set()

            # Wait for claude to be rebound to the target session.
            for _ in range(500):
                c = fake_claude_holder.get("client")
                if c is not None and c.session_id == target_id:
                    break
                await real_sleep(0.01)

            # The original session is the only row in the DB that isn't
            # target_id — lazy creation wrote it, then load_session swapped
            # the binding to target_id.
            rows_check = SessionStore.open(db_path)
            try:
                all_rows = await rows_check.list_sessions()
            finally:
                await rows_check.close()
            non_target = [r["id"] for r in all_rows if r["id"] != target_id]
            assert non_target, "lazy creation never fired — no non-target session row"
            original_sid = non_target[0]

            # Give the background summary task time to write.
            for _ in range(200):
                store_check = SessionStore.open(db_path)
                try:
                    row = store_check._conn.execute(
                        "SELECT title FROM sessions WHERE id = ?",
                        (original_sid,),
                    ).fetchone()
                finally:
                    await store_check.close()
                if row is not None and row[0] is not None:
                    break
                await real_sleep(0.01)
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task

    # The original (non-target) session was summarized by fire_summary.
    store = SessionStore.open(db_path)
    try:
        row = store._conn.execute(
            "SELECT title FROM sessions WHERE id = ?", (original_sid,)
        ).fetchone()
    finally:
        await store.close()
    assert row is not None and row[0] == _StubAsyncAnthropic._STUB_SUMMARY_TITLE
    assert fake_claude_holder["client"].session_id == target_id
