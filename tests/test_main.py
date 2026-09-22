"""Tests for `meeko.main`, the entry point and composition root.

Covers `setup_logging`, the sync `main()` entrypoint, and a happy-path
drive-through of `run()` with all external services (PyAudio, Deepgram
STT/TTS, Anthropic) mocked.

The state machine itself lives in `meeko/orchestrator/state.py` — see test_state.py.
"""

import asyncio
import contextlib
import functools
import json
import logging
import logging.handlers
import signal
import time
from importlib import resources
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from meeko import main as meeko_main
from meeko.config import Profile
from meeko.main import setup_logging
from meeko.sessions import UNTITLED
from meeko.tools.profile import ProfileManager


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
    # INFO, not DEBUG: conversation content is logged at DEBUG and the
    # stream handler's output is the system journal under systemd.
    assert clean_logger.level == logging.INFO


def test_setup_logging_file_target(monkeypatch, tmp_path, clean_logger):
    monkeypatch.chdir(tmp_path)
    setup_logging(log_level="INFO", log_target="file")
    assert isinstance(clean_logger.handlers[0], logging.handlers.RotatingFileHandler)
    assert clean_logger.level == logging.INFO
    # Close the file handler so tmp_path can be cleaned up on Windows/macOS.
    clean_logger.handlers[0].close()


def test_setup_logging_invalid_level_falls_back_to_info(clean_logger):
    setup_logging(log_level="NOT_A_REAL_LEVEL")
    assert clean_logger.level == logging.INFO


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
        # on_session's FIRST_COMPLETED cancel an in-flight turn
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
    """Stands in for ClaudeClient. Each turn replies with `reply`, then
    dispatches `tool_calls` (``(name, args)`` pairs) in order, the way Sonnet
    acknowledges before a session tool fires. With `persist=False` the turn
    skips lazy session creation and persistence, as if Sonnet called the
    tools before any turn was saved. `_tool_calling_claude` builds these."""

    def __init__(
        self,
        api_key,
        system_prompt,
        dispatcher,
        *,
        reply="Hi.",
        tool_calls=(),
        persist=True,
        **kwargs,
    ):
        self.api_key = api_key
        self.system_prompt = system_prompt
        self.dispatcher = dispatcher
        self.store = kwargs.get("store")
        self.session_id = kwargs.get("session_id")
        self._create_session_fn = kwargs.get("create_session_fn")
        self.turns: list[str] = []
        self.loaded_history: list[dict] | None = None
        self.reset_count = 0  # incremented each time reset_session() is called
        self._reply = reply
        self._tool_calls = tool_calls
        self._persist = persist

    def stream_turn(self, text):
        self.turns.append(text)
        return self._turn(text)

    async def _turn(self, text):
        # Mirror the real ClaudeClient's lazy session creation + persistence
        # so downstream consumers (end-of-session summarization) see a
        # turn log and tests cover the lazy-create path.
        store = self.store if self._persist else None
        if store is not None:
            if self.session_id is None and self._create_session_fn is not None:
                self.session_id = await self._create_session_fn()
            if self.session_id is not None:
                await store.persist_turn(self.session_id, "user", text)
        yield self._reply
        if store is not None and self.session_id is not None:
            await store.persist_turn(
                self.session_id, "assistant", [{"type": "text", "text": self._reply}]
            )
        for name, args in self._tool_calls:
            await self.dispatcher.dispatch(name, args)

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


def _tool_calling_claude(reply, *tool_calls, persist=True):
    """A `claude=` factory for `_running_meeko` whose turn replies with
    `reply` and then dispatches each ``(name, args)`` in `tool_calls`."""
    return functools.partial(
        _FakeClaudeClient, reply=reply, tool_calls=tool_calls, persist=persist
    )


_END_SESSION = _tool_calling_claude("Goodnight!", ("end_session", {}))


@pytest.fixture(autouse=True)
def _disable_wake_word_by_default(monkeypatch):
    """Existing orchestrator tests predate wake-word gating — default them
    into the legacy flow. Tests that exercise the wake-word path opt back
    in by deleting this env var in their own body."""
    monkeypatch.setenv("MEEKO_WAKE_WORD_DISABLED", "1")


@pytest.fixture(autouse=True)
def _isolate_from_host(monkeypatch, tmp_path):
    """Keep run() off the contributor's machine: read a private copy of the
    bundled default config instead of their ~/.config/meeko/meeko.toml
    (which run() would otherwise read, or create if missing), and never
    touch a real XVF3800 LED ring that happens to be plugged in."""
    config_path = tmp_path / "meeko.toml"
    config_path.write_bytes(
        resources.files("meeko").joinpath("default_config.toml").read_bytes()
    )
    monkeypatch.setenv("MEEKO_CONFIG", str(config_path))
    monkeypatch.setenv("MEEKO_LED_DISABLED", "1")


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
            prompt="system",
            voice=None,
            # Disable the post-turn idle monitor: orchestrator tests
            # patch asyncio.sleep, which would otherwise short-circuit
            # the idle timer and trigger spurious session ends.
            idle_timeout_seconds=0,
        )
    }


_REAL_SLEEP = asyncio.sleep
# One stereo int16 frame, as the PyAudio mic callback delivers it.
_FRAME = b"\x00\x00\x00\x00"


async def _fast_sleep(delay, *a, **kw):
    """Stand-in for `asyncio.sleep` in run(): collapses the long waits (the
    1s tail drain, reconnect holds) to a yield and keeps short ones real."""
    if delay >= 1.0:
        return await _REAL_SLEEP(0)
    return await _REAL_SLEEP(delay, *a, **kw)


async def _wait_until(predicate, *, timeout=5.0, msg="predicate never became true"):
    """Poll until predicate() is true, or raise TimeoutError(msg). Uses the
    real asyncio.sleep, so it keeps working while run()'s is patched."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not predicate():
        if loop.time() >= deadline:
            raise TimeoutError(msg)
        await _REAL_SLEEP(0.005)


def _set_env(monkeypatch, tmp_path, *, wake_word=False) -> Path:
    """Set the API keys and DB path run() needs; return the DB path.
    `wake_word=True` undoes the autouse fixture that disables the gate."""
    db_path = tmp_path / "meeko.db"
    monkeypatch.setenv("DEEPGRAM_API_KEY", "dg-test")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-test")
    monkeypatch.setenv("MEEKO_DB_PATH", str(db_path))
    if wake_word:
        monkeypatch.delenv("MEEKO_WAKE_WORD_DISABLED", raising=False)
    return db_path


def _profile(name="query", **overrides) -> Profile:
    """A profile with both silence windows off, so the patched asyncio.sleep
    can't fire them early."""
    fields = {
        "name": name,
        "prompt": "system",
        "voice": None,
        "idle_timeout_seconds": 0,
        "post_wake_timeout_seconds": 0,
    }
    return Profile(**(fields | overrides))


class _FakeDetector:
    """Wake-word detector that fires on every chunk. Counts `process()` and
    `reset()` calls; subclasses override `_fires` to change when it fires."""

    def __init__(self, *a, **kw):
        self.calls = 0
        self.resets = 0

    def process(self, pcm: bytes) -> bool:
        self.calls += 1
        return self._fires()

    def _fires(self) -> bool:
        return True

    def reset(self) -> None:
        self.resets += 1


class _FireOnceDetector(_FakeDetector):
    def _fires(self) -> bool:
        return self.calls == 1


class _NeverFiresDetector(_FakeDetector):
    def _fires(self) -> bool:
        return False


class _BlockingTTSClient:
    """Yields one chunk, then blocks until `release` is set: holds the turn
    worker in SPEAKING for as long as the test needs."""

    def __init__(self, api_key="dg-test"):
        self.calls: list[tuple[str, str]] = []
        self.release = asyncio.Event()

    async def stream(self, text, voice):
        self.calls.append((text, voice))
        yield b"\x01\x02\x03\x04"
        await self.release.wait()


class _SignalingSTTClient(_FakeSTTClient):
    """`_FakeSTTClient` with no canned events that sets `entered` once a
    session is open, i.e. once run() has finished starting up."""

    def __init__(self, api_key="dg-test"):
        super().__init__(api_key)
        self.events = []
        self.entered = asyncio.Event()

    @contextlib.asynccontextmanager
    async def session(self):
        async with super().session() as s:
            self.entered.set()
            yield s


class _Harness:
    """What `_running_meeko` hands a test: the run() task, the PyAudio mocks,
    and the fakes run() was built with (`claude` and `detector` are filled in
    when run() constructs them)."""

    def __init__(self, stt, tts):
        self.stt = stt
        self.tts = tts
        self.task: asyncio.Task | None = None
        self.claude = None
        self.detector = None
        self.mic_cb = None
        self.pa = MagicMock()
        self.mic = MagicMock()
        self.speaker = MagicMock()
        self.first_write = asyncio.Event()
        self.speaker.write.side_effect = lambda data: self.first_write.set()
        self.pa.open.side_effect = self._open_stream

    def _open_stream(self, **kwargs):
        if kwargs.get("input"):
            self.mic_cb = kwargs["stream_callback"]
            return self.mic
        return self.speaker

    def push_mic(self, frame=_FRAME) -> None:
        """Deliver one chunk the way PyAudio's callback thread would."""
        self.mic_cb(frame, 1, None, 0)

    async def stt_session(self):
        await _wait_until(
            lambda: self.stt.session_obj is not None, msg="STT session never opened"
        )
        return self.stt.session_obj

    async def wake(self) -> None:
        """Push one mic chunk once the mic and detector are up, and wait for
        the detector to see it: with a firing detector, state is LISTENING."""
        await _wait_until(
            lambda: self.mic_cb is not None and self.detector is not None,
            msg="mic stream or wake detector never came up",
        )
        self.push_mic()
        await _wait_until(lambda: self.detector.calls >= 1)

    async def claude_reset(self) -> None:
        """Wait for a session tool to reset the Claude client (the stable
        signal that the post-turn session change ran)."""
        await _wait_until(
            lambda: self.claude is not None and self.claude.reset_count >= 1,
            msg="session change never called reset_session",
        )


@contextlib.asynccontextmanager
async def _running_meeko(
    profiles,
    *,
    stt,
    tts=None,
    claude=_FakeClaudeClient,
    detector=None,
    sleep=_fast_sleep,
    extra_patches=(),
    **run_kwargs,
):
    """Run `meeko_main.run(**run_kwargs)` as a task with every external
    service faked, yield a `_Harness`, and cancel the task on exit.

    `claude` and `detector` are called the way run() calls ClaudeClient and
    WakeWordDetector; `detector=None` leaves the detector unpatched (fine
    while the autouse fixture disables the gate). `sleep` replaces
    `asyncio.sleep` inside run(); None leaves it alone.
    """
    h = _Harness(stt, tts if tts is not None else _FakeTTSClient("dg-test"))

    def make_claude(*a, **kw):
        h.claude = claude(*a, **kw)
        return h.claude

    def make_detector(*a, **kw):
        h.detector = detector(*a, **kw)
        return h.detector

    with contextlib.ExitStack() as stack:
        for p in (
            patch("meeko.audio_io.pyaudio.PyAudio", return_value=h.pa),
            patch("meeko.main.load_profiles", return_value=(profiles, "query")),
            patch("meeko.main.load_dotenv"),
            patch("meeko.main.DeepgramSTT", return_value=h.stt),
            patch("meeko.main.DeepgramTTS", return_value=h.tts),
            patch("meeko.main.ClaudeClient", side_effect=make_claude),
            patch("meeko.main.setup_logging"),  # avoid mutating the real logger
            *extra_patches,
        ):
            stack.enter_context(p)
        if detector is not None:
            stack.enter_context(
                patch("meeko.main.WakeWordDetector", side_effect=make_detector)
            )
        if sleep is not None:
            stack.enter_context(patch("meeko.main.asyncio.sleep", new=sleep))

        h.task = asyncio.create_task(meeko_main.run(**run_kwargs))
        try:
            yield h
        finally:
            h.task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await h.task


async def _seed_session(
    db_path,
    user="prior question",
    assistant="prior answer",
    profile_name="query",
    **metadata,
) -> str:
    """Write a two-turn session (plus summary metadata, if given) and return
    its id."""
    from meeko.sessions import SessionStore

    store = SessionStore.open(db_path)
    try:
        sid = await store.create_session(profile_name)
        await store.persist_turn(sid, "user", user)
        await store.persist_turn(
            sid, "assistant", [{"type": "text", "text": assistant}]
        )
        if metadata:
            await store.update_session_metadata(sid, **metadata)
    finally:
        await store.close()
    return sid


async def _list_sessions(db_path) -> list[dict]:
    from meeko.sessions import SessionStore

    store = SessionStore.open(db_path)
    try:
        return await store.list_sessions()
    finally:
        await store.close()


async def _abandoned_session(db_path, target_id) -> str:
    """After a load_session swap, the session the user abandoned is the only
    row that isn't the target: lazy creation wrote it, then the load
    rebound the client to `target_id`."""
    others = [r["id"] for r in await _list_sessions(db_path) if r["id"] != target_id]
    assert others, "lazy creation never fired — no non-target session row"
    return others[0]


async def _get_session(db_path, sid) -> dict | None:
    from meeko.sessions import SessionStore

    store = SessionStore.open(db_path)
    try:
        return await store.get_session(sid)
    finally:
        await store.close()


async def _wait_for_title(db_path, sid, *, timeout=5.0) -> str:
    """Poll until the background summary has titled `sid`; return the title."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while True:
        row = await _get_session(db_path, sid)
        if row is not None and row["title"] is not None:
            return row["title"]
        if loop.time() >= deadline:
            raise TimeoutError(f"session {sid} was never summarized")
        await _REAL_SLEEP(0.01)


async def test_run_drives_one_turn_end_to_end(monkeypatch, fake_profiles, tmp_path):
    _set_env(monkeypatch, tmp_path)
    tts = _FakeTTSClient("dg-test")

    # Mic callback is never exercised here; mic_queue stays empty and
    # MicPump just spins on its 100ms timeout.
    async with _running_meeko(
        fake_profiles, stt=_FakeSTTClient("dg-test"), tts=tts
    ) as h:
        await asyncio.wait_for(h.first_write.wait(), timeout=5)

    assert h.claude.turns == ["hello"]
    assert tts.calls == [("Hi.", "asteria")]
    # Speaker.write receives the TTS chunk duplicated into stereo
    # (L/R from the mono TTS sample).
    h.speaker.write.assert_any_call(b"\x01\x02\x01\x02\x03\x04\x03\x04")
    # Cleanup was executed in run()'s finally block.
    h.mic.stop_stream.assert_called()
    h.mic.close.assert_called()
    h.speaker.stop_stream.assert_called()
    h.speaker.close.assert_called()
    h.pa.terminate.assert_called()


async def test_mute_mic_while_speaking_drops_chunks_during_speaking(
    monkeypatch, fake_profiles, tmp_path
):
    """With MEEKO_MUTE_MIC_WHILE_SPEAKING=1, mic audio captured while
    the assistant is SPEAKING must not reach the STT session."""
    _set_env(monkeypatch, tmp_path)
    monkeypatch.setenv("MEEKO_MUTE_MIC_WHILE_SPEAKING", "1")
    stt = _FakeSTTClient("dg-test")

    # The blocking TTS keeps the state machine in SPEAKING while we push
    # mic audio.
    async with _running_meeko(fake_profiles, stt=stt, tts=_BlockingTTSClient()) as h:
        await asyncio.wait_for(h.first_write.wait(), timeout=5)
        # We're now inside SPEAKING. Snapshot STT send count, then
        # push mic chunks via the captured PyAudio callback and
        # give MicPump a chance to see them.
        session = stt.session_obj
        baseline = session.sent_audio_count
        for _ in range(5):
            h.push_mic()
        for _ in range(50):
            await _REAL_SLEEP(0)
        assert session.sent_audio_count == baseline


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
    _set_env(monkeypatch, tmp_path)
    stt = _ReconnectSTTClient("dg-test")

    async with _running_meeko(fake_profiles, stt=stt) as h:
        await asyncio.wait_for(h.first_write.wait(), timeout=5)

    # Session was re-established after the ConnectionClosed on the first.
    assert stt.session_count >= 2


async def test_run_backs_off_on_repeated_stt_failures(
    monkeypatch, fake_profiles, tmp_path
):
    """Repeated connect failures should sleep with increasing delays
    (0.5, 1, 2, ...) and only log a full traceback on the first failure."""
    _set_env(monkeypatch, tmp_path)

    class _AlwaysFailSTT:
        def __init__(self, api_key):
            self.attempts = 0

        @contextlib.asynccontextmanager
        async def session(self):
            self.attempts += 1
            raise OSError("dns down")
            yield  # pragma: no cover

    stt = _AlwaysFailSTT("dg-test")
    sleep_calls: list[float] = []

    async def recording_sleep(delay, *a, **kw):
        sleep_calls.append(delay)
        # collapse real sleep so the test finishes quickly
        return await _REAL_SLEEP(0)

    async with _running_meeko(fake_profiles, stt=stt, sleep=recording_sleep):
        # Let the loop churn through several reconnect attempts.
        await _wait_until(
            lambda: stt.attempts >= 4, msg="run() did not reach 4 reconnect attempts"
        )

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
    _set_env(monkeypatch, tmp_path)

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

    stt = _FailThenSucceedSTT("dg-test")

    async def zero_sleep(delay, *a, **kw):
        # Collapse every real asyncio.sleep so the grace task fires
        # immediately after a disconnect.
        return await _REAL_SLEEP(0)

    async with _running_meeko(fake_profiles, stt=stt, sleep=zero_sleep) as h:
        await _wait_until(
            lambda: stt.attempts >= 6, msg="run() did not reach 6 reconnect attempts"
        )

    # mic_stream was stopped during the outage and restarted after
    # reconnect — so start_stream() was called at least twice (initial
    # + post-cutoff resume) and stop_stream() at least twice (grace
    # cutoff + shutdown).
    assert h.mic.start_stream.call_count >= 2
    assert h.mic.stop_stream.call_count >= 2


async def test_a_dead_turn_worker_ends_run_with_an_error(
    monkeypatch, fake_profiles, tmp_path, caplog
):
    """If the long-lived turn worker dies, Meeko would keep listening but
    never answer. run() must notice, clean up, and raise, so the process
    exits non-zero and systemd restarts it."""
    _set_env(monkeypatch, tmp_path)
    died = AsyncMock(side_effect=RuntimeError("worker blew up"))

    async with _running_meeko(
        fake_profiles,
        stt=_QueueDrivenSTTClient("dg-test"),
        extra_patches=[patch("meeko.main.TurnWorker.run", new=died)],
    ) as h:
        with pytest.raises(RuntimeError, match="Turn worker stopped") as excinfo:
            await asyncio.wait_for(h.task, timeout=5)

    assert str(excinfo.value.__cause__) == "worker blew up"
    # Logged, with the cause, not just raised: a file log would otherwise
    # show a restart with no reason.
    logged = [r for r in caplog.records if "Turn worker stopped" in r.getMessage()]
    assert logged and logged[0].levelno == logging.ERROR
    assert str(logged[0].exc_info[1]) == "worker blew up"
    h.mic.close.assert_called()
    h.pa.terminate.assert_called()


async def test_a_turn_worker_ending_cancelled_still_ends_run_with_an_error(
    monkeypatch, fake_profiles, tmp_path
):
    """A CancelledError leaking out of a turn (not a barge-in) ends the
    worker cancelled. Asking it for .exception() would raise CancelledError,
    which run() swallows, and Meeko would exit 0 without a restart."""
    _set_env(monkeypatch, tmp_path)
    cancelled = AsyncMock(side_effect=asyncio.CancelledError())

    async with _running_meeko(
        fake_profiles,
        stt=_QueueDrivenSTTClient("dg-test"),
        extra_patches=[patch("meeko.main.TurnWorker.run", new=cancelled)],
    ) as h:
        with pytest.raises(RuntimeError, match="Turn worker stopped") as excinfo:
            await asyncio.wait_for(h.task, timeout=5)

    assert excinfo.value.__cause__ is None
    h.pa.terminate.assert_called()


async def test_a_turn_worker_returning_during_shutdown_is_not_a_crash(
    monkeypatch, fake_profiles, tmp_path
):
    """The worker returns normally once stop_event is set (the mic-queue
    overflow shutdown sets it). Finishing before the supervisor then is
    shutdown, not a dead worker: no "stopped unexpectedly" error."""
    _set_env(monkeypatch, tmp_path)
    release = asyncio.Event()

    async def stop_then_return(self) -> None:
        await release.wait()
        self._stop_event.set()

    async with _running_meeko(
        fake_profiles,
        stt=_QueueDrivenSTTClient("dg-test"),
        extra_patches=[patch("meeko.main.TurnWorker.run", new=stop_then_return)],
    ) as h:
        # The supervisor is mid-session, as in a real overflow, so the
        # worker's return lands first.
        await h.stt_session()
        release.set()
        await asyncio.wait_for(h.task, timeout=5)  # returns; doesn't raise

    h.pa.terminate.assert_called()


async def test_mic_queue_full_triggers_shutdown(monkeypatch, fake_profiles, tmp_path):
    """If mic_callback can't enqueue because the queue is at MIC_QUEUE_MAX,
    it logs and sets stop_event so run() exits cleanly."""
    _set_env(monkeypatch, tmp_path)
    stt = _FakeSTTClient("dg-test")
    # No events, so the session just holds open and MicPump never drains.
    stt.events = []

    async with _running_meeko(
        fake_profiles,
        stt=stt,
        sleep=None,
        extra_patches=[patch("meeko.audio_io.MIC_QUEUE_MAX", 2)],
    ) as h:
        await _wait_until(
            lambda: h.mic_cb is not None, msg="mic stream was never opened"
        )
        # Overflow the queue from the "PyAudio thread" side.
        for _ in range(10):
            h.push_mic(b"\x00\x00")
            await asyncio.sleep(0)
        # run() should exit on its own because stop_event was set.
        try:
            await asyncio.wait_for(h.task, timeout=3)
        except TimeoutError:
            pytest.fail("run() did not exit after mic_queue overflow")


async def test_run_resume_preloads_history(monkeypatch, fake_profiles, tmp_path):
    db_path = _set_env(monkeypatch, tmp_path)
    session_id = await _seed_session(db_path, assistant="prior reply")
    stt = _SignalingSTTClient()  # no turns needed — we just need run() to start up
    tts = _FakeTTSClient("dg-test")

    async with _running_meeko(
        {"query": _profile()}, stt=stt, tts=tts, resume=session_id
    ) as h:
        await asyncio.wait_for(stt.entered.wait(), timeout=5)

    assert h.claude.session_id == session_id
    assert h.claude.loaded_history == [
        {"role": "user", "content": "prior question"},
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "prior reply"}],
        },
    ]
    assert tts.calls == []


def _capture_profile_manager() -> tuple[list, object]:
    """Patch target for meeko.main.ProfileManager that records the instance
    run() builds, so a test can inspect which profile it made active."""
    built: list[ProfileManager] = []

    def make(*a, **kw):
        built.append(ProfileManager(*a, **kw))
        return built[-1]

    return built, make


async def test_run_resume_activates_the_sessions_profile(monkeypatch, tmp_path):
    """Resuming a session recorded under a non-default profile makes that
    profile active everywhere — not just its prompt, but the ProfileManager
    that drives idle behavior and list_profiles."""
    db_path = _set_env(monkeypatch, tmp_path)
    session_id = await _seed_session(db_path, profile_name="brainstorm")
    profiles = {
        "query": _profile(),
        "brainstorm": _profile("brainstorm", prompt="brainstorm system"),
    }
    built, make = _capture_profile_manager()
    stt = _SignalingSTTClient()

    with patch("meeko.main.ProfileManager", side_effect=make):
        async with _running_meeko(
            profiles, stt=stt, tts=_FakeTTSClient("dg-test"), resume=session_id
        ) as h:
            await asyncio.wait_for(stt.entered.wait(), timeout=5)

    assert h.claude.system_prompt == "brainstorm system"
    assert built[0].active_profile.name == "brainstorm"


async def test_run_resume_falls_back_when_sessions_profile_was_removed(
    monkeypatch, tmp_path, caplog
):
    """A session whose profile has since been renamed or deleted from the
    config resumes under the default profile instead of crashing startup."""
    db_path = _set_env(monkeypatch, tmp_path)
    session_id = await _seed_session(
        db_path, assistant="prior reply", profile_name="retired"
    )
    built, make = _capture_profile_manager()
    stt = _SignalingSTTClient()

    with (
        caplog.at_level(logging.WARNING, logger="meeko"),
        patch("meeko.main.ProfileManager", side_effect=make),
    ):
        async with _running_meeko(
            {"query": _profile(prompt="default system")},
            stt=stt,
            tts=_FakeTTSClient("dg-test"),
            resume=session_id,
        ) as h:
            await asyncio.wait_for(stt.entered.wait(), timeout=5)

    assert h.claude.session_id == session_id
    assert h.claude.loaded_history[-1]["content"][0]["text"] == "prior reply"
    assert h.claude.system_prompt == "default system"
    assert built[0].active_profile.name == "query"
    assert any("'retired'" in r.getMessage() for r in caplog.records)


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
        titled = await seed.create_session("conversation")
        await seed.persist_turn(titled, "user", "let's plan")
        await seed.update_session_metadata(
            session_id=titled,
            title="Planning the todo app",
            summary="s",
            transcript="t",
        )
    finally:
        await seed.close()

    with patch("meeko.main.setup_logging"):
        await meeko_main.run(list_sessions=True)

    out = capsys.readouterr().out
    assert sid in out
    assert "query" in out
    # The title is what identifies a session when picking one to --resume.
    lines = {line.split()[0]: line for line in out.splitlines()[1:]}
    assert lines[titled].endswith("Planning the todo app")
    assert lines[sid].endswith(UNTITLED)


@pytest.fixture
def new_york_tz(monkeypatch):
    """Pin local time so timestamp tests don't depend on the host's zone."""
    monkeypatch.setenv("TZ", "America/New_York")
    time.tzset()
    yield
    monkeypatch.undo()
    time.tzset()


@pytest.mark.parametrize(
    ("stored", "shown"),
    [
        # last_active is stored as UTC with microseconds.
        ("2026-09-18T01:42:29.123456+00:00", "2026-09-17 21:42 EDT"),
        # Across a DST boundary the zone name changes with the offset.
        ("2026-01-15T12:00:00+00:00", "2026-01-15 07:00 EST"),
    ],
)
def test_list_sessions_shows_last_active_in_local_time(new_york_tz, stored, shown):
    assert meeko_main._local_timestamp(stored) == shown


def test_list_sessions_keeps_an_unparseable_timestamp_visible():
    """A bad value shouldn't hide the row or crash the listing."""
    assert meeko_main._local_timestamp("not a timestamp") == "not a timestamp"


async def test_run_backfills_untitled_sessions_on_startup(
    monkeypatch, fake_profiles, tmp_path
):
    """A session left untitled-but-non-empty by a prior killed run should
    be summarized at the next startup, so `list_sessions` stops reading
    it back as '(no title)'."""
    db_path = _set_env(monkeypatch, tmp_path)
    untitled_sid = await _seed_session(
        db_path, user="leftover question", assistant="leftover answer"
    )
    stt = _SignalingSTTClient()

    async with _running_meeko(fake_profiles, stt=stt):
        await asyncio.wait_for(stt.entered.wait(), timeout=5)
        # Backfill is fire-and-forget — poll the DB until the title appears.
        title = await _wait_for_title(db_path, untitled_sid)

    # _StubAsyncAnthropic returns "stub title" for any summary call.
    assert title == "stub title"


async def test_run_gates_stt_on_wake_word(monkeypatch, fake_profiles, tmp_path):
    """With wake word enabled the orchestrator starts in IDLE and holds
    audio back from STT until the detector fires. After the detector
    fires, mic audio flows to STT (so a question spoken right after
    the wake word is transcribed) and no greeting TTS is emitted."""
    _set_env(monkeypatch, tmp_path, wake_word=True)

    # Use a holding STT client so the session stays open while the test
    # pushes post-wake chunks — otherwise the empty events list combined
    # with fast_sleep-patched asyncio.sleep causes the supervisor to
    # churn through reconnects and drop the mic chunks mid-cycle.
    stt = _HoldingSTTClient("dg-test")
    tts = _FakeTTSClient("dg-test")

    # The first process() call captures the idle-state snapshot (STT
    # count, TTS calls) so the test can assert those on the pre-wake path
    # without racing the event loop. The second call fires the wake word.
    class _SnapshotDetector(_FakeDetector):
        def __init__(self, *a, **kw):
            super().__init__()
            self.fired = asyncio.Event()
            self.first_call_snapshot: dict | None = None

        def _fires(self) -> bool:
            if self.calls == 1:
                self.first_call_snapshot = {
                    "stt": stt.session_obj.sent_audio_count,
                    "tts": list(tts.calls),
                }
                return False
            self.fired.set()
            return True

    async with _running_meeko(
        {"query": _profile()}, stt=stt, tts=tts, detector=_SnapshotDetector
    ) as h:
        # Wait for detector + mic callback + STT session to be live.
        await _wait_until(
            lambda: (
                h.mic_cb is not None
                and h.detector is not None
                and stt.session_obj is not None
            )
        )

        # Push chunks until the detector fires on the second call. A small
        # amount of real time keeps MicPump ticking under load.
        for _ in range(20):
            if h.detector.fired.is_set():
                break
            h.push_mic()
            await _REAL_SLEEP(0.02)
        await asyncio.wait_for(h.detector.fired.wait(), timeout=5)

        # First call happened while IDLE — snapshot proves STT and
        # TTS were both untouched up to that point.
        assert h.detector.first_call_snapshot == {"stt": 0, "tts": []}

        # After the wake word fires, mic chunks should flow through
        # to STT so a question spoken right after "Hey Meeko" gets
        # transcribed. No greeting TTS should be emitted.
        for _ in range(50):
            h.push_mic()
            await _REAL_SLEEP(0.02)
            if stt.session_obj.sent_audio_count > 0:
                break
        assert stt.session_obj.sent_audio_count > 0
        assert tts.calls == []


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
    db_path = _set_env(monkeypatch, tmp_path, wake_word=True)
    stt = _HoldingSTTClient("dg-test")
    stt.transcript = "Meeko stop"

    async with _running_meeko(
        {"query": _profile()},
        stt=stt,
        claude=_END_SESSION,
        detector=_FakeDetector,
    ) as h:
        # Drive: the first mic chunk fires the wake detector → LISTENING.
        # Only then release the EndOfTurn from the fake STT; otherwise it'd
        # arrive while still in IDLE and get discarded by the safety belt.
        await h.wake()
        stt.ready.set()
        # EndOfTurn("Meeko stop") fires end_session; the detector being
        # reset is our signal that the IDLE transition ran.
        await _wait_until(
            lambda: h.detector.resets >= 1, msg="wake detector was never reset"
        )

    # Claude fake got the user's stop message and its session_id was
    # cleared by reset_session (lazy creation: the next user turn would
    # create a fresh row; none happens in this test).
    assert h.claude.turns == ["Meeko stop"]
    assert h.claude.session_id is None

    # Only the original session row exists — the one the user spoke into.
    # No empty stub for the "fresh" post-rotation session under lazy
    # creation.
    rows = await _list_sessions(db_path)
    assert len(rows) == 1
    assert rows[0]["turn_count"] >= 1


async def test_post_wake_timeout_returns_to_idle_without_session_row(
    monkeypatch, fake_profiles, tmp_path
):
    """Wake word fires → LISTENING, but the user never speaks. The
    post-wake monitor expires, ends the session silently (no TTS), and
    returns to IDLE (detector reset). Since no turn happened, lazy row
    creation never fires, so no session row is written to the DB."""
    db_path = _set_env(monkeypatch, tmp_path, wake_word=True)
    tts = _FakeTTSClient("dg-test")

    # Short post-wake window; idle monitor disabled so only the
    # post-wake path can end the (turn-less) session. The user never
    # speaks: the holding STT's `ready` is never set, so no EndOfTurn.
    async with _running_meeko(
        {"query": _profile(post_wake_timeout_seconds=0.05)},
        stt=_HoldingSTTClient("dg-test"),
        tts=tts,
        detector=_FireOnceDetector,
    ) as h:
        # One mic frame fires the wake detector → LISTENING + arms the
        # post-wake monitor. We push no further frames, so the user
        # "says nothing".
        await h.wake()
        # Post-wake monitor expires → end_session → IDLE → detector reset.
        await _wait_until(
            lambda: h.detector.resets >= 1,
            msg="post-wake timeout never ended the session",
        )
        # No TTS: the silent close speaks nothing.
        assert tts.calls == []

    # No turn ever happened, so lazy row creation never fired: the DB
    # has no session row.
    assert await _list_sessions(db_path) == []


async def test_end_session_without_wake_word_transitions_to_listening(
    monkeypatch, fake_profiles, tmp_path
):
    """MEEKO_WAKE_WORD_DISABLED=1: end_session still resets the session
    but state transitions to LISTENING (no wake gate to re-arm)."""
    db_path = _set_env(monkeypatch, tmp_path)  # autouse fixture disables the gate
    stt = _HoldingSTTClient("dg-test")
    stt.transcript = "stop"

    async with _running_meeko(fake_profiles, stt=stt, claude=_END_SESSION) as h:
        # Wake word disabled → state starts in LISTENING, so we can
        # release the EndOfTurn immediately.
        await h.stt_session()
        stt.ready.set()
        await h.claude_reset()

    # Only the original (user-spoken) session row exists. Lazy creation
    # means the post-rotation session has no row until the next user turn.
    assert len(await _list_sessions(db_path)) == 1


async def test_new_session_tool_rotates_session_and_stays_listening(
    monkeypatch, fake_profiles, tmp_path
):
    """User says 'let's start fresh' → fake Claude dispatches new_session →
    after SPEAKING completes, a fresh SQLite session is created, claude
    client rebinds to it, and state stays in LISTENING (wake detector is
    NOT reset — Meeko keeps listening without requiring the wake word)."""
    db_path = _set_env(monkeypatch, tmp_path, wake_word=True)
    stt = _HoldingSTTClient("dg-test")
    stt.transcript = "let's start fresh"

    async with _running_meeko(
        {"query": _profile()},
        stt=stt,
        claude=_tool_calling_claude("Starting fresh.", ("new_session", {})),
        detector=_FakeDetector,
    ) as h:
        await h.wake()
        stt.ready.set()
        await h.claude_reset()

    # Wake detector must NOT have been reset — new_session stays in
    # LISTENING, not IDLE.
    assert h.detector.resets == 0

    # Only the original (user-spoken) session row exists. Lazy creation
    # means the rotated session has no row until the next user turn.
    assert len(await _list_sessions(db_path)) == 1
    assert h.claude.session_id is None


async def test_end_session_fires_background_summary_for_finalized_session(
    monkeypatch, fake_profiles, tmp_path
):
    """After end_session rotates the session, the background summary
    task should run and write title/summary/FTS for the *finalized*
    session (not the freshly-created one)."""
    from meeko.sessions import SessionStore

    db_path = _set_env(monkeypatch, tmp_path)  # autouse fixture disables the gate
    stt = _HoldingSTTClient("dg-test")
    stt.transcript = "stop"

    async with _running_meeko(fake_profiles, stt=stt, claude=_END_SESSION) as h:
        await h.stt_session()
        stt.ready.set()
        await h.claude_reset()

        # The finalized row is the only session in the DB at this point
        # (lazy creation means nothing else was written).
        rows = await _list_sessions(db_path)
        assert len(rows) == 1, "expected exactly one finalized session row"
        original_sid = rows[0]["id"]

        # The detached summary task awaits the stubbed anthropic client
        # (immediate) and then writes SQLite.
        await _wait_for_title(db_path, original_sid)

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
    db_path = _set_env(monkeypatch, tmp_path)  # autouse fixture disables the gate
    two_profiles = {
        "query": _profile(prompt="default system"),
        "pirate": _profile("pirate", prompt="pirate system"),
    }
    stt = _HoldingSTTClient("dg-test")
    stt.transcript = "switch to pirate and start fresh"

    async with _running_meeko(
        two_profiles,
        stt=stt,
        claude=_tool_calling_claude(
            "Okay, switching and starting fresh.",
            ("switch_profile", {"profile_name": "pirate"}),
            ("new_session", {}),
        ),
    ) as h:
        await h.stt_session()
        stt.ready.set()
        await h.claude_reset()

        # Drive a second lazy creation by invoking the create_session_fn
        # the orchestrator wired up. After switch_profile + new_session,
        # the next user-turn's lazy create must use the *switched*
        # profile.
        new_sid = await h.claude._create_session_fn()

    row = await _get_session(db_path, new_sid)
    assert row is not None
    assert row["profile_name"] == "pirate", (
        f"fresh session recorded stale profile {row['profile_name']!r}; "
        "expected 'pirate' after switch_profile"
    )


# ---------------------------------------------------------------------------
# switch_profile records the switch on the live session row
# ---------------------------------------------------------------------------


def _profile_dispatcher(session_id="sid-1", store=None):
    """The real dispatcher from _build_dispatcher over a query + pirate
    profile set, with a fake store so the row write can be inspected."""
    profiles = {
        "query": _profile(prompt="default system"),
        "pirate": _profile("pirate", prompt="pirate system"),
    }
    manager = ProfileManager(profiles, active_name="query")
    store = store if store is not None else MagicMock(set_session_profile=AsyncMock())
    dispatcher = meeko_main._build_dispatcher(
        profiles,
        manager,
        MagicMock(),
        store,
        lambda: session_id,
        meeko_main.MeekoConfig(),
    )
    return dispatcher, manager, store


async def test_switch_profile_records_the_new_profile_on_the_session():
    dispatcher, manager, store = _profile_dispatcher()

    result = await dispatcher.dispatch("switch_profile", {"profile_name": "pirate"})

    assert result == "Switched to the pirate profile."
    assert manager.active_profile.name == "pirate"
    store.set_session_profile.assert_awaited_once_with("sid-1", "pirate")


@pytest.mark.parametrize(
    ("fn_name", "args"),
    [
        ("switch_profile", {"profile_name": "query"}),  # already active
        ("switch_profile", {"profile_name": "nope"}),  # unknown profile
        ("list_profiles", {}),
    ],
)
async def test_profile_tools_that_do_not_switch_leave_the_row_alone(fn_name, args):
    dispatcher, manager, store = _profile_dispatcher()

    await dispatcher.dispatch(fn_name, args)

    assert manager.active_profile.name == "query"
    store.set_session_profile.assert_not_awaited()


async def test_switch_profile_before_any_row_exists_writes_nothing():
    """No session id means lazy creation hasn't fired yet; it will create
    the row under the now-active profile, so there is nothing to update."""
    dispatcher, manager, store = _profile_dispatcher(session_id=None)

    await dispatcher.dispatch("switch_profile", {"profile_name": "pirate"})

    assert manager.active_profile.name == "pirate"
    store.set_session_profile.assert_not_awaited()


async def test_switch_profile_survives_a_failed_row_write(caplog):
    """The user hears the switch confirmed and the in-memory switch has
    happened, so a store failure is logged rather than failing the tool."""
    failing = MagicMock(set_session_profile=AsyncMock(side_effect=OSError("disk")))
    dispatcher, manager, _ = _profile_dispatcher(store=failing)

    with caplog.at_level(logging.WARNING, logger="meeko"):
        result = await dispatcher.dispatch("switch_profile", {"profile_name": "pirate"})

    assert result == "Switched to the pirate profile."
    assert manager.active_profile.name == "pirate"
    messages = [r.getMessage() for r in caplog.records]
    assert any("Failed to record profile switch" in m for m in messages)


async def test_resume_restores_a_profile_switched_to_mid_session(monkeypatch, tmp_path):
    """The reported bug, end to end. The session's row is created under
    query when its first utterance is saved, and the switch happens after,
    during that same turn. Resuming it must come back in pirate."""
    db_path = _set_env(monkeypatch, tmp_path)  # autouse fixture disables the gate
    two_profiles = {
        "query": _profile(prompt="default system"),
        "pirate": _profile("pirate", prompt="pirate system"),
    }
    stt = _HoldingSTTClient("dg-test")
    stt.transcript = "talk like a pirate"

    async with _running_meeko(
        two_profiles,
        stt=stt,
        # end_session as a trailing marker: reset_session only runs after
        # every tool in the turn, including the row write, has finished.
        claude=_tool_calling_claude(
            "Arr.",
            ("switch_profile", {"profile_name": "pirate"}),
            ("end_session", {}),
        ),
    ) as h:
        await h.stt_session()
        stt.ready.set()
        await h.claude_reset()

    [row] = await _list_sessions(db_path)
    assert row["profile_name"] == "pirate"

    built, make = _capture_profile_manager()
    stt = _SignalingSTTClient()
    with patch("meeko.main.ProfileManager", side_effect=make):
        async with _running_meeko(
            two_profiles, stt=stt, tts=_FakeTTSClient("dg-test"), resume=row["id"]
        ) as h:
            await asyncio.wait_for(stt.entered.wait(), timeout=5)

    assert h.claude.system_prompt == "pirate system"
    assert built[0].active_profile.name == "pirate"


async def test_load_session_tool_swaps_history_and_stays_listening(
    monkeypatch, fake_profiles, tmp_path
):
    """User says 'go back to the todo session' → Sonnet calls load_session →
    after SPEAKING, claude.load_history is populated from the target session's
    turns, the client is rebound to target's session_id, and the abandoned
    session is summarized in the background so it stays in the recall index.

    The abandoned session already has turns, so session_id is non-None when
    the post-turn hook runs: this also covers the summary log branch in
    apply_post_turn_session_change."""
    db_path = _set_env(monkeypatch, tmp_path)
    target_id = await _seed_session(
        db_path,
        user="let's plan a todo app",
        assistant="Sure, let's start!",
        title="Todo app planning",
        summary="Discussed architecture options.",
        transcript="USER: let's plan a todo app\nASSISTANT: Sure, let's start!",
    )
    stt = _HoldingSTTClient("dg-test")
    stt.transcript = "go back to the todo session"

    async with _running_meeko(
        fake_profiles,
        stt=stt,
        claude=_tool_calling_claude(
            "Picking up where we left off.", ("load_session", {"id": target_id})
        ),
    ) as h:
        await h.stt_session()
        stt.ready.set()
        await _wait_until(
            lambda: h.claude is not None and h.claude.session_id == target_id
        )
        original_sid = await _abandoned_session(db_path, target_id)
        title = await _wait_for_title(db_path, original_sid)

    assert title == _StubAsyncAnthropic._STUB_SUMMARY_TITLE
    assert h.claude.session_id == target_id
    assert h.claude.loaded_history == [
        {"role": "user", "content": "let's plan a todo app"},
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "Sure, let's start!"}],
        },
    ]


async def test_end_then_load_session_fires_summary_for_current_session(
    monkeypatch, fake_profiles, tmp_path
):
    """Sonnet chains end_session + load_session in one turn.

    Equivalent in effect to load_session alone (the orchestrator always
    summarizes the abandoned session on load), but the chain must still
    succeed: claude ends up bound to the target session and the original
    session has been summarized in the background."""
    db_path = _set_env(monkeypatch, tmp_path)
    target_id = await _seed_session(db_path)
    stt = _HoldingSTTClient("dg-test")
    stt.transcript = "wrap up and go back to the prior session"

    async with _running_meeko(
        fake_profiles,
        stt=stt,
        claude=_tool_calling_claude(
            "Wrapping up and switching over.",
            ("end_session", {}),
            ("load_session", {"id": target_id}),
        ),
    ) as h:
        await h.stt_session()
        stt.ready.set()
        await _wait_until(
            lambda: h.claude is not None and h.claude.session_id == target_id
        )
        # The original session (not target) gets a summary from the stub.
        original_sid = await _abandoned_session(db_path, target_id)
        title = await _wait_for_title(db_path, original_sid)

    assert h.claude.session_id == target_id
    assert h.claude.loaded_history == [
        {"role": "user", "content": "prior question"},
        {"role": "assistant", "content": [{"type": "text", "text": "prior answer"}]},
    ]
    assert title == _StubAsyncAnthropic._STUB_SUMMARY_TITLE


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


def _stt_event(event, transcript=""):
    return SimpleNamespace(event=event, transcript=transcript)


def _logged(caplog, text) -> bool:
    return any(text in rec.getMessage() for rec in caplog.records)


async def test_puller_keeps_draining_events_while_worker_is_in_tts(
    monkeypatch, fake_profiles, tmp_path
):
    """Regression test for the queue-backpressure bug. While
    the turn worker is parked awaiting TTS, pull_stt_events must keep
    consuming events from stt_session.events() so the underlying
    websockets recv queue doesn't fill and starve pong frames."""
    _set_env(monkeypatch, tmp_path)

    # The blocking TTS keeps the turn worker parked in speak_stream.
    async with _running_meeko(
        fake_profiles, stt=_QueueDrivenSTTClient("dg-test"), tts=_BlockingTTSClient()
    ) as h:
        session = await h.stt_session()

        # Drive a turn.
        await session.event_queue.put(_stt_event("StartOfTurn"))
        await session.event_queue.put(_stt_event("EndOfTurn", "hello"))

        # Wait until the turn worker has actually entered TTS — the
        # first chunk reaching the speaker proves it.
        await asyncio.wait_for(h.first_write.wait(), timeout=5)

        # While the turn worker is parked awaiting `release`, push a
        # batch of events. If pull_stt_events were parked too (the
        # old behavior), `observed` would not grow; the queue would
        # fill and block on `put`. With the fix, all events are
        # consumed promptly.
        baseline = len(session.observed)
        for _ in range(50):
            await session.event_queue.put(_stt_event("Update", "..."))
        await _wait_until(lambda: len(session.observed) >= baseline + 50)

        # The turn worker must still be in TTS (not advanced past it).
        assert h.claude.turns == ["hello"]


async def test_endofturn_during_speaking_is_logged_as_echo(
    monkeypatch, fake_profiles, tmp_path, caplog
):
    """An EndOfTurn observed while state==SPEAKING must be logged as
    `[echo?]` and NOT drive a second Claude turn (AEC observation
    mode). The check lives in the puller — decided synchronously at
    observation time so it can't race with the worker."""
    _set_env(monkeypatch, tmp_path)
    # DEBUG: the transcript itself is privacy-sensitive and only
    # logged at that level (see tests/test_logging_privacy.py).
    caplog.set_level(logging.DEBUG, logger="meeko")

    async with _running_meeko(
        fake_profiles, stt=_QueueDrivenSTTClient("dg-test"), tts=_BlockingTTSClient()
    ) as h:
        session = await h.stt_session()
        await session.event_queue.put(_stt_event("EndOfTurn", "hello"))

        # Wait until SPEAKING (first TTS chunk reached the speaker).
        await asyncio.wait_for(h.first_write.wait(), timeout=5)

        # Now fire a second EndOfTurn — should be suppressed as echo.
        await session.event_queue.put(_stt_event("EndOfTurn", "echo leakage"))
        await _wait_until(lambda: len(session.observed) >= 2)

        # Echo turn must be logged as such, never reach Claude.
        assert _logged(caplog, "[echo?] echo leakage"), (
            "echo EndOfTurn was not logged as [echo?]"
        )
        assert h.claude.turns == ["hello"]


async def test_endofturn_while_idle_does_not_drive_a_turn(
    monkeypatch, fake_profiles, tmp_path
):
    """When the wake-word gate has not fired (state==IDLE), any
    EndOfTurn that somehow arrives must not drive a Claude turn — the
    puller's IDLE safety belt drops it."""
    _set_env(monkeypatch, tmp_path, wake_word=True)
    tts = _FakeTTSClient("dg-test")

    # The detector never fires, which keeps state pinned at IDLE.
    async with _running_meeko(
        {"query": _profile()},
        stt=_QueueDrivenSTTClient("dg-test"),
        tts=tts,
        detector=_NeverFiresDetector,
    ) as h:
        session = await h.stt_session()
        await session.event_queue.put(_stt_event("EndOfTurn", "ignored while idle"))
        await _wait_until(lambda: len(session.observed) >= 1)

        # Give the turn worker a few scheduling cycles to be wrong if it
        # were going to. It shouldn't run — turn was filtered.
        for _ in range(20):
            await _REAL_SLEEP(0.001)

        assert h.claude.turns == []
        assert tts.calls == []


async def test_start_of_turn_during_speaking_triggers_barge_in(
    monkeypatch, fake_profiles, tmp_path, caplog
):
    """A StartOfTurn observed while state==SPEAKING must cancel the
    in-flight speak task, flush the speaker buffer, and return the
    session to LISTENING. A subsequent EndOfTurn drives a fresh turn."""
    _set_env(monkeypatch, tmp_path)
    caplog.set_level(logging.INFO, logger="meeko")
    tts = _BlockingTTSClient()

    async with _running_meeko(
        fake_profiles, stt=_QueueDrivenSTTClient("dg-test"), tts=tts
    ) as h:
        session = await h.stt_session()

        # Drive the first turn into SPEAKING.
        await session.event_queue.put(_stt_event("EndOfTurn", "hello"))
        await asyncio.wait_for(h.first_write.wait(), timeout=5)

        # User barges in: StartOfTurn while SPEAKING. The speak task
        # should be cancelled even though the TTS stream is still
        # blocked on `release`.
        await session.event_queue.put(_stt_event("StartOfTurn"))
        await _wait_until(lambda: _logged(caplog, "Barge-in: turn cancelled"))

        # Barge-in mutes further writes via the audio_io flag rather
        # than touching the PortAudio stream — stop_stream from the
        # asyncio thread races a blocking write_stream still running
        # in the to_thread executor and corrupts the stream on ALSA.
        assert not h.speaker.stop_stream.called
        assert not h.speaker.start_stream.called

        # Now the user finishes their interruption. The EndOfTurn
        # arrives in LISTENING (not SPEAKING) and drives a fresh turn —
        # the barge-in utterance is the new instruction and reaches Claude.
        tts.release.set()  # unblock any residual TTS so new turn can run
        await session.event_queue.put(_stt_event("EndOfTurn", "wait, actually"))
        await _wait_until(lambda: h.claude.turns == ["hello", "wait, actually"])


async def test_start_of_turn_during_processing_triggers_barge_in(
    monkeypatch, fake_profiles, tmp_path, caplog
):
    """A StartOfTurn observed while state==PROCESSING (after EndOfTurn,
    before first audio plays — the window covers Claude TTFT plus
    Deepgram TTS first-byte synthesis) must cancel the in-flight turn
    and return the session to LISTENING so the user's interruption is
    captured rather than silently dropped."""
    _set_env(monkeypatch, tmp_path)
    caplog.set_level(logging.INFO, logger="meeko")

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

    async with _running_meeko(
        fake_profiles, stt=_QueueDrivenSTTClient("dg-test"), claude=_SlowClaude
    ) as h:
        session = await h.stt_session()

        # Drive the first turn into PROCESSING.
        await session.event_queue.put(_stt_event("EndOfTurn", "hello"))
        await asyncio.wait_for(claude_called.wait(), timeout=5)

        # No audio has played — state is still PROCESSING. User
        # barges in: StartOfTurn must cancel the in-flight turn.
        await session.event_queue.put(_stt_event("StartOfTurn"))
        await _wait_until(lambda: _logged(caplog, "Barge-in: turn cancelled"))

        # Release Claude in case any partial cleanup is awaiting it.
        release_claude.set()

        # The user finishes their interruption. The EndOfTurn arrives
        # in LISTENING (not PROCESSING/SPEAKING) and drives a fresh turn
        # — the user's change-of-mind was captured, not silently dropped.
        await session.event_queue.put(_stt_event("EndOfTurn", "wait, actually"))
        await _wait_until(lambda: h.claude.turns == ["hello", "wait, actually"])


async def test_end_of_turn_immediately_after_barge_in_is_not_dropped(
    monkeypatch, fake_profiles, tmp_path, caplog
):
    """An EndOfTurn arriving before the barge-in cancel has propagated
    through the turn worker must still be processed as a real turn, not
    dropped as `[echo?]`. request_barge_in() flips state to LISTENING
    synchronously so pull_stt_events sees the right state when it drains
    the EndOfTurn that follows StartOfTurn."""
    _set_env(monkeypatch, tmp_path)
    caplog.set_level(logging.INFO, logger="meeko")
    tts = _BlockingTTSClient()

    async with _running_meeko(
        fake_profiles, stt=_QueueDrivenSTTClient("dg-test"), tts=tts
    ) as h:
        session = await h.stt_session()

        # Drive into SPEAKING.
        await session.event_queue.put(_stt_event("EndOfTurn", "hello"))
        await asyncio.wait_for(h.first_write.wait(), timeout=5)

        # Queue StartOfTurn and EndOfTurn back-to-back so the
        # EndOfTurn is available the moment pull_stt_events finishes
        # handling the StartOfTurn. Without the synchronous state
        # flip in request_barge_in, the EndOfTurn would be processed
        # while state is still SPEAKING and dropped as `[echo?]`.
        await session.event_queue.put(_stt_event("StartOfTurn"))
        await session.event_queue.put(_stt_event("EndOfTurn", "stop"))
        tts.release.set()  # let the cancelled speak unwind

        await _wait_until(lambda: h.claude.turns == ["hello", "stop"])

        # Defensive: ensure the interruption was not logged as echo.
        assert not _logged(caplog, "[echo?] stop"), (
            "EndOfTurn that arrived during cancel propagation was dropped as echo"
        )


async def test_start_of_turn_while_listening_does_not_cancel(
    monkeypatch, fake_profiles, tmp_path, caplog
):
    """StartOfTurn while LISTENING must not trigger barge-in (there is
    nothing to cancel) and must not log a barge-in line. PROCESSING is
    a separate case — it does trigger barge-in, covered by
    test_start_of_turn_during_processing_triggers_barge_in."""
    _set_env(monkeypatch, tmp_path)
    caplog.set_level(logging.INFO, logger="meeko")

    async with _running_meeko(fake_profiles, stt=_QueueDrivenSTTClient("dg-test")) as h:
        session = await h.stt_session()

        # StartOfTurn arriving while LISTENING — no barge-in.
        await session.event_queue.put(_stt_event("StartOfTurn"))
        await session.event_queue.put(_stt_event("EndOfTurn", "hi"))
        await _wait_until(lambda: h.claude.turns == ["hi"])

        assert not _logged(caplog, "Barge-in")
        assert not h.speaker.stop_stream.called


@pytest.mark.parametrize(
    ("present", "missing"),
    [
        ({}, ["DEEPGRAM_API_KEY", "ANTHROPIC_API_KEY"]),
        ({"DEEPGRAM_API_KEY": "dg-test"}, ["ANTHROPIC_API_KEY"]),
        ({"ANTHROPIC_API_KEY": "anthropic-test"}, ["DEEPGRAM_API_KEY"]),
    ],
)
async def test_run_missing_api_key_exits_with_env_hint(
    monkeypatch, capsys, present, missing
):
    """A missing key exits 1 with a message naming exactly the missing keys
    and pointing at `.env`, not a KeyError traceback."""
    for name in ("DEEPGRAM_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    for name, value in present.items():
        monkeypatch.setenv(name, value)
    with (
        patch("meeko.main.load_dotenv"),
        patch("meeko.main.setup_logging"),
    ):
        with pytest.raises(SystemExit) as excinfo:
            await meeko_main.run()

    assert excinfo.value.code == 1
    err = capsys.readouterr().err
    assert f"Missing {', '.join(missing)}." in err
    assert ".env" in err


async def test_end_session_before_any_turn_fire_summary_is_noop(
    monkeypatch, fake_profiles, tmp_path
):
    """If end_session fires before any user turn is persisted, session_id
    is None and fire_summary must no-op (lazy creation never ran).
    Covers the `if finalized_sid is None: return` branch in fire_summary."""
    db_path = _set_env(monkeypatch, tmp_path)
    stt = _HoldingSTTClient("dg-test")
    stt.transcript = "stop"

    async with _running_meeko(
        fake_profiles,
        stt=stt,
        claude=_tool_calling_claude("Goodbye.", ("end_session", {}), persist=False),
    ) as h:
        await h.stt_session()
        stt.ready.set()
        await h.claude_reset()

    # No session row was created — lazy creation never fired and
    # fire_summary(None) was a no-op rather than crashing.
    assert await _list_sessions(db_path) == []


def test_main_registers_sigint_and_sigterm():
    """main() must register signal handlers for both SIGINT and SIGTERM."""
    signals_registered: dict[int, object] = {}

    class FakeLoop:
        def add_signal_handler(self, sig, handler):
            signals_registered[sig] = handler

        def run_until_complete(self, coro):
            coro.close()

        def close(self):
            pass

    with (
        patch("asyncio.new_event_loop", return_value=FakeLoop()),
        patch(
            "meeko.main._parse_args",
            return_value=SimpleNamespace(resume=None, list_sessions=False),
        ),
    ):
        meeko_main.main()

    assert signal.SIGINT in signals_registered
    assert signal.SIGTERM in signals_registered
