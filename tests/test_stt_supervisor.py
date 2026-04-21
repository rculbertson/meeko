"""Unit tests for `meeko.stt_supervisor.STTSupervisor`."""

import asyncio
import contextlib
from unittest.mock import MagicMock, patch

import pytest

from meeko import stt_supervisor as supervisor_mod
from meeko.stt_supervisor import STTSupervisor


@pytest.fixture
def audio_mock():
    audio = MagicMock()
    audio.mic_capturing = True  # default: mic already running
    return audio


async def _drive_until(task, predicate, *, tries=200):
    """Yield to the loop until predicate() is true or tries exhausts."""
    for _ in range(tries):
        if predicate():
            return
        await asyncio.sleep(0)
    raise AssertionError("predicate never became true")


class _OneShotSTT:
    """STT whose session() yields once, runs on_session, then the
    supervisor should exit when stop_event is set from outside."""

    def __init__(self):
        self.session_count = 0

    @contextlib.asynccontextmanager
    async def session(self):
        self.session_count += 1
        yield MagicMock()


async def test_happy_path_runs_on_session_then_exits_on_stop(audio_mock):
    stop = asyncio.Event()
    stt = _OneShotSTT()
    on_session_calls = []

    async def on_session(sess):
        on_session_calls.append(sess)
        stop.set()

    sup = STTSupervisor(stt, audio_mock, stop, on_session, lambda: False)
    await asyncio.wait_for(sup.run(), timeout=2)

    assert stt.session_count == 1
    assert len(on_session_calls) == 1


class _FailThenSucceedSTT:
    def __init__(self, fail_count: int):
        self.attempts = 0
        self._fail_count = fail_count

    @contextlib.asynccontextmanager
    async def session(self):
        self.attempts += 1
        if self.attempts <= self._fail_count:
            raise OSError("dns")
        yield MagicMock()


async def test_reconnects_with_exponential_backoff(audio_mock):
    stop = asyncio.Event()
    stt = _FailThenSucceedSTT(fail_count=4)
    sleep_calls: list[float] = []
    real_sleep = asyncio.sleep

    async def recording_sleep(delay, *a, **kw):
        sleep_calls.append(delay)
        return await real_sleep(0)

    async def on_session(sess):
        stop.set()

    sup = STTSupervisor(stt, audio_mock, stop, on_session, lambda: False)
    with patch("meeko.stt_supervisor.asyncio.sleep", new=recording_sleep):
        await asyncio.wait_for(sup.run(), timeout=2)

    backoff_delays = [d for d in sleep_calls if d != supervisor_mod.RECONNECT_GRACE_S]
    assert backoff_delays[:4] == [0.5, 1, 2, 4]
    assert stt.attempts == 5  # 4 failures + 1 success


async def test_restarts_mic_after_grace_cutoff(audio_mock):
    """If the outage exceeds RECONNECT_GRACE_S the supervisor stops the
    mic; on successful reconnect it resumes capture."""
    stop = asyncio.Event()
    stt = _FailThenSucceedSTT(fail_count=5)
    real_sleep = asyncio.sleep

    # Collapse every sleep so grace fires immediately.
    async def fast_sleep(delay, *a, **kw):
        return await real_sleep(0)

    # Track mic_capturing transitions driven by supervisor.
    def on_start():
        audio_mock.mic_capturing = True

    def on_stop():
        audio_mock.mic_capturing = False

    audio_mock.start_mic.side_effect = on_start
    audio_mock.stop_mic.side_effect = on_stop
    audio_mock.mic_capturing = True  # running at the start

    async def on_session(sess):
        stop.set()

    sup = STTSupervisor(stt, audio_mock, stop, on_session, lambda: False)
    with patch("meeko.stt_supervisor.asyncio.sleep", new=fast_sleep):
        await asyncio.wait_for(sup.run(), timeout=2)

    # Grace cutoff stopped the mic during the outage; reconnect restarted it.
    audio_mock.stop_mic.assert_called()
    audio_mock.start_mic.assert_called()


async def test_drains_mic_on_reconnect_if_was_speaking(audio_mock):
    """If the disconnect happened during SPEAKING, on reconnect the
    supervisor drains the mic queue so TTS echo doesn't flush as a
    phantom user turn."""
    stop = asyncio.Event()
    stt = _FailThenSucceedSTT(fail_count=1)
    real_sleep = asyncio.sleep

    async def fast_sleep(delay, *a, **kw):
        return await real_sleep(0)

    async def on_session(sess):
        stop.set()

    # is_speaking() returns True at the moment the first attempt fails.
    was_speaking = [True]

    def is_speaking():
        return was_speaking[0]

    sup = STTSupervisor(stt, audio_mock, stop, on_session, is_speaking)
    with patch("meeko.stt_supervisor.asyncio.sleep", new=fast_sleep):
        await asyncio.wait_for(sup.run(), timeout=2)

    audio_mock.drain_mic_queue.assert_called()


async def test_exits_when_stop_event_set_before_connect(audio_mock):
    """Setting stop_event before run() starts should make it return
    without opening any session."""
    stop = asyncio.Event()
    stop.set()
    stt = _OneShotSTT()

    async def on_session(sess):
        pytest.fail("on_session should not be called")

    sup = STTSupervisor(stt, audio_mock, stop, on_session, lambda: False)
    await asyncio.wait_for(sup.run(), timeout=1)

    assert stt.session_count == 0


async def test_cancellation_propagates(audio_mock):
    """CancelledError raised mid-session must bubble out of run()."""
    stop = asyncio.Event()

    class _BlockingSTT:
        @contextlib.asynccontextmanager
        async def session(self):
            yield MagicMock()

    async def on_session(sess):
        await asyncio.sleep(10)  # block until cancelled

    sup = STTSupervisor(_BlockingSTT(), audio_mock, stop, on_session, lambda: False)
    task = asyncio.create_task(sup.run())
    await asyncio.sleep(0)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
