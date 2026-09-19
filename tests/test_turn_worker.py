"""Tests for meeko.orchestrator.turn_worker: the turn worker and barge-in.

The turn is a fake coroutine that blocks on an Event, so each test can put
a cancel exactly where it wants one: mid-turn, between turns, or in the
window after a turn finishes but before the worker resumes. That makes it
possible to test the barge-in vs. shutdown distinction directly. In the
run() drive-throughs in test_main.py a broken flag mostly shows up
as a hang or a quietly dead worker.

Uses the real StateManager (with a recording LED stand-in) rather than a
mock, so the state transitions under test are the real ones.

A worker that's still running when it should have exited, or has died
when it should still be running, is the failure these tests look for.
`_finished` waits with a timeout rather than awaiting outright, so a
swallowed shutdown cancel fails the assertion instead of hanging the
suite.
"""

import asyncio

import pytest

from meeko.leds import LedState
from meeko.orchestrator.idle import IDLE_TIMEOUT_SENTINEL
from meeko.orchestrator.state import State, StateManager
from meeko.orchestrator.turn_worker import TurnWorker

ERROR = "error"


class _RecordingLeds:
    """Records state cues and error flashes in one list, so tests can
    check their relative order."""

    def __init__(self):
        self.calls: list[LedState | str] = []

    def set_state(self, state: LedState) -> None:
        self.calls.append(state)

    def error(self) -> None:
        self.calls.append(ERROR)


class _FakeIdle:
    def __init__(self):
        self.cancels = 0
        self.post_turn_starts = 0

    def cancel(self) -> None:
        self.cancels += 1

    def start_post_turn(self) -> None:
        self.post_turn_starts += 1


class _Harness:
    """A TurnWorker wired to fakes.

    By default each turn blocks until `release` is set. Tests replace
    `turn` for other behaviour; `_start_turn` records the text
    synchronously, the way `claude.stream_turn(text)` is called before
    the sub-task exists in production.
    """

    def __init__(
        self,
        state: State = State.LISTENING,
        next_state: State = State.LISTENING,
    ):
        self.leds = _RecordingLeds()
        self.state_manager = StateManager(state, self.leds)
        self.idle = _FakeIdle()
        self.turn_queue: asyncio.Queue = asyncio.Queue()
        self.stop_event = asyncio.Event()
        self.next_state = next_state
        self.session_change_error: Exception | None = None
        self.session_changes = 0
        self.started: list[str] = []
        self.cancelled: list[str] = []
        self.completed: list[str] = []
        self.release = asyncio.Event()
        self.turn = self._blocking_turn
        self.worker = TurnWorker(
            turn_queue=self.turn_queue,
            state_manager=self.state_manager,
            idle=self.idle,
            leds=self.leds,
            start_turn=self._start_turn,
            apply_session_change=self._apply_session_change,
            stop_event=self.stop_event,
        )
        self.task: asyncio.Task | None = None

    def _start_turn(self, text: str):
        self.started.append(text)
        return self.turn(text)

    async def _blocking_turn(self, text: str) -> None:
        try:
            await self.release.wait()
        except asyncio.CancelledError:
            self.cancelled.append(text)
            raise
        self.completed.append(text)

    async def _apply_session_change(self) -> State:
        self.session_changes += 1
        if self.session_change_error is not None:
            raise self.session_change_error
        return self.next_state

    def start(self) -> asyncio.Task:
        self.task = asyncio.create_task(self.worker.run())
        return self.task

    async def submit(self, text: str) -> None:
        """Queue a turn and wait until the worker has started it."""
        n = len(self.started)
        await self.turn_queue.put(text)
        await _wait_until(lambda: len(self.started) > n)

    async def idle_again(self) -> None:
        """Wait until the worker is back on the queue with nothing in flight."""
        await _wait_until(lambda: self.turn_queue.empty() and self._turn_settled())
        await _yield(5)

    def _turn_settled(self) -> bool:
        # done(), not `is None`: clearing the handle isn't behaviour, and
        # waiting on it would make those lines look tested when they aren't.
        task = self.worker._current_speak_task
        return task is None or task.done()


async def _yield(n: int = 1) -> None:
    for _ in range(n):
        await asyncio.sleep(0)


async def _wait_until(predicate, timeout: float = 1.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not predicate():
        if loop.time() >= deadline:
            raise TimeoutError("predicate never became true")
        await asyncio.sleep(0.001)


async def _finished(task: asyncio.Task, timeout: float = 0.5) -> bool:
    done, _ = await asyncio.wait({task}, timeout=timeout)
    return task in done


@pytest.fixture
async def harnesses():
    """Build harnesses and cancel any worker a test leaves running."""
    made: list[_Harness] = []

    def make(**kwargs) -> _Harness:
        h = _Harness(**kwargs)
        made.append(h)
        return h

    yield make
    for h in made:
        h.release.set()
        if h.task is not None and not h.task.done():
            h.task.cancel()
            await asyncio.wait({h.task}, timeout=1.0)


# ---------------------------------------------------------------------------
# Normal turns
# ---------------------------------------------------------------------------


async def test_turn_runs_in_processing_then_applies_session_change(harnesses):
    h = harnesses()
    h.start()

    await h.submit("hello")
    assert h.state_manager.state is State.PROCESSING
    assert h.session_changes == 0

    h.release.set()
    await h.idle_again()

    assert h.completed == ["hello"]
    assert h.session_changes == 1
    assert h.idle.post_turn_starts == 1
    assert h.state_manager.state is State.LISTENING
    assert not h.task.done()


async def test_dequeuing_a_turn_cancels_the_idle_window(harnesses):
    h = harnesses()
    h.start()

    await h.submit("hello")

    assert h.idle.cancels == 1


async def test_session_ending_in_idle_does_not_arm_the_idle_window(harnesses):
    """IDLE means the session is over and the next interaction needs the
    wake word, so there's no post-turn window to run."""
    h = harnesses(next_state=State.IDLE)
    h.release.set()
    h.start()

    await h.submit("goodbye")
    await h.idle_again()

    assert h.session_changes == 1
    assert h.idle.post_turn_starts == 0
    assert h.state_manager.state is State.IDLE


async def test_idle_sentinel_applies_session_change_without_a_turn(harnesses):
    h = harnesses(next_state=State.IDLE)
    h.start()

    await h.turn_queue.put(IDLE_TIMEOUT_SENTINEL)
    await _wait_until(lambda: h.session_changes == 1)
    await _yield(5)

    assert h.started == []
    assert h.idle.cancels == 1
    assert h.idle.post_turn_starts == 0
    assert h.state_manager.state is State.IDLE

    # The worker is still serving turns afterwards.
    await h.submit("hey")
    assert h.started == ["hey"]


async def test_failed_turn_flashes_error_after_listening_and_keeps_worker(
    harnesses,
):
    """LISTENING must reach the LED worker before the error flash — the
    reverse order makes the flash abort immediately (see the comment in
    TurnWorker.run)."""

    async def failing_turn(text: str) -> None:
        raise RuntimeError("Claude fell over")

    h = harnesses()
    h.turn = failing_turn
    h.start()

    await h.submit("hello")
    await h.idle_again()

    assert h.leds.calls[-2:] == [LedState.LISTENING, ERROR]
    assert h.state_manager.state is State.LISTENING
    assert h.session_changes == 0
    assert h.idle.post_turn_starts == 0
    assert not h.task.done()

    h.turn = h._blocking_turn
    await h.submit("again")
    assert h.started == ["hello", "again"]


async def test_a_failing_session_change_ends_the_worker(harnesses):
    """Deliberately not caught: a failure here takes a bug or a failing
    local database, both rare. The worker ends with the exception, and
    run() turns that into a non-zero exit so systemd restarts Meeko with
    clean state (see test_main.py)."""
    h = harnesses()
    h.session_change_error = RuntimeError("bug in the hook")
    h.start()

    await h.submit("hello")
    h.release.set()

    assert await _finished(h.task)
    with pytest.raises(RuntimeError, match="bug in the hook"):
        h.task.result()


# ---------------------------------------------------------------------------
# Barge-in
# ---------------------------------------------------------------------------


async def test_barge_in_cancels_turn_and_keeps_worker_running(harnesses):
    h = harnesses()
    h.start()
    await h.submit("hello")

    h.worker.request_barge_in()
    await _wait_until(lambda: h.cancelled == ["hello"])
    await h.idle_again()

    assert not h.task.done()
    # A cancelled turn has no reply to act on: no session change, no
    # idle window.
    assert h.session_changes == 0
    assert h.idle.post_turn_starts == 0
    assert h.state_manager.state is State.LISTENING

    # The interruption itself is the next turn.
    await h.submit("wait, actually")
    assert h.started == ["hello", "wait, actually"]


async def test_barge_in_flips_to_listening_before_any_await(harnesses):
    """The router drains the EndOfTurn that follows a StartOfTurn without
    yielding in between, so the state must already be LISTENING when
    request_barge_in returns, or that EndOfTurn is dropped as echo."""
    h = harnesses()
    h.start()
    await h.submit("hello")
    assert h.state_manager.state is State.PROCESSING
    cancels_before = h.idle.cancels

    h.worker.request_barge_in()

    assert h.state_manager.state is State.LISTENING
    assert h.idle.cancels == cancels_before + 1


async def test_barge_in_with_no_turn_in_flight_still_flips_to_listening(
    harnesses,
):
    """A timer's expiry announcement ("The 5 second timer is done!")
    enters SPEAKING through speaker.speak(), outside the worker, so
    there's nothing to cancel. The user's interruption must still be
    captured rather than dropped as echo."""
    h = harnesses(state=State.SPEAKING)
    h.start()
    await _yield()

    h.worker.request_barge_in()

    assert h.state_manager.state is State.LISTENING
    assert h.idle.cancels == 1

    # And it must not leave a barge-in pending: the next turn's shutdown
    # cancel still has to propagate.
    await h.submit("hello")
    h.task.cancel()
    assert await _finished(h.task)
    assert h.task.cancelled()


async def test_state_is_listening_after_a_barge_in_unwinds(harnesses):
    """speak_stream's finally calls exit_speaking(prev) with the state it
    entered from, PROCESSING, which undoes request_barge_in's LISTENING.
    The worker has to put LISTENING back."""
    h = harnesses()
    speaking = asyncio.Event()

    async def speaking_turn(text: str) -> None:
        prev = h.state_manager.enter_speaking()
        try:
            speaking.set()
            await asyncio.Event().wait()
        finally:
            h.state_manager.exit_speaking(prev)

    h.turn = speaking_turn
    h.start()
    await h.turn_queue.put("hello")
    await asyncio.wait_for(speaking.wait(), timeout=1.0)
    assert h.state_manager.state is State.SPEAKING

    h.worker.request_barge_in()
    await h.idle_again()

    assert h.state_manager.state is State.LISTENING
    assert not h.task.done()


# ---------------------------------------------------------------------------
# Shutdown
# ---------------------------------------------------------------------------


async def test_shutdown_while_waiting_for_a_turn_propagates(harnesses):
    h = harnesses()
    h.start()
    await _yield()

    h.task.cancel()

    assert await _finished(h.task)
    assert h.task.cancelled()


async def test_shutdown_mid_turn_cancels_the_turn_and_propagates(harnesses):
    """Without a barge-in flag, a CancelledError out of the turn is a
    shutdown: the worker must exit, not loop back for the next turn."""
    h = harnesses()
    h.start()
    await h.submit("hello")

    h.task.cancel()

    assert await _finished(h.task)
    assert h.task.cancelled()
    assert h.cancelled == ["hello"]
    assert h.state_manager.state is State.LISTENING


async def test_shutdown_after_a_barge_in_still_propagates(harnesses):
    """The barge-in flag is consumed by the cancel it was set for. If it
    stuck, the next shutdown would be mistaken for a barge-in."""
    h = harnesses()
    h.start()
    await h.submit("first")
    h.worker.request_barge_in()
    await _wait_until(lambda: h.cancelled == ["first"])
    await h.idle_again()

    await h.submit("second")
    h.task.cancel()

    assert await _finished(h.task)
    assert h.task.cancelled()


async def test_barge_in_after_turn_finished_is_not_a_barge_in(harnesses):
    """There's a window where the turn task is done but the worker hasn't
    resumed yet. A StartOfTurn landing there must not set the barge-in
    flag: no cancel is coming to consume it, so it would stay set and
    swallow the next shutdown."""
    h = harnesses()
    turn_returning = asyncio.Event()

    async def quick_turn(text: str) -> None:
        # No await after this: the task finishes in the same step, and
        # the worker's wake-up is queued behind ours.
        turn_returning.set()

    h.turn = quick_turn
    h.start()
    await h.turn_queue.put("first")
    await asyncio.wait_for(turn_returning.wait(), timeout=1.0)

    # Precondition: we really are in the window.
    speak_task = h.worker._current_speak_task
    assert speak_task is not None and speak_task.done()

    h.worker.request_barge_in()
    await h.idle_again()
    assert h.session_changes == 1  # the turn completed normally

    h.turn = h._blocking_turn
    await h.submit("second")
    h.task.cancel()

    assert await _finished(h.task)
    assert h.task.cancelled()


async def test_worker_exits_after_the_in_flight_turn_once_stopping(harnesses):
    """run()'s finally sets stop_event before cancelling the worker. A
    worker that sees it between turns returns rather than waiting on an
    empty queue."""
    h = harnesses()
    h.start()
    await h.submit("hello")

    h.stop_event.set()
    h.release.set()

    assert await _finished(h.task)
    assert not h.task.cancelled()
    assert h.completed == ["hello"]
