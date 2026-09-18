"""Tests for meeko.orchestrator.stt_events: what each STT event means per state.

Routing is a decision table over (event, state), so these feed TurnEvent
objects straight to `handle()`. No STT session, no Claude, no audio — the
whole table used to require driving `run()` with fakes for all three.

Uses the real StateManager (with a recording LED stand-in) rather than a
mock, so the state transitions under test are the real ones.
"""

import asyncio

import pytest

from meeko.deepgram_stt import TurnEvent
from meeko.leds import LedState
from meeko.orchestrator.state import State, StateManager
from meeko.orchestrator.stt_events import SttEventRouter


class _RecordingLeds:
    def __init__(self):
        self.calls: list[LedState] = []

    def set_state(self, state: LedState) -> None:
        self.calls.append(state)


class _FakeIdle:
    def __init__(self):
        self.cancels = 0

    def cancel(self) -> None:
        self.cancels += 1


class _Harness:
    def __init__(self, state: State):
        self.leds = _RecordingLeds()
        self.state_manager = StateManager(state, self.leds)
        self.idle = _FakeIdle()
        self.turn_queue: asyncio.Queue = asyncio.Queue()
        self.barge_ins = 0
        self.stop_event = asyncio.Event()
        self.router = SttEventRouter(
            state_manager=self.state_manager,
            idle=self.idle,
            turn_queue=self.turn_queue,
            request_barge_in=self._barge_in,
            stop_event=self.stop_event,
        )
        self.leds.calls.clear()

    def _barge_in(self) -> None:
        self.barge_ins += 1
        # Mirror the real request_barge_in's synchronous state flip.
        self.state_manager.set(State.LISTENING)

    def queued(self) -> list[str]:
        out = []
        while not self.turn_queue.empty():
            out.append(self.turn_queue.get_nowait())
        return out


def _start() -> TurnEvent:
    return TurnEvent(event="StartOfTurn", transcript="")


def _end(transcript: str) -> TurnEvent:
    return TurnEvent(event="EndOfTurn", transcript=transcript)


# ---------------------------------------------------------------------------
# StartOfTurn
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("state", [State.SPEAKING, State.PROCESSING])
async def test_start_of_turn_barges_in_while_replying(state):
    """SPEAKING is the obvious case; PROCESSING covers the user changing
    their mind during Claude TTFT, before any audio exists."""
    h = _Harness(state)

    await h.router.handle(_start())

    assert h.barge_ins == 1
    assert h.idle.cancels == 1
    assert h.leds.calls[-1] == LedState.LISTENING_ACTIVE


async def test_start_of_turn_while_listening_does_not_barge_in():
    h = _Harness(State.LISTENING)

    await h.router.handle(_start())

    assert h.barge_ins == 0
    assert h.idle.cancels == 1
    assert h.leds.calls == [LedState.LISTENING_ACTIVE]


async def test_start_of_turn_while_idle_only_cancels_the_idle_window():
    """Nothing to barge in on, and the ring must stay dark — showing a
    LISTENING cue in IDLE would claim Meeko is awake when it isn't."""
    h = _Harness(State.IDLE)

    await h.router.handle(_start())

    assert h.barge_ins == 0
    assert h.idle.cancels == 1
    assert h.leds.calls == []
    assert h.state_manager.state is State.IDLE


# ---------------------------------------------------------------------------
# EndOfTurn
# ---------------------------------------------------------------------------


async def test_end_of_turn_queues_the_transcript():
    h = _Harness(State.LISTENING)

    await h.router.handle(_end("what's the weather"))

    assert h.queued() == ["what's the weather"]
    assert h.leds.calls == [LedState.LISTENING]


async def test_end_of_turn_strips_surrounding_whitespace():
    h = _Harness(State.LISTENING)

    await h.router.handle(_end("  hello there \n"))

    assert h.queued() == ["hello there"]


async def test_end_of_turn_while_idle_does_not_start_a_turn():
    """Mic audio is gated behind the wake word; a stray transcript here
    must not start a conversation."""
    h = _Harness(State.IDLE)

    await h.router.handle(_end("stray transcript"))

    assert h.queued() == []
    assert h.leds.calls == []


async def test_end_of_turn_while_speaking_is_dropped_as_echo():
    """Reaching this branch means no StartOfTurn triggered a barge-in, so
    with AEC on this is the assistant's own voice coming back."""
    h = _Harness(State.SPEAKING)

    await h.router.handle(_end("...the forecast calls for rain"))

    assert h.queued() == []
    assert h.leds.calls == []


async def test_end_of_turn_while_processing_is_queued():
    """The post-barge-in path: request_barge_in flips to LISTENING, but a
    turn arriving while still PROCESSING must not be dropped."""
    h = _Harness(State.PROCESSING)

    await h.router.handle(_end("actually, never mind"))

    assert h.queued() == ["actually, never mind"]


async def test_empty_transcript_is_not_queued_but_still_clears_the_led():
    """Deepgram closes turns on silence too. The ring must drop back to
    the steady cue or it stays on "hearing you" with nobody speaking."""
    h = _Harness(State.LISTENING)

    await h.router.handle(_end("   "))

    assert h.queued() == []
    assert h.leds.calls == [LedState.LISTENING]


# ---------------------------------------------------------------------------
# Other events / the drain loop
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("event", ["Update", "EagerEndOfTurn", "TurnResumed", "Junk"])
async def test_other_event_types_are_ignored(event):
    h = _Harness(State.LISTENING)

    await h.router.handle(TurnEvent(event=event, transcript="partial text"))

    assert h.queued() == []
    assert h.barge_ins == 0
    assert h.idle.cancels == 0
    assert h.leds.calls == []


class _FakeSession:
    def __init__(self, events: list[TurnEvent]):
        self._events = events

    async def events(self):
        for ev in self._events:
            yield ev


async def test_run_drains_every_event_in_order():
    h = _Harness(State.LISTENING)
    session = _FakeSession([_start(), _end("first"), _start(), _end("second")])

    await h.router.run(session)

    assert h.queued() == ["first", "second"]


async def test_run_stops_once_stop_event_is_set():
    """Shutdown must not keep queueing turns nobody will drain."""
    h = _Harness(State.LISTENING)
    h.stop_event.set()
    session = _FakeSession([_end("too late")])

    await h.router.run(session)

    assert h.queued() == []
