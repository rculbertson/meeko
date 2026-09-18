"""Tests for meeko.mic_pump: where each mic chunk goes, per state.

`route_chunk` takes raw bytes and a session stand-in, so the whole
routing table is reachable without PyAudio, Deepgram, or a running
orchestrator.

Uses the real StateManager so the wake-word transition under test is the
real one.
"""

import asyncio

import pytest

from meeko.leds import LedState
from meeko.mic_pump import MicPump
from meeko.state import State, StateManager

CHUNK = b"\x00\x01" * 800  # one 50ms frame of 16-bit PCM


class _RecordingLeds:
    def __init__(self):
        self.calls: list[LedState] = []

    def set_state(self, state: LedState) -> None:
        self.calls.append(state)


class _FakeIdle:
    def __init__(self):
        self.post_wake_arms = 0

    def start_post_wake(self) -> None:
        self.post_wake_arms += 1


class _FakeWakeDetector:
    """Fires on the Nth chunk it sees; never, if detect_on is None."""

    def __init__(self, detect_on: int | None = None):
        self.detect_on = detect_on
        self.seen = 0

    def process(self, data: bytes) -> bool:
        self.seen += 1
        return self.seen == self.detect_on


class _FakeSttSession:
    def __init__(self):
        self.sent: list[bytes] = []

    async def send_audio(self, data: bytes) -> None:
        self.sent.append(data)


class _FakeAudio:
    def __init__(self):
        self.mic_queue: asyncio.Queue = asyncio.Queue()


class _Harness:
    def __init__(
        self,
        state: State,
        *,
        detect_on: int | None = None,
        wake_detector: object | None = ...,
        mute_mic_while_speaking: bool = False,
    ):
        self.leds = _RecordingLeds()
        self.state_manager = StateManager(state, self.leds)
        self.idle = _FakeIdle()
        self.audio = _FakeAudio()
        self.detector = (
            _FakeWakeDetector(detect_on) if wake_detector is ... else wake_detector
        )
        self.session = _FakeSttSession()
        self.stop_event = asyncio.Event()
        self.pump = MicPump(
            audio=self.audio,
            state_manager=self.state_manager,
            idle=self.idle,
            wake_detector=self.detector,
            stop_event=self.stop_event,
            mute_mic_while_speaking=mute_mic_while_speaking,
        )
        self.leds.calls.clear()


# ---------------------------------------------------------------------------
# IDLE: chunks go to the wake detector, never to Deepgram
# ---------------------------------------------------------------------------


async def test_idle_feeds_the_detector_and_streams_nothing():
    """Nothing reaches Deepgram before the wake word — that's the whole
    point of the gate, for privacy and for billing."""
    h = _Harness(State.IDLE)

    await h.pump.route_chunk(CHUNK, h.session)

    assert h.detector.seen == 1
    assert h.session.sent == []
    assert h.state_manager.state is State.IDLE
    assert h.idle.post_wake_arms == 0


async def test_wake_word_transitions_to_listening_and_arms_post_wake():
    h = _Harness(State.IDLE, detect_on=1)

    await h.pump.route_chunk(CHUNK, h.session)

    assert h.state_manager.state is State.LISTENING
    assert h.idle.post_wake_arms == 1
    assert h.leds.calls == [LedState.LISTENING]


async def test_the_chunk_that_fires_the_wake_word_is_not_forwarded():
    """It contains the wake phrase itself, which isn't part of the
    question the user is asking."""
    h = _Harness(State.IDLE, detect_on=1)

    await h.pump.route_chunk(CHUNK, h.session)

    assert h.session.sent == []


async def test_chunks_after_the_wake_word_are_forwarded():
    h = _Harness(State.IDLE, detect_on=1)

    await h.pump.route_chunk(CHUNK, h.session)  # fires the wake word
    await h.pump.route_chunk(b"question audio", h.session)

    assert h.session.sent == [b"question audio"]
    # The detector is not consulted once out of IDLE.
    assert h.detector.seen == 1


async def test_idle_without_a_detector_raises_a_named_error():
    """Unreachable in normal wiring, but if it ever happens the error must
    say what's wrong. It surfaces through STTSupervisor, which logs only
    str(exc) on retries: a bare AssertionError there reads as an empty,
    unexplained STT failure. Pinning RuntimeError also fails the test if
    this is reverted to an assert, which `python -O` would strip."""
    h = _Harness(State.IDLE, wake_detector=None)

    with pytest.raises(RuntimeError, match="no wake-word detector"):
        await h.pump.route_chunk(CHUNK, h.session)

    assert h.session.sent == []


# ---------------------------------------------------------------------------
# SPEAKING: the AEC / no-AEC split
# ---------------------------------------------------------------------------


async def test_speaking_drops_chunks_when_muting_is_enabled():
    """No-hardware-AEC path: forwarding here would make Deepgram hear the
    assistant's own voice and report it as a user turn."""
    h = _Harness(State.SPEAKING, mute_mic_while_speaking=True)

    await h.pump.route_chunk(CHUNK, h.session)

    assert h.session.sent == []


async def test_speaking_forwards_chunks_when_muting_is_disabled():
    """Default path: hardware AEC keeps playback out of the mic signal,
    and keeping the mic hot is what makes barge-in possible."""
    h = _Harness(State.SPEAKING, mute_mic_while_speaking=False)

    await h.pump.route_chunk(CHUNK, h.session)

    assert h.session.sent == [CHUNK]


@pytest.mark.parametrize("state", [State.LISTENING, State.PROCESSING])
async def test_muting_only_applies_to_speaking(state):
    h = _Harness(state, mute_mic_while_speaking=True)

    await h.pump.route_chunk(CHUNK, h.session)

    assert h.session.sent == [CHUNK]


@pytest.mark.parametrize("state", [State.LISTENING, State.PROCESSING])
async def test_non_idle_states_forward_to_stt(state):
    h = _Harness(state)

    await h.pump.route_chunk(CHUNK, h.session)

    assert h.session.sent == [CHUNK]
    assert h.detector.seen == 0


# ---------------------------------------------------------------------------
# The drain loop
# ---------------------------------------------------------------------------


async def test_run_drains_queued_chunks_in_order():
    h = _Harness(State.LISTENING)
    for chunk in (b"one", b"two", b"three"):
        h.audio.mic_queue.put_nowait(chunk)

    task = asyncio.create_task(h.pump.run(h.session))
    while len(h.session.sent) < 3:
        await asyncio.sleep(0.01)
    h.stop_event.set()
    await task

    assert h.session.sent == [b"one", b"two", b"three"]


async def test_run_returns_promptly_when_the_queue_stays_empty():
    """The mic legitimately goes quiet during an STT outage, so shutdown
    can't depend on a chunk arriving to unblock the loop."""
    h = _Harness(State.LISTENING)
    h.stop_event.set()

    await asyncio.wait_for(h.pump.run(h.session), timeout=1.0)

    assert h.session.sent == []


async def test_run_keeps_polling_across_an_empty_window():
    """A queue timeout must continue the loop, not end the pump."""
    h = _Harness(State.LISTENING)

    task = asyncio.create_task(h.pump.run(h.session))
    await asyncio.sleep(0.15)  # longer than the 0.1s queue poll timeout
    assert not task.done()

    h.audio.mic_queue.put_nowait(b"late chunk")
    while not h.session.sent:
        await asyncio.sleep(0.01)
    h.stop_event.set()
    await task

    assert h.session.sent == [b"late chunk"]
