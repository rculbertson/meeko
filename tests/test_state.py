"""Tests for meeko.orchestrator.state: the orchestrator state machine and
its LED mirror.

StateManager is the single writer for both the logical state and the LED
ring, so these assert the two can't drift apart.
"""

from meeko.leds import LedState
from meeko.orchestrator.state import State, StateManager


class _RecordingLeds:
    """Stands in for LedController — only set_state is exercised here."""

    def __init__(self):
        self.calls: list[LedState] = []

    def set_state(self, state: LedState) -> None:
        self.calls.append(state)


def _manager(initial: State = State.IDLE) -> tuple[StateManager, _RecordingLeds]:
    leds = _RecordingLeds()
    return StateManager(initial, leds), leds


def test_state_enum_members():
    assert {s.name for s in State} == {"IDLE", "LISTENING", "PROCESSING", "SPEAKING"}


def test_set_updates_state_and_pushes_the_matching_led_cue():
    manager, leds = _manager()

    manager.set(State.LISTENING)

    assert manager.state is State.LISTENING
    assert leds.calls == [LedState.LISTENING]


def test_set_to_the_current_state_still_reasserts_the_led():
    """The ring is write-only hardware — re-asserting is how a cue that
    was overwritten (e.g. by the error flash) gets restored."""
    manager, leds = _manager(State.LISTENING)

    manager.set(State.LISTENING)
    manager.set(State.LISTENING)

    assert manager.state is State.LISTENING
    assert leds.calls == [LedState.LISTENING, LedState.LISTENING]


def test_every_state_maps_to_an_led_cue():
    """A new State member without an LED mapping would KeyError at
    runtime, inside the state transition."""
    for state in State:
        manager, leds = _manager()
        manager.set(state)
        assert len(leds.calls) == 1


def test_enter_speaking_returns_the_previous_state():
    manager, _ = _manager(State.PROCESSING)

    prev = manager.enter_speaking()

    assert prev is State.PROCESSING
    assert manager.state is State.SPEAKING


def test_exit_speaking_restores_the_previous_state():
    manager, _ = _manager(State.PROCESSING)

    prev = manager.enter_speaking()
    manager.exit_speaking(prev)

    assert manager.state is State.PROCESSING


def test_exit_speaking_falls_back_to_listening_when_prev_was_speaking():
    """Guards against a nested speak (e.g. a timer chime firing mid-reply)
    restoring SPEAKING and stranding the session there with nothing
    playing."""
    manager, _ = _manager(State.SPEAKING)

    manager.exit_speaking(State.SPEAKING)

    assert manager.state is State.LISTENING


def test_set_listening_active_switches_the_led_without_changing_state():
    manager, leds = _manager(State.LISTENING)
    leds.calls.clear()

    manager.set_listening_active(True)
    assert manager.state is State.LISTENING
    assert leds.calls == [LedState.LISTENING_ACTIVE]

    manager.set_listening_active(False)
    assert manager.state is State.LISTENING
    assert leds.calls == [LedState.LISTENING_ACTIVE, LedState.LISTENING]


def test_set_listening_active_is_a_noop_outside_listening():
    """Otherwise the ring would show a LISTENING cue while the session is
    actually PROCESSING or SPEAKING — the LED lying about the state."""
    for state in (State.IDLE, State.PROCESSING, State.SPEAKING):
        manager, leds = _manager(state)
        leds.calls.clear()

        manager.set_listening_active(True)

        assert manager.state is state
        assert leds.calls == []
