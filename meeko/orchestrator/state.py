"""The orchestrator's state machine and its LED mirror.

`IDLE → (wake word) → LISTENING → PROCESSING → SPEAKING`, with barge-in
returning to LISTENING. See docs/architecture.md §5.1 for the full diagram
and the three paths back to IDLE.

`StateManager` is the single writer: it holds the current state and
pushes the matching cue to the LED ring on every change, so the ring can
never disagree with the logical state. Lives here rather than in
`meeko/main.py` so the modules that branch on `State` — the mic pump,
the STT event router, the turn worker — can import it without a cycle
back through `meeko/main.py`.
"""

import logging
from enum import Enum, auto
from typing import ClassVar

from meeko.leds import LedController, LedState

logger = logging.getLogger("meeko")


class State(Enum):
    IDLE = auto()
    LISTENING = auto()
    PROCESSING = auto()
    SPEAKING = auto()


class StateManager:
    _STATE_TO_LED: ClassVar[dict[State, LedState]] = {
        State.IDLE: LedState.IDLE,
        State.LISTENING: LedState.LISTENING,
        State.PROCESSING: LedState.PROCESSING,
        State.SPEAKING: LedState.SPEAKING,
    }

    def __init__(self, initial: State, leds: LedController) -> None:
        self._state = initial
        self._leds = leds

    @property
    def state(self) -> State:
        return self._state

    def set(self, new: State) -> None:
        if new != self._state:
            # Content-free, one line per transition: this is the trace
            # that makes a journal at the default INFO level readable.
            logger.info("[state] %s → %s", self._state.name, new.name)
            self._state = new
        self._leds.set_state(self._STATE_TO_LED[new])

    def enter_speaking(self) -> State:
        prev = self._state
        self.set(State.SPEAKING)
        return prev

    def exit_speaking(self, prev: State) -> None:
        self.set(prev if prev != State.SPEAKING else State.LISTENING)

    def set_listening_active(self, active: bool) -> None:
        """LISTENING_ACTIVE is a LED sub-state of LISTENING (brighter
        cyan while the user is actually speaking, vs. the steady cyan
        "ready" cue). The orchestrator stays in State.LISTENING either
        way — only the LED display changes. Calls from outside
        LISTENING are no-ops so the LED never diverges from the
        logical state."""
        if self._state != State.LISTENING:
            return
        self._leds.set_state(
            LedState.LISTENING_ACTIVE if active else LedState.LISTENING
        )
