"""Routing for Deepgram STT turn events.

Decides what each `StartOfTurn` / `EndOfTurn` means given the session's
current state, and queues the transcripts that should actually drive a
Claude turn.

The loop must always drain `stt_session.events()` — backpressure here
parks the websocket's transfer_data task, starves pong frames and trips
Deepgram's keepalive watchdog with a 1011 mid-reply. So routing is a
synchronous decision per event and the slow Claude+TTS work happens in
the turn worker, on the far side of `turn_queue`.

The interesting cases are all about *not* driving a turn:

* `EndOfTurn` in IDLE — mic audio is gated behind the wake word, so a
  stray transcript here must not start a conversation.
* `EndOfTurn` in SPEAKING — a barge-in would have flipped the state to
  LISTENING already, so reaching this branch means no `StartOfTurn`
  triggered one. With AEC on, that is the assistant's own voice.
* An empty transcript — Deepgram closes turns on silence too.
"""

import asyncio
import logging
from collections.abc import Callable

from meeko.orchestrator.idle import IdleController
from meeko.orchestrator.state import State, StateManager

logger = logging.getLogger("meeko")


class SttEventRouter:
    """Turns STT events into state changes and queued user turns."""

    def __init__(
        self,
        *,
        state_manager: StateManager,
        idle: IdleController,
        turn_queue: asyncio.Queue,
        request_barge_in: Callable[[], None],
        stop_event: asyncio.Event,
    ) -> None:
        self._state = state_manager
        self._idle = idle
        self._turn_queue = turn_queue
        self._request_barge_in = request_barge_in
        self._stop_event = stop_event

    async def run(self, stt_session) -> None:
        """Drain the session's events until it ends or we're shutting down."""
        # If we want to make it faster, we can also use EagerEndOfTurn and
        # TurnResumed events which allows us to send text to the LLM eagerly.
        # If they're done talking, great, we already sent the text to the LLM.
        # If not, we cancel the LLM request (or discard the result), and send
        # the complete text. So may cost more since we throw away some results.
        async for ev in stt_session.events():
            if self._stop_event.is_set():
                return
            await self.handle(ev)

    async def handle(self, ev) -> None:
        """Route one turn event. Ignores event types we don't act on."""
        if ev.event == "StartOfTurn":
            self._on_start_of_turn()
            return
        if ev.event == "EndOfTurn":
            await self._on_end_of_turn(ev)

    def _on_start_of_turn(self) -> None:
        logger.info("User started speaking (state=%s)", self._state.state.name)
        # User activity always cancels a pending idle close; the branch
        # below handles barge-in for the SPEAKING case.
        self._idle.cancel()
        if self._state.state in (State.SPEAKING, State.PROCESSING):
            # Barge-in: user is talking over the assistant, or they
            # changed their mind during the window between EndOfTurn and
            # first audio (Claude TTFT + Deepgram TTS first-byte
            # synthesis, often >1s now that the Speaker defers SPEAKING
            # entry until first chunk). Cancel the in-flight speak task
            # in either case; the eventual EndOfTurn arrives in
            # LISTENING and flows through normally.
            self._request_barge_in()
            # User is already mid-utterance; show the "hearing you" cyan
            # rather than the steady "ready" cyan that request_barge_in's
            # state change would otherwise leave on the ring.
            self._state.set_listening_active(True)
        elif self._state.state == State.LISTENING:
            # Switch from the "I heard the wake word" cyan to the
            # brighter "I'm hearing you speak" cyan. Reverts on
            # EndOfTurn → PROCESSING.
            self._state.set_listening_active(True)

    async def _on_end_of_turn(self, ev) -> None:
        text = ev.transcript.strip()
        if self._state.state == State.IDLE:
            # Safety belt: Deepgram shouldn't emit turns while we're
            # gating mic audio behind the wake word, but any stray
            # transcripts must not start a Claude turn.
            return
        if self._state.state == State.SPEAKING:
            # No preceding StartOfTurn triggered barge-in (otherwise
            # state would already be LISTENING). With AEC on, this is
            # residual echo; drop it.
            # Transcript content: debug only (see setup_logging).
            logger.debug("[echo?] %s", text)
            return
        # Speech is over — drop back to the steady "ready" cyan. Without
        # this the ring stays on the brighter "hearing you" cue for the
        # empty-transcript path, and briefly for the window before the
        # turn worker picks up the turn and transitions to PROCESSING.
        self._state.set_listening_active(False)
        if not text:
            return
        await self._turn_queue.put(text)
