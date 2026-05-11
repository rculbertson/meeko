"""Meeko orchestrator: mic → Deepgram STT → Claude → Deepgram TTS → speaker.

This replaces the Deepgram Voice Agent wiring from the prototype. Claude
is called directly; STT and TTS are Deepgram-only. Audio runs over
PyAudio.

By default the mic stays open during TTS — we rely on the ReSpeaker
XVF3800's hardware AEC to suppress echo. A `StartOfTurn` while SPEAKING
is treated as the user barging in: the in-flight Claude+TTS turn is
cancelled, the speaker buffer is flushed, and the session transitions
back to LISTENING so the eventual `EndOfTurn` flows through the normal
path. An `EndOfTurn` while SPEAKING with no preceding `StartOfTurn`
having triggered barge-in is still echo and is dropped. For Mac /
no-AEC development, set `MEEKO_MUTE_MIC_WHILE_SPEAKING=1`: mic chunks
are dropped while SPEAKING and the mic queue is drained after playback.
"""

import argparse
import asyncio
import contextlib
import functools
import logging
import logging.handlers
import os
import signal
import sys
import time
from collections.abc import Awaitable, Callable
from enum import Enum, auto

import anthropic
from dotenv import load_dotenv
from websockets.exceptions import ConnectionClosed

from meeko.audio_io import AudioIO
from meeko.claude_client import ClaudeClient
from meeko.config import MeekoConfig, Profile, load_config, load_profiles
from meeko.deepgram_stt import DeepgramSTT
from meeko.deepgram_tts import DeepgramTTS
from meeko.leds import LedController, LedState
from meeko.session_summary import summarize_session
from meeko.sessions import SessionStore
from meeko.speaker import Speaker
from meeko.stt_supervisor import KEEPALIVE_INTERVAL_S, STTSupervisor
from meeko.tools.dispatch import ToolDispatcher
from meeko.tools.profile import ProfileManager
from meeko.tools.profile import get_tool_definitions as profile_tools
from meeko.tools.profile import handle as profile_handle
from meeko.tools.session import SessionManager
from meeko.tools.session import get_tool_definitions as session_tools
from meeko.tools.session import handle as session_handle
from meeko.tools.timer import get_tool_definitions as timer_tools
from meeko.tools.timer import handle as timer_handle
from meeko.tools.timer import timer_manager
from meeko.wake_word import WakeWordDetector

LOG_FILE = "meeko.log"

logger = logging.getLogger("meeko")


def setup_logging(log_level: str = "DEBUG", log_target: str | None = None) -> None:
    formatter = logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M:%S")

    if log_target == "file":
        handler = logging.handlers.RotatingFileHandler(
            LOG_FILE, maxBytes=5_000_000, backupCount=3
        )
    else:
        handler = logging.StreamHandler()

    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(getattr(logging, log_level.upper(), logging.DEBUG))

    # Route asyncio's own warnings through the same handler so they get
    # Meeko's timestamp format and land in the rotating log file when
    # MEEKO_LOG_TARGET=file. Covers slow-callback warnings (when run
    # with PYTHONASYNCIODEBUG=1) plus the always-on ones like
    # "task was destroyed but it is pending" and "coroutine was never
    # awaited" — useful signal that would otherwise hit stderr only.
    asyncio_logger = logging.getLogger("asyncio")
    asyncio_logger.addHandler(handler)
    asyncio_logger.setLevel(logging.WARNING)


class State(Enum):
    IDLE = auto()
    LISTENING = auto()
    PROCESSING = auto()
    SPEAKING = auto()


class StateManager:
    _STATE_TO_LED = {
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
            logger.debug("[state] %s → %s", self._state.name, new.name)
            self._state = new
        self._leds.set_state(self._STATE_TO_LED[new])

    def enter_speaking(self) -> State:
        prev = self._state
        self.set(State.SPEAKING)
        return prev

    def exit_speaking(self, prev: State) -> None:
        self.set(prev if prev != State.SPEAKING else State.LISTENING)

    def set_listening_active(self, active: bool) -> None:
        """LISTENING_ACTIVE is a LED sub-state of LISTENING (DoA mode
        on, vs. solid cyan). The orchestrator stays in State.LISTENING
        either way — only the LED display changes. Calls from outside
        LISTENING are no-ops so the LED never diverges from the
        logical state."""
        if self._state != State.LISTENING:
            return
        self._leds.set_state(
            LedState.LISTENING_ACTIVE if active else LedState.LISTENING
        )


RESUME_LATEST = "__latest__"

# Sentinel posted to the turn queue by the idle-timeout monitor to wake
# drive_turns and run post-turn session handling (which finalizes the
# session) without spending a Claude/TTS round-trip first.
_IDLE_TIMEOUT_SENTINEL: object = object()

CONVERSATION_PROMPT_TEXT = (
    "Would you like to continue, or should we end the session now?"
)
CONVERSATION_CLOSE_TEXT = "Okay, ending the session now."


async def _idle_monitor(
    profile: Profile,
    on_timeout: Callable[[], None],
    speak: Callable[[str], Awaitable[None]] | None = None,
) -> None:
    """Wait for the active mode's idle window, then call on_timeout.

    Cancellation at any await bails cleanly without firing. Called from
    drive_turns after each turn that left the state in LISTENING.

    Query mode: silent close after `idle_timeout_seconds`.
    Conversation mode: after `conversation_idle_seconds`, speak a prompt;
    if no response within `conversation_close_seconds` after the prompt
    finishes, speak a closing line and end the session.

    A non-positive timeout (`idle_timeout_seconds` for query,
    `conversation_idle_seconds` for conversation) disables the monitor —
    used as a test escape hatch and an operator override.
    """
    if profile.mode == "query":
        if profile.idle_timeout_seconds <= 0:
            return
        await asyncio.sleep(profile.idle_timeout_seconds)
        on_timeout()
        return

    elif profile.mode == "conversation":
        if profile.conversation_idle_seconds <= 0:
            return
        if speak is None:
            raise ValueError("conversation mode requires a speak callback")
        await asyncio.sleep(profile.conversation_idle_seconds)
        await speak(CONVERSATION_PROMPT_TEXT)
        # Full close window after the prompt finishes — Meeko's own
        # talking does not eat into the user's response time.
        await asyncio.sleep(profile.conversation_close_seconds)
        await speak(CONVERSATION_CLOSE_TEXT)
        on_timeout()
        return


async def _list_sessions_cmd(store: SessionStore) -> None:
    rows = await store.list_sessions()
    if not rows:
        print("No sessions yet.")
        return
    print(f"{'id':36}  {'profile':12}  {'last_active':32}  turns")
    for r in rows:
        print(
            f"{r['id']:36}  {r['profile_name']:12}  {r['last_active']:32}  "
            f"{r['turn_count']}"
        )


async def _resolve_resume(store: SessionStore, resume: str) -> dict[str, object] | None:
    """Return the session row to resume, or None to fall back to new."""
    if resume == RESUME_LATEST:
        row = await store.get_latest_session()
        if row is None:
            logger.info("No prior sessions, starting new")
            return None
        return row
    row = await store.get_session(resume)
    if row is None:
        print(f"No session with id {resume!r}", file=sys.stderr)
        raise SystemExit(1)
    return row


async def _init_session_state(
    store: SessionStore,
    resume: str | None,
    profiles: dict[str, Profile],
) -> tuple[Profile, str, list[dict] | None]:
    """Resolve the active profile, session id, and prior history (if any)."""
    resumed_row: dict[str, object] | None = None
    if resume is not None:
        try:
            resumed_row = await _resolve_resume(store, resume)
        except SystemExit:
            await store.close()
            raise

    if resumed_row is not None:
        session_id = str(resumed_row["id"])
        stored_profile = str(resumed_row["profile_name"])
        if stored_profile in profiles:
            profile = profiles[stored_profile]
        else:
            logger.warning(
                "Session %s profile %r missing; falling back to default",
                session_id,
                stored_profile,
            )
            profile = profiles["default"]
        await store.touch_session(session_id)
        history = await store.load_turns(session_id)
        logger.info(
            "Resumed session %s (profile=%s, %d turns)",
            session_id[:8],
            profile.name,
            len(history),
        )
        return profile, session_id, history

    profile = profiles["default"]
    session_id = await store.create_session(profile.name)
    logger.info("Started session %s (profile=%s)", session_id, profile.name)
    return profile, session_id, None


def _build_dispatcher(
    profiles: dict[str, Profile],
    profile_manager: ProfileManager,
    session_manager: SessionManager,
    store: SessionStore,
    get_session_id: Callable[[], str],
) -> ToolDispatcher:
    dispatcher = ToolDispatcher()
    dispatcher.register(timer_tools(), timer_handle)
    dispatcher.register(
        profile_tools(profiles),
        functools.partial(profile_handle, manager=profile_manager),
    )

    async def session_handle_wrapper(fn_name: str, args: dict) -> str:
        return await session_handle(
            fn_name,
            args,
            manager=session_manager,
            store=store,
            current_session_id=get_session_id(),
        )

    dispatcher.register(session_tools(), session_handle_wrapper)
    return dispatcher


def _create_wake_detector(
    config: MeekoConfig,
) -> tuple[WakeWordDetector | None, State]:
    if config.wake_word_disabled:
        return None, State.LISTENING
    return (
        WakeWordDetector(
            threshold=config.wake_word_threshold,
            model_path=str(config.wake_word_model),
        ),
        State.IDLE,
    )


async def _apply_post_turn_session_change(
    *,
    session_manager: SessionManager,
    profile_manager: ProfileManager,
    claude: ClaudeClient,
    store: SessionStore,
    wake_detector: WakeWordDetector | None,
    session_id: str,
    fire_summary: Callable[[str], None],
) -> tuple[str, State]:
    """Apply pending session-management actions queued during the last turn.

    Returns the (possibly updated) session id and the next State to enter.
    `should_load` can coexist with `should_end` (chain: finalize current
    then load a prior one in one turn). Always summarize the abandoned
    session so it stays in the recall index — `summarize_session` no-ops
    on empty sessions, so this is safe even when the user loads after
    only a turn or two.
    """
    if session_manager.should_load():
        target_id = session_manager.get_load_target()
        assert target_id is not None  # guaranteed by should_load()
        fire_summary(session_id)
        logger.info(
            "load_session: fired summary for abandoned %s",
            session_id[:8],
        )
        turns = await store.load_turns(target_id)
        claude.load_history(turns)
        claude.rebind_session(target_id)
        await store.touch_session(target_id)
        session_manager.clear()
        logger.info(
            "load_session: swapped history to %s (%d turns), continuing in LISTENING",
            target_id[:8],
            len(turns),
        )
        return target_id, State.LISTENING

    if session_manager.should_end():
        active = profile_manager.active_profile
        finalized_sid = session_id
        new_sid = await store.create_session(active.name)
        claude.reset_session(new_sid)
        session_manager.clear()
        fire_summary(finalized_sid)
        if wake_detector is not None:
            wake_detector.reset()
            logger.info(
                "Session ended; returning to IDLE "
                "(say '%s' to start a new conversation)",
                active.wake_word,
            )
            return new_sid, State.IDLE
        logger.info("Session ended; wake word disabled, returning to LISTENING")
        return new_sid, State.LISTENING

    if session_manager.should_start_new():
        active = profile_manager.active_profile
        finalized_sid = session_id
        new_sid = await store.create_session(active.name)
        claude.reset_session(new_sid)
        session_manager.clear()
        fire_summary(finalized_sid)
        logger.info(
            "new_session: rotated to %s (profile=%s), continuing in LISTENING",
            new_sid[:8],
            active.name,
        )
        return new_sid, State.LISTENING

    return session_id, State.LISTENING


async def run(resume: str | None = None, list_sessions: bool = False):
    load_dotenv()
    config = load_config()
    setup_logging(log_level=config.log_level, log_target=config.log_target)

    if list_sessions:
        store = SessionStore.open(config.db_path)
        try:
            await _list_sessions_cmd(store)
        finally:
            await store.close()
        return

    deepgram_key = os.environ["DEEPGRAM_API_KEY"]
    anthropic_key = os.environ["ANTHROPIC_API_KEY"]

    profiles = load_profiles()
    store = SessionStore.open(config.db_path)
    profile, session_id, history = await _init_session_state(store, resume, profiles)

    profile_manager = ProfileManager(profiles)
    session_manager = SessionManager()

    dispatcher = _build_dispatcher(
        profiles,
        profile_manager,
        session_manager,
        store,
        lambda: session_id,
    )

    claude = ClaudeClient(
        api_key=anthropic_key,
        system_prompt=profile.prompt,
        dispatcher=dispatcher,
        store=store,
        session_id=session_id,
        compaction_trigger_tokens=config.compaction_trigger_tokens,
        web_search_enabled=config.web_search_enabled,
        web_search_max_uses=config.web_search_max_uses,
    )
    if history:
        claude.load_history(history)
    profile_manager.set_claude_client(claude)

    # Separate Anthropic client for background summarization — sidesteps
    # any concern about concurrent use with the live conversation client
    # and lets summarize_session own its own lifecycle.
    summary_client = anthropic.AsyncAnthropic(api_key=anthropic_key)
    summary_tasks: set[asyncio.Task] = set()

    def fire_summary(finalized_sid: str) -> None:
        task = asyncio.create_task(
            summarize_session(store, finalized_sid, summary_client)
        )
        summary_tasks.add(task)
        task.add_done_callback(summary_tasks.discard)

    stt = DeepgramSTT(deepgram_key)
    tts = DeepgramTTS(deepgram_key)

    stop_event = asyncio.Event()
    audio = AudioIO(
        stop_event,
        input_channels=config.input_channels,
        output_channels=config.output_channels,
        input_device_index=config.input_device_index,
        output_device_index=config.output_device_index,
    )

    leds = LedController(disabled=config.led_disabled)
    leds.start()

    wake_detector, initial_state = _create_wake_detector(config)
    state_manager = StateManager(initial_state, leds)
    state_manager.set(initial_state)

    speaker = Speaker(
        tts,
        audio,
        profile,
        state_manager.enter_speaking,
        state_manager.exit_speaking,
        mute_mic_while_speaking=config.mute_mic_while_speaking,
    )
    timer_manager.set_speak_callback(speaker.speak)

    async def pump_mic(stt_session):
        """Forward mic chunks to STT.

        While in IDLE, chunks are fed to the wake-word detector instead
        of STT. On detection the session transitions to LISTENING so
        subsequent mic audio (including any question the user spoke
        right after the wake word) flows to Deepgram. With
        MEEKO_MUTE_MIC_WHILE_SPEAKING set, chunks are dropped while the
        assistant is SPEAKING (Mac / no-AEC dev path). Otherwise the
        pump stays on and we rely on hardware AEC to suppress echo."""
        while not stop_event.is_set():
            try:
                data = await asyncio.wait_for(audio.mic_queue.get(), timeout=0.1)
            except TimeoutError:
                continue
            if state_manager.state == State.IDLE:
                assert wake_detector is not None
                if wake_detector.process(data):
                    state_manager.set(State.LISTENING)
                    logger.info("Wake word accepted; entering LISTENING")
                continue
            if config.mute_mic_while_speaking and state_manager.state == State.SPEAKING:
                continue
            await stt_session.send_audio(data)

    async def keepalive_pump(stt_session):
        """Send a Deepgram KeepAlive every few seconds so the session
        stays open when we're not streaming audio."""
        while not stop_event.is_set():
            try:
                await asyncio.sleep(KEEPALIVE_INTERVAL_S)
                await stt_session.send_keepalive()
            except ConnectionClosed:
                return

    # Queue of user-turn transcripts handed from the STT puller to the
    # turn-driving worker. Decouples the (always-fast) STT recv loop
    # from the (slow) Claude+TTS path so the websockets recv queue
    # never backs up while the worker is awaiting TTS — backpressure
    # there used to fill the queue, park transfer_data, starve pong
    # frames, and trip the keepalive watchdog with 1011 mid-reply.
    # Unbounded: payloads are short transcript strings, and a long
    # backlog would only happen if Claude fell catastrophically behind,
    # which is a separate problem. Also carries _IDLE_TIMEOUT_SENTINEL
    # from the idle monitor to ask drive_turns to run post-turn session
    # handling without a Claude round-trip.
    turn_queue: asyncio.Queue[str | object] = asyncio.Queue()

    # Set while an idle monitor task is sleeping between turns. Cancelled
    # on user activity (StartOfTurn), barge-in, the next turn dequeuing,
    # and shutdown. See _idle_monitor.
    idle_monitor_task: asyncio.Task | None = None

    def cancel_idle_monitor() -> None:
        nonlocal idle_monitor_task
        if idle_monitor_task is not None and not idle_monitor_task.done():
            idle_monitor_task.cancel()
        idle_monitor_task = None

    def on_idle_timeout() -> None:
        # Race guard: a turn may have arrived in the gap between
        # asyncio.sleep waking and on_timeout running. Skip if state
        # is no longer LISTENING (turn already in flight) or if a
        # user transcript is already queued ahead of us — letting the
        # sentinel land behind real text would end a session right
        # after a successful turn.
        if state_manager.state != State.LISTENING or not turn_queue.empty():
            return
        active = profile_manager.active_profile
        if active.mode == "query":
            logger.info(
                "Idle timeout (%.1fs, mode=query); ending session",
                active.idle_timeout_seconds,
            )
        elif active.mode == "conversation":
            logger.info(
                "Idle timeout (%.1fs after prompt, mode=conversation); ending session",
                active.conversation_close_seconds,
            )
        else:
            logger.warning(
                "Idle timeout for unknown mode: %s; ending session", active.mode
            )
        session_manager.request_end()
        turn_queue.put_nowait(_IDLE_TIMEOUT_SENTINEL)

    def start_idle_monitor() -> None:
        nonlocal idle_monitor_task
        cancel_idle_monitor()
        idle_monitor_task = asyncio.create_task(
            _idle_monitor(
                profile_manager.active_profile,
                on_idle_timeout,
                speak=speaker.speak,
            )
        )

    # Set in drive_turns while a Claude+TTS turn is running so
    # request_barge_in() can cancel just that turn without tearing down
    # the long-lived drive_turns worker.
    current_speak_task: asyncio.Task | None = None
    # Set by request_barge_in() so drive_turns can tell the difference
    # between a real barge-in (continue worker) and a shutdown cancel
    # (re-raise). Checking stop_event isn't reliable because asyncio
    # shutdown cancels drive_turns_task directly, before run()'s finally
    # has a chance to set stop_event.
    barge_in_requested = False

    def request_barge_in() -> None:
        """Cancel the in-flight speak task, if any. Called from
        pull_stt_events when StartOfTurn fires during SPEAKING (the
        assistant is talking) or PROCESSING (the assistant's reply is
        still being generated; user has changed their mind)."""
        nonlocal barge_in_requested
        # Flip state synchronously so any EndOfTurn arriving before the
        # cancel propagates through drive_turns isn't dropped as echo
        # by pull_stt_events. This must happen even when there's no
        # current_speak_task (e.g. a timer chime is playing via
        # speaker.speak() — that path enters SPEAKING but is not
        # cancellable from here): the chime keeps playing, but the
        # user's interruption is at least captured into turn_queue
        # instead of silently dropped. drive_turns and speak_stream's
        # exit_speaking will re-assert LISTENING when they unwind; the
        # brief PROCESSING window in between is harmless (PROCESSING-
        # state EndOfTurns are queued normally).
        cancel_idle_monitor()
        state_manager.set(State.LISTENING)
        if current_speak_task is not None and not current_speak_task.done():
            logger.info("Barge-in: cancelling in-flight reply")
            barge_in_requested = True
            current_speak_task.cancel()

    async def pull_stt_events(stt_session):
        """Always drain stt_session.events(); decide synchronously
        whether each EndOfTurn should drive a turn, and queue the ones
        that should."""
        # If we want to make it faster, we can also use EagerEndOfTurn and
        # TurnResumed events which allows us to send text to the LLM eagerly.
        # If they're done talking, great, we already sent the text to the LLM.
        # If not, we cancel the LLM request (or discard the result), and send
        # the complete text. So may cost more since we throw away some results.
        async for ev in stt_session.events():
            if stop_event.is_set():
                return
            if ev.event == "StartOfTurn":
                logger.info(
                    "User started speaking (state=%s)", state_manager.state.name
                )
                # User activity always cancels a pending idle close; the
                # branch below handles barge-in for the SPEAKING case.
                cancel_idle_monitor()
                if state_manager.state in (State.SPEAKING, State.PROCESSING):
                    # Barge-in: user is talking over the assistant, or
                    # they changed their mind during the window between
                    # EndOfTurn and first audio (Claude TTFT + Deepgram
                    # TTS first-byte synthesis, often >1s now that the
                    # Speaker defers SPEAKING entry until first chunk).
                    # Cancel the in-flight speak task in either case;
                    # the eventual EndOfTurn arrives in LISTENING and
                    # flows through normally.
                    request_barge_in()
                    # User is already mid-utterance; show DoA tracking
                    # rather than the solid "ready" cyan that
                    # request_barge_in's state change would otherwise
                    # leave on the ring.
                    state_manager.set_listening_active(True)
                elif state_manager.state == State.LISTENING:
                    # Switch from solid "I heard the wake word" to DoA
                    # mode so the ring tracks the user's direction while
                    # they speak. Reverts on EndOfTurn → PROCESSING.
                    state_manager.set_listening_active(True)
                continue
            if ev.event != "EndOfTurn":
                continue
            text = ev.transcript.strip()
            if state_manager.state == State.IDLE:
                # Safety belt: Deepgram shouldn't emit turns while
                # we're gating mic audio behind the wake word, but
                # any stray transcripts must not start a Claude turn.
                continue
            if state_manager.state == State.SPEAKING:
                # No preceding StartOfTurn triggered barge-in (otherwise
                # state would already be LISTENING). With AEC on, this
                # is residual echo; drop it.
                logger.info("[echo?] %s", text)
                continue
            # Speech is over — leave DoA mode so the ring shows the
            # solid "ready" cue. (DoA's firmware only lights the ring
            # while speech is detected, so without this the ring goes
            # dark for the empty-transcript path, and briefly for the
            # window before drive_turns picks up the turn and
            # transitions to PROCESSING.)
            state_manager.set_listening_active(False)
            if not text:
                continue
            await turn_queue.put(text)

    async def drive_turns():
        """Drive Claude + TTS for queued user turns, one at a time."""
        nonlocal session_id, current_speak_task, barge_in_requested
        while not stop_event.is_set():
            item = await turn_queue.get()
            cancel_idle_monitor()
            if item is _IDLE_TIMEOUT_SENTINEL:
                # Idle window expired without user activity; on_idle_timeout
                # already set session_manager.request_end(). Run the
                # post-turn block to finalize and (with wake word) return
                # to IDLE. No Claude/TTS round-trip on this path, so we
                # do not start a fresh idle window after.
                session_id, new_state = await _apply_post_turn_session_change(
                    session_manager=session_manager,
                    profile_manager=profile_manager,
                    claude=claude,
                    store=store,
                    wake_detector=wake_detector,
                    session_id=session_id,
                    fire_summary=fire_summary,
                )
                state_manager.set(new_state)
                continue
            text = item
            assert isinstance(text, str)
            logger.info("[user] %s", text)
            state_manager.set(State.PROCESSING)
            t_turn = time.perf_counter()
            # Run the turn as a sub-task so request_barge_in() can
            # cancel just this turn without tearing down drive_turns.
            current_speak_task = asyncio.create_task(
                speaker.speak_stream(claude.stream_turn(text))
            )
            try:
                await current_speak_task
            except asyncio.CancelledError:
                # speak_stream's finally already flushed TTS subtasks
                # and exit_speaking() restored state. Distinguish a real
                # barge-in (continue worker) from a shutdown cancel
                # (re-raise) by the explicit flag — stop_event is racy
                # because asyncio's shutdown cancels drive_turns_task
                # before run()'s finally has set it.
                state_manager.set(State.LISTENING)
                current_speak_task = None
                if not barge_in_requested:
                    raise
                barge_in_requested = False
                logger.info("Barge-in: turn cancelled, returning to LISTENING")
                continue
            except Exception:
                logger.exception("Claude turn failed")
                # Set state first so the worker applies LISTENING before
                # the error animation, and the post-flash restore picks
                # up LISTENING as _current_state. Reversing the order
                # makes _sleep_or_interrupt see the queued state action
                # and abort the breath immediately.
                state_manager.set(State.LISTENING)
                leds.error()
                current_speak_task = None
                continue
            current_speak_task = None
            logger.debug(
                "[timing] turn_total_eot_to_speak_done=%dms",
                int((time.perf_counter() - t_turn) * 1000),
            )
            session_id, new_state = await _apply_post_turn_session_change(
                session_manager=session_manager,
                profile_manager=profile_manager,
                claude=claude,
                store=store,
                wake_detector=wake_detector,
                session_id=session_id,
                fire_summary=fire_summary,
            )
            # Start the post-turn idle window. Only when state is
            # LISTENING — IDLE means the session already ended and the
            # next interaction needs the wake word.
            if new_state == State.LISTENING:
                start_idle_monitor()
            state_manager.set(new_state)

    async def on_session(stt_session) -> None:
        # drive_turns is intentionally NOT in this group — it lives at
        # run() scope and outlives individual STT sessions, so an STT
        # blip mid-reply doesn't cut TTS off mid-sentence and doesn't
        # lose the in-flight turn. The session-scoped tasks are the
        # ones that legitimately need the live stt_session handle.
        session_tasks = [
            asyncio.create_task(pump_mic(stt_session)),
            asyncio.create_task(pull_stt_events(stt_session)),
            asyncio.create_task(keepalive_pump(stt_session)),
        ]
        try:
            done, pending = await asyncio.wait(
                session_tasks, return_when=asyncio.FIRST_COMPLETED
            )
            for t in pending:
                t.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            for t in done:
                exc = t.exception()
                if exc is not None:
                    raise exc
        finally:
            for t in session_tasks:
                if not t.done():
                    t.cancel()
            await asyncio.gather(*session_tasks, return_exceptions=True)

    supervisor = STTSupervisor(
        stt,
        audio,
        stop_event,
        on_session,
        lambda: state_manager.state == State.SPEAKING,
    )

    audio.start_mic()
    logger.info("Mic active.")

    # Long-lived turn worker — survives STT reconnects so a connection
    # blip mid-reply doesn't truncate TTS or lose the in-flight turn.
    drive_turns_task = asyncio.create_task(drive_turns())

    try:
        await supervisor.run()
    except asyncio.CancelledError:
        pass
    finally:
        stop_event.set()
        # Cancel and await the idle monitor directly; can't use
        # cancel_idle_monitor() here because it nulls the reference
        # before we can await it.
        if idle_monitor_task is not None:
            if not idle_monitor_task.done():
                idle_monitor_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await idle_monitor_task
        drive_turns_task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await drive_turns_task
        timer_manager.cancel_all_timers()
        # Cancel in-flight summary tasks before closing the SQLite
        # connection; letting them run into a closed store would crash
        # and a half-written summary is not worth the wait at shutdown.
        for task in summary_tasks:
            task.cancel()
        if summary_tasks:
            await asyncio.gather(*summary_tasks, return_exceptions=True)
        audio.close()
        leds.close()
        await store.close()
        try:
            await summary_client.close()
        except Exception:
            logger.exception("Failed to close summary client")
        logger.info("Shutting down.")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="meeko")
    parser.add_argument(
        "--resume",
        nargs="?",
        const=RESUME_LATEST,
        default=None,
        metavar="SESSION_ID",
        help="Resume a prior session. With no value, resumes the most recent.",
    )
    parser.add_argument(
        "--list-sessions",
        action="store_true",
        help="Print prior sessions and exit.",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = _parse_args()
    loop = asyncio.new_event_loop()

    def handle_sigint():
        for task in asyncio.all_tasks(loop):
            task.cancel()

    loop.add_signal_handler(signal.SIGINT, handle_sigint)

    try:
        loop.run_until_complete(
            run(resume=args.resume, list_sessions=args.list_sessions)
        )
    except KeyboardInterrupt:
        pass
    finally:
        loop.close()


if __name__ == "__main__":
    main()
