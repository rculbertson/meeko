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
from collections.abc import Callable

import anthropic
from dotenv import load_dotenv

from meeko.audio_io import AudioIO
from meeko.claude_client import ClaudeClient
from meeko.config import (
    MeekoConfig,
    Profile,
    ensure_config_exists,
    load_config,
    load_profiles,
)
from meeko.deepgram_stt import DeepgramSTT
from meeko.deepgram_tts import DeepgramTTS
from meeko.idle import IDLE_TIMEOUT_SENTINEL, IdleController
from meeko.leds import LedController
from meeko.mic_pump import MicPump
from meeko.session_summary import SummaryScheduler
from meeko.sessions import UNTITLED, SessionStore
from meeko.speaker import Speaker
from meeko.state import State, StateManager
from meeko.stt_events import SttEventRouter
from meeko.stt_supervisor import (
    STTSupervisor,
    keepalive_pump,
    run_session_workers,
)
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
from meeko.tools.weather import get_tool_definitions as weather_tools
from meeko.tools.weather import handle as weather_handle
from meeko.tools.weather import weather_client
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


RESUME_LATEST = "__latest__"


async def _list_sessions_cmd(store: SessionStore) -> None:
    rows = await store.list_sessions()
    if not rows:
        print("No sessions yet.")
        return
    # Title last: it's free text of unbounded length, so anywhere else it
    # would push the fixed-width columns out of line.
    print(f"{'id':36}  {'profile':12}  {'last_active':32}  {'turns':>5}  title")
    for r in rows:
        print(
            f"{r['id']:36}  {r['profile_name']:12}  {r['last_active']:32}  "
            f"{r['turn_count']:>5}  {r['title'] or UNTITLED}"
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
    default_profile_name: str,
) -> tuple[Profile, str | None, list[dict] | None]:
    """Resolve the active profile, session id, and prior history (if any).

    For fresh starts (no resume), session_id is None — the row is
    created lazily on the first persisted turn (see ClaudeClient)."""
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
        profile = profiles[stored_profile]
        await store.touch_session(session_id)
        history = await store.load_turns(session_id)
        logger.info(
            "Resumed session %s (profile=%s, %d turns)",
            session_id[:8],
            profile.name,
            len(history),
        )
        return profile, session_id, history

    profile = profiles[default_profile_name]
    logger.info(
        "Started new session (profile=%s, row deferred until first turn)",
        profile.name,
    )
    return profile, None, None


def _build_dispatcher(
    profiles: dict[str, Profile],
    profile_manager: ProfileManager,
    session_manager: SessionManager,
    store: SessionStore,
    get_session_id: Callable[[], str | None],
    config: MeekoConfig,
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

    weather_client.configure(
        latitude=config.latitude,
        longitude=config.longitude,
        units=config.weather_units,
    )
    dispatcher.register(weather_tools(), weather_handle)

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
    session_id: str | None,
    fire_summary: Callable[[str | None], None],
) -> State:
    """Apply pending session-management actions queued during the last turn.

    Returns the next State to enter. `claude.session_id` tracks the live
    bound row internally (cleared by `reset_session`, rebound by
    `rebind_session`), so the caller never needs to thread the id back
    out. `should_load` can coexist with `should_end` (chain: finalize
    current then load a prior one in one turn). When the current session
    has no turns yet (session_id is None — lazy creation hasn't fired),
    there's nothing to summarize; `fire_summary` no-ops on None.
    """
    if session_manager.should_load():
        target_id = session_manager.get_load_target()
        assert target_id is not None  # guaranteed by should_load()
        fire_summary(session_id)
        if session_id is not None:
            logger.info(
                "load_session: fired summary for abandoned %s",
                session_id[:8],
            )
        # Rebind the runtime profile (system prompt + voice) to match
        # the loaded session's persisted profile_name before swapping
        # history, so subsequent turns persist back into a row whose
        # profile_name matches what is actually driving the model. A
        # missing row would mean Sonnet handed us a bogus id (or the
        # session got deleted between list_sessions and load_session) —
        # abort rather than binding Claude to a nonexistent session.
        target_row = await store.get_session(target_id)
        if target_row is None:
            logger.error(
                "load_session: target session %s not found; aborting load",
                target_id[:8],
            )
            session_manager.clear()
            return State.LISTENING
        profile_manager.rebind_profile(str(target_row["profile_name"]))
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
        return State.LISTENING

    if session_manager.should_end():
        active = profile_manager.active_profile
        finalized_sid = session_id
        claude.reset_session()
        session_manager.clear()
        fire_summary(finalized_sid)
        if wake_detector is not None:
            wake_detector.reset()
            logger.info(
                "Session ended; returning to IDLE "
                "(say '%s' to start a new conversation)",
                active.wake_word,
            )
            return State.IDLE
        logger.info("Session ended; wake word disabled, returning to LISTENING")
        return State.LISTENING

    if session_manager.should_start_new():
        active = profile_manager.active_profile
        finalized_sid = session_id
        claude.reset_session()
        session_manager.clear()
        fire_summary(finalized_sid)
        logger.info(
            "new_session: rotated (profile=%s, row deferred), continuing in LISTENING",
            active.name,
        )
        return State.LISTENING

    return State.LISTENING


async def run(resume: str | None = None, list_sessions: bool = False):
    load_dotenv()
    config_path, config_created = ensure_config_exists()
    config = load_config(config_path)
    setup_logging(log_level=config.log_level, log_target=config.log_target)
    if config_created:
        # Logged after setup_logging so it reaches the configured handler
        # (stderr or the rotating file) rather than being suppressed.
        logger.info("No config found; wrote default config to %s", config_path)

    if list_sessions:
        store = SessionStore.open(config.db_path)
        try:
            await _list_sessions_cmd(store)
        finally:
            await store.close()
        return

    deepgram_key = os.environ["DEEPGRAM_API_KEY"]
    anthropic_key = os.environ["ANTHROPIC_API_KEY"]

    profiles, default_profile_name = load_profiles(config_path)
    store = SessionStore.open(config.db_path)
    profile, session_id, history = await _init_session_state(
        store, resume, profiles, default_profile_name
    )

    profile_manager = ProfileManager(profiles, active_name=default_profile_name)
    session_manager = SessionManager()

    # Lazy session creation: the row is INSERTed on the first persisted
    # turn so killing the process before any conversation doesn't litter
    # the DB with empty, forever-untitled sessions.
    async def create_session_for_active_profile() -> str:
        sid = await store.create_session(profile_manager.active_profile.name)
        logger.info(
            "Created session %s (profile=%s)",
            sid[:8],
            profile_manager.active_profile.name,
        )
        return sid

    dispatcher = _build_dispatcher(
        profiles,
        profile_manager,
        session_manager,
        store,
        lambda: claude.session_id,
        config,
    )

    claude = ClaudeClient(
        api_key=anthropic_key,
        system_prompt=profile.prompt,
        dispatcher=dispatcher,
        store=store,
        session_id=session_id,
        create_session_fn=create_session_for_active_profile,
        compaction_trigger_tokens=config.compaction_trigger_tokens,
        web_search_enabled=config.web_search_enabled,
        web_search_max_uses=config.web_search_max_uses,
        latitude=config.latitude,
        longitude=config.longitude,
    )
    if history:
        claude.load_history(history)
    profile_manager.set_claude_client(claude)

    # Separate Anthropic client for background summarization — sidesteps
    # any concern about concurrent use with the live conversation client.
    summaries = SummaryScheduler(
        store=store,
        client=anthropic.AsyncAnthropic(api_key=anthropic_key),
    )
    await summaries.backfill(active_session_id=session_id)

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
    profile_manager.set_speaker(speaker)
    timer_manager.set_speak_callback(speaker.speak)

    # Queue of user-turn transcripts handed from the STT puller to the
    # turn-driving worker. Decouples the (always-fast) STT recv loop
    # from the (slow) Claude+TTS path so the websockets recv queue
    # never backs up while the worker is awaiting TTS — backpressure
    # there used to fill the queue, park transfer_data, starve pong
    # frames, and trip the keepalive watchdog with 1011 mid-reply.
    # Unbounded: payloads are short transcript strings, and a long
    # backlog would only happen if Claude fell catastrophically behind,
    # which is a separate problem. Also carries IDLE_TIMEOUT_SENTINEL
    # from the idle monitor to ask drive_turns to run post-turn session
    # handling without a Claude round-trip.
    turn_queue: asyncio.Queue[str | object] = asyncio.Queue()

    # Owns the post-turn and post-wake silence windows (one task slot).
    # Cancelled on user activity (StartOfTurn), barge-in, the next turn
    # dequeuing, and shutdown. See meeko/idle.py.
    idle = IdleController(
        is_listening=lambda: state_manager.state == State.LISTENING,
        profile_manager=profile_manager,
        session_manager=session_manager,
        turn_queue=turn_queue,
        speak=speaker.speak,
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
        SttEventRouter when StartOfTurn fires during SPEAKING (the
        assistant is talking) or PROCESSING (the assistant's reply is
        still being generated; user has changed their mind)."""
        nonlocal barge_in_requested
        # Flip state synchronously so any EndOfTurn arriving before the
        # cancel propagates through drive_turns isn't dropped as echo
        # by the router. This must happen even when there's no
        # current_speak_task (e.g. a timer chime is playing via
        # speaker.speak() — that path enters SPEAKING but is not
        # cancellable from here): the chime keeps playing, but the
        # user's interruption is at least captured into turn_queue
        # instead of silently dropped. drive_turns and speak_stream's
        # exit_speaking will re-assert LISTENING when they unwind; the
        # brief PROCESSING window in between is harmless (PROCESSING-
        # state EndOfTurns are queued normally).
        idle.cancel()
        state_manager.set(State.LISTENING)
        if current_speak_task is not None and not current_speak_task.done():
            logger.info("Barge-in: cancelling in-flight reply")
            barge_in_requested = True
            current_speak_task.cancel()

    # Drains the mic queue into the wake detector or the STT session.
    mic_pump = MicPump(
        audio=audio,
        state_manager=state_manager,
        idle=idle,
        wake_detector=wake_detector,
        stop_event=stop_event,
        mute_mic_while_speaking=config.mute_mic_while_speaking,
    )

    # Routes STT turn events into state changes and queued user turns.
    # Constructed here because it needs request_barge_in above.
    stt_router = SttEventRouter(
        state_manager=state_manager,
        idle=idle,
        turn_queue=turn_queue,
        request_barge_in=request_barge_in,
        stop_event=stop_event,
    )

    async def drive_turns():
        """Drive Claude + TTS for queued user turns, one at a time."""
        nonlocal current_speak_task, barge_in_requested
        while not stop_event.is_set():
            item = await turn_queue.get()
            idle.cancel()
            if item is IDLE_TIMEOUT_SENTINEL:
                # Idle window expired without user activity; IdleController
                # already set session_manager.request_end(). Run the
                # post-turn block to finalize and (with wake word) return
                # to IDLE. No Claude/TTS round-trip on this path, so we
                # do not start a fresh idle window after.
                new_state = await _apply_post_turn_session_change(
                    session_manager=session_manager,
                    profile_manager=profile_manager,
                    claude=claude,
                    store=store,
                    wake_detector=wake_detector,
                    session_id=claude.session_id,
                    fire_summary=summaries.fire,
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
            new_state = await _apply_post_turn_session_change(
                session_manager=session_manager,
                profile_manager=profile_manager,
                claude=claude,
                store=store,
                wake_detector=wake_detector,
                session_id=claude.session_id,
                fire_summary=summaries.fire,
            )
            # Start the post-turn idle window. Only when state is
            # LISTENING — IDLE means the session already ended and the
            # next interaction needs the wake word.
            if new_state == State.LISTENING:
                idle.start_post_turn()
            state_manager.set(new_state)

    async def on_session(stt_session) -> None:
        # drive_turns is intentionally NOT in this group — it lives at
        # run() scope and outlives individual STT sessions, so an STT
        # blip mid-reply doesn't cut TTS off mid-sentence and doesn't
        # lose the in-flight turn. The session-scoped workers are the
        # ones that legitimately need the live stt_session handle.
        await run_session_workers(
            mic_pump.run(stt_session),
            stt_router.run(stt_session),
            keepalive_pump(stt_session, stop_event),
        )

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
        await idle.aclose()
        drive_turns_task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await drive_turns_task
        timer_manager.cancel_all_timers()
        # Before store.close(): an in-flight summary would otherwise write
        # into a closed SQLite connection.
        await summaries.aclose()
        audio.close()
        leds.close()
        await store.close()
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
