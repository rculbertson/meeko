"""Meeko entry point: mic → Deepgram STT → Claude → Deepgram TTS → speaker.

This module is the composition root: `run()` builds every component and
injects its dependencies. The orchestration logic itself — state machine,
mic pump, STT event routing, idle windows, turn worker, post-turn session
changes — lives in `meeko/orchestrator/`. For barge-in and echo
handling see `MicPump` (wake-word gate, mute-while-speaking),
`SttEventRouter` (what each STT event means per state) and `TurnWorker`
(cancelling the in-flight turn).
"""

import argparse
import asyncio
import contextlib
import logging
import logging.handlers
import os
import signal
import sys
from collections.abc import Callable
from datetime import datetime

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
from meeko.leds import LedController
from meeko.orchestrator.idle import IdleController
from meeko.orchestrator.mic_pump import MicPump
from meeko.orchestrator.session_change import apply_post_turn_session_change
from meeko.orchestrator.state import State, StateManager
from meeko.orchestrator.stt_events import SttEventRouter
from meeko.orchestrator.turn_worker import TurnWorker
from meeko.session_summary import SummaryScheduler
from meeko.sessions import UNTITLED, SessionStore
from meeko.speaker import Speaker
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
from meeko.tools.timer import TimerManager
from meeko.tools.timer import get_tool_definitions as timer_tools
from meeko.tools.timer import handle as timer_handle
from meeko.tools.weather import WeatherClient
from meeko.tools.weather import get_tool_definitions as weather_tools
from meeko.tools.weather import handle as weather_handle
from meeko.wake_word import WakeWordDetector

LOG_FILE = "meeko.log"

logger = logging.getLogger("meeko")


def setup_logging(log_level: str = "INFO", log_target: str | None = None) -> None:
    """Configure the `meeko` logger.

    The default level is INFO on purpose: transcripts, assistant replies
    and tool arguments are logged at DEBUG, so under systemd (stderr →
    journald) nothing a user said or heard reaches the system log unless
    the operator explicitly opts in with `log_level = "DEBUG"`.
    """
    formatter = logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M:%S")

    if log_target == "file":
        handler = logging.handlers.RotatingFileHandler(
            LOG_FILE, maxBytes=5_000_000, backupCount=3
        )
    else:
        handler = logging.StreamHandler()

    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

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


def _local_timestamp(utc_iso: str) -> str:
    """Render a stored UTC ISO-8601 timestamp as local time for the CLI.

    `last_active` is written as UTC with microseconds, which is precise
    but hard to scan. The zone name is kept so local time can't be
    mistaken for UTC. Unparseable input is returned unchanged rather than
    hiding the row.
    """
    try:
        return (
            datetime.fromisoformat(utc_iso).astimezone().strftime("%Y-%m-%d %H:%M %Z")
        )
    except ValueError:
        return utc_iso


async def _list_sessions_cmd(store: SessionStore) -> None:
    rows = await store.list_sessions()
    if not rows:
        print("No sessions yet.")
        return
    # Title last: it's free text of unbounded length, so anywhere else it
    # would push the fixed-width columns out of line.
    print(f"{'id':36}  {'profile':12}  {'last_active':22}  {'turns':>5}  title")
    for r in rows:
        print(
            f"{r['id']:36}  {r['profile_name']:12}  "
            f"{_local_timestamp(r['last_active']):22}  "
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


def _require_api_keys() -> tuple[str, str]:
    """Return (Deepgram, Anthropic) keys, or exit with a pointer to `.env`
    naming every missing key — a fresh clone without `.env` would otherwise
    die on a bare KeyError traceback."""
    names = ("DEEPGRAM_API_KEY", "ANTHROPIC_API_KEY")
    missing = [name for name in names if not os.environ.get(name)]
    if missing:
        print(
            f"Missing {', '.join(missing)}. Add to a .env file in the project "
            "root (see README, Setup).",
            file=sys.stderr,
        )
        raise SystemExit(1)
    return os.environ[names[0]], os.environ[names[1]]


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
        profile = profiles.get(stored_profile)
        if profile is None:
            # The session's profile was renamed or removed from the config
            # since it was recorded. Resume the transcript under the default
            # profile rather than refusing to start. (ProfileManager.rebind_profile
            # handles the same case mid-session by keeping the active profile.)
            profile = profiles[default_profile_name]
            logger.warning(
                "Resumed session references unknown profile %r; using default "
                "profile %r",
                stored_profile,
                profile.name,
            )
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
    timer_manager: TimerManager,
    weather_client: WeatherClient,
    store: SessionStore,
    get_session_id: Callable[[], str | None],
) -> ToolDispatcher:
    dispatcher = ToolDispatcher()

    async def timer_handle_wrapper(fn_name: str, args: dict) -> str:
        return await timer_handle(fn_name, args, manager=timer_manager)

    dispatcher.register(timer_tools(), timer_handle_wrapper)

    async def profile_handle_wrapper(fn_name: str, args: dict) -> str:
        before = profile_manager.active_profile.name
        result = await profile_handle(fn_name, args, manager=profile_manager)
        after = profile_manager.active_profile.name
        # Record a real switch on the live session row so resume and
        # load_session restore this profile, not the one the session
        # started in. No row yet means lazy creation hasn't fired; it will
        # create the row under the now-active profile.
        session_id = get_session_id()
        if after != before and session_id is not None:
            try:
                await store.set_session_profile(session_id, after)
            except Exception:
                # The in-memory switch already happened and the user will
                # hear it confirmed, so don't fail the tool call. The cost
                # is only that a later resume restores the prior profile.
                logger.warning(
                    "Failed to record profile switch on session %s",
                    session_id[:8],
                    exc_info=True,
                )
        return result

    dispatcher.register(profile_tools(profiles), profile_handle_wrapper)

    async def session_handle_wrapper(fn_name: str, args: dict) -> str:
        return await session_handle(
            fn_name,
            args,
            manager=session_manager,
            store=store,
            current_session_id=get_session_id(),
        )

    dispatcher.register(session_tools(), session_handle_wrapper)

    async def weather_handle_wrapper(fn_name: str, args: dict) -> str:
        return await weather_handle(fn_name, args, client=weather_client)

    dispatcher.register(weather_tools(), weather_handle_wrapper)

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

    deepgram_key, anthropic_key = _require_api_keys()

    profiles, default_profile_name = load_profiles(config_path)
    store = SessionStore.open(config.db_path)
    profile, session_id, history = await _init_session_state(
        store, resume, profiles, default_profile_name
    )

    profile_manager = ProfileManager(profiles, active_name=profile.name)
    session_manager = SessionManager()
    timer_manager = TimerManager()
    weather_client = WeatherClient()
    weather_client.configure(
        latitude=config.latitude,
        longitude=config.longitude,
        units=config.weather_units,
    )

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
        timer_manager,
        weather_client,
        store,
        lambda: claude.session_id,
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
    # from the idle monitor to ask the turn worker to run post-turn session
    # handling without a Claude round-trip.
    turn_queue: asyncio.Queue[str | object] = asyncio.Queue()

    # Owns the post-turn and post-wake silence windows (one task slot).
    # Cancelled on user activity (StartOfTurn), barge-in, the next turn
    # dequeuing, and shutdown. See meeko/orchestrator/idle.py.
    idle = IdleController(
        is_listening=lambda: state_manager.state == State.LISTENING,
        profile_manager=profile_manager,
        session_manager=session_manager,
        turn_queue=turn_queue,
        speak=speaker.speak,
    )

    async def apply_session_change() -> State:
        return await apply_post_turn_session_change(
            session_manager=session_manager,
            profile_manager=profile_manager,
            claude=claude,
            store=store,
            wake_detector=wake_detector,
            session_id=claude.session_id,
            fire_summary=summaries.fire,
        )

    # Drives Claude + TTS for queued user turns and owns barge-in: its
    # request_barge_in cancels just the in-flight turn. See
    # meeko/orchestrator/turn_worker.py.
    turn_worker = TurnWorker(
        turn_queue=turn_queue,
        state_manager=state_manager,
        idle=idle,
        leds=leds,
        start_turn=lambda text: speaker.speak_stream(claude.stream_turn(text)),
        apply_session_change=apply_session_change,
        stop_event=stop_event,
    )

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
    # Constructed here because it needs turn_worker above.
    stt_router = SttEventRouter(
        state_manager=state_manager,
        idle=idle,
        turn_queue=turn_queue,
        request_barge_in=turn_worker.request_barge_in,
        stop_event=stop_event,
    )

    async def on_session(stt_session) -> None:
        # The turn worker is intentionally NOT in this group — it lives at
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
    turn_worker_task = asyncio.create_task(turn_worker.run())
    supervisor_task = asyncio.create_task(supervisor.run())

    try:
        # Wait on the worker too. Outside shutdown it never finishes, so if it
        # finishes first it died, and Meeko would otherwise keep listening
        # while never answering. Raise so the process exits non-zero and
        # systemd (Restart=on-failure) brings it back. It can return once
        # stop_event is set (it checks between turns, e.g. when the mic-queue
        # overflow shutdown lands mid-turn), which is not a death: wait for
        # the supervisor as usual.
        await asyncio.wait(
            {supervisor_task, turn_worker_task}, return_when=asyncio.FIRST_COMPLETED
        )
        if (
            turn_worker_task.done()
            and not supervisor_task.done()
            and not stop_event.is_set()
        ):
            # A worker that ended cancelled (a CancelledError leaking out of
            # a turn, not a barge-in) has no exception to chain, and calling
            # .exception() on it would raise CancelledError, which the except
            # below swallows: a clean exit that systemd won't restart. A real
            # shutdown never gets here: it cancels run() itself, so the wait
            # above raises first.
            cause = (
                None if turn_worker_task.cancelled() else turn_worker_task.exception()
            )
            # Log it too: the raise alone only reaches stderr at exit, so
            # with log_target = "file" meeko.log would show a restart with
            # no reason.
            logger.error(
                "Turn worker stopped unexpectedly; exiting so systemd restarts Meeko",
                exc_info=cause,
            )
            raise RuntimeError("Turn worker stopped unexpectedly") from cause
        await supervisor_task
    except asyncio.CancelledError:
        pass
    finally:
        stop_event.set()
        supervisor_task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await supervisor_task
        await idle.aclose()
        turn_worker_task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await turn_worker_task
        timer_manager.cancel_all_timers()
        # Before store.close(): an in-flight summary would otherwise write
        # into a closed SQLite connection.
        await summaries.aclose()
        audio.close()
        leds.close()
        await weather_client.aclose()
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

    def handle_signal():
        for task in asyncio.all_tasks(loop):
            task.cancel()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, handle_signal)

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
