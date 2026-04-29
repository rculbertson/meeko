"""Meeko orchestrator: mic → Deepgram STT → Claude → Deepgram TTS → speaker.

This replaces the Deepgram Voice Agent wiring from the prototype. Claude
is called directly; STT and TTS are Deepgram-only. Audio runs over
PyAudio.

By default the mic stays open during TTS — we rely on the ReSpeaker
XVF3800's hardware AEC to suppress echo, and any EndOfTurn that still
fires while SPEAKING is logged as `[echo?]` and ignored so it can't
start a spurious Claude turn. For Mac / no-AEC development, set
`MEEKO_MUTE_MIC_WHILE_SPEAKING=1`: mic chunks are dropped while
SPEAKING and the mic queue is drained after playback. True barge-in
(acting on `SpeechStarted` during SPEAKING) is a later step.
"""

import argparse
import asyncio
import functools
import logging
import logging.handlers
import os
import signal
import sys
import time
from enum import Enum, auto

import anthropic
from dotenv import load_dotenv
from websockets.exceptions import ConnectionClosed

from meeko.audio_io import AudioIO
from meeko.claude_client import ClaudeClient
from meeko.deepgram_stt import DeepgramSTT
from meeko.deepgram_tts import DeepgramTTS
from meeko.profiles import load_profiles
from meeko.session_summary import summarize_session
from meeko.sessions import SessionStore, default_db_path
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
from meeko.wake_word import DEFAULT_THRESHOLD, WakeWordDetector, default_model_path

LOG_FILE = "meeko.log"

logger = logging.getLogger("meeko")


def setup_logging() -> None:
    formatter = logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M:%S")

    if os.environ.get("MEEKO_LOG_TARGET") == "file":
        handler = logging.handlers.RotatingFileHandler(
            LOG_FILE, maxBytes=5_000_000, backupCount=3
        )
    else:
        handler = logging.StreamHandler()

    handler.setFormatter(formatter)
    logger.addHandler(handler)
    level = os.environ.get("MEEKO_LOG_LEVEL", "DEBUG").upper()
    logger.setLevel(getattr(logging, level, logging.DEBUG))


class State(Enum):
    IDLE = auto()
    LISTENING = auto()
    PROCESSING = auto()
    SPEAKING = auto()


RESUME_LATEST = "__latest__"


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


async def run(resume: str | None = None, list_sessions: bool = False):
    setup_logging()

    if list_sessions:
        store = SessionStore.open(default_db_path())
        try:
            await _list_sessions_cmd(store)
        finally:
            await store.close()
        return

    load_dotenv()

    deepgram_key = os.environ["DEEPGRAM_API_KEY"]
    anthropic_key = os.environ["ANTHROPIC_API_KEY"]
    mute_mic_while_speaking = os.environ.get(
        "MEEKO_MUTE_MIC_WHILE_SPEAKING", ""
    ).strip().lower() in {"1", "true", "yes"}
    wake_word_disabled = os.environ.get(
        "MEEKO_WAKE_WORD_DISABLED", ""
    ).strip().lower() in {"1", "true", "yes"}

    profiles = load_profiles()

    store = SessionStore.open(default_db_path())

    resumed_row: dict[str, object] | None = None
    if resume is not None:
        try:
            resumed_row = await _resolve_resume(store, resume)
        except SystemExit:
            # _resolve_resume raises SystemExit on unknown ids; make sure
            # the store we just opened doesn't leak past that exit.
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
    else:
        profile = profiles["default"]
        session_id = await store.create_session(profile.name)
        history = None
        logger.info("Started session %s (profile=%s)", session_id, profile.name)

    profile_manager = ProfileManager(profiles)
    session_manager = SessionManager()

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
            current_session_id=session_id,
        )

    dispatcher.register(session_tools(), session_handle_wrapper)

    claude = ClaudeClient(
        api_key=anthropic_key,
        system_prompt=profile.prompt,
        dispatcher=dispatcher,
        store=store,
        session_id=session_id,
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
    audio = AudioIO(stop_event)

    if wake_word_disabled:
        wake_detector = None
        state = State.LISTENING
    else:
        wake_model_path = os.environ.get("MEEKO_WAKE_WORD_MODEL", default_model_path())
        wake_threshold = float(
            os.environ.get("MEEKO_WAKE_WORD_THRESHOLD", DEFAULT_THRESHOLD)
        )
        wake_detector = WakeWordDetector(
            threshold=wake_threshold, model_path=wake_model_path
        )
        state = State.IDLE

    def enter_speaking() -> State:
        nonlocal state
        prev = state
        state = State.SPEAKING
        return prev

    def exit_speaking(prev: State) -> None:
        nonlocal state
        state = prev if prev != State.SPEAKING else State.LISTENING

    speaker = Speaker(
        tts,
        audio,
        profile,
        enter_speaking,
        exit_speaking,
        mute_mic_while_speaking=mute_mic_while_speaking,
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
        nonlocal state
        while not stop_event.is_set():
            try:
                data = await asyncio.wait_for(audio.mic_queue.get(), timeout=0.1)
            except TimeoutError:
                continue
            if state == State.IDLE:
                assert wake_detector is not None
                if wake_detector.process(data):
                    state = State.LISTENING
                    logger.info("Wake word accepted; entering LISTENING")
                continue
            if mute_mic_while_speaking and state == State.SPEAKING:
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

    async def handle_turns(stt_session):
        """Consume STT events, drive a Claude turn on EndOfTurn."""
        nonlocal state, session_id
        # If we want to make it faster, we can also use EagerEndOfTurn and
        # TurnResumed events which allows us to send text to the LLM eagerly.
        # If they're done talking, great, we already sent the text to the LLM.
        # If not, we cancel the LLM request (or discard the result), and send
        # the complete text. So may cost more since we throw away some results.
        async for ev in stt_session.events():
            if stop_event.is_set():
                return
            if ev.event == "StartOfTurn":
                logger.info("User started speaking (state=%s)", state.name)
            elif ev.event == "EndOfTurn":
                text = ev.transcript.strip()
                if state == State.IDLE:
                    # Safety belt: Deepgram shouldn't emit turns while
                    # we're gating mic audio behind the wake word, but
                    # any stray transcripts must not start a Claude turn.
                    continue
                if state == State.SPEAKING:
                    # AEC observation mode: don't start a Claude turn while
                    # we're talking. Log so we can gauge echo leakage.
                    logger.info("[echo?] %s", text)
                    continue
                logger.info("[user] %s", text)
                if not text:
                    continue
                state = State.PROCESSING
                t_turn = time.perf_counter()
                try:
                    await speaker.speak_stream(claude.stream_turn(text))
                except Exception:
                    logger.exception("Claude turn failed")
                    state = State.LISTENING
                    continue
                logger.debug(
                    "[timing] turn_total_eot_to_speak_done=%dms",
                    int((time.perf_counter() - t_turn) * 1000),
                )
                # should_load can coexist with should_end (chain: finalize
                # current session then load a prior one in one turn). Always
                # summarize the abandoned session so it stays in the recall
                # index — summarize_session no-ops on empty sessions, so
                # this is safe even when the user loads after only a turn
                # or two.
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
                    session_id = target_id
                    await store.touch_session(target_id)
                    session_manager.clear()
                    state = State.LISTENING
                    logger.info(
                        "load_session: swapped history to %s (%d turns), "
                        "continuing in LISTENING",
                        target_id[:8],
                        len(turns),
                    )
                elif session_manager.should_end():
                    active = profile_manager.active_profile
                    finalized_sid = session_id
                    new_sid = await store.create_session(active.name)
                    claude.reset_session(new_sid)
                    session_id = new_sid
                    session_manager.clear()
                    fire_summary(finalized_sid)
                    if wake_detector is not None:
                        wake_detector.reset()
                        state = State.IDLE
                        logger.info(
                            "Session ended; returning to IDLE "
                            "(say '%s' to start a new conversation)",
                            active.wake_word,
                        )
                    else:
                        state = State.LISTENING
                        logger.info(
                            "Session ended; wake word disabled, returning to LISTENING"
                        )
                elif session_manager.should_start_new():
                    active = profile_manager.active_profile
                    finalized_sid = session_id
                    new_sid = await store.create_session(active.name)
                    claude.reset_session(new_sid)
                    session_id = new_sid
                    session_manager.clear()
                    fire_summary(finalized_sid)
                    state = State.LISTENING
                    logger.info(
                        "new_session: rotated to %s (profile=%s), "
                        "continuing in LISTENING",
                        new_sid[:8],
                        active.name,
                    )
                else:
                    state = State.LISTENING

    async def on_session(stt_session) -> None:
        session_tasks = [
            asyncio.create_task(pump_mic(stt_session)),
            asyncio.create_task(handle_turns(stt_session)),
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
        stt, audio, stop_event, on_session, lambda: state == State.SPEAKING
    )

    audio.start_mic()
    logger.info("Mic active.")

    try:
        await supervisor.run()
    except asyncio.CancelledError:
        pass
    finally:
        stop_event.set()
        timer_manager.cancel_all_timers()
        # Cancel in-flight summary tasks before closing the SQLite
        # connection; letting them run into a closed store would crash
        # and a half-written summary is not worth the wait at shutdown.
        for task in summary_tasks:
            task.cancel()
        if summary_tasks:
            await asyncio.gather(*summary_tasks, return_exceptions=True)
        audio.close()
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
