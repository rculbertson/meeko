"""Meeko orchestrator: mic → Deepgram STT → Claude → Deepgram TTS → speaker.

This replaces the Deepgram Voice Agent wiring from the prototype. Claude
is called directly; STT and TTS are Deepgram-only. Audio still runs over
PyAudio with the default input/output device; ReSpeaker/sounddevice is
a later step. Mic is muted (silence fed to STT) while the assistant
speaks — true barge-in is also a later step, driven by Deepgram's
SpeechStarted event once hardware AEC (ReSpeaker XVF3800) is in place.
"""

import asyncio
import functools
import logging
import logging.handlers
import os
import signal
import time
from enum import Enum, auto

from dotenv import load_dotenv
from websockets.exceptions import ConnectionClosed

from meeko.audio_io import AudioIO
from meeko.claude_client import ClaudeClient
from meeko.deepgram_stt import DeepgramSTT
from meeko.deepgram_tts import DeepgramTTS
from meeko.profiles import load_profiles
from meeko.speaker import Speaker
from meeko.stt_supervisor import KEEPALIVE_INTERVAL_S, STTSupervisor
from meeko.tools.dispatch import ToolDispatcher
from meeko.tools.profile import ProfileManager
from meeko.tools.profile import get_tool_definitions as profile_tools
from meeko.tools.profile import handle as profile_handle
from meeko.tools.timer import get_tool_definitions as timer_tools
from meeko.tools.timer import handle as timer_handle
from meeko.tools.timer import timer_manager

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
    LISTENING = auto()
    PROCESSING = auto()
    SPEAKING = auto()


async def run():
    setup_logging()
    load_dotenv()

    deepgram_key = os.environ["DEEPGRAM_API_KEY"]
    anthropic_key = os.environ["ANTHROPIC_API_KEY"]

    profiles = load_profiles()
    profile = profiles["default"]

    profile_manager = ProfileManager(profiles)

    dispatcher = ToolDispatcher()
    dispatcher.register(timer_tools(), timer_handle)
    dispatcher.register(
        profile_tools(profiles),
        functools.partial(profile_handle, manager=profile_manager),
    )

    claude = ClaudeClient(
        api_key=anthropic_key,
        system_prompt=profile.prompt,
        dispatcher=dispatcher,
    )
    profile_manager.set_claude_client(claude)

    stt = DeepgramSTT(deepgram_key)
    tts = DeepgramTTS(deepgram_key)

    stop_event = asyncio.Event()
    audio = AudioIO(stop_event)

    state = State.LISTENING

    def enter_speaking() -> State:
        nonlocal state
        prev = state
        state = State.SPEAKING
        return prev

    def exit_speaking(prev: State) -> None:
        nonlocal state
        state = prev if prev != State.SPEAKING else State.LISTENING

    speaker = Speaker(tts, audio, profile, enter_speaking, exit_speaking)
    timer_manager.set_speak_callback(speaker.speak)

    async def pump_mic(stt_session):
        """Forward mic chunks to STT; drop them while SPEAKING.

        The keepalive_pump task holds the socket open while SPEAKING, so
        we don't need to fill the audio channel with silence — and
        dropping mic audio during speech removes the echo path on mics
        without hardware AEC."""
        while not stop_event.is_set():
            try:
                data = await asyncio.wait_for(audio.mic_queue.get(), timeout=0.1)
            except TimeoutError:
                continue
            if state == State.SPEAKING:
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
        nonlocal state
        # If we want to make it faster, we can also use EagerEndOfTurn and
        # TurnResumed events which allows us to send text to the LLM eagerly.
        # If they're done talking, great, we already sent the text to the LLM.
        # If not, we cancel the LLM request (or discard the result), and send
        # the complete text. So may cost more since we throw away some results.
        async for ev in stt_session.events():
            if stop_event.is_set():
                return
            if ev.event == "StartOfTurn":
                logger.info("User started speaking")
            elif ev.event == "EndOfTurn":
                text = ev.transcript.strip()
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
        await speaker.speak(profile.greeting)
        await supervisor.run()
    except asyncio.CancelledError:
        pass
    finally:
        stop_event.set()
        timer_manager.cancel_all_timers()
        audio.close()
        logger.info("Shutting down.")


def main() -> None:
    loop = asyncio.new_event_loop()

    def handle_sigint():
        for task in asyncio.all_tasks(loop):
            task.cancel()

    loop.add_signal_handler(signal.SIGINT, handle_sigint)

    try:
        loop.run_until_complete(run())
    except KeyboardInterrupt:
        pass
    finally:
        loop.close()


if __name__ == "__main__":
    main()
