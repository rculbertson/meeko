"""Meeko orchestrator: mic → Deepgram STT → Claude → Deepgram TTS → speaker.

This replaces the Deepgram Voice Agent wiring from the prototype. Claude
is called directly; STT and TTS are Deepgram-only. Audio still runs over
PyAudio with the default input/output device; ReSpeaker/sounddevice is
a later step. Mic is muted (silence fed to STT) while the assistant
speaks — true barge-in is also a later step.
"""

import asyncio
import functools
import logging
import logging.handlers
import os
import signal
import time
from enum import Enum, auto

import pyaudio
from dotenv import load_dotenv

from meeko.claude_client import ClaudeClient
from meeko.deepgram_stt import DeepgramSTT
from meeko.deepgram_tts import DEFAULT_VOICE, DeepgramTTS
from meeko.profiles import load_profiles
from meeko.tools.dispatch import ToolDispatcher
from meeko.tools.profile import ProfileManager
from meeko.tools.profile import get_tool_definitions as profile_tools
from meeko.tools.profile import handle as profile_handle
from meeko.tools.timer import get_tool_definitions as timer_tools
from meeko.tools.timer import handle as timer_handle
from meeko.tools.timer import timer_manager

RATE = 16000
CHANNELS = 1
FORMAT = pyaudio.paInt16
CHUNK = 800  # 50ms at 16kHz (800 samples * 2 bytes = 1600 bytes per chunk)
SILENCE_CHUNK = b"\x00" * (CHUNK * 2)

# Silence inserted between pipelined TTS sentences so back-to-back
# synthesis doesn't run into the next sentence without a natural pause.
# 200ms at 16kHz 16-bit mono = 6400 bytes.
INTER_SENTENCE_PAUSE_MS = 200
INTER_SENTENCE_SILENCE = b"\x00" * (INTER_SENTENCE_PAUSE_MS * RATE * 2 // 1000)

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

    pa = pyaudio.PyAudio()
    mic_queue: asyncio.Queue[bytes] = asyncio.Queue()
    loop = asyncio.get_event_loop()

    def mic_callback(in_data, frame_count, time_info, status):
        loop.call_soon_threadsafe(mic_queue.put_nowait, in_data)
        return (None, pyaudio.paContinue)

    mic_stream = pa.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK,
        stream_callback=mic_callback,
    )
    speaker_stream = pa.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        output=True,
        frames_per_buffer=CHUNK,
    )

    state = State.LISTENING
    stop_event = asyncio.Event()
    speak_lock = asyncio.Lock()

    async def speak_stream(texts) -> None:
        """Pipelined TTS + playback for an async iterator of text chunks.

        A producer task synthesizes each sentence into its own inner
        queue; a consumer drains inner queues in order and writes chunks
        to the speaker. At most pipeline_depth sentences are synthesized
        concurrently, so by the time sentence N finishes playing,
        sentence N+1's bytes are already queued — eliminating the TTS
        time-to-first-byte gap at sentence boundaries."""
        pipeline_depth = 2
        end_marker = object()

        nonlocal state
        async with speak_lock:
            prev = state
            state = State.SPEAKING
            try:
                voice = profile.voice or DEFAULT_VOICE
                t_start = time.perf_counter()
                t_first_play: float | None = None

                outer: asyncio.Queue = asyncio.Queue(maxsize=pipeline_depth)
                tts_tasks: list[asyncio.Task] = []

                async def tts_into(sentence: str, inner: asyncio.Queue) -> None:
                    try:
                        async for chunk in tts.stream(sentence, voice=voice):
                            await inner.put(chunk)
                    finally:
                        await inner.put(None)

                async def produce() -> None:
                    async for sentence in texts:
                        if not sentence:
                            continue
                        logger.info("[assistant] %s", sentence)
                        inner: asyncio.Queue = asyncio.Queue()
                        tts_tasks.append(asyncio.create_task(tts_into(sentence, inner)))
                        await outer.put(inner)
                    await outer.put(end_marker)

                async def consume() -> None:
                    nonlocal t_first_play
                    first_sentence = True
                    while True:
                        inner = await outer.get()
                        if inner is end_marker:
                            return
                        if not first_sentence:
                            await asyncio.to_thread(
                                speaker_stream.write, INTER_SENTENCE_SILENCE
                            )
                        first_sentence = False
                        while True:
                            chunk = await inner.get()
                            if chunk is None:
                                break
                            if t_first_play is None:
                                t_first_play = time.perf_counter()
                                logger.debug(
                                    "[timing] first_audio_to_speaker=%dms",
                                    int((t_first_play - t_start) * 1000),
                                )
                            # PyAudio write is blocking; run in a thread
                            # so the mic silence pump + STT loop run.
                            await asyncio.to_thread(speaker_stream.write, chunk)

                try:
                    await asyncio.gather(produce(), consume())
                finally:
                    # Surface any TTS task errors; also ensures tasks
                    # are cleaned up if produce/consume raised.
                    for t in tts_tasks:
                        if not t.done():
                            t.cancel()
                    await asyncio.gather(*tts_tasks, return_exceptions=True)

                logger.debug(
                    "[timing] speak_total=%dms",
                    int((time.perf_counter() - t_start) * 1000),
                )
                # Tail-drain: speaker buffer may still be flushing.
                await asyncio.sleep(1.0)
                while not mic_queue.empty():
                    mic_queue.get_nowait()
            finally:
                state = prev if prev != State.SPEAKING else State.LISTENING
                logger.debug("speak complete, state=%s", state.name)

    async def speak(text: str) -> None:
        """Single-utterance convenience wrapper (greeting, timer)."""
        if not text:
            return

        async def _one() -> asyncio.AsyncIterator[str]:
            yield text

        await speak_stream(_one())

    timer_manager.set_speak_callback(speak)

    logger.info("Connecting to Deepgram STT (Flux)...")

    async with stt.session() as stt_session:

        async def pump_mic():
            """Forward mic chunks to STT; send silence while SPEAKING."""
            while not stop_event.is_set():
                try:
                    data = await asyncio.wait_for(mic_queue.get(), timeout=0.1)
                except TimeoutError:
                    continue
                if state == State.SPEAKING:
                    await stt_session.send_audio(SILENCE_CHUNK)
                else:
                    await stt_session.send_audio(data)

        async def handle_turns():
            """Consume STT events, drive a Claude turn on EndOfTurn."""
            nonlocal state
            await speak(profile.greeting)
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
                        await speak_stream(claude.stream_turn(text))
                    except Exception:
                        logger.exception("Claude turn failed")
                        state = State.LISTENING
                        continue
                    logger.debug(
                        "[timing] turn_total_eot_to_speak_done=%dms",
                        int((time.perf_counter() - t_turn) * 1000),
                    )
                    state = State.LISTENING

        mic_stream.start_stream()
        logger.info("Mic active.")

        try:
            await asyncio.gather(pump_mic(), handle_turns())
        except asyncio.CancelledError:
            pass
        finally:
            stop_event.set()
            timer_manager.cancel_all_timers()
            mic_stream.stop_stream()
            mic_stream.close()
            speaker_stream.stop_stream()
            speaker_stream.close()
            pa.terminate()
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
