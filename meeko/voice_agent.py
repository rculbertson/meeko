"""Deepgram Voice Agent session with live mic input and speaker output."""

import asyncio
import functools
import logging
import logging.handlers
import os
import signal

import pyaudio
from deepgram import AsyncDeepgramClient
from deepgram.agent.v1.types import (
    AgentV1Settings,
    AgentV1SettingsAgent,
    AgentV1SettingsAgentListen,
    AgentV1SettingsAgentListenProvider_V1,
    AgentV1SettingsAgentSpeakEndpoint,
    AgentV1SettingsAgentSpeakEndpointProvider_Deepgram,
    AgentV1SettingsAgentThinkOneItem,
    AgentV1SettingsAgentThinkOneItemEndpoint,
    AgentV1SettingsAudio,
    AgentV1SettingsAudioInput,
    AgentV1SettingsAudioOutput,
)
from dotenv import load_dotenv

from meeko.profiles import Profile, load_profiles
from meeko.tools.dispatch import ToolDispatcher
from meeko.tools.profile import ProfileManager
from meeko.tools.profile import get_tool_definitions as profile_tools
from meeko.tools.profile import handle as profile_handle
from meeko.tools.timer import get_tool_definitions as timer_tools
from meeko.tools.timer import handle as timer_handle

RATE = 16000
CHANNELS = 1
FORMAT = pyaudio.paInt16
CHUNK = 800  # 50ms at 16kHz (800 samples * 2 bytes = 1600 bytes per chunk)


def build_settings(
    anthropic_api_key: str,
    profile: Profile,
    tool_definitions: list,
) -> AgentV1Settings:
    return AgentV1Settings(
        type="Settings",
        audio=AgentV1SettingsAudio(
            input=AgentV1SettingsAudioInput(
                encoding="linear16",
                sample_rate=RATE,
            ),
            output=AgentV1SettingsAudioOutput(
                encoding="linear16",
                sample_rate=RATE,
                container="none",
            ),
        ),
        agent=AgentV1SettingsAgent(
            listen=AgentV1SettingsAgentListen(
                provider=AgentV1SettingsAgentListenProvider_V1(
                    type="deepgram",
                    model="nova-3",
                ),
            ),
            think=[
                AgentV1SettingsAgentThinkOneItem(
                    provider={"type": "anthropic", "model": "claude-sonnet-4-6"},
                    endpoint=AgentV1SettingsAgentThinkOneItemEndpoint(
                        url="https://api.anthropic.com/v1/messages",
                        headers={"x-api-key": anthropic_api_key},
                    ),
                    prompt=profile.prompt,
                    functions=tool_definitions,
                )
            ],
            speak=AgentV1SettingsAgentSpeakEndpoint(
                provider=AgentV1SettingsAgentSpeakEndpointProvider_Deepgram(
                    model="aura-2-asteria-en",
                ),
            ),
            greeting=profile.greeting,
        ),
    )


LOG_FILE = "meeko.log"

logger = logging.getLogger("meeko")


def setup_logging():
    formatter = logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M:%S")

    if os.environ.get("MEEKO_LOG_TARGET") == "file":
        # Rotating file: 5 MB per file, keep 3 backups (20 MB max)
        handler = logging.handlers.RotatingFileHandler(
            LOG_FILE, maxBytes=5_000_000, backupCount=3
        )
    else:
        handler = logging.StreamHandler()

    handler.setFormatter(formatter)
    logger.addHandler(handler)
    level = os.environ.get("MEEKO_LOG_LEVEL", "DEBUG").upper()
    logger.setLevel(getattr(logging, level, logging.DEBUG))


async def run():
    setup_logging()
    load_dotenv()

    deepgram_key = os.environ["DEEPGRAM_API_KEY"]
    anthropic_key = os.environ["CLAUDE_API_KEY"]

    # Load profiles and set up tool dispatcher
    profiles = load_profiles()
    profile = profiles["default"]

    profile_manager = ProfileManager(profiles)

    dispatcher = ToolDispatcher()
    dispatcher.register(timer_tools(), timer_handle)
    dispatcher.register(
        profile_tools(profiles),
        functools.partial(profile_handle, manager=profile_manager),
    )

    client = AsyncDeepgramClient(api_key=deepgram_key)
    settings = build_settings(anthropic_key, profile, dispatcher.get_all_definitions())

    pa = pyaudio.PyAudio()

    # Queue to pass mic audio from PyAudio callback thread to async send loop
    mic_queue: asyncio.Queue[bytes] = asyncio.Queue()
    loop = asyncio.get_event_loop()

    def mic_callback(in_data, frame_count, time_info, status):
        loop.call_soon_threadsafe(mic_queue.put_nowait, in_data)
        return (None, pyaudio.paContinue)

    # Open mic stream (callback mode — non-blocking)
    mic_stream = pa.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK,
        stream_callback=mic_callback,
    )

    # Open speaker stream
    speaker_stream = pa.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        output=True,
        frames_per_buffer=CHUNK,
    )

    stop_event = asyncio.Event()

    # Mute the mic while the agent is speaking to prevent the mic from picking
    # up TTS audio and feeding it back to Deepgram (echo/feedback loop). This
    # prevents barge-in (talking over the agent), which we eventually want.
    # Hardware with built-in acoustic echo cancellation (e.g. a conference
    # speakerphone) may eliminate this problem and allow us to remove the mute.
    agent_speaking = False

    logger.info("Connecting to Deepgram Voice Agent...")

    async with client.agent.v1.connect() as connection:
        logger.info("Connected. Sending settings...")
        await connection.send_settings(settings)

        async def send_mic_audio():
            """Read mic chunks from queue and send to Voice Agent."""
            nonlocal agent_speaking
            chunks_sent = 0
            while not stop_event.is_set():
                try:
                    data = await asyncio.wait_for(mic_queue.get(), timeout=0.1)
                    if agent_speaking:
                        # Send silence to keep the WebSocket alive
                        await connection.send_media(b"\x00" * len(data))
                    else:
                        await connection.send_media(data)
                    chunks_sent += 1
                    if chunks_sent % 100 == 0:
                        logger.debug("mic: sent %d chunks", chunks_sent)
                except TimeoutError:
                    continue

        async def receive_messages():
            """Receive and handle messages from Voice Agent."""
            nonlocal agent_speaking
            async for message in connection:
                if stop_event.is_set():
                    break

                if isinstance(message, bytes):
                    agent_speaking = True
                    logger.debug("recv: audio %d bytes", len(message))
                    speaker_stream.write(message)
                    continue

                # The SDK delivers all non-bytes messages as
                # AgentV1PromptUpdated, so isinstance checks against
                # specific types (e.g. AgentV1AgentAudioDone) never
                # match. Use the ``type`` attribute instead.
                msg_type = getattr(message, "type", "")
                logger.debug("recv: %s %s", msg_type, message)

                if msg_type == "Welcome":
                    logger.info("Connected to Voice Agent")
                elif msg_type == "SettingsApplied":
                    logger.info("Settings applied. Listening...")
                elif msg_type == "ConversationText":
                    role = getattr(message, "role", "?")
                    content = getattr(message, "content", "")
                    logger.info("[%s] %s", role, content)
                elif msg_type == "UserStartedSpeaking":
                    logger.info("User started speaking")
                elif msg_type == "AgentThinking":
                    logger.debug("Agent thinking...")
                elif msg_type == "AgentStartedSpeaking":
                    logger.debug("Agent started speaking")
                    agent_speaking = True
                elif msg_type == "AgentAudioDone":
                    logger.debug("Agent audio done")
                    # Delay before unmuting mic — the speaker buffer
                    # may still be playing the tail end of TTS audio.
                    await asyncio.sleep(1.0)
                    # Drain any mic audio captured during agent speech
                    while not mic_queue.empty():
                        mic_queue.get_nowait()
                    agent_speaking = False
                    logger.debug("mic unmuted")
                elif msg_type == "FunctionCallRequest":
                    await dispatcher.handle_function_call_request(message, connection)
                elif msg_type == "Error":
                    logger.error("Agent error: %s", message)
                else:
                    logger.debug("Unhandled message type: %s", msg_type)

        mic_stream.start_stream()
        logger.info("Mic active.")

        try:
            await asyncio.gather(send_mic_audio(), receive_messages())
        except asyncio.CancelledError:
            pass
        finally:
            stop_event.set()
            mic_stream.stop_stream()
            mic_stream.close()
            speaker_stream.stop_stream()
            speaker_stream.close()
            pa.terminate()
            logger.info("Shutting down.")


def main():
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
