import asyncio
import logging

import pytest
from rapidfuzz import fuzz

logger = logging.getLogger(__name__)

EXPECTED_TRANSCRIPT = "hello meeko what is the capital of france"
SILENCE_CHUNK = b"\x00" * 1600  # 50ms of silence at 16kHz mono 16-bit


@pytest.mark.integration
async def test_voice_agent_responds_to_question(
    dg_client, agent_settings, wav_chunks_hello
):
    """Send a WAV question to the Voice Agent, validate STT + LLM."""
    events = []
    settings_applied = asyncio.Event()
    got_response = asyncio.Event()
    greeting_done = asyncio.Event()

    async with dg_client.agent.v1.connect() as connection:
        await connection.send_settings(agent_settings)

        async def send_audio():
            await settings_applied.wait()
            # Wait for greeting audio to finish before sending user audio
            await greeting_done.wait()
            # Stream audio at real-time pace (50ms per 1600-byte chunk)
            for chunk in wav_chunks_hello:
                await connection.send_media(chunk)
                await asyncio.sleep(0.05)
            # Send silence to trigger end-of-speech detection
            for _ in range(20):  # 1 second of silence
                await connection.send_media(SILENCE_CHUNK)
                await asyncio.sleep(0.05)

        async def receive_events():
            has_user_text = False
            async for message in connection:
                if isinstance(message, bytes):
                    continue  # skip TTS audio
                msg_type = getattr(message, "type", "")
                logger.info("Event: %s", msg_type)
                if msg_type == "SettingsApplied":
                    settings_applied.set()
                elif msg_type == "ConversationText":
                    role = getattr(message, "role", "")
                    content = getattr(message, "content", "")
                    logger.info("  [%s] %s", role, content)
                    events.append({"role": role, "content": content})
                    if role == "user":
                        has_user_text = True
                elif msg_type == "AgentAudioDone":
                    if not greeting_done.is_set():
                        greeting_done.set()
                    elif has_user_text:
                        got_response.set()
                elif msg_type == "Error":
                    logger.warning("Error event received")
                    # Connection may close after this — if we already have
                    # user text and agent responses, treat as done
                    if has_user_text:
                        got_response.set()

        send_task = asyncio.create_task(send_audio())
        recv_task = asyncio.create_task(receive_events())

        try:
            await asyncio.wait_for(got_response.wait(), timeout=30.0)
        except TimeoutError:
            pytest.fail(
                f"Timed out waiting for agent response. Events so far: {events}"
            )
        finally:
            send_task.cancel()
            recv_task.cancel()
            for task in (send_task, recv_task):
                try:
                    await task
                except asyncio.CancelledError:
                    pass

    # --- Assertions ---
    user_texts = [e for e in events if e["role"] == "user"]
    agent_texts = [e for e in events if e["role"] == "assistant"]

    assert len(user_texts) >= 1, f"Expected at least one user transcript, got: {events}"
    assert len(agent_texts) >= 2, (
        f"Expected greeting + at least one response, got: {events}"
    )

    # Fuzzy match STT transcript against known input text
    transcript = " ".join(e["content"] for e in user_texts).lower()
    score = fuzz.partial_ratio(transcript, EXPECTED_TRANSCRIPT)
    assert score > 70, f"STT transcript too different (score={score}): {transcript!r}"

    # Check all non-greeting assistant messages for the expected answer
    all_responses = " ".join(e["content"] for e in agent_texts[1:]).lower()
    assert "paris" in all_responses, (
        f"Expected 'paris' in agent response: {all_responses!r}"
    )
