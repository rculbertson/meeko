import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock

import pytest

from meeko.tools.dispatch import ToolDispatcher, _get
from meeko.tools.timer import (
    TimerManager,
    get_tool_definitions,
    timer_manager,
)
from meeko.tools.timer import (
    handle as timer_handle,
)

logger = logging.getLogger(__name__)

SILENCE_CHUNK = b"\x00" * 1600  # 50ms of silence at 16kHz mono 16-bit


# ---------------------------------------------------------------------------
# Unit tests — mock connection, patch asyncio.sleep for instant expiry
# ---------------------------------------------------------------------------


async def test_named_timer_announcement():
    """A timer with a user-provided label announces both name and duration."""
    mgr = TimerManager()
    conn = MagicMock()
    conn.send_inject_agent_message = AsyncMock()

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(asyncio, "sleep", AsyncMock())
        await mgr.set_timer(
            duration_seconds=600,
            duration_display="10 minutes",
            label="pasta",
            connection=conn,
        )
        # Await the background task so it completes
        task = mgr._timers["pasta"][0]
        await task

    conn.send_inject_agent_message.assert_called_once()
    msg = conn.send_inject_agent_message.call_args[0][0].message
    assert msg == "The 10 minute pasta timer is done!"


async def test_unnamed_timer_announcement():
    """A timer without a label announces only the duration."""
    mgr = TimerManager()
    conn = MagicMock()
    conn.send_inject_agent_message = AsyncMock()

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(asyncio, "sleep", AsyncMock())
        await mgr.set_timer(
            duration_seconds=30,
            duration_display="30 seconds",
            label=None,
            connection=conn,
        )
        # Auto-labeled as "timer 1"
        task = list(mgr._timers.values())[0][0]
        await task

    conn.send_inject_agent_message.assert_called_once()
    msg = conn.send_inject_agent_message.call_args[0][0].message
    assert msg == "The 30 second timer is done!"


async def test_duration_display_singularized():
    """The announcement singularizes plural time units."""
    mgr = TimerManager()
    conn = MagicMock()
    conn.send_inject_agent_message = AsyncMock()

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(asyncio, "sleep", AsyncMock())
        await mgr.set_timer(
            duration_seconds=120,
            duration_display="120 seconds",
            label="eggs",
            connection=conn,
        )
        task = mgr._timers["eggs"][0]
        await task

    msg = conn.send_inject_agent_message.call_args[0][0].message
    assert "120 second" in msg
    assert msg == "The 120 second eggs timer is done!"


async def test_cancel_all_timers():
    """cancel_all_timers cancels every active timer and clears the dict."""
    mgr = TimerManager()
    conn = MagicMock()
    conn.send_inject_agent_message = AsyncMock()

    with pytest.MonkeyPatch.context() as mp:
        real_sleep = asyncio.sleep

        async def long_sleep(duration):
            await real_sleep(3600)

        mp.setattr(asyncio, "sleep", long_sleep)
        await mgr.set_timer(300, "5 minute", "pasta", conn)
        await mgr.set_timer(600, "10 minute", "eggs", conn)
        await mgr.set_timer(60, "1 minute", None, conn)

    assert len(mgr._timers) == 3
    result = mgr.cancel_all_timers()
    assert "3 timers cancelled" in result
    assert len(mgr._timers) == 0


async def test_cancel_all_timers_empty():
    """cancel_all_timers on an empty manager returns a no-timers message."""
    mgr = TimerManager()
    result = mgr.cancel_all_timers()
    assert result == "No active timers."


@pytest.mark.integration
async def test_set_timer_function_call(dg_client, agent_settings, wav_chunks_set_timer):
    """Send a 'set a timer' request and verify the function calling round-trip.

    Validates:
    - Voice Agent emits a FunctionCallRequest for set_timer
    - Our handler sends FunctionCallResponse with correct id
    - Agent acknowledges the timer in a spoken response
    """
    dispatcher = ToolDispatcher()
    dispatcher.register(get_tool_definitions(), timer_handle)

    events = []
    function_calls = []
    settings_applied = asyncio.Event()
    got_response = asyncio.Event()
    greeting_done = asyncio.Event()

    async with dg_client.agent.v1.connect() as connection:
        await connection.send_settings(agent_settings)

        async def send_audio():
            await settings_applied.wait()
            await greeting_done.wait()
            for chunk in wav_chunks_set_timer:
                await connection.send_media(chunk)
                await asyncio.sleep(0.05)
            # Send silence to trigger end-of-speech detection
            for _ in range(20):
                await connection.send_media(SILENCE_CHUNK)
                await asyncio.sleep(0.05)

        async def receive_events():
            has_function_call = False
            async for message in connection:
                if isinstance(message, bytes):
                    continue
                msg_type = getattr(message, "type", "")
                logger.info("Event: %s", msg_type)

                if msg_type == "SettingsApplied":
                    settings_applied.set()
                elif msg_type == "ConversationText":
                    role = getattr(message, "role", "")
                    content = getattr(message, "content", "")
                    logger.info("  [%s] %s", role, content)
                    events.append({"role": role, "content": content})
                elif msg_type == "FunctionCallRequest":
                    functions = _get(message, "functions", [])
                    for fn in functions:
                        function_calls.append(
                            {
                                "id": _get(fn, "id"),
                                "name": _get(fn, "name"),
                                "arguments": _get(fn, "arguments"),
                            }
                        )
                    await dispatcher.handle_function_call_request(message, connection)
                    has_function_call = True
                elif msg_type == "AgentAudioDone":
                    if not greeting_done.is_set():
                        greeting_done.set()
                    elif has_function_call:
                        got_response.set()
                elif msg_type == "Error":
                    logger.warning("Error event: %s", message)
                    if has_function_call:
                        got_response.set()

        send_task = asyncio.create_task(send_audio())
        recv_task = asyncio.create_task(receive_events())

        try:
            await asyncio.wait_for(got_response.wait(), timeout=30.0)
        except TimeoutError:
            pytest.fail(
                f"Timed out waiting for agent response. "
                f"Events: {events}, Function calls: {function_calls}"
            )
        finally:
            # Cancel any timers that were set during the test
            for label in list(timer_manager._timers):
                timer_manager.cancel_timer(label)
            send_task.cancel()
            recv_task.cancel()
            for task in (send_task, recv_task):
                try:
                    await task
                except asyncio.CancelledError:
                    pass

    # --- Assertions ---

    # A FunctionCallRequest for set_timer was received
    assert len(function_calls) >= 1, (
        f"Expected at least one function call, got: {function_calls}"
    )
    timer_calls = [fc for fc in function_calls if fc["name"] == "set_timer"]
    assert len(timer_calls) >= 1, (
        f"Expected a set_timer function call, got: {function_calls}"
    )

    # The function call had a non-empty id (needed for response correlation)
    assert timer_calls[0]["id"], "Function call id should not be empty"

    # Agent produced a spoken acknowledgment after the function call
    agent_texts = [e for e in events if e["role"] == "assistant"]
    assert len(agent_texts) >= 2, (
        f"Expected greeting + at least one response, got: {events}"
    )
