import asyncio
from unittest.mock import AsyncMock

import pytest

from meeko.tools.timer import TimerManager


async def test_named_timer_announcement():
    """A labeled timer announces both name and duration via speak_callback."""
    speak = AsyncMock()
    mgr = TimerManager(speak_callback=speak)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(asyncio, "sleep", AsyncMock())
        await mgr.set_timer(
            duration_seconds=600,
            duration_display="10 minutes",
            label="pasta",
        )
        task = mgr._timers["pasta"][0]
        await task

    speak.assert_awaited_once_with("The 10 minute pasta timer is done!")


async def test_unnamed_timer_announcement():
    """An unlabeled timer announces only the duration."""
    speak = AsyncMock()
    mgr = TimerManager(speak_callback=speak)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(asyncio, "sleep", AsyncMock())
        await mgr.set_timer(
            duration_seconds=30,
            duration_display="30 seconds",
            label=None,
        )
        task = list(mgr._timers.values())[0][0]
        await task

    speak.assert_awaited_once_with("The 30 second timer is done!")


async def test_duration_display_singularized():
    speak = AsyncMock()
    mgr = TimerManager(speak_callback=speak)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(asyncio, "sleep", AsyncMock())
        await mgr.set_timer(
            duration_seconds=120,
            duration_display="120 seconds",
            label="eggs",
        )
        await mgr._timers["eggs"][0]

    speak.assert_awaited_once_with("The 120 second eggs timer is done!")


async def test_cancel_all_timers():
    mgr = TimerManager(speak_callback=AsyncMock())

    with pytest.MonkeyPatch.context() as mp:
        real_sleep = asyncio.sleep

        async def long_sleep(_duration):
            await real_sleep(3600)

        mp.setattr(asyncio, "sleep", long_sleep)
        await mgr.set_timer(300, "5 minute", "pasta")
        await mgr.set_timer(600, "10 minute", "eggs")
        await mgr.set_timer(60, "1 minute", None)

    assert len(mgr._timers) == 3
    result = mgr.cancel_all_timers()
    assert "3 timers cancelled" in result
    assert len(mgr._timers) == 0


async def test_cancel_all_timers_empty():
    mgr = TimerManager()
    assert mgr.cancel_all_timers() == "No active timers."


async def test_expiry_without_callback_does_not_raise():
    """If no speak_callback is configured, expiry logs a warning and cleans up."""
    mgr = TimerManager()

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(asyncio, "sleep", AsyncMock())
        await mgr.set_timer(10, "10 second", "orphan")
        task = mgr._timers["orphan"][0]
        await task

    assert "orphan" not in mgr._timers
