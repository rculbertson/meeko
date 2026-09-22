import asyncio
from unittest.mock import AsyncMock

import pytest

from meeko.tools.timer import TimerManager, handle


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
        task = next(iter(mgr._timers.values()))[0]
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


async def test_resetting_a_label_keeps_the_new_timer_listed_and_cancellable():
    """Re-setting "pasta" cancels the old task, whose cleanup used to pop
    the label after the new timer was stored under it — so the new timer
    kept running but list_timers/cancel_timer could no longer see it."""
    mgr = TimerManager(speak_callback=AsyncMock())
    await mgr.set_timer(60, "1 minute", "pasta")
    old_task = mgr._timers["pasta"][0]
    await asyncio.sleep(0)  # the first timer is running, as it would be
    await mgr.set_timer(120, "2 minutes", "pasta")
    new_task = mgr._timers["pasta"][0]

    # Let the cancelled task run its cleanup.
    await asyncio.gather(old_task, return_exceptions=True)

    assert "pasta" in mgr.list_timers()
    assert mgr.cancel_timer("pasta") == "Timer 'pasta' cancelled."
    await asyncio.gather(new_task, return_exceptions=True)
    assert new_task.done()
    assert mgr.list_timers() == "No active timers."


async def test_failed_announcement_is_logged_and_the_timer_cleaned_up(caplog):
    """A TTS failure at expiry used to escape the timer task and surface only
    as "Task exception was never retrieved" when it was garbage-collected."""
    speak = AsyncMock(side_effect=RuntimeError("TTS unavailable"))
    mgr = TimerManager(speak_callback=speak)

    with pytest.MonkeyPatch.context() as mp, caplog.at_level("ERROR", "meeko"):
        mp.setattr(asyncio, "sleep", AsyncMock())
        await mgr.set_timer(60, "1 minute", "pasta")
        task = mgr._timers["pasta"][0]
        await task  # would raise RuntimeError before the fix

    assert "Timer announcement failed" in caplog.text
    # The label is user-chosen and stays out of the system log.
    assert "pasta" not in caplog.text
    assert mgr.list_timers() == "No active timers."


async def test_handle_with_injected_manager():
    mgr = TimerManager()
    res = await handle(
        "set_timer",
        {"duration_seconds": 60, "duration_display": "1 minute", "label": "tea"},
        manager=mgr,
    )
    assert "Timer 'tea' set" in res
    assert "tea" in mgr.list_timers()

    listed = await handle("list_timers", {}, manager=mgr)
    assert "tea" in listed

    cancelled = await handle("cancel_timer", {"label": "tea"}, manager=mgr)
    assert "Timer 'tea' cancelled." in cancelled
    assert mgr.list_timers() == "No active timers."
