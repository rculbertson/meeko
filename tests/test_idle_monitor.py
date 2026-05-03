"""Tests for the post-turn idle monitor (Stage 1: query mode)."""

import asyncio

import pytest

from meeko.main import _idle_monitor
from meeko.profiles import Profile


def _query_profile(timeout: float = 0.05) -> Profile:
    return Profile(
        name="test",
        wake_word="meeko",
        prompt="...",
        mode="query",
        idle_timeout_seconds=timeout,
    )


def _conversation_profile() -> Profile:
    return Profile(
        name="test",
        wake_word="meeko",
        prompt="...",
        mode="conversation",
        conversation_idle_seconds=60.0,
    )


async def test_query_mode_fires_after_timeout():
    """Monitor sleeps for the idle window then calls on_timeout."""
    fired = []

    def on_timeout() -> None:
        fired.append(True)

    profile = _query_profile(timeout=0.02)
    await _idle_monitor(profile, on_timeout)

    assert fired == [True]


async def test_query_mode_cancellation_does_not_fire():
    """Cancelling the task before the timer expires must not fire on_timeout."""
    fired = []

    def on_timeout() -> None:
        fired.append(True)

    profile = _query_profile(timeout=10.0)
    task = asyncio.create_task(_idle_monitor(profile, on_timeout))
    # Let the task start its sleep
    await asyncio.sleep(0.01)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert fired == []


async def test_conversation_mode_no_op_in_stage_1():
    """Conversation mode is a no-op in Stage 1; monitor returns immediately."""
    fired = []

    def on_timeout() -> None:
        fired.append(True)

    profile = _conversation_profile()
    await asyncio.wait_for(_idle_monitor(profile, on_timeout), timeout=0.5)

    assert fired == []
