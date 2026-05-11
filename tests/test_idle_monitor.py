"""Tests for the post-turn idle monitor (query and conversation modes)."""

import asyncio

import pytest

from meeko.config import Profile
from meeko.main import (
    CONVERSATION_CLOSE_TEXT,
    CONVERSATION_PROMPT_TEXT,
    _idle_monitor,
)


def _query_profile(timeout: float = 0.05) -> Profile:
    return Profile(
        name="test",
        wake_word="meeko",
        prompt="...",
        mode="query",
        idle_timeout_seconds=timeout,
    )


def _conversation_profile(idle: float = 0.02, close: float = 0.02) -> Profile:
    return Profile(
        name="test",
        wake_word="meeko",
        prompt="...",
        mode="conversation",
        conversation_idle_seconds=idle,
        conversation_close_seconds=close,
    )


def _recording_speak() -> tuple[list[str], callable]:
    calls: list[str] = []

    async def speak(text: str) -> None:
        calls.append(text)

    return calls, speak


# ---------------------------------------------------------------------------
# Query mode
# ---------------------------------------------------------------------------


async def test_query_mode_fires_after_timeout():
    """Monitor sleeps for the idle window then calls on_timeout."""
    fired = []

    def on_timeout() -> None:
        fired.append(True)

    await _idle_monitor(_query_profile(timeout=0.02), on_timeout)

    assert fired == [True]


async def test_query_mode_cancellation_does_not_fire():
    """Cancelling the task before the timer expires must not fire on_timeout."""
    fired = []

    def on_timeout() -> None:
        fired.append(True)

    task = asyncio.create_task(_idle_monitor(_query_profile(timeout=10.0), on_timeout))
    await asyncio.sleep(0.01)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert fired == []


async def test_query_mode_disabled_when_timeout_non_positive():
    """idle_timeout_seconds <= 0 disables the monitor entirely."""
    fired = []

    def on_timeout() -> None:
        fired.append(True)

    await _idle_monitor(_query_profile(timeout=0), on_timeout)

    assert fired == []


# ---------------------------------------------------------------------------
# Conversation mode
# ---------------------------------------------------------------------------


async def test_conversation_mode_full_flow():
    """Idle window → prompt → close window → close line → on_timeout."""
    fired = []
    calls, speak = _recording_speak()

    def on_timeout() -> None:
        fired.append(True)

    await _idle_monitor(
        _conversation_profile(idle=0.02, close=0.02),
        on_timeout,
        speak=speak,
    )

    assert calls == [CONVERSATION_PROMPT_TEXT, CONVERSATION_CLOSE_TEXT]
    assert fired == [True]


async def test_conversation_mode_cancel_during_idle_wait():
    """Cancelling during the initial idle window: no prompt, no close, no end."""
    fired = []
    calls, speak = _recording_speak()

    def on_timeout() -> None:
        fired.append(True)

    task = asyncio.create_task(
        _idle_monitor(
            _conversation_profile(idle=10.0, close=10.0),
            on_timeout,
            speak=speak,
        )
    )
    await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert calls == []
    assert fired == []


async def test_conversation_mode_cancel_during_close_window():
    """Cancelling after the prompt but before the close window expires."""
    fired = []
    calls, speak = _recording_speak()

    def on_timeout() -> None:
        fired.append(True)

    task = asyncio.create_task(
        _idle_monitor(
            _conversation_profile(idle=0.01, close=10.0),
            on_timeout,
            speak=speak,
        )
    )
    # Wait long enough for the prompt to be spoken but well within the
    # 10s close window.
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert calls == [CONVERSATION_PROMPT_TEXT]
    assert fired == []


async def test_conversation_mode_cancel_during_prompt_speak():
    """Cancelling while the prompt is being spoken: no close, no end."""
    fired = []
    calls: list[str] = []
    started = asyncio.Event()

    async def slow_speak(text: str) -> None:
        calls.append(text)
        started.set()
        await asyncio.sleep(10.0)  # simulate long TTS

    def on_timeout() -> None:
        fired.append(True)

    task = asyncio.create_task(
        _idle_monitor(
            _conversation_profile(idle=0.01, close=0.5),
            on_timeout,
            speak=slow_speak,
        )
    )
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert calls == [CONVERSATION_PROMPT_TEXT]
    assert fired == []


async def test_conversation_mode_disabled_when_idle_non_positive():
    """conversation_idle_seconds <= 0 disables the monitor entirely."""
    fired = []
    calls, speak = _recording_speak()

    def on_timeout() -> None:
        fired.append(True)

    await _idle_monitor(
        _conversation_profile(idle=0, close=10.0),
        on_timeout,
        speak=speak,
    )

    assert calls == []
    assert fired == []


async def test_conversation_mode_close_window_runs_after_prompt_finishes():
    """The close window starts after the prompt-speak completes, so a
    slow prompt does not eat into the user's response time."""
    fired = []
    calls: list[str] = []
    prompt_speak_duration = 0.1
    close_window = 0.05

    async def slow_speak(text: str) -> None:
        calls.append(text)
        if text == CONVERSATION_PROMPT_TEXT:
            await asyncio.sleep(prompt_speak_duration)

    def on_timeout() -> None:
        fired.append(True)

    loop = asyncio.get_running_loop()
    t_start = loop.time()
    await _idle_monitor(
        _conversation_profile(idle=0.01, close=close_window),
        on_timeout,
        speak=slow_speak,
    )
    elapsed = loop.time() - t_start

    # Total ≈ idle (0.01) + prompt-speak (0.1) + full close window
    # (0.05) ≈ 0.16s. If the close window were absolute from prompt
    # start instead, the wait would be skipped (close < prompt_speak)
    # and total would be ≈ 0.11s.
    assert calls == [CONVERSATION_PROMPT_TEXT, CONVERSATION_CLOSE_TEXT]
    assert fired == [True]
    assert elapsed >= 0.01 + prompt_speak_duration + close_window, (
        f"close window not given after prompt finishes: {elapsed:.3f}s"
    )
