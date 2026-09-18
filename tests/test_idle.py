"""Tests for meeko.orchestrator.idle: the idle windows and the controller
that arms them.
"""

import asyncio

import pytest

from meeko.config import Profile
from meeko.orchestrator.idle import (
    CONVERSATION_CLOSE_TEXT,
    CONVERSATION_PROMPT_TEXT,
    IDLE_TIMEOUT_SENTINEL,
    IdleController,
    run_idle_window,
)


def _query_profile(timeout: float = 0.05) -> Profile:
    return Profile(
        name="query",
        wake_word="meeko",
        prompt="...",
        idle_timeout_seconds=timeout,
    )


def _conversation_profile(idle: float = 0.02, close: float = 0.02) -> Profile:
    return Profile(
        name="conversation",
        wake_word="meeko",
        prompt="...",
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

    await run_idle_window(_query_profile(timeout=0.02), on_timeout)

    assert fired == [True]


async def test_query_mode_cancellation_does_not_fire():
    """Cancelling the task before the timer expires must not fire on_timeout."""
    fired = []

    def on_timeout() -> None:
        fired.append(True)

    task = asyncio.create_task(
        run_idle_window(_query_profile(timeout=10.0), on_timeout)
    )
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

    await run_idle_window(_query_profile(timeout=0), on_timeout)

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

    await run_idle_window(
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
        run_idle_window(
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
        run_idle_window(
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
        run_idle_window(
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

    await run_idle_window(
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
    await run_idle_window(
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


# ---------------------------------------------------------------------------
# IdleController
#
# These exercise the arming/cancelling/race-guard logic that previously
# lived in closures inside run() and could only be reached by driving the
# whole orchestrator with fake STT/TTS/Anthropic clients.
# ---------------------------------------------------------------------------


class _FakeProfileManager:
    def __init__(self, profile: Profile):
        self.active_profile = profile


class _RecordingSessionManager:
    def __init__(self):
        self.end_requested = 0

    def request_end(self) -> None:
        self.end_requested += 1


def _controller(
    profile: Profile,
    *,
    listening: bool = True,
    queue: asyncio.Queue | None = None,
) -> tuple[IdleController, _RecordingSessionManager, asyncio.Queue, list[str]]:
    sessions = _RecordingSessionManager()
    turn_queue = queue if queue is not None else asyncio.Queue()
    spoken, speak = _recording_speak()
    controller = IdleController(
        is_listening=lambda: listening,
        profile_manager=_FakeProfileManager(profile),
        session_manager=sessions,
        turn_queue=turn_queue,
        speak=speak,
    )
    return controller, sessions, turn_queue, spoken


async def test_post_turn_timeout_ends_session_and_posts_sentinel():
    controller, sessions, turn_queue, _ = _controller(_query_profile(timeout=0.01))

    controller.start_post_turn()
    await asyncio.sleep(0.05)

    assert sessions.end_requested == 1
    assert turn_queue.get_nowait() is IDLE_TIMEOUT_SENTINEL


async def test_post_wake_timeout_ends_session_and_posts_sentinel():
    profile = _query_profile()
    profile = Profile(
        name=profile.name,
        wake_word=profile.wake_word,
        prompt=profile.prompt,
        post_wake_timeout_seconds=0.01,
    )
    controller, sessions, turn_queue, _ = _controller(profile)

    controller.start_post_wake()
    await asyncio.sleep(0.05)

    assert sessions.end_requested == 1
    assert turn_queue.get_nowait() is IDLE_TIMEOUT_SENTINEL


async def test_timeout_does_not_fire_when_no_longer_listening():
    """A turn is already in flight — the session must not be torn down."""
    controller, sessions, turn_queue, _ = _controller(
        _query_profile(timeout=0.01), listening=False
    )

    controller.start_post_turn()
    await asyncio.sleep(0.05)

    assert sessions.end_requested == 0
    assert turn_queue.empty()


async def test_timeout_does_not_fire_when_a_transcript_is_already_queued():
    """The race this guard exists for: the user spoke in the gap between
    the sleep waking and the callback running. Letting the sentinel land
    behind their text would end the session right after a good turn."""
    queue: asyncio.Queue = asyncio.Queue()
    queue.put_nowait("what's the weather")
    controller, sessions, turn_queue, _ = _controller(
        _query_profile(timeout=0.01), queue=queue
    )

    controller.start_post_turn()
    await asyncio.sleep(0.05)

    assert sessions.end_requested == 0
    assert turn_queue.get_nowait() == "what's the weather"
    assert turn_queue.empty()


async def test_start_post_wake_is_a_noop_when_timeout_non_positive():
    profile = Profile(
        name="query",
        wake_word="meeko",
        prompt="...",
        post_wake_timeout_seconds=0,
    )
    controller, sessions, turn_queue, _ = _controller(profile)

    controller.start_post_wake()
    await asyncio.sleep(0.05)

    assert sessions.end_requested == 0
    assert turn_queue.empty()


async def test_cancel_stops_a_pending_timeout():
    controller, sessions, turn_queue, _ = _controller(_query_profile(timeout=0.05))

    controller.start_post_turn()
    await asyncio.sleep(0.01)
    controller.cancel()
    await asyncio.sleep(0.1)

    assert sessions.end_requested == 0
    assert turn_queue.empty()


async def test_cancel_is_safe_before_any_window_is_armed():
    controller, _, _, _ = _controller(_query_profile())
    controller.cancel()
    controller.cancel()


async def test_arming_again_replaces_the_running_window():
    """Both windows share one task slot, so the second arm must cancel
    the first — otherwise two timers race to end the same session."""
    profile = Profile(
        name="query",
        wake_word="meeko",
        prompt="...",
        idle_timeout_seconds=0.02,
        post_wake_timeout_seconds=0.02,
    )
    controller, sessions, turn_queue, _ = _controller(profile)

    controller.start_post_turn()
    first = controller._task
    controller.start_post_wake()

    assert first is not controller._task
    # cancel() is scheduled, not immediate — let the loop deliver it.
    await asyncio.sleep(0)
    assert first.cancelled()
    await asyncio.sleep(0.1)

    # Exactly one window survived to fire.
    assert sessions.end_requested == 1
    assert turn_queue.qsize() == 1


async def test_aclose_awaits_the_cancelled_window():
    """Shutdown path: the task must be finished, not merely cancelled,
    or asyncio logs "Task was destroyed but it is pending"."""
    controller, _, _, _ = _controller(_query_profile(timeout=10.0))

    controller.start_post_turn()
    task = controller._task
    await asyncio.sleep(0.01)
    await controller.aclose()

    assert task.done()
    assert controller._task is None


async def test_aclose_is_safe_with_no_window_armed():
    controller, _, _, _ = _controller(_query_profile())
    await controller.aclose()


async def test_conversation_check_in_is_spoken_before_the_close():
    controller, sessions, turn_queue, spoken = _controller(
        _conversation_profile(idle=0.01, close=0.01)
    )

    controller.start_post_turn()
    await asyncio.sleep(0.1)

    assert spoken == [CONVERSATION_PROMPT_TEXT, CONVERSATION_CLOSE_TEXT]
    assert sessions.end_requested == 1
    assert turn_queue.get_nowait() is IDLE_TIMEOUT_SENTINEL


async def test_post_wake_timeout_does_not_fire_when_a_transcript_is_queued():
    """Same race as the post-turn guard: the user finally spoke just as
    the post-wake window expired."""
    profile = Profile(
        name="query",
        wake_word="meeko",
        prompt="...",
        post_wake_timeout_seconds=0.01,
    )
    queue: asyncio.Queue = asyncio.Queue()
    queue.put_nowait("hey, sorry — what's the weather")
    controller, sessions, turn_queue, _ = _controller(profile, queue=queue)

    controller.start_post_wake()
    await asyncio.sleep(0.05)

    assert sessions.end_requested == 0
    assert turn_queue.qsize() == 1
