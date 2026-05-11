"""Unit tests for `meeko.speaker.Speaker`."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from meeko.config import Profile
from meeko.speaker import INTER_SENTENCE_SILENCE, Speaker


@pytest.fixture
def profile():
    return Profile(
        name="default",
        wake_word="meeko",
        prompt="sys",
        voice=None,
    )


@pytest.fixture
def audio_mock():
    audio = MagicMock()
    audio.write_speaker = AsyncMock()
    audio.drain_mic_queue = MagicMock()
    return audio


@pytest.fixture
def tts_mock():
    """TTS that yields one bytes chunk per sentence (the sentence's
    text encoded) so tests can assert ordering from writes."""
    tts = MagicMock()

    def stream(text, voice):
        async def _gen():
            yield text.encode()

        return _gen()

    tts.stream.side_effect = stream
    return tts


@pytest.fixture
def fast_sleep():
    """Collapse the 1s tail-drain in speak_stream."""
    real_sleep = asyncio.sleep

    async def _fast(delay, *a, **kw):
        if delay >= 1.0:
            return await real_sleep(0)
        return await real_sleep(delay, *a, **kw)

    with patch("meeko.speaker.asyncio.sleep", new=_fast):
        yield


def _state_hooks():
    """Returns (enter, exit, calls) where calls records entry/exit pairs."""
    calls: list[tuple[str, object]] = []

    def enter():
        calls.append(("enter", "PREV"))
        return "PREV"

    def exit_(prev):
        calls.append(("exit", prev))

    return enter, exit_, calls


async def _aiter(items):
    for x in items:
        yield x


async def test_speak_empty_is_noop(profile, audio_mock, tts_mock, fast_sleep):
    enter, exit_, calls = _state_hooks()
    sp = Speaker(tts_mock, audio_mock, profile, enter, exit_)

    await sp.speak("")

    tts_mock.stream.assert_not_called()
    audio_mock.write_speaker.assert_not_awaited()
    assert calls == []


async def test_speak_stream_resets_audio_buffer_before_writes(
    profile, audio_mock, tts_mock, fast_sleep
):
    """Each utterance must start from a clean leftover buffer so a stale
    odd byte from a prior errored TTS stream can't byte-swap this
    utterance's samples. The reset must happen before the first
    write_speaker call."""
    enter, exit_, _ = _state_hooks()
    sp = Speaker(tts_mock, audio_mock, profile, enter, exit_)

    call_order: list[str] = []
    audio_mock.reset_speaker_buffer = MagicMock(
        side_effect=lambda: call_order.append("reset")
    )
    audio_mock.write_speaker = AsyncMock(
        side_effect=lambda _chunk: call_order.append("write")
    )

    await sp.speak_stream(_aiter(["Hi."]))

    assert call_order[0] == "reset"
    assert "write" in call_order


async def test_speak_stream_plays_sentences_in_order(
    profile, audio_mock, tts_mock, fast_sleep
):
    enter, exit_, _ = _state_hooks()
    sp = Speaker(tts_mock, audio_mock, profile, enter, exit_)

    await sp.speak_stream(_aiter(["First.", "Second.", "Third."]))

    # Collect only the non-silence payloads in order.
    writes = [c.args[0] for c in audio_mock.write_speaker.await_args_list]
    payloads = [w for w in writes if w != INTER_SENTENCE_SILENCE]
    assert payloads == [b"First.", b"Second.", b"Third."]


async def test_inter_sentence_silence_between_but_not_before_first(
    profile, audio_mock, tts_mock, fast_sleep
):
    enter, exit_, _ = _state_hooks()
    sp = Speaker(tts_mock, audio_mock, profile, enter, exit_)

    await sp.speak_stream(_aiter(["A.", "B."]))

    writes = [c.args[0] for c in audio_mock.write_speaker.await_args_list]
    # Expected: [b"A.", SILENCE, b"B."] — no silence before the first sentence.
    assert writes[0] == b"A."
    assert writes[1] == INTER_SENTENCE_SILENCE
    assert writes[2] == b"B."
    assert writes.count(INTER_SENTENCE_SILENCE) == 1


async def test_speak_stream_skips_empty_sentences(
    profile, audio_mock, tts_mock, fast_sleep
):
    enter, exit_, _ = _state_hooks()
    sp = Speaker(tts_mock, audio_mock, profile, enter, exit_)

    await sp.speak_stream(_aiter(["Hi.", "", "There."]))

    payloads = [
        c.args[0]
        for c in audio_mock.write_speaker.await_args_list
        if c.args[0] != INTER_SENTENCE_SILENCE
    ]
    assert payloads == [b"Hi.", b"There."]


async def test_state_hooks_called_symmetrically(
    profile, audio_mock, tts_mock, fast_sleep
):
    enter, exit_, calls = _state_hooks()
    sp = Speaker(tts_mock, audio_mock, profile, enter, exit_)

    await sp.speak_stream(_aiter(["Hi."]))

    assert [c[0] for c in calls] == ["enter", "exit"]
    assert calls[1][1] == "PREV"


async def test_state_hooks_called_on_exception(
    profile, audio_mock, tts_mock, fast_sleep
):
    enter, exit_, calls = _state_hooks()
    sp = Speaker(tts_mock, audio_mock, profile, enter, exit_)

    audio_mock.write_speaker.side_effect = RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        await sp.speak_stream(_aiter(["Hi."]))

    # Exit must still have run so the caller's state is restored.
    assert [c[0] for c in calls] == ["enter", "exit"]


async def test_speak_stream_does_not_drain_mic_when_mute_disabled(
    profile, audio_mock, tts_mock, fast_sleep
):
    """Default (hardware-AEC) path: speak_stream must leave the mic
    queue alone so STT sees everything the mic hears."""
    enter, exit_, _ = _state_hooks()
    sp = Speaker(tts_mock, audio_mock, profile, enter, exit_)

    await sp.speak_stream(_aiter(["Hi."]))

    audio_mock.drain_mic_queue.assert_not_called()


async def test_speak_stream_drains_mic_when_mute_enabled(
    profile, audio_mock, tts_mock, fast_sleep
):
    """Mac / no-AEC path: speak_stream drains buffered echo after
    playback so it isn't flushed to STT as a phantom user turn."""
    enter, exit_, _ = _state_hooks()
    sp = Speaker(
        tts_mock, audio_mock, profile, enter, exit_, mute_mic_while_speaking=True
    )

    await sp.speak_stream(_aiter(["Hi."]))

    audio_mock.drain_mic_queue.assert_called_once()


async def test_speak_lock_serializes_concurrent_callers(
    profile, audio_mock, tts_mock, fast_sleep
):
    """Two speak() calls started concurrently must not interleave:
    the second waits until the first fully exits."""
    enter, exit_, calls = _state_hooks()
    sp = Speaker(tts_mock, audio_mock, profile, enter, exit_)

    await asyncio.gather(sp.speak("first"), sp.speak("second"))

    # Exactly one entry per speak — no nesting.
    assert [c[0] for c in calls] == ["enter", "exit", "enter", "exit"]


async def test_default_voice_used_when_profile_voice_none(
    profile, audio_mock, tts_mock, fast_sleep
):
    enter, exit_, _ = _state_hooks()
    sp = Speaker(tts_mock, audio_mock, profile, enter, exit_)

    await sp.speak_stream(_aiter(["Hi."]))

    _, kwargs = tts_mock.stream.call_args
    from meeko.deepgram_tts import DEFAULT_VOICE

    assert kwargs["voice"] == DEFAULT_VOICE


async def test_profile_voice_overrides_default(audio_mock, tts_mock, fast_sleep):
    profile = Profile(
        name="custom",
        wake_word="meeko",
        prompt="sys",
        voice="aura-2-andromeda-en",
    )
    enter, exit_, _ = _state_hooks()
    sp = Speaker(tts_mock, audio_mock, profile, enter, exit_)

    await sp.speak_stream(_aiter(["Hi."]))

    _, kwargs = tts_mock.stream.call_args
    assert kwargs["voice"] == "aura-2-andromeda-en"


async def test_state_hooks_not_called_when_no_sentences(
    profile, audio_mock, tts_mock, fast_sleep
):
    """If the text iterator yields nothing, no audio is ever played, so
    the SPEAKING transition must not happen — state stays at PROCESSING
    and the orchestrator handles the post-turn flow."""
    enter, exit_, calls = _state_hooks()
    sp = Speaker(tts_mock, audio_mock, profile, enter, exit_)

    await sp.speak_stream(_aiter([]))

    assert calls == []
    audio_mock.write_speaker.assert_not_awaited()


async def test_state_hooks_not_called_when_only_empty_sentences(
    profile, audio_mock, tts_mock, fast_sleep
):
    """Empty sentences are skipped before TTS. With no chunks ever
    written, SPEAKING must not be entered."""
    enter, exit_, calls = _state_hooks()
    sp = Speaker(tts_mock, audio_mock, profile, enter, exit_)

    await sp.speak_stream(_aiter(["", ""]))

    assert calls == []
    audio_mock.write_speaker.assert_not_awaited()


async def test_enter_speaking_deferred_until_first_chunk(
    profile, audio_mock, tts_mock, fast_sleep
):
    """SPEAKING transition must coincide with the first audio write,
    not with the start of speak_stream — otherwise PROCESSING LEDs
    are hidden during Claude TTFT + TTS first-byte synthesis."""
    events: list[str] = []

    def enter():
        events.append("enter")
        return "PREV"

    def exit_(prev):
        events.append("exit")

    audio_mock.reset_speaker_buffer = MagicMock(
        side_effect=lambda: events.append("reset")
    )
    audio_mock.write_speaker = AsyncMock(
        side_effect=lambda chunk: events.append(f"write:{chunk!r}")
    )

    sp = Speaker(tts_mock, audio_mock, profile, enter, exit_)
    await sp.speak_stream(_aiter(["Hi."]))

    # reset must come before enter (enter is deferred until first
    # chunk), and enter must happen immediately before the first
    # write_speaker call.
    assert events[0] == "reset"
    assert events[1] == "enter"
    assert events[2].startswith("write:")
    assert events[-1] == "exit"


async def test_no_exit_when_tts_fails_before_first_chunk(
    profile, audio_mock, fast_sleep
):
    """If TTS raises before producing any chunk, no audio plays, so
    enter/exit must not run. State stays at PROCESSING and the
    orchestrator handles cleanup."""
    tts = MagicMock()

    def stream(text, voice):
        async def _gen():
            raise RuntimeError("tts down")
            yield  # pragma: no cover - make this an async generator

        return _gen()

    tts.stream.side_effect = stream

    enter, exit_, calls = _state_hooks()
    sp = Speaker(tts, audio_mock, profile, enter, exit_)

    await sp.speak_stream(_aiter(["Hi."]))

    assert calls == []
    audio_mock.write_speaker.assert_not_awaited()
