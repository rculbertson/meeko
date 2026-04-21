"""Unit tests for `meeko.speaker.Speaker`."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from meeko.profiles import Profile
from meeko.speaker import INTER_SENTENCE_SILENCE, Speaker


@pytest.fixture
def profile():
    return Profile(
        name="default",
        wake_word="meeko",
        prompt="sys",
        greeting="",
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


async def test_speak_stream_drains_mic_after_tail(
    profile, audio_mock, tts_mock, fast_sleep
):
    enter, exit_, _ = _state_hooks()
    sp = Speaker(tts_mock, audio_mock, profile, enter, exit_)

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
        greeting="",
        voice="aura-2-andromeda-en",
    )
    enter, exit_, _ = _state_hooks()
    sp = Speaker(tts_mock, audio_mock, profile, enter, exit_)

    await sp.speak_stream(_aiter(["Hi."]))

    _, kwargs = tts_mock.stream.call_args
    assert kwargs["voice"] == "aura-2-andromeda-en"
