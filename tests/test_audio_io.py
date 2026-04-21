"""Unit tests for `meeko.audio_io.AudioIO`."""

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from meeko import audio_io as audio_io_mod
from meeko.audio_io import AudioIO


@pytest.fixture
def pa_factory():
    """Patches pyaudio.PyAudio and returns (pa_instance, mic_stream,
    speaker_stream) so tests can inspect stream method calls."""
    pa_instance = MagicMock()
    mic_stream = MagicMock()
    speaker_stream = MagicMock()
    pa_instance.open.side_effect = [mic_stream, speaker_stream]
    with patch("meeko.audio_io.pyaudio.PyAudio", return_value=pa_instance):
        yield pa_instance, mic_stream, speaker_stream


async def test_starts_not_capturing(pa_factory):
    _, mic_stream, _ = pa_factory
    io = AudioIO(asyncio.Event())
    assert io.mic_capturing is False
    mic_stream.start_stream.assert_not_called()


async def test_start_and_stop_mic_toggle(pa_factory):
    _, mic_stream, _ = pa_factory
    io = AudioIO(asyncio.Event())

    io.start_mic()
    assert io.mic_capturing is True
    mic_stream.start_stream.assert_called_once()

    io.stop_mic()
    assert io.mic_capturing is False
    mic_stream.stop_stream.assert_called_once()


async def test_drain_mic_queue_empties_all_items(pa_factory):
    io = AudioIO(asyncio.Event())
    for i in range(5):
        io.mic_queue.put_nowait(bytes([i]))
    assert io.mic_queue.qsize() == 5

    io.drain_mic_queue()
    assert io.mic_queue.empty()


async def test_mic_callback_overflow_sets_stop_event(pa_factory):
    stop_event = asyncio.Event()
    with patch("meeko.audio_io.MIC_QUEUE_MAX", 2):
        io = AudioIO(stop_event)
        # Simulate the PyAudio thread calling the callback more times
        # than the queue can hold. call_soon_threadsafe schedules onto
        # the loop so we need to yield to let those enqueues run.
        for _ in range(5):
            io._mic_callback(b"\x00\x00", 1, None, 0)
        for _ in range(10):
            if stop_event.is_set():
                break
            await asyncio.sleep(0)

    assert stop_event.is_set()


async def test_write_speaker_delegates_to_stream(pa_factory):
    _, _, speaker_stream = pa_factory
    io = AudioIO(asyncio.Event())
    await io.write_speaker(b"\x01\x02\x03")
    speaker_stream.write.assert_called_once_with(b"\x01\x02\x03")


async def test_close_tears_down_both_streams(pa_factory):
    pa_instance, mic_stream, speaker_stream = pa_factory
    io = AudioIO(asyncio.Event())
    io.start_mic()

    io.close()

    mic_stream.stop_stream.assert_called()
    mic_stream.close.assert_called_once()
    speaker_stream.stop_stream.assert_called_once()
    speaker_stream.close.assert_called_once()
    pa_instance.terminate.assert_called_once()
    assert io.mic_capturing is False


async def test_close_without_active_capture_does_not_double_stop(pa_factory):
    _, mic_stream, _ = pa_factory
    io = AudioIO(asyncio.Event())
    # Never called start_mic — stop_mic on the mic stream must not fire.
    io.close()
    mic_stream.stop_stream.assert_not_called()
    mic_stream.close.assert_called_once()


def test_audio_constants_are_sane():
    # Sanity check the 50ms/16kHz chunk invariants that the rest of the
    # pipeline (tail-drain, inter-sentence silence) relies on.
    assert audio_io_mod.RATE == 16000
    assert audio_io_mod.CHUNK == 800
