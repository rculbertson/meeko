"""Unit tests for `meeko.audio_io.AudioIO`."""

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from meeko import audio_io as audio_io_mod
from meeko.audio_io import AudioIO, _device_index, _left_channel, _mono_to_stereo


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
        # 4 bytes = one stereo int16 frame (L=0, R=0).
        for _ in range(5):
            io._mic_callback(b"\x00\x00\x00\x00", 1, None, 0)
        for _ in range(10):
            if stop_event.is_set():
                break
            await asyncio.sleep(0)

    assert stop_event.is_set()


async def test_mic_callback_extracts_left_channel(pa_factory):
    """Stereo input (L=0x0101, R=0x7f7f per frame) should be
    deinterleaved to mono containing only the left samples."""
    io = AudioIO(asyncio.Event())
    # Two stereo frames: [L=0x0101, R=0x7f7f, L=0x0202, R=0x7f7f]
    stereo = b"\x01\x01\x7f\x7f\x02\x02\x7f\x7f"
    io._mic_callback(stereo, 2, None, 0)
    for _ in range(10):
        if not io.mic_queue.empty():
            break
        await asyncio.sleep(0)
    mono = io.mic_queue.get_nowait()
    assert mono == b"\x01\x01\x02\x02"


async def test_write_speaker_duplicates_mono_to_stereo(pa_factory):
    _, _, speaker_stream = pa_factory
    io = AudioIO(asyncio.Event())
    # One mono int16 sample 0x0201 — expect it duplicated across L/R.
    await io.write_speaker(b"\x01\x02")
    speaker_stream.write.assert_called_once_with(b"\x01\x02\x01\x02")


def test_left_channel_helper_extracts_every_nth_sample():
    # int16 LE: samples 1, 2, 3, 4 interleaved as stereo frames.
    stereo = b"\x01\x00\x02\x00\x03\x00\x04\x00"
    assert _left_channel(stereo, 2) == b"\x01\x00\x03\x00"
    # Mono passthrough.
    assert _left_channel(b"\x01\x00\x02\x00", 1) == b"\x01\x00\x02\x00"


def test_mono_to_stereo_helper_duplicates_samples():
    mono = b"\x01\x00\x02\x00"
    assert _mono_to_stereo(mono) == b"\x01\x00\x01\x00\x02\x00\x02\x00"


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


def test_device_index_unset_returns_none(monkeypatch):
    monkeypatch.delenv("MEEKO_TEST_INDEX", raising=False)
    assert _device_index("MEEKO_TEST_INDEX") is None


def test_device_index_empty_returns_none(monkeypatch):
    monkeypatch.setenv("MEEKO_TEST_INDEX", "")
    assert _device_index("MEEKO_TEST_INDEX") is None


def test_device_index_whitespace_returns_none(monkeypatch):
    monkeypatch.setenv("MEEKO_TEST_INDEX", "   ")
    assert _device_index("MEEKO_TEST_INDEX") is None


def test_device_index_parses_int(monkeypatch):
    monkeypatch.setenv("MEEKO_TEST_INDEX", "3")
    assert _device_index("MEEKO_TEST_INDEX") == 3


def test_audio_constants_are_sane():
    # Sanity check the 50ms/16kHz chunk invariants that the rest of the
    # pipeline (tail-drain, inter-sentence silence) relies on.
    assert audio_io_mod.RATE == 16000
    assert audio_io_mod.CHUNK == 800
