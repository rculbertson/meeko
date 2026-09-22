"""PyAudio lifecycle + mic queue.

Owns the input/output streams and the bounded queue that couples the
PyAudio callback thread to the asyncio event loop.
"""

import array
import asyncio
import concurrent.futures
import logging

import pyaudio

logger = logging.getLogger("meeko")

RATE = 16000
FORMAT = pyaudio.paInt16
CHUNK = 800  # 50ms at 16kHz (800 samples * 2 bytes = 1600 bytes per chunk)

# PortAudio does not silently rate/channel-convert for us the way
# CoreAudio's system mixer does for apps like Spotify, so we open both
# streams at the device's native channel count and do the mono <->
# stereo conversion in Python: take the left channel on input,
# duplicate mono TTS to both channels on output. AudioIO is parameterized
# on input/output channel count + device index; defaults assume the
# ReSpeaker XVF3800 (2-ch in, 2-ch out, OS default device). Run
# `python -m meeko.audio_io` to list device indices.


def _left_channel(data: bytes, channels: int) -> bytes:
    """Extract the left channel of interleaved int16 PCM."""
    if channels == 1:
        return data
    samples = array.array("h")
    samples.frombytes(data)
    return samples[::channels].tobytes()


def _mono_to_stereo(data: bytes) -> bytes:
    """Duplicate mono int16 PCM into interleaved stereo."""
    samples = array.array("h")
    samples.frombytes(data)
    stereo = array.array("h", [0] * (len(samples) * 2))
    stereo[0::2] = samples
    stereo[1::2] = samples
    return stereo.tobytes()


# Hard ceiling on buffered mic chunks. At 50ms chunks this is ~8min of
# audio — we should never come close. Hitting it means something is
# very wrong (e.g. MicPump stuck); log and exit.
MIC_QUEUE_MAX = 10000


class AudioIO:
    def __init__(
        self,
        stop_event: asyncio.Event,
        *,
        input_channels: int = 2,
        output_channels: int = 2,
        input_device_index: int | None = None,
        output_device_index: int | None = None,
    ):
        self._stop_event = stop_event
        self._input_channels = input_channels
        self._output_channels = output_channels
        self._pa = pyaudio.PyAudio()
        self.mic_queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=MIC_QUEUE_MAX)
        self._loop = asyncio.get_running_loop()
        self._mic_capturing = False

        self._mic_stream = self._pa.open(
            format=FORMAT,
            channels=input_channels,
            rate=RATE,
            input=True,
            input_device_index=input_device_index,
            frames_per_buffer=CHUNK,
            stream_callback=self._mic_callback,
        )
        self._speaker_stream = self._pa.open(
            format=FORMAT,
            channels=output_channels,
            rate=RATE,
            output=True,
            output_device_index=output_device_index,
            frames_per_buffer=CHUNK,
        )
        # TTS chunks arrive at arbitrary byte boundaries; an int16 sample
        # can be split across two chunks. Buffer any odd trailing byte
        # and prepend it to the next chunk so frombytes always sees an
        # even-length buffer. Reset on barge-in (abort_speaker) so a
        # leftover byte doesn't bleed into the next utterance.
        self._spk_leftover = b""
        # Set by abort_speaker() to drop any further writes from the
        # cancelled utterance. Cleared at the start of the next utterance
        # via reset_speaker_buffer(). We can't safely flush PortAudio's
        # output buffer mid-playback (stop_stream from the asyncio thread
        # races a blocking write_stream still running in the to_thread
        # executor — that race left the stream in a closed/errored state
        # on ALSA), so we accept the ~one-buffer tail of audio that
        # PortAudio has already queued.
        self._spk_muted = False
        # Dedicated single-worker executor for speaker writes. PortAudio's
        # blocking write API is not thread-safe on the same stream
        # (single-threaded spec; concurrent calls are documented as
        # illegal). The default to_thread pool has multiple workers, so
        # if a write is still running in the executor when the asyncio
        # task is cancelled (barge-in) and the next utterance submits
        # another write, the two could land on different workers and
        # call pa.write_stream concurrently. A single-worker executor
        # serializes writes by FIFO regardless of asyncio cancellation.
        self._spk_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="meeko-spk"
        )

    def _mic_callback(self, in_data, frame_count, time_info, status):
        mono = _left_channel(in_data, self._input_channels)

        def enqueue() -> None:
            try:
                self.mic_queue.put_nowait(mono)
            except asyncio.QueueFull:
                logger.error("mic_queue reached max size (%d); aborting", MIC_QUEUE_MAX)
                self._stop_event.set()

        self._loop.call_soon_threadsafe(enqueue)
        return (None, pyaudio.paContinue)

    @property
    def mic_capturing(self) -> bool:
        return self._mic_capturing

    def start_mic(self) -> None:
        self._mic_stream.start_stream()
        self._mic_capturing = True

    def stop_mic(self) -> None:
        self._mic_stream.stop_stream()
        self._mic_capturing = False

    def drain_mic_queue(self) -> None:
        while not self.mic_queue.empty():
            self.mic_queue.get_nowait()

    async def write_speaker(self, chunk: bytes) -> None:
        # PyAudio write is blocking; run in a thread so the mic silence
        # pump + STT loop run.
        if self._spk_muted:
            return
        if self._spk_leftover:
            chunk = self._spk_leftover + chunk
            self._spk_leftover = b""
        if len(chunk) % 2:
            self._spk_leftover = chunk[-1:]
            chunk = chunk[:-1]
        if not chunk:
            return
        # Slice into CHUNK-sized pieces (50ms each) and re-check the
        # mute flag between slices so barge-in cancellation latency is
        # bounded by one slice rather than by the (unbounded) size of
        # the TTS chunk handed in. abort_speaker() only blocks future
        # submissions — without slicing, a multi-hundred-ms TTS chunk
        # already in flight would play to completion.
        mono_slice_bytes = CHUNK * 2
        for offset in range(0, len(chunk), mono_slice_bytes):
            if self._spk_muted:
                return
            piece = chunk[offset : offset + mono_slice_bytes]
            stereo = _mono_to_stereo(piece) if self._output_channels == 2 else piece
            await self._loop.run_in_executor(
                self._spk_executor, self._speaker_stream.write, stereo
            )

    def reset_speaker_buffer(self) -> None:
        """Reset speaker write state at the start of a new utterance.

        Drops the odd-byte carry from a prior utterance so a producer
        that errored mid-sample (e.g. TTS websocket drop) can't bleed a
        stale byte into the next utterance and byte-swap every int16
        sample that follows. Also clears the barge-in mute so writes
        flow again."""
        self._spk_leftover = b""
        self._spk_muted = False

    def abort_speaker(self) -> None:
        """Drop any further writes from the cancelled utterance.

        On barge-in we cancel the speak task, but a blocking write_stream
        may still be in flight in the to_thread executor and any chunks
        already queued in speak_stream's consume() will follow. Setting
        _spk_muted causes write_speaker() to no-op for the remainder of
        the utterance; reset at the start of the next one. We don't
        touch the PortAudio stream itself — stop_stream/start_stream
        races the in-flight write and corrupts the stream. The user
        will still hear whatever PortAudio has already buffered (~one
        CHUNK plus driver buffer, ~100ms on the Pi)."""
        self._spk_muted = True

    def close(self) -> None:
        if self._mic_capturing:
            self._mic_stream.stop_stream()
            self._mic_capturing = False
        self._mic_stream.close()
        # Drain any pending speaker writes before closing the stream so
        # the worker doesn't try to write into a closed handle.
        self._spk_executor.shutdown(wait=True)
        self._speaker_stream.stop_stream()
        self._speaker_stream.close()
        self._pa.terminate()


def _list_devices() -> None:
    pa = pyaudio.PyAudio()
    try:
        try:
            default_in = pa.get_default_input_device_info().get("index")
        except OSError:
            default_in = None
        try:
            default_out = pa.get_default_output_device_info().get("index")
        except OSError:
            default_out = None
        for i in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(i)
            tags = []
            in_ch = int(info.get("maxInputChannels", 0))
            out_ch = int(info.get("maxOutputChannels", 0))
            if in_ch > 0:
                tags.append(f"in={in_ch}ch")
            if out_ch > 0:
                tags.append(f"out={out_ch}ch")
            if i == default_in:
                tags.append("DEFAULT-IN")
            if i == default_out:
                tags.append("DEFAULT-OUT")
            print(f"[{i}] {info['name']}  ({', '.join(tags)})")
    finally:
        pa.terminate()


if __name__ == "__main__":
    _list_devices()
