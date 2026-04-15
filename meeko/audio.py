"""Mic capture and speaker playback using PyAudio."""

import pyaudio

RATE = 16000
CHANNELS = 1
FORMAT = pyaudio.paInt16
CHUNK = 1024
RECORD_SECONDS = 3


def list_devices():
    """Print available audio devices."""
    p = pyaudio.PyAudio()
    print("Audio devices:")
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        direction = []
        if info["maxInputChannels"] > 0:
            direction.append("in")
        if info["maxOutputChannels"] > 0:
            direction.append("out")
        print(f"  [{i}] {info['name']} ({', '.join(direction)})")
    p.terminate()


def record_and_playback():
    """Record from mic for a few seconds, then play it back."""
    p = pyaudio.PyAudio()

    # Record
    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK,
    )
    print(f"Recording for {RECORD_SECONDS} seconds... speak now!")
    frames = []
    for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)
    stream.stop_stream()
    stream.close()
    print("Recording done.")

    # Playback
    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        output=True,
        frames_per_buffer=CHUNK,
    )
    print("Playing back...")
    for frame in frames:
        stream.write(frame)
    stream.stop_stream()
    stream.close()
    print("Playback done.")

    p.terminate()


if __name__ == "__main__":
    list_devices()
    print()
    record_and_playback()
