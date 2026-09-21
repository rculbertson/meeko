"""Unit tests for the streaming Deepgram TTS wrapper."""

import os
from unittest.mock import MagicMock, patch

import pytest

from meeko.deepgram_tts import DeepgramTTS


class _FakeGenerate:
    """Mimics ``client.speak.v1.audio.generate(...)`` — a callable
    returning an async iterator of PCM chunks."""

    def __init__(self, chunks: list[bytes]):
        self.chunks = chunks
        self.call_kwargs: dict | None = None

    def __call__(self, **kwargs):
        self.call_kwargs = kwargs

        async def _iter():
            for c in self.chunks:
                yield c

        return _iter()


def _build_tts(chunks: list[bytes]) -> tuple[DeepgramTTS, _FakeGenerate]:
    fake = _FakeGenerate(chunks)
    mock_client = MagicMock()
    mock_client.speak.v1.audio.generate = fake
    with patch("meeko.deepgram_tts.AsyncDeepgramClient", return_value=mock_client):
        tts = DeepgramTTS(api_key="x")
    return tts, fake


async def test_stream_yields_chunks_in_order():
    chunks = [b"\x01\x02", b"\x03\x04", b"\x05\x06"]
    tts, _ = _build_tts(chunks)

    received = [c async for c in tts.stream("hello", voice="asteria")]

    assert received == chunks


async def test_stream_passes_voice_and_audio_params():
    tts, fake = _build_tts([b"\x00\x00"])

    async for _ in tts.stream("hi", voice="zeus"):
        pass

    assert fake.call_kwargs["text"] == "hi"
    assert fake.call_kwargs["model"] == "aura-2-zeus-en"
    assert fake.call_kwargs["encoding"] == "linear16"
    assert fake.call_kwargs["container"] == "none"
    assert fake.call_kwargs["sample_rate"] == 16000


async def test_stream_handles_empty_response():
    tts, _ = _build_tts([])

    received = [c async for c in tts.stream("silent")]

    assert received == []


@pytest.mark.integration
@pytest.mark.asyncio
async def test_live_tts_stream_yields_audio():
    api_key = os.environ.get("DEEPGRAM_API_KEY")
    if not api_key:
        pytest.skip("DEEPGRAM_API_KEY not set")
    tts = DeepgramTTS(api_key=api_key)
    chunks = [
        c async for c in tts.stream("Testing Meeko text to speech.", voice="asteria")
    ]
    assert len(chunks) > 0
    total_bytes = sum(len(c) for c in chunks)
    assert total_bytes > 0
    assert total_bytes % 2 == 0
