import os
import wave
from pathlib import Path

import pytest
import pytest_asyncio
from deepgram import AsyncDeepgramClient
from dotenv import load_dotenv

from meeko.profiles import Profile
from meeko.tools.timer import get_tool_definitions as timer_tools
from meeko.voice_agent import build_settings

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session", autouse=True)
def _load_env():
    load_dotenv()


@pytest_asyncio.fixture
async def dg_client():
    key = os.environ["DEEPGRAM_API_KEY"]
    return AsyncDeepgramClient(api_key=key)


@pytest.fixture
def agent_settings():
    key = os.environ["CLAUDE_API_KEY"]
    profile = Profile(
        name="default",
        wake_word="meeko",
        prompt=(
            "You are Meeko, an intelligent voice assistant. "
            "Your responses will be spoken aloud via text-to-speech, "
            "so never use emojis, markdown, or other formatting. "
            "Keep your responses minimal."
        ),
        greeting="Hello! I'm Meeko. How can I help?",
    )
    return build_settings(key, profile, timer_tools())


def read_wav_as_chunks(path: Path, chunk_bytes: int = 1600) -> list[bytes]:
    """Read a WAV file and return raw PCM data as a list of chunks."""
    with wave.open(str(path), "rb") as wf:
        assert wf.getsampwidth() == 2, "Expected 16-bit audio"
        assert wf.getnchannels() == 1, "Expected mono audio"
        assert wf.getframerate() == 16000, "Expected 16kHz sample rate"
        raw = wf.readframes(wf.getnframes())
    return [raw[i : i + chunk_bytes] for i in range(0, len(raw), chunk_bytes)]


@pytest.fixture
def wav_chunks_hello():
    path = FIXTURES_DIR / "hello_meeko.wav"
    assert path.exists(), (
        f"Fixture not found: {path}. Run: uv run python tests/generate_fixtures.py"
    )
    return read_wav_as_chunks(path)


@pytest.fixture
def wav_chunks_set_timer():
    path = FIXTURES_DIR / "set_timer.wav"
    assert path.exists(), (
        f"Fixture not found: {path}. Run: uv run python tests/generate_fixtures.py"
    )
    return read_wav_as_chunks(path)
