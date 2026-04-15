"""One-time script to generate WAV test fixtures via Deepgram TTS."""

import os
import wave
from pathlib import Path

import requests
from dotenv import load_dotenv

FIXTURES_DIR = Path(__file__).parent / "fixtures"

FIXTURES = [
    {
        "filename": "hello_meeko.wav",
        "text": "Hello Meeko, what is the capital of France?",
    },
]


def generate():
    load_dotenv()
    key = os.environ["DEEPGRAM_API_KEY"]

    FIXTURES_DIR.mkdir(exist_ok=True)

    for fixture in FIXTURES:
        path = FIXTURES_DIR / fixture["filename"]
        print(f"Generating {path} ...")

        resp = requests.post(
            "https://api.deepgram.com/v1/speak",
            params={
                "model": "aura-2-asteria-en",
                "encoding": "linear16",
                "sample_rate": "16000",
            },
            headers={
                "Authorization": f"Token {key}",
                "Content-Type": "application/json",
            },
            json={"text": fixture["text"]},
        )
        resp.raise_for_status()

        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(resp.content)

        print(f"  Wrote {path.stat().st_size} bytes")

    print("Done.")


if __name__ == "__main__":
    generate()
