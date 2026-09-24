# Meeko

[![CI](https://github.com/rculbertson/meeko/actions/workflows/ci.yml/badge.svg)](https://github.com/rculbertson/meeko/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.14+](https://img.shields.io/badge/python-3.14+-blue.svg)](https://www.python.org/downloads/)

Meeko is a screenless, tabletop voice assistant powered by Claude. Built as a dedicated appliance for the **Raspberry Pi**, Meeko acts as an always-ready thinking partner that remembers the full context of every conversation. Brainstorm an idea, step away for days, and pick up right where you left off—just say, *"let's go back to the conversation about the app I'm building,"* and Meeko seamlessly resumes.

*(You can also test and develop Meeko on a Mac before assembling dedicated hardware — see [Testing on macOS](#testing-on-macos).)*

## Features

- **Natural voice session management** — start, end, switch, and resume conversations without rigid commands or syntax; Claude infers session intents naturally from context.
- **Hardware-level barge-in** — onboard acoustic echo cancellation (AEC) lets you speak over and interrupt Meeko naturally while it is talking.
- **Local-first privacy** — full conversation transcripts and metadata are stored in a local SQLite database on your hardware, never in a vendor cloud.
- **Configurable personality profiles** — switch between quick query-and-answer responses or deep, open-ended conversational modes, or define your own.
- **Automatic context summarization** — long-running sessions are summarized dynamically in-flight so conversations never overflow Claude's context window.
- **Cost-efficient prompt caching** — Anthropic prompt caching keeps input token costs negligible even across extended, multi-turn discussions.
- **Direct provider billing (BYOK)** — connect directly to Deepgram and Anthropic with your own API keys, with zero middleman markup or subscription fees.

## Hardware & System Requirements

Meeko is engineered as a dedicated hardware appliance:

- **Target Hardware**:
  - **Raspberry Pi 5** (8 GB recommended, with active cooler and NVMe SSD HAT)
  - **[ReSpeaker XVF3800](https://www.seeedstudio.com/ReSpeaker-XVF3800-USB-4-Mic-Array-With-Case-p-6490.html)** USB 4-Mic Array with onboard hardware AEC and WS2812 LED ring
  - **Powered speaker** plugged into the XVF3800's 3.5mm jack
- **Software Dependencies**:
  - Python 3.14+
  - [uv](https://docs.astral.sh/uv/) package manager
  - PortAudio (`sudo apt install portaudio19-dev` on Debian/Ubuntu/Raspberry Pi OS, `brew install portaudio` on macOS)
- **API Keys & Operating Costs**:
  - [Deepgram](https://deepgram.com/) (STT & TTS; free tier includes ~$200 credit, covering hundreds of hours of voice)
  - [Anthropic](https://www.anthropic.com/) (Claude Sonnet 5; prompt caching discounts cached input tokens by 90%, keeping daily conversational costs to pennies)

---

## Getting Started

1. **Clone and install**:

   ```bash
   git clone https://github.com/rculbertson/meeko.git
   cd meeko
   uv sync
   ```

2. **Add API keys**: Create a `.env` file in the project root:

   ```bash
   DEEPGRAM_API_KEY=your-deepgram-api-key
   ANTHROPIC_API_KEY=your-anthropic-api-key
   ```

---

## Raspberry Pi Appliance Setup

Meeko's built-in defaults are preconfigured for the Raspberry Pi + ReSpeaker XVF3800 setup (2-channel audio, hardware AEC, and WS2812 LED control).

Follow the **[Raspberry Pi Setup Guide](docs/raspberry-pi-setup.md)** to complete the one-time hardware configuration:
- Tune the ReSpeaker's AEC sensitivity registers (`xvf_host PP_DTSENSITIVE 12`)
- Install the udev rule for LED ring permissions (`scripts/99-meeko-xvf3800.rules`)
- Configure the systemd user service to start Meeko automatically on boot

---

## Testing on macOS

You do not need a Raspberry Pi to try Meeko out. You can run Meeko on a Mac using its built-in microphone and speakers for testing and development:

1. Run Meeko once to generate the default config at `~/.config/meeko/meeko.toml`:

   ```bash
   uv run python -m meeko.main
   ```

2. Add an `[audio]` section to `~/.config/meeko/meeko.toml` to match your single-channel built-in mic and enable software echo suppression:

   ```toml
   [audio]
   input_channels = 1
   mute_mic_while_speaking = true
   ```

*(Note: `mute_mic_while_speaking = true` prevents Meeko from hearing itself talk on hardware lacking acoustic echo cancellation; true barge-in is disabled in this mode).*

---

## Running

Start Meeko:

```bash
uv run python -m meeko.main
```

Say **"Hey Meeko"** to wake it. Press `Ctrl+C` to quit.

On a Raspberry Pi with the ReSpeaker XVF3800, the onboard LED ring indicates Meeko's live state (listening, thinking, speaking, errors). See [What the LED Ring Shows](docs/raspberry-pi-setup.md#3-what-the-led-ring-shows).

### What You Can Say

Meeko understands natural conversational phrasing with no rigid syntax, routing intents dynamically to built-in tools:

- **Sessions & Memory**:
  - *"Let's go back to our brainstorm about the mobile app from yesterday."*
  - *"What were we discussing earlier this morning?"*
  - *"Wrap up this session."*
- **Built-in Utilities**:
  - *"What's the weather forecast for tomorrow afternoon?"* (Open-Meteo)
  - *"Set a 15-minute timer for baking."*
  - *"Search the web for the latest Python 3.14 release notes."* (Claude web search)
- **Personality Profiles**:
  - *"Switch to conversation profile."* (For open-ended brainstorming)
  - *"Switch to query profile."* (For concise, 1-2 sentence answers)

### Managing Sessions from the Terminal

You can also inspect or resume prior sessions directly from the CLI:

```bash
uv run python -m meeko.main --list-sessions        # print prior sessions and exit
uv run python -m meeko.main --resume               # resume the most recent session
uv run python -m meeko.main --resume SESSION_ID    # resume a specific session
```

---

## Configuration

All non-secret settings live in `meeko.toml` (searched in `$MEEKO_CONFIG`, `~/.config/meeko/meeko.toml`, then `./meeko.toml`). Every setting can also be overridden at runtime with a corresponding `MEEKO_*` environment variable.

Configuration tables include:
- `[system]` — logging levels, rotating log files, SQLite database path, LED toggle.
- `[audio]` — mic/speaker device selection, channel counts, software mute toggle.
- `[wake_word]` — ONNX model path, confidence threshold, bypass switch.
- `[claude]` — context compaction token trigger, web search toggle and limits.
- `[location]` & `[weather]` — home coordinates and imperial/metric units for forecasts via Open-Meteo.
- `[profiles.<name>]` — custom personas, TTS voice IDs, and pause/idle timeout behaviors.

See [docs/configuration.md](docs/configuration.md) for the complete list of options, default values, and a guide to authoring custom profiles.

---

## How It Works

Each conversational turn flows through a direct, low-latency pipeline:

> **Mic Audio** → **Deepgram STT** (Flux, with semantic end-of-turn) → **Claude** (Anthropic SDK) → **Deepgram TTS** (Aura-2) → **Speaker**

- **On-Device Wake Word:** Gated locally by [openWakeWord](https://github.com/dscripka/openWakeWord) running on CPU—audio is only streamed externally after "Hey Meeko" is detected.
- **Tool-Based Session Management:** Intents like starting, ending, listing, or resuming sessions are exposed as Claude tools. Claude decides when to invoke them from conversational context rather than relying on a rigid classifier.
- **Immediate Local Persistence:** Turns are committed to SQLite instantly, ensuring the local transcript remains the authoritative source of truth.

See [docs/architecture.md](docs/architecture.md) for the full design — state machine, session persistence, prompt caching, server-side compaction, and the rationale behind key design decisions.

---

## Troubleshooting

- **Meeko doesn't wake up:** Lower `[wake_word] threshold` (default `0.96`) in `meeko.toml` to increase sensitivity. If it wakes spontaneously from room noise, raise it. Run `uv run python -m meeko.wake_word` to verify preprocessor models are properly cached.
- **Meeko talks over you or ignores interruptions:** Barge-in requires acoustic isolation. On the Pi with an XVF3800, ensure the speaker is plugged into the XVF3800's 3.5mm jack (not the Pi's audio out) and AEC is tuned. On macOS or hardware without AEC, ensure `mute_mic_while_speaking = true` is set in `[audio]`.
- **Wrong mic/speaker or silent audio:** Run `uv run python -m meeko.audio_io` to inspect device indices:

  ```bash
  uv run python -m meeko.audio_io
  ```

  Set `input_device_index` and `output_device_index` in `[audio]` to pin devices, and make sure `input_channels` matches your hardware (`2` for ReSpeaker, `1` for built-in Mac mics). A channel mismatch is the most common cause of silence or distorted audio.
- **LED ring doesn't light up:** LED control is Linux/XVF3800-specific and requires the udev rule; see [docs/raspberry-pi-setup.md](docs/raspberry-pi-setup.md). The ring remains off when no XVF3800 is detected (expected on macOS).

---

## Privacy

Meeko stores all conversation transcripts and summaries locally in SQLite on your device — nothing is uploaded to a third-party assistant cloud. Three external services see data:

- **Deepgram** receives microphone audio for speech-to-text, and the text of Claude's responses for text-to-speech.
- **Anthropic** receives the conversation history sent to Claude.
- **Open-Meteo** receives coordinates when you ask about the weather — your `[location]` home coordinates, or coordinates Claude provides for a named place. No account or API key is involved, and no conversation text is sent.

Deepgram and Anthropic are accessed using your own API keys under their respective terms of service.

### Logs & Content Privacy

Conversation content — user speech, Claude's responses, tool arguments, timer labels, and session titles — is logged at `DEBUG` level only. The default `log_level` is `INFO`, which records only a content-free operational trace (state transitions, tool names, and errors). When Meeko runs under systemd, private conversation content never enters the system journal.

To enable content logging for debugging, set `log_level = "DEBUG"` (or `MEEKO_LOG_LEVEL=DEBUG`). Set `log_target = "file"` to keep debug logs in `meeko.log` rather than system journals.

---

## Project Status

Meeko is a personal project, maintained in spare time. It works and I use it daily, but I review issues and pull requests in batches — expect weeks, not days. Contributions are welcome; please read [CONTRIBUTING.md](CONTRIBUTING.md) first, and open an issue before starting large changes.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for dev setup, tests, complexity gates, and coding conventions.

## License

MIT — see [LICENSE](LICENSE). Note that the bundled wake-word model (`models/hey_meeko.onnx`) and openWakeWord's pre-trained preprocessor models downloaded at runtime are CC-BY-NC-SA-4.0; see [docs/wake-word.md](docs/wake-word.md) before using Meeko commercially.

## Inspiration

After a concussion left me unable to look at screens for a week, I couldn't use my laptop, my phone, or even watch TV. What saved my sanity was talking to Claude's voice mode — hours of conversation about whatever was on my mind, no screen required.

That experience inspired Meeko: a physical, screenless appliance purpose-built for deep, open-ended brainstorming, with no display, no browser, and zero screen dependency.