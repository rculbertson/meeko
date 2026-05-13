# Meeko

Meeko is a personal voice assistant backed by Claude. It handles the usual quick questions, as well as extended, open-ended thinking sessions where context accumulates over time. Meeko remembers the full text of every conversation, so you can pause mid-thought, come back days later, and pick up exactly where you left off. Just say "let's go back to the conversation about the app I'm building" and Meeko finds it and resumes.


## Features

- **Named, resumable sessions** — every conversation is saved and searchable by voice.
- **Natural session management** — start, end, and resume sessions by voice with no rigid command syntax; Claude decides when to call session tools from full conversation context
- **Transcripts stay on your device** — every conversation is saved to SQLite on your own Raspberry Pi or Mac.
- **Bring your own API keys** — talks to Deepgram and Anthropic directly, billed at provider rates with no assistant-vendor markup or middleman account.
- **Stays cheap on long conversations** — prompt caching keeps token costs low even as sessions grow to 50k–150k tokens
- **Never fills the context window** — long sessions are summarized automatically in-flight, without losing the full transcript on disk

## Prerequisites

- Python 3.14+
- [uv](https://docs.astral.sh/uv/)
- PortAudio — `brew install portaudio` on macOS, `sudo apt install portaudio19-dev` on Debian/Ubuntu/Raspberry Pi OS
- A microphone and speakers — anything PyAudio can see. Meeko was built using a Raspberry Pi 5 with a [ReSpeaker XVF3800](https://www.seeedstudio.com/ReSpeaker-USB-Mic-Array-p-4247.html) USB mic array (see "Notes for the ReSpeaker XVF3800" below). But using any device's built-in mic and speakers works fine with `mute_mic_while_speaking = true`.
- A [Deepgram](https://deepgram.com/) API key (the free tier includes ~$200 of credit)
- An [Anthropic](https://www.anthropic.com/) API key

## Setup

```
uv sync
```

Create a `.env` file in the project root with your API keys:

```
DEEPGRAM_API_KEY=your-deepgram-api-key
ANTHROPIC_API_KEY=your-anthropic-api-key
```

## Running

```
uv run python -m meeko.main
```

Say "Hey Meeko" to wake it. Press `Ctrl+C` to quit.

The shipped `meeko.toml` provides two profiles — `query` (short replies, auto-closes after a short silence) and `conversation` (substantive thinking-partner persona that stays open through pauses). You can switch between them mid-session by asking — for example "switch to conversation mode", "let's have a long conversation", "switch back to query mode", or "just quick questions from now on". Ask "what modes are available?" to list profiles.

## How it works

A turn flows: mic audio → Deepgram STT (Flux, with semantic end-of-turn detection) → Claude (called directly via the Anthropic SDK, with prompt caching, tool use, web search, and server-side compaction) → Deepgram TTS (Aura-2) → speaker. Every turn is persisted to SQLite immediately, so the on-disk transcript is always the source of truth. Wake-word gating ("Hey Meeko") runs on-device with [openWakeWord](https://github.com/dscripka/openWakeWord). Session-management intents (end, new, list, load) are exposed to Claude as tools, so Sonnet decides when to call them based on the full conversation rather than a separate classifier.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full design — state machine, session persistence, prompt caching, server-side compaction, and the rationale behind the key design decisions.

## Configuration

All non-secret settings live in `meeko.toml` at the repo root. Each setting also has a `MEEKO_*` environment variable that overrides the TOML value at runtime — useful for one-off testing without editing the file.

Settings are grouped into four tables: `[system]`, `[audio]`, `[wake_word]`, and `[claude]`. Profiles (personas) are defined under `[profiles.<name>]`, and a top-level `default_profile = "<name>"` key selects which profile a fresh session starts in.

### `[system]`

| TOML key       | Env var              | Default              | Description                                                                 |
|----------------|----------------------|----------------------|-----------------------------------------------------------------------------|
| `log_level`    | `MEEKO_LOG_LEVEL`    | `DEBUG`              | One of `DEBUG`, `INFO`, `WARNING`, `ERROR`.                                 |
| `log_target`   | `MEEKO_LOG_TARGET`   | unset (stderr)       | Set to `file` to log to `meeko.log` (rotating, 5 MB × 3 files).             |
| `db_path`      | `MEEKO_DB_PATH`      | `~/.meeko/meeko.db`  | SQLite database location for sessions and transcripts.                      |
| `led_disabled` | `MEEKO_LED_DISABLED` | `false`              | Skip LED control. Auto-disabled when the XVF3800 isn't found.               |

### `[audio]`

| TOML key                  | Env var                          | Default | Description                                                              |
|---------------------------|----------------------------------|---------|--------------------------------------------------------------------------|
| `input_device_index`      | `MEEKO_INPUT_DEVICE_INDEX`       | unset   | PyAudio device index for the mic. Unset = OS default. Run `uv run python -m meeko.audio_io` to list devices. |
| `output_device_index`     | `MEEKO_OUTPUT_DEVICE_INDEX`      | unset   | PyAudio device index for the speaker.                                    |
| `input_channels`          | `MEEKO_INPUT_CHANNELS`           | `2`     | Native input channel count. Use `1` for most built-in laptop mics.       |
| `output_channels`         | `MEEKO_OUTPUT_CHANNELS`          | `2`     | Native output channel count.                                             |
| `mute_mic_while_speaking` | `MEEKO_MUTE_MIC_WHILE_SPEAKING`  | `false` | Software-mute the mic during TTS playback. Set `true` on hardware with no AEC (e.g. Mac built-in mic). |

### `[wake_word]`

| TOML key    | Env var                       | Default                  | Description                                                                |
|-------------|-------------------------------|--------------------------|----------------------------------------------------------------------------|
| `model`     | `MEEKO_WAKE_WORD_MODEL`       | `models/hey_meeko.onnx`  | Path to the openWakeWord ONNX model.                                       |
| `threshold` | `MEEKO_WAKE_WORD_THRESHOLD`   | `0.96`                   | Confidence threshold (0–1). Lower = more sensitive, more false wakes.      |
| `disabled`  | `MEEKO_WAKE_WORD_DISABLED`    | `false`                  | Skip the wake gate; start directly in `LISTENING` (useful for development).|

### `[claude]`

| TOML key                    | Env var                            | Default  | Description                                                                |
|-----------------------------|------------------------------------|----------|----------------------------------------------------------------------------|
| `compaction_trigger_tokens` | `MEEKO_COMPACTION_TRIGGER_TOKENS`  | `150000` | Input-token threshold that triggers server-side compaction.                |
| `web_search_enabled`        | `MEEKO_WEB_SEARCH_ENABLED`         | `true`   | Expose Anthropic's server-side web search tool to the model.               |
| `web_search_max_uses`       | `MEEKO_WEB_SEARCH_MAX_USES`        | `2`      | Max web-search calls per turn.                                             |

### Profiles

`[profiles.<name>]` blocks define personas. The profile name *is* the conversation mode, so it must be either `query` (auto-closes after a short silence) or `conversation` (stays open through pauses, asks before closing). A top-level `default_profile = "<name>"` key selects which profile a fresh session starts in and is required. Each profile has:

| Key                           | Description                                                                        |
|-------------------------------|------------------------------------------------------------------------------------|
| `wake_word`                   | Wake-phrase label (advisory; the ONNX model determines the actual phrase).         |
| `voice`                       | Aura-2 voice id, e.g. `mars`, `andromeda`.                                          |
| `prompt`                      | The system prompt that defines the persona.                                        |
| `idle_timeout_seconds`        | (query profile) silence window before silent close. Default `5.0`.                 |
| `conversation_idle_seconds`   | (conversation profile) silence before the verbal check-in. Default `60.0`.         |
| `conversation_close_seconds`  | (conversation profile) silence after the check-in before closing. Default `20.0`.  |

See the shipped `meeko.toml` for working examples of both profiles.

## Notes for the ReSpeaker XVF3800

If you're using the ReSpeaker XVF3800 USB mic array (4-mic, hardware AEC), two one-time setup steps are required.

### Tune the AEC sensitivity

On the Pi + XVF3800 path, the chip's default AEC tuning suppresses near-end speech aggressively during far-end playback, which prevents barge-in: your voice never reaches Deepgram while Meeko is talking. Raise the double-talk sensitivity once and persist it to flash.

Grab `xvf_host` from the [reSpeaker XVF3800 repo](https://github.com/respeaker/reSpeaker_XVF3800_USB_4MIC_ARRAY/tree/master/host_control) (`host_control/rpi_64bit/`) and run:

```bash
sudo ./xvf_host PP_DTSENSITIVE 12
sudo ./xvf_host SAVE_CONFIGURATION 1
```

`PP_DTSENSITIVE 12` enables the chip's extra near-end speech detector and biases the AEC toward double-talk performance; `SAVE_CONFIGURATION 1` writes the value to flash so it survives power cycles. This is the only parameter that needed changing. To revert, `sudo ./xvf_host CLEAR_CONFIGURATION 1` and reboot the device.

The speaker must be connected to the XVF3800's **3.5mm audio jack**, not the Pi's audio output — the chip uses the speaker signal as the AEC reference.

### Install the udev rule for LED control

Meeko talks to the XVF3800's WS2812 LED ring directly over libusb. Linux defaults the USB control interface to root-only; install the udev rule once so a non-root user (in `plugdev`) can drive it:

```bash
sudo cp scripts/99-meeko-xvf3800.rules /etc/udev/rules.d/
sudo udevadm control --reload
sudo udevadm trigger --action=add --subsystem-match=usb
```

No reboot or replug needed.

To skip LED control entirely, set `led_disabled = true` in `[system]`. The controller also auto-disables when no XVF3800 is found, so no special handling is needed on a Mac dev machine.

## Privacy

Meeko stores all conversation transcripts and summaries locally in SQLite on your device — nothing is uploaded to a third-party assistant cloud. Two external services do see data per turn:

- **Deepgram** receives microphone audio for speech-to-text, and the text of Claude's responses for text-to-speech.
- **Anthropic** receives the conversation history sent to Claude.

Both are accessed with your own API keys; their handling of your data is governed by their respective terms of service.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for dev setup, tests, and lint.

## Inspiration
After a concussion left me unable to look at screens for a week, I couldn't use my laptop, my phone, or even watch TV. What saved my sanity was talking to Claude's voice mode — hours of conversation about whatever was on my mind, no screen required.

That experience made me want to build something purpose-built for it: a voice assistant designed from the ground up for long, open-ended conversation, with no screen dependency at all. Meeko is that.