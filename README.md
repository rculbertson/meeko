# Meeko

Meeko is a personal voice assistant backed by Claude. It runs on Raspberry Pi and Mac. Meeko remembers the full text of every conversation, so you can pause mid-thought, come back days later, and pick up exactly where you left off. Just say "let's go back to the conversation about the app I'm building" and Meeko finds it and resumes.

## Contents

- [Features](#features)
- [Prerequisites](#prerequisites)
- [Setup](#setup)
- [Running](#running)
- [How it works](#how-it-works)
- [Configuration](#configuration)
- [Raspberry Pi](#raspberry-pi)
- [Troubleshooting](#troubleshooting)
- [Privacy](#privacy)

## Features

- **Named, resumable sessions** — every conversation is saved and searchable by voice.
- **Natural session management** — start, end, and resume sessions by voice with no rigid command syntax; Claude decides when to call session tools from full conversation context.
- **Transcripts stay on your device** — every conversation is saved to SQLite on your own Raspberry Pi or Mac.
- **Customizable behavior** — define new profiles to change Meeko's behavior. Comes with two default profiles: `query` for quick questions, and `conversation` for open-ended discussions.
- **Bring your own API keys** — talks to Deepgram and Anthropic directly, billed at provider rates with no assistant-vendor markup or middleman account.
- **Stays cheap on long conversations** — prompt caching keeps token costs low even as sessions grow.
- **Never fills the context window** — long sessions are summarized automatically in-flight, without losing the full transcript on disk.

## Prerequisites

- Python 3.14+
- [uv](https://docs.astral.sh/uv/)
- PortAudio — `brew install portaudio` on macOS, `sudo apt install portaudio19-dev` on Debian/Ubuntu/Raspberry Pi OS
- A microphone and speakers — anything PyAudio can see. Meeko was built using a Raspberry Pi 5 with a [ReSpeaker XVF3800](https://www.seeedstudio.com/ReSpeaker-USB-Mic-Array-p-4247.html) USB mic array (see [Raspberry Pi](#raspberry-pi) below). But using any device's built-in mic and speakers works fine with `mute_mic_while_speaking = true`.
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

On first run Meeko creates a config file at `~/.config/meeko/meeko.toml` (copied from the bundled `meeko/default_config.toml`) and starts with working defaults; edit that file to add personal settings like your home location.

The defaults assume a ReSpeaker XVF3800. **On a Mac's built-in mic and speakers**, add an `[audio]` table to that file — the bundled default ships none:

```toml
[audio]
input_channels = 1
mute_mic_while_speaking = true
```

`input_channels = 1` matches the built-in mic's single channel (the default `2` is the ReSpeaker's), and `mute_mic_while_speaking` stands in for the hardware echo cancellation a Mac doesn't have, so Meeko doesn't hear itself talk.

The first run that uses the wake word also downloads openWakeWord's preprocessor models (~6.7 MB) into its own cache. If you're deploying to a device with no internet, run `uv run python -m meeko.wake_word` on a connected machine first to pre-populate it — see [models/README.md](models/README.md).

## Running

```
uv run python -m meeko.main
```

Say "Hey Meeko" to wake it. Press `Ctrl+C` to quit.

Prior sessions can also be worked with from the terminal:

```
uv run python -m meeko.main --list-sessions        # print prior sessions and exit
uv run python -m meeko.main --resume               # resume the most recent session
uv run python -m meeko.main --resume SESSION_ID    # resume a specific session
```

## How it works

A turn flows: mic audio → Deepgram STT (Flux, with semantic end-of-turn detection) → Claude (called directly via the Anthropic SDK, with prompt caching, tool use, web search, and server-side compaction) → Deepgram TTS (Aura-2) → speaker. Every turn is persisted to SQLite immediately, so the on-disk transcript is always the source of truth. Wake-word gating ("Hey Meeko") runs on-device with [openWakeWord](https://github.com/dscripka/openWakeWord). Session-management intents (end, new, list, load) are exposed to Claude as tools, so Sonnet decides when to call them based on the full conversation rather than a separate classifier.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full design — state machine, session persistence, prompt caching, server-side compaction, and the rationale behind the key design decisions.

## Configuration

All non-secret settings live in `meeko.toml`. Meeko looks for it in this order: `$MEEKO_CONFIG`, then `$XDG_CONFIG_HOME/meeko/meeko.toml` (i.e. `~/.config/meeko/meeko.toml`), then `./meeko.toml` in the working directory. If none exist, it auto-creates `~/.config/meeko/meeko.toml` from the bundled `meeko/default_config.toml` on first run. Edit that file to customize. Each setting also has a `MEEKO_*` environment variable that overrides the TOML value at runtime — useful for one-off testing without editing the file. (API keys stay in `.env` in the project root.)

Settings are grouped into six tables: `[system]`, `[audio]`, `[wake_word]`, `[claude]`, `[location]`, and `[weather]`. Profiles are defined under `[profiles.<name>]`, and a top-level `default_profile = "<name>"` key selects which profile a fresh session starts in.

### `[system]`

| TOML key       | Env var              | Default              | Description                                                                 |
|----------------|----------------------|----------------------|-----------------------------------------------------------------------------|
| `log_level`    | `MEEKO_LOG_LEVEL`    | `INFO`               | One of `DEBUG`, `INFO`, `WARNING`, `ERROR`. `DEBUG` logs conversation content — see [Privacy](#privacy). |
| `log_target`   | `MEEKO_LOG_TARGET`   | unset (stderr)       | Set to `file` to log to `meeko.log` (rotating at 5 MB, keeping 3 older files), relative to the working directory. |
| `db_path`      | `MEEKO_DB_PATH`      | `~/.local/share/meeko/meeko.db` | SQLite database location for sessions and transcripts (honors `$XDG_DATA_HOME`). |
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
| `model`     | `MEEKO_WAKE_WORD_MODEL`       | `models/hey_meeko.onnx`  | Path to the openWakeWord ONNX model. The default is relative to the working directory; use an absolute path to run Meeko from elsewhere. See [models/README.md](models/README.md) to train your own phrase. |
| `threshold` | `MEEKO_WAKE_WORD_THRESHOLD`   | `0.96`                   | Confidence threshold (0–1). Lower = more sensitive, more false wakes.      |
| `disabled`  | `MEEKO_WAKE_WORD_DISABLED`    | `false`                  | Skip the wake gate; start directly in `LISTENING` (useful for development).|

### `[claude]`

| TOML key                    | Env var                            | Default  | Description                                                                |
|-----------------------------|------------------------------------|----------|----------------------------------------------------------------------------|
| `compaction_trigger_tokens` | `MEEKO_COMPACTION_TRIGGER_TOKENS`  | `150000` | Input-token threshold that triggers server-side compaction.                |
| `web_search_enabled`        | `MEEKO_WEB_SEARCH_ENABLED`         | `true`   | Expose Anthropic's server-side web search tool to the model.               |
| `web_search_max_uses`       | `MEEKO_WEB_SEARCH_MAX_USES`        | `4`      | Max web-search *attempts* per turn. A circuit breaker on searches that hang server-side, not a quality dial — raising it makes slow turns far slower (see `meeko/claude_client.py`). |

### `[location]`

| TOML key    | Env var            | Default | Description                                                                                                                                          |
|-------------|--------------------|---------|------------------------------------------------------------------------------------------------------------------------------------------------------|
| `latitude`  | `MEEKO_LATITUDE`   | unset   | Home latitude (decimal degrees). Used by the weather tool when you don't name a place, and injected into Claude's system prompt so it can answer location-aware questions (sunset/sunrise, regional references, climate, etc.). Must be set with `longitude` or not at all. |
| `longitude` | `MEEKO_LONGITUDE`  | unset   | Home longitude (decimal degrees). Look both up once via Google Maps (right-click → copy coordinates).                                                |

### `[weather]`

| TOML key | Env var               | Default    | Description                                            |
|----------|-----------------------|------------|--------------------------------------------------------|
| `units`  | `MEEKO_WEATHER_UNITS` | `imperial` | `imperial` (°F, mph, inch) or `metric` (°C, km/h, mm). |

Forecasts come from [Open-Meteo](https://open-meteo.com) (free, no API key). Claude can ask for either a single day — today by default, any day up to two weeks out — or an hour-by-hour forecast covering the next 48 hours, which it picks automatically for questions about part of a day ("what's it doing this afternoon?"). Name a place ("weather in Tokyo") and Claude supplies the coordinates itself; otherwise `[location]` is used.

### Profiles

Each profile is defined by a `[profiles.<name>]` block. The default config ships two profiles — `query` (auto-closes after a short silence) and `conversation` (stays open through pauses, asks before closing). You can modify these or add new ones. To switch between profiles, just ask Meeko, e.g. "switch to the conversation profile". You can also ask "what profiles are available?" to hear the list and which one is active.

A top-level `default_profile = "<name>"` key selects which one a fresh session starts in, and is required. Within a profile, `prompt` is the only required key; every other key below is optional and falls back to the default shown.

| Key                           | Description                                                                        |
|-------------------------------|------------------------------------------------------------------------------------|
| `prompt`                      | **Required.** The system prompt that defines the profile.                          |
| `voice`                       | Aura-2 voice id, e.g. `mars`, `andromeda`. Defaults to `asteria`.                   |
| `description`                 | One line telling Claude when to switch to this profile. Recommended; without it Claude only has the name to go on. |
| `idle_timeout_seconds`        | silence after a turn before Meeko acts (check-in or close). Default `5.0`. Non-positive disables. |
| `idle_prompt`                 | spoken check-in after that silence. Unset (default) skips straight to closing.     |
| `idle_close_seconds`          | further silence after `idle_prompt` before closing. Only used with `idle_prompt`. Default `20.0`. |
| `idle_close_text`             | spoken just before the session closes. Unset (default) closes silently.            |
| `post_wake_timeout_seconds`   | silence window after the wake word, before the first turn; on expiry the session closes silently and returns to IDLE. Default `15.0`. Non-positive disables. |

With no `idle_*` keys a profile closes silently after 5 seconds. See the bundled `meeko/default_config.toml` to see how the default profiles are configured.

## Raspberry Pi

Optional — skip this section if you're running on a Mac; nothing here is needed beyond the `[audio]` block in [Setup](#setup).

Meeko was built on a Raspberry Pi 5 with a ReSpeaker XVF3800 USB mic array, and this section covers that pairing. The XVF3800 is a USB device that works on any host, so the AEC tuning below applies wherever you use it; the udev rule and the systemd unit are Linux-specific.

### Tune the ReSpeaker's AEC sensitivity

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

### What the LED ring shows

The LED ring mirrors the state machine, which on a device with no screen is the only way to tell what Meeko is doing:

| Ring                    | State              | Meaning                                              |
|-------------------------|--------------------|------------------------------------------------------|
| Off                     | `IDLE`             | Waiting for the wake word. Nothing reaches Deepgram. |
| Solid cyan              | `LISTENING`        | Awake and ready for your turn.                       |
| Solid brighter cyan     | `LISTENING_ACTIVE` | Hearing you speak right now.                         |
| Breathing blue          | `PROCESSING`       | Waiting on Claude. Speak to cancel the turn.         |
| Solid green             | `SPEAKING`         | Talking back. Speak over it to barge in.             |
| Breathing red (~3s)     | —                  | A turn failed; Meeko returns to `LISTENING` after.   |

Colors and the breath speed are in `PALETTE` at the top of `meeko/leds.py`.

### Run at boot with systemd

To have Meeko start automatically when your Pi boots — and keep running across reboots without you SSH'ing in — install the shipped systemd user service.

1. Copy the unit into your user systemd directory:

   ```bash
   mkdir -p ~/.config/systemd/user
   cp scripts/meeko.service ~/.config/systemd/user/
   ```

   The unit assumes the repo is at `~/meeko`. Edit `WorkingDirectory=` if you cloned somewhere else — it is what the default wake-word model path and the `meeko.log` file resolve against.

2. Enable lingering so your user manager starts at boot without a login session:

   ```bash
   sudo loginctl enable-linger $USER
   ```

3. Enable and start the service:

   ```bash
   systemctl --user daemon-reload
   systemctl --user enable --now meeko
   ```

Useful commands:

- Tail logs: `journalctl --user-unit=meeko -f`
- Logs since boot: `journalctl --user-unit=meeko -b`
- Status: `systemctl --user status meeko`
- Restart (e.g. to pick up code changes): `systemctl --user restart meeko`
- Stop: `systemctl --user stop meeko`

## Troubleshooting

**Meeko doesn't wake up.** Lower `[wake_word] threshold` from its default `0.96` — the lower the value, the more readily it fires, at the cost of more false wakes. If it wakes on its own instead, raise it. `uv run python -m meeko.wake_word` confirms the preprocessor models are cached; a first run with no network fails here.

**Meeko talks over me, or won't let me interrupt.** Barge-in needs Meeko's own voice kept out of the mic. On the Pi with an XVF3800, that's hardware echo cancellation — see [Tune the ReSpeaker's AEC sensitivity](#tune-the-respeakers-aec-sensitivity), and check the speaker is in the XVF3800's 3.5mm jack rather than the Pi's own output. On anything without hardware AEC (a Mac's built-in mic, most USB mics), set `mute_mic_while_speaking = true` in `[audio]` instead.

**Meeko uses the wrong mic or speaker, or hears nothing.** List what PyAudio can see:

```
uv run python -m meeko.audio_io
```

Each line gives an index, a name, and the device's channel count — `[0] MacBook Air Microphone  (in=1ch, DEFAULT-IN)`. Set `input_device_index` / `output_device_index` in `[audio]` to pin a device, and make `input_channels` / `output_channels` match the `in=`/`out=` counts shown. A channel-count mismatch is the usual cause of silence or garbled audio: the default of `2` is the ReSpeaker's, and most built-in mics are `1`.

**The LED ring does nothing on Linux.** The USB control interface is root-only by default — install the udev rule, see [Install the udev rule for LED control](#install-the-udev-rule-for-led-control). The ring also stays dark when no XVF3800 is present, which is expected on a Mac.

## Privacy

Meeko stores all conversation transcripts and summaries locally in SQLite on your device — nothing is uploaded to a third-party assistant cloud. Three external services see data:

- **Deepgram** receives microphone audio for speech-to-text, and the text of Claude's responses for text-to-speech.
- **Anthropic** receives the conversation history sent to Claude.
- **Open-Meteo** receives coordinates when you ask about the weather — your `[location]` home coordinates, or the ones Claude supplies for a place you named. No account or API key is involved, and nothing else about the conversation is sent.

Deepgram and Anthropic are accessed with your own API keys; their handling of your data is governed by their respective terms of service.

### Logs

Conversation content — your transcribed speech, Claude's replies, tool-call arguments, timer labels, and session titles — is logged at `DEBUG` only. The default `log_level` is `INFO`, so when Meeko runs under systemd (stderr goes to journald) none of it reaches the system journal. What `journalctl --user-unit=meeko` does show is the content-free trace: state transitions, which tool was called (not with what), how long a dropped echo transcript was (not what it said), and errors. Per-turn `[timing]` lines are `DEBUG`.

Setting `log_level = "DEBUG"` (or `MEEKO_LOG_LEVEL=DEBUG`) turns that content logging on. Pair it with `log_target = "file"` to keep it in `meeko.log` instead of the journal, and remember journald retains what it captured until its own rotation — `journalctl --user --vacuum-time=1s` clears your user journal.

## Project status

Meeko is a personal project, maintained in spare time. It works and I use it daily, but I review issues and pull requests in batches — expect weeks, not days. Contributions are genuinely welcome anyway; please read [CONTRIBUTING.md](CONTRIBUTING.md) first, and open an issue before starting anything large so you don't build something I end up declining.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for dev setup, tests, and lint.

## License

MIT — see [LICENSE](LICENSE). Note that openWakeWord's pre-trained preprocessor models, which the wake-word path downloads at runtime, are CC-BY-NC-SA-4.0; see [models/README.md](models/README.md) before using Meeko commercially.

## Inspiration
After a concussion left me unable to look at screens for a week, I couldn't use my laptop, my phone, or even watch TV. What saved my sanity was talking to Claude's voice mode — hours of conversation about whatever was on my mind, no screen required.

That experience made me want to build something purpose-built for it: a voice assistant designed from the ground up for long, open-ended conversation, with no screen dependency at all. Meeko is that.
