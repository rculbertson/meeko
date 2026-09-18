# Meeko

Meeko is a personal voice assistant backed by Claude. It handles the usual quick questions, as well as extended, open-ended thinking sessions where context accumulates over time. Meeko remembers the full text of every conversation, so you can pause mid-thought, come back days later, and pick up exactly where you left off. Just say "let's go back to the conversation about the app I'm building" and Meeko finds it and resumes.


## Project status

Meeko is a personal project, maintained in spare time. It works and I use it daily, but I review issues and pull requests in batches — expect weeks, not days. Contributions are genuinely welcome anyway; please read [CONTRIBUTING.md](CONTRIBUTING.md) first, and open an issue before starting anything large so you don't build something I end up declining.

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

That's all the setup required. On first run Meeko creates a config file at `~/.config/meeko/meeko.toml` (copied from the bundled `meeko/default_config.toml`) and starts with working defaults; edit that file to add personal settings like your home location.

## Running

```
uv run python -m meeko.main
```

Say "Hey Meeko" to wake it. Press `Ctrl+C` to quit.

Two flags are available for working with prior sessions from the terminal:

```
uv run python -m meeko.main --list-sessions        # print prior sessions and exit
uv run python -m meeko.main --resume               # resume the most recent session
uv run python -m meeko.main --resume SESSION_ID    # resume a specific session
```

Resuming by voice works too — just ask ("let's go back to the conversation about the app I'm building") and Meeko searches and loads it.

The default config provides two profiles — `query` (short replies, auto-closes after a short silence) and `conversation` (substantive thinking-partner persona that stays open through pauses). You can switch between them mid-session by asking — for example "switch to conversation mode", "let's have a long conversation", "switch back to query mode", or "just quick questions from now on". Ask "what modes are available?" to list profiles.

## How it works

A turn flows: mic audio → Deepgram STT (Flux, with semantic end-of-turn detection) → Claude (called directly via the Anthropic SDK, with prompt caching, tool use, web search, and server-side compaction) → Deepgram TTS (Aura-2) → speaker. Every turn is persisted to SQLite immediately, so the on-disk transcript is always the source of truth. Wake-word gating ("Hey Meeko") runs on-device with [openWakeWord](https://github.com/dscripka/openWakeWord). Session-management intents (end, new, list, load) are exposed to Claude as tools, so Sonnet decides when to call them based on the full conversation rather than a separate classifier.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full design — state machine, session persistence, prompt caching, server-side compaction, and the rationale behind the key design decisions.

## Configuration

All non-secret settings live in `meeko.toml`. Meeko looks for it in this order: `$MEEKO_CONFIG`, then `$XDG_CONFIG_HOME/meeko/meeko.toml` (i.e. `~/.config/meeko/meeko.toml`), then `./meeko.toml` in the working directory. If none exist, it auto-creates `~/.config/meeko/meeko.toml` from the bundled `meeko/default_config.toml` on first run. Edit that file to customize. Each setting also has a `MEEKO_*` environment variable that overrides the TOML value at runtime — useful for one-off testing without editing the file. (API keys stay in `.env` in the project root.)

Settings are grouped into six tables: `[system]`, `[audio]`, `[wake_word]`, `[claude]`, `[location]`, and `[weather]`. Profiles (personas) are defined under `[profiles.<name>]`, and a top-level `default_profile = "<name>"` key selects which profile a fresh session starts in.

### `[system]`

| TOML key       | Env var              | Default              | Description                                                                 |
|----------------|----------------------|----------------------|-----------------------------------------------------------------------------|
| `log_level`    | `MEEKO_LOG_LEVEL`    | `DEBUG`              | One of `DEBUG`, `INFO`, `WARNING`, `ERROR`.                                 |
| `log_target`   | `MEEKO_LOG_TARGET`   | unset (stderr)       | Set to `file` to log to `meeko.log` (rotating, 5 MB × 3 files).             |
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
| `model`     | `MEEKO_WAKE_WORD_MODEL`       | `models/hey_meeko.onnx`  | Path to the openWakeWord ONNX model.                                       |
| `threshold` | `MEEKO_WAKE_WORD_THRESHOLD`   | `0.96`                   | Confidence threshold (0–1). Lower = more sensitive, more false wakes.      |
| `disabled`  | `MEEKO_WAKE_WORD_DISABLED`    | `false`                  | Skip the wake gate; start directly in `LISTENING` (useful for development).|

### `[claude]`

| TOML key                    | Env var                            | Default  | Description                                                                |
|-----------------------------|------------------------------------|----------|----------------------------------------------------------------------------|
| `compaction_trigger_tokens` | `MEEKO_COMPACTION_TRIGGER_TOKENS`  | `150000` | Input-token threshold that triggers server-side compaction.                |
| `web_search_enabled`        | `MEEKO_WEB_SEARCH_ENABLED`         | `true`   | Expose Anthropic's server-side web search tool to the model.               |
| `web_search_max_uses`       | `MEEKO_WEB_SEARCH_MAX_USES`        | `2`      | Max web-search calls per turn.                                             |

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

`[profiles.<name>]` blocks define personas. The name is free-form: the default config ships `query` (auto-closes after a short silence) and `conversation` (stays open through pauses, asks before closing), and you can add more — each is just another block, and "switch to <name> mode" switches to it by voice. How a profile behaves when you go quiet is set entirely by its `idle_*` keys, not its name. A top-level `default_profile = "<name>"` key selects which profile a fresh session starts in and is required. Each profile has:

| Key                           | Description                                                                        |
|-------------------------------|------------------------------------------------------------------------------------|
| `wake_word`                   | Wake-phrase label (advisory; the ONNX model determines the actual phrase).         |
| `voice`                       | Aura-2 voice id, e.g. `mars`, `andromeda`. Optional; defaults to `asteria`.          |
| `prompt`                      | The system prompt that defines the persona.                                        |
| `description`                 | One line telling Claude when to switch to this profile. Optional but recommended; without it Claude only has the name to go on. |
| `idle_timeout_seconds`        | silence after a turn before Meeko acts (check-in or close). Default `5.0`. Non-positive disables. |
| `idle_prompt`                 | spoken check-in after that silence. Unset (default) skips straight to closing.     |
| `idle_close_seconds`          | further silence after `idle_prompt` before closing. Only used with `idle_prompt`. Default `20.0`. |
| `idle_close_text`             | spoken just before the session closes. Unset (default) closes silently.            |
| `post_wake_timeout_seconds`   | silence window after the wake word, before the first turn; on expiry the session closes silently and returns to IDLE. Default `15.0`. Non-positive disables. |

With no `idle_*` keys a profile closes silently after 5 seconds; the shipped `conversation` profile sets all four to get its spoken check-in. See the bundled `meeko/default_config.toml` for working examples of both profiles.

**Upgrading from an older config:** `conversation_idle_seconds` and `conversation_close_seconds` were renamed to `idle_timeout_seconds` and `idle_close_seconds`, and Meeko refuses to start while they are present. A `[profiles.conversation]` block that set no timing keys used to get the spoken check-in automatically and now gets the silent 5-second close — copy the four `idle_*` lines from the bundled `[profiles.conversation]` to keep it.

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

## Run Meeko at boot (Raspberry Pi)

To have Meeko start automatically when your Pi boots — and keep running across reboots without you SSH'ing in — install the shipped systemd user service.

1. Copy the unit into your user systemd directory:

   ```bash
   mkdir -p ~/.config/systemd/user
   cp scripts/meeko.service ~/.config/systemd/user/
   ```

   The unit assumes the repo is at `~/meeko`. Edit `WorkingDirectory=` if you cloned somewhere else.

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

## Privacy

Meeko stores all conversation transcripts and summaries locally in SQLite on your device — nothing is uploaded to a third-party assistant cloud. Two external services do see data per turn:

- **Deepgram** receives microphone audio for speech-to-text, and the text of Claude's responses for text-to-speech.
- **Anthropic** receives the conversation history sent to Claude.

Both are accessed with your own API keys; their handling of your data is governed by their respective terms of service.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for dev setup, tests, and lint.

## License

MIT — see [LICENSE](LICENSE). Note that openWakeWord's pre-trained preprocessor models, which the wake-word path downloads at runtime, are CC-BY-NC-SA-4.0; see [models/README.md](models/README.md) before using Meeko commercially.

## Inspiration
After a concussion left me unable to look at screens for a week, I couldn't use my laptop, my phone, or even watch TV. What saved my sanity was talking to Claude's voice mode — hours of conversation about whatever was on my mind, no screen required.

That experience made me want to build something purpose-built for it: a voice assistant designed from the ground up for long, open-ended conversation, with no screen dependency at all. Meeko is that.