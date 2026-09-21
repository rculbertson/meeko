# Configuration Reference

All non-secret settings in Meeko are managed through a TOML configuration file and corresponding environment variables. Secret API keys are stored separately in `.env`.

---

## File Location & Precedence

Meeko searches for its configuration file in the following order:

1. `$MEEKO_CONFIG` (if set, points to an explicit file path)
2. `$XDG_CONFIG_HOME/meeko/meeko.toml` (defaults to `~/.config/meeko/meeko.toml`)
3. `./meeko.toml` (in the current working directory, primarily for local development)

If no configuration file is found, Meeko automatically creates `~/.config/meeko/meeko.toml` on first run by copying the bundled `meeko/default_config.toml`. It starts with working defaults; you can edit that file to add personal settings like your home location.

The SQLite database for session transcripts and search indexes lives separately in your XDG data directory at `~/.local/share/meeko/meeko.db` (honoring `$XDG_DATA_HOME`).

---

## Runtime Overrides

Every configuration key in `meeko.toml` has a corresponding `MEEKO_*` environment variable. Setting an environment variable overrides the TOML file value at runtime without modifying the file on disk — ideal for one-off testing and debugging:

```bash
MEEKO_LOG_LEVEL=DEBUG uv run python -m meeko.main
```

---

## Configuration Tables

Settings are organized into six tables: `[system]`, `[audio]`, `[wake_word]`, `[claude]`, `[location]`, and `[weather]`, alongside profile definitions under `[profiles.<name>]`.

### `[system]`

General process and persistence settings.

| TOML key | Env var | Default | Description |
|---|---|---|---|
| `log_level` | `MEEKO_LOG_LEVEL` | `INFO` | Logging verbosity: `DEBUG`, `INFO`, `WARNING`, or `ERROR`. `DEBUG` logs full conversation text (transcripts and Claude replies); `INFO` logs only operational events (state transitions, tool names). |
| `log_target` | `MEEKO_LOG_TARGET` | unset (stderr) | Set to `file` to log to `meeko.log` in the working directory (rotates at 5 MB, keeping 3 backups). Leave unset to stream to standard error / systemd journal. |
| `db_path` | `MEEKO_DB_PATH` | `~/.local/share/meeko/meeko.db` | SQLite database file location for storing sessions, transcripts, and FTS5 search indexes (honors `$XDG_DATA_HOME`). |
| `led_disabled` | `MEEKO_LED_DISABLED` | `false` | Skip LED ring control. Auto-disabled when running on systems without a ReSpeaker XVF3800. |

### `[audio]`

Audio input and output device routing.

| TOML key | Env var | Default | Description |
|---|---|---|---|
| `input_device_index` | `MEEKO_INPUT_DEVICE_INDEX` | unset | PyAudio device index for the microphone. Unset uses the OS default input. Run `uv run python -m meeko.audio_io` to view device indices. |
| `output_device_index` | `MEEKO_OUTPUT_DEVICE_INDEX` | unset | PyAudio device index for the speaker. Unset uses the OS default output. |
| `input_channels` | `MEEKO_INPUT_CHANNELS` | `2` | Native input channel count. Use `1` for most built-in laptop microphones. The default of `2` is configured for the ReSpeaker XVF3800. |
| `output_channels` | `MEEKO_OUTPUT_CHANNELS` | `2` | Native output channel count. |
| `mute_mic_while_speaking` | `MEEKO_MUTE_MIC_WHILE_SPEAKING` | `false` | Software-mutes the microphone during TTS playback. **Set to `true` on devices without hardware AEC** (e.g. Mac built-in mic) to prevent Meeko from hearing itself talk. |

### `[wake_word]`

On-device wake-word detection using openWakeWord.

| TOML key | Env var | Default | Description |
|---|---|---|---|
| `model` | `MEEKO_WAKE_WORD_MODEL` | `models/hey_meeko.onnx` | Path to the openWakeWord ONNX model. Resolved relative to the process working directory; use an absolute path when running from elsewhere. See [wake-word.md](wake-word.md) to train your own phrase. |
| `threshold` | `MEEKO_WAKE_WORD_THRESHOLD` | `0.96` | Confidence threshold (0.0 to 1.0). Lower values make detection more sensitive at the cost of potential false wakes. |
| `disabled` | `MEEKO_WAKE_WORD_DISABLED` | `false` | When `true`, skips wake-word gating and starts directly in `LISTENING` mode (useful for development). |

### `[claude]`

Anthropic API parameters and server-side features.

| TOML key | Env var | Default | Description |
|---|---|---|---|
| `compaction_trigger_tokens` | `MEEKO_COMPACTION_TRIGGER_TOKENS` | `150000` | Input token count that triggers Claude server-side context compaction for long sessions. |
| `web_search_enabled` | `MEEKO_WEB_SEARCH_ENABLED` | `true` | Exposes Anthropic's server-side web search tool to Claude. |
| `web_search_max_uses` | `MEEKO_WEB_SEARCH_MAX_USES` | `4` | Maximum web search attempts per turn. Serves as a circuit breaker for stalled searches, not a quality dial. |

### `[location]`

Home coordinates used for location awareness. Both keys must be set together or left unset.

| TOML key | Env var | Default | Description |
|---|---|---|---|
| `latitude` | `MEEKO_LATITUDE` | unset | Home latitude in decimal degrees (e.g., `40.7484`). |
| `longitude` | `MEEKO_LONGITUDE` | unset | Home longitude in decimal degrees (e.g., `-73.9857`). |

When set, these coordinates are:
1. Used as the default location for the weather tool when you don't explicitly name a city.
2. Injected into Claude's system prompt so it can naturally answer location-dependent questions (local sunrise/sunset, climate, regional references). Look them up once on Google Maps (right-click a location → copy coordinates).

### `[weather]`

Settings for the Open-Meteo weather tool.

| TOML key | Env var | Default | Description |
|---|---|---|---|
| `units` | `MEEKO_WEATHER_UNITS` | `imperial` | Measurement units: `imperial` (°F, mph, inches) or `metric` (°C, km/h, mm). |

Forecasts are retrieved from [Open-Meteo](https://open-meteo.com) (free, no API key required).

---

## Profiles & Idle Behavior

Profiles configure Claude's persona, the text-to-speech voice, and how Meeko handles pauses and silence.

A top-level `default_profile` key specifies which profile fresh sessions start in:

```toml
default_profile = "query"
```

Each profile is defined in a `[profiles.<name>]` table. Profile names are arbitrary; you can add as many profiles as you like and switch between them by voice (*"switch to conversation profile"* or *"what profiles are available?"*).

### Profile Keys

Within each `[profiles.<name>]` block:

| Key | Type | Default | Description |
|---|---|---|---|
| `prompt` | String | **Required** | The system prompt defining the persona and behavior. |
| `voice` | String | `asteria` | Deepgram Aura-2 voice ID (e.g., `mars`, `asteria`, `andromeda`). |
| `description` | String | unset | One-line description telling Claude when to switch to this profile. Strongly recommended. |
| `idle_timeout_seconds` | Float | `5.0` | Seconds of silence in `LISTENING` after a turn before Meeko acts. Non-positive disables it. |
| `idle_prompt` | String | unset | Spoken check-in phrase after `idle_timeout_seconds`. If unset, Meeko skips straight to closing. |
| `idle_close_seconds` | Float | `20.0` | Seconds of additional silence after `idle_prompt` finishes before closing the session. |
| `idle_close_text` | String | unset | Spoken phrase just before the session closes. If unset, Meeko closes silently. |
| `post_wake_timeout_seconds` | Float | `15.0` | Seconds of silence allowed between the wake word and the first turn. If this expires with no speech, the session returns silently to `IDLE`. Non-positive disables it. |

### Shipped Profiles

Meeko ships with two pre-configured profiles in `meeko/default_config.toml`:

1. **`query`** (Quick Q&A):
   - Designed for fast, one-off questions.
   - Closes silently after 5 seconds of silence (`idle_timeout_seconds = 5.0`).
2. **`conversation`** (Deep Brainstorming):
   - Designed for long, open-ended discussions where you need time to think.
   - Waits 60 seconds before speaking a check-in (*"Would you like to continue, or should we end the session now?"*), then waits another 20 seconds before saying *"Okay, ending the session now"* and closing.

### Adding a Custom Profile

To create a new profile, add a new table to `meeko.toml`:

```toml
[profiles.tutor]
voice = "andromeda"
description = "Socratic tutor for learning new topics and explaining complex concepts."
idle_timeout_seconds = 30.0
idle_prompt = "Let me know when you're ready to continue, or if you have any questions."
idle_close_seconds = 15.0
idle_close_text = "Talk to you later!"
prompt = """
You are Meeko, acting as a patient Socratic tutor. Guide the user step by step \
through difficult concepts by asking leading questions rather than simply lecturing. \
Keep responses spoken and conversational without markdown formatting.
"""
```

Once added, you can switch to it simply by asking: *"Hey Meeko, switch to tutor mode."*
