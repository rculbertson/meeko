# Meeko — Claude Code Instructions

## Project Overview

Meeko is a personal voice assistant for macOS and Raspberry Pi 5. It is designed for long, deep brainstorming conversations — not command-and-control. The architecture overview — component tour, state machine, session persistence, and the rationale behind the key design decisions — is in `ARCHITECTURE.md`. Read it before making significant changes.

`meeko/main.py` is the entry point and composition root: `run()` builds every component and injects its dependencies. The orchestration logic (state machine, mic pump, STT event routing, idle windows, turn worker, post-turn session changes) lives in the `meeko/orchestrator/` package. `meeko/stt_supervisor.py` stays top-level as the STT connection layer.

## Current State

Meeko uses **Deepgram STT (Flux, v2 live)** and **Deepgram TTS (Aura-2)** directly, with **Claude called directly via the Anthropic SDK** (`claude-sonnet-4-6`). Wake-word gating (openWakeWord), SQLite turn persistence, end-of-session summarization with FTS recall, CLI session resume, and a Claude-native tool-use loop for timers, profile switching, and session management (`end_session`, `new_session`, `list_sessions`, `load_session`) are all implemented.

## Tech Stack

- **Python 3.14+**, async/await throughout
- **Deepgram SDK** — STT (Flux model) and TTS (Aura-2) only; do not use Deepgram's Voice Agent API or managed LLM
- **Anthropic SDK** — direct Claude API calls
- **SQLite** — session and transcript storage; FTS5 for resume-by-voice recall
- **PyAudio** — audio I/O (input from ReSpeaker XVF3800 left channel, output via the 3.5mm jack)
- **openWakeWord** — on-device wake-word gating ("Hey Meeko") before STT

## Key Conventions

### Configuration
- Secrets (`DEEPGRAM_API_KEY`, `ANTHROPIC_API_KEY`) live in `.env`.
- Everything else (audio, wake word, LEDs, logging, DB path, compaction threshold, home location) lives in `meeko.toml` under `[system]`, `[audio]`, `[wake_word]`, `[claude]`, `[location]`, `[weather]` tables. See `README.md §Configuration` for the full list of options and defaults. `meeko/default_config.toml` (the bundled default, copied to the XDG location on first run) shows the minimum required config (profiles only); all other tables are optional and fall back to defaults when absent.
- **Config location** follows XDG. `meeko.config.resolve_config_path()` resolves it by precedence: `$MEEKO_CONFIG` → `$XDG_CONFIG_HOME/meeko/meeko.toml` (i.e. `~/.config/meeko/meeko.toml`) → `./meeko.toml` (cwd fallback, kept for dev). On first run `ensure_config_exists()` copies the bundled `meeko/default_config.toml` into the XDG location, so a fresh checkout starts with working defaults — no manual copy step. Do not re-track `meeko.toml` (still gitignored for the cwd-fallback dev case). The DB lives separately in the XDG data dir (`meeko.sessions.default_db_path()`, `~/.local/share/meeko/`).
- Any `MEEKO_*` environment variable overrides the corresponding TOML value at runtime — useful for one-off testing without editing the file.
- Loading happens once in `run()`: `ensure_config_exists()` resolves/creates the path, then `load_config()` and `load_profiles()` are both passed that path. Components receive their values via constructor kwargs (no module-level `os.environ.get` reads).

### Claude API calls
- Model: `claude-sonnet-4-6` for main conversation and end-of-session summarization (long transcripts + summary quality drives resume-by-voice recall)
- Prompt caching (`cache_control` on the stable prefix) and server-side compaction are wired up in `meeko/claude_client.py` (compaction beta `compact-2026-01-12`, strategy `compact_20260112`). Both operate on the in-memory message array only
- The on-disk SQLite transcript is the source of truth — the server-emitted `compaction` block stays in-memory and is filtered out before persistence so transcripts remain verbatim
- Compaction threshold: `[claude] compaction_trigger_tokens` in `meeko.toml` (default `150000`). Env override: `MEEKO_COMPACTION_TRIGGER_TOKENS`.

### Turn persistence
- Write every turn to SQLite immediately on completion, not buffered, not at session end (see `meeko/sessions.py` and `ClaudeClient._persist`)
- This is load-bearing: server-side compaction replaces early turns in memory with a summary block; SQLite is the only full record

### Session management (Claude tool-use)
- Session-management intents are exposed to Sonnet as Claude-native tools, not detected by a separate classifier (see `meeko/tools/session.py`). Sonnet decides when to call them based on full conversation context.
- Tools: `end_session`, `new_session`, `list_sessions(query)`, `load_session(id)`. Per-profile system prompts instruct Sonnet to acknowledge verbally before `new_session` / `load_session`; `end_session` is called silently so Meeko turns off without speaking.
- Handlers only set flags on `SessionManager`; the orchestrator drains TTS first, then performs the actual session transition after SPEAKING ends (`apply_post_turn_session_change` in `meeko/orchestrator/session_change.py`, which `TurnWorker` calls after each turn).
- `load_session` semantics: orchestrator finalizes (summarizes) the abandoned session in the background, then swaps the in-memory message array to the loaded transcript and rebinds `ClaudeClient` to the loaded session's SQLite row.
- End-of-session summarization runs on Sonnet via `meeko/session_summary.py`, fire-and-forget. The summary + transcript are written to the standalone FTS5 `sessions_fts` table for keyword recall.
- `SummaryScheduler` (same module) owns the background tasks: `fire()` after a session ends, a startup `backfill()` for sessions a killed run never summarized (capped at 3 concurrent), and `aclose()` at shutdown — which must run before `store.close()`, or an in-flight summary writes into a closed database.

### Audio
- Read from the **left channel** of the ReSpeaker XVF3800 USB device — this is the AEC-processed output
- Speaker must route through the XVF3800's **3.5mm jack**, not the Pi's audio output (required for hardware AEC)
- Barge-in: `StartOfTurn` during SPEAKING **or PROCESSING** → stop TTS, cancel Claude request, transition to LISTENING. (Deepgram Flux emits `StartOfTurn`, not `SpeechStarted`.)
- Barge-in and the turn worker live together in `TurnWorker` (`meeko/orchestrator/turn_worker.py`). Each turn runs as a sub-task so `request_barge_in()` can cancel just that turn. A barge-in cancel and a shutdown cancel reach the worker as the same `CancelledError`, and only the `_barge_in_requested` flag tells them apart: `stop_event` can't, because asyncio's shutdown cancels the worker before `run()`'s `finally` sets it. `request_barge_in()` flips state to LISTENING synchronously even with no turn in flight (e.g. a timer's "timer is done" announcement), so the following `EndOfTurn` isn't dropped as echo. Mistakes here tend to pass the `run()` drive-throughs and show up as a dropped utterance or a quietly dead worker; `tests/test_turn_worker.py` tests them directly.
- Which STT event does what in which state is decided by `SttEventRouter` in `meeko/orchestrator/stt_events.py`, not in `main.py`. The state machine it branches on (`State`, `StateManager`) lives in `meeko/orchestrator/state.py`.
- The mic side is `MicPump` in `meeko/orchestrator/mic_pump.py`: it owns the wake-word gate (nothing reaches Deepgram while IDLE) and the `mute_mic_while_speaking` drop. The Deepgram keepalive pump lives in `meeko/stt_supervisor.py` alongside `KEEPALIVE_INTERVAL_S`.
- Per STT session, the mic pump, event router and keepalive run together under `run_session_workers` (`meeko/stt_supervisor.py`): the first to finish ends the session, and if it raised, that exception must reach `STTSupervisor.run()` — that's what triggers backoff and reconnect. Don't swallow it.
- **Device selection** — `[audio]` in `meeko.toml` (env-var overrides in parens, all optional):
  - `input_device_index` / `output_device_index` (env: `MEEKO_INPUT_DEVICE_INDEX` / `MEEKO_OUTPUT_DEVICE_INDEX`) — pin a specific PyAudio device. Unset = OS default. Run `uv run python -m meeko.audio_io` to list indices.
  - `input_channels` / `output_channels` (env: `MEEKO_INPUT_CHANNELS` / `MEEKO_OUTPUT_CHANNELS`) — native channel counts. Default `2` / `2` (ReSpeaker).
  - `mute_mic_while_speaking = true` (env: `MEEKO_MUTE_MIC_WHILE_SPEAKING=1`) — for devices without hardware AEC (e.g. Mac built-in).
  - Mac built-in mic/speaker: `input_channels = 1`, `mute_mic_while_speaking = true` (output channels default `2` is fine for stereo speakers).

### Wake word
- On startup Meeko sits in `IDLE` — mic is open but audio is fed to an openWakeWord detector, not Deepgram STT. Saying "Hey Meeko" transitions the session to `LISTENING` (one-shot per session).
- Configured in `[wake_word]` (env-var overrides in parens):
  - `model` (env: `MEEKO_WAKE_WORD_MODEL`) — path to the ONNX model. Default `models/hey_meeko.onnx`.
  - `threshold` (env: `MEEKO_WAKE_WORD_THRESHOLD`) — confidence (0–1). Default `0.96`.
  - `disabled = true` (env: `MEEKO_WAKE_WORD_DISABLED=1`) — skip the gate; start directly in `LISTENING`.
- `post_wake_timeout_seconds` is a **per-profile** key (not `[wake_word]`), default `15.0`: the silence window between the wake word firing and the user's first turn. On expiry the session always closes silently and returns to `IDLE`, whatever the profile's `idle_*` keys say. Non-positive disables it.
- First run downloads openWakeWord's melspectrogram, embedding and VAD models (~6.7 MB; its six bundled wake words are suppressed, see `models/README.md`). Run `uv run python -m meeko.wake_word` to pre-populate the cache on a network-connected host before deploying offline (e.g. Pi image bake).

### LEDs
- The XVF3800's WS2812 ring is driven by `meeko/leds.py` to mirror the state machine: IDLE off, LISTENING solid cyan, LISTENING_ACTIVE solid brighter cyan while the user is speaking, PROCESSING blue breath, SPEAKING solid green. Errors get a ~3s red breath.
- We talk to the chip directly over libusb (pyusb vendor control transfers on resid 20) — not via ReSpeaker's `xvf_host.py`.
- Linux defaults the USB control interface to root-only. Install `scripts/99-meeko-xvf3800.rules` once (see README) so the `plugdev` group can drive it.
- `[system] led_disabled = true` (env: `MEEKO_LED_DISABLED=1`) — skip LED control entirely. Also auto-disabled if the XVF3800 isn't found or pyusb/libusb is unavailable, so Mac dev runs need no special handling.

### Home location
- `[location] latitude` / `longitude` (env: `MEEKO_LATITUDE` / `MEEKO_LONGITUDE`) — decimal degrees, must both be set or both unset.
- Consumed in two places:
  - **Weather tool** uses them as the default when the user doesn't name a place.
  - **System prompt** gets a small `_location_block` (in `meeko/claude_client.py`) inside the cached prefix, so Sonnet can answer location-aware questions (sunset/sunrise, regional references, climate zone, etc.) without the user repeating their location. Sonnet reverse-geocodes the decimal coords from training data — one source of truth, no separate `city`/`region` field.

### Weather tool
- `meeko/tools/weather.py` exposes a single `get_weather` tool to Sonnet, backed by [Open-Meteo](https://open-meteo.com)'s forecast endpoint (free, no API key). No geocoder.
- Two modes on the one tool. Default: current conditions + a single requested day (today, or any `date` up to 14 days out), bounded with `start_date`/`end_date`. `hourly = true`: one compact row per hour (`Mon 3 PM  54°F  rain  90%`) for the next 48 hours starting at the current hour — `date` is ignored there. Sonnet is told to set `hourly` for part-of-day questions and to summarize the rows rather than read them out.
- Both paths reconstruct "now" at the *forecast location* from the API's `utc_offset_seconds` (`_now_local`), not from this machine's clock — the hourly window and the precipitation-timing cutoff both depend on it.
- Home coords come from `[location]` (see above). `[weather]` only holds:
  - `units` (env: `MEEKO_WEATHER_UNITS`) — `imperial` (default) or `metric`. Drives the Open-Meteo query params and the symbols (`°F`/`mph`/`in` vs `°C`/`km/h`/`mm`) in the formatted response.
- For ad-hoc places ("weather in Tokyo"), **Sonnet supplies `latitude`, `longitude`, and `place_label` directly from its own geographic knowledge** — the tool description tells Sonnet not to web-search for coordinates. This avoids the comma-disambiguation traps a geocoder runs into (`"Portland, Maine"` vs `"Portland, Oregon"`) and saves a network round trip.
- All error paths return a friendly string (partial coord args, out-of-range coords, network failure, no home location) — never raise — so the tool-use turn stays intact.

### State machine
States: `IDLE → (wake word) → LISTENING → PROCESSING → SPEAKING → (barge-in back to LISTENING)`. The wake-word gate is one-shot per session — follow-up turns do not require re-wakeing.

Three paths return to `IDLE`: the `end_session` tool, the post-turn idle timeout (silently, or after a spoken check-in, per the active profile's `idle_*` keys), and the post-wake timeout when no first turn ever arrives. See `ARCHITECTURE.md` §5.1.

Both silence windows live in `meeko/orchestrator/idle.py` (`IdleController`), not in `main.py`. They don't end the session themselves — they set the `SessionManager` end flag and post `IDLE_TIMEOUT_SENTINEL` to the turn queue, and `TurnWorker` (`meeko/orchestrator/turn_worker.py`) runs the normal post-turn handling without a Claude/TTS round-trip.

### Debugging
- Run with `PYTHONASYNCIODEBUG=1` (Python's built-in env var) to enable asyncio debug mode. The loop will then log a WARNING (`Executing <Handle ...> took N.NNN seconds`) whenever a synchronous callback holds it ≥100 ms — useful for diagnosing loop stalls (e.g. STT websocket keepalive timeouts). Off in normal operation; debug mode wraps every coroutine creation with traceback capture and is a real cost on hot paths. Meeko routes asyncio's own warnings through the same logging handler as `meeko.*` logs, so they pick up the timestamp format and land in the rotating log when `MEEKO_LOG_TARGET=file`.

## What's Out of Scope (v1)

Do not add: web/mobile UI, multi-user support, semantic search over sessions, session deletion or editing by voice, cross-device sync. See `ARCHITECTURE.md` §9.

## Testing

- Any non-trivial change — new behavior, bug fix, or refactor of existing logic — must include updated or new tests
- Tests live in `tests/`; use **pytest** and **pytest-asyncio** for async cases
- Tests that require live API keys (Deepgram, Anthropic) must be marked `@pytest.mark.integration`; all other tests must be runnable offline
- Run the full test suite before committing: `uv run pytest`
- Trivial changes (docstrings, comments, config tweaks, logging) do not require tests

## Git Workflow
- Always implement features on a new branch, never directly on main
- Branch naming: `<github-username>/<short-description>`
- Commit when a discrete, working piece is complete; each commit should run correctly on its own

## Before Opening a PR
1. Commit any outstanding changes
2. `gh pr create` — ruff and tests run automatically via pre-commit hooks