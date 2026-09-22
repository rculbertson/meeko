# Meeko — Instructions for Claude Code

## Project Overview

Meeko is a personal voice assistant for macOS and Raspberry Pi 5. It is designed for long, deep brainstorming conversations — not command-and-control. The architecture overview — component tour, state machine, session persistence, and design rationale — is in `docs/architecture.md`. Read it before making significant changes.

### Architecture & Subsystems

- `meeko/main.py`: Entry point and composition root — `run()` builds every component and injects its dependencies.
- `meeko/orchestrator/`: Core orchestration logic — state machine, mic wake-word gating, STT event routing, async turn worker with barge-in cancellation, post-turn session transitions, and idle silence windows.
- `meeko/tools/`: Claude-native tool implementations and dispatching (`session`, `weather`, `timer`, `profile`).
- `meeko/config.py`: TOML configuration (`MeekoConfig`), profile parsing (`Profile`), and XDG path resolution.
- External services & hardware integration: Deepgram STT/TTS connections, PyAudio device streaming, on-device openWakeWord detection, ReSpeaker XVF3800 LED control, and SQLite turn persistence with FTS5 search.

## Tech Stack

- **Python 3.14+**, async/await throughout
- **Deepgram SDK** — STT (Flux model, v2 live) and TTS (Aura-2) only; do not use Deepgram's Voice Agent API or managed LLM
- **Anthropic SDK** — direct Claude API calls (`claude-sonnet-5`)
- **SQLite** — session and transcript storage; FTS5 for resume-by-voice recall
- **PyAudio** — audio I/O (input from ReSpeaker XVF3800 left channel, output via 3.5mm jack)
- **openWakeWord** — on-device wake-word gating ("Hey Meeko") before STT

---

## Essential Commands

### Running Meeko & Dev Helpers
```bash
uv run meeko                          # Run the assistant (or: uv run python -m meeko.main)
uv run python -m meeko.audio_io       # List PyAudio device indices (input/output)
uv run python -m meeko.wake_word      # Download/pre-populate openWakeWord base models
```

### Testing
```bash
uv run pytest                                            # Run offline test suite with coverage
uv run pytest --no-cov                                   # Fast run (skips coverage)
uv run pytest tests/test_turn_worker.py -k test_barge_in # Run a specific test
uv run pytest -m integration                             # Run tests requiring live API keys (Deepgram, Anthropic)
```

### Linting, Formatting, Typing & Complexity
```bash
uv run ruff check .                   # Lint check
uv run ruff check --fix .             # Lint autofix
uv run ruff format .                  # Format code (88-char line length)
uv run pyright                        # Type check meeko/ (basic mode)
./scripts/check_complexity.sh         # Cyclomatic complexity gate (xenon)
```

---

## Key Conventions & Invariants

### Configuration
- Secrets (`DEEPGRAM_API_KEY`, `ANTHROPIC_API_KEY`) live in `.env`.
- Everything else (audio, wake word, LEDs, logging, DB path, compaction threshold, home location, web search) lives in `meeko.toml` under `[system]`, `[audio]`, `[wake_word]`, `[claude]`, `[location]`, `[weather]`, and `[profiles.<name>]`. See `docs/configuration.md` for options and defaults. `meeko/default_config.toml` shows the minimum required config; all other tables are optional and fall back to defaults when absent.
- **Config location** follows XDG: `$MEEKO_CONFIG` → `$XDG_CONFIG_HOME/meeko/meeko.toml` (`~/.config/meeko/meeko.toml`) → `./meeko.toml` (cwd fallback, kept for dev). On first run `ensure_config_exists()` copies `meeko/default_config.toml` into the XDG location. Do not re-track `meeko.toml`. The DB lives separately in `$XDG_DATA_HOME/meeko/meeko.db` (`~/.local/share/meeko/meeko.db`).
- Any `MEEKO_*` environment variable overrides the corresponding TOML value at runtime.
- Loading happens once in `run()`: `ensure_config_exists()` resolves/creates the path, then `load_config()` and `load_profiles()` are passed that path. Components receive values via constructor kwargs (no module-level `os.environ.get` reads).

### Claude API & Serialization Invariants
- Model: `claude-sonnet-5` for main conversation and end-of-session summarization.
- Prompt caching (`cache_control` on the stable prefix) and server-side compaction (`compact-2026-01-12`, strategy `compact_20260112`) are wired in `meeko/claude_client.py`. Both operate on the in-memory message array only.
- Compaction threshold: `[claude] compaction_trigger_tokens` in `meeko.toml` (default `150000`, env: `MEEKO_COMPACTION_TRIGGER_TOKENS`). Web search is configured via `[claude] web_search_enabled` (default `true`) and `web_search_max_uses` (default `4`).
- **Replay block fidelity:** `_serialize_block` rebuilds assistant blocks from canonical fields for replay. **Never drop `caller`** from `server_tool_use` / `web_search_tool_result`: web search's dynamic filtering nests searches inside `code_execution`, and without `caller` the API 400s the *next* turn of the session. See `docs/architecture.md` §4.3 and `tests/test_claude_api_contract.py`.
- **Immediate turn persistence:** Write every turn to SQLite immediately on completion, not buffered, not at session end (see `meeko/sessions.py` and `ClaudeClient._persist`). Server-side compaction replaces early turns in memory with a summary block; SQLite is the only full verbatim record.

### Session Management (Claude Tool-Use)
- Session-management intents are exposed to Sonnet as Claude-native tools (`meeko/tools/session.py`): `end_session`, `new_session`, `list_sessions(query)`, `load_session(id)`.
- Per-profile system prompts instruct Sonnet to acknowledge verbally before `new_session` / `load_session`; `end_session` is called silently so Meeko turns off without speaking.
- Handlers only set flags on `SessionManager`; the orchestrator drains TTS first, then performs the actual session transition after `SPEAKING` ends (`apply_post_turn_session_change` in `meeko/orchestrator/session_change.py`, called by `TurnWorker`).
- `load_session` semantics: orchestrator finalizes (summarizes) the abandoned session in the background, then swaps the in-memory message array to the loaded transcript and rebinds `ClaudeClient` to the loaded session's SQLite row.
- End-of-session summarization runs on Sonnet via `meeko/session_summary.py`, fire-and-forget. Summary and transcript are written to the FTS5 `sessions_fts` table.
- `SummaryScheduler` owns background summary tasks: `fire()` after session end, startup `backfill()`, and `aclose()` at shutdown (must run before `store.close()`).

### Audio & Barge-In
- Read from the **left channel** of the ReSpeaker XVF3800 USB device (AEC-processed output).
- Speaker must route through the XVF3800's **3.5mm jack**, not the Pi's audio output (required for hardware AEC).
- Device selection in `[audio]` (`input_device_index`, `output_device_index`, `input_channels`, `output_channels`, `mute_mic_while_speaking`). Set `mute_mic_while_speaking = true` (env: `MEEKO_MUTE_MIC_WHILE_SPEAKING=1`) for devices without hardware AEC (e.g. Mac built-in mic).
- Barge-in: `StartOfTurn` during `SPEAKING` or `PROCESSING` → stop TTS, cancel Claude request, transition to `LISTENING`. (Deepgram Flux emits `StartOfTurn`, not `SpeechStarted`.)
- Barge-in and the turn worker live in `TurnWorker` (`meeko/orchestrator/turn_worker.py`). Each turn runs as a sub-task so `request_barge_in()` can cancel just that turn. A barge-in cancel and a shutdown cancel reach the worker as the same `CancelledError`; only the `_barge_in_requested` flag tells them apart (`stop_event` cannot, because asyncio shutdown cancels the worker before `run()`'s `finally` sets it).
- STT event routing is managed by `SttEventRouter` in `meeko/orchestrator/stt_events.py`.
- Per STT session, the mic pump, event router, and keepalive run together under `run_session_workers` (`meeko/stt_supervisor.py`). The first to finish ends the session, and if it raised, that exception must reach `STTSupervisor.run()` to trigger backoff and reconnect. Do not swallow it.

### Hardware-Dependent Code & User Manual Testing
- Changes to `meeko/leds.py`, `meeko/audio_io.py`, or `meeko/speaker.py` touch physical hardware that unit tests and CI cannot fully validate.
- When modifying these files, **inform the user that manual verification on a physical Raspberry Pi 5 with the ReSpeaker XVF3800 is required**.
- PRs touching these files should include hardware observation notes (what device was tested, audio clarity, LED behavior).

### Wake Word
- On startup Meeko sits in `IDLE` — mic audio feeds the openWakeWord detector, not STT. Saying "Hey Meeko" transitions to `LISTENING` (one-shot per session).
- Configured in `[wake_word]` (`model`, `threshold`, `disabled = true` to skip gate).
- `post_wake_timeout_seconds` is a per-profile key (default `15.0`): silence window between wake word and user's first turn. Expiry returns to `IDLE` silently.

### LEDs
- XVF3800 WS2812 ring is driven by `meeko/leds.py` to mirror the state machine: IDLE off, LISTENING solid cyan, LISTENING_ACTIVE bright cyan, PROCESSING blue breath, SPEAKING solid green, errors red breath (~3s).
- Direct control over libusb (pyusb vendor control transfers on resid 20).
- `[system] led_disabled = true` (env: `MEEKO_LED_DISABLED=1`) skips LED control. Auto-disabled if XVF3800 isn't found or pyusb/libusb is unavailable.

### Home Location & Weather Tool
- `[location] latitude` / `longitude` (decimal degrees). Sonnet reverse-geocodes coordinates from its training data.
- Injected into Claude's cached system prompt prefix (`_location_block`) and used as the default location for `get_weather`.
- `meeko/tools/weather.py` exposes `get_weather` backed by Open-Meteo (free, no API key). Units set via `[weather] units = "imperial" | "metric"`. Reconstructs local time from the API's `utc_offset_seconds`.
- All tool handlers must return friendly error strings on failure — never raise.

### State Machine & Silence Windows
- States: `IDLE → (wake word) → LISTENING → PROCESSING → SPEAKING → (barge-in back to LISTENING)`.
- Three paths return to `IDLE`: `end_session` tool, post-turn idle timeout (spoken check-in or silent per profile), and post-wake timeout when no first turn arrives.
- Silence windows live in `IdleController` (`meeko/orchestrator/idle.py`). They set `SessionManager` flags and post `IDLE_TIMEOUT_SENTINEL` to the turn queue so `TurnWorker` handles session teardown without extra Claude/TTS calls.

### Logging and Privacy
- Conversation content — transcripts, assistant sentences, tool arguments, timer labels, weather coordinates, session titles, and summarizer responses — must be logged at **DEBUG only**. `[system] log_level` defaults to `INFO`.
- Keep content-free `INFO` logs beside debug logs (tool name without arguments, state transitions, HTTP status without query string).
- Validated by `tests/test_logging_privacy.py`.

### Debugging
- Run with `PYTHONASYNCIODEBUG=1` to enable asyncio debug mode and catch callbacks holding the loop ≥100 ms.

---

## Code Style & Standards

- **Target:** Python 3.14+ (`target-version = "py314"` in Ruff).
- **Formatting & Linting:** 88-char line length. Use standard hyphens (`-`) for ranges; em dashes (`—`) are permitted in prose and comments.
- **Type Annotations:** Checked with Pyright (`typeCheckingMode = "basic"` over `meeko/`). All functions and methods in `meeko/` should have type annotations. Tests use duck typing and fakes.
- **Async Hygiene:** Never perform blocking I/O on the asyncio event loop. SQLite queries run in worker threads (`SessionStore._run`), and hardware streams run in dedicated threads.

---

## Git Workflow & Pre-PR Verification

- Always implement features on a new branch: `<github-username>/<short-description>`.
- Commit discrete, working changes.
- **Git hooks:** Running `uv run pre-commit install` installs both `pre-commit` (Ruff lint & format) and `pre-push` (Pyright, complexity gate, pytest) hooks.
- **Verification before pushing / opening PR:** Run the standalone checks locally to ensure clean CI:
  ```bash
  uv run ruff check .
  uv run ruff format --check .
  uv run pyright
  ./scripts/check_complexity.sh
  uv run pytest
  ```
- If changes touch hardware-dependent code (`leds.py`, `audio_io.py`, `speaker.py`), document physical Raspberry Pi test observations in the PR.

---

## Complexity Gate

`scripts/check_complexity.sh` (xenon) runs at pre-push and in CI:
- **Ceiling:** Fails on any block in `meeko/` with cyclomatic complexity above 20 (radon rank D or worse).
- **Ratchet:** Fails if package average complexity exceeds `AVERAGE` in `scripts/check_complexity.sh`.
- Re-pin `AVERAGE` in the script downward as the package mean improves. Do not loosen to Xenon's letter-grade `--max-average A` (which allows up to CC 5.0).
- Prefer extracting cohesive helper functions over arbitrary splitting or inlining.

Diagnostic inspection (if the gate fails or to check scores before re-pinning):
```bash
uv run radon cc -s -n C meeko/             # List blocks ranked C or worse
uv run radon cc meeko --total-average -n F   # Print package average
```
