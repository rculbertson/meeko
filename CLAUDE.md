# Meeko — Claude Code Instructions

## Project Overview

Meeko is a personal voice assistant for macOS and Raspberry Pi 5. It is designed for long, deep brainstorming conversations — not command-and-control. The full architecture and design rationale are in `meeko-design.md`. Read it before making significant changes.

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
- Everything else (audio, wake word, LEDs, logging, DB path, compaction threshold) lives in `meeko.toml` under `[system]`, `[audio]`, `[wake_word]`, `[claude]` tables. See the commented examples at the top of `meeko.toml`.
- Any `MEEKO_*` environment variable overrides the corresponding TOML value at runtime — useful for one-off testing without editing the file.
- Loading happens once in `run()` via `meeko.config.load_config()`; components receive their values via constructor kwargs (no module-level `os.environ.get` reads).

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
- Tools: `end_session`, `new_session`, `list_sessions(query)`, `load_session(id)`. Per-profile system prompts instruct Sonnet to acknowledge verbally before calling them.
- Handlers only set flags on `SessionManager`; the orchestrator drains TTS first, then performs the actual session transition after SPEAKING ends (see the post-turn block in `meeko/main.py`).
- `load_session` semantics: orchestrator finalizes (summarizes) the abandoned session in the background, then swaps the in-memory message array to the loaded transcript and rebinds `ClaudeClient` to the loaded session's SQLite row.
- End-of-session summarization runs on Sonnet via `meeko/session_summary.py`, fire-and-forget. The summary + transcript are written to the standalone FTS5 `sessions_fts` table for keyword recall.

### Audio
- Read from the **left channel** of the ReSpeaker XVF3800 USB device — this is the AEC-processed output
- Speaker must route through the XVF3800's **3.5mm jack**, not the Pi's audio output (required for hardware AEC)
- Barge-in: `SpeechStarted` during SPEAKING state → stop TTS, cancel Claude request, transition to LISTENING
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
- First run downloads openWakeWord's preprocessor ONNX files (~3 MB). Run `uv run python -m meeko.wake_word` to pre-populate the cache on a network-connected host before deploying offline (e.g. Pi image bake).

### LEDs
- The XVF3800's WS2812 ring is driven by `meeko/leds.py` to mirror the state machine: IDLE off, LISTENING solid cyan, LISTENING_ACTIVE DoA (cyan indicator on darker cyan) while the user is speaking, PROCESSING blue breath, SPEAKING solid green. Errors get a ~3s red breath.
- We talk to the chip directly over libusb (pyusb vendor control transfers on resid 20) — not via ReSpeaker's `xvf_host.py`.
- Linux defaults the USB control interface to root-only. Install `scripts/99-meeko-xvf3800.rules` once (see README) so the `plugdev` group can drive it.
- `[system] led_disabled = true` (env: `MEEKO_LED_DISABLED=1`) — skip LED control entirely. Also auto-disabled if the XVF3800 isn't found or pyusb/libusb is unavailable, so Mac dev runs need no special handling.

### State machine
States: `IDLE → (wake word) → LISTENING → PROCESSING → SPEAKING → (barge-in back to LISTENING)`. The wake-word gate is one-shot per session — follow-up turns do not require re-wakeing.

### Debugging
- Run with `PYTHONASYNCIODEBUG=1` (Python's built-in env var) to enable asyncio debug mode. The loop will then log a WARNING (`Executing <Handle ...> took N.NNN seconds`) whenever a synchronous callback holds it ≥100 ms — useful for diagnosing loop stalls (e.g. STT websocket keepalive timeouts). Off in normal operation; debug mode wraps every coroutine creation with traceback capture and is a real cost on hot paths. Meeko routes asyncio's own warnings through the same logging handler as `meeko.*` logs, so they pick up the timestamp format and land in the rotating log when `MEEKO_LOG_TARGET=file`.

## What's Out of Scope (v1)

Do not add: web/mobile UI, multi-user support, semantic search over sessions, session deletion by voice, cross-device sync. See `meeko-design.md` §9.

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