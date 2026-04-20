# Meeko — Claude Code Instructions

## Project Overview

Meeko is a personal voice assistant for macOS and Raspberry Pi 5. It is designed for long, deep brainstorming conversations — not command-and-control. The full architecture and design rationale are in `meeko-design.md`. Read it before making significant changes.

## Current State

The prototype uses **Deepgram STT (Flux, v2 live)** and **Deepgram TTS (Aura-2, REST)** directly, with **Claude called directly via the Anthropic SDK** (`claude-sonnet-4-6`). Timers and profile switching are wired through a Claude-native tool-use loop.

## Primary Goal

The next milestones, in rough order, are: SQLite turn persistence → prompt caching + auto compaction → intent detection (keyword + Haiku) → session resume → barge-in → ReSpeaker/sounddevice. See `private/meeko-design.md` for the full target architecture.

## Tech Stack

- **Python 3.14+**, async/await throughout
- **Deepgram SDK** — STT (Flux model) and TTS (Aura-2) only; do not use Deepgram's Voice Agent API or managed LLM
- **Anthropic SDK** — direct Claude API calls
- **SQLite** — session and transcript storage
- **sounddevice** — audio I/O from ReSpeaker XVF3800

## Key Conventions

### Claude API calls
- Model: `claude-sonnet-4-6` for main conversation
- Model: `claude-haiku-4-5` for intent classification and session summarization
- Always apply `cache_control` on conversation history — cache hits are ~90% after turn 1
- Enable auto compaction via the Anthropic SDK; it operates on the in-memory message array only
- The on-disk SQLite transcript is the source of truth — auto compaction must never cause data loss

### Turn persistence
- Write every turn to SQLite immediately on completion, not buffered, not at session end
- This is load-bearing: auto compaction replaces early turns in memory; SQLite is the only full record

### Intent detection
- Two-stage: keyword filter first (instant, no API call), then Haiku classification only if keywords matched
- The intent detector must be abstracted behind an interface — a local model (Gemma3 via Ollama) may replace Haiku later
- Supported intents: `end_session`, `resume_session`, `new_session`, `list_sessions`, `none`

### Audio
- Read from the **left channel** of the ReSpeaker XVF3800 USB device — this is the AEC-processed output
- Speaker must route through the XVF3800's **3.5mm jack**, not the Pi's audio output (required for hardware AEC)
- Barge-in: `SpeechStarted` during SPEAKING state → stop TTS, cancel Claude request, transition to LISTENING

### State machine
States: `IDLE → LISTENING → PROCESSING → SPEAKING → (barge-in back to LISTENING)`

## What's Out of Scope (v1)

Do not add: wake word detection, web/mobile UI, multi-user support, semantic search over sessions, session deletion by voice, cross-device sync. See `meeko-design.md` §9.

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