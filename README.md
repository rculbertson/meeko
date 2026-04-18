# Meeko

A self-hosted voice assistant built on Deepgram and Claude.

## Status

Milestone 1 in progress — core Voice Agent loop (no wake word yet).

---

## For Users

### Prerequisites

- Python 3.14+
- [uv](https://docs.astral.sh/uv/)
- Homebrew: `brew install portaudio`
- Deepgram API key (free tier includes $200 credit)
- Anthropic API key

### Setup

```bash
uv sync             # installs dependencies
```

Create a `.env` file in the project root with your API keys:

```bash
DEEPGRAM_API_KEY=your-deepgram-api-key
ANTHROPIC_API_KEY=your-anthropic-api-key
```

### Running

```bash
uv run python -m meeko.main
```

Speak to the assistant. Press `Ctrl+C` to quit.

### Logging

Two environment variables control logging behavior.

**`MEEKO_LOG_LEVEL`** — verbosity (default: `DEBUG`)

| Value | What you see |
|---|---|
| `DEBUG` | All messages including mic chunk counts, raw WebSocket messages |
| `INFO` | Conversation text, connection events, mic mute/unmute |
| `WARNING` / `ERROR` | Errors only |

**`MEEKO_LOG_TARGET`** — destination (default: stderr)

| Value | Behavior |
|---|---|
| _(unset)_ | Logs to stderr |
| `file` | Logs to `meeko.log` (rotating, 5 MB × 3 files) |

Example — quiet console output:

```bash
MEEKO_LOG_LEVEL=INFO uv run python -m meeko.main
```

Example — log to file at debug level:

```bash
MEEKO_LOG_TARGET=file uv run python -m meeko.main
```

---

## For Developers

### Dev Setup

```bash
mise install           # installs Python 3.14 via .python-version
uv sync --all-groups   # includes dev dependencies (pytest, ruff, pre-commit)
pre-commit install
```

### Tests

Generate the WAV test fixtures (one-time, requires `DEEPGRAM_API_KEY`):

```bash
uv run python tests/generate_fixtures.py
```

Run the unit tests (with coverage report):

```bash
uv run pytest
```

Run the integration tests:

```bash
uv run pytest -m integration    # calls external APIs (Deepgram, Claude)
```

Skip coverage for a faster run:

```bash
uv run pytest --no-cov
```

### Linting

```bash
uv run ruff check .
uv run ruff format .
```
