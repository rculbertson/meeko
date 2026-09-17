# Contributing to Meeko

Thanks for your interest in contributing. See [ARCHITECTURE.md](ARCHITECTURE.md) for the design rationale and a tour of the components before making non-trivial changes.

## Before you start

Meeko is maintained in spare time, and reviews happen in batches — expect weeks, not days. For anything beyond a small fix, **open an issue before writing code**. It costs you a minute and saves you from building something that gets declined.

## Scope

Meeko is deliberately a voice-only, single-user, single-device assistant. These are out of scope and PRs adding them will most likely be declined:

- Web or mobile UI
- Multi-user support
- Semantic search over sessions (SQLite FTS5 is sufficient at expected volumes)
- Session deletion or editing by voice
- Cross-device sync

See [ARCHITECTURE.md](ARCHITECTURE.md) §9 for the reasoning. If you think real-world usage justifies revisiting one of these, open an issue and make the case — the list isn't permanent, it just isn't v1.

Windows is also unsupported; the code targets macOS and Raspberry Pi OS.

## Dev setup

```bash
mise install            # installs Python 3.14 via .python-version
uv sync --all-groups    # includes dev dependencies (pytest, ruff, pre-commit)
pre-commit install
```

## Tests

Tests live in `tests/` and use **pytest** + **pytest-asyncio**. Anything that requires live API keys (Deepgram, Anthropic) is marked `@pytest.mark.integration`; everything else runs offline.

Generate the WAV test fixtures once (requires `DEEPGRAM_API_KEY`):

```bash
uv run python tests/generate_fixtures.py
```

Run the unit tests (with coverage report):

```bash
uv run pytest
```

Run the integration tests (calls external APIs):

```bash
uv run pytest -m integration
```

Skip coverage for a faster run:

```bash
uv run pytest --no-cov
```

Any non-trivial change — new behavior, bug fix, or refactor of existing logic — should include updated or new tests. Trivial changes (docstrings, comments, config tweaks, logging) do not require tests.

### Hardware-dependent code

Neither CI nor the maintainer can validate every path. Changes to `meeko/leds.py`, `meeko/audio_io.py`, or `meeko/speaker.py` need real hardware — a Raspberry Pi 5 with a ReSpeaker XVF3800 for the LED and AEC paths. If your PR touches these, say in the description what you tested on and what you observed. A PR in these files with no hardware report can sit indefinitely, because there may be nobody able to confirm it works.

## Linting

```bash
uv run ruff check .
uv run ruff format .
```

Both run automatically via pre-commit hooks; the same checks run in CI.

## Git workflow

- Implement features on a new branch, never directly on `main`.
- Branch naming: `<github-username>/<short-description>`.
- Commit when a discrete, working piece is complete; each commit should run correctly on its own.
- Before opening a PR, commit outstanding changes and run `gh pr create` — ruff and tests run automatically via pre-commit hooks.

## AI-assisted contributions

Meeko's repo-specific conventions for AI-assisted work (when to read the design doc, how the config layer is wired, what's load-bearing) are documented in [CLAUDE.md](CLAUDE.md). Treat it as the canonical guide when using Claude Code or similar tools in this repo.

This repo's tracked `.claude/settings.json` also registers Deepgram's [skills marketplace](https://github.com/deepgram/skills) and enables its plugin, which gives Claude Code the Deepgram API reference used throughout `meeko/deepgram_*.py`. Claude Code asks you to trust the workspace before any of it loads, so nothing is enabled behind your back, and the plugin ships documentation skills only — no hooks or commands that execute. To opt out, decline the trust prompt, or manage it with `/plugin` inside Claude Code.
