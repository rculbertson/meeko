# Contributing to Meeko

Thanks for your interest in contributing. See [ARCHITECTURE.md](ARCHITECTURE.md) for the design rationale and a tour of the components before making non-trivial changes.

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
