# Contributing to Meeko

Thanks for your interest in contributing. See [docs/architecture.md](docs/architecture.md) for the design rationale and a tour of the components before making non-trivial changes.

## Before you start

Meeko is maintained in spare time, and reviews happen in batches — expect weeks, not days. For anything beyond a small fix, **open an issue before writing code**. It costs you a minute and saves you from building something that gets declined.

## Scope

Meeko is deliberately a voice-only, single-user, single-device assistant. These are out of scope and PRs adding them will most likely be declined:

- Web or mobile UI
- Multi-user support
- Semantic search over sessions (SQLite FTS5 is sufficient at expected volumes)
- Session deletion or editing by voice
- Cross-device sync

See [docs/architecture.md §9](docs/architecture.md#9-deliberate-scope) for the reasoning. If you think real-world usage justifies revisiting one of these, open an issue and make the case — the list isn't permanent, it just isn't v1.

Windows is also unsupported; the code targets macOS and Raspberry Pi OS.

## Dev setup

First install PortAudio, which `pyaudio` compiles against — `brew install portaudio` on macOS, `sudo apt install portaudio19-dev` on Debian/Ubuntu/Raspberry Pi OS. Without it `uv sync` fails while building `pyaudio`.

```bash
uv sync --all-groups        # installs Python 3.14 if needed, plus dev deps (pytest, ruff, pre-commit)
uv run pre-commit install   # installs both the pre-commit and pre-push hooks
```

`uv` provisions Python itself from the tracked `.python-version`, so no separate version manager is required. If you already use [mise](https://mise.jdx.dev/), `mise install` reads the same file.

## Tests

Tests live in `tests/` and use **pytest** + **pytest-asyncio**. Anything that requires live API keys (Deepgram, Anthropic) is marked `@pytest.mark.integration`; everything else runs offline.

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

## Linting and type checking

```bash
uv run ruff check .
uv run ruff format .           # format files in-place
uv run ruff format --check .   # check formatting (runs in CI)
uv run pyright
./scripts/check_complexity.sh
```

Ruff runs on commit, and pyright (basic mode, over `meeko/`), the complexity gate and pytest run on push, via the hooks installed above; the same checks run in CI.

### Complexity

`scripts/check_complexity.sh` wraps [xenon](https://github.com/rubik/xenon) and holds the thresholds, so CI and the pre-push hook can't drift apart. It fails if any block in `meeko/` exceeds cyclomatic complexity 20 (radon rank D or worse), or if the package average exceeds **3.1** (currently 2.82 over 289 blocks).

Two different jobs. The **absolute** ceiling is deliberately loose, because cyclomatic complexity can't tell a flat dispatch ladder from a branch buried in a hot loop and scores the two identically — so it only catches unambiguous runaway, and judgment calls are left to review. The **average** is the ratchet, and it has to be numeric to be worth anything: xenon's letter-grade `--max-average A` runs all the way to CC 5.0, which would let 42 more CC-20 functions through while the build stayed green. Re-pin it downward as the real average improves.

Neither gate catches everything, and it's worth knowing the hole: because the average is a per-block mean, inlining two small helpers back into their caller can leave both checks green while the code gets worse. Complexity gates catch decay in bulk; they don't replace review.

`tests/` is excluded. Test functions are linear setup-then-assert and score badly for reasons that don't indicate a maintenance problem.

To see where a change landed rather than just whether it passed:

```bash
uv run radon cc -s -n C meeko/            # blocks ranked C or worse (silent when clean)
uv run radon cc meeko --total-average -n F  # just the package average
```

## Git workflow

- Implement features on a new branch, never directly on `main`.
- Branch naming: `<github-username>/<short-description>`.
- Commit when a discrete, working piece is complete; each commit should run correctly on its own.
- Push your branch to GitHub (or your fork) and open a pull request (or run `gh pr create`) — ruff runs on commit, and the test suite, pyright and the complexity gate run on push and in CI.

## AI-assisted contributions

Meeko's repo-specific conventions for AI-assisted work (how the config layer is wired, what's load-bearing, which invariants not to break) are documented in [CLAUDE.md](CLAUDE.md). Treat it as the canonical guide when using Claude Code or similar tools in this repo.

This repo tracks a `.claude/settings.json`, and trusting the workspace applies all of it. Worth knowing what that grants:

- **A `permissions.allow` list** that pre-authorizes some tool calls without prompting: `WebSearch`, `WebFetch` to `api.deepgram.com`, and `Bash` for `uv`, `cat`, `head`, `tail`, `grep`, `awk`, `sed`, `wc`. Note `Bash(uv *)` covers any `uv` subcommand and `Bash(sed *)` can write files. Fine for this repo's workflow, but it is a real grant — read it before accepting.
- **Deepgram's [skills marketplace](https://github.com/deepgram/skills)** and its plugin, which supply the Deepgram API reference used throughout `meeko/deepgram_stt.py` and `meeko/deepgram_tts.py`. The plugin ships documentation skills only — no hooks or commands that execute — and `skillOverrides` turns off the two skills that would run setup actions.

Claude Code asks you to trust the workspace before any of this loads, so nothing applies behind your back. To opt out, decline the trust prompt, or manage plugins with `/plugin`.

## Code of Conduct

Please review and follow our [Code of Conduct](CODE_OF_CONDUCT.md) in all project spaces.

## License

Meeko is MIT licensed. By contributing, you agree that your contributions are licensed under the same terms.
