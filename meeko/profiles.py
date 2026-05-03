"""Profile loading and management.

A profile defines a persona for the voice assistant: wake word, system
prompt, voice, and conversation mode (query / conversation). Profiles
are stored in a TOML config file.
"""

import tomllib
from dataclasses import dataclass
from pathlib import Path

VALID_MODES = {"query", "conversation"}


@dataclass(frozen=True)
class Profile:
    name: str
    wake_word: str
    prompt: str
    voice: str | None = None
    # Mode controls post-turn idle behavior (see meeko/main.py _idle_monitor).
    # "query" auto-closes silently after `idle_timeout_seconds` of silence.
    # "conversation" prompts after `conversation_idle_seconds` then closes
    # after `conversation_close_seconds` more silence (Stage 2).
    mode: str = "query"
    idle_timeout_seconds: float = 8.0
    conversation_idle_seconds: float = 120.0
    conversation_close_seconds: float = 30.0


def load_profiles(path: str | Path = "profiles.toml") -> dict[str, Profile]:
    """Load profiles from a TOML config file.

    Returns a dict keyed by profile name. Raises ValueError if no 'default'
    profile is defined or if any profile has an invalid `mode`.
    """
    with open(path, "rb") as f:
        data = tomllib.load(f)

    raw_profiles = data.get("profiles", {})
    if not raw_profiles:
        raise ValueError(f"No profiles defined in {path}")

    profiles = {}
    for name, fields in raw_profiles.items():
        mode = fields.get("mode", "query")
        if mode not in VALID_MODES:
            raise ValueError(
                f"Profile {name!r} has invalid mode={mode!r}; "
                f"must be one of {sorted(VALID_MODES)}"
            )
        profiles[name] = Profile(
            name=name,
            wake_word=fields["wake_word"],
            prompt=fields["prompt"],
            voice=fields.get("voice"),
            mode=mode,
            idle_timeout_seconds=float(fields.get("idle_timeout_seconds", 8.0)),
            conversation_idle_seconds=float(
                fields.get("conversation_idle_seconds", 120.0)
            ),
            conversation_close_seconds=float(
                fields.get("conversation_close_seconds", 30.0)
            ),
        )

    if "default" not in profiles:
        raise ValueError(
            f"No 'default' profile in {path}. Available: {', '.join(profiles)}"
        )

    return profiles
