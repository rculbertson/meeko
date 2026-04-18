"""Profile loading and management.

A profile defines a persona for the voice assistant: wake word, system prompt,
and greeting. Profiles are stored in a TOML config file.
"""

import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Profile:
    name: str
    wake_word: str
    prompt: str
    greeting: str
    voice: str | None = None


def load_profiles(path: str | Path = "profiles.toml") -> dict[str, Profile]:
    """Load profiles from a TOML config file.

    Returns a dict keyed by profile name. Raises ValueError if no 'default'
    profile is defined.
    """
    with open(path, "rb") as f:
        data = tomllib.load(f)

    raw_profiles = data.get("profiles", {})
    if not raw_profiles:
        raise ValueError(f"No profiles defined in {path}")

    profiles = {}
    for name, fields in raw_profiles.items():
        profiles[name] = Profile(
            name=name,
            wake_word=fields["wake_word"],
            prompt=fields["prompt"],
            greeting=fields["greeting"],
            voice=fields.get("voice"),
        )

    if "default" not in profiles:
        raise ValueError(
            f"No 'default' profile in {path}. Available: {', '.join(profiles)}"
        )

    return profiles
