import textwrap
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from meeko.profiles import Profile, load_profiles
from meeko.tools.profile import ProfileManager, get_tool_definitions

# ---------------------------------------------------------------------------
# load_profiles tests
# ---------------------------------------------------------------------------


def test_load_profiles_default(tmp_path: Path):
    """Loading a TOML with just a default profile works."""
    toml_file = tmp_path / "profiles.toml"
    toml_file.write_text(
        textwrap.dedent("""\
        [profiles.default]
        wake_word = "meeko"
        prompt = "You are Meeko."
    """)
    )

    profiles = load_profiles(toml_file)
    assert "default" in profiles
    assert profiles["default"].name == "default"
    assert profiles["default"].wake_word == "meeko"
    assert profiles["default"].prompt == "You are Meeko."


def test_load_profiles_multiple(tmp_path: Path):
    """Loading a TOML with multiple profiles returns all of them."""
    toml_file = tmp_path / "profiles.toml"
    toml_file.write_text(
        textwrap.dedent("""\
        [profiles.default]
        wake_word = "meeko"
        prompt = "You are Meeko."

        [profiles.pirate]
        wake_word = "ahoy"
        prompt = "You are a pirate."
    """)
    )

    profiles = load_profiles(toml_file)
    assert len(profiles) == 2
    assert "default" in profiles
    assert "pirate" in profiles
    assert profiles["pirate"].wake_word == "ahoy"


def test_load_profiles_missing_default(tmp_path: Path):
    """Loading a TOML without a default profile raises ValueError."""
    toml_file = tmp_path / "profiles.toml"
    toml_file.write_text(
        textwrap.dedent("""\
        [profiles.pirate]
        wake_word = "ahoy"
        prompt = "You are a pirate."
    """)
    )

    with pytest.raises(ValueError, match="No 'default' profile"):
        load_profiles(toml_file)


def test_load_profiles_empty(tmp_path: Path):
    """Loading a TOML with no profiles section raises ValueError."""
    toml_file = tmp_path / "profiles.toml"
    toml_file.write_text("")

    with pytest.raises(ValueError, match="No profiles defined"):
        load_profiles(toml_file)


# ---------------------------------------------------------------------------
# ProfileManager tests
# ---------------------------------------------------------------------------


def _make_profiles():
    return {
        "default": Profile("default", "meeko", "You are Meeko."),
        "pirate": Profile("pirate", "ahoy", "You are a pirate."),
    }


async def test_switch_profile():
    """Switching to a valid profile updates the Claude system prompt."""
    profiles = _make_profiles()
    claude = MagicMock()
    mgr = ProfileManager(profiles, claude_client=claude)

    result = await mgr.switch_profile("pirate")

    assert result == "Switched to pirate mode."
    claude.set_system_prompt.assert_called_once_with("You are a pirate.")
    assert mgr.active_profile.name == "pirate"


async def test_switch_profile_unknown():
    profiles = _make_profiles()
    mgr = ProfileManager(profiles, claude_client=MagicMock())

    result = await mgr.switch_profile("nonexistent")

    assert "Unknown profile" in result
    assert "default" in result
    assert "pirate" in result
    assert mgr.active_profile.name == "default"


async def test_switch_profile_already_active():
    profiles = _make_profiles()
    mgr = ProfileManager(profiles, claude_client=MagicMock())

    result = await mgr.switch_profile("default")

    assert "Already using" in result


async def test_list_profiles():
    profiles = _make_profiles()
    mgr = ProfileManager(profiles)

    result = mgr.list_profiles()

    assert "default (active)" in result
    assert "pirate" in result


async def test_list_profiles_after_switch():
    profiles = _make_profiles()
    claude = MagicMock()
    mgr = ProfileManager(profiles, claude_client=claude)

    await mgr.switch_profile("pirate")
    result = mgr.list_profiles()

    assert "pirate (active)" in result
    assert "default" in result
    assert "(active)" not in result.split("pirate (active)")[0].split("default")[-1]


# ---------------------------------------------------------------------------
# Tool definitions tests
# ---------------------------------------------------------------------------


def test_tool_definitions_include_profile_names():
    """Tool definitions include available profile names."""
    profiles = _make_profiles()
    defs = get_tool_definitions(profiles)

    names = [d["name"] for d in defs]
    assert "switch_profile" in names
    assert "list_profiles" in names

    switch_def = next(d for d in defs if d["name"] == "switch_profile")
    assert "default" in switch_def["description"]
    assert "pirate" in switch_def["description"]
