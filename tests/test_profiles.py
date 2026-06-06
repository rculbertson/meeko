import textwrap
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from meeko.config import Profile, load_profiles
from meeko.tools.profile import ProfileManager, get_tool_definitions

# ---------------------------------------------------------------------------
# load_profiles tests
# ---------------------------------------------------------------------------


def test_load_profiles_single(tmp_path: Path):
    """Loading a TOML with a single profile + default_profile works."""
    toml_file = tmp_path / "profiles.toml"
    toml_file.write_text(
        textwrap.dedent("""\
        default_profile = "query"

        [profiles.query]
        wake_word = "meeko"
        prompt = "You are Meeko."
    """)
    )

    profiles, default_name = load_profiles(toml_file)
    assert default_name == "query"
    assert "query" in profiles
    assert profiles["query"].name == "query"
    assert profiles["query"].wake_word == "meeko"
    assert profiles["query"].prompt == "You are Meeko."


def test_load_profiles_multiple(tmp_path: Path):
    """Loading a TOML with multiple profiles returns all of them."""
    toml_file = tmp_path / "profiles.toml"
    toml_file.write_text(
        textwrap.dedent("""\
        default_profile = "query"

        [profiles.query]
        wake_word = "meeko"
        prompt = "You are Meeko."

        [profiles.conversation]
        wake_word = "meeko"
        prompt = "You are a thinking partner."
    """)
    )

    profiles, default_name = load_profiles(toml_file)
    assert len(profiles) == 2
    assert default_name == "query"
    assert "query" in profiles
    assert "conversation" in profiles


def test_load_profiles_missing_default_profile_key(tmp_path: Path):
    """Loading a TOML without a top-level default_profile raises ValueError."""
    toml_file = tmp_path / "profiles.toml"
    toml_file.write_text(
        textwrap.dedent("""\
        [profiles.query]
        wake_word = "meeko"
        prompt = "You are Meeko."
    """)
    )

    with pytest.raises(ValueError, match="Missing top-level 'default_profile'"):
        load_profiles(toml_file)


def test_load_profiles_default_profile_unknown(tmp_path: Path):
    """default_profile pointing at an undefined profile raises ValueError."""
    toml_file = tmp_path / "profiles.toml"
    toml_file.write_text(
        textwrap.dedent("""\
        default_profile = "nope"

        [profiles.query]
        wake_word = "meeko"
        prompt = "You are Meeko."
    """)
    )

    with pytest.raises(ValueError, match="default_profile='nope'"):
        load_profiles(toml_file)


def test_load_profiles_invalid_name(tmp_path: Path):
    """A profile whose name is not a valid mode raises ValueError."""
    toml_file = tmp_path / "profiles.toml"
    toml_file.write_text(
        textwrap.dedent("""\
        default_profile = "pirate"

        [profiles.pirate]
        wake_word = "ahoy"
        prompt = "You are a pirate."
    """)
    )

    with pytest.raises(ValueError, match="not a valid mode"):
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
        "query": Profile("query", "meeko", "You are Meeko."),
        "conversation": Profile("conversation", "meeko", "You are a thinking partner."),
    }


async def test_switch_profile():
    """Switching to a valid profile updates the Claude system prompt."""
    profiles = _make_profiles()
    claude = MagicMock()
    mgr = ProfileManager(profiles, claude_client=claude, active_name="query")

    result = await mgr.switch_profile("conversation")

    assert result == "Switched to conversation mode."
    claude.set_system_prompt.assert_called_once_with("You are a thinking partner.")
    assert mgr.active_profile.name == "conversation"


async def test_switch_profile_persists_to_store():
    """A mid-session switch must be written back to the live session row,
    so a later resume/load restores the switched mode rather than the
    row's creation-time mode (the query/conversation idle-timeout bug)."""
    profiles = _make_profiles()
    claude = MagicMock()
    claude.session_id = "sess-1"
    store = MagicMock()
    store.set_profile_name = AsyncMock()
    mgr = ProfileManager(profiles, claude_client=claude, active_name="query")
    mgr.set_store(store)

    await mgr.switch_profile("conversation")

    store.set_profile_name.assert_awaited_once_with("sess-1", "conversation")


async def test_switch_profile_skips_persist_when_no_live_row():
    """Lazy session creation: no row exists yet (session_id is None), so
    there's nothing to update — the row inherits the active profile when
    it's eventually created."""
    profiles = _make_profiles()
    claude = MagicMock()
    claude.session_id = None
    store = MagicMock()
    store.set_profile_name = AsyncMock()
    mgr = ProfileManager(profiles, claude_client=claude, active_name="query")
    mgr.set_store(store)

    await mgr.switch_profile("conversation")

    store.set_profile_name.assert_not_called()


async def test_switch_profile_unknown():
    profiles = _make_profiles()
    mgr = ProfileManager(profiles, claude_client=MagicMock(), active_name="query")

    result = await mgr.switch_profile("nonexistent")

    assert "Unknown profile" in result
    assert "query" in result
    assert "conversation" in result
    assert mgr.active_profile.name == "query"


async def test_switch_profile_already_active():
    profiles = _make_profiles()
    mgr = ProfileManager(profiles, claude_client=MagicMock(), active_name="query")

    result = await mgr.switch_profile("query")

    assert "Already using" in result


async def test_list_profiles():
    profiles = _make_profiles()
    mgr = ProfileManager(profiles, active_name="query")

    result = mgr.list_profiles()

    assert "query (active)" in result
    assert "conversation" in result


async def test_list_profiles_after_switch():
    profiles = _make_profiles()
    claude = MagicMock()
    mgr = ProfileManager(profiles, claude_client=claude, active_name="query")

    await mgr.switch_profile("conversation")
    result = mgr.list_profiles()

    assert "conversation (active)" in result
    assert "query" in result
    assert "(active)" not in result.split("conversation (active)")[0].split("query")[-1]


def test_profile_manager_rejects_unknown_active_name():
    profiles = _make_profiles()
    with pytest.raises(ValueError, match="not a defined profile"):
        ProfileManager(profiles, active_name="nope")


async def test_switch_profile_updates_speaker_voice():
    """switch_profile must also update the Speaker so the next TTS call
    uses the new profile's voice (sibling latent bug to #50 — fixed in
    the same change because both rely on the same rebind plumbing)."""
    profiles = _make_profiles()
    claude = MagicMock()
    speaker = MagicMock()
    mgr = ProfileManager(profiles, claude_client=claude, active_name="query")
    mgr.set_speaker(speaker)

    await mgr.switch_profile("conversation")

    speaker.set_profile.assert_called_once_with(profiles["conversation"])


def test_rebind_profile_updates_claude_and_speaker():
    """Rebinding to a different profile updates Claude's system prompt
    and Speaker's profile so subsequent turns match the loaded session."""
    profiles = _make_profiles()
    claude = MagicMock()
    speaker = MagicMock()
    mgr = ProfileManager(profiles, claude_client=claude, active_name="query")
    mgr.set_speaker(speaker)

    returned = mgr.rebind_profile("conversation")

    assert returned.name == "conversation"
    assert mgr.active_profile.name == "conversation"
    claude.set_system_prompt.assert_called_once_with("You are a thinking partner.")
    speaker.set_profile.assert_called_once_with(profiles["conversation"])


def test_rebind_profile_no_op_when_already_active():
    profiles = _make_profiles()
    claude = MagicMock()
    speaker = MagicMock()
    mgr = ProfileManager(profiles, claude_client=claude, active_name="query")
    mgr.set_speaker(speaker)

    returned = mgr.rebind_profile("query")

    assert returned.name == "query"
    claude.set_system_prompt.assert_not_called()
    speaker.set_profile.assert_not_called()


def test_rebind_profile_unknown_falls_back_to_active(caplog):
    """A loaded session may reference a profile that's since been
    removed from meeko.toml; keep the current active profile and log."""
    profiles = _make_profiles()
    claude = MagicMock()
    speaker = MagicMock()
    mgr = ProfileManager(profiles, claude_client=claude, active_name="query")
    mgr.set_speaker(speaker)

    with caplog.at_level("WARNING", logger="meeko"):
        returned = mgr.rebind_profile("retired-profile")

    assert returned.name == "query"
    assert mgr.active_profile.name == "query"
    claude.set_system_prompt.assert_not_called()
    speaker.set_profile.assert_not_called()
    assert any("retired-profile" in r.getMessage() for r in caplog.records)


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
    assert "query" in switch_def["description"]
    assert "conversation" in switch_def["description"]
