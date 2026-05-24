"""Tests for `MeekoConfig` loading and env-var override precedence."""

import os
import textwrap
from pathlib import Path

import pytest

from meeko.config import (
    MeekoConfig,
    default_config_path,
    ensure_config_exists,
    load_config,
    load_profiles,
    resolve_config_path,
)


@pytest.fixture(autouse=True)
def _clear_meeko_env(monkeypatch):
    """Strip every MEEKO_* env var before each test so .env or shell
    settings can't bleed into precedence checks."""
    for key in [k for k in os.environ if k.startswith("MEEKO_")]:
        monkeypatch.delenv(key, raising=False)


def test_defaults_match_historical_env_defaults(tmp_path: Path):
    """With no TOML file and no env vars set, MeekoConfig matches the
    documented defaults (regression guard against accidental drift)."""
    cfg = load_config(tmp_path / "missing.toml")

    assert cfg.log_level == "DEBUG"
    assert cfg.log_target is None
    assert cfg.db_path.name == "meeko.db"
    assert cfg.db_path.parent.name == "meeko"
    assert cfg.led_disabled is False
    assert cfg.input_channels == 2
    assert cfg.output_channels == 2
    assert cfg.input_device_index is None
    assert cfg.output_device_index is None
    assert cfg.mute_mic_while_speaking is False
    assert cfg.wake_word_model == Path("models/hey_meeko.onnx")
    assert cfg.wake_word_threshold == 0.96
    assert cfg.wake_word_disabled is False
    assert cfg.compaction_trigger_tokens == 150000
    assert cfg.web_search_enabled is True
    assert cfg.web_search_max_uses == 2


def test_loads_values_from_toml(tmp_path: Path):
    toml_file = tmp_path / "meeko.toml"
    toml_file.write_text(
        textwrap.dedent("""\
        [system]
        log_level = "INFO"
        log_target = "file"
        led_disabled = true

        [audio]
        input_channels = 1
        output_device_index = 3
        mute_mic_while_speaking = true

        [wake_word]
        threshold = 0.5
        disabled = true

        [claude]
        compaction_trigger_tokens = 99999
    """)
    )

    cfg = load_config(toml_file)
    assert cfg.log_level == "INFO"
    assert cfg.log_target == "file"
    assert cfg.led_disabled is True
    assert cfg.input_channels == 1
    assert cfg.output_device_index == 3
    assert cfg.mute_mic_while_speaking is True
    assert cfg.wake_word_threshold == 0.5
    assert cfg.wake_word_disabled is True
    assert cfg.compaction_trigger_tokens == 99999


def test_env_var_overrides_toml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    toml_file = tmp_path / "meeko.toml"
    toml_file.write_text(
        textwrap.dedent("""\
        [audio]
        input_channels = 1
    """)
    )

    monkeypatch.setenv("MEEKO_INPUT_CHANNELS", "4")
    cfg = load_config(toml_file)
    assert cfg.input_channels == 4


def test_env_overrides_when_no_toml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MEEKO_WAKE_WORD_DISABLED", "1")
    monkeypatch.setenv("MEEKO_COMPACTION_TRIGGER_TOKENS", "42")
    cfg = load_config(tmp_path / "missing.toml")
    assert cfg.wake_word_disabled is True
    assert cfg.compaction_trigger_tokens == 42


@pytest.mark.parametrize("val", ["1", "true", "TRUE", "yes", "YES", "True"])
def test_bool_env_truthy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, val: str):
    monkeypatch.setenv("MEEKO_LED_DISABLED", val)
    cfg = load_config(tmp_path / "missing.toml")
    assert cfg.led_disabled is True


@pytest.mark.parametrize("val", ["0", "no", "false", "off"])
def test_bool_env_falsy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, val: str):
    monkeypatch.setenv("MEEKO_LED_DISABLED", val)
    cfg = load_config(tmp_path / "missing.toml")
    assert cfg.led_disabled is False


def test_bool_env_empty_falls_back_to_toml(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Empty (or whitespace-only) env vars are treated as unset, so the
    TOML/default value wins. Matters for default-True flags like
    web_search_enabled where empty=False would silently flip the feature."""
    toml_file = tmp_path / "meeko.toml"
    toml_file.write_text("[claude]\nweb_search_enabled = true\n")
    monkeypatch.setenv("MEEKO_WEB_SEARCH_ENABLED", "")
    cfg = load_config(toml_file)
    assert cfg.web_search_enabled is True


def test_bool_env_invalid_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MEEKO_LED_DISABLED", "maybe")
    with pytest.raises(ValueError, match="MEEKO_LED_DISABLED"):
        load_config(tmp_path / "missing.toml")


def test_db_path_expansion(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("MEEKO_DB_PATH", "~/custom.db")
    cfg = load_config(tmp_path / "missing.toml")
    assert cfg.db_path == tmp_path / "custom.db"


def test_db_path_from_toml_expands_tilde(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("HOME", str(tmp_path))
    toml_file = tmp_path / "meeko.toml"
    toml_file.write_text('[system]\ndb_path = "~/meeko-config.db"\n')
    cfg = load_config(toml_file)
    assert cfg.db_path == tmp_path / "meeko-config.db"


def test_missing_toml_falls_back_to_defaults(tmp_path: Path):
    cfg = load_config(tmp_path / "does-not-exist.toml")
    assert cfg == MeekoConfig()


def test_web_search_enabled_env_overrides_toml_true(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """MEEKO_WEB_SEARCH_ENABLED=0 overrides a TOML opt-in."""
    toml_file = tmp_path / "meeko.toml"
    toml_file.write_text("[claude]\nweb_search_enabled = true\n")
    monkeypatch.setenv("MEEKO_WEB_SEARCH_ENABLED", "0")
    cfg = load_config(toml_file)
    assert cfg.web_search_enabled is False


def test_web_search_enabled_env_overrides_toml_false(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """MEEKO_WEB_SEARCH_ENABLED=1 overrides a TOML opt-out."""
    toml_file = tmp_path / "meeko.toml"
    toml_file.write_text("[claude]\nweb_search_enabled = false\n")
    monkeypatch.setenv("MEEKO_WEB_SEARCH_ENABLED", "1")
    cfg = load_config(toml_file)
    assert cfg.web_search_enabled is True


def test_load_profiles_missing_file_friendly_error(tmp_path: Path):
    """A missing explicit config path should produce a helpful error that
    names the path and points at the README, not a raw FileNotFoundError."""
    missing = tmp_path / "meeko.toml"
    with pytest.raises(FileNotFoundError, match="not found"):
        load_profiles(missing)


# --- XDG config path resolution ---


def test_default_config_path_follows_xdg(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    path = default_config_path()
    assert path.name == "meeko.toml"
    assert path.parent.name == "meeko"
    assert path.parent.parent == Path.home() / ".config"


def test_default_config_path_honors_xdg_config_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    assert default_config_path() == tmp_path / "meeko" / "meeko.toml"


def test_default_config_path_ignores_relative_xdg(monkeypatch: pytest.MonkeyPatch):
    # Per the XDG spec, a relative $XDG_CONFIG_HOME is invalid; fall back.
    monkeypatch.setenv("XDG_CONFIG_HOME", "relative/config")
    assert default_config_path() == Path.home() / ".config" / "meeko" / "meeko.toml"


def test_resolve_config_path_meeko_config_wins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("MEEKO_CONFIG", str(tmp_path / "custom.toml"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    assert resolve_config_path() == tmp_path / "custom.toml"


def test_resolve_config_path_prefers_xdg_when_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.delenv("MEEKO_CONFIG", raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    xdg_file = tmp_path / "meeko" / "meeko.toml"
    xdg_file.parent.mkdir(parents=True)
    xdg_file.write_text('default_profile = "query"\n')
    # Even with a cwd meeko.toml present, XDG takes precedence.
    monkeypatch.chdir(tmp_path)
    (tmp_path / "meeko.toml").write_text("")
    assert resolve_config_path() == xdg_file


def test_resolve_config_path_falls_back_to_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.delenv("MEEKO_CONFIG", raising=False)
    # Point XDG at an empty dir so the XDG file does not exist.
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    monkeypatch.chdir(tmp_path)
    (tmp_path / "meeko.toml").write_text("")
    assert resolve_config_path() == Path("meeko.toml")


def test_resolve_config_path_defaults_to_xdg_when_none_exist(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.delenv("MEEKO_CONFIG", raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    monkeypatch.chdir(tmp_path)  # no cwd meeko.toml here
    assert resolve_config_path() == tmp_path / "xdg" / "meeko" / "meeko.toml"


def test_ensure_config_exists_auto_creates_from_example(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.delenv("MEEKO_CONFIG", raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)  # ensure no cwd meeko.toml interferes

    created = ensure_config_exists()

    assert created == tmp_path / "meeko" / "meeko.toml"
    assert created.exists()
    # The copied example must be a loadable config with profiles defined.
    profiles, default_name = load_profiles(created)
    assert profiles
    assert default_name in profiles


def test_ensure_config_exists_noop_when_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.delenv("MEEKO_CONFIG", raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    xdg_file = tmp_path / "meeko" / "meeko.toml"
    xdg_file.parent.mkdir(parents=True)
    xdg_file.write_text('default_profile = "query"\n')
    monkeypatch.chdir(tmp_path)

    assert ensure_config_exists() == xdg_file
    # Untouched: still our minimal content, not the example.
    assert xdg_file.read_text() == 'default_profile = "query"\n'
