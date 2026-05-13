"""Tests for `MeekoConfig` loading and env-var override precedence."""

import os
import textwrap
from pathlib import Path

import pytest

from meeko.config import MeekoConfig, load_config


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
    assert cfg.db_path.parent.name == ".meeko"
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


@pytest.mark.parametrize("val", ["", "0", "no", "false", "off"])
def test_bool_env_falsy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, val: str):
    monkeypatch.setenv("MEEKO_LED_DISABLED", val)
    cfg = load_config(tmp_path / "missing.toml")
    assert cfg.led_disabled is False


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


def test_web_search_disabled_env_overrides_toml_true(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """MEEKO_WEB_SEARCH_DISABLED=1 is a one-way override: it forces
    web_search off even if the TOML opts in."""
    toml_file = tmp_path / "meeko.toml"
    toml_file.write_text("[claude]\nweb_search_enabled = true\n")
    monkeypatch.setenv("MEEKO_WEB_SEARCH_DISABLED", "1")
    cfg = load_config(toml_file)
    assert cfg.web_search_enabled is False
