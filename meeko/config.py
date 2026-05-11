"""Meeko configuration loading.

Loads both per-host runtime config (`MeekoConfig`) and the conversation
personas (`Profile`) from a single `meeko.toml` file. Secrets stay in
`.env`; everything else lives in `meeko.toml`. Any `MEEKO_*` environment
variable overrides the corresponding TOML value, so existing env-var
setups keep working.
"""

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

VALID_MODES = {"query", "conversation"}

DEFAULT_CONFIG_PATH = "meeko.toml"


def _env_flag(name: str) -> bool:
    """Parse a MEEKO_* bool env var. True for 1/true/yes (case-insensitive)."""
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes"}


def _default_db_path() -> Path:
    return Path.home() / ".meeko" / "meeko.db"


def _default_wake_word_model() -> Path:
    return Path("models/hey_meeko.onnx")


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
    idle_timeout_seconds: float = 5.0
    conversation_idle_seconds: float = 60.0
    conversation_close_seconds: float = 20.0


@dataclass(frozen=True)
class MeekoConfig:
    """Per-host runtime config. All fields optional; defaults mirror the
    historical env-var defaults so unset == prior behavior."""

    # system
    log_level: str = "DEBUG"
    log_target: str | None = None  # None = stderr; "file" = rotating file
    db_path: Path = field(default_factory=_default_db_path)
    led_disabled: bool = False
    # audio
    input_channels: int = 2
    output_channels: int = 2
    input_device_index: int | None = None
    output_device_index: int | None = None
    mute_mic_while_speaking: bool = False
    # wake word
    wake_word_model: Path = field(default_factory=_default_wake_word_model)
    wake_word_threshold: float = 0.96
    wake_word_disabled: bool = False
    # claude
    compaction_trigger_tokens: int = 150000


def load_profiles(path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Profile]:
    """Load conversation profiles from `meeko.toml`.

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
            idle_timeout_seconds=float(
                fields.get("idle_timeout_seconds", Profile.idle_timeout_seconds)
            ),
            conversation_idle_seconds=float(
                fields.get(
                    "conversation_idle_seconds", Profile.conversation_idle_seconds
                )
            ),
            conversation_close_seconds=float(
                fields.get(
                    "conversation_close_seconds", Profile.conversation_close_seconds
                )
            ),
        )

    if "default" not in profiles:
        raise ValueError(
            f"No 'default' profile in {path}. Available: {', '.join(profiles)}"
        )

    return profiles


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> MeekoConfig:
    """Load `MeekoConfig` from `meeko.toml`, with env-var overrides.

    Precedence: env var > meeko.toml > dataclass default. Missing tables
    and keys silently fall through to defaults; the file may not even
    define any of `[system]`, `[audio]`, `[wake_word]`, `[claude]`.
    """
    try:
        with open(path, "rb") as f:
            data = tomllib.load(f)
    except FileNotFoundError:
        data = {}

    system = data.get("system", {})
    audio = data.get("audio", {})
    wake = data.get("wake_word", {})
    claude = data.get("claude", {})

    defaults = MeekoConfig()

    def _str_env(name: str, fallback: str) -> str:
        return os.environ.get(name, fallback)

    def _opt_str_env(name: str, fallback: str | None) -> str | None:
        v = os.environ.get(name)
        return v if v is not None else fallback

    def _path_env(name: str, fallback: Path) -> Path:
        v = os.environ.get(name)
        return Path(v).expanduser() if v else fallback

    def _int_env(name: str, fallback: int) -> int:
        v = os.environ.get(name)
        return int(v) if v else fallback

    def _opt_int_env(name: str, fallback: int | None) -> int | None:
        v = os.environ.get(name, "").strip()
        if v:
            return int(v)
        return fallback

    def _float_env(name: str, fallback: float) -> float:
        v = os.environ.get(name)
        return float(v) if v else fallback

    def _bool_env(name: str, fallback: bool) -> bool:
        if name in os.environ:
            return _env_flag(name)
        return fallback

    # system
    log_level = _str_env(
        "MEEKO_LOG_LEVEL", str(system.get("log_level", defaults.log_level))
    ).upper()
    log_target = _opt_str_env(
        "MEEKO_LOG_TARGET", system.get("log_target", defaults.log_target)
    )
    raw_db = system.get("db_path")
    db_path = _path_env(
        "MEEKO_DB_PATH",
        Path(raw_db).expanduser() if raw_db else defaults.db_path,
    )
    led_disabled = _bool_env(
        "MEEKO_LED_DISABLED", bool(system.get("led_disabled", defaults.led_disabled))
    )

    # audio
    input_channels = _int_env(
        "MEEKO_INPUT_CHANNELS",
        int(audio.get("input_channels", defaults.input_channels)),
    )
    output_channels = _int_env(
        "MEEKO_OUTPUT_CHANNELS",
        int(audio.get("output_channels", defaults.output_channels)),
    )
    input_device_index = _opt_int_env(
        "MEEKO_INPUT_DEVICE_INDEX",
        audio.get("input_device_index", defaults.input_device_index),
    )
    output_device_index = _opt_int_env(
        "MEEKO_OUTPUT_DEVICE_INDEX",
        audio.get("output_device_index", defaults.output_device_index),
    )
    mute_mic_while_speaking = _bool_env(
        "MEEKO_MUTE_MIC_WHILE_SPEAKING",
        bool(audio.get("mute_mic_while_speaking", defaults.mute_mic_while_speaking)),
    )

    # wake_word
    raw_model = wake.get("model")
    wake_word_model = _path_env(
        "MEEKO_WAKE_WORD_MODEL",
        Path(raw_model).expanduser() if raw_model else defaults.wake_word_model,
    )
    wake_word_threshold = _float_env(
        "MEEKO_WAKE_WORD_THRESHOLD",
        float(wake.get("threshold", defaults.wake_word_threshold)),
    )
    wake_word_disabled = _bool_env(
        "MEEKO_WAKE_WORD_DISABLED",
        bool(wake.get("disabled", defaults.wake_word_disabled)),
    )

    # claude
    compaction_trigger_tokens = _int_env(
        "MEEKO_COMPACTION_TRIGGER_TOKENS",
        int(
            claude.get("compaction_trigger_tokens", defaults.compaction_trigger_tokens)
        ),
    )

    return MeekoConfig(
        log_level=log_level,
        log_target=log_target,
        db_path=db_path,
        led_disabled=led_disabled,
        input_channels=input_channels,
        output_channels=output_channels,
        input_device_index=input_device_index,
        output_device_index=output_device_index,
        mute_mic_while_speaking=mute_mic_while_speaking,
        wake_word_model=wake_word_model,
        wake_word_threshold=wake_word_threshold,
        wake_word_disabled=wake_word_disabled,
        compaction_trigger_tokens=compaction_trigger_tokens,
    )
