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


_BOOL_TRUE = {"1", "true", "yes"}
_BOOL_FALSE = {"0", "false", "no", "off"}


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
    idle_timeout_seconds: float = 5.0
    conversation_idle_seconds: float = 60.0
    conversation_close_seconds: float = 20.0

    # The profile name is the mode. "query" auto-closes silently after
    # `idle_timeout_seconds` of silence; "conversation" prompts after
    # `conversation_idle_seconds` then closes after
    # `conversation_close_seconds` more silence (see meeko/main.py
    # _idle_monitor).
    @property
    def mode(self) -> str:
        return self.name


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
    web_search_enabled: bool = True
    web_search_max_uses: int = 2


def load_profiles(
    path: str | Path = DEFAULT_CONFIG_PATH,
) -> tuple[dict[str, Profile], str]:
    """Load conversation profiles from `meeko.toml`.

    Returns `(profiles, default_profile_name)`. The profile name is the
    mode, so each profile name must be one of `VALID_MODES`. The top-level
    `default_profile` key selects which profile a fresh session starts in.
    """
    try:
        with open(path, "rb") as f:
            data = tomllib.load(f)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Config file '{path}' not found. Copy meeko.toml.example to "
            f"'{path}' and edit as needed (see README.md §Setup)."
        ) from exc

    raw_profiles = data.get("profiles", {})
    if not raw_profiles:
        raise ValueError(f"No profiles defined in {path}")

    profiles = {}
    for name, fields in raw_profiles.items():
        if name not in VALID_MODES:
            raise ValueError(
                f"Profile name {name!r} is not a valid mode; "
                f"must be one of {sorted(VALID_MODES)}"
            )
        profiles[name] = Profile(
            name=name,
            wake_word=fields["wake_word"],
            prompt=fields["prompt"],
            voice=fields.get("voice"),
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

    default_profile = data.get("default_profile")
    if default_profile is None:
        raise ValueError(
            f"Missing top-level 'default_profile' in {path}. "
            f"Available profiles: {', '.join(sorted(profiles))}"
        )
    if default_profile not in profiles:
        raise ValueError(
            f"default_profile={default_profile!r} in {path} is not a defined "
            f"profile. Available: {', '.join(sorted(profiles))}"
        )

    return profiles, default_profile


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
        v = os.environ.get(name, "").strip().lower()
        if not v:
            return fallback
        if v in _BOOL_TRUE:
            return True
        if v in _BOOL_FALSE:
            return False
        raise ValueError(f"{name} must be one of 1/true/yes/0/false/no/off, got {v!r}")

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
    web_search_enabled = _bool_env(
        "MEEKO_WEB_SEARCH_ENABLED",
        bool(claude.get("web_search_enabled", defaults.web_search_enabled)),
    )
    web_search_max_uses = _int_env(
        "MEEKO_WEB_SEARCH_MAX_USES",
        int(claude.get("web_search_max_uses", defaults.web_search_max_uses)),
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
        web_search_enabled=web_search_enabled,
        web_search_max_uses=web_search_max_uses,
    )
