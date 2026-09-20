"""Meeko configuration loading.

Loads both per-host runtime config (`MeekoConfig`) and the conversation
personas (`Profile`) from a single `meeko.toml` file. The file is located
via `resolve_config_path()`: `$MEEKO_CONFIG`, then the XDG path
(`~/.config/meeko/meeko.toml`), then `./meeko.toml`. Secrets stay in
`.env` (cwd-relative); everything else lives in `meeko.toml`. Any `MEEKO_*`
environment variable overrides the corresponding TOML value, so existing
env-var setups keep working.
"""

import os
import shutil
import tomllib
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path

from meeko.sessions import default_db_path as _default_db_path

# Profile keys from when the profile name selected a hardcoded idle mode.
# Rejected at load with a pointer to the replacement, so a config that set
# them fails loudly instead of silently falling back to default timings.
# A profile that set *neither* old nor new keys can't be told apart from
# one that wants the defaults, so it gets them without complaint — e.g. a
# [profiles.conversation] that relied on the old name-derived check-in now
# closes silently. README.md §Profiles covers that upgrade step.
_RENAMED_PROFILE_KEYS = {
    "conversation_idle_seconds": "idle_timeout_seconds",
    "conversation_close_seconds": "idle_close_seconds",
}

DEFAULT_CONFIG_FILENAME = "meeko.toml"


_BOOL_TRUE = {"1", "true", "yes"}
_BOOL_FALSE = {"0", "false", "no", "off"}


def default_config_path() -> Path:
    """Default config location, following the XDG Base Directory spec.

    `$XDG_CONFIG_HOME/meeko/meeko.toml`, falling back to
    `~/.config/meeko/meeko.toml` when `$XDG_CONFIG_HOME` is unset or, per
    the spec, set to a relative (non-absolute) path.
    """
    xdg = os.environ.get("XDG_CONFIG_HOME")
    base = Path(xdg).expanduser() if xdg else None
    if base is None or not base.is_absolute():
        base = Path.home() / ".config"
    return base / "meeko" / DEFAULT_CONFIG_FILENAME


def resolve_config_path() -> Path:
    """Effective config path by precedence.

    1. `$MEEKO_CONFIG` (explicit override)
    2. the XDG path (`default_config_path()`) if it exists
    3. `./meeko.toml` in the cwd if it exists (dev / legacy fallback)

    Falls through to the XDG path (the auto-create target) when none exist.
    """
    override = os.environ.get("MEEKO_CONFIG")
    if override:
        return Path(override).expanduser()
    xdg_path = default_config_path()
    if xdg_path.is_file():
        return xdg_path
    cwd_path = Path(DEFAULT_CONFIG_FILENAME)
    if cwd_path.is_file():
        return cwd_path
    return xdg_path


def ensure_config_exists() -> tuple[Path, bool]:
    """Resolve the config path; if nothing exists anywhere, copy the bundled
    `default_config.toml` to the XDG location.

    Returns `(path, created)` where `created` is True iff the file was just
    written. The caller is responsible for any user-facing notification —
    logging here would be suppressed since `setup_logging()` runs later.

    An explicit `$MEEKO_CONFIG` override is never auto-created: if it points
    at a missing file (e.g. a typo), that surfaces via the loaders' "not
    found" error rather than silently writing a default to that path.
    """
    path = resolve_config_path()
    if path.is_file() or os.environ.get("MEEKO_CONFIG"):
        return path, False
    # path is the XDG default here (resolve_config_path fell through to it).
    # If it exists but isn't a regular file (e.g. a directory), copying would
    # raise an opaque IsADirectoryError — surface a clear message instead.
    if path.exists():
        raise IsADirectoryError(
            f"Config path {path} exists but is not a regular file; "
            f"remove it so Meeko can create a default config there."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    # Locate the bundled default via importlib.resources so it resolves
    # correctly however the package is installed (source tree, wheel, zip).
    source = resources.files("meeko") / "default_config.toml"
    with resources.as_file(source) as src:
        # copyfile (not copy2): the generated user config should get a fresh
        # mtime and the user's umask, not the packaged file's metadata.
        shutil.copyfile(src, path)
    return path, True


def _default_wake_word_model() -> Path:
    return Path("models/hey_meeko.onnx")


@dataclass(frozen=True)
class Profile:
    name: str
    wake_word: str
    prompt: str
    voice: str | None = None
    description: str | None = None
    idle_timeout_seconds: float = 5.0
    idle_prompt: str | None = None
    idle_close_seconds: float = 20.0
    idle_close_text: str | None = None
    post_wake_timeout_seconds: float = 15.0

    # `description` tells Sonnet when to switch to this profile; it is
    # composed into the switch_profile tool description.
    #
    # The post-turn idle window (see meeko/orchestrator/idle.py
    # run_idle_window): after `idle_timeout_seconds` of silence, speak
    # `idle_prompt` if set and wait `idle_close_seconds` more, then speak
    # `idle_close_text` if set and end the session. With neither text set
    # the session closes silently. Non-positive `idle_timeout_seconds`
    # disables the window.
    #
    # `post_wake_timeout_seconds` is the silence window right after the
    # wake word, before the user's first turn. On expiry the session
    # closes silently and returns to IDLE (re-wake required). Non-positive
    # disables it.


@dataclass(frozen=True)
class MeekoConfig:
    """Per-host runtime config. All fields optional; defaults mirror the
    historical env-var defaults so unset == prior behavior."""

    # system
    log_level: str = "INFO"
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
    # These two repeat `claude_client`'s constants rather than importing
    # them: config is a leaf that components import, and reaching into
    # the client would pull the Anthropic SDK in behind it. This is the
    # default that actually applies — `main.py` passes
    # `config.web_search_max_uses` into the client — so
    # `test_defaults_match_the_client_constants` pins the two together.
    web_search_max_uses: int = 4
    # location
    latitude: float | None = None
    longitude: float | None = None
    # weather
    weather_units: str = "imperial"


def load_profiles(
    path: str | Path | None = None,
) -> tuple[dict[str, Profile], str]:
    """Load conversation profiles from `meeko.toml`.

    Returns `(profiles, default_profile_name)`. Profile names are
    free-form. The top-level `default_profile` key selects which profile a
    fresh session starts in.

    With no explicit `path`, resolves via `resolve_config_path()`.
    """
    if path is None:
        path = resolve_config_path()
    try:
        with open(path, "rb") as f:
            data = tomllib.load(f)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Config file '{path}' not found (see README.md §Configuration)."
        ) from exc
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"Invalid TOML in '{path}': {exc}") from exc

    raw_profiles = data.get("profiles", {})
    if not raw_profiles:
        raise ValueError(f"No profiles defined in {path}")

    profiles = {}
    for name, fields in raw_profiles.items():
        for old_key, new_key in _RENAMED_PROFILE_KEYS.items():
            if old_key in fields:
                raise ValueError(
                    f"[profiles.{name}] in {path} uses {old_key!r}, which was "
                    f"renamed to {new_key!r}. Also set 'idle_prompt' (and "
                    f"optionally 'idle_close_text') to keep the spoken check-in; "
                    f"see README.md §Configuration."
                )
        profiles[name] = Profile(
            name=name,
            wake_word=fields["wake_word"],
            prompt=fields["prompt"],
            voice=fields.get("voice"),
            description=fields.get("description"),
            idle_timeout_seconds=float(
                fields.get("idle_timeout_seconds", Profile.idle_timeout_seconds)
            ),
            idle_prompt=fields.get("idle_prompt"),
            idle_close_seconds=float(
                fields.get("idle_close_seconds", Profile.idle_close_seconds)
            ),
            idle_close_text=fields.get("idle_close_text"),
            post_wake_timeout_seconds=float(
                fields.get(
                    "post_wake_timeout_seconds", Profile.post_wake_timeout_seconds
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


def load_config(path: str | Path | None = None) -> MeekoConfig:
    """Load `MeekoConfig` from `meeko.toml`, with env-var overrides.

    Precedence: env var > meeko.toml > dataclass default. Missing tables
    and keys silently fall through to defaults; the file may not even
    define any of `[system]`, `[audio]`, `[wake_word]`, `[claude]`,
    `[location]`, `[weather]`. With no explicit `path`, resolves via
    `resolve_config_path()`.
    """
    if path is None:
        path = resolve_config_path()
    try:
        with open(path, "rb") as f:
            data = tomllib.load(f)
    except FileNotFoundError:
        data = {}
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"Invalid TOML in '{path}': {exc}") from exc

    system = data.get("system", {})
    audio = data.get("audio", {})
    wake = data.get("wake_word", {})
    claude = data.get("claude", {})
    location = data.get("location", {})
    weather = data.get("weather", {})

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

    # location
    def _opt_float_env(name: str, fallback: float | None) -> float | None:
        v = os.environ.get(name, "").strip()
        if v:
            return float(v)
        return fallback

    latitude = _opt_float_env(
        "MEEKO_LATITUDE",
        location.get("latitude", defaults.latitude),
    )
    longitude = _opt_float_env(
        "MEEKO_LONGITUDE",
        location.get("longitude", defaults.longitude),
    )
    if (latitude is None) != (longitude is None):
        raise ValueError(
            "location.latitude and location.longitude must both be set, or both unset"
        )

    # weather
    weather_units = _str_env(
        "MEEKO_WEATHER_UNITS", str(weather.get("units", defaults.weather_units))
    ).lower()
    if weather_units not in {"imperial", "metric"}:
        raise ValueError(
            f"weather units must be 'imperial' or 'metric', got {weather_units!r}"
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
        latitude=latitude,
        longitude=longitude,
        weather_units=weather_units,
    )
