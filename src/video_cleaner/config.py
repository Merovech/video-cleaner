"""Settings: defaults, then TOML file, then ``VC_*`` environment overrides."""

from __future__ import annotations

import math
import os
import tomllib
from collections.abc import Callable, Mapping
from dataclasses import MISSING, dataclass, fields
from pathlib import Path
from typing import Any

ENV_PREFIX = "VC_"


def env_var(setting: str) -> str:
    """The environment variable that overrides *setting* (``vmaf_target`` -> ``VC_VMAF_TARGET``)."""
    return ENV_PREFIX + setting.upper()


APP_DIR_NAME = "video-cleaner"
X265_PRESETS = frozenset(
    {
        "ultrafast",
        "superfast",
        "veryfast",
        "faster",
        "fast",
        "medium",
        "slow",
        "slower",
        "veryslow",
        "placebo",
    }
)


class ConfigError(ValueError):
    """Raised for any invalid, unknown or missing configuration."""


@dataclass(frozen=True)
class Settings:
    library_root: Path
    scratch_dir: Path
    completed_dir: Path
    state_db: Path
    log_dir: Path
    scratch_cap_bytes: int = 2 * 10**12
    min_file_size: int = 0
    bitrate_floor_kbps: int = 2500  # placeholder, set by calibration
    vmaf_target: float = 95.0
    crf_min: int = 16
    crf_max: int = 24
    guard_mean_floor: float = 93.0
    guard_worst_floor: float = 88.0  # placeholder, set by calibration
    min_saving: float = 0.10
    preset: str = "slow"
    heartbeat_s: float = 15.0
    plain_progress_s: float = 30.0


def _to_int(name: str, value: object) -> int:
    if isinstance(value, bool):
        raise ConfigError(f"{name}: expected an integer, got {value!r}")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if value.is_integer():
            return int(value)
        raise ConfigError(f"{name}: expected an integer, got {value!r}")
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            pass
        try:
            as_float = float(value.strip())
        except ValueError:
            raise ConfigError(f"{name}: expected an integer, got {value!r}") from None
        if math.isfinite(as_float) and as_float.is_integer():
            return int(as_float)
    raise ConfigError(f"{name}: expected an integer, got {value!r}")


def _to_float(name: str, value: object) -> float:
    if isinstance(value, bool):
        raise ConfigError(f"{name}: expected a number, got {value!r}")
    if isinstance(value, (int, float)):
        result = float(value)
    elif isinstance(value, str):
        try:
            result = float(value.strip())
        except ValueError:
            raise ConfigError(f"{name}: expected a number, got {value!r}") from None
    else:
        raise ConfigError(f"{name}: expected a number, got {value!r}")
    if not math.isfinite(result):
        raise ConfigError(f"{name}: must be a finite number, got {value!r}")
    return result


def _to_str(name: str, value: object) -> str:
    if not isinstance(value, str):
        raise ConfigError(f"{name}: expected a string, got {value!r}")
    return value


def _to_path(name: str, value: object) -> Path:
    if not isinstance(value, str):
        raise ConfigError(f"{name}: expected a path string, got {value!r}")
    if not value.strip():
        raise ConfigError(f"{name}: path must not be empty")
    return Path(value)


_CONVERTERS: dict[str, Callable[[str, object], Any]] = {
    "int": _to_int,
    "float": _to_float,
    "str": _to_str,
    "Path": _to_path,
}


def _field_converters() -> dict[str, Callable[[str, object], Any]]:
    # `from __future__ import annotations` makes f.type a string such as "int".
    return {f.name: _CONVERTERS[str(f.type)] for f in fields(Settings)}


def _require(cond: bool, message: str) -> None:
    if not cond:
        raise ConfigError(message)


def _validate(values: Mapping[str, Any]) -> None:
    _require(0 <= values["crf_min"] <= 51, "crf_min must be within 0..51")
    _require(0 <= values["crf_max"] <= 51, "crf_max must be within 0..51")
    _require(
        values["crf_min"] <= values["crf_max"],
        f"crf_min ({values['crf_min']}) must not exceed crf_max ({values['crf_max']})",
    )
    _require(0 < values["vmaf_target"] <= 100, "vmaf_target must be in (0, 100]")
    _require(0 <= values["guard_mean_floor"] <= 100, "guard_mean_floor must be within 0..100")
    _require(0 <= values["guard_worst_floor"] <= 100, "guard_worst_floor must be within 0..100")
    _require(
        values["guard_worst_floor"] <= values["guard_mean_floor"],
        "guard_worst_floor must not exceed guard_mean_floor",
    )
    _require(0 <= values["min_saving"] < 1, "min_saving must be within [0, 1)")
    _require(values["scratch_cap_bytes"] > 0, "scratch_cap_bytes must be positive")
    _require(values["min_file_size"] >= 0, "min_file_size must not be negative")
    _require(values["bitrate_floor_kbps"] >= 0, "bitrate_floor_kbps must not be negative")
    _require(values["heartbeat_s"] > 0, "heartbeat_s must be positive")
    _require(values["plain_progress_s"] > 0, "plain_progress_s must be positive")
    _require(
        values["preset"] in X265_PRESETS,
        f"preset {values['preset']!r} is not an x265 preset ({', '.join(sorted(X265_PRESETS))})",
    )


def default_config_path(env: Mapping[str, str] | None = None) -> Path | None:
    """The config file `load_settings` reads when no path is given, or None without APPDATA.

    The single source of this rule: `doctor` checks the permissions of the same file.
    """
    environ: Mapping[str, str] = os.environ if env is None else env
    appdata = environ.get("APPDATA")
    return Path(appdata) / APP_DIR_NAME / "config.toml" if appdata else None


def _read_toml(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as fh:
            return tomllib.load(fh)
    except FileNotFoundError:
        raise ConfigError(f"config file not found: {path}") from None
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"invalid TOML in {path}: {exc}") from exc
    except OSError as exc:
        raise ConfigError(f"cannot read config file {path}: {exc}") from exc


def load_settings(
    config_path: Path | None = None, env: Mapping[str, str] | None = None
) -> Settings:
    """Build `Settings` from defaults, a TOML file and ``VC_<FIELD>`` env variables.

    Precedence (lowest first): dataclass defaults, TOML file, environment. When *env* is
    given it is the only environment consulted (``os.environ`` is not read); when None,
    ``os.environ`` is used. An explicit *config_path* must exist; the default path
    (``%APPDATA%\\video-cleaner\\config.toml``) is optional.
    """
    environ: Mapping[str, str] = os.environ if env is None else env
    converters = _field_converters()

    raw: dict[str, tuple[str, object]] = {}  # field -> (source label, raw value)

    if config_path is not None:
        file_path: Path | None = config_path
        explicit = True
    else:
        file_path = default_config_path(environ)
        explicit = False
    if file_path is not None and (explicit or file_path.is_file()):
        table = _read_toml(file_path)
        unknown = sorted(set(table) - set(converters))
        if unknown:
            raise ConfigError(
                f"unknown key(s) in {file_path}: {', '.join(unknown)}; "
                f"valid keys: {', '.join(converters)}"
            )
        for key, value in table.items():
            raw[key] = (f"{file_path}", value)

    for name in converters:
        env_name = env_var(name)
        if env_name in environ:
            raw[name] = (f"environment variable {env_name}", environ[env_name])

    defaults = {f.name: f.default for f in fields(Settings) if f.default is not MISSING}
    values: dict[str, Any] = dict(defaults)
    for name, (source, value) in raw.items():
        try:
            values[name] = converters[name](name, value)
        except ConfigError as exc:
            raise ConfigError(f"{exc} (from {source})") from None

    local_appdata = environ.get("LOCALAPPDATA")
    app_local = Path(local_appdata) / APP_DIR_NAME if local_appdata else None
    derived = {"state_db": ("state.db", app_local), "log_dir": ("logs", app_local)}
    for name, (leaf, base) in derived.items():
        if name not in values:
            if base is None:
                raise ConfigError(
                    f"{name} is not set and LOCALAPPDATA is unavailable to derive a default; "
                    f"set it in the config file or as {env_var(name)}"
                )
            values[name] = base / leaf

    for f in fields(Settings):
        if f.name not in values:
            raise ConfigError(
                f"{f.name} is required: set it in the config file or as {env_var(f.name)}"
            )

    _validate(values)
    return Settings(**values)
