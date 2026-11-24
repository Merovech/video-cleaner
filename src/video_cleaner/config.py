"""Settings: defaults, then TOML file, then ``VC_*`` environment overrides."""

from __future__ import annotations

import math
import os
import re
import tomllib
from collections.abc import Callable, Mapping
from dataclasses import MISSING, dataclass, fields
from pathlib import Path, PureWindowsPath
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


LOG_FILE_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR")
# Where the subtitle service lives. The service has changed domain before, so these are
# settings; the defaults are today's values.
DEFAULT_OPENSUBTITLES_DOMAIN = "opensubtitles.com"
DEFAULT_OPENSUBTITLES_API_HOST = "api.opensubtitles.com"
DEFAULT_OPENSUBTITLES_API_PATH = "/api/v1"
DEFAULT_SUBTITLE_LANGUAGE = "en"
MAX_HEARTBEAT_S = 15.0  # plan 4.4: never silent longer than 15 s


class ConfigError(ValueError):
    """Raised for any invalid, unknown or missing configuration."""


@dataclass(frozen=True)
class Settings:
    library_root: Path
    scratch_dir: Path
    completed_dir: Path
    state_db: Path
    log_dir: Path
    # `load_settings` derives the real default (under LOCALAPPDATA); this relative value only
    # serves code that builds `Settings` directly and passes no cache folder.
    subtitle_cache_dir: Path = Path("subtitles")
    scratch_cap_bytes: int = 2 * 10**12
    min_file_size: int = 0
    bitrate_floor_kbps: int = 2500  # placeholder the user sets; calibration does not change it
    vmaf_target: float = 95.0
    crf_min: int = 16
    crf_max: int = 24
    guard_mean_floor: float = 93.0
    guard_worst_floor: float = 88.0  # placeholder the user sets after reviewing the calibration
    min_saving: float = 0.10
    preset: str = "slow"
    heartbeat_s: float = 15.0
    plain_progress_s: float = 30.0
    log_file_level: str = "DEBUG"
    local_subtitle_dirs: tuple[Path, ...] = ()
    forced_full_cues_per_hour: float = 250.0  # placeholder, calibrated against the real library
    align_max_offset_s: float = 120.0
    align_scale_tolerance: float = 0.05
    align_min_score: float = 0.0  # 0 disables the score check
    subtitle_max_bytes: int = 2_000_000
    subtitle_min_cues: int = 20
    subtitle_candidates_per_file: int = 3
    opensubtitles_domain: str = DEFAULT_OPENSUBTITLES_DOMAIN
    opensubtitles_api_host: str = DEFAULT_OPENSUBTITLES_API_HOST
    opensubtitles_api_path: str = DEFAULT_OPENSUBTITLES_API_PATH
    subtitle_language: str = DEFAULT_SUBTITLE_LANGUAGE


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
    """An absolute path with no control characters.

    A control character is what a TOML basic string makes of a Windows path ("D:\\temp" gives
    a TAB); a relative path would depend on the folder the command happens to run in. Both are
    refused. Windows forms (``D:\\x``, ``\\\\nas\\share``) count as absolute on any platform.
    """
    if not isinstance(value, str):
        raise ConfigError(f"{name}: expected a path string, got {value!r}")
    if not value.strip():
        raise ConfigError(f"{name}: path must not be empty")
    bad = next((c for c in value if ord(c) < 0x20 or ord(c) == 0x7F), None)
    if bad is not None:
        raise ConfigError(
            f"{name}: path {value!r} contains the control character {bad!r}; use a TOML "
            "literal string (single quotes) for Windows paths"
        )
    if not (Path(value).is_absolute() or PureWindowsPath(value).is_absolute()):
        raise ConfigError(f"{name}: path {value!r} must be absolute")
    return Path(value)


def _to_path_tuple(name: str, value: object) -> tuple[Path, ...]:
    """A TOML array of absolute path strings (an environment value is split first)."""
    if not isinstance(value, (list, tuple)):
        raise ConfigError(f"{name}: expected an array of path strings, got {value!r}")
    return tuple(_to_path(f"{name}[{index}]", item) for index, item in enumerate(value))


_CONVERTERS: dict[str, Callable[[str, object], Any]] = {
    "int": _to_int,
    "float": _to_float,
    "str": _to_str,
    "Path": _to_path,
    "tuple[Path, ...]": _to_path_tuple,
}
_LIST_TYPE = "tuple[Path, ...]"


def _list_fields() -> frozenset[str]:
    return frozenset(f.name for f in fields(Settings) if str(f.type) == _LIST_TYPE)


def _split_env_list(value: str) -> list[str]:
    """An environment value holding several paths: split on ``os.pathsep``, empty parts dropped."""
    return [part for part in value.split(os.pathsep) if part.strip()]


def _field_converters() -> dict[str, Callable[[str, object], Any]]:
    # `from __future__ import annotations` makes f.type a string such as "int".
    return {f.name: _CONVERTERS[str(f.type)] for f in fields(Settings)}


def _require(cond: bool, message: str) -> None:
    if not cond:
        raise ConfigError(message)


_DNS_LABEL = r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?"
# At least two labels, so a bare top-level name (which would trust every site under it) is out.
_DOMAIN = re.compile(rf"(?:{_DNS_LABEL}\.)+{_DNS_LABEL}")
_API_PATH = re.compile(r"/[A-Za-z0-9._~-]+(?:/[A-Za-z0-9._~-]+)*")
_LANGUAGE = re.compile(r"[a-z]{2,3}(?:-[a-z0-9]{2,4})?")


def _is_domain(value: str) -> bool:
    """A lower-case ASCII DNS name of two or more labels that is not an IP address."""
    return (
        len(value) <= 253
        and _DOMAIN.fullmatch(value) is not None
        and not value.rsplit(".", 1)[1].isdigit()  # "127.0.0.1" and "10.0.0.1.1" are numbers
    )


def _normalise_endpoint(values: dict[str, Any]) -> None:
    """Trim and lower-case the host names and language; drop a trailing "/" from the path."""
    for name in ("opensubtitles_domain", "opensubtitles_api_host", "subtitle_language"):
        values[name] = str(values[name]).strip().lower()
    path = str(values["opensubtitles_api_path"]).strip()
    values["opensubtitles_api_path"] = path[:-1] if len(path) > 1 and path.endswith("/") else path


def _validate_endpoint(values: Mapping[str, Any]) -> None:
    domain, host = values["opensubtitles_domain"], values["opensubtitles_api_host"]
    _require(
        _is_domain(domain),
        f"opensubtitles_domain {domain!r} must be a DNS name such as 'opensubtitles.com': "
        "letters, digits, hyphens and dots, at least two labels, no scheme, port or path",
    )
    _require(
        _is_domain(host) and (host == domain or host.endswith("." + domain)),
        f"opensubtitles_api_host {host!r} must be {domain!r} or a host under it "
        "(requests carrying your credentials only go to that domain)",
    )
    _require(
        _API_PATH.fullmatch(values["opensubtitles_api_path"]) is not None
        and ".." not in values["opensubtitles_api_path"].split("/"),
        f"opensubtitles_api_path {values['opensubtitles_api_path']!r} must start with '/' and "
        "hold only letters, digits, '.', '_', '~', '-' and '/' (no '..', query or fragment)",
    )
    _require(
        _LANGUAGE.fullmatch(values["subtitle_language"]) is not None,
        f"subtitle_language {values['subtitle_language']!r} must be one language code such as "
        "'en' or 'pt-br'",
    )


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
    _require(
        0 < values["heartbeat_s"] <= MAX_HEARTBEAT_S,
        f"heartbeat_s must be within (0, {MAX_HEARTBEAT_S:g}]: the console is never silent longer",
    )
    _require(values["plain_progress_s"] > 0, "plain_progress_s must be positive")
    _require(
        values["log_file_level"] in LOG_FILE_LEVELS,
        f"log_file_level {values['log_file_level']!r} must be one of {', '.join(LOG_FILE_LEVELS)}",
    )
    _require(values["forced_full_cues_per_hour"] > 0, "forced_full_cues_per_hour must be positive")
    _require(values["align_max_offset_s"] > 0, "align_max_offset_s must be positive")
    _require(
        0 <= values["align_scale_tolerance"] < 1, "align_scale_tolerance must be within [0, 1)"
    )
    _require(values["align_min_score"] >= 0, "align_min_score must not be negative (0 disables)")
    _require(values["subtitle_max_bytes"] > 0, "subtitle_max_bytes must be positive")
    _require(values["subtitle_min_cues"] >= 0, "subtitle_min_cues must not be negative")
    _require(
        values["subtitle_candidates_per_file"] >= 1,
        "subtitle_candidates_per_file must be at least 1",
    )
    _validate_endpoint(values)
    _require(
        values["preset"] in X265_PRESETS,
        f"preset {values['preset']!r} is not an x265 preset ({', '.join(sorted(X265_PRESETS))})",
    )


def default_config_path(env: Mapping[str, str] | None = None) -> Path | None:
    """The config file `load_settings` reads when no path is given, or None without LOCALAPPDATA.

    The config sits beside the state database and the logs: it holds paths of this machine, so
    it is not kept in the roaming profile (``APPDATA``).

    The single source of this rule: `doctor` checks the permissions of the same file.
    """
    environ: Mapping[str, str] = os.environ if env is None else env
    local_appdata = environ.get("LOCALAPPDATA")
    return Path(local_appdata) / APP_DIR_NAME / "config.toml" if local_appdata else None


def _read_toml(path: Path) -> dict[str, Any]:
    try:
        # utf-8-sig: Notepad and PowerShell may save the file with a byte order mark.
        return tomllib.loads(path.read_text(encoding="utf-8-sig"))
    except UnicodeDecodeError as exc:
        raise ConfigError(f"config file {path} is not valid UTF-8: {exc}") from exc
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
    (``%LOCALAPPDATA%\\video-cleaner\\config.toml``) is optional.
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

    list_fields = _list_fields()
    for name in converters:
        env_name = env_var(name)
        if env_name in environ:
            text = environ[env_name]
            raw[name] = (
                f"environment variable {env_name}",
                _split_env_list(text) if name in list_fields else text,
            )

    defaults = {f.name: f.default for f in fields(Settings) if f.default is not MISSING}
    values: dict[str, Any] = dict(defaults)
    for name, (source, value) in raw.items():
        try:
            values[name] = converters[name](name, value)
        except ConfigError as exc:
            raise ConfigError(f"{exc} (from {source})") from None

    local_appdata = environ.get("LOCALAPPDATA")
    app_local = Path(local_appdata) / APP_DIR_NAME if local_appdata else None
    derived = {
        "state_db": ("state.db", app_local),
        "log_dir": ("logs", app_local),
        "subtitle_cache_dir": ("subtitles", app_local),
    }
    for name, (leaf, base) in derived.items():
        if name not in raw:
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

    values["log_file_level"] = str(values["log_file_level"]).strip().upper()
    _normalise_endpoint(values)
    _validate(values)
    return Settings(**values)
