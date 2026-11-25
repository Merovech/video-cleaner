from __future__ import annotations

import dataclasses
import os
import sys
import time
from collections.abc import Sequence
from pathlib import Path

import keyring.backends.fail
import keyring.backends.null
import pytest
from typer.testing import CliRunner

from video_cleaner import credentials, doctor
from video_cleaner.cli import app
from video_cleaner.config import Settings, default_config_path, load_settings
from video_cleaner.doctor import Check, run_doctor
from video_cleaner.logarchive import ARCHIVE_DIR_NAME

ENCODERS_OK = """Encoders:
 V..... = Video
 ------
 V....D libx264              libx264 H.264 / AVC / MPEG-4 AVC (codec h264)
 V....D libx265              libx265 H.265 / HEVC (codec hevc)
 V....D hevc_nvenc           NVIDIA NVENC hevc encoder (codec hevc)
 A....D aac                  AAC (Advanced Audio Coding)
"""
FILTERS_OK = """Filters:
  T.. = Timeline support
 ... scale             V->V       Scale the input video size.
 ... libvmaf           VV->V      Calculate the VMAF between two video streams.
"""
ENCODERS_NO_X265 = """Encoders:
 V....D libx264              libx264 H.264 / AVC / MPEG-4 AVC (codec h264)
 V....D libx265_fake         not the real one
"""
FILTERS_NO_VMAF = """Filters:
 ... scale             V->V       Scale the input video size.
 ... vmafmotion        V->V       Calculate the VMAF Motion score.
"""
FFSUBSYNC_HELP = (0, "usage: ffsubsync.py [-h] [-i [SRTIN ...]] [-o SRTOUT]\n")
ICACLS_TIGHT = (
    "C:\\cfg\\config.toml DESKTOP\\ari:(F)\n"
    "                    NT AUTHORITY\\SYSTEM:(F)\n\n"
    "Successfully processed 1 files; Failed processing 0 files\n"
)
ICACLS_LOOSE = (
    "C:\\cfg\\config.toml BUILTIN\\Users:(I)(RX)\n"
    "                    DESKTOP\\ari:(I)(F)\n\n"
    "Successfully processed 1 files; Failed processing 0 files\n"
)


class FakeRunner:
    def __init__(
        self,
        encoders: str = ENCODERS_OK,
        filters: str = FILTERS_OK,
        gpu: tuple[int, str] = (0, "NVIDIA GeForce RTX 4080 SUPER, 16376 MiB\n"),
        icacls: str = ICACLS_TIGHT,
        ffsubsync: tuple[int, str] = FFSUBSYNC_HELP,
    ) -> None:
        self.encoders = encoders
        self.filters = filters
        self.gpu = gpu
        self.icacls = icacls
        self.ffsubsync = ffsubsync
        self.calls: list[list[str]] = []

    def __call__(self, cmd: Sequence[str]) -> tuple[int, str]:
        self.calls.append(list(cmd))
        if "ffsubsync.ffsubsync" in cmd:
            return self.ffsubsync
        if "-encoders" in cmd:
            return 0, self.encoders
        if "-filters" in cmd:
            return 0, self.filters
        if Path(cmd[0]).stem == "nvidia-smi":
            return self.gpu
        if Path(cmd[0]).stem == "icacls":
            return 0, self.icacls
        raise AssertionError(f"unexpected command {cmd}")


def _make_tools(directory: Path, names: Sequence[str] = ("ffmpeg", "ffprobe", "mkvmerge")) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for name in names:
        (directory / f"{name}.exe").write_bytes(b"")


def _settings(tmp_path: Path, **overrides: object) -> Settings:
    library = tmp_path / "library"
    library.mkdir(exist_ok=True)
    (library / "movie.mkv").write_bytes(b"x")
    values: dict[str, object] = {
        "library_root": library,
        "scratch_dir": tmp_path / "scratch",
        "completed_dir": tmp_path / "completed",
        "state_db": tmp_path / "state.db",
        "log_dir": tmp_path / "logs",
        "scratch_cap_bytes": 10**9,
        "subtitle_cache_dir": tmp_path / "subtitles",
    }
    values.update(overrides)
    return Settings(**values)  # type: ignore[arg-type]


def _usage(free: int) -> doctor.DiskUsageFn:
    def usage(path: Path) -> tuple[int, int, int]:
        return (free * 2, free, free)

    return usage


def _env(tmp_path: Path) -> dict[str, str]:
    bin_dir = tmp_path / "bin"
    _make_tools(bin_dir)
    return {"PATH": str(bin_dir)}


def _by_name(checks: Sequence[Check]) -> dict[str, Check]:
    return {c.name: c for c in checks}


def _run(tmp_path: Path, **kwargs: object) -> dict[str, Check]:
    settings = kwargs.pop("settings", None) or _settings(tmp_path)
    params: dict[str, object] = {
        "env": _env(tmp_path),
        "runner": FakeRunner(),
        "disk_usage": _usage(10**12),
        "config_path": None,
        "python_version": (3, 12),
        "windows": True,
    }
    params.update(kwargs)
    return _by_name(run_doctor(settings, **params))  # type: ignore[arg-type]


@pytest.fixture(autouse=True)
def _isolated_appdata(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Whatever a test runs through the real environment sees an empty LOCALAPPDATA of its own,
    never the user's config, credentials file or logs."""
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "isolated-localappdata"))


def test_doctor_all_ok(tmp_path: Path) -> None:
    checks = _run(tmp_path)
    failed = [c for c in checks.values() if c.required and not c.ok]
    assert failed == []
    for name in (
        "python",
        "ffmpeg",
        "libx265",
        "libvmaf",
        "hevc_nvenc",
        "ffprobe",
        "mkvmerge",
        "gpu",
        "scratch space",
        "library_root",
        "completed_dir",
        "scratch_dir",
        "config permissions",
        "keyring",
        "ffsubsync",
        "subtitle_cache_dir",
    ):
        assert name in checks
    assert checks["hevc_nvenc"].ok and not checks["hevc_nvenc"].required
    assert checks["gpu"].ok and "4080" in checks["gpu"].detail
    assert not checks["keyring"].required


def test_doctor_reports_missing_tool_without_crashing(tmp_path: Path) -> None:
    bin_dir = tmp_path / "partial-bin"
    _make_tools(bin_dir, ("ffprobe",))
    env = {"PATH": str(bin_dir), "LOCALAPPDATA": str(tmp_path / "nolocal")}
    env["ProgramFiles"] = str(tmp_path / "nopf")
    checks = _run(tmp_path, env=env)
    assert checks["ffmpeg"].ok is False and checks["ffmpeg"].required
    assert checks["mkvmerge"].ok is False and checks["mkvmerge"].required
    assert checks["ffprobe"].ok is True
    # Without ffmpeg the codec checks fail cleanly instead of raising.
    assert checks["libx265"].ok is False
    assert "ffmpeg" in checks["libx265"].detail


def test_doctor_survives_a_runner_that_raises(tmp_path: Path) -> None:
    def boom(cmd: Sequence[str]) -> tuple[int, str]:
        raise OSError("cannot start")

    checks = _run(tmp_path, runner=boom)
    assert checks["libx265"].ok is False
    assert checks["gpu"].ok is False and not checks["gpu"].required
    assert checks["config permissions"].ok is True  # no config file given


def test_doctor_flags_insufficient_scratch_space(tmp_path: Path) -> None:
    settings = _settings(tmp_path, scratch_cap_bytes=2 * 10**12)
    checks = _run(tmp_path, settings=settings, disk_usage=_usage(500 * 10**9))
    space = checks["scratch space"]
    assert space.ok is False and space.required
    assert "500" in space.detail


def test_doctor_scratch_space_uses_existing_ancestor(tmp_path: Path) -> None:
    seen: list[Path] = []

    def usage(path: Path) -> tuple[int, int, int]:
        seen.append(path)
        return (10**13, 0, 10**13)

    settings = _settings(tmp_path, scratch_dir=tmp_path / "a" / "b" / "scratch")
    checks = _run(tmp_path, settings=settings, disk_usage=usage)
    assert checks["scratch space"].ok
    assert all(p.exists() for p in seen)


def test_doctor_requires_libx265_and_libvmaf(tmp_path: Path) -> None:
    runner = FakeRunner(encoders=ENCODERS_NO_X265, filters=FILTERS_NO_VMAF)
    checks = _run(tmp_path, runner=runner)
    assert checks["libx265"].ok is False and checks["libx265"].required
    assert checks["libvmaf"].ok is False and checks["libvmaf"].required
    assert checks["hevc_nvenc"].ok is False and not checks["hevc_nvenc"].required
    assert all(isinstance(part, str) for call in runner.calls for part in call)


def test_doctor_gpu_not_visible_is_informational(tmp_path: Path) -> None:
    checks = _run(tmp_path, runner=FakeRunner(gpu=(9, "NVIDIA-SMI has failed")))
    assert checks["gpu"].ok is False and not checks["gpu"].required


def test_doctor_flags_unreadable_library_root(tmp_path: Path) -> None:
    settings = _settings(tmp_path, library_root=tmp_path / "missing")
    checks = _run(tmp_path, settings=settings)
    assert checks["library_root"].ok is False and checks["library_root"].required


def test_doctor_write_probe_leaves_no_file(tmp_path: Path) -> None:
    (tmp_path / "completed").mkdir()
    (tmp_path / "scratch").mkdir()
    checks = _run(tmp_path)
    assert checks["completed_dir"].ok and checks["scratch_dir"].ok
    assert list((tmp_path / "completed").iterdir()) == []
    assert list((tmp_path / "scratch").iterdir()) == []


def test_doctor_reports_missing_folders_without_creating_them(tmp_path: Path) -> None:
    checks = _run(tmp_path)
    for name in ("completed_dir", "scratch_dir"):
        assert checks[name].ok, checks[name].detail
        assert "does not exist (will be created)" in checks[name].detail
    assert not (tmp_path / "completed").exists()
    assert not (tmp_path / "scratch").exists()


def test_doctor_refuses_output_dir_inside_library(tmp_path: Path) -> None:
    library = tmp_path / "library"
    settings = _settings(tmp_path, completed_dir=library / "completed")
    checks = _run(tmp_path, settings=settings)
    assert checks["completed_dir"].ok is False
    assert "library" in checks["completed_dir"].detail
    assert not (library / "completed").exists()


@pytest.mark.parametrize("field", ["state_db", "log_dir"])
def test_doctor_refuses_state_or_logs_inside_library(tmp_path: Path, field: str) -> None:
    settings = _settings(tmp_path, **{field: tmp_path / "library" / "vc" / "x"})
    check = _run(tmp_path, settings=settings)["library overlap"]
    assert check.ok is False and check.required
    assert "library root" in check.detail


def test_doctor_refuses_library_inside_completed(tmp_path: Path) -> None:
    settings = _settings(tmp_path, completed_dir=tmp_path)
    check = _run(tmp_path, settings=settings)["library overlap"]
    assert check.ok is False
    assert "is inside the completed folder" in check.detail


def test_doctor_library_overlap_ok(tmp_path: Path) -> None:
    check = _run(tmp_path)["library overlap"]
    assert check.ok and check.required


def test_doctor_unwritable_dir_fails(tmp_path: Path) -> None:
    blocker = tmp_path / "file"
    blocker.write_bytes(b"")
    settings = _settings(tmp_path, scratch_dir=blocker / "scratch")
    checks = _run(tmp_path, settings=settings)
    assert checks["scratch_dir"].ok is False and checks["scratch_dir"].required


def test_doctor_python_version(tmp_path: Path) -> None:
    assert _run(tmp_path, python_version=(3, 11))["python"].ok
    bad = _run(tmp_path, python_version=(3, 14))["python"]
    assert bad.ok is False and bad.required


def test_doctor_warns_on_loose_windows_acl(tmp_path: Path) -> None:
    config = tmp_path / "config.toml"
    config.write_text("", encoding="utf-8")
    loose = _run(tmp_path, config_path=config, runner=FakeRunner(icacls=ICACLS_LOOSE))
    assert loose["config permissions"].ok is False
    assert not loose["config permissions"].required
    assert "BUILTIN\\Users" in loose["config permissions"].detail
    tight = _run(tmp_path, config_path=config, runner=FakeRunner(icacls=ICACLS_TIGHT))
    assert tight["config permissions"].ok is True


SYSTEM_TOOLS = ("icacls", "nvidia-smi")


def test_doctor_runs_windows_system_tools_by_full_path(tmp_path: Path) -> None:
    # A bare "icacls" is also looked up in the current directory (CreateProcess search
    # order), so a planted icacls.exe there would run instead of the real one.
    config = tmp_path / "config.toml"
    config.write_text("", encoding="utf-8")
    runner = FakeRunner()
    system_root = tmp_path / "Windows"
    env = {**_env(tmp_path), "SystemRoot": str(system_root)}
    _run(tmp_path, config_path=config, runner=runner, env=env)
    called = {Path(c[0]).stem: c[0] for c in runner.calls if Path(c[0]).stem in SYSTEM_TOOLS}
    assert called == {n: str(system_root / "System32" / f"{n}.exe") for n in SYSTEM_TOOLS}


def test_doctor_system_tools_default_root_without_systemroot(tmp_path: Path) -> None:
    config = tmp_path / "config.toml"
    config.write_text("", encoding="utf-8")
    runner = FakeRunner()
    _run(tmp_path, config_path=config, runner=runner)
    icacls = next(c[0] for c in runner.calls if Path(c[0]).stem == "icacls")
    assert icacls == str(Path("C:/Windows") / "System32" / "icacls.exe")


def test_doctor_runs_posix_tools_by_name(tmp_path: Path) -> None:
    runner = FakeRunner()
    _run(tmp_path, runner=runner, windows=False, file_mode=lambda p: 0o600)
    assert [c[0] for c in runner.calls if Path(c[0]).stem == "nvidia-smi"] == ["nvidia-smi"]


@pytest.mark.parametrize(
    ("acl", "loose"),
    [
        ("x Everyone:(R)\n", True),
        ("x Everyone:(OI)(CI)(F)\n", True),
        ("x Everyone:(DENY)(R)\n", False),
        ("x BUILTIN\\Users:(W)\n", False),
        ("x NT AUTHORITY\\Authenticated Users:(M)\n", True),
        ("x BUILTIN\\Users:(RD,REA)\n", True),
    ],
)
def test_loose_acl_parser(acl: str, loose: bool) -> None:
    assert bool(doctor.loose_acl_entries(acl)) is loose


def test_doctor_config_mode_posix(tmp_path: Path) -> None:
    config = tmp_path / "config.toml"
    config.write_text("", encoding="utf-8")
    checks = _run(tmp_path, config_path=config, windows=False, file_mode=lambda p: 0o644)
    assert checks["config permissions"].ok is False
    checks = _run(tmp_path, config_path=config, windows=False, file_mode=lambda p: 0o600)
    assert checks["config permissions"].ok is True


# --- CLI ----------------------------------------------------------------------------------


def _config(tmp_path: Path) -> Path:
    (tmp_path / "library").mkdir(exist_ok=True)
    config = tmp_path / "config.toml"
    config.write_text(
        f"library_root = '{tmp_path / 'library'}'\n"
        f"scratch_dir = '{tmp_path / 'scratch'}'\n"
        f"completed_dir = '{tmp_path / 'completed'}'\n"
        f"state_db = '{tmp_path / 'state.db'}'\n"
        f"log_dir = '{tmp_path / 'logs'}'\n",
        encoding="utf-8",
    )
    return config


def _fake_doctor(checks: list[Check]) -> object:
    def fake(settings: Settings, **kwargs: object) -> list[Check]:
        assert kwargs.get("config_path") is not None
        return checks

    return fake


def test_cli_doctor_ok(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    checks = [Check("ffmpeg", True, "found"), Check("gpu", False, "none", required=False)]
    monkeypatch.setattr(doctor, "run_doctor", _fake_doctor(checks))
    logs = tmp_path / "logs"
    logs.mkdir()
    old = logs / "run-old.log"
    old.write_bytes(b"x" * 10)
    stamp = time.time() - 60 * 86_400
    os.utime(old, (stamp, stamp))
    result = CliRunner().invoke(app, ["doctor", "--config", str(_config(tmp_path))])
    assert result.exit_code == 0, result.output
    assert "[ OK ] ffmpeg: found" in result.output
    assert "[WARN] gpu: none" in result.output
    assert "archived 1" in result.output
    assert not old.exists()
    assert (logs / ARCHIVE_DIR_NAME).is_dir()


def test_cli_doctor_required_failure_exits_1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checks = [Check("libx265", False, "missing")]
    monkeypatch.setattr(doctor, "run_doctor", _fake_doctor(checks))
    result = CliRunner().invoke(app, ["doctor", "--config", str(_config(tmp_path))])
    assert result.exit_code == 1
    assert "[FAIL] libx265: missing" in result.output


def test_cli_doctor_config_error_exits_2(tmp_path: Path) -> None:
    result = CliRunner().invoke(app, ["doctor", "--config", str(tmp_path / "missing.toml")])
    assert result.exit_code == 2
    assert "config" in result.output.lower()


def test_doctor_icacls_failure_is_a_warning(tmp_path: Path) -> None:
    config = tmp_path / "config.toml"
    config.write_text("", encoding="utf-8")

    def runner(cmd: Sequence[str]) -> tuple[int, str]:
        if Path(cmd[0]).stem == "icacls":
            return 5, "Access is denied.\n"
        return FakeRunner()(cmd)

    check = _run(tmp_path, config_path=config, runner=runner)["config permissions"]
    assert check.ok is False and not check.required
    assert "Access is denied" in check.detail


def test_doctor_checks_the_config_file_load_settings_reads(tmp_path: Path) -> None:
    appdata = tmp_path / "appdata"
    config = appdata / "video-cleaner" / "config.toml"
    config.parent.mkdir(parents=True)
    config.write_text(_config(tmp_path).read_text(encoding="utf-8"), encoding="utf-8")
    env = {**_env(tmp_path), "LOCALAPPDATA": str(appdata)}
    settings = load_settings(env=env)
    assert settings.library_root == tmp_path / "library"  # load_settings read this file
    assert default_config_path(env) == config
    seen: list[list[str]] = []
    fake = FakeRunner()

    def runner(cmd: Sequence[str]) -> tuple[int, str]:
        seen.append(list(cmd))
        return fake(cmd)

    run_doctor(settings, env=env, runner=runner, disk_usage=_usage(10**12), windows=True)
    assert [c[1:] for c in seen if Path(c[0]).stem == "icacls"] == [[str(config)]]


def test_cli_doctor_checks_the_default_config_it_loaded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    appdata = tmp_path / "appdata"
    config = appdata / "video-cleaner" / "config.toml"
    config.parent.mkdir(parents=True)
    config.write_text(_config(tmp_path).read_text(encoding="utf-8"), encoding="utf-8")
    monkeypatch.setenv("LOCALAPPDATA", str(appdata))
    received: list[object] = []

    def fake(settings: Settings, **kwargs: object) -> list[Check]:
        received.append(kwargs.get("config_path"))
        return [Check("python", True, "3.12")]

    monkeypatch.setattr(doctor, "run_doctor", fake)
    result = CliRunner().invoke(app, ["doctor"])
    assert result.exit_code == 0, result.output
    assert received == [config]


def test_default_runner_captures_output() -> None:
    code, output = doctor.default_runner([sys.executable, "-c", "print('hello')"])
    assert code == 0
    assert "hello" in output


def test_default_helpers_read_the_real_system(tmp_path: Path) -> None:
    total, _, free = doctor._disk_usage(tmp_path)
    assert total >= free > 0
    probe = tmp_path / "f"
    probe.write_bytes(b"")
    assert 0 <= doctor._file_mode(probe) <= 0o777


def test_a_crashing_check_becomes_a_failed_check(tmp_path: Path) -> None:
    def broken(path: Path) -> tuple[int, int, int]:
        raise RuntimeError("boom")

    checks = _run(tmp_path, disk_usage=broken)
    assert checks["scratch space"].ok is False
    assert "boom" in checks["scratch space"].detail
    assert "keyring" in checks  # later checks still ran


# --- credentials ----------------------------------------------------------------------


class _Store:
    def __init__(self, values: dict[str, str] | None = None) -> None:
        self.values = values or {}

    def get(self, name: str) -> str | None:
        return self.values.get(name)

    def set(self, name: str, value: str) -> None:
        self.values[name] = value

    def delete(self, name: str) -> None:
        self.values.pop(name, None)


_CREDS = {
    "opensubtitles_api_key": "key-value-123",
    "opensubtitles_username": "someuser",
    "opensubtitles_password": "pass-value-123",
}


def test_doctor_credentials_set_is_ok_and_never_shows_values(tmp_path: Path) -> None:
    check = _run(tmp_path, credential_store=_Store(dict(_CREDS)))["credentials"]
    assert check.ok is True
    assert not check.required
    for value in _CREDS.values():
        assert value not in check.detail


def test_doctor_missing_credentials_are_a_warning_not_a_failure(tmp_path: Path) -> None:
    store = _Store({"opensubtitles_username": "someuser"})
    check = _run(tmp_path, credential_store=store)["credentials"]
    assert check.ok is False
    assert check.required is False
    assert "opensubtitles_api_key" in check.detail
    assert "opensubtitles_password" in check.detail
    assert "opensubtitles_username" not in check.detail
    assert "auth" in check.detail


def test_doctor_credentials_from_the_environment_count(tmp_path: Path) -> None:
    env = {
        **_env(tmp_path),
        "VC_OPENSUBTITLES_API_KEY": "k",
        "VC_OPENSUBTITLES_USERNAME": "u",
        "VC_OPENSUBTITLES_PASSWORD": "p",
    }
    assert _run(tmp_path, env=env, credential_store=_Store())["credentials"].ok is True


def test_doctor_unusable_credential_store_is_a_warning(tmp_path: Path) -> None:
    # No injected store: the keyring is unusable in tests and there is no LOCALAPPDATA.
    check = _run(tmp_path)["credentials"]
    assert check.ok is False
    assert check.required is False
    assert "LOCALAPPDATA" in check.detail


def test_doctor_credential_store_that_raises_is_a_warning(tmp_path: Path) -> None:
    class Broken(_Store):
        def get(self, name: str) -> str | None:
            raise credentials.CredentialError("keyring locked")

    check = _run(tmp_path, credential_store=Broken())["credentials"]
    assert check.ok is False
    assert check.required is False
    assert "keyring locked" in check.detail


def _credentials_env(tmp_path: Path) -> tuple[dict[str, str], Path]:
    appdata = tmp_path / "appdata"
    path = appdata / "video-cleaner" / "credentials.json"
    path.parent.mkdir(parents=True)
    path.write_text("{}", encoding="utf-8")
    return {**_env(tmp_path), "LOCALAPPDATA": str(appdata)}, path


def test_doctor_without_a_credentials_file_has_nothing_to_warn_about(tmp_path: Path) -> None:
    check = _run(tmp_path, credential_store=_Store())["credentials file permissions"]
    assert check.ok is True
    assert not check.required


def test_doctor_warns_on_a_loose_windows_credentials_file(tmp_path: Path) -> None:
    env, path = _credentials_env(tmp_path)
    runner = FakeRunner(icacls=ICACLS_LOOSE)
    check = _run(tmp_path, env=env, runner=runner, credential_store=_Store())[
        "credentials file permissions"
    ]
    assert check.ok is False
    assert check.required is False
    assert str(path) in check.detail
    assert [c[1] for c in runner.calls if Path(c[0]).stem == "icacls"] == [str(path)]


def test_doctor_accepts_a_tight_windows_credentials_file(tmp_path: Path) -> None:
    env, _ = _credentials_env(tmp_path)
    check = _run(tmp_path, env=env, credential_store=_Store())["credentials file permissions"]
    assert check.ok is True


def test_doctor_warns_on_a_loose_posix_credentials_file(tmp_path: Path) -> None:
    env, _ = _credentials_env(tmp_path)
    name = "credentials file permissions"
    loose = _run(
        tmp_path, env=env, windows=False, file_mode=lambda p: 0o644, credential_store=_Store()
    )[name]
    assert loose.ok is False and not loose.required
    assert "chmod 600" in loose.detail
    tight = _run(
        tmp_path, env=env, windows=False, file_mode=lambda p: 0o600, credential_store=_Store()
    )[name]
    assert tight.ok is True


def test_doctor_credentials_file_icacls_failure_is_a_warning(tmp_path: Path) -> None:
    env, _ = _credentials_env(tmp_path)

    def runner(cmd: Sequence[str]) -> tuple[int, str]:
        if Path(cmd[0]).stem == "icacls":
            return 5, "Access is denied.\n"
        return FakeRunner()(cmd)

    check = _run(tmp_path, env=env, runner=runner, credential_store=_Store())[
        "credentials file permissions"
    ]
    assert check.ok is False and not check.required
    assert "Access is denied" in check.detail


class _Vault:
    """Stands for a usable keyring backend; reading a credential from it fails the test."""

    def get_password(self, service: str, name: str) -> str:
        raise AssertionError("doctor must not read a credential")


def test_doctor_keeps_the_informational_keyring_check(tmp_path: Path) -> None:
    check = _run(tmp_path, credential_store=_Store(), keyring_backend=_Vault)["keyring"]
    assert check.ok is True
    assert not check.required


def test_doctor_names_the_keyring_backend(tmp_path: Path) -> None:
    check = _run(tmp_path, credential_store=_Store(), keyring_backend=_Vault)["keyring"]
    assert f"{_Vault.__module__}.{_Vault.__qualname__}" in check.detail


@pytest.mark.parametrize(
    ("backend", "name"),
    [
        (keyring.backends.fail.Keyring, "keyring.backends.fail.Keyring"),
        (keyring.backends.null.Keyring, "keyring.backends.null.Keyring"),
    ],
)
def test_doctor_says_when_the_keyring_backend_is_unusable(
    tmp_path: Path, backend: type, name: str
) -> None:
    appdata = tmp_path / "appdata"
    env = {**_env(tmp_path), "LOCALAPPDATA": str(appdata)}
    check = _run(tmp_path, env=env, credential_store=_Store(), keyring_backend=backend)["keyring"]
    assert check.ok is False
    assert not check.required
    assert name in check.detail
    assert "no usable keyring backend" in check.detail
    assert str(appdata / "video-cleaner" / "credentials.json") in check.detail
    assert "owner" in check.detail


def test_doctor_unusable_keyring_without_localappdata_says_environment_only(
    tmp_path: Path,
) -> None:
    check = _run(
        tmp_path,
        credential_store=_Store(),
        keyring_backend=keyring.backends.fail.Keyring,
    )["keyring"]
    assert check.ok is False
    assert not check.required
    assert "LOCALAPPDATA" in check.detail
    assert "VC_OPENSUBTITLES_" in check.detail


def test_doctor_keyring_backend_that_fails_to_load_is_unusable(tmp_path: Path) -> None:
    def broken() -> object:
        raise RuntimeError("secret-detail-from-the-backend")

    check = _run(tmp_path, credential_store=_Store(), keyring_backend=broken)["keyring"]
    assert check.ok is False
    assert not check.required
    assert "RuntimeError" in check.detail
    assert "secret-detail" not in check.detail


def test_doctor_credential_store_with_any_other_error_is_a_warning(tmp_path: Path) -> None:
    class PyWinError(Exception):  # what the Windows keyring backend lets through
        pass

    class Broken(_Store):
        def get(self, name: str) -> str | None:
            raise PyWinError(f"(1312, 'CredRead', secret-detail-{name})")

    checks = _run(tmp_path, credential_store=Broken())
    check = checks["credentials"]
    assert check.ok is False
    assert check.required is False
    assert "PyWinError" in check.detail
    assert "secret-detail" not in check.detail
    assert "credentials file permissions" in checks  # later checks still ran


def test_doctor_fails_when_the_subtitle_cache_is_inside_the_library(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    settings = dataclasses.replace(settings, subtitle_cache_dir=settings.library_root / "subs")
    check = _run(tmp_path, settings=settings)["library overlap"]
    assert check.ok is False
    assert check.required is True
    assert "subtitle cache" in check.detail


def test_doctor_fails_when_the_library_is_inside_the_subtitle_cache(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    settings = dataclasses.replace(settings, subtitle_cache_dir=tmp_path)
    check = _run(tmp_path, settings=settings)["library overlap"]
    assert check.ok is False
    assert "subtitle cache" in check.detail


def test_cli_writing_command_refuses_a_subtitle_cache_in_the_library(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("VC_SUBTITLE_CACHE_DIR")  # the conftest default would beat the file
    config = _config(tmp_path)
    config.write_text(
        config.read_text(encoding="utf-8")
        + f"subtitle_cache_dir = '{tmp_path / 'library' / 'subs'}'\n",
        encoding="utf-8",
    )
    result = CliRunner().invoke(app, ["status", "--config", str(config)])
    assert result.exit_code == 2
    assert "subtitle cache" in result.output
    assert "library" in result.output


def test_cli_subtitle_cache_in_the_library_from_the_environment(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        app,
        ["status", "--config", str(_config(tmp_path))],
        env={"VC_SUBTITLE_CACHE_DIR": str(tmp_path / "library" / "subs")},
    )
    assert result.exit_code == 2
    assert "subtitle cache" in result.output


# --- ffsubsync and the subtitle cache folder ----------------------------------------------


def test_doctor_checks_that_ffsubsync_runs(tmp_path: Path) -> None:
    runner = FakeRunner()
    check = _run(tmp_path, runner=runner)["ffsubsync"]
    assert check.ok is True
    assert check.required
    (call,) = [c for c in runner.calls if "ffsubsync.ffsubsync" in c]
    assert call[0] == sys.executable
    assert call[-1] == "--help"
    assert all(isinstance(part, str) for part in call)


def test_doctor_ffsubsync_failure_is_required_and_says_how_to_install(tmp_path: Path) -> None:
    runner = FakeRunner(ffsubsync=(1, "ModuleNotFoundError: No module named 'ffsubsync'\n"))
    check = _run(tmp_path, runner=runner)["ffsubsync"]
    assert check.ok is False
    assert check.required
    assert "ModuleNotFoundError" in check.detail
    assert "uv sync" in check.detail
    assert "uv pip install ffsubsync" in check.detail


def test_doctor_ffsubsync_that_cannot_be_started_is_a_failed_check(tmp_path: Path) -> None:
    fake = FakeRunner()

    def runner(cmd: Sequence[str]) -> tuple[int, str]:
        if "ffsubsync.ffsubsync" in cmd:
            raise OSError("cannot start")
        return fake(cmd)

    checks = _run(tmp_path, runner=runner)
    assert checks["ffsubsync"].ok is False
    assert checks["ffsubsync"].required
    assert "uv sync" in checks["ffsubsync"].detail
    assert checks["libx265"].ok  # the other checks still ran


def test_doctor_reports_a_missing_subtitle_cache_without_creating_it(tmp_path: Path) -> None:
    check = _run(tmp_path)["subtitle_cache_dir"]
    assert check.ok, check.detail
    assert "does not exist (will be created)" in check.detail
    assert not (tmp_path / "subtitles").exists()


def test_doctor_subtitle_cache_write_probe_leaves_no_file(tmp_path: Path) -> None:
    (tmp_path / "subtitles").mkdir()
    check = _run(tmp_path)["subtitle_cache_dir"]
    assert check.ok
    assert "writable" in check.detail
    assert list((tmp_path / "subtitles").iterdir()) == []


def test_doctor_unwritable_subtitle_cache_is_a_warning(tmp_path: Path) -> None:
    blocker = tmp_path / "file"
    blocker.write_bytes(b"")
    settings = _settings(tmp_path, subtitle_cache_dir=blocker / "subtitles")
    check = _run(tmp_path, settings=settings)["subtitle_cache_dir"]
    assert check.ok is False
    assert not check.required  # runs and encodes do not need it; `subs resolve` does


def test_doctor_does_not_create_a_subtitle_cache_inside_the_library(tmp_path: Path) -> None:
    library = tmp_path / "library"
    settings = _settings(tmp_path, subtitle_cache_dir=library / "subs")
    checks = _run(tmp_path, settings=settings)
    assert checks["subtitle_cache_dir"].ok is False
    assert checks["library overlap"].ok is False
    assert not (library / "subs").exists()
