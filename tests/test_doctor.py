from __future__ import annotations

import os
import sys
import time
from collections.abc import Sequence
from pathlib import Path

import pytest
from typer.testing import CliRunner

from video_cleaner import doctor
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
    ) -> None:
        self.encoders = encoders
        self.filters = filters
        self.gpu = gpu
        self.icacls = icacls
        self.calls: list[list[str]] = []

    def __call__(self, cmd: Sequence[str]) -> tuple[int, str]:
        self.calls.append(list(cmd))
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
