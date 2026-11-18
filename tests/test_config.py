from __future__ import annotations

from pathlib import Path

import pytest

from video_cleaner.config import ConfigError, Settings, load_settings

REQUIRED_PATHS = {
    "library_root": "Z:/library",
    "scratch_dir": "D:/vc/scratch",
    "completed_dir": "D:/vc/completed",
}


def _env(tmp_path: Path, **extra: str) -> dict[str, str]:
    """Isolated environment: never falls through to the real os.environ."""
    env = {
        "LOCALAPPDATA": str(tmp_path / "local"),
        "VC_LIBRARY_ROOT": REQUIRED_PATHS["library_root"],
        "VC_SCRATCH_DIR": REQUIRED_PATHS["scratch_dir"],
        "VC_COMPLETED_DIR": REQUIRED_PATHS["completed_dir"],
    }
    env.update(extra)
    return env


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def _paths_toml() -> str:
    return (
        f"library_root = '{REQUIRED_PATHS['library_root']}'\n"
        f"scratch_dir = '{REQUIRED_PATHS['scratch_dir']}'\n"
        f"completed_dir = '{REQUIRED_PATHS['completed_dir']}'\n"
    )


def test_defaults_match_spec(tmp_path: Path) -> None:
    s = load_settings(None, _env(tmp_path))
    assert (s.crf_min, s.crf_max) == (16, 24)
    assert s.vmaf_target == 95.0
    assert s.scratch_cap_bytes == 2 * 10**12
    assert s.preset == "slow"
    assert s.min_saving == 0.10
    assert s.min_file_size == 0
    assert s.bitrate_floor_kbps == 2500
    assert s.guard_mean_floor == 93.0
    assert s.guard_worst_floor == 88.0
    assert s.heartbeat_s == 15.0
    assert s.plain_progress_s == 30.0
    assert s.state_db == tmp_path / "local" / "video-cleaner" / "state.db"
    assert s.log_dir == tmp_path / "local" / "video-cleaner" / "logs"


def test_settings_is_frozen_and_directly_constructible() -> None:
    s = Settings(
        library_root=Path("a"),
        scratch_dir=Path("b"),
        completed_dir=Path("c"),
        state_db=Path("d"),
        log_dir=Path("e"),
    )
    with pytest.raises(AttributeError):
        s.preset = "fast"  # type: ignore[misc]


def test_toml_overrides_defaults(tmp_path: Path) -> None:
    cfg = _write(
        tmp_path / "config.toml",
        _paths_toml() + "vmaf_target = 94.5\ncrf_max = 26\npreset = 'medium'\n"
        "scratch_cap_bytes = 5000000000\nmin_saving = 0.2\n",
    )
    env = {"LOCALAPPDATA": str(tmp_path)}
    s = load_settings(cfg, env)
    assert s.vmaf_target == 94.5
    assert s.crf_max == 26
    assert s.preset == "medium"
    assert s.scratch_cap_bytes == 5_000_000_000
    assert s.min_saving == 0.2
    assert s.library_root == Path(REQUIRED_PATHS["library_root"])
    assert s.crf_min == 16  # untouched default


def test_env_overrides_file(tmp_path: Path) -> None:
    cfg = _write(tmp_path / "config.toml", _paths_toml() + "vmaf_target = 95\n")
    s = load_settings(cfg, _env(tmp_path, VC_VMAF_TARGET="94"))
    assert s.vmaf_target == 94.0
    assert isinstance(s.vmaf_target, float)


def test_env_converts_types(tmp_path: Path) -> None:
    s = load_settings(
        None,
        _env(
            tmp_path,
            VC_CRF_MIN="18",
            VC_SCRATCH_CAP_BYTES="1e12",
            VC_PRESET="medium",
            VC_STATE_DB=str(tmp_path / "x.db"),
        ),
    )
    assert s.crf_min == 18
    assert s.scratch_cap_bytes == 10**12
    assert s.preset == "medium"
    assert s.state_db == tmp_path / "x.db"


def test_load_is_pure_wrt_injected_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VC_VMAF_TARGET", "50")
    monkeypatch.setenv("VC_PRESET", "ultrafast")
    s = load_settings(None, _env(tmp_path))
    assert s.vmaf_target == 95.0
    assert s.preset == "slow"


def test_env_none_reads_os_environ(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for k, v in _env(tmp_path, VC_VMAF_TARGET="91").items():
        monkeypatch.setenv(k, v)
    assert load_settings(None, None).vmaf_target == 91.0


def test_default_config_path_is_under_localappdata(tmp_path: Path) -> None:
    cfg_dir = tmp_path / "local" / "video-cleaner"
    cfg_dir.mkdir(parents=True)
    _write(cfg_dir / "config.toml", "crf_max = 22\n")
    s = load_settings(None, _env(tmp_path))
    assert s.crf_max == 22


def test_missing_default_config_is_fine_but_explicit_missing_is_error(tmp_path: Path) -> None:
    assert load_settings(None, _env(tmp_path)).crf_max == 24
    with pytest.raises(ConfigError, match="not found"):
        load_settings(tmp_path / "nope.toml", _env(tmp_path))


def test_rejects_inverted_crf_range(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="crf_min"):
        load_settings(None, _env(tmp_path, VC_CRF_MIN="25", VC_CRF_MAX="20"))
    cfg = _write(tmp_path / "c.toml", _paths_toml() + "crf_min = 30\ncrf_max = 18\n")
    with pytest.raises(ConfigError, match="crf_min"):
        load_settings(cfg, {"LOCALAPPDATA": str(tmp_path)})


def test_paths_with_spaces_and_unicode_roundtrip(tmp_path: Path) -> None:
    lib = "Z:/Films & Séries/日本語 映画/The First Omen (2024).en.MX]"
    unc = r"\\nas\media\Ünï cödé"
    cfg = _write(
        tmp_path / "config.toml",
        f"library_root = '{unc}'\nscratch_dir = 'D:/my scratch/ü'\ncompleted_dir = '{lib}'\n",
    )
    s = load_settings(cfg, {"LOCALAPPDATA": str(tmp_path)})
    assert str(s.library_root) == str(Path(unc))
    assert s.scratch_dir == Path("D:/my scratch/ü")
    assert s.completed_dir == Path(lib)
    s2 = load_settings(None, _env(tmp_path, VC_LIBRARY_ROOT=lib))
    assert s2.library_root == Path(lib)


@pytest.mark.parametrize("name", ["library_root", "scratch_dir", "completed_dir"])
def test_missing_required_path_raises(tmp_path: Path, name: str) -> None:
    env = _env(tmp_path)
    del env["VC_" + name.upper()]
    with pytest.raises(ConfigError, match=name):
        load_settings(None, env)


def test_missing_localappdata_without_state_paths_raises(tmp_path: Path) -> None:
    env = _env(tmp_path)
    del env["LOCALAPPDATA"]
    with pytest.raises(ConfigError, match="state_db"):
        load_settings(None, env)
    env["VC_STATE_DB"] = str(tmp_path / "s.db")
    with pytest.raises(ConfigError, match="log_dir"):
        load_settings(None, env)


def test_unknown_toml_key_raises(tmp_path: Path) -> None:
    cfg = _write(tmp_path / "c.toml", _paths_toml() + "vmaf_targt = 95\n")
    with pytest.raises(ConfigError, match="vmaf_targt"):
        load_settings(cfg, {"LOCALAPPDATA": str(tmp_path)})


def test_invalid_toml_raises(tmp_path: Path) -> None:
    cfg = _write(tmp_path / "c.toml", "this is = = not toml")
    with pytest.raises(ConfigError, match="invalid TOML"):
        load_settings(cfg, _env(tmp_path))


@pytest.mark.parametrize(
    ("var", "value"),
    [
        ("VC_VMAF_TARGET", "abc"),
        ("VC_VMAF_TARGET", "nan"),
        ("VC_VMAF_TARGET", "101"),
        ("VC_CRF_MIN", "1.5"),
        ("VC_CRF_MAX", "99"),
        ("VC_MIN_SAVING", "1.0"),
        ("VC_MIN_SAVING", "-0.1"),
        ("VC_SCRATCH_CAP_BYTES", "0"),
        ("VC_MIN_FILE_SIZE", "-1"),
        ("VC_HEARTBEAT_S", "0"),
        ("VC_PLAIN_PROGRESS_S", "-5"),
        ("VC_PRESET", "warp"),
        ("VC_GUARD_WORST_FLOOR", "99"),
        ("VC_LIBRARY_ROOT", ""),
    ],
)
def test_invalid_values_raise(tmp_path: Path, var: str, value: str) -> None:
    with pytest.raises(ConfigError):
        load_settings(None, _env(tmp_path, **{var: value}))


@pytest.mark.parametrize(
    "line",
    ["crf_min = true", "vmaf_target = 'high'", "preset = 3", "library_root = 5", "crf_max = 20.5"],
)
def test_wrong_toml_types_raise(tmp_path: Path, line: str) -> None:
    cfg = _write(tmp_path / "c.toml", _paths_toml() + line + "\n")
    with pytest.raises(ConfigError):
        load_settings(cfg, {"LOCALAPPDATA": str(tmp_path)})


def test_toml_int_accepted_for_float_and_integral_float_for_int(tmp_path: Path) -> None:
    cfg = _write(
        tmp_path / "c.toml", _paths_toml() + "vmaf_target = 94\nscratch_cap_bytes = 2e12\n"
    )
    s = load_settings(cfg, {"LOCALAPPDATA": str(tmp_path)})
    assert s.vmaf_target == 94.0
    assert s.scratch_cap_bytes == 2 * 10**12


def test_example_config_loads(tmp_path: Path) -> None:
    example = Path(__file__).parent.parent / "config.example.toml"
    s = load_settings(example, {"LOCALAPPDATA": str(tmp_path)})
    assert s.crf_min == 16
    assert "nas" in str(s.library_root)
    assert "\\\\" not in str(s.library_root).lstrip("\\")  # no doubled separators from escaping


def test_env_var_name_for_a_setting() -> None:
    from video_cleaner.config import env_var

    assert env_var("vmaf_target") == "VC_VMAF_TARGET"


@pytest.mark.parametrize(
    ("raw", "level"), [("DEBUG", "DEBUG"), ("info", "INFO"), ("Warning", "WARNING")]
)
def test_log_file_level_setting(tmp_path: Path, raw: str, level: str) -> None:
    s = load_settings(None, _env(tmp_path, VC_LOG_FILE_LEVEL=raw))
    assert s.log_file_level == level


def test_log_file_level_default_and_rejects_unknown(tmp_path: Path) -> None:
    assert load_settings(None, _env(tmp_path)).log_file_level == "DEBUG"
    with pytest.raises(ConfigError, match="log_file_level"):
        load_settings(None, _env(tmp_path, VC_LOG_FILE_LEVEL="LOUD"))


# --- heartbeat bound, absolute clean paths, BOM ------------------------------------


def test_heartbeat_s_is_at_most_15(tmp_path: Path) -> None:
    assert load_settings(None, _env(tmp_path, VC_HEARTBEAT_S="15")).heartbeat_s == 15.0
    with pytest.raises(ConfigError, match="heartbeat_s"):
        load_settings(None, _env(tmp_path, VC_HEARTBEAT_S="15.5"))


@pytest.mark.parametrize("bad", ["relative/library", "library"])
def test_path_settings_must_be_absolute(tmp_path: Path, bad: str) -> None:
    with pytest.raises(ConfigError, match=r"library_root.*absolute"):
        load_settings(None, _env(tmp_path, VC_LIBRARY_ROOT=bad))


def test_path_settings_reject_control_characters(tmp_path: Path) -> None:
    # What a TOML basic string makes of "D:\temp": a TAB instead of backslash-t.
    cfg = _write(
        tmp_path / "c.toml",
        _paths_toml().replace("scratch_dir = 'D:/vc/scratch'", 'scratch_dir = "D:\\temp"'),
    )
    env = _env(tmp_path)
    del env["VC_SCRATCH_DIR"]  # the file's value must be the one used
    with pytest.raises(ConfigError, match=r"scratch_dir.*control character"):
        load_settings(cfg, env)


def test_unc_and_drive_paths_are_absolute(tmp_path: Path) -> None:
    s = load_settings(None, _env(tmp_path, VC_LIBRARY_ROOT="\\\\nas\\media"))
    assert str(s.library_root).startswith("\\\\") or s.library_root.as_posix().startswith("//")


def test_config_file_with_utf8_bom_is_read(tmp_path: Path) -> None:
    cfg = tmp_path / "bom.toml"
    cfg.write_bytes(b"\xef\xbb\xbf" + (_paths_toml() + "vmaf_target = 94.0\n").encode("utf-8"))
    assert load_settings(cfg, _env(tmp_path)).vmaf_target == 94.0
