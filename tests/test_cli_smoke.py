from pathlib import Path

import pytest
from typer.testing import CliRunner

import video_cleaner
from video_cleaner.cli import app

PHASE1_COMMANDS = ("doctor", "survey", "plan", "calibrate", "run", "status", "report")


def test_help_lists_phase1_commands() -> None:
    out = CliRunner().invoke(app, ["--help"]).stdout
    for cmd in PHASE1_COMMANDS:
        assert cmd in out


def test_every_command_is_registered() -> None:
    names = {c.name or (c.callback.__name__ if c.callback else "") for c in app.registered_commands}
    assert names == set(PHASE1_COMMANDS)


@pytest.mark.parametrize("cmd", PHASE1_COMMANDS)
def test_no_command_is_a_stub(cmd: str, tmp_path: Path) -> None:
    # A missing explicit config stops every command at the config check, before any real work.
    result = CliRunner().invoke(app, [cmd, "--config", str(tmp_path / "missing.toml")])
    assert "not implemented" not in result.output
    assert result.exit_code == 2
    assert "config" in result.output


def test_version_is_a_string() -> None:
    assert isinstance(video_cleaner.__version__, str)
    assert video_cleaner.__version__


@pytest.mark.parametrize("cmd", ["survey", "calibrate", "run"])
def test_long_commands_offer_a_progress_flag(cmd: str) -> None:
    result = CliRunner().invoke(app, [cmd, "--help"], env={"COLUMNS": "200"})
    assert result.exit_code == 0
    assert "--progress" in result.stdout
    assert "auto" in result.stdout and "rich" in result.stdout and "plain" in result.stdout


def test_progress_flag_rejects_an_unknown_mode() -> None:
    result = CliRunner().invoke(app, ["run", "--progress", "fancy"])
    assert result.exit_code == 2
