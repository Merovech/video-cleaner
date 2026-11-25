import re
from pathlib import Path

import pytest
from typer.testing import CliRunner

import video_cleaner
from video_cleaner.cli import app

COMMANDS = ("doctor", "survey", "plan", "calibrate", "run", "status", "report", "auth")
# Command groups (a sub-app with commands of its own).
GROUPS = ("subs",)


def test_help_lists_every_command() -> None:
    out = CliRunner().invoke(app, ["--help"]).stdout
    for cmd in (*COMMANDS, *GROUPS):
        assert cmd in out


def test_every_command_is_registered() -> None:
    names = {c.name or (c.callback.__name__ if c.callback else "") for c in app.registered_commands}
    assert names == set(COMMANDS)
    assert {g.name for g in app.registered_groups} == set(GROUPS)


@pytest.mark.parametrize("sub", ["resolve", "status", "retry"])
def test_no_subs_command_is_a_stub(sub: str, tmp_path: Path) -> None:
    args = ["subs", sub, "--config", str(tmp_path / "missing.toml")]
    result = CliRunner().invoke(app, [*args, "--path", "x.mkv"] if sub == "retry" else args)
    assert "not implemented" not in result.output
    assert result.exit_code == 2
    assert "config" in result.output


@pytest.mark.parametrize("cmd", COMMANDS)
def test_no_command_is_a_stub(cmd: str, tmp_path: Path) -> None:
    # A missing explicit config stops every command at the config check, before any real work.
    result = CliRunner().invoke(app, [cmd, "--config", str(tmp_path / "missing.toml")])
    assert "not implemented" not in result.output
    assert result.exit_code == 2
    assert "config" in result.output


def test_version_is_a_string() -> None:
    assert isinstance(video_cleaner.__version__, str)
    assert video_cleaner.__version__


@pytest.mark.parametrize("cmd", ["survey", "calibrate", "run", "subs resolve"])
def test_long_commands_offer_a_progress_flag(cmd: str) -> None:
    # CI sets colour variables, so the help may carry ANSI styling that splits "--progress".
    result = CliRunner().invoke(
        app, [*cmd.split(), "--help"], env={"COLUMNS": "200", "NO_COLOR": "1"}
    )
    assert result.exit_code == 0
    text = re.sub(r"\x1b\[[0-9;]*m", "", result.stdout)
    assert "--progress" in text
    assert "auto" in text and "rich" in text and "plain" in text


def test_progress_flag_rejects_an_unknown_mode() -> None:
    result = CliRunner().invoke(app, ["run", "--progress", "fancy"])
    assert result.exit_code == 2
