import pytest
from typer.testing import CliRunner

import video_cleaner
from video_cleaner.cli import app


def test_help_lists_phase1_commands() -> None:
    out = CliRunner().invoke(app, ["--help"]).stdout
    for cmd in ("doctor", "survey", "plan", "calibrate", "run", "status", "report"):
        assert cmd in out


@pytest.mark.parametrize("cmd", ["plan", "calibrate", "status", "report"])
def test_stub_commands_exit_2_not_implemented(cmd: str) -> None:
    result = CliRunner().invoke(app, [cmd])
    assert result.exit_code == 2
    assert "not implemented" in result.output


def test_version_is_a_string() -> None:
    assert isinstance(video_cleaner.__version__, str)
    assert video_cleaner.__version__
