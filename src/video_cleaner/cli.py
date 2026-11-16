"""Command-line interface for video-cleaner (Phase 1 command stubs)."""

import os
from pathlib import Path
from typing import Annotated, NoReturn

import typer
from rich.console import Console

from video_cleaner import doctor as doctor_mod
from video_cleaner.config import ConfigError, default_config_path, load_settings

app = typer.Typer(
    name="video-cleaner",
    help="Re-encode a video library to HEVC with x265, safely and non-destructively.",
    no_args_is_help=True,
    add_completion=False,
)

_err = Console(stderr=True)


def _not_implemented(command: str) -> NoReturn:
    _err.print(f"{command}: not implemented")
    raise typer.Exit(code=2)


def _check_tag(check: doctor_mod.Check) -> str:
    if check.ok:
        return "[ OK ]"
    return "[FAIL]" if check.required else "[WARN]"


@app.command()
def doctor(
    config: Annotated[
        Path | None,
        typer.Option(
            "--config", help=r"Config file (default %APPDATA%\video-cleaner\config.toml)."
        ),
    ] = None,
) -> None:
    """Check the environment (ffmpeg, libx265, libvmaf, paths)."""
    # Resolve once with config's own rule, so the permission check covers the loaded file.
    # The explicit/default distinction is kept: a missing default file is not an error.
    config_path = config if config is not None else default_config_path(os.environ)
    try:
        settings = load_settings(config, os.environ)
    except ConfigError as exc:
        _err.print(f"config error: {exc}", markup=False, highlight=False)
        raise typer.Exit(code=2) from None
    checks = doctor_mod.run_doctor(settings, config_path=config_path)
    for check in checks:
        typer.echo(f"{_check_tag(check)} {check.name}: {check.detail}")
    removed = doctor_mod.prune_logs(settings.log_dir)
    typer.echo(f"logs: removed {len(removed)} old log file(s) from {settings.log_dir}")
    if any(c.required and not c.ok for c in checks):
        raise typer.Exit(code=1)


@app.command()
def survey() -> None:
    """Scan the library and probe every file."""
    _not_implemented("survey")


@app.command()
def plan() -> None:
    """Decide what to do with each file and project savings."""
    _not_implemented("plan")


@app.command()
def calibrate() -> None:
    """Calibrate CRF selection on sample clips."""
    _not_implemented("calibrate")


@app.command()
def run() -> None:
    """Execute the acknowledged plan."""
    _not_implemented("run")


@app.command()
def status() -> None:
    """Show progress of the current or last run."""
    _not_implemented("status")


@app.command()
def report() -> None:
    """Summarise results of a run."""
    _not_implemented("report")
