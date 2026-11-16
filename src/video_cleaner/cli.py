"""Command-line interface for video-cleaner (Phase 1 command stubs)."""

from typing import NoReturn

import typer
from rich.console import Console

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


@app.command()
def doctor() -> None:
    """Check the environment (ffmpeg, libx265, libvmaf, paths)."""
    _not_implemented("doctor")


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
