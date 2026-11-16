"""Command-line interface for video-cleaner."""

import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Annotated, NoReturn

import typer
from rich.console import Console

from video_cleaner import doctor as doctor_mod
from video_cleaner import runner as _runner
from video_cleaner import survey as _survey
from video_cleaner.config import ConfigError, default_config_path, load_settings
from video_cleaner.decide import Action
from video_cleaner.logging_setup import setup_logging
from video_cleaner.progress import RichReporter, _encodable, make_reporter
from video_cleaner.state import State, StateError
from video_cleaner.tools import ToolNotFound, locate_tools

app = typer.Typer(
    name="video-cleaner",
    help="Re-encode a video library to HEVC with x265, safely and non-destructively.",
    no_args_is_help=True,
    add_completion=False,
)

_err = Console(stderr=True)

# Unprobeable files listed in the survey summary; the CSV has all of them.
MAX_LISTED_UNPROBEABLE = 20
EXIT_INTERRUPTED = 130  # 128 + SIGINT, the shell convention for Ctrl+C

RootOption = Annotated[
    Path | None, typer.Option("--root", help="Survey this folder instead of library_root.")
]
ConfigOption = Annotated[
    Path | None,
    typer.Option("--config", help=r"Config file (default %APPDATA%\video-cleaner\config.toml)."),
]
VerboseOption = Annotated[
    int, typer.Option("--verbose", "-v", count=True, help="More console logging.")
]
QuietOption = Annotated[bool, typer.Option("--quiet", "-q", help="Warnings only on the console.")]


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


def _fail(message: str) -> NoReturn:
    _err.print(message, markup=False, highlight=False)
    raise typer.Exit(code=2)


def _say(text: str = "") -> None:
    """Print a plain line to stdout; file names never break a narrow-encoding console."""
    typer.echo(_encodable(text, sys.stdout))


def _gib(n: int) -> str:
    return f"{n / 1024**3:.1f} GiB"


def _print_survey_summary(root: Path, result: _survey.SurveyResult) -> None:
    files = sum(result.counts.values())
    _say(f"Survey of {root}: {files} files, {_gib(result.total_bytes)}")
    for action in Action:
        _say(f"  {action.value:<12} {result.counts.get(action, 0)}")
    _say(f"Estimated saving: {_gib(result.est_saving_bytes)} (placeholder until calibration)")
    _say(
        f"Flagged: {result.counts.get(Action.FLAG, 0)} "
        f"(unprobeable: {len(result.unprobeable)}, "
        f"unrepresentable: {len(result.unrepresentable)})"
    )
    for path, reason in result.unprobeable[:MAX_LISTED_UNPROBEABLE]:
        _say(f"  unprobeable: {path}: {reason}")
    if len(result.unprobeable) > MAX_LISTED_UNPROBEABLE:
        _say(f"  ... {len(result.unprobeable) - MAX_LISTED_UNPROBEABLE} more in the CSV")
    for path in result.unrepresentable:
        _say(f"  unrepresentable: {path!r}")
    _say(f"Multi-part groups: {len(result.groups)}")
    for group in result.groups:
        _say(f"  {group.output_name}: {', '.join(p.name for p in group.parts)}")
    _say(f"CSV: {result.csv_path}")


@app.command()
def survey(
    root: RootOption = None,
    config: ConfigOption = None,
    verbose: VerboseOption = 0,
    quiet: QuietOption = False,
) -> None:
    """Scan the library and probe every file (read-only)."""
    try:
        settings = load_settings(config)
    except ConfigError as exc:
        _fail(f"config error: {exc}")
    surveyed = settings.library_root if root is None else root
    try:
        _survey.check_outputs_outside(settings, surveyed)
    except _survey.SurveyError as exc:
        _fail(str(exc))
    setup_logging(settings.log_dir, verbosity=-1 if quiet else verbose)
    try:
        tools = locate_tools()
    except ToolNotFound as exc:
        _fail(str(exc))
    reporter = make_reporter(settings, is_tty=sys.stdout.isatty())
    try:
        with State.open(settings.state_db) as state:
            result = _survey.run_survey(settings, tools, state, reporter, root=root)
    except _survey.SurveyError as exc:
        _fail(str(exc))
    except ToolNotFound as exc:
        _fail(str(exc))
    except KeyboardInterrupt:
        _err.print("survey interrupted; files already surveyed stay recorded", highlight=False)
        raise typer.Exit(code=EXIT_INTERRUPTED) from None
    finally:
        if isinstance(reporter, RichReporter):
            reporter.close()
    _print_survey_summary(surveyed, result)


@app.command()
def plan() -> None:
    """Decide what to do with each file and project savings."""
    _not_implemented("plan")


@app.command()
def calibrate() -> None:
    """Calibrate CRF selection on sample clips."""
    _not_implemented("calibrate")


def _print_run_summary(summary: _runner.RunSummary) -> None:
    _say(
        f"Run: completed {summary.completed}, left alone {summary.left_alone}, "
        f"flagged {summary.flagged}, retry {summary.retry}; saved {_gib(summary.bytes_saved)}"
    )
    if summary.stop_reason:
        _say(f"Run stopped: {summary.stop_reason}")
    elif summary.stopped_by_deadline:
        _say("Run ended: deadline reached (the file in progress was finished)")
    else:
        _say("Run ended: queue finished")


@app.command()
def run(
    until: Annotated[
        str | None,
        typer.Option("--until", help="Start no new file after this time of day (HH:MM)."),
    ] = None,
    max_hours: Annotated[
        float | None, typer.Option("--max-hours", help="Start no new file after N hours.")
    ] = None,
    path: Annotated[
        Path | None, typer.Option("--path", help="Only process this file or folder.")
    ] = None,
    config: ConfigOption = None,
    verbose: VerboseOption = 0,
    quiet: QuietOption = False,
) -> None:
    """Execute the acknowledged plan."""
    try:
        settings = load_settings(config)
    except ConfigError as exc:
        _fail(f"config error: {exc}")
    try:
        deadline = _runner.deadline_from(until, max_hours, datetime.now())
    except ValueError as exc:
        _fail(str(exc))
    log_path = setup_logging(settings.log_dir, verbosity=-1 if quiet else verbose)
    try:
        tools = locate_tools()
    except ToolNotFound as exc:
        _fail(str(exc))
    reporter = make_reporter(settings, is_tty=sys.stdout.isatty())
    try:
        with State.open(settings.state_db) as state:
            summary = _runner.run_queue(
                settings,
                tools,
                state,
                reporter,
                deadline=deadline,
                only=path,
                current_log=log_path,
            )
    except (_runner.PlanNotAcknowledged, _runner.RunRefused, StateError) as exc:
        _fail(str(exc))
    except KeyboardInterrupt:
        _err.print(
            "run interrupted; the file in progress is retried on the next run", highlight=False
        )
        raise typer.Exit(code=EXIT_INTERRUPTED) from None
    finally:
        if isinstance(reporter, RichReporter):
            reporter.close()
    _print_run_summary(summary)
    _say(f"Log: {log_path}")
    if summary.stop_reason:
        raise typer.Exit(code=1)


@app.command()
def status() -> None:
    """Show progress of the current or last run."""
    _not_implemented("status")


@app.command()
def report() -> None:
    """Summarise results of a run."""
    _not_implemented("report")
