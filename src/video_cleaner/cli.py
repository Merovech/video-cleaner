"""Command-line interface for video-cleaner."""

import logging
import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Annotated, NoReturn

import typer
from rich.console import Console

from video_cleaner import calibrate as _calibrate
from video_cleaner import doctor as doctor_mod
from video_cleaner import planning, power
from video_cleaner import runner as _runner
from video_cleaner import survey as _survey
from video_cleaner.config import ConfigError, Settings, default_config_path, load_settings
from video_cleaner.decide import Action
from video_cleaner.encode import X265_CRF_MAX
from video_cleaner.held import HELD_REASON
from video_cleaner.instancelock import AlreadyRunning
from video_cleaner.logging_setup import setup_logging
from video_cleaner.manifest import ManifestError, read_manifest
from video_cleaner.paths import OutputInLibrary, outputs_outside_library
from video_cleaner.progress import Reporter, RichReporter, encodable, make_reporter
from video_cleaner.state import State, StateError
from video_cleaner.tools import ToolNotFound, Tools, locate_tools

app = typer.Typer(
    name="video-cleaner",
    help="Re-encode a video library to HEVC with x265, safely and non-destructively.",
    no_args_is_help=True,
    add_completion=False,
)

_err = Console(stderr=True, emoji=False)

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
LogFileOption = Annotated[
    Path | None,
    typer.Option("--log-file", help="Write this run's log here instead of log_dir."),
]


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
    settings = _load(config)
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


def _load(config: Path | None) -> Settings:
    """Settings from *config*, or exit 2 with the configuration error."""
    try:
        return load_settings(config)
    except ConfigError as exc:
        _fail(f"config error: {exc}")


def _logging(settings: Settings, verbose: int, quiet: bool, log_file: Path | None) -> Path:
    """Start the run log; ``--quiet`` wins over ``-v``. Returns the log file.

    A ``--log-file`` under the library root is refused (exit 2) before it is created.
    """
    if log_file is not None:
        try:
            outputs_outside_library(settings, extra=(("log file", log_file),))
        except OutputInLibrary as exc:
            _fail(str(exc))
    return setup_logging(
        settings.log_dir,
        verbosity=-1 if quiet else verbose,
        log_file=log_file,
        file_level=logging.getLevelName(settings.log_file_level),
    )


def _tools() -> Tools:
    """The located ffmpeg, ffprobe and mkvmerge, or exit 2 naming the missing one."""
    try:
        return locate_tools()
    except ToolNotFound as exc:
        _fail(str(exc))


@contextmanager
def _open_reporter(settings: Settings) -> Iterator[Reporter]:
    """The console reporter for a long command, closed (live display stopped) on the way out."""
    reporter = make_reporter(settings, is_tty=sys.stdout.isatty())
    try:
        yield reporter
    finally:
        if isinstance(reporter, RichReporter):
            reporter.close()


def _say(text: str = "") -> None:
    """Print a plain line to stdout; file names never break a narrow-encoding console."""
    typer.echo(encodable(text, sys.stdout))


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
    log_file: LogFileOption = None,
) -> None:
    """Scan the library and probe every file (read-only)."""
    settings = _load(config)
    surveyed = settings.library_root if root is None else root
    try:
        _survey.check_outputs_outside(settings, surveyed)
    except _survey.SurveyError as exc:
        _fail(str(exc))
    _logging(settings, verbose, quiet, log_file)
    tools = _tools()
    with _open_reporter(settings) as reporter:
        try:
            with State.open(settings.state_db) as state:
                result = _survey.run_survey(settings, tools, state, reporter, root=root)
        except (_survey.SurveyError, ToolNotFound) as exc:
            _fail(str(exc))
        except KeyboardInterrupt:
            _err.print(
                "survey interrupted; files are recorded only once all of them are probed, so "
                "nothing from this survey was recorded: run it again",
                highlight=False,
            )
            raise typer.Exit(code=EXIT_INTERRUPTED) from None
    _print_survey_summary(surveyed, result)


def _load_outside_library(config: Path | None) -> Settings:
    """Settings, refusing (exit 2) any output location under the library root.

    Called by every command before it creates a log file, the state database or any output.
    """
    settings = _load(config)
    try:
        _runner.refuse_outputs_in_library(settings)
    except _runner.RunRefused as exc:
        _fail(str(exc))
    return settings


def _require_state(settings: Settings) -> None:
    if not settings.state_db.is_file():
        _fail(f"no state database at {settings.state_db}: run `survey` first")


MAX_LISTED_FLAGGED = 20
HASH_PREFIX = 12


def _print_plan_summary(report: planning.PlanReport, csv_path: Path, digest: str) -> None:
    _say(f"Plan: {len(report.rows)} files, {_gib(report.total_bytes)}")
    for action in Action:
        _say(f"  {action.value:<12} {report.counts.get(action.value, 0)}")
    _say(f"Estimated saving still to do: {_gib(report.est_saving_bytes)}")
    _say(f"Multi-part groups: {len(report.groups)}")
    for group in report.groups:
        _say(f"  {group.output_name} ({group.action}): {', '.join(p.name for p in group.parts)}")
    flagged = [r for r in report.rows if r.action == Action.FLAG.value]
    _say(f"Flagged: {len(flagged)}")
    for row in flagged[:MAX_LISTED_FLAGGED]:
        _say(f"  {row.path}: {planning.LIST_SEPARATOR.join(row.reasons)}")
    if len(flagged) > MAX_LISTED_FLAGGED:
        _say(f"  ... {len(flagged) - MAX_LISTED_FLAGGED} more in the CSV")
    for note in report.notes:
        _say(f"Note: {note}")
    _say(f"CSV: {csv_path}")
    _say(f"Plan hash: {digest}")
    _say("Review the CSV, then acknowledge it with `video-cleaner plan --ack`.")


@app.command()
def plan(
    ack: Annotated[
        bool, typer.Option("--ack", help="Acknowledge the plan last written by `plan`.")
    ] = False,
    retry: Annotated[
        Path | None,
        typer.Option(
            "--retry",
            help="Put a file a run left alone or flagged back in the queue (surveyed path).",
        ),
    ] = None,
    config: ConfigOption = None,
    verbose: VerboseOption = 0,
    quiet: QuietOption = False,
    log_file: LogFileOption = None,
) -> None:
    """Write the plan (what happens to each file, and why) or acknowledge it."""
    settings = _load_outside_library(config)
    _require_state(settings)
    _logging(settings, verbose, quiet, log_file)
    if retry is not None:
        if ack:
            _fail("give either --ack or --retry, not both")
        try:
            with State.open(settings.state_db) as state:
                rec = state.requeue(retry)
        except KeyError as exc:
            _fail(str(exc.args[0]))
        except StateError as exc:
            _fail(str(exc))
        _say(f"Requeued {rec.path}: {rec.status.value} ({rec.decision})")
        return
    csv_path = settings.state_db.parent / planning.PLAN_CSV
    hash_path = settings.state_db.parent / planning.PLAN_HASH
    try:
        with State.open(settings.state_db) as state:
            report = planning.build_plan(state)
            if ack:
                current = planning.plan_hash(report)
                if planning.read_plan_hash(hash_path) != current:
                    _fail("run `plan` first and review it (the plan changed or was not written)")
                if not report.rows:
                    _fail("the plan is empty: run `survey` first")
                state.acknowledge_plan(current)
                _say(f"Plan acknowledged: {current}")
                return
            digest = planning.write_plan_report(report, csv_path)
            planning.write_plan_hash(hash_path, digest)
    except (planning.PlanError, StateError) as exc:
        _fail(str(exc))
    _print_plan_summary(report, csv_path, digest)


CALIBRATION_HELD_MESSAGE = (
    f"{HELD_REASON}: calibration needs real encodes; it will run once the hold is lifted"
)


def _approve_calibration(
    settings: Settings, vmaf_target: float | None, crf_max: int | None
) -> None:
    if vmaf_target is None or crf_max is None:
        _fail("--approve needs both --vmaf-target and --crf-max")
    low, high = _calibrate.VMAF_TARGET_RANGE
    if not low <= vmaf_target <= high:
        _fail(f"--vmaf-target must be between {low:g} and {high:g}, got {vmaf_target}")
    if vmaf_target < settings.guard_mean_floor:
        _fail(
            f"--vmaf-target {vmaf_target:g} is below guard_mean_floor "
            f"{settings.guard_mean_floor:g}: the quality guard would reject every encode"
        )
    if not settings.crf_min <= crf_max <= X265_CRF_MAX:
        _fail(
            f"--crf-max must be between crf_min ({settings.crf_min}) and "
            f"{X265_CRF_MAX}, got {crf_max}"
        )
    _require_state(settings)  # never creates an empty database
    try:
        with State.open(settings.state_db) as state:
            if state.calibration_swept() is None:
                _fail(
                    "no calibration sweep has been run: run `calibrate --samples N` and review "
                    "the table before approving"
                )
            state.approve_calibration(vmaf_target, crf_max)
    except StateError as exc:
        _fail(str(exc))
    _say(f"Calibration approved: vmaf_target {vmaf_target:g}, crf_max {crf_max}")
    _say(f"Stored in {settings.state_db}; `run` uses these values instead of the config ones.")


@app.command()
def calibrate(
    samples: Annotated[
        int | None, typer.Option("--samples", help="Encode N sample files at CRF 17/20/23/26.")
    ] = None,
    approve: Annotated[
        bool, typer.Option("--approve", help="Approve a VMAF target and CRF ceiling for runs.")
    ] = False,
    vmaf_target: Annotated[
        float | None,
        typer.Option(
            "--vmaf-target", help="With --approve: VMAF target (guard_mean_floor to 100)."
        ),
    ] = None,
    crf_max: Annotated[
        int | None, typer.Option("--crf-max", help="With --approve: highest CRF a run may use.")
    ] = None,
    config: ConfigOption = None,
    verbose: VerboseOption = 0,
    quiet: QuietOption = False,
    log_file: LogFileOption = None,
) -> None:
    """Calibrate CRF selection on sample clips, or approve the calibration."""
    settings = _load_outside_library(config)
    if approve:
        if samples is not None:
            _fail("give either --samples N or --approve, not both")
        _approve_calibration(settings, vmaf_target, crf_max)
        return
    if vmaf_target is not None or crf_max is not None:
        _fail("--vmaf-target and --crf-max are only used with --approve")
    if samples is None:
        _fail("give --samples N to run the calibration, or --approve to approve one")
    if samples < 1:
        _fail(f"--samples must be at least 1, got {samples}")
    engine = _calibrate.default_engine()
    if engine.held_stages():
        typer.echo(CALIBRATION_HELD_MESSAGE, err=True)
        raise typer.Exit(code=1)
    _require_state(settings)
    _logging(settings, verbose, quiet, log_file)
    tools = _tools()
    with _open_reporter(settings) as reporter:
        try:
            with State.open(settings.state_db) as state:
                rows = _calibrate.run_calibration(
                    settings, tools, state, reporter, samples=samples, engine=engine
                )
        except (
            _calibrate.CalibrationError,
            _runner.RunRefused,
            AlreadyRunning,
            StateError,
        ) as exc:
            _fail(str(exc))
        except KeyboardInterrupt:
            _err.print("calibration interrupted; nothing was approved", highlight=False)
            raise typer.Exit(code=EXIT_INTERRUPTED) from None
    _say(_calibrate.format_calibration_table(rows))
    _say(f"Table: {settings.state_db.parent / _calibrate.CALIBRATION_TXT}")
    _say(
        f"Side-by-side clips: {settings.completed_dir / _calibrate.CLIPS_FOLDER} "
        "(newest folder: this sweep)"
    )
    _say("Approve with `video-cleaner calibrate --approve --vmaf-target X --crf-max Y`.")


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
    log_file: LogFileOption = None,
) -> None:
    """Execute the acknowledged plan."""
    settings = _load_outside_library(config)
    _require_state(settings)  # never creates an empty database
    try:
        deadline = _runner.deadline_from(until, max_hours, datetime.now())
    except ValueError as exc:
        _fail(str(exc))
    log_path = _logging(settings, verbose, quiet, log_file)
    tools = _tools()
    with _open_reporter(settings) as reporter, power.keep_awake():
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
        except (
            _runner.PlanNotAcknowledged,
            _runner.CalibrationNotApproved,
            _runner.RunRefused,
            AlreadyRunning,
            StateError,
        ) as exc:
            _fail(str(exc))
        except KeyboardInterrupt:
            _err.print(
                "run interrupted; the file in progress is retried on the next run",
                highlight=False,
            )
            raise typer.Exit(code=EXIT_INTERRUPTED) from None
    _print_run_summary(summary)
    _say(f"Log: {log_path}")
    if summary.stop_reason:
        raise typer.Exit(code=1)


LAST_EVENTS = 10


@app.command()
def status(config: ConfigOption = None) -> None:
    """Show file counts, plan and calibration approval, and the latest events."""
    settings = _load_outside_library(config)
    if not settings.state_db.is_file():
        _say(f"No state database at {settings.state_db} yet: run `survey` first.")
        return
    try:
        with State.open(settings.state_db) as state:
            counts, members = planning.status_counts(state)
            acknowledged = state.plan_acknowledged()
            current = planning.plan_hash(planning.build_plan(state))
            calibration = state.calibration()
            events = state.events(LAST_EVENTS)
    except StateError as exc:
        _fail(str(exc))
    _say(f"Files: {sum(counts.values())} (stale history left out)")
    for st, n in counts.items():
        _say(f"  {st.value:<12} {n}")
    _say(f"Multi-part members: {members} (counted with their group)")
    if acknowledged is None:
        _say("Plan acknowledged: no (run `plan`, review it, then `plan --ack`)")
    else:
        match = (
            "matches the current plan"
            if acknowledged == current
            else "the plan changed since: run `plan` and acknowledge it again"
        )
        _say(f"Plan acknowledged: yes ({acknowledged[:HASH_PREFIX]}), {match}")
    if calibration is None:
        _say("Calibration approved: no (run `calibrate`)")
    else:
        vmaf_target, crf_max = calibration
        _say(f"Calibration approved: yes (vmaf_target {vmaf_target:g}, crf_max {crf_max})")
    _say(f"Last {len(events)} event(s), newest first:")
    for line in events:
        _say(f"  {line}")


def _print_results(results: planning.ResultsReport, csv_path: Path) -> None:
    _say(f"Completed: {len(results.entries)}")
    _say(f"Saved: {_gib(results.bytes_saved)}")
    if results.unchanged:
        tally = {v: results.unchanged.count(v) for v in planning.UNCHANGED_VERDICTS}
        _say(
            "Original unchanged: "
            + ", ".join(f"{verdict} {n}" for verdict, n in tally.items())
            + " (only a 'yes' entry may replace its original)"
        )
        for entry, verdict in zip(results.entries, results.unchanged, strict=True):
            _say(f"  {verdict}: {entry.completed}")
    _say(f"Left alone: {results.left_alone}")
    if results.replace:
        _say(f"Replace (Dolby Vision profile 5, no HDR10 fallback): {len(results.replace)}")
        for path in results.replace:
            _say(f"  {path}")
    _say(f"Flagged: {len(results.flagged)}")
    for path, reason in results.flagged:
        _say(f"  {path}: {reason}")
    _say(f"Retry: {len(results.retry)}")
    for path, reason in results.retry:
        _say(f"  {path}: {reason}")
    _say(f"CSV: {csv_path}")


@app.command()
def report(
    config: ConfigOption = None,
    output: Annotated[
        Path | None,
        typer.Option("--output", help="Results CSV (default report.csv beside the state DB)."),
    ] = None,
) -> None:
    """Write a CSV of the results so far and print a summary."""
    settings = _load_outside_library(config)
    csv_path = settings.state_db.parent / planning.REPORT_CSV if output is None else output
    try:
        outputs_outside_library(settings, extra=(("report", csv_path),))
    except OutputInLibrary as exc:
        _fail(str(exc))
    _require_state(settings)
    try:
        entries = read_manifest(settings.completed_dir / _runner.MANIFEST_NAME)
    except FileNotFoundError:
        entries = []
    except (ManifestError, OSError) as exc:
        _fail(f"manifest unreadable: {exc}")
    try:
        with State.open(settings.state_db) as state:
            results = planning.build_results(entries, state, library_root=settings.library_root)
        planning.write_results_csv(results, csv_path)
    except (planning.PlanError, StateError) as exc:
        _fail(str(exc))
    _print_results(results, csv_path)
