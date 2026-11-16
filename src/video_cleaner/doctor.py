"""Environment self-check (`video-cleaner doctor`) and log retention (spec sections 4.2-4.4).

Every external command goes through an injectable *runner* so tests never start a process.
No check raises: a problem, including an unexpected exception inside a check, becomes a
`Check` with ``ok=False``. The doctor only reads, except for a probe file it creates and
deletes in ``completed_dir`` and ``scratch_dir``; it never writes under ``library_root``.
"""

from __future__ import annotations

import importlib.util
import os
import re
import shutil
import stat
import subprocess  # nosec B404
import sys
import tempfile
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from video_cleaner.config import Settings, default_config_path
from video_cleaner.paths import is_under, path_key
from video_cleaner.tools import ToolNotFound, _locate

Runner = Callable[[Sequence[str]], tuple[int, str]]
DiskUsageFn = Callable[[Path], tuple[int, int, int]]  # (total, used, free), as shutil
FileModeFn = Callable[[Path], int]

RUNNER_TIMEOUT_S = 20.0
SUPPORTED_PYTHON = ((3, 11), (3, 12))
LOG_GLOB = "run-*.log"

# Used when SystemRoot is not set (it always is on a normal Windows session).
DEFAULT_SYSTEM_ROOT = "C:/Windows"

# Principals that mean "other users on this machine" and read-type rights (icacls notation).
_BROAD_PRINCIPALS = ("Everyone", "BUILTIN\\Users", "NT AUTHORITY\\Authenticated Users")
_READ_RIGHTS = frozenset({"F", "M", "RX", "R", "GR", "GA", "RD"})
_ACE = re.compile(
    r"(?P<who>"
    + "|".join(re.escape(p) for p in _BROAD_PRINCIPALS)
    + r"):(?P<rights>(?:\([^)]*\))+)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Check:
    name: str
    ok: bool
    detail: str
    required: bool = True


def default_runner(cmd: Sequence[str]) -> tuple[int, str]:
    """Run *cmd* (argument list, no shell) and return (exit code, stdout + stderr)."""
    # Fixed argument list, shell=False.
    proc = subprocess.run(  # noqa: S603  # nosec B603
        [str(part) for part in cmd],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=RUNNER_TIMEOUT_S,
        check=False,
        shell=False,
    )
    return proc.returncode, proc.stdout + proc.stderr


def _disk_usage(path: Path) -> tuple[int, int, int]:
    usage = shutil.disk_usage(path)
    return usage.total, usage.used, usage.free


def _file_mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def _gb(n: int) -> str:
    return f"{n / 10**9:,.0f} GB"


def _first_line(text: str) -> str:
    return next((line.strip() for line in text.splitlines() if line.strip()), "")


def _listed_names(output: str) -> set[str]:
    """Names from ``ffmpeg -encoders`` / ``-filters`` lines: the token after the flags column."""
    names = set()
    for line in output.splitlines():
        parts = line.split()
        if len(parts) >= 2:
            names.add(parts[1])
    return names


def loose_acl_entries(icacls_output: str) -> list[str]:
    """Return ``principal:(rights)`` entries that let broad groups read the file."""
    loose = []
    for match in _ACE.finditer(icacls_output):
        tokens = {
            token.strip().upper()
            for group in re.findall(r"\(([^)]*)\)", match["rights"])
            for token in group.split(",")
        }
        if "DENY" in tokens:
            continue
        if tokens & _READ_RIGHTS:
            loose.append(f"{match['who']}:{match['rights']}")
    return loose


def _existing_ancestor(path: Path) -> Path:
    current = path.absolute()
    while not current.exists() and current.parent != current:
        current = current.parent
    return current


class _Doctor:
    def __init__(
        self,
        settings: Settings,
        env: Mapping[str, str],
        runner: Runner,
        disk_usage: DiskUsageFn,
        config_path: Path | None,
        python_version: tuple[int, int],
        windows: bool,
        file_mode: FileModeFn,
    ) -> None:
        self.settings = settings
        self.env = env
        self.runner = runner
        self.disk_usage = disk_usage
        self.config_path = config_path
        self.python_version = python_version
        self.windows = windows
        self.file_mode = file_mode
        self.ffmpeg: Path | None = None

    def _run(self, cmd: Sequence[str]) -> tuple[int, str]:
        """The runner, with launch failures turned into a non-zero result."""
        try:
            return self.runner(cmd)
        except (OSError, subprocess.SubprocessError) as exc:
            return -1, f"{type(exc).__name__}: {exc}"

    def _system_tool(self, name: str) -> str:
        """*name* on POSIX (PATH only); on Windows the full path in ``System32``, because a bare
        name is also looked up in the current directory, where a planted copy would run."""
        if not self.windows:
            return name
        root = self.env.get("SystemRoot") or DEFAULT_SYSTEM_ROOT
        return str(Path(root) / "System32" / f"{name}.exe")

    def python(self) -> Iterator[Check]:
        major, minor = self.python_version
        version = f"{major}.{minor}"
        ok = (major, minor) in SUPPORTED_PYTHON
        yield Check("python", ok, version if ok else f"{version} (3.11 or 3.12 required)")

    def _tool(self, name: str) -> tuple[Check, Path | None]:
        try:
            path = _locate(name, self.env)
        except ToolNotFound as exc:
            return Check(name, False, str(exc)), None
        return Check(name, True, str(path)), path

    def ffmpeg_and_tools(self) -> Iterator[Check]:
        check, self.ffmpeg = self._tool("ffmpeg")
        yield check
        yield from self.ffmpeg_features()
        for name in ("ffprobe", "mkvmerge"):
            yield self._tool(name)[0]

    def ffmpeg_features(self) -> Iterator[Check]:
        wanted = (
            ("libx265", "-encoders", True),
            ("libvmaf", "-filters", True),
            ("hevc_nvenc", "-encoders", False),
        )
        if self.ffmpeg is None:
            for name, _, required in wanted:
                yield Check(name, False, "cannot check: ffmpeg not found", required)
            return
        listings: dict[str, tuple[int, str]] = {}
        for flag in ("-encoders", "-filters"):
            listings[flag] = self._run([str(self.ffmpeg), "-hide_banner", flag])
        for name, flag, required in wanted:
            code, output = listings[flag]
            if code != 0:
                detail = f"ffmpeg {flag} failed (exit {code}): {_first_line(output)}"
                yield Check(name, False, detail, required)
            elif name in _listed_names(output):
                yield Check(name, True, f"available in ffmpeg {flag[1:]}", required)
            else:
                yield Check(name, False, f"not in ffmpeg {flag[1:]}", required)

    def gpu(self) -> Iterator[Check]:
        code, output = self._run(
            [
                self._system_tool("nvidia-smi"),
                "--query-gpu=name,memory.total",
                "--format=csv,noheader",
            ]
        )
        line = _first_line(output)
        if code == 0 and line:
            yield Check("gpu", True, line, required=False)
        else:
            yield Check("gpu", False, f"no NVIDIA GPU visible ({line or f'exit {code}'})", False)

    def scratch_space(self) -> Iterator[Check]:
        cap = self.settings.scratch_cap_bytes
        where = _existing_ancestor(self.settings.scratch_dir)
        free = self.disk_usage(where)[2]
        detail = f"{_gb(free)} free on {where.anchor or where}, cap {_gb(cap)}"
        yield Check("scratch space", free >= cap, detail)

    def library_root(self) -> Iterator[Check]:
        root = self.settings.library_root
        try:
            with os.scandir(root) as entries:
                next(entries, None)  # one entry is enough; a NAS listing can be large
        except OSError as exc:
            yield Check("library_root", False, f"{root} not readable: {exc}")
            return
        yield Check("library_root", True, f"{root} readable")

    def writable_dirs(self) -> Iterator[Check]:
        for name in ("completed_dir", "scratch_dir"):
            directory: Path = getattr(self.settings, name)
            yield self._writable(name, directory)

    def _writable(self, name: str, directory: Path) -> Check:
        if is_under(directory, self.settings.library_root):
            return Check(name, False, f"{directory} is inside library_root; it must be local")
        try:
            directory.mkdir(parents=True, exist_ok=True)
            fd, probe = tempfile.mkstemp(prefix=".vc-doctor-", dir=directory)
            os.close(fd)
            Path(probe).unlink()
        except OSError as exc:
            return Check(name, False, f"{directory} not writable: {exc}")
        return Check(name, True, f"{directory} writable")

    def config_permissions(self) -> Iterator[Check]:
        name = "config permissions"
        path = self.config_path
        if path is None or not path.is_file():
            yield Check(name, True, "no config file (defaults and VC_* only)", required=False)
            return
        if self.windows:
            code, output = self._run([self._system_tool("icacls"), str(path)])
            if code != 0:
                detail = f"icacls failed (exit {code}): {_first_line(output)}"
                yield Check(name, False, detail, required=False)
                return
            loose = loose_acl_entries(output)
            if loose:
                detail = f"{path} readable by {', '.join(loose)}; restrict it with icacls"
                yield Check(name, False, detail, required=False)
                return
        else:
            mode = self.file_mode(path)
            if mode & 0o077:
                detail = f"{path} mode {mode:o}; run chmod 600"
                yield Check(name, False, detail, required=False)
                return
        yield Check(name, True, f"{path} restricted to the owner", required=False)

    def keyring(self) -> Iterator[Check]:
        found = importlib.util.find_spec("keyring") is not None
        detail = "importable" if found else "not installed (needed from Phase 2 for credentials)"
        yield Check("keyring", found, detail, required=False)


def run_doctor(
    settings: Settings,
    *,
    env: Mapping[str, str] | None = None,
    runner: Runner | None = None,
    disk_usage: DiskUsageFn | None = None,
    config_path: Path | None = None,
    python_version: tuple[int, int] | None = None,
    windows: bool | None = None,
    file_mode: FileModeFn | None = None,
) -> list[Check]:
    """Run every environment check and return the results; never raises.

    *config_path* defaults to `config.default_config_path` for *env*, the file `load_settings`
    reads when it is given no path. The other
    keyword arguments exist for tests and default to the real system.
    """
    environ: Mapping[str, str] = os.environ if env is None else env
    doc = _Doctor(
        settings,
        environ,
        runner or default_runner,
        disk_usage or _disk_usage,
        config_path if config_path is not None else default_config_path(environ),
        python_version or (sys.version_info.major, sys.version_info.minor),
        os.name == "nt" if windows is None else windows,
        file_mode or _file_mode,
    )
    groups: list[tuple[str, Callable[[], Iterator[Check]]]] = [
        ("python", doc.python),
        ("ffmpeg", doc.ffmpeg_and_tools),
        ("gpu", doc.gpu),
        ("scratch space", doc.scratch_space),
        ("library_root", doc.library_root),
        ("output dirs", doc.writable_dirs),
        ("config permissions", doc.config_permissions),
        ("keyring", doc.keyring),
    ]
    checks: list[Check] = []
    for label, group in groups:
        try:
            checks.extend(group())
        except Exception as exc:  # a broken check must not hide the others
            checks.append(Check(label, False, f"check crashed: {type(exc).__name__}: {exc}"))
    return checks


def prune_logs(
    log_dir: Path,
    *,
    keep_days: int = 30,
    max_bytes: int = 200 * 10**6,
    now: float | None = None,
    exclude: Path | None = None,
) -> list[Path]:
    """Delete ``run-*.log`` files older than *keep_days*, then the oldest until the total
    size is at most *max_bytes*. Other files are never touched, nor is *exclude* (the current
    run's log). Files that cannot be deleted (for example a log still open) are skipped.
    Returns the deleted paths; never raises for a missing directory.
    """
    current = time.time() if now is None else now
    cutoff = current - keep_days * 86_400
    try:
        skip = None if exclude is None else path_key(exclude)
        candidates = [p for p in log_dir.glob(LOG_GLOB) if p.is_file() and path_key(p) != skip]
        logs = sorted((p.stat().st_mtime, p.stat().st_size, p) for p in candidates)
    except OSError:
        return []

    removed: list[Path] = []
    kept: list[tuple[float, int, Path]] = []

    def delete(path: Path) -> bool:
        try:
            path.unlink()
        except OSError:
            return False
        removed.append(path)
        return True

    for mtime, size, path in logs:
        if not (mtime < cutoff and delete(path)):
            kept.append((mtime, size, path))
    total = sum(size for _, size, _ in kept)
    for _, size, path in kept:  # oldest first
        if total <= max_bytes:
            break
        if delete(path):
            total -= size
    return removed
