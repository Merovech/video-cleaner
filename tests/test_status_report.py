"""`status` and `report` commands and the results numbers behind `report` (no media, no tools)."""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from helpers import add_record, write_config
from video_cleaner import planning
from video_cleaner.cli import app
from video_cleaner.manifest import ManifestEntry, write_manifest
from video_cleaner.planning import build_results
from video_cleaner.state import State, Status

GIB = 1024**3


def _invoke(*args: str) -> Any:
    return CliRunner().invoke(app, list(args))


def _state(tmp_path: Path) -> State:
    return State.open(tmp_path / "appdata" / "state.db")


def _populate(state: State, lib: Path) -> dict[str, Path]:
    p = {
        "enc": lib / "Heat.avi",
        "enc2": lib / "Ran.avi",
        "left": lib / "Up.mkv",
        "flag": lib / "Dune.mkv",
        "retry": lib / "Alien.avi",
        "pt1": lib / "Kill Bill pt1.avi",
        "pt2": lib / "Kill Bill pt2.avi",
    }
    add_record(state, p["enc"], "encode", ("h264",), size=4000, est=2000)
    add_record(state, p["enc2"], "encode", ("h264",), size=3000, est=1500)
    add_record(state, p["left"], "leave_alone", ("already hevc",), size=3000)
    add_record(state, p["flag"], "flag", ("HDR (transfer smpte2084)",), size=5000)
    add_record(state, p["retry"], "encode", ("h264",), size=2000, est=1000)
    add_record(state, p["pt1"], "remux", ("multi-part file to join",), parts=(p["pt1"], p["pt2"]))
    add_record(state, p["pt2"], "leave_alone", ("part of group Kill Bill.mkv",))
    add_record(state, p["enc2"], "encode", ("h264",), size=3100, est=1500)  # older one now stale
    state.set_status(p["retry"], Status.RETRY, "scratch full: no room")
    state.set_status(p["enc"], Status.COMPLETED, "encode: saved 2000 bytes")
    return p


# --- status --------------------------------------------------------------------------------


def _count_lines(output: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for line in output.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[1].isdigit():
            counts[parts[0]] = int(parts[1])
    return counts


def test_status_lists_counts(tmp_path: Path) -> None:
    cfg = write_config(tmp_path)
    with _state(tmp_path) as st:
        _populate(st, tmp_path / "library")
        digest = planning.plan_hash(planning.build_plan(st))
        st.acknowledge_plan(digest)
        st.approve_calibration(94.5, 23)
    result = _invoke("status", "--config", str(cfg))
    assert result.exit_code == 0, result.output
    counts = _count_lines(result.output)
    assert counts["pending"] == 2  # Ran (live record only) and the multi-part group
    assert counts["completed"] == 1
    assert counts["left_alone"] == 1  # Up only: the member part is counted with its group
    assert counts["flagged"] == 1
    assert counts["retry"] == 1
    assert "stale" not in counts
    assert sum(counts.values()) == 6
    assert "multi-part members: 1" in result.output.lower()
    assert f"Plan acknowledged: yes ({digest[:12]})" in result.output
    assert "matches the current plan" in result.output
    assert "Calibration approved: yes (vmaf_target 94.5, crf_max 23)" in result.output
    assert "calibration approved" in result.output  # the last events include it


def test_status_without_ack_or_calibration(tmp_path: Path) -> None:
    cfg = write_config(tmp_path)
    with _state(tmp_path) as st:
        add_record(st, tmp_path / "library" / "a.avi", "encode", ("h264",))
    result = _invoke("status", "--config", str(cfg))
    assert result.exit_code == 0, result.output
    assert "Plan acknowledged: no" in result.output
    assert "Calibration approved: no" in result.output


def test_status_says_when_the_plan_changed_since_ack(tmp_path: Path) -> None:
    cfg = write_config(tmp_path)
    lib = tmp_path / "library"
    with _state(tmp_path) as st:
        add_record(st, lib / "a.avi", "encode", ("h264",))
        st.acknowledge_plan(planning.plan_hash(planning.build_plan(st)))
        add_record(st, lib / "a.avi", "encode", ("h264",), size=2000)
    result = _invoke("status", "--config", str(cfg))
    assert "plan changed since" in result.output


def test_status_shows_only_the_last_10_events(tmp_path: Path) -> None:
    cfg = write_config(tmp_path)
    with _state(tmp_path) as st:
        for i in range(15):
            st.acknowledge_plan(f"hash-{i:02d}")
    result = _invoke("status", "--config", str(cfg))
    assert result.exit_code == 0, result.output
    shown = [n for n in range(15) if f"hash-{n:02d}" in result.output]
    assert shown == list(range(5, 15))


def test_status_without_state_creates_nothing(tmp_path: Path) -> None:
    cfg = write_config(tmp_path)
    result = _invoke("status", "--config", str(cfg))
    assert result.exit_code == 0
    assert "survey" in result.output
    assert not (tmp_path / "appdata").exists()


def test_status_refuses_outputs_in_library(tmp_path: Path) -> None:
    library = tmp_path / "library"
    cfg = write_config(tmp_path, log_dir=library / "logs")
    result = _invoke("status", "--config", str(cfg))
    assert result.exit_code == 2
    assert "library root" in result.output
    assert os.listdir(library) == []


# --- report --------------------------------------------------------------------------------


def _manifest(tmp_path: Path, lib: Path) -> list[ManifestEntry]:
    done = tmp_path / "completed"
    entries = [
        ManifestEntry(
            original=(lib / "Heat.avi",),
            original_sizes=(4 * GIB,),
            original_mtimes=(1.0,),
            completed=done / "Heat.mkv",
            completed_size=1 * GIB,
            vmaf_mean=95.5,
            vmaf_worst=91.25,
            crf=21,
            verified=True,
            action="encode",
        ),
        ManifestEntry(
            original=(lib / "Kill Bill pt1.avi", lib / "Kill Bill pt2.avi"),
            original_sizes=(GIB, GIB),
            original_mtimes=(1.0, 2.0),
            completed=done / "Kill Bill.mkv",
            completed_size=2 * GIB,
            vmaf_mean=None,
            vmaf_worst=None,
            crf=None,
            verified=True,
            action="remux",
        ),
    ]
    write_manifest(done / "manifest.csv", entries)
    return entries


def test_build_results_numbers(tmp_path: Path) -> None:
    lib = tmp_path / "library"
    entries = _manifest(tmp_path, lib)
    with _state(tmp_path) as st:
        p = _populate(st, lib)
        results = build_results(entries, st)
    assert results.bytes_saved == 3 * GIB
    assert results.left_alone == 1  # the member part is not a left-alone file
    assert results.flagged == ((p["flag"], "HDR (transfer smpte2084)"),)
    assert results.retry == ((p["retry"], "scratch full: no room"),)


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def test_report_writes_results_csv_and_summary(tmp_path: Path) -> None:
    cfg = write_config(tmp_path)
    lib = tmp_path / "library"
    _manifest(tmp_path, lib)
    with _state(tmp_path) as st:
        _populate(st, lib)
    result = _invoke("report", "--config", str(cfg))
    assert result.exit_code == 0, result.output
    rows = _read(tmp_path / "appdata" / "report.csv")
    assert [r["completed"] for r in rows] == [
        str(tmp_path / "completed" / "Heat.mkv"),
        str(tmp_path / "completed" / "Kill Bill.mkv"),
    ]
    heat, kill_bill = rows
    assert heat["original_size"] == str(4 * GIB)
    assert heat["completed_size"] == str(GIB)
    assert heat["saving_pct"] == "75.0"
    assert (heat["vmaf_mean"], heat["vmaf_worst"], heat["crf"]) == ("95.5", "91.25", "21")
    assert heat["action"] == "encode"
    assert kill_bill["original"] == f"{lib / 'Kill Bill pt1.avi'}; {lib / 'Kill Bill pt2.avi'}"
    assert (kill_bill["saving_pct"], kill_bill["vmaf_mean"], kill_bill["crf"]) == ("0.0", "", "")
    out = result.output
    assert "Completed: 2" in out
    assert "Saved: 3.0 GiB" in out
    assert "Left alone: 1" in out
    assert "Flagged: 1" in out and "Dune.mkv: HDR (transfer smpte2084)" in out
    assert "Retry: 1" in out and "Alien.avi: scratch full: no room" in out
    assert str(tmp_path / "appdata" / "report.csv") in out


def test_report_without_manifest_has_no_completed_rows(tmp_path: Path) -> None:
    cfg = write_config(tmp_path)
    with _state(tmp_path) as st:
        add_record(st, tmp_path / "library" / "a.avi", "encode", ("h264",))
    result = _invoke("report", "--config", str(cfg))
    assert result.exit_code == 0, result.output
    assert "Completed: 0" in result.output
    assert _read(tmp_path / "appdata" / "report.csv") == []
    assert not (tmp_path / "completed").exists()  # report never creates the completed folder


def test_report_output_option(tmp_path: Path) -> None:
    cfg = write_config(tmp_path)
    with _state(tmp_path):
        pass
    target = tmp_path / "elsewhere" / "results.csv"
    result = _invoke("report", "--config", str(cfg), "--output", str(target))
    assert result.exit_code == 0, result.output
    assert target.is_file()
    assert not (tmp_path / "appdata" / "report.csv").exists()


def test_report_output_under_library_is_refused(tmp_path: Path) -> None:
    cfg = write_config(tmp_path)
    with _state(tmp_path):
        pass
    library = tmp_path / "library"
    result = _invoke("report", "--config", str(cfg), "--output", str(library / "r.csv"))
    assert result.exit_code == 2
    assert "library root" in result.output
    assert os.listdir(library) == []


def test_report_corrupt_manifest_exits_2(tmp_path: Path) -> None:
    cfg = write_config(tmp_path)
    with _state(tmp_path):
        pass
    (tmp_path / "completed").mkdir()
    (tmp_path / "completed" / "manifest.csv").write_text("not,a,manifest\n", encoding="utf-8")
    result = _invoke("report", "--config", str(cfg))
    assert result.exit_code == 2
    assert "manifest" in result.output


def test_report_without_state_exits_2_and_creates_nothing(tmp_path: Path) -> None:
    cfg = write_config(tmp_path)
    result = _invoke("report", "--config", str(cfg))
    assert result.exit_code == 2
    assert "survey" in result.output
    assert not (tmp_path / "appdata").exists()


@pytest.mark.parametrize("field", ["state_db", "completed_dir"])
def test_report_refuses_outputs_in_library(tmp_path: Path, field: str) -> None:
    library = tmp_path / "library"
    cfg = write_config(tmp_path, **{field: library / "inside"})
    result = _invoke("report", "--config", str(cfg))
    assert result.exit_code == 2
    assert "library root" in result.output
    assert os.listdir(library) == []


# --- original-unchanged check before the nightly overwrite ---------------------------------


def _real_original(lib: Path, name: str, data: bytes) -> tuple[Path, int, float]:
    path = lib / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    st = path.stat()
    return path, st.st_size, st.st_mtime


def _entry(tmp_path: Path, originals: list[tuple[Path, int, float]], name: str) -> ManifestEntry:
    return ManifestEntry(
        original=tuple(p for p, _, _ in originals),
        original_sizes=tuple(s for _, s, _ in originals),
        original_mtimes=tuple(m for _, _, m in originals),
        completed=tmp_path / "completed" / name,
        completed_size=1,
        vmaf_mean=None,
        vmaf_worst=None,
        crf=None,
        verified=True,
        action="remux",
    )


def test_report_says_per_entry_whether_the_original_is_unchanged(tmp_path: Path) -> None:
    cfg = write_config(tmp_path)
    lib = tmp_path / "library"
    same = _real_original(lib, "Same.avi", b"same")
    changed_path, size, mtime = _real_original(lib, "Changed.avi", b"before")
    gone = (lib / "Gone.avi", 5, 1.0)
    entries = [
        _entry(tmp_path, [same], "Same.mkv"),
        _entry(tmp_path, [(changed_path, size, mtime)], "Changed.mkv"),
        _entry(tmp_path, [gone], "Gone.mkv"),
    ]
    changed_path.write_bytes(b"after, longer")
    write_manifest(tmp_path / "completed" / "manifest.csv", entries)
    with _state(tmp_path):
        pass
    result = _invoke("report", "--config", str(cfg))
    assert result.exit_code == 0, result.output
    rows = {Path(r["completed"]).name: r for r in _read(tmp_path / "appdata" / "report.csv")}
    assert rows["Same.mkv"]["original_unchanged"] == "yes"
    assert rows["Changed.mkv"]["original_unchanged"] == "no"
    assert rows["Gone.mkv"]["original_unchanged"] == "no"
    out = " ".join(result.output.split())
    assert "Original unchanged: yes 1, no 2, unknown 0" in out
    assert f"no: {tmp_path / 'completed' / 'Changed.mkv'}" in out
    assert f"yes: {tmp_path / 'completed' / 'Same.mkv'}" in out


def test_original_check_is_unknown_when_library_is_unreachable(tmp_path: Path) -> None:
    lib = tmp_path / "library"  # never created: unreachable
    entries = [_entry(tmp_path, [(lib / "a.avi", 1, 1.0)], "a.mkv")]
    assert planning.originals_unchanged(entries, lib) == ("unknown",)


def test_original_check_is_unknown_on_other_os_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lib = tmp_path / "library"
    lib.mkdir()
    entries = [_entry(tmp_path, [(lib / "a.avi", 1, 1.0)], "a.mkv")]

    def denied(entry: ManifestEntry) -> bool:
        raise PermissionError(13, "denied")

    monkeypatch.setattr(planning, "check_unchanged", denied)
    assert planning.originals_unchanged(entries, lib) == ("unknown",)
