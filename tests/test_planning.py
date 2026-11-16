"""plan report, plan hash and the `plan` command (no media, no tools)."""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from helpers import add_record as _add
from helpers import write_config
from video_cleaner import planning
from video_cleaner.cli import app
from video_cleaner.planning import (
    PlanError,
    PlanGroup,
    build_plan,
    plan_hash,
    read_plan_hash,
    write_plan_report,
)
from video_cleaner.state import State, Status

LIB = Path("/lib")
ENC = LIB / "Heat (1995).avi"
LEFT = LIB / "Up (2009).mkv"
FLAG = LIB / "Dune (2021).mkv"
REMUX = LIB / "Alien (1979).mp4"
PT1 = LIB / "Kill Bill pt1.avi"
PT2 = LIB / "Kill Bill pt2.avi"


def _populate(state: State) -> None:
    _add(state, ENC, "encode", ("h264 1080p at 8000 kbps (floor 2500 kbps)",), size=4000, est=2000)
    _add(state, LEFT, "leave_alone", ("already hevc",), size=3000)
    _add(state, FLAG, "flag", ("HDR (transfer smpte2084)",), size=5000)
    _add(state, REMUX, "remux", ("new subtitles to mux in",), size=2000)
    _add(
        state,
        PT1,
        "encode",
        ("h264 1080p at 8000 kbps (floor 2500 kbps)", "multi-part file to join"),
        size=700,
        est=650,
        parts=(PT1, PT2),
    )
    _add(state, PT2, "leave_alone", ("part of group Kill Bill.mkv",), size=600)


@pytest.fixture
def state(tmp_path: Path) -> Any:
    st = State.open(tmp_path / "state.db")
    yield st
    st.close()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


# --- build_plan ----------------------------------------------------------------------------


def test_plan_report_contains_reason_for_every_file(state: State, tmp_path: Path) -> None:
    _populate(state)
    report = build_plan(state)
    out = tmp_path / "plan.csv"
    write_plan_report(report, out)
    rows = _read_csv(out)
    assert {r["path"] for r in rows} == {str(p) for p in (ENC, LEFT, FLAG, REMUX, PT1)}
    for row in rows:
        assert row["reasons"].strip(), row
    by_path = {r["path"]: r for r in rows}
    assert by_path[str(FLAG)]["action"] == "flag"
    assert "HDR" in by_path[str(FLAG)]["reasons"]
    assert by_path[str(LEFT)]["reasons"] == "already hevc"
    assert by_path[str(ENC)]["status"] == "pending"


def test_row_without_reasons_gets_one(state: State) -> None:
    _add(state, ENC, "encode", ())
    (row,) = build_plan(state).rows
    assert row.reasons and all(r.strip() for r in row.reasons)


def test_multipart_group_listed_once_and_members_not_counted(state: State, tmp_path: Path) -> None:
    _populate(state)
    report = build_plan(state)
    assert [r.path for r in report.rows if r.parts] == [PT1]
    assert PT2 not in {r.path for r in report.rows}
    assert report.counts["leave_alone"] == 1  # LEFT only, not the member part
    assert report.counts == {"leave_alone": 1, "encode": 2, "remux": 1, "flag": 1}
    assert report.groups == (PlanGroup("Kill Bill.mkv", (PT1, PT2), "encode"),)
    group_row = next(r for r in report.rows if r.path == PT1)
    assert group_row.size == 1300  # both parts
    assert report.total_bytes == 4000 + 3000 + 5000 + 2000 + 1300
    assert any("multi-part" in n and "Part 1" in n for n in report.notes)
    out = tmp_path / "plan.csv"
    write_plan_report(report, out)
    (csv_group,) = [r for r in _read_csv(out) if r["group"]]
    assert csv_group["group"] == "Kill Bill.mkv"
    assert csv_group["parts"] == f"{PT1}; {PT2}"


def test_member_not_in_any_live_group_is_listed(state: State) -> None:
    _add(state, PT2, "leave_alone", ("part of group Kill Bill.mkv",))
    (row,) = build_plan(state).rows
    assert row.path == PT2  # nothing would explain it otherwise


def test_flagged_files_are_listed_with_reasons(state: State) -> None:
    _populate(state)
    report = build_plan(state)
    flagged = [r for r in report.rows if r.action == "flag"]
    assert [(r.path, r.reasons) for r in flagged] == [(FLAG, ("HDR (transfer smpte2084)",))]
    assert any("flagged" in n for n in report.notes)


def test_stale_records_are_not_in_the_plan(state: State) -> None:
    _add(state, ENC, "encode", ("x",), size=1000)
    _add(state, ENC, "encode", ("x",), size=1200)  # the file changed: old record stale
    (row,) = build_plan(state).rows
    assert row.size == 1200


def test_estimated_saving_counts_outstanding_work_only(state: State) -> None:
    _add(state, ENC, "encode", ("x",), est=2000)
    _add(state, REMUX, "encode", ("x",), est=500)
    state.set_status(REMUX, Status.COMPLETED)
    assert build_plan(state).est_saving_bytes == 2000


def test_empty_state_says_run_survey(state: State) -> None:
    report = build_plan(state)
    assert report.rows == ()
    assert any("survey" in n for n in report.notes)


# --- hash ----------------------------------------------------------------------------------


def test_plan_hash_changes_when_a_record_changes(state: State) -> None:
    _populate(state)
    first = plan_hash(build_plan(state))
    assert plan_hash(build_plan(state)) == first  # deterministic
    _add(state, LEFT, "leave_alone", ("already hevc",), size=3001)  # re-survey: size changed
    second = plan_hash(build_plan(state))
    assert second != first
    _add(state, ENC, "flag", ("interlaced (field order tt)",), size=4000, est=0)
    assert plan_hash(build_plan(state)) != second


def test_plan_hash_ignores_run_progress(state: State) -> None:
    _populate(state)
    before = plan_hash(build_plan(state))
    state.set_status(ENC, Status.COMPLETED)
    assert plan_hash(build_plan(state)) == before


def test_write_plan_report_returns_hash_and_stores_nothing(state: State, tmp_path: Path) -> None:
    _populate(state)
    events = state.events(1000)
    report = build_plan(state)
    out = tmp_path / "out" / "plan.csv"
    digest = write_plan_report(report, out)
    assert digest == plan_hash(report)
    assert len(digest) == 64
    assert state.events(1000) == events
    assert state.plan_acknowledged() is None
    assert [p.name for p in out.parent.iterdir()] == ["plan.csv"]  # no temp left behind


def test_plan_csv_is_utf8_with_odd_names(state: State, tmp_path: Path) -> None:
    odd = LIB / "The First Omen (2024).en.MX]" / "Amélie ✓ [1080p].mkv"
    _add(state, odd, "encode", ("x",))
    out = tmp_path / "plan.csv"
    write_plan_report(build_plan(state), out)
    assert out.read_bytes()[:3] != b"\xef\xbb\xbf"
    assert _read_csv(out)[0]["path"] == str(odd)


def test_locked_plan_csv_raises_plan_error(
    state: State, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _populate(state)
    out = tmp_path / "plan.csv"

    def locked(src: Any, dst: Any) -> None:
        raise PermissionError(13, "in use", str(dst))

    monkeypatch.setattr(os, "replace", locked)
    with pytest.raises(PlanError, match=r"plan.csv"):
        write_plan_report(build_plan(state), out)
    assert list(tmp_path.glob("*.tmp")) == []


def test_read_plan_hash_missing_or_blank(tmp_path: Path) -> None:
    assert read_plan_hash(tmp_path / "plan.hash") is None
    (tmp_path / "plan.hash").write_text("  \n", encoding="utf-8")
    assert read_plan_hash(tmp_path / "plan.hash") is None


# --- CLI -----------------------------------------------------------------------------------


def _cli_state(tmp_path: Path) -> State:
    return State.open(tmp_path / "appdata" / "state.db")


def _invoke(*args: str) -> Any:
    return CliRunner().invoke(app, list(args))


def test_ack_requires_matching_hash(tmp_path: Path) -> None:
    cfg = write_config(tmp_path)
    with _cli_state(tmp_path) as st:
        _populate(st)
    appdata = tmp_path / "appdata"

    result = _invoke("plan", "--ack", "--config", str(cfg), "-q")
    assert result.exit_code == 2
    assert "run `plan` first and review it" in result.output

    result = _invoke("plan", "--config", str(cfg), "-q")
    assert result.exit_code == 0, result.output
    digest = (appdata / "plan.hash").read_text(encoding="utf-8").strip()
    assert digest in result.output
    assert (appdata / "plan.csv").is_file()
    with _cli_state(tmp_path) as st:
        assert st.plan_acknowledged() is None  # writing the plan acknowledges nothing

    result = _invoke("plan", "--ack", "--config", str(cfg), "-q")
    assert result.exit_code == 0, result.output
    with _cli_state(tmp_path) as st:
        assert st.plan_acknowledged() == digest
        _add(st, LEFT, "leave_alone", ("already hevc",), size=3001)  # survey saw a change

    result = _invoke("plan", "--ack", "--config", str(cfg), "-q")
    assert result.exit_code == 2
    assert "run `plan` first and review it" in result.output
    with _cli_state(tmp_path) as st:
        assert st.plan_acknowledged() == digest  # unchanged

    assert _invoke("plan", "--config", str(cfg), "-q").exit_code == 0
    assert _invoke("plan", "--ack", "--config", str(cfg), "-q").exit_code == 0
    with _cli_state(tmp_path) as st:
        new = st.plan_acknowledged()
    assert new is not None and new != digest


def test_cli_plan_prints_summary_groups_and_flagged(tmp_path: Path) -> None:
    cfg = write_config(tmp_path)
    with _cli_state(tmp_path) as st:
        _populate(st)
    result = _invoke("plan", "--config", str(cfg), "-q")
    assert result.exit_code == 0, result.output
    out = result.output
    assert "Plan: 5 files" in out
    assert "Kill Bill.mkv" in out and "Kill Bill pt2.avi" in out
    assert "Dune (2021).mkv" in out and "HDR" in out
    assert str(tmp_path / "appdata" / "plan.csv") in out
    assert "plan --ack" in out


def test_cli_ack_of_empty_plan_is_refused(tmp_path: Path) -> None:
    cfg = write_config(tmp_path)
    with _cli_state(tmp_path):
        pass
    assert _invoke("plan", "--config", str(cfg), "-q").exit_code == 0
    result = _invoke("plan", "--ack", "--config", str(cfg), "-q")
    assert result.exit_code == 2
    assert "survey" in result.output
    with _cli_state(tmp_path) as st:
        assert st.plan_acknowledged() is None


def test_cli_plan_without_state_says_survey_and_creates_nothing(tmp_path: Path) -> None:
    cfg = write_config(tmp_path)
    result = _invoke("plan", "--config", str(cfg), "-q")
    assert result.exit_code == 2
    assert "survey" in result.output
    assert not (tmp_path / "appdata").exists()


def test_cli_plan_refuses_outputs_in_library_before_anything(tmp_path: Path) -> None:
    library = tmp_path / "library"
    cfg = write_config(tmp_path, state_db=library / "state" / "state.db", log_dir=library / "logs")
    result = _invoke("plan", "--config", str(cfg), "-q")
    assert result.exit_code == 2
    assert "library root" in result.output
    assert sorted(os.listdir(library)) == []


def test_cli_plan_locked_csv_exits_2(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = write_config(tmp_path)
    with _cli_state(tmp_path) as st:
        _populate(st)

    def locked(*a: Any, **k: Any) -> str:
        raise PlanError("cannot replace plan.csv: it is open in another program")

    monkeypatch.setattr(planning, "write_plan_report", locked)
    result = _invoke("plan", "--config", str(cfg), "-q")
    assert result.exit_code == 2
    assert "open in another program" in result.output
    assert not (tmp_path / "appdata" / "plan.hash").exists()


def test_cli_plan_config_error_exits_2(tmp_path: Path) -> None:
    result = _invoke("plan", "--config", str(tmp_path / "missing.toml"))
    assert result.exit_code == 2
    assert "config error" in result.output


def test_group_name_falls_back_when_parts_no_longer_group(state: State) -> None:
    a, b = LIB / "Alpha.avi", LIB / "Beta.avi"  # no part markers: find_part_groups finds nothing
    _add(state, a, "remux", ("multi-part file to join",), parts=(a, b))
    _add(state, b, "leave_alone", ("part of group Alpha Beta.mkv",))
    assert [g.output_name for g in build_plan(state).groups] == ["Alpha Beta.mkv"]
    c = LIB / "Gamma.avi"
    _add(state, c, "remux", ("multi-part file to join",), parts=(c, LIB / "Delta.avi"))
    names = [g.output_name for g in build_plan(state).groups]
    assert names == ["Alpha Beta.mkv", "Gamma (parts)"]


def test_unwritable_folder_raises_plan_error(state: State, tmp_path: Path) -> None:
    blocker = tmp_path / "file"
    blocker.write_text("x", encoding="utf-8")
    with pytest.raises(PlanError, match="cannot write"):
        write_plan_report(build_plan(state), blocker / "plan.csv")


def test_plan_lists_at_most_20_flagged_on_the_console(tmp_path: Path) -> None:
    cfg = write_config(tmp_path)
    with _cli_state(tmp_path) as st:
        for i in range(23):
            _add(st, LIB / f"hdr{i:02d}.mkv", "flag", ("HDR",))
    result = _invoke("plan", "--config", str(cfg), "-q")
    assert result.exit_code == 0, result.output
    assert "... 3 more in the CSV" in result.output


def test_cli_plan_retry_requeues_a_run_outcome(tmp_path: Path) -> None:
    cfg = write_config(tmp_path)
    with _cli_state(tmp_path) as st:
        _populate(st)
        st.set_status(LEFT, Status.FLAGGED, "verify failed")
    result = _invoke("plan", "--retry", str(LEFT), "--config", str(cfg), "-q")
    assert result.exit_code == 0, result.output
    assert "left_alone" in result.output
    with _cli_state(tmp_path) as st:
        rec = st.get(LEFT)
    assert rec is not None and (rec.status, rec.note) == (Status.LEFT_ALONE, "")


def test_cli_plan_retry_unknown_path_exits_2(tmp_path: Path) -> None:
    cfg = write_config(tmp_path)
    with _cli_state(tmp_path) as st:
        _populate(st)
    result = _invoke("plan", "--retry", str(tmp_path / "nope.mkv"), "--config", str(cfg), "-q")
    assert result.exit_code == 2
    assert "no live state record" in result.output


def test_plan_notes_dolby_vision_profile_5_files_to_replace(state: State) -> None:
    from video_cleaner.decide import DV5_REPLACE_REASON

    _add(state, LIB / "DV5.mkv", "leave_alone", ("already hevc", DV5_REPLACE_REASON))
    report = build_plan(state)
    assert any("1 file(s) are Dolby Vision profile 5" in n for n in report.notes)
